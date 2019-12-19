from collections import OrderedDict

import numpy as np
import tensorflow as tf

from gym import spaces

from imitation.helpers.tf_util import TheanoFunction, flatgrad, clip
from imitation.helpers.networks import ActorNN, CriticNN
from imitation.helpers.math_util import huber_loss, rmsify, dermsify
from imitation.helpers import logger
from imitation.helpers.mpi_moments import mpi_mean_like
from imitation.helpers.mpi_running_mean_std import MpiRunningMeanStd
from imitation.helpers.misc_util import onehotify, zipsame
from imitation.helpers.mpi_adam import MpiAdamOptimizer
from imitation.helpers.console_util import log_module_info
from imitation.imitation_algorithms import memory as XP


def get_target_updates(vars_, targ_vars, polyak):
    """Return assignment ops for target network updates.
    Hard updates are used for initialization only, while soft updates are
    used throughout the training process, at every iteration.
    Note that DQN uses hard updates while training, but those updates
    are not performed every iteration (only once every XX iterations).
    """
    logger.info("setting up target updates")
    hard_updates = []
    soft_updates = []
    assert len(vars_) == len(targ_vars)
    for var_, targ_var in zipsame(vars_, targ_vars):
        logger.info('  {} <- {}'.format(targ_var.name, var_.name))
        hard_updates.append(tf.assign(targ_var, var_))
        soft_updates.append(tf.assign(targ_var, (1. - polyak) * targ_var + polyak * var_))
    assert len(hard_updates) == len(vars_)
    assert len(soft_updates) == len(vars_)
    return tf.group(*hard_updates), tf.group(*soft_updates)  # ops that group ops


def get_p_actor_updates(actor, perturbed_actor, pn_std):
    """Return assignment ops for actor parameters noise perturbations.
    The perturbations consist in applying additive gaussian noise the the perturbable
    actor variables, while simply leaving the non-perturbable ones untouched.
    """
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var_, perturbed_var in zipsame(actor.vars, perturbed_actor.vars):
        if var_ in actor.perturbable_vars:
            logger.info("  {} <- {} + noise".format(perturbed_var.name, var_.name))
            noised_up_var = var_ + tf.random_normal(tf.shape(var_), mean=0., stddev=pn_std)
            updates.append(tf.assign(perturbed_var, noised_up_var))
        else:
            logger.info("  {} <- {}".format(perturbed_var.name, var_.name))
            updates.append(tf.assign(perturbed_var, var_))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class SAMAgent(object):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        # Define everything in a specific scope
        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._init(*args, **kwargs)

    def _init(self, comm, env, hps, d):
        # Parameters
        self.comm = comm
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space

        if isinstance(self.ac_space, spaces.Box):
            self.ac_shape = self.ac_space.shape
        elif isinstance(self.ac_space, spaces.Discrete):
            self.ac_shape = (self.ac_space.n,)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        self.hps = hps
        assert self.hps.n > 1 or not self.hps.n_step_returns

        # Retrieve the synthetic reward network
        self.d = d.reward_nn

        # Assemble clipping functions
        unlimited_range = (-np.infty, np.infty)
        if isinstance(self.ac_space, spaces.Box):
            self.clip_obs = clip((-5., 5.))
            assert all(self.ac_space.low == -self.ac_space.high), "non-admissible ac space bounds"
            ac_valid_bound = self.ac_space.high[0]
            assert all(ac_comp == ac_valid_bound for ac_comp in self.ac_space.high)
            self.clip_acs = clip((-ac_valid_bound, ac_valid_bound))
            self.clip_rets = clip(unlimited_range)
        elif isinstance(self.ac_space, spaces.Discrete):
            self.clip_obs = clip(unlimited_range)
            self.clip_acs = lambda x: tf.cast(tf.identity(x), dtype=tf.float32)  # identity
            self.clip_rets = clip(unlimited_range)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Create inputs (`obs0`: s (s_t), `obs1`: s' (s_{t+1}))
        # Transitions atomic components placeholders
        self.obs0 = tf.placeholder(name='obs0', dtype=tf.float32, shape=(None,) + self.ob_shape)
        self.obs1 = tf.placeholder(name='obs1', dtype=tf.float32, shape=(None,) + self.ob_shape)
        self.dones1 = tf.placeholder(name='dones1', dtype=tf.float32, shape=(None, 1))
        self.rews = tf.placeholder(name='rews', dtype=tf.float32, shape=(None, 1))
        self.acs = tf.placeholder(name='acs', dtype=tf.float32, shape=(None,) + self.ac_shape)
        # Target critic 1-step td value placeholder
        self.tc1s = tf.placeholder(name='tc1s', dtype=tf.float32, shape=(None, 1))
        # TD length placeholder
        self.td_len = tf.placeholder(name='td_len', dtype=tf.float32, shape=(None, 1))
        # Target critic n-step td value placeholder
        self.tcns = tf.placeholder(name='tcns', dtype=tf.float32, shape=(None, 1))
        # Importance weights placeholder
        self.iws = tf.placeholder(name='iws', dtype=tf.float32, shape=(None, 1))
        if self.param_noise is not None:
            # Parameter noise placeholder
            self.pn_std = tf.placeholder(name='pn_std', dtype=tf.float32, shape=())

        # Create main actor and critic (need XOR-different name)
        self.actor = ActorNN(scope=self.scope, name='online_actor',
                             ac_space=self.ac_space, hps=self.hps)
        self.critic = CriticNN(scope=self.scope, name='online_critic', hps=self.hps)
        # Create target actor and critic
        self.targ_actor = ActorNN(scope=self.scope, name='targ_actor',
                                  ac_space=self.ac_space, hps=self.hps)
        self.targ_critic = CriticNN(scope=self.scope, name='targ_critic', hps=self.hps)
        if self.param_noise is not None:
            # Create parameter-noise-perturbed ('pnp') actor
            self.pnp_actor = ActorNN(scope=self.scope, name='pnp_actor',
                                     ac_space=self.ac_space, hps=self.hps)
            # Create adaptive-parameter-noise-perturbed ('apnp') actor
            self.apnp_actor = ActorNN(scope=self.scope, name='apnp_actor',
                                      ac_space=self.ac_space, hps=self.hps)

        # Process observations
        if self.hps.from_raw_pixels:
            # Scale pixel values
            self.obz0 = tf.cast(self.obs0, tf.float32) / 255.0
            self.obz1 = tf.cast(self.obs1, tf.float32) / 255.0
        else:
            # Smooth out observations using running statistics and clip
            if self.hps.rmsify_obs:
                with tf.variable_scope("apply_obs_rms"):
                    self.obs_rms = MpiRunningMeanStd(shape=self.ob_shape)
                # Smooth out observations using running statistics and clip
                self.obz0 = self.clip_obs(rmsify(self.obs0, self.obs_rms))
                self.obz1 = self.clip_obs(rmsify(self.obs1, self.obs_rms))
            else:
                self.obz0 = self.obs0
                self.obz1 = self.obs1

        # Process returns
        if self.hps.rmsify_rets:
            with tf.variable_scope("apply_ret_rms"):
                self.ret_rms = MpiRunningMeanStd()  # scalar, no shape to provide
            # Normalize and clip the 1-step (and optionaly n-step) critic target(s) value(s)
            self.tc1z = self.clip_rets(rmsify(self.tc1s, self.ret_rms))
            if self.hps.n_step_returns:
                self.tcnz = self.clip_rets(rmsify(self.tcns, self.ret_rms))
        else:
            self.tc1z = self.clip_rets(self.tc1s)
            if self.hps.n_step_returns:
                self.tcnz = self.clip_rets(self.tcns)

        # Build graph

        # Actor prediction from state input
        self.actor_pred = self.clip_acs(self.actor(self.obz0))

        # Critic prediction from state and action inputs
        self.critic_pred = self.clip_rets(self.critic(self.obz0, self.acs))
        if self.hps.rmsify_rets:
            self.critic_pred = dermsify(self.critic_pred, self.ret_rms)

        # Critic prediction from observation input and action outputed by the actor
        # critic(s, actor(s)), i.e. only dependent on state input
        self.critic_pred_w_actor = self.clip_rets(self.critic(self.obz0, self.actor_pred))
        if self.hps.rmsify_rets:
            self.critic_pred_w_actor = dermsify(self.critic_pred_w_actor, self.ret_rms)

        # Discriminator prediction from observation input and action outputed by the actor
        # critic(s, actor(s)), i.e. only dependent on state input
        self.d_pred_w_actor = self.clip_rets(self.d(self.obz0, self.actor_pred))
        if self.hps.rmsify_rets:
            self.d_pred_w_actor = dermsify(self.d_pred_w_actor, self.ret_rms)

        # Create target Q value defined as reward + gamma * Q' (1-step TD lookahead)
        # Q' (Q_{t+1}) is defined w/ s' (s_{t+1}) and a' (a_{t+1}),
        # where a' is the output of the target actor evaluated on s' (s_{t+1})
        self.q_prime = self.targ_critic(self.obz1, self.targ_actor(self.obz1))
        if self.hps.rmsify_rets:
            self.q_prime = dermsify(self.q_prime, self.ret_rms)
        self.mask = tf.ones_like(self.dones1) - self.dones1
        assert self.mask.get_shape().as_list() == self.q_prime.get_shape().as_list()
        self.masked_q_prime = self.mask * self.q_prime  # mask out Q's when terminal state reached
        self.targ_q = self.rews + (tf.pow(self.hps.gamma, self.td_len) * self.masked_q_prime)

        # Create callable objects
        if isinstance(self.ac_space, spaces.Box):
            self.act = TheanoFunction(inputs=[self.obs0],
                                      outputs=[self.actor_pred])
            self.act_q = TheanoFunction(inputs=[self.obs0],
                                        outputs=[self.actor_pred, self.critic_pred_w_actor])
        elif isinstance(self.ac_space, spaces.Discrete):
            # Note: actor network outputs softmax -> take argmax to pick one action
            self._actor_pred = tf.argmax(self.actor_pred, axis=-1)
            self.act = TheanoFunction(inputs=[self.obs0],
                                      outputs=[self._actor_pred])
            self.act_q = TheanoFunction(inputs=[self.obs0],
                                        outputs=[self._actor_pred, self.critic_pred_w_actor])

        if self.hps.rmsify_rets:
            self.old_ret_stats = TheanoFunction(inputs=[self.rews],
                                                outputs=[self.ret_rms.mean, self.ret_rms.std])

        self.get_targ_q = TheanoFunction(inputs=[self.obs1, self.rews, self.dones1, self.td_len],
                                         outputs=self.targ_q)

        # Set up training components
        if self.param_noise is not None:
            self.setup_param_noise()
        self.setup_replay_buffer()
        self.actor_ops = self.setup_actor()
        self.critic_ops = self.setup_critic()
        if self.hps.rmsify_rets and self.hps.enable_popart:
            self.setup_popart()
        self.setup_target_network_updates()

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        ac_noise = None
        param_noise = None
        if isinstance(self.ac_space, spaces.Box):
            ac_dim = self.ac_space.shape[-1]  # num dims
        elif isinstance(self.ac_space, spaces.Discrete):
            ac_dim = self.ac_space.n  # num ac choices
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")
        logger.info("parsing noise type")
        # Parse the comma-seprated (with possible whitespaces) list of noise params
        for cur_noise_type in noise_type.split(','):
            cur_noise_type = cur_noise_type.strip()  # remove all whitespaces (start and end)
            # If the specified noise type is litterally 'none'
            if cur_noise_type == 'none':
                pass
            # If 'adaptive-param' is in the specified string for noise type
            elif 'adaptive-param' in cur_noise_type:
                # Set parameter noise
                from imitation.imitation_algorithms.param_noise import AdaptiveParamNoise
                if isinstance(self.ac_space, spaces.Box):
                    _, std = cur_noise_type.split('_')
                    std = float(std)
                    param_noise = AdaptiveParamNoise(initial_std=std, delta=std)
                elif isinstance(self.ac_space, spaces.Discrete):
                    _, init_eps = cur_noise_type.split('_')
                    init_eps = float(init_eps)
                    # Compute param noise thres depending on eps, as explained in Appendix C.1
                    # of the paper 'Parameter Space Noise for Exploration', Plappert, ICLR 2017
                    init_delta = -np.log(1. - init_eps + (init_eps / float(ac_dim)))
                    param_noise = AdaptiveParamNoise(delta=init_delta)
                    self.setup_eps_greedy(init_eps)
                logger.info("  {} configured".format(param_noise))
            elif 'normal' in cur_noise_type:
                assert isinstance(self.ac_space, spaces.Box), "must be continuous ac space"
                _, std = cur_noise_type.split('_')
                # Spherical (isotropic) gaussian action noise
                from imitation.imitation_algorithms.ac_noise import NormalAcNoise
                ac_noise = NormalAcNoise(mu=np.zeros(ac_dim), sigma=float(std) * np.ones(ac_dim))
                logger.info("  {} configured".format(ac_noise))
            elif 'ou' in cur_noise_type:
                assert isinstance(self.ac_space, spaces.Box), "must be continuous ac space"
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                from imitation.imitation_algorithms.ac_noise import OUAcNoise
                ac_noise = OUAcNoise(mu=np.zeros(ac_dim), sigma=float(std) * np.ones(ac_dim))
                logger.info("  {} configured".format(ac_noise))
            else:
                raise RuntimeError("unknown specified noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def setup_eps_greedy(self, init_eps):
        # Define fraction of training period over which the exploration rate is annealed
        explo_frac = 0.1
        sched_timesteps = int(explo_frac * self.hps.num_timesteps)
        # Define final value of random action probability
        explo_final_eps = 0.02
        from imitation.helpers.linear_schedule import LinearSchedule
        self.eps_greedy_sched = LinearSchedule(sched_timesteps=sched_timesteps,
                                               init_p=init_eps,
                                               final_p=explo_final_eps)

    def adapt_eps_greedy(self, t):
        # Fetch a new eps value from eps-greedy schedule
        new_eps = self.eps_greedy_sched.value(t)
        # Compute a new param noise threshold depending on eps, as explained in Appendix C.1
        # of the paper 'Parameter Space Noise for Exploration', Plappert, ICLR 2017
        new_delta = -np.log(1. - new_eps + (new_eps / float(self.ac_space.n)))
        # Adapt the param noise threshold
        self.param_noise.adapt_delta(new_delta)

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info("setting up replay buffer")
        # In the discrete actions case, we store the acs indices
        if isinstance(self.ac_space, spaces.Box):
            ac_shape_ = self.ac_shape
        elif isinstance(self.ac_space, spaces.Discrete):
            ac_shape_ = ()
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")
        xp_params = {'limit': self.hps.mem_size,
                     'ob_shape': self.ob_shape,
                     'ac_shape': ac_shape_}
        extra_xp_params = {'alpha': self.hps.alpha,
                           'beta': self.hps.beta,
                           'ranked': self.hps.ranked}

        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                self.replay_buffer = XP.UnrealRB(**xp_params)
            else:  # Vanilla prioritized experience replay
                self.replay_buffer = XP.PrioritizedRB(**xp_params, **extra_xp_params)
        else:  # Vanilla experience replay
            self.replay_buffer = XP.RB(**xp_params)
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("  {} configured".format(self.replay_buffer))

    def setup_target_network_updates(self):
        logger.info("setting up target network updates")
        actor_args = [self.actor.vars, self.targ_actor.vars, self.hps.polyak]
        critic_args = [self.critic.vars, self.targ_critic.vars, self.hps.polyak]
        actor_hard_updates, actor_soft_updates = get_target_updates(*actor_args)
        critic_hard_updates, critic_soft_updates = get_target_updates(*critic_args)
        self.targ_hard_updates = [actor_hard_updates, critic_hard_updates]
        self.targ_soft_updates = [actor_soft_updates, critic_soft_updates]

        # Create callable objects
        self.perform_targ_hard_updates = TheanoFunction(inputs=[],
                                                        outputs=[self.targ_hard_updates])
        self.perform_targ_soft_updates = TheanoFunction(inputs=[],
                                                        outputs=[self.targ_soft_updates])

    def setup_param_noise(self):
        """Setup two separate perturbed actors, one which be used only for interacting
        with the environment, while the other will be used exclusively for std adaption.
        We use two instead of one for clarity-related purposes.
        """
        # Define parameter corresponding to the current parameter noise stddev
        self.pn_cur_std = self.param_noise.cur_std  # real value, not the placeholder

        logger.info("setting up param noise")
        # Configure parameter-noise-perturbed ('pnp') actor
        # Use: interact with the environment
        self.pnp_actor_pred = self.clip_acs(self.pnp_actor(self.obz0))
        self.p_actor_updates = get_p_actor_updates(self.actor, self.pnp_actor, self.pn_std)

        logger.info("setting up adaptive param noise")
        # Configure adaptive-parameter-noise-perturbed ('apnp') actor
        # Use: adapt the standard deviation
        self.apnp_actor_pred = self.clip_acs(self.apnp_actor(self.obz0))
        self.a_p_actor_updates = get_p_actor_updates(self.actor, self.apnp_actor, self.pn_std)
        self.a_dist = tf.sqrt(tf.reduce_mean(tf.square(self.actor_pred - self.apnp_actor_pred)))

        # Create callable objects
        # Act (and compute Q) according to the parameter-noise-perturbed actor
        self.p_act = TheanoFunction(inputs=[self.obs0],
                                    outputs=[self.pnp_actor_pred])
        self.p_act_q = TheanoFunction(inputs=[self.obs0],
                                      outputs=[self.pnp_actor_pred,
                                               self.critic_pred_w_actor])

        if isinstance(self.ac_space, spaces.Box):
            self.p_act = TheanoFunction(inputs=[self.obs0],
                                        outputs=[self.pnp_actor_pred])
            self.p_act_q = TheanoFunction(inputs=[self.obs0],
                                          outputs=[self.pnp_actor_pred,
                                                   self.critic_pred_w_actor])
        elif isinstance(self.ac_space, spaces.Discrete):
            # Note: actor network outputs softmax -> take argmax to pick one action
            self.pnp_actor_pred_ = tf.argmax(self.pnp_actor_pred, axis=-1)
            self.p_act = TheanoFunction(inputs=[self.obs0],
                                        outputs=[self.pnp_actor_pred_])
            self.p_act_q = TheanoFunction(inputs=[self.obs0],
                                          outputs=[self.pnp_actor_pred_,
                                                   self.critic_pred_w_actor])

        # Create distance between actor and adaptive-parameter-noise-perturbed actor predictions
        self.get_a_p_dist = TheanoFunction(inputs=[self.obs0, self.pn_std],
                                           outputs=self.a_dist)
        # Retrieve parameter-noise-perturbation updates
        self.apply_p_actor_updates = TheanoFunction(inputs=[self.pn_std],
                                                    outputs=[self.p_actor_updates])
        # Retrieve adaptive-parameter-noise-perturbation updates
        self.apply_a_p_actor_updates = TheanoFunction(inputs=[self.pn_std],
                                                      outputs=[self.a_p_actor_updates])

    def setup_actor(self):
        logger.info("setting up actor optimizer")

        losses = OrderedDict()

        # Create the Q loss as the negative of the cumulated Q values
        q_loss = -tf.reduce_mean(self.critic_pred_w_actor)
        q_loss *= self.hps.q_actor_loss_scale

        # Create the actor loss w/ the scaled Q loss
        loss = q_loss

        losses.update({'actor_q_loss': q_loss})

        # Create the D loss as the negative of the cumulated D values
        d_loss = -tf.reduce_mean(self.d_pred_w_actor)
        d_loss *= self.hps.d_actor_loss_scale

        # Add the D loss to the actor loss
        loss += d_loss

        losses.update({'actor_d_loss': d_loss})

        # Add assembled actor loss
        losses.update({'actor_total_loss': loss})

        # Create gradients
        grads = flatgrad(loss, self.actor.trainable_vars, self.hps.clip_norm)

        # Create mpi adam optimizer
        optimizer = MpiAdamOptimizer(comm=self.comm,
                                     clip_norm=self.hps.clip_norm,
                                     learning_rate=self.hps.actor_lr,
                                     name='actor_adam')
        optimize_ = optimizer.minimize(loss=loss, var_list=self.actor.trainable_vars)

        # Create callable objects
        get_losses = TheanoFunction(inputs=[self.obs0], outputs=list(losses.values()))
        get_grads = TheanoFunction(inputs=[self.obs0], outputs=grads)
        optimize = TheanoFunction(inputs=[self.obs0], outputs=optimize_)

        # Log statistics
        log_module_info(logger, self.name, self.actor)

        # Return the actor ops
        return {'names': list(losses.keys()),
                'losses': get_losses,
                'grads': get_grads,
                'optimizer': optimizer,
                'optimize': optimize}

    def setup_critic(self):
        logger.info("setting up critic optimizer")

        losses = OrderedDict()

        phs = [self.obs0, self.acs]

        if self.hps.prioritized_replay:
            phs.append(self.iws)

        # Create the 1-step look-ahead TD error loss
        td_errors_1 = self.critic_pred - self.tc1z
        hubered_td_errors_1 = huber_loss(td_errors_1)
        if self.hps.prioritized_replay:
            # Adjust with importance weights
            hubered_td_errors_1 *= self.iws
        td_loss_1 = tf.reduce_mean(hubered_td_errors_1)
        td_loss_1 *= self.hps.td_loss_1_scale

        # Create the critic loss w/ the scaled 1-step TD loss
        loss = td_loss_1

        losses.update({'critic_td_loss_1': td_loss_1})

        phs.append(self.tc1s)

        if self.hps.n_step_returns:
            # Create the n-step look-ahead TD error loss
            td_errors_n = self.critic_pred - self.tcnz
            hubered_td_errors_n = huber_loss(td_errors_n)
            if self.hps.prioritized_replay:
                # Adjust with importance weights
                hubered_td_errors_n *= self.iws
            td_loss_n = tf.reduce_mean(hubered_td_errors_n)
            td_loss_n *= self.hps.td_loss_n_scale

            # Add the scaled n-step TD loss to the critic loss
            loss += td_loss_n

            losses.update({'critic_td_loss_n': td_loss_n})

            phs.append(self.tcns)

        # Fetch critic's regularization losses (@property of the network)
        wd_loss = tf.reduce_sum(self.critic.regularization_losses)
        # Note: no need to multiply by a scale as it has already been scaled
        logger.info("setting up weight decay")
        if self.hps.wd_scale > 0:
            for var in self.critic.trainable_vars:
                if var in self.critic.decayable_vars:
                    logger.info("  {} <- wd w/ scale {}".format(var.name, self.hps.wd_scale))
                else:
                    logger.info("  {}".format(var.name))

        # Add critic weight decay regularization to the critic loss
        loss += wd_loss

        losses.update({'critic_wd': wd_loss})

        # Add assembled critic loss
        losses.update({'critic_total_loss': loss})

        # Create gradients
        grads = flatgrad(loss, self.critic.trainable_vars, self.hps.clip_norm)

        # Create mpi adam optimizer
        optimizer = MpiAdamOptimizer(comm=self.comm,
                                     clip_norm=self.hps.clip_norm,
                                     learning_rate=self.hps.critic_lr,
                                     name='critic_adam')
        optimize_ = optimizer.minimize(loss=loss, var_list=self.critic.trainable_vars)

        # Create callable objects
        get_losses = TheanoFunction(inputs=phs, outputs=list(losses.values()))
        get_grads = TheanoFunction(inputs=phs, outputs=grads)
        optimize = TheanoFunction(inputs=phs, outputs=optimize_)

        if self.hps.prioritized_replay:
            td_errors_ops = [td_errors_1] + ([td_errors_n]
                                             if self.hps.n_step_returns
                                             else [])
            get_td_errors = TheanoFunction(inputs=phs, outputs=td_errors_ops)

        # Log statistics
        log_module_info(logger, self.name, self.critic)

        # Return the critic ops
        out = {'names': list(losses.keys()),
               'losses': get_losses,
               'grads': get_grads,
               'optimizer': optimizer,
               'optimize': optimize}
        if self.hps.prioritized_replay:
            out.update({'td_errors': get_td_errors})

        return out

    def setup_popart(self):
        """Play w/ the magnitude of the return @ the critic output
        by renormalizing the critic output vars (w + b) w/ old running statistics
        Reference paper: https://arxiv.org/pdf/1602.07714.pdf
        """
        logger.info("setting up popart")

        # Setting old and new stds and means
        self.old_std = tf.placeholder(name='old_std', dtype=tf.float32, shape=[1])
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(name='old_mean', dtype=tf.float32, shape=[1])
        new_mean = self.ret_rms.mean

        self.popart_op = []
        # Pass once in the critic and once in the target critic -> 2 loop steps
        for output_vars in [self.critic.output_vars, self.targ_critic.output_vars]:
            # Ensure the network only has 2 vars w/ 'final' in their names (w + b of output layer)
            assert len(output_vars) == 2, "only w + b of the critic output layer \
                                           should be caught -> 2 vars"
            out_names = [var.name for var in output_vars]
            for out_name in out_names:
                # Log output variables on which popart involved in popart
                logger.info("  {}".format(out_name))
            # Unpack weight and bias of output layer
            w, b = output_vars
            # Ensure that w is indeed a weight, and that b is indeed a bias
            assert 'kernel' in w.name, "'w' not in w.name"
            assert 'bias' in b.name, "'b' not in b.name"
            # Ensure that both w and b are compatible w/ the critic spitting out a scalar
            assert w.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.popart_op += [w.assign(w * self.old_std / new_std)]
            self.popart_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

            # Create callable objects
            self.popart = TheanoFunction(inputs=[self.old_mean, self.old_std],
                                         outputs=[self.popart_op])

    def predict(self, obs, apply_noise, compute_q):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated Q value.
        """
        if apply_noise and self.param_noise is not None:
            # Predict following a parameter-noise-perturbed actor
            if compute_q:
                ac, q = self.p_act_q({self.obs0: obs[None]})
            else:
                ac, q = (self.p_act({self.obs0: obs[None]}), None)
        else:
            # Predict following the non-perturbed actor
            if compute_q:
                ac, q = self.act_q({self.obs0: obs[None]})
            else:
                ac, q = (self.act({self.obs0: obs[None]}), None)

        # Collapse the returned action into one dimension
        ac = ac.flatten()

        if apply_noise and self.ac_noise is not None:
            # Apply additive action noise once the action has been predicted,
            # with a parameter-noise-perturbed agent or not.
            noise = self.ac_noise.generate()
            assert noise.shape == ac.shape
            ac += noise

        return ac, np.asscalar(q.flatten()) if compute_q else (ac, None)

    def store_transition(self, ob0, ac, rew, ob1, done1):
        """Store a experiental transition in the replay buffer"""
        # Scale the reward
        rew *= self.hps.reward_scale
        # Store the transition in the replay buffer
        self.replay_buffer.append(ob0, ac, rew, ob1, done1)
        if self.hps.rmsify_obs:
            # Update running mean and std over observations
            self.obs_rms.update(np.array([ob0]), self.comm)

    def train(self):
        """Train the SAM agent"""
        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            out = self.replay_buffer.n_step_lookahead_sample(batch_size=self.hps.batch_size,
                                                             n=self.hps.n,
                                                             gamma=self.hps.gamma)
            batch, batchn = out
        else:
            batch = self.replay_buffer.sample(batch_size=self.hps.batch_size)

        if isinstance(self.ac_space, spaces.Discrete):
            # Actions are stored as scalars in the replay buffer for storage reasons
            # but the critic processes one-hot versions of those scalars
            batch['acs'] = onehotify(batch['acs'], self.ac_space.n)

        # Create vector containing the lengths of the 1-step lookaheads -> 1 for all
        onesies = np.ones(shape=(self.hps.batch_size, 1))

        # Compute target Q values
        targ_q_1 = self.get_targ_q({self.obs1: batch['obs1'],
                                    self.rews: batch['rews'],
                                    self.dones1: batch['dones1'].astype(dtype='float32'),
                                    self.td_len: onesies})

        if self.hps.n_step_returns:
            targ_q_n = self.get_targ_q({self.obs1: batchn['obs1'],
                                        self.rews: batchn['rews'],
                                        self.dones1: batchn['dones1'].astype(dtype='float32'),
                                        self.td_len: batchn['td_len']})

        if self.hps.rmsify_rets and self.hps.enable_popart:
            # Compute old return statistics
            old_ret_mean, old_ret_std = self.old_ret_stats({self.rews: batch['rews']})
            # The values are stored as `old_*`, an update is now performed
            # on the return stats, using the freshly computed target Q value
            self.ret_rms.update(targ_q_1.flatten(), self.comm)
            if self.hps.n_step_returns:
                self.ret_rms.update(targ_q_n.flatten(), self.comm)
            # Perform popart critic output parameters rectifications
            self.popart({self.old_mean: np.array([old_ret_mean]),
                         self.old_std: np.array([old_ret_std])})

        # Compute losses and gradients
        actor_losses = self.actor_ops['losses']({self.obs0: batch['obs0']})
        actor_grads = self.actor_ops['grads']({self.obs0: batch['obs0']})

        feeds = {self.obs0: batch['obs0'],
                 self.acs: batch['acs']}
        if self.hps.prioritized_replay:
            feeds.update({self.iws: batch['iws']})
        feeds.update({self.tc1s: targ_q_1})
        if self.hps.n_step_returns:
            feeds.update({self.tcns: targ_q_n})

        critic_losses = self.critic_ops['losses'](feeds)
        critic_grads = self.critic_ops['grads'](feeds)

        # Perform mpi gradient descent update
        self.actor_ops['optimize']({self.obs0: batch['obs0']})
        self.critic_ops['optimize'](feeds)

        if self.hps.prioritized_replay:
            # Update priorities
            td_errors = self.critic_ops['td_errors'](feeds)
            if self.hps.n_step_returns:  # `td_errors` = [td_errors_1, td_errors_n]
                # Sum `td_errors_1` and `td_errors_n` element-wise
                td_errors = np.sum(td_errors, axis=0)
            else:  # `td_errors` = [td_errors_1]
                # Extract the only element of the list
                td_errors = td_errors[0]
            new_priorities = np.abs(td_errors) + 1e-6  # epsilon from paper
            self.replay_buffer.update_priorities(batch['idxs'], new_priorities)

        # Aggregate the elements to return
        losses_and_grads = {'actor_grads': actor_grads,
                            'actor_losses': actor_losses,
                            'critic_grads': critic_grads,
                            'critic_losses': critic_losses}

        return losses_and_grads

    def sync_from_root(self):
        """Initialize both actor and critic, as well as their target counterparts"""
        # Synchronize the optimizers across all mpi workers
        self.actor_ops['optimizer'].sync_from_root(self.actor.trainable_vars)
        self.critic_ops['optimizer'].sync_from_root(self.critic.trainable_vars)

    def initialize_target_nets(self):
        """Initialize target networks as hard copies of the main networks"""
        self.perform_targ_hard_updates({})

    def update_target_net(self):
        """Update the target networks by slowly tracking their non-target counterparts"""
        self.perform_targ_soft_updates({})

    def adapt_param_noise(self, comm):
        """Adapt the parameter noise standard deviation"""
        # Perturb separate copy of the policy to adjust the scale for the next 'real' perturbation
        batch = self.replay_buffer.sample(batch_size=self.hps.batch_size)
        # Align the adaptive-parameter-noise-perturbed weights with the actor
        self.apply_a_p_actor_updates({self.pn_std: self.param_noise.cur_std})
        # Compute distance between actor and adaptive-parameter-noise-perturbed actor predictions
        dist = self.get_a_p_dist({self.obs0: batch['obs0'],
                                  self.pn_std: self.param_noise.cur_std})
        # Average the computed distance between the mpi workers
        self.pn_dist = mpi_mean_like(dist, comm)
        # Adapt the parameter noise
        self.param_noise.adapt_std(self.pn_dist)

    def reset_noise(self):
        """Reset noise processes at episode termination"""
        # Reset action noise
        if self.ac_noise is not None:
            self.ac_noise.reset()
        # Reset parameter-noise-perturbed actor vars by redefining the pnp actor
        # w.r.t. the actor (by applying additive gaussian noise with current std)
        if self.param_noise is not None:
            self.apply_p_actor_updates({self.pn_std: self.param_noise.cur_std})
