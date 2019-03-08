from collections import OrderedDict

import numpy as np
import tensorflow as tf

from gym import spaces

from imitation.helpers.tf_util import clip, flatgrad, TheanoFunction
from imitation.helpers.distributions import BernoulliPd
from imitation.helpers.networks import RewardNN
from imitation.helpers.mpi_adam import MpiAdamOptimizer
from imitation.helpers.mpi_running_mean_std import MpiRunningMeanStd
from imitation.helpers.math_util import rmsify
from imitation.helpers.console_util import log_module_info
from imitation.helpers import logger


class Discriminator(object):

    def __init__(self, name, env, hps, comm):
        self.name = name
        # Define everything in a specific scope
        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._init(env=env, hps=hps, comm=comm)

    def _init(self, env, hps, comm):
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        if "NoFrameskip" in env.spec.id:
            # Expand the dimension for Atari
            self.ac_shape = (1,) + self.ac_shape
        self.hps = hps
        assert self.hps.ent_reg_scale >= 0, "'ent_reg_scale' must be non-negative"
        self.comm = comm

        # Assemble clipping functions
        unlimited_range = (-np.infty, np.infty)
        if isinstance(self.ac_space, spaces.Box):
            self.clip_obs = clip((-5., 5.))
        elif isinstance(self.ac_space, spaces.Discrete):
            self.clip_obs = clip(unlimited_range)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        # Define the synthetic reward network
        self.reward_nn = RewardNN(scope=self.scope, name='sr', hps=self.hps)

        # Create inputs
        self.p_obs = tf.placeholder(name='p_obs', dtype=tf.float32,
                                    shape=(None,) + self.ob_shape)
        self.p_acs = tf.placeholder(name='p_acs', dtype=tf.float32,
                                    shape=(None,) + self.ac_shape)
        self.e_obs = tf.placeholder(name='e_obs', dtype=tf.float32,
                                    shape=(None,) + self.ob_shape)
        self.e_acs = tf.placeholder(name='e_acs', dtype=tf.float32,
                                    shape=(None,) + self.ac_shape)

        # Rescale observations
        if self.hps.from_raw_pixels:
            # Scale de pixel values
            p_obz = self.p_obs / 255.0
            e_obz = self.e_obs / 255.0
        else:
            if self.hps.rmsify_obs:
                # Smooth out observations using running statistics and clip
                with tf.variable_scope("apply_obs_rms"):
                    self.obs_rms = MpiRunningMeanStd(shape=self.ob_shape)
                p_obz = self.clip_obs(rmsify(self.p_obs, self.obs_rms))
                e_obz = self.clip_obs(rmsify(self.e_obs, self.obs_rms))
            else:
                p_obz = self.p_obs
                e_obz = self.e_obs

        # Build graph
        p_scores = self.reward_nn(p_obz, self.p_acs)
        e_scores = self.reward_nn(e_obz, self.e_acs)
        scores = tf.concat([p_scores, e_scores], axis=0)
        # `scores` define the conditional distribution D(s,a) := p(label|(state,action))

        # Create entropy loss
        bernouilli_pd = BernoulliPd(logits=scores)
        ent_mean = tf.reduce_mean(bernouilli_pd.entropy())
        ent_loss = -self.hps.ent_reg_scale * ent_mean

        # Create labels
        fake_labels = tf.zeros_like(p_scores)
        real_labels = tf.ones_like(e_scores)
        if self.hps.label_smoothing:
            # Label smoothing, suggested in 'Improved Techniques for Training GANs',
            # Salimans 2016, https://arxiv.org/abs/1606.03498
            # The paper advises on the use of one-sided label smoothing (positive targets side)
            # Extra comment explanation: https://github.com/openai/improved-gan/blob/
            # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
            if not self.hps.one_sided_label_smoothing:
                # Fake labels (negative targets)
                soft_fake_u_b = 0.0  # standard, hyperparameterization not needed
                soft_fake_l_b = 0.3  # standard, hyperparameterization not needed
                fake_labels = tf.random_uniform(shape=tf.shape(fake_labels),
                                                name="fake_labels_smoothing",
                                                minval=soft_fake_l_b, maxval=soft_fake_u_b)
            # Real labels (positive targets)
            soft_real_u_b = 0.7  # standard, hyperparameterization not needed
            soft_real_l_b = 1.2  # standard, hyperparameterization not needed
            real_labels = tf.random_uniform(shape=tf.shape(real_labels),
                                            name="real_labels_smoothing",
                                            minval=soft_real_l_b, maxval=soft_real_u_b)

        # # Build accuracies
        p_acc = tf.reduce_mean(tf.sigmoid(p_scores))
        e_acc = tf.reduce_mean(tf.sigmoid(e_scores))

        # Build binary classification (cross-entropy) losses, equal to the negative log likelihood
        # for random variables following a Bernoulli law, divided by the batch size
        p_bernoulli_pd = BernoulliPd(logits=p_scores)
        p_loss_mean = tf.reduce_mean(p_bernoulli_pd.neglogp(fake_labels))
        e_bernoulli_pd = BernoulliPd(logits=e_scores)
        e_loss_mean = tf.reduce_mean(e_bernoulli_pd.neglogp(real_labels))

        # Add a gradient penalty (motivation from WGANs (Gulrajani),
        # but empirically useful in JS-GANs (Lucic et al. 2017))

        def batch_size(x):
            """Returns an int corresponding to the batch size of the input tensor"""
            return tf.to_float(tf.shape(x)[0], name='get_batch_size_in_fl32')

        shape_obz = (tf.to_int64(batch_size(p_obz)),) + self.ob_shape
        eps_obz = tf.random_uniform(shape=shape_obz, minval=0.0, maxval=1.0)
        obz_interp = eps_obz * p_obz + (1. - eps_obz) * e_obz
        shape_acs = (tf.to_int64(batch_size(self.p_acs)),) + self.ac_shape
        eps_acs = tf.random_uniform(shape=shape_acs, minval=0.0, maxval=1.0)
        acs_interp = eps_acs * self.p_acs + (1. - eps_acs) * self.e_acs
        interp_scores = self.reward_nn(obz_interp, acs_interp)
        grads = tf.gradients(interp_scores, [obz_interp, acs_interp], name="interp_grads")
        assert len(grads) == 2, "length must be exacty 2"
        grad_squared_norms = [tf.reduce_mean(tf.square(grad)) for grad in grads]
        grad_norm = tf.sqrt(tf.reduce_sum(grad_squared_norms))
        grad_pen = tf.reduce_mean(tf.square(grad_norm - 1.0))

        losses = OrderedDict()

        # Add losses
        losses.update({'d_policy_loss': p_loss_mean,
                       'd_expert_loss': e_loss_mean,
                       'd_ent_mean': ent_mean,
                       'd_ent_loss': ent_loss,
                       'd_policy_acc': p_acc,
                       'd_expert_acc': e_acc,
                       'd_grad_pen': grad_pen})

        # Assemble discriminator loss
        loss = p_loss_mean + e_loss_mean + ent_loss + 10 * grad_pen
        # gradient penalty coefficient aligned with the value used in Gulrajani et al.

        # Add assembled disciminator loss
        losses.update({'d_total_loss': loss})

        # Compute gradients
        grads = flatgrad(loss, self.trainable_vars, self.hps.clip_norm)

        # Create mpi adam optimizer
        self.optimizer = MpiAdamOptimizer(comm=self.comm,
                                          clip_norm=self.hps.clip_norm,
                                          learning_rate=self.hps.d_lr,
                                          name='d_adam')
        optimize_ = self.optimizer.minimize(loss=loss, var_list=self.trainable_vars)

        # Create callable objects
        phs = [self.p_obs, self.p_acs, self.e_obs, self.e_acs]
        self.get_losses = TheanoFunction(inputs=phs, outputs=list(losses.values()))
        self.get_grads = TheanoFunction(inputs=phs, outputs=grads)
        self.optimize = TheanoFunction(inputs=phs, outputs=optimize_)

        # Make loss names graspable from outside
        self.names = list(losses.keys())

        # Define synthetic reward
        if self.hps.non_satur_grad:
            # Recommended in the original GAN paper and later in Fedus et al. 2017 (Many Paths...)
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            reward = tf.log_sigmoid(p_scores)
        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            reward = -tf.log(1. - tf.sigmoid(p_scores) + 1e-8)  # HAXX: avoids log(0)

        # Create Theano-like op that compute the synthetic reward
        self.compute_reward = TheanoFunction(inputs=[self.p_obs, self.p_acs],
                                             outputs=reward)

        # Summarize module information in logs
        log_module_info(logger, self.name, self.reward_nn)

    def get_reward(self, ob, ac):
        """Compute synthetic reward from a single observation
        The network is able to process observations and actions in minibatches, but the RL
        paradigm enforces the agent to see observations and perform actions in sequences,
        therefore seeing only one at a time.
        `ob` and `ac` are structured as np.array([a, b, c, ...]). Since the network work with
        minibatches, we have to construct a minibatch of size 1, e.g. by using `ob[None]`
        (do not use `[ob]`!) and `ac[None]` which are structured as np.array([[a, b, c, ...]]).
        The network outputs a single synthetic_reward per single observation-action pair in the
        joined input minibatch: np.array([[d]])
        Since a minibatch will later be sequentially construted out of the outputs, we extract
        the output from the returned minibatch of size 1. The extraction can be done by taking
        the first element (one or several times) with `.[0]` or by collapsing the returned array
        into one dimension with numpy's `flatten()` function:
            `synthetic_reward`: np.array([[d]]) -> np.array([d])
        Note that we only manipulate numpy arrays and not lists. We therefore do not need to
        extract the scalar `d` from `np.array([d])`, as for arithmetic operations a numpy array
        of size 1 is equivalent to a scalar (`np.array([1]) + np.array([2]) = np.array([3])` but
        `[1] + [2] = [1, 2]`).
        For safety reason, the scalar is still extracted from the singleton numpy array with
        `np.asscalar`.
        """
        synthetic_reward = self.compute_reward({self.p_obs: ob[None],
                                                self.p_acs: ac[None]})

        return np.asscalar(synthetic_reward.flatten())

    def sync_from_root(self):
        """Synchronize the optimizer across all mpi workers"""
        self.optimizer.sync_from_root(self.trainable_vars)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
