import time
import copy
import os.path as osp
from collections import namedtuple, deque, OrderedDict

from gym import spaces

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from imitation.helpers.tf_util import initialize
from imitation.helpers.tf_util import save_state, load_model, load_latest_checkpoint
from imitation.helpers import logger
from imitation.helpers.misc_util import zipsame
from imitation.helpers.console_util import timed_cm_wrapper, pretty_iter, pretty_elapsed, columnize
from imitation.helpers.mpi_moments import mpi_mean_like, mpi_mean_reduce, mpi_moments


def traj_segment_generator(env, mu, d, timesteps_per_batch, rew_aug_coeff, expert_dataset):

    def reset_with_demos_():
        """Get an observation from expert demos"""
        ob_, _ = expert_dataset.get_next_pair_batch(batch_size=1)
        ob_ = ob_[0]
        if hasattr(env, 'k'):
            # Environment wrapped with 'FrameStack' wrapper
            # Repeat the observation `k` times
            ob_ = np.repeat(ob_, env.k, axis=-1)
        return ob_

    t = 0
    ac = env.action_space.sample()
    done = True
    syn_rew = 0.0
    env_rew = 0.0
    mu.reset_noise()
    ob = env.reset()
    if expert_dataset is not None:
        # Override with an observation from expert demos
        ob = reset_with_demos_()
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    qs = np.zeros(timesteps_per_batch, 'float32')
    syn_rews = np.zeros(timesteps_per_batch, 'float32')
    env_rews = np.zeros(timesteps_per_batch, 'float32')
    dones = np.zeros(timesteps_per_batch, 'int32')

    while True:
        ac, q_pred = mu.predict(ob, apply_noise=True, compute_q=True)
        if t > 0 and t % timesteps_per_batch == 0:
            yield {"obs": obs,
                   "acs": acs,
                   "qs": qs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "dones": dones}
            _, q_pred = mu.predict(ob, apply_noise=True, compute_q=True)
        i = t % timesteps_per_batch
        obs[i] = ob
        acs[i] = ac
        qs[i] = q_pred
        dones[i] = done

        syn_rew = d.get_reward(ob, ac)
        new_ob, env_rew, done, _ = env.step(ac)
        syn_rews[i] = syn_rew
        env_rews[i] = env_rew

        stored_rew = syn_rew + (rew_aug_coeff * env_rew)
        mu.store_transition(ob, ac, stored_rew, new_ob, done)
        ob = copy.copy(new_ob)
        if done:
            mu.reset_noise()
            ob = env.reset()
            if expert_dataset is not None:
                # Override with an observation from expert demos
                ob = reset_with_demos_()
        t += 1


def traj_ep_generator(env, mu, d, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_syn_ret = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    qs = []
    syn_rews = []
    env_rews = []

    while True:
        ac, q = mu.predict(ob, apply_noise=False, compute_q=True)
        obs.append(ob)
        acs.append(ac)
        qs.append(q)
        if render:
            env.render()
        syn_rew = d.get_reward(ob, ac)
        new_ob, env_rew, done, _ = env.step(ac)
        syn_rews.append(syn_rew)
        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_syn_ret += syn_rew
        cur_ep_env_ret += env_rew
        ob = copy.copy(new_ob)
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            syn_rews = np.array(syn_rews)
            env_rews = np.array(env_rews)
            yield {"obs": obs,
                   "acs": acs,
                   "qs": qs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_syn_ret": cur_ep_syn_ret,
                   "ep_env_ret": cur_ep_env_ret}
            cur_ep_len = 0
            cur_ep_syn_ret = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            syn_rews = []
            env_rews = []
            mu.reset_noise()
            ob = env.reset()


def evaluate(env,
             discriminator_wrapper,
             sam_agent_wrapper,
             num_trajs,
             render,
             exact_model_path=None,
             model_ckpt_dir=None):
    """Evaluate a trained SAM agent"""

    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph
    # Create discriminator
    d = discriminator_wrapper('d')
    # Create a sam agent, taking `d` as input
    mu = sam_agent_wrapper('mu', d)
    # Create episode generator
    traj_gen = traj_ep_generator(env, mu, d, render)
    # Initialize and load the previously learned weights into the freshly re-built graph
    initialize()
    if exact_model_path is not None:
        load_model(exact_model_path)
        logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
    else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
        load_latest_checkpoint(model_ckpt_dir)
        logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))
    # Initialize the history data structures
    ep_lens = []
    ep_syn_rets = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = traj_gen.__next__()
        ep_len, ep_syn_ret, ep_env_ret = traj['ep_len'], traj['ep_syn_ret'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_syn_rets.append(ep_syn_ret)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_syn_ret_mean = np.mean(ep_syn_rets)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_syn_ret_mean", ep_syn_ret_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(comm,
          env,
          eval_env,
          discriminator_wrapper,
          sam_agent_wrapper,
          experiment_name,
          ckpt_dir,
          summary_dir,
          expert_dataset,
          reset_with_demos,
          add_demos_to_mem,
          save_frequency,
          rew_aug_coeff,
          param_noise_adaption_frequency,
          timesteps_per_batch,
          batch_size,
          window,
          g_steps,
          d_steps,
          training_steps_per_iter,
          eval_steps_per_iter,
          eval_frequency,
          render,
          max_iters,
          preload=False,
          exact_model_path=None,
          model_ckpt_dir=None):

    # If preload = True (1), only one should be specified (sum = 1)
    # If preload = False (0), none should be specified (sum = 0)
    assert sum([exact_model_path is not None, model_ckpt_dir is not None]) == preload

    rank = comm.Get_rank()

    # Create discriminator
    d = discriminator_wrapper('d')
    # Create sam agent, taking `d` as input
    mu = sam_agent_wrapper('mu', d)

    # Initialise variables
    initialize()

    if preload:
        # Load tensor values from previous run
        if exact_model_path is not None:
            load_model(exact_model_path)
            logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
        else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
            load_latest_checkpoint(model_ckpt_dir)
            logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))

    # Initialize target networks
    mu.initialize_target_nets()

    # Sync params of all processes with the params of the root process
    mu.sync_from_root()
    d.sync_from_root()

    if add_demos_to_mem:
        # Add demonstrations to memory
        mu.replay_buffer.add_demo_transitions_to_mem(expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm, logger)

    if rank == 0:
        # Create summary writer
        summary_writer = tf.summary.FileWriterCache.get(summary_dir)

    if isinstance(env.action_space, spaces.Box):
        # Logging the action scale + performing shape sanity check
        ac_scale = env.action_space.high
        logger.info("env-specific action scale (actor outputs in [-1, 1]): {}".format(ac_scale))
        _ac_ = env.action_space.sample()
        assert _ac_.shape == ac_scale.shape  # gym sanity check

    # Create segment generator for training the agent
    assert 0 <= rew_aug_coeff <= 1
    expert_dataset_ = expert_dataset if reset_with_demos else None
    seg_gen = traj_segment_generator(env, mu, d, timesteps_per_batch,
                                     rew_aug_coeff, expert_dataset_)
    if eval_env is not None:
        # Create episode generator for evaluating the agent
        eval_ep_gen = traj_ep_generator(eval_env, mu, d, render)

    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    keys = ['ac', 'q',
            'actor_grads', 'actor_losses',
            'critic_grads', 'critic_losses',
            'd_grads', 'd_losses']
    if mu.param_noise is not None:
        keys.extend(['pn_dist', 'pn_cur_std'])
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        keys.extend(['eval_ac', 'eval_q', 'eval_len', 'eval_syn_ret', 'eval_env_ret'])
    Deques = namedtuple('Deques', keys)
    deques = Deques(**{k: deque(maxlen=maxlen) for k in keys})

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        if hasattr(mu, 'eps_greedy_sched'):
            # Adapt the param noise threshold
            mu.adapt_eps_greedy(iters_so_far * timesteps_per_batch)

        # Verify that the processes are still in sync
        if iters_so_far > 0 and iters_so_far % 10 == 0:
            logger.info("checking param sync across processes")
            mu.actor_ops['optimizer'].check_synced(mu.actor.trainable_vars)
            mu.critic_ops['optimizer'].check_synced(mu.critic.trainable_vars)
            d.optimizer.check_synced(d.trainable_vars)
            logger.info("  sync check passed")

        # Save the model
        if rank == 0 and iters_so_far % save_frequency == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        # Make non-zero-rank workers wait for rank zero
        comm.Barrier()

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("sampling mini-batch"):
            seg = seg_gen.__next__()

        # Extend deques with collected experiential data
        deques.ac.extend(seg['acs'])
        deques.q.extend(seg['qs'])

        for training_step in range(training_steps_per_iter):

            logger.info("training [{}/{}]".format(training_step + 1, training_steps_per_iter))

            for d_step in range(d_steps):
                # Update discriminator

                if window is not None:
                    # Train d on recent pairs sampled from the replay buffer

                    # Collect recent pairs uniformly from the experience replay buffer
                    assert window > batch_size, "must have window > batch_size"
                    xp_batch = mu.replay_buffer.sample_recent(batch_size=batch_size,
                                                              window=window)
                    # Update discriminator w/ most recently collected samples & expert dataset
                    ob_pol, ac_pol = xp_batch['obs0'], xp_batch['acs']
                    # Collect expert data w/ identical batch size (GAN's equal mixture)
                    ob_exp, ac_exp = expert_dataset.get_next_pair_batch(batch_size=len(ob_pol))

                    if hasattr(env, 'k'):
                        # Environment wrapped with 'FrameStack' wrapper
                        # Repeat the observation `k` times
                        ob_exp = np.repeat(ob_exp, env.k, axis=-1)

                    assert len(ob_exp) == len(ob_pol), "length mismatch"

                    # Update running mean and std on states
                    if hasattr(d, "obs_rms"):
                        d.obs_rms.update(np.concatenate((ob_pol, ob_exp), axis=0), comm)

                    feeds = {d.p_obs: ob_pol,
                             d.p_acs: ac_pol,
                             d.e_obs: ob_exp,
                             d.e_acs: ac_exp}

                    # Compute losses and gradients
                    grads = d.get_grads(feeds)
                    losses = d.get_losses(feeds)

                    # Update discriminator
                    d.optimize(feeds)

                    # Store the losses and gradients in their respective deques
                    deques.d_grads.append(grads)
                    deques.d_losses.append(losses)

                # Train d on pairs sampled from the replay buffer

                # Collect pairs uniformly from the experience replay buffer
                sample_ = mu.replay_buffer.sample
                if hasattr(mu.replay_buffer, 'sample_uniform'):
                    # Executed iff prioritization is used, for which `sample` is overridden
                    sample_ = mu.replay_buffer.sample_uniform
                xp_batch_ = sample_(batch_size=batch_size)
                ob_pol_, ac_pol_ = xp_batch_['obs0'], xp_batch_['acs']
                # Collect expert data w/ identical batch size (GAN's equal mixture)
                ob_exp_, ac_exp_ = expert_dataset.get_next_pair_batch(batch_size=batch_size)

                if hasattr(env, 'k'):
                    # Environment wrapped with 'FrameStack' wrapper
                    # Repeat the observation `k` times
                    ob_exp_ = np.repeat(ob_exp_, env.k, axis=-1)

                assert len(ob_exp_) == len(ob_pol_), "length mismatch"

                # Update running mean and std on states
                if hasattr(d, "obs_rms"):
                    d.obs_rms.update(np.concatenate((ob_pol_, ob_exp_), axis=0), comm)

                feeds_ = {d.p_obs: ob_pol_,
                          d.p_acs: ac_pol_,
                          d.e_obs: ob_exp_,
                          d.e_acs: ac_exp_}

                # Compute losses and gradients
                d_grads = d.get_grads(feeds_)
                d_losses = d.get_losses(feeds_)

                # Update discriminator
                d.optimize(feeds_)

                # Store the losses and gradients in their respective deques
                deques.d_grads.append(d_grads)
                deques.d_losses.append(d_losses)

            if mu.param_noise is not None:
                if training_step % param_noise_adaption_frequency == 0:
                    # Adapt parameter noise
                    mu.adapt_param_noise(comm)
                    # Store the action-space distance between perturbed and non-perturbed actors
                    deques.pn_dist.append(mu.pn_dist)
                    # Store the new std resulting from the adaption
                    deques.pn_cur_std.append(mu.pn_cur_std)

            for g_step in range(g_steps):
                # Update agent w/ samples from replay buffer

                # Train the actor-critic architecture
                losses_and_grads = mu.train()
                # Unpack the returned training gradients and losses
                actor_grads = losses_and_grads['actor_grads']
                actor_losses = losses_and_grads['actor_losses']
                critic_grads = losses_and_grads['critic_grads']
                critic_losses = losses_and_grads['critic_losses']
                # Store the losses and gradients in their respective deques
                deques.actor_grads.append(actor_grads)
                deques.actor_losses.append(actor_losses)
                deques.critic_grads.append(critic_grads)
                deques.critic_losses.append(critic_losses)
                # Update the target networks
                mu.update_target_net()

        if eval_env is not None:  # `eval_env` not None iff rank = 0
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:
                for eval_step in range(eval_steps_per_iter):
                    logger.info("evaluating [{}/{}]".format(eval_step + 1, eval_steps_per_iter))

                    # Sample an episode w/ non-perturbed actor w/o storing anything
                    eval_ep = eval_ep_gen.__next__()

                    # Aggregate data collected during the evaluation to the buffers
                    deques.eval_ac.extend(eval_ep['acs'])
                    deques.eval_q.extend(eval_ep['qs'])
                    deques.eval_len.append(eval_ep['ep_len'])
                    deques.eval_syn_ret.append(eval_ep['ep_syn_ret'])
                    deques.eval_env_ret.append(eval_ep['ep_env_ret'])

        # Make non-zero-rank workers wait for rank zero
        comm.Barrier()

        # Log statistics

        stats = OrderedDict()
        mpi_stats = OrderedDict()

        # Add min, max and mean of the components of the average action
        ac_np_mean = np.mean(deques.ac, axis=0)  # vector
        ac_mpi_mean, _, _ = mpi_moments(list(deques.ac), comm)  # vector
        stats.update({'min_ac_comp': np.amin(ac_np_mean)})
        stats.update({'max_ac_comp': np.amax(ac_np_mean)})
        stats.update({'mean_ac_comp': np.mean(ac_np_mean)})
        mpi_stats.update({'min_ac_comp': np.amin(ac_mpi_mean)})
        mpi_stats.update({'max_ac_comp': np.amax(ac_mpi_mean)})
        mpi_stats.update({'mean_ac_comp': np.mean(ac_mpi_mean)})

        # Add Q values mean and std
        q_mpi_mean, q_mpi_std, _ = mpi_moments(list(deques.q), comm)  # scalars
        stats.update({'q_value': np.mean(deques.q)})
        stats.update({'q_deviation': np.std(deques.q)})
        mpi_stats.update({'q_value': q_mpi_mean})
        mpi_stats.update({'q_deviation': q_mpi_std})

        # Add gradient norms
        stats.update({'actor_gradnorm': norm(np.mean(deques.actor_grads, axis=0))})
        stats.update({'critic_gradnorm': norm(np.mean(deques.critic_grads, axis=0))})
        stats.update({'d_gradnorm': norm(np.mean(deques.d_grads, axis=0))})

        mpi_mean_actor_grad = mpi_mean_reduce(list(deques.actor_grads), comm)
        mpi_mean_critic_grad = mpi_mean_reduce(list(deques.critic_grads), comm)
        mpi_mean_d_grad = mpi_mean_reduce(list(deques.d_grads), comm)
        mpi_stats.update({'actor_gradnorm': norm(mpi_mean_actor_grad)})
        mpi_stats.update({'critic_gradnorm': norm(mpi_mean_critic_grad)})
        mpi_stats.update({'d_gradnorm': norm(mpi_mean_d_grad)})

        # Add replay buffer num entries
        stats.update({'mem_num_entries': np.mean(mu.replay_buffer.num_entries)})
        mpi_stats.update({'mem_num_entries': mpi_mean_like(mu.replay_buffer.num_entries, comm)})

        # Log dictionary content
        col_names = ['name', 'value']
        col_widths = [24, 16]  # no hp
        logger.info("logging misc training stats (local) (log)")
        logger.info(columnize(col_names, stats.items(), col_widths))
        logger.info("logging misc training stats (global) (log)")
        logger.info(columnize(col_names, mpi_stats.items(), col_widths))

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:
                # Use the logger object to log the eval stats (will appear in `progress{}.csv`)
                logger.info("logging misc eval stats (log + csv)")
                # Add min, max and mean of the components of the average action
                ac_np_mean = np.mean(deques.eval_ac, axis=0)  # vector
                logger.record_tabular('min_ac_comp', np.amin(ac_np_mean))
                logger.record_tabular('max_ac_comp', np.amax(ac_np_mean))
                logger.record_tabular('mean_ac_comp', np.mean(ac_np_mean))
                # Add Q values mean and std
                logger.record_tabular('q_value', np.mean(deques.eval_q))
                logger.record_tabular('q_deviation', np.std(deques.eval_q))
                # Add episodic stats
                logger.record_tabular('ep_len', np.mean(deques.eval_len))
                logger.record_tabular('ep_syn_ret', np.mean(deques.eval_syn_ret))
                logger.record_tabular('ep_env_ret', np.mean(deques.eval_env_ret))
                logger.dump_tabular()

        # Mark the end of the iter in the logs
        logger.info('')

        iters_so_far += 1

        if rank == 0:

            # Add summaries
            summary = tf.summary.Summary()
            tab = 'sam'

            if iters_so_far % eval_frequency == 0:
                summary.value.add(tag="{}/{}".format(tab, 'mean_ep_len'),
                                  simple_value=np.mean(deques.eval_len))
                summary.value.add(tag="{}/{}".format(tab, 'mean_ep_syn_ret'),
                                  simple_value=np.mean(deques.eval_syn_ret))
                summary.value.add(tag="{}/{}".format(tab, 'mean_ep_env_ret'),
                                  simple_value=np.mean(deques.eval_env_ret))
            if mu.param_noise is not None:
                # Add param noise stats to summary
                summary.value.add(tag="{}/{}".format(tab, 'mean_pn_cur_std'),
                                  simple_value=np.mean(deques.pn_cur_std))
                summary.value.add(tag="{}/{}".format(tab, 'mean_pn_dist'),
                                  simple_value=np.mean(deques.pn_dist))
            # Losses
            for name, loss in zipsame(mu.actor_ops['names'],
                                      np.mean(deques.actor_losses, axis=0)):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)
            for name, loss in zipsame(mu.critic_ops['names'],
                                      np.mean(deques.critic_losses, axis=0)):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)
            for name, loss in zipsame(d.names, np.mean(deques.d_losses, axis=0)):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)

            summary_writer.add_summary(summary, iters_so_far)
