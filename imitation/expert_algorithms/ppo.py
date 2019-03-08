import time
import os.path as osp
from collections import deque, OrderedDict

import tensorflow as tf
import numpy as np

from imitation.helpers.tf_util import (initialize, get_placeholder_cached,
                                       save_state, TheanoFunction)
from imitation.helpers import logger
from imitation.helpers.feeder import Feeder
from imitation.helpers.misc_util import zipsame, flatten_lists, prettify_time
from imitation.helpers.math_util import explained_variance, augment_segment_gae_stats
from imitation.helpers.console_util import columnize, timed_cm_wrapper, pretty_iter, pretty_elapsed
from imitation.helpers.mpi_adam import MpiAdamOptimizer
from imitation.helpers.mpi_moments import mpi_mean_like, mpi_mean_reduce
from imitation.expert_algorithms.xpo_util import traj_segment_generator


def learn(comm,
          env,
          xpo_agent_wrapper,
          sample_or_mode,
          gamma,
          save_frequency,
          ckpt_dir,
          summary_dir,
          timesteps_per_batch,
          batch_size,
          optim_epochs_per_iter,
          lr,
          experiment_name,
          ent_reg_scale,
          clipping_eps,
          gae_lambda,
          schedule,
          max_iters):

    rank = comm.Get_rank()

    # Create policies
    pi = xpo_agent_wrapper('pi')
    old_pi = xpo_agent_wrapper('old_pi')

    # Create and retrieve already-existing placeholders
    ob = get_placeholder_cached(name='ob')
    ac = pi.pd_type.sample_placeholder([None])
    adv = tf.placeholder(name='adv', dtype=tf.float32, shape=[None])
    ret = tf.placeholder(name='ret', dtype=tf.float32, shape=[None])
    # Adaptive learning rate multiplier, updated with schedule
    lr_mult = tf.placeholder(name='lr_mult', dtype=tf.float32, shape=[])

    # Build graphs
    kl_mean = tf.reduce_mean(old_pi.pd_pred.kl(pi.pd_pred))
    ent_mean = tf.reduce_mean(pi.pd_pred.entropy())
    ent_pen = (-ent_reg_scale) * ent_mean
    vf_err = tf.reduce_mean(tf.square(pi.v_pred - ret))  # MC error
    # The surrogate objective is defined as: advantage * pnew / pold
    ratio = tf.exp(pi.pd_pred.logp(ac) - old_pi.pd_pred.logp(ac))  # IS
    surr_gain = ratio * adv  # surrogate objective (CPI)
    # Annealed clipping parameter epsilon
    clipping_eps = clipping_eps * lr_mult
    surr_gain_w_clipping = tf.clip_by_value(ratio,
                                            1.0 - clipping_eps,
                                            1.0 + clipping_eps) * adv
    # PPO's pessimistic surrogate (L^CLIP in paper)
    surr_loss = -tf.reduce_mean(tf.minimum(surr_gain, surr_gain_w_clipping))
    # Assemble losses (including the value function loss)
    loss = surr_loss + ent_pen + vf_err

    losses = OrderedDict()

    # Add losses
    losses.update({'pol_kl_mean': kl_mean,
                   'pol_ent_mean': ent_mean,
                   'pol_ent_pen': ent_pen,
                   'pol_surr_loss': surr_loss,
                   'pol_vf_err': vf_err,
                   'pol_total_loss': loss})

    # Make the current `pi` become the next `old_pi`
    zipped = zipsame(old_pi.vars, pi.vars)
    updates_op = []
    for k, v in zipped:
        # Populate list of assignment operations
        logger.info("assignment: {} <- {}".format(k, v))
        assign_op = tf.assign(k, v)
        updates_op.append(assign_op)
    assert len(updates_op) == len(pi.vars)

    # Create mpi adam optimizer
    optimizer = MpiAdamOptimizer(comm=comm,
                                 clip_norm=5.0,
                                 learning_rate=lr * lr_mult,
                                 name='adam')
    optimize = optimizer.minimize(loss=loss, var_list=pi.trainable_vars)

    # Create callable objects
    assign_old_eq_new = TheanoFunction(inputs=[], outputs=updates_op)
    compute_losses = TheanoFunction(inputs=[ob, ac, adv, ret, lr_mult],
                                    outputs=list(losses.values()))
    optimize = TheanoFunction(inputs=[ob, ac, adv, ret, lr_mult],
                              outputs=optimize)

    # Initialise variables
    initialize()

    # Sync params of all processes with the params of the root process
    optimizer.sync_from_root(pi.trainable_vars)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm, logger)

    if rank == 0:
        # Create summary writer
        summary_writer = tf.summary.FileWriterCache.get(summary_dir)

    seg_gen = traj_segment_generator(env, pi, timesteps_per_batch, sample_or_mode)

    eps_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for recent stats aggregation
    maxlen = 100
    len_buffer = deque(maxlen=maxlen)
    env_ret_buffer = deque(maxlen=maxlen)
    pol_losses_buffer = deque(maxlen=maxlen)

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        # Verify that the processes are still in sync
        if iters_so_far > 0 and iters_so_far % 10 == 0:
            optimizer.check_synced(pi.trainable_vars)
            logger.info("params still in sync across processes")

        # Manage lr multiplier schedule
        if schedule == 'constant':
            curr_lr_mult = 1.0
        elif schedule == 'linear':
            curr_lr_mult = max(1.0 - float(iters_so_far * timesteps_per_batch) /
                               max_iters * timesteps_per_batch, 0)
        else:
            raise NotImplementedError

        # Save the model
        if rank == 0 and iters_so_far % save_frequency == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        with timed("sampling mini-batch"):
            seg = seg_gen.__next__()

        augment_segment_gae_stats(seg, gamma, gae_lambda, rew_key="env_rews")

        # Standardize advantage function estimate
        seg['advs'] = (seg['advs'] - seg['advs'].mean()) / (seg['advs'].std() + 1e-8)

        # Update running mean and std
        if hasattr(pi, 'obs_rms'):
            with timed("normalizing obs via rms"):
                pi.obs_rms.update(seg['obs'], comm)

        assign_old_eq_new({})

        # Create Feeder object to iterate over (ob, ac, adv, td_lam_ret) tuples
        data_map = {'obs': seg['obs'],
                    'acs': seg['acs'],
                    'advs': seg['advs'],
                    'td_lam_rets': seg['td_lam_rets']}
        feeder = Feeder(data_map=data_map, enable_shuffle=True)

        # Update policy and state-value function
        with timed("updating policy and value function"):
            for _ in range(optim_epochs_per_iter):
                for minibatch in feeder.get_feed(batch_size=batch_size):

                    feeds = {ob: minibatch['obs'],
                             ac: minibatch['acs'],
                             adv: minibatch['advs'],
                             ret: minibatch['td_lam_rets'],
                             lr_mult: curr_lr_mult}

                    # Compute losses
                    pol_losses = compute_losses(feeds)

                    # Update the policy and value function
                    optimize(feeds)

                    # Store the losses
                    pol_losses_buffer.append(pol_losses)

        # Log policy update statistics
        logger.info("logging training losses (log)")
        pol_losses_np_mean = np.mean(pol_losses_buffer, axis=0)
        pol_losses_mpi_mean = mpi_mean_reduce(pol_losses_buffer, comm, axis=0)
        zipped_pol_losses = zipsame(list(losses.keys()), pol_losses_np_mean, pol_losses_mpi_mean)
        logger.info(columnize(names=['name', 'local', 'global'],
                              tuples=zipped_pol_losses,
                              widths=[20, 16, 16]))

        # Log statistics

        logger.info("logging misc training stats (log + csv)")
        # Gather statistics across workers
        local_lens_rets = (seg['ep_lens'], seg['ep_env_rets'])
        gathered_lens_rets = comm.allgather(local_lens_rets)
        lens, env_rets = map(flatten_lists, zip(*gathered_lens_rets))
        # Extend the deques of recorded statistics
        len_buffer.extend(lens)
        env_ret_buffer.extend(env_rets)
        ep_len_mpi_mean = np.mean(len_buffer)
        ep_env_ret_mpi_mean = np.mean(env_ret_buffer)
        logger.record_tabular('ep_len_mpi_mean', ep_len_mpi_mean)
        logger.record_tabular('ep_env_ret_mpi_mean', ep_env_ret_mpi_mean)
        eps_this_iter = len(lens)
        timesteps_this_iter = sum(lens)
        eps_so_far += eps_this_iter
        timesteps_so_far += timesteps_this_iter
        eps_this_iter_mpi_mean = mpi_mean_like(eps_this_iter, comm)
        timesteps_this_iter_mpi_mean = mpi_mean_like(timesteps_this_iter, comm)
        eps_so_far_mpi_mean = mpi_mean_like(eps_so_far, comm)
        timesteps_so_far_mpi_mean = mpi_mean_like(timesteps_so_far, comm)
        logger.record_tabular('eps_this_iter_mpi_mean', eps_this_iter_mpi_mean)
        logger.record_tabular('timesteps_this_iter_mpi_mean', timesteps_this_iter_mpi_mean)
        logger.record_tabular('eps_so_far_mpi_mean', eps_so_far_mpi_mean)
        logger.record_tabular('timesteps_so_far_mpi_mean', timesteps_so_far_mpi_mean)
        logger.record_tabular('elapsed time', prettify_time(time.time() - tstart))  # no mpi mean
        logger.record_tabular('ev_td_lam_before', explained_variance(seg['vs'],
                                                                     seg['td_lam_rets']))
        iters_so_far += 1

        if rank == 0:
            logger.dump_tabular()

        if rank == 0:
            # Add summaries
            summary = tf.summary.Summary()
            tab = 'ppo'
            # Episode stats
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_len'),
                              simple_value=ep_len_mpi_mean)
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_env_ret'),
                              simple_value=ep_env_ret_mpi_mean)
            # Losses
            for name, loss in zipsame(list(losses.keys()), pol_losses_mpi_mean):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)

            summary_writer.add_summary(summary, iters_so_far)
