import time
import os.path as osp
from collections import deque, OrderedDict

import tensorflow as tf
import numpy as np

from imitation.helpers.tf_util import (initialize, get_placeholder_cached, save_state,
                                       TheanoFunction, GetFlat, SetFromFlat, flatgrad)
from imitation.helpers import logger
from imitation.helpers.feeder import Feeder
from imitation.helpers.misc_util import zipsame, flatten_lists, prettify_time, intprod
from imitation.helpers.math_util import (explained_variance, conjugate_gradient,
                                         augment_segment_gae_stats)
from imitation.helpers.console_util import columnize, timed_cm_wrapper, pretty_iter, pretty_elapsed
from imitation.helpers.mpi_adam import MpiAdamOptimizer
from imitation.helpers.mpi_moments import mpi_mean_like, mpi_mean_reduce
from imitation.expert_algorithms.xpo_util import traj_segment_generator


def learn(comm,
          env,
          xpo_agent_wrapper,
          sample_or_mode,
          gamma,
          max_kl,
          save_frequency,
          ckpt_dir,
          summary_dir,
          timesteps_per_batch,
          batch_size,
          experiment_name,
          ent_reg_scale,
          gae_lambda,
          cg_iters,
          cg_damping,
          vf_iters,
          vf_lr,
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
    flat_tangent = tf.placeholder(name='flat_tan', dtype=tf.float32, shape=[None])

    # Build graphs
    kl_mean = tf.reduce_mean(old_pi.pd_pred.kl(pi.pd_pred))
    ent_mean = tf.reduce_mean(pi.pd_pred.entropy())
    ent_bonus = ent_reg_scale * ent_mean
    vf_err = tf.reduce_mean(tf.square(pi.v_pred - ret))  # MC error
    # The surrogate objective is defined as: advantage * pnew / pold
    ratio = tf.exp(pi.pd_pred.logp(ac) - old_pi.pd_pred.logp(ac))  # IS
    surr_gain = tf.reduce_mean(ratio * adv)  # surrogate objective (CPI)
    # Add entropy bonus
    optim_gain = surr_gain + ent_bonus

    losses = OrderedDict()

    # Add losses
    losses.update({'pol_kl_mean': kl_mean,
                   'pol_ent_mean': ent_mean,
                   'pol_ent_bonus': ent_bonus,
                   'pol_surr_gain': surr_gain,
                   'pol_optim_gain': optim_gain,
                   'pol_vf_err': vf_err})

    # Build natural gradient material
    get_flat = GetFlat(pi.pol_trainable_vars)
    set_from_flat = SetFromFlat(pi.pol_trainable_vars)
    kl_grads = tf.gradients(kl_mean, pi.pol_trainable_vars)
    shapes = [var.get_shape().as_list() for var in pi.pol_trainable_vars]
    start = 0
    tangents = []
    for shape in shapes:
        sz = intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    # Create the gradient vector product
    gvp = tf.add_n([tf.reduce_sum(g * tangent)
                    for (g, tangent) in zipsame(kl_grads, tangents)])
    # Create the Fisher vector product
    fvp = flatgrad(gvp, pi.pol_trainable_vars)

    # Make the current `pi` become the next `old_pi`
    zipped = zipsame(old_pi.vars, pi.vars)
    updates_op = []
    for k, v in zipped:
        # Populate list of assignment operations
        logger.info("assignment: {} <- {}".format(k, v))
        assign_op = tf.assign(k, v)
        updates_op.append(assign_op)
    assert len(updates_op) == len(pi.vars)

    # Create mpi adam optimizer for the value function
    vf_optimizer = MpiAdamOptimizer(comm=comm,
                                    clip_norm=5.0,
                                    learning_rate=vf_lr,
                                    name='vf_adam')
    optimize_vf = vf_optimizer.minimize(loss=vf_err, var_list=pi.vf_trainable_vars)

    # Create gradients
    grads = flatgrad(optim_gain, pi.pol_trainable_vars)

    # Create callable objects
    assign_old_eq_new = TheanoFunction(inputs=[], outputs=updates_op)
    compute_losses = TheanoFunction(inputs=[ob, ac, adv, ret], outputs=list(losses.values()))
    compute_losses_grads = TheanoFunction(inputs=[ob, ac, adv, ret],
                                          outputs=list(losses.values()) + [grads])
    compute_fvp = TheanoFunction(inputs=[flat_tangent, ob, ac, adv], outputs=fvp)
    optimize_vf = TheanoFunction(inputs=[ob, ret], outputs=optimize_vf)

    # Initialise variables
    initialize()

    # Sync params of all processes with the params of the root process
    theta_init = get_flat()
    comm.Bcast(theta_init, root=0)
    set_from_flat(theta_init)

    vf_optimizer.sync_from_root(pi.vf_trainable_vars)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm, logger)

    if rank == 0:
        # Create summary writer
        summary_writer = tf.summary.FileWriterCache.get(summary_dir)

    # Create segment generator
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
            vf_optimizer.check_synced(pi.vf_trainable_vars)
            logger.info("vf params still in sync across processes")

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

        def fisher_vector_product(p):
            computed_fvp = compute_fvp({flat_tangent: p,
                                        ob: seg['obs'],
                                        ac: seg['acs'],
                                        adv: seg['advs']})
            return mpi_mean_like(computed_fvp, comm) + cg_damping * p

        assign_old_eq_new({})

        # Compute gradients
        with timed("computing gradients"):
            *loss_before, g = compute_losses_grads({ob: seg['obs'],
                                                    ac: seg['acs'],
                                                    adv: seg['advs'],
                                                    ret: seg['td_lam_rets']})

        loss_before = mpi_mean_like(loss_before, comm)

        g = mpi_mean_like(g, comm)

        if np.allclose(g, 0):
            logger.info("got zero gradient -> not updating")
        else:
            with timed("performing conjugate gradient procedure"):
                step_direction = conjugate_gradient(f_Ax=fisher_vector_product,
                                                    b=g,
                                                    cg_iters=cg_iters,
                                                    verbose=(rank == 0))
            assert np.isfinite(step_direction).all()
            shs = 0.5 * step_direction.dot(fisher_vector_product(step_direction))
            # shs is (1/2)*s^T*A*s in the paper
            lm = np.sqrt(shs / max_kl)
            # lm is 1/beta in the paper (max_kl is user-specified delta)
            full_step = step_direction / lm  # beta*s
            expected_improve = g.dot(full_step)  # project s on g
            surr_before = loss_before[4]  # 5-th in loss list
            step_size = 1.0
            theta_before = get_flat()

            with timed("updating policy"):
                for _ in range(10):  # trying (10 times max) until the stepsize is OK
                    # Update the policy parameters
                    theta_new = theta_before + full_step * step_size
                    set_from_flat(theta_new)
                    pol_losses = compute_losses({ob: seg['obs'],
                                                 ac: seg['acs'],
                                                 adv: seg['advs'],
                                                 ret: seg['td_lam_rets']})

                    pol_losses_buffer.append(pol_losses)

                    pol_losses_mpi_mean = mpi_mean_like(pol_losses, comm)
                    surr = pol_losses_mpi_mean[4]
                    kl = pol_losses_mpi_mean[0]
                    actual_improve = surr - surr_before
                    logger.info("  expected: {:.3f} | actual: {:.3f}".format(expected_improve,
                                                                             actual_improve))
                    if not np.isfinite(pol_losses_mpi_mean).all():
                        logger.info("  got non-finite value of losses :(")
                    elif kl > max_kl * 1.5:
                        logger.info("  violated KL constraint -> shrinking step.")
                    elif actual_improve < 0:
                        logger.info("  surrogate didn't improve -> shrinking step.")
                    else:
                        logger.info("  stepsize fine :)")
                        break
                    step_size *= 0.5  # backtracking when the step size is deemed inappropriate
                else:
                    logger.info("  couldn't compute a good step")
                    set_from_flat(theta_before)

        # Create Feeder object to iterate over (ob, ret) pairs
        feeder = Feeder(data_map={'obs': seg['obs'], 'td_lam_rets': seg['td_lam_rets']},
                        enable_shuffle=True)

        # Update state-value function
        with timed("updating value function"):
            for _ in range(vf_iters):
                for minibatch in feeder.get_feed(batch_size=batch_size):
                    optimize_vf({ob: minibatch['obs'],
                                 ret: minibatch['td_lam_rets']})

        # Log policy update statistics
        logger.info("logging pol training losses (log)")
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
            tab = 'trpo'
            # Episode stats
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_len'),
                              simple_value=ep_len_mpi_mean)
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_env_ret'),
                              simple_value=ep_env_ret_mpi_mean)
            # Losses
            for name, loss in zipsame(list(losses.keys()), pol_losses_mpi_mean):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)

            summary_writer.add_summary(summary, iters_so_far)
