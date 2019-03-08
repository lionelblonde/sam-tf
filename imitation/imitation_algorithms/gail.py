import time
import copy
import os.path as osp
from collections import deque, OrderedDict

import tensorflow as tf
import numpy as np

from imitation.helpers.tf_util import (initialize, get_placeholder_cached,
                                       save_state, load_model, load_latest_checkpoint,
                                       TheanoFunction, GetFlat, SetFromFlat, flatgrad)
from imitation.helpers import logger
from imitation.helpers.feeder import Feeder
from imitation.helpers.misc_util import zipsame, flatten_lists, prettify_time, intprod
from imitation.helpers.math_util import (explained_variance, conjugate_gradient,
                                         augment_segment_gae_stats)
from imitation.helpers.console_util import columnize, timed_cm_wrapper, pretty_iter, pretty_elapsed
from imitation.helpers.mpi_adam import MpiAdamOptimizer
from imitation.helpers.mpi_moments import mpi_mean_like, mpi_mean_reduce


def traj_segment_generator(env, pi, d, timesteps_per_batch, sample_or_mode):
    t = 0
    ac = env.action_space.sample()
    done = True
    syn_rew = 0.0
    env_rew = 0.0
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_syn_ret = 0
    cur_ep_env_ret = 0
    ep_lens = []
    ep_syn_rets = []
    ep_env_rets = []
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    vs = np.zeros(timesteps_per_batch, 'float32')
    syn_rews = np.zeros(timesteps_per_batch, 'float32')
    env_rews = np.zeros(timesteps_per_batch, 'float32')
    dones = np.zeros(timesteps_per_batch, 'int32')

    while True:
        ac, v_pred = pi.predict(sample_or_mode, ob)
        if t > 0 and t % timesteps_per_batch == 0:
            yield {"obs": obs,
                   "acs": acs,
                   "vs": vs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "dones": dones,
                   "next_v": v_pred * (1 - done),
                   "ep_lens": ep_lens,
                   "ep_syn_rets": ep_syn_rets,
                   "ep_env_rets": ep_env_rets}
            _, v_pred = pi.predict(sample_or_mode, ob)
            ep_lens = []
            ep_syn_rets = []
            ep_env_rets = []
        i = t % timesteps_per_batch
        obs[i] = ob
        acs[i] = ac
        vs[i] = v_pred
        dones[i] = done
        syn_rew = d.get_reward(ob, ac)
        ob, env_rew, done, _ = env.step(ac)
        syn_rews[i] = syn_rew
        env_rews[i] = env_rew
        cur_ep_len += 1
        cur_ep_syn_ret += syn_rew
        cur_ep_env_ret += env_rew
        if done:
            ep_lens.append(cur_ep_len)
            ep_syn_rets.append(cur_ep_syn_ret)
            ep_env_rets.append(cur_ep_env_ret)
            cur_ep_len = 0
            cur_ep_syn_ret = 0
            cur_ep_env_ret = 0
            ob = env.reset()
        t += 1


def traj_ep_generator(env, pi, d, sample_or_mode, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_syn_ret = 0
    cur_ep_env_ret = 0
    obs0 = []
    acs = []
    syn_rews = []
    env_rews = []
    dones1 = []
    obs1 = []

    while True:
        ac, _ = pi.predict(sample_or_mode, ob)
        obs0.append(ob)
        acs.append(ac)
        if render:
            env.render()
        syn_rew = d.get_reward(ob, ac)
        new_ob, env_rew, done, _ = env.step(ac)
        obs1.append(new_ob)
        ob = copy.copy(new_ob)
        syn_rews.append(syn_rew)
        env_rews.append(env_rew)
        dones1.append(done)
        cur_ep_len += 1
        cur_ep_syn_ret += syn_rew
        cur_ep_env_ret += env_rew
        if done:
            obs0 = np.array(obs0)
            acs = np.array(acs)
            syn_rews = np.array(syn_rews)
            env_rews = np.array(env_rews)
            dones1 = np.array(dones1)
            obs1 = np.array(obs1)
            yield {"obs0": obs0,
                   "acs": acs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "dones1": dones1,
                   "obs1": obs1,
                   "ep_len": cur_ep_len,
                   "ep_syn_ret": cur_ep_syn_ret,
                   "ep_env_ret": cur_ep_env_ret}
            ob = env.reset()
            cur_ep_len = 0
            cur_ep_syn_ret = 0
            cur_ep_env_ret = 0
            obs0 = []
            acs = []
            syn_rews = []
            env_rews = []
            dones1 = []
            obs1 = []


def evaluate(env, trpo_agent_wrapper, discriminator_wrapper, num_trajs, sample_or_mode, render,
             exact_model_path=None, model_ckpt_dir=None):
    """Evaluate a trained GAIL agent"""

    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph to gain evaluation access to a learned and saved policy
    pi = trpo_agent_wrapper('pi')
    d = discriminator_wrapper('d')
    traj_gen = traj_ep_generator(env=env, pi=pi, d=d, sample_or_mode=sample_or_mode, render=render)
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
    sample_or_mode = 'sample' if sample_or_mode else 'mode'
    logger.info("action picking: {}".format(sample_or_mode))
    ep_len_mean = np.mean(ep_lens)
    ep_syn_ret_mean = np.mean(ep_syn_rets)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_syn_ret_mean", ep_syn_ret_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(comm,
          env,
          trpo_agent_wrapper,
          discriminator_wrapper,
          sample_or_mode,
          gamma,
          max_kl,
          expert_dataset,
          g_steps,
          d_steps,
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
    pi = trpo_agent_wrapper('pi')
    old_pi = trpo_agent_wrapper('old_pi')
    # Create discriminator
    d = discriminator_wrapper('d')

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
                                    clip_norm=pi.hps.clip_norm,
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

    d.sync_from_root()

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm, logger)

    if rank == 0:
        # Create summary writer
        summary_writer = tf.summary.FileWriterCache.get(summary_dir)

    seg_gen = traj_segment_generator(env, pi, d, timesteps_per_batch, sample_or_mode)

    eps_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for recent stats aggregation
    maxlen = 100
    len_buffer = deque(maxlen=maxlen)
    syn_ret_buffer = deque(maxlen=maxlen)
    env_ret_buffer = deque(maxlen=maxlen)
    pol_losses_buffer = deque(maxlen=maxlen)
    d_losses_buffer = deque(maxlen=maxlen)

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        # Verify that the processes are still in sync
        if iters_so_far > 0 and iters_so_far % 10 == 0:
            logger.info("checking param sync across processes")
            vf_optimizer.check_synced(pi.vf_trainable_vars)
            d.optimizer.check_synced(d.trainable_vars)
            logger.info("  sync check passed")

        # Save the model
        if rank == 0 and iters_so_far % save_frequency == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        for g_step in range(g_steps):
            logger.info("updating g [{}/{}]".format(g_step + 1, g_steps))

            with timed("sampling mini-batch"):
                seg = seg_gen.__next__()

            augment_segment_gae_stats(seg, gamma, gae_lambda, rew_key="syn_rews")

            # Standardize advantage estimate
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
                surr_before = loss_before[4]
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
            g_feeder = Feeder(data_map={'obs': seg['obs'], 'td_lam_rets': seg['td_lam_rets']},
                              enable_shuffle=True)

            # Update state-value function
            with timed("updating value function"):
                for _ in range(vf_iters):
                    for minibatch in g_feeder.get_feed(batch_size=batch_size):
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

        for d_step in range(d_steps):
            # Update discriminator
            logger.info("updating d [{}/{}]".format(d_step + 1, d_steps))

            # Create Feeder object to iterate over (ob, ret) pairs
            d_feeder = Feeder(data_map={'obs': seg['obs'], 'acs': seg['acs']},
                              enable_shuffle=True)

            for minibatch in d_feeder.get_feed(batch_size=batch_size):
                ob_pol, ac_pol = minibatch['obs'], minibatch['acs']
                # Collect as many data demonstrations (pairs) as experienced (GAN's equal mixture)
                ob_exp, ac_exp = expert_dataset.get_next_pair_batch(batch_size=len(ob_pol))

                if hasattr(env, 'k'):
                    # Environment wrapped with 'FrameStack' wrapper
                    # Repeat the observation `k` times
                    ob_exp = np.repeat(ob_exp, env.k, axis=-1)

                if "NoFrameskip" in env.spec.id:
                    # Expand the dimension for Atari
                    ac_pol = np.expand_dims(ac_pol, axis=-1)

                assert len(ob_exp) == len(ob_pol), "length mismatch"

                # Update running mean and std on states
                if hasattr(d, "obs_rms"):
                    d.obs_rms.update(np.concatenate((ob_pol, ob_exp), axis=0), comm)

                feeds = {d.p_obs: ob_pol,
                         d.p_acs: ac_pol,
                         d.e_obs: ob_exp,
                         d.e_acs: ac_exp}

                # Compute losses
                d_losses = d.get_losses(feeds)

                # Update discriminator
                d.optimize(feeds)

                # Store the losses
                d_losses_buffer.append(d_losses)

        # Assemble discriminator losses
        logger.info("logging d training losses (log)")
        d_losses_np_mean = np.mean(d_losses_buffer, axis=0)
        d_losses_mpi_mean = mpi_mean_reduce(d_losses_buffer, comm, axis=0)
        zipped_d_losses = zipsame(d.names, d_losses_np_mean, d_losses_mpi_mean)
        logger.info(columnize(names=['name', 'local', 'global'],
                              tuples=zipped_d_losses,
                              widths=[20, 16, 16]))

        # Log statistics

        logger.info("logging misc training stats (log + csv)")
        # Gather statistics across workers
        local_lens_rets = (seg['ep_lens'], seg['ep_syn_rets'], seg['ep_env_rets'])
        gathered_lens_rets = comm.allgather(local_lens_rets)
        lens, syn_rets, env_rets = map(flatten_lists, zip(*gathered_lens_rets))
        # Extend the deques of recorded statistics
        len_buffer.extend(lens)
        syn_ret_buffer.extend(syn_rets)
        env_ret_buffer.extend(env_rets)
        ep_len_mpi_mean = np.mean(len_buffer)
        ep_syn_ret_mpi_mean = np.mean(syn_ret_buffer)
        ep_env_ret_mpi_mean = np.mean(env_ret_buffer)
        logger.record_tabular('ep_len_mpi_mean', ep_len_mpi_mean)
        logger.record_tabular('ep_syn_ret_mpi_mean', ep_syn_ret_mpi_mean)
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
        logger.record_tabular("ev_td_lam_before", explained_variance(seg['vs'],
                                                                     seg['td_lam_rets']))
        iters_so_far += 1

        if rank == 0:
            logger.dump_tabular()

        if rank == 0:
            # Add summaries
            summary = tf.summary.Summary()
            tab = 'gail'
            # Episode stats
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_len'),
                              simple_value=ep_len_mpi_mean)
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_syn_ret'),
                              simple_value=ep_syn_ret_mpi_mean)
            summary.value.add(tag="{}/{}".format(tab, 'mean_ep_env_ret'),
                              simple_value=ep_env_ret_mpi_mean)
            # Losses
            for name, loss in zipsame(list(losses.keys()), pol_losses_mpi_mean):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)
            for name, loss in zipsame(d.names, d_losses_mpi_mean):
                summary.value.add(tag="{}/{}".format(tab, name), simple_value=loss)

            summary_writer.add_summary(summary, iters_so_far)
