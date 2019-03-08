import copy
import os.path as osp

import numpy as np

from imitation.helpers.tf_util import initialize, load_model, load_latest_checkpoint
from imitation.helpers import logger


def traj_segment_generator(env, pi, timesteps_per_batch, sample_or_mode):
    t = 0
    ac = env.action_space.sample()
    done = True
    env_rew = 0.0
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_env_ret = 0
    ep_lens = []
    ep_env_rets = []
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    vs = np.zeros(timesteps_per_batch, 'float32')
    env_rews = np.zeros(timesteps_per_batch, 'float32')
    dones = np.zeros(timesteps_per_batch, 'int32')

    while True:
        ac, v_pred = pi.predict(sample_or_mode, ob)
        if t > 0 and t % timesteps_per_batch == 0:
            yield {"obs": obs,
                   "acs": acs,
                   "vs": vs,
                   "env_rews": env_rews,
                   "dones": dones,
                   "next_v": v_pred * (1 - done),
                   "ep_lens": ep_lens,
                   "ep_env_rets": ep_env_rets}
            _, v_pred = pi.predict(sample_or_mode, ob)
            ep_lens = []
            ep_env_rets = []
        i = t % timesteps_per_batch
        obs[i] = ob
        acs[i] = ac
        vs[i] = v_pred
        dones[i] = done
        ob, env_rew, done, _ = env.step(ac)
        env_rews[i] = env_rew
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        if done:
            ep_lens.append(cur_ep_len)
            ep_env_rets.append(cur_ep_env_ret)
            cur_ep_len = 0
            cur_ep_env_ret = 0
            ob = env.reset()
        t += 1


def traj_ep_generator(env, pi, sample_or_mode, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs0 = []
    acs = []
    env_rews = []
    dones1 = []
    obs1 = []

    while True:
        ac, _ = pi.predict(sample_or_mode, ob)
        obs0.append(ob)
        acs.append(ac)
        if render:
            env.render()
        new_ob, env_rew, done, _ = env.step(ac)
        obs1.append(new_ob)
        ob = copy.copy(new_ob)
        env_rews.append(env_rew)
        dones1.append(done)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        if done:
            obs0 = np.array(obs0)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            dones1 = np.array(dones1)
            obs1 = np.array(obs1)
            yield {"obs0": obs0,
                   "acs": acs,
                   "env_rews": env_rews,
                   "dones1": dones1,
                   "obs1": obs1,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            ob = env.reset()
            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs0 = []
            acs = []
            env_rews = []
            dones1 = []
            obs1 = []


def evaluate(env, xpo_agent_wrapper, num_trajs, sample_or_mode, render,
             exact_model_path=None, model_ckpt_dir=None):
    """Evaluate a trained TRPO agent"""

    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph to gain evaluation access to a learned and saved policy
    pi = xpo_agent_wrapper('pi')
    # Create episode generator
    traj_gen = traj_ep_generator(env=env, pi=pi, sample_or_mode=sample_or_mode, render=render)
    # Initialize and load the previously learned weights into the freshly re-built graph
    initialize()
    if exact_model_path is not None:
        load_model(exact_model_path)
        logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
    else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
        load_latest_checkpoint(model_ckpt_dir)
        logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))
    # Initialize the history data structures
    ep_env_rets = []
    ep_lens = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = traj_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    sample_or_mode = 'sample' if sample_or_mode else 'mode'
    logger.info("action picking: {}".format(sample_or_mode))
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def gather_trajectories(env, xpo_agent_wrapper, demos_dir, num_trajs, sample_or_mode, render,
                        expert_arxiv_name, exact_model_path=None, model_ckpt_dir=None):
    """Gather trajectories from a trained `mlp_policy` agent"""
    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph to gain evaluation access to a learned and saved policy
    pi = xpo_agent_wrapper('pi')
    # Create episode generator
    traj_gen = traj_ep_generator(env=env, pi=pi, sample_or_mode=sample_or_mode, render=render)
    # Initialize and load the previously learned weights into the freshly re-built graph
    initialize()
    if exact_model_path is not None:
        load_model(exact_model_path)
        logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
    else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
        load_latest_checkpoint(model_ckpt_dir)
        logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))
    # Initialize the history data structures
    obs0 = []
    acs = []
    env_rews = []
    dones1 = []
    obs1 = []
    ep_env_rets = []
    ep_lens = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("gathering [{}/{}]".format(i + 1, num_trajs))
        traj = traj_gen.__next__()
        # Next two steps are separated to shrink line length
        ep_obs0, ep_acs, ep_env_rews = traj['obs0'], traj['acs'], traj['env_rews']
        ep_dones1, ep_obs1 = traj['dones1'], traj['obs1']
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        obs0.append(ep_obs0)
        acs.append(ep_acs)
        env_rews.append(ep_env_rews)
        dones1.append(ep_dones1)
        obs1.append(ep_obs1)
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    sample_or_mode = 'sample' if sample_or_mode else 'mode'
    logger.info("action picking: {}".format(sample_or_mode))
    ep_len_mean = np.mean(ep_lens)
    ep_len_std = np.std(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    ep_env_ret_std = np.std(ep_env_rets)
    ep_env_ret_min = np.amin(ep_env_rets)
    ep_env_ret_max = np.amax(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_len_std", ep_len_std)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.record_tabular("ep_env_ret_std", ep_env_ret_std)
    logger.record_tabular("ep_env_ret_min", ep_env_ret_min)
    logger.record_tabular("ep_env_ret_max", ep_env_ret_max)
    logger.dump_tabular()
    # Assemble the file name
    path = osp.join(demos_dir, "{}.{}".format(expert_arxiv_name, sample_or_mode))
    # Save the gathered data collections to the filesystem
    np.savez(path,
             obs0=np.array(obs0),
             acs=np.array(acs),
             env_rews=np.array(env_rews),
             dones1=np.array(dones1),
             obs1=np.array(obs1),
             ep_lens=np.array(ep_lens),
             ep_env_rets=np.array(ep_env_rets))
    logger.info("saving demonstrations")
    logger.info("  @: {}.npz".format(path))
