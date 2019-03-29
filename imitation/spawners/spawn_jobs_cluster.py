"""Example launch
    python -m imitation.spawners.spawn_jobs_cluster \
        --task=sam \
        --benchmark=mujoco \
        --cluster=cscs \
        --demos_dir=/code/sam-tf/DEMOS \
        --no-mpi \
        --num_workers=4 \
        --partition=shared-gpu \
        --time=12:00:00 \
        --num_seeds=5 \
        --no-call \
        --no-rand \
        --num_demos="4, 16"
"""

import argparse
import os.path as osp
import numpy as np
from subprocess import call
from copy import copy
import yaml

from imitation.helpers.misc_util import flatten_lists, zipsame, boolean_flag
from imitation.helpers.experiment_initializer import rand_id


parser = argparse.ArgumentParser(description="SAM Job Orchestrator + HP Search")
parser.add_argument('--task', type=str, choices=['ppo', 'gail', 'sam'], default='ppo',
                    help="whether to train an expert with PPO or an imitator with GAIL or SAM")
parser.add_argument('--benchmark', type=str, choices=['ale', 'mujoco'], default='mujoco',
                    help="benchmark on which to run experiments")
parser.add_argument('--cluster', type=str, choices=['baobab', 'cscs', 'gengar'], default='gengar',
                    help="cluster on which the experiments will be launched")
parser.add_argument('--num_rand_trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
boolean_flag(parser, 'mpi', default=False, help="whether to use mpi")
parser.add_argument('--num_workers', type=int, default=1,
                    help="number of parallel mpi workers (actors) to use for each job")
parser.add_argument('--partition', type=str, default=None, help="partition to launch jobs on")
parser.add_argument('--time', type=str, default=None, help="duration of the jobs")
parser.add_argument('--num_seeds', type=int, default=5,
                    help="amount of seeds across which jobs are replicated ('range(num_seeds)'')")
boolean_flag(parser, 'call', default=False, help="whether to launch the jobs once created")
boolean_flag(parser, 'rand', default=False, help="whether to perform hyperparameter search")
parser.add_argument('--num_demos', default="4, 16", help="comma-separated list of num of demos")
args = parser.parse_args()

# Load the content of the config file
envs = yaml.load(open("admissible_envs.yml"))['environments']

MUJOCO_ENVS_SET = list(envs['mujoco'].keys())
ALE_ENVS_SET = list(envs['ale'].keys())
MUJOCO_EXPERT_DEMOS = list(envs['mujoco'].values())
ALE_EXPERT_DEMOS = list(envs['ale'].values())


def zipsame_prune(l1, l2):
    out = []
    for a, b in zipsame(l1, l2):
        if b is None:
            continue
        out.append((a, b))
    return out


def parse_num_demos(num_demos):
    """Parse the `num_demos` hyperparameter"""
    return [n.strip() for n in num_demos.split(',')]


def fmt_path(args, meta, dir_):
    """Transform a relative path into an absolute path"""
    return osp.join("data/{}".format(meta), dir_)


def dup_hps_for_env(hpmap, env):
    """Return a separate copy of the HP map after adding extra key-value pair
    for the key 'env_id'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    return hpmap_


def dup_hps_for_env_w_demos(args, hpmap, env, demos):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the keys 'env_id' and 'expert_path'
    """
    demos = osp.join("DEMOS", demos)
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    hpmap_.update({'expert_path': demos})
    return hpmap_


def dup_hps_for_seed(hpmap, seed):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the key 'seed'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'seed': seed})
    return hpmap_


def dup_hps_for_num_demos(hpmap, num_demos):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the key 'num_demos'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'num_demos': num_demos})
    return hpmap_


def rand_tuple_from_list(list_):
    """Return a random tuple from a list of tuples
    (Function created because `np.random.choice` does not work on lists of tuples)
    """
    assert all(isinstance(v, tuple) for v in list_), "not a list of tuples"
    return list_[np.random.randint(low=0, high=len(list_))]


def get_rand_hps(args, meta, num_seeds):
    """Return a list of maps of hyperparameters selected by random search
    Example of hyperparameter dictionary:
        {'hid_widths': rand_tuple_from_list([(64, 64)]),  # list of tuples
         'hid_nonlin': np.random.choice(['relu', 'leaky_relu']),
         'hid_w_init': np.random.choice(['he_normal', 'he_uniform']),
         'polyak': np.random.choice([0.001, 0.01]),
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    # Atari
    if args.benchmark == 'ale':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 0,
                     'save_frequency': 20,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'nums_filters': (8, 16),
                     'filter_shapes': (8, 4),
                     'stride_shapes': (4, 2),
                     'hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env)
                      for env in ALE_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_nums_filters': (8, 16),
                     'pol_filter_shapes': (8, 4),
                     'pol_stride_shapes': (4, 2),
                     'pol_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(ALE_ENVS_SET, ALE_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'training_steps_per_iter': np.random.choice([25, 50]),
                     'eval_steps_per_iter': 10,
                     'eval_frequency': 100,
                     'render': 0,
                     'timesteps_per_batch': np.random.choice([2, 4, 8, 16, 32]),
                     'batch_size': np.random.choice([32, 64, 128, 256]),
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_nums_filters': (8, 16),
                     'actorcritic_filter_shapes': (8, 4),
                     'actorcritic_stride_shapes': (4, 2),
                     'actorcritic_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'polyak': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': np.random.choice([0, 1]),
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': np.random.choice(['adaptive-param_0.1', 'adaptive-param_0.2']),
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': np.random.choice([0.98, 0.99, 0.995]),
                     'mem_size': int(1e4),
                     'prioritized_replay': np.random.choice([0, 1]),
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'reset_with_demos': np.random.choice([0, 1]),
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': np.random.choice([0.01, 0.001]),
                     'n_step_returns': 1,
                     'n': np.random.choice([36, 72, 96])}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(ALE_ENVS_SET, ALE_EXPERT_DEMOS)]
    # MuJoCo
    elif args.benchmark == 'mujoco':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 1,
                     'save_frequency': 20,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env)
                      for env in MUJOCO_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_hid_widths': (100, 100),
                     'd_hid_widths': (100, 100),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'training_steps_per_iter': np.random.choice([25, 50]),
                     'eval_steps_per_iter': 10,
                     'eval_frequency': 100,
                     'render': 0,
                     'timesteps_per_batch': np.random.choice([2, 4, 8, 16, 32]),
                     'batch_size': np.random.choice([32, 64, 128, 256]),
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_hid_widths': (64, 64),
                     'd_hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'polyak': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': np.random.choice([0, 1]),
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': np.random.choice(['"adaptive-param_0.2"',
                                                     '"adaptive-param_0.2, ou_0.2"']),
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': np.random.choice([0.98, 0.99, 0.995]),
                     'mem_size': int(1e4),
                     'prioritized_replay': np.random.choice([0, 1]),
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'reset_with_demos': np.random.choice([0, 1]),
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': np.random.choice([0.01, 0.001]),
                     'n_step_returns': 1,
                     'n': np.random.choice([36, 72, 96])}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
    else:
        raise RuntimeError("unknown benchmark")

    # Duplicate every hyperparameter map of the list to span the range of seeds
    output = [dup_hps_for_seed(hpmap_, seed)
              for seed in range(num_seeds)
              for hpmap_ in hpmaps]

    if args.task in ['gail', 'sam']:
        # Duplicate every hyperparameter map of the list to span the range of num of demos
        output = [dup_hps_for_num_demos(hpmap_, num_demos)
                  for num_demos in parse_num_demos(args.num_demos)
                  for hpmap_ in output]

    return output


def get_spectrum_hps(args, meta, num_seeds):
    """Return a list of maps of hyperparameters selected deterministically
    and spanning the specified range of seeds
    Example of hyperparameter dictionary:
        {'hid_widths': (64, 64),
         'hid_nonlin': 'relu',
         'hid_w_init': 'he_normal',
         'polyak': 0.001,
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    # Atari
    if args.benchmark == 'ale':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 0,
                     'save_frequency': 20,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'nums_filters': (8, 16),
                     'filter_shapes': (8, 4),
                     'stride_shapes': (4, 2),
                     'hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env) for env in ALE_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_nums_filters': (8, 16),
                     'pol_filter_shapes': (8, 4),
                     'pol_stride_shapes': (4, 2),
                     'pol_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(ALE_ENVS_SET, ALE_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'training_steps_per_iter': 20,
                     'eval_steps_per_iter': 10,
                     'eval_frequency': 100,
                     'render': 0,
                     'timesteps_per_batch': 4,
                     'batch_size': 32,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_nums_filters': (8, 16),
                     'actorcritic_filter_shapes': (8, 4),
                     'actorcritic_stride_shapes': (4, 2),
                     'actorcritic_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'polyak': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': 1,
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': 'adaptive-param_0.4',
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': 0.99,
                     'mem_size': int(1e4),
                     'prioritized_replay': 0,
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'reset_with_demos': 1,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': 0.001,
                     'n_step_returns': 1,
                     'n': 96}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(ALE_ENVS_SET, ALE_EXPERT_DEMOS)]
    # MuJoCo
    elif args.benchmark == 'mujoco':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 1,
                     'save_frequency': 20,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env) for env in MUJOCO_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_hid_widths': (100, 100),
                     'd_hid_widths': (100, 100),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_iters': int(1e6),
                     'training_steps_per_iter': 20,
                     'eval_steps_per_iter': 10,
                     'eval_frequency': 100,
                     'render': 0,
                     'timesteps_per_batch': 4,
                     'batch_size': 32,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_hid_widths': (64, 64),
                     'd_hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'polyak': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': 1,
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': '"adaptive-param_0.2, ou_0.2"',
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': 0.99,
                     'mem_size': int(1e4),
                     'prioritized_replay': 0,
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'reset_with_demos': 1,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': 0.001,
                     'n_step_returns': 1,
                     'n': 96}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame_prune(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
    else:
        raise RuntimeError("unknown benchmark")

    # Duplicate every hyperparameter map of the list to span the range of seeds
    output = [dup_hps_for_seed(hpmap_, seed)
              for seed in range(num_seeds)
              for hpmap_ in hpmaps]

    if args.task in ['gail', 'sam']:
        # Duplicate every hyperparameter map of the list to span the range of num of demos
        output = [dup_hps_for_num_demos(hpmap_, num_demos)
                  for num_demos in parse_num_demos(args.num_demos)
                  for hpmap_ in output]
    return output


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    indent = 4 * ' '  # indents are defined as 4 spaces
    base_str = ""
    no_value_keys = ['from_raw_pixels',
                     'rmsify_obs',
                     'sample_or_mode',
                     'gaussian_fixed_var',
                     'render',
                     'non_satur_grad',
                     'with_layernorm',
                     'rmsify_rets',
                     'enable_popart',
                     'label_smoothing',
                     'one_sided_label_smoothing',
                     'prioritized_replay',
                     'ranked',
                     'reset_with_demos',
                     'add_demos_to_mem',
                     'unreal',
                     'n_step_returns']
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            base_str += "{}--no-{} \\\n".format(indent, k)
            continue
        elif k in no_value_keys:
            base_str += "{}--{} \\\n".format(indent, k)
            continue
        if isinstance(v, tuple):
            base_str += "{}--{} ".format(indent, k) + " ".join(str(v_) for v_ in v) + ' \\\n'
            continue
        base_str += "{}--{}={} \\\n".format(indent, k, v)
    return base_str


def format_job_str(args, job_map, run_str):
    """Build the batch script that launches a job"""

    if args.cluster == 'baobab':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --mem=32000\n')
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/6.3.0-2.27\n\n')
        # Launch command
        if args.mpi:
            bash_script_str += ('mpirun ')
        else:
            bash_script_str += ('srun ')
        bash_script_str += ('{}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)[:-2]

    elif args.cluster == 'cscs':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --constraint=gpu\n\n')
        # Load modules
        bash_script_str += ('module load daint-gpu\n')
        bash_script_str += ('\n')
        # Launch command
        if args.mpi:
            bash_script_str += ('mpirun {}')
        else:
            bash_script_str += ('srun {}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)[:-2]

    elif args.cluster == 'gengar':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpirun -np {} --allow-run-as-root {}')
        return bash_script_str.format(job_map['job-name'],
                                      job_map['ntasks'],
                                      run_str)[:-2]

    else:
        raise NotImplementedError("cluster selected is not covered")


def format_exp_str(args, hpmap):
    """Build the experiment name"""
    hpmap_str = unroll_options(hpmap)
    # Parse task name
    if args.task == 'ppo':
        script = "imitation.expert_algorithms.run_xpo_expert"
    elif args.task == 'gail':
        script = "imitation.imitation_algorithms.run_gail"
    elif args.task == 'sam':
        script = "imitation.imitation_algorithms.run_sam"

    return "python -m {} \\\n{}".format(script, hpmap_str)


def get_job_map(args, meta, i, env, seed, num_demos, type_exp):
    if not args.mpi:
        # Override the number of parallel workers to 1 when mpi is off
        args.num_workers = 1
    return {'ntasks': args.num_workers,
            'partition': args.partition,
            'time': args.time,
            'job-name': "{}_{}{}_{}_{}_s{}_d{}".format(meta,
                                                       type_exp,
                                                       i,
                                                       args.task,
                                                       env.split('-')[0],
                                                       seed,
                                                       num_demos)}


def run(args):
    """Spawn jobs"""
    # Create meta-experiment identifier
    meta = rand_id()
    # Define experiment type
    if args.rand:
        type_exp = 'hpsearch'
    else:
        type_exp = 'sweep'
    # Get hyperparameter configurations
    if args.rand:
        # Get a number of random hyperparameter configurations
        hpmaps = [get_rand_hps(args, meta, args.num_seeds) for _ in range(args.num_rand_trials)]
        # Flatten into a 1-dim list
        hpmaps = flatten_lists(hpmaps)
    else:
        # Get the deterministic spectrum of specified hyperparameters
        hpmaps = get_spectrum_hps(args, meta, args.num_seeds)
    # Create associated task strings
    exp_strs = [format_exp_str(args, hpmap) for hpmap in hpmaps]
    if not len(exp_strs) == len(set(exp_strs)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again :)")
    # Create the job maps
    job_maps = [get_job_map(args,
                            meta,
                            i,
                            hpmap['env_id'],
                            hpmap['seed'],
                            '0' if args.task == 'ppo' else hpmap['num_demos'],
                            type_exp)
                for i, hpmap in enumerate(hpmaps)]
    # Finally get all the required job strings
    job_strs = [format_job_str(args, jm, es) for jm, es in zipsame(job_maps, exp_strs)]
    # Spawn the jobs
    for i, (jm, js) in enumerate(zipsame(job_maps, job_strs)):
        print('-' * 10 + "> job #{} launcher content:".format(i))
        print(js + "\n")
        job_name = "{}.sh".format(jm['job-name'])
        with open(job_name, 'w') as f:
            f.write(js)
        if args.call:
            # Spawn the job!
            call(["sbatch", "./{}".format(job_name)])
    # Summarize the number of jobs spawned
    print("total num job (successfully) spawned: {}".format(len(job_strs)))


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
