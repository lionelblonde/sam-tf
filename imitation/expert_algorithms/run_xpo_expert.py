from os import makedirs
import os.path as osp

from imitation.helpers.tf_util import single_threaded_session
from imitation.helpers.argparsers import xpo_expert_argparser
from imitation.helpers.experiment_initializer import ExperimentInitializer
from imitation.helpers.env_makers import make_env
from imitation.helpers.misc_util import set_global_seeds
from imitation.expert_algorithms.xpo_agent import XPOAgent
from imitation.expert_algorithms import xpo_util


def train_xpo_expert(args):
    """Train a XPO expert policy"""
    # Create a single-threaded session
    single_threaded_session().__enter__()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, comm=comm)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_long_name()

    # Seedify
    rank = comm.Get_rank()
    worker_seed = args.seed + 1000000 * rank
    set_global_seeds(worker_seed)
    # Create environment
    env = make_env(args.env_id, worker_seed, args.horizon)

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Train XPO expert policy
    if args.algo == 'ppo':
        from imitation.expert_algorithms import ppo
        ppo.learn(comm=comm,
                  env=env,
                  xpo_agent_wrapper=xpo_agent_wrapper,
                  sample_or_mode=args.sample_or_mode,
                  gamma=args.gamma,
                  save_frequency=args.save_frequency,
                  ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                  summary_dir=osp.join(args.summary_dir, experiment_name),
                  timesteps_per_batch=args.timesteps_per_batch,
                  batch_size=args.batch_size,
                  optim_epochs_per_iter=args.optim_epochs_per_iter,
                  lr=args.lr,
                  experiment_name=experiment_name,
                  ent_reg_scale=args.ent_reg_scale,
                  clipping_eps=args.clipping_eps,
                  gae_lambda=args.gae_lambda,
                  schedule=args.schedule,
                  max_iters=int(args.num_iters))
    elif args.algo == 'trpo':
        from imitation.expert_algorithms import trpo
        trpo.learn(comm=comm,
                   env=env,
                   xpo_agent_wrapper=xpo_agent_wrapper,
                   experiment_name=experiment_name,
                   ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                   summary_dir=osp.join(args.summary_dir, experiment_name),
                   sample_or_mode=args.sample_or_mode,
                   gamma=args.gamma,
                   max_kl=args.max_kl,
                   save_frequency=args.save_frequency,
                   timesteps_per_batch=args.timesteps_per_batch,
                   batch_size=args.batch_size,
                   ent_reg_scale=args.ent_reg_scale,
                   gae_lambda=args.gae_lambda,
                   cg_iters=args.cg_iters,
                   cg_damping=args.cg_damping,
                   vf_iters=args.vf_iters,
                   vf_lr=args.vf_lr,
                   max_iters=int(args.num_iters))
    else:
        raise RuntimeError("unknown algorithm")

    # Close environment
    env.close()


def evaluate_xpo_expert(args):
    """Evaluate a trained XPO expert policy"""
    assert args.render + args.record <= 1, "either record video or render"

    # Create a single-threaded session
    single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()

    # Seedify
    set_global_seeds(args.seed)
    # Create environment
    env = make_env(args.env_id, args.seed, args.horizon)

    if args.record:
        # Create experiment name
        experiment_name = experiment.get_long_name()
        save_dir = osp.join(args.video_dir, experiment_name)
        makedirs(save_dir, exist_ok=True)
        # Wrap the environment again to record videos
        from imitation.helpers.video_recorder_wrapper import VideoRecorder
        video_length = args.horizon if args.horizon is not None else env.env._max_episode_steps
        env = VideoRecorder(env=env,
                            save_dir=save_dir,
                            record_video_trigger=lambda x: x % x == 0,  # record at the very start
                            video_length=video_length,
                            prefix="video_{}".format(args.env_id))

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Evaluate trained XPO expert policy
    xpo_util.evaluate(env=env,
                      xpo_agent_wrapper=xpo_agent_wrapper,
                      num_trajs=args.num_trajs,
                      sample_or_mode=args.sample_or_mode,
                      render=args.render,
                      exact_model_path=args.exact_model_path,
                      model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


def gather_xpo_expert(args):
    """Gather trajectories from a trained XPO expert policy"""
    # Create a single-threaded session
    single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()

    # Seedify
    set_global_seeds(args.seed)
    # Create environment
    env = make_env(args.env_id, args.seed, args.horizon)

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Prepare trajectories destination
    expert_arxiv_name = experiment.get_expert_arxiv_name()

    # Gather trajectories from a trained XPO expert policy
    xpo_util.gather_trajectories(env=env,
                                 xpo_agent_wrapper=xpo_agent_wrapper,
                                 num_trajs=args.num_trajs,
                                 sample_or_mode=args.sample_or_mode,
                                 render=args.render,
                                 exact_model_path=args.exact_model_path,
                                 model_ckpt_dir=args.model_ckpt_dir,
                                 demos_dir=args.demos_dir,
                                 expert_arxiv_name=expert_arxiv_name)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = xpo_expert_argparser().parse_args()
    if _args.task == 'train_xpo_expert':
        train_xpo_expert(_args)
    elif _args.task == 'evaluate_xpo_expert':
        evaluate_xpo_expert(_args)
    elif _args.task == 'gather_xpo_expert':
        gather_xpo_expert(_args)
    else:
        raise NotImplementedError
