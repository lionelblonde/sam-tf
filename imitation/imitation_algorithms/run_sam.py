from os import makedirs
import os.path as osp

from imitation.helpers.tf_util import single_threaded_session
from imitation.helpers.argparsers import sam_argparser, disambiguate
from imitation.helpers.experiment_initializer import ExperimentInitializer
from imitation.helpers.env_makers import make_env
from imitation.helpers.misc_util import set_global_seeds
from imitation.imitation_algorithms.sam_agent import SAMAgent
from imitation.imitation_algorithms.discriminator import Discriminator
from imitation.imitation_algorithms import sam
from imitation.imitation_algorithms.demo_dataset import DemoDataset


def imitate_via_sam(args):
    """Train a SAM imitation policy"""
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

    # Refine hps to avoid ambiguities
    actorcritic_hps, d_hps = disambiguate(kvs=args, tokens=['actorcritic', 'd'])

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps, comm=comm)

    # Create a SAM agent wrapper (note the second input)
    def sam_agent_wrapper(name, d):
        return SAMAgent(name=name, env=env, hps=actorcritic_hps, d=d, comm=comm)

    # Create the expert demonstrations dataset from expert trajectories
    dataset = DemoDataset(expert_arxiv=args.expert_path, size=args.num_demos,
                          train_fraction=None, randomize=True, full=args.add_demos_to_mem)

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, args.seed, args.horizon)

    comm.Barrier()

    # Train SAM imitation agent
    sam.learn(comm=comm,
              env=env,
              eval_env=eval_env,
              discriminator_wrapper=discriminator_wrapper,
              sam_agent_wrapper=sam_agent_wrapper,
              experiment_name=experiment_name,
              ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
              summary_dir=osp.join(args.summary_dir, experiment_name),
              expert_dataset=dataset,
              reset_with_demos=args.reset_with_demos,
              add_demos_to_mem=args.add_demos_to_mem,
              save_frequency=args.save_frequency,
              rew_aug_coeff=args.rew_aug_coeff,
              param_noise_adaption_frequency=args.param_noise_adaption_frequency,
              timesteps_per_batch=args.timesteps_per_batch,
              batch_size=args.batch_size,
              window=args.window,
              g_steps=args.g_steps,
              d_steps=args.d_steps,
              training_steps_per_iter=args.training_steps_per_iter,
              eval_steps_per_iter=args.eval_steps_per_iter,
              eval_frequency=args.eval_frequency,
              render=args.render,
              max_iters=int(args.num_iters),
              preload=args.preload,
              exact_model_path=args.exact_model_path,
              model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate_sam_policy(args):
    """Evaluate a trained SAM imitation policy"""
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

    # Refine hps to avoid ambiguities
    actorcritic_hps, d_hps = disambiguate(kvs=args, tokens=['actorcritic', 'd'])

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps)

    # Create a SAM agent wrapper (note the second input)
    def sam_agent_wrapper(name, d):
        return SAMAgent(name=name, env=env, hps=actorcritic_hps, d=d, comm=None)

    # Evaluate agent trained via SAM
    sam.evaluate(env=env,
                 discriminator_wrapper=discriminator_wrapper,
                 sam_agent_wrapper=sam_agent_wrapper,
                 num_trajs=args.num_trajs,
                 render=args.render,
                 exact_model_path=args.exact_model_path,
                 model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = sam_argparser().parse_args()
    if _args.task == 'imitate_via_sam':
        imitate_via_sam(_args)
    elif _args.task == 'evaluate_sam_policy':
        evaluate_sam_policy(_args)
    else:
        raise NotImplementedError
