from collections import namedtuple

from imitation.helpers.misc_util import boolean_flag


def disambiguate(kvs, tokens):
    """For each token, return a `namedtuple` whose keys do not start
    with any other token from the list of tokens, and wrap in a tuple
    """
    # List out ambiguous hyperparameter stems
    amb_hp_stems = ['nums_filters',
                    'filter_shapes',
                    'stride_shapes',
                    'hid_widths',
                    'ent_reg_scale']
    import argparse
    if isinstance(kvs, argparse.Namespace):
        kvs = kvs.__dict__
    assert len(tokens) > 0, "empty token list"
    assert len(tokens) == len(set(tokens)), "duplicates in token list"
    assert all(token.split('_')[0] == token for token in tokens), "`_` in token"
    hps = []

    for token in tokens:
        subdict = {}
        for k, v in kvs.items():
            # Split the key around the (unique per hp) special delimiter
            k_split = k.split('_')
            # Isolate the head and queue
            k_h = k_split[0]
            k_q = '_'.join(k_split[1:])
            # In the following, priority has been given to readability (no code golf)
            if k_h not in tokens:
                # head not in tokens
                subdict.update({k: v})
            else:
                # head in tokens
                if k_q in amb_hp_stems:
                    # queue listed as ambiguous
                    if k_h == token:
                        subdict.update({k_q: v})
                    else:
                        pass
                else:
                    subdict.update({k: v})

        HyperParameters = namedtuple('HyperParameters', subdict.keys())
        subdict = HyperParameters(**subdict)
        hps.append(subdict)
    return tuple(hps)


def argparse(description):
    """Create an empty argparse.ArgumentParser"""
    import argparse
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def xpo_expert_argparser(description="XPO Expert Experiment"):
    """Create an argparse.ArgumentParser for XPO-expert-related tasks"""
    parser = argparse(description)
    parser.add_argument('--note', help='w/e', type=str, default=None)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    boolean_flag(parser, 'from_raw_pixels', default=False)
    parser.add_argument('--horizon', help='maximum number of timesteps in an episode',
                        type=int, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save the models',
                        default=None)
    parser.add_argument('--log_dir', help='directory to save the log files',
                        default='data/logs')
    parser.add_argument('--summary_dir', help='directory to save the summaries',
                        default='data/summaries')
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='whether to record the interaction traces', default=False)
    parser.add_argument('--video_dir', help='directory to save the video recordings',
                        default='data/videos')
    parser.add_argument('--task', help='task to carry out', type=str,
                        choices=['train_xpo_expert',
                                 'evaluate_xpo_expert',
                                 'gather_xpo_expert'],
                        default='train_xpo_expert')
    parser.add_argument('--algo', help='pick an algorithm', type=str,
                        choices=['ppo', 'trpo'], default='ppo')
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_iters', help='cummulative number of iters since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--timesteps_per_batch', help='number of interactions per iteration',
                        type=int, default=1024)
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=64)
    parser.add_argument('--optim_epochs_per_iter', type=int, default=10,
                        help='optimization epochs per iteraction')
    parser.add_argument('--lr', help='adam learning rate', type=float, default=3e-4)
    boolean_flag(parser, 'sample_or_mode', default=True,
                 help='whether to pick actions by sampling or taking the mode')
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate/gather',
                        type=int, default=10)
    parser.add_argument('--exact_model_path', help='exact path of the model',
                        type=str, default=None)
    parser.add_argument('--model_ckpt_dir', help='checkpoint directory containing the models',
                        type=str, default=None)
    parser.add_argument('--demos_dir', type=str, help='directory to save the demonstrations',
                        default='data/expert_demonstrations')
    boolean_flag(parser, 'rmsify_obs', default=True)
    parser.add_argument('--nums_filters', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--filter_shapes', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--stride_shapes', nargs='+', type=int, default=[4, 2])
    parser.add_argument('--hid_widths', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--hid_nonlin', type=str, default='leaky_relu',
                        choices=['relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'tanh'])
    parser.add_argument('--hid_w_init', type=str, default='he_normal',
                        choices=['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform'])
    boolean_flag(parser, 'gaussian_fixed_var', default=True)
    boolean_flag(parser, 'with_layernorm', default=False)
    parser.add_argument('--cg_iters', type=int, default=10,
                        help='number of conjugate gradient iterations')
    parser.add_argument('--cg_damping', type=float, default=0.1, help='conjugate gradient damping')
    parser.add_argument('--vf_iters', type=int, default=10,
                        help='number of iterations for value function adam optimization')
    parser.add_argument('--vf_lr', type=float, default=3e-4,
                        help='value function adam learning rate')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--ent_reg_scale', help='scale of the policy entropy term',
                        type=float, default=0.)
    parser.add_argument('--clipping_eps', help='ppo annealed clipping parameter epsilon',
                        type=float, default=3e-1)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='gae lamdba parameter', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='constant',
                        choices=['constant', 'linear'])
    return parser


def gail_argparser(description="GAIL Experiment"):
    """Create an argparse.ArgumentParser for GAIL-related tasks"""
    parser = argparse(description)
    parser.add_argument('--note', help='w/e note', type=str, default=None)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    boolean_flag(parser, 'from_raw_pixels', default=False)
    parser.add_argument('--horizon', help='maximum number of timesteps in an episode',
                        type=int, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save the models',
                        default=None)
    parser.add_argument('--log_dir', help='directory to save the log files',
                        default='data/logs')
    parser.add_argument('--summary_dir', help='directory to save the summaries',
                        default='data/summaries')
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='whether to record the interaction traces', default=False)
    parser.add_argument('--video_dir', help='directory to save the video recordings',
                        default='data/videos')
    parser.add_argument('--task', help='task to carry out', type=str,
                        choices=['imitate_via_gail',
                                 'evaluate_gail_policy'],
                        default='imitate_via_gail')
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_iters', help='cummulative number of iters since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--timesteps_per_batch', help='number of interactions per iteration',
                        type=int, default=1024)
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=32)
    boolean_flag(parser, 'sample_or_mode', default=True,
                 help='whether to pick actions by sampling or taking the mode')
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate/gather',
                        type=int, default=10)
    parser.add_argument('--exact_model_path', help='exact path of the model',
                        type=str, default=None)
    parser.add_argument('--model_ckpt_dir', help='checkpoint directory containing the models',
                        type=str, default=None)
    parser.add_argument('--expert_path', help='.npz archive containing the demos',
                        type=str, default=None)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    parser.add_argument('--g_steps', type=int, default=3)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--d_lr', type=float, default=3e-4)
    boolean_flag(parser, 'non_satur_grad', help='whether to use non-saturating gradients in d')
    boolean_flag(parser, 'rmsify_obs', default=True)
    parser.add_argument('--pol_nums_filters', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--pol_filter_shapes', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--pol_stride_shapes', nargs='+', type=int, default=[4, 2])
    parser.add_argument('--pol_hid_widths', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--d_nums_filters', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--d_filter_shapes', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--d_stride_shapes', nargs='+', type=int, default=[4, 2])
    parser.add_argument('--d_hid_widths', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--hid_nonlin', type=str, default='leaky_relu',
                        choices=['relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'tanh'])
    parser.add_argument('--hid_w_init', type=str, default='he_normal',
                        choices=['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform'])
    boolean_flag(parser, 'gaussian_fixed_var', default=True)
    boolean_flag(parser, 'with_layernorm', default=False)
    parser.add_argument('--cg_iters', type=int, default=10,
                        help='number of conjugate gradient iterations')
    parser.add_argument('--cg_damping', type=float, default=0.1, help='conjugate gradient damping')
    parser.add_argument('--vf_iters', type=int, default=10,
                        help='number of iterations for value function adam optimization')
    parser.add_argument('--vf_lr', type=float, default=3e-4,
                        help='value function adam learning rate')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--pol_ent_reg_scale', type=float, default=0.,
                        help='scale of the policy entropy term')
    parser.add_argument('--d_ent_reg_scale', type=float, default=0.,
                        help='scale of the dicriminator entropy term')
    boolean_flag(parser, 'label_smoothing', default=True)
    boolean_flag(parser, 'one_sided_label_smoothing', default=True)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='gae lamdba parameter', type=float, default=0.97)
    parser.add_argument('--clip_norm', type=float, default=None)
    return parser


def sam_argparser(description="SAM Experiment"):
    """Create an argparse.ArgumentParser for SAM-related tasks"""
    parser = argparse(description)
    parser.add_argument('--note', help='w/e note', type=str, default=None)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    boolean_flag(parser, 'from_raw_pixels', default=False)
    parser.add_argument('--horizon', help='maximum number of timesteps in an episode',
                        type=int, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save the models',
                        default=None)
    parser.add_argument('--log_dir', help='directory to save the log files',
                        default='data/logs')
    parser.add_argument('--summary_dir', help='directory to save the summaries',
                        default='data/summaries')
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='whether to record the interaction traces', default=False)
    parser.add_argument('--video_dir', help='directory to save the video recordings',
                        default='data/videos')
    parser.add_argument('--task', help='task to carry out', type=str,
                        choices=['imitate_via_sam',
                                 'evaluate_sam_policy'],
                        default=None)
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    boolean_flag(parser, 'preload', help='whether to preload with trained tensors', default=False)
    parser.add_argument('--num_iters', help='cummulative number of iters since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--timesteps_per_batch', help='number of interactions per iteration',
                        type=int, default=16)
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=32)
    parser.add_argument('--window', help='window size for optional d training on recent data',
                        type=int, default=None)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate/gather',
                        type=int, default=10)
    parser.add_argument('--exact_model_path', help='exact path of the model',
                        type=str, default=None)
    parser.add_argument('--model_ckpt_dir', help='checkpoint directory containing the models',
                        type=str, default=None)
    parser.add_argument('--expert_path', help='.npz archive containing the demos',
                        type=str, default=None)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    parser.add_argument('--g_steps', type=int, default=3)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--d_lr', type=float, default=3e-4)
    boolean_flag(parser, 'non_satur_grad', help='whether to use non-saturating gradients in d')
    parser.add_argument('--actorcritic_nums_filters', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--actorcritic_filter_shapes', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--actorcritic_stride_shapes', nargs='+', type=int, default=[4, 2])
    parser.add_argument('--actorcritic_hid_widths', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--d_nums_filters', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--d_filter_shapes', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--d_stride_shapes', nargs='+', type=int, default=[4, 2])
    parser.add_argument('--d_hid_widths', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--hid_nonlin', type=str, default='leaky_relu',
                        choices=['relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'tanh'])
    parser.add_argument('--hid_w_init', type=str, default='he_normal',
                        choices=['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform'])
    parser.add_argument('--ac_branch_in', type=int, default=1)
    boolean_flag(parser, 'with_layernorm', default=False)
    parser.add_argument('--d_ent_reg_scale', type=float, default=0.,
                        help='scale of the dicriminator entropy term')
    boolean_flag(parser, 'label_smoothing', default=True)
    boolean_flag(parser, 'one_sided_label_smoothing', default=True)
    boolean_flag(parser, 'rmsify_rets', default=True)
    boolean_flag(parser, 'enable_popart', default=True)
    boolean_flag(parser, 'rmsify_obs', default=True)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.995)
    parser.add_argument('--mem_size', type=int, default=int(1e6))
    boolean_flag(parser, 'reset_with_demos', default=False)
    boolean_flag(parser, 'add_demos_to_mem', default=False)
    boolean_flag(parser, 'prioritized_replay', default=False)
    parser.add_argument('--alpha', help='how much prioritized', type=float, default=0.3)
    boolean_flag(parser, 'ranked', default=False)
    boolean_flag(parser, 'unreal', default=False)
    parser.add_argument('--beta', help='importance weights usage', default=1.0, type=float)
    parser.add_argument('--reward_scale', type=float, default=1.)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--noise_type', help='choices: adaptive-param_xx, normal_xx, ou_xx, none',
                        type=str, default='adaptive-param_0.2, ou_0.1, normal_0.1')
    parser.add_argument('--rew_aug_coeff', type=float, default=0.)
    parser.add_argument('--param_noise_adaption_frequency', type=float, default=50)
    parser.add_argument('--polyak', type=float, default=0.001, help='target networks tracking')
    parser.add_argument('--q_actor_loss_scale', type=float, default=1.)
    parser.add_argument('--d_actor_loss_scale', type=float, default=0.)
    parser.add_argument('--wd_scale', help='critic wd scale', type=float, default=0.001)
    parser.add_argument('--td_loss_1_scale', type=float, default=1.)
    parser.add_argument('--td_loss_n_scale', type=float, default=1.)
    boolean_flag(parser, 'n_step_returns', default=True)
    parser.add_argument('--n', help='number of steps for the TD lookahead', type=int, default=10)
    parser.add_argument('--training_steps_per_iter', type=int, default=50)
    parser.add_argument('--eval_steps_per_iter', type=int, default=100)
    parser.add_argument('--eval_frequency', type=int, default=500)
    return parser
