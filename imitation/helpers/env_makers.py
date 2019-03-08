import yaml

import gym

from imitation.helpers import logger


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    envs = yaml.load(open("admissible_envs.yml"))['environments']
    benchmark = None
    for k, v in envs.items():
        if env_id in list(v.keys()):
            benchmark = k
    assert benchmark is not None, "env not found in 'project_root/admissible_envs.yml'"
    logger.info("env_id = {} <- admissibility check passed!".format(env_id))
    return benchmark


def make_mujoco_env(env_id, seed, horizon=None):
    """Create a wrapped gym.Env for MuJoCo"""
    env = gym.make(env_id)
    if horizon is not None:
        # Override the default episode horizon
        # by hacking the private attribute of the `TimeLimit` wrapped env
        env._max_episode_steps = horizon
    env.seed(seed)
    return env


def make_ale_env(env_id, seed, horizon=None):
    """Create a wrapped gym.Env for ALE"""
    from imitation.helpers.ale_wrappers import make_ale, wrap_deepmind
    env = make_ale(env_id)
    if horizon is not None:
        # Override the default episode horizon
        # by hacking the private attribute of the `TimeLimit` wrapped env
        env._max_episode_steps = horizon
    env.seed(seed)
    # Wrap (second wrapper) with DeepMind's wrapper
    env = wrap_deepmind(env, frame_stack=True)
    env.seed(seed)
    return env


def make_env(env_id, seed, horizon=None):
    """Create an environment"""
    benchmark = get_benchmark(env_id)
    if benchmark == 'mujoco':
        make_env_ = make_mujoco_env
    elif benchmark == 'ale':
        make_env_ = make_ale_env
    else:
        raise RuntimeError("unknown benchmark")
    env = make_env_(env_id, seed, horizon)
    return env
