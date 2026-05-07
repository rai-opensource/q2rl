"""
Environment utilities and wrappers.

Provides functions for creating and configuring gym environments,
evaluating policies, and handling environment-specific configurations.
"""
from typing import Optional

import d4rl
import gym
import mj_envs
import numpy as np
from absl import flags

from backend.envs.wrappers import (
    AdroitTerminalWrapper,
    KitchenTerminalWrapper,
    ScaledRewardWrapper,
    TruncationWrapper,
    RobosuiteStateWrapper,
)

FLAGS = flags.FLAGS


def make_gym_env(
    env_name: str,
    reward_scale: Optional[float] = None,
    reward_bias: Optional[float] = None,
    scale_and_clip_action: bool = False,
    action_clip_lim: Optional[float] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    seed: int = 0,
):
    """
    Create and configure a gym environment.
    
    Creates environments for Antmaze, Kitchen, Adroit, and Locomotion tasks.
    Applies appropriate wrappers for terminal state handling, action rescaling,
    reward scaling, and episode statistics.
    
    Args:
        env_name: Name of the environment.
        reward_scale: Optional reward scaling factor.
        reward_bias: Optional reward bias.
        scale_and_clip_action: Whether to rescale and clip actions.
        action_clip_lim: Action clipping limit if scaling.
        max_episode_steps: Optional max episode length.
        render_mode: Optional render mode.
        seed: Random seed.
        
    Returns:
        Configured gym environment.
    """
    try:
        env = gym.make(env_name, seed=seed, render_mode=render_mode)
    except TypeError:
        # some envs don't take in seed as argument
        env = gym.make(env_name)

    # fix the done signal
    if "kitchen" in env_name:
        env = KitchenTerminalWrapper(env)
    if "robosuite-state" in env_name:
        env = RobosuiteStateWrapper(env)
    if "binary" in env_name:
        # adroit
        env = AdroitTerminalWrapper(env)

    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if scale_and_clip_action:
        # avoid NaNs for dist.log_prob(1.0) for tanh policies
        env = gym.wrappers.RescaleAction(env, -action_clip_lim, action_clip_lim)
        env = gym.wrappers.ClipAction(env)

    if reward_scale is not None and reward_bias is not None:
        env = ScaledRewardWrapper(env, reward_scale, reward_bias)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # 4-tuple to 5-tuple return
    env = TruncationWrapper(env)

    return env


def get_env_type(env_name):
    """
    Classify environment type for logging and success conditions.
    
    Args:
        env_name: Name of environment.
        
    Returns:
        Environment type string ('adroit-binary', 'antmaze', 'kitchen', 'locomotion').
        
    Raises:
        RuntimeError: If environment type is unknown.
    """
    if env_name in (
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
    ):
        env_type = "adroit-binary"
    elif "antmaze" in env_name:
        env_type = "antmaze"
    elif "kitchen" in env_name:
        env_type = "kitchen"
    elif "halfcheetah" in env_name or "hopper" in env_name or "walker" in env_name:
        env_type = "locomotion"
    else:
        raise RuntimeError(f"Unknown environment type for {env_name}")

    return env_type


def _determine_whether_sparse_reward(env_name):
    """
    Determine if environment has sparse or dense rewards.
    
    Args:
        env_name: Name of environment.
        
    Returns:
        True if sparse-reward environment, False otherwise.
    """
    # return True if the environment is sparse-reward
    # determine if the env is sparse-reward or not
    if "antmaze" in env_name or env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
        "can",
        "lift", 
        "square",]:
        is_sparse_reward = True
    elif (
        "halfcheetah" in env_name
        or "hopper" in env_name
        or "walker" in env_name
        or "kitchen" in env_name
    ):
        is_sparse_reward = False
    else:
        raise NotImplementedError

    return is_sparse_reward


# used to calculate the MC return for sparse-reward tasks.
# Assumes that the environment issues two reward values: reward_pos when the
# task is completed, and reward_neg at all the other steps.
ENV_REWARD_INFO = {
    "antmaze": {  # antmaze default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {  # adroit default is -1/0 reward
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    },
}


def _get_negative_reward(env_name, reward_scale, reward_bias):
    """
    Given an environment with sparse rewards (aka there's only two reward values,
    the goal reward when the task is done, or the step penalty otherwise).
    Args:
        env_name: the name of the environment
        reward_scale: the reward scale
        reward_bias: the reward bias. The reward_scale and reward_bias are not applied
            here to scale the reward, but to determine the correct negative reward value.

    NOTE: this function should only be called on sparse-reward environments
    """
    if env_name in [
        "antmaze",
        "can", 
        "lift", 
        "square",]:
        reward_neg = (
            ENV_REWARD_INFO["antmaze"]["reward_neg"] * reward_scale + reward_bias
        )
    elif env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
    ]:
        reward_neg = (
            ENV_REWARD_INFO["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        raise NotImplementedError(
            """
            If you want to try on a sparse reward env,
            please add the reward_neg value in the ENV_REWARD_INFO dict.
        """
        )

    return reward_neg


def calc_return_to_go(
    env_name,
    rewards,
    masks,
    gamma,
    reward_scale=None,
    reward_bias=None,
    infinite_horizon=False,
):
    """
    Calculat the Monte Carlo return to go given a list of reward for a single trajectory.
    Args:
        env_name: the name of the environment
        rewards: a list of rewards
        masks: a list of done masks
        gamma: the discount factor used to discount rewards
        reward_scale, reward_bias: the reward scale and bias used to determine
            the negative reward value for sparse-reward environments. If None,
            default from FLAGS values. Leave None unless for special cases.
        infinite_horizon: whether the MDP has inifite horizion (and therefore infinite return to go)
    """
    if len(rewards) == 0:
        return np.array([])

    # process sparse-reward envs
    if reward_scale is None or reward_bias is None:
        # scale and bias not applied, but used to determien the negative reward value
        assert reward_scale is None and reward_bias is None  # both should be unset
        reward_scale = FLAGS.reward_scale
        reward_bias = FLAGS.reward_bias
    is_sparse_reward = _determine_whether_sparse_reward(env_name)
    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        # sum up the rewards backwards as the return to go
        return_to_go = [0] * len(rewards)
        prev_return = 0 if not infinite_horizon else float(rewards[-1] / (1 - gamma))
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                masks[-i - 1]
            )
            prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)
