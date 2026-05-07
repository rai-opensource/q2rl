import collections
from typing import Optional

import d4rl
import gym
import gym.wrappers
import numpy as np

from backend.envs.env_common import calc_return_to_go
from backend.utils.train_utils import concatenate_batches


def get_d4rl_dataset_by_trajectory(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    This function heavily inherits from d4rl.qlearning_dataset
    Instead of returning a flat dataset that is a dictionary, this function
    returns a list of trajectories, each of which is a small dict-dataset.

    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episodes = []
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        if done_bool or final_timestep:
            # record this episode and reset the stats
            episode = {
                "observations": np.array(obs_),
                "actions": np.array(action_),
                "next_observations": np.array(next_obs_),
                "rewards": np.array(reward_),
                "terminals": np.array(done_),
            }
            episodes.append(episode)

            episode_step = 0
            obs_ = []
            next_obs_ = []
            action_ = []
            reward_ = []
            done_ = []

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return episodes


def get_d4rl_dataset(
    env,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
):
    dataset = d4rl.qlearning_dataset(gym.make(env).unwrapped)

    if clip_action:
        dataset["actions"] = np.clip(dataset["actions"], -clip_action, clip_action)

    dones_float = np.zeros_like(dataset["rewards"])

    if "kitchen" in env:
        # kitchen envs don't set the done signal correctly
        dones_float = dataset["rewards"] == 4

    else:
        # antmaze / locomotion envs
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

    # reward scale and bias
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=np.logical_or(dataset["terminals"], dones_float),
        masks=1 - dataset["terminals"].astype(np.float32),
    )


def get_d4rl_dataset_with_mc_calculation(
    env_name, reward_scale, reward_bias, clip_action, gamma
):
    dataset = qlearning_dataset_and_calc_mc(
        gym.make(env_name).unwrapped,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
    )

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
        mc_returns=dataset["mc_returns"],
        masks=1 - dataset["terminals"].astype(np.float32),
    )

def get_d4rl_dataset_with_mc_calculation_and_bc_probs(
    env_name, reward_scale, reward_bias, clip_action, gamma, bc_agent
):
    dataset = qlearning_dataset_and_calc_mc(
        gym.make(env_name).unwrapped,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
    )
    

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
        mc_returns=dataset["mc_returns"],
        masks=1 - dataset["terminals"].astype(np.float32),
    )

def qlearning_dataset_and_calc_mc(
    env,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset=None,
    terminate_on_end=False,
    **kwargs
):
    # this funtion follows d4rl.qlearning_dataset
    # and adds the logic to calculate the return to go

    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    # manually update the terminals for kitchen envs
    if "kitchen" in env.unwrapped.spec.id:
        # kitchen envs don't set the done signal correctly
        dataset["terminals"] = np.logical_or(
            dataset["terminals"], dataset["rewards"] == 4
        )

    # iterate over transitions, put them into trajectories
    episode_step = 0
    for i in range(N):

        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            is_final_timestep = dataset["timeouts"][i]
        else:
            is_final_timestep = episode_step == env._max_episode_steps - 1

        if (not terminate_on_end) and is_final_timestep or i == N - 1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in (
                    "actions",
                    "next_observations",
                    "observations",
                    "rewards",
                    "terminals",
                    "timeouts",
                ):
                    data_[k].append(dataset[k][i])
            if "next_observations" not in dataset.keys():
                data_["next_observations"].append(dataset["observations"][i + 1])
            episode_step += 1
        if (done_bool or is_final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = (
                episode_data["rewards"] * reward_scale + reward_bias
            )
            episode_data["mc_returns"] = calc_return_to_go(
                env.spec.name,
                episode_data["rewards"],
                1 - episode_data["terminals"],
                gamma,
                reward_scale,
                reward_bias,
            )
            episode_data["actions"] = np.clip(
                episode_data["actions"], -clip_action, clip_action
            )
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)
    return concatenate_batches(episodes_dict_list)
