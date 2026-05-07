"""
source: https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/replay_buffer.py
"""
import os

import numpy as np
from absl import flags

from backend.envs.env_common import calc_return_to_go

DEMO_PATHS = os.environ.get("DATA_DIR_PREFIX", os.path.expanduser("~/adroit_data"))

FLAGS = flags.FLAGS


def get_hand_dataset_with_mc_calculation(
    env_name,
    gamma,
    add_expert_demos=True,
    add_bc_demos=True,
    reward_scale=1.0,
    reward_bias=0.0,
    pos_ind=-1,
    clip_action=None,
):
    assert env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]

    expert_demo_paths = {
        "pen-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/pen2_sparse.npy",
        "door-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/door2_sparse.npy",
        "relocate-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/relocate2_sparse.npy",
    }

    bc_demo_paths = {
        "pen-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/pen_bc_sparse4.npy",
        "door-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/door_bc_sparse4.npy",
        "relocate-binary-v0": f"{DEMO_PATHS}/offpolicy_hand_data/relocate_bc_sparse4.npy",
    }

    def truncate_traj(
        env_name,
        dataset,
        i,
        gamma,
        start_index=None,
        end_index=None,
    ):
        """
        This function truncates the i'th trajectory in dataset from start_index to end_index.
        Since in Adroit-binary datasets, we have trajectories like [-1, -1, -1, -1, 0, 0, 0, -1, -1] which transit from neg -> pos -> neg,
        we truncate the trajcotry from the beginning to the last positive reward, i.e., [-1, -1, -1, -1, 0, 0, 0]
        """
        observations = np.array(dataset[i]["observations"])[start_index:end_index]
        next_observations = np.array(dataset[i]["next_observations"])[
            start_index:end_index
        ]
        rewards = dataset[i]["rewards"][start_index:end_index]
        dones = rewards == 0  # by default, adroit has -1/0 rewards
        actions = np.array(dataset[i]["actions"])[start_index:end_index]
        mc_returns = calc_return_to_go(
            env_name,
            rewards * FLAGS.reward_scale + FLAGS.reward_bias,
            1 - dones,
            gamma,
            infinite_horizon=False,
        )

        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            masks=1 - dones,
            mc_returns=mc_returns,
        )

    dataset_list = []
    dataset_bc_list = []

    if add_expert_demos:
        print("loading expert demos from:", expert_demo_paths[env_name])
        dataset = np.load(expert_demo_paths[env_name], allow_pickle=True)

        for i in range(len(dataset)):
            N = len(dataset[i]["observations"])
            for j in range(len(dataset[i]["observations"])):
                dataset[i]["observations"][j] = dataset[i]["observations"][j][
                    "state_observation"
                ]
                dataset[i]["next_observations"][j] = dataset[i]["next_observations"][j][
                    "state_observation"
                ]
            if (
                np.array(dataset[i]["rewards"]).shape
                != np.array(dataset[i]["terminals"]).shape
            ):
                dataset[i]["rewards"] = dataset[i]["rewards"][:N]

            assert (
                np.array(dataset[i]["rewards"]).shape
                == np.array(dataset[i]["terminals"]).shape
            )
            dataset[i].pop("terminals", None)

            if not (0 in dataset[i]["rewards"]):
                continue

            trunc_ind = np.where(dataset[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(
                env_name,
                dataset,
                i,
                gamma,
                start_index=None,
                end_index=trunc_ind,
            )
            dataset_list.append(d_pos)

    if add_bc_demos:
        print("loading BC demos from:", bc_demo_paths[env_name])
        dataset_bc = np.load(bc_demo_paths[env_name], allow_pickle=True)
        for i in range(len(dataset_bc)):
            dataset_bc[i]["rewards"] = dataset_bc[i]["rewards"].squeeze()
            dataset_bc[i]["dones"] = dataset_bc[i]["terminals"].squeeze()
            dataset_bc[i].pop("terminals", None)

            if not (0 in dataset_bc[i]["rewards"]):
                continue
            trunc_ind = np.where(dataset_bc[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(
                env_name,
                dataset_bc,
                i,
                gamma,
                start_index=None,
                end_index=trunc_ind,
            )
            dataset_bc_list.append(d_pos)

    dataset = np.concatenate([dataset_list, dataset_bc_list])

    print("num offline trajs:", len(dataset))
    concatenated = {}
    for key in dataset[0].keys():
        if key in ["agent_infos", "env_infos"]:
            continue
        concatenated[key] = np.concatenate(
            [batch[key] for batch in dataset], axis=0
        ).astype(np.float32)

    # global transforms
    if clip_action:
        concatenated["actions"] = np.clip(
            concatenated["actions"], -clip_action, clip_action
        )
    concatenated["rewards"] = concatenated["rewards"] * reward_scale + reward_bias

    return concatenated
