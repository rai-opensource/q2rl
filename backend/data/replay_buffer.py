"""
Replay buffer implementations for online RL.

Provides ReplayBuffer class for storing and sampling experience tuples during
online training. Supports gym Box and Dict observation spaces with circular
buffer semantics.
"""
import os
from typing import Iterable, Optional, Union

import gym
import gym.spaces
import numpy as np
from absl import flags

from backend.data.dataset import Dataset, DatasetDict, _sample
from backend.envs.env_common import calc_return_to_go
from typing import Iterable, Optional


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    """
    Initialize empty buffer arrays matching observation space structure.
    
    Recursively creates numpy arrays for nested Box spaces and dicts for
    nested Dict spaces.
    
    Args:
        obs_space: Gym observation space (Box or Dict).
        capacity: Buffer capacity (size of first dimension).
        
    Returns:
        Numpy array or nested dict of arrays with appropriate shapes.
        
    Raises:
        TypeError: If observation space type is unsupported.
    """
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    """
    Insert data into nested buffer structure at given index.
    
    Recursively inserts data into all arrays in nested dict structure
    at the specified index.
    
    Args:
        dataset_dict: Nested dict of buffer arrays to insert into.
        data_dict: Nested dict of data to insert.
        insert_index: Index to insert at.
        
    Raises:
        AssertionError: If key structures don't match.
        TypeError: If structure contains unsupported types.
    """
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert (
            dataset_dict.keys() == data_dict.keys()
        ), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    """
    Circular replay buffer for online RL training.
    
    Stores experience tuples (observations, actions, rewards, etc.) with
    circular overwriting when capacity is reached. Supports batched sampling
    and sampling without replacement for different training scenarios.
    
    Attributes:
        dataset_dict: Dictionary with keys for observations, actions, rewards, etc.
        _size: Current number of valid samples in buffer.
        _capacity: Maximum buffer capacity.
        _insert_index: Current position for next insert (circular).
        _sequential_index: Current index for sequential sampling.
        unsampled_indices: Indices not yet sampled in current epoch.
        _discount: Discount factor for return-to-go computation.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict, seed)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._sequential_index = 0
        self.unsampled_indices = list(range(self._size))
        self._discount = discount

    def __len__(self) -> int:
        """Return the current number of valid samples in the buffer."""
        return self._size

    def insert(self, data_dict: DatasetDict):
        """
        Add experience tuple to buffer.
        
        Inserts data at current insertion point and advances pointer with
        circular wraparound. Updates size up to capacity.
        
        Args:
            data_dict: Dictionary with keys for observations, actions, etc.
        """
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample_without_repeat(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
    ) -> dict:
        """
        Sample batch without replacement from unsampled indices.
        
        Maintains list of unsampled indices and removes each as it's selected.
        Useful for epoch-based training without replacement.
        
        Args:
            batch_size: Number of samples to draw.
            keys: Optional keys to sample (defaults to all keys).
            
        Returns:
            Dictionary mapping keys to sampled arrays.
            
        Raises:
            ValueError: If fewer unsampled indices than batch_size.
        """
        if keys is None:
            keys = self.dataset_dict.keys()

        batch = dict()
        if len(self.unsampled_indices) < batch_size:
            raise ValueError("Not enough samples left to sample without repeat.")
        selected_indices = []
        for _ in range(batch_size):
            idx = self.np_random.randint(len(self.unsampled_indices))
            selected_indices.append(self.unsampled_indices[idx])
            # Swap the selected index with the last unselected index
            self.unsampled_indices[idx], self.unsampled_indices[-1] = (
                self.unsampled_indices[-1],
                self.unsampled_indices[idx],
            )
            # Remove the last unselected index (which is now the selected index)
            self.unsampled_indices.pop()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], np.array(selected_indices))

        return batch

    def save(self, save_dir):
        """
        Save buffer to disk.
        
        Args:
            save_dir: Directory to save buffer files to.
        """
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        np.save(save_buffer_file, self.dataset_dict)
        np.save(save_size_file, self._size)

    def load(self, save_dir):
        """
        Load buffer from disk.
        
        Args:
            save_dir: Directory to load buffer files from.
        """
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        self.dataset_dict = np.load(save_buffer_file, allow_pickle=True).item()
        self._size = np.load(save_size_file, allow_pickle=True).item()
        self.unsampled_indices = list(range(self._size))


class ReplayBuffer_IBRL(ReplayBuffer):
    """
    Replay buffer variant for IBRL with behavioral cloning actions.
    
    Extends ReplayBuffer with additional fields for behavioral cloning actions
    and next base actions used in imitation-based RL training.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        """
        Args:
            observation_space: Gym observation space.
            action_space: Gym action space.
            capacity: Maximum buffer capacity.
            next_observation_space: Space for next observations (defaults to observation_space).
            seed: Optional random seed.
            discount: Discount factor (required).
        """
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        bc_actions = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
        self.dataset_dict["bc_actions"] = bc_actions

        next_bc_actions = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
        self.dataset_dict["next_bc_actions"] = next_bc_actions

        self._allow_idxs = []
        self._traj_start_idx = 0


class ReplayBufferMC(ReplayBuffer):
    """
    Replay buffer with Monte Carlo return computation.

    Extends ReplayBuffer to store and compute MC returns (return-to-go) at
    episode boundaries. Only completed trajectories are available for sampling,
    ensuring every sample has a valid MC return.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        """
        Args:
            observation_space: Gym observation space.
            action_space: Gym action space.
            capacity: Maximum buffer capacity (must exceed total online steps).
            next_observation_space: Space for next observations (defaults to observation_space).
            seed: Optional random seed.
            discount: Discount factor for MC return computation (required).
        """
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        """
        Insert a transition and compute MC returns at episode end.

        MC returns are computed and stored for all transitions in the trajectory
        when a terminal step (dones=1.0) is encountered. The transition indices
        are added to the sample-eligible set only after the trajectory completes.

        Args:
            data_dict: Transition dict with 'dones' flag and standard fields.
        """
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            FLAGS = flags.FLAGS
            rewards = self.dataset_dict["rewards"][
                self._traj_start_idx : self._insert_index + 1
            ]
            masks = self.dataset_dict["masks"][
                self._traj_start_idx : self._insert_index + 1
            ]
            self.dataset_dict["mc_returns"][
                self._traj_start_idx : self._insert_index + 1
            ] = calc_return_to_go(
                FLAGS.env,
                rewards,
                masks,
                self._discount,
            )

            self._allow_idxs.extend(
                list(range(self._traj_start_idx, self._insert_index + 1))
            )
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Sample a batch only from completed-trajectory indices.

        Args:
            batch_size: Number of samples to draw.
            keys: Optional subset of keys to include (defaults to all keys).
            indx: Optional explicit indices to use instead of random sampling.

        Returns:
            Dict mapping keys to arrays of shape (batch_size, ...).
        """
        if indx is None:
            indx = self.np_random.choice(
                self._allow_idxs, size=batch_size, replace=True
            )
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch


class ReplayBuffer_Q2RL(ReplayBuffer):
    """
    Replay buffer for Q2RL with MC returns and BC policy statistics.

    Extends ReplayBuffer with fields for Monte Carlo returns, behavioral cloning
    action log-probabilities, entropy, and actions. Used to combine offline BC
    data with online RL in the Q2RL algorithm.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        """
        Args:
            observation_space: Gym observation space.
            action_space: Gym action space.
            capacity: Maximum buffer capacity (must exceed total online steps).
            next_observation_space: Space for next observations (defaults to observation_space).
            seed: Optional random seed.
            discount: Discount factor for MC return computation (required).
        """
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        bc_probs = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["bc_probs"] = bc_probs

        bc_entropy = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["bc_entropy"] = bc_entropy

        bc_actions = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
        self.dataset_dict["bc_actions"] = bc_actions

        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        """
        Insert a transition and compute MC returns at episode end.

        MC returns are computed and stored for all transitions in the trajectory
        when a terminal step (dones=1.0) is encountered. The transition indices
        are added to the sample-eligible set only after the trajectory completes.

        Args:
            data_dict: Transition dict with 'dones' flag and standard fields.
        """
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            FLAGS = flags.FLAGS
            rewards = self.dataset_dict["rewards"][
                self._traj_start_idx : self._insert_index + 1
            ]
            masks = self.dataset_dict["masks"][
                self._traj_start_idx : self._insert_index + 1
            ]
            self.dataset_dict["mc_returns"][
                self._traj_start_idx : self._insert_index + 1
            ] = calc_return_to_go(
                FLAGS.env,
                rewards,
                masks,
                self._discount,
            )

            self._allow_idxs.extend(
                list(range(self._traj_start_idx, self._insert_index + 1))
            )
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Sample a batch only from completed-trajectory indices.

        Args:
            batch_size: Number of samples to draw.
            keys: Optional subset of keys to include (defaults to all keys).
            indx: Optional explicit indices to use instead of random sampling.

        Returns:
            Dict mapping keys to arrays of shape (batch_size, ...).
        """
        if indx is None:
            indx = self.np_random.choice(
                self._allow_idxs, size=batch_size, replace=True
            )
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch
