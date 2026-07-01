import collections
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import numpy as np
from flax.core import frozen_dict
from gym.utils import seeding

from backend.common.typing import Data

DatasetDict = Dict[str, Data]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    """
    Verify all arrays in a dataset dict have consistent lengths.

    Args:
        dataset_dict: Nested dict of numpy arrays.
        dataset_len: Expected length (inferred from first array if None).

    Returns:
        The common length of all arrays.

    Raises:
        AssertionError: If any array has a different length.
        TypeError: If dict contains unsupported types.
    """
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    """
    Select a subset of data by index from a nested dataset dict.

    Args:
        dataset_dict: Nested dict of numpy arrays.
        index: Array of indices to select.

    Returns:
        New dataset dict with only the selected rows.

    Raises:
        TypeError: If dict contains unsupported types.
    """
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    """
    Sample rows by index from a numpy array or nested dataset dict.

    Args:
        dataset_dict: Array or nested dict of arrays to sample from.
        indx: Array of indices to sample.

    Returns:
        Sampled array or nested dict of sampled arrays.

    Raises:
        TypeError: If input contains unsupported types.
    """
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):
    """
    Base dataset class for offline RL data.

    Wraps a dictionary of numpy arrays and provides seeded random sampling
    and train/test splitting utilities.

    Attributes:
        dataset_dict: Nested dict of numpy arrays containing dataset fields.
        dataset_len: Common length of all arrays in the dataset.
    """

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        """
        Args:
            dataset_dict: Nested dict of numpy arrays (all must have the same length).
            seed: Optional random seed for reproducible sampling.
        """
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        """Lazily initialized numpy random state, seeded on first access if not set."""
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set the random seed for sampling.

        Args:
            seed: Random seed (randomly chosen if None).

        Returns:
            List containing the seed used.
        """
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Sample a random batch from the dataset.

        Args:
            batch_size: Number of samples to draw.
            keys: Optional subset of keys to include (defaults to all keys).
            indx: Optional explicit indices to use instead of random sampling.

        Returns:
            Dict mapping keys to arrays of shape (batch_size, ...).
        """
        if indx is None:
            indx = self.np_random.choice(len(self), size=batch_size, replace=True)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        """
        Randomly split the dataset into train and test subsets.

        Args:
            ratio: Fraction of data to use for training (must be in (0, 1)).

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        assert 0 < ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio) :]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)
