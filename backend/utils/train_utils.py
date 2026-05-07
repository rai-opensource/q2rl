"""
Utility functions for batch manipulation and sampling.

Provides helpers for concatenating batches, indexing, and subsampling
from training datasets.
"""
from collections.abc import Mapping

import numpy as np


def concatenate_batches(batches):
    """
    Concatenate list of batches into single batch.
    
    Handles nested dictionaries for hierarchical observation structures.
    Concatenates along batch dimension (axis 0).
    
    Args:
        batches: List of batch dictionaries.
        
    Returns:
        Single concatenated batch dictionary.
    """
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            concatenated[key] = np.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(np.float32)
    return concatenated


def index_batch(batch, indices):
    """
    Index into batch using provided indices.
    
    Handles nested dictionaries for hierarchical observation structures.
    
    Args:
        batch: Batch dictionary.
        indices: Indices to select.
        
    Returns:
        Indexed batch with same structure.
    """
    indexed = {}
    for key in batch.keys():
        if isinstance(batch[key], Mapping):
            # to index into batch["observations"]["image"], etc.
            indexed[key] = index_batch(batch[key], indices)
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    """
    Randomly subsample batch to given size.
    
    Args:
        batch: Input batch dictionary.
        size: Number of samples to draw.
        
    Returns:
        Subsampled batch of size 'size'.
    """
    indices = np.random.randint(batch["rewards"].shape[0], size=size)
    return index_batch(batch, indices)
