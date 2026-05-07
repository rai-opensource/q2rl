"""
Neural network initialization functions.

Provides various weight initialization strategies for neural networks.
Supports orthogonal, variance scaling, Xavier, and Kaiming initializations.
"""
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


def var_scaling_init(scale: Optional[float] = 1.0):
    """
    Variance scaling initialization.
    
    Args:
        scale: Scaling factor (default 1.0).
        
    Returns:
        Flax initialization function.
    """
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def orthogonal_init(scale: Optional[float] = jnp.sqrt(2.0)):
    """
    Orthogonal matrix initialization.
    
    Args:
        scale: Scaling factor (default sqrt(2)).
        
    Returns:
        Flax initialization function.
    """
    return nn.initializers.orthogonal(scale)


def xavier_normal_init():
    """
    Xavier normal initialization.
    
    Returns:
        Flax initialization function.
    """
    return nn.initializers.xavier_normal()


def kaiming_init():
    """
    Kaiming/He normal initialization.
    
    Returns:
        Flax initialization function.
    """
    return nn.initializers.kaiming_normal()


def xavier_uniform_init():
    """
    Xavier uniform initialization.
    
    Returns:
        Flax initialization function.
    """
    return nn.initializers.xavier_uniform()


init_fns = {
    None: orthogonal_init,
    "var_scaling": var_scaling_init,
    "orthogonal": orthogonal_init,
    "xavier_normal": xavier_normal_init,
    "kaiming": kaiming_init,
    "xavier_uniform": xavier_uniform_init,
}
