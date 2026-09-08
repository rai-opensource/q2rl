# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Differential entropy of the diagonal-Gaussian BC policy (gym entrypoints).

Q2RL's Q-estimation target is ``Q_bc = V + log pi_bc(a) + H[pi_bc]``. The
original ``H`` (:func:`legacy_diag_gaussian_entropy`) is in bits while
``log_prob`` is in nats, and averages the per-dimension variances instead of
summing over them, making it roughly ``action_dim`` times too small.
:func:`diag_gaussian_entropy` is the exact joint entropy, in nats. Both are kept
so the change can be toggled.
"""

from __future__ import annotations

import jax.numpy as jnp


def diag_gaussian_entropy(variance: jnp.ndarray) -> jnp.ndarray:
    """Exact differential entropy of a diagonal Gaussian, in nats.

    Args:
        variance: per-dimension variances, ``(..., D)``.

    Returns:
        ``0.5 * sum_i log(2 * pi * e * var_i)``, shape ``(...)``.
    """
    return 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * variance + 1e-10), axis=-1)


def legacy_diag_gaussian_entropy(variance: jnp.ndarray) -> jnp.ndarray:
    """The original entropy formula, in bits.

    Args:
        variance: per-dimension variances, ``(..., D)``.

    Returns:
        ``0.5 * log2(2 * pi * e * mean_i(var_i))``, shape ``(...)``.
    """
    return 0.5 * jnp.log2(
        2 * jnp.e * jnp.pi * jnp.mean(variance, axis=-1) + 1e-10
    )
