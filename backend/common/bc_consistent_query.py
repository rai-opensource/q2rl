# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Self-consistent ``(action, log_prob, entropy)`` queries for robomimic GMM BC.

robomimic's ``BC_RNN_GMM.get_action_and_log_prob`` samples the action from the
carried RNN hidden state but scores it under a second distribution built from a
fresh zero state, so past the first step of an episode the action is scored
under a distribution it was not drawn from. The helpers here build the
distribution once and sample, score and take its entropy from that one object.

Legacy variants of the entropy formula are kept so the change can be toggled.
Nothing under ``robomimic/`` is modified.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_nets import MIMO_Transformer, RNN_MIMO_MLP


def gmm_entropy_upper_bound(dist: D.MixtureSameFamily) -> torch.Tensor:
    """Upper-bound entropy for a GMM: ``H(weights) + sum_k w_k H_k``, in nats.

    Same units as ``dist.log_prob``.
    """
    cat = dist.mixture_distribution
    comps = dist.component_distribution
    w = cat.probs  # (..., K)
    discrete_h = -(w * w.clamp_min(1e-20).log()).sum(dim=-1)
    expected_comp_h = (w * comps.entropy()).sum(dim=-1)
    return discrete_h + expected_comp_h


def legacy_gmm_entropy(dist: D.MixtureSameFamily) -> torch.Tensor:
    """The original GMM entropy formula (``log2`` of the mean variance), in bits.

    Mirrors ``BC_RNN_GMM.get_action_and_log_prob``. Not comparable to
    ``log_prob``, which is in nats.
    """
    cat = dist.mixture_distribution
    comps = dist.component_distribution
    w = cat.probs
    discrete_h = -(w * w.clamp_min(1e-20).log()).sum(dim=-1)
    comp_entropy = 0.5 * torch.log2(
        2 * torch.e * torch.pi * torch.mean(comps.base_dist.variance, axis=-1) + 1e-10
    )
    expected_comp_h = (w * comp_entropy).sum(dim=-1)
    return discrete_h + expected_comp_h


def consistent_gmm_dist(policy: Any, ob: dict) -> D.MixtureSameFamily:
    """Build the GMM the policy acts under at this step.

    For RNN policies this threads and advances the algo's carried hidden state
    exactly like ``get_action``, in a single ``forward_train`` pass. MLP and
    transformer policies are already stateless in this sense.

    Args:
        policy: a robomimic ``RolloutPolicy``.
        ob: a single un-batched policy obs dict.

    Returns:
        A ``MixtureSameFamily``. RNN/transformer nets carry a unit time
        dimension (batch_shape ``[B, 1]``); MLP nets give ``[B]``.
    """
    algo = policy.policy
    net = algo.nets["policy"]
    obs_dict = policy._prepare_observation(ob)

    if isinstance(net, RNN_MIMO_MLP):
        # Same hidden-state management as BC_RNN(_GMM).get_action, so the
        # rollout's per-step state advancement is unchanged.
        if algo._rnn_hidden_state is None or algo._rnn_counter % algo._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            algo._rnn_hidden_state = net.get_rnn_init_state(
                batch_size=batch_size, device=algo.device
            )
            if algo._rnn_is_open_loop:
                algo._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if algo._rnn_is_open_loop:
            obs_to_use = algo._open_loop_obs

        algo._rnn_counter += 1
        seq_obs = TensorUtils.to_sequence(obs_to_use)  # [B, 1, ...]
        with torch.no_grad():
            dists, algo._rnn_hidden_state = net.forward_train(
                obs_dict=seq_obs,
                goal_dict=None,
                rnn_init_state=algo._rnn_hidden_state,
                return_state=True,
            )
        return dists

    obs_in = obs_dict
    if isinstance(net, MIMO_Transformer):
        obs_in = TensorUtils.to_sequence(obs_dict)
    with torch.no_grad():
        dists = net.forward_train(obs_in, goal_dict=None)
    return dists


def consistent_query_bc(
    policy: Any, ob: dict, old_gmm_entropy: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an action and score it under the same distribution.

    Shapes match ``RolloutPolicy.__call__(..., log_prob=True)`` -- ``(A,)``,
    ``(1,)``, ``(1,)`` -- so this is a drop-in replacement at the call sites.

    Args:
        policy: robomimic ``RolloutPolicy``.
        ob: single built policy obs dict.
        old_gmm_entropy: report the legacy ``log2`` entropy (bits) instead of
            the ``H(weights)+sum_k w_k H_k`` upper bound (nats).

    Returns:
        ``(action, log_prob, entropy)`` as float32 numpy arrays.
    """
    dists = consistent_gmm_dist(policy, ob)
    if not isinstance(dists, D.MixtureSameFamily):
        raise TypeError(
            "consistent_query_bc requires a GMM BC policy whose forward_train "
            f"returns a MixtureSameFamily, got {type(dists).__name__}."
        )
    with torch.no_grad():
        action = dists.sample()
        log_prob = dists.log_prob(action)
        entropy = (
            legacy_gmm_entropy(dists) if old_gmm_entropy
            else gmm_entropy_upper_bound(dists)
        )

    action_np = TensorUtils.to_numpy(action).reshape(-1).astype(np.float32)
    log_prob_np = TensorUtils.to_numpy(log_prob).reshape(1).astype(np.float32)
    entropy_np = TensorUtils.to_numpy(entropy).reshape(1).astype(np.float32)
    return action_np, log_prob_np, entropy_np
