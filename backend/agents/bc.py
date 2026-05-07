"""
Behavioral Cloning (BC) agent implementation.

This module implements a Behavioral Cloning agent that learns to imitate expert
demonstrations by maximizing the likelihood of expert actions given observations.
Uses JAX-based neural networks with Flax for efficient training.
"""
from functools import partial
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from backend.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from backend.common.typing import Batch, PRNGKey
from backend.networks.actor_critic_nets import Policy
from backend.networks.mlp import MLP


class BCAgent(flax.struct.PyTreeNode):
    """
    Behavioral Cloning Agent.
    
    Learns a policy by maximizing likelihood of expert actions. Uses policy 
    networks with distribution output and optimizes negative log-likelihood loss.
    Supports temperature-scaled action sampling and deterministic modes.
    
    Attributes:
        state: JaxRLTrainState containing network parameters and optimizer state.
        lr_schedule: Learning rate schedule function callable with step count.
    """
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        """
        Update agent parameters using behavioral cloning loss.
        
        Computes negative log-likelihood loss between predicted and expert actions,
        then updates network parameters using gradient descent.
        
        Args:
            batch: Dictionary containing 'observations' and 'actions' tensors.
            pmap_axis: Optional axis name for parallel map operations.
            
        Returns:
            Tuple of (updated_agent, info_dict) where info_dict contains training metrics.
        """
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
                "entropy": -dist.log_prob(pi_actions).mean(),
                "log_probs": log_probs,
                "pi_actions": pi_actions,
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),
            }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        """
        Sample actions from policy given observations.
        
        Args:
            observations: Observation tensor of shape (batch_size, obs_dim).
            seed: JAX random key for stochastic sampling.
            temperature: Temperature for policy scaling (>1 increases entropy).
            argmax: If True, return mode; else sample stochastically.
            
        Returns:
            Action tensor of shape (batch_size, action_dim).
        """
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions
    
    @partial(jax.jit, static_argnames="argmax")
    def sample_actions_and_log_probs(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=True,
    ) -> jnp.ndarray:
        """
        Sample actions and return log probabilities, mean, and variance.
        
        Args:
            observations: Observation tensor.
            seed: JAX random key for stochastic sampling.
            temperature: Temperature for policy scaling.
            argmax: If True, return mode; else sample.
            
        Returns:
            Tuple of (actions, log_probs, mean, variance).
        """
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions, dist.log_prob(actions), dist.mean(), dist.variance()
    
    @partial(jax.jit, static_argnames="argmax")
    def policy_actions_and_log_probs(
        self,
        observations: np.ndarray,
        policy_action: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=True,
    ) -> jnp.ndarray:
        """
        Compute log probabilities of given actions under current policy.
        
        Args:
            observations: Observation tensor.
            policy_action: Actions to evaluate under the policy.
            seed: JAX random key (unused).
            temperature: Temperature for policy scaling.
            argmax: Not used.
            
        Returns:
            Tuple of (clipped_log_probs, mean, variance, mode).
        """
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        return jnp.clip(dist.log_prob(policy_action),-100, None), dist.mean(), dist.variance(), dist.mode()

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """
        Compute debug metrics for batches of data.
        
        Args:
            batch: Dictionary with 'observations' and 'actions' keys.
            **kwargs: Additional unused arguments.
            
        Returns:
            Dictionary with mse, log_probs, and pi_actions per sample.
        """
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            # "activations": "relu",
            # "use_layer_norm": True,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        **kwargs,
    ):
        """
        Create and initialize a BC agent.
        
        Args:
            rng: JAX random key for parameter initialization.
            observations: Sample observations for encoder initialization.
            actions: Sample actions for determining action dimension.
            encoder_def: Neural network encoder module.
            network_kwargs: Arguments for MLP (hidden_dims, activations, etc).
            policy_kwargs: Arguments for policy distribution.
            learning_rate: Peak learning rate.
            warmup_steps: Optimizer warmup steps.
            decay_steps: Optimizer decay steps.
            **kwargs: Additional unused arguments.
            
        Returns:
            Initialized BCAgent instance.
        """
        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            )
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, lr_schedule)
