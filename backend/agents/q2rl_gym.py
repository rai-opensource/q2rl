# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Q2RL (Conservative Q-learning with Offline-to-Online RL) agent.

Combines behavioral cloning with Q-learning for offline-to-online RL.
Integrates BC agent with SAC-like critic and policy learning.
"""
import copy
from functools import partial
from typing import Optional, Tuple, Union

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from absl import flags

from backend.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from backend.common.diag_gaussian_entropy import (
    diag_gaussian_entropy,
    legacy_diag_gaussian_entropy,
)
from backend.common.optimizers import make_optimizer
from backend.common.typing import Batch, Data, Params, PRNGKey
from backend.networks.actor_critic_nets import Critic, Policy, ensemblize, ValueCritic
from backend.networks.lagrange import GeqLagrangeMultiplier
from backend.networks.mlp import MLP
from backend.agents.bc import BCAgent
import optax
import random

FLAGS = flags.FLAGS


class Q2RLAgent(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    bc_agent: BCAgent

    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        if jnp.ndim(actions) == 3:
            # forward the q function with multiple actions on each state
            q = jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": grad_params or self.state.params},
                    observations,
                    a,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                ),
                in_axes=1,
                out_axes=-1,
            )(
                actions
            )  # (ensemble_size, batch_size, n_actions)
        else:
            # forward the q function on 1 action on each state
            q = self.state.apply_fn(
                {"params": grad_params or self.state.params},
                observations,
                actions,
                name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
            )  # (ensemble_size, batch_size)

        return q

    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_policy_and_sample(
        self,
        obs: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        repeat=None,
    ):
        rng, sample_rng = jax.random.split(rng)
        action_dist = self.forward_policy(obs, rng, grad_params=grad_params)#, train=False)
        if repeat:
            new_actions, log_pi = action_dist.sample_and_log_prob(
                seed=sample_rng, sample_shape=repeat
            )
            new_actions = jnp.transpose(
                new_actions, (1, 0, 2)
            )  # (batch, repeat, action_dim)
            log_pi = jnp.transpose(log_pi, (1, 0))  # (batch, repeat)
        else:
            new_actions, log_pi = action_dist.sample_and_log_prob(seed=sample_rng)
        return new_actions, log_pi
    
    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    @jax.jit
    def forward_value(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        *,
        train: bool = False,
    ) -> jax.Array:
        """
        Get the option state-value function
        This is never needed in training, only for evaluation
        """
        pi_dist = self.forward_policy(observations, train=False)
        action = pi_dist.mode()
        q = self.forward_critic(observations, action, train=False)
        q = q.min(axis=0)
        return q
    
    @partial(jax.jit, static_argnames="train")
    def forward_value_critic(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for value network.
        Pass grad_params
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="value",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]
        sample_n_actions = (
            self.config["n_actions"] if self.config["max_target_backup"] else None
        )

        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            rng,
            repeat=sample_n_actions,
        )

        if sample_n_actions:
            chex.assert_shape(next_actions_log_probs, (batch_size, sample_n_actions))
        else:
            chex.assert_shape(next_actions_log_probs, (batch_size,))
        return next_actions, next_actions_log_probs

    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """classes that inherit this class can add to this function
        e.g. CQL will add the cql_max_target_backup option
        """
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_next_qs = target_next_qs - temperature * next_actions_log_probs

        if self.config["max_target_backup"]:
            max_target_indices = jnp.expand_dims(
                jnp.argmax(target_next_qs, axis=-1), axis=-1
            )
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)
            next_actions_log_probs = jnp.take_along_axis(
                next_actions_log_probs, max_target_indices, axis=-1
            ).squeeze(-1)

        return target_next_qs

    def bc_differential_entropy(self, dist_var: jnp.ndarray) -> jnp.ndarray:
        """Entropy of the diagonal-Gaussian BC policy, per sample.

        Args:
            dist_var: per-dimension BC policy variances, ``(..., D)``.
        """
        if self.config.get("fix_logprob_entropy", True) and not self.config.get(
            "old_gmm_entropy", False
        ):
            return diag_gaussian_entropy(dist_var)
        return legacy_diag_gaussian_entropy(dist_var)

    def qest_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        action_log_probs, dist_mean, dist_var, bc_action = self.bc_agent.policy_actions_and_log_probs(
            batch["observations"],
            batch["actions"],
            argmax=True,
        )
        differential_entropy = self.bc_differential_entropy(dist_var)
        rng, key = jax.random.split(rng)
        target_q_bc = batch["mc_returns"] + action_log_probs  + differential_entropy
        batch_size = batch["rewards"].shape[0]

        chex.assert_shape(target_q_bc, (batch_size,))

        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )

        # MSE loss
        target_qs = target_q_bc[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
        }

        return critic_loss, info
    
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )
        # (batch_size, ) for sac, (batch_size, cql_n_actions) for cql

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_equal_shape([target_next_min_q, next_actions_log_probs])
        # (batch_size,) for sac, (batch_size, cql_n_actions) for cql

        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )

        # MSE loss
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_q),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"],
            rng=policy_rng,
            grad_params=params,
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)
        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        action_log_probs, dist_mean, dist_var, bc_action = self.bc_agent.policy_actions_and_log_probs(
            batch["observations"],
            batch["actions"],
            argmax=True,
        )
        predicted_q = predicted_qs.min(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        nll_objective = -jnp.mean(
            action_distributions.log_prob(jnp.clip(bc_action, -0.99, 0.99))
        )
        bc_weight = self.config["bc_weight"]
        actor_objective = predicted_q
        actor_loss = (1- bc_weight) * -jnp.mean(actor_objective) + bc_weight * nll_objective

        info = {
            "actor_loss": actor_loss,
            "bc_weight": bc_weight,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "predicted_q_actor" : jnp.mean(predicted_q),
            "bc_log_probs" : action_log_probs.mean(),
            "actor_log_probs" : log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(axis=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
            "actions": actions,
        }

        return actor_loss, info
    
    def qest_policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"],
            rng=policy_rng,
            grad_params=params,
        )
        
        nll_objective = -jnp.mean(
            action_distributions.log_prob(jnp.clip(batch["actions"], -0.99, 0.99))
        )
        actor_loss = nll_objective
        
        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
        }
        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
    
    def value_loss_fn(self, batch, params: Params, rng: PRNGKey):        
        rng, key = jax.random.split(rng)
        v = self.forward_value_critic(batch["observations"], key, grad_params=params)

        value_critic_loss = jnp.mean((v - batch["mc_returns"]) ** 2)

        return value_critic_loss, {"value_critic_loss": value_critic_loss}

    def loss_fns(self, batch, q_est: bool = False):
        if not q_est:
            return {
                "critic": partial(self.critic_loss_fn, batch),
                "actor": partial(self.policy_loss_fn, batch),
                "value": partial(self.value_loss_fn, batch),
                "temperature": partial(self.temperature_loss_fn, batch),
            }
        else:
            return {
                "critic": partial(self.qest_critic_loss_fn, batch),
                "actor": partial(self.qest_policy_loss_fn, batch),
                "value": partial(self.value_loss_fn, batch),
                "temperature": partial(self.temperature_loss_fn, batch),
            }

    @partial(jax.jit, static_argnames=("q_est","pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        q_est: bool = False,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
    ) -> Tuple["Q2RLAgent", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, key = jax.random.split(self.state.rng)

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, q_est)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"


        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

      
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions_og(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        log: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Original sample action function of SAC.
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """
        dist = self.forward_policy(observations,rng=seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            return dist.mode(), None
        else:
            return dist.sample(seed=seed), None
        
    @partial(jax.jit, static_argnames=("argmax", "log"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        log: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions according to Q2RL logic
        """
        # Calculate Q_rl
        dist = self.forward_policy(observations,rng=seed, train=False)
        if argmax: 
            rl_action =  dist.mode()
        else:
            rl_action = dist.sample(seed=seed)
        rl_q =  self.forward_critic(observations, rl_action, train=False).min(axis=0)

        # Calculate Q_bc
        bc_action, log_prob, mean, dist_var = self.bc_agent.sample_actions_and_log_probs(observations, argmax=True)
        differential_entropy = self.bc_differential_entropy(dist_var)
        v = self.forward_value_critic(observations, train=False)
        bc_q =  v + log_prob + differential_entropy

        # Take the action with higher Q value
        final_action = jnp.where((rl_q - bc_q)>0, rl_action, bc_action)

        if log:
            info = {
                "rl_q": jnp.mean(rl_q),
                "bc_q": jnp.mean(bc_q),
                "differential_entropy": jnp.mean(differential_entropy),
                "bc_actions": jnp.mean((bc_q-rl_q>0))
            }
            return final_action, info
        else:
            return final_action
        
    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        rng, critic_rng, actor_rng = jax.random.split(self.state.rng, 3)
        critic_loss, critic_info = self.critic_loss_fn(
            batch, self.state.params, critic_rng
        )
        policy_loss, policy_info = self.policy_loss_fn(
            batch, self.state.params, actor_rng
        )

        metrics = {**critic_info, **policy_info}

        return metrics

    def update_config(self, new_config):
        """update the frozen self.config"""
        object.__setattr__(self, "config", self.config.copy(new_config))

    @classmethod
    def _create_common(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        value_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.99,
        n_actions: int = 10,
        max_target_backup: bool = False,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        # bc loss:
        bc_weight: float = 0.0,
        bc_agent: BCAgent = None,
        # see backend/common/diag_gaussian_entropy.py
        fix_logprob_entropy: bool = True,
        old_gmm_entropy: bool = False,
        **kwargs,
    ):
        """common part of both create() methods.
        for real create, call create() or create_states()"""
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "value": value_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "value": make_optimizer(**critic_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        network_input = observations
        params = model_def.init(
            init_rng,
            actor=[network_input],
            value=[network_input],
            critic=[network_input, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        if target_entropy is None or target_entropy >= 0.0:
            target_entropy = -actions.shape[-1]

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                bc_weight=bc_weight,
                n_actions=n_actions,
                max_target_backup=max_target_backup,
                fix_logprob_entropy=fix_logprob_entropy,
                old_gmm_entropy=old_gmm_entropy,
                **kwargs,
            ),
            bc_agent=bc_agent,
        )

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        bc_agent: BCAgent = None,
        bc_weight: float = 0.0,
        **kwargs,
    ):
        """
        Create a new agent, with no encoders.
        This is the default create.
        Call cls.create_states to create a state-based agent.
        """

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        value_def = ValueCritic(encoders["value"], MLP(**critic_network_kwargs))

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic,
            encoder=encoders["critic"],
            network=critic_backbone,
        )(name="critic")

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        return cls._create_common(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            value_def=value_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            bc_agent=bc_agent,
            bc_weight=bc_weight,
            **kwargs,
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["Q2RLAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[Q2RLAgent], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(
                minibatch,
                pmap_axis=pmap_axis,
                networks_to_update=frozenset({"critic"}),
            )
            return (agent,), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        critic_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        del critic_infos["actor"]
        del critic_infos["temperature"]

        # Take one gradient descent step on the actor and temperature
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"actor", "temperature"}),
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos
