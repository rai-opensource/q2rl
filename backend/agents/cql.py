"""
Implementation of CQL in continuous action spaces.
"""
import copy
from functools import partial
from typing import Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from overrides import overrides

from backend.agents.sac import SACAgent
from backend.common.common import JaxRLTrainState, ModuleDict
from backend.common.optimizers import make_optimizer
from backend.common.typing import *
from backend.networks.actor_critic_nets import Critic, Policy, ensemblize
from backend.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from backend.networks.mlp import MLP


class CQLAgent(SACAgent):
    """
    Conservative Q-Learning (CQL) agent for offline RL.

    Extends SACAgent with a CQL penalty that penalizes Q-values of OOD actions,
    keeping them below in-distribution Q-values. Supports both fixed and
    auto-tuned CQL alpha via a Lagrange multiplier, and Cal-QL (a variant that
    lower-bounds Q-values using MC returns).
    """

    def forward_cql_alpha_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the CQL alpha Lagrange multiplier
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="cql_alpha_lagrange",
        )

    def _get_cql_q_diff(
        self, batch, rng: PRNGKey, grad_params: Optional[Params] = None
    ):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        batch_size = batch["rewards"].shape[0]
        q_pred = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng,
            grad_params=grad_params,
        )
        chex.assert_shape(q_pred, (self.config["critic_ensemble_size"], batch_size))

        """sample random actions"""
        action_dim = batch["actions"].shape[-1]
        rng, action_rng = jax.random.split(rng)
        if self.config["cql_action_sample_method"] == "uniform":
            cql_random_actions = jax.random.uniform(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        elif self.config["cql_action_sample_method"] == "normal":
            cql_random_actions = jax.random.normal(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
            )
        else:
            raise NotImplementedError

        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            batch["observations"],
            current_a_rng,
            repeat=self.config["cql_n_actions"],
        )
        chex.assert_shape(
            cql_current_log_pis, (batch_size, self.config["cql_n_actions"])
        )

        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            batch["next_observations"],
            next_a_rng,
            repeat=self.config["cql_n_actions"],
        )

        all_sampled_actions = jnp.concatenate(
            [
                cql_random_actions,
                cql_current_actions,
                cql_next_actions,
            ],
            axis=1,
        )

        """q values of randomly sampled actions"""
        rng, q_rng = jax.random.split(rng)
        cql_q_samples = self.forward_critic(
            batch["observations"],
            all_sampled_actions,  # this is being vmapped over in sac.py
            q_rng,
            grad_params=grad_params,
            train=True,
        )
        chex.assert_shape(
            cql_q_samples,
            (
                self.config["critic_ensemble_size"],
                batch_size,
                self.config["cql_n_actions"] * 3,
            ),
        )

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            cql_q_samples = cql_q_samples[subsample_idcs]
            q_pred = q_pred[subsample_idcs]
            critic_size = self.config["critic_subsample_size"]
        else:
            critic_size = self.config["critic_ensemble_size"]
        """Cal-QL"""
        if self.config["use_calql"]:
            if self.config["calql_bound_random_actions"]:
                 lower_bound = jnp.repeat(
                    batch["mc_returns"].reshape(-1, 1),
                    self.config["cql_n_actions"] * 3,
                    axis=1,
                )
            else:
                fake_lower_bound = jnp.repeat(
                    jnp.ones_like(batch["mc_returns"].reshape(-1, 1)) * (-jnp.inf),
                    self.config["cql_n_actions"],
                    axis=1,
                )
                mc_lower_bound = jnp.repeat(
                    batch["mc_returns"].reshape(-1, 1),
                    self.config["cql_n_actions"] * 2,
                    axis=1,
                )
                mc_lower_bound = jnp.concatenate(
                    [fake_lower_bound, mc_lower_bound], axis=1
                )
            chex.assert_shape(
                mc_lower_bound, (batch_size, self.config["cql_n_actions"] * 3)
            )

            num_vals = jnp.size(cql_q_samples)
            calql_bound_rate = jnp.sum(cql_q_samples < mc_lower_bound) / num_vals
            cql_q_samples = jnp.maximum(cql_q_samples, mc_lower_bound)

        if self.config["cql_importance_sample"]:
            random_density = jnp.log(0.5**action_dim)

            importance_prob = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        random_density, (batch_size, self.config["cql_n_actions"])
                    ),
                    cql_current_log_pis,
                    cql_next_log_pis,  # this order matters, should match all_sampled_actions
                ],
                axis=1,
            )
            cql_q_samples = cql_q_samples - importance_prob  # broadcast over dim 0
        else:
            cql_q_samples = jnp.concatenate(
                [
                    cql_q_samples,
                    jnp.expand_dims(q_pred, -1),
                ],
                axis=-1,
            )
            cql_q_samples -= jnp.log(cql_q_samples.shape[-1]) * self.config["cql_temp"]
            chex.assert_shape(
                cql_q_samples,
                (
                    critic_size,
                    batch_size,
                    3 * self.config["cql_n_actions"] + 1,
                ),
            )

        """log sum exp of the ood actions"""
        cql_ood_values = (
            jax.scipy.special.logsumexp(
                cql_q_samples / self.config["cql_temp"], axis=-1
            )
            * self.config["cql_temp"]
        )
        chex.assert_shape(cql_ood_values, (critic_size, batch_size))

        cql_q_diff = cql_ood_values - q_pred
        info = {
            "cql_ood_values": cql_ood_values.mean(),
        }
        if self.config["use_calql"]:
            info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

    @overrides
    def _compute_next_actions(self, batch, rng):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        sample_n_actions = (
            self.config["cql_n_actions"]
            if self.config["cql_max_target_backup"]
            else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            rng,
            repeat=sample_n_actions,
        )
        return next_actions, next_actions_log_probs

    @overrides
    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config["cql_max_target_backup"]:
            max_target_indices = jnp.expand_dims(
                jnp.argmax(target_next_qs, axis=-1), axis=-1
            )
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)
            next_actions_log_probs = jnp.take_along_axis(
                next_actions_log_probs, max_target_indices, axis=-1
            ).squeeze(-1)

        assert not self.config["backup_entropy"], "Need to call the super() fn"

        return target_next_qs

    @overrides
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """add CQL loss on top of SAC loss"""
        if self.config["use_td_loss"]:
            td_loss, td_loss_info = super().critic_loss_fn(batch, params, rng)
        else:
            td_loss, td_loss_info = 0.0, {}

        if self.config["use_cql_loss"]:

            cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(
                batch, rng, params
            )

            """auto tune cql alpha"""
            if self.config["cql_autotune_alpha"]:
                alpha = self.forward_cql_alpha_lagrange()
                cql_loss = (cql_q_diff - self.config["cql_target_action_gap"]).mean()
            else:
                alpha = self.config["cql_alpha"]
                cql_loss = jnp.clip(
                    cql_q_diff,
                    self.config["cql_clip_diff_min"],
                    self.config["cql_clip_diff_max"],
                ).mean()

            critic_loss = td_loss + alpha * cql_loss
            cql_loss_info = {
                "cql_loss": cql_loss,
                "cql_alpha": alpha,
                "cql_diff": cql_q_diff.mean(),
                **cql_intermediate_results,
            }
        else:
            critic_loss = td_loss
            cql_loss_info = {}

        info = {
            **td_loss_info,
            **cql_loss_info,
            "critic_loss": critic_loss,
            "td_loss": td_loss,
        }

        return critic_loss, info

    def cql_alpha_lagrange_penalty(
        self, qvals_diff, *, grad_params: Optional[Params] = None
    ):
        """
        Compute the Lagrange penalty for CQL alpha auto-tuning.

        Args:
            qvals_diff: Difference between OOD and in-distribution Q-values.
            grad_params: Optional non-default parameters for gradient computation.

        Returns:
            Scalar Lagrange penalty value.
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=qvals_diff,
            rhs=self.config["cql_target_action_gap"],
            name="cql_alpha_lagrange",
        )

    def cql_alpha_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """recompute cql_q_diff without gradients (not optimal for runtime)"""
        cql_q_diff, _ = self._get_cql_q_diff(batch, rng)

        cql_alpha_loss = self.cql_alpha_lagrange_penalty(
            qvals_diff=cql_q_diff.mean(),
            grad_params=params,
        )
        lmbda = self.forward_cql_alpha_lagrange()

        return cql_alpha_loss, {
            "cql_alpha_loss": cql_alpha_loss,
            "cql_alpha_lagrange_multiplier": lmbda,
        }

    @overrides
    def loss_fns(self, batch):
        """
        Return loss functions for all networks, including CQL alpha lagrange if enabled.

        Returns:
            Dict mapping network names to partially applied loss functions.
        """
        losses = super().loss_fns(batch)
        if self.config["cql_autotune_alpha"]:
            losses["cql_alpha_lagrange"] = partial(self.cql_alpha_loss_fn, batch)

        return losses

    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        networks_to_update: set = set({"actor", "critic"}),
    ):
        """update super() to perhaps include updating CQL lagrange multiplier"""
        if not isinstance(networks_to_update, frozenset):
            if self.config["autotune_entropy"]:
                networks_to_update.add("temperature")
            if self.config["cql_autotune_alpha"]:
                networks_to_update.add("cql_alpha_lagrange")

        return super().update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset(networks_to_update),
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgent", dict]:
        """
        same as super().update_high_utd, but also considers the CQL alpha lagrange loss
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[SACAgent], data: Tuple[Batch]):
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

        # Take one gradient descent step on the actor, temperature, and cql_alpha_lagrange
        networks_to_update = set(("actor", "temperature"))
        if self.config["cql_autotune_alpha"]:  # only diff from super().update_high_utd
            networks_to_update.add("cql_alpha_lagrange")
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset(networks_to_update),
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model arch
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        critic_network_type: str = "mlp",
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
        **kwargs,
    ):
        """
        Create a new CQLAgent with actor, critic, temperature, and optional CQL alpha networks.

        Args:
            rng: Random number generator key.
            observations: Example observations for network initialization.
            actions: Example actions for network initialization.
            encoder_def: Encoder module for observation preprocessing.
            shared_encoder: Whether actor and critic share the encoder.
            critic_network_type: Type of critic network ('mlp').
            critic_network_kwargs: Keyword arguments for critic MLP.
            policy_network_kwargs: Keyword arguments for policy MLP.
            policy_kwargs: Additional policy kwargs (e.g. std_parameterization).
            **kwargs: Additional config kwargs (discount, CQL hyperparams, etc.).

        Returns:
            A new CQLAgent instance.
        """
        # update algorithm config
        config = ConfigDict(kwargs)

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
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

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )

        critic_def = partial(
            Critic,
            encoder=encoders["critic"],
            network=critic_backbone,
        )(name="critic")

        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )
        if config["cql_autotune_alpha"]:
            cql_alpha_lagrange_def = LeqLagrangeMultiplier(
                init_value=config.cql_alpha_lagrange_init,
                constraint_shape=(),
                name="cql_alpha_lagrange",
            )

        # model def
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        if config["cql_autotune_alpha"]:
            networks["cql_alpha_lagrange"] = cql_alpha_lagrange_def
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
        }
        if config["cql_autotune_alpha"]:
            txs["cql_alpha_lagrange"] = make_optimizer(
                **config.cql_alpha_lagrange_otpimizer_kwargs
            )

        # init params
        rng, init_rng = jax.random.split(rng)
        extra_kwargs = {}
        if config["cql_autotune_alpha"]:
            extra_kwargs["cql_alpha_lagrange"] = []
        network_input = observations
        params = model_def.init(
            init_rng,
            actor=[network_input],
            critic=[network_input, actions],
            temperature=[],
            **extra_kwargs,
        )["params"]

        # create
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # config
        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config = flax.core.FrozenDict(config)
        return cls(state, config)
