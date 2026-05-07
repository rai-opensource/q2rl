import dataclasses
import copy

from functools import partial

from typing import Dict, Iterable, Optional, Tuple, OrderedDict, Union

import chex
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from backend.common.encoding import EncodingWrapper
from backend.common.typing import Batch, Data, Params, PRNGKey

from backend.agents.sac_image import SACImageAgent
from backend.vision.data_augmentations import batched_random_crop
from backend.vision.small_encoders import SmallEncoder
from backend.networks.actor_critic_nets import Critic, Policy, ensemblize, ValueCritic
from backend.networks.lagrange import GeqLagrangeMultiplier
from backend.networks.mlp import MLP


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

class IBRLImageAgent(SACImageAgent):

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
        image_keys: Optional[list] = None,
        use_proprio: bool = True,
        padding: int = 4,
        **kwargs,
    ):  
        encoders = {
            image_key: SmallEncoder(
                features=(32, 64, 128, 256),
                kernel_sizes=(3, 3, 3, 3),
                strides=(2, 2, 2, 2),
                padding="VALID",
                pool_method="avg",
                bottleneck_dim=256,
                spatial_block_size=8,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=False,
            image_keys=image_keys,
        )

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

        sac_agent = SACImageAgent._create_common(
                    rng= rng,
                    observations= observations,
                    actions= actions,
                    # Model architecture
                    actor_def=policy_def,
                    critic_def=critic_def,
                    temperature_def=temperature_def,
                    critic_ensemble_size=critic_ensemble_size,
                    critic_subsample_size=critic_subsample_size,
                    padding=padding,
                    image_keys=image_keys,
                    **kwargs,
                )
        agent = cls(state=sac_agent.state, config=sac_agent.config)
        return agent

    def calc_q_scores(
        self,
        observations: Data,
        rl_actions,
        bc_actions,
        seed: Optional[PRNGKey] = None,
        train=False,
    ):
        if seed is not None:
            seed, rl_q_seed = jax.random.split(seed)
            seed, bc_q_seed = jax.random.split(seed)
        else:
            rl_q_seed = None
            bc_q_seed = None
        q_scores_rl = self.forward_target_critic_no_train(
            observations, rl_actions, rng=rl_q_seed, train=train
        )
        q_scores_bc = self.forward_target_critic_no_train(
            observations, bc_actions, rng=bc_q_seed, train=train
        )

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None and seed is not None:
            rng, subsample_key = jax.random.split(seed)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            q_scores_rl = q_scores_rl[subsample_idcs]
            q_scores_bc = q_scores_bc[subsample_idcs]

        

        q_score_rl = q_scores_rl.min(axis=0)
        q_score_bc = q_scores_bc.min(axis=0)

        return q_score_rl, q_score_bc


    def forward_target_critic_no_train(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: PRNGKey,
        train: bool = False,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params, train=train
        )
    
    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        bc_actions = None,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:

        if seed is not None and argmax is not True:
            seed, rl_seed = jax.random.split(seed)
            seed, bc_seed = jax.random.split(seed)
        else:
            rl_seed = None
            bc_seed = None       
        
        rl_actions = super().sample_actions(observations, seed=rl_seed, argmax=argmax)

       
        rl_q_action = rl_actions 

        q_score_rl, q_score_bc = self.calc_q_scores(
            observations,
            rl_actions=rl_q_action,
            bc_actions=bc_actions,
            seed=seed,
            train=False,
        )
        
        ## For image based env, we directly take max
        actions  = jnp.where((q_score_rl - q_score_bc)>0, rl_actions, bc_actions)
        diff  = q_score_rl - q_score_bc
        return actions, diff

    def _compute_next_actions(self, batch, rng, temperature=1.0):
        batch_size = batch["rewards"].shape[0]

        observations = batch["next_observations"]

        next_rl_action_distributions = self.forward_policy(observations, rng=rng)

        rl_rng, rng = jax.random.split(rng)
        (
            next_rl_actions,
            next_rl_actions_log_probs,
        ) = next_rl_action_distributions.sample_and_log_prob(seed=rl_rng)

        next_bc_actions = batch["next_bc_actions"]
        rl_q_action = next_rl_actions
         
        q_score_rl, q_score_bc = self.calc_q_scores(
            observations,
            rl_actions=rl_q_action,
            bc_actions=next_bc_actions,
            seed=rng,
            train=True,
        )

        next_actions = jnp.where(
            q_score_rl.reshape(-1,1) > q_score_bc.reshape(-1,1), next_rl_actions, next_bc_actions
        )

        next_actions_log_probs = next_rl_actions_log_probs
        chex.assert_equal_shape([batch["actions"], next_actions])
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs
    
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
        predicted_q = predicted_qs.min(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        nll_objective = -jnp.mean(
            action_distributions.log_prob(jnp.clip(batch["actions"], -0.99, 0.99))
        )
        actor_objective = predicted_q
        ## Remove entropy regularization
        actor_loss = -jnp.mean(actor_objective) 

        info = {
            "actor_loss": actor_loss,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(axis=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
            "actions": actions,
        }

        return actor_loss, info
