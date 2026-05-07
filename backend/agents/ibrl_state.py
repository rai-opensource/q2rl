import dataclasses
from functools import partial

from typing import Dict, Iterable, Optional, Tuple, OrderedDict, Union

import chex
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from backend.common.typing import Batch, Data, Params, PRNGKey

from backend.agents.sac import SACAgent


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

class IBRLStateAgent(SACAgent):

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
        soft_ibrl_beta: float = 10,
        **kwargs,
    ):

        sac_agent = SACAgent.create(
                    rng= rng,
                    observations= observations,
                    actions= actions,
                    # Model architecture
                    encoder_def= encoder_def,
                    shared_encoder= shared_encoder,
                    critic_network_kwargs= critic_network_kwargs,
                    policy_network_kwargs= policy_network_kwargs,
                    policy_kwargs= policy_kwargs,
                    critic_ensemble_size= critic_ensemble_size,
                    critic_subsample_size= critic_subsample_size,
                    temperature_init= temperature_init,
                    **kwargs,
                )
        sac_agent.config["soft_ibrl_beta"] = soft_ibrl_beta
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
      
        # decide which action to take
        if seed is not None:
            ## IBRL state based soft action selection
            qa  =  jnp.stack([q_score_rl, q_score_bc])
            p_center = jax.nn.softmax(qa * self.config["soft_ibrl_beta"])
            center_idx = jax.random.categorical(key=seed , logits=p_center)
            actions = rl_actions * (1 - center_idx) + bc_actions * center_idx
        else:
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


        qa  =  jnp.stack([q_score_rl, q_score_bc], axis = 1)
        p_center = jax.nn.softmax(qa * self.config["soft_ibrl_beta"])
        action_rng, rng = jax.random.split(rng)
        center_idx = jax.random.categorical(key=rng, logits=p_center)
        expanded_center_idx = jnp.expand_dims(center_idx, axis = 1)
        next_actions = next_rl_actions * (1 - expanded_center_idx) + next_bc_actions * expanded_center_idx
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
            action_distributions.log_prob(jnp.clip(batch["bc_actions"], -0.99, 0.99))
        )
        actor_objective = predicted_q
        ## Remove entropy regularization since IBRL uses TD3 style deterministic policy optimization
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
