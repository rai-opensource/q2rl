"""
Actor-critic network architectures.

Provides Policy and Critic network modules for RL agents, supporting various
standard deviations parameterizations and ensemble methods.
"""
from typing import Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp

from backend.common.initialization import init_fns


class ValueCritic(nn.Module):
    """
    Value function network (critic).
    
    Outputs a scalar value estimate given observations. Optionally uses an
    encoder to preprocess observations before feeding to the network.
    
    Attributes:
        encoder: Optional encoder module for observation preprocessing.
        network: MLP backbone network.
        init_final: Optional uniform initialization range for final layer.
        kernel_init_type: Type of kernel initialization (e.g., 'lecun_normal').
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass returning scalar value estimate.

        Args:
            observations: Input observations.
            train: Whether in training mode (enables dropout).

        Returns:
            Scalar value estimate of shape (batch_size,).
        """
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        outputs = self.network(obs_enc, train=train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=self.init_fn())(outputs)

        return jnp.squeeze(value, -1)


class Critic(nn.Module):
    """
    State-action critic (Q-function).
    
    Outputs a scalar Q-value given observations and actions. Concatenates
    encoded observations with actions before feeding to network.
    
    Attributes:
        encoder: Optional encoder module for observation preprocessing.
        network: MLP backbone network.
        init_final: Optional uniform initialization range for final layer.
        kernel_init_type: Type of kernel initialization.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass returning scalar Q-value.

        Args:
            observations: Input observations.
            actions: Input actions, concatenated with encoded observations.
            train: Whether in training mode (enables dropout).

        Returns:
            Scalar Q-value of shape (batch_size,).
        """
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train=train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=self.init_fn())(outputs)

        return jnp.squeeze(value, -1)


def ensemblize(cls, num_qs, out_axes=0):
    """
    Create ensemble of network modules via vectorization.
    
    Uses Flax's vmap to create an ensemble where each ensemble member has
    independent parameters. Useful for ensemble methods like REDQ.
    
    Args:
        cls: Module class to ensemblize.
        num_qs: Number of ensemble members.
        out_axes: Output axis for ensemble dimension (default 0).
        
    Returns:
        Ensemblized module with separate parameters per member.
    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
    )


class Policy(nn.Module):
    """
    Stochastic policy network.
    
    Outputs action distribution (mean and std) given observations.
    Supports multiple std parameterizations (exp, softplus, fixed, uniform).
    Can optionally apply tanh squashing for bounded actions.
    
    Attributes:
        encoder: Optional encoder for observation preprocessing.
        network: MLP backbone network.
        action_dim: Dimensionality of action space.
        init_final: Optional uniform initialization for final layer.
        std_parameterization: How to parameterize std ('exp', 'softplus', 'fixed', 'uniform').
        std_min: Minimum std value for clamping.
        std_max: Maximum std value for clamping.
        tanh_squash_distribution: Whether to apply tanh squashing.
        fixed_std: Fixed std value when std_parameterization='fixed'.
        kernel_init_type: Type of kernel initialization.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False
    ) -> distrax.Distribution:
        """
        Forward pass returning an action distribution.

        Args:
            observations: Input observations.
            temperature: Temperature scaling for std (default 1.0).
            train: Whether in training mode (enables dropout).

        Returns:
            Action distribution (TanhMultivariateNormalDiag or MultivariateNormalDiag).
        """
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)

        outputs = self.network(obs_enc, train=train)

        means = nn.Dense(self.action_dim, kernel_init=self.init_fn())(outputs)
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = nn.Dense(self.action_dim, kernel_init=self.init_fn())(
                    outputs
                )

                # # mitsuhiko ablation
                # base_network_output = nn.Dense(2 * self.action_dim, kernel_init=self.init_fn())(
                #     outputs
                # )
                # means, log_stds = jnp.split(base_network_output, 2, axis=-1)
                # log_stds = jnp.clip(log_stds + Scalar(-1.0)(), -20.0, 2.0)

                stds = jnp.exp(log_stds)

            elif self.std_parameterization == "softplus":
                stds = nn.Dense(self.action_dim, kernel_init=self.init_fn())(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            if type(self.fixed_std) == list:
                stds = jnp.array(self.fixed_std)
            else:
                # self.fixed_std is a float
                assert isinstance(
                    self.fixed_std, (int, float)
                ), "fixed std must be a number"
                stds = jnp.array([self.fixed_std] * self.action_dim)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution


class TanhMultivariateNormalDiag(distrax.Transformed):
    """
    Multivariate normal distribution with diagonal covariance and tanh bijector.

    Transforms a MultivariateNormalDiag through tanh (and optional affine rescaling)
    to produce a bounded action distribution suitable for continuous control.
    """

    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        """
        Args:
            loc: Mean of the underlying normal distribution.
            scale_diag: Diagonal standard deviations.
            low: Optional lower bound for affine rescaling after tanh.
            high: Optional upper bound for affine rescaling after tanh.
        """
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        """Return the mode of the distribution (tanh of the normal mode)."""
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        """Return the stddev passed through the bijector (approximate)."""
        return self.bijector.forward(self.distribution.stddev())


class Scalar(nn.Module):
    """Learnable scalar parameter module."""

    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        """Return the scalar parameter value."""
        return self.value
