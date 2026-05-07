"""
Multilayer perceptron network architectures.

Provides MLP, MLPResNet, and related network blocks for use in RL agents.
Supports various normalization and regularization techniques.
"""
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from backend.common.initialization import init_fns


class MLP(nn.Module):
    """
    Multi-layer perceptron.
    
    Standard feedforward network with configurable hidden dimensions, 
    activations, normalization, and regularization.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions.
        activations: Activation function (callable or string like 'relu').
        activate_final: Whether to apply activation to final layer.
        use_layer_norm: Whether to use layer normalization.
        use_group_norm: Whether to use group normalization.
        dropout_rate: Dropout rate (0-1) or None for no dropout.
        kernel_init_type: Type of kernel initialization.
        kernel_scale_final: Optional scale factor for final layer initialization.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.relu
    activate_final: bool = True
    use_layer_norm: bool = False
    use_group_norm: bool = False
    dropout_rate: Optional[float] = None
    kernel_init_type: Optional[str] = None
    kernel_scale_final: Optional[float] = None

    def setup(self):
        assert not (self.use_layer_norm and self.use_group_norm)
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor.
            train: Whether in training mode (enables dropout).

        Returns:
            Output tensor after all hidden layers (and optional final activation).
        """
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        for i, size in enumerate(self.hidden_dims):

            # optinally final layer have different init scale
            if i + 1 == len(self.hidden_dims) and self.kernel_scale_final is not None:
                x = nn.Dense(size, kernel_init=self.init_fn(self.kernel_scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=self.init_fn())(x)

            # normalization and activation after each layer
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                elif self.use_group_norm:
                    x = nn.GroupNorm()(x)
                x = activations(x)
        return x


class MLPResNetBlock(nn.Module):
    """
    Residual block for MLPResNet.
    
    Standard residual block with optional layer normalization and dropout.
    Expands dimension by 4x in hidden layer.
    
    Attributes:
        features: Number of features/hidden dimension.
        act: Activation function.
        dropout_rate: Dropout rate or None.
        use_layer_norm: Whether to use layer normalization.
    """
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor.
            train: Whether in training mode (enables dropout).

        Returns:
            Output tensor with residual connection added.
        """
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    """
    Residual MLP network.
    
    MLP constructed from stacked residual blocks. Useful for deep networks
    where residual connections improve training stability.
    
    Attributes:
        num_blocks: Number of residual blocks.
        out_dim: Output dimension.
        dropout_rate: Dropout rate or None.
        use_layer_norm: Whether to use layer normalization.
        hidden_dim: Hidden dimension for residual blocks.
        activations: Activation function (default Swish).
        kernel_init_type: Type of kernel initialization.
    """
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.swish
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Forward pass through stacked residual blocks.

        Args:
            x: Input tensor.
            train: Whether in training mode (enables dropout).

        Returns:
            Output tensor of shape (..., out_dim).
        """
        x = nn.Dense(self.hidden_dim, kernel_init=self.init_fn())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=self.init_fn())(x)
        return x


class Scalar(nn.Module):
    """Learnable scalar parameter module."""

    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        """Return the scalar parameter value."""
        return self.value
