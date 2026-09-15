from flax import linen as nn
from typing import Tuple, Optional
import jax.numpy as jnp


def build_network(
    phi: nn.Module,
    network_type: str = "mlp",
    hidden_dims: Tuple[int, ...] = (256, 256, 128),
    use_layernorm: bool = True,
    last_scale: float = 1e-1,
    activation: str = "silu",
    leaky_slope: float = 0.01,
    softplus_beta: float = 1.0,
    max_dist: float = jnp.pi,
) -> nn.Module:
    """Factory function to create network based on type.

    Args:
        phi: Embedding module (e.g., GromovDistanceEmbedding)
        network_type: "mlp" or "residual"
        hidden_dims: Tuple of hidden layer widths
        use_layernorm: Whether to use layer normalization
        last_scale: Scale for output layer initialization
        activation: Activation function name
        leaky_slope: Slope for leaky_relu
        softplus_beta: Beta for softplus
        max_dist: Max geodesic distance for normalization (ResidualMLP only)

    Returns:
        Configured network module
    """
    if network_type.lower() == "mlp":
        return MLP(
            phi=phi,
            hidden_dims=hidden_dims,
            use_layernorm=use_layernorm,
            last_scale=last_scale,
            activation=activation,
            leaky_slope=leaky_slope,
            softplus_beta=softplus_beta,
        )
    elif network_type.lower() == "residual":
        return ResidualMLP(
            phi=phi,
            hidden_dims=hidden_dims,
            use_layernorm=use_layernorm,
            last_scale=last_scale,
            activation=activation,
            leaky_slope=leaky_slope,
            softplus_beta=softplus_beta,
            max_dist=max_dist,
        )
    else:
        raise ValueError(f"Unknown network_type: {network_type}. Use 'mlp' or 'residual'.")


def _get_activation(name: str, leaky_slope: float = 0.01, softplus_beta: float = 1.0):
    """Returns activation function based on name"""
    act = name.lower()
    if act == "leaky_relu":
        return lambda u: nn.leaky_relu(u, negative_slope=leaky_slope)
    elif act == "selu":
        return nn.selu
    elif act == "silu":
        return nn.silu
    elif act == "hard_swish":
        return nn.hard_swish
    elif act == "gelu":
        return nn.gelu
    elif act == "softplus":
        return lambda u: nn.softplus(softplus_beta * u) / softplus_beta
    else:
        raise ValueError(f"Unknown activation: {act}")


class MLP(nn.Module):
    """Standard MLP with manifold-aware embedding"""
    phi: nn.Module
    hidden_dims: Tuple[int, ...] = (128, 128)
    out_dim: int = 1
    use_layernorm: bool = True
    last_scale: float = 1e-1
    activation: str = "silu"
    leaky_slope: float = 0.01
    softplus_beta: float = 1.0

    @nn.compact
    def __call__(self, xs):
        z = self.phi(xs)

        if self.use_layernorm:
            z = nn.LayerNorm()(z)

        activation = _get_activation(self.activation, self.leaky_slope, self.softplus_beta)
        h = z

        for width in self.hidden_dims:
            h = nn.Dense(
                width,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.zeros,
            )(h)
            h = activation(h)

        y = nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.normal(self.last_scale),
            bias_init=nn.initializers.zeros
        )(h)

        return y[..., 0] if self.out_dim == 1 else y


class ResidualMLP(nn.Module):
    """MLP with residual connections and normalized landmark distances.

    Features:
    - Normalizes landmark distances by max_dist (π for S2)
    - Residual connections between hidden layers of same width
    - LayerNorm before each residual block
    """
    phi: nn.Module
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    out_dim: int = 1
    use_layernorm: bool = True
    last_scale: float = 1e-1
    activation: str = "silu"
    leaky_slope: float = 0.01
    softplus_beta: float = 1.0
    max_dist: float = jnp.pi  # For S2, max geodesic distance is π

    @nn.compact
    def __call__(self, xs):
        # Get embedding and normalize by max geodesic distance
        z = self.phi(xs) / self.max_dist

        if self.use_layernorm:
            z = nn.LayerNorm()(z)

        activation = _get_activation(self.activation, self.leaky_slope, self.softplus_beta)

        # Project to first hidden dim
        h = nn.Dense(
            self.hidden_dims[0],
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
        )(z)
        h = activation(h)

        # Hidden layers with residual connections
        prev_width = self.hidden_dims[0]
        for width in self.hidden_dims[1:]:
            # Pre-norm residual block
            h_in = h
            if self.use_layernorm:
                h = nn.LayerNorm()(h)

            h = nn.Dense(
                width,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.zeros,
            )(h)
            h = activation(h)

            # Add residual if dimensions match
            if width == prev_width:
                h = h + h_in

            prev_width = width

        # Output layer with small initialization
        y = nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.normal(self.last_scale),
            bias_init=nn.initializers.zeros
        )(h)

        return y[..., 0] if self.out_dim == 1 else y