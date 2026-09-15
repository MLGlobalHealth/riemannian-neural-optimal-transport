"""Intrinsic Riemannian barycenters and entropic barycentric maps.

Provides scan-based (JIT/autodiff-friendly) routines for computing
Fréchet means on Cartan-Hadamard manifolds and entropic barycentric
projections of discrete conditional laws.
"""

import jax
import jax.numpy as jnp
from typing import Any, Optional


def _normalize_weights(w: jnp.ndarray) -> jnp.ndarray:
    """Normalize non-negative weights to sum to 1."""
    w = jnp.asarray(w)
    w = jnp.maximum(w, 0.0)
    return w / jnp.sum(w)


def barycenter_objective(
    manifold: Any,
    z: jnp.ndarray,
    ys: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Fréchet objective: 0.5 * sum_j w_j d(z, y_j)^2."""
    d2 = jax.vmap(lambda y: manifold.dist(z, y) ** 2)(ys)
    return 0.5 * jnp.dot(weights, d2)


def intrinsic_barycenter(
    ys: jnp.ndarray,
    weights: jnp.ndarray,
    manifold: Any,
    z0: Optional[jnp.ndarray] = None,
    num_steps: int = 32,
    step_size: float = 0.5,
) -> jnp.ndarray:
    """Approximate the intrinsic Fréchet/Karcher mean via Riemannian gradient
    descent using ``jax.lax.scan`` (JIT and autodiff friendly).

        argmin_z  0.5 * sum_j w_j d(z, y_j)^2

    Args:
        ys: (K, D) support points on the manifold.
        weights: (K,) non-negative weights (will be normalized).
        manifold: object exposing ``log``, ``exponential_map``, ``projx``.
        z0: optional initial point; defaults to the heaviest support point.
        num_steps: number of gradient descent iterations.
        step_size: step size (1.0 = full Riemannian gradient step).

    Returns:
        z: (D,) approximate Fréchet mean.
    """
    ys = jnp.asarray(ys)
    w = _normalize_weights(weights)

    if z0 is None:
        z0 = ys[jnp.argmax(w)]

    def one_step(z, _):
        logs = jax.vmap(lambda y: manifold.log(z, y))(ys)
        v = jnp.einsum("k,k...->...", w, logs)
        z_next = manifold.exponential_map(z, step_size * v)
        z_next = manifold.projx(z_next)
        return z_next, z_next

    z_final, _ = jax.lax.scan(one_step, z0, xs=None, length=num_steps)
    return z_final


def entropic_conditional_weights(
    x: jnp.ndarray,
    psi_params: Any,
    psi_module: Any,
    y_samples: jnp.ndarray,
    manifold: Any,
    epsilon: float,
    log_target_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Discrete entropic conditional weights (Gibbs kernel).

    Computes softmax over logits:
        (psi(y_j) - c(x, y_j)) / epsilon  [+ log(alpha_j)]

    Args:
        x: (D,) source point.
        psi_params: parameters of the dual potential network.
        psi_module: Flax module with ``.apply({"params": ...}, y)``.
        y_samples: (K, D) target support points.
        manifold: manifold with ``dist`` method.
        epsilon: entropic regularization strength.
        log_target_weights: optional (K,) log-masses of target support.

    Returns:
        w: (K,) conditional weights summing to 1.
    """
    psi_y = psi_module.apply({"params": psi_params}, y_samples).reshape(-1)
    costs = jax.vmap(lambda y: 0.5 * manifold.dist(x, y) ** 2)(y_samples)
    logits = (psi_y - costs) / epsilon

    if log_target_weights is not None:
        logits = logits + log_target_weights

    return jax.nn.softmax(logits, axis=0)


def intrinsic_entropic_barycentric_map(
    x: jnp.ndarray,
    psi_params: Any,
    psi_module: Any,
    manifold: Any,
    y_samples: jnp.ndarray,
    epsilon: float,
    log_target_weights: Optional[jnp.ndarray] = None,
    z0: Optional[jnp.ndarray] = None,
    num_steps: int = 32,
    step_size: float = 0.5,
):
    """Intrinsic barycentric projection of the entropic conditional law.

    Computes the discrete entropic conditional weights and then finds
    their intrinsic Fréchet mean on the manifold.

    Args:
        x: (D,) source point.
        psi_params: dual-potential network parameters.
        psi_module: Flax module for the dual potential.
        manifold: Riemannian manifold.
        y_samples: (K, D) target support points.
        epsilon: entropic regularization.
        log_target_weights: optional (K,) log-masses of target support.
        z0: optional initial point for the barycenter solver.
        num_steps: number of Riemannian gradient steps.
        step_size: gradient step size.

    Returns:
        z: (D,) intrinsic barycenter.
        w: (K,) conditional weights.
    """
    w = entropic_conditional_weights(
        x=x,
        psi_params=psi_params,
        psi_module=psi_module,
        y_samples=y_samples,
        manifold=manifold,
        epsilon=epsilon,
        log_target_weights=log_target_weights,
    )

    if z0 is None:
        z0 = y_samples[jnp.argmax(w)]

    z = intrinsic_barycenter(
        ys=y_samples,
        weights=w,
        manifold=manifold,
        z0=z0,
        num_steps=num_steps,
        step_size=step_size,
    )
    return z, w
