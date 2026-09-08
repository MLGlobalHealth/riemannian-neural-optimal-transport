"""Post-training evaluation metrics including KL divergence via IFT on Riemannian manifolds."""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Any

from src.barycenter import intrinsic_entropic_barycentric_map


class KLResult(NamedTuple):
    """Results from KL divergence computation."""
    kl: float                   # KL(T#mu || nu)
    log_mu: float               # E[log mu(x)]
    log_nu: float               # E[log nu(T(x))]
    log_det_J: float            # E[log|det J_T(x)|]
    kl_per_sample: jnp.ndarray  # Per-sample KL values
    ess: float                  # Effective Sample Size
    ess_ratio: float            # ESS / N (normalized, in [0, 1])


def compute_ess(log_weights: jnp.ndarray) -> tuple[float, float]:
    """
    Compute Effective Sample Size from log importance weights.

    ESS measures how many "effective" independent samples we have when
    using importance sampling. For a perfect transport map where T#μ = ν,
    all weights are equal and ESS = N.

    The importance weights for pushforward T#μ targeting ν are:
        w(x) = ν(T(x)) * |det J_T(x)| / μ(x)

    In log space:
        log w(x) = log ν(T(x)) + log|det J_T(x)| - log μ(x)
                 = -kl_per_sample(x)

    ESS = (Σ w)² / Σ w² = 1 / Σ w̃²

    where w̃ = w / Σw are normalized weights.

    Args:
        log_weights: Log importance weights, shape (N,)
                     Note: log_weights = -kl_per_sample

    Returns:
        ess: Effective sample size
        ess_ratio: ESS / N (normalized to [0, 1])
    """
    n = log_weights.shape[0]

    # Normalize log weights using logsumexp for numerical stability
    # log w̃ = log w - log(Σ w)
    log_sum_weights = jax.scipy.special.logsumexp(log_weights)
    log_normalized = log_weights - log_sum_weights

    # ESS = 1 / Σ w̃² = exp(-log(Σ w̃²))
    # log(Σ w̃²) = logsumexp(2 * log w̃)
    log_sum_sq = jax.scipy.special.logsumexp(2 * log_normalized)
    ess = jnp.exp(-log_sum_sq)

    # Clamp to valid range [1, N]
    ess = jnp.clip(ess, 1.0, float(n))
    ess_ratio = ess / n

    return float(ess), float(ess_ratio)


def _psi_scalar(psi_params, psi_module, y):
    """Evaluate psi at a single point."""
    return psi_module.apply({"params": psi_params}, y[None])[0]


def _grad_psi_riemannian(psi_params, psi_module, manifold, y):
    """Compute Riemannian gradient of psi (projected onto tangent space)."""
    g_amb = jax.grad(lambda yy: _psi_scalar(psi_params, psi_module, yy))(y)
    return manifold.tangent_projection(y, g_amb)


def _stationarity(x, y, psi_params, psi_module, manifold):
    """
    Stationarity condition F(x, y) = 0 for the optimal transport map.

    F(x, y) = -log_y(x) - grad_psi(y)

    This is a tangent vector at y. At the optimal y*, F(x, y*) = 0.

    Note: This matches solvers.py:53-55
    """
    # Riemannian gradient of cost c(x,y) = 0.5*d(x,y)^2 w.r.t. y is -log_y(x)
    grad_cost = -manifold.log(y, x)

    # Riemannian gradient of psi at y
    grad_psi = _grad_psi_riemannian(psi_params, psi_module, manifold, y)

    # Stationarity: grad_cost - grad_psi = 0
    F = grad_cost - grad_psi

    # Project to ensure it's in tangent space (should already be, but for safety)
    return manifold.tangent_projection(y, F)


def compute_transport_jacobian_ift(
    x: jnp.ndarray,          # single point (D,)
    y_star: jnp.ndarray,     # transported point (D,)
    psi_params: Any,
    manifold: Any,
    psi_module: Any,
) -> tuple[float, jnp.ndarray]:
    """
    Compute log|det J_T(x)| using Implicit Function Theorem on the manifold.

    The transport map T(x) = y*(x) is defined implicitly by:
        F(x, y*) = -log_{y*}(x) - grad_psi(y*) = 0

    By IFT: J_T = dy*/dx = -[dF/dy]^{-1} [dF/dx]

    We compute this in tangent space coordinates to get a d×d Jacobian
    where d is the intrinsic dimension of the manifold.

    Args:
        x: Source point (D,) in ambient coordinates
        y_star: Transported point T(x) (D,) in ambient coordinates
        psi_params: Neural network parameters
        manifold: Manifold object with Riemannian operations
        psi_module: Flax module for psi

    Returns:
        log_abs_det: log|det J_T(x)| (intrinsic Jacobian determinant)
        J_T: The Jacobian matrix in tangent coordinates (d, d)
    """
    # Get tangent space bases at x and y*
    # tangent_orthonormal_basis expects batch input (B, D) and direction hint
    x_batch = x[None, :]  # (1, D)
    y_batch = y_star[None, :]  # (1, D)

    # Use the log direction as hint for basis orientation
    log_xy = manifold.log(x, y_star)  # tangent at x pointing to y*
    log_yx = manifold.log(y_star, x)  # tangent at y* pointing to x

    E_x = manifold.tangent_orthonormal_basis(x_batch, log_xy[None, :])[0]  # (D, d)
    E_y = manifold.tangent_orthonormal_basis(y_batch, log_yx[None, :])[0]  # (D, d)

    # Define stationarity with fixed parameters
    def F_xy(x_, y_):
        return _stationarity(x_, y_, psi_params, psi_module, manifold)

    # Compute ambient Jacobians
    # dF/dy: (D, D) - how F changes as y changes in ambient space
    dF_dy_amb = jax.jacobian(lambda y_: F_xy(x, y_))(y_star)  # (D, D)

    # dF/dx: (D, D) - how F changes as x changes in ambient space
    dF_dx_amb = jax.jacobian(lambda x_: F_xy(x_, y_star))(x)  # (D, D)

    # Project to tangent coordinates:
    # F outputs a tangent vector at y, so we project with E_y^T
    # x perturbations are in T_x M, so we use E_x
    # y perturbations are in T_y M, so we use E_y

    # dF/dy in tangent coords: E_y^T @ dF_dy_amb @ E_y -> (d, d)
    dF_dy = E_y.T @ dF_dy_amb @ E_y

    # dF/dx in tangent coords: E_y^T @ dF_dx_amb @ E_x -> (d, d)
    dF_dx = E_y.T @ dF_dx_amb @ E_x

    # IFT: J_T = -[dF/dy]^{-1} [dF/dx]
    # Solve dF_dy @ J_T = -dF_dx for J_T
    J_T = -jnp.linalg.solve(dF_dy, dF_dx)  # (d, d)

    # log|det J_T|
    sign, log_abs_det = jnp.linalg.slogdet(J_T)

    return log_abs_det, J_T


def compute_transport_jacobian_autodiff(
    x: jnp.ndarray,          # single point (D,)
    psi_params: Any,
    solver: Any,
    manifold: Any,
    y_samples: Any = None,
) -> tuple[float, jnp.ndarray]:
    """
    Compute log|det J_T(x)| by differentiating through the solver (autodiff).

    For embedded manifolds (Sphere, Product S1×S1, etc.), we compute the ambient
    Jacobian and project to tangent space to get the intrinsic Jacobian.

    Args:
        x: Source point (D,) in ambient coordinates
        psi_params: Neural network parameters
        solver: ArgminSolver instance
        manifold: Manifold object with Riemannian operations
        y_samples: Optional target samples for warm-start (K, D)

    Returns:
        log_abs_det: log|det J_T(x)| (intrinsic Jacobian determinant)
        J_T: The Jacobian matrix in tangent coordinates (d, d)
    """
    def transport(x_):
        y, _ = solver(psi_params, x_, y_samples)
        return y

    # Compute y* = T(x)
    y_star = transport(x)

    # Jacobian in ambient coordinates: dT/dx (D, D)
    J_ambient = jax.jacfwd(transport)(x)

    # Get tangent space bases at x and y*
    log_xy = manifold.log(x, y_star)
    log_yx = manifold.log(y_star, x)

    E_x = manifold.tangent_orthonormal_basis(x[None, :], log_xy[None, :])[0]  # (D, d)
    E_y = manifold.tangent_orthonormal_basis(y_star[None, :], log_yx[None, :])[0]  # (D, d)

    # Project to tangent coordinates: E_y^T @ J_ambient @ E_x -> (d, d)
    J_T = E_y.T @ J_ambient @ E_x

    # log|det J_T|
    sign, log_abs_det = jnp.linalg.slogdet(J_T)

    return log_abs_det, J_T


def compute_kl_divergence(
    psi_params: Any,
    xs: jnp.ndarray,             # (N, D) source samples
    ys: jnp.ndarray,             # (N, D) transported samples T(x)
    base_density: Any,           # has .log_prob
    target_density: Any,         # has .log_prob
    manifold: Any,
    psi_module: Any,
) -> KLResult:
    """
    Compute KL(T#mu || nu) using the pushforward (mu) formulation.

    KL(T#mu || nu) = E_x~mu[log mu(x) - log nu(T(x)) - log|det J_T(x)|]

    The Jacobian is computed in tangent space coordinates using the
    Implicit Function Theorem, giving the correct intrinsic determinant.

    Args:
        psi_params: Trained neural network parameters
        xs: Source samples from mu, shape (N, D)
        ys: Transported samples T(xs), shape (N, D)
        base_density: Source distribution with .log_prob method
        target_density: Target distribution with .log_prob method
        manifold: Manifold object with Riemannian operations
        psi_module: Flax module for the potential psi

    Returns:
        KLResult with KL divergence and component terms
    """

    # Term 1: E[log mu(x)]
    log_mu = base_density.log_prob(xs)  # (N,)

    # Term 2: E[log nu(T(x))]
    log_nu = target_density.log_prob(ys)  # (N,)

    # Term 3: E[log|det J_T(x)|] via IFT (vectorized over samples)
    compute_single = partial(
        compute_transport_jacobian_ift,
        psi_params=psi_params,
        manifold=manifold,
        psi_module=psi_module,
    )
    log_det_J, _ = jax.vmap(compute_single)(xs, ys)  # (N,)

    # KL = E[log mu - log nu - log|det J|]
    kl_per_sample = log_mu - log_nu - log_det_J

    # ESS: importance weights are w = nu(T(x)) * |det J| / mu(x)
    # log w = log nu + log|det J| - log mu = -kl_per_sample
    log_weights = -kl_per_sample
    ess, ess_ratio = compute_ess(log_weights)

    return KLResult(
        kl=float(jnp.mean(kl_per_sample)),
        log_mu=float(jnp.mean(log_mu)),
        log_nu=float(jnp.mean(log_nu)),
        log_det_J=float(jnp.mean(log_det_J)),
        kl_per_sample=kl_per_sample,
        ess=ess,
        ess_ratio=ess_ratio,
    )


def compute_transport_jacobian_entropic(
    x: jnp.ndarray,          # single point (D,)
    psi_params: Any,
    psi_module: Any,
    manifold: Any,
    y_samples: jnp.ndarray,  # target support (K, D)
    epsilon: float,
    intrinsic: bool = True,
    num_steps: int = 32,
    step_size: float = 0.5,
) -> tuple[float, jnp.ndarray]:
    """
    Compute log|det J_T(x)| for the entropic barycentric projection via autodiff.

    When ``intrinsic=True`` (default), uses the intrinsic Riemannian barycenter:
        T^ε(x) = argmin_z  0.5 Σ_j w_j(x) d(z, y_j)²
    When ``intrinsic=False``, uses the extrinsic projection:
        T^ε(x) = projx( Σ_j w_j(x) y_j )

    Both are fully differentiable so jacfwd works directly.

    Args:
        x: Source point (D,) in ambient coordinates
        psi_params: Neural network parameters
        psi_module: Flax module for the potential psi
        manifold: Manifold object with Riemannian operations
        y_samples: Target support points (K, D)
        epsilon: Entropic regularization strength
        intrinsic: If True, use intrinsic Riemannian barycenter
        num_steps: Number of Riemannian gradient steps (intrinsic mode)
        step_size: Step size for barycenter solver (intrinsic mode)

    Returns:
        log_abs_det: log|det J_T(x)| (intrinsic Jacobian determinant)
        J_T: The Jacobian matrix in tangent coordinates (d, d)
    """
    def transport(x_):
        if intrinsic:
            z, _ = intrinsic_entropic_barycentric_map(
                x=x_,
                psi_params=psi_params,
                psi_module=psi_module,
                manifold=manifold,
                y_samples=y_samples,
                epsilon=epsilon,
                num_steps=num_steps,
                step_size=step_size,
            )
            return z
        else:
            psi_y = psi_module.apply({"params": psi_params}, y_samples)
            costs = jax.vmap(lambda yj: 0.5 * manifold.dist(x_, yj) ** 2)(y_samples)
            logits = (psi_y - costs) / epsilon
            weights = jax.nn.softmax(logits)
            y_avg = jnp.einsum('k,kd->d', weights, y_samples)
            return manifold.projx(y_avg)

    y_star = transport(x)

    # Jacobian in ambient coordinates: dT/dx (D, D)
    J_ambient = jax.jacfwd(transport)(x)

    # Project to tangent space
    log_xy = manifold.log(x, y_star)
    log_yx = manifold.log(y_star, x)

    E_x = manifold.tangent_orthonormal_basis(x[None, :], log_xy[None, :])[0]  # (D, d)
    E_y = manifold.tangent_orthonormal_basis(y_star[None, :], log_yx[None, :])[0]  # (D, d)

    # Intrinsic Jacobian: E_y^T @ J_ambient @ E_x -> (d, d)
    J_T = E_y.T @ J_ambient @ E_x

    _, log_abs_det = jnp.linalg.slogdet(J_T)

    return log_abs_det, J_T


def compute_kl_divergence_entropic(
    psi_params: Any,
    xs: jnp.ndarray,             # (N, D) source samples
    base_density: Any,           # has .log_prob
    target_density: Any,         # has .log_prob
    psi_module: Any,
    manifold: Any,
    y_samples: jnp.ndarray,     # (K, D) target support for barycentric projection
    epsilon: float,
    intrinsic: bool = True,
    num_steps: int = 32,
    step_size: float = 0.5,
) -> KLResult:
    """
    Compute KL(T#mu || nu) for entropic OT via autodiff through the barycentric projection.

    When ``intrinsic=True`` (default), uses the intrinsic Riemannian barycenter
    for both the Jacobian and the transported points.

    Args:
        psi_params: Trained neural network parameters
        xs: Source samples from mu, shape (N, D)
        base_density: Source distribution with .log_prob method
        target_density: Target distribution with .log_prob method
        psi_module: Flax module for the potential psi
        manifold: Manifold object with Riemannian operations
        y_samples: Target support points (K, D) — typically a fresh sample from nu
        epsilon: Entropic regularization strength
        intrinsic: If True, use intrinsic Riemannian barycenter
        num_steps: Number of Riemannian gradient steps (intrinsic mode)
        step_size: Step size for barycenter solver (intrinsic mode)

    Returns:
        KLResult with KL divergence and component terms
    """
    # Term 1: E[log mu(x)]
    log_mu = base_density.log_prob(xs)  # (N,)

    # Compute transport map and Jacobian via autodiff
    def compute_single(x, y_samp, eps):
        return compute_transport_jacobian_entropic(
            x=x,
            psi_params=psi_params,
            psi_module=psi_module,
            manifold=manifold,
            y_samples=y_samp,
            epsilon=eps,
            intrinsic=intrinsic,
            num_steps=num_steps,
            step_size=step_size,
        )
    log_det_J, _ = jax.vmap(compute_single, in_axes=(0, None, None))(
        xs, y_samples, epsilon
    )  # (N,)

    # Compute transported points T^ε(x)
    def transport_single(x_):
        if intrinsic:
            z, _ = intrinsic_entropic_barycentric_map(
                x=x_, psi_params=psi_params, psi_module=psi_module,
                manifold=manifold, y_samples=y_samples, epsilon=epsilon,
                num_steps=num_steps, step_size=step_size,
            )
            return z
        else:
            psi_y = psi_module.apply({"params": psi_params}, y_samples)
            costs = jax.vmap(lambda yj: 0.5 * manifold.dist(x_, yj) ** 2)(y_samples)
            logits = (psi_y - costs) / epsilon
            weights = jax.nn.softmax(logits)
            y_avg = jnp.einsum('k,kd->d', weights, y_samples)
            return manifold.projx(y_avg)

    ys = jax.vmap(transport_single)(xs)  # (N, D)

    # Term 2: E[log nu(T(x))]
    log_nu = target_density.log_prob(ys)  # (N,)

    # KL = E[log mu - log nu - log|det J|]
    kl_per_sample = log_mu - log_nu - log_det_J

    # ESS
    log_weights = -kl_per_sample
    ess, ess_ratio = compute_ess(log_weights)

    return KLResult(
        kl=float(jnp.mean(kl_per_sample)),
        log_mu=float(jnp.mean(log_mu)),
        log_nu=float(jnp.mean(log_nu)),
        log_det_J=float(jnp.mean(log_det_J)),
        kl_per_sample=kl_per_sample,
        ess=ess,
        ess_ratio=ess_ratio,
    )


def compute_kl_divergence_autodiff(
    psi_params: Any,
    xs: jnp.ndarray,             # (N, D) source samples
    base_density: Any,           # has .log_prob
    target_density: Any,         # has .log_prob
    solver: Any,
    manifold: Any,
    y_samples: Any = None,       # optional warm-start samples
) -> KLResult:
    """
    Compute KL(T#mu || nu) using autodiff through the solver.

    This is an alternative to compute_kl_divergence that uses jacfwd
    to differentiate through the solver iterations instead of IFT.

    Pros: Simpler, no need to derive stationarity condition
    Cons: Higher memory usage (stores intermediate states)

    Args:
        psi_params: Trained neural network parameters
        xs: Source samples from mu, shape (N, D)
        base_density: Source distribution with .log_prob method
        target_density: Target distribution with .log_prob method
        solver: ArgminSolver instance
        manifold: Manifold object with Riemannian operations
        y_samples: Optional target samples for warm-start (K, D)

    Returns:
        KLResult with KL divergence and component terms
    """
    # Term 1: E[log mu(x)]
    log_mu = base_density.log_prob(xs)  # (N,)

    # Compute transport and Jacobian via autodiff
    # NOTE: We must use in_axes to explicitly tell vmap NOT to map over y_samples.
    # When len(xs) == len(y_samples), JAX vmap can incorrectly infer that y_samples
    # should be mapped, causing wrong Jacobian computation.
    def compute_single_jacobian(x, y_samp):
        return compute_transport_jacobian_autodiff(
            x=x,
            psi_params=psi_params,
            solver=solver,
            manifold=manifold,
            y_samples=y_samp,
        )
    # in_axes=(0, None) means: vmap over x (axis 0), broadcast y_samples (None)
    log_det_J, _ = jax.vmap(compute_single_jacobian, in_axes=(0, None))(xs, y_samples)  # (N,)

    # Get transported points (need to call solver again, or extract from autodiff)
    # For simplicity, call solver to get ys
    ys, _ = solver.batch_solve(psi_params, xs, y_samples)

    # Term 2: E[log nu(T(x))]
    log_nu = target_density.log_prob(ys)  # (N,)

    # KL = E[log mu - log nu - log|det J|]
    kl_per_sample = log_mu - log_nu - log_det_J

    # ESS
    log_weights = -kl_per_sample
    ess, ess_ratio = compute_ess(log_weights)

    return KLResult(
        kl=float(jnp.mean(kl_per_sample)),
        log_mu=float(jnp.mean(log_mu)),
        log_nu=float(jnp.mean(log_nu)),
        log_det_J=float(jnp.mean(log_det_J)),
        kl_per_sample=kl_per_sample,
        ess=ess,
        ess_ratio=ess_ratio,
    )