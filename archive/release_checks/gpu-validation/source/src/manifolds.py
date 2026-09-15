import os
import re
import itertools
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
try:
    from spherical_kde import SphericalKDE
except ImportError:
    SphericalKDE = None


eps = 1e-8


def _safe_sinc(x):
    """
    Compute sin(x)/x safely, using Taylor expansion near 0.

    sin(x)/x = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...

    For autodiff stability:
    - Use Taylor expansion for |x| < 0.1 (threshold chosen so Taylor is accurate)
    - Use exact formula sin(x)/x for |x| >= 0.1
    - CRITICAL: Both branches must have bounded gradients because jnp.where
      computes gradients for BOTH branches during autodiff. When differentiating
      through 1000+ solver iterations, unbounded gradients in the "unselected"
      branch cause NaN accumulation.
    - Use jnp.maximum(x², 0.01) to ensure x_abs >= 0.1 even in the Taylor-selected
      regime, giving bounded gradients for the exact branch.

    At x=0.1: Taylor = 0.998334166..., exact = 0.998334166... (15 digits match)
    """
    x2 = x**2
    # Taylor: 1 - x²/6 + x⁴/120 (accurate to O(x⁶) ≈ 10⁻¹² for x=0.1)
    taylor = 1.0 - x2 / 6.0 + x2 * x2 / 120.0

    # Exact formula: use jnp.maximum to ensure bounded gradients in both branches.
    # When x² < 0.01, we select Taylor, but the exact branch gradient is still computed.
    # Without clamping, d/dx(1/sqrt(x²)) ~ 1/x² -> infinity as x->0.
    # By using max(x², 0.01), we ensure x_abs >= 0.1 -> bounded gradient.
    x2_safe = jnp.maximum(x2, 0.01)  # Same threshold as branch selection
    x_abs = jnp.sqrt(x2_safe)
    exact = jnp.sin(x) / x_abs

    return jnp.where(x2 < 0.01, taylor, exact)  # x² < 0.01 means |x| < 0.1


def _normalize(v, axis=-1, eps=1e-8):
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)


@dataclass
class Manifold(ABC):
    D: int  # Dimension of the ambient Euclidean space

    @abstractmethod
    def exponential_map(self, x, v):
        pass

    @abstractmethod
    def log(self, x, y):
        pass

    @abstractmethod
    def tangent_projection(self, x, v):
        pass

    @abstractmethod
    def projx(self, x):
        pass

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def tangent_orthonormal_basis(self, x, dF):
        pass

    def barycenter(self, weights, points, n_iters=5, lr=1.0):
        """Riemannian barycenter via gradient descent on the manifold.

        Minimises f(z) = Σ w_j d²(z, y_j) by iterating:
            v = Σ w_j log_z(y_j),   z ← exp_z(lr · v)

        Uses ``jax.lax.scan`` so the loop is JIT and autodiff friendly.

        Args:
            weights: (K,) non-negative weights summing to 1
            points:  (K, D) target points on the manifold
            n_iters: number of gradient steps
            lr:      step size (1.0 = full Riemannian gradient step)

        Returns:
            z: (D,) approximate Fréchet mean
        """
        z0 = self.projx(jnp.einsum('k,kd->d', weights, points))

        def one_step(z, _):
            logs = jax.vmap(lambda y: self.log(z, y))(points)
            v = jnp.einsum('k,kd->d', weights, logs)
            z_next = self.exponential_map(z, lr * v)
            z_next = self.projx(z_next)
            return z_next, z_next

        z_final, _ = jax.lax.scan(one_step, z0, xs=None, length=n_iters)
        return z_final


# -------------------------
# Euclidean: R^n with flat metric
# -------------------------
@dataclass
class Euclidean(Manifold):

    def exponential_map(self, x, v):
        return x + v

    def log(self, x, y):
        return y - x

    def tangent_projection(self, x, v):
        return v

    def dist(self, x, y):
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x (B,D), y (D,M) -> (B,M)
            diff = x[:, :, None] - y[None, :, :]  # (B, D, M)
            return jnp.sqrt(jnp.sum(diff**2, axis=1) + eps)
        return jnp.sqrt(jnp.sum((x - y)**2, axis=-1) + eps)

    def cost(self, x, y):
        d = self.dist(x, y)
        return 0.5 * d**2

    def projx(self, x):
        return x

    def tangent_orthonormal_basis(self, x, dF):
        B, D = x.shape
        return jnp.broadcast_to(jnp.eye(D), (B, D, D))


# -------------------------
# Sphere: supports S^n embedded in R^{n+1} by setting D=n+1
# -------------------------
@dataclass
class Sphere(Manifold):
    jitter: float = 1e-8

    # plotting grid resolution
    NUM_POINTS: int = 200

    # plotting-only cached grid (NumPy arrays)
    _plot_grid_ready: bool = field(default=False, init=False, repr=False)
    _theta_grid: np.ndarray = field(default=None, init=False, repr=False)  # (2N,)
    _phi_grid: np.ndarray = field(default=None, init=False, repr=False)  # (N,)
    _tp_grid: np.ndarray = field(
        default=None, init=False, repr=False
    )  # (2N*N, 2) [theta,phi]

    def _init_plot_grid(self):
        """Create a spherical grid in (theta, phi) matching utils.* convention."""
        N = int(self.NUM_POINTS)

        # theta: [-pi, pi), phi: [0, pi]
        theta = np.linspace(-np.pi, np.pi, 2 * N, endpoint=False)
        phi = np.linspace(0.0, np.pi, N, endpoint=True)

        tt, pp = np.meshgrid(theta, phi, indexing="ij")  # (2N, N)
        tp = np.stack([tt.reshape(-1), pp.reshape(-1)], -1)  # (2N*N, 2)

        self._theta_grid = theta
        self._phi_grid = phi
        self._tp_grid = tp
        self._plot_grid_ready = True

    def exponential_map(self, x, v):
        # x: (..., D), v: (..., D)
        # Normalize x first (defensive against numerical drift)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        # Project v to tangent space (formula requires v ⟂ x)
        v = self.tangent_projection(x, v)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return x * jnp.cos(v_norm) + v * _safe_sinc(v_norm)

    def log(self, x, y):
        """
        Riemannian log map on S^{D-1}.
        x,y: (..., D) broadcastable
        returns: (..., D) tangent vectors at x

        Uses atan2 for numerical stability in both forward and backward passes.
        The arccos formulation has gradient singularity d/dx arccos(x) = -1/sqrt(1-x²)
        which blows up when x→±1, causing autodiff through solver to explode for
        points near poles.
        """
        # Normalize to unit sphere (defensive against numerical drift)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

        # Tangent direction: u = y - <x,y>*x (unnormalized, in tangent space at x)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)  # (..., 1)
        u = y - xy * x  # (..., D)

        # ||u|| = sin(theta), add eps inside sqrt for gradient stability at u=0
        u_norm = jnp.sqrt(jnp.sum(u**2, axis=-1, keepdims=True) + eps)  # (..., 1)

        # Stable theta via atan2: theta = atan2(sin(theta), cos(theta)) = atan2(||u||, xy)
        # Unlike arccos, atan2 has bounded gradients everywhere
        theta = jnp.arctan2(u_norm, xy)  # (..., 1)

        # log_x(y) = theta * (u / ||u||) = (theta / ||u||) * u
        # For small ||u||, use Taylor: theta/sin(theta) ≈ 1 + theta²/6
        small = u_norm < 1e-6
        coef_small = 1.0 + (theta**2) / 6.0
        safe_u_norm = jnp.maximum(u_norm, 1e-6)  # Same threshold as 'small'
        coef_large = theta / safe_u_norm
        coef = jnp.where(small, coef_small, coef_large)

        return coef * u

    def tangent_projection(self, x, u):
        # batch-safe projection: u - <x,u>x
        # Normalize x first (defensive against numerical drift)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        xu = jnp.sum(x * u, axis=-1, keepdims=True)
        return u - xu * x

    def dist(self, x, y):
        """
        Geodesic distance on the sphere using numerically stable atan2 formula.

        Uses: dist = 2 * atan2(||x - y||, ||x + y||)
        This is stable both when x ≈ y (near 0) and x ≈ -y (near π).
        (arccos explodes near x·y→1, arcsin explodes near antipodes)

        Inputs are normalized to unit vectors first (defensive against numerical drift).

        Supports:
          x: (B,D), y: (D,M)  -> (B,M)   (landmark case)
          x: (B,D), y: (B,D)  -> (B,)
          x: (...,D), y: (...,D) -> (...)
        """
        # Normalize to unit sphere (formula requires unit vectors)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x (B,D), y (D,M) -> (B,M)
            # Normalize y along axis=0 (each column is a landmark)
            y = y / (jnp.linalg.norm(y, axis=0, keepdims=True) + eps)
            diff = x[:, :, None] - y[None, :, :]  # (B, D, M)
            summ = x[:, :, None] + y[None, :, :]  # (B, D, M)
            # eps inside sqrt for gradient stability at ||v||=0
            norm_diff = jnp.sqrt(jnp.sum(diff**2, axis=1) + eps)  # (B, M)
            norm_sum = jnp.sqrt(jnp.sum(summ**2, axis=1) + eps)   # (B, M)
        else:
            # Standard case: normalize y along last axis
            y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)
            diff = x - y
            summ = x + y
            norm_diff = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)
            norm_sum = jnp.sqrt(jnp.sum(summ**2, axis=-1) + eps)

        return 2 * jnp.arctan2(norm_diff, norm_sum)

    def cost(self, x, y):
        d = self.dist(x, y)
        return 0.5 * d**2

    def projx(self, x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def tangent_orthonormal_basis(self, x, dF):
        """
        Returns (B, D, D-1) orthonormal tangent basis at x.
        Special-case D=3 (S^2) to avoid QR; fallback uses vmapped QR.
        """
        assert x.ndim == 2 and dF.ndim == 2 and x.shape == dF.shape
        B, D = x.shape
        n = D - 1

        # Normalize x first (defensive against numerical drift)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

        # u0: align with tangent gradient when possible, else fallback
        dF_tan = self.tangent_projection(x, dF)
        nrm = jnp.linalg.norm(dF_tan, axis=-1, keepdims=True)

        I = jnp.eye(D)
        e0 = jnp.broadcast_to(I[0], (B, D))
        e1 = jnp.broadcast_to(I[1 % D], (B, D))

        u0a = self.tangent_projection(x, e0)
        u0b = self.tangent_projection(x, e1)
        na = jnp.linalg.norm(u0a, axis=-1, keepdims=True)
        nb = jnp.linalg.norm(u0b, axis=-1, keepdims=True)
        # Safe normalize both branches BEFORE jnp.where to ensure valid gradients
        # (jnp.where evaluates gradients for both branches during autodiff)
        u0a_safe = u0a / jnp.maximum(na, eps)
        u0b_safe = u0b / jnp.maximum(nb, eps)
        u0_fb = jnp.where(na >= nb, u0a_safe, u0b_safe)

        u0_from_dF = dF_tan / (nrm + eps)
        u0 = jnp.where(nrm > eps, u0_from_dF, u0_fb)  # (B,D)

        # ---- S^2 case: D=3 => tangent dimension n=2
        if D == 3:
            # u1 = normalize(x × u0); if degenerate, use x × fallback
            u1 = jnp.cross(x, u0)
            u1n = jnp.linalg.norm(u1, axis=-1, keepdims=True)

            u1_fb = jnp.cross(x, u0_fb)
            u1_fb = _normalize(u1_fb, eps=eps)

            u1 = jnp.where(u1n > eps, u1 / (u1n + eps), u1_fb)
            E = jnp.stack([u0, u1], axis=-1)  # (B,3,2)
            return E

        # ---- General D: complete basis via projected canonical basis + vmapped QR
        C = jnp.broadcast_to(I, (B, D, D))  # (B,D,D)
        xTC = jnp.einsum("bi,bij->bj", x, C)  # (B,D)
        Ctan = C - jnp.einsum("bi,bj->bij", x, xTC)  # tangent projection

        u0TC = jnp.einsum("bi,bij->bj", u0, Ctan)
        Crem = Ctan - jnp.einsum("bi,bj->bij", u0, u0TC)

        # IMPORTANT: vmap QR over batch (avoid batched QR custom_call)
        def qr_one(A):
            Q, R = jnp.linalg.qr(A)
            return Q, R

        Q, _ = jax.vmap(qr_one)(Crem)  # Q: (B,D,D)
        rest = Q[..., : max(n - 1, 0)]  # (B,D,n-1)
        E = jnp.concatenate([u0[..., None], rest], axis=-1)  # (B,D,n)
        return E

    def zero(self):
        y = jnp.zeros((self.D,))
        y = y.at[0].set(-1.0)
        return y

    def zero_like(self, x):
        # x can be (D,) or (B,D)
        y = jnp.zeros_like(x)
        y = y.at[..., 0].set(-1.0)
        return y

    def squeeze_tangent(self, v):
        # tangent at zero() is {0} x R^{D-1}
        return v[..., 1:]

    def unsqueeze_tangent(self, w):
        # w: (..., D-1) -> (..., D) with leading 0
        return jnp.concatenate((jnp.zeros_like(w[..., :1]), w), axis=-1)

    def transp(self, x, y, u):
        # parallel transport on sphere along minimal geodesic (when defined)
        # Near cut locus (xy ≈ -1), transport is undefined; we clamp to avoid explosion
        # Normalize x and y first (defensive against numerical drift)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)
        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        # Clamp denominator away from zero (cut locus)
        denom = jnp.maximum(1.0 + xy, 0.01)
        return u - yu / denom * (x + y)

    def logdetexp(self, x, u):
        """
        Log determinant of exponential map Jacobian on S^{D-1}.

        Formula: (D-2) * log(sin(r)/r) where r = ||u||.

        Numerical stability: For high-dimensional spheres (e.g., S^100), the
        multiplier (D-2)=99 amplifies floating point errors. We use Taylor
        expansion log(sinc(r)) ≈ -r²/6 - r⁴/180 + O(r⁶) for small r, which
        avoids catastrophic cancellation in log(≈1). Error is O(r⁶) ≈ 10⁻¹²
        for r < 0.01. This is standard practice (cf. log1p, numpy.sinc).
        """
        r = jnp.linalg.norm(u, axis=-1)

        # Taylor expansion for small r (avoids log(1-ε) cancellation)
        small = jnp.abs(r) < 1e-2
        log_sinc_taylor = -r**2 / 6.0 - r**4 / 180.0

        # Direct computation for larger r, with safe clipping
        # Use abs(sin(r)) for correct |det| when r > π
        sinc_r = jnp.abs(jnp.sin(r)) / jnp.clip(r, 1e-10, None)
        sinc_r = jnp.clip(sinc_r, 1e-10, 1.0)  # |sinc| ∈ (0, 1]
        log_sinc_direct = jnp.log(sinc_r)

        val = jnp.where(small, log_sinc_taylor, log_sinc_direct)
        return (u.shape[-1] - 2) * val

    def plot_samples(self, model_samples, kde_factor=0.1, save="t.png"):
        """
        model_samples: (B,3) points on S^2 (can be JAX array).
        """
        if not self._plot_grid_ready:
            self._init_plot_grid()

        # Convert to JAX then to NumPy in spherical coords (theta, phi)
        spherical = utils.euclidean_to_spherical(
            jnp.asarray(model_samples)
        )  # (B,2) JAX
        spherical = np.asarray(spherical)  # NumPy for KDE

        kde = SphericalKDE(
            spherical[:, 0],  # theta
            spherical[:, 1],  # phi (colatitude)
            bandwidth=float(kde_factor),
        )

        heatmap = np.exp(
            kde(self._tp_grid[:, 0], self._tp_grid[:, 1]).reshape(
                2 * self.NUM_POINTS, self.NUM_POINTS
            )
        )

        self.plot_mollweide(heatmap, save=save)

    def plot_density(self, log_prob_fn, save="t.png"):
        """
        log_prob_fn: callable accepting (N,3) JAX array and returning (N,) log-density.
        """
        if not self._plot_grid_ready:
            self._init_plot_grid()

        # Grid points to Euclidean using your utils convention
        tp_jax = jnp.asarray(self._tp_grid)  # (2N*N,2)
        xyz = utils.spherical_to_euclidean(tp_jax)  # (2N*N,3) JAX

        density = jnp.exp(log_prob_fn(xyz))  # (2N*N,) JAX
        heatmap = np.asarray(density).reshape(2 * self.NUM_POINTS, self.NUM_POINTS)

        self.plot_mollweide(heatmap, save=save)

    def plot_mollweide(self, heatmap, save):
        """
        heatmap: (2N, N) NumPy array
        """
        if not self._plot_grid_ready:
            self._init_plot_grid()

        # Mollweide expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
        lon = self._theta_grid  # theta already in [-pi, pi)
        lat = (np.pi / 2.0) - self._phi_grid  # latitude = pi/2 - colatitude

        Lon, Lat = np.meshgrid(lon, lat, indexing="ij")  # (2N, N)

        fig = plt.figure(figsize=(3, 2), dpi=200)
        ax = fig.add_subplot(111, projection="mollweide")
        norm = matplotlib.colors.Normalize()

        ax.pcolormesh(Lon, Lat, heatmap, cmap="magma", norm=norm, shading="auto")
        ax.set_axis_off()

        plt.savefig(save, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # optional trim (ignore if ImageMagick not installed)
        try:
            os.system(f"convert {save} -trim {save} >/dev/null 2>&1")
        except Exception:
            pass


# -------------------------
# SO(3): Rotation group as unit quaternions (q ≡ -q identified)
# -------------------------
@dataclass
class SO3(Manifold):
    """
    SO(3) represented by unit quaternions q = [w, x, y, z], modulo q ≡ -q.

    Geometry convention:
      - dist(x, y) is the physical rotation angle in [0, pi]
      - ||log_x(y)|| equals that rotation angle
      - exp_x(v) interprets ||v|| as a rotation angle

    Points live in ambient R^4, tangent vectors are represented as ambient
    tangent vectors in R^4 orthogonal to q.
    """
    D: int = 4

    def _normalize(self, q):
        return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)

    def _to_canonical(self, q):
        """
        Normalize and choose the representative with nonnegative scalar part.
        """
        q = self._normalize(q)
        return jnp.where(q[..., :1] < 0.0, -q, q)

    def projx(self, x):
        return self._to_canonical(x)

    def tangent_projection(self, x, v):
        x = self._normalize(x)
        xv = jnp.sum(x * v, axis=-1, keepdims=True)
        return v - xv * x

    def exponential_map(self, x, v):
        """
        Riemannian exp for the rotation-angle metric.

        Since quaternions use half-angles internally, this is the sphere exp
        evaluated at v / 2.
        """
        x = self._to_canonical(x)
        v = self.tangent_projection(x, v)

        r = jnp.linalg.norm(v, axis=-1, keepdims=True)      # physical angle
        half_r = 0.5 * r

        # exp_SO3(x, v) = exp_S3(x, v/2)
        y = x * jnp.cos(half_r) + 0.5 * v * _safe_sinc(half_r)
        return self._to_canonical(y)

    def log(self, x, y):
        """
        Riemannian log for the rotation-angle metric.

        Returns v in T_x with ||v|| = physical rotation angle in [0, pi].
        """
        x = self._to_canonical(x)
        y = self._normalize(y)

        # choose the closer representative of y or -y
        dot = jnp.sum(x * y, axis=-1, keepdims=True)
        y = jnp.where(dot < 0.0, -y, y)

        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        xy = jnp.clip(xy, 0.0, 1.0)

        u = y - xy * x
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)

        # theta is the quaternion half-angle
        theta = jnp.arctan2(u_norm, xy)

        small = u_norm < 1e-6
        coef_small = 2.0 * (1.0 + theta**2 / 6.0)
        coef_large = 2.0 * theta / jnp.maximum(u_norm, 1e-6)
        coef = jnp.where(small, coef_small, coef_large)

        return coef * u

    def dist(self, x, y):
        """
        Geodesic distance = physical rotation angle in [0, pi].

        Pairwise:
          x: (..., 4), y: (..., 4) -> (...)

        Landmark:
          x: (B, 4), y: (4, M) -> (B, M)
        """
        x = self._normalize(x)

        # landmark case: x (B,4), y (4,M) -> (B,M)
        if x.ndim == 2 and y.ndim == 2 and x.shape != y.shape and x.shape[1] == 4 and y.shape[0] == 4:
            y = y / (jnp.linalg.norm(y, axis=0, keepdims=True) + eps)
            dot = x @ y  # (B, M)
            sign = jnp.sign(dot + eps)  # hemisphere selection
            y_adj = y[None, :, :] * sign[:, None, :]  # (B, 4, M)
            diff = x[:, :, None] - y_adj
            summ = x[:, :, None] + y_adj
            norm_diff = jnp.sqrt(jnp.clip(jnp.sum(diff**2, axis=1), 0.0))
            norm_sum = jnp.sqrt(jnp.clip(jnp.sum(summ**2, axis=1), eps))
            return 4.0 * jnp.arctan2(norm_diff, norm_sum)

        # pairwise case
        y = self._normalize(y)
        dot = jnp.sum(x * y, axis=-1, keepdims=True)
        y = jnp.where(dot < 0.0, -y, y)
        diff = x - y
        summ = x + y
        norm_diff = jnp.sqrt(jnp.clip(jnp.sum(diff**2, axis=-1), 0.0))
        norm_sum = jnp.sqrt(jnp.clip(jnp.sum(summ**2, axis=-1), eps))
        return 4.0 * jnp.arctan2(norm_diff, norm_sum)

    def cost(self, x, y):
        d = self.dist(x, y)
        return 0.5 * d**2

    def tangent_orthonormal_basis(self, x, dF):
        """
        Returns (B, 4, 3) ambient-orthonormal tangent basis at x.

        At unit quaternion q, the vectors q*i, q*j, q*k form an orthonormal
        basis of T_q S^3, and under this convention they also represent the
        natural SO(3) rotation-angle tangent directions.
        """
        assert x.ndim == 2 and x.shape[-1] == 4

        q = self._normalize(x)
        w, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        e1 = jnp.stack([-qx,  w,  qz, -qy], axis=-1)
        e2 = jnp.stack([-qy, -qz,  w,  qx], axis=-1)
        e3 = jnp.stack([-qz,  qy, -qx,  w], axis=-1)

        return jnp.stack([e1, e2, e3], axis=-1)  # (B, 4, 3)

    def zero(self):
        return jnp.array([1.0, 0.0, 0.0, 0.0])

    def zero_like(self, x):
        z = jnp.zeros_like(x)
        return z.at[..., 0].set(1.0)

    def squeeze_tangent(self, v):
        """
        At the identity, T_e SO(3) = {0} x R^3.
        """
        return v[..., 1:]

    def unsqueeze_tangent(self, w):
        """
        (..., 3) -> (..., 4), tangent at identity.
        """
        return jnp.concatenate([jnp.zeros_like(w[..., :1]), w], axis=-1)

    def transp(self, x, y, u):
        """
        Parallel transport along the minimal geodesic.

        Same formula as on the sphere; constant metric scaling does not change
        the Levi-Civita connection.
        """
        x = self._to_canonical(x)
        y = self._normalize(y)

        dot = jnp.sum(x * y, axis=-1, keepdims=True)
        y = jnp.where(dot < 0.0, -y, y)
        y = self._normalize(y)

        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)

        out = u - yu / jnp.maximum(1.0 + xy, eps) * (x + y)
        return self.tangent_projection(y, out)

    def logdetexp(self, x, u):
        """
        Log Jacobian determinant of the exponential map in rotation-angle coords.

        For r = ||u||:
          det(d exp_x(u)) = (sin(r/2) / (r/2))^2
        """
        r = jnp.linalg.norm(u, axis=-1)
        half_r = 0.5 * r

        small = jnp.abs(r) < 1e-2
        log_sinc_half_taylor = -r**2 / 24.0 - r**4 / 2880.0

        sinc_half = jnp.abs(jnp.sin(half_r)) / jnp.maximum(half_r, 1e-10)
        sinc_half = jnp.clip(sinc_half, 1e-10, 1.0)
        log_sinc_half_direct = jnp.log(sinc_half)

        val = jnp.where(small, log_sinc_half_taylor, log_sinc_half_direct)
        return 2.0 * val


# -------------------------
# SE(3): Rigid body motions = SO(3) ⋉ R³
# Representation: [q0, q1, q2, q3, t1, t2, t3] ∈ R⁷
#   - q ∈ S³ (unit quaternion, SO(3) double cover)
#   - t ∈ R³ (translation)
# -------------------------
@dataclass
class SE3(Manifold):
    """
    Product-manifold SE(3) = SO(3) x R^3, represented as [q, t] in R^7.

    IMPORTANT:
      This is the product-manifold geometry, not the Lie-group twist exp/log.

    Metric:
      d^2((q, t), (q', t')) = alpha^2 * d_SO3(q, q')^2 + ||t - t'||^2

    where d_SO3 is the physical rotation-angle distance.
    """
    D: int = 7
    alpha: float = 1.0

    def __post_init__(self):
        self._so3 = SO3()

    def _split(self, x):
        return x[..., :4], x[..., 4:]

    def _join(self, q, t):
        return jnp.concatenate([q, t], axis=-1)

    def projx(self, x):
        q, t = self._split(x)
        q = self._so3.projx(q)
        return self._join(q, t)

    def tangent_projection(self, x, v):
        q, _ = self._split(x)
        vq, vt = self._split(v)
        vq = self._so3.tangent_projection(q, vq)
        return self._join(vq, vt)

    def exponential_map(self, x, v):
        """
        Product-manifold exponential:
          - SO(3) exponential on rotation
          - Euclidean addition on translation
        """
        q, t = self._split(x)
        vq, vt = self._split(v)

        q_new = self._so3.exponential_map(q, vq)
        t_new = t + vt
        return self._join(q_new, t_new)

    def log(self, x, y):
        """
        Product-manifold logarithm:
          - SO(3) logarithm on rotation
          - Euclidean subtraction on translation
        """
        qx, tx = self._split(x)
        qy, ty = self._split(y)

        vq = self._so3.log(qx, qy)
        vt = ty - tx
        return self._join(vq, vt)

    def dist(self, x, y):
        """
        Geodesic distance for the weighted product metric.

        Pairwise:
          x: (..., 7), y: (..., 7) -> (...)

        Landmark:
          x: (B, 7), y: (7, M) -> (B, M)
        """
        alpha2 = self.alpha**2

        # landmark case
        if x.ndim == 2 and y.ndim == 2 and x.shape != y.shape and x.shape[1] == 7 and y.shape[0] == 7:
            qx, tx = x[:, :4], x[:, 4:]   # (B,4), (B,3)
            qy, ty = y[:4, :], y[4:, :]   # (4,M), (3,M)

            d_rot2 = self._so3.dist(qx, qy) ** 2
            dt = tx[:, :, None] - ty[None, :, :]
            d_trans2 = jnp.sum(dt**2, axis=1)

            return jnp.sqrt(jnp.maximum(alpha2 * d_rot2 + d_trans2, 0.0))

        qx, tx = self._split(x)
        qy, ty = self._split(y)

        d_rot2 = self._so3.dist(qx, qy) ** 2
        d_trans2 = jnp.sum((tx - ty) ** 2, axis=-1)

        return jnp.sqrt(jnp.maximum(alpha2 * d_rot2 + d_trans2, 0.0))

    def cost(self, x, y):
        """Squared geodesic distance / 2, bypassing sqrt for gradient stability."""
        alpha2 = self.alpha**2

        # landmark case
        if x.ndim == 2 and y.ndim == 2 and x.shape != y.shape and x.shape[1] == 7 and y.shape[0] == 7:
            qx, tx = x[:, :4], x[:, 4:]
            qy, ty = y[:4, :], y[4:, :]
            d_rot2 = self._so3.dist(qx, qy) ** 2
            dt = tx[:, :, None] - ty[None, :, :]
            d_trans2 = jnp.sum(dt**2, axis=1)
            return 0.5 * (alpha2 * d_rot2 + d_trans2)

        qx, tx = self._split(x)
        qy, ty = self._split(y)
        d_rot2 = self._so3.dist(qx, qy) ** 2
        d_trans2 = jnp.sum((tx - ty) ** 2, axis=-1)
        return 0.5 * (alpha2 * d_rot2 + d_trans2)

    def tangent_orthonormal_basis(self, x, dF):
        """
        Returns (B, 7, 6) basis orthonormal for the SE(3) product metric.

        Because the metric weights rotation by alpha^2, the rotational basis
        columns are scaled by 1 / alpha.
        """
        assert x.ndim == 2 and x.shape[-1] == 7
        B = x.shape[0]

        q, _ = self._split(x)
        E_rot = self._so3.tangent_orthonormal_basis(q, dF[..., :4]) / self.alpha  # (B,4,3)
        E_trans = jnp.broadcast_to(jnp.eye(3), (B, 3, 3))                          # (B,3,3)

        top = jnp.concatenate([E_rot, jnp.zeros((B, 4, 3))], axis=-1)              # (B,4,6)
        bot = jnp.concatenate([jnp.zeros((B, 3, 3)), E_trans], axis=-1)            # (B,3,6)
        return jnp.concatenate([top, bot], axis=1)                                  # (B,7,6)

    def zero(self):
        return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def zero_like(self, x):
        z = jnp.zeros_like(x)
        return z.at[..., 0].set(1.0)

    def squeeze_tangent(self, v):
        """
        At the identity, return a 6-vector whose Euclidean norm matches the
        SE(3) product metric norm.
        """
        vq, vt = self._split(v)
        return jnp.concatenate([self.alpha * vq[..., 1:], vt], axis=-1)

    def unsqueeze_tangent(self, w):
        """
        Inverse of squeeze_tangent at the identity.
        """
        wq = w[..., :3] / self.alpha
        wt = w[..., 3:]
        vq = jnp.concatenate([jnp.zeros_like(wq[..., :1]), wq], axis=-1)
        return jnp.concatenate([vq, wt], axis=-1)

    def transp(self, x, y, u):
        """
        Product-manifold parallel transport.
        """
        qx, _ = self._split(x)
        qy, _ = self._split(y)
        uq, ut = self._split(u)

        uq_transp = self._so3.transp(qx, qy, uq)
        return self._join(uq_transp, ut)

    def logdetexp(self, x, u):
        """
        Product manifold:
          logdetexp_SE3 = logdetexp_SO3 + 0
        """
        q, _ = self._split(x)
        uq, _ = self._split(u)
        return self._so3.logdetexp(q, uq)


# -------------------------
# Product: fixes slicing so it works for batched (B,D)
# -------------------------
@dataclass
class Product(Manifold):
    manifolds_str: str = "S1,S1"

    def __post_init__(self):
        self.manifolds = [get(m.strip()) for m in self.manifolds_str.split(",")]
        self.D = sum(m.D for m in self.manifolds)

    def exponential_map(self, x, v):
        parts = []
        d = 0
        for man in self.manifolds:
            xs = x[..., d : d + man.D]
            vs = v[..., d : d + man.D]
            parts.append(man.exponential_map(xs, vs))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def tangent_projection(self, x, u):
        parts = []
        d = 0
        for man in self.manifolds:
            xs = x[..., d : d + man.D]
            us = u[..., d : d + man.D]
            parts.append(man.tangent_projection(xs, us))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def cost(self, x, y):
        # Handle two cases:
        # 1. x: (B,D), y: (D,M) -> (B,M)  [landmark case]
        # 2. x: (B,D), y: (B,D) -> (B,)   [standard case]
        is_landmark_case = (x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1])

        cost_prod = 0.0
        d = 0
        for man in self.manifolds:
            if is_landmark_case:
                # x[:, d:d+D] is (B, D_i), y[d:d+D, :] is (D_i, M)
                cost_prod = cost_prod + man.cost(x[:, d : d + man.D], y[d : d + man.D, :])
            else:
                # Both x and y have same shape, use standard slicing
                cost_prod = cost_prod + man.cost(x[..., d : d + man.D], y[..., d : d + man.D])
            d += man.D
        return cost_prod

    def dist(self, x, y):
        """
        Product distance: sqrt(sum_i dist_i^2).

        Supports:
          x: (B, D), y: (D, M)  -> (B, M)   [landmark case]
          x: (B, D), y: (B, D)  -> (B,)     [standard pairwise]
          x: (..., D), y: (..., D) -> (...)  [broadcasted]
        """
        # Detect landmark case: x (B, D), y (D, M)
        is_landmark = (x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1])

        acc = 0.0
        d = 0
        for man in self.manifolds:
            xi = x[..., d : d + man.D]
            if is_landmark:
                # Landmark case: slice rows of y, not columns
                yi = y[d : d + man.D, :]
            else:
                yi = y[..., d : d + man.D]
            di = man.dist(xi, yi)
            acc = acc + di**2
            d += man.D
        return jnp.sqrt(acc + eps)

    def tangent_orthonormal_basis(self, x, dF):
        d = 0
        blocks = []
        for man in self.manifolds:
            blocks.append(
                man.tangent_orthonormal_basis(x[:, d : d + man.D], dF[:, d : d + man.D])
            )
            d += man.D
        # batch block diag
        map_block_diag = jax.vmap(block_diag)
        return map_block_diag(*blocks)

    def projx(self, x):
        # Handle both single point (D,) and batch (B, D)
        parts = []
        d = 0
        for man in self.manifolds:
            parts.append(man.projx(x[..., d : d + man.D]))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def zero(self):
        parts = []
        for man in self.manifolds:
            if not hasattr(man, "zero"):
                raise NotImplementedError(
                    f"{type(man)} has no zero() needed by WrappedNormal."
                )
            parts.append(man.zero())  # (D_i,)
        return jnp.concatenate(parts, axis=-1)  # (D,)

    def zero_like(self, x):
        # x can be (D,) or (B,D)
        parts = []
        d = 0
        for man in self.manifolds:
            xs = x[..., d : d + man.D]
            if not hasattr(man, "zero_like"):
                raise NotImplementedError(
                    f"{type(man)} has no zero_like() needed by WrappedNormal."
                )
            parts.append(man.zero_like(xs))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def log(self, x, y):
        # x,y: (..., D) -> (..., D) tangent vectors concatenated
        parts = []
        d = 0
        for man in self.manifolds:
            xi = x[..., d : d + man.D]
            yi = y[..., d : d + man.D]
            if not hasattr(man, "log"):
                raise NotImplementedError(
                    f"{type(man)} has no log() needed by WrappedNormal."
                )
            parts.append(man.log(xi, yi))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def transp(self, x, y, u):
        # parallel transport: componentwise on product
        parts = []
        d = 0
        for man in self.manifolds:
            xi = x[..., d : d + man.D]
            yi = y[..., d : d + man.D]
            ui = u[..., d : d + man.D]
            if not hasattr(man, "transp"):
                raise NotImplementedError(
                    f"{type(man)} has no transp() needed by WrappedNormal."
                )
            parts.append(man.transp(xi, yi, ui))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def squeeze_tangent(self, v):
        # v: (..., D) in tangent at product zero -> (..., sum_i (D_i-1))
        parts = []
        d = 0
        for man in self.manifolds:
            vi = v[..., d : d + man.D]
            if not hasattr(man, "squeeze_tangent"):
                raise NotImplementedError(
                    f"{type(man)} has no squeeze_tangent() needed by WrappedNormal."
                )
            parts.append(man.squeeze_tangent(vi))  # (..., D_i-1)
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def unsqueeze_tangent(self, w):
        # w: (..., sum_i (D_i-1)) -> (..., D) tangent at product zero
        parts = []
        d = 0
        t = 0
        for man in self.manifolds:
            ti = man.D - 1
            wi = w[..., t : t + ti]
            if not hasattr(man, "unsqueeze_tangent"):
                raise NotImplementedError(
                    f"{type(man)} has no unsqueeze_tangent() needed by WrappedNormal."
                )
            parts.append(man.unsqueeze_tangent(wi))  # (..., D_i)
            t += ti
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def logdetexp(self, x, u):
        # log |det DExp_x(u)| for product = sum over factors
        out = 0.0
        d = 0
        for man in self.manifolds:
            xi = x[..., d : d + man.D]
            ui = u[..., d : d + man.D]
            if not hasattr(man, "logdetexp"):
                raise NotImplementedError(
                    f"{type(man)} has no logdetexp() needed by WrappedNormal."
                )
            out = out + man.logdetexp(xi, ui)  # (...,)
            d += man.D
        return out


# -------------------------
# Hyperboloid: H^n in the Lorentz model (Minkowski space R^{n,1})
#
# Points x ∈ R^{n+1} satisfy  -x₀² + x₁² + ... + xₙ² = -1, x₀ > 0.
# Convention: x[0] is the "time" coordinate.
# Ambient dimension D = n+1 (same as Sphere).
# -------------------------
@dataclass
class Hyperboloid(Manifold):
    """Hyperboloid model H^n of curvature -1 in Minkowski space R^{n,1}.

    Points x in R^{n+1} satisfy <x,x>_L = -1, x₀ > 0.
    Convention: x[..., 0] is the "time" coordinate.
    Ambient dimension D = n+1.

    All operations are batch-safe via ``...`` indexing, supporting
    scalar (D,), batched (B,D), and arbitrary leading dimensions.
    """

    def _lorentz_inner(self, u, v):
        """Minkowski inner product: <u,v>_L = -u₀v₀ + u₁v₁ + ... + uₙvₙ.

        Batch-safe: works for any leading dimensions.
        """
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def projx(self, x):
        """Project onto the hyperboloid: keep spatial part, recompute x₀."""
        spatial = x[..., 1:]
        x0 = jnp.sqrt(jnp.sum(spatial * spatial, axis=-1, keepdims=True) + 1.0)
        return jnp.concatenate([x0, spatial], axis=-1)

    def tangent_projection(self, x, v):
        """Project v onto the tangent space at x.

        Tangent space: {v : <x,v>_L = 0}.
        Projection: v + <x,v>_L · x  (since <x,x>_L = -1).
        """
        coeff = self._lorentz_inner(x, v)
        return v + coeff[..., None] * x

    def dist(self, x, y):
        """Geodesic distance on H^n using autodiff-stable log formulation.

        Uses d = log(α + √(α²-1)) instead of arccosh(α), where α = -⟨x,y⟩_L.
        The log formulation has bounded gradients as α → 1⁺ (d → 0), unlike
        arccosh whose derivative 1/√(α²-1) → ∞.  This mirrors the Sphere's
        atan2 trick for arccos.

        Supports:
          x: (D,), y: (D,)       -> scalar
          x: (B,D), y: (B,D)     -> (B,)
          x: (B,D), y: (D,M)     -> (B,M)   (landmark/pairwise case)
        """
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x (B,D), y (D,M)
            inner = -x[:, 0:1] * y[0:1, :] + x[:, 1:] @ y[1:, :]  # (B, M)
            alpha = jnp.maximum(-inner, 1.0)
            # arccosh(α) = log(α + √(α²-1)); clamp inside sqrt for safety
            return jnp.log(alpha + jnp.sqrt(jnp.maximum(alpha * alpha - 1.0, 0.0)) + eps)
        alpha = jnp.maximum(-self._lorentz_inner(x, y), 1.0)
        return jnp.log(alpha + jnp.sqrt(jnp.maximum(alpha * alpha - 1.0, 0.0)) + eps)

    def cost(self, x, y):
        d = self.dist(x, y)
        return 0.5 * d ** 2

    def exponential_map(self, x, v):
        """Exp map: exp_x(v) = cosh(||v||_L) x + sinh(||v||_L)/||v||_L · v."""
        v_norm = jnp.sqrt(jnp.maximum(self._lorentz_inner(v, v), 0.0) + eps)
        return jnp.cosh(v_norm)[..., None] * x + _safe_sinch(v_norm)[..., None] * v

    def log(self, x, y):
        """Log map: log_x(y) = d / ||u||_L · u, where u = y + <x,y>_L · x.

        Uses d = log(α + ||u||_L) where α = cosh(d) and ||u||_L = sinh(d).
        This is the H² analog of the Sphere's atan2(||u||, <x,y>) trick:
        both partial derivatives 1/(α + ||u||_L) are bounded everywhere,
        unlike arccosh'(α) = 1/√(α²-1) which → ∞ as α → 1.
        """
        inner_xy = self._lorentz_inner(x, y)
        alpha = jnp.maximum(-inner_xy, 1.0)
        u = y + inner_xy[..., None] * x  # tangent direction at x
        u_norm = jnp.sqrt(jnp.maximum(self._lorentz_inner(u, u), 0.0) + eps)

        # d = arccosh(α) = log(α + sinh(d)) = log(α + ||u||_L)
        # Gradients: ∂d/∂α = ∂d/∂||u|| = 1/(α + ||u||_L + eps), bounded.
        d = jnp.log(alpha + u_norm + eps)

        # d / u_norm with Taylor branch for small u_norm
        # When u_norm → 0: d ≈ u_norm (to first order), so ratio → 1
        small = u_norm < 1e-6
        safe_u_norm = jnp.maximum(u_norm, 1e-6)
        coef_large = d / safe_u_norm
        coef_small = 1.0 + d ** 2 / 6.0
        coef = jnp.where(small, coef_small, coef_large)
        return coef[..., None] * u

    def tangent_orthonormal_basis(self, x, dF):
        """Returns (B, D, D-1) orthonormal tangent basis at x on H^n.

        Tangent space: {v : <x,v>_L = 0}, dimension n = D-1.
        Uses Gram-Schmidt in Lorentz metric, seeded by projected canonical basis.
        """
        assert x.ndim == 2 and dF.ndim == 2 and x.shape == dF.shape
        B, D = x.shape
        n = D - 1  # intrinsic dimension

        # Project dF to tangent space as seed direction
        dF_tan = self.tangent_projection(x, dF)
        nrm = jnp.sqrt(jnp.maximum(
            jnp.sum(dF_tan ** 2, axis=-1, keepdims=True), 0.0) + eps)

        # Fallback: project e_1 (first spatial coordinate) onto tangent space
        e1 = jnp.zeros((B, D))
        e1 = e1.at[:, 1].set(1.0)
        u0_fb = self.tangent_projection(x, e1)
        u0_fb_norm = jnp.sqrt(jnp.maximum(
            jnp.sum(u0_fb ** 2, axis=-1, keepdims=True), 0.0) + eps)
        u0_fb = u0_fb / jnp.maximum(u0_fb_norm, eps)

        u0 = jnp.where(nrm > eps, dF_tan / jnp.maximum(nrm, eps), u0_fb)

        # Complete to ONB via projected canonical basis + QR
        # Project all D canonical vectors to tangent space at x
        I = jnp.eye(D)
        C = jnp.broadcast_to(I, (B, D, D))  # (B, D, D)

        # Tangent-project each canonical vector
        # x shape (B, D), C shape (B, D, D)
        xTC = jnp.einsum("bi,bij->bj", x, C)  # <x, e_j>_L not quite right
        # Proper Lorentz projection: v_tan = v + <x,v>_L * x
        # <x, e_j>_L for each j: need Lorentz inner
        # For canonical e_j: <x, e_j>_L = -x[0]*delta_{j,0} + x[j]*(1-delta_{j,0})
        lorentz_sign = jnp.ones(D).at[0].set(-1.0)  # (-1, 1, 1, ..., 1)
        xL = x * lorentz_sign[None, :]  # (B, D) — x with Lorentz sign
        inner_xej = jnp.einsum("bi,ji->bj", xL, I)  # (B, D) — <x, e_j>_L
        # Tangent projection: e_j + <x, e_j>_L * x
        Ctan = C + jnp.einsum("bj,bi->bij", inner_xej, x)  # (B, D, D)

        # Remove u0 component
        # Lorentz inner of u0 with each projected canonical vector
        u0L = u0 * lorentz_sign[None, :]  # (B, D)
        u0_dot_Ctan = jnp.einsum("bi,bij->bj", u0L, Ctan)  # (B, D)
        Crem = Ctan - jnp.einsum("bi,bj->bij", u0, u0_dot_Ctan)  # (B, D, D)

        # QR to get orthonormal basis (Euclidean QR is fine as starting point,
        # then we re-project to ensure Lorentz orthogonality)
        def qr_one(A):
            Q, R = jnp.linalg.qr(A)
            return Q

        Q = jax.vmap(qr_one)(Crem)  # (B, D, D)
        rest = Q[..., :n - 1]  # (B, D, n-1)

        # Re-project all vectors to tangent space (QR was Euclidean, not Lorentz)
        # Vectorized: for each column c of rest, project onto tangent space at x.
        # tangent_projection(x, v) = v + <x,v>_L * x
        # rest shape: (B, D, n-1), x shape: (B, D)
        # <x, rest_col>_L for all columns at once:
        xL_rest = x * lorentz_sign[None, :]  # (B, D)
        inner_x_rest = jnp.einsum("bi,bij->bj", xL_rest, rest)  # (B, n-1)
        rest = rest + jnp.einsum("bj,bi->bij", inner_x_rest, x)  # (B, D, n-1)

        # Normalize each column
        rest_norms = jnp.sqrt(jnp.maximum(
            jnp.sum(rest ** 2, axis=1, keepdims=True), 0.0) + eps)
        rest = rest / jnp.maximum(rest_norms, eps)

        E = jnp.concatenate([u0[..., None], rest], axis=-1)  # (B, D, n)
        return E

    # --- WrappedNormal support ---

    def zero(self):
        """Origin of H^n: (1, 0, ..., 0)."""
        z = jnp.zeros(self.D)
        return z.at[0].set(1.0)

    def zero_like(self, x):
        z = jnp.zeros_like(x)
        return z.at[..., 0].set(1.0)

    def squeeze_tangent(self, v):
        """Tangent space at origin is {0} × R^n — drop the time component."""
        return v[..., 1:]

    def unsqueeze_tangent(self, w):
        """w: (..., n) → (..., n+1) tangent at origin with leading 0."""
        return jnp.concatenate([jnp.zeros_like(w[..., :1]), w], axis=-1)

    def transp(self, x, y, u):
        """Parallel transport on H^n along the minimal geodesic from x to y.

        Formula (hyperbolic analogue of the spherical transport):
            P_{x→y}(u) = u + <y, u>_L / (1 − <x, y>_L) · (x + y)

        Note the sign flip vs. the sphere: denominator uses (1 − ⟨x,y⟩_L)
        since ⟨x,y⟩_L ≤ −1 on the hyperboloid, so 1 − ⟨x,y⟩_L ≥ 2.
        """
        inner_yu = self._lorentz_inner(y, u)
        inner_xy = self._lorentz_inner(x, y)
        denom = jnp.maximum(1.0 - inner_xy, 2.0)  # ≥ 2 since <x,y>_L ≤ -1
        return u + (inner_yu / denom)[..., None] * (x + y)

    def logdetexp(self, x, u):
        """Log |det d(exp_x)_u| on H^n.

        Formula: (n − 1) · log(sinh(r)/r) where r = ||u||_L
        and n = D − 1 is the intrinsic dimension.

        sinh(r)/r → 1 as r → 0, so logdetexp → 0 (flat near origin).
        """
        r = jnp.sqrt(jnp.maximum(self._lorentz_inner(u, u), 0.0) + eps)
        n = self.D - 1  # intrinsic dimension

        small = r < 1e-2
        # Taylor: log(sinh(r)/r) = r²/6 + ... (note positive, unlike sphere)
        log_sinch_taylor = r ** 2 / 6.0 + r ** 4 / 180.0

        r_safe = jnp.maximum(r, 1e-10)
        sinch_r = jnp.sinh(r_safe) / r_safe
        log_sinch_direct = jnp.log(jnp.maximum(sinch_r, 1e-10))

        val = jnp.where(small, log_sinch_taylor, log_sinch_direct)
        return (n - 1) * val


# -------------------------
# SPD: Symmetric Positive Definite matrices with affine-invariant metric
#
# Points are n×n SPD matrices, stored as flat (n²,) vectors for framework
# compatibility.  The affine-invariant (Fisher-Rao) metric is:
#   d(P,Q) = || log(P^{-1/2} Q P^{-1/2}) ||_F
# -------------------------
def _sym(X: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise a square matrix."""
    return 0.5 * (X + X.T)


def _eigh_spd(X: jnp.ndarray, lo: float = 1e-10):
    """Eigen-decomposition with eigenvalues clipped above lo."""
    evals, evecs = jnp.linalg.eigh(_sym(X))
    evals = jnp.clip(evals, lo, None)
    return evals, evecs


def _matrix_exp(S: jnp.ndarray) -> jnp.ndarray:
    """Matrix exponential of a symmetric matrix."""
    evals, evecs = jnp.linalg.eigh(_sym(S))
    return _sym(evecs @ jnp.diag(jnp.exp(evals)) @ evecs.T)


def _matrix_log(P: jnp.ndarray, lo: float = 1e-10) -> jnp.ndarray:
    """Matrix logarithm of an SPD matrix."""
    evals, evecs = _eigh_spd(P, lo)
    return _sym(evecs @ jnp.diag(jnp.log(evals)) @ evecs.T)


def _matrix_sqrt(P: jnp.ndarray, lo: float = 1e-10) -> jnp.ndarray:
    """Matrix square root of an SPD matrix."""
    evals, evecs = _eigh_spd(P, lo)
    return _sym(evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T)


def _matrix_invsqrt(P: jnp.ndarray, lo: float = 1e-10) -> jnp.ndarray:
    """Inverse matrix square root of an SPD matrix."""
    evals, evecs = _eigh_spd(P, lo)
    return _sym(evecs @ jnp.diag(1.0 / jnp.sqrt(evals)) @ evecs.T)


@dataclass
class SPD(Manifold):
    """Symmetric Positive Definite manifold with affine-invariant metric.

    Points are n×n SPD matrices stored as flat (D,) = (n²,) vectors.
    Set D = n² where n is the matrix size.
    """
    lo: float = 1e-10  # eigenvalue floor

    @property
    def n(self) -> int:
        """Matrix size n such that D = n²."""
        n = int(round(self.D ** 0.5))
        return n

    def _to_mat(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(self.n, self.n)

    def _to_vec(self, M: jnp.ndarray) -> jnp.ndarray:
        return M.reshape(-1)

    def projx(self, x):
        """Project onto SPD: symmetrise and clip eigenvalues.

        Supports (D,) and (B, D) inputs.
        """
        if x.ndim == 2:
            return jax.vmap(self.projx)(x)
        M = self._to_mat(x)
        evals, evecs = jnp.linalg.eigh(_sym(M))
        evals = jnp.clip(evals, self.lo, None)
        return self._to_vec(_sym(evecs @ jnp.diag(evals) @ evecs.T))

    def tangent_projection(self, x, v):
        """Tangent space of SPD at P is the space of symmetric matrices.

        Supports (D,) and (B, D) inputs.
        """
        if v.ndim == 2:
            return jax.vmap(self.tangent_projection)(x, v)
        V = self._to_mat(v)
        return self._to_vec(_sym(V))

    def dist(self, x, y):
        """Affine-invariant distance: || log(P^{-1/2} Q P^{-1/2}) ||_F.

        Supports:
          x: (D,), y: (D,)   -> scalar
          x: (B,D), y: (B,D) -> (B,)
        """
        if x.ndim == 2 and y.ndim == 2:
            return jax.vmap(self.dist)(x, y)
        P, Q = self._to_mat(x), self._to_mat(y)
        Pinv2 = _matrix_invsqrt(P, self.lo)
        return jnp.linalg.norm(_matrix_log(Pinv2 @ Q @ Pinv2, self.lo), ord="fro")

    def cost(self, x, y):
        """Squared geodesic cost: 0.5 * d(P, Q)².

        Supports:
          x: (B,D), y: (D,M)  -> (B,M)   (landmark/pairwise case)
          Otherwise delegates to 0.5 * dist²
        """
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x (B,D), y (D,M)
            # Reshape y columns as points: (M, D)
            M = y.shape[1]
            y_points = y.T  # (M, D)
            # Compute pairwise costs: (B, M)
            return jax.vmap(
                lambda xi: jax.vmap(lambda yj: 0.5 * self.dist(xi, yj) ** 2)(y_points)
            )(x)
        d = self.dist(x, y)
        return 0.5 * d ** 2

    def exponential_map(self, x, v):
        """Exp map: exp_P(V) = P^{1/2} expm(P^{-1/2} V P^{-1/2}) P^{1/2}.

        Supports (D,) and (B, D) inputs.
        """
        if x.ndim == 2:
            return jax.vmap(self.exponential_map)(x, v)
        P, V = self._to_mat(x), self._to_mat(v)
        P2 = _matrix_sqrt(P, self.lo)
        Pinv2 = _matrix_invsqrt(P, self.lo)
        result = _sym(P2 @ _matrix_exp(Pinv2 @ V @ Pinv2) @ P2)
        return self._to_vec(result)

    def log(self, x, y):
        """Log map: log_P(Q) = P^{1/2} logm(P^{-1/2} Q P^{-1/2}) P^{1/2}.

        Supports (D,) and (B, D) inputs.
        """
        if x.ndim == 2:
            return jax.vmap(self.log)(x, y)
        P, Q = self._to_mat(x), self._to_mat(y)
        P2 = _matrix_sqrt(P, self.lo)
        Pinv2 = _matrix_invsqrt(P, self.lo)
        result = _sym(P2 @ _matrix_log(Pinv2 @ Q @ Pinv2, self.lo) @ P2)
        return self._to_vec(result)

    def tangent_orthonormal_basis(self, x, dF):
        raise NotImplementedError(
            "tangent_orthonormal_basis not implemented for SPD manifold")

    # --- WrappedNormal support ---

    @property
    def tangent_dim(self) -> int:
        """Intrinsic dimension: n(n+1)/2 for symmetric n×n matrices."""
        return self.n * (self.n + 1) // 2

    def zero(self):
        """Identity matrix I_n (flattened)."""
        return jnp.eye(self.n).reshape(-1)

    def zero_like(self, x):
        """Identity matrix with same batch shape as x."""
        I = jnp.eye(self.n).reshape(-1)
        return jnp.broadcast_to(I, x.shape) + jnp.zeros_like(x)

    def squeeze_tangent(self, v):
        """Extract upper-triangle of a symmetric tangent matrix.

        The tangent space at I is the space of symmetric n×n matrices (dim = n(n+1)/2).
        We extract the upper triangle with √2 scaling on off-diagonal entries
        so that the Frobenius inner product is preserved:
            <V, W>_F = Σ_ij V_ij W_ij = Σ_{i≤j} s_ij v_ij w_ij
        where s_ij = 1 if i==j, 2 if i<j.
        """
        n = self.n
        if v.ndim == 1:
            V = v.reshape(n, n)
            idx = jnp.triu_indices(n)
            w = V[idx]
            # Scale off-diagonal by sqrt(2)
            diag_mask = (idx[0] == idx[1]).astype(v.dtype)
            scale = jnp.where(diag_mask, 1.0, jnp.sqrt(2.0))
            return w * scale
        else:
            return jax.vmap(self.squeeze_tangent)(v)

    def unsqueeze_tangent(self, w):
        """Reconstruct symmetric matrix from upper-triangle vector.

        Inverse of squeeze_tangent.
        """
        n = self.n
        if w.ndim == 1:
            idx = jnp.triu_indices(n)
            diag_mask = (idx[0] == idx[1]).astype(w.dtype)
            scale = jnp.where(diag_mask, 1.0, 1.0 / jnp.sqrt(2.0))
            vals = w * scale
            V = jnp.zeros((n, n))
            V = V.at[idx].set(vals)
            # Symmetrise by copying upper to lower (not averaging via _sym)
            V = V + V.T - jnp.diag(jnp.diag(V))
            return V.reshape(-1)
        else:
            return jax.vmap(self.unsqueeze_tangent)(w)

    def transp(self, x, y, u):
        """Parallel transport of tangent vector u from T_x SPD to T_y SPD.

        P_{x→y}(U) = E U E^T  where E = (YX^{-1})^{1/2}
        """
        if u.ndim == 2:
            return jax.vmap(self.transp)(x, y, u)
        X, Y = self._to_mat(x), self._to_mat(y)
        U = self._to_mat(u)
        Xinv = jnp.linalg.inv(_sym(X) + self.lo * jnp.eye(self.n))
        E = _matrix_sqrt(_sym(Y @ Xinv), self.lo)
        result = _sym(E @ U @ E.T)
        return self._to_vec(result)

    def logdetexp(self, x, u):
        """Log |det d(exp_x)_u| on SPD(n) with the affine-invariant metric.

        At the identity (x = I), exp_I(V) = expm(V) and the Jacobian
        determinant in the upper-triangle (intrinsic) coordinates is:

            log|det| = Σ_{i<j} log(sinh((λ_i − λ_j)/2) / ((λ_i − λ_j)/2))

        where λ_1,...,λ_n are eigenvalues of V. For general x, the transport
        to identity is an isometry, so the formula is the same applied to
        X^{-1/2} U X^{-1/2}.
        """
        if u.ndim == 2:
            return jax.vmap(self.logdetexp)(x, u)
        X = self._to_mat(x)
        U = self._to_mat(u)
        Xinv2 = _matrix_invsqrt(X, self.lo)
        V = _sym(Xinv2 @ U @ Xinv2)
        evals = jnp.linalg.eigvalsh(V)  # (n,)
        n = self.n
        logdet = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                half_diff = (evals[i] - evals[j]) / 2.0
                # log(sinh(t)/t): Taylor for small t, direct otherwise
                t = half_diff
                t2 = t ** 2
                small_val = t2 / 6.0 + t2 * t2 / 180.0
                t_safe = jnp.maximum(jnp.abs(t), 1e-6)
                large_val = jnp.log(jnp.abs(jnp.sinh(t_safe)) / t_safe)
                logdet += jnp.where(jnp.abs(t) < 1e-3, small_val, large_val)
        return logdet


@dataclass
class SPDLogEuclidean(Manifold):
    """SPD manifold with the Log-Euclidean metric.

    The Log-Euclidean metric turns SPD(n) into a flat (Euclidean) space via
    the matrix logarithm:  d(P, Q) = || logm(P) - logm(Q) ||_F.

    This makes barycenters closed-form:
        bary(w, {P_j}) = expm( Σ_j w_j logm(P_j) )

    Points are n×n SPD matrices stored as flat (D,) = (n²,) vectors.
    """
    lo: float = 1e-10

    @property
    def n(self) -> int:
        return int(round(self.D ** 0.5))

    def _to_mat(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(self.n, self.n)

    def _to_vec(self, M: jnp.ndarray) -> jnp.ndarray:
        return M.reshape(-1)

    def projx(self, x):
        M = self._to_mat(x)
        evals, evecs = jnp.linalg.eigh(_sym(M))
        evals = jnp.clip(evals, self.lo, None)
        return self._to_vec(_sym(evecs @ jnp.diag(evals) @ evecs.T))

    def tangent_projection(self, x, v):
        V = self._to_mat(v)
        return self._to_vec(_sym(V))

    def dist(self, x, y):
        """Log-Euclidean distance: || logm(P) - logm(Q) ||_F.

        Supports:
          x: (D,), y: (D,)   -> scalar
          x: (B,D), y: (B,D) -> (B,)
        """
        if x.ndim == 2 and y.ndim == 2:
            return jax.vmap(self.dist)(x, y)
        P, Q = self._to_mat(x), self._to_mat(y)
        return jnp.linalg.norm(
            _matrix_log(P, self.lo) - _matrix_log(Q, self.lo), ord="fro")

    def cost(self, x, y):
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            y_points = y.T
            return jax.vmap(
                lambda xi: jax.vmap(lambda yj: 0.5 * self.dist(xi, yj) ** 2)(y_points)
            )(x)
        d = self.dist(x, y)
        return 0.5 * d ** 2

    def exponential_map(self, x, v):
        """Exp map: expm(logm(P) + V), where V is a symmetric matrix in the
        Log-Euclidean tangent representation."""
        P, V = self._to_mat(x), self._to_mat(v)
        return self._to_vec(_matrix_exp(_matrix_log(P, self.lo) + V))

    def log(self, x, y):
        """Log map: logm(Q) - logm(P)."""
        P, Q = self._to_mat(x), self._to_mat(y)
        return self._to_vec(_matrix_log(Q, self.lo) - _matrix_log(P, self.lo))

    def log_euclidean_barycenter(self, weights, points):
        """Closed-form Log-Euclidean barycenter.

        bary(w, {P_j}) = expm( Σ_j w_j logm(P_j) )

        Args:
            weights: (K,) non-negative weights (will be normalized).
            points:  (K, D) SPD matrices as flat vectors.

        Returns:
            z: (D,) the Log-Euclidean Fréchet mean.
        """
        w = weights / jnp.sum(weights)
        mats = points.reshape(-1, self.n, self.n)
        logs = jax.vmap(lambda M: _matrix_log(M, self.lo))(mats)
        mean_log = jnp.einsum("k,kij->ij", w, logs)
        return self._to_vec(_matrix_exp(mean_log))

    def tangent_orthonormal_basis(self, x, dF):
        raise NotImplementedError(
            "tangent_orthonormal_basis not implemented for SPDLogEuclidean")


def _log_sinc(x):
    """log(|sinc(x)|) = log(|sin(x)/x|), numerically stable near x=0.

    Taylor: log(sinc(x)) = -x²/6 - x⁴/180 - x⁶/2835 - ...
    """
    x2 = x ** 2
    taylor = -x2 / 6.0 - x2 * x2 / 180.0
    x_safe = jnp.maximum(jnp.abs(x), 0.1)
    sinc_val = jnp.abs(jnp.sin(x_safe)) / x_safe
    sinc_val = jnp.maximum(sinc_val, 1e-30)
    exact = jnp.log(sinc_val)
    return jnp.where(jnp.abs(x) < 0.1, taylor, exact)


# -------------------------
def _safe_sinch(x):
    """Compute sinh(x)/x safely, using Taylor expansion near 0.

    sinh(x)/x = 1 + x²/6 + x⁴/120 + ...
    """
    x2 = x ** 2
    taylor = 1.0 + x2 / 6.0 + x2 * x2 / 120.0
    x_safe = jnp.maximum(jnp.abs(x), 0.1)
    exact = jnp.sinh(x_safe) / x_safe
    return jnp.where(jnp.abs(x) < 0.1, taylor, exact)


# -------------------------
# get(): supports S1,S2,... plus general S<n> and T<n>
#   - "S5" means S^5 embedded in R^6  => Sphere(D=6)
#   - "T3" means (S^1)^3 embedded in R^6 via PRODUCT => Product(manifolds_str="S1,S1,S1")
# -------------------------
def get(manifold: str):
    manifold = manifold.strip()

    # keep your original aliases
    if manifold == "S1":
        return Sphere(D=2)
    if manifold == "S2":
        return Sphere(D=3)

    # SO(3) rotation group (quaternion representation)
    if manifold == "SO3":
        return SO3(D=4)

    # SE(3) rigid body motions
    if manifold == "SE3":
        return SE3(D=7)

    # SPD manifold (affine-invariant): "SPD<n>" = n×n SPD matrices, D = n²
    m = re.fullmatch(r"SPD(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("SPD0 not meaningful.")
        return SPD(D=n * n)

    # SPD manifold (Log-Euclidean): "SPDLE<n>"
    m = re.fullmatch(r"SPDLE(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("SPDLE0 not meaningful.")
        return SPDLogEuclidean(D=n * n)

    # Hyperbolic space: "H<n>" = H^n in R^{n+1} (Lorentz model)
    m = re.fullmatch(r"H(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("H0 not meaningful.")
        return Hyperboloid(D=n + 1)

    # general sphere: "S<n>"
    m = re.fullmatch(r"S(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("S0 not supported here.")
        return Sphere(D=n + 1)

    # n-torus via product embedding: "T<n>" = (S1)^n in R^{2n}
    m = re.fullmatch(r"T(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("T0 not meaningful.")
        manifolds_str = ",".join(["S1"] * n)
        return Product(D=2 * n, manifolds_str=manifolds_str)

    # Comma-separated product: "SO3,S1,S1,..." or "S2,S1,S1"
    if "," in manifold:
        components = [get(c.strip()) for c in manifold.split(",")]
        D_total = sum(c.D for c in components)
        return Product(D=D_total, manifolds_str=manifold)

    raise ValueError(f"Unknown manifold spec: {manifold}")