import os
import re
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
from spherical_kde import SphericalKDE


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
    SO(3) represented as unit quaternions in R^4 with ||q||=1.

    Quaternions q and -q represent the same rotation, so we identify them
    by always projecting to the hemisphere with q[0] >= 0 (scalar part positive).

    The manifold dimension is 3, but ambient dimension is D=4.
    Geodesic distance is the rotation angle in [0, π].

    Convention: q = [w, x, y, z] where w is the scalar part.
    """
    D: int = 4

    def _to_canonical(self, q):
        """Project to canonical hemisphere (w >= 0) to handle q ≡ -q."""
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
        sign = jnp.sign(q[..., :1] + eps)  # +eps to break tie at exactly 0
        return q * sign

    def projx(self, x):
        """Normalize to unit quaternion and pick canonical hemisphere."""
        return self._to_canonical(x)

    def exponential_map(self, x, v):
        """
        Exponential map on SO(3): exp_q(v) where v ∈ T_q SO(3).

        v is a 4D vector tangent to S³ at q (i.e., v·q = 0), representing
        a 3D rotation. Same formula as sphere exponential.
        """
        x = self._to_canonical(x)
        v = self.tangent_projection(x, v)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        result = x * jnp.cos(v_norm) + v * _safe_sinc(v_norm)
        return self._to_canonical(result)

    def log(self, x, y):
        """
        Riemannian log map on SO(3).

        Returns v ∈ T_x such that exp_x(v) = y.
        Handles antipodal identification by using the closer of y or -y.
        Uses atan2 for numerical stability (same as Sphere).
        """
        x = self._to_canonical(x)
        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

        # Handle q ≡ -q: use whichever is closer to x
        dot_pos = jnp.sum(x * y, axis=-1, keepdims=True)
        dot_neg = jnp.sum(x * (-y), axis=-1, keepdims=True)
        y = jnp.where(dot_pos >= dot_neg, y, -y)

        # Now compute log as on sphere (reuse stable atan2 formula)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        u = y - xy * x  # tangent direction

        u_norm = jnp.sqrt(jnp.sum(u**2, axis=-1, keepdims=True) + eps)
        theta = jnp.arctan2(u_norm, xy)

        # Coefficient theta / sin(theta), with Taylor expansion near 0
        small = u_norm < 1e-6
        coef_small = 1.0 + (theta**2) / 6.0
        safe_u_norm = jnp.maximum(u_norm, 1e-6)
        coef_large = theta / safe_u_norm
        coef = jnp.where(small, coef_small, coef_large)

        return coef * u

    def tangent_projection(self, x, v):
        """Project v onto tangent space at x: v - <x,v>x."""
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        xv = jnp.sum(x * v, axis=-1, keepdims=True)
        return v - xv * x

    def dist(self, x, y):
        """
        Geodesic distance on SO(3) = rotation angle in [0, π].

        For unit quaternions: angle = 2 * arccos(|q1 · q2|)
        The factor of 2 accounts for the double cover S³ → SO(3).

        Uses atan2 formula for numerical stability (same as Sphere):
            dist = 2 * atan2(||x - y||, ||x + y||)
        This is stable both near 0 and near π (unlike arccos).

        For SO(3), we first pick the closer of y or -y, then apply the formula.

        Supports:
          x: (B,4), y: (4,M)  -> (B,M)   [landmark case]
          x: (B,4), y: (B,4)  -> (B,)
          x: (...,4), y: (...,4) -> (...)
        """
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x (B,4), y (4,M) -> (B,M)
            y = y / (jnp.linalg.norm(y, axis=0, keepdims=True) + eps)

            # For each (x_i, y_j) pair, pick closer of y_j or -y_j
            # dot_pos[i,j] = x_i · y_j
            dot_pos = x @ y  # (B, M)

            # Use sign of dot to pick hemisphere: if dot < 0, use -y
            # Equivalent to using |dot| but allows stable atan2 formula
            sign = jnp.sign(dot_pos + eps)  # (B, M)

            # Compute ||x - y'|| and ||x + y'|| where y' = sign * y
            # x[:,:,None] is (B,4,1), y[None,:,:] is (1,4,M), sign is (B,1,M)
            x_exp = x[:, :, None]  # (B, 4, 1)
            y_exp = y[None, :, :]  # (1, 4, M)
            sign_exp = sign[:, None, :]  # (B, 1, M)

            y_adj = y_exp * sign_exp  # (B, 4, M) - y adjusted to closer hemisphere

            diff = x_exp - y_adj  # (B, 4, M)
            summ = x_exp + y_adj  # (B, 4, M)

            norm_diff = jnp.sqrt(jnp.sum(diff**2, axis=1) + eps)  # (B, M)
            norm_sum = jnp.sqrt(jnp.sum(summ**2, axis=1) + eps)   # (B, M)

            # Factor of 2 for S³ geodesic, another 2 for SO(3) double cover
            return 4 * jnp.arctan2(norm_diff, norm_sum)
        else:
            y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

            # Pick closer of y or -y
            dot = jnp.sum(x * y, axis=-1, keepdims=True)
            y = jnp.where(dot >= 0, y, -y)

            # Stable atan2 formula
            # 2 * arctan2 gives S³ geodesic = arccos(x·y)
            # Another factor of 2 for SO(3) double cover (rotation angle = 2 * quaternion angle)
            diff = x - y
            summ = x + y
            norm_diff = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)
            norm_sum = jnp.sqrt(jnp.sum(summ**2, axis=-1) + eps)

            return 4 * jnp.arctan2(norm_diff, norm_sum)

    def cost(self, x, y):
        """Squared geodesic distance / 2."""
        d = self.dist(x, y)
        return 0.5 * d**2

    def tangent_orthonormal_basis(self, x, dF):
        """
        Returns (B, 4, 3) orthonormal tangent basis at x.
        Tangent space to S³ at x is 3-dimensional.
        """
        assert x.ndim == 2 and dF.ndim == 2 and x.shape == dF.shape
        B, D = x.shape
        assert D == 4

        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

        # Start with gradient direction if nonzero
        dF_tan = self.tangent_projection(x, dF)
        nrm = jnp.linalg.norm(dF_tan, axis=-1, keepdims=True)

        # Fallback: project canonical basis vectors
        I = jnp.eye(4)
        e0 = jnp.broadcast_to(I[0], (B, 4))
        e1 = jnp.broadcast_to(I[1], (B, 4))

        u0a = self.tangent_projection(x, e0)
        u0b = self.tangent_projection(x, e1)
        na = jnp.linalg.norm(u0a, axis=-1, keepdims=True)
        nb = jnp.linalg.norm(u0b, axis=-1, keepdims=True)
        u0a_safe = u0a / jnp.maximum(na, eps)
        u0b_safe = u0b / jnp.maximum(nb, eps)
        u0_fb = jnp.where(na >= nb, u0a_safe, u0b_safe)

        u0_from_dF = dF_tan / (nrm + eps)
        u0 = jnp.where(nrm > eps, u0_from_dF, u0_fb)

        # Complete to ONB via QR on projected canonical basis
        C = jnp.broadcast_to(I, (B, 4, 4))
        xTC = jnp.einsum("bi,bij->bj", x, C)
        Ctan = C - jnp.einsum("bi,bj->bij", x, xTC)

        u0TC = jnp.einsum("bi,bij->bj", u0, Ctan)
        Crem = Ctan - jnp.einsum("bi,bj->bij", u0, u0TC)

        def qr_one(A):
            Q, R = jnp.linalg.qr(A)
            return Q, R

        Q, _ = jax.vmap(qr_one)(Crem)
        rest = Q[..., :2]  # need 2 more vectors for 3D tangent space
        E = jnp.concatenate([u0[..., None], rest], axis=-1)  # (B, 4, 3)
        return E

    def zero(self):
        """Identity rotation: quaternion [1, 0, 0, 0]."""
        return jnp.array([1.0, 0.0, 0.0, 0.0])

    def zero_like(self, x):
        """Identity rotation with same batch shape as x."""
        z = jnp.zeros_like(x)
        return z.at[..., 0].set(1.0)

    def squeeze_tangent(self, v):
        """
        Tangent space at identity [1,0,0,0] is {0} × R³.
        Remove the first (scalar) component.
        """
        return v[..., 1:]

    def unsqueeze_tangent(self, w):
        """
        w: (..., 3) -> (..., 4) tangent at identity with leading 0.
        """
        return jnp.concatenate((jnp.zeros_like(w[..., :1]), w), axis=-1)

    def transp(self, x, y, u):
        """
        Parallel transport on SO(3) along minimal geodesic.
        Same formula as sphere, but handle antipodal identification.
        """
        x = self._to_canonical(x)
        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

        # Use closer representative
        dot = jnp.sum(x * y, axis=-1, keepdims=True)
        y = jnp.where(dot >= 0, y, -y)
        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        denom = jnp.maximum(1.0 + xy, 0.01)  # clamp near cut locus
        return u - yu / denom * (x + y)

    def logdetexp(self, x, u):
        """
        Log determinant of exponential map Jacobian on SO(3).

        Same as S³: (D-2) * log(sinc(r)) = 2 * log(sinc(||u||))
        where D=4 is the ambient dimension.

        Note: ||u|| is half the rotation angle due to quaternion double cover,
        so this gives the correct volume element for SO(3).
        """
        r = jnp.linalg.norm(u, axis=-1)

        small = jnp.abs(r) < 1e-2
        log_sinc_taylor = -r**2 / 6.0 - r**4 / 180.0

        sinc_r = jnp.abs(jnp.sin(r)) / jnp.clip(r, 1e-10, None)
        sinc_r = jnp.clip(sinc_r, 1e-10, 1.0)
        log_sinc_direct = jnp.log(sinc_r)

        val = jnp.where(small, log_sinc_taylor, log_sinc_direct)
        return 2 * val  # (D-2) = 4-2 = 2


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
# FlatTorus: T^n as quotient R^n/Z^n (intrinsic angle coordinates)
# -------------------------
@dataclass
class FlatTorus(Manifold):
    """
    n-dimensional flat torus T^n = R^n / Z^n represented intrinsically
    as angle coordinates θ ∈ [0, 2π)^n.

    This uses the MINIMAL n-dimensional representation (not the 2n-dimensional
    product embedding). All operations account for periodic boundary conditions.

    Examples:
        T^2: D=2 (angles [θ₁, θ₂])
        T^3: D=3 (angles [θ₁, θ₂, θ₃])

    Advantages over Product embedding:
        - Minimal dimensions (n instead of 2n)
        - Simpler Jacobian (flat → logdetexp = 0)
        - Direct angle manipulation
    """

    def _wrap_diff(self, x, y):
        """
        Compute shortest difference y - x accounting for periodicity.
        Maps differences to [-π, π).
        """
        diff = y - x
        return jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi

    def exponential_map(self, x, v):
        """
        Exponential map on flat torus: straight line with periodic wrapping.
        exp_x(v) = (x + v) mod 2π
        """
        return jnp.mod(x + v, 2 * jnp.pi)

    def log(self, x, y):
        """
        Riemannian log map: shortest tangent vector from x to y.
        Returns v ∈ T_x such that exp_x(v) = y (modulo wrapping).
        """
        return self._wrap_diff(x, y)

    def tangent_projection(self, x, v):
        """
        Tangent projection on flat torus: identity (tangent space is just R^n).
        """
        return v

    def dist(self, x, y):
        """
        Geodesic distance with periodic boundary conditions.

        Supports:
          x: (B, n), y: (n, M)  -> (B, M)   [landmark/batch-to-batch case]
          x: (B, n), y: (B, n)  -> (B,)     [standard pairwise case]
          x: (..., n), y: (..., n) -> (...)  [broadcasted case]
        """
        # Check if this is the landmark case: x: (B, n), y: (n, M)
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            # Landmark case: x is (B, n), y is (n, M)
            # y stores M landmarks as columns: y[:, j] is the j-th landmark
            # Need to compute distances from each of B points to each of M landmarks
            # Result: (B, M)

            # Reshape: x -> (B, 1, n), y.T -> (1, M, n)
            x_exp = x[:, None, :]  # (B, 1, n)
            y_exp = y.T[None, :, :]   # (1, M, n)

            # Compute differences with wrapping: (B, M, n)
            diff = y_exp - x_exp
            diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi

            # Compute distances: sqrt(sum over n dimension)
            dist_sq = jnp.sum(diff**2, axis=2)  # (B, M)
            return jnp.sqrt(dist_sq + eps)
        else:
            # Standard case: element-wise or broadcasted
            diff = self._wrap_diff(x, y)
            return jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)

    def cost(self, x, y):
        """
        Squared geodesic distance / 2.

        Handles both:
          - Landmark case: x: (B, n), y: (n, M) -> (B, M)
          - Standard case: x: (B, n), y: (B, n) -> (B,)
        """
        d = self.dist(x, y)
        return 0.5 * d**2

    def projx(self, x):
        """
        Project to [0, 2π)^n.
        """
        return jnp.mod(x, 2 * jnp.pi)

    def tangent_orthonormal_basis(self, x, dF):
        """
        Orthonormal tangent basis on flat torus: canonical basis.
        Returns (B, n, n) identity matrices.
        """
        B, n = x.shape
        assert n == self.D
        return jnp.broadcast_to(jnp.eye(n), (B, n, n))

    def zero(self):
        """
        Reference point: center of torus (all angles = π).
        This matches the Product T^n behavior where each S¹ factor has zero at angle π.
        """
        return jnp.full((self.D,), jnp.pi)

    def zero_like(self, x):
        """
        Zero point with same shape as x (all angles = π).
        """
        return jnp.full_like(x, jnp.pi)

    def squeeze_tangent(self, v):
        """
        Flat torus: tangent space is R^n itself, no squeezing needed.
        Identity operation for interface compatibility.
        """
        return v

    def unsqueeze_tangent(self, w):
        """
        Flat torus: tangent space is R^n itself, no unsqueezing needed.
        Identity operation for interface compatibility.
        """
        return w

    def transp(self, x, y, u):
        """
        Parallel transport on flat torus: identity (flat connection).
        """
        return u

    def logdetexp(self, x, u):
        """
        Log determinant of exponential map Jacobian.
        Flat torus: exponential map is just translation → Jacobian is identity → log det = 0.
        """
        if u.ndim == 1:
            return 0.0
        return jnp.zeros(u.shape[:-1])


# -------------------------
# get(): supports S1,S2,... plus general S<n> and T<n>
#   - "S5" means S^5 embedded in R^6  => Sphere(D=6)
#   - "T3" means (S^1)^3 embedded in R^6 via PRODUCT => Product(manifolds_str="S1,S1,S1")
#   - "FlatT3" means T^3 as quotient R^3/Z^3 (MINIMAL 3D angles) => FlatTorus(D=3)
# -------------------------
def get(manifold: str):
    manifold = manifold.strip()

    # keep your original aliases
    if manifold == "S1":
        return Sphere(D=2)
    if manifold == "S2":
        return Sphere(D=3)

    # SO(3) rotation group
    if manifold == "SO3":
        return SO3(D=4)

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

    # n-torus via quotient (flat): "FlatT<n>" = R^n/Z^n (minimal dimension n)
    m = re.fullmatch(r"FlatT(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("FlatT0 not meaningful.")
        return FlatTorus(D=n)

    raise ValueError(f"Unknown manifold spec: {manifold}")
