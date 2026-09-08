
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import numpy as np
from functools import partial
import sys

from dataclasses import dataclass
from abc import ABC, abstractmethod
from src.manifolds import Manifold, Sphere, Product, SE3
import jax.scipy.special as jsp_special

import src.utils

from scipy.stats import gaussian_kde
from jax.numpy import newaxis


def get(manifold, name):
    if name == "SphereWrappedNormal":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Sphere", f"Expected Sphere, got {type(manifold).__name__}"
        loc = manifold.zero()
        scale = jnp.full(manifold.D - 1, 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    if name == "TorusWrappedNormal":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Product", f"Expected Product, got {type(manifold).__name__}"
        loc = manifold.zero()
        # Tangent dimension for Product of spheres: sum of (D_i - 1)
        # For T^n = (S^1)^n: each S^1 has D=2, tangent_dim=1, so total = n
        tangent_dim = sum(m.D - 1 for m in manifold.manifolds)
        scale = jnp.full(tangent_dim, 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    elif name == "SphereUniform":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Sphere", f"Expected Sphere, got {type(manifold).__name__}"
        return SphereUniform(manifold=manifold)

    elif name == "TorusUniform":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Product", f"Expected Product, got {type(manifold).__name__}"
        return ProductUniformComponents(manifold=manifold)

    elif name == "SphereFourModes":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Sphere", f"Expected Sphere, got {type(manifold).__name__}"
        return SphereFourModes(manifold=manifold)

    elif name == "SOUniform":
        class_name = type(manifold).__name__
        if class_name == "SO3":
            return SO3Uniform(manifold=manifold)
        else:
            raise ValueError(f"SOUniform requires SO3 manifold, got {class_name}")

    elif name == "SOWrappedNormal":
        class_name = type(manifold).__name__
        loc = manifold.zero()
        if class_name == "SO3":
            tangent_dim = 3  # D - 1
        else:
            raise ValueError(f"SOWrappedNormal requires SO3 manifold, got {class_name}")
        scale = jnp.full(tangent_dim, 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    elif name == "SE3Uniform":
        assert type(manifold).__name__ == "SE3", f"Expected SE3, got {type(manifold).__name__}"
        return SE3Uniform(manifold=manifold)

    elif name == "SE3WrappedNormal":
        assert type(manifold).__name__ == "SE3", f"Expected SE3, got {type(manifold).__name__}"
        loc = manifold.zero()
        # 6D tangent: 3 rotation + 3 translation
        scale = jnp.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    elif name == "SE3FactorizedCompact":
        assert type(manifold).__name__ == "SE3", f"Expected SE3, got {type(manifold).__name__}"
        loc = manifold.zero()
        return SE3FactorizedCompact(
            manifold=manifold,
            loc=loc,
            rot_scale=jnp.array([0.3, 0.3, 0.3]),
            trans_scale=jnp.array([0.5, 0.5, 0.5]),
            trans_low=jnp.array([-2.0, -2.0, -2.0]),
            trans_high=jnp.array([2.0, 2.0, 2.0]),
            include_alpha_volume_correction=False,
        )

    elif name == "HyperboloidWrappedNormal":
        assert type(manifold).__name__ == "Hyperboloid", \
            f"Expected Hyperboloid, got {type(manifold).__name__}"
        loc = manifold.zero()
        tangent_dim = manifold.D - 1  # H^n has intrinsic dim n = D-1
        scale = jnp.full(tangent_dim, 0.5)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    elif name == "SPDWrappedNormal":
        assert type(manifold).__name__ == "SPD", \
            f"Expected SPD, got {type(manifold).__name__}"
        loc = manifold.zero()
        tangent_dim = manifold.tangent_dim
        scale = jnp.full(tangent_dim, 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    else:
        try:
            return getattr(sys.modules[__name__], name)(manifold=manifold)
        except:
            print(f"Error loading data class {name}")
            raise


def get_uniform(manifold):
    # Use class name to support both our manifolds and RCPM manifolds
    class_name = type(manifold).__name__
    if class_name == "Sphere" or class_name.startswith("S") and class_name[1:].isdigit():
        return SphereUniform(manifold=manifold)
    if class_name == "SO3":
        return SO3Uniform(manifold=manifold)
    if class_name in ("Product", "Torus"):
        return ProductUniformComponents(manifold=manifold)
    if class_name == "SE3":
        return SE3Uniform(manifold=manifold)
    raise NotImplementedError(f"Uniform not implemented for {type(manifold)}")


@dataclass
class Density(ABC):
    manifold: Manifold

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self, key, n_samples):
        pass

    def __hash__(self):
        return 0  # For jitting


class SphereUniform(Density):
    def log_prob(self, xs):
        assert xs.ndim == 2
        n_batch, D = xs.shape
        assert D == self.manifold.D

        # Surface area of unit sphere S^{D-1} in R^D:
        # SA = 2 * pi^{D/2} / Gamma(D/2)
        log_SA = (
            jnp.log(2.0) + (D / 2.0) * jnp.log(jnp.pi) - jsp_special.gammaln(D / 2.0)
        )
        return jnp.full((n_batch,), -log_SA)

    def sample(self, key, n_samples):
        xs = random.normal(key, shape=(n_samples, self.manifold.D))
        return self.manifold.projx(xs)


class SO3Uniform(Density):
    """Uniform (Haar) measure on SO(3) via unit quaternions.

    SO(3) = S^3 / Z_2, so volume = vol(S^3) / 2 = 2pi^2 / 2 = pi^2.
    Sampling: project Gaussian in R^4 to unit sphere, canonicalize to w >= 0.
    """

    def log_prob(self, xs):
        assert xs.ndim == 2
        n_batch = xs.shape[0]
        # vol(SO(3)) = pi^2
        return jnp.full((n_batch,), -2.0 * jnp.log(jnp.pi))

    def sample(self, key, n_samples):
        xs = random.normal(key, shape=(n_samples, 4))
        return self.manifold.projx(xs)


@dataclass
class SE3Uniform(Density):
    """Uniform measure on SE(3) within a bounded translation region.

    Rotation: Haar measure on SO(3) (uniform quaternions).
    Translation: uniform in [-t_range, t_range]^3.

    This is a product measure: vol = vol(SO(3)) * (2*t_range)^3 = π² * (2*t_range)^3.
    """
    t_range: float = 2.0  # translation sampled from [-t_range, t_range]^3

    def log_prob(self, xs):
        assert xs.ndim == 2
        n_batch = xs.shape[0]
        # vol = pi^2 * (2*t_range)^3
        log_vol = 2.0 * jnp.log(jnp.pi) + 3.0 * jnp.log(2.0 * self.t_range)
        return jnp.full((n_batch,), -log_vol)

    def sample(self, key, n_samples):
        k1, k2 = jax.random.split(key)
        # Uniform rotation: project Gaussian to unit quaternion
        q = jax.random.normal(k1, (n_samples, 4))
        q = self.manifold._so3.projx(q)
        # Uniform translation
        t = jax.random.uniform(k2, (n_samples, 3), minval=-self.t_range, maxval=self.t_range)
        return jnp.concatenate([q, t], axis=-1)

    def __hash__(self):
        return 0


@dataclass
class EmpiricalDensity(Density):
    """Empirical distribution: sample uniformly from a stored dataset.

    For use as a target density in semi-dual OT training where only
    .sample() is needed. log_prob is not supported.
    """
    data: jnp.ndarray = None  # (N, D) stored data points

    def log_prob(self, xs):
        raise NotImplementedError(
            "EmpiricalDensity does not support log_prob. "
            "Use a KDE or parametric fit for density evaluation."
        )

    def sample(self, key, n_samples):
        n_data = self.data.shape[0]
        indices = jax.random.randint(key, (n_samples,), 0, n_data)
        return self.data[indices]


@dataclass
class WrappedNormal(Density):
    loc: jnp.ndarray  # (D,)
    scale: jnp.ndarray  # (D-1,)

    def log_prob(self, z):
        assert z.ndim == 2
        loc = jnp.broadcast_to(self.loc, z.shape)  # (B,D)

        u = self.manifold.log(loc, z)  # (B,D)
        y = self.manifold.zero_like(loc)  # (B,D)

        v = self.manifold.transp(loc, y, u)  # (B,D)
        v = self.manifold.squeeze_tangent(v)  # (B,d_tangent)

        assert v.shape[-1] == self.scale.shape[0], (
            f"scale has length {self.scale.shape[0]} but tangent coords have length {v.shape[-1]}"
        )

        n_logprob = norm.logpdf(v, scale=self.scale).sum(axis=-1)  # (B,)
        logdet = self.manifold.logdetexp(loc, u)  # (B,)
        return n_logprob - logdet

    def sample(self, key, n_samples):
        tdim = self.scale.shape[0]
        v = self.scale * random.normal(key, (n_samples, tdim))  # (B,tdim)
        v = self.manifold.unsqueeze_tangent(v)  # (B,D)

        loc = jnp.broadcast_to(self.loc, (n_samples, self.manifold.D))
        x0 = self.manifold.zero_like(loc)

        u = self.manifold.transp(x0, loc, v)
        z = self.manifold.exponential_map(loc, u)
        return z

    def __hash__(self):
        return 0


@dataclass
class SphereFourModes(Density):
    def __post_init__(self):
        self.modes = []
        locs = [
            jnp.array([0.3, 1.0, 1.0]),
            jnp.array([0.3, -1.0, 1.0]),
            jnp.array([0.3, 1.0, -1.0]),
            jnp.array([0.3, -1.0, -1.0]),
        ]
        locs = [self.manifold.projx(loc) for loc in locs]
        scale = jnp.full(self.manifold.D - 1, 0.3)
        self.dists = [
            WrappedNormal(manifold=self.manifold, loc=loc, scale=scale) for loc in locs
        ]

    def log_prob(self, z):
        raise NotImplementedError()

    def sample(self, key, n_samples):
        keys = random.split(key, len(self.dists))
        n = int(np.ceil(n_samples / len(self.dists)))
        samples = jnp.concatenate(
            [d.sample(key, n) for key, d in zip(keys, self.dists)], axis=0
        )
        samples = random.permutation(key, samples)
        return samples[:n_samples]

    def __hash__(self):
        return 0  # For jitting


@dataclass
class ProductUniformComponents(Density):
    def __post_init__(self):
        self.base_dists = []
        for man in self.manifold.manifolds:
            self.base_dists.append(get_uniform(man))

    def log_prob(self, xs):
        # Note this is not necessarily uniform
        assert xs.ndim == 2
        n_batch = xs.shape[0]
        log_probas = jnp.zeros([n_batch])
        d = 0
        for i, base_dist in enumerate(self.base_dists):
            D = self.manifold.manifolds[i].D
            log_probas += base_dist.log_prob(xs[:, d : d + D])
            d = d + D
        return log_probas

    def sample(self, key, n_samples):
        # Note this is not necessarily uniform
        xs = []
        keys = jax.random.split(key, len(self.base_dists))
        for key, base_dist in zip(keys, self.base_dists):
            samples_man = base_dist.sample(key=key, n_samples=n_samples)
            xs.append(samples_man)
        xs = jnp.concatenate(xs, 1)
        return xs

    def __hash__(self):
        return 0  # For jitting


def _as_1d_array(x, length):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.full((length,), x)
    assert x.shape == (length,), f"Expected shape {(length,)}, got {x.shape}"
    return x


def _diag_truncnorm_log_prob(x, loc, scale, low, high):
    """
    Diagonal truncated normal on R^d with box support [low, high].

    Args:
        x:     (B, d)
        loc:   (d,)
        scale: (d,)
        low:   (d,)
        high:  (d,)

    Returns:
        (B,) log density wrt Lebesgue measure.
    """
    assert x.ndim == 2
    d = x.shape[-1]

    loc = _as_1d_array(loc, d)
    scale = _as_1d_array(scale, d)
    low = _as_1d_array(low, d)
    high = _as_1d_array(high, d)

    a = (low - loc) / scale
    b = (high - loc) / scale

    cdf_a = jsp_special.ndtr(a)
    cdf_b = jsp_special.ndtr(b)
    Z = jnp.maximum(cdf_b - cdf_a, jnp.finfo(x.dtype).tiny)
    logZ = jnp.log(Z).sum()

    inside = jnp.all((x >= low) & (x <= high), axis=-1)

    base_logp = norm.logpdf(x, loc=loc, scale=scale).sum(axis=-1)
    logp = base_logp - logZ
    return jnp.where(inside, logp, -jnp.inf)


def _diag_truncnorm_sample(key, n_samples, loc, scale, low, high):
    """
    Sample from a diagonal truncated normal on a box [low, high]
    using inverse-CDF sampling.

    Returns:
        (n_samples, d)
    """
    loc = jnp.asarray(loc)
    d = loc.shape[0]

    scale = _as_1d_array(scale, d)
    low = _as_1d_array(low, d)
    high = _as_1d_array(high, d)

    a = (low - loc) / scale
    b = (high - loc) / scale

    cdf_a = jsp_special.ndtr(a)
    cdf_b = jsp_special.ndtr(b)

    u = random.uniform(key, shape=(n_samples, d))
    p = cdf_a[None, :] + u * (cdf_b - cdf_a)[None, :]

    tiny = jnp.finfo(loc.dtype).eps
    p = jnp.clip(p, tiny, 1.0 - tiny)

    z = jsp_special.ndtri(p)
    return loc[None, :] + scale[None, :] * z


@dataclass
class SE3FactorizedCompact(Density):
    """
    Factorized compact-support density on product-manifold SE(3).

    Factorization:
        p(q, t) = p_rot(q) * p_trans(t)

    where:
      - p_rot is a WrappedNormal on SO(3)
      - p_trans is a diagonal truncated normal on R^3 with box support

    Notes
    -----
    1) This is for the product-manifold SE(3) implementation, not the true
       Lie-group twist density.

    2) By default, log_prob is returned with respect to the raw product
       measure dmu_SO3(q) * dt, i.e. Haar/rotation measure times Lebesgue
       in translation.

       If you want the density with respect to the alpha-weighted SE(3)
       Riemannian volume, set include_alpha_volume_correction=True.
       That adds the constant correction -3*log(alpha).
    """
    loc: jnp.ndarray                 # (7,) = [q_loc, t_loc]
    rot_scale: jnp.ndarray           # (3,)
    trans_scale: jnp.ndarray         # (3,)
    trans_low: jnp.ndarray           # (3,) or scalar
    trans_high: jnp.ndarray          # (3,) or scalar
    include_alpha_volume_correction: bool = False

    def __post_init__(self):
        assert (type(self.manifold).__name__ == "SE3"), \
            f"Expected SE3, got {type(self.manifold).__name__}"

        self.loc = self.manifold.projx(jnp.asarray(self.loc))
        self.rot_scale = _as_1d_array(self.rot_scale, 3)
        self.trans_scale = _as_1d_array(self.trans_scale, 3)
        self.trans_low = _as_1d_array(self.trans_low, 3)
        self.trans_high = _as_1d_array(self.trans_high, 3)

        assert jnp.all(self.rot_scale > 0), "rot_scale must be positive"
        assert jnp.all(self.trans_scale > 0), \
            "trans_scale must be positive"
        assert jnp.all(self.trans_low < self.trans_high), \
            "Need trans_low < trans_high"
        assert self.manifold.alpha > 0, "SE3 alpha must be > 0"

        q_loc, t_loc = self.manifold._split(self.loc)
        self.q_loc = q_loc
        self.t_loc = t_loc

        self.rot_dist = WrappedNormal(
            manifold=self.manifold._so3,
            loc=self.q_loc,
            scale=self.rot_scale,
        )

    def log_prob(self, z):
        assert z.ndim == 2 and z.shape[-1] == 7

        q, t = self.manifold._split(z)

        logp_rot = self.rot_dist.log_prob(q)
        logp_trans = _diag_truncnorm_log_prob(
            x=t,
            loc=self.t_loc,
            scale=self.trans_scale,
            low=self.trans_low,
            high=self.trans_high,
        )

        logp = logp_rot + logp_trans

        if self.include_alpha_volume_correction:
            logp = logp - 3.0 * jnp.log(self.manifold.alpha)

        return logp

    def sample(self, key, n_samples):
        k1, k2 = random.split(key)

        q = self.rot_dist.sample(k1, n_samples)
        t = _diag_truncnorm_sample(
            key=k2,
            n_samples=n_samples,
            loc=self.t_loc,
            scale=self.trans_scale,
            low=self.trans_low,
            high=self.trans_high,
        )
        return jnp.concatenate([q, t], axis=-1)

    def __hash__(self):
        return 0


