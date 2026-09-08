
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import numpy as np
from functools import partial
import sys

from dataclasses import dataclass
from abc import ABC, abstractmethod
from src.manifolds import Manifold, Sphere, Product, FlatTorus
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

    elif name == "FlatTorusUniform":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "FlatTorus", f"Expected FlatTorus, got {type(manifold).__name__}"
        return FlatTorusUniform(manifold=manifold)

    elif name == "FlatTorusWrappedNormal":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "FlatTorus", f"Expected FlatTorus, got {type(manifold).__name__}"
        loc = manifold.zero()  # Center at [π, π, ...]
        scale = jnp.full(manifold.D, 0.3)
        return FlatTorusWrappedNormal(manifold=manifold, loc=loc, scale=scale)

    elif name == "SphereFourModes":
        # Check by class name to handle autoreload
        assert type(manifold).__name__ == "Sphere", f"Expected Sphere, got {type(manifold).__name__}"
        return SphereFourModes(manifold=manifold)

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
    if class_name in ("Product", "Torus"):
        return ProductUniformComponents(manifold=manifold)
    if class_name == "FlatTorus":
        return FlatTorusUniform(manifold=manifold)
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


@dataclass
class FlatTorusUniform(Density):
    """
    Uniform distribution on flat torus T^n = [0, 2π)^n.

    For the flat torus represented as angle coordinates,
    the uniform distribution is simply uniform angles in [0, 2π)^n.

    Log density = -n * log(2π)
    """

    def log_prob(self, xs):
        """
        Compute log probability of points on flat torus.
        xs: (B, n) array of angles in [0, 2π)^n
        """
        assert xs.ndim == 2
        n_batch, n = xs.shape
        assert n == self.manifold.D
        # Uniform on [0, 2π)^n has density 1/(2π)^n
        # log prob = -n * log(2π)
        log_density = -n * jnp.log(2 * jnp.pi)
        return jnp.full((n_batch,), log_density)

    def sample(self, key, n_samples):
        """
        Sample uniformly from [0, 2π)^n.
        """
        return jax.random.uniform(key, (n_samples, self.manifold.D)) * 2 * jnp.pi

    def __hash__(self):
        return 0  # For jitting


@dataclass
class FlatTorusWrappedNormal(Density):
    """
    Wrapped normal distribution on flat torus T^n = [0, 2π)^n.

    The exact wrapped normal is: p(θ) = Σ_{k∈Z^n} N(θ - μ + 2πk; 0, σ²I)

    This implementation uses the k=0 term only:
        p(θ) ≈ N(wrap(θ - μ); 0, σ²I)

    This is a good approximation when σ << π. For σ = 0.3 and period 2π,
    the Gaussian density at ±π is exp(-π²/(2*0.09)) ≈ exp(-55) ≈ 0,
    so the truncation error is negligible.

    For larger σ where periodic copies matter, use von Mises distribution instead.
    """
    loc: jnp.ndarray    # (n,) center angles in [0, 2π)
    scale: jnp.ndarray  # (n,) standard deviations

    def _wrap_diff(self, theta):
        """Compute wrapped difference theta - loc, mapped to [-π, π)^n."""
        diff = theta - self.loc
        return jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi

    def _log_prob_with_periodic_sum(self, theta):
        """
        Log probability with sum over k ∈ {-1, 0, 1} periodic copies.

        Uses factorization: since dimensions are independent,
            log p(θ) = Σᵢ log[Σₖ N(θᵢ - μᵢ + 2πk; 0, σᵢ²)]

        This is O(3n) instead of O(3^n) for the naive mesh approach.
        Only needed when σ is not small (σ ≳ 1).
        """
        diff = self._wrap_diff(theta)  # (B, n)
        offsets = jnp.array([-2*jnp.pi, 0.0, 2*jnp.pi])  # (3,)

        # Shapes: diff (B, n), offsets (3,), scale (n,)
        # We want shifted (B, n, 3) and scale (1, n, 1) for proper broadcasting
        shifted = diff[:, :, None] + offsets  # (B, n, 3)
        scale = self.scale[None, :, None]     # (1, n, 1)

        # Gaussian log prob for each (B, n, 3)
        log_probs_per_k = (
            -0.5 * (shifted / scale) ** 2
            - jnp.log(scale)
            - 0.5 * jnp.log(2 * jnp.pi)
        )  # (B, n, 3)

        # logsumexp over k for each dimension, then sum over dimensions
        log_prob_per_dim = jax.scipy.special.logsumexp(log_probs_per_k, axis=-1)  # (B, n)
        return jnp.sum(log_prob_per_dim, axis=-1)  # (B,)

    def log_prob(self, theta):
        """
        Compute log probability of wrapped normal on flat torus.

        Uses the k=0 approximation: p(θ) ≈ N(wrap(θ - μ); 0, σ²I)

        Accurate when σ << π (e.g., σ = 0.3).

        Args:
            theta: (B, n) array of angles in [0, 2π)^n

        Returns:
            (B,) log probabilities
        """
        assert theta.ndim == 2
        n = theta.shape[1]
        assert n == self.manifold.D

        # Wrapped difference to [-π, π)^n
        diff = self._wrap_diff(theta)  # (B, n)

        # Gaussian log probability: log N(x; 0, σ²) = -x²/(2σ²) - log(σ) - 0.5*log(2π)
        log_prob = jnp.sum(
            -0.5 * (diff / self.scale) ** 2 - jnp.log(self.scale) - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1
        )  # (B,)

        return log_prob

    def sample(self, key, n_samples):
        """
        Sample from wrapped normal distribution.

        Draw from Gaussian centered at loc and wrap to [0, 2π)^n.
        """
        n = self.manifold.D
        # Sample from Gaussian
        v = self.scale * jax.random.normal(key, (n_samples, n))  # (B, n)
        # Add to loc and wrap to [0, 2π)
        theta = jnp.mod(self.loc + v, 2 * jnp.pi)  # (B, n)
        return theta

    def __hash__(self):
        return 0  # For jitting
