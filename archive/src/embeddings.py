from typing import Callable, Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------
# Landmark sampling methods
# ---------------------------

def sample_landmarks_random(base, key, M: int) -> jnp.ndarray:
    """Sample M random landmarks from the base distribution."""
    return base.sample(key, M)


def farthest_point_sampling(
    manifold,
    candidates: jnp.ndarray,
    M: int,
    key,
) -> jnp.ndarray:
    """
    Greedy farthest point sampling on a manifold.

    Selects M points from candidates that are maximally spread out
    according to the manifold's distance function.

    Args:
        manifold: Manifold with a dist(x, y) method
        candidates: (N, D) array of candidate points
        M: Number of landmarks to select
        key: JAX random key for initial point selection

    Returns:
        (M, D) array of selected landmark points
    """
    N = candidates.shape[0]
    if M > N:
        raise ValueError(f"FPS requires M <= N, got M={M}, N={N}")

    idx0 = jax.random.randint(key, shape=(), minval=0, maxval=N)

    def dists_to_point(p):
        return jax.vmap(lambda x: manifold.dist(x, p))(candidates)

    d0 = dists_to_point(candidates[idx0])
    idxs0 = jnp.zeros((M,), dtype=jnp.int32).at[0].set(idx0)

    def body(i, carry):
        idxs, min_dists = carry
        idx_next = jnp.argmax(min_dists)
        idxs = idxs.at[i].set(idx_next)
        d_new = dists_to_point(candidates[idx_next])
        min_dists = jnp.minimum(min_dists, d_new)
        return (idxs, min_dists)

    idxs, _ = jax.lax.fori_loop(1, M, body, (idxs0, d0))
    return candidates[idxs]


# ---------------------------
# Landmark builder registry
# ---------------------------

LandmarkFn = Callable[[Any, Any, dict, Any], Tuple[jnp.ndarray, Any]]


def _landmarks_random(manifold, base, cfg: dict, key):
    """Build landmarks via random sampling from base distribution."""
    key, subkey = jax.random.split(key)
    M = int(cfg["n_landmarks"])
    return sample_landmarks_random(base, subkey, M), key


def _landmarks_fps(manifold, base, cfg: dict, key):
    """Build landmarks via farthest point sampling."""
    M = int(cfg["n_landmarks"])
    N = int(cfg.get("fps_candidates", 4096))

    key, k_cand, k_fps = jax.random.split(key, 3)
    candidates = base.sample(k_cand, N)
    landmarks = farthest_point_sampling(manifold, candidates, M, k_fps)
    return landmarks, key


LANDMARK_METHODS: Dict[str, LandmarkFn] = {
    "random": _landmarks_random,
    "fps": _landmarks_fps,
}


def build_landmarks(manifold, base, cfg: dict, key):
    """
    Build landmark points for embedding.

    Args:
        manifold: Manifold object with dist method
        base: Base distribution with sample method
        cfg: Config dict with:
            - n_landmarks: int, number of landmarks
            - method: "random" or "fps"
            - fps_candidates: int, candidate pool size (fps only, default 4096)
        key: JAX random key

    Returns:
        (landmarks, new_key) tuple
    """
    method = str(cfg.get("method", "random")).lower()
    if method not in LANDMARK_METHODS:
        raise ValueError(
            f"Unknown landmark method '{method}'. "
            f"Available: {sorted(LANDMARK_METHODS.keys())}"
        )
    return LANDMARK_METHODS[method](manifold, base, cfg, key)


# ---------------------------
# Distance embedding
# ---------------------------

def dist_to_landmarks(manifold, xs, landmarks):
    """Compute distances from points to landmarks.

    Uses the manifold's batched landmark path: dist(xs, landmarks.T)
    where xs is (N, D) and landmarks.T is (D, M), returning (N, M).
    All manifolds (Sphere, SO3, SE3, Product, Hyperboloid)
    support this path.
    """
    single = (xs.ndim == 1)
    if single:
        xs = xs[None, :]
    d = manifold.dist(xs, landmarks.T)  # (N, M) via landmark path
    return d[0] if single else d

class GromovDistanceEmbedding(nn.Module):
    """Embedding via distances to landmark points.

    Suitable for compact manifolds (S^n, SO(3), tori, etc.) where geodesic
    distances are bounded.  On non-compact Cartan-Hadamard manifolds use
    LogarithmicEmbedding instead.
    """
    manifold: object
    landmarks: jnp.ndarray

    @nn.compact
    def __call__(self, xs):
        return dist_to_landmarks(self.manifold, xs, self.landmarks)


# ---------------------------
# Logarithmic (tangent-space) embedding for Cartan-Hadamard manifolds
# ---------------------------

class LogarithmicEmbedding(nn.Module):
    """Embedding via Riemannian logarithm at a fixed basepoint.

    For Cartan-Hadamard manifolds (complete, simply connected, K <= 0) the
    log map at any point is a global diffeomorphism M -> T_pM ~ R^d, making
    it a natural, non-redundant feature map that respects the geometry.

    Unlike GromovDistanceEmbedding this does *not* require landmarks and
    outputs vectors (not scalars), preserving directional information.

    Output dimension = D (ambient).  For Euclidean R^d this is just
    translation by the basepoint: log_p(x) = x - p.
    """
    manifold: object
    basepoint: jnp.ndarray  # (D,) reference point on the manifold

    @nn.compact
    def __call__(self, xs):
        single = (xs.ndim == 1)
        if single:
            xs = xs[None, :]
        v = jax.vmap(lambda y: self.manifold.log(self.basepoint, y))(xs)
        return v[0] if single else v