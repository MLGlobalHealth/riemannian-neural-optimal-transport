# src/empirical.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import newaxis
from scipy.stats import gaussian_kde
from src.densities import Density


def latlon_deg_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon in degrees to unit-sphere xyz."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    xyz = np.stack([x, y, z], axis=-1)

    # numerical safety
    nrm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz = xyz / np.clip(nrm, 1e-12, None)
    return xyz


@dataclass(frozen=True)
class EmpiricalSpherePointCloud:
    """Empirical distribution on S^2 backed by a fixed point cloud."""
    points: jnp.ndarray  # (N,3) on unit sphere

    @classmethod
    def from_csv_latlon(cls, csv_path: str) -> "EmpiricalSpherePointCloud":
        # Try reading with header first
        df = pd.read_csv(csv_path)

        # If file had no header, columns won't be named lat/lon; fallback
        if not {"lat", "lon"}.issubset(df.columns):
            df = pd.read_csv(csv_path, header=None, names=["lat", "lon"])

        # Coerce to numeric and drop bad rows (or set errors="raise" to be strict)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        xyz = latlon_deg_to_xyz(df["lat"].to_numpy(dtype=float),
                                df["lon"].to_numpy(dtype=float))
        return cls(points=jnp.asarray(xyz, dtype=jnp.float32))

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        N = self.points.shape[0]
        if n >= N:
            return self.points  # Return all points, no sampling
        idx = jax.random.choice(key, N, shape=(n,), replace=False)
        return self.points[idx]


def load_empirical_from_csv(manifold, csv_path, kde_bandwidth=0.1):
    """Load an empirical density from a CSV file with lat/lon columns."""
    return EmpiricalSphereDensity(
        manifold=manifold,
        csv_path=csv_path,
        kde_bandwidth=kde_bandwidth
    )

@dataclass
class EmpiricalSphereDensity(Density):
    """
    Empirical density on S^{d-1} from a CSV file.

    Supports two formats:
    1. Lat/lon format (S^2 only): CSV with 'lat' and 'lon' columns
    2. Cartesian format (any S^{d-1}): CSV with d columns of Cartesian coordinates
       - Optional: last column can be 'weight' for weighted empirical measures

    Args:
        csv_path: Path to CSV file
        kde_bandwidth: Bandwidth for KDE (default 0.1)
        n_mc_samples: Number of MC samples for normalization (default 10000)
        use_kde: If True, use KDE for log_prob. If False, use empirical (default True)
    """
    csv_path: str = ""
    kde_bandwidth: float = 0.1
    n_mc_samples: int = 10000
    use_kde: bool = True

    # Note: manifold is inherited from Density base class

    def __post_init__(self):
        import pandas as pd

        # Load CSV
        df = pd.read_csv(self.csv_path)

        # Detect format: lat/lon vs Cartesian
        if 'lat' in df.columns and 'lon' in df.columns:
            # Lat/lon format (S^2 only)
            if self.manifold.D != 3:
                raise ValueError(f"Lat/lon format only supports S^2 (D=3), got D={self.manifold.D}")

            lat = df['lat'].values
            lon = df['lon'].values

            # Convert to xyz on unit sphere
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            self.data = jnp.array(np.stack([x, y, z], axis=1))
            self.sample_weights = None  # uniform weights

        else:
            # Cartesian format: assume CSV has D or D+1 columns
            # If D+1 columns, last column is 'weight'

            if df.shape[1] == self.manifold.D:
                # No weights, just coordinates
                self.data = jnp.array(df.values)
                self.sample_weights = None

            elif df.shape[1] == self.manifold.D + 1:
                # Last column is weight
                coords = df.iloc[:, :-1].values
                weights = df.iloc[:, -1].values

                # Normalize weights to sum to 1
                weights = weights / weights.sum()

                self.data = jnp.array(coords)
                self.sample_weights = jnp.array(weights)

                print(f"  Using weighted samples (weights sum to {weights.sum():.6f})")

            else:
                raise ValueError(
                    f"CSV has {df.shape[1]} columns but manifold has D={self.manifold.D}. "
                    f"Expected {self.manifold.D} (coordinates) or {self.manifold.D + 1} (coordinates + weight)"
                )

            # Verify points are on sphere
            norms = np.linalg.norm(self.data, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                print(f"  WARNING: Points not exactly on sphere (norm range: [{norms.min():.6f}, {norms.max():.6f}])")
                print(f"           Projecting to unit sphere...")
                self.data = self.data / norms[:, np.newaxis]

        if self.use_kde:
            # Build KDE for log_prob (using sample weights if provided)
            if self.sample_weights is not None:
                # scipy gaussian_kde doesn't support weights directly
                # Approximate by replicating samples proportionally
                # For now, fallback to uniform KDE
                print("  WARNING: KDE doesn't support weighted samples yet. Using uniform KDE.")

            self.kde = gaussian_kde(self.data.T, self.kde_bandwidth)
            L = jnp.linalg.cholesky(jnp.array(self.kde.covariance) * 2 * jnp.pi)
            self.log_det = 2 * jnp.log(jnp.diag(L)).sum()
            self.inv_cov = jnp.array(self.kde.inv_cov)
            self.kde_weights = jnp.array(self.kde.weights)

            # Estimate normalization constant
            self.log_Z = self._estimate_log_normalization()
            print(f"Loaded {len(self.data)} points from {self.csv_path}")
            print(f"  Manifold: S^{self.manifold.D - 1} (D={self.manifold.D})")
            print(f"  Estimated log(Z) = {self.log_Z:.4f} (normalization constant)")
        else:
            # No KDE, just empirical sampling
            print(f"Loaded {len(self.data)} points from {self.csv_path}")
            print(f"  Manifold: S^{self.manifold.D - 1} (D={self.manifold.D})")
            print(f"  Using empirical sampling only (no KDE)")
            self.log_Z = None

    def _estimate_log_normalization(self):
        """Estimate log(Z) where Z = ∫_{S^{d-1}} kde(x) dσ via Monte Carlo."""
        # Sample uniform points on S^{d-1}
        key = jax.random.PRNGKey(42)
        uniform_samples = self._sample_uniform_sphere(key, self.n_mc_samples)

        # Evaluate unnormalized log_prob at these points
        log_kde_vals = self._log_prob_unnormalized(uniform_samples)

        # Z = (surface area of S^{d-1}) * E[kde(x)]
        # Surface area of S^{d-1}: SA = 2π^{d/2} / Γ(d/2)
        # log Z = log(SA) + logsumexp(log_kde) - log(N)
        d = self.manifold.D
        log_surface_area = (
            jnp.log(2.0) + (d / 2.0) * jnp.log(jnp.pi) - jax.scipy.special.gammaln(d / 2.0)
        )
        log_Z = log_surface_area + jax.scipy.special.logsumexp(log_kde_vals) - np.log(self.n_mc_samples)

        return float(log_Z)

    def _sample_uniform_sphere(self, key, n):
        """Sample uniformly from S^{d-1} in R^d."""
        # Use Gaussian projection method (works for any dimension)
        samples = jax.random.normal(key, (n, self.manifold.D))
        norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
        return samples / norms

    def _log_prob_unnormalized(self, xs):
        """Compute unnormalized log KDE values."""
        def single_log_prob(point):
            diff = self.data.T - point[:, newaxis]
            tdiff = jnp.dot(self.inv_cov, diff)
            energy = jnp.sum(diff * tdiff, axis=0)
            log_to_sum = 2.0 * jnp.log(self.kde_weights) - self.log_det - energy
            return jax.scipy.special.logsumexp(0.5 * log_to_sum)
        return jax.vmap(single_log_prob)(xs)

    def log_prob(self, xs):
        """Compute normalized log probability on S^{d-1}."""
        if not self.use_kde:
            raise NotImplementedError("log_prob requires use_kde=True")
        # Unnormalized log KDE - log(Z) = normalized log prob
        return self._log_prob_unnormalized(xs) - self.log_Z

    def sample(self, key, n_samples):
        """Sample from empirical distribution (with weights if provided)."""
        if self.sample_weights is None:
            # Uniform sampling
            indices = jax.random.randint(key, [n_samples], 0, len(self.data))
        else:
            # Weighted sampling
            indices = jax.random.choice(
                key, len(self.data), shape=(n_samples,),
                replace=True, p=self.sample_weights
            )
        return self.data[indices]

    def __hash__(self): return 0
