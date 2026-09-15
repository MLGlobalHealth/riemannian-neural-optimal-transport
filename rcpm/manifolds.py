# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import jax.numpy as jnp
from jax import random
import jax
from jax.scipy.linalg import block_diag

from spherical_kde import SphericalKDE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

import utils
import cartopy.crs as ccrs

import matplotlib

@dataclass
class Manifold(ABC):
    D: int # Dimension of the ambient Euclidean space

    @abstractmethod
    def exponential_map(self, x, v):
        pass

    @abstractmethod
    def tangent_projection(self, x, v):
        pass

    @abstractmethod
    def projx(self, x):
        pass

    # @abstractmethod
    # def dist(self, x, y):
    #     pass

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def tangent_orthonormal_basis(self, x, dF):
        pass


eps = 1e-5 # TODO: Other stabilization?
divsin = lambda x: x / jnp.sin(x)
sindiv = lambda x: jnp.sin(x) / (x + eps)
divsinh = lambda x: x / jnp.sinh(x)
sinhdiv = lambda x: jnp.sinh(x) / (x + eps)

def _normalize(v, axis=-1, eps=1e-8):
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)

def lorentz_cross(x, y):
    z = jnp.cross(x, y)
    z = z.at[...,0].set(-z[...,0])
    return z

@dataclass
class Sphere(Manifold):
    jitter: float = 1e-2

    NUM_POINTS = 100

    theta = jnp.linspace(0, 2 * np.pi, 2 * NUM_POINTS)
    phi = jnp.linspace(0, np.pi, NUM_POINTS)
    tp = jnp.array(np.meshgrid(theta, phi, indexing='ij'))
    tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

    def exponential_map(self, x, v):
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return x * jnp.cos(v_norm) + v * sindiv(v_norm)

    def log(self, x, y):
        xy = (x * y).sum(axis=-1, keepdims=True)
        xy = jnp.clip(xy, a_min=-1 + 1e-6, a_max=1 - 1e-6)
        val = jnp.arccos(xy)
        return divsin(val) * (y - xy * x)

    def tangent_projection(self, x, u):
        # Batch-safe projection: u - <x,u>x
        xu = jnp.sum(x * u, axis=-1, keepdims=True)
        return u - xu * x

    def tangent_orthonormal_basis(self, x, dF):
        """
        Returns (B, D, D-1) orthonormal tangent basis at x.
        Special-case D=2,3 for efficiency; general D uses vmapped QR.
        """
        assert x.ndim == 2 and dF.ndim == 2 and x.shape == dF.shape
        B, D = x.shape
        n = D - 1

        if D == 2:
            # S1 case
            E = x[:, jnp.array([1,0])] * jnp.array([-1., 1.])
            E = E.reshape(*E.shape, 1)
            return E

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
        u0_fb = jnp.where(na >= nb, u0a, u0b)
        u0_fb = _normalize(u0_fb, eps=eps)

        u0_from_dF = dF_tan / (nrm + eps)
        u0 = jnp.where(nrm > 1e-6, u0_from_dF, u0_fb)  # (B,D)

        if D == 3:
            # S2 case: u1 = normalize(x × u0)
            u1 = jnp.cross(x, u0)
            u1n = jnp.linalg.norm(u1, axis=-1, keepdims=True)
            u1_fb = jnp.cross(x, u0_fb)
            u1_fb = _normalize(u1_fb, eps=eps)
            u1 = jnp.where(u1n > 1e-6, u1 / (u1n + eps), u1_fb)
            E = jnp.stack([u0, u1], axis=-1)  # (B,3,2)
            return E

        # General D: complete basis via projected canonical basis + vmapped QR
        C = jnp.broadcast_to(I, (B, D, D))  # (B,D,D)
        xTC = jnp.einsum("bi,bij->bj", x, C)  # (B,D)
        Ctan = C - jnp.einsum("bi,bj->bij", x, xTC)  # tangent projection

        u0TC = jnp.einsum("bi,bij->bj", u0, Ctan)
        Crem = Ctan - jnp.einsum("bi,bj->bij", u0, u0TC)

        # vmap QR over batch
        def qr_one(A):
            Q, R = jnp.linalg.qr(A)
            return Q, R

        Q, _ = jax.vmap(qr_one)(Crem)  # Q: (B,D,D)
        rest = Q[..., : max(n - 1, 0)]  # (B,D,n-1)
        E = jnp.concatenate([u0[..., None], rest], axis=-1)  # (B,D,n)
        return E

    def dist(self, x, y):
        inner = jnp.matmul(x, y)
        inner = inner/(1 + self.jitter)
        return jnp.arccos(inner)

    def cost(self, x, y):
        return self.dist(x, y)**2 / 2.

    def projx(self, x):
        x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def transp(self, x, y, u):
        yu = jnp.sum(y * u, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        return u - yu/(1 + xy) * (x + y)

    def logdetexp(self, x, u):
        norm_u = jnp.linalg.norm(u, axis=-1)
        val = jnp.log(jnp.abs(sindiv(norm_u)))
        return (u.shape[-1]-2) * val


    def zero(self):
        y = jnp.zeros(self.D)
        y = y.at[...,0].set(-1.)
        return y

    def zero_like(self, x):
        y = jnp.zeros_like(x)
        y = y.at[...,0].set(-1.)
        return y

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return jnp.concatenate((jnp.zeros_like(x[..., :1]), x), axis=-1)

    def plot_samples(self, model_samples, kde_factor=0.1, save='t.png'):
        spherical_samples = utils.euclidean_to_spherical(model_samples)
        kde = SphericalKDE(
            spherical_samples[:,0], spherical_samples[:,1], bandwidth=kde_factor)
        heatmap = np.exp(kde(self.tp[:,0], self.tp[:,1]).reshape(
            2 * self.NUM_POINTS, self.NUM_POINTS))
        self.plot_mollweide(heatmap, save=save)

    def plot_density(self, log_prob_fn, save='t.png'):
        density = log_prob_fn(utils.spherical_to_euclidean(self.tp))
        density = jnp.exp(density)
        heatmap = density.reshape(2 * self.NUM_POINTS, self.NUM_POINTS)
        self.plot_mollweide(heatmap, save=save)

    def plot_mollweide(self, heatmap, save):
        tt, pp = np.meshgrid(
            self.theta - np.pi, self.phi - np.pi / 2, indexing='ij')

        proj = ccrs.Mollweide()
        fig = plt.figure(figsize=(3,2), dpi=200)
        ax = fig.add_subplot(111, projection='mollweide')
        norm = matplotlib.colors.Normalize()
        ax.pcolormesh(tt, pp, heatmap, cmap='magma', norm = norm)
        ax.set_axis_off()
        plt.savefig(save)
        os.system(f"convert {save} -trim {save} &")
        plt.close(fig)



class Euclidean(Manifold):
    def exponential_map(self, x, v):
        return x + v

    def tangent_projection(self, x, u):
        return u

    def cost(self, x, y):
        return 0.5 * self.dist(x,y)**2

    def dist(self, x, y):
        return - jnp.matmul(x, y)

    def tangent_orthonormal_basis(self, x, dF):
        tang_vecs = [jnp.eye(x.shape[1]) for i in range(x.shape[0])]
        return jnp.stack(tang_vecs, 0)



import re as _re


# -------------------------
# SO(n): General rotation group as flattened n×n matrices
# -------------------------
@dataclass
class SOn(Manifold):
    """
    SO(n) represented as flattened n×n rotation matrices in R^{n²}.

    Bi-invariant metric: <A,B> = (1/2) tr(A^T B) for A,B ∈ so(n).
    Manifold dimension: d = n(n-1)/2, ambient dimension D = n².
    """
    D: int  # = n²

    def __post_init__(self):
        n = int(jnp.sqrt(self.D + 0.5))
        assert n * n == self.D
        self.n = n

    def _to_matrix(self, x):
        return x.reshape(*x.shape[:-1], self.n, self.n)

    def _to_flat(self, M):
        return M.reshape(*M.shape[:-2], self.n * self.n)

    def _expm_batch(self, A):
        from jax.scipy.linalg import expm
        return jax.vmap(expm)(A)

    def _logm_ortho_single(self, R):
        eigvals, V = jnp.linalg.eig(R)
        log_eigvals = jnp.log(eigvals + 0j)
        V_inv = jnp.conj(V).T
        logR = (V @ jnp.diag(log_eigvals) @ V_inv).real
        return (logR - logR.T) / 2

    def _logm_batch(self, R):
        return jax.vmap(self._logm_ortho_single)(R)

    def projx(self, x):
        M = self._to_matrix(x)
        orig_shape = M.shape
        M_flat = M.reshape(-1, self.n, self.n)

        def _proj_single(m):
            U, _, Vt = jnp.linalg.svd(m)
            R = U @ Vt
            det_R = jnp.linalg.det(R)
            sign = jnp.where(det_R >= 0, 1.0, -1.0)
            correction = jnp.ones(self.n).at[-1].set(sign)
            return (U * correction) @ Vt

        R_flat = jax.vmap(_proj_single)(M_flat)
        return self._to_flat(R_flat.reshape(orig_shape))

    def exponential_map(self, x, v):
        Q = self._to_matrix(x)
        V = self._to_matrix(v)
        orig_shape = Q.shape
        Q_flat = Q.reshape(-1, self.n, self.n)
        V_flat = V.reshape(-1, self.n, self.n)
        A = jnp.einsum('bki,bkj->bij', Q_flat, V_flat)
        A = (A - jnp.swapaxes(A, -2, -1)) / 2
        eA = self._expm_batch(A)
        R = jnp.einsum('bik,bkj->bij', Q_flat, eA)
        return self._to_flat(R.reshape(orig_shape))

    def log(self, x, y):
        Q = self._to_matrix(x)
        P = self._to_matrix(y)
        orig_shape = Q.shape
        Q_flat = Q.reshape(-1, self.n, self.n)
        P_flat = P.reshape(-1, self.n, self.n)
        R = jnp.einsum('bki,bkj->bij', Q_flat, P_flat)
        A = self._logm_batch(R)
        V = jnp.einsum('bik,bkj->bij', Q_flat, A)
        return self._to_flat(V.reshape(orig_shape))

    def tangent_projection(self, x, u):
        Q = self._to_matrix(x)
        V = self._to_matrix(u)
        A = jnp.swapaxes(Q, -2, -1) @ V
        A_skew = (A - jnp.swapaxes(A, -2, -1)) / 2
        return self._to_flat(Q @ A_skew)

    def dist(self, x, y):
        n = self.n
        if x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[1]:
            B = x.shape[0]
            M = y.shape[1]
            X = x.reshape(B, n, n)
            Y = y.T.reshape(M, n, n)
            R = jnp.einsum('bki,mkj->bmij', X, Y)
            R_flat = R.reshape(B * M, n, n)
            eigvals = jax.vmap(jnp.linalg.eigvals)(R_flat)
            angles = jnp.angle(eigvals)
            dist_sq = 0.5 * jnp.sum(angles**2, axis=-1)
            return jnp.sqrt(dist_sq + eps).reshape(B, M)
        else:
            X = self._to_matrix(x)
            Y = self._to_matrix(y)
            R = jnp.swapaxes(X, -2, -1) @ Y
            orig_batch = R.shape[:-2]
            R_flat = R.reshape(-1, n, n)
            eigvals = jax.vmap(jnp.linalg.eigvals)(R_flat)
            angles = jnp.angle(eigvals)
            dist_sq = 0.5 * jnp.sum(angles**2, axis=-1)
            return jnp.sqrt(dist_sq + eps).reshape(orig_batch)

    def cost(self, x, y):
        return self.dist(x, y)**2 / 2.

    def tangent_orthonormal_basis(self, x, dF):
        n = self.n
        d = n * (n - 1) // 2
        B_size, D = x.shape
        Q = x.reshape(B_size, n, n)
        idx_i, idx_j = jnp.triu_indices(n, k=1)
        k_idx = jnp.arange(d)
        basis_skew = jnp.zeros((d, n, n))
        basis_skew = basis_skew.at[k_idx, idx_i, idx_j].set(1.0)
        basis_skew = basis_skew.at[k_idx, idx_j, idx_i].set(-1.0)
        tangent = jnp.einsum('bim,kmj->bkij', Q, basis_skew)
        tangent = tangent.reshape(B_size, d, D)
        return jnp.swapaxes(tangent, 1, 2)

    def zero(self):
        return self._to_flat(jnp.eye(self.n))

    def zero_like(self, x):
        I_flat = self._to_flat(jnp.eye(self.n))
        return jnp.broadcast_to(I_flat, x.shape) + jnp.zeros_like(x)

    def squeeze_tangent(self, v):
        V = self._to_matrix(v)
        idx_i, idx_j = jnp.triu_indices(self.n, k=1)
        return V[..., idx_i, idx_j]

    def unsqueeze_tangent(self, w):
        n = self.n
        batch_shape = w.shape[:-1]
        w_flat = w.reshape(-1, w.shape[-1])
        B = w_flat.shape[0]
        idx_i, idx_j = jnp.triu_indices(n, k=1)
        A = jnp.zeros((B, n, n))
        A = A.at[:, idx_i, idx_j].set(w_flat)
        A = A - jnp.swapaxes(A, -2, -1)
        return A.reshape(*batch_shape, n * n)

    def transp(self, x, y, u):
        Q = self._to_matrix(x)
        P = self._to_matrix(y)
        U = self._to_matrix(u)
        orig_shape = Q.shape
        Q_flat = Q.reshape(-1, self.n, self.n)
        P_flat = P.reshape(-1, self.n, self.n)
        U_flat = U.reshape(-1, self.n, self.n)
        R = jnp.einsum('bki,bkj->bij', Q_flat, P_flat)
        A = self._logm_batch(R)
        B_lie = jnp.einsum('bki,bkj->bij', Q_flat, U_flat)
        half_A = A / 2
        eA2 = self._expm_batch(half_A)
        emA2 = self._expm_batch(-half_A)
        B_new = jnp.einsum('bij,bjk,bkl->bil', emA2, B_lie, eA2)
        W = jnp.einsum('bik,bkj->bij', P_flat, B_new)
        return self._to_flat(W.reshape(orig_shape))

    def _log_sinc(self, x):
        x2 = x**2
        taylor = -x2 / 6.0 - x2 * x2 / 180.0
        x_safe = jnp.maximum(jnp.abs(x), 0.1)
        sinc_val = jnp.abs(jnp.sin(x_safe)) / x_safe
        sinc_val = jnp.maximum(sinc_val, 1e-30)
        exact = jnp.log(sinc_val)
        return jnp.where(jnp.abs(x) < 0.1, taylor, exact)

    def logdetexp(self, x, u):
        n = self.n
        k = n // 2
        Q = self._to_matrix(x)
        V = self._to_matrix(u)
        A = jnp.swapaxes(Q, -2, -1) @ V
        A = (A - jnp.swapaxes(A, -2, -1)) / 2
        orig_batch = A.shape[:-2]
        A_flat = A.reshape(-1, n, n)

        def _get_thetas(a):
            s = jnp.linalg.svd(a, compute_uv=False)
            return s[0::2][:k]

        thetas = jax.vmap(_get_thetas)(A_flat)
        if k > 1:
            theta_i = thetas[:, :, None]
            theta_j = thetas[:, None, :]
            diff_half = (theta_i - theta_j) / 2
            sum_half = (theta_i + theta_j) / 2
            mask = jnp.triu(jnp.ones((k, k), dtype=bool), k=1)
            pairwise = jnp.where(mask,
                self._log_sinc(diff_half) + self._log_sinc(sum_half), 0.0)
            logdet = 2.0 * jnp.sum(pairwise, axis=(-2, -1))
        else:
            logdet = jnp.zeros(A_flat.shape[0])
        if n % 2 == 1:
            logdet = logdet + 2.0 * jnp.sum(self._log_sinc(thetas / 2), axis=-1)
        return logdet.reshape(orig_batch)


def get(manifold):
    if manifold == 'S1':
        return Sphere(D = 2)
    elif manifold == 'S2':
        return Sphere(D = 3)
    elif manifold == 'R':
        return Euclidean(D = 1)

    # General SO(n): "SO<n>"
    m = _re.fullmatch(r"SO(\d+)", manifold)
    if m is not None:
        n = int(m.group(1))
        if n < 2:
            raise ValueError("SO(n) requires n >= 2")
        return SOn(D=n * n)

    raise ValueError(f"Unknown manifold: {manifold}")

@dataclass
class Product(Manifold):
    manifolds_str: str = 'S1,S1'

    def __post_init__(self):
        self.manifolds = []
        for man in self.manifolds_str.split(','):
            self.manifolds.append(get(man))

    def exponential_map(self, x, v):
        exp_prod = []
        d = 0
        for man in self.manifolds:
            exp_man = man.exponential_map(x[d:d+man.D], v[d:d+man.D])
            exp_prod.append(exp_man)
            d = d + man.D
        exp_prod = jnp.concatenate(exp_prod)
        return exp_prod

    def tangent_projection(self, x, u):
        proj_prod = []
        d = 0
        for man in self.manifolds:
            proj_man = man.tangent_projection(x[d:d+man.D], u[d:d+man.D])
            proj_prod.append(proj_man)
            d = d + man.D
        proj_prod = jnp.concatenate(proj_prod)
        return proj_prod

    def cost(self, x, y):
        cost_prod = jnp.zeros([x.shape[0], y.T.shape[0]])
        d = 0
        for man in self.manifolds:
            cost_prod += man.cost(x[:,d:d+man.D], y[d:d+man.D,:])
            d = d + man.D
        return cost_prod

    def dist(self, x, y):
        pass

    def tangent_orthonormal_basis(self, x, dF):
        d = 0
        map_block_diag = jax.vmap(block_diag)
        blocks = []
        for man in self.manifolds:
            onb_man = man.tangent_orthonormal_basis(x[:,d:d+man.D], dF[:,d:d+man.D])
            blocks.append(onb_man)
            d = d + man.D
        onb = map_block_diag(*(blocks))
        return onb

    def projx(self, x):
        x_proj = []
        d = 0
        for man in self.manifolds:
            x_proj_man = man.projx(x[:,d:d+man.D])
            d = d + man.D
            x_proj.append(x_proj_man)
        x_proj = jnp.concatenate(x_proj, 1)
        return x_proj

    def zero(self):
        """Zero point: concatenate zeros of each component."""
        parts = [man.zero() for man in self.manifolds]
        return jnp.concatenate(parts, axis=-1)

    def zero_like(self, x):
        """Zero point broadcast to batch shape."""
        parts = []
        d = 0
        for man in self.manifolds:
            parts.append(man.zero_like(x[..., d:d+man.D]))
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def log(self, x, y):
        """Logarithmic map: componentwise."""
        log_prod = []
        d = 0
        for man in self.manifolds:
            log_man = man.log(x[..., d:d+man.D], y[..., d:d+man.D])
            log_prod.append(log_man)
            d += man.D
        return jnp.concatenate(log_prod, axis=-1)

    def transp(self, x, y, u):
        """Parallel transport: componentwise."""
        transp_prod = []
        d = 0
        for man in self.manifolds:
            transp_man = man.transp(x[..., d:d+man.D], y[..., d:d+man.D], u[..., d:d+man.D])
            transp_prod.append(transp_man)
            d += man.D
        return jnp.concatenate(transp_prod, axis=-1)

    def logdetexp(self, x, u):
        """Log determinant of exponential map: sum over components."""
        logdet = 0.0
        d = 0
        for man in self.manifolds:
            logdet += man.logdetexp(x[..., d:d+man.D], u[..., d:d+man.D])
            d += man.D
        return logdet

    def squeeze_tangent(self, x):
        """Remove first coordinate from each component's tangent vector.

        For Product of spheres: (B, D_total) -> (B, tangent_dim_total)
        where tangent_dim_total = sum(D_i - 1) for each component.
        """
        parts = []
        d = 0
        for man in self.manifolds:
            # Each component: drop first coord (the normal direction)
            parts.append(x[..., d+1:d+man.D])
            d += man.D
        return jnp.concatenate(parts, axis=-1)

    def unsqueeze_tangent(self, x):
        """Insert zero as first coordinate for each component.

        For Product of spheres: (B, tangent_dim_total) -> (B, D_total)
        """
        parts = []
        t = 0  # tangent index
        for man in self.manifolds:
            tangent_dim = man.D - 1
            # Insert zero, then tangent coords for this component
            zero_part = jnp.zeros_like(x[..., t:t+1])
            parts.append(zero_part)
            parts.append(x[..., t:t+tangent_dim])
            t += tangent_dim
        return jnp.concatenate(parts, axis=-1)

    def plot_samples(self, model_samples, save='t.png'):
        pass

    def plot_density(self, log_prob_fn, save='t.png'):
        pass



@dataclass
class Torus(Product):
    manifolds: str = 'S1,S1'

    NUM_POINTS = 160

    theta = jnp.linspace(0, 2 * np.pi, 2 * NUM_POINTS)
    phi = jnp.linspace(0, 2 * np.pi, NUM_POINTS)
    tp = jnp.array(np.meshgrid(theta, phi, indexing='ij'))
    tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

    def plot_samples(self, model_samples, save='t.png'):
        theta1 = utils.S1euclideantospherical(model_samples[:,:2])
        theta2 = utils.S1euclideantospherical(model_samples[:,2:])

        x, y, z = utils.productS1toTorus(theta1, theta2)
        data = jnp.stack((x, y, z), 1)
        estimated_density = gaussian_kde(
                data.T, 0.2)

        x_grid, y_grid, z_grid = utils.productS1toTorus(self.tp[:,0], self.tp[:,1])
        grid = jnp.stack((x_grid, y_grid, z_grid), 1)
        probas_grid = estimated_density(grid.T)

        fig = plt.figure()
        ax = Axes3D(fig)
        #TODO: fix this - I negate become the mode is at the bottom of the torus in unimodal density
        ax.scatter(-x_grid, -y_grid, -z_grid, alpha = 0.2, c = probas_grid)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.axis('off')
        plt.savefig(save)


    def plot_density(self, log_prob_fn, save='t.png'):
        euc1 = jnp.stack((jnp.cos(self.tp[:,0]), jnp.sin(self.tp[:,0])),1)
        euc2 = jnp.stack((jnp.cos(self.tp[:,1]), jnp.sin(self.tp[:,1])),1)
        prod_euc = jnp.concatenate((euc1,euc2),1)

        density = log_prob_fn(prod_euc)
        density = jnp.exp(density)

        x_grid, y_grid, z_grid = utils.productS1toTorus(self.tp[:,0], self.tp[:,1])
        grid = jnp.stack((x_grid, y_grid, z_grid), 1)

        fig = plt.figure()
        plt.savefig(save)
        ax = Axes3D(fig)
        #TODO: fix this - I negate become the mode is at the bottom of the torus in unimodal density
        ax.scatter(-x_grid, -y_grid, -z_grid, alpha = 0.2, c = density)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.axis('off')

        plt.savefig(save)


@dataclass
class InfCylinder(Product):
    manifolds: str = 'S1,R'
