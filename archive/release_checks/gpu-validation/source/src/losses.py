import jax
import jax.numpy as jnp

from src.barycenter import intrinsic_entropic_barycentric_map


class EntropicSemiDualLoss:
    """Entropic semi-dual OT loss via Monte Carlo logsumexp.

    Replaces the inner argmin solver with:
        φ^ε(x) = -ε · logsumexp_j [(ψ(y_j) - c(x_i, y_j)) / ε] + ε · log(K)

    Fully differentiable — no envelope theorem or stop_gradient needed.
    """

    def __init__(self, manifold, psi_module, epsilon: float,
                 intrinsic_barycenter: bool = False,
                 barycenter_iters: int = 32,
                 barycenter_step_size: float = 0.5):
        self.manifold = manifold
        self.psi_module = psi_module
        self.epsilon = epsilon
        self.intrinsic_barycenter = intrinsic_barycenter
        self.barycenter_iters = barycenter_iters
        self.barycenter_step_size = barycenter_step_size

    def _compute_phi_batch(self, psi_params, x_batch, y_samples):
        """Compute entropic c-transform φ^ε(x) for a batch.

        Args:
            psi_params: Parameters of the potential network
            x_batch: Source points (N, D)
            y_samples: Target samples (K, D)

        Returns:
            phi_values: (N,) array of φ^ε values
        """
        # ψ(y_j) for all target samples
        psi_y = self.psi_module.apply({"params": psi_params}, y_samples)  # (K,)
        # Gauge fix: center g_θ(y) = a_θ(y) - E_ν[a_θ] (§3, paper)
        psi_y = psi_y - jnp.mean(psi_y)
        # Pairwise costs c(x_i, y_j) = 0.5 * d(x_i, y_j)^2
        costs = self.manifold.cost(x_batch, y_samples.T)  # (N, K)
        # φ^ε(x_i) = -ε * logsumexp((g_θ(y_j) - c(x_i, y_j)) / ε) + ε * log(K)
        K = y_samples.shape[0]
        logits = (psi_y[None, :] - costs) / self.epsilon  # (N, K)
        phi = -self.epsilon * jax.nn.logsumexp(logits, axis=-1)  # (N,)
        phi += self.epsilon * jnp.log(K)  # MC normalization
        return phi

    def _compute_transport_map(self, psi_params, x_batch, y_samples):
        """Compute barycentric projection (soft transport map) T^ε(x).

        Returns the softmin-weighted average of target samples.

        Args:
            psi_params: Parameters of the potential network
            x_batch: Source points (N, D)
            y_samples: Target samples (K, D)

        Returns:
            y_transported: (N, D) soft transport map output
            weights: (N, K) soft assignment weights
        """
        psi_y = self.psi_module.apply({"params": psi_params}, y_samples)  # (K,)
        psi_y = psi_y - jnp.mean(psi_y)  # gauge centering
        costs = self.manifold.cost(x_batch, y_samples.T)  # (N, K)
        logits = (psi_y[None, :] - costs) / self.epsilon  # (N, K)
        weights = jax.nn.softmax(logits, axis=-1)  # (N, K)
        if self.intrinsic_barycenter:
            # Intrinsic entropic barycentric map (scan-based, autodiff friendly)
            def _transport_one(x_i):
                z, _ = intrinsic_entropic_barycentric_map(
                    x=x_i,
                    psi_params=psi_params,
                    psi_module=self.psi_module,
                    manifold=self.manifold,
                    y_samples=y_samples,
                    epsilon=self.epsilon,
                    num_steps=self.barycenter_iters,
                    step_size=self.barycenter_step_size,
                )
                return z
            y_transported = jax.vmap(_transport_one)(x_batch)
        else:
            # Extrinsic weighted mean, projected to manifold
            y_avg = jnp.einsum('nk,kd->nd', weights, y_samples)  # (N, D)
            y_transported = jax.vmap(self.manifold.projx)(y_avg)
        return y_transported, weights

    def compute_transport_map_chunked(self, psi_params, x_batch, y_samples,
                                      y_chunk_size=256):
        """Memory-efficient barycentric projection, chunked over y_support.

        Decomposes the global softmax into per-chunk softmax + cross-chunk
        reweighting using the logsumexp identity:
            global_avg = Σ_c (Z_c / Z_total) · avg_within_chunk_c

        Peak memory: O(N × y_chunk_size × D) instead of O(N × K × D).

        Args:
            psi_params: Parameters of the potential network
            x_batch: Source points (N, D)
            y_samples: Target samples (K, D) — full support, shared across x chunks
            y_chunk_size: Number of target points per chunk

        Returns:
            y_transported: (N, D) soft transport map output
            diagnostics: dict with effective_K (N,)
        """
        K = y_samples.shape[0]

        # Fall back to non-chunked path if support fits in one chunk
        if K <= y_chunk_size:
            y_transported, weights = self._compute_transport_map(
                psi_params, x_batch, y_samples)
            log_w = jnp.log(weights + 1e-30)
            eff_K = jnp.exp(-jnp.sum(weights * log_w, axis=-1))
            return y_transported, {"effective_K": eff_K}

        # JIT-compiled per-chunk computation
        @jax.jit
        def _chunk_stats(x_batch, y_chunk, psi_y_chunk):
            costs = self.manifold.cost(x_batch, y_chunk.T)         # (N, C)
            logits = (psi_y_chunk[None, :] - costs) / self.epsilon # (N, C)
            lse = jax.nn.logsumexp(logits, axis=-1)                # (N,)
            w = jax.nn.softmax(logits, axis=-1)                    # (N, C)
            numer = jnp.einsum('nc,cd->nd', w, y_chunk)            # (N, D)
            return lse, numer

        # Compute centered psi for all y_samples (gauge fix)
        psi_y_all = self.psi_module.apply({"params": psi_params}, y_samples)
        psi_y_all = psi_y_all - jnp.mean(psi_y_all)

        # Pass 1: per-chunk logsumexp and within-chunk weighted average
        chunk_lses = []
        chunk_numers = []
        for j0 in range(0, K, y_chunk_size):
            y_chunk = y_samples[j0:j0 + y_chunk_size]
            psi_y_chunk = psi_y_all[j0:j0 + y_chunk_size]
            lse, numer = _chunk_stats(x_batch, y_chunk, psi_y_chunk)
            chunk_lses.append(lse)
            chunk_numers.append(numer)

        # Pass 2: combine via cross-chunk softmax
        all_lses = jnp.stack(chunk_lses, axis=-1)             # (N, n_chunks)
        chunk_weights = jax.nn.softmax(all_lses, axis=-1)     # (N, n_chunks)

        y_avg = sum(chunk_weights[:, i:i+1] * chunk_numers[i]
                    for i in range(len(chunk_numers)))

        y_transported = jax.vmap(self.manifold.projx)(y_avg)

        # Effective K from global logsumexp (approximate)
        global_lse = jax.nn.logsumexp(all_lses, axis=-1)      # (N,)
        # H = log(K) - (1/K_eff equivalent), approximate via chunk entropies
        chunk_entropy = -jnp.sum(
            chunk_weights * jnp.log(chunk_weights + 1e-30), axis=-1)
        eff_K_approx = jnp.exp(chunk_entropy) * y_chunk_size

        return y_transported, {"effective_K": eff_K_approx}

    def _loss_with_aux(self, psi_params, x_batch, y_batch,
                       compute_transport=False):
        """Compute entropic semi-dual loss with auxiliary outputs.

        Loss = -E[φ^ε(x)] - E[ψ(y)]

        Args:
            compute_transport: If True, compute barycentric transport map
                and transport cost (expensive). If False, only compute
                effective_K from softmax weights (cheap).

        Returns:
            loss: scalar loss value
            aux: dict with effective_K, and transport_cost if requested
        """
        psi_y = self.psi_module.apply({"params": psi_params}, y_batch)
        psi_y = psi_y - jnp.mean(psi_y)  # gauge centering: g_θ = a_θ - E_ν[a_θ]
        phi_x = self._compute_phi_batch(psi_params, x_batch, y_batch)
        # After centering, E[g_θ] = 0, so loss = -E[φ^ε(x)] (the g term vanishes)
        loss = -jnp.mean(phi_x) - jnp.mean(psi_y)

        # Effective K from softmax weights (cheap — reuses logits)
        costs = self.manifold.cost(x_batch, y_batch.T)  # (N, K)
        logits = (psi_y[None, :] - costs) / self.epsilon  # (N, K)
        log_w = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        entropy = -jnp.sum(jnp.exp(log_w) * log_w, axis=-1)  # (N,)
        effective_K = jnp.mean(jnp.exp(entropy))

        aux = {"effective_K": effective_K}

        if compute_transport:
            y_transported, _ = self._compute_transport_map(
                psi_params, x_batch, y_batch
            )
            d = jax.vmap(self.manifold.dist)(x_batch, y_transported)
            aux["transport_cost"] = jnp.mean(0.5 * d * d)
        else:
            aux["transport_cost"] = jnp.float32(0.0)

        return loss, aux

    def __call__(self, psi_params, x_batch, y_batch, return_aux=False,
                 compute_transport=False):
        """Compute loss, optionally with auxiliary outputs."""
        if return_aux:
            return self._loss_with_aux(psi_params, x_batch, y_batch,
                                       compute_transport=compute_transport)
        else:
            loss, _ = self._loss_with_aux(psi_params, x_batch, y_batch)
            return loss


class SemiDualLoss:
    """Semi-dual optimal transport loss with envelope trick"""

    def __init__(self, manifold, psi_module, solver):
        self.manifold = manifold
        self.psi_module = psi_module
        self.solver = solver

    def _compute_phi_batch_with_aux(self, psi_params, x_batch, y_samples):
        """Compute φ[ψ](x) for a batch of x, returning auxiliary info.

        Args:
            psi_params: Parameters of the potential network
            x_batch: Source points (N, D)
            y_samples: Target samples for logsumexp warm-start (K, D)

        Returns:
            phi_values: (N,) array of φ values
            aux: dict with y_star, residuals, transport_cost
        """
        # Solve for y* with optional logsumexp warm-start
        # Stop gradients on INPUTS to prevent JAX from tracing through the solver.
        # Envelope theorem: we only need y*, not dy*/dψ. The gradient flows through
        # psi_module.apply(psi_params, y_star) below, not through the solver.
        y_star, residuals = self.solver.batch_solve(
            jax.lax.stop_gradient(psi_params),
            jax.lax.stop_gradient(x_batch),
            jax.lax.stop_gradient(y_samples) if y_samples is not None else None
        )

        # Compute φ(x) = 0.5*d(x,y*)^2 - ψ(y*)
        psi_y = self.psi_module.apply({"params": psi_params}, y_star)
        d = jax.vmap(self.manifold.dist)(x_batch, y_star)
        half_d_sq = 0.5 * d * d
        phi_values = half_d_sq - psi_y

        aux = {
            "y_star": y_star,
            "residuals": residuals,
            "mean_residual": jnp.mean(residuals),
            "transport_cost": jnp.mean(half_d_sq),
        }

        return phi_values, aux

    def _loss_with_aux(self, psi_params, x_batch, y_batch):
        """Compute semi-dual loss with auxiliary outputs.

        Returns:
            loss: scalar loss value
            aux: dict with y_star, residuals, transport_cost, mean_residual
        """
        psi_y = self.psi_module.apply({"params": psi_params}, y_batch)
        phi_x, aux = self._compute_phi_batch_with_aux(psi_params, x_batch, y_batch)
        loss = -jnp.mean(phi_x) - jnp.mean(psi_y)
        return loss, aux

    def __call__(self, psi_params, x_batch, y_batch, return_aux=False):
        """Compute loss, optionally with auxiliary outputs.

        Args:
            psi_params: Network parameters
            x_batch: Source samples (N, D)
            y_batch: Target samples (N, D)
            return_aux: If True, return (loss, aux) tuple

        Returns:
            loss if return_aux=False, else (loss, aux)
        """
        if return_aux:
            return self._loss_with_aux(psi_params, x_batch, y_batch)
        else:
            # Fast path without aux (for backward compat)
            loss, _ = self._loss_with_aux(psi_params, x_batch, y_batch)
            return loss