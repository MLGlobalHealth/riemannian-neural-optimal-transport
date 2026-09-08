import jax
import jax.numpy as jnp
from typing import Optional
import numpy as np


class ArgminSolver:
    """Solver for argmin_y [0.5*d(x,y)^2 - psi(y)] (c-transform in semi-dual OT).

    Supports:
    - Vanilla Riemannian GD
    - Riemannian Heavy Ball (momentum > 0)
    - Riemannian Adam (use_adam=True) - per-coordinate adaptive learning rates

    Optionally warm-starts from soft-argmin (logsumexp) over target samples.
    """

    def __init__(
        self,
        manifold,
        psi_module,
        inner_steps: int,
        inner_lr: float,
        grad_clip: Optional[float],
        lr_decay: bool,
        tolerance: float,
        min_steps: int,
        momentum: float = 0.0,
        eps: float = 1e-12,
        logsumexp_init: bool = False,
        logsumexp_gamma: float = 0.1,
        use_adam: bool = False,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
    ):
        self.manifold = manifold
        self.psi_module = psi_module
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.tolerance = tolerance
        self.min_steps = min_steps
        self.momentum = momentum
        self.eps = eps
        self.logsumexp_init = logsumexp_init
        self.logsumexp_gamma = logsumexp_gamma
        self.use_adam = use_adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        # Build solver functions
        self._solve_one_fn = self._build_solver()
        self.solve_one = jax.jit(self._solve_one_fn)
        # JIT the vmapped batch solve together for better fusion
        self._batch_solve_jit = jax.jit(jax.vmap(self._solve_one_fn, in_axes=(None, 0, 0)))

    def _psi_scalar(self, psi_params, y):
        return self.psi_module.apply({"params": psi_params}, y[None, :])[0]

    def _grad_psi(self, psi_params, y):
        g_amb = jax.grad(lambda yy: self._psi_scalar(psi_params, yy))(y)
        return self.manifold.tangent_projection(y, g_amb)

    def _stationarity_residual(self, psi_params, x, y):
        g = -self.manifold.log(y, x) - self._grad_psi(psi_params, y)
        return self.manifold.tangent_projection(y, g)

    def _build_solver(self):
        """Build the solver function.

        Supports three modes:
        1. use_adam=True: Riemannian Adam with per-coordinate adaptive LR
        2. momentum > 0: Riemannian Heavy Ball
        3. Otherwise: Vanilla Riemannian GD

        Accepts y_init for warm-starting (e.g., from logsumexp soft-argmin).
        """
        use_adam = self.use_adam
        use_momentum = (not use_adam) and (self.momentum > 0)

        if use_adam:
            # Adam solver
            def solve_one(psi_params, x, y_init):
                y0 = y_init
                m0 = jnp.zeros_like(x)  # first moment
                v0 = jnp.zeros_like(x)  # second moment
                done0 = jnp.array(False)
                res0 = jnp.array(jnp.inf, dtype=x.dtype)

                def body(i, carry):
                    y, m, v, done, res = carry

                    def do_nothing(_):
                        return (y, m, v, done, res)

                    def do_step(_):
                        g = self._stationarity_residual(psi_params, x, y)
                        res_new = jnp.linalg.norm(g)

                        if self.grad_clip is not None:
                            gn = res_new + self.eps
                            g = g * jnp.minimum(1.0, self.grad_clip / gn)

                        can_stop = (i + 1) >= self.min_steps
                        stop_now = jnp.logical_and(can_stop, res_new <= self.tolerance)

                        # Adam update
                        t = i + 1.0  # 1-indexed for bias correction
                        m_new = self.adam_beta1 * m + (1 - self.adam_beta1) * g
                        v_new = self.adam_beta2 * v + (1 - self.adam_beta2) * (g ** 2)

                        # Bias correction
                        m_hat = m_new / (1 - self.adam_beta1 ** t)
                        v_hat = v_new / (1 - self.adam_beta2 ** t)

                        # Adaptive update (note: negative because we minimize)
                        lr = self.inner_lr / jnp.sqrt(t) if self.lr_decay else self.inner_lr
                        update = -lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

                        y_new = self.manifold.projx(
                            self.manifold.exponential_map(y, update)
                        )

                        # Parallel transport first moment (tangent vector) to T_{y_new}
                        # Note: v (second moment = g**2) is NOT a tangent vector, so we don't transport
                        m_transported = self.manifold.transp(y, y_new, m_new)

                        return (y_new, m_transported, v_new, stop_now, res_new)

                    return jax.lax.cond(done, do_nothing, do_step, operand=None)

                y, _, _, _, final_res = jax.lax.fori_loop(
                    0, self.inner_steps, body, (y0, m0, v0, done0, res0)
                )
                return y, final_res

        elif use_momentum:
            # Heavy Ball solver
            def solve_one(psi_params, x, y_init):
                y0 = y_init
                v0 = jnp.zeros_like(x)  # velocity
                done0 = jnp.array(False)
                res0 = jnp.array(jnp.inf, dtype=x.dtype)

                def body(i, carry):
                    y, v, done, res = carry

                    def do_nothing(_):
                        return (y, v, done, res)

                    def do_step(_):
                        g = self._stationarity_residual(psi_params, x, y)
                        res_new = jnp.linalg.norm(g)

                        if self.grad_clip is not None:
                            gn = res_new + self.eps
                            g = g * jnp.minimum(1.0, self.grad_clip / gn)

                        can_stop = (i + 1) >= self.min_steps
                        stop_now = jnp.logical_and(can_stop, res_new <= self.tolerance)

                        lr = self.inner_lr / jnp.sqrt(i + 1.0) if self.lr_decay else self.inner_lr

                        # Riemannian Heavy Ball: v_new = momentum * v - lr * g
                        v_new = self.momentum * v - lr * g
                        y_new = self.manifold.projx(
                            self.manifold.exponential_map(y, v_new)
                        )
                        # Parallel transport v_new to tangent space at y_new
                        v_transported = self.manifold.transp(y, y_new, v_new)

                        return (y_new, v_transported, stop_now, res_new)

                    return jax.lax.cond(done, do_nothing, do_step, operand=None)

                y, _, _, final_res = jax.lax.fori_loop(
                    0, self.inner_steps, body, (y0, v0, done0, res0)
                )
                return y, final_res

        else:
            # Vanilla GD solver
            def solve_one(psi_params, x, y_init):
                y0 = y_init
                done0 = jnp.array(False)
                res0 = jnp.array(jnp.inf, dtype=x.dtype)

                def body(i, carry):
                    y, done, res = carry

                    def do_nothing(_):
                        return (y, done, res)

                    def do_step(_):
                        g = self._stationarity_residual(psi_params, x, y)
                        res_new = jnp.linalg.norm(g)

                        if self.grad_clip is not None:
                            gn = res_new + self.eps
                            g = g * jnp.minimum(1.0, self.grad_clip / gn)

                        can_stop = (i + 1) >= self.min_steps
                        stop_now = jnp.logical_and(can_stop, res_new <= self.tolerance)

                        lr = self.inner_lr / jnp.sqrt(i + 1.0) if self.lr_decay else self.inner_lr
                        y_new = self.manifold.projx(
                            self.manifold.exponential_map(y, -lr * g)
                        )

                        return (y_new, stop_now, res_new)

                    return jax.lax.cond(done, do_nothing, do_step, operand=None)

                y, _, final_res = jax.lax.fori_loop(
                    0, self.inner_steps, body, (y0, done0, res0)
                )
                return y, final_res

        return solve_one

    def _soft_argmin_init(self, psi_params, x, y_samples):
        """Compute soft-argmin initialization via logsumexp over target samples.

        Args:
            psi_params: Parameters of the potential network
            x: Source point (D,)
            y_samples: Target samples (K, D)

        Returns:
            y_init: Warm-start point on manifold (D,)
        """
        # Compute ψ(y) for all samples
        psi_vals = self.psi_module.apply({"params": psi_params}, y_samples)  # (K,)

        # Compute costs c(x, y_j) = 0.5 * d(x, y_j)^2 using batched cost
        # x[None, :] is (1, D), y_samples.T is (D, K) -> costs is (1, K)
        costs = self.manifold.cost(x[None, :], y_samples.T)[0]  # (K,)

        # Soft-argmin weights: softmax((ψ(y) - c(x,y)) / γ)
        logits = (psi_vals - costs) / self.logsumexp_gamma
        weights = jax.nn.softmax(logits)  # (K,)

        # Extrinsic weighted mean, projected back to manifold
        y_avg = jnp.sum(weights[:, None] * y_samples, axis=0)
        return self.manifold.projx(y_avg)

    def _soft_argmin_init_batch(self, psi_params, xs, y_samples):
        """Vectorized soft-argmin initialization for a batch of source points.

        Args:
            psi_params: Parameters of the potential network
            xs: Source points (N, D)
            y_samples: Target samples (K, D)

        Returns:
            y_inits: Warm-start points (N, D)
        """
        # Compute ψ(y) for all samples (shared across all x)
        psi_vals = self.psi_module.apply({"params": psi_params}, y_samples)  # (K,)

        # Compute pairwise costs efficiently using batched manifold.cost
        # This uses matrix multiplication for spheres: xs @ y_samples.T -> (N, K)
        costs = self.manifold.cost(xs, y_samples.T)  # (N, K)

        # Soft-argmin weights for each x: (N, K)
        logits = (psi_vals[None, :] - costs) / self.logsumexp_gamma
        weights = jax.nn.softmax(logits, axis=-1)

        # Extrinsic weighted mean for each x: (N, D)
        y_avgs = jnp.einsum('nk,kd->nd', weights, y_samples)
        return jax.vmap(self.manifold.projx)(y_avgs)

    def __call__(self, psi_params, x, y_samples=None):
        """Solve for a single point, returns (y, residual).

        Args:
            psi_params: Parameters of the potential network
            x: Source point (D,)
            y_samples: Optional target samples for logsumexp warm-start (K, D)
        """
        if self.logsumexp_init and y_samples is not None:
            # stop_gradient here since envelope trick doesn't need grads through init
            y_init = jax.lax.stop_gradient(
                self._soft_argmin_init(psi_params, x, y_samples)
            )
        else:
            y_init = jax.lax.stop_gradient(x)
        return self.solve_one(psi_params, x, y_init)

    def batch_solve(self, psi_params, xs, y_samples=None):
        """Solve for a batch of points, returns (ys, residuals).

        Args:
            psi_params: Parameters of the potential network
            xs: Source points (N, D)
            y_samples: Optional target samples for logsumexp warm-start (K, D)
        """
        if self.logsumexp_init and y_samples is not None:
            # stop_gradient here since envelope trick doesn't need grads through init
            y_inits = jax.lax.stop_gradient(
                self._soft_argmin_init_batch(psi_params, xs, y_samples)
            )
        else:
            y_inits = jax.lax.stop_gradient(xs)
        return self._batch_solve_jit(psi_params, xs, y_inits)

    def tighten(
        self,
        inner_steps: int = None,
        inner_lr: float = None,
        tolerance: float = None,
        logsumexp_gamma: float = None,
    ) -> "ArgminSolver":
        """
        Return a new solver with tightened convergence parameters.

        Useful for evaluation/inference where you want more accurate solutions
        than during training.

        Args:
            inner_steps: Override inner optimization steps (default: 2x current)
            inner_lr: Override learning rate (default: keep current)
            tolerance: Override convergence tolerance (default: 0.1x current)
            logsumexp_gamma: Override logsumexp temperature (default: 0.1x current)

        Returns:
            New ArgminSolver with tightened parameters
        """
        return ArgminSolver(
            manifold=self.manifold,
            psi_module=self.psi_module,
            inner_steps=inner_steps if inner_steps is not None else self.inner_steps * 2,
            inner_lr=inner_lr if inner_lr is not None else self.inner_lr,
            grad_clip=self.grad_clip,
            lr_decay=self.lr_decay,
            tolerance=tolerance if tolerance is not None else self.tolerance * 0.1,
            min_steps=self.min_steps,
            momentum=self.momentum,
            eps=self.eps,
            logsumexp_init=self.logsumexp_init,
            logsumexp_gamma=logsumexp_gamma if logsumexp_gamma is not None else self.logsumexp_gamma * 0.1,
            use_adam=self.use_adam,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
        )

    def analyze_convergence(
        self,
        psi_params,
        xs: jnp.ndarray,
        y_samples: jnp.ndarray = None,
        threshold: float = None,
    ) -> dict:
        """
        Analyze convergence across a batch of samples.

        Args:
            psi_params: Network parameters
            xs: Source points (N, D)
            y_samples: Target samples for logsumexp init (K, D)
            threshold: Residual threshold for "converged" (default: 10 * tolerance)

        Returns:
            dict with:
                'residuals': (N,) array of final residuals
                'ys': (N, D) solved points
                'converged_mask': (N,) boolean mask
                'converged_frac': fraction that converged
                'failing_indices': indices of non-converged samples
                'init_dists': (N,) distances from x to y_init
                'final_dists': (N,) distances from x to y_final
                'stats': dict with min/max/mean/median residuals
        """
        import numpy as np

        if threshold is None:
            threshold = 10 * self.tolerance

        # Get initializations
        if self.logsumexp_init and y_samples is not None:
            y_inits = self._soft_argmin_init_batch(psi_params, xs, y_samples)
        else:
            y_inits = xs

        # Compute init distances
        init_dists = jax.vmap(self.manifold.dist)(xs, y_inits)

        # Run JIT batch solve
        ys, residuals = self._batch_solve_jit(psi_params, xs, y_inits)

        # Compute final distances
        final_dists = jax.vmap(self.manifold.dist)(xs, ys)

        # Analyze convergence
        residuals_np = np.array(residuals)
        converged_mask = residuals_np < threshold
        converged_frac = np.mean(converged_mask)
        failing_indices = np.where(~converged_mask)[0]

        stats = {
            'min': float(np.min(residuals_np)),
            'max': float(np.max(residuals_np)),
            'mean': float(np.mean(residuals_np)),
            'median': float(np.median(residuals_np)),
            'std': float(np.std(residuals_np)),
            'p90': float(np.percentile(residuals_np, 90)),
            'p99': float(np.percentile(residuals_np, 99)),
        }

        return {
            'residuals': residuals,
            'ys': ys,
            'y_inits': y_inits,
            'converged_mask': converged_mask,
            'converged_frac': converged_frac,
            'failing_indices': failing_indices,
            'init_dists': init_dists,
            'final_dists': final_dists,
            'stats': stats,
            'xs': xs,
        }

    def print_convergence_report(self, analysis: dict, plot: bool = False, **plot_kwargs):
        """
        Print a formatted convergence report from analyze_convergence output.

        Args:
            analysis: Output dict from analyze_convergence()
            plot: If True, also display residual histogram
            **plot_kwargs: Passed to plot_residual_histogram (bins, log_scale, etc.)

        Returns:
            (fig, ax) if plot=True, else None
        """
        print("=" * 70)
        print("CONVERGENCE ANALYSIS")
        print("=" * 70)
        opt_str = "Adam" if self.use_adam else (f"HeavyBall(mom={self.momentum})" if self.momentum > 0 else "GD")
        print(f"Solver: {opt_str}, steps={self.inner_steps}, lr={self.inner_lr}, tol={self.tolerance}")
        print()

        stats = analysis['stats']
        print(f"Residual Statistics (N={len(analysis['residuals'])}):")
        print(f"  Min:    {stats['min']:.2e}")
        print(f"  Median: {stats['median']:.2e}")
        print(f"  Mean:   {stats['mean']:.2e}")
        print(f"  P90:    {stats['p90']:.2e}")
        print(f"  P99:    {stats['p99']:.2e}")
        print(f"  Max:    {stats['max']:.2e}")
        print()

        print(f"Convergence: {analysis['converged_frac']*100:.1f}% "
              f"({int(analysis['converged_mask'].sum())}/{len(analysis['residuals'])})")

        if len(analysis['failing_indices']) > 0:
            print(f"\nFailing samples ({len(analysis['failing_indices'])}):")
            for idx in analysis['failing_indices'][:10]:  # Show first 10
                res = float(analysis['residuals'][idx])
                init_d = float(analysis['init_dists'][idx])
                final_d = float(analysis['final_dists'][idx])
                print(f"  [{idx:4d}] res={res:.2e}, init_dist={init_d:.3f}, final_dist={final_d:.3f}")
            if len(analysis['failing_indices']) > 10:
                print(f"  ... and {len(analysis['failing_indices']) - 10} more")

        # Compare failing vs passing
        if len(analysis['failing_indices']) > 0 and analysis['converged_frac'] > 0:
            failing = analysis['failing_indices']
            passing = np.where(analysis['converged_mask'])[0]

            print(f"\nInit distance comparison:")
            print(f"  Passing samples: mean={np.mean(analysis['init_dists'][passing]):.3f}")
            print(f"  Failing samples: mean={np.mean(analysis['init_dists'][failing]):.3f}")

        if plot:
            return self.plot_residual_histogram(analysis, **plot_kwargs)
        return None

    def plot_residual_histogram(
        self,
        analysis: dict,
        bins: int = 50,
        log_scale: bool = True,
        show_threshold: bool = True,
        threshold: float = None,
        figsize: tuple = (10, 6),
        title: str = None,
    ):
        """
        Plot histogram of gradient residuals from analyze_convergence output.

        Args:
            analysis: Output dict from analyze_convergence()
            bins: Number of histogram bins
            log_scale: Use log scale for x-axis (residuals)
            show_threshold: Show convergence threshold line
            threshold: Override threshold (default: 10 * tolerance)
            figsize: Figure size tuple
            title: Custom title (auto-generated if None)

        Returns:
            fig, ax: matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        import numpy as np

        residuals = np.array(analysis['residuals'])

        if threshold is None:
            threshold = 10 * self.tolerance

        fig, ax = plt.subplots(figsize=figsize)

        if log_scale:
            # Use log-spaced bins for better visualization
            log_res = np.log10(residuals + 1e-16)
            bin_edges = np.logspace(log_res.min(), log_res.max(), bins + 1)
            ax.hist(residuals, bins=bin_edges, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xscale('log')
        else:
            ax.hist(residuals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        # Add threshold line
        if show_threshold:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold: {threshold:.1e}')

        # Add statistics annotations
        stats = analysis['stats']
        stats_text = (
            f"N = {len(residuals)}\n"
            f"Min: {stats['min']:.2e}\n"
            f"Median: {stats['median']:.2e}\n"
            f"Mean: {stats['mean']:.2e}\n"
            f"P99: {stats['p99']:.2e}\n"
            f"Max: {stats['max']:.2e}\n"
            f"Converged: {analysis['converged_frac']*100:.1f}%"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')

        ax.set_xlabel('Gradient Residual (||∇f||)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)

        if title is None:
            opt_str = "Adam" if self.use_adam else (f"HeavyBall(β={self.momentum})" if self.momentum > 0 else "GD")
            title = f"Residual Distribution | {opt_str}, {self.inner_steps} steps, lr={self.inner_lr}"
        ax.set_title(title, fontsize=12)

        if show_threshold:
            ax.legend(loc='upper left')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, ax

    def compute_monge_gap(
        self,
        psi_params,
        x_samples: jnp.ndarray,
        y_samples: jnp.ndarray,
        y_target: jnp.ndarray = None,
    ) -> dict:
        """
        Compute Monge gap: M(T) = E[c(x, T(x))] - W_c(μ, ν)

        The Monge gap measures how suboptimal the transport map is.
        M(T) = 0 iff T is the optimal Monge map.

        Args:
            psi_params: Network parameters
            x_samples: Source samples (N, D)
            y_samples: Target samples for logsumexp init (K, D)
            y_target: Target samples for ψ evaluation (M, D), defaults to y_samples

        Returns:
            dict with:
                'monge_gap': M(T) = transport_cost - dual_value (should be ≥ 0)
                'transport_cost': E[c(x, T(x))] = E[0.5 * d(x,y*)²]
                'dual_value': E[φ(x)] + E[ψ(y)] ≈ W_c (lower bound)
                'psi_transported': E[ψ(T(x))]
                'psi_target': E[ψ(y)]
                'phi_values': E[φ(x)] = E[c(x,y*) - ψ(y*)]
        """
        if y_target is None:
            y_target = y_samples

        # Solve for transported points y* = T(x)
        y_star, residuals = self.batch_solve(psi_params, x_samples, y_samples)

        # Compute transport cost: c(x, y*) = 0.5 * d(x, y*)²
        dists_sq = jax.vmap(lambda x, y: self.manifold.dist(x, y) ** 2)(x_samples, y_star)
        c_xy = 0.5 * dists_sq
        transport_cost = jnp.mean(c_xy)

        # Compute ψ(y*) for transported points
        psi_y_star = self.psi_module.apply({"params": psi_params}, y_star)
        psi_transported = jnp.mean(psi_y_star)

        # Compute φ(x) = c(x, y*) - ψ(y*)  [c-transform]
        phi_x = c_xy - psi_y_star
        phi_mean = jnp.mean(phi_x)

        # Compute ψ(y) for target samples
        psi_y = self.psi_module.apply({"params": psi_params}, y_target)
        psi_target = jnp.mean(psi_y)

        # Dual value: W_c ≈ E[φ(x)] + E[ψ(y)]
        dual_value = phi_mean + psi_target

        # Monge gap: should be ≥ 0, = 0 at optimality
        # Equivalent forms: M(T) = transport_cost - dual_value = E[ψ(T(x))] - E[ψ(y)]
        monge_gap = psi_transported - psi_target  # simplified form

        return {
            'monge_gap': float(monge_gap),
            'transport_cost': float(transport_cost),
            'dual_value': float(dual_value),
            'psi_transported': float(psi_transported),
            'psi_target': float(psi_target),
            'phi_mean': float(phi_mean),
            'mean_residual': float(jnp.mean(residuals)),
            'mean_dist': float(jnp.mean(jnp.sqrt(dists_sq))),
        }
