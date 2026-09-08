import jax
import jax.numpy as jnp

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
