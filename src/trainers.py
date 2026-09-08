import time
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.training import train_state


# -----------------------------
# EMA utilities
# -----------------------------

def ema_init(params):
    """Initialize EMA params as a copy."""
    return jtu.tree_map(lambda p: jnp.array(p), params)


def ema_update(ema_params, params, decay=0.999):
    """Update EMA: ema = decay * ema + (1 - decay) * params"""
    return jtu.tree_map(
        lambda e, p: decay * e + (1 - decay) * p,
        ema_params, params
    )


class EMACallback:
    """Track exponential moving average of parameters during training.

    Usage:
        ema_cb = EMACallback(decay=0.999)
        state, history = trainer.train(..., callbacks=[ema_cb])

        # Use EMA params for final evaluation
        ema_params = ema_cb.ema_params
    """

    def __init__(self, decay: float = 0.999, eval_with_ema: bool = True):
        """
        Args:
            decay: EMA decay rate (0.999 = slow update, 0.99 = faster)
            eval_with_ema: If True, print EMA metrics during eval
        """
        self.decay = decay
        self.eval_with_ema = eval_with_ema
        self.ema_params = None
        self.history = {
            "eval_loss_ema": [],
            "eval_transport_cost_ema": [],
        }

    def on_train_begin(self, trainer, state):
        self.ema_params = ema_init(state.params)

    def on_step(self, trainer, state, step):
        """Update EMA after every training step."""
        self.ema_params = ema_update(self.ema_params, state.params, self.decay)

    def on_log(self, trainer, state, step, history):
        pass  # EMA update happens in on_step

    def on_eval(self, trainer, state, step, x_eval, y_eval, history):
        if not self.eval_with_ema:
            return

        # Evaluate with EMA params
        eval_loss_ema, transport_cost_ema, residual_ema = trainer.eval_fn(
            self.ema_params, x_eval, y_eval
        )

        self.history["eval_loss_ema"].append(float(eval_loss_ema))
        self.history["eval_transport_cost_ema"].append(float(transport_cost_ema))

        print(f"       [EMA] loss={eval_loss_ema:.6f} | transport={transport_cost_ema:.6f}")

    def on_train_end(self, trainer, state, history):
        # Merge EMA history into main history
        history["eval_loss_ema"] = self.history["eval_loss_ema"]
        history["eval_transport_cost_ema"] = self.history["eval_transport_cost_ema"]


class EntropicTrainer:
    """Trainer for entropic semi-dual OT (no inner solver loop)."""

    def __init__(
        self,
        manifold,
        psi_module,
        loss_fn,
        learning_rate: float,
        lr_decay: bool = False,
        lr_decay_alpha: float = 0.01,
        n_steps: int = 1500,
        grad_accum_steps: int = 0,
    ):
        self.manifold = manifold
        self.psi_module = psi_module
        self.loss_fn = loss_fn
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        if lr_decay:
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=n_steps,
                alpha=lr_decay_alpha,
            )
            self.tx = optax.adamw(schedule)
        else:
            self.tx = optax.adamw(learning_rate)

        self._build_step_fn()
        self._build_eval_fn()

    def _build_step_fn(self):
        """Build JIT-compiled training step.

        Training steps skip the barycentric transport map (expensive).
        Only effective_K is computed as a cheap diagnostic from softmax weights.
        Transport cost is computed at eval time only.
        """

        def loss_with_aux(params, x_batch, y_batch):
            return self.loss_fn(params, x_batch, y_batch,
                                return_aux=True, compute_transport=False)

        if self.grad_accum_steps == 1:
            @jax.jit
            def step(state, x_batch, y_batch):
                (loss, aux), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
                    state.params, x_batch, y_batch
                )
                gnorm = optax.global_norm(grads)
                state = state.apply_gradients(grads=grads)
                return state, loss, gnorm, aux["effective_K"]

            self.step_fn = step
        else:
            @jax.jit
            def grad_fn(params, x_batch, y_batch):
                (loss, aux), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
                    params, x_batch, y_batch
                )
                return grads, loss, aux["effective_K"]

            @jax.jit
            def apply_fn(state, grads):
                gnorm = optax.global_norm(grads)
                state = state.apply_gradients(grads=grads)
                return state, gnorm

            self._grad_fn = grad_fn
            self._apply_fn = apply_fn
            self.step_fn = None

    def _build_eval_fn(self):
        """Build JIT-compiled evaluation.

        Computes entropic loss and transport cost (via barycentric projection).
        Transport cost is only computed here, not during training steps.
        """

        @jax.jit
        def eval_metrics(params, x_eval, y_eval):
            loss, aux = self.loss_fn(params, x_eval, y_eval,
                                     return_aux=True, compute_transport=True)
            return loss, aux["transport_cost"], aux["effective_K"]

        self.eval_fn = eval_metrics

    def init_state(self, params):
        """Initialize training state"""
        return train_state.TrainState.create(
            apply_fn=self.psi_module.apply,
            params=params,
            tx=self.tx,
        )

    def train(
        self,
        state,
        base_density,
        target_density,
        key,
        n_steps: int,
        batch_size: int,
        log_every: int = 50,
        eval_every: int = None,
        eval_size: int = 4096,
        callbacks: Optional[list] = None,
    ):
        """Main training loop"""
        callbacks = callbacks or []

        history = {
            "iterations": [],
            "train_loss": [],
            "train_effective_K": [],
            "grad_norm": [],
            "eval_loss": [],
            "eval_transport_cost": [],
            "eval_effective_K": [],
        }

        for callback in callbacks:
            callback.on_train_begin(self, state)

        t0 = time.time()

        for it in range(1, n_steps + 1):
            if self.grad_accum_steps == 1:
                key, kx, ky = jax.random.split(key, 3)
                x = base_density.sample(kx, batch_size)
                y = target_density.sample(ky, batch_size)
                state, loss, gnorm, eff_K = self.step_fn(state, x, y)
            else:
                accum_grads = None
                accum_loss = 0.0
                accum_eff_K = 0.0
                K = self.grad_accum_steps

                for _ in range(K):
                    key, kx, ky = jax.random.split(key, 3)
                    x_micro = base_density.sample(kx, batch_size)
                    y_micro = target_density.sample(ky, batch_size)

                    grads, loss_i, ek_i = self._grad_fn(
                        state.params, x_micro, y_micro
                    )

                    if accum_grads is None:
                        accum_grads = grads
                    else:
                        accum_grads = jtu.tree_map(
                            lambda a, g: a + g, accum_grads, grads
                        )
                    accum_loss += float(loss_i)
                    accum_eff_K += float(ek_i)

                accum_grads = jtu.tree_map(lambda g: g / K, accum_grads)
                state, gnorm = self._apply_fn(state, accum_grads)
                loss = accum_loss / K
                eff_K = accum_eff_K / K

            for callback in callbacks:
                if hasattr(callback, 'on_step'):
                    callback.on_step(self, state, it)

            if it % log_every == 0:
                history["iterations"].append(it)
                history["train_loss"].append(float(loss))
                history["train_effective_K"].append(float(eff_K))
                history["grad_norm"].append(float(gnorm))

                dt = time.time() - t0
                print(
                    f"[{it:6d}] loss={float(loss):.4f} "
                    f"| eff_K={float(eff_K):.1f} | grad={float(gnorm):.2e} | {dt:.1f}s"
                )
                t0 = time.time()

                for callback in callbacks:
                    callback.on_log(self, state, it, history)

            if eval_every is not None and it % eval_every == 0:
                key, k_eval_x, k_eval_y = jax.random.split(key, 3)
                x_eval = base_density.sample(k_eval_x, eval_size)
                y_eval = target_density.sample(k_eval_y, eval_size)

                eval_loss, transport_cost, eff_K = self.eval_fn(state.params, x_eval, y_eval)
                eval_loss = float(eval_loss)
                transport_cost = float(transport_cost)
                eff_K = float(eff_K)

                history["eval_loss"].append(eval_loss)
                history["eval_transport_cost"].append(transport_cost)
                history["eval_effective_K"].append(eff_K)

                print(f" ---> eval_loss={eval_loss:.6f} | transport={transport_cost:.6f} | eff_K={eff_K:.1f}")

                for callback in callbacks:
                    callback.on_eval(self, state, it, x_eval, y_eval, history)

        for callback in callbacks:
            callback.on_train_end(self, state, history)

        return state, history


class SemiDualTrainer:
    """Trainer for semi-dual optimal transport"""

    def __init__(
        self,
        manifold,
        psi_module,
        loss_fn,
        solver,
        learning_rate: float,
        lr_decay: bool = False,
        lr_decay_alpha: float = 0.01,
        n_steps: int = 1500,
        grad_accum_steps: int = 0,
    ):
        self.manifold = manifold
        self.psi_module = psi_module
        self.loss_fn = loss_fn
        self.solver = solver
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        if lr_decay:
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=n_steps,
                alpha=lr_decay_alpha,
            )
            self.tx = optax.adamw(schedule)
        else:
            self.tx = optax.adamw(learning_rate)

        self._build_step_fn()
        self._build_eval_fn()

    def _build_step_fn(self):
        """Build JIT-compiled training step with auxiliary outputs."""

        def loss_with_aux(params, x_batch, y_batch):
            return self.loss_fn(params, x_batch, y_batch, return_aux=True)

        if self.grad_accum_steps == 1:
            # Fast path: single step, no accumulation overhead
            @jax.jit
            def step(state, x_batch, y_batch):
                (loss, aux), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
                    state.params, x_batch, y_batch
                )
                gnorm = optax.global_norm(grads)
                state = state.apply_gradients(grads=grads)
                return state, loss, gnorm, aux["mean_residual"], aux["transport_cost"]

            self.step_fn = step
        else:
            # Gradient accumulation: separate grad computation from apply
            @jax.jit
            def grad_fn(params, x_batch, y_batch):
                (loss, aux), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
                    params, x_batch, y_batch
                )
                return grads, loss, aux["mean_residual"], aux["transport_cost"]

            @jax.jit
            def apply_fn(state, grads):
                gnorm = optax.global_norm(grads)
                state = state.apply_gradients(grads=grads)
                return state, gnorm

            self._grad_fn = grad_fn
            self._apply_fn = apply_fn
            self.step_fn = None  # use accum path in train()

    def _build_eval_fn(self):
        """Build JIT-compiled evaluation metrics.

        Computes loss and transport cost in a single solver pass.
        Loss = -E[φ(x)] - E[ψ(y)] = E[ψ(y*) - 0.5*d(x,y*)²] - E[ψ(y)]
        """

        @jax.jit
        def eval_metrics(params, x_eval, y_eval):
            # Run solver once
            y_star, residuals = self.solver.batch_solve(params, x_eval, y_eval)

            # Compute distances and transport cost
            d = jax.vmap(self.manifold.dist)(x_eval, y_star)
            half_d_sq = 0.5 * d * d
            transport_cost = jnp.mean(half_d_sq)

            # Compute loss: -E[φ(x)] - E[ψ(y)]
            # where φ(x) = 0.5*d(x,y*)² - ψ(y*)
            psi_y_star = self.psi_module.apply({"params": params}, y_star)
            psi_y = self.psi_module.apply({"params": params}, y_eval)
            eval_loss = jnp.mean(psi_y_star - half_d_sq) - jnp.mean(psi_y)

            mean_residual = jnp.mean(residuals)

            return eval_loss, transport_cost, mean_residual

        self.eval_fn = eval_metrics

    def init_state(self, params):
        """Initialize training state"""
        return train_state.TrainState.create(
            apply_fn=self.psi_module.apply,
            params=params,
            tx=self.tx,
        )

    def train(
        self,
        state,
        base_density,
        target_density,
        key,
        n_steps: int,
        batch_size: int,
        log_every: int = 50,
        eval_every: int = None,
        eval_size: int = 4096,
        callbacks: Optional[list] = None,
    ):
        """Main training loop"""
        callbacks = callbacks or []

        history = {
            "iterations": [],
            "train_loss": [],
            "train_residual": [],
            "train_transport_cost": [],
            "grad_norm": [],
            "eval_loss": [],
            "eval_transport_cost": [],
            "eval_solver_residual": [],
        }

        for callback in callbacks:
            callback.on_train_begin(self, state)

        t0 = time.time()

        for it in range(1, n_steps + 1):
            if self.grad_accum_steps == 1:
                # Standard single-batch step
                key, kx, ky = jax.random.split(key, 3)
                x = base_density.sample(kx, batch_size)
                y = target_density.sample(ky, batch_size)
                state, loss, gnorm, residual, transport_cost = self.step_fn(state, x, y)
            else:
                # Gradient accumulation over micro-batches
                accum_grads = None
                accum_loss = 0.0
                accum_residual = 0.0
                accum_transport = 0.0
                K = self.grad_accum_steps

                for _ in range(K):
                    key, kx, ky = jax.random.split(key, 3)
                    x_micro = base_density.sample(kx, batch_size)
                    y_micro = target_density.sample(ky, batch_size)

                    grads, loss_i, res_i, tc_i = self._grad_fn(
                        state.params, x_micro, y_micro
                    )

                    if accum_grads is None:
                        accum_grads = grads
                    else:
                        accum_grads = jtu.tree_map(
                            lambda a, g: a + g, accum_grads, grads
                        )
                    accum_loss += float(loss_i)
                    accum_residual += float(res_i)
                    accum_transport += float(tc_i)

                # Average gradients and apply
                accum_grads = jtu.tree_map(lambda g: g / K, accum_grads)
                state, gnorm = self._apply_fn(state, accum_grads)
                loss = accum_loss / K
                residual = accum_residual / K
                transport_cost = accum_transport / K

            # Call on_step for every training step (used by EMA)
            for callback in callbacks:
                if hasattr(callback, 'on_step'):
                    callback.on_step(self, state, it)

            if it % log_every == 0:
                # Use values from step_fn (pre-update params, but avoids recomputation)
                history["iterations"].append(it)
                history["train_loss"].append(float(loss))
                history["train_residual"].append(float(residual))
                history["train_transport_cost"].append(float(transport_cost))
                history["grad_norm"].append(float(gnorm))

                dt = time.time() - t0
                print(
                    f"[{it:6d}] loss={float(loss):.4f} "
                    f"| res={float(residual):.2e} | grad={float(gnorm):.2e} | {dt:.1f}s"
                )
                t0 = time.time()

                for callback in callbacks:
                    callback.on_log(self, state, it, history)

            if eval_every is not None and it % eval_every == 0:
                # Sample fresh eval data each time
                key, k_eval_x, k_eval_y = jax.random.split(key, 3)
                x_eval = base_density.sample(k_eval_x, eval_size)
                y_eval = target_density.sample(k_eval_y, eval_size)

                eval_loss, transport_cost, mean_residual = self.eval_fn(state.params, x_eval, y_eval)
                eval_loss = float(eval_loss)
                transport_cost = float(transport_cost)
                mean_residual = float(mean_residual)

                history["eval_loss"].append(eval_loss)
                history["eval_transport_cost"].append(transport_cost)
                history["eval_solver_residual"].append(mean_residual)

                print(f" ---> eval_loss={eval_loss:.6f} | transport_cost={transport_cost:.6f} | solver_res={mean_residual:.3e}")

                for callback in callbacks:
                    callback.on_eval(self, state, it, x_eval, y_eval, history)

        for callback in callbacks:
            callback.on_train_end(self, state, history)

        return state, history
