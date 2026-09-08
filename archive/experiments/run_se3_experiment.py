#!/usr/bin/env python3
"""Run SE(3) comparison experiment: our semi-dual method vs RCPM."""
# conda activate rcpms-jax && nohup python -u run_se3_experiment.py --gpu 0 > se3_run.log 2>&1 &

# =============================================================================
# GPU SELECTION - Must be FIRST before any imports!
# =============================================================================
import sys
import os

for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        gpu_id = sys.argv[i + 1]
        os.environ['HIP_VISIBLE_DEVICES'] = gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Setting GPU: {gpu_id}")
        break

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_gpu_enable_command_buffer='
    + ' --xla_gpu_graph_level=0'
)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import time

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rcpm'))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random

import numpy as np
import optax

import flows as rcpm_flows

from src.base import ExperimentConfig, ModelConfig, SolverConfig, TrainingConfig
from src.manifolds import get as get_manifold, SE3
from src.densities import SE3Uniform, WrappedNormal, SE3FactorizedCompact
from src.embeddings import GromovDistanceEmbedding, build_landmarks
from src.networks import build_network
from src.solvers import ArgminSolver
from src.losses import SemiDualLoss
from src.trainers import SemiDualTrainer
from src.metrics import compute_kl_divergence, compute_ess

# =============================================================================
# CONFIGURATION
# =============================================================================
GAMMAS = [1.0, 0.1, 0.05, 0.01, 0.005, 0.001]

N_ITERS = 5000  # for RCPM
BATCH_SIZE = 256
EVAL_SIZE = 1024
N_EVAL_BATCHES = 5
TRAINING_SEED = 12345
EVAL_SEED = 12345

LANDMARK_METHOD = "fps"
FPS_CANDIDATES = 4096

# SE(3) target: SE3FactorizedCompact centered at a nontrivial rigid transform.
# Rotation: ~60° about z-axis; Translation: [1.0, 0.5, -0.5]
# quaternion for 60° about z: [cos(30°), 0, 0, sin(30°)] = [√3/2, 0, 0, 0.5]
TARGET_LOC = jnp.array([jnp.sqrt(3.0)/2, 0.0, 0.0, 0.5,   1.0, 0.5, -0.5])
TARGET_ROT_SCALE = jnp.array([0.3, 0.3, 0.3])
TARGET_TRANS_SCALE = jnp.array([0.5, 0.5, 0.5])
TRANS_LOW = jnp.array([-4.0, -4.0, -4.0])
TRANS_HIGH = jnp.array([4.0, 4.0, 4.0])


def build_experiment():
    manifold = get_manifold("SE3")

    base = SE3Uniform(manifold=manifold, t_range=4.0)

    target = SE3FactorizedCompact(
        manifold=manifold,
        loc=TARGET_LOC,
        rot_scale=TARGET_ROT_SCALE,
        trans_scale=TARGET_TRANS_SCALE,
        trans_low=TRANS_LOW,
        trans_high=TRANS_HIGH,
    )

    # max geodesic distance on SE(3): sqrt(α²·π² + (2·t_range·√3)²)
    # With α=1, t_range=4: sqrt(π² + 192) ≈ 14.3
    max_dist = float(jnp.sqrt(jnp.pi**2 + (2 * 4.0 * jnp.sqrt(3.0))**2))

    model_cfg = ModelConfig(
        network_type="mlp",
        hidden_dims=(128, 128),
        n_landmarks=128,
        use_layernorm=True,
        last_scale=0.01,
        activation="silu",
        max_dist=max_dist,
    )

    solver_cfg = SolverConfig(
        inner_steps=100,
        inner_lr=5e-3,
        tolerance=1e-6,
        min_steps=500,
        logsumexp_init=True,
        logsumexp_gamma=0.01,
        use_adam=False,
        use_line_search=True,
    )

    training_cfg = TrainingConfig(
        n_steps=200,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        lr_decay=True,
        lr_decay_alpha=0.05,
        log_every=1,
        eval_every=None,
        eval_size=EVAL_SIZE,
        seed=TRAINING_SEED,
    )

    key = jax.random.PRNGKey(training_cfg.seed)

    # Build landmarks from both distributions
    n_landmarks = model_cfg.n_landmarks
    n_base = n_landmarks // 2
    n_target = n_landmarks - n_base

    if LANDMARK_METHOD == "fps":
        lm_cfg_base = {"n_landmarks": n_base, "method": "fps",
                       "fps_candidates": FPS_CANDIDATES}
        lm_cfg_target = {"n_landmarks": n_target, "method": "fps",
                         "fps_candidates": FPS_CANDIDATES}
        landmarks_base, key = build_landmarks(manifold, base, lm_cfg_base, key)
        landmarks_target, key = build_landmarks(manifold, target, lm_cfg_target, key)
    else:
        key, k1, k2 = jax.random.split(key, 3)
        landmarks_base = base.sample(k1, n_base)
        landmarks_target = target.sample(k2, n_target)

    landmarks = jnp.concatenate([landmarks_base, landmarks_target], axis=0)

    emb = GromovDistanceEmbedding(manifold=manifold, landmarks=landmarks)
    psi = build_network(
        phi=emb,
        network_type=model_cfg.network_type,
        hidden_dims=model_cfg.hidden_dims,
        use_layernorm=model_cfg.use_layernorm,
        last_scale=model_cfg.last_scale,
        activation=model_cfg.activation,
        leaky_slope=model_cfg.leaky_slope,
        softplus_beta=model_cfg.softplus_beta,
        max_dist=model_cfg.max_dist,
    )

    key, kx_init, kparams = jax.random.split(key, 3)
    x_init = base.sample(kx_init, 16)
    psi_vars = psi.init(kparams, x_init)
    psi_params = psi_vars["params"]

    solver = ArgminSolver(
        manifold=manifold, psi_module=psi,
        inner_steps=solver_cfg.inner_steps,
        inner_lr=solver_cfg.inner_lr,
        grad_clip=solver_cfg.grad_clip,
        lr_decay=solver_cfg.lr_decay,
        tolerance=solver_cfg.tolerance,
        min_steps=solver_cfg.min_steps,
        momentum=solver_cfg.momentum,
        logsumexp_init=solver_cfg.logsumexp_init,
        logsumexp_gamma=solver_cfg.logsumexp_gamma,
        use_adam=solver_cfg.use_adam,
        adam_beta1=solver_cfg.adam_beta1,
        adam_beta2=solver_cfg.adam_beta2,
        use_line_search=solver_cfg.use_line_search,
        line_search_steps=solver_cfg.line_search_steps,
    )

    loss_obj = SemiDualLoss(manifold=manifold, psi_module=psi, solver=solver)
    trainer = SemiDualTrainer(
        manifold=manifold, psi_module=psi, loss_fn=loss_obj, solver=solver,
        learning_rate=training_cfg.learning_rate,
        lr_decay=training_cfg.lr_decay,
        lr_decay_alpha=training_cfg.lr_decay_alpha,
        n_steps=training_cfg.n_steps,
    )

    state = trainer.init_state(psi_params)
    return {
        "manifold": manifold, "base": base, "target": target,
        "psi": psi, "solver": solver, "trainer": trainer,
        "state": state, "key": key,
        "training_cfg": training_cfg, "model_cfg": model_cfg,
        "solver_cfg": solver_cfg,
    }


def train(experiment):
    trainer = experiment["trainer"]
    cfg = experiment["training_cfg"]
    state, history = trainer.train(
        state=experiment["state"],
        base_density=experiment["base"],
        target_density=experiment["target"],
        key=experiment["key"],
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        log_every=cfg.log_every,
        eval_every=cfg.eval_every,
        eval_size=cfg.eval_size,
        callbacks=[],
    )
    experiment["state"] = state
    return experiment


def evaluate(experiment, key, batch_size=1024):
    k1, k2 = jax.random.split(key)
    xs = experiment["base"].sample(k1, batch_size)
    ys_target = experiment["target"].sample(k2, batch_size)

    ys, residuals = experiment["solver"].batch_solve(
        experiment["state"].params, xs, ys_target)

    kl_ift = compute_kl_divergence(
        psi_params=experiment["state"].params,
        xs=xs, ys=ys,
        base_density=experiment["base"],
        target_density=experiment["target"],
        manifold=experiment["manifold"],
        psi_module=experiment["psi"],
    )

    return {
        "kl": kl_ift.kl,
        "ess": kl_ift.ess,
        "ess_ratio": kl_ift.ess_ratio,
        "mean_residual": float(jnp.mean(residuals)),
    }


# =============================================================================
# RCPM METHOD
# =============================================================================
def build_rcpm_experiment(gamma=0.1, n_components=68, n_transforms=5, seed=TRAINING_SEED):
    """Build RCPM experiment on SE(3)."""
    manifold = SE3(D=7)

    base = SE3Uniform(manifold=manifold, t_range=4.0)
    target = SE3FactorizedCompact(
        manifold=manifold,
        loc=TARGET_LOC,
        rot_scale=TARGET_ROT_SCALE,
        trans_scale=TARGET_TRANS_SCALE,
        trans_low=TRANS_LOW,
        trans_high=TRANS_HIGH,
    )

    potential_cfg = {
        '_target_': 'flows.InfAffine',
        'n_components': n_components,
        'init_alpha_mode': 'uniform',
        'init_alpha_linear_scale': 1.0,
        'init_alpha_minval': 0.4,
        'init_alpha_range': 0.01,
        'cost_gamma': gamma,
        'min_zero_gamma': None,
    }

    single_transform_cfg = {
        '_target_': 'flows.ExpMapFlow',
        'potential_cfg': potential_cfg,
    }

    flow = rcpm_flows.SequentialFlow(
        n_transforms=n_transforms,
        manifold=manifold,
        single_transform_cfg=single_transform_cfg,
    )

    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    batch = base.sample(k1, 16)
    params = flow.init(k2, batch)

    return {
        'manifold': manifold, 'base': base, 'target': target,
        'flow': flow, 'params': params, 'key': key,
    }


def train_rcpm(experiment, n_iters=500, batch_size=128, lr=1e-3, log_every=500):
    flow = experiment['flow']
    base = experiment['base']
    target = experiment['target']
    params = experiment['params']
    key = experiment['key']

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, base_samples, base_log_probs):
        z, ldjs = flow.apply(params, base_samples)
        return (base_log_probs - ldjs - target.log_prob(z)).mean()

    @jax.jit
    def update(params, opt_state, base_samples, base_log_probs):
        loss, grads = jax.value_and_grad(loss_fn)(params, base_samples, base_log_probs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    for i in range(n_iters):
        key, subkey = jax.random.split(key)
        base_samples = base.sample(subkey, batch_size)
        base_log_probs = base.log_prob(base_samples)
        loss, params, opt_state = update(params, opt_state, base_samples, base_log_probs)
        if (i + 1) % log_every == 0:
            print(f"  [{i+1:5d}] loss = {loss:.4f}")

    experiment['params'] = params
    experiment['key'] = key
    return experiment


def compute_kl_rcpm(experiment, key, batch_size=1024):
    xs = experiment['base'].sample(key, batch_size)
    ys_transport, ldjs = experiment['flow'].apply(experiment['params'], xs)

    log_mu = experiment['base'].log_prob(xs)
    log_nu = experiment['target'].log_prob(ys_transport)

    kl_per_sample = log_mu - ldjs - log_nu
    kl = float(jnp.mean(kl_per_sample))

    log_weights = -kl_per_sample
    ess, ess_ratio = compute_ess(log_weights)

    return kl, ess, ess_ratio


# =============================================================================
# MAIN
# =============================================================================
def main():
    global LANDMARK_METHOD

    parser = argparse.ArgumentParser(description='Run SE(3) comparison experiment')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default=EXPERIMENTS_DIR,
                        help='Output directory for results')
    parser.add_argument('--ours-only', action='store_true', help='Only run our method')
    parser.add_argument('--rcpm-only', action='store_true', help='Only run RCPM')
    parser.add_argument('--landmark-method', type=str, default=LANDMARK_METHOD,
                        choices=['random', 'fps'], help='Landmark sampling method')
    args = parser.parse_args()

    LANDMARK_METHOD = args.landmark_method
    run_ours = not args.rcpm_only
    run_rcpm = not args.ours_only
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running on: {jax.devices()}")
    print(f"SE(3) comparison experiment")
    print(f"  Source: SE3Uniform (Haar x Uniform[-2,2]^3)")
    print(f"  Target: SE3WrappedNormal at 60 deg z-rotation + [1, 0.5, -0.5] translation")
    print(f"  Scale:  rot={TARGET_ROT_SCALE.tolist()}, trans={TARGET_TRANS_SCALE.tolist()}")
    print(f"  LANDMARK_METHOD = {LANDMARK_METHOD}")

    # =========================================================================
    # OUR METHOD
    # =========================================================================
    if run_ours:
        print(f"\n{'='*70}")
        print(f"RUNNING OUR METHOD ON SE(3)")
        print(f"{'='*70}")

        exp = build_experiment()

        start_time = time.time()
        exp = train(exp)
        ours_runtime = time.time() - start_time

        key = jax.random.PRNGKey(EVAL_SEED)
        kl_vals, ess_vals, ess_ratio_vals, residual_vals = [], [], [], []
        for _ in range(N_EVAL_BATCHES):
            key, subkey = jax.random.split(key)
            result = evaluate(exp, subkey, batch_size=EVAL_SIZE)
            kl_vals.append(result["kl"])
            ess_vals.append(result["ess"])
            ess_ratio_vals.append(result["ess_ratio"])
            residual_vals.append(result["mean_residual"])

        ours_kl_mean = np.mean(kl_vals)
        ours_kl_se = np.std(kl_vals) / np.sqrt(N_EVAL_BATCHES)
        ours_ess_mean = np.mean(ess_vals)
        ours_ess_ratio_mean = np.mean(ess_ratio_vals)
        ours_residual_mean = np.mean(residual_vals)

        print(f"  KL = {ours_kl_mean:.4f} +/- {ours_kl_se:.4f}  |  ESS ratio = {ours_ess_ratio_mean:.3f}")
        print(f"  Mean solver residual = {ours_residual_mean:.2e}")
        print(f"  Runtime = {ours_runtime:.1f}s")

        out_path = os.path.join(output_dir, 'ours_se3.npz')
        np.savez(out_path,
                 kl_mean=ours_kl_mean, kl_se=ours_kl_se,
                 ess_mean=ours_ess_mean, ess_ratio=ours_ess_ratio_mean,
                 residual_mean=ours_residual_mean, runtime=ours_runtime)
        print(f"Saved: {out_path}")

    # =========================================================================
    # RCPM METHOD
    # =========================================================================
    if run_rcpm:
        print(f"\n{'='*70}")
        print(f"RUNNING RCPM ON SE(3)")
        print(f"{'='*70}")

        all_rcpm_results = {}

        for gamma in GAMMAS:
            print(f"\n{'#'*60}")
            print(f"# GAMMA = {gamma}")
            print(f"{'#'*60}")

            print(f"\n[RCPM gamma={gamma}] Training on SE(3)...")
            exp = build_rcpm_experiment(gamma=gamma, seed=TRAINING_SEED)

            start_time = time.time()
            exp = train_rcpm(exp, n_iters=N_ITERS, batch_size=BATCH_SIZE,
                             log_every=N_ITERS//10)
            runtime = time.time() - start_time

            key = jax.random.PRNGKey(EVAL_SEED)
            kl_vals, ess_vals, ess_ratio_vals = [], [], []
            for _ in range(N_EVAL_BATCHES):
                key, subkey = jax.random.split(key)
                kl, ess, ess_ratio = compute_kl_rcpm(exp, subkey, EVAL_SIZE)
                kl_vals.append(kl)
                ess_vals.append(ess)
                ess_ratio_vals.append(ess_ratio)

            kl_mean = np.mean(kl_vals)
            kl_se = np.std(kl_vals) / np.sqrt(N_EVAL_BATCHES)
            ess_mean = np.mean(ess_vals)
            ess_se = np.std(ess_vals) / np.sqrt(N_EVAL_BATCHES)
            ess_ratio_mean = np.mean(ess_ratio_vals)

            print(f"  KL divergence = {kl_mean:.4f} +/- {kl_se:.4f}")
            print(f"  ESS = {ess_mean:.1f} +/- {ess_se:.1f} (ratio: {ess_ratio_mean:.3f})")
            print(f"  Runtime = {runtime:.1f}s")

            all_rcpm_results[gamma] = {
                'kl_mean': kl_mean, 'kl_se': kl_se,
                'ess_mean': ess_mean, 'ess_se': ess_se,
                'ess_ratio': ess_ratio_mean, 'runtime': runtime,
            }

        rcpm_output = os.path.join(output_dir, 'rcpm_se3.npz')
        np.savez(rcpm_output,
                 gammas=GAMMAS,
                 **{f'gamma_{g}_kl': all_rcpm_results[g]['kl_mean'] for g in GAMMAS},
                 **{f'gamma_{g}_kl_se': all_rcpm_results[g]['kl_se'] for g in GAMMAS},
                 **{f'gamma_{g}_ess': all_rcpm_results[g]['ess_mean'] for g in GAMMAS},
                 **{f'gamma_{g}_ess_se': all_rcpm_results[g]['ess_se'] for g in GAMMAS},
                 **{f'gamma_{g}_ess_ratio': all_rcpm_results[g]['ess_ratio'] for g in GAMMAS},
                 **{f'gamma_{g}_runtime': all_rcpm_results[g]['runtime'] for g in GAMMAS})
        print(f"\nSaved: {rcpm_output}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"SE(3) EXPERIMENT COMPLETE")
    print(f"{'='*70}")

    if run_ours:
        print(f"\n[Ours]  KL = {ours_kl_mean:.4f} +/- {ours_kl_se:.4f}  |  ESS ratio = {ours_ess_ratio_mean:.3f}  |  Runtime = {ours_runtime:.1f}s")

    if run_rcpm:
        print(f"\n[RCPM] Results by gamma:")
        for gamma in GAMMAS:
            r = all_rcpm_results[gamma]
            print(f"  gamma={gamma:<6}  KL = {r['kl_mean']:.4f} +/- {r['kl_se']:.4f}  |  ESS ratio = {r['ess_ratio']:.3f}  |  Runtime = {r['runtime']:.1f}s")


if __name__ == '__main__':
    main()
