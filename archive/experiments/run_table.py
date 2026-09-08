#!/usr/bin/env python3
"""Run table experiments: 5 independent runs on S2 and T2 for paper table."""
# conda activate rcpms-jax && nohup python -u run_table.py --gpu 3 > table_run.log 2>&1 &

# =============================================================================
# GPU SELECTION - Must be FIRST before any imports!
# =============================================================================
import sys
import os

# Parse --gpu from sys.argv directly (before importing anything else)
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        gpu_id = sys.argv[i + 1]
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

# Add project root and RCPM to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rcpm'))

import jax
import jax.numpy as jnp
from jax import random

import numpy as np
import optax

import flows as rcpm_flows
import manifolds as rcpm_manifolds

from src.base import ExperimentConfig
from src.manifolds import get as get_manifold
import src.densities as densities
from src.embeddings import GromovDistanceEmbedding, build_landmarks
from src.networks import build_network
from src.solvers import ArgminSolver
from src.losses import SemiDualLoss
from src.trainers import SemiDualTrainer
from src.metrics import compute_kl_divergence, compute_ess
from src.densities import SphereUniform, WrappedNormal, ProductUniformComponents

# =============================================================================
# CONFIGURATION
# =============================================================================
N_RUNS = 5  # Number of independent training runs
RCPM_GAMMA = 1.0  # Only gamma=0.1 for RCPM

N_ITERS = 5000  # for RCPM
BATCH_SIZE = 256
EVAL_SIZE = 1024
N_EVAL_BATCHES = 5  # Batches per run for KL evaluation

# Landmark sampling method: "random" or "fps"
LANDMARK_METHOD = "random"
FPS_CANDIDATES = 4096  # Candidate pool size for FPS

# Seeds for independent runs
RUN_SEEDS = [12345, 23456, 34567, 45678, 56789]

# Manifold configs - use lambdas to create fresh manifolds for each run
MANIFOLDS = {
    'S2': {
        'ours_manifold_name': 'S2',
        'rcpm_manifold': lambda: rcpm_manifolds.Sphere(D=3, jitter=1e-2),
        'is_torus': False,
        'base_density': 'SphereUniform',
        'target_density': 'SphereWrappedNormal',
    },
    'T2': {
        'ours_manifold_name': 'T2',
        'rcpm_manifold': lambda: rcpm_manifolds.Product(D=4, manifolds_str="S1,S1"),
        'is_torus': True,
        'base_density': 'TorusUniform',
        'target_density': 'TorusWrappedNormal',
    },
}


# =============================================================================
# PRINT CONFIG
# =============================================================================
def print_config():
    """Print all configuration parameters from base.py"""
    from src.base import ModelConfig, SolverConfig, TrainingConfig

    print("\n" + "="*70)
    print("DEFAULT CONFIGURATION PARAMETERS (from src/base.py)")
    print("="*70)

    model = ModelConfig()
    print(f"\n[ModelConfig]")
    print(f"  network_type:   {model.network_type}")
    print(f"  hidden_dims:    {model.hidden_dims}")
    print(f"  n_landmarks:    {model.n_landmarks}")
    print(f"  use_layernorm:  {model.use_layernorm}")
    print(f"  last_scale:     {model.last_scale}")
    print(f"  activation:     {model.activation}")
    print(f"  leaky_slope:    {model.leaky_slope}")
    print(f"  softplus_beta:  {model.softplus_beta}")
    print(f"  max_dist:       {model.max_dist}")

    solver = SolverConfig()
    print(f"\n[SolverConfig]")
    print(f"  inner_steps:      {solver.inner_steps}")
    print(f"  inner_lr:         {solver.inner_lr}")
    print(f"  grad_clip:        {solver.grad_clip}")
    print(f"  lr_decay:         {solver.lr_decay}")
    print(f"  tolerance:        {solver.tolerance}")
    print(f"  min_steps:        {solver.min_steps}")
    print(f"  momentum:         {solver.momentum}")
    print(f"  logsumexp_init:   {solver.logsumexp_init}")
    print(f"  logsumexp_gamma:  {solver.logsumexp_gamma}")
    print(f"  use_adam:         {solver.use_adam}")
    print(f"  adam_beta1:       {solver.adam_beta1}")
    print(f"  adam_beta2:       {solver.adam_beta2}")

    training = TrainingConfig()
    print(f"\n[TrainingConfig]")
    print(f"  n_steps:        {training.n_steps}")
    print(f"  batch_size:     {training.batch_size}")
    print(f"  learning_rate:  {training.learning_rate}")
    print(f"  lr_decay:       {training.lr_decay}")
    print(f"  lr_decay_alpha: {training.lr_decay_alpha}")
    print(f"  log_every:      {training.log_every}")
    print(f"  eval_every:     {training.eval_every}")
    print(f"  eval_size:      {training.eval_size}")
    print(f"  seed:           {training.seed}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# OUR METHOD
# =============================================================================
def build_ours_experiment(manifold_name, base_density, target_density, seed):
    exp_cfg = ExperimentConfig(
        manifold_name=manifold_name,
        base_density=base_density,
        target_density=target_density,
        jax_platform="gpu",
    )
    # Override the seed
    exp_cfg.training.seed = seed

    manifold = get_manifold(exp_cfg.manifold_name)
    base = densities.get(manifold, exp_cfg.base_density)
    target = densities.get(manifold, exp_cfg.target_density)

    key = jax.random.PRNGKey(seed)

    # Build landmarks using configured method
    n_landmarks = exp_cfg.model.n_landmarks
    n_base = n_landmarks // 2
    n_target = n_landmarks - n_base

    if LANDMARK_METHOD == "fps":
        landmark_cfg_base = {"n_landmarks": n_base, "method": "fps", "fps_candidates": FPS_CANDIDATES}
        landmark_cfg_target = {"n_landmarks": n_target, "method": "fps", "fps_candidates": FPS_CANDIDATES}
        landmarks_base, key = build_landmarks(manifold, base, landmark_cfg_base, key)
        landmarks_target, key = build_landmarks(manifold, target, landmark_cfg_target, key)
    else:
        key, k1, k2 = jax.random.split(key, 3)
        landmarks_base = base.sample(k1, n_base)
        landmarks_target = target.sample(k2, n_target)

    landmarks = jnp.concatenate([landmarks_base, landmarks_target], axis=0)

    emb = GromovDistanceEmbedding(manifold=manifold, landmarks=landmarks)
    psi = build_network(
        phi=emb,
        network_type=exp_cfg.model.network_type,
        hidden_dims=exp_cfg.model.hidden_dims,
        use_layernorm=exp_cfg.model.use_layernorm,
        last_scale=exp_cfg.model.last_scale,
        activation=exp_cfg.model.activation,
        leaky_slope=exp_cfg.model.leaky_slope,
        softplus_beta=exp_cfg.model.softplus_beta,
        max_dist=exp_cfg.model.max_dist,
    )

    key, kx_init, kparams = jax.random.split(key, 3)
    x_init = base.sample(kx_init, 16)
    psi_vars = psi.init(kparams, x_init)
    psi_params = psi_vars["params"]

    solver = ArgminSolver(
        manifold=manifold, psi_module=psi,
        inner_steps=exp_cfg.solver.inner_steps,
        inner_lr=exp_cfg.solver.inner_lr,
        grad_clip=exp_cfg.solver.grad_clip,
        lr_decay=exp_cfg.solver.lr_decay,
        tolerance=exp_cfg.solver.tolerance,
        min_steps=exp_cfg.solver.min_steps,
        momentum=exp_cfg.solver.momentum,
        logsumexp_init=exp_cfg.solver.logsumexp_init,
        logsumexp_gamma=exp_cfg.solver.logsumexp_gamma,
        use_adam=exp_cfg.solver.use_adam,
        adam_beta1=exp_cfg.solver.adam_beta1,
        adam_beta2=exp_cfg.solver.adam_beta2,
        use_line_search=exp_cfg.solver.use_line_search,
        line_search_steps=exp_cfg.solver.line_search_steps,
    )

    loss_obj = SemiDualLoss(manifold=manifold, psi_module=psi, solver=solver)
    trainer = SemiDualTrainer(
        manifold=manifold, psi_module=psi, loss_fn=loss_obj, solver=solver,
        learning_rate=exp_cfg.training.learning_rate,
        lr_decay=exp_cfg.training.lr_decay,
        lr_decay_alpha=exp_cfg.training.lr_decay_alpha,
        n_steps=exp_cfg.training.n_steps,
    )

    state = trainer.init_state(psi_params)
    return {
        "cfg": exp_cfg, "manifold": manifold, "base": base, "target": target,
        "psi": psi, "solver": solver, "trainer": trainer, "state": state, "key": key,
    }


def train_ours(experiment):
    trainer = experiment["trainer"]
    exp_cfg = experiment["cfg"]
    state, history = trainer.train(
        state=experiment["state"],
        base_density=experiment["base"],
        target_density=experiment["target"],
        key=experiment["key"],
        n_steps=exp_cfg.training.n_steps,
        batch_size=exp_cfg.training.batch_size,
        log_every=exp_cfg.training.log_every,
        eval_every=exp_cfg.training.eval_every,
        eval_size=exp_cfg.training.eval_size,
        callbacks=[],
    )
    experiment["state"] = state
    return experiment


def compute_kl_ours(experiment, key, batch_size=1024):
    k1, k2 = jax.random.split(key)
    xs = experiment["base"].sample(k1, batch_size)
    ys_target = experiment["target"].sample(k2, batch_size)

    eval_solver = experiment["solver"]

    # Solve for transported points
    ys, _ = eval_solver.batch_solve(experiment["state"].params, xs, ys_target)

    # KL via IFT (uses implicit function theorem)
    kl_ift = compute_kl_divergence(
        psi_params=experiment["state"].params,
        xs=xs,
        ys=ys,
        base_density=experiment["base"],
        target_density=experiment["target"],
        manifold=experiment["manifold"],
        psi_module=experiment["psi"],
    )

    return {
        "kl": kl_ift.kl,
        "ess": kl_ift.ess,
        "ess_ratio": kl_ift.ess_ratio,
    }


# =============================================================================
# RCPM METHOD
# =============================================================================
def build_rcpm_experiment(rcpm_manifold, is_torus=False, gamma=0.1, seed=12345):
    manifold = rcpm_manifold

    if is_torus:
        base = ProductUniformComponents(manifold=manifold)
        loc = manifold.zero()
        tangent_dim = sum(m.D - 1 for m in manifold.manifolds)
        scale = jnp.full(tangent_dim, 0.3)
        target = WrappedNormal(manifold=manifold, loc=loc, scale=scale)
    else:
        base = SphereUniform(manifold=manifold)
        loc = manifold.zero()
        scale = jnp.full(manifold.D - 1, 0.3)
        target = WrappedNormal(manifold=manifold, loc=loc, scale=scale)

    potential_cfg = {
        '_target_': 'flows.InfAffine',
        'n_components': 68,
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
        n_transforms=5,
        manifold=manifold,
        single_transform_cfg=single_transform_cfg,
    )

    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    batch = base.sample(k1, 16)
    params = flow.init(k2, batch)

    return {
        'manifold': manifold, 'base': base, 'target': target,
        'flow': flow, 'params': params, 'key': key, 'is_torus': is_torus,
    }


def train_rcpm(experiment, n_iters=1000, batch_size=256, lr=1e-3, log_every=500):
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
            print(f"    [{i+1:5d}] loss = {loss:.4f}")

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

    parser = argparse.ArgumentParser(description='Run table experiments (5 runs on S2 and T2)')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--ours-only', action='store_true', help='Only run our method')
    parser.add_argument('--rcpm-only', action='store_true', help='Only run RCPM')
    parser.add_argument('--landmark-method', type=str, default=LANDMARK_METHOD,
                        choices=['random', 'fps'], help='Landmark sampling method')
    args = parser.parse_args()

    LANDMARK_METHOD = args.landmark_method
    run_ours = not args.rcpm_only
    run_rcpm = not args.ours_only

    # Create output directory if needed
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running on: {jax.devices()}")
    print(f"N_RUNS = {N_RUNS}")
    print(f"RCPM_GAMMA = {RCPM_GAMMA}")
    print(f"LANDMARK_METHOD = {LANDMARK_METHOD}" + (f" (candidates={FPS_CANDIDATES})" if LANDMARK_METHOD == "fps" else ""))
    print(f"Manifolds: S2, T2")

    # Print all config parameters at startup
    print_config()

    results = {}

    for manifold_name, cfg in MANIFOLDS.items():
        print(f"\n{'='*70}")
        print(f"MANIFOLD: {manifold_name}")
        print(f"{'='*70}")

        results[manifold_name] = {}

        # =================================================================
        # OUR METHOD - 5 independent runs
        # =================================================================
        if run_ours:
            print(f"\n[Ours] Running {N_RUNS} independent training runs...")

            ours_kl_runs = []
            ours_ess_runs = []
            ours_runtime_runs = []

            for run_idx, seed in enumerate(RUN_SEEDS):
                print(f"\n  --- Run {run_idx+1}/{N_RUNS} (seed={seed}) ---")

                exp = build_ours_experiment(
                    cfg['ours_manifold_name'],
                    cfg['base_density'],
                    cfg['target_density'],
                    seed=seed
                )

                start_time = time.time()
                exp = train_ours(exp)
                runtime = time.time() - start_time
                ours_runtime_runs.append(runtime)

                # Evaluate with multiple batches and take mean
                key = jax.random.PRNGKey(seed + 1000)  # Different seed for eval
                kl_vals = []
                ess_vals = []

                for _ in range(N_EVAL_BATCHES):
                    key, subkey = jax.random.split(key)
                    result = compute_kl_ours(exp, subkey, batch_size=EVAL_SIZE)
                    kl_vals.append(result["kl"])
                    ess_vals.append(result["ess_ratio"])

                # Mean over eval batches for this run
                ours_kl_runs.append(np.mean(kl_vals))
                ours_ess_runs.append(np.mean(ess_vals))

                print(f"    KL = {ours_kl_runs[-1]:.4f}")
                print(f"    ESS ratio = {ours_ess_runs[-1]:.3f}")
                print(f"    Runtime = {runtime:.1f}s")

            # Compute mean and SE across runs
            results[manifold_name]['ours'] = {
                'kl_mean': np.mean(ours_kl_runs),
                'kl_se': np.std(ours_kl_runs) / np.sqrt(N_RUNS),
                'ess_mean': np.mean(ours_ess_runs),
                'ess_se': np.std(ours_ess_runs) / np.sqrt(N_RUNS),
                'runtime_mean': np.mean(ours_runtime_runs),
                'runtime_se': np.std(ours_runtime_runs) / np.sqrt(N_RUNS),
            }

            print(f"\n  [Ours Summary for {manifold_name}]")
            print(f"    KL = {results[manifold_name]['ours']['kl_mean']:.4f} ± {results[manifold_name]['ours']['kl_se']:.4f}")
            print(f"    ESS = {results[manifold_name]['ours']['ess_mean']:.3f} ± {results[manifold_name]['ours']['ess_se']:.3f}")
            print(f"    Runtime = {results[manifold_name]['ours']['runtime_mean']:.1f} ± {results[manifold_name]['ours']['runtime_se']:.1f}s")

        # =================================================================
        # RCPM - 5 independent runs (gamma=0.1 only)
        # =================================================================
        if run_rcpm:
            print(f"\n[RCPM gamma={RCPM_GAMMA}] Running {N_RUNS} independent training runs...")

            rcpm_kl_runs = []
            rcpm_ess_runs = []
            rcpm_runtime_runs = []

            for run_idx, seed in enumerate(RUN_SEEDS):
                print(f"\n  --- Run {run_idx+1}/{N_RUNS} (seed={seed}) ---")

                exp = build_rcpm_experiment(
                    cfg['rcpm_manifold'](),  # Call lambda to create fresh manifold
                    is_torus=cfg['is_torus'],
                    gamma=RCPM_GAMMA,
                    seed=seed
                )

                start_time = time.time()
                exp = train_rcpm(exp, n_iters=N_ITERS, batch_size=BATCH_SIZE, log_every=N_ITERS)
                runtime = time.time() - start_time
                rcpm_runtime_runs.append(runtime)

                # Evaluate with multiple batches
                key = jax.random.PRNGKey(seed + 1000)
                kl_vals = []
                ess_vals = []

                for _ in range(N_EVAL_BATCHES):
                    key, subkey = jax.random.split(key)
                    kl, ess, ess_ratio = compute_kl_rcpm(exp, subkey, EVAL_SIZE)
                    kl_vals.append(kl)
                    ess_vals.append(ess_ratio)

                rcpm_kl_runs.append(np.mean(kl_vals))
                rcpm_ess_runs.append(np.mean(ess_vals))

                print(f"    KL = {rcpm_kl_runs[-1]:.4f}")
                print(f"    ESS ratio = {rcpm_ess_runs[-1]:.3f}")
                print(f"    Runtime = {runtime:.1f}s")

            # Compute mean and SE across runs
            results[manifold_name]['rcpm'] = {
                'kl_mean': np.mean(rcpm_kl_runs),
                'kl_se': np.std(rcpm_kl_runs) / np.sqrt(N_RUNS),
                'ess_mean': np.mean(rcpm_ess_runs),
                'ess_se': np.std(rcpm_ess_runs) / np.sqrt(N_RUNS),
                'runtime_mean': np.mean(rcpm_runtime_runs),
                'runtime_se': np.std(rcpm_runtime_runs) / np.sqrt(N_RUNS),
            }

            print(f"\n  [RCPM Summary for {manifold_name}]")
            print(f"    KL = {results[manifold_name]['rcpm']['kl_mean']:.4f} ± {results[manifold_name]['rcpm']['kl_se']:.4f}")
            print(f"    ESS = {results[manifold_name]['rcpm']['ess_mean']:.3f} ± {results[manifold_name]['rcpm']['ess_se']:.3f}")
            print(f"    Runtime = {results[manifold_name]['rcpm']['runtime_mean']:.1f} ± {results[manifold_name]['rcpm']['runtime_se']:.1f}s")

    # =================================================================
    # FINAL TABLE OUTPUT
    # =================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Manifold':<10} {'Method':<10} {'KL':<20} {'ESS Ratio':<20} {'Runtime (s)':<15}")
    print("-" * 75)

    for manifold_name in ['S2', 'T2']:
        if manifold_name not in results:
            continue

        if 'ours' in results[manifold_name]:
            r = results[manifold_name]['ours']
            kl_str = f"{r['kl_mean']:.4f} ± {r['kl_se']:.4f}"
            ess_str = f"{r['ess_mean']:.3f} ± {r['ess_se']:.3f}"
            rt_str = f"{r['runtime_mean']:.1f} ± {r['runtime_se']:.1f}"
            print(f"{manifold_name:<10} {'Ours':<10} {kl_str:<20} {ess_str:<20} {rt_str:<15}")

        if 'rcpm' in results[manifold_name]:
            r = results[manifold_name]['rcpm']
            kl_str = f"{r['kl_mean']:.4f} ± {r['kl_se']:.4f}"
            ess_str = f"{r['ess_mean']:.3f} ± {r['ess_se']:.3f}"
            rt_str = f"{r['runtime_mean']:.1f} ± {r['runtime_se']:.1f}"
            print(f"{manifold_name:<10} {'RCPM':<10} {kl_str:<20} {ess_str:<20} {rt_str:<15}")

    # Save results
    npz_output = os.path.join(output_dir, 'table_results.npz')
    np.savez(npz_output, results=results)
    print(f"\nSaved: {npz_output}")

    # =================================================================
    # LATEX TABLE OUTPUT
    # =================================================================
    def fmt_val(mean, se, decimals=2):
        """Format value as mean ± se with specified decimals."""
        return f"{mean:.{decimals}f} $\\pm$ {se:.{decimals}f}"

    latex_lines = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r" & \multicolumn{3}{c|}{$\mathbb{S}^{2}$} & \multicolumn{3}{c}{$\mathbb{T}^{2}$} \\",
        r"\cline{2-7}",
        r" \textbf{Model} & KL $\downarrow$ & ESS $\uparrow$ & Time (s) $\downarrow$",
        r" & KL $\downarrow$ & ESS $\uparrow$ & Time (s) $\downarrow$\\",
        r"\hline",
    ]

    # Ours row
    s2_ours = results.get('S2', {}).get('ours', {})
    t2_ours = results.get('T2', {}).get('ours', {})
    if s2_ours and t2_ours:
        row = "Ours "
        row += f"& {fmt_val(s2_ours['kl_mean'], s2_ours['kl_se'], 2)} "
        row += f"& {fmt_val(s2_ours['ess_mean'], s2_ours['ess_se'], 2)} "
        row += f"& {fmt_val(s2_ours['runtime_mean'], s2_ours['runtime_se'], 0)} "
        row += f"& {fmt_val(t2_ours['kl_mean'], t2_ours['kl_se'], 2)} "
        row += f"& {fmt_val(t2_ours['ess_mean'], t2_ours['ess_se'], 2)} "
        row += f"& {fmt_val(t2_ours['runtime_mean'], t2_ours['runtime_se'], 0)} \\\\"
        latex_lines.append(row)

    # RCPM row
    s2_rcpm = results.get('S2', {}).get('rcpm', {})
    t2_rcpm = results.get('T2', {}).get('rcpm', {})
    if s2_rcpm and t2_rcpm:
        row = r"RCPM~\citep{lou2020neural} "
        row += f"& {fmt_val(s2_rcpm['kl_mean'], s2_rcpm['kl_se'], 2)} "
        row += f"& {fmt_val(s2_rcpm['ess_mean'], s2_rcpm['ess_se'], 2)} "
        row += f"& {fmt_val(s2_rcpm['runtime_mean'], s2_rcpm['runtime_se'], 0)} "
        row += f"& {fmt_val(t2_rcpm['kl_mean'], t2_rcpm['kl_se'], 2)} "
        row += f"& {fmt_val(t2_rcpm['ess_mean'], t2_rcpm['ess_se'], 2)} "
        row += f"& {fmt_val(t2_rcpm['runtime_mean'], t2_rcpm['runtime_se'], 0)} \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
    ])

    latex_table = "\n".join(latex_lines)

    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")
    print(latex_table)

    # Save LaTeX table to file
    tex_output = os.path.join(output_dir, 'table_results.tex')
    with open(tex_output, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved: {tex_output}")


if __name__ == '__main__':
    main()
