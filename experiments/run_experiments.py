#!/usr/bin/env python3
"""Run unified experiments for sphere or torus manifolds."""
# Sphere on GPU 3
# conda activate rcpms-jax && nohup python -u run_experiments.py sphere --gpu 3 > sphere_run.log 2>&1 &

# Torus on GPU 4
# conda activate rcpms-jax && nohup python -u run_experiments.py torus --gpu 4 > torus_run.log 2>&1 &

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
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
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
SPHERE_DIMS = list(range(2, 11))
TORUS_DIMS = list(range(2, 11))# Product torus (S1 x S1 x ...)


def print_config():
    """Print all configuration parameters from base.py"""
    from src.base import ModelConfig, SolverConfig, TrainingConfig, ExperimentConfig

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


GAMMAS = [1.0, 0.1, 0.05, 0.01, 0.005, 0.001]

N_ITERS = 5000  # for RCPM
BATCH_SIZE = 256
EVAL_SIZE = 1024
N_EVAL_BATCHES = 5
TRAINING_SEED = 12345
EVAL_SEED = 12345

# Landmark sampling method: "random" or "fps"
LANDMARK_METHOD = "fps"
FPS_CANDIDATES = 4096  # Candidate pool size for FPS


def get_config(manifold_type, output_dir):
    if manifold_type == "sphere":
        return {
            'dimensions': SPHERE_DIMS,
            'ours_manifold_name': lambda d: f"S{d}",
            'rcpm_manifold': lambda d: rcpm_manifolds.Sphere(D=d+1, jitter=1e-2),
            'is_torus': False,
            'base_density': "SphereUniform",
            'target_density': "SphereWrappedNormal",
            'ours_output': os.path.join(output_dir, 'ours_sweep_sphere.npz'),
            'rcpm_output': os.path.join(output_dir, 'rcpm_sweep_sphere.npz'),
        }
    elif manifold_type == "torus":
        return {
            'dimensions': TORUS_DIMS,
            'ours_manifold_name': lambda d: f"T{d}",
            'rcpm_manifold': lambda d: rcpm_manifolds.Product(D=2*d, manifolds_str=",".join(["S1"] * d)),
            'is_torus': True,
            'base_density': "TorusUniform",
            'target_density': "TorusWrappedNormal",
            'ours_output': os.path.join(output_dir, 'ours_sweep_torus.npz'),
            'rcpm_output': os.path.join(output_dir, 'rcpm_sweep_torus.npz'),
        }
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


# =============================================================================
# OUR METHOD
# =============================================================================
def build_ours_experiment(manifold_name, base_density, target_density):
    exp_cfg = ExperimentConfig(
        manifold_name=manifold_name,
        base_density=base_density,
        target_density=target_density,
        jax_platform="gpu",
    )

    manifold = get_manifold(exp_cfg.manifold_name)
    base = densities.get(manifold, exp_cfg.base_density)
    target = densities.get(manifold, exp_cfg.target_density)

    key = jax.random.PRNGKey(exp_cfg.training.seed)

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
def batched_sphere_dist(x, y, jitter=1e-2):
    inner = jnp.sum(x * y, axis=-1)
    inner = inner / (1 + jitter)
    inner = jnp.clip(inner, -1.0, 1.0)
    return jnp.arccos(inner)


def rcpm_product_dist(manifold, x, y):
    total_dist_sq = jnp.zeros(x.shape[0])
    d = 0
    for m in manifold.manifolds:
        D = m.D
        jitter = getattr(m, 'jitter', 1e-2)
        dist_i = batched_sphere_dist(x[:, d:d+D], y[:, d:d+D], jitter=jitter)
        total_dist_sq += dist_i ** 2
        d += D
    return jnp.sqrt(total_dist_sq)


def build_rcpm_experiment(rcpm_manifold, is_torus=False, n_components=68, n_transforms=5, gamma=0.1, seed=TRAINING_SEED):
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
        'flow': flow, 'params': params, 'key': key, 'is_torus': is_torus,
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

    # Compute ESS from importance weights: log w = -kl_per_sample
    log_weights = -kl_per_sample
    ess, ess_ratio = compute_ess(log_weights)

    return kl, ess, ess_ratio


# =============================================================================
# MAIN
# =============================================================================
def main():
    global LANDMARK_METHOD

    parser = argparse.ArgumentParser(description='Run unified experiments')
    parser.add_argument('manifold', choices=['sphere', 'torus'], help='Manifold type')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID (e.g., 0, 1, 2)')
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

    # Create output directory if needed
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = get_config(args.manifold, output_dir)
    print(f"Running on: {jax.devices()}")
    print(f"Running experiments on: {args.manifold}")
    print(f"Dimensions: {cfg['dimensions']}")
    print(f"LANDMARK_METHOD = {LANDMARK_METHOD}" + (f" (candidates={FPS_CANDIDATES})" if LANDMARK_METHOD == "fps" else ""))

    # Print all config parameters at startup
    print_config()

    # Run our method
    if run_ours:
        print(f"\n{'='*70}")
        print(f"RUNNING OUR METHOD ON {args.manifold.upper()}")
        print(f"{'='*70}")

        ours_kl_values = []
        ours_kl_errors = []
        ours_ess_values = []
        ours_ess_errors = []
        ours_ess_ratio_values = []
        ours_runtimes = []

        for dim in cfg['dimensions']:
            manifold_name = cfg['ours_manifold_name'](dim)
            print(f"\n[Ours] Training on {manifold_name}...")

            exp = build_ours_experiment(manifold_name, cfg['base_density'], cfg['target_density'])

            start_time = time.time()
            exp = train_ours(exp)
            runtime = time.time() - start_time
            ours_runtimes.append(runtime)

            key = jax.random.PRNGKey(EVAL_SEED)
            kl_vals = []
            ess_vals = []
            ess_ratio_vals = []
            for _ in range(N_EVAL_BATCHES):
                key, subkey = jax.random.split(key)
                result = compute_kl_ours(exp, subkey, batch_size=EVAL_SIZE)
                kl_vals.append(result["kl"])
                ess_vals.append(result["ess"])
                ess_ratio_vals.append(result["ess_ratio"])

            kl_mean = np.mean(kl_vals)
            kl_se = np.std(kl_vals) / np.sqrt(N_EVAL_BATCHES)
            ess_mean = np.mean(ess_vals)
            ess_se = np.std(ess_vals) / np.sqrt(N_EVAL_BATCHES)
            ess_ratio_mean = np.mean(ess_ratio_vals)

            ours_kl_values.append(kl_mean)
            ours_kl_errors.append(kl_se)
            ours_ess_values.append(ess_mean)
            ours_ess_errors.append(ess_se)
            ours_ess_ratio_values.append(ess_ratio_mean)

            print(f"  KL = {kl_mean:.4f} +/- {kl_se:.4f}  |  ESS = {ess_mean:.1f} (ratio: {ess_ratio_mean:.3f})")
            print(f"  Runtime = {runtime:.1f}s")

        np.savez(cfg['ours_output'],
                 dimensions=cfg['dimensions'],
                 kl_values=ours_kl_values,
                 kl_errors=ours_kl_errors,
                 ess_values=ours_ess_values,
                 ess_errors=ours_ess_errors,
                 ess_ratio_values=ours_ess_ratio_values,
                 runtimes=ours_runtimes)
        print(f"\nSaved: {cfg['ours_output']}")

    # Run RCPM
    if run_rcpm:
        print(f"\n{'='*70}")
        print(f"RUNNING RCPM ON {args.manifold.upper()}")
        print(f"{'='*70}")

        all_rcpm_results = {}

        for gamma in GAMMAS:
            print(f"\n{'#'*60}")
            print(f"# GAMMA = {gamma}")
            print(f"{'#'*60}")

            rcpm_kl_values = []
            rcpm_kl_errors = []
            rcpm_ess_values = []
            rcpm_ess_errors = []
            rcpm_ess_ratio_values = []
            rcpm_runtimes = []

            for dim in cfg['dimensions']:
                rcpm_manifold = cfg['rcpm_manifold'](dim)
                print(f"\n[RCPM gamma={gamma}] Training on dim={dim}...")

                exp = build_rcpm_experiment(rcpm_manifold, is_torus=cfg['is_torus'], gamma=gamma, seed=TRAINING_SEED)

                start_time = time.time()
                exp = train_rcpm(exp, n_iters=N_ITERS, batch_size=BATCH_SIZE, log_every=N_ITERS//10)
                runtime = time.time() - start_time
                rcpm_runtimes.append(runtime)

                key = jax.random.PRNGKey(EVAL_SEED)
                kl_vals = []
                ess_vals = []
                ess_ratio_vals = []
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

                rcpm_kl_values.append(kl_mean)
                rcpm_kl_errors.append(kl_se)
                rcpm_ess_values.append(ess_mean)
                rcpm_ess_errors.append(ess_se)
                rcpm_ess_ratio_values.append(ess_ratio_mean)

                print(f"  KL divergence = {kl_mean:.4f} +/- {kl_se:.4f}")
                print(f"  ESS = {ess_mean:.1f} +/- {ess_se:.1f} (ratio: {ess_ratio_mean:.3f})")
                print(f"  Runtime = {runtime:.1f}s")

            all_rcpm_results[gamma] = {
                'kl_values': rcpm_kl_values,
                'kl_errors': rcpm_kl_errors,
                'ess_values': rcpm_ess_values,
                'ess_errors': rcpm_ess_errors,
                'ess_ratio_values': rcpm_ess_ratio_values,
                'runtimes': rcpm_runtimes,
            }

        np.savez(cfg['rcpm_output'],
                 dimensions=cfg['dimensions'],
                 gammas=GAMMAS,
                 **{f'gamma_{g}_kl': all_rcpm_results[g]['kl_values'] for g in GAMMAS},
                 **{f'gamma_{g}_kl_errors': all_rcpm_results[g]['kl_errors'] for g in GAMMAS},
                 **{f'gamma_{g}_ess': all_rcpm_results[g]['ess_values'] for g in GAMMAS},
                 **{f'gamma_{g}_ess_errors': all_rcpm_results[g]['ess_errors'] for g in GAMMAS},
                 **{f'gamma_{g}_ess_ratio': all_rcpm_results[g]['ess_ratio_values'] for g in GAMMAS},
                 **{f'gamma_{g}_runtimes': all_rcpm_results[g]['runtimes'] for g in GAMMAS})
        print(f"\nSaved: {cfg['rcpm_output']}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {args.manifold.upper()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
