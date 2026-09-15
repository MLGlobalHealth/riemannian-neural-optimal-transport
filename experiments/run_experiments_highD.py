#!/usr/bin/env python3
"""Run experiments for our method only (no RCPM baseline)."""
# Sphere on GPU 3
# conda activate rcpms-jax && nohup python -u run_experiments_ours.py sphere --gpu 3 > sphere_ours.log 2>&1 &

# Torus on GPU 4
# conda activate rcpms-jax && nohup python -u run_experiments_ours.py torus --gpu 4 > torus_ours.log 2>&1 &

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
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import random

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.base import ExperimentConfig
from src.manifolds import get as get_manifold
import src.densities as densities
from src.embeddings import GromovDistanceEmbedding, build_landmarks
from src.networks import build_network
from src.solvers import ArgminSolver
from src.losses import SemiDualLoss
from src.trainers import SemiDualTrainer
from src.metrics import compute_kl_divergence

# =============================================================================
# CONFIGURATION
# =============================================================================
SPHERE_DIMS = [2,5,10,20,30,40]
TORUS_DIMS = [2,5,10,20,30,40]

EVAL_SIZE = 1024
N_EVAL_BATCHES = 5
EVAL_SEED = 12345

# Landmark sampling method: "random" or "fps"
LANDMARK_METHOD = "fps"
FPS_CANDIDATES = 4096  # Candidate pool size for FPS


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


def get_config(manifold_type, output_dir):
    if manifold_type == "sphere":
        return {
            'dimensions': SPHERE_DIMS,
            'manifold_name': lambda d: f"S{d}",
            'is_torus': False,
            'base_density': "SphereUniform",
            'target_density': "SphereWrappedNormal",
            'output': os.path.join(output_dir, 'ours_highD_sphere.npz'),
        }
    elif manifold_type == "torus":
        return {
            'dimensions': TORUS_DIMS,
            'manifold_name': lambda d: f"T{d}",
            'is_torus': True,
            'base_density': "TorusUniform",
            'target_density': "TorusWrappedNormal",
            'output': os.path.join(output_dir, 'ours_highD_torus.npz'),
        }
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


# =============================================================================
# OUR METHOD
# =============================================================================
def build_experiment(manifold_name, base_density, target_density, dim):
    """Build experiment with dimension-dependent scaling for high-D."""
    exp_cfg = ExperimentConfig(
        manifold_name=manifold_name,
        base_density=base_density,
        target_density=target_density,
        jax_platform="gpu",
    )

    # Scale model parameters for high dimensions
    base_n_landmarks = exp_cfg.model.n_landmarks
    base_hidden_dims = exp_cfg.model.hidden_dims

    # n_landmarks: use base for dim < 30, 512 for dim >= 30
    scaled_n_landmarks = base_n_landmarks if dim < 30 else 512

    # hidden_dims = (256, 256) if dim > 30 else base
    scaled_hidden_dims = (128, 128) if dim > 30 else base_hidden_dims

    if scaled_n_landmarks != base_n_landmarks or scaled_hidden_dims != base_hidden_dims:
        print(f"  [High-D scaling] n_landmarks: {base_n_landmarks} -> {scaled_n_landmarks}, "
              f"hidden_dims: {base_hidden_dims} -> {scaled_hidden_dims}")
        new_model = replace(exp_cfg.model,
                           n_landmarks=scaled_n_landmarks,
                           hidden_dims=scaled_hidden_dims)
        exp_cfg = replace(exp_cfg, model=new_model)

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


def train(experiment):
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


def compute_kl(experiment, key, batch_size=512):
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
# MAIN
# =============================================================================
def main():
    global LANDMARK_METHOD

    parser = argparse.ArgumentParser(description='Run experiments (our method only)')
    parser.add_argument('manifold', choices=['sphere', 'torus'], help='Manifold type')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID (e.g., 0, 1, 2)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--landmark-method', type=str, default=LANDMARK_METHOD,
                        choices=['random', 'fps'], help='Landmark sampling method')
    args = parser.parse_args()

    LANDMARK_METHOD = args.landmark_method

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

    print(f"\n{'='*70}")
    print(f"RUNNING OUR METHOD ON {args.manifold.upper()}")
    print(f"{'='*70}")

    kl_values = []
    kl_errors = []
    ess_values = []
    ess_errors = []
    ess_ratio_values = []
    runtimes = []

    for dim in cfg['dimensions']:
        manifold_name = cfg['manifold_name'](dim)
        print(f"\n[Ours] Training on {manifold_name}...")

        exp = build_experiment(manifold_name, cfg['base_density'], cfg['target_density'], dim)

        start_time = time.time()
        exp = train(exp)
        runtime = time.time() - start_time
        runtimes.append(runtime)

        key = jax.random.PRNGKey(EVAL_SEED)
        kl_vals = []
        ess_vals = []
        ess_ratio_vals = []
        for _ in range(N_EVAL_BATCHES):
            key, subkey = jax.random.split(key)
            result = compute_kl(exp, subkey, batch_size=EVAL_SIZE)
            kl_vals.append(result["kl"])
            ess_vals.append(result["ess"])
            ess_ratio_vals.append(result["ess_ratio"])

        kl_mean = np.mean(kl_vals)
        kl_se = np.std(kl_vals) / np.sqrt(N_EVAL_BATCHES)
        ess_mean = np.mean(ess_vals)
        ess_se = np.std(ess_vals) / np.sqrt(N_EVAL_BATCHES)
        ess_ratio_mean = np.mean(ess_ratio_vals)

        kl_values.append(kl_mean)
        kl_errors.append(kl_se)
        ess_values.append(ess_mean)
        ess_errors.append(ess_se)
        ess_ratio_values.append(ess_ratio_mean)

        print(f"  KL = {kl_mean:.4f} +/- {kl_se:.4f}  |  ESS = {ess_mean:.1f} (ratio: {ess_ratio_mean:.3f})")
        print(f"  Runtime = {runtime:.1f}s")

    np.savez(cfg['output'],
             dimensions=cfg['dimensions'],
             kl_values=kl_values,
             kl_errors=kl_errors,
             ess_values=ess_values,
             ess_errors=ess_errors,
             ess_ratio_values=ess_ratio_values,
             runtimes=runtimes)
    print(f"\nSaved: {cfg['output']}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {args.manifold.upper()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
