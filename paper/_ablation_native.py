#!/usr/bin/env python3
"""
Ablation study for Riemannian Neural OT on S² and T².

Usage:
    conda activate rcpms-jax && nohup python -u run_ablations.py --gpu 3 > ablations.log 2>&1 &
"""
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

import argparse
import time
import json
from dataclasses import replace
from typing import Dict, Any

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
# ABLATION CONFIGURATIONS
# =============================================================================
ABLATIONS = {
    # Baseline (current best config)
    "baseline": {},

    # === Solver ablations (most important) ===
    "no_logsumexp_init": {
        "solver": {"logsumexp_init": False}
    },
    "fewer_inner_steps_500": {
        "solver": {"inner_steps": 500, "min_steps": 500}
    },
    "more_inner_steps_4000": {
        "solver": {"inner_steps": 4000, "min_steps": 4000}
    },
    "inner_lr_5e-2": {
        "solver": {"inner_lr": 5e-2}
    },
    "inner_lr_1e-2": {
        "solver": {"inner_lr": 1e-2}
    },
    "inner_lr_1e-3": {
        "solver": {"inner_lr": 1e-3}
    },
    "logsumexp_gamma_1": {
        "solver": {"logsumexp_gamma": 1.0}
    },
    "logsumexp_gamma_0.1": {
        "solver": {"logsumexp_gamma": 0.1}
    },
    "logsumexp_gamma_0.01": {
        "solver": {"logsumexp_gamma": 0.01}
    },
    "logsumexp_gamma_0.001": {
        "solver": {"logsumexp_gamma": 0.001}
    },
    "no_adam": {
        "solver": {"use_adam": False}
    },

    # === Architecture ablations ===
    "landmarks_fps": {
        "landmark_method": "fps"
    },
    "landmarks_32": {
        "model": {"n_landmarks": 32}
    },
    "landmarks_64": {
        "model": {"n_landmarks": 64}
    },
    "landmarks_256": {
        "model": {"n_landmarks": 256}
    },
    "hidden_32_32": {
        "model": {"hidden_dims": (32, 32)}
    },
    "hidden_64_64": {
        "model": {"hidden_dims": (64, 64)}
    },
    "hidden_256_256": {
        "model": {"hidden_dims": (256, 256)}
    },
    "hidden_128_128_128": {
        "model": {"hidden_dims": (128, 128, 128)}
    },
    # === Training ablations ===
    "lr_1e-3": {
        "training": {"learning_rate": 1e-3}
    },
    "lr_1e-4": {
        "training": {"learning_rate": 1e-4}
    },
    "batch_512": {
        "training": {"batch_size": 512}
    },
    "batch_128": {
        "training": {"batch_size": 128}
    },
}

# Which ablations to run (edit this list)
ABLATIONS_TO_RUN = [
    "baseline",
    # === Solver ===
    "no_logsumexp_init",
    "fewer_inner_steps_500",
    "more_inner_steps_4000",
    "inner_lr_5e-2",
    "inner_lr_1e-3",
    "logsumexp_gamma_1",
    "logsumexp_gamma_0.01",
    "logsumexp_gamma_0.001",
    "no_adam",
    # === Architecture ===
    "landmarks_fps",
    "landmarks_32",
    "landmarks_64",
    "landmarks_256",
    "hidden_32_32",
    "hidden_64_64",
    "hidden_256_256",
    "hidden_128_128_128",
    # === Training ===
    "lr_1e-4",
    "batch_512",
    "batch_128",
]

# Manifolds to test
MANIFOLDS = ["S2", "T2"]

# Evaluation settings
EVAL_SIZE = 1024
N_EVAL_BATCHES = 5
EVAL_SEED = 12345

# Landmark FPS settings
FPS_CANDIDATES = 10000


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================
def apply_overrides(base_cfg: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    """Apply ablation overrides to config."""
    cfg = base_cfg

    if "model" in overrides:
        new_model = replace(cfg.model, **overrides["model"])
        cfg = replace(cfg, model=new_model)

    if "solver" in overrides:
        new_solver = replace(cfg.solver, **overrides["solver"])
        cfg = replace(cfg, solver=new_solver)

    if "training" in overrides:
        new_training = replace(cfg.training, **overrides["training"])
        cfg = replace(cfg, training=new_training)

    return cfg


def get_base_config(manifold_name: str) -> ExperimentConfig:
    """Get base config for a manifold."""
    if manifold_name.startswith("S"):
        return ExperimentConfig(
            manifold_name=manifold_name,
            base_density="SphereUniform",
            target_density="SphereWrappedNormal",
            jax_platform="gpu",
        )
    elif manifold_name.startswith("T"):
        return ExperimentConfig(
            manifold_name=manifold_name,
            base_density="TorusUniform",
            target_density="TorusWrappedNormal",
            jax_platform="gpu",
        )
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


def build_experiment(cfg: ExperimentConfig, landmark_method: str = "random"):
    """Build experiment from config."""
    manifold = get_manifold(cfg.manifold_name)
    base = densities.get(manifold, cfg.base_density)
    target = densities.get(manifold, cfg.target_density)

    key = jax.random.PRNGKey(cfg.training.seed)

    # Build landmarks using configured method
    n_landmarks = cfg.model.n_landmarks
    n_base = n_landmarks // 2
    n_target = n_landmarks - n_base

    if landmark_method == "fps":
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
        network_type=cfg.model.network_type,
        hidden_dims=cfg.model.hidden_dims,
        use_layernorm=cfg.model.use_layernorm,
        last_scale=cfg.model.last_scale,
        activation=cfg.model.activation,
        leaky_slope=cfg.model.leaky_slope,
        softplus_beta=cfg.model.softplus_beta,
        max_dist=cfg.model.max_dist,
    )

    key, kx_init, kparams = jax.random.split(key, 3)
    x_init = base.sample(kx_init, 16)
    psi_vars = psi.init(kparams, x_init)
    psi_params = psi_vars["params"]

    solver = ArgminSolver(
        manifold=manifold, psi_module=psi,
        inner_steps=cfg.solver.inner_steps,
        inner_lr=cfg.solver.inner_lr,
        grad_clip=cfg.solver.grad_clip,
        lr_decay=cfg.solver.lr_decay,
        tolerance=cfg.solver.tolerance,
        min_steps=cfg.solver.min_steps,
        momentum=cfg.solver.momentum,
        logsumexp_init=cfg.solver.logsumexp_init,
        logsumexp_gamma=cfg.solver.logsumexp_gamma,
        use_adam=cfg.solver.use_adam,
        adam_beta1=cfg.solver.adam_beta1,
        adam_beta2=cfg.solver.adam_beta2,
    )

    loss_obj = SemiDualLoss(manifold=manifold, psi_module=psi, solver=solver)
    trainer = SemiDualTrainer(
        manifold=manifold, psi_module=psi, loss_fn=loss_obj, solver=solver,
        learning_rate=cfg.training.learning_rate,
        lr_decay=cfg.training.lr_decay,
        lr_decay_alpha=cfg.training.lr_decay_alpha,
        n_steps=cfg.training.n_steps,
    )

    state = trainer.init_state(psi_params)
    return {
        "cfg": cfg, "manifold": manifold, "base": base, "target": target,
        "psi": psi, "solver": solver, "trainer": trainer, "state": state, "key": key,
    }


def train(experiment):
    """Train the model."""
    trainer = experiment["trainer"]
    cfg = experiment["cfg"]
    state, history = trainer.train(
        state=experiment["state"],
        base_density=experiment["base"],
        target_density=experiment["target"],
        key=experiment["key"],
        n_steps=cfg.training.n_steps,
        batch_size=cfg.training.batch_size,
        log_every=cfg.training.log_every,
        eval_every=cfg.training.eval_every,
        eval_size=cfg.training.eval_size,
        callbacks=[],
    )
    experiment["state"] = state
    return experiment


def evaluate(experiment, n_batches=N_EVAL_BATCHES, batch_size=EVAL_SIZE):
    """Evaluate KL divergence and loss."""
    key = jax.random.PRNGKey(EVAL_SEED)

    kl_vals = []
    ess_vals = []
    loss_vals = []

    loss_fn = experiment["trainer"].loss_fn

    for _ in range(n_batches):
        key, k1, k2 = jax.random.split(key, 3)
        xs = experiment["base"].sample(k1, batch_size)
        ys_target = experiment["target"].sample(k2, batch_size)

        ys, _ = experiment["solver"].batch_solve(
            experiment["state"].params, xs, ys_target
        )

        kl_ift = compute_kl_divergence(
            psi_params=experiment["state"].params,
            xs=xs,
            ys=ys,
            base_density=experiment["base"],
            target_density=experiment["target"],
            manifold=experiment["manifold"],
            psi_module=experiment["psi"],
        )

        kl_vals.append(kl_ift.kl)
        ess_vals.append(kl_ift.ess_ratio)

        loss = loss_fn(experiment["state"].params, xs, ys_target)
        loss_vals.append(float(loss))

    return {
        "kl": float(np.mean(kl_vals)),
        "kl_se": float(np.std(kl_vals) / np.sqrt(n_batches)),
        "ess_ratio": float(np.mean(ess_vals)),
        "loss": float(np.mean(loss_vals)),
        "loss_se": float(np.std(loss_vals) / np.sqrt(n_batches)),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--manifolds', type=str, nargs='+', default=MANIFOLDS,
                        help='Manifolds to test (e.g., --manifolds S10 T10)')
    args = parser.parse_args()

    # Create output directory if needed
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    manifolds = args.manifolds

    print(f"Running on: {jax.devices()}")
    print(f"Ablations: {ABLATIONS_TO_RUN}")
    print(f"Manifolds: {manifolds}")

    results = {}

    for manifold_name in manifolds:
        print(f"\n{'='*70}")
        print(f"MANIFOLD: {manifold_name}")
        print(f"{'='*70}")

        results[manifold_name] = {}
        base_cfg = get_base_config(manifold_name)

        for ablation_name in ABLATIONS_TO_RUN:
            print(f"\n[{manifold_name}] Ablation: {ablation_name}")

            overrides = ABLATIONS.get(ablation_name, {}).copy()
            landmark_method = overrides.pop("landmark_method", "random")
            cfg = apply_overrides(base_cfg, overrides)

            if overrides or landmark_method != "random":
                print(f"  Overrides: {overrides}" + (f", landmark_method={landmark_method}" if landmark_method != "random" else ""))

            try:
                start_time = time.time()
                exp = build_experiment(cfg, landmark_method=landmark_method)
                exp = train(exp)
                runtime = time.time() - start_time

                metrics = evaluate(exp)
                metrics["runtime"] = runtime

                results[manifold_name][ablation_name] = metrics

                print(f"  KL = {metrics['kl']:.4f} ± {metrics['kl_se']:.4f}")
                print(f"  ESS ratio = {metrics['ess_ratio']:.3f}")
                print(f"  Loss = {metrics['loss']:.4f} ± {metrics['loss_se']:.4f}")
                print(f"  Runtime = {runtime:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[manifold_name][ablation_name] = {"error": str(e)}

    # Save results
    # Build output filename from manifolds
    manifold_suffix = "_".join(manifolds)
    json_output = os.path.join(output_dir, f'ablation_{manifold_suffix}.json')
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_output}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for manifold_name in manifolds:
        print(f"\n{manifold_name}:")
        print(f"{'Ablation':<25} {'KL↓':>10} {'ESS↑':>10} {'Loss':>10} {'Time':>8}")
        print("-" * 65)

        baseline_kl = results[manifold_name].get("baseline", {}).get("kl", float('inf'))

        for ablation_name in ABLATIONS_TO_RUN:
            r = results[manifold_name].get(ablation_name, {})
            if "error" in r:
                print(f"{ablation_name:<25} {'ERROR':>10}")
            else:
                kl = r.get("kl", float('nan'))
                ess = r.get("ess_ratio", float('nan'))
                loss = r.get("loss", float('nan'))
                rt = r.get("runtime", float('nan'))
                marker = "" if kl <= baseline_kl * 1.1 else " ↑"
                print(f"{ablation_name:<25} {kl:>9.4f}{marker} {ess:>10.3f} {loss:>10.4f} {rt:>7.1f}s")

    # =================================================================
    # LATEX TABLE OUTPUT
    # =================================================================
    def fmt_val(mean, se=None, decimals=4):
        """Format value as mean ± se with specified decimals."""
        if se is not None:
            if decimals == 2:
                return f"{mean:.2f} $\\pm$ {se:.2f}"
            elif decimals == 3:
                return f"{mean:.3f} $\\pm$ {se:.3f}"
            else:
                return f"{mean:.4f} $\\pm$ {se:.4f}"
        else:
            if decimals == 2:
                return f"{mean:.2f}"
            elif decimals == 3:
                return f"{mean:.3f}"
            else:
                return f"{mean:.4f}"

    def ablation_display_name(name):
        """Convert ablation name to display format for LaTeX."""
        display_map = {
            "baseline": "Baseline",
            "no_logsumexp_init": "No LogSumExp init",
            "fewer_inner_steps_500": "Inner steps: 500",
            "more_inner_steps_4000": "Inner steps: 4000",
            "inner_lr_5e-2": "Inner LR: $5 \\times 10^{-2}$",
            "inner_lr_1e-2": "Inner LR: $10^{-2}$",
            "inner_lr_1e-3": "Inner LR: $10^{-3}$",
            "logsumexp_gamma_1": "$\\gamma_{\\text{LSE}}$: 1.0",
            "logsumexp_gamma_0.1": "$\\gamma_{\\text{LSE}}$: 0.1",
            "logsumexp_gamma_0.01": "$\\gamma_{\\text{LSE}}$: 0.01",
            "logsumexp_gamma_0.001": "$\\gamma_{\\text{LSE}}$: 0.001",
            "no_adam": "No Adam (SGD)",
            "landmarks_fps": "Landmarks: FPS",
            "landmarks_32": "Landmarks: 32",
            "landmarks_64": "Landmarks: 64",
            "landmarks_256": "Landmarks: 256",
            "hidden_32_32": "Hidden: [32, 32]",
            "hidden_64_64": "Hidden: [64, 64]",
            "hidden_256_256": "Hidden: [256, 256]",
            "hidden_128_128_128": "Hidden: [128, 128, 128]",
            "no_layernorm": "No LayerNorm",
            "activation_relu": "Activation: ReLU",
            "activation_tanh": "Activation: Tanh",
        }
        return display_map.get(name, name.replace("_", " ").title())

    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")

    # Build LaTeX table with ablations as rows, manifolds as column groups
    latex_lines = [
        r"\begin{tabular}{l" + "cccc" * len(manifolds) + "}",
        r"\toprule",
    ]

    # Header row
    header = r"\textbf{Ablation}"
    for manifold_name in manifolds:
        header += f" & \\multicolumn{{4}}{{c}}{{\\textbf{{{manifold_name}}}}}"
    header += r" \\"
    latex_lines.append(header)

    # Sub-header for metrics
    subheader = ""
    for _ in manifolds:
        subheader += r" & Loss $\downarrow$ & KL $\downarrow$ & ESS $\uparrow$ & Time (s)"
    subheader += r" \\"
    latex_lines.append(subheader)
    latex_lines.append(r"\midrule")

    # Data rows
    for ablation_name in ABLATIONS_TO_RUN:
        row = ablation_display_name(ablation_name)

        for manifold_name in manifolds:
            r = results.get(manifold_name, {}).get(ablation_name, {})
            if "error" in r or not r:
                row += r" & -- & -- & -- & --"
            else:
                loss = r.get("loss", float('nan'))
                loss_se = r.get("loss_se", 0)
                kl = r.get("kl", float('nan'))
                kl_se = r.get("kl_se", 0)
                ess = r.get("ess_ratio", float('nan'))
                rt = r.get("runtime", float('nan'))
                row += f" & {fmt_val(loss, loss_se, 2)} & {fmt_val(kl, kl_se, 2)} & {ess:.2f} & {rt:.0f}"

        row += r" \\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    latex_table = "\n".join(latex_lines)
    print(latex_table)

    # Save LaTeX table to file
    latex_output = os.path.join(output_dir, f'ablation_{manifold_suffix}.tex')
    with open(latex_output, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved: {latex_output}")


if __name__ == '__main__':
    main()
