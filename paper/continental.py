"""Train the recovered continental-drift recipe and export its transport samples.

The notebook's referenced checkpoint is unavailable. This fresh recipe preserves
its native builder/trainer and printed settings; it is not a reproduced figure.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

NOTEBOOK_SHA256 = "e73b9ecf125a3808a35a67dda860c55b8889ff8dd1cb1809eba08c7508a19975"
LANDMARK_METHOD = "fps"
FPS_CANDIDATES = 50000


def _load_backend():
    # CLI parsing and dry runs remain available without scientific dependencies.
    global jax, jnp, np, ExperimentConfig, get_manifold, densities
    global GromovDistanceEmbedding, build_landmarks, build_network, ArgminSolver
    global SemiDualLoss, SemiDualTrainer, EMACallback, EmpiricalSphereDensity
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.base import ExperimentConfig
    from src.manifolds import get as get_manifold
    import src.densities as densities
    from src.embeddings import GromovDistanceEmbedding, build_landmarks
    from src.networks import build_network
    from src.solvers import ArgminSolver
    from src.losses import SemiDualLoss
    from src.trainers import SemiDualTrainer, EMACallback
    from src.empirical import EmpiricalSphereDensity
    jax.config.update("jax_enable_x64", True)


# The next three function bodies are unchanged from notebook cell 0.
def build_experiment(cfg: ExperimentConfig, csv_path_base: str | None = None, csv_path_target: str | None = None):
    """Build manifold, densities, model, solver, loss, trainer, and initial state."""
    # Set device *before* creating arrays
    if cfg.jax_platform is not None:
        jax.config.update("jax_platform_name", cfg.jax_platform)

    manifold = get_manifold(cfg.manifold_name)

    if csv_path_base is None:
        base = densities.get(manifold, cfg.base_density)
    else:
        base = EmpiricalSphereDensity(manifold=manifold, csv_path=csv_path_base)
    if csv_path_target is None:
        target = densities.get(manifold, cfg.target_density)
    else:
        target = EmpiricalSphereDensity(manifold=manifold, csv_path=csv_path_target)

    # RNG
    key = jax.random.PRNGKey(cfg.training.seed)
    
    n_landmarks = cfg.model.n_landmarks
    n_base = n_landmarks // 2
    n_target = n_landmarks - n_base
    
    # Build landmarks using configured method
    if LANDMARK_METHOD == "fps":
        print(f"Using FPS landmarks: {n_base} from base, {n_target} from target (candidates={FPS_CANDIDATES})")
        landmark_cfg_base = {"n_landmarks": n_base, "method": "fps", "fps_candidates": FPS_CANDIDATES}
        landmark_cfg_target = {"n_landmarks": n_target, "method": "fps", "fps_candidates": FPS_CANDIDATES}
        landmarks_base, key = build_landmarks(manifold, base, landmark_cfg_base, key)
        landmarks_target, key = build_landmarks(manifold, target, landmark_cfg_target, key)
    else:
        print(f"Using random landmarks: {n_base} from base, {n_target} from target")
        key, k1, k2 = jax.random.split(key, 3)
        landmarks_base = base.sample(k1, n_base)
        landmarks_target = target.sample(k2, n_target)
    
    landmarks = jnp.concatenate([landmarks_base, landmarks_target], axis=0)
    print(f"✓ Generated {len(landmarks)} total landmarks")
    
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

    # init params
    key, kx_init, kparams = jax.random.split(key, 3)
    x_init = base.sample(kx_init, 16)
    psi_vars = psi.init(kparams, x_init)
    psi_params = psi_vars["params"]

    # solver + loss
    solver = ArgminSolver(
        manifold=manifold,
        psi_module=psi,
        inner_steps=cfg.solver.inner_steps,
        inner_lr=cfg.solver.inner_lr,
        grad_clip=cfg.solver.grad_clip,
        lr_decay=cfg.solver.lr_decay,
        tolerance=cfg.solver.tolerance,
        min_steps=cfg.solver.min_steps,
        momentum=cfg.solver.momentum,
        logsumexp_init=cfg.solver.logsumexp_init,
        logsumexp_gamma=cfg.solver.logsumexp_gamma,
    )

    loss_obj = SemiDualLoss(
        manifold=manifold,
        psi_module=psi,
        solver=solver,
    )

    trainer = SemiDualTrainer(
        manifold=manifold,
        psi_module=psi,
        loss_fn=loss_obj,
        solver=solver,
        learning_rate=cfg.training.learning_rate,
        lr_decay=cfg.training.lr_decay,
        lr_decay_alpha=cfg.training.lr_decay_alpha,
        n_steps=cfg.training.n_steps,
    )

    state = trainer.init_state(psi_params)

    return {
        "cfg": cfg,
        "manifold": manifold,
        "base": base,
        "target": target,
        "psi": psi,
        "solver": solver,
        "loss": loss_obj,
        "trainer": trainer,
        "state": state,
        "key": key,
        "landmarks": landmarks,
        "landmarks_base": landmarks_base,
        "landmarks_target": landmarks_target,
    }


def transport(solver, params, xs, y_samples):
    """Compute y*(x) for a batch xs."""
    ys, _ = solver.batch_solve(params, xs, y_samples)
    return ys


def train(experiment):
    """Run training and update exp dict in place."""
    trainer = experiment["trainer"]
    base = experiment["base"]
    target = experiment["target"]
    key = experiment["key"]
    state = experiment["state"]
    cfg = experiment["cfg"]

    # Create EMA callback
    ema_cb = EMACallback(decay=0.99, eval_with_ema=False)

    state, history = trainer.train(
        state=state,
        base_density=base,
        target_density=target,
        key=key,
        n_steps=cfg.training.n_steps,
        batch_size=cfg.training.batch_size,
        log_every=cfg.training.log_every,
        eval_every=cfg.training.eval_every,
        eval_size=cfg.training.eval_size,
        callbacks=[ema_cb],
    )

    experiment["state"] = state
    experiment["history"] = history
    experiment["ema_params"] = ema_cb.ema_params  # <-- Store EMA params
    return state, history



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit continental JSON recipe.")
    parser.add_argument("--output", type=Path, required=True, help="New result directory.")
    parser.add_argument("--gpu", help="One physical CUDA GPU index; required except for dry run.")
    parser.add_argument("--smoke", action="store_true", help="Tiny execution check, not the recovered full experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the recipe and data without importing JAX.")
    args = parser.parse_args(argv)
    if not args.dry_run and (args.gpu is None or not args.gpu.isdigit()):
        parser.error("Execution requires a single numeric --gpu index.")
    if args.output.exists():
        parser.error("The output directory must be new.")
    return args


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_record(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"lat", "lon"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected lat and lon columns: {path}")
        rows = sum(1 for _ in reader)
    if rows < 256:
        raise ValueError(f"Continental CSV requires at least 256 rows: {path}")
    return {"path": str(path), "sha256": _sha(path), "rows": rows, "bytes": path.stat().st_size}


def load_plan(args):
    root = Path(__file__).resolve().parent.parent
    recipe = json.loads(args.config.read_text())
    expected = {"schema_version", "label", "configuration", "landmark_method", "fps_candidates", "data", "transport", "provenance"}
    if set(recipe) != expected or recipe["schema_version"] != 1:
        raise ValueError("Unexpected continental recipe structure or version.")
    if recipe["landmark_method"] != "fps" or recipe["configuration"]["manifold_name"] != "S2":
        raise ValueError("This recovered recipe uses FPS on S2.")
    if recipe["transport"] != {"source_selection": "first_csv_rows", "size": 1024, "target_seed": 42, "parameters": "state.params"}:
        raise ValueError("The recovered transport recipe uses first 1024 CSV points, target seed 42, and state.params.")
    data = {role: _csv_record(root / entry["path"]) for role, entry in recipe["data"].items()}
    if set(data) != {"past", "present"}:
        raise ValueError("Expected past and present CSV files.")
    for role, item in data.items():
        if item["sha256"] != recipe["data"][role]["sha256"]:
            raise ValueError(f"CSV hash differs from explicit recipe: {role}")
    return root, recipe, data


def _smoke_csv(source, output):
    with source.open(newline="") as reader_handle, output.open("w", newline="") as writer_handle:
        reader = csv.DictReader(reader_handle)
        writer = csv.DictWriter(writer_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader):
            if index == 256:
                break
            writer.writerow(row)


def compute(args, recipe, data, record):
    from flax import serialization
    from paper._runtime import replace_config, save_json
    import pandas as pd
    global FPS_CANDIDATES, LANDMARK_METHOD
    cfg_values = recipe["configuration"]
    cfg = ExperimentConfig(**{k: v for k, v in cfg_values.items() if k not in ("model", "solver", "training")})
    for section in ("model", "solver", "training"):
        setattr(cfg, section, replace_config(getattr(cfg, section), cfg_values[section]))
    FPS_CANDIDATES = recipe["fps_candidates"]
    LANDMARK_METHOD = recipe["landmark_method"]
    paths = {role: Path(item["path"]) for role, item in data.items()}
    if args.smoke:
        cfg.model = dataclasses.replace(cfg.model, n_landmarks=8, hidden_dims=(16, 16))
        cfg.solver = dataclasses.replace(cfg.solver, inner_steps=2, min_steps=2)
        cfg.training = dataclasses.replace(cfg.training, n_steps=2, batch_size=8, log_every=1, eval_every=None)
        FPS_CANDIDATES = 64
        for role in paths:
            target = args.output / f"smoke_{role}.csv"
            _smoke_csv(paths[role], target)
            paths[role] = target
    record["configuration"] = dataclasses.asdict(cfg)
    record["used_data"] = {role: _csv_record(path) for role, path in paths.items()}
    record["landmarks"] = {"method": LANDMARK_METHOD, "fps_candidates": FPS_CANDIDATES}
    record["environment"].update(jax_backend=jax.default_backend(), jax_devices=[str(d) for d in jax.devices()],
        jax_enable_x64=bool(jax.config.jax_enable_x64), jax_threefry_partitionable=bool(jax.config.jax_threefry_partitionable))
    save_json(args.output / "result.json", record)
    started = time.perf_counter()
    experiment = build_experiment(cfg, csv_path_base=str(paths["past"]), csv_path_target=str(paths["present"]))
    jax.block_until_ready(experiment["state"].params)
    record["build_seconds"] = time.perf_counter() - started
    solver = experiment["solver"]
    record["effective_solver"] = {k: getattr(solver, k) for k in (
        "inner_steps", "inner_lr", "grad_clip", "lr_decay", "tolerance", "min_steps", "momentum", "eps",
        "logsumexp_init", "logsumexp_gamma", "use_adam", "adam_beta1", "adam_beta2", "use_line_search", "line_search_steps")}
    record["empirical_densities"] = {role: {"class": type(d).__name__, "points": int(d.data.shape[0]),
        "kde_bandwidth": d.kde_bandwidth, "n_mc_samples": d.n_mc_samples, "use_kde": d.use_kde, "log_Z": d.log_Z}
        for role, d in (("past", experiment["base"]), ("present", experiment["target"]))}
    record["training_key"] = np.asarray(experiment["key"]).tolist()
    save_json(args.output / "result.json", record)
    started = time.perf_counter()
    state, history = train(experiment)
    jax.block_until_ready(state.params)
    record["training_seconds"] = time.perf_counter() - started
    record["history"] = history
    payload = {name: experiment[name] for name in ("key", "landmarks", "landmarks_base", "landmarks_target", "ema_params")}
    payload.update(params=state.params, state=state)
    checkpoint_path = args.output / "checkpoint.msgpack"
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(payload))
    temporary.replace(checkpoint_path)
    record["checkpoint"] = {"file": checkpoint_path.name, "sha256": _sha(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size, "note": "Regular and EMA parameters retained. Transport uses regular state.params. Stored key is the native retained pre-training key; this is not an exact training-resume checkpoint."}
    save_json(args.output / "result.json", record)
    # Notebook cell 6: first CSV rows, float64 latitude/longitude conversion,
    # and corresponding target sample array used directly as solver hints.
    df_past = pd.read_csv(paths["past"])
    eval_size = min(16 if args.smoke else 1024, len(df_past))
    lat = np.deg2rad(df_past["lat"].values[:eval_size])
    lon = np.deg2rad(df_past["lon"].values[:eval_size])
    source = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    xs = jnp.array(source)
    target_key = jax.random.PRNGKey(42)
    targets = experiment["target"].sample(target_key, eval_size)
    started = time.perf_counter()
    ys = transport(solver, state.params, xs, targets)
    jax.block_until_ready(ys)
    record["transport_seconds"] = time.perf_counter() - started
    arrays = {"source_samples": np.asarray(xs), "target_samples": np.asarray(targets), "transported_samples": np.asarray(ys)}
    samples_path = args.output / "samples.npz"
    np.savez_compressed(samples_path, **arrays)
    record["samples"] = {"file": samples_path.name, "sha256": _sha(samples_path), "bytes": samples_path.stat().st_size,
        "size": eval_size, "target_key": np.asarray(target_key).tolist(), "source_selection": "first_csv_rows",
        "target_hint": "The native solver receives target_samples row-for-row.", "parameters": "state.params",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
    save_json(args.output / "samples.json", {"manifold": "S2", "label": record["label"], "smoke": args.smoke,
        "provenance": record["provenance"], "source": record["source"], "configuration": record["configuration"],
        "samples": record["samples"], "result_file": "result.json"})


def main(argv=None):
    args = parse_args(argv)
    root, recipe, data = load_plan(args)
    if args.dry_run:
        print(json.dumps({"recipe": recipe, "data": data, "smoke": args.smoke, "will_train": False}, indent=2))
        return 0
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cuda", JAX_ENABLE_X64="True",
        JAX_THREEFRY_PARTITIONABLE="False", XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", MPLBACKEND="Agg")
    from paper._runtime import environment, save_json, source_manifest, utc_now, validate_environment
    validate_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "record_type": "paper_continental_run", "status": "running", "started_at": utc_now(),
        "fresh_experiment": True, "smoke": args.smoke, "manifold": "S2",
        "label": "Continental smoke execution check" if args.smoke else recipe["label"],
        "requested_recipe": recipe, "config_sha256": _sha(args.config), "data": data,
        "source": source_manifest(root), "wrapper_sha256": _sha(Path(__file__)), "environment": environment(),
        "provenance": {**recipe["provenance"], "notebook_sha256": NOTEBOOK_SHA256,
            "native_cells": [0, 1, 6], "ema": {"decay": 0.99, "eval_with_ema": False},
            "solver_note": "Native builder omits Adam and line-search options; actual constructor defaults are recorded.",
            "smoke_changes": "First 256 CSV rows, 8 landmarks, 64 FPS candidates, hidden dimensions [16,16], 2 outer/inner/min steps, batch 8, 16 transport samples." if args.smoke else None}}
    save_json(args.output / "result.json", record)
    try:
        _load_backend()
        compute(args, recipe, data, record)
        record.update(status="complete", finished_at=utc_now())
        save_json(args.output / "result.json", record)
    except Exception as exc:
        record.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_at=utc_now())
        save_json(args.output / "result.json", record)
        raise
    print(args.output / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
