#!/usr/bin/env python3
"""Run one independently recorded experiment using the supplied ZIP sources.

No archive files are edited. Default execution delegates to the archive's own
builders, training functions, and evaluation functions. Explicit configuration
overrides use the same library classes with a separately labeled configuration.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_manifest(root):
    files = sorted(p for subdir in ("src", "experiments", "rcpm")
                   for p in (root / subdir).rglob("*")
                   if p.is_file() and (p.suffix in (".py", ".txt") or p.name == "LICENSE"))
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": manifest}


def package_versions():
    names = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin",
             "flax", "optax", "numpy", "scipy", "matplotlib", "cartopy", "spherical-kde", "pandas")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment():
    data = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "hostname": platform.node(), "packages": package_versions(),
            "environment": {k: os.environ.get(k) for k in (
                "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS", "MPLBACKEND")}}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        data["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        data["nvidia_smi"] = str(exc)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent / "archive")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("table", "sweep", "highD"), default="table")
    parser.add_argument("--manifold", choices=("sphere", "torus"), required=True)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--method", choices=("ours", "rcpm"), default="ours")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gamma", type=float, help="Default: table constant, or first sweep gamma (1.0).")
    parser.add_argument("--landmark-method", choices=("random", "fps"))
    parser.add_argument("--gpu", help="Physical GPU ID; set before importing JAX.")
    parser.add_argument("--overrides", type=Path, help="JSON object with model/solver/training/evaluation/rcpm overrides.")
    parser.add_argument("--label", help="Optional descriptive label, e.g. paper_settings.")
    parser.add_argument("--smoke", action="store_true", help="Tiny diagnostic configuration; never paper reproduction.")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.dimension < 2:
        parser.error("--dimension must be at least 2")
    if args.suite == "table" and args.dimension != 2:
        parser.error("the shipped table only covers dimension 2")
    if args.suite == "highD" and args.method != "ours":
        parser.error("the shipped highD suite only implements ours")
    if args.method == "ours" and args.gamma is not None:
        parser.error("--gamma applies only to RCPM")
    return args


def read_overrides(args):
    value = json.loads(args.overrides.read_text()) if args.overrides else {}
    if not isinstance(value, dict):
        raise ValueError("Override file must contain a JSON object")
    allowed = {"model", "solver", "training", "evaluation", "rcpm"}
    if set(value) - allowed:
        raise ValueError(f"Unknown override sections: {set(value) - allowed}")
    if any(not isinstance(v, dict) for v in value.values()):
        raise ValueError("Each override section must be a JSON object")
    if args.method == "rcpm" and any(value.get(k) for k in ("model", "solver", "training")):
        raise ValueError("RCPM overrides belong under rcpm, not model/solver/training")
    if args.method == "ours" and value.get("rcpm"):
        raise ValueError("rcpm overrides do not apply to ours")
    if "seed" in value.get("training", {}):
        raise ValueError("Use --seed instead of training.seed")
    if args.smoke:
        smoke = {"evaluation": {"batch_size": 16, "n_batches": 1}}
        if args.method == "ours":
            smoke.update({"model": {"n_landmarks": 32, "hidden_dims": [32, 32]},
                          "solver": {"inner_steps": 10, "min_steps": 5},
                          "training": {"n_steps": 2, "batch_size": 16}})
        else:
            smoke["rcpm"] = {"n_iters": 2, "batch_size": 16, "log_every": 1}
        # Explicit values take precedence; every smoke run remains labeled smoke.
        for section, fields in value.items():
            smoke.setdefault(section, {}).update(fields)
        value = smoke
    return value


def load_experiment_module(root, suite):
    name = {"table": "run_table", "sweep": "run_experiments", "highD": "run_experiments_highD"}[suite]
    spec = importlib.util.spec_from_file_location("zip_" + name, root / "experiments" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_config(cfg, values):
    known = {f.name for f in dataclasses.fields(cfg)}
    if set(values) - known:
        raise ValueError(f"Unknown {type(cfg).__name__} fields: {set(values) - known}")
    updates = {k: tuple(v) if isinstance(getattr(cfg, k), tuple) else v for k, v in values.items()}
    return dataclasses.replace(cfg, **updates)


def construct_overridden_ours(m, args, overrides, manifold_name, base_name, target_name):
    """Archive builder's exact order of RNG splits and library constructors."""
    cfg = m.ExperimentConfig(manifold_name=manifold_name, base_density=base_name,
                             target_density=target_name, jax_platform="gpu")
    cfg.training.seed = args.seed
    if args.suite == "highD":
        cfg.model.n_landmarks = cfg.model.n_landmarks if args.dimension < 30 else 512
        cfg.model.hidden_dims = (128, 128) if args.dimension > 30 else cfg.model.hidden_dims
    for section in ("model", "solver", "training"):
        setattr(cfg, section, replace_config(getattr(cfg, section), overrides.get(section, {})))
    if cfg.training.grad_accum_steps > 1:
        raise ValueError("Shipped experiment builders do not wire grad_accum_steps; unsupported override")
    manifold = m.get_manifold(manifold_name)
    base = m.densities.get(manifold, base_name)
    target = m.densities.get(manifold, target_name)
    key = m.jax.random.PRNGKey(args.seed)
    n_base = cfg.model.n_landmarks // 2
    n_target = cfg.model.n_landmarks - n_base
    if m.LANDMARK_METHOD == "fps":
        landmarks_base, key = m.build_landmarks(manifold, base, {
            "n_landmarks": n_base, "method": "fps", "fps_candidates": m.FPS_CANDIDATES}, key)
        landmarks_target, key = m.build_landmarks(manifold, target, {
            "n_landmarks": n_target, "method": "fps", "fps_candidates": m.FPS_CANDIDATES}, key)
    else:
        key, k1, k2 = m.jax.random.split(key, 3)
        landmarks_base = base.sample(k1, n_base)
        landmarks_target = target.sample(k2, n_target)
    landmarks = m.jnp.concatenate([landmarks_base, landmarks_target], axis=0)
    embedding = m.GromovDistanceEmbedding(manifold=manifold, landmarks=landmarks)
    network_config = dataclasses.asdict(cfg.model)
    network_config.pop("n_landmarks")
    psi = m.build_network(phi=embedding, **network_config)
    key, kx_init, kparams = m.jax.random.split(key, 3)
    params = psi.init(kparams, base.sample(kx_init, 16))["params"]
    solver = m.ArgminSolver(manifold=manifold, psi_module=psi, **dataclasses.asdict(cfg.solver))
    loss = m.SemiDualLoss(manifold=manifold, psi_module=psi, solver=solver)
    trainer = m.SemiDualTrainer(manifold=manifold, psi_module=psi, loss_fn=loss, solver=solver,
        **{k: getattr(cfg.training, k) for k in ("learning_rate", "lr_decay", "lr_decay_alpha", "n_steps")})
    return {"cfg": cfg, "manifold": manifold, "base": base, "target": target,
            "psi": psi, "solver": solver, "trainer": trainer, "state": trainer.init_state(params), "key": key}


def build_experiment(m, args, overrides):
    manifold_name = ("S" if args.manifold == "sphere" else "T") + str(args.dimension)
    base_name = "SphereUniform" if args.manifold == "sphere" else "TorusUniform"
    target_name = "SphereWrappedNormal" if args.manifold == "sphere" else "TorusWrappedNormal"
    if args.method == "rcpm":
        rcpm_manifold = (m.rcpm_manifolds.Sphere(D=args.dimension + 1, jitter=1e-2)
                        if args.manifold == "sphere" else m.rcpm_manifolds.Product(
                            D=2 * args.dimension, manifolds_str=",".join(["S1"] * args.dimension)))
        exp = m.build_rcpm_experiment(rcpm_manifold, is_torus=args.manifold == "torus",
                                      gamma=args.gamma, seed=args.seed)
        rcpm_cfg = {"n_iters": m.N_ITERS, "batch_size": m.BATCH_SIZE, "lr": 1e-3,
                    "log_every": m.N_ITERS if args.suite == "table" else m.N_ITERS // 10}
        if set(overrides.get("rcpm", {})) - set(rcpm_cfg):
            raise ValueError("Supported rcpm overrides: n_iters, batch_size, lr, log_every")
        rcpm_cfg.update(overrides.get("rcpm", {}))
        config = {"manifold_name": manifold_name, "base_density": base_name, "target_density": target_name,
                  "target_scale": 0.3, "seed": args.seed, "training": rcpm_cfg,
                  "n_components": 68, "n_transforms": 5, "cost_gamma": args.gamma,
                  "init_alpha_mode": "uniform", "init_alpha_linear_scale": 1.0,
                  "init_alpha_minval": 0.4, "init_alpha_range": 0.01, "min_zero_gamma": None}
        return exp, config, "native_archive_builder"
    custom = any(overrides.get(k) for k in ("model", "solver", "training"))
    custom |= args.suite != "table" and args.seed != m.ExperimentConfig().training.seed
    if custom:
        exp = construct_overridden_ours(m, args, overrides, manifold_name, base_name, target_name)
        builder = "explicit_configuration_builder_using_archive_classes"
    elif args.suite == "table":
        exp = m.build_ours_experiment(manifold_name, base_name, target_name, seed=args.seed)
        builder = "native_archive_builder"
    elif args.suite == "highD":
        exp = m.build_experiment(manifold_name, base_name, target_name, args.dimension)
        builder = "native_archive_builder"
    else:
        exp = m.build_ours_experiment(manifold_name, base_name, target_name)
        builder = "native_archive_builder"
    return exp, dataclasses.asdict(exp["cfg"]), builder


def residual_diagnostics(m, exp, key, batch_size):
    k1, k2 = m.jax.random.split(key)
    xs = exp["base"].sample(k1, batch_size)
    target_samples = exp["target"].sample(k2, batch_size)
    ys, residuals = exp["solver"].batch_solve(exp["state"].params, xs, target_samples)
    residuals = m.np.asarray(residuals)
    finite = m.np.isfinite(residuals)
    values = residuals[finite]
    return {"n": int(residuals.size), "nonfinite_count": int((~finite).sum()),
            "transport_nonfinite_count": int((~m.np.isfinite(m.np.asarray(ys))).sum()),
            "tolerance": exp["solver"].tolerance,
            "fraction_at_tolerance": float((finite & (residuals <= exp["solver"].tolerance)).mean()),
            "mean_finite": float(values.mean()) if values.size else None,
            "quantiles_finite": {str(q): float(m.np.quantile(values, q)) if values.size else None
                                 for q in (0, .5, .9, .95, .99, 1)}}


def checkpoint(m, exp, method, path):
    from flax import serialization
    payload = {"params": exp["state"].params, "state": exp["state"], "key": exp["key"],
               "landmarks": exp["psi"].phi.landmarks} if method == "ours" else {
                   "params": exp["params"], "key": exp["key"]}
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(payload))
    temporary.replace(path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "note": "Ours key is the archive's retained pre-training key; checkpoint supports evaluation, not exact training continuation."
                    if method == "ours" else "Archive RCPM training does not return optimizer state."}


def main():
    args = parse_args()
    # Importing the archive must not create __pycache__ files in its directory.
    sys.dont_write_bytecode = True
    args.source_root = args.source_root.resolve()
    overrides = read_overrides(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("MPLBACKEND", "Agg")
    source = source_manifest(args.source_root)
    if not source["files"]:
        raise ValueError("No source files found beneath --source-root")
    m = load_experiment_module(args.source_root, args.suite)
    if args.landmark_method:
        m.LANDMARK_METHOD = args.landmark_method
    if args.method == "rcpm" and args.gamma is None:
        args.gamma = m.RCPM_GAMMA if args.suite == "table" else m.GAMMAS[0]
    eval_cfg = {"batch_size": m.EVAL_SIZE, "n_batches": m.N_EVAL_BATCHES,
                "seed": args.seed + 1000 if args.suite == "table" else m.EVAL_SEED}
    if set(overrides.get("evaluation", {})) - set(eval_cfg):
        raise ValueError("Supported evaluation overrides: batch_size, n_batches, seed")
    eval_cfg.update(overrides.get("evaluation", {}))
    if eval_cfg["batch_size"] < 1 or eval_cfg["n_batches"] < 1:
        raise ValueError("Evaluation batch_size and n_batches must be positive")
    selection = {"suite": args.suite, "manifold": args.manifold, "dimension": args.dimension,
                 "method": args.method, "seed": args.seed, "gamma": args.gamma,
                 "landmark_method": m.LANDMARK_METHOD if args.method == "ours" else None}
    label = args.label or ("smoke" if args.smoke else "overrides" if overrides else "zip_defaults")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("--label may only contain letters, numbers, dot, underscore and hyphen")
    identity = {"selection": selection, "overrides": overrides, "source_sha256": source["sha256"], "label": label}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    task_id = f"{args.suite}_{args.method}_{'S' if args.manifold == 'sphere' else 'T'}{args.dimension}_seed{args.seed}_{label}_{digest}"
    run_dir = args.output_dir.resolve() / task_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    record = {"schema_version": 1, "task_id": task_id, "status": "building", "started_at": utc_now(),
              "selection": selection, "label": label, "smoke": args.smoke,
              "paper_reproduction": False,
              "paper_reproduction_note": "Numerical comparison with paper is a separate verification step; smoke runs are ineligible.",
              "explicit_overrides": overrides, "evaluation_config": eval_cfg,
              "source_root": str(args.source_root), "source": source, "environment": environment(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": sys.argv, "evaluation_batches": [], "timings_seconds": {}}
    record["environment"]["jax_devices"] = [str(device) for device in m.jax.devices()]
    record["environment"]["jax_default_backend"] = m.jax.default_backend()
    record["environment"]["jax_enable_x64"] = bool(m.jax.config.jax_enable_x64)
    save_json(result_path, record)
    print(f"VERIFICATION_RESULT {result_path}", flush=True)
    try:
        start = time.perf_counter()
        exp, config, builder = build_experiment(m, args, overrides)
        record["timings_seconds"]["build"] = time.perf_counter() - start
        record["configuration"] = config
        record["builder"] = builder
        record["density_configuration"] = {
            "base_class": type(exp["base"]).__name__, "target_class": type(exp["target"]).__name__,
            "target_loc": exp["target"].loc, "target_scale": exp["target"].scale}
        record["geometry_configuration"] = {
            "class": type(exp["manifold"]).__name__, "ambient_dimension": exp["manifold"].D,
            "jitter": getattr(exp["manifold"], "jitter", None),
            "components": [{"class": type(manifold).__name__, "ambient_dimension": manifold.D,
                            "jitter": getattr(manifold, "jitter", None)}
                           for manifold in getattr(exp["manifold"], "manifolds", [])]}
        if args.method == "ours":
            count = exp["cfg"].model.n_landmarks
            record["landmark_configuration"] = {
                "method": m.LANDMARK_METHOD, "base_count": count // 2, "target_count": count - count // 2,
                "fps_candidates_per_density": m.FPS_CANDIDATES if m.LANDMARK_METHOD == "fps" else None}
        params = exp["state"].params if args.method == "ours" else exp["params"]
        record["parameter_count"] = sum(int(x.size) for x in m.jax.tree_util.tree_leaves(params))
        record["parameter_dtypes"] = sorted({str(x.dtype) for x in m.jax.tree_util.tree_leaves(params)})
        record["status"] = "training"
        save_json(result_path, record)
        start = time.perf_counter()
        if args.method == "rcpm":
            exp = m.train_rcpm(exp, **config["training"])
        elif args.suite == "highD":
            exp = m.train(exp)
        else:
            exp = m.train_ours(exp)
        record["timings_seconds"]["training_native_return"] = time.perf_counter() - start
        params = exp["state"].params if args.method == "ours" else exp["params"]
        m.jax.block_until_ready(params)
        record["timings_seconds"]["training_synchronized"] = time.perf_counter() - start
        record["parameter_nonfinite_count"] = sum(int((~m.np.isfinite(m.np.asarray(x))).sum())
                                                   for x in m.jax.tree_util.tree_leaves(params))
        record["status"] = "trained"
        save_json(result_path, record)
        if not args.no_checkpoint:
            record["checkpoint"] = checkpoint(m, exp, args.method, run_dir / "checkpoint.msgpack")
        record["status"] = "evaluating"
        save_json(result_path, record)
        key = m.jax.random.PRNGKey(eval_cfg["seed"])
        for batch_index in range(eval_cfg["n_batches"]):
            key, subkey = m.jax.random.split(key)
            start = time.perf_counter()
            if args.method == "rcpm":
                kl, ess, ratio = m.compute_kl_rcpm(exp, subkey, eval_cfg["batch_size"])
                metrics = {"kl": kl, "ess": ess, "ess_ratio": ratio}
            else:
                compute = m.compute_kl if args.suite == "highD" else m.compute_kl_ours
                metrics = compute(exp, subkey, batch_size=eval_cfg["batch_size"])
            batch = {"index": batch_index, "prng_key": m.np.asarray(subkey).tolist(), "metrics": metrics,
                     "metric_nonfinite": {k: not math.isfinite(float(v)) for k, v in metrics.items()},
                     "evaluation_seconds": time.perf_counter() - start}
            record["evaluation_batches"].append(batch)
            # Persist the original metrics before optional additional diagnostics.
            save_json(result_path, record)
            if args.method == "ours":
                start = time.perf_counter()
                batch["residual_diagnostics"] = residual_diagnostics(m, exp, subkey, eval_cfg["batch_size"])
                batch["diagnostic_seconds"] = time.perf_counter() - start
                save_json(result_path, record)
            print(f"EVAL_BATCH {batch_index + 1}/{eval_cfg['n_batches']} {json.dumps(json_safe(metrics))}", flush=True)
        summary = {}
        for name in ("kl", "ess", "ess_ratio"):
            values = m.np.asarray([b["metrics"][name] for b in record["evaluation_batches"]], dtype=float)
            summary[name] = {"mean": float(values.mean()),
                             "se_ddof0": float(values.std(ddof=0) / m.np.sqrt(values.size)),
                             "se_ddof1": float(values.std(ddof=1) / m.np.sqrt(values.size)) if values.size > 1 else None,
                             "raw_batches": values.tolist()}
        record["summary"] = summary
        record["summary_note"] = "Batch errors are evaluation Monte Carlo uncertainty. Table comparison requires aggregating independent training seeds."
        record["status"] = "complete"
        record["finished_at"] = utc_now()
        save_json(result_path, record)
        print(f"VERIFICATION_COMPLETE {result_path}", flush=True)
    except BaseException as exc:
        record["status"] = "failed"
        record["finished_at"] = utc_now()
        record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        save_json(result_path, record)
        raise


if __name__ == "__main__":
    main()
