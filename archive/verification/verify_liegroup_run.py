#!/usr/bin/env python3
"""Record one native SO(3)/SE(3) experiment from the recovered paper runners.

Calls the unchanged builders, trainers and evaluators. One training seed and
five evaluation batches constitute a native row; these are not five training
replicates. RCPM gamma is selected only to distribute the original sweep across
fresh processes. No configuration, optimizer, loss or metric is replaced.
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
import sys
import time
import traceback

import verify_run as evidence

HELPER_SHA256 = "cd3671a45ecca9b55542d75353991ad591c61b5f18f2e3cc3a855875a61d7589"
CORE_VERSIONS = {"jax": "0.4.35", "jaxlib": "0.4.34", "flax": "0.8.4", "optax": "0.2.3"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).resolve().parent / "provenance/recovered_liegroup_source_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for this one row.")
    parser.add_argument("--manifold", choices=("SO3", "SE3"), required=True)
    parser.add_argument("--method", choices=("ours", "rcpm"), required=True)
    parser.add_argument("--gamma", type=float, help="Required for RCPM; one of the recovered sweep values.")
    parser.add_argument("--gpu", required=True, help="One physical NVIDIA GPU index.")
    args = parser.parse_args(argv)
    if not args.gpu.isdigit():
        parser.error("--gpu must be a single physical GPU index")
    if (args.method == "rcpm") != (args.gamma is not None):
        parser.error("--gamma is required only for RCPM")
    if args.gamma is not None and args.gamma not in (1.0, 0.1, 0.05, 0.01, 0.005, 0.001):
        parser.error("--gamma must be a value in the original six-value sweep")
    return args


def verify_sources(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    actual = evidence.source_manifest(root)
    if actual["files"] != manifest["files"] or actual["sha256"] != manifest["sha256"]:
        raise ValueError("Source files differ from the frozen recovered-runner manifest")
    helper_digest = hashlib.sha256(Path(evidence.__file__).read_bytes()).hexdigest()
    if helper_digest != HELPER_SHA256:
        raise ValueError("The original evidence helper has changed")
    return actual


def load_native(root, manifold):
    path = root / "experiments" / f"run_{manifold.lower()}_experiment.py"
    spec = importlib.util.spec_from_file_location(f"recovered_{manifold.lower()}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_configuration(module, experiment, args):
    if args.method == "ours":
        if args.manifold == "SO3":
            return dataclasses.asdict(experiment["cfg"])
        return {name: dataclasses.asdict(experiment[f"{name}_cfg"])
                for name in ("model", "solver", "training")}
    return {
        "builder": {"gamma": args.gamma, "n_components": 68, "n_transforms": 5, "seed": module.TRAINING_SEED},
        "flow": {"n_transforms": experiment["flow"].n_transforms,
                 "single_transform_cfg": dict(experiment["flow"].single_transform_cfg)},
        "training": {"n_iters": module.N_ITERS, "batch_size": module.BATCH_SIZE,
                     "lr": 1e-3, "log_every": module.N_ITERS // 10},
    }


def density_configuration(density):
    fields = ("loc", "scale", "rot_scale", "trans_scale", "trans_low", "trans_high", "t_range", "include_alpha_volume_correction")
    return {"class": type(density).__name__, **{name: evidence.json_safe(getattr(density, name))
            for name in fields if hasattr(density, name)}}


def main():
    args = parse_args()
    sys.dont_write_bytecode = True
    root = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "result.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HIP_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("MPLBACKEND", "Agg")
    record = {
        "schema_version": 1, "record_type": "native_liegroup_training_run",
        "status": "preparing", "started_at": evidence.utc_now(), "command": sys.argv,
        "label": "recovered_liegroup_native_cuda_jax435",
        "selection": {"manifold": args.manifold, "method": args.method, "gamma": args.gamma},
        "source_root": str(root), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": HELPER_SHA256, "explicit_configuration_overrides": {},
        "evaluation_batches": [], "timings_seconds": {},
        "protocol_note": "Native single-training-seed row. Five evaluation batches are Monte Carlo evaluations of that one model, not independent training runs.",
        "execution_note": "One gamma per fresh process. The original native builders/trainers/evaluators and their argument values are unchanged; the six-gamma main-loop cache history is not recreated.",
    }
    evidence.save_json(record_path, record)
    print(f"LIEGROUP_RESULT {record_path}", flush=True)
    try:
        record["source"] = verify_sources(root, args.source_manifest)
        record["environment"] = evidence.environment()
        actual = {name: importlib.metadata.version(name) for name in CORE_VERSIONS}
        if actual != CORE_VERSIONS:
            raise RuntimeError(f"Use the pinned restored environment: expected {CORE_VERSIONS}, got {actual}")
        module = load_native(root, args.manifold)
        if module.LANDMARK_METHOD != "fps" or module.N_EVAL_BATCHES != 5 or module.EVAL_SIZE != 1024:
            raise ValueError("Recovered native constants changed")
        if module.TRAINING_SEED != 12345 or module.EVAL_SEED != 12345:
            raise ValueError("Recovered native seed constants changed")
        record["environment"] = evidence.environment()
        record["environment"].update({
            "all_package_versions": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
            "jax_devices": [str(device) for device in module.jax.devices()],
            "jax_default_backend": module.jax.default_backend(),
            "jax_enable_x64": bool(module.jax.config.jax_enable_x64),
            "jax_threefry_partitionable": bool(module.jax.config.jax_threefry_partitionable),
            "jax_default_prng_impl": str(module.jax.config.jax_default_prng_impl),
        })
        if module.jax.default_backend() != "gpu" or not module.jax.config.jax_enable_x64:
            raise RuntimeError("The native CUDA reproduction requires a GPU and x64 enabled")
        if module.jax.config.jax_threefry_partitionable:
            raise RuntimeError("Use the original JAX 0.4.35 default Threefry behavior")
        record["evaluation_config"] = {"seed": module.EVAL_SEED, "batch_size": module.EVAL_SIZE,
                                       "n_batches": module.N_EVAL_BATCHES, "seed_offset": None}
        record["native_runner"] = {"file": f"experiments/run_{args.manifold.lower()}_experiment.py",
            "sha256": record["source"]["files"][f"experiments/run_{args.manifold.lower()}_experiment.py"]}
        record["status"] = "building"
        evidence.save_json(record_path, record)
        start = time.perf_counter()
        if args.method == "rcpm":
            experiment = module.build_rcpm_experiment(gamma=args.gamma, seed=module.TRAINING_SEED)
        else:
            builder = module.build_ours_experiment if args.manifold == "SO3" else module.build_experiment
            experiment = builder()
        record["timings_seconds"]["build"] = time.perf_counter() - start
        config = native_configuration(module, experiment, args)
        record["configuration"] = config
        record["selection"]["seed"] = config["training"]["seed"] if args.method == "ours" else module.TRAINING_SEED
        record["selection"]["landmark_method"] = module.LANDMARK_METHOD if args.method == "ours" else None
        record["density_configuration"] = {name: density_configuration(experiment[name]) for name in ("base", "target")}
        geometry = experiment["manifold"]
        record["geometry_configuration"] = {"class": type(geometry).__name__, "D": geometry.D,
            **{name: evidence.json_safe(getattr(geometry, name)) for name in ("alpha", "jitter") if hasattr(geometry, name)}}
        if args.method == "ours":
            record["landmark_configuration"] = {"method": module.LANDMARK_METHOD,
                "count": config["model"]["n_landmarks"], "fps_candidates_per_density": module.FPS_CANDIDATES}
        params = experiment["state"].params if args.method == "ours" else experiment["params"]
        record["parameter_count"] = sum(int(value.size) for value in module.jax.tree_util.tree_leaves(params))
        record["parameter_dtypes"] = sorted({str(value.dtype) for value in module.jax.tree_util.tree_leaves(params)})
        record["status"] = "training"
        evidence.save_json(record_path, record)
        print("NATIVE_CONFIGURATION " + json.dumps(evidence.json_safe(config), sort_keys=True), flush=True)
        start = time.perf_counter()
        if args.method == "rcpm":
            experiment = module.train_rcpm(experiment, n_iters=module.N_ITERS,
                batch_size=module.BATCH_SIZE, log_every=module.N_ITERS // 10)
        else:
            train = module.train_ours if args.manifold == "SO3" else module.train
            experiment = train(experiment)
        record["timings_seconds"]["training_native_return"] = time.perf_counter() - start
        params = experiment["state"].params if args.method == "ours" else experiment["params"]
        module.jax.block_until_ready(params)
        record["timings_seconds"]["training_synchronized"] = time.perf_counter() - start
        record["parameter_nonfinite_count"] = sum(int((~module.np.isfinite(module.np.asarray(value))).sum())
                                                  for value in module.jax.tree_util.tree_leaves(params))
        record["status"] = "trained"
        evidence.save_json(record_path, record)
        record["checkpoint"] = evidence.checkpoint(module, experiment, args.method, output / "checkpoint.msgpack")
        record["status"] = "evaluating"
        evidence.save_json(record_path, record)
        key = module.jax.random.PRNGKey(module.EVAL_SEED)
        for index in range(module.N_EVAL_BATCHES):
            key, subkey = module.jax.random.split(key)
            start = time.perf_counter()
            if args.method == "rcpm":
                kl, ess, ratio = module.compute_kl_rcpm(experiment, subkey, module.EVAL_SIZE)
                metrics = {"kl": kl, "ess": ess, "ess_ratio": ratio}
            else:
                evaluate = module.compute_kl_ours if args.manifold == "SO3" else module.evaluate
                metrics = evaluate(experiment, subkey, batch_size=module.EVAL_SIZE)
            batch = {"index": index, "prng_key": module.np.asarray(subkey).tolist(),
                "metrics": {name: float(value) for name, value in metrics.items()},
                "evaluation_seconds": time.perf_counter() - start}
            batch["metric_nonfinite"] = {name: not math.isfinite(value) for name, value in batch["metrics"].items()}
            record["evaluation_batches"].append(batch)
            evidence.save_json(record_path, record)
            print("NATIVE_EVAL_BATCH " + json.dumps(evidence.json_safe(batch)), flush=True)
        summary = {}
        for name in record["evaluation_batches"][0]["metrics"]:
            values = module.np.asarray([batch["metrics"][name] for batch in record["evaluation_batches"]])
            summary[name] = {"mean": float(module.np.mean(values)),
                "se_ddof0": float(module.np.std(values) / module.np.sqrt(module.N_EVAL_BATCHES)),
                "se_ddof1": float(module.np.std(values, ddof=1) / module.np.sqrt(module.N_EVAL_BATCHES)),
                "raw_batches": values.tolist()}
        record["summary"] = summary
        record["summary_note"] = "Original mean and population-standard-deviation/sqrt(5) across the five evaluation batches. Sample SE is supplemental. No training-seed aggregation."
        record["outcome"] = "nonfinite_output" if any(any(batch["metric_nonfinite"].values()) for batch in record["evaluation_batches"]) else "finite_output"
        record["source_unchanged_after_run"] = verify_sources(root, args.source_manifest) == record["source"]
        record["status"] = "complete"
        record["finished_at"] = evidence.utc_now()
        evidence.save_json(record_path, record)
        print("LIEGROUP_COMPLETE " + json.dumps(evidence.json_safe(summary)), flush=True)
    except BaseException as exc:
        record["failed_during"] = record["status"]
        record["status"] = "failed"
        record["finished_at"] = evidence.utc_now()
        record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        evidence.save_json(record_path, record)
        raise


if __name__ == "__main__":
    main()
