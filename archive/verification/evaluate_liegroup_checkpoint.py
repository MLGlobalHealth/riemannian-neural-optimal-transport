#!/usr/bin/env python3
"""Replay five recorded native SO3/SE3 batches from fixed saved state; never train."""
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
import re
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
CORE = {"jax": "0.4.35", "jaxlib": "0.4.34", "flax": "0.8.4", "optax": "0.2.3"}
HELPER_SHA = "cd3671a45ecca9b55542d75353991ad591c61b5f18f2e3cc3a855875a61d7589"
WRAPPER_SHA = "a5cd6e5eae86918a5ebb788e6249bd65f9def4b52edfa12bf624870e54a1da33"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_packages(packages):
    return {re.sub(r"[-_.]+", "-", name).lower(): version for name, version in packages.items()}


def read_pins(path):
    pins = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", line.strip()):
                raise ValueError("The runtime freeze must contain exact package==version pins.")
            name, version = line.strip().split("==")
            pins[name] = version
    pins = canonical_packages(pins)
    if len(pins) != 80 or any(pins.get(name) != version for name, version in CORE.items()):
        raise ValueError("Requires the original 80-pin restored runtime freeze.")
    return pins


def validate_packages(saved, actual, pins, python_version):
    if tuple(python_version[:2]) != (3, 11):
        raise ValueError("Replay requires Python 3.11; patch and build are recorded.")
    for label, packages in (("original", saved["environment"]["all_package_versions"]), ("replay", actual)):
        packages = canonical_packages(packages)
        mismatches = {name: {"required": version, "found": packages.get(name)}
                      for name, version in pins.items() if packages.get(name) != version}
        if mismatches:
            raise ValueError(f"{label} runtime differs from the frozen 80 pins: {mismatches}")


def validate_record(saved):
    selection = saved["selection"]
    if saved.get("record_type") != "native_liegroup_training_run" or saved.get("status") != "complete":
        raise ValueError("Requires a completed native Lie-group training record.")
    if saved.get("label") != "recovered_liegroup_native_cuda_jax435" or saved.get("explicit_configuration_overrides") != {}:
        raise ValueError("Requires the unchanged recovered native recipe.")
    if selection.get("manifold") not in ("SO3", "SE3") or selection.get("method") not in ("ours", "rcpm"):
        raise ValueError("Only SO3/SE3 RNOT or RCPM gamma=1 are supported.")
    if selection.get("gamma") != (1.0 if selection["method"] == "rcpm" else None):
        raise ValueError("Only the four declared models are supported.")
    if selection.get("seed") != 12345 or selection.get("landmark_method") != ("fps" if selection["method"] == "ours" else None):
        raise ValueError("Unexpected native training seed or landmark recipe.")
    if saved.get("source_unchanged_after_run") is not True:
        raise ValueError("Original source preservation was not established.")
    if saved.get("runner_sha256") != WRAPPER_SHA or saved.get("helper_sha256") != HELPER_SHA:
        raise ValueError("Input helper/wrapper hashes differ from the frozen tools.")
    if saved.get("evaluation_config") != {"seed": 12345, "batch_size": 1024, "n_batches": 5, "seed_offset": None}:
        raise ValueError("Requires exactly the native five batches and seed schedule.")
    env = saved["environment"]
    if env.get("jax_enable_x64") is not True or env.get("jax_threefry_partitionable") is not False or env.get("jax_default_prng_impl") != "threefry2x32":
        raise ValueError("Input precision/PRNG settings differ from the declared runtime.")
    if any(env.get("packages", {}).get(name) != version for name, version in CORE.items()):
        raise ValueError("Input core runtime versions differ.")
    batches = saved.get("evaluation_batches", [])
    if len(batches) != 5:
        raise ValueError("Requires all five recorded evaluation subkeys.")
    for index, batch in enumerate(batches):
        key = batch.get("prng_key")
        if batch.get("index") != index or not isinstance(key, list) or len(key) != 2 or any(
                type(value) is not int or not 0 <= value < 2**32 for value in key):
            raise ValueError("Each batch requires its indexed uint32[2] evaluation key.")
        if not {"kl", "ess", "ess_ratio"} <= batch.get("metrics", {}).keys():
            raise ValueError("An original batch is missing native metrics.")
    relative = f"experiments/run_{selection['manifold'].lower()}_experiment.py"
    if saved["native_runner"] != {"file": relative, "sha256": saved["source"]["files"][relative]}:
        raise ValueError("Native runner provenance does not match the source manifest.")


def verify_inputs(saved, source_root, checkpoint, helper_path, wrapper_path, helper, expected_hash=None):
    if sha256(helper_path) != HELPER_SHA or sha256(wrapper_path) != WRAPPER_SHA:
        raise ValueError("Helper/wrapper files differ from the frozen input tools.")
    digest = sha256(checkpoint)
    if digest != saved["checkpoint"]["sha256"] or (expected_hash and digest != expected_hash):
        raise ValueError("Checkpoint SHA256 differs from the recorded/common checkpoint.")
    if checkpoint.stat().st_size != saved["checkpoint"]["bytes"]:
        raise ValueError("Checkpoint byte count differs from the input record.")
    if helper.source_manifest(source_root) != saved["source"]:
        raise ValueError("Native source differs from the original source manifest.")
    return digest


def array_fingerprint(value, np):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "size": int(array.size),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "nonfinite_count": int((~np.isfinite(array)).sum())}


def parameter_fingerprints(tree, np, prefix="params"):
    if isinstance(tree, dict):
        return {path: value for key in sorted(tree) for path, value in
                parameter_fingerprints(tree[key], np, prefix + "/" + str(key)).items()}
    if isinstance(tree, (tuple, list)):
        return {path: value for index, item in enumerate(tree) for path, value in
                parameter_fingerprints(item, np, prefix + "/" + str(index)).items()}
    return {prefix: array_fingerprint(tree, np)}


def restore_experiment(module, wrapper, saved, payload):
    selection = SimpleNamespace(**saved["selection"])
    if selection.method == "rcpm":
        experiment = module.build_rcpm_experiment(gamma=selection.gamma, seed=selection.seed)
    else:
        builder = module.build_ours_experiment if selection.manifold == "SO3" else module.build_experiment
        experiment = builder()
    helper = wrapper.evidence
    if helper.json_safe(wrapper.native_configuration(module, experiment, selection)) != saved["configuration"]:
        raise ValueError("Reconstructed native configuration differs from the original record.")
    density = {name: wrapper.density_configuration(experiment[name]) for name in ("base", "target")}
    geometry = experiment["manifold"]
    geometry_config = {"class": type(geometry).__name__, "D": geometry.D,
        **{name: helper.json_safe(getattr(geometry, name)) for name in ("alpha", "jitter") if hasattr(geometry, name)}}
    if helper.json_safe(density) != saved["density_configuration"] or helper.json_safe(geometry_config) != saved["geometry_configuration"]:
        raise ValueError("Reconstructed densities or geometry differ from the recorded recipe.")
    original_params = experiment["state"].params if selection.method == "ours" else experiment["params"]
    before = parameter_fingerprints(payload["params"], module.np)
    initialized = parameter_fingerprints(original_params, module.np)
    signature = lambda arrays: {path: (value["shape"], value["dtype"]) for path, value in arrays.items()}
    if signature(before) != signature(initialized):
        raise ValueError("Saved parameter paths/shapes/dtypes differ from the native model.")
    params = module.jax.tree_util.tree_map(module.jnp.asarray, payload["params"])
    if selection.method == "ours":
        if parameter_fingerprints(payload["state"]["params"], module.np) != before:
            raise ValueError("Checkpoint state.params and top-level params disagree.")
        cfg = experiment["cfg"] if selection.manifold == "SO3" else None
        model_cfg = cfg.model if cfg is not None else experiment["model_cfg"]
        solver_cfg = cfg.solver if cfg is not None else experiment["solver_cfg"]
        lm_config = {"method": module.LANDMARK_METHOD, "count": model_cfg.n_landmarks,
                     "fps_candidates_per_density": module.FPS_CANDIDATES}
        if lm_config != saved["landmark_configuration"]:
            raise ValueError("Reconstructed landmark recipe differs from the original.")
        landmarks = module.jnp.asarray(payload["landmarks"])
        old_lm = array_fingerprint(experiment["psi"].phi.landmarks, module.np)
        new_lm = array_fingerprint(landmarks, module.np)
        if (old_lm["shape"], old_lm["dtype"]) != (new_lm["shape"], new_lm["dtype"]):
            raise ValueError("Saved landmark shapes/dtypes differ from the native model.")
        embedding = module.GromovDistanceEmbedding(manifold=geometry, landmarks=landmarks)
        psi = experiment["psi"].clone(phi=embedding)
        experiment["psi"] = psi
        experiment["state"] = experiment["state"].replace(params=params)
        # ArgminSolver captures psi in its compiled closures; construct it afresh.
        experiment["solver"] = module.ArgminSolver(manifold=geometry, psi_module=psi,
                                                     **dataclasses.asdict(solver_cfg))
    else:
        experiment["params"] = params
    restored = {"parameters": parameter_fingerprints(params, module.np)}
    if restored["parameters"] != before:
        raise ValueError("Parameter restoration changed saved array bytes or metadata.")
    if selection.method == "ours":
        restored["landmarks"] = array_fingerprint(experiment["psi"].phi.landmarks, module.np)
        if restored["landmarks"] != array_fingerprint(payload["landmarks"], module.np):
            raise ValueError("Landmark restoration changed saved array bytes or metadata.")
    if sum(value["size"] for value in before.values()) != saved["parameter_count"] or sorted({value["dtype"] for value in before.values()}) != saved["parameter_dtypes"]:
        raise ValueError("Restored parameter count/dtypes differ from the input record.")
    if sum(value["nonfinite_count"] for value in before.values()) != saved["parameter_nonfinite_count"]:
        raise ValueError("Restored nonfinite parameter count differs from the input record.")
    return experiment, restored


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--helper", type=Path, default=Path(__file__).with_name("verify_run.py"))
    parser.add_argument("--wrapper", type=Path, default=Path(__file__).with_name("verify_liegroup_run.py"))
    parser.add_argument("--requirements", type=Path, default=Path(__file__).parent / "environments/reproduction-cuda.txt")
    parser.add_argument("--output", type=Path, required=True, help="Fresh evaluation-only JSON file.")
    parser.add_argument("--gpu", required=True, help="One physical NVIDIA GPU index.")
    args = parser.parse_args(argv)
    if not args.gpu.isdigit() or args.output.exists():
        parser.error("Requires one numeric GPU index and a fresh output file.")
    saved = json.loads(args.input_result.read_text())
    validate_record(saved)
    checkpoint = args.checkpoint or args.input_result.with_name(saved["checkpoint"]["file"])
    # Verify executable helper files before importing them (both use stdlib only).
    if sha256(args.helper) != HELPER_SHA or sha256(args.wrapper) != WRAPPER_SHA:
        raise ValueError("Frozen helper/wrapper SHA256 mismatch.")
    helper = load_file("verify_run", args.helper)
    wrapper = load_file("checkpoint_liegroup_wrapper", args.wrapper)
    checkpoint_hash = verify_inputs(saved, args.source_root, checkpoint, args.helper, args.wrapper, helper,
                                    args.expected_checkpoint_sha256)
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    pins = read_pins(args.requirements)
    validate_packages(saved, packages, pins, sys.version_info)
    os.environ.update(JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES=args.gpu, HIP_VISIBLE_DEVICES=args.gpu,
        JAX_ENABLE_X64="True", JAX_THREEFRY_PARTITIONABLE="False", JAX_DEFAULT_PRNG_IMPL="threefry2x32",
        JAX_RANDOM_SEED_OFFSET="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg")
    record = {"schema_version": 1, "record_type": "liegroup_checkpoint_evaluation", "evaluation_only": True,
        "independent_training_run": False, "label": "recovered_liegroup_fixed_checkpoint_cuda_jax435",
        "status": "evaluation_only_building", "started_at": helper.utc_now(), "command": sys.argv,
        "input_result": str(args.input_result.resolve()), "input_result_sha256": sha256(args.input_result),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "source_root": str(args.source_root.resolve()), "source": saved["source"],
        "runner_sha256": WRAPPER_SHA, "helper_sha256": HELPER_SHA, "evaluator_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(args.requirements), "validated_runtime_pins": pins,
        "selection": saved["selection"], "configuration": saved["configuration"],
        "density_configuration": saved["density_configuration"], "geometry_configuration": saved["geometry_configuration"],
        "evaluation_config": saved["evaluation_config"], "original_environment": saved["environment"],
        "evaluation_batches": [], "interpretation": "Fixed saved model and the exact five recorded evaluation keys. Identical sampled arrays across hosts are not assumed. No training is performed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    try:
        module = wrapper.load_native(args.source_root.resolve(), saved["selection"]["manifold"])
        from flax import serialization
        module.jax.config.update("jax_default_matmul_precision", None)
        config = {name: getattr(module.jax.config, name) for name in ("jax_enable_x64", "jax_threefry_partitionable",
            "jax_default_prng_impl", "jax_random_seed_offset", "jax_default_matmul_precision")}
        if config != {"jax_enable_x64": True, "jax_threefry_partitionable": False, "jax_default_prng_impl": "threefry2x32",
                      "jax_random_seed_offset": 0, "jax_default_matmul_precision": None} or module.jax.default_backend() != "gpu":
            raise ValueError("Native runtime precision/PRNG/backend guards failed.")
        if (module.TRAINING_SEED, module.EVAL_SEED, module.EVAL_SIZE, module.N_EVAL_BATCHES, module.LANDMARK_METHOD) != (12345, 12345, 1024, 5, "fps"):
            raise ValueError("Native constants differ from the recorded protocol.")
        record["environment"] = helper.environment()
        record["environment"].update(all_package_versions=packages, jax_config=config,
            jax_devices=[str(device) for device in module.jax.devices()], jax_default_backend=module.jax.default_backend())
        record["original_host_replay"] = record["environment"]["hostname"] == saved["environment"]["hostname"]
        # Independently validate the recorded subkey schedule before replaying its keys directly.
        key = module.jax.random.PRNGKey(12345)
        for original in saved["evaluation_batches"]:
            key, subkey = module.jax.random.split(key)
            if module.np.asarray(subkey).tolist() != original["prng_key"]:
                raise ValueError("Recorded evaluation subkeys differ from the native seed schedule.")
        payload = serialization.msgpack_restore(checkpoint.read_bytes())
        experiment, restored = restore_experiment(module, wrapper, saved, payload)
        record["restored_arrays"] = restored
        record["status"] = "evaluation_only_evaluating"
        helper.save_json(args.output, record)
        for original in saved["evaluation_batches"]:
            subkey = module.jnp.asarray(original["prng_key"], dtype=module.jnp.uint32)
            start = time.perf_counter()
            if saved["selection"]["method"] == "rcpm":
                kl, ess, ratio = module.compute_kl_rcpm(experiment, subkey, 1024)
                metrics = {"kl": kl, "ess": ess, "ess_ratio": ratio}
            else:
                evaluate = module.compute_kl_ours if saved["selection"]["manifold"] == "SO3" else module.evaluate
                metrics = evaluate(experiment, subkey, batch_size=1024)
            metrics = {name: float(value) for name, value in metrics.items()}
            if metrics.keys() != original["metrics"].keys():
                raise ValueError("Native evaluation returned a different set of metric names.")
            batch = {"index": original["index"], "prng_key": original["prng_key"], "metrics": metrics,
                "evaluation_seconds": time.perf_counter() - start, "original_metrics": original["metrics"],
                "metric_nonfinite": {name: not math.isfinite(value) for name, value in metrics.items()},
                "comparison_to_original": {name: {"exact_equal": value == float(original["metrics"][name]),
                    "difference": value - float(original["metrics"][name])} for name, value in metrics.items()}}
            record["evaluation_batches"].append(batch)
            helper.save_json(args.output, record)
            print("LIEGROUP_EVALUATION_ONLY_BATCH", json.dumps(helper.json_safe(batch)), flush=True)
        record["summary"] = {}
        for name in record["evaluation_batches"][0]["metrics"]:
            values = module.np.asarray([batch["metrics"][name] for batch in record["evaluation_batches"]])
            record["summary"][name] = {"mean": float(module.np.mean(values)),
                "se_ddof0": float(module.np.std(values) / module.np.sqrt(5)), "raw_batches": values.tolist()}
        verify_inputs(saved, args.source_root, checkpoint, args.helper, args.wrapper, helper, checkpoint_hash)
        if sha256(args.input_result) != record["input_result_sha256"]:
            raise ValueError("Original input record changed during evaluation.")
        record["source_and_checkpoint_unchanged_after_replay"] = True
        record["all_five_batches_exactly_match_original"] = all(check["exact_equal"] for batch in record["evaluation_batches"]
            for check in batch["comparison_to_original"].values())
        record["status"] = "evaluation_only_nonfinite" if any(any(batch["metric_nonfinite"].values()) for batch in record["evaluation_batches"]) else "evaluation_only_complete"
        record["finished_at"] = helper.utc_now()
        helper.save_json(args.output, record)
        return 0 if record["status"] == "evaluation_only_complete" else 1
    except BaseException as exc:
        record.update(status="evaluation_only_failed", finished_at=helper.utc_now(),
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        helper.save_json(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
