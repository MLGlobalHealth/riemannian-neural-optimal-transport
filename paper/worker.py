"""Execute one declared paper task. Use paper.run to create its manifest."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from . import _runtime as runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    if not args.gpu.isdigit():
        parser.error("GPU must be one physical numeric index")
    plan = json.loads(args.manifest.read_text())
    matches = [job for job in plan["jobs"] if job["job_id"] == args.job_id]
    if len(matches) != 1:
        parser.error("Job must be declared exactly once in the manifest")
    job = matches[0]
    root = Path(plan["source_root"])
    output = Path(job["result"]).parent
    output.mkdir(parents=True, exist_ok=False)
    path = output / "result.json"
    os.environ.update(plan["environment_overrides"])
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HIP_VISIBLE_DEVICES=args.gpu)
    sys.dont_write_bytecode = True
    selection, recipe, smoke = job["selection"], plan["recipe"], plan["smoke"]
    record = {"schema_version": 1, "record_type": "paper_experiment", "status": "preparing",
        "job_id": args.job_id, "started_at": runtime.utc_now(), "selection": selection,
        "smoke": smoke, "label": "smoke_not_paper_reproduction" if smoke else recipe["label"],
        "recipe": recipe, "recipe_sha256": plan["recipe_sha256"], "source_root": str(root),
        "paper_source": plan["paper_source"], "evaluation_batches": [], "timings_seconds": {},
        "execution_note": plan["execution_note"], "command": sys.argv,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()}
    runtime.save_json(path, record)
    try:
        runtime.validate_environment()
        source = runtime.source_manifest(root)
        if source != plan["source"]:
            raise ValueError("Native source changed after this task was declared")
        record["source"] = source
        for relative, expected in plan["paper_source"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Paper tooling changed after declaration: {relative}")
        module = runtime.load_native(root, selection["suite"], selection["manifold"])
        record["environment"] = runtime.environment()
        record["environment"].update(all_package_versions={item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
            jax_devices=[str(device) for device in module.jax.devices()], jax_default_backend=module.jax.default_backend(),
            jax_enable_x64=bool(module.jax.config.jax_enable_x64),
            jax_threefry_partitionable=bool(module.jax.config.jax_threefry_partitionable),
            jax_default_prng_impl=str(module.jax.config.jax_default_prng_impl),
            jax_random_seed_offset=module.jax.config.jax_random_seed_offset,
            jax_default_matmul_precision=module.jax.config.jax_default_matmul_precision)
        if (bool(module.jax.config.jax_enable_x64) != recipe["x64"] or module.jax.config.jax_threefry_partitionable
                or module.jax.config.jax_default_prng_impl != "threefry2x32" or module.jax.config.jax_random_seed_offset != 0):
            raise ValueError("Runtime precision/PRNG settings differ from the declared recipe")
        if module.jax.default_backend() != ("cpu" if plan["platform"] == "cpu" else "gpu"):
            raise ValueError("The selected execution backend is unavailable")
        record["status"] = "building"
        runtime.save_json(path, record)
        start = time.perf_counter()
        exp = runtime.build_experiment(module, selection, recipe, smoke)
        record["timings_seconds"]["build"] = time.perf_counter() - start
        record["configuration"] = runtime.configuration(module, exp, selection)
        record["density_configuration"] = {name: runtime.density_configuration(exp[name]) for name in ("base", "target")}
        geom = exp["manifold"]
        record["geometry_configuration"] = {"class": type(geom).__name__, "D": geom.D,
            **{name: getattr(geom, name) for name in ("alpha", "jitter") if hasattr(geom, name)}}
        record["evaluation_config"] = runtime.evaluation_recipe(selection, recipe, smoke)
        if selection["method"] == "ours":
            record["landmark_configuration"] = {"method": selection["landmark_method"],
                "count": int(exp["psi"].phi.landmarks.shape[0]),
                "fps_candidates_per_density": module.FPS_CANDIDATES if selection["landmark_method"] == "fps" else None}
        params = exp["state"].params if selection["method"] == "ours" else exp["params"]
        record["parameter_count"] = sum(int(value.size) for value in module.jax.tree_util.tree_leaves(params))
        record["parameter_dtypes"] = sorted({str(value.dtype) for value in module.jax.tree_util.tree_leaves(params)})
        record["status"] = "training"
        runtime.save_json(path, record)
        print("PAPER_CONFIGURATION", json.dumps(runtime.json_safe(record["configuration"])), flush=True)
        start = time.perf_counter()
        exp = runtime.train(module, exp, selection)
        record["timings_seconds"]["training_native_return"] = time.perf_counter() - start
        params = exp["state"].params if selection["method"] == "ours" else exp["params"]
        module.jax.block_until_ready(params)
        record["timings_seconds"]["training_synchronized"] = time.perf_counter() - start
        record["parameter_nonfinite_count"] = sum(int((~module.np.isfinite(module.np.asarray(value))).sum())
            for value in module.jax.tree_util.tree_leaves(params))
        record["status"] = "trained"
        runtime.save_json(path, record)
        record["checkpoint"] = runtime.checkpoint(module, exp, selection["method"], output / "checkpoint.msgpack")
        record["status"] = "evaluating"
        runtime.save_json(path, record)
        start = time.perf_counter()
        for batch in runtime.evaluation_batches(module, exp, selection, record["evaluation_config"]):
            batch["evaluation_seconds"] = time.perf_counter() - start
            batch["metric_nonfinite"] = {name: not math.isfinite(value) for name, value in batch["metrics"].items()}
            record["evaluation_batches"].append(batch)
            runtime.save_json(path, record)
            print("PAPER_EVAL_BATCH", json.dumps(runtime.json_safe(batch)), flush=True)
            start = time.perf_counter()
        record["summary"] = runtime.summarize(record["evaluation_batches"], module.np)
        record["summary_note"] = "SE is across evaluation batches of one fitted model. Tables require aggregation across the declared training seeds. Smoke runs are excluded."
        record["outcome"] = "nonfinite_output" if any(any(b["metric_nonfinite"].values()) for b in record["evaluation_batches"]) else "finite_output"
        record["source_unchanged_after_run"] = runtime.source_manifest(root) == source
        if not record["source_unchanged_after_run"]:
            raise ValueError("Native source changed during execution")
        record.update(status="complete", finished_at=runtime.utc_now())
        runtime.save_json(path, record)
        print("PAPER_COMPLETE", str(path), flush=True)
        return 0
    except BaseException as exc:
        record.update(failed_during=record["status"], status="failed", finished_at=runtime.utc_now(),
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        runtime.save_json(path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
