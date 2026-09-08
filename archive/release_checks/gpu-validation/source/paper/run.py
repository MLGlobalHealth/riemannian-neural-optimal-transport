"""Declare and run paper experiments: python -m paper.run SUITE --gpus 0 --output runs/NAME."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from ._runtime import save_json, source_manifest, utc_now

SUITES = ("tables", "liegroups", "sweep", "highd", "ablations")
CORE = {"jax": "0.4.35", "jaxlib": "0.4.34", "flax": "0.8.4", "optax": "0.2.3"}


def load_recipe(suite, path=None):
    path = Path(path) if path else Path(__file__).with_name("configs") / f"{suite}.json"
    recipe = json.loads(path.read_text())
    if recipe.get("suite") != suite:
        raise ValueError("Recipe suite does not match the requested command")
    if not isinstance(recipe.get("x64"), bool) or not recipe.get("label"):
        raise ValueError("Recipe must declare its precision and interpretation label")
    if not recipe.get("seeds") or not recipe.get("manifolds") or not recipe.get("landmark_methods"):
        raise ValueError("Recipe must declare nonempty seeds, manifolds and landmark methods")
    if suite in ("sweep", "highd") and (any(f not in ("sphere", "torus") for f in recipe["manifolds"])
            or not recipe.get("dimensions") or any(type(d) is not int or d < 2 for d in recipe["dimensions"])):
        raise ValueError("Dimension recipes require sphere/torus families and integer dimensions >=2")
    if recipe.get("evaluation") != {"n_batches": 5, "batch_size": 1024,
            "seed_offset": 1000 if suite == "tables" else None, "seed": None if suite == "tables" else 12345}:
        raise ValueError("Keep the native evaluation protocol; use --smoke for a small validation run")
    if suite != "tables" and recipe.get("seeds") != [12345]:
        raise ValueError("The supplied non-table recipes train only seed12345")
    if recipe.get("rnot_overrides") and suite not in ("tables", "ablations"):
        raise ValueError("The supplied sweep/Lie-group builders do not accept configuration overrides")
    if suite == "ablations" and recipe["rnot_overrides"].get("solver", {}).get("use_line_search") is not False:
        raise ValueError("Historical ablations require effective use_line_search=false")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in recipe.get("seeds", [])):
        raise ValueError("Training seeds must be uint32 integers")
    if any(method not in ("random", "fps") for method in recipe.get("landmark_methods", [])):
        raise ValueError("Unknown landmark method")
    if suite == "liegroups" and recipe.get("landmark_methods") != ["fps"]:
        raise ValueError("The recovered native Lie-group recipe uses FPS")
    return recipe, path.resolve()


def declare_jobs(recipe):
    suite = recipe["suite"]
    jobs = []
    if suite in ("sweep", "highd"):
        manifolds = [("S" if family == "sphere" else "T") + str(dim)
            for family in recipe["manifolds"] for dim in recipe["dimensions"]]
    else:
        manifolds = recipe["manifolds"]
    for manifold in manifolds:
        if manifold in ("SO3", "SE3"):
            dimension = 3 if manifold == "SO3" else 6
        elif len(manifold) > 1 and manifold[0] in "ST" and manifold[1:].isdigit() and int(manifold[1:]) >= 2:
            dimension = int(manifold[1:])
        else:
            raise ValueError(f"Unsupported manifold {manifold}")
        if suite == "tables" and manifold not in ("S2", "T2"):
            raise ValueError("The main table recipe covers S2/T2")
        if suite == "liegroups" and manifold not in ("SO3", "SE3"):
            raise ValueError("Lie-group recipe requires SO3/SE3")
        ablations = recipe["ablations"] if suite == "ablations" else [None]
        for ablation in ablations:
            landmarks = ["fps" if ablation == "landmarks_fps" else "random"] if suite == "ablations" else recipe["landmark_methods"]
            for landmark in landmarks:
                for seed in recipe["seeds"]:
                    selection = {"suite": suite, "manifold": manifold, "dimension": dimension,
                        "method": "ours", "seed": seed, "gamma": None, "landmark_method": landmark, "ablation": ablation}
                    suffix = f"_{ablation}" if ablation else ""
                    jobs.append({"job_id": f"{suite}_{manifold}_ours_{landmark}_seed{seed}{suffix}", "selection": selection})
        if suite not in ("highd", "ablations"):
            for gamma in recipe["gammas"]:
                if gamma not in (1.0, .1, .05, .01, .005, .001) or suite == "tables" and gamma != 1.0:
                    raise ValueError("Gamma is outside the native recipe")
                for seed in recipe["seeds"]:
                    selection = {"suite": suite, "manifold": manifold, "dimension": dimension,
                        "method": "rcpm", "seed": seed, "gamma": gamma, "landmark_method": None, "ablation": None}
                    jobs.append({"job_id": f"{suite}_{manifold}_rcpm_gamma{gamma:g}_seed{seed}", "selection": selection})
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("The recipe declares duplicate jobs")
    if not jobs:
        raise ValueError("The recipe declares no jobs")
    return jobs


def make_plan(suite, output, gpus, source_root=None, config=None, python=sys.executable,
              smoke=False, platform="cuda", job=None, manifolds=None, methods=None, landmarks=None, seeds=None):
    if not gpus or len(set(gpus)) != len(gpus) or any(not str(gpu).isdigit() for gpu in gpus):
        raise ValueError("Choose explicit, unique numeric GPU IDs")
    if platform != "cuda" and not smoke:
        raise ValueError("CPU execution is available only for explicitly labeled smoke checks")
    recipe, recipe_path = load_recipe(suite, config)
    declared = declare_jobs(recipe)
    jobs = []
    for item in declared:
        s = item["selection"]
        if job and item["job_id"] != job:
            continue
        if manifolds and s["manifold"] not in manifolds:
            continue
        if methods and s["method"] not in methods:
            continue
        if landmarks and s["landmark_method"] not in landmarks:
            continue
        if seeds and s["seed"] not in seeds:
            continue
        jobs.append(item)
    if not jobs:
        raise ValueError("No declared native jobs match the requested filters")
    source_root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
    output = Path(output).resolve()
    env = {"JAX_PLATFORMS": platform, "JAX_ENABLE_X64": str(recipe["x64"]),
        "JAX_THREEFRY_PARTITIONABLE": "False", "JAX_DEFAULT_PRNG_IMPL": "threefry2x32",
        "JAX_RANDOM_SEED_OFFSET": "0", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    for item in jobs:
        item["result"] = str(output / "results" / item["job_id"] / "result.json")
        item["log"] = str(output / "logs" / (item["job_id"] + ".log"))
        item["command_without_gpu"] = [str(python), "-u", "-m", "paper.worker", "--manifest", str(output / "manifest.json"), "--job-id", item["job_id"]]
    return {"schema_version": 1, "suite": suite, "source_root": str(source_root), "output_dir": str(output),
        "python": str(python), "gpus": [str(gpu) for gpu in gpus], "recipe": recipe,
        "recipe_file": str(recipe_path), "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "smoke": bool(smoke), "platform": platform, "environment_overrides": env,
        "unfiltered_job_count": len(declared), "jobs": jobs,
        "execution_note": "One native configuration per fresh process; historical in-process compilation/cache order is not recreated.",
        "interpretation": "Tiny validation only; not paper reproduction" if smoke else recipe["label"]}


def inspect_interpreter(python):
    code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':list(sys.version_info[:3]),'packages':{n:m.version(n) for n in ('jax','jaxlib','flax','optax')}}))"
    result = subprocess.run([str(python), "-I", "-c", code], text=True, capture_output=True, timeout=30, check=True)
    data = json.loads(result.stdout)
    if data["python"][:2] != [3, 11] or data["packages"] != CORE:
        raise ValueError(f"Use the pinned Python3.11 environment: found {data}")
    return data


def run_plan(plan, poll_seconds=.2):
    output = Path(plan["output_dir"])
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    root = Path(plan["source_root"])
    if not all((root / part).is_dir() for part in ("src", "experiments", "rcpm", "paper")):
        raise ValueError("Source root must contain paper/, experiments/, src/ and rcpm/")
    interpreter = inspect_interpreter(plan["python"])
    output.mkdir(parents=True)
    (output / "results").mkdir()
    (output / "logs").mkdir()
    manifest = dict(plan, created_at=utc_now(), interpreter=interpreter, source=source_manifest(root))
    manifest["paper_source"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "paper").rglob("*")) if p.is_file() and p.suffix in (".py", ".json")}
    save_json(output / "manifest.json", manifest)
    states = [dict(item, status="pending", gpu=None, pid=None, exit_code=None) for item in plan["jobs"]]
    status = {"status": "running", "started_at": utc_now(), "stop_reason": None, "jobs": states}
    pending, active = deque(range(len(states))), {}
    save_json(output / "status.json", status)

    def stop(signum, frame):
        status["stop_reason"] = status["stop_reason"] or f"signal_{signum}"

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            for gpu, (index, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                states[index].update(status="complete" if code == 0 else "failed", exit_code=code, finished_at=utc_now())
                del active[gpu]
                if code != 0:
                    status["stop_reason"] = status["stop_reason"] or f"job_failed:{states[index]['job_id']}"
            if not status["stop_reason"]:
                for gpu in plan["gpus"]:
                    if gpu in active or not pending:
                        continue
                    index = pending.popleft()
                    entry = states[index]
                    command = entry["command_without_gpu"] + ["--gpu", gpu]
                    env = dict(os.environ, **plan["environment_overrides"], CUDA_VISIBLE_DEVICES=gpu, HIP_VISIBLE_DEVICES=gpu)
                    log = open(entry["log"], "wb")
                    entry.update(gpu=gpu, command=command, started_at=utc_now())
                    try:
                        process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    except OSError as exc:
                        log.write((str(exc) + "\n").encode()); log.close()
                        entry.update(status="failed", error=str(exc))
                        status["stop_reason"] = f"job_start_failed:{entry['job_id']}"
                        break
                    entry.update(status="running", pid=process.pid)
                    active[gpu] = (index, process, log)
            if status["stop_reason"]:
                status["status"] = "waiting_for_active_jobs"
                while pending:
                    states[pending.popleft()]["status"] = "not_started"
            save_json(output / "status.json", status)
            if active:
                time.sleep(poll_seconds)
        failed = any(entry["status"] == "failed" for entry in states)
        status.update(status="failed" if failed else "interrupted" if status["stop_reason"] else "complete", finished_at=utc_now())
        save_json(output / "status.json", status)
        return 0 if status["status"] == "complete" else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="A fresh run directory")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two training updates and one 8-point evaluation; not paper reproduction")
    parser.add_argument("--platform", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--job", help="Select one exact declared job ID")
    parser.add_argument("--manifolds", nargs="+")
    parser.add_argument("--methods", choices=("ours", "rcpm"), nargs="+")
    parser.add_argument("--landmarks", choices=("fps", "random"), nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.suite, args.output, args.gpus, args.source_root, args.config, args.python,
            args.smoke, args.platform, args.job, args.manifolds, args.methods, args.landmarks, args.seeds)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return run_plan(plan)
    except (ValueError, FileExistsError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
