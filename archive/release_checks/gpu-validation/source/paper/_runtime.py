"""Runtime support for native paper experiments; importing this module needs only stdlib."""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
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
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_ROOT", "JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS",
                "JAX_THREEFRY_PARTITIONABLE", "JAX_DEFAULT_PRNG_IMPL", "JAX_RANDOM_SEED_OFFSET", "JAX_DEFAULT_MATMUL_PRECISION",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND")}}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        data["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        data["nvidia_smi"] = str(exc)
    return data

def replace_config(cfg, values):
    known = {f.name for f in dataclasses.fields(cfg)}
    if set(values) - known:
        raise ValueError(f"Unknown {type(cfg).__name__} fields: {set(values) - known}")
    updates = {k: tuple(v) if isinstance(getattr(cfg, k), tuple) else v for k, v in values.items()}
    return dataclasses.replace(cfg, **updates)

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
            "note": "Ours key is the native builder's retained pre-training key; checkpoint supports evaluation, not exact training continuation."
                    if method == "ours" else "Native RCPM training does not return optimizer state."}


CORE_VERSIONS = {"jax": "0.4.35", "jaxlib": "0.4.34", "flax": "0.8.4", "optax": "0.2.3"}


def validate_environment():
    versions = package_versions()
    if sys.version_info[:2] != (3, 11) or any(versions.get(k) != v for k, v in CORE_VERSIONS.items()):
        raise ValueError(f"Use the pinned Python3.11/JAX0.4.35 environment; found {sys.version}, {versions}")
    return versions


def load_native(source_root, suite, manifold):
    root = Path(source_root).resolve()
    if suite == "ablations":
        path = Path(__file__).with_name("_ablation_native.py")
    else:
        name = {"tables": "run_table", "sweep": "run_experiments", "highd": "run_experiments_highD"}.get(suite)
        name = f"run_{manifold.lower()}_experiment" if suite == "liegroups" else name
        path = root / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("paper_native_" + suite, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configured_ours(m, selection, overrides):
    """Preserve the tested native builder's RNG order and library constructor calls."""
    name = selection["manifold"]
    sphere = name.startswith("S")
    cfg = m.ExperimentConfig(manifold_name=name, base_density="SphereUniform" if sphere else "TorusUniform",
        target_density="SphereWrappedNormal" if sphere else "TorusWrappedNormal", jax_platform="gpu")
    cfg.training.seed = selection["seed"]
    for section in ("model", "solver", "training"):
        setattr(cfg, section, replace_config(getattr(cfg, section), overrides.get(section, {})))
    if cfg.training.grad_accum_steps > 1:
        raise ValueError("The native paper builders use ordinary single-batch updates.")
    geometry = m.get_manifold(name)
    base = m.densities.get(geometry, cfg.base_density)
    target = m.densities.get(geometry, cfg.target_density)
    key = m.jax.random.PRNGKey(selection["seed"])
    nb = cfg.model.n_landmarks // 2
    nt = cfg.model.n_landmarks - nb
    if m.LANDMARK_METHOD == "fps":
        lb, key = m.build_landmarks(geometry, base, {"n_landmarks": nb, "method": "fps", "fps_candidates": m.FPS_CANDIDATES}, key)
        lt, key = m.build_landmarks(geometry, target, {"n_landmarks": nt, "method": "fps", "fps_candidates": m.FPS_CANDIDATES}, key)
    else:
        key, k1, k2 = m.jax.random.split(key, 3)
        lb, lt = base.sample(k1, nb), target.sample(k2, nt)
    embedding = m.GromovDistanceEmbedding(manifold=geometry, landmarks=m.jnp.concatenate([lb, lt], axis=0))
    network_cfg = dataclasses.asdict(cfg.model)
    network_cfg.pop("n_landmarks")
    psi = m.build_network(phi=embedding, **network_cfg)
    key, kx, kp = m.jax.random.split(key, 3)
    params = psi.init(kp, base.sample(kx, 16))["params"]
    solver = m.ArgminSolver(manifold=geometry, psi_module=psi, **dataclasses.asdict(cfg.solver))
    loss = m.SemiDualLoss(manifold=geometry, psi_module=psi, solver=solver)
    trainer = m.SemiDualTrainer(manifold=geometry, psi_module=psi, loss_fn=loss, solver=solver,
        **{k: getattr(cfg.training, k) for k in ("learning_rate", "lr_decay", "lr_decay_alpha", "n_steps")})
    return {"cfg": cfg, "manifold": geometry, "base": base, "target": target, "psi": psi,
            "solver": solver, "trainer": trainer, "state": trainer.init_state(params), "key": key}


def build_experiment(module, selection, recipe, smoke=False):
    m, s = module, selection
    suite, method, name = s["suite"], s["method"], s["manifold"]
    if suite != "ablations":
        m.LANDMARK_METHOD = s["landmark_method"] or "fps"
    if method == "rcpm":
        if suite == "liegroups":
            exp = m.build_rcpm_experiment(gamma=s["gamma"], seed=s["seed"])
        else:
            dim = s["dimension"]
            geom = m.rcpm_manifolds.Sphere(D=dim + 1, jitter=1e-2) if name.startswith("S") else m.rcpm_manifolds.Product(
                D=2 * dim, manifolds_str=",".join(["S1"] * dim))
            exp = m.build_rcpm_experiment(geom, is_torus=name.startswith("T"), gamma=s["gamma"], seed=s["seed"])
        exp["_paper_rcpm_training"] = {"n_iters": 2 if smoke else m.N_ITERS,
            "batch_size": 8 if smoke else m.BATCH_SIZE, "lr": 1e-3,
            "log_every": 1 if smoke else m.N_ITERS if suite == "tables" else m.N_ITERS // 10}
        return exp
    if suite == "ablations":
        cfg = m.get_base_config(name)
        cfg = m.apply_overrides(cfg, recipe["rnot_overrides"])
        cfg.training.seed = s["seed"]
        delta = dict(m.ABLATIONS[s["ablation"]])
        landmarks = delta.pop("landmark_method", "random")
        cfg = m.apply_overrides(cfg, delta)
        exp = m.build_experiment(cfg, landmark_method=landmarks)
    elif suite == "liegroups":
        exp = m.build_ours_experiment() if name == "SO3" else m.build_experiment()
    elif suite == "tables":
        exp = _configured_ours(m, s, recipe["rnot_overrides"])
    else:
        base = "SphereUniform" if name.startswith("S") else "TorusUniform"
        target = "SphereWrappedNormal" if name.startswith("S") else "TorusWrappedNormal"
        exp = m.build_experiment(name, base, target, s["dimension"]) if suite == "highd" else m.build_ours_experiment(name, base, target)
    if smoke:
        _make_smoke(m, exp)
    return exp


def _make_smoke(m, exp):
    """Explicit tiny validation run; never included in paper reproduction claims."""
    cfg = exp.get("cfg")
    training = cfg.training if cfg else exp["training_cfg"]
    solver_cfg = cfg.solver if cfg else exp["solver_cfg"]
    training = dataclasses.replace(training, n_steps=2, batch_size=8, log_every=1, eval_every=None)
    solver_cfg = dataclasses.replace(solver_cfg, inner_steps=2, min_steps=2)
    if cfg:
        exp["cfg"] = dataclasses.replace(cfg, training=training, solver=solver_cfg)
    else:
        exp.update(training_cfg=training, solver_cfg=solver_cfg)
    solver = m.ArgminSolver(manifold=exp["manifold"], psi_module=exp["psi"], **dataclasses.asdict(solver_cfg))
    loss = m.SemiDualLoss(manifold=exp["manifold"], psi_module=exp["psi"], solver=solver)
    trainer = m.SemiDualTrainer(manifold=exp["manifold"], psi_module=exp["psi"], loss_fn=loss, solver=solver,
        **{k: getattr(training, k) for k in ("learning_rate", "lr_decay", "lr_decay_alpha", "n_steps")})
    exp.update(solver=solver, trainer=trainer, state=trainer.init_state(exp["state"].params))


def configuration(module, exp, selection):
    if selection["method"] == "rcpm":
        return {"builder": {"gamma": selection["gamma"], "seed": selection["seed"], "n_components": 68, "n_transforms": 5},
            "flow": {"n_transforms": exp["flow"].n_transforms, "single_transform_cfg": dict(exp["flow"].single_transform_cfg)},
            "training": exp["_paper_rcpm_training"]}
    if "cfg" in exp:
        return dataclasses.asdict(exp["cfg"])
    return {name: dataclasses.asdict(exp[name + "_cfg"]) for name in ("model", "solver", "training")}


def density_configuration(density):
    fields = ("loc", "scale", "rot_scale", "trans_scale", "trans_low", "trans_high", "t_range", "include_alpha_volume_correction")
    return {"class": type(density).__name__, **{key: json_safe(getattr(density, key)) for key in fields if hasattr(density, key)}}


def train(module, exp, selection):
    if selection["method"] == "rcpm":
        return module.train_rcpm(exp, **exp["_paper_rcpm_training"])
    func = module.train if selection["suite"] in ("highd", "ablations") or selection["manifold"] == "SE3" else module.train_ours
    return func(exp)


def evaluation_recipe(selection, recipe, smoke=False):
    cfg = dict(recipe["evaluation"])
    cfg["seed"] = selection["seed"] + cfg["seed_offset"] if cfg["seed_offset"] is not None else cfg["seed"]
    cfg["key_protocol"] = "split3_source_target" if selection["suite"] == "ablations" else "split2_native_evaluator"
    if smoke:
        cfg.update(n_batches=1, batch_size=8)
    return cfg


def evaluation_batches(m, exp, selection, cfg):
    """Yield native batches without diagnostic solves or alternate metrics."""
    key = m.jax.random.PRNGKey(cfg["seed"])
    for index in range(cfg["n_batches"]):
        if selection["suite"] == "ablations":
            parent = key
            key, k1, k2 = m.jax.random.split(key, 3)
            xs = exp["base"].sample(k1, cfg["batch_size"])
            targets = exp["target"].sample(k2, cfg["batch_size"])
            ys, _ = exp["solver"].batch_solve(exp["state"].params, xs, targets)
            kl = m.compute_kl_divergence(psi_params=exp["state"].params, xs=xs, ys=ys,
                base_density=exp["base"], target_density=exp["target"], manifold=exp["manifold"], psi_module=exp["psi"])
            # The historical native evaluator also computes its existing training loss.
            loss = exp["trainer"].loss_fn(exp["state"].params, xs, targets)
            metrics = {"kl": kl.kl, "ess_ratio": kl.ess_ratio, "loss": float(loss)}
            keys = {"prng_key": m.np.asarray(parent).tolist(), "source_key": m.np.asarray(k1).tolist(), "target_key": m.np.asarray(k2).tolist()}
        else:
            key, subkey = m.jax.random.split(key)
            if selection["method"] == "rcpm":
                kl, ess, ratio = m.compute_kl_rcpm(exp, subkey, cfg["batch_size"])
                metrics = {"kl": kl, "ess": ess, "ess_ratio": ratio}
            else:
                func = m.compute_kl if selection["suite"] == "highd" else m.evaluate if selection["manifold"] == "SE3" else m.compute_kl_ours
                metrics = func(exp, subkey, batch_size=cfg["batch_size"])
            keys = {"prng_key": m.np.asarray(subkey).tolist()}
        yield {"index": index, **keys, "metrics": {name: float(value) for name, value in metrics.items()}}


def summarize(batches, np):
    result = {}
    for name in batches[0]["metrics"]:
        values = np.asarray([batch["metrics"][name] for batch in batches])
        result[name] = {"mean": float(np.mean(values)), "se_ddof0": float(np.std(values) / np.sqrt(len(values))),
            "se_ddof1": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "raw_batches": values.tolist()}
    return result


def array_fingerprint(value, np):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def restore_experiment(module, selection, recipe, payload, saved_configuration, smoke=False):
    """Build native objects and restore checkpoint arrays for evaluation/transport only."""
    exp = build_experiment(module, selection, recipe, smoke=smoke)
    if json_safe(configuration(module, exp, selection)) != saved_configuration:
        raise ValueError("Checkpoint configuration differs from the selected native recipe")
    initialized = exp["state"].params if selection["method"] == "ours" else exp["params"]
    signature = lambda a: (list(module.np.asarray(a).shape), str(module.np.asarray(a).dtype))
    if module.jax.tree_util.tree_map(signature, initialized) != module.jax.tree_util.tree_map(signature, payload["params"]):
        raise ValueError("Saved parameter paths/shapes/dtypes differ from the native model")
    params = module.jax.tree_util.tree_map(module.jnp.asarray, payload["params"])
    original = module.jax.tree_util.tree_map(lambda a: array_fingerprint(a, module.np), payload["params"])
    restored = module.jax.tree_util.tree_map(lambda a: array_fingerprint(a, module.np), params)
    if original != restored:
        raise ValueError("Restoring checkpoint parameters changed bytes or dtype")
    if selection["method"] == "rcpm":
        exp["params"] = params
    else:
        state_params = module.jax.tree_util.tree_map(lambda a: array_fingerprint(a, module.np), payload["state"]["params"])
        if state_params != original:
            raise ValueError("Checkpoint state parameters disagree with its top-level parameters")
        landmarks = module.jnp.asarray(payload["landmarks"])
        if signature(landmarks) != signature(exp["psi"].phi.landmarks):
            raise ValueError("Saved landmark shapes/dtypes differ from the native model")
        if array_fingerprint(landmarks, module.np) != array_fingerprint(payload["landmarks"], module.np):
            raise ValueError("Restoring checkpoint landmarks changed bytes or dtype")
        psi = exp["psi"].clone(phi=module.GromovDistanceEmbedding(manifold=exp["manifold"], landmarks=landmarks))
        exp["psi"] = psi
        exp["state"] = exp["state"].replace(params=params)
        solver_cfg = exp["cfg"].solver if "cfg" in exp else exp["solver_cfg"]
        solver = module.ArgminSolver(manifold=exp["manifold"], psi_module=psi, **dataclasses.asdict(solver_cfg))
        loss = module.SemiDualLoss(manifold=exp["manifold"], psi_module=psi, solver=solver)
        training = exp["cfg"].training if "cfg" in exp else exp["training_cfg"]
        trainer = module.SemiDualTrainer(manifold=exp["manifold"], psi_module=psi, loss_fn=loss, solver=solver,
            **{k: getattr(training, k) for k in ("learning_rate", "lr_decay", "lr_decay_alpha", "n_steps")})
        exp.update(solver=solver, trainer=trainer)
    exp["key"] = module.jnp.asarray(payload["key"])
    if array_fingerprint(exp["key"], module.np) != array_fingerprint(payload["key"], module.np):
        raise ValueError("Restoring checkpoint key changed bytes or dtype")
    return exp
