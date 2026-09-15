# Public runner implementation

Staged under `stage/paper/`; no active source, archive, verification code, or remote job was changed. SHA256 declarations are in `runner_staging_manifest.json`.

## Commands

```sh
python -m paper.run tables --gpus 0 1 --output runs/tables --dry-run
python -m paper.run tables --gpus 0 --output runs/smoke-s2 --smoke --manifolds S2 --methods ours --landmarks random --seeds 12345
python -m paper.run liegroups --gpus 0 --output runs/se3-native --manifolds SE3 --methods ours
python -m paper.run ablations --gpus 0 1 --output runs/ablations
```

`--output` must be fresh. Dry runs import only the standard library and create no files. Explicit GPU IDs are required; one process runs per selected GPU. A failed job prevents remaining jobs from starting, while already active jobs finish; all declarations and statuses remain recorded. SIGINT/SIGTERM stop scheduling and let active work finish. CLI filters select from the full declared recipe; an empty selection fails. CPU execution is permitted only with `--smoke`. The pinned Python 3.11/JAX 0.4.35/jaxlib 0.4.34/Flax 0.8.4/Optax 0.2.3 core stack is checked before execution. Full resolved package versions are recorded.

## Protocols

- Tables: 30 distinct tasks, two manifolds, five prescribed training seeds, FPS/random RNOT and gamma1 RCPM. RNOT uses exactly the documented recovered overrides. Each fitted model uses five 1024-sample evaluations reset to training seed+1000, then native split2/evaluator. RCPM is deduplicated across landmark-method invocations.
- Lie groups: 14 tasks, native FPS RNOT and six native RCPM gammas per group. No historical RNOT overrides. One training seed12345 and five native split2 evaluations reset to12345. Native SE3 residual metric is retained; no diagnostic solve is added.
- Sweep/highd: 126/12 tasks, exact supplied native builders and native fixed-seed evaluation, explicitly labeled unverified. Supplied precision is x64=False; high-dimensional model scaling remains inside its native builder.
- Ablations: 84 tasks (21 historical cases on each S2/T2/S10/T10), historical baseline injected before the historical ablation delta. Explicit effective line-search=False prevents ZIP-default drift. The historical driver is byte-identical to dbf533d. Evaluation preserves its split3 source/target schedule and existing native loss computation. Fresh ablations using the current library are labeled unverified.

Smoke overrides are applied last, after full native/recipe initialization and ablation deltas: two outer updates, batch8, two inner steps with minimum2, and one eight-sample evaluation. Optimizer and model initialization choices remain those of the selected recipe. Smoke is explicitly excluded from reproduction claims.

## Recording and restoration

Each `results/<job_id>/result.json` stores `record_type=paper_experiment`, selection, actual configuration, density/geometry/landmarks metadata, recipe, source and tooling hashes, precision/PRNG/backend/package versions, parameter count/dtypes/nonfinite count, synchronized training time, checkpoint hash, raw per-batch native metrics/keys, population and sample SE, and failure/nonfinite outcomes. Checkpoint is adjacent; logs and manifest/status remain at the run root. There is no pooling of runs inside the worker. Batch SE is not training-seed uncertainty.

`paper._runtime.restore_experiment(module, selection, recipe, payload, saved_configuration, smoke=False)` restores a native fitted model for evaluation/transport only. It validates configuration and array paths/shapes/dtypes/bytes, restores saved landmarks and parameters, checks duplicated checkpoint state parameters, rebuilds the potential-dependent solver/loss/trainer closures, and restores the saved key. It does not restore optimizer state for training continuation.

## Validation

All 12 standard-library tests pass: complete grids/deduplication, SE3 and smoke filtering, invalid GPU/CPU/recipe guards, fresh-output enforcement, no-side-effect dry run, queue failure handling, archive exclusion, fixed evaluation seeds, last-applied smoke overrides, native split2 SE3 metrics, and historical ablation split3/native-loss dispatch. All implementation modules parse. No new training or native GPU evaluation was performed by this implementation agent; root coordinates the actual smoke and SE3 parity checks.
