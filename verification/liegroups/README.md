# Recovered SO(3) and SE(3) experiments

The two Lie-group runners were recovered from the author's Downloads folder on 8 September 2026 and added unchanged in commit `3ae0f9d07d22fdc9d1c0a4ae84c5a77be1a273fd`. They were absent from the original 22-file ZIP. The original library remains unchanged; [the recovered-source manifest](../provenance/recovered_liegroup_source_manifest.json) records the resulting 24 source files and both supplied runner hashes.

These runners cover the SO(3)/SE(3) RNOT and six-gamma RCPM configurations associated with Table 5. All 14 native configurations completed on RTX 3090: nine finite and five nonfinite. The five SE(3) RCPM configurations below gamma 1 reproduce the reported instability; the nine numeric KL means do not match the paper at its printed precision. RCNF/Moser remain outside this recovered experiment coverage. The [Table 5 comparison](comparison/comparison.md) retains each recorded outcome and the differences from the paper. The [reproduction report](../REPORT.md) explains the remaining coverage.

Use [the root README](../../README.md)'s tested NVIDIA installation instructions and [pinned environment](../environments/reproduction-cuda.txt): Python 3.11.15, JAX 0.4.35, jaxlib 0.4.34, Flax 0.8.4, and Optax 0.2.3. This is the supported NVIDIA software candidate, not a reconstruction of the original AMD/ROCm runtime. Keep its `CUDA_ROOT` setting. The recovered runners enable x64 themselves; the recording wrapper requires JAX's 0.4.35 default non-partitionable Threefry behavior.

Run commands from the repository root with the environment activated. Choose a GPU available for your use. The recorded jobs also use these thread and backend settings:

```sh
export JAX_PLATFORMS=cuda MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
```

## Original recovered entry points

Each command trains one FPS RNOT model and then six RCPM models, using the recovered native settings:

```sh
python experiments/run_so3_experiment.py --gpu 0 --output-dir runs/native_so3
python experiments/run_se3_experiment.py --gpu 0 --output-dir runs/native_se3
```

Use `--ours-only` or `--rcpm-only` to select one method; do not combine them. `--landmark-method random` is available for RNOT, while the recovered default is FPS. The initial Table 5 verification uses that FPS default. The original CLI has no training-seed or gamma selector. Its fixed NPZ filenames overwrite prior outputs, so choose a fresh directory for each invocation.

## Record the full native protocol

The recording wrapper calls the unchanged native builders, trainers, and evaluators. It preserves all five evaluation batches and their keys, configurations, source/package hashes, timings, nonfinite flags, and a trained checkpoint. Each output directory must be new. This serial example records all 14 model/configuration trials on one available GPU:

```sh
for manifold in SO3 SE3; do
  python verification/verify_liegroup_run.py \
    --source-root . --gpu 0 --manifold "$manifold" --method ours \
    --output-dir "runs/native_liegroup/${manifold}_ours"

  for gamma in 1.0 0.1 0.05 0.01 0.005 0.001; do
    python verification/verify_liegroup_run.py \
      --source-root . --gpu 0 --manifold "$manifold" --method rcpm --gamma "$gamma" \
      --output-dir "runs/native_liegroup/${manifold}_rcpm_gamma${gamma}"
  done
done
```

The wrapper uses one gamma per fresh process; the original six-gamma main's compilation/cache history is therefore different. It records that distinction. Keep failed or nonfinite records, including the SE(3) RCPM configurations described as unstable in the paper. A failed invocation writes its phase and traceback; it is not evidence of successful reproduction.

## Native settings and uncertainty

| Setting | SO(3) RNOT | SE(3) RNOT |
|---|---|---|
| Network | MLP 128/128, SiLU, LayerNorm | Same |
| Landmarks | 256, FPS | 128, FPS |
| FPS candidates | 4096 per source/target distribution | Same |
| Outer updates / batch | 500 / 256 | 200 / 256 |
| Outer learning rate | .001, no schedule | .001, cosine schedule to .05 of initial rate |
| Inner steps / minimum | 500 / 50 | 100 / 500 |
| Inner learning rate / tolerance | .005 / 1e-6 | Same |
| Inner update | Line search enabled, Adam disabled | Same |
| Log-sum-exp initialization | Disabled | Enabled, gamma .01 |

SO(3) inherits the unchanged ZIP's `ExperimentConfig` defaults. SE(3) supplies its own configuration, including the minimum-steps value shown above; the recovered recipe is retained exactly. Neither uses `verification/historical_settings.json`, which belongs to the earlier S²/T² main-table restoration. RCPM uses 68 components, five transforms, 5000 updates, batch 256, and Adam learning rate .001 for every gamma.

Each model uses training seed 12345 and is evaluated on five batches of 1024, beginning from evaluation seed 12345 with no offset. The reported KL uncertainty is the population standard deviation of those five batch estimates divided by sqrt(5). These are five evaluations of one fitted model, unlike the five independently trained seeds per row in the S²/T² main-table suite. Table 5 does not supply the exact training settings or landmark strategy needed to resolve differences independently of the recovered scripts.

The actual SE(3) source is uniform over translations `[-4,4]^3`; its target is `SE3FactorizedCompact` centered at a 60-degree z rotation and translation `[1,.5,-.5]`, with rotation scales .3 and translation scales .5. The runner's older console banner says `[-2,2]` and `SE3WrappedNormal`; the constructor values and recorded density configuration are authoritative. SE(3)'s native mean solver residual is retained as an additional output.

## Compare the recorded results

Run from the repository root:

```sh
python verification/compare_liegroup_results.py \
  --results-root runs/native_liegroup \
  --paper-targets verification/paper_targets.json \
  --output-dir runs/native_liegroup/comparison
```

To regenerate the committed comparison, use `--results-root verification/liegroups/results` and `--output-dir verification/liegroups/comparison`. The comparator retains incomplete, failed, duplicate, and nonfinite records, and separates source/configuration/environment variants. It compares the nine numeric Table 5 rows and reports the actual outcomes for all five SE(3) RCPM configurations described as numerically unstable. Import or build failures alone do not reproduce that claim.

The [run manifest](provenance/run_manifest.json) records hashes of unchanged results, checkpoints and full logs. The committed result JSONs are under `results/`; checkpoints and logs are retained in ignored `verification/artifacts/liegroups/` and in the isolated server run directory. The eight [fixed-checkpoint evaluations](provenance/paired_replay_comparison.md) are separate evaluation-only evidence and do not add training repetitions. Both NVIDIA hosts reproduce the four saved models' metrics to floating-point precision; this does not reconstruct the AMD environment.

Adding the two recovered runners creates a new 24-file source fingerprint. Earlier S²/T² saved records still identify the original 22-file snapshot, and their checkpoint replay guards require that exact source tree. Fresh runs from this branch record the new fingerprint; earlier raw evidence remains unchanged.

## Replay a saved Lie-group checkpoint

Use the same pinned environment and its `CUDA_ROOT`. For a new run, the checkpoint is adjacent to its result JSON:

```sh
python verification/evaluate_liegroup_checkpoint.py \
  --source-root . --gpu 0 \
  --input-result runs/native_liegroup/SE3_ours/result.json \
  --output runs/replays/SE3_ours.json
```

This evaluator supports the four models used in the paired check: SO(3)/SE(3) RNOT and RCPM gamma 1. It restores the saved parameters and, for RNOT, landmarks and solver references, then calls the unchanged native metric function using each recorded key. The output file must be new. A copy of the same source tree, result JSON, and checkpoint can be replayed on another compatible host; hashes, all 80 package pins, array dtypes/bytes, and the native precision/PRNG settings are checked. It never trains or changes the saved model.
