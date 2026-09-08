# Riemannian Neural Optimal Transport

JAX/Flax code for learning transport maps on Riemannian manifolds, with RCPM comparison experiments.

The `finalMLGH` branch starts with the exact 22 files from `riemannian-neural-ot-library.zip`; its experiment and library files remain unchanged. [The archive manifest](verification/provenance/archive_manifest.json) records their hashes. Reproduction results for [the paper](https://openreview.net/pdf?id=ez4oLq7PR3), including differences and missing experiment coverage, are in [verification/REPORT.md](verification/REPORT.md).

The ZIP defaults differ from the historical experiment settings. The reproduction command below applies the settings recovered from historical logs and uses a separately pinned software environment. The original algorithms and metric functions are retained.

## Install the tested NVIDIA environment

The restored experiments use Linux, Python 3.11.15, NVIDIA CUDA 12, JAX 0.4.35, jaxlib 0.4.34, Flax 0.8.4, and Optax 0.2.3. The complete [80-package freeze](verification/environments/reproduction-cuda.txt) was installed in fresh environments on both `nvidia7` and `nvidia6`; both passed `pip check`.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r verification/environments/reproduction-cuda.txt
python -m pip check
export CUDA_ROOT="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
```

`CUDA_ROOT` points this older JAX release to the pip-installed CUDA toolkit. The jaxlib version follows JAX 0.4.35's official CUDA packaging. This is a supported NVIDIA counterpart to the recovered historical software candidate, not the original AMD/ROCm environment. [Environment provenance](verification/provenance/amd_environment.md) describes what was recovered and what remains unknown.

The recorded execution environment also matters: matched package versions produced different torus RNOT ESS values when evaluating an identical saved model on RTX 3090 and RTX 6000 Ada hosts. [The report](verification/REPORT.md) keeps these environments and their results separate.

## Run all six available main-table rows

Choose GPU IDs available for your use; one experiment runs at a time on each selected GPU. The output directory must be new. Preview the commands without starting work:

```sh
python verification/run_table_suite.py \
  --source-root . --output-dir runs/restored_table --gpus 0 --dry-run
```

Run the full suite:

```sh
python verification/run_table_suite.py \
  --source-root . --output-dir runs/restored_table --gpus 0
```

This launches 30 full experiments: five prescribed training seeds for S²/T² RNOT with FPS/random landmarks and S²/T² RCPM with γ=1. It enables x64 and the historical Threefry PRNG behavior, applies [historical_settings.json](verification/historical_settings.json) only to RNOT, and keeps per-run configurations, source hashes, package versions, checkpoints, logs, and all evaluation batches. For multiple available GPUs, use e.g. `--gpus 0 1`. A failure stops new launches while active experiments finish; existing outputs are preserved.

Generate the comparison:

```sh
python verification/compare_results.py \
  --results-root runs/restored_table/results \
  --paper-targets verification/paper_targets.json \
  --output-dir runs/restored_table/comparison --relative-paths
```

The comparator keeps different source snapshots, environments, and configurations separate and reports incomplete seed sets. Its rounding comparisons are descriptive; the paper's rounded uncertainties do not define an equivalence test. GPU timings are recorded without expecting NVIDIA and AMD to match.

## Original experiment entry points

These retain the ZIP's supplied defaults, which differ from the restored main-table configuration above:

```sh
python experiments/run_table.py --gpu 0 --landmark-method random --output-dir runs/table_random
python experiments/run_table.py --gpu 0 --landmark-method fps --output-dir runs/table_fps
python experiments/run_experiments.py sphere --gpu 0 --output-dir runs/sphere
python experiments/run_experiments.py torus --gpu 0 --output-dir runs/torus
python experiments/run_experiments_highD.py sphere --gpu 0 --output-dir runs/highD_sphere
python experiments/run_experiments_highD.py torus --gpu 0 --output-dir runs/highD_torus
```

`verification/verify_run.py --help` exposes individual runs and short smoke checks. Smoke checks are excluded from paper comparisons. `verification/evaluate_checkpoint.py --help` describes replaying a saved model from its result JSON and checkpoint on another compatible host; it retains the original evaluation function and recorded keys, and labels outputs as evaluation-only. The separate [historical source candidate](verification/snapshots/dbf533d/) includes the recovered ablation driver and is preserved for provenance; it does not replace the ZIP library.

The author supplied the SO(3)/SE(3) runners separately; both are now included unchanged. Their native Table 5 protocol uses one training seed per configuration and has separate [run instructions and evidence](verification/liegroups/README.md). These 14 configurations are outside the 30-run main-table command above. RCNF/Moser runners remain unrecovered. The original RCPM license is included at [rcpm/LICENSE](rcpm/LICENSE).

The verification tooling can be checked without running experiments:

```sh
python -m unittest discover -s verification -p 'test_*.py'
```

## Local project page

A visual research page is available under [website/](website/README.md), with interactive manifold illustrations, original paper figures, and local paper/code downloads. Start it from the repository root:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory website
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). The website is self-contained and has no build step or deployment configuration.
