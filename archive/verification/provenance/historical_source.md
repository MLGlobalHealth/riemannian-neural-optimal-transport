# Exact historical source candidate

Candidate: **`dbf533d86b7b259b42282f0d55c95915c181add6`**, committed 2026-01-27.

Prepared without editing or executing historical code:

- `../snapshots/dbf533d/`: exact selected Git source files, ready to use as
  `verify_run.py --source-root`.
- The selected source files were produced by `git archive` and checked against Git.
- `historical_snapshot_manifest.json`: revision, archive checksum, all file
  checksums, and ZIP comparison.
- The source comparison is recorded in the snapshot manifest.

The snapshot contains **24 files**, including **20 Python files**: all historical
core `src/*.py`, the four table/sweep/highD/ablation drivers, required RCPM
library modules and license/requirements, and original root environment files.
Every selected file was verified byte-for-byte against `git show REV:PATH`.
Every Python file passes AST parsing. Images, notebooks, data, historical outputs,
and unused RCPM application entry points were omitted. No source was rewritten.

## Why this revision

The numerical evidence is documented in `historical_results.md`: all 79
numeric Table 3 cells and multiple table/figure values match committed results.

`a7327bc` commits the historical logs, but its default RNOT outer steps are
**50**, whereas those logs explicitly record **1000**. `dbf533d` restores 1000
while retaining the logged 128 landmarks, 2500 inner steps, Adam LR 0.05,
minimum 1000 inner steps, warm start with gamma 0.1, and explicit x64 in drivers.
It also has the logged table RCPM gamma 1.0 and 5000 steps. Selecting it avoids
silently correcting the older commit's configuration.

Between the log commit `a7327bc` and `dbf533d`, common table execution components
are unchanged: `Manifold`, `Sphere`, `Product`, safe sphere helpers, `Density`,
`SphereUniform`, `ProductUniformComponents`, `WrappedNormal`, `get_uniform`,
`SemiDualLoss`, and utility functions have **identical ASTs**. The other changes
are cleanup of unused Euclidean/data code, comments, an added notebook, and the
outer-step restoration. Drivers, solvers, metrics, networks, embeddings, trainers,
and RCPM sources did not change between those revisions.

This makes `dbf533d` a strong **provenance-based source candidate**, not a proven
execution commit: logs do not store a source hash, and their commit already had
different defaults. No candidate was chosen by comparing fresh run outcomes.

## Quantitative difference from the ZIP

Of the union of selected candidate files and ZIP files:

- 7 files are byte-identical.
- 14 shared files changed, totaling **2307 added / 439 removed lines** from
  historical source to ZIP.
- Candidate-only: `experiments/run_ablations.py`, `environment.yml`, and root
  `requirements.txt`.
- ZIP-only: `src/barycenter.py`.

The reproduction-relevant differences include changed model/training/solver
defaults, deletion of drivers' explicit x64, a different landmark-distance
evaluation path (nested `vmap` historically; batched distances in ZIP), new line
search taking precedence over Adam, and changed returned residual reporting.
Many other added lines support additional manifolds and entropic OT unused by
the main historical table.

Therefore a ZIP run with historical settings is a useful separate experiment,
but it is **not identical source execution** to this snapshot.

## Reproduction command

Use the existing verification runner against the untouched snapshot with a
separate label and output directory, without JSON overrides:

```sh
python -u verification/verify_run.py \
  --source-root verification/snapshots/dbf533d \
  --output-dir results/historical_source \
  --suite table --manifold sphere --method ours --seed 12345 \
  --landmark-method random --label historical_source_dbf533d --gpu 0
```

The historical driver itself enables x64. Native default construction uses the
historical configuration and RNG sequence; the runner records the effective
source hash and environment. Repeat FPS/T2/other seeds only under this separate
historical-source identity. Historical dependencies record Python 3.10,
JAX/JAXlib 0.4.28, Flax 0.8.4, and Optax 0.2.3, but installed versions of the
original ROCm runs are not proven by the logs. A modern NVIDIA run must report
its actual environment and retain that distinction.

The candidate preparation records provenance and execution with the original
source preserved. Later [GPU setup documentation](amd_environment.md) identifies
JAX/ROCm packages 0.4.35; the generic dependency files above do not establish the
installed versions of the historical runs.
