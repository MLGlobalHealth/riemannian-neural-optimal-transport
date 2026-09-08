# Rebuttal code location audit — 2026-09-08

The requested RCNF/Moser implementations and the camera-ready SO(3)/SE(3)
KL/ESS experiment runners were **not located in the inspected branches,
histories, or bounded server code/output searches**. The servers do contain
SO(3)/SE(3) rebuttal code for a separate entropic-RNOT comparison, with existing
W1/W2 results. Those files provide useful related code, but do not establish
reproduction of the requested paper rows.

This was a code-location/provenance audit. No GPU jobs were launched, no remote
files or Git branches were modified, and no mathematical review was performed.
The parent agent refreshed the local origin refs; this audit used read-only
Git commands against those refs. Local audit artifacts were written only under
`/private/tmp/mlgh-verification`.

## Git branches and history

After the parent's fetch, the relevant refs are:

| Ref | Commit |
|---|---|
| `main` | `811406518ad3787b378eefd3fe2c1fa39aa135c3` |
| `finalMLGH` | `ca8eb58a7f20b7d0c22912892588335d48700ba4` |
| `origin/main`, `origin/HEAD` | `9efac207305adc84f4d07526a5891d0207b28c34` |

Upstream returned only `main`; no rebuttal branch refs were available. All 80
reachable commits were inspected through 435 unique text/code/notebook-source
blobs, including reachable stash history. Searches for RCNF, Moser, rebuttal,
continuous-normalizing-flow, neural-ODE, `odeint` and `diffeqsolve` found no
matches. Notebook source cells were parsed separately from embedded images.
The machine-readable scan is `rebuttal_history_scan.json`.

Related, recoverable historical files:

| Ref / commit | File | Relevance |
|---|---|---|
| `origin/main` / `9efac207305adc84f4d07526a5891d0207b28c34` | `run_pdbbind_benchmark.py` | SE(3) entropic ligand-pose experiment; RMSD metrics and external PDBbind data. |
| Same | `se3_entropic.ipynb` | PDBbind 6AYO docking demo. |
| Same | `se3_affinity_ranking.ipynb` | Affinity-ranking/application experiments, including Vina rescoring. |
| `origin/main` history / `52e65540b38fbceba967418f2583dd68e5749b6a` | `so3_figure1.ipynb` | Synthetic rotation visualization; seed 42, 100 target landmarks, 1000 training steps, target scale 0.25. No recovered RCNF/Moser table runner. |

Exact Git blob IDs and SHA256 values for these files are recorded in
`rebuttal_branch_provenance.json`. The historical SO(3) notebook has Git blob
`314024f1db9b5f8385ccae6bb8554a57d3692edc` and SHA256
`7250f695fc0f7053fcc2108f89e5b404e5c1698636ebe2d54e83f53dd8f52bc3`.

The current upstream requirements still specify JAX/jaxlib 0.4.28, Flax 0.8.4
and Optax 0.2.3. The PDBbind runner additionally imports Biopython, absent from
that requirements file. The affinity notebook also imports RDKit, Vina, Meeko
and OpenBabel. Its data paths include `data/refined-set/`, PDBbind index files,
and ligand/pocket structures. The batch runner expects
`data/PDBbind_v2020_refined.tar.gz`. These application inputs are not part of the
supplied ZIP and were not downloaded by this audit.

## Server repositories and bounded output search

SSH aliases used were `nvidia6`, `nvidia7`, and `dide` (the configured alias for
`wpia-didelx01.dide.ic.ac.uk`; `dide1` itself does not resolve).

Code-directory name searches and discovery of `.git` directories/worktree
files were bounded to depth four, excluding environments, caches and private
configuration directories. Git remote URLs were inspected to detect renamed
clones of `alemicheli/riemannian-neural-ot`.

| Server | Git locations inspected | Matching clone / result |
|---|---:|---|
| nvidia6 | 2 | No matching Riemannian-OT clone. |
| nvidia7 | 6 | No matching Git clone; related source-only deployment below. |
| dide | 12 | `/home/am1118/projects/2026/riemannian-neural-ot` |

The dide clone has only `main`, `origin/main` and `origin/HEAD`, all at
`044b0d5fa7158b9b96e75a84f2add07c3cb2ff2e` (2026-01-05). Its all-ref listing,
recent reflog and worktree listing show no newer/rebuttal branch or alternate
worktree. The only untracked run artifacts are two sphere configuration YAMLs;
the other untracked files are bytecode caches. Its commit is already included
in the local history scan.

An additional bounded search covered experiment/result files and notebook
source plus plain-text outputs under user-owned code/research directories,
excluding datasets/private documents: 99 candidate files on nvidia6, 20 on
nvidia7, and 1617 on dide, including 11 dide notebooks. No relevant RCNF/Moser
output was found. One Moser string occurred in an unrelated question-answer
corpus output and is not baseline evidence. All SO(3)/SE(3) matches were in the
entropic deployment described next. Search records are
`rebuttal_output_scan_{nvidia6,nvidia7,dide}.json`.

The specifically requested source-worktree locations `~/.codex/worktrees` and
`~/.claude/worktrees` do not exist on any of the three servers. A further exact
filename/directory search for `rcnf`, `moser` and `rebut` under the identified
research/code roots (depth eight, excluding environments/datasets) found only
the already inspected entropic `rebuttals_experiments` directories. Records are
`rebuttal_worktree_scan_{nvidia6,nvidia7,dide}.json`.

A final explicitly authorized provenance check filtered standard Bash, Zsh and
Fish shell-history files on each server for RCNF/Moser/rebuttal or SO3/SE3
experiment references. The filtering ran on the server and emitted only
candidate Python/notebook/YAML paths and relevant `cd` directories; no history
commands, environment values or unrelated entries were retained or executed.
It returned no candidate paths or directories on any server. The empty records
are `rebuttal_history_paths_{nvidia6,nvidia7,dide}.json`. No further search was
performed from this route.

A later bounded source-bundle check also found no matching `.zip`, `.tar`,
`.tar.gz` or `.tgz` archive names under the identified code/project roots on
any server. Patterns, roots and exclusions are documented separately in
[the archive search report](rebuttal_archive_search.md); no archive was extracted or executed.

These bounds do not prove that files are absent from deeper directories,
unmounted storage, or differently named locations outside the inspected code
roots. A more specific path or original launch command would resolve that gap.

## Related source-only rebuttal deployments

Found at:

- nvidia7: `/data/am1118/entropic-riemannian-neural-ot`
- dide: `/home/am1118/entropic-riemannian-neural-ot`

Neither directory contains Git metadata, so no branch or commit attribution is
available. They are distinct snapshots: the comparison runner and density file
differ between servers. Their source/result hashes are retained separately in
`rebuttal_branch_provenance.json`; do not combine the two snapshots implicitly.

| Server | File | SHA256 |
|---|---|---|
| nvidia7 | `rebuttals_experiments/compare_chnot_rnot_manifolds.py` | `9dd6e3ad46437f7c5baf5c949a12d3080c21c65b3d108c9d1d63076fa72d48d8` |
| dide | Same path | `74e164cb788a49ff9ffa3dbb45ffdb716cc885cbfda1ed6b8d64b6b2175e2f19` |
| Both | `experiments/run_benchmark_table.py` | `c6cc1dfb5dd5c56b4ff08f9ec4e26e2725d5540d9423f54cef5d602901bdd04f` |
| Both | `src/baselines.py` | `fca2e267948c9e92f7d4bbfa6cbfafd2cb088caca936c94456dfe1dbbb422001` |

The comparison runs Entropic RNOT and non-entropic RNOT on S2/SO3/SE3, reporting
W1, W2, transport cost and execution-cost metrics. `src/baselines.py` implements
Sinkhorn-based baselines, not RCNF or Moser. Searching the related deployment's
code and result text also found no RCNF/Moser or ODE implementation references.

The SO3 target uses a tangent shift 2.5 and scale 0.8, with a wrapped-normal
source. SE3 uses a bounded Haar/uniform source with translation range 4 and a
factorized compact target centered at quaternion
`[0.8660254037844387, 0, 0, 0.5]` and translation `[1, 0.5, -0.5]`, with rotation
scale 0.3, translation scale 0.5 and `alpha=2`. These synthetic runners generate
their own distributions and require no external dataset.

Native defaults include five seeds from 42, 200 support points, batch size 256,
3000 entropic steps, 200 RNOT steps, 1000 inner steps and minimum 500. Heat time
defaults to epsilon in the comparison runner, while the separate benchmark
runner uses 100 times epsilon. Those are different experiment configurations.

Existing nvidia7 pinned-result files include:

- `rebuttals_experiments/results_pinned/compare_chnot_rnot_SO3.json`, SHA256
  `95e633ecd0bc7596b38f8345d0c89e74207a69df19cde14369ec6344d6ef31fb`.
  Stored W1 means are 0.406568 for CHNOT and 0.462983 for RNOT.
- `rebuttals_experiments/results_pinned/compare_chnot_rnot_SE3.json`, SHA256
  `e59adf8eebf58d61ba80fade32f2e51058f5f01d9ea86831939c6c500f7f3afd`.

The dide `results_dide/` version differs; for example its SO3 RNOT W1 mean is
0.470567. These are pre-existing results, not results rerun during this audit,
and they contain no requested camera-ready KL/ESS comparison.

## Candidate rerun commands and the remaining gap

Both hosts have `~/venvs/ernot-pinned/bin/python`. Package metadata confirms
Python 3.11.15, JAX/jaxlib 0.6.2, Flax 0.10.6, Optax 0.2.4, NumPy 2.4.6,
SciPy 1.17.1, ott-jax 0.6.0 and POT 0.9.7.post1. The comparison imports both
`ot` (POT) and `ott`; its README's older anaconda environment path is not needed.

The following is a source-supported command for rerunning the **related
entropic comparison**, after copying one complete version into an isolated
directory and selecting an available GPU. It was not executed in this audit:

```sh
cd /path/to/isolated/entropic-riemannian-neural-ot
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ~/venvs/ernot-pinned/bin/python -u \
  rebuttals_experiments/compare_chnot_rnot_manifolds.py \
  --manifolds SO3 SE3 --gpu GPU_ID \
  --output-dir /path/to/new/results
```

Add `--quick` for the script's explicitly reduced diagnostic run. Keep the
matching `src/` and `experiments/run_benchmark_table.py` with the runner; it
imports entropic helpers not supplied in the original ZIP.

No faithful reproduction command can yet be supplied for the missing RCNF,
Moser or camera-ready SO3/SE3 KL/ESS rows: their actual runner/configuration
has not been found. The concrete remaining requirement is its original source
location or launch command, including method settings and seeds. This does not
prevent running the ZIP's available main tables or the separately recovered
historical configurations.
