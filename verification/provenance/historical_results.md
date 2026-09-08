# Historical experiment provenance audit — 2026-09-08

**These are previously stored results, not new reproductions.** This read-only
audit compared the original repository at `8114065` with
`verification/paper_targets.json` and the supplied ZIP. Neither
worktree nor historical code was modified or executed.

All historical evidence paths below refer to the original repository at the stated revisions, not to the ZIP import.

## Strong numerical provenance

- **Table 3: every one of the 79 numeric paper cells matches historical JSON at
  the paper's displayed rounding.** Checked all 20 paper ablations across
  `experiments/final_results/ablation_{S2,T2,S10,T10}.json`: three decimals for
  S2/T2/S10, two for T10. Baseline raw KL values are respectively
  `0.04025391775865537`, `0.12526493671553487`, `0.0476635847732733`, and
  `1.0721334077897062`. FPS raw KL values are `0.018638380474244827`,
  `0.08265259590489349`, `0.03452566291407259`, and `0.029350181451723946`.
  The paper's one blank cell is excluded from this 79-cell comparison.
- These ablation JSON files entered Git in `5fe05b8` on 2026-01-25. Their logs
  entered in `a7327bc` on 2026-01-26, alongside table and dimension-sweep logs.
  The four JSON files and both table logs byte-match their committed `a7327bc`
  versions; they are not later local replacements.
- **Table 1's S2 RCPM row matches `table_fps.log:10243`:** KL
  `0.0037 ± 0.0008`, ESS `0.996 ± 0.000`, runtime `37.3 ± 0.6` seconds.
- **Table 2's T2 RCPM row matches `table_random.log:10205` at paper precision:**
  KL `0.9310 ± 0.0172`, ESS `0.548 ± 0.030`, runtime `62.3 ± 0.5` seconds.
- **Camera-ready Table 4's six RNOT KL cells match the high-dimensional logs at
  paper precision:** S10/S20/S30 = `0.0365/0.0226/0.0297`; T10/T20/T30 =
  `0.9230/5.2451/3.2147`. Evidence is `sphere_highD.log` and `torus_highD.log`,
  lines 3072, 4080 and 5104. This establishes provenance for those RNOT values,
  not for the table's newly added RCPM semi-dual baselines.

## Tables 1–2 are not completely explained by these stored files

Historical final table summaries:

| Stored run | Manifold | KL mean ± SE | ESS mean ± SE | Time mean ± SE (s) |
|---|---|---|---|---|
| random | S2 | 0.0443 ± 0.0024 | 0.948 ± 0.004 | 1096.7 ± 5.2 |
| FPS | S2 | 0.0270 ± 0.0016 | 0.965 ± 0.005 | 1024.4 ± 12.9 |
| random | T2 | 0.2242 ± 0.0350 | 0.818 ± 0.032 | 1070.5 ± 1.3 |
| FPS | T2 | 0.1267 ± 0.0068 | 0.906 ± 0.024 | 1038.6 ± 1.6 |

Evidence: `experiments/final_results/table_random.log:10202` and `:10204`;
`table_fps.log:10242` and `:10244`. All four KL means and their uncertainties
round to the paper values. However, paper T2 ESS is `0.85 ± 0.02` for random and
`0.93 ± 0.01` for FPS, whereas these stored logs give `0.818 ± 0.032` and
`0.906 ± 0.024`. Paper RNOT timing rows also differ. S2 FPS ESS is only printed
to three decimals in the log, and its stored LaTeX rounds to `0.96 ± 0.01`,
versus paper `0.97 ± 0.00`; do not infer an exact match from the abbreviated log.

The current `table_results.tex` is the **random** run's output; both table jobs
wrote the same `table_results.npz` and `table_results.tex` filenames. It therefore
does not contain independent evidence for FPS, and neither stored log includes
a complete package freeze or source-commit hash. RCPM results also differ between
the two logs despite identical displayed RCPM configurations. Their cause is
not established by this audit.

## Recorded settings explain substantial ZIP drift

`table_random.log:12–46` and `table_fps.log:12–46` print the same model/solver/
training configuration. These settings also agree with paper F.2 and the
ablation descriptions. The historical table driver explicitly enables x64
(`git show dbf533d:experiments/run_table.py`, line 32).

| Setting | Recorded historical experiment | ZIP default |
|---|---:|---:|
| landmarks | 128 | 256 |
| outer steps | 1000 | 500 |
| inner steps | 2500 | 500 |
| inner LR | 0.05 | 0.005 |
| minimum inner steps | 1000 | 50 |
| soft-argmin initialization | true | false |
| soft-argmin gamma | 0.1 | 0.01 |
| inner Adam | true | false |
| line search | absent in historical solver | true |
| JAX x64 | explicitly enabled in historical drivers | not enabled by ZIP drivers |

Shared settings: MLP `(128,128)`, SiLU, input LayerNorm, last-layer scale 0.01,
outer AdamW LR 0.001, batch 256, tolerance 1e-6, no LR decay, Adam betas
0.9/0.999, wrapped-normal scale 0.3. Table seeds are
`12345,23456,34567,45678,56789`, with evaluation seed `training_seed+1000`,
five batches of 1024. Table/sweep FPS uses **4096 candidates per density**;
the historical **ablation driver uses 10000** (`experiments/run_ablations.py:169`).

An evidence-based, separately labeled historical-settings trial on the ZIP
would use `JAX_ENABLE_X64=True` and these explicit overrides:

```json
{
  "model": {"n_landmarks": 128},
  "training": {"n_steps": 1000},
  "solver": {
    "inner_steps": 2500,
    "inner_lr": 0.05,
    "min_steps": 1000,
    "logsumexp_init": true,
    "logsumexp_gamma": 0.1,
    "use_adam": true,
    "use_line_search": false
  }
}
```

Setting `use_line_search=false` is essential: the ZIP gives line search priority
over Adam. These values are transcribed from recorded settings and history;
they were not selected by fitting the current run's outcomes.

## Git chronology and remaining implementation ambiguity

- `119f041` (Jan 22) and `5fe05b8` (Jan 25) contain the recorded 128-landmark,
  1000-outer/2500-inner Adam configuration. `5fe05b8` still has table RCPM
  gamma 0.1 and 20000 steps; this is **not** the configuration printed in logs.
- `f56f8b0` (Jan 26) changes table RCPM to gamma 1.0 and 5000 steps, and
  temporarily changes the RNOT default to only 50 outer steps.
- `a7327bc` adds the logs while that 50-step default remains. Logged runs
  nevertheless show 1000 steps, so commit date alone does not pin execution.
- `dbf533d86b7b259b42282f0d55c95915c181add6` (Jan 27) restores 1000 outer
  steps. It is a strong historical source candidate whose defaults and table
  constants match the logs, but **not a proven logged execution commit**.
- February commits introduce line search, reduce outer/inner steps and change
  warm-start temperature. ZIP differs again from the latest original repository.
- Comparing `dbf533d` with ZIP: sphere/product distance and log methods,
  `WrappedNormal.sample/log_prob`, the MLP forward method and IFT Jacobian
  function have identical ASTs. The ZIP changes landmark-distance evaluation
  from nested `vmap` to a batched path, adds line search, and recomputes residuals
  at the returned inner solution. Thus restoring settings does not establish
  identical source execution or bitwise equivalence.
- Original `environment.yml`/`requirements.txt` name Python 3.10, JAX/JAXlib
  0.4.28, Flax 0.8.4 and Optax 0.2.3. Logs show ROCm and a Python 3.10
  environment, but do not establish that these package pins were installed on
  the machines that produced them.
  Later [GPU setup documentation](amd_environment.md) records the JAX/ROCm
  0.4.35 candidate used for the restored CUDA trials.

## Figure provenance

`experiments/plot_results.ipynb` loads `ours_sweep_*`, `rcpm_sweep_*` and
`ours_highD_*` NPZ files and plots **KL + 1**, matching Figures 3 and 5's axes.
Those NPZ files are absent from the inspected final-results directory; logs
retain rounded coordinates. The supplied paper targets contain curves without
raw figure coordinates, so no exact figure comparison is established here.

Selected evidence SHA-256:

```text
ablation_S2.json    52ee1e949323c6a814ad9a1e221b8d59eb549346f2786beb6f8c668502c194f6
ablation_T2.json    ece12011a460332c707ccb6b6fc164238d036369b4c783fdcba700aeb7ca3247
ablation_S10.json   74292677adca1cbf4a1eb4409454cb54050aaf7c35fa4f54b9319b4669f7ca48
ablation_T10.json   8a6e91dc5d680c7463051888396985fe95cab828557266dc687d60541c727bd7
table_random.log   3f625f12817470696d9cd8c8064b1bf13729d9aae2075e57572aa173ec84e8cc
table_fps.log      da31a46875cd445e2dbe438612033ef52de9264a1c815f87b22c34d325d4fe5d
```
