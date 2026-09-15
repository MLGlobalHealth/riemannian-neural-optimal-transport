# Paper companion runner inventory

Read-only inventory of the finalMLGH experiment/library source and exact historical snapshot `dbf533d86b7b259b42282f0d55c95915c181add6`. No source was moved or changed; no imports, native builds, training, evaluation, or remote jobs were run. Standard-library AST comparisons are retained in `runner_api_inventory.json`.

## Minimal active interface

Keep the current `src/` and `rcpm/` implementations intact, including `rcpm/LICENSE`, and expose a small experiment layer with explicit JSON recipes. Existing native builders/trainers/evaluators should remain the computational entry points. A thin launcher can select a recipe and a device/output directory, declare every task before launch, and preserve the native evaluation protocol. Moving reports, old environments, scheduling records, alternate source snapshots, website material, and completed diagnostic evidence to `archive/` does not require changing model, optimizer, or metric code.

Suggested active recipe layout (a proposal, not existing CLI/files):

| Recipe | Native implementation | Configuration that must stay explicit |
|---|---|---|
| `configs/main_tables.json` | `run_table.py` plus tested per-seed recording/queue helpers | Recovered historical RNOT config; two landmark strategies; prescribed five seeds; RCPM gamma1 deduplicated across landmark strategies; x64/native old Threefry |
| `configs/ablations.json` | Historical ablation driver's existing functions against selected, clearly identified library | Historical baseline followed by each ablation delta; random default/FPS10000; fixed21-case native list; S2/T2/S10/T10; native evaluation split3 |
| `configs/dimension_sweep.json` | `run_experiments.py` | Shipped dimensions2–10, six RCPM gammas, native FPS4096 and one-seed protocol; label supplied ZIP defaults unless separately configured |
| `configs/high_dimension_sweep.json` | `run_experiments_highD.py` | Dimensions2/5/10/20/30/40, RNOT only, native dimension scaling and one-seed protocol |
| `configs/liegroups.json` | Recovered `run_so3_experiment.py` and `run_se3_experiment.py` | Distinct native SO3/SE3 recipes; FPS4096; one seed and five batches; six RCPM gammas; no S2/T2 historical override |

Store complete effective model/solver/training/evaluation/landmark settings in each result even if recipes use shared defaults internally. A wrapper must compose baseline settings **before** each ablation delta. Do not change global `src/base.py` defaults to make one recipe work: SO3 and sweep builders inherit them and would change too. Preserve already recorded source/configuration identities separately from the reorganized source tree's new fingerprint.

A small separate rendering command should consume results and emit figures/tables without training or recomputing transport metrics. None of the five active experiment drivers currently calls a figure-saving function: they produce NPZ/TeX summaries, not the paper's figure PDFs. Retain any genuine plotting/data assets identified separately, and distinguish plotting historical stored values from plotting newly reproduced results.

## Native experiment coverage and outputs

| Current entry point | Default scope / repetition | Native files |
|---|---|---|
| `experiments/run_table.py` | S2 and T2; RNOT + RCPM gamma1; five training seeds12345/23456/34567/45678/56789; random landmarks by default | `table_results.npz` containing nested object `results`; `table_results.tex` |
| `experiments/run_experiments.py sphere\|torus` | Nine dimensions2–10; one RNOT + six RCPM models per dimension; FPS default; one training seed12345 | `ours_sweep_{sphere,torus}.npz`, `rcpm_sweep_{sphere,torus}.npz` |
| `experiments/run_experiments_highD.py sphere\|torus` | Six dimensions2/5/10/20/30/40; RNOT only; FPS; one seed | `ours_highD_{sphere,torus}.npz` |
| `experiments/run_so3_experiment.py` | SO3 RNOT + RCPM gamma1/.1/.05/.01/.005/.001; native FPS; one seed | `ours_so3.npz`, `rcpm_so3.npz` |
| `experiments/run_se3_experiment.py` | Same seven configurations on SE3, with separate explicit RNOT recipe | `ours_se3.npz`, `rcpm_se3.npz` |
| Historical `experiments/run_ablations.py` | 21 ablations per selected manifold; defaults S2/T2, accepts `--manifolds S2 T2 S10 T10`; one seed | `ablation_<joined_manifolds>.json` and `.tex` |

A raw `run_table.py` invocation trains20 models. Running it once for FPS and once for random also retrains the identical RCPM recipe; the existing canonical main suite correctly declares30 total models by sharing that RCPM row. The dimension sweep declares63 models per manifold family,126 across both; highD adds12 RNOT models. A complete native ablation run over all four published manifolds declares84 models; its default two-manifold run declares42.

The native sweep NPZs contain dimensions, KL means/batch SE, absolute ESS means/batch SE, ESS ratios and runtimes. RCPM fields are prefixed `gamma_<g>_...`. HighD uses the corresponding RNOT field names. Lie-group NPZs retain aggregate means/SE/runtime; SE3 RNOT additionally stores `residual_mean`. Ablation JSON is nested as `{manifold: {ablation: {kl, kl_se, ess_ratio, loss, loss_se, runtime}}}` or `{error: ...}`. Its native JSON/TeX writes occur only after the whole selected loop; its runtime includes build+train, while the other drivers normally time training after building. Plain JSON per-task records and checkpoints provide more durable output than these summaries without changing computation.

Source references: table constants59–94, evaluation451–456/510–515, saves570–631; low-dimensional sweep constants64–126, configuration129–156, saves494–580; highD constants58–67, scaling143–168, saves369–378; SO3 constants61–72 and native builder125; SE3 constants58–77 and explicit configuration80–129.

## Ablation compatibility: runnable API, different inherited defaults

The historical driver can call the current library APIs unchanged: every imported class/function exists; every explicit network/solver/trainer keyword remains accepted. It uses `SemiDualLoss` as a scalar in evaluation, which the current implementation still supports. Sphere and product-torus constructors and required source/target distributions remain available. **Copying the historical driver into active experiments and executing its main unchanged does not restore the historical recipe**, because `get_base_config()` inherits whichever `src/base.py` is imported.

| Baseline field | Historical dbf533d default | Current ZIP default |
|---|---:|---:|
| Landmarks |128|256|
| Outer steps |1000|500|
| Inner maximum/minimum |2500/1000|500/50|
| Inner learning rate |.05|.005|
| Log-sum-exp initialization / gamma |true/.1|false/.01|
| Inner Adam |true|false|

Both share MLP128/128, SiLU, LayerNorm, last scale.01, outer AdamW learning rate.001, batch256, no outer decay, tolerance1e-6, zero momentum, no clipping, Adam betas.9/.999, and seed12345. The existing `historical_settings.json` captures the differing baseline fields. For a current-library ablation adapter, also explicitly record `use_line_search=false` and ordinary updates (`grad_accum_steps=0`, normalized by the trainer to one).

The line-search detail matters: historical `build_experiment()` passes neither `use_line_search` nor `line_search_steps` (`run_ablations.py:257–273`). Current `ArgminSolver.__init__` defaults line search to **false**, whereas current `SolverConfig` defaults it to **true**. Consequently a copied driver's saved config could misleadingly say true although its effective solver is false. Set and record the effective false value in the recipe; do not add line search to this historical protocol.

Historical API/code differences relevant to faithful reproduction:

- `Sphere`, `Product`, `SphereUniform`, `ProductUniformComponents`, `WrappedNormal`, the network constructors/MLP, `SemiDualLoss`, and native IFT/ESS/KL functions have identical AST definitions across dbf533d/current source.
- Current embedding distance evaluation uses `manifold.dist(xs, landmarks.T)` instead of the historical nested vmap (`src/embeddings.py:122–133`). That is an implementation difference, so API compatibility is not a byte-identical historical-source claim.
- Current ArgminSolver adds optional line search and recomputes its returned residual at the final returned point; historical Adam/momentum/GD returned the loop's stored residual. Record the source difference; do not restore or change either implementation as part of packaging.
- Current SemiDualTrainer adds optional gradient accumulation. Its ordinary branch retains the native single-batch update; the ablation driver passes no accumulation option. The changed train default `eval_every=None` does not affect this driver because it explicitly passes its config value.
- Current source adds unrelated entropic/geometric functionality. In particular `src/barycenter.py` cannot simply be removed: current `src/losses.py` and `src/metrics.py` import it unconditionally, even for these nonentropic experiments.

The recovered driver is identical between dbf533d and stored-output revision `a7327bc829837c62d5e41ba222e89ea7b4900d83`, but the latter's `src/base.py` changes outer steps1000→50. Therefore the revision storing the JSONs is not itself proof of the execution settings. Use the documented historical baseline with its provenance; retain the exact archived snapshot for users who need historical source execution. No fresh ablation cohort has established agreement of a current-library reconstruction with the79 stored/published cells.

## Ablation-specific protocol and rendering mapping

Historical `ABLATIONS_TO_RUN` contains21 cases (`:133–157`), though `ABLATIONS` defines24 possible cases. The active list includes baseline; no LSE;500/4000 inner steps with matching minimum; inner LR.05/.001; LSE gamma1/.01/.001; no Adam; FPS;32/64/256 landmarks; four hidden-width/depth variants; outer LR.0001; batches512/128. The extra inner LR.05 case duplicates the historical baseline's LR and is not a published Table3 row. Table3 has20 rows across four manifolds,79 numeric cells and one blank; the stored files retain all21 cases. Preserve native run coverage and use an explicit row-name map when rendering the paper table rather than deleting stored cases or filling the blank.

Ablation landmarks are random except the FPS case; FPS uses10000 candidates **per distribution**, not the4096 used elsewhere (`:169,224–229`). Training restarts from12345 for each case. Evaluation resets to12345 and performs `key,k1,k2 = split(key,3)` each batch (`:309–347`), drawing source and target directly from those keys. This differs from both the table seed+1000/split2 protocol and the Lie-group/sweep seed12345/split2 protocol. It calls the native loss as well as native KL/ESS and computes KL/loss population batch SE over five batches1024. Do not reuse another suite's key loop or substitute five training seeds.

## Path and scope constraints

Every driver prepends a project root inferred from its own file location. The historical driver in an archived snapshot will prepend that snapshot's `src`, so importing it into an active-library wrapper can select the wrong library. Keep an unchanged active copy under `experiments/` if its functions are to use active `src`, with explicit baseline composition outside its main; otherwise launch the archived snapshot in its own fresh process. Do not import two different `src` trees into one interpreter. A fresh-process task changes historical compilation/cache order; retain that execution note when relevant to timings.

All covered distributions are synthetic; no external dataset is required for these runners. Existing missing RCNF/Moser, semi-dual RCPM, continental-drift data/runner, and unverified full figure-sweep coverage remain unresolved. Archived evidence should preserve that distinction. Generated comparison plots can present retained outputs accurately, but should not be described as fresh experiment or exact paper-figure reproduction solely because plotting succeeds.
