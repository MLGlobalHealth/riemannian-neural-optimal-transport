# Plot inventory and practical release interface

Read-only inventory, 2026-09-08. No source edits, model execution, remote jobs, or new scientific figures were performed. The inspected release is `/Users/alessandromicheli/Documents/dev/ml/projects/riemannian-neural-ot-finalMLGH`; “original worktree” below means its sibling `riemannian-neural-ot`. Original notebooks are useful implementation references, not proof of the exact execution that produced a published image.

The practical first release can plot the completed table experiments directly from committed result JSONs, plot archived dimension/ablation results with explicit historical labels, and offer a separate checkpoint-to-samples step for new sphere/torus qualitative plots. These are distinct from copying the paper's existing figure assets.

## Available code and outputs

The five release drivers (`experiments/run_table.py`, `run_experiments.py`, `run_experiments_highD.py`, `run_so3_experiment.py`, `run_se3_experiment.py`) contain no plotting functions. Their native outputs are numerical NPZ summaries. The audited wrappers add complete per-batch JSON, source/runtime/configuration provenance and checkpoints.

| Original implementation | Inputs and outputs | Release implication |
|---|---|---|
| `experiments/plot_results.ipynb`, cells 0–4 | Loads `ours_sweep_{sphere,torus}.npz`, `rcpm_sweep_{sphere,torus}.npz`, `ours_highD_{sphere,torus}.npz`; produces `kl_combined.{pdf,png}`, `sphere_highD.{pdf,png}`, `torus_highD.{pdf,png}`. | Existing figure style and plotting logic for Figures 3/5. Replace notebook cwd dependence and mandatory LaTeX with CLI paths and Matplotlib mathtext. |
| `sphere_figure1.ipynb`, cells 3, 10, 12–13 | Sphere KDE/surface renderer, transport trajectories and individual four-panel outputs. Cell 8 generates source/target/transported arrays from live trained models. | Useful Figure 2 renderer. Extract only rendering; do not execute the notebook wholesale, which also trains baselines and runs unrelated diagnostic cells. |
| `torus_figure1.ipynb`, cells 3, 14–16 | Periodic angular KDE, embedded torus surface, trajectory renderer and four individual panels. Cell 9 samples a live model. | Same separation for the torus. Native representation is `(cos θ1,sin θ1,cos θ2,sin θ2)`; convert to angles only for display. |
| `Gromov Embeddings E1.ipynb`, cells 4–5 | `coverage_radius`, `sweep_M_diagnostics`, `plot_diagnostics`; outputs `embedding_diagnostics.{pdf,png}` plus printed numerical table. | Figure 4 computation and plotting reference; archived printed values are available, but no separate full-precision array file was found in the bounded inspection. |
| `continental_drift.ipynb`, cells 4, 6, 11–12, 14–17; `continental_drift_output.ipynb` | Loads a saved continental model, transports samples, plots source/target/pushforward/paths and plate labels; outputs continental-drift panels. | Source data and plotting code exist in the original worktree, but referenced saved model is missing locally. Do not label a paper crop as regenerated transport. |
| `plot_earth_spheres.py:131,300` | `earth_plot` and `make_figure2`, source data, learned samples, log-density grid. | Reusable geographic plotting utility from the original worktree, not part of the supplied ZIP. `run_earth_ot.py` targets volcano/earthquake/flood/fire datasets and is not the paper's continental-drift recipe. |
| `plot-demo.py`, `plot-components.py` | Historical RCPM demos expect their own pickle/config environment. | Archive as examples rather than presenting as the paper's current figure commands. |

## Paper figures: what can be regenerated

The reference is camera-ready `website/assets/paper.pdf`, SHA256 `318386d9f934d70a7758df4f931058b68b5182b4bd60e9996b3c524b569f3e89`. PDF page numbers below are one-based. Existing `website/assets/figure-provenance.json` explicitly identifies PDF crops/preexisting assets as original paper artwork.

| Paper item | Actual scope | Can produce now without new training? | Missing inputs / precise limits |
|---|---|---|---|
| Figure 1, p7 | Continental drift on S2: source at150Ma, present target, pushforward and trajectories. Appendix F.1 also discusses plate purity/Monge gap. | Source/target geographic point plots: yes, from original CSVs. Full transported figure: not from a verified available continental checkpoint. | `saved_runs/continental_drift_20260121_211950.pkl` and older `...20260114_233624.pkl` are referenced by notebooks; the original worktree's `saved_runs` directory is absent. Need a saved model plus config/landmarks/source version, or a declared fresh training run following F.1. Published 90% purity and <0.1% gap are not implied by input CSVs or figure artwork. |
| Figure 2, p8 | RNOT source, target, geodesic trajectories, pushforward for S2 and T2. No RCNF/Moser/RCPM panel in this figure. | Yes in principle from the verified S2/T2 RNOT checkpoints retained locally, by running an explicit native transport-only sample export; rendering the exported arrays then uses CPU only. | Existing result JSONs contain scalar metrics, not transported sample arrays. Checkpoints are local ignored artifacts, so a clean public clone will need a new run or separately distributed checkpoint bundle. A new plot from a declared seed is a new visualization of that saved run, not necessarily the exact representative run in the PDF. |
| Figure 3, p9 | RNOT plus six RCPM gammas, spheres/tori, dimensions2…10. | Historical reconstruction from logs is possible now at their printed precision. A fresh sweep plot becomes possible as new `suite=sweep` records are produced. | No verified fresh dimension-sweep cohort is in canonical current result roots. Six original NPZ filenames are referenced but absent from checked `experiments/final_results` and release numerical assets. Do not substitute the p2 table cohort for the distinct dimension-sweep recipe. |
| Figure 4, p23 | S2 embedding coverage/separation/near-collision statistics versus landmark count, random/FPS. | Archived printed-value plot is possible, explicitly historical and rounded. Fresh numerical regeneration requires executing the recovered standalone notebook computation, not model training. | Full per-repeat arrays/uncertainty used by the original renderer are not separately saved. Printed means cannot reconstruct missing error bands. The notebook output includes small nonzero random near-collision values, so do not replace all values by zero to match a caption. |
| Figure 5, p32 | RNOT on dimensions2,5,10,20,30,40; six RCPM curves only2…10. | Historical rounded-log reconstruction is possible. Future fresh highD runs can be plotted. | Do not extrapolate RCPM to40 or mix highD RNOT points with separately trained main-sweep points at overlapping dimensions. Retain which native suite produced each series. |

Figure 3/5 notebook code uses **`KL + 1` on a logarithmic axis**, despite captions describing KL scaling. Keep this explicit on the axis. Preserve negative reported KL estimates in the underlying JSON/CSV; do not replace them with zero. If a transformed value is nonpositive or nonfinite, annotate/exclude it from that axis with a recorded reason; do not silently clip the measured values.

Figure 4 recovered recipe: `PRNGKey(0)`, S2 uniform validation set2048, landmark counts `[8,16,32,64,128,256,512,1024,2048]`, FPS candidate pool4096, epsilon0.001, diagnostic subset512, five repeats. It is a separate embedding experiment, not five trained transport seeds.

## Tables and comparisons available now

- **Tables 1/2:** complete restored ZIP main-table RNOT FPS/random and RCPM gamma1 records are available. Use `verification/results/restored_zip_jax435` for the declared primary six rows and explicit second-host `verification/results/nvidia6_portability_jax435` when requested. The two hosts must remain separate series. RCNF and Moser rows exist as paper targets, not reproduced models/outputs; any reference markers must say “paper-reported”. Do not import historical snapshots or ignored artifact copies by broad unfiltered traversal and count them as new runs.
- **Table 3:** four committed `verification/provenance/stored_ablation_results/ablation_{S2,T2,S10,T10}.json` files support an explicitly historical ablation heatmap/table now. The previous audit found all79 numeric paper cells match at printed rounding. The recovered historical `run_ablations.py` is in the archived snapshot, not the supplied22-file ZIP; label any future run by the source actually used.
- **Table 4:** paper targets plus historical highD RNOT summaries exist; the additional semi-dual-trained RCPM runner/output provenance remains unrecovered. Do not use the native forward-KL RCPM dimension sweep as if it were this experiment.
- **Table 5:** all14 recovered native Lie-group records are now committed under `verification/liegroups/results`, with9 finite paper target rows and5 SE3 gamma<1 qualitative target slots. Plot actual finite/nonfinite outcomes; do not draw zeros for failures or reuse the paper's instability description when the fresh record is finite. Each row is **one trained model**, summarized over **five evaluation batches**, not five training replicates.
- **Host comparison:** evaluation-only replay records can support a separate fixed-checkpoint comparison plot. They must never be aggregated into training-run tables. Same-checkpoint hashes, exact keys and runtime provenance are available for those comparisons.

For Tables1/2, calculate a mean per trained seed from its five batches, then the prescribed five-seed row mean and native population-SE/sqrt(5), retaining individual seeds. For Table5 and native sweep/highD drivers, the displayed native uncertainty is population-SE/sqrt(5) across evaluation batches of one model. Use wording specific to that level. The sample-SE alternatives already recorded are supplemental, not a newly inferred paper confidence interval.

## Concrete saved schemas

`run_experiments.py:494` and `run_experiments_highD.py:369` write RNOT NPZ fields:

```
dimensions, kl_values, kl_errors,
ess_values, ess_errors, ess_ratio_values, runtimes
```

`run_experiments.py:571` writes RCPM fields:

```
dimensions, gammas,
gamma_{g}_kl, gamma_{g}_kl_errors,
gamma_{g}_ess, gamma_{g}_ess_errors,
gamma_{g}_ess_ratio, gamma_{g}_runtimes
```

The Lie-group NPZ files use the same summary concepts without a dimension sweep. Native `run_table.py:571` writes `results=results`, a nested object-valued NPZ; prefer the audited JSON outputs for a portable plot input and keep legacy object-NPZ support separate and explicit.

Audited native JSONs supply selection/suite/dimension/method/landmark method/gamma/seed, source/runner/helper hashes, exact config, host/device/software, raw indexed evaluation batches and keys, summary statistics, checkpoint SHA and measured runtime. Existing comparison JSONs already preserve group identity and coverage; consume these for table plots rather than implementing a second inconsistent aggregation rule. Main and Lie-group comparison schemas differ and need explicit adapters.

A useful qualitative sample bundle contains numeric arrays `source_samples`, `target_samples`, `transported_samples`, `trajectory_sources`, `trajectory_targets` and optional precomputed `trajectory_points`; save a JSON sidecar with source/config/checkpoint hashes, exact sampling keys, sample counts, native representation, manifold/dimension, checkpoint seed/host, sampling host/software, plot KDE bandwidth/grid, projection/radii/camera, and renderer version. Preserve RNOT solver hint arrays or their exact keys/counts in the sidecar. Export samples by invoking the unchanged native model/solver, then render those arrays without importing JAX. Never infer transport arrays from scalar KL/ESS summaries.

## Recommended small CLI

A practical split keeps ordinary plotting independent of GPU/JAX startup:

```text
python scripts/plot_results.py tables --comparison PATH/comparison.json --output-dir figures/main
python scripts/plot_results.py liegroups --comparison PATH/comparison.json --output-dir figures/liegroups
python scripts/plot_results.py sweep --results-root RESULTS --suite sweep --group GROUP --output-dir figures/dimensions
python scripts/plot_results.py sweep --results-root RESULTS --suite highD --group GROUP --output-dir figures/highD
python scripts/plot_results.py ablations --input-dir ARCHIVED_ABLATION_JSON --provenance historical --output-dir figures/ablations
python scripts/plot_results.py qualitative --samples samples.npz --metadata samples.json --output-dir figures/transport
```

A separate future `export_transport_samples.py --input-result result.json --checkpoint checkpoint.msgpack --output-dir ... --seed ... --n-samples ...` can reuse audited checkpoint restoration and native transport, record a new sampling manifest, and avoid retraining. It should require a single declared model, not choose a seed by appearance or metric. The plotter needs NumPy/Matplotlib; SciPy is sufficient for recovered KDE/interpolation renderers. Cartopy is optional for the continental projection. Default headless `Agg`, PDF+300dpi PNG, mathtext without system LaTeX, sensible white backgrounds, explicit titles/legends, and a machine-readable plot-data JSON/CSV alongside the image make outputs reviewable and portable.

For historical sweep recovery, use a small explicit importer rather than passing logs off as raw fresh results. Store every extracted dimension/gamma/metric/SE, its source log SHA and line number, and the fact that KL/SE were printed to4decimals and ESS ratio to3. The two sweep logs each contain63 metric summaries (9 RNOT,54 RCPM); the highD logs contain6 each. Imports must verify the entire expected grid and reject duplicate/missing sections. An exact paper-figure numerical match remains unproven without original NPZs; reconstructed curves are historical logged outputs.

Every plotting mode should preserve source/helper/wrapper/config/label/host/software groups, show missing/nonfinite rows, reject ambiguous duplicate executions, and output the selected group IDs. Do not pool different environments merely because a display label is the same. Identical artifact copies may be evidence aliases, never extra seeds.

## Local historical input hashes and packaging boundaries

The following original-worktree inputs were directly inspected; they are not all currently shipped in finalMLGH:

| File | SHA256 | Extent |
|---|---|---|
| `experiments/final_results/sphere_run.log` | `1783a18c229de00c87f0ed9cc5b08f784bbf86239997c65d077755ae9fc90e90` |63 completed metric summaries |
| `experiments/final_results/torus_run.log` | `73e1ac291241aaae24b31e260cb59aa40c479aa95d6a20a33aa973e389142bf0` |63 completed metric summaries |
| `experiments/final_results/sphere_highD.log` | `402f846ba2b31758c0422bc6335b96ca4b2501938d5cda92d5fec11b554de0fb` |6 completed metric summaries |
| `experiments/final_results/torus_highD.log` | `a781f7bc8ca753f3edd8fadedd5fe13f9bff906293779c58ac430d76d739c5c3` |6 completed metric summaries |
| `data/land_points_past.csv` | `1afe72623f68d42ded0c1c90de223fb050f4298ff272ca6401d32cb40e1d2fee` |50,000 rows; `lat,lon,plate_id` |
| `data/land_points_present.csv` | `17ff2ebfa9fa7acb367b9e21adb994e1a501777e97143decc8f437a5fa33357f` |50,000 rows; `lat,lon,plate_id` |

Keep small numerical JSONs, plot scripts, provenance manifests and explicit paper targets accessible in the runnable release. Place old notebooks, exploratory scripts, historical source snapshots and audit-only probes in a clearly named archive, retaining paths or a migration map for existing evidence links. Large ignored checkpoints/logs can remain separately distributed artifacts with hash manifests and clear acquisition instructions. Original PDF crops and artwork belong in paper/reference assets; they are useful illustrations but must remain visibly distinct from regenerated experiment figures.
