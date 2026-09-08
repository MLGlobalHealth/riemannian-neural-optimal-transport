# Experiments and figures

Run these commands from the repository root after the installation in [README](../README.md). Choose GPU IDs available for your use and a **new output directory for each training invocation**. Plotting an existing result or reference file requires only the [plotting dependencies](../requirements/plots.txt).

The public commands call the supplied training and metric implementations. Their recipes, execution metadata and saved outputs make a run inspectable; they do not imply that every published number has been reproduced. See [reproduction status](reproducibility.md) for measured agreement and remaining gaps.

## Map from the paper to commands

| Paper item | Command/workflow | Scope |
|---|---|---|
| Tables 1–2 | `paper.run tables`, `paper.plot tables` | RNOT FPS/random and native RCPM gamma 1 on S2/T2; 30 trained models. RCNF and Moser have paper reference values only. |
| Table 3 | `paper.run ablations`, `paper.plot ablations` | Recovered historical driver against the packaged library. The native suite has 21 settings × 4 manifolds =84 models; one setting repeats the baseline inner learning rate. The reference plot contains the 20 published rows. |
| Table 4 | Numeric targets in `reference/paper_targets.json` | The semi-dual RCPM runner remains unavailable. `highd` supplies RNOT runs, but the existing forward-KL RCPM sweep is a different experiment. There is no command claiming a complete Table 4 reproduction. |
| Table 5 | `paper.run liegroups`, `paper.plot liegroups` | SO(3)/SE(3): one RNOT model and six RCPM gamma values per manifold, 14 models total. |
| Figure 1 | `paper.continental`, then `paper.transport --samples` | Fresh continental-drift training and geographic sample plots from the packaged point clouds. The original saved model is missing; the published transported image is not claimed to be exactly recreated. |
| Figure 2 | `paper.transport --input` | Source, target, transported samples and endpoint geodesics from one explicitly selected saved S2/T2 RNOT checkpoint. |
| Figure 3 | `paper.run sweep`, `paper.plot sweep` | Dimensions 2–10, spheres/tori, RNOT and six RCPM gammas: 126 models. |
| Figure 4 | `paper.embeddings`, `paper.plot embeddings` | Fresh random/FPS landmark computation on S2; no transport-model training. Historical printed values are also supplied. |
| Figure 5 | `paper.run highd`, `paper.plot highd` | RNOT dimensions 2,5,10,20,30,40: 12 models. Add a `sweep` result tree to show RCPM points up to 10 only. |

The dimension plots use **KL + 1 on a logarithmic axis**, as in the original plotting notebook. The exported data retains the original signed KL estimates.

## Preview a plan and run a smoke check

A dry run reads the recipe, prints all selected job IDs and commands, and creates no run directory or subprocesses:

```sh
python -m paper.run tables --gpus 0 1 --output runs/tables --dry-run
```

A bounded smoke check selects one native job and reduces the execution size. It checks the pipeline; it is not a paper result:

```sh
python -m paper.run tables --gpus 0 --output runs/smoke-s2 \
  --job tables_S2_ours_fps_seed12345 --smoke
```

Training smoke records are excluded by the table/sweep plot readers. Inspect their `result.json` and log, or render the smoke checkpoint with `paper.transport`; the qualitative output is labeled as a smoke check. The launcher also allows `--platform cpu` together with `--smoke`; full training recipes require CUDA.

## Tables 1–2

Run all six supported rows with the five prescribed training seeds:

```sh
python -m paper.run tables --gpus 0 1 --output runs/tables
python -m paper.plot tables --input runs/tables --output figures/tables
```

Seeds are `12345 23456 34567 45678 56789`. Each trained model is evaluated on five batches of 1024 points. The table plot first averages those batches within each model, then aggregates the five seed means. A partial seed set is displayed as partial; it is not filled with seeds from another configuration or host.

Filters select jobs already declared by the recipe. For example, run the complete S2 FPS row alone:

```sh
python -m paper.run tables --gpus 0 --output runs/s2-fps \
  --manifolds S2 --methods ours --landmarks fps
python -m paper.plot tables --input runs/s2-fps --output figures/s2-fps
```

To select one trained model, add `--seeds 12345`, or use its exact `--job` ID. Such a run supports checkpoint visualization but does not complete a five-seed table row. A landmark filter selects RNOT jobs only, since RCPM does not have that selection field.

## Table 5: Lie groups

```sh
python -m paper.run liegroups --gpus 0 1 --output runs/liegroups
python -m paper.plot liegroups --input runs/liegroups --output figures/liegroups
```

The native gamma grid is `1, 0.1, 0.05, 0.01, 0.005, 0.001`. One exact model can be selected without altering its settings:

```sh
python -m paper.run liegroups --gpus 0 --output runs/so3-rcpm-gamma1 \
  --job liegroups_SO3_rcpm_gamma1_seed12345
```

Each Lie-group row is one trained model with five evaluation batches. Its error bars describe evaluation sampling of that model, not variation across five independently trained seeds. Actual nonfinite results remain nonfinite; a build failure is reported as an execution failure. Published qualitative instability is not converted into a numeric zero.

## Figure 3 and Figure 5: dimension sweeps

```sh
python -m paper.run sweep --gpus 0 1 --output runs/dimensions/sweep
python -m paper.plot sweep --input runs/dimensions/sweep --output figures/sweep

python -m paper.run highd --gpus 0 1 --output runs/dimensions/highd
python -m paper.plot highd --input runs/dimensions --output figures/highd
```

The last command reads the common parent directory: RNOT comes from `highd`, and RCPM comes from `sweep`. It does not add high-dimensional RCPM points or combine overlapping RNOT points from the two recipes.

These commands expose the supplied sweep defaults, with x64 disabled, one training seed and no historical RNOT overrides. The historical logged curves were produced under an incompletely recovered execution environment. A fresh sweep is therefore labeled as an unverified supplied recipe until its actual outputs are compared.

For a small subset of the declared sweep:

```sh
python -m paper.run sweep --gpus 0 --output runs/sweep-s2-s3 \
  --manifolds S2 S3 --methods ours
```

Missing dimensions remain gaps in the plot. A line connects separately reported points from a common recipe/source/environment; it does not average them.

## Table 3: ablations

```sh
python -m paper.run ablations --gpus 0 1 --output runs/ablations
python -m paper.plot ablations --input runs/ablations --output figures/ablations
```

The public recipe retains all 21 native driver settings, including `inner_lr_5e-2`, which repeats the baseline learning rate. The paper's 20-row reference excludes that redundant setting. Native ablation names such as `fewer_inner_steps_500` are kept in new run records; the reference file records their mapping to the published names.

Select a single configuration by its declared job ID:

```sh
python -m paper.run ablations --gpus 0 --output runs/ablation-s2-fps \
  --job ablations_S2_ours_fps_seed12345_landmarks_fps
```

The heatmap cells show recorded KL, including signed values. Color uses a separately labeled signed logarithmic transform. Failed or ambiguous cells are not filled from another run.

## Figure 2: a saved model's transport

After a completed public S2/T2 RNOT run, select its exact result record:

```sh
python -m paper.transport \
  --input runs/tables/results/tables_S2_ours_fps_seed12345/result.json \
  --gpu 0 --seed 123 --n-samples 1024 --output runs/s2-transport
```

The command verifies the native source, restoration code and checkpoint, restores its configuration/parameters/landmarks, samples source and target points, and calls the unchanged solver. It writes `samples.npz`, `samples.json`, `transport.png`, `transport.pdf` and rendering provenance. There is no retraining or seed selection by visual quality.

Render the saved arrays again without JAX or a GPU:

```sh
python -m paper.transport --samples runs/s2-transport/samples.npz \
  --output figures/s2-transport
```

The renderer shows point samples, not an estimated density. Colors track source longitude through transport. Curves join recorded source/transport endpoints by geodesic interpolation; they are not internal optimizer trajectories. A new visualization describes the declared model and sampling seed, not necessarily the representative checkpoint used for the original paper image.

## Figure 4: landmark embeddings

Run the recovered computation and plot its numerical output:

```sh
python -m paper.embeddings --gpu 0 --output runs/embeddings.json
python -m paper.plot embeddings --input runs/embeddings.json --output figures/embeddings
```

The defaults use seed 0, validation size 2048, FPS candidate pool 4096, landmark counts 8–2048, subset size 512, five repeats and epsilon 0.001. The command defaults to float32 because the recovered notebook did not enable x64; the original notebook kernel state is unconfirmed. `--x64` is an explicit precision variant recorded in the output.

For an execution check that requires no GPU:

```sh
python -m paper.embeddings --platform cpu --smoke --output runs/embeddings-smoke.json
python -m paper.plot embeddings --input runs/embeddings-smoke.json \
  --output figures/embeddings-smoke
```

This still requires the experiment environment's JAX installation. Zero near-collision rates use a labeled display floor on the logarithmic axis; the JSON/CSV retains their raw zeros.

## Figure 1: continental drift

Install the empirical-data dependency in the experiment environment:

```sh
python -m pip install -r requirements/empirical.txt
```

Validate the explicit recipe and input CSV hashes, run it, then render its sample bundle:

```sh
python -m paper.continental --config paper/configs/continental.json \
  --output runs/continental --dry-run
python -m paper.continental --config paper/configs/continental.json \
  --gpu 0 --output runs/continental
python -m paper.transport --samples runs/continental/samples.npz \
  --output figures/continental
```

For a small execution check, use a separate directory and `--smoke`:

```sh
python -m paper.continental --config paper/configs/continental.json \
  --gpu 0 --smoke --output runs/continental-smoke
python -m paper.transport --samples runs/continental-smoke/samples.npz \
  --output figures/continental-smoke
```

The CSVs contain 50,000 source/target points each; their provenance is described in [data/README](../data/README.md). The recovered notebook recipe explicitly uses 1000 outer updates,128 landmarks, batch 256 and up to 2500 inner steps. Paper Appendix F.1 instead states 500 updates,1024 landmarks, batch 1024 and up to 1000 inner steps. The command exposes the recovered notebook recipe with this distinction recorded; it does not silently replace settings to claim an exact Figure 1 reproduction. Some unprinted notebook settings follow the packaged library defaults.

The full sample export takes the first 1024 source CSV rows and a target sample using seed 42, then transports them using saved `state.params`. The original continental checkpoint is unavailable. This workflow does not compute or claim the paper's plate-purity or Monge-gap results.

## Reference figures without model execution

```sh
python -m paper.plot tables --reference --output figures/reference-tables
python -m paper.plot liegroups --reference --output figures/reference-liegroups
python -m paper.plot sweep --reference --output figures/reference-sweep
python -m paper.plot highd --reference --output figures/reference-highd
python -m paper.plot ablations --reference --output figures/reference-ablations
python -m paper.plot embeddings --reference --output figures/reference-embeddings
```

Tables use published numeric targets; sweeps, ablations and embedding plots use recorded historical outputs. Source hashes, original paths and numeric precision are documented in [reference/README](../reference/README.md). No paper PDF crop is substituted for newly computed output. Figure 1/2 reference artwork is retained in the full repository's archive, outside the standalone code distribution.

## Results, groups and failures

A training directory contains:

```text
manifest.json                         # Declared jobs, source hashes, recipe and commands
status.json                           # Queue state and per-job exit status
results/JOB_ID/result.json            # Effective config, source/runtime, keys and raw metrics
results/JOB_ID/checkpoint.msgpack     # Parameters and model state needed for evaluation
logs/JOB_ID.log                       # Native training/evaluation output
```

Each GPU runs at most one worker from that invocation. A failed worker stops new jobs from starting; active workers finish and their outputs are retained. The launcher does not resume or overwrite an existing directory. Completed jobs can be selected by exact ID in a fresh invocation when a bounded rerun is needed; the prior execution remains separate evidence.

Plotters keep source, public worker/runtime hashes, configuration, label, host and software groups distinct. Identical copied result files are evidence aliases, not extra seeds. Conflicting repeated executions are marked ambiguous, and partial five-seed groups are not pooled. GPU indices can differ within the same recorded host/model/driver group. A `--group GROUP_OR_COHORT_ID` plot filter can be repeated to select explicit groups; full identifiers are retained in plot JSONs and CSVs.

Every numerical plot writes PNG, PDF, CSV and a JSON data/provenance summary. Existing audit comparison JSONs are also accepted for main-table/Lie-group plots, but standalone plotting does not depend on importing the archive. Custom recipes use `paper.run SUITE --config PATH ...` and must retain the supported native protocol; preserve their labels and new output directories when interpreting results.
