# Reference plot data

These files contain published targets or previously stored results. They are not new experiment runs. The plot commands read these compact JSON files directly; they do not load code or data from `archive/`.

| File | Origin | Precision and scope |
|---|---|---|
| `paper_targets.json` | Paper Tables1–5, checked against local PDF copies; Table4/5 use the May2026 camera-ready version. | All published table values are retained unchanged. A blank or qualitative cell is not a numeric zero. Published plus/minus values are displayed descriptively; a confidence level is not inferred. |
| `dimension_logs.json` | Original `sphere_run.log`, `torus_run.log`, `sphere_highD.log`, `torus_highD.log`. |126 sweep records (2families ×9dimensions ×7methods/settings),12 highD RNOT records. KL/SE printed to4decimals; ESS ratio to3. Each record identifies the log hash and line. The original full-precision NPZ files were absent from the checked folders. |
| `ablation_results.json` | Four original `ablation_{S2,T2,S10,T10}.json` files. |20 paper ablations ×4manifolds, preserving stored numeric precision, native batch SE and original names. The historical T10/256landmark KL is negative; the paper cell is blank. Both facts are retained. |
| `embedding_results.json` | Saved textual output of `Gromov Embeddings E1.ipynb`, cell5. |18 rows, random/FPS at9landmark counts. Coverage/separation were printed to6decimals. Near-collision values preserve printed scientific notation, including exact printed zeros. Missing repeat-level values/error bars are not reconstructed. |

The historical source paths in each provenance entry point to `archive/original/` in the full research repository. A standalone source export can retain the hashes and numerical JSONs without including that archive. Original paper PDF crops and artwork remain archived reference figures; no plotting command presents those images as newly generated experiment output.

Render any reference plot without JAX or a GPU:

```bash
python -m paper.plot tables --reference --output figures/reference-tables
python -m paper.plot liegroups --reference --output figures/reference-liegroups
python -m paper.plot sweep --reference --output figures/reference-sweep
python -m paper.plot highd --reference --output figures/reference-highd
python -m paper.plot ablations --reference --output figures/reference-ablations
python -m paper.plot embeddings --reference --output figures/reference-embeddings
```

Each command writes PNG, PDF, CSV and provenance/data JSON. Figure3/5-style plots use **KL +1 on a logarithmic axis**. Underlying signed KL values are preserved. Embedding plots display zero near-collision rates at a labeled10⁻⁸ floor while keeping raw zeros in CSV/JSON.

For newly recorded experiments, replace `--reference` with `--input runs/results` or a specific comparison/plot-data JSON. Source, public worker/runtime hashes, recipe, configuration, host and software groups remain distinct. Tables aggregate the five prescribed training seeds only; Lie-group and sweep rows retain one trained model and its five evaluation batches. Replays, smoke training records, conflicting duplicates, failures and missing batches cannot silently fill a complete training row.

For fresh embedding computation, first run `python -m paper.embeddings --help`, then pass its output JSON to `python -m paper.plot embeddings --input ... --output ...`.

For qualitative plots, `python -m paper.transport --samples RUN/samples.npz --output figures/transport` renders exported source/target/transport samples offline. `python -m paper.transport --input RUNS/results/JOB/result.json --gpu 0 --output NEW_DIRECTORY` restores a public S2/T2 RNOT checkpoint, samples with a recorded seed, calls its unchanged native solver and exports the arrays before rendering. These are sample plots of a declared saved run; they are not asserted to be the exact representative Figure2 run from the paper. Continental sample bundles use the same offline renderer.
