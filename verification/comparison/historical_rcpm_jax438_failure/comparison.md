# Experimental verification comparison

Generated: 2026-09-08T12:59:05.409585+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| f41535767bd626d2 | nvidia6 | table sphere2 rcpm 1.0 [historical_source_dbf533d] | 0 | partial_with_failed_or_unfinished_runs | — ± — | — ± — | kl: unavailable; ess_ratio: unavailable (partial) (configured) |
| 7efdddd1144e6332 | nvidia6 | table torus2 rcpm 1.0 [historical_source_dbf533d] | 0 | partial_with_failed_or_unfinished_runs | — ± — | — ± — | kl: unavailable; ess_ratio: unavailable (partial) (configured) |

## Completeness

Read 2 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

Overall row coverage: 0/6 rows have a complete five-seed group.
Missing rows across the provided groups: S2:RNOT_FPS, S2:RNOT_RND, S2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND, T2:RCPM_gamma1.

Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.

| Row | Complete native groups | Complete configured groups |
|---|---|---|
| S2:RNOT_FPS | none | none |
| S2:RNOT_RND | none | none |
| S2:RCPM_gamma1 | none | none |
| T2:RNOT_FPS | none | none |
| T2:RNOT_RND | none | none |
| T2:RCPM_gamma1 | none | none |

Group provenance for row coverage:

| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |
|---|---|---|---|---|---|---:|---|
| S2:RCPM_gamma1 | f41535767bd626d2 | nvidia6 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 9536cf7f54e587ba | 0 | partial_with_failed_or_unfinished_runs |
| T2:RCPM_gamma1 | 7efdddd1144e6332 | nvidia6 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 9536cf7f54e587ba | 0 | partial_with_failed_or_unfinished_runs |

Environment `9536cf7f54e587ba`: host `nvidia6`, label `historical_source_dbf533d`.
Complete native table rows: none.
Complete configured table rows: none.
Missing rows in this environment across native/configured groups: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND.
Missing native-default-only rows in this environment: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND.

## Scope and interpretation

Paper Tables 1–3 agree across the arXiv preprint, the historical download from the requested OpenReview URL, and the local camera-ready PDF. The currently served OpenReview revision has not been retrieved.

The paper labels ± quantities as confidence intervals without specifying their level or construction. Shipped SE uses population standard deviation divided by √5; sample SE is also preserved in the JSON. Rounded ±0.00 does not mean zero uncertainty.

Training time is descriptive: the paper used AMD MI300X 192GB, while fresh-process compilation, device and software differences affect these reruns.

- **table_1_2_RNOT_RCPM**: runners available; defaults conflict with paper.

- **table_1_2_RCNF_Moser**: code absent.

- **table_3**: absent from ZIP; historical driver recovered in verification/snapshots/dbf533d; no fresh ablation rerun.

- **table_4**: semi-dual RCPM experiment runner absent.

- **table_5**: SO3/SE3 geometry components present but experiment runners absent.

- **continental_drift**: runner and original point clouds absent.

- **figures_3_5**: dimension sweep runners available; defaults conflict; raw figure values absent.

## Per-seed evidence

### f41535767bd626d2

Seeds: []; missing shipped seeds: [12345, 23456, 34567, 45678, 56789].

KL raw run means: []; sample SE: —.
ESS raw run means: []; sample SE: —.

- Seed 12345: [failed](../../results/historical_rcpm_jax438_failure/table_rcpm_S2_seed12345_historical_source_dbf533d_eeb9297b6b3c/result.json) — unfinished_or_failed

### 7efdddd1144e6332

Seeds: []; missing shipped seeds: [12345, 23456, 34567, 45678, 56789].

KL raw run means: []; sample SE: —.
ESS raw run means: []; sample SE: —.

- Seed 12345: [failed](../../results/historical_rcpm_jax438_failure/table_rcpm_T2_seed12345_historical_source_dbf533d_04e2fac62813/result.json) — unfinished_or_failed
