# Experimental verification comparison

Generated: 2026-09-08T12:59:05.572768+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| 0d722cf6df6d60ad | nvidia6 | table sphere2 ours random [historical_source_dbf533d] | 1 | partial_fewer_than_five_seeds | 1.32148 ± 0 | 0.520784 ± 0 | kl: differs; ess_ratio: differs (partial) |
| 77b12e4d4dba4339 | nvidia6 | table torus2 ours fps [historical_source_dbf533d] | 1 | partial_fewer_than_five_seeds | 0.119206 ± 0 | 0.840474 ± 0 | kl: differs; ess_ratio: differs (partial) |

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
| S2:RNOT_RND | 0d722cf6df6d60ad | nvidia6 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 40214715a3bb05a7 | 1 | partial_fewer_than_five_seeds |
| T2:RNOT_FPS | 77b12e4d4dba4339 | nvidia6 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 40214715a3bb05a7 | 1 | partial_fewer_than_five_seeds |

Environment `40214715a3bb05a7`: host `nvidia6`, label `historical_source_dbf533d`.
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

### 0d722cf6df6d60ad

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [1.3214831343238518]; sample SE: —.
ESS raw run means: [0.5207842076399488]; sample SE: —.

- Seed 12345: [complete](../../results/historical_source_jax438/table_ours_S2_seed12345_historical_source_dbf533d_da8c697005eb/result.json)

### 77b12e4d4dba4339

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.11920590637269664]; sample SE: —.
ESS raw run means: [0.8404742253033808]; sample SE: —.

- Seed 12345: [complete](../../results/historical_source_jax438/table_ours_T2_seed12345_historical_source_dbf533d_39d585aca286/result.json)
