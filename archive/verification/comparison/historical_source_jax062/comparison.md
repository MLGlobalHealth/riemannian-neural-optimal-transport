# Experimental verification comparison

Generated: 2026-09-08T12:59:05.520068+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| 22e59407fbfa94c8 | nvidia7 | table sphere2 rcpm 1.0 [historical_source_dbf533d] | 5 | complete_five_seeds | 0.004747 ± 0.000901671 | 0.996314 ± 0.000512992 | kl: differs; ess_ratio: matches |
| a68be1d7b7aa01bd | nvidia7 | table torus2 rcpm 1.0 [historical_source_dbf533d] | 5 | complete_five_seeds | 0.919366 ± 0.0209664 | 0.573904 ± 0.0133958 | kl: differs; ess_ratio: differs |

## Completeness

Read 10 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

Overall row coverage: 2/6 rows have a complete five-seed group.
Missing rows across the provided groups: S2:RNOT_FPS, S2:RNOT_RND, T2:RNOT_FPS, T2:RNOT_RND.

Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.

| Row | Complete native groups | Complete configured groups |
|---|---|---|
| S2:RNOT_FPS | none | none |
| S2:RNOT_RND | none | none |
| S2:RCPM_gamma1 | 22e59407fbfa94c8 | none |
| T2:RNOT_FPS | none | none |
| T2:RNOT_RND | none | none |
| T2:RCPM_gamma1 | a68be1d7b7aa01bd | none |

Group provenance for row coverage:

| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |
|---|---|---|---|---|---|---:|---|
| S2:RCPM_gamma1 | 22e59407fbfa94c8 | nvidia7 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 37b68e4b1991c9a1 | 5 | complete_five_seeds |
| T2:RCPM_gamma1 | a68be1d7b7aa01bd | nvidia7 | historical_source_dbf533d | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 37b68e4b1991c9a1 | 5 | complete_five_seeds |

Environment `37b68e4b1991c9a1`: host `nvidia7`, label `historical_source_dbf533d`.
Complete native table rows: S2:RCPM_gamma1, T2:RCPM_gamma1.
Complete configured table rows: none.
Missing rows in this environment across native/configured groups: S2:RNOT_FPS, S2:RNOT_RND, T2:RNOT_FPS, T2:RNOT_RND.
Missing native-default-only rows in this environment: S2:RNOT_FPS, S2:RNOT_RND, T2:RNOT_FPS, T2:RNOT_RND.

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

### 22e59407fbfa94c8

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.0035172677233256205, 0.008515486481507609, 0.0036997300457513316, 0.002905240031055456, 0.005097259093571588]; sample SE: 0.0010081.
ESS raw run means: [0.9974542667995809, 0.9941252540435543, 0.9963831626565242, 0.9967661668898131, 0.9968421656506894]; sample SE: 0.000573542.

- Seed 12345: [complete](../../results/historical_source_jax062/table_rcpm_S2_seed12345_historical_source_dbf533d_eeb9297b6b3c/result.json)

- Seed 23456: [complete](../../results/historical_source_jax062/table_rcpm_S2_seed23456_historical_source_dbf533d_d3198fc4be02/result.json)

- Seed 34567: [complete](../../results/historical_source_jax062/table_rcpm_S2_seed34567_historical_source_dbf533d_7de191ed86fb/result.json)

- Seed 45678: [complete](../../results/historical_source_jax062/table_rcpm_S2_seed45678_historical_source_dbf533d_29f813049204/result.json)

- Seed 56789: [complete](../../results/historical_source_jax062/table_rcpm_S2_seed56789_historical_source_dbf533d_895ebf574be9/result.json)

### a68be1d7b7aa01bd

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.9354391704703964, 0.89541963309522, 0.9192471170604462, 0.8522069127281903, 0.9945168200881731]; sample SE: 0.0234412.
ESS raw run means: [0.6076349103635547, 0.6000380910370138, 0.5616606229875507, 0.5238100153888198, 0.5763769495641254]; sample SE: 0.014977.

- Seed 12345: [complete](../../results/historical_source_jax062/table_rcpm_T2_seed12345_historical_source_dbf533d_04e2fac62813/result.json)

- Seed 23456: [complete](../../results/historical_source_jax062/table_rcpm_T2_seed23456_historical_source_dbf533d_4511c83e15e2/result.json)

- Seed 34567: [complete](../../results/historical_source_jax062/table_rcpm_T2_seed34567_historical_source_dbf533d_e379803e1ea3/result.json)

- Seed 45678: [complete](../../results/historical_source_jax062/table_rcpm_T2_seed45678_historical_source_dbf533d_83e5789c9d5c/result.json)

- Seed 56789: [complete](../../results/historical_source_jax062/table_rcpm_T2_seed56789_historical_source_dbf533d_fcef7c77e885/result.json)
