# Experimental verification comparison

Generated: 2026-09-08T12:59:05.359637+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| cdd561267ccae27d | nvidia7 | table sphere2 rcpm 1.0 [closest_official_cuda_jax435] | 1 | partial_fewer_than_five_seeds | 0.00676858 ± 0 | 0.996514 ± 0 | kl: differs; ess_ratio: differs (partial) |
| 209a78cbc5a9786a | nvidia7 | table sphere2 ours random [closest_official_cuda_jax435] | 1 | partial_fewer_than_five_seeds | 0.0393168 ± 0 | 0.95139 ± 0 | kl: matches; ess_ratio: matches (partial) |

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
| S2:RNOT_RND | 209a78cbc5a9786a | nvidia7 | closest_official_cuda_jax435 | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 62f41246dabae1f7 | 1 | partial_fewer_than_five_seeds |
| S2:RCPM_gamma1 | cdd561267ccae27d | nvidia7 | closest_official_cuda_jax435 | 48b62f9f4c81b2dd410ac7c623172ece18bafd22452a041c334f8bdebc3ab23b | 62f41246dabae1f7 | 1 | partial_fewer_than_five_seeds |

Environment `62f41246dabae1f7`: host `nvidia7`, label `closest_official_cuda_jax435`.
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

### cdd561267ccae27d

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.006768577020761767]; sample SE: —.
ESS raw run means: [0.9965136084187709]; sample SE: —.

- Seed 12345: [complete](../../results/closest_official_cuda_jax435/table_rcpm_S2_seed12345_closest_official_cuda_jax435_f23e2ce5aabd/result.json)

### 209a78cbc5a9786a

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.03931675014307971]; sample SE: —.
ESS raw run means: [0.951389850537429]; sample SE: —.

- Seed 12345: [complete](../../results/closest_official_cuda_jax435/table_ours_S2_seed12345_closest_official_cuda_jax435_d4432e52de6c/result.json)
