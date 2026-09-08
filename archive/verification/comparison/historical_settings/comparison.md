# Experimental verification comparison

Generated: 2026-09-08T12:59:05.461331+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| 412de9fc0373d6c8 | nvidia7 | table sphere2 ours fps [historical_settings_x64] | 1 | partial_fewer_than_five_seeds | 28.8043 ± 0 | 0.343748 ± 0 | kl: differs; ess_ratio: differs (partial) (configured) |
| df3db1dde6030da4 | nvidia7 | table sphere2 ours random [historical_settings_x64] | 1 | partial_fewer_than_five_seeds | 27.2074 ± 0 | 0.318405 ± 0 | kl: differs; ess_ratio: differs (partial) (configured) |
| 7487da1b0c19b7f5 | nvidia7 | table torus2 ours fps [historical_settings_x64] | 1 | partial_fewer_than_five_seeds | 0.0790563 ± 0 | 0.952204 ± 0 | kl: differs; ess_ratio: differs (partial) (configured) |
| 7109cb1681a99ff3 | nvidia7 | table torus2 ours random [historical_settings_x64] | 1 | partial_fewer_than_five_seeds | 0.155506 ± 0 | 0.837985 ± 0 | kl: differs; ess_ratio: differs (partial) (configured) |

## Completeness

Read 4 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

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
| S2:RNOT_FPS | 412de9fc0373d6c8 | nvidia7 | historical_settings_x64 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | eeff02d5dd8f0ddb | 1 | partial_fewer_than_five_seeds |
| S2:RNOT_RND | df3db1dde6030da4 | nvidia7 | historical_settings_x64 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | eeff02d5dd8f0ddb | 1 | partial_fewer_than_five_seeds |
| T2:RNOT_FPS | 7487da1b0c19b7f5 | nvidia7 | historical_settings_x64 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | eeff02d5dd8f0ddb | 1 | partial_fewer_than_five_seeds |
| T2:RNOT_RND | 7109cb1681a99ff3 | nvidia7 | historical_settings_x64 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | eeff02d5dd8f0ddb | 1 | partial_fewer_than_five_seeds |

Environment `eeff02d5dd8f0ddb`: host `nvidia7`, label `historical_settings_x64`.
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

### 412de9fc0373d6c8

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [28.80433641142211]; sample SE: —.
ESS raw run means: [0.34374843594410803]; sample SE: —.

- Seed 12345: [complete](../../results/historical_settings/table_ours_S2_seed12345_historical_settings_x64_0a43fdb0aa48/result.json)

### df3db1dde6030da4

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [27.207391841727873]; sample SE: —.
ESS raw run means: [0.3184048889529447]; sample SE: —.

- Seed 12345: [complete](../../results/historical_settings/table_ours_S2_seed12345_historical_settings_x64_b0891303e33c/result.json)

### 7487da1b0c19b7f5

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.07905629591680634]; sample SE: —.
ESS raw run means: [0.9522044782179702]; sample SE: —.

- Seed 12345: [complete](../../results/historical_settings/table_ours_T2_seed12345_historical_settings_x64_11df8b4cd73b/result.json)

### 7109cb1681a99ff3

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.15550552744822493]; sample SE: —.
ESS raw run means: [0.8379848380645256]; sample SE: —.

- Seed 12345: [complete](../../results/historical_settings/table_ours_T2_seed12345_historical_settings_x64_0d0f4fd4aa16/result.json)
