# Experimental verification comparison

Generated: 2026-09-08T12:59:05.634442+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| e740b5d9546babc5 | nvidia6 | table torus2 ours fps [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.126792 ± 0.0070617 | 0.926916 ± 0.00988707 | kl: matches; ess_ratio: matches (configured) |
| 67456ac801272931 | nvidia6 | table torus2 ours random [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.222998 ± 0.0349288 | 0.79668 ± 0.0327878 | kl: matches; ess_ratio: differs (configured) |
| ac5289de918616a7 | nvidia6 | table sphere2 rcpm 1.0 [zip_x64_cuda_jax435] | 1 | partial_fewer_than_five_seeds | 0.00676858 ± 0 | 0.996514 ± 0 | kl: differs; ess_ratio: differs (partial) |
| 166c9db2e27fc646 | nvidia6 | table torus2 rcpm 1.0 [zip_x64_cuda_jax435] | 1 | partial_fewer_than_five_seeds | 0.900383 ± 0 | 0.595137 ± 0 | kl: differs; ess_ratio: differs (partial) |

## Completeness

Read 12 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

Overall row coverage: 2/6 rows have a complete five-seed group.
Missing rows across the provided groups: S2:RNOT_FPS, S2:RNOT_RND, S2:RCPM_gamma1, T2:RCPM_gamma1.

Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.

| Row | Complete native groups | Complete configured groups |
|---|---|---|
| S2:RNOT_FPS | none | none |
| S2:RNOT_RND | none | none |
| S2:RCPM_gamma1 | none | none |
| T2:RNOT_FPS | none | e740b5d9546babc5 |
| T2:RNOT_RND | none | 67456ac801272931 |
| T2:RCPM_gamma1 | none | none |

Group provenance for row coverage:

| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |
|---|---|---|---|---|---|---:|---|
| S2:RCPM_gamma1 | ac5289de918616a7 | nvidia6 | zip_x64_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 10eebdd73c35fcce | 1 | partial_fewer_than_five_seeds |
| T2:RNOT_FPS | e740b5d9546babc5 | nvidia6 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | bf52d4aa5cd8f392 | 5 | complete_five_seeds |
| T2:RNOT_RND | 67456ac801272931 | nvidia6 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | bf52d4aa5cd8f392 | 5 | complete_five_seeds |
| T2:RCPM_gamma1 | 166c9db2e27fc646 | nvidia6 | zip_x64_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 10eebdd73c35fcce | 1 | partial_fewer_than_five_seeds |

Environment `bf52d4aa5cd8f392`: host `nvidia6`, label `zip_historical_settings_cuda_jax435`.
Complete native table rows: none.
Complete configured table rows: T2:RNOT_FPS, T2:RNOT_RND.
Missing rows in this environment across native/configured groups: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1.
Missing native-default-only rows in this environment: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND.

Environment `10eebdd73c35fcce`: host `nvidia6`, label `zip_x64_cuda_jax435`.
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

### e740b5d9546babc5

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.12129077546086486, 0.11708712751501604, 0.1278625066918664, 0.1564564099338291, 0.11126395437336718]; sample SE: 0.00789522.
ESS raw run means: [0.9448833410945905, 0.9346861080168862, 0.8906438954815815, 0.9136156327128802, 0.9507506504937012]; sample SE: 0.0110541.

- Seed 12345: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json)

- Seed 23456: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fc4a448fa84b/result.json)

- Seed 34567: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_035145fdc01d/result.json)

- Seed 45678: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_847d5106879f/result.json)

- Seed 56789: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_45c22dbf9443/result.json)

### 67456ac801272931

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.1273211332307107, 0.15809615697071022, 0.24011934121984627, 0.35195945194000683, 0.23749206465263514]; sample SE: 0.0390516.
ESS raw run means: [0.6814796058338255, 0.8885916656987016, 0.860670484927768, 0.7871498760152282, 0.7655107429032018]; sample SE: 0.0366579.

- Seed 12345: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/result.json)

- Seed 23456: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fa120bdddebe/result.json)

- Seed 34567: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_7d0f21afb99d/result.json)

- Seed 45678: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_ddbac3d03fab/result.json)

- Seed 56789: [complete](../../results/nvidia6_portability_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_4a181a372502/result.json)

### ac5289de918616a7

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.006768577020761767]; sample SE: —.
ESS raw run means: [0.9965136084187709]; sample SE: —.

- Seed 12345: [complete](../../results/nvidia6_portability_jax435/table_rcpm_S2_seed12345_zip_x64_cuda_jax435_dd37783d77d1/result.json)

### 166c9db2e27fc646

Seeds: [12345]; missing shipped seeds: [23456, 34567, 45678, 56789].

KL raw run means: [0.90038261216891]; sample SE: —.
ESS raw run means: [0.595137276527734]; sample SE: —.

- Seed 12345: [complete](../../results/nvidia6_portability_jax435/table_rcpm_T2_seed12345_zip_x64_cuda_jax435_563daa63d810/result.json)
