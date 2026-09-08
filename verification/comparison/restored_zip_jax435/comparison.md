# Experimental verification comparison

Generated: 2026-09-08T12:59:05.707213+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| f93bd4238029889f | nvidia7 | table sphere2 ours fps [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.0269593 ± 0.00157974 | 0.965099 ± 0.00473029 | kl: matches; ess_ratio: matches (configured) |
| d2b0bd2d89b35fc4 | nvidia7 | table sphere2 ours random [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.0441433 ± 0.00224148 | 0.948114 ± 0.00368129 | kl: matches; ess_ratio: matches (configured) |
| a42e85edcd0849b4 | nvidia7 | table torus2 ours fps [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.127375 ± 0.00709924 | 0.893765 ± 0.0282074 | kl: matches; ess_ratio: differs (configured) |
| 9cc4e7180b4edd02 | nvidia7 | table torus2 ours random [zip_historical_settings_cuda_jax435] | 5 | complete_five_seeds | 0.222871 ± 0.0352441 | 0.792746 ± 0.0337315 | kl: matches; ess_ratio: differs (configured) |
| cdc40555dcd2babe | nvidia7 | table sphere2 rcpm 1.0 [zip_x64_cuda_jax435] | 5 | complete_five_seeds | 0.00381521 ± 0.000815151 | 0.996418 ± 0.000421595 | kl: differs; ess_ratio: matches |
| 8f0eda66316ae598 | nvidia7 | table torus2 rcpm 1.0 [zip_x64_cuda_jax435] | 5 | complete_five_seeds | 0.879901 ± 0.0286128 | 0.617496 ± 0.0164906 | kl: differs; ess_ratio: differs |

## Completeness

Read 30 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

Overall row coverage: 6/6 rows have a complete five-seed group.
Missing rows across the provided groups: none.

Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.

| Row | Complete native groups | Complete configured groups |
|---|---|---|
| S2:RNOT_FPS | none | f93bd4238029889f |
| S2:RNOT_RND | none | d2b0bd2d89b35fc4 |
| S2:RCPM_gamma1 | cdc40555dcd2babe | none |
| T2:RNOT_FPS | none | a42e85edcd0849b4 |
| T2:RNOT_RND | none | 9cc4e7180b4edd02 |
| T2:RCPM_gamma1 | 8f0eda66316ae598 | none |

Group provenance for row coverage:

| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |
|---|---|---|---|---|---|---:|---|
| S2:RNOT_FPS | f93bd4238029889f | nvidia7 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 425ed50554812ec5 | 5 | complete_five_seeds |
| S2:RNOT_RND | d2b0bd2d89b35fc4 | nvidia7 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 425ed50554812ec5 | 5 | complete_five_seeds |
| S2:RCPM_gamma1 | cdc40555dcd2babe | nvidia7 | zip_x64_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | bb0f06d960e2e744 | 5 | complete_five_seeds |
| T2:RNOT_FPS | a42e85edcd0849b4 | nvidia7 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 425ed50554812ec5 | 5 | complete_five_seeds |
| T2:RNOT_RND | 9cc4e7180b4edd02 | nvidia7 | zip_historical_settings_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 425ed50554812ec5 | 5 | complete_five_seeds |
| T2:RCPM_gamma1 | 8f0eda66316ae598 | nvidia7 | zip_x64_cuda_jax435 | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | bb0f06d960e2e744 | 5 | complete_five_seeds |

Environment `425ed50554812ec5`: host `nvidia7`, label `zip_historical_settings_cuda_jax435`.
Complete native table rows: none.
Complete configured table rows: S2:RNOT_FPS, S2:RNOT_RND, T2:RNOT_FPS, T2:RNOT_RND.
Missing rows in this environment across native/configured groups: S2:RCPM_gamma1, T2:RCPM_gamma1.
Missing native-default-only rows in this environment: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND.

Environment `bb0f06d960e2e744`: host `nvidia7`, label `zip_x64_cuda_jax435`.
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

### f93bd4238029889f

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.02072790187323534, 0.030345056031174034, 0.025502257204180128, 0.028512596760487778, 0.02970888659375474]; sample SE: 0.0017662.
ESS raw run means: [0.978000623509868, 0.9643418971228648, 0.9734337069576553, 0.9473359682762759, 0.9623837910980046]; sample SE: 0.00528862.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_ours_S2_seed12345_zip_historical_settings_cuda_jax435_2411dbc94059/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_ours_S2_seed23456_zip_historical_settings_cuda_jax435_e3b69b1f0fd1/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_ours_S2_seed34567_zip_historical_settings_cuda_jax435_441b57612e45/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_ours_S2_seed45678_zip_historical_settings_cuda_jax435_061e8e18efc4/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_ours_S2_seed56789_zip_historical_settings_cuda_jax435_6d7d390c678e/result.json)

### d2b0bd2d89b35fc4

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.03931675014307971, 0.04385364818949553, 0.041628595784698554, 0.04217987061089966, 0.05373744762201053]; sample SE: 0.00250605.
ESS raw run means: [0.9513898505374285, 0.9628407489286085, 0.9435215377865658, 0.9411334940178294, 0.9416819655430644]; sample SE: 0.00411581.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_ours_S2_seed12345_zip_historical_settings_cuda_jax435_ae44af77b0c9/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_ours_S2_seed23456_zip_historical_settings_cuda_jax435_5615df2eb2e3/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_ours_S2_seed34567_zip_historical_settings_cuda_jax435_0b45516f8cfc/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_ours_S2_seed45678_zip_historical_settings_cuda_jax435_e3da3b789275/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_ours_S2_seed56789_zip_historical_settings_cuda_jax435_8741bd6f228e/result.json)

### a42e85edcd0849b4

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.12069804356254883, 0.11743133411632918, 0.12949083772595119, 0.15706484293678102, 0.11219089498001174]; sample SE: 0.00793719.
ESS raw run means: [0.7723879863960003, 0.9334150336421725, 0.907306362023977, 0.9046304319023678, 0.9510848116996856]; sample SE: 0.0315369.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fc4a448fa84b/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_035145fdc01d/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_847d5106879f/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_45c22dbf9443/result.json)

### 9cc4e7180b4edd02

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.12672628200580083, 0.15704639330917355, 0.23984221994584703, 0.3531487003752704, 0.2375890056641531]; sample SE: 0.0394041.
ESS raw run means: [0.7038575790571031, 0.7156940710242181, 0.8596872682077119, 0.791128790423546, 0.8933616299977043]; sample SE: 0.037713.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fa120bdddebe/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_7d0f21afb99d/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_ddbac3d03fab/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_4a181a372502/result.json)

### cdc40555dcd2babe

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.006768577020761767, 0.0025545824345464796, 0.00215126763703543, 0.00247208299369579, 0.005129556549095912]; sample SE: 0.000911366.
ESS raw run means: [0.9965136084187709, 0.9960934175261322, 0.9980802354909859, 0.9962191078664941, 0.9951824885243342]; sample SE: 0.000471358.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_rcpm_S2_seed12345_zip_x64_cuda_jax435_dd37783d77d1/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_rcpm_S2_seed23456_zip_x64_cuda_jax435_755e8cfe6fa5/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_rcpm_S2_seed34567_zip_x64_cuda_jax435_a0af9098c89f/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_rcpm_S2_seed45678_zip_x64_cuda_jax435_ffab489bc864/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_rcpm_S2_seed56789_zip_x64_cuda_jax435_8cf0fb324f83/result.json)

### 8f0eda66316ae598

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.90038261216891, 0.9323565837185551, 0.8504139578269708, 0.9461599663113092, 0.7701899055616892]; sample SE: 0.0319901.
ESS raw run means: [0.595137276527734, 0.5979921276273072, 0.669902713430675, 0.5727519704961646, 0.6516972756729679]; sample SE: 0.0184371.

- Seed 12345: [complete](../../results/restored_zip_jax435/table_rcpm_T2_seed12345_zip_x64_cuda_jax435_563daa63d810/result.json)

- Seed 23456: [complete](../../results/restored_zip_jax435/table_rcpm_T2_seed23456_zip_x64_cuda_jax435_7a545acb8856/result.json)

- Seed 34567: [complete](../../results/restored_zip_jax435/table_rcpm_T2_seed34567_zip_x64_cuda_jax435_6864300643c7/result.json)

- Seed 45678: [complete](../../results/restored_zip_jax435/table_rcpm_T2_seed45678_zip_x64_cuda_jax435_87fe3a67b19f/result.json)

- Seed 56789: [complete](../../results/restored_zip_jax435/table_rcpm_T2_seed56789_zip_x64_cuda_jax435_e9a985af4a97/result.json)
