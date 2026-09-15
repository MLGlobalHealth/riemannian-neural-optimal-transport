# Experimental verification comparison

Generated: 2026-09-08T12:59:05.781363+00:00

Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.

These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.

| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |
|---|---|---|---:|---|---|---|---|---|
| 2fb86a384def923b | nvidia7 | table sphere2 rcpm 1.0 [zip_defaults] | 5 | complete_five_seeds | 0.0085077 ± 0.000788035 | 0.996008 ± 0.0005016 | kl: differs; ess_ratio: matches |
| 831431cdc70bd86b | nvidia7 | table sphere2 ours fps [zip_defaults] | 5 | complete_five_seeds | 0.0396821 ± 0.00394427 | 0.89982 ± 0.0293001 | kl: differs; ess_ratio: differs |
| 5e85c2d921f8a75f | nvidia7 | table sphere2 ours random [zip_defaults] | 5 | complete_five_seeds | 0.0678439 ± 0.00687402 | 0.763097 ± 0.0731542 | kl: differs; ess_ratio: differs |
| 55d845eb2a613a41 | nvidia7 | table torus2 rcpm 1.0 [zip_defaults] | 5 | complete_five_seeds | 0.93351 ± 0.0434081 | 0.545083 ± 0.0283626 | kl: matches; ess_ratio: matches |
| a3230348aa48d1ca | nvidia7 | table torus2 ours fps [zip_defaults] | 5 | complete_five_seeds | 0.425912 ± 0.0175998 | 0.793395 ± 0.0341689 | kl: differs; ess_ratio: differs |
| 8b8657bc978c1f72 | nvidia7 | table torus2 ours random [zip_defaults] | 5 | complete_five_seeds | 0.471589 ± 0.0233457 | 0.700423 ± 0.0389923 | kl: differs; ess_ratio: differs |

## Completeness

Read 30 result files; excluded 0 smoke runs, deduplicated 0 identical copies, and encountered 0 read errors.

Overall row coverage: 6/6 rows have a complete five-seed group.
Missing rows across the provided groups: none.

Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.

| Row | Complete native groups | Complete configured groups |
|---|---|---|
| S2:RNOT_FPS | 831431cdc70bd86b | none |
| S2:RNOT_RND | 5e85c2d921f8a75f | none |
| S2:RCPM_gamma1 | 2fb86a384def923b | none |
| T2:RNOT_FPS | a3230348aa48d1ca | none |
| T2:RNOT_RND | 8b8657bc978c1f72 | none |
| T2:RCPM_gamma1 | 55d845eb2a613a41 | none |

Group provenance for row coverage:

| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |
|---|---|---|---|---|---|---:|---|
| S2:RNOT_FPS | 831431cdc70bd86b | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |
| S2:RNOT_RND | 5e85c2d921f8a75f | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |
| S2:RCPM_gamma1 | 2fb86a384def923b | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |
| T2:RNOT_FPS | a3230348aa48d1ca | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |
| T2:RNOT_RND | 8b8657bc978c1f72 | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |
| T2:RCPM_gamma1 | 55d845eb2a613a41 | nvidia7 | zip_defaults | 5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0 | 209529b2ecf1cc9d | 5 | complete_five_seeds |

Environment `209529b2ecf1cc9d`: host `nvidia7`, label `zip_defaults`.
Complete native table rows: S2:RCPM_gamma1, S2:RNOT_FPS, S2:RNOT_RND, T2:RCPM_gamma1, T2:RNOT_FPS, T2:RNOT_RND.
Complete configured table rows: none.
Missing rows in this environment across native/configured groups: none.
Missing native-default-only rows in this environment: none.

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

### 2fb86a384def923b

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.008967555686831474, 0.006203321111388505, 0.008052751398645341, 0.007766315760090947, 0.01154854502528906]; sample SE: 0.00088105.
ESS raw run means: [0.9970454096794128, 0.9973085522651672, 0.9961712718009949, 0.9943304419517517, 0.9951824426651001]; sample SE: 0.000560806.

- Seed 12345: [complete](../../results/zip_defaults/table_rcpm_S2_seed12345_zip_defaults_0d945495fc09/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_rcpm_S2_seed23456_zip_defaults_0a5584d68448/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_rcpm_S2_seed34567_zip_defaults_bf35bb541692/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_rcpm_S2_seed45678_zip_defaults_29ad38d15ee6/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_rcpm_S2_seed56789_zip_defaults_715416696674/result.json)

### 831431cdc70bd86b

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.039093632251024246, 0.03290243446826935, 0.029744068160653114, 0.05519462674856186, 0.041475965082645415]; sample SE: 0.00440982.
ESS raw run means: [0.965373158454895, 0.9618959069252014, 0.8951546669006347, 0.7848122358322144, 0.8918651342391968]; sample SE: 0.0327586.

- Seed 12345: [complete](../../results/zip_defaults/table_ours_S2_seed12345_zip_defaults_769765e2d52f/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_ours_S2_seed23456_zip_defaults_1b0b06fa631d/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_ours_S2_seed34567_zip_defaults_2a38f312eac5/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_ours_S2_seed45678_zip_defaults_94dbb7102a5d/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_ours_S2_seed56789_zip_defaults_835103629fb0/result.json)

### 5e85c2d921f8a75f

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.04926144778728485, 0.06490588262677192, 0.06626407578587531, 0.09609814286231995, 0.06268976852297783]; sample SE: 0.00768538.
ESS raw run means: [0.9238246440887451, 0.8636424779891968, 0.7394968390464782, 0.45845919065177443, 0.8300629496574402]; sample SE: 0.0817889.

- Seed 12345: [complete](../../results/zip_defaults/table_ours_S2_seed12345_zip_defaults_7417ab3c8c6d/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_ours_S2_seed23456_zip_defaults_e4684d6e775e/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_ours_S2_seed34567_zip_defaults_90ad28ff6a57/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_ours_S2_seed45678_zip_defaults_8c6b393f4297/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_ours_S2_seed56789_zip_defaults_ccb538400cea/result.json)

### 55d845eb2a613a41

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [1.0049230933189393, 0.7726993918418884, 1.0195032596588134, 0.8690840482711792, 1.0013421297073364]; sample SE: 0.0485318.
ESS raw run means: [0.5412155389785767, 0.6155619144439697, 0.6026856899261475, 0.5282299518585205, 0.43772080540657043]; sample SE: 0.0317104.

- Seed 12345: [complete](../../results/zip_defaults/table_rcpm_T2_seed12345_zip_defaults_95923f3cebeb/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_rcpm_T2_seed23456_zip_defaults_3426d08ce645/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_rcpm_T2_seed34567_zip_defaults_8f9d82f2563c/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_rcpm_T2_seed45678_zip_defaults_8c04b6f4d9b3/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_rcpm_T2_seed56789_zip_defaults_97ba2753c811/result.json)

### a3230348aa48d1ca

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.3657603144645691, 0.43679836988449094, 0.46541803479194643, 0.4654227375984192, 0.39615935683250425]; sample SE: 0.0196771.
ESS raw run means: [0.8314434170722962, 0.8064245581626892, 0.8513123512268066, 0.6433044325560331, 0.8344895839691162]; sample SE: 0.038202.

- Seed 12345: [complete](../../results/zip_defaults/table_ours_T2_seed12345_zip_defaults_f2f9a924eaf9/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_ours_T2_seed23456_zip_defaults_8f24e5513209/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_ours_T2_seed34567_zip_defaults_6b38c5f27d6a/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_ours_T2_seed45678_zip_defaults_3b69daa7bc32/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_ours_T2_seed56789_zip_defaults_1984dd02271e/result.json)

### 8b8657bc978c1f72

Seeds: [12345, 23456, 34567, 45678, 56789]; missing shipped seeds: [].

KL raw run means: [0.4220570147037506, 0.4843824923038483, 0.5282326757907867, 0.5232876062393188, 0.39998445510864256]; sample SE: 0.0261013.
ESS raw run means: [0.7972240090370178, 0.7258748173713684, 0.5773397624492645, 0.7809679269790649, 0.6207062713801861]; sample SE: 0.0435947.

- Seed 12345: [complete](../../results/zip_defaults/table_ours_T2_seed12345_zip_defaults_37c3ecb52ad2/result.json)

- Seed 23456: [complete](../../results/zip_defaults/table_ours_T2_seed23456_zip_defaults_043f35be1d9b/result.json)

- Seed 34567: [complete](../../results/zip_defaults/table_ours_T2_seed34567_zip_defaults_aff3fb3af70f/result.json)

- Seed 45678: [complete](../../results/zip_defaults/table_ours_T2_seed45678_zip_defaults_b1eb64252e9a/result.json)

- Seed 56789: [complete](../../results/zip_defaults/table_ours_T2_seed56789_zip_defaults_52314f1198b4/result.json)
