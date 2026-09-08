# T2 seed 12345: recorded training trajectories

This compares the first restored ZIP FPS/random runs with the same-seed sections of committed `a7327bc` AMD logs. Losses are compared exactly at their printed four-decimal precision.

| Pair | Landmarks | Matching losses / 1000 | First differing step | Largest printed absolute difference |
|---|---|---:|---:|---:|
| Committed AMD / nvidia7 | fps | 671 | 12 | 0.0005 |
| nvidia7 / nvidia6 | fps | 1000 | none | 0.0000 |
| Committed AMD / nvidia7 | random | 448 | 12 | 0.0008 |
| nvidia7 / nvidia6 | random | 1000 | none | 0.0000 |

| Run | Landmarks | Final KL | Final ESS ratio |
|---|---|---:|---:|
| committed_amd | fps | 0.1215 | 0.951 |
| nvidia7 | fps | 0.120698043563 | 0.772387986396 |
| nvidia6 | fps | 0.121290775461 | 0.944883341095 |
| committed_amd | random | 0.1316 | 0.69 |
| nvidia7 | random | 0.126726282006 | 0.703857579057 |
| nvidia6 | random | 0.127321133231 | 0.681479605834 |

Evidence sections:

- committed_amd, fps: [table_fps.log](../artifacts/committed_amd_logs_a7327bc/table_fps.log#L5150), lines 5150–6158; SHA256 `da31a46875cd445e2dbe438612033ef52de9264a1c815f87b22c34d325d4fe5d`.
- nvidia7, fps: [table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9.log](../artifacts/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/training.log#L1), lines 1–1013; SHA256 `6bec61b50c13e34f4f2512315f85e777dbcaf0eb476c4eb9bec15fc85d803067`.
- nvidia6, fps: [torus_fps_seed12345.log](../artifacts/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/training.log#L1), lines 1–1013; SHA256 `c93a0b7561854e402f34d21d4bafd36c517836bd683ab90a0a03dc61a17a8594`.
- committed_amd, random: [table_random.log](../artifacts/committed_amd_logs_a7327bc/table_random.log#L5130), lines 5130–6134; SHA256 `3f625f12817470696d9cd8c8064b1bf13729d9aae2075e57572aa173ec84e8cc`.
- nvidia7, random: [table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f.log](../artifacts/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/training.log#L1), lines 1–1009; SHA256 `0795133dd7b72929dd04360735eafc5675c73a8628cd2d8a8b4e209534552381`.
- nvidia6, random: [torus_random_seed12345.log](../artifacts/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/training.log#L1), lines 1–1009; SHA256 `578857f27f4ff646321c1f9d4bb488cd972838b632081e6b22b9f5c1309fb95d`.

All per-step printed values, first-divergence values/line references, final result hashes and environment/configuration records are retained in [the JSON comparison](t2_seed12345_trajectory_comparison.json).

Equal printed losses do not establish identical parameters; differing losses show differences already appear during training. This comparison does not isolate GPU vendor or identify the numerical cause. The final values above belong to one seed, not five-seed paper means.

Full training logs are retained locally under ignored `verification/artifacts/`; the raw printed loss values and comparison summary are committed in the JSON evidence.
