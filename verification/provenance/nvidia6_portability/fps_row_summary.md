# Full nvidia6 T2 FPS portability row

All five prescribed training seeds completed in one unchanged source/settings/package/host group. The aggregate means round to the paper’s displayed KL 0.13 and ESS ratio 0.93.

| Metric | Five-seed mean | SE across training seeds (ddof0) | Paper displayed mean |
|---|---:|---:|---:|
| KL | 0.126792154795 | 0.007061703544 | 0.13 |
| ESS ratio | 0.926915925560 | 0.009887074036 | 0.93 |

These errors describe variation across five training seeds, distinct from each run’s five-batch Monte Carlo errors. Paper interval construction is not specified; the displayed-mean match is a numerical rounding comparison.

| Seed | KL | ESS ratio | Native result |
|---|---:|---:|---|
| 12345 | 0.121290775461 | 0.944883341095 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json) |
| 23456 | 0.117087127515 | 0.934686108017 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fc4a448fa84b/result.json) |
| 34567 | 0.127862506692 | 0.890643895482 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_035145fdc01d/result.json) |
| 45678 | 0.156456409934 | 0.913615632713 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_847d5106879f/result.json) |
| 56789 | 0.111263954373 | 0.950750650494 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_45c22dbf9443/result.json) |

The exact 80 frozen package versions, source22 hashes, x64 setting and native Threefry setting match the supported nvidia7 stack. All five FPS runs here use nvidia6 RTX 3090, driver 550.90.07. Host and driver differ from nvidia7 RTX 6000 Ada, driver 570.195.03; this does not isolate GPU architecture. No host results are pooled.

The [initial capacity decision](capacity_rebalance_decision.json) was made before either first-seed T2 result. Later, the [full FPS manifest](fps_row_completion_manifest.json) retained seed 12345 and declared every remaining prescribed FPS seed in response to the paired-host ESS difference, superseding only the FPS cancellations. The FPS extension contained no additional random training; a later [random-row declaration](random_row_completion_manifest.json) now covers every remaining prescribed random seed.

Twelve native records are delivered: this full FPS row, the subsequently completed five-seed random row, and one S2 and T2 RCPM check. All raw JSON, checkpoint and log hashes match remote evidence; the 22 source files and frozen inputs remain unchanged. See [delivery checksums](delivery_checksums.json) and the [paired-host report](paired_comparison.md). The separate checkpoint replay reproduced all five original nvidia6 batches exactly and is excluded from training-seed counts.
