# Full nvidia6 T2 random portability row

All five prescribed training seeds completed in one unchanged source/settings/package/host group. The aggregate KL matches the paper’s displayed mean; ESS ratio does not.

| Metric | Five-seed mean | SE across training seeds (ddof0) | Paper displayed mean |
|---|---:|---:|---:|
| KL | 0.222997629603 | 0.034928834154 | 0.22 |
| ESS ratio | 0.796680475076 | 0.032787810612 | 0.85 |

KL rounds to 0.22 and ESS ratio rounds to 0.80. No seeds or batches are omitted. Across-seed SE is separate from each run’s five-batch Monte Carlo uncertainty; the paper does not specify its interval construction.

| Seed | KL | ESS ratio | Native record |
|---|---:|---:|---|
| 12345 | 0.127321133231 | 0.681479605834 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/result.json) |
| 23456 | 0.158096156971 | 0.888591665699 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed23456_zip_historical_settings_cuda_jax435_fa120bdddebe/result.json) |
| 34567 | 0.240119341220 | 0.860670484928 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed34567_zip_historical_settings_cuda_jax435_7d0f21afb99d/result.json) |
| 45678 | 0.351959451940 | 0.787149876015 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed45678_zip_historical_settings_cuda_jax435_ddbac3d03fab/result.json) |
| 56789 | 0.237492064653 | 0.765510742903 | [JSON](../../results/nvidia6_portability_jax435/table_ours_T2_seed56789_zip_historical_settings_cuda_jax435_4a181a372502/result.json) |

The [random-row prelaunch manifest](random_row_completion_manifest.json) retained seed 12345 and explicitly declared all four remaining prescribed seeds, superseding the earlier random capacity cancellations. No source/settings/version changes, sphere training or additional baseline jobs were part of this extension.

All 12 native records, checkpoints and logs are delivered with verified hashes. The [final release confirmation](twelve_check_release_confirmation.json) confirms all final workers/supervisors/collector exited, GPUs 0–3 were empty, and all 80 frozen package versions were unchanged. See [delivery checksums](delivery_checksums.json), the [complete FPS row](fps_row_summary.md), and the [paired host comparison](paired_comparison.md). No results are pooled across hosts.
