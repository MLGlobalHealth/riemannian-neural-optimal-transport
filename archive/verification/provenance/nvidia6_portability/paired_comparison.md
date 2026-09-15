# Same-seed NVIDIA host portability checks

Same package versions, source and settings on nvidia6 RTX 3090 with driver 550.90.07 versus nvidia7 RTX 6000 Ada with driver 570.195.03. Host and driver differ; this is not an isolated GPU-architecture, GPU-vendor or AMD experiment.

All four records use seed 12345 and five unchanged native evaluation batches. These paired checks are separate from the nvidia7 five-seed main table.

| Check | nvidia6 KL | nvidia7 KL | nvidia6 ESS ratio | nvidia7 ESS ratio |
|---|---:|---:|---:|---:|
| [T2_RNOT_FPS](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json) | 0.121290775461 | 0.120698043563 | 0.944883341095 | 0.772387986396 |
| [T2_RNOT_random](../../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_a3b0d254ed4f/result.json) | 0.127321133231 | 0.126726282006 | 0.681479605834 | 0.703857579057 |
| [S2_RCPM](../../results/nvidia6_portability_jax435/table_rcpm_S2_seed12345_zip_x64_cuda_jax435_dd37783d77d1/result.json) | 0.006768577021 | 0.006768577021 | 0.996513608419 | 0.996513608419 |
| [T2_RCPM](../../results/nvidia6_portability_jax435/table_rcpm_T2_seed12345_zip_x64_cuda_jax435_563daa63d810/result.json) | 0.900382612169 | 0.900382612169 | 0.595137276528 | 0.595137276528 |

Every declared source/configuration/selection/evaluation field compared in `paired_comparison.json` matches. Both T2 RNOT runs match all 1000 nvidia7 losses at printed precision; their final native ESS ratios differ. The two RCPM native metric summaries match exactly. All four paired serialized checkpoints have identical SHA256 hashes across hosts. These observations do not isolate a hardware cause.

All four raw JSON files, checkpoints and training logs were verified against remote SHA256 records. The 22 source files and original inputs are unchanged. All four initial workers and supervisors exited, and GPUs 4/5 were empty after these checks. The original ten-job manifest, the earlier capacity-only cancellations, and the two added RCPM job manifest are retained.

A subsequent separately declared expansion at 11:59:15 UTC launches all four remaining prescribed T2 FPS seeds on nvidia6 GPUs 0–3. Its manifest records the paired ESS difference as the rationale and supersedes only the FPS part of the earlier capacity decision. The first FPS seed is retained to form a complete five-seed nvidia6 row. No additional random training is authorized by that manifest. Those four runs subsequently completed; the [full five-seed FPS report](fps_row_summary.md) gives KL 0.126792154795 and ESS ratio 0.926915925560. This paired document still compares only the common first seed across hosts.

The [initial manifest](initial_t2_manifest.json), [capacity-only decision](capacity_rebalance_decision.json), [two RCPM checks](rcpm_portability_manifest.json), and [later complete FPS row manifest](fps_row_completion_manifest.json) retain the scheduling chronology. The latter supersedes only the FPS cancellations. [Environment verification](environment_verification.json) confirms all 80 frozen package versions match the supported candidate. [Delivery checksums](delivery_checksums.json) record the currently completed files; all 12 native records are now complete and verified.

Both five-seed torus rows have now completed: [FPS](fps_row_summary.md) matches the displayed paper means, while [random](random_row_summary.md) matches KL but not ESS ratio. The original four paired checks above remain unchanged.
