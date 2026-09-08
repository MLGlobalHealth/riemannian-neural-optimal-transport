# T2 FPS: five independent seeds on each NVIDIA host

Each host row contains exactly seeds 12345, 23456, 34567, 45678 and 56789 in one source/configuration/label/environment group. Host rows are aggregated separately; each seed contributes its native mean over five held-out batches.

| Host | KL mean ± SE (ddof0) | ESS ratio mean ± SE (ddof0) | Paper KL rounding | Paper ESS rounding |
| --- | ---: | ---: | --- | --- |
| nvidia7 | 0.127375191 ± 0.0070992 | 0.893764925 ± 0.028207 | matches | differs |
| nvidia6 | 0.126792155 ± 0.0070617 | 0.926915926 ± 0.0098871 | matches | matches |

Paper Table 2 reports FPS KL 0.13 ± 0.01 and ESS ratio 0.93 ± 0.01. The rounding comparisons check the printed means only. The paper does not specify the confidence level or construction of its intervals; the SE values above follow the shipped across-seed calculation.

| Seed | Checkpoint bytes identical across hosts | Five evaluation keys identical |
| ---: | --- | --- |
| 12345 | True | True |
| 23456 | True | True |
| 34567 | True | True |
| 45678 | True | True |
| 56789 | True | True |

[Complete per-seed values, ddof0/sample SE, timing, result/checkpoint hashes, host identities and paper comparisons](t2_fps_five_seed_host_comparison.json). The [FPS seed 12345 evaluation-only replay](checkpoint_replay_comparison.md) separately checks original-host reproduction of one common saved checkpoint; no other checkpoint was replayed.

The comparison preserves every prescribed seed and does not pool hosts. GPU architecture, drivers, OS and Python builds differ between these recorded host environments; this does not isolate an AMD-versus-NVIDIA effect. Timings are included descriptively in JSON and are not claimed to reproduce AMD timings.
