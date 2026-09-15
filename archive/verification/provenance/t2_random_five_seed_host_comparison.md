# T2 random: five independent seeds on each NVIDIA host

Each host row contains exactly seeds 12345, 23456, 34567, 45678 and 56789 in one source/configuration/label/environment group. Host rows are aggregated separately; each seed contributes its native mean over five held-out batches.

| Host | KL mean ± SE (ddof0) | ESS ratio mean ± SE (ddof0) | Paper KL rounding | Paper ESS rounding |
| --- | ---: | ---: | --- | --- |
| nvidia7 | 0.22287052 ± 0.035244 | 0.792745868 ± 0.033732 | matches | differs |
| nvidia6 | 0.22299763 ± 0.034929 | 0.796680475 ± 0.032788 | matches | differs |

Paper Table 2 reports random KL 0.22 ± 0.04 and ESS ratio 0.85 ± 0.02. The rounding comparisons check the printed means only. The paper does not specify the confidence level or construction of its intervals; the SE values above follow the shipped across-seed calculation.

| Seed | Checkpoint bytes identical across hosts | Five evaluation keys identical |
| ---: | --- | --- |
| 12345 | True | True |
| 23456 | True | True |
| 34567 | True | True |
| 45678 | True | True |
| 56789 | True | True |

[Complete per-seed values, ddof0/sample SE, timing, result/checkpoint hashes, host identities and paper comparisons](t2_random_five_seed_host_comparison.json). The [FPS seed 12345 evaluation-only replay](checkpoint_replay_comparison.md) separately checks original-host reproduction of one common saved checkpoint; no other checkpoint was replayed.

The comparison preserves every prescribed seed and does not pool hosts. GPU architecture, drivers, OS and Python builds differ between these recorded host environments; this does not isolate an AMD-versus-NVIDIA effect. Timings are included descriptively in JSON and are not claimed to reproduce AMD timings.
