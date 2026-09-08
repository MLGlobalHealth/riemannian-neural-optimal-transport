# Both torus RNOT rows across the two NVIDIA hosts

All four host/method groups contain the same five prescribed training seeds. Each group is aggregated separately; all 20 runs are retained. The ten paired checkpoint files are byte-for-byte identical across hosts, and each pair has the same five evaluation keys.

| Method | Host | KL mean ± shipped SE | ESS ratio mean ± shipped SE |
| --- | --- | ---: | ---: |
| FPS | nvidia6 | 0.12679215 ± 0.00706170 | 0.92691593 ± 0.00988707 |
| FPS | nvidia7 | 0.12737519 ± 0.00709924 | 0.89376493 ± 0.02820744 |
| random | nvidia6 | 0.22299763 ± 0.03492883 | 0.79668048 ± 0.03278781 |
| random | nvidia7 | 0.22287052 ± 0.03524407 | 0.79274587 ± 0.03373153 |

On RTX 3090, FPS means and shipped standard errors round to the paper’s displayed KL/ESS values. Random KL rounds to 0.22 on both hosts, while random ESS rounds to 0.80 on RTX 3090 and 0.79 on RTX 6000 Ada, compared with the paper’s 0.85. Thus the second host does not reproduce every torus metric at the paper’s printed precision.

[FPS: raw seeds, batch values, hashes and paper checks](t2_fps_five_seed_host_comparison.md); [random: corresponding complete evidence](t2_random_five_seed_host_comparison.md); [combined machine-readable comparison](t2_five_seed_host_comparison.json).

The saved states agree across hosts for all ten paired models. A separate [FPS seed 12345 checkpoint replay](checkpoint_replay_comparison.md) reproduced all 15 original native scalar outputs exactly on each original host, confirming repeatable evaluation sensitivity for that case. Other checkpoints were compared as saved files without additional replay.

GPU architecture, drivers, OS/system libraries and Python builds differ between these host environments; sampling and compiler contributions were not isolated. Neither host is AMD. The paper does not specify the construction or confidence level of its intervals, so matching printed values is descriptive rather than statistical equivalence. Timings are retained separately and are not claimed to match AMD timings.
