# Four fixed Lie-group models across two NVIDIA hosts

Eight evaluation-only records replay the same four saved models, with the original five keys, on nvidia6 (RTX 3090) and nvidia7 (RTX 6000 Ada). No model is retrained. Each source/checkpoint/configuration and restored array fingerprint is checked, and each recorded mean below is independently recomputed from its five batches.

| Model | Paper KL / ESS | RTX 3090 replay KL / ESS | RTX 6000 Ada replay KL / ESS | Original-host exact replay |
|---|---:|---:|---:|---|
| SO3 RNOT | 2.96 / 0.225 | [3.801323208 / 0.780428518](<../evaluations/nvidia6/SO3_ours_nativeJAX435.json>) | [3.801323208 / 0.780428518](<../evaluations/nvidia7/SO3_ours_nativeJAX435.json>) | yes |
| SO3 RCPM γ=1 | 2.86 / 0.391 | [2.820084209 / 0.775483452](<../evaluations/nvidia6/SO3_rcpm_gamma1_nativeJAX435.json>) | [2.820084209 / 0.775483452](<../evaluations/nvidia7/SO3_rcpm_gamma1_nativeJAX435.json>) | yes |
| SE3 RNOT | 2.50 / 0.683 | [2.405148242 / 0.670147436](<../evaluations/nvidia6/SE3_ours_nativeJAX435.json>) | [2.405148242 / 0.670147436](<../evaluations/nvidia7/SE3_ours_nativeJAX435.json>) | yes |
| SE3 RCPM γ=1 | 14.38 / 0.007 | [13.226344407 / 0.007267948](<../evaluations/nvidia6/SE3_rcpm_gamma1_nativeJAX435.json>) | [13.226344407 / 0.007267948](<../evaluations/nvidia7/SE3_rcpm_gamma1_nativeJAX435.json>) | yes |

The [machine-readable comparison](paired_replay_comparison.json) preserves every paired metric, key, and source link. The [evaluation manifest](evaluation_manifest.json) identifies the eight unchanged raw records. All eight records share the same frozen replay tool. Matching the original host empirically checks that restoring the saved model reproduces its original native evaluation.

These checks measure evaluation of fixed models across two recorded NVIDIA environments. They do not test training on the second host, isolate GPU hardware from driver/compiler/OS/Python-build effects, or reconstruct the original AMD runtime. Five evaluation batches remain evaluations of one fitted model, not five independent training repetitions.
