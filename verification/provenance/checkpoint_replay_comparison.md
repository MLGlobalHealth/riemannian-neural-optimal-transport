# Common checkpoint replay on both original NVIDIA hosts

Both hosts reproduce all five original evaluation batches exactly: 15 native scalar values per host (KL, ESS, and ESS ratio), checked directly against the original result JSON. The replay uses unchanged native `compute_kl_ours`, the same saved checkpoint bytes and recorded evaluation subkeys, without training.

| Original host replay | Mean KL | Mean ESS ratio | Exact original scalar matches |
| --- | ---: | ---: | ---: |
| nvidia7 | 0.12069804356255 | 0.772387986396 | 15 / 15 |
| nvidia6 | 0.12129077546086 | 0.94488334109459 | 15 / 15 |

The difference is reproduced during evaluation across these two recorded NVIDIA host environments. The common saved training state cannot explain the difference in this case. The two replays also have identical restored parameter/landmark fingerprints, recorded package versions, effective JAX settings, configuration, and evaluation keys.

nvidia7 uses an RTX 6000 Ada GPU with driver 570.195.03; nvidia6 uses an RTX 3090 with driver 550.90.07. Sample generation, compiler behavior, GPU architecture, drivers, system libraries and their interactions were not isolated. This is not an isolated AMD-versus-NVIDIA comparison.

Common checkpoint SHA256: `358939c83c3cf560dcb1ac4cc096d15b61a8e8413586f8cb4bcfebe6894e8c8f`. Evaluator SHA256: `fb500e51ed9aa2da9d42d78363e9ccdb44fdf67cf3b2cf33d16d42d47c9f168c`. Source manifest SHA256: `5205da3999ed08aa6e58e0201096d77bad38a85a02525c5443b4728b4e07a9c0`. [Full hashes, host environments, exact scalar comparisons and batch values](checkpoint_replay_comparison.json).

- nvidia7: [original result](../results/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json), [unchanged replay result](../results/checkpoint_replay/nvidia7/evaluation.json), [locally retained replay log](../artifacts/checkpoint_replay/nvidia7/evaluation.log).
- nvidia6: [original result](../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json), [unchanged replay result](../results/checkpoint_replay/nvidia6/evaluation.json), [locally retained replay log](../artifacts/checkpoint_replay/nvidia6/evaluation.log).

These two evaluation-only records are excluded from independent training-run counts.
