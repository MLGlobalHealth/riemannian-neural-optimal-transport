# T2 FPS seed 12345: saved state across NVIDIA hosts

The nvidia7 and nvidia6 checkpoints are byte-for-byte identical: 539,794 bytes, SHA256 `358939c83c3cf560dcb1ac4cc096d15b61a8e8413586f8cb4bcfebe6894e8c8f`. Both original result records report that same checkpoint hash.

CPU msgpack restoration independently confirms all eight parameter arrays (33,409 float32 values), all 128×4 float64 landmark coordinates, and the retained key match exactly. The maximum parameter and landmark absolute difference is zero. The serialized file identity also covers the stored optimizer state.

The five recorded evaluation subkeys and effective configuration also agree. This rules out differing saved state or subkeys for this paired FPS case. It does not establish identical generated samples, compiled computation or driver behavior; the next check replays the common checkpoint on its original host before comparison across hosts.

[nvidia7 result](../results/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json), [nvidia6 result](../results/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/result.json), [array/hash evidence](t2_fps_saved_checkpoint_comparison.json). Full checkpoints are retained locally in ignored [nvidia7 artifacts](../artifacts/restored_zip_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/checkpoint.msgpack) and [nvidia6 artifacts](../artifacts/nvidia6_portability_jax435/table_ours_T2_seed12345_zip_historical_settings_cuda_jax435_201d894144e9/checkpoint.msgpack).

This comparison decoded existing files on two CPU cores and did not evaluate a model or train.

The [four-pair checkpoint manifest](paired_checkpoint_hash_manifest.md) extends the serialized-state and evaluation-key check to T2 RNOT random and S2/T2 RCPM, without additional model probes.
