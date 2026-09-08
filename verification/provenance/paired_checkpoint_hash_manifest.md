# Four paired saved checkpoints across NVIDIA hosts

All four seed 12345 pairs have byte-for-byte identical checkpoint files and the same five recorded evaluation subkeys. Each local file hash matches its original result record. Source, effective configuration, evaluation configuration, label, and verifier hash also agree within each pair.

| Pair | Checkpoint SHA256 | Bytes | Native KL: nvidia7 / nvidia6 | Native ESS ratio: nvidia7 / nvidia6 |
| --- | --- | ---: | ---: | ---: |
| T2 RNOT fps | `358939c83c3cf560dcb1ac4cc096d15b61a8e8413586f8cb4bcfebe6894e8c8f` | 539794 | 0.120698043563 / 0.121290775461 | 0.772387986396 / 0.944883341095 |
| T2 RNOT random | `4e7a6f657bd2ca9076f83a218b54a73e0c408d50c1b2e3a96a0607e0fbcc8543` | 539794 | 0.126726282006 / 0.127321133231 | 0.703857579057 / 0.681479605834 |
| S2 RCPM | `91f4cfaf352ecf15581b4439c2f50d4a1d81016f77bb1edfa5acc1d4f55a3bad` | 9944 | 0.00676857702076 / 0.00676857702076 | 0.996513608419 / 0.996513608419 |
| T2 RCPM | `5d8488e817b767e4a42c39021afea4eb1a7fb3f816ca892e6ad35aeef6864654` | 12664 | 0.900382612169 / 0.900382612169 | 0.595137276528 / 0.595137276528 |

[Machine-readable hashes, original result links, and all recorded keys](paired_checkpoint_hash_manifest.json) preserve each host and environment separately. Full checkpoint files remain in ignored local artifacts. The [T2 FPS array comparison](t2_fps_saved_checkpoint_comparison.md) additionally decoded that pair on CPU; the other three pairs use exact serialized bytes, without further probes.

These comparisons establish equality of the saved checkpoint files and recorded inputs used for evaluation. The different original output metrics therefore cannot be explained by differing saved checkpoint bytes or evaluation subkeys. Identical generated sample arrays, compiled computation, process state, and driver behavior have not been established by this file comparison. A separate evaluation-only replay of the common T2 FPS checkpoint checks whether each original host reproduces its recorded five batches.
