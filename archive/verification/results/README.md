# Recorded experiment outputs

Each `result.json` is an unchanged output from `verify_run.py`, including source and runner hashes, effective configuration, environment, all evaluation batches, and a summary.

The `paper_reproduction: false` field is the runner's initial assessment placeholder. Its adjacent note explicitly delegates numerical agreement to the separate comparison step. Consult [the report](../REPORT.md) and generated comparisons for the findings.

The result groups retain distinct protocols:

- `restored_zip_jax435/`: the prescribed 30-run ZIP main-table cohort on `nvidia7`, with recovered settings, x64, and the supported older CUDA environment.
- `nvidia6_portability_jax435/`: same-software RTX 3090 checks, including both complete five-seed torus FPS/random rows. The two RCPM rows in this group are single-seed checks.
- `zip_defaults/`: the initial complete 30-run ZIP-default suite under JAX 0.6.2 and float32 defaults.
- `checkpoint_replay/`: two evaluation-only records replaying the common saved model on the original hosts; excluded from independent training-run counts.
- Other directories: historical-settings and exact historical-source trials in separately recorded environments, plus retained initialization failures.

These groups must not be pooled. Checkpoint replay outputs are evaluation-only evidence and never independent training runs.

Checkpoints and original training logs are retained in the local, ignored `verification/artifacts/` directory and isolated server directories. Their checksums are recorded in the JSON files. Committed JSON records contain the original server paths for provenance; use the root README's commands with paths appropriate to your checkout when rerunning.
