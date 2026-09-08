# Preserved research material

This folder preserves material outside the public paper experiment workflow. The active code, recipes and figure commands are documented in the repository root README. `tools/build_release.py` and Git source exports exclude this entire archive.

| Path | Contents |
|---|---|
| [README.md](README.md) | Previous branch README, frozen before packaging |
| [verification/](verification/) | Reproduction records, numerical comparisons, historical source snapshots and old audit tools |
| [website/](website/) | Local project page and its assets; publication remains on hold |
| `src/`, `rcpm/`, `experiments/` | Frozen pre-packaging native files, retained for historical hash checks and relative imports |
| [original/](original/) | Original notebooks and experiment outputs supporting the recovered plotting references |
| `release_checks/` | Packaging validation records, added with the release |

The frozen branch state is commit `6e6b7aa04a029aa6fcd3147e9c6b9a7ca2e6fa25`. Historical audit commands should be interpreted in that context; run old relative-path workflows from this directory, not from the new repository root. Historical records retain their original paths, dates and hashes rather than being rewritten as new experiments.

`verification/artifacts/` contains local ignored checkpoints and logs. These files remain on disk but are not added to Git or distributed. The website and previously committed verification records are moved without changing their bytes. The original main worktree is left untouched; supporting notebooks and logs here are copies.
