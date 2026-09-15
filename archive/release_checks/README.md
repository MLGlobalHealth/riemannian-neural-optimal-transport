# Paper-package validation

The release keeps the available experiment code, explicit recipes, data, plotting references and documentation at the repository root. Research history, earlier audits, notebooks, website and development checks are archived. The original main worktree and its untracked files remain unchanged. No remote branch or website was published.

The standalone ZIP contains **59 files (2,129,123 bytes)**, including a file-by-file checksum manifest. Its SHA256 is `2665f27bed3a0879d8e36bcb0ac78b2d2421607640e5e28a3f09f9600e1b5502`. Build it from the repository root with:

```sh
python tools/build_release.py --output dist/riemannian-neural-optimal-transport-paper.zip
```

Choose a new output filename if that artifact already exists. The archive is excluded both by the explicit ZIP builder and by Git export attributes.

Validation completed on 2026-09-08:

- All 20 public-workflow tests pass, including the relocated test layout. Run from the repository root: `python -m unittest discover -s archive/release_checks/tests -v`. The tests need the plotting dependencies; training is not part of the unit suite.
- The final ZIP was extracted into an independent temporary directory. All checksums, 33 Python modules, 14 local documentation links, five CLI help commands, all five suite plans and the continental CSV dry run pass. All six numerical reference plot modes produce PNG, PDF, CSV and provenance JSON without the archive. Details: [export-qa/summary.json](export-qa/summary.json).
- Seven bounded NVIDIA computations pass, including a full native SE(3) run, the full embedding computation, four smoke training runs and a fresh-process checkpoint visualization. The SE(3) run reproduces all 200 losses, five evaluation keys, 20 batch scalar values and checkpoint bytes from the earlier run. All owned jobs exited and GPU resources were released. Details: [GPU validation](gpu-validation/VALIDATION.md).
- All 23 active native files retain their pre-packaging bytes. The older broad `rcpm/requirements.txt` is preserved in the frozen 24-file archive mirror. All 873 moved audit/website files retain their original bytes, including ignored local artifacts. Supporting original notebooks/logs are copied without modifying the main worktree.

The GPU check used frozen numerical code. Subsequent changes were documentation, removal of an unused plotting-only SciPy requirement, and embedding legend/footnote layout. The final ZIP uses the same native library, recipes and numerical wrappers. The frozen GPU export is retained under `gpu-validation/source/` for exact provenance.

Generated figures are available locally under the repository's ignored `figures/` directory. Historical reference plots, the fresh full embedding plot and explicitly labeled smoke transport panels remain distinct. Neither generated outputs nor this validation archive are included in the public ZIP.

Known missing runners and numerical differences remain documented in `docs/reproducibility.md`. These packaging checks do not replace a full rerun of every paper result.
