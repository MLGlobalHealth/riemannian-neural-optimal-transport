# Recovered Lie-group recording wrapper review

Reviewed `verify_liegroup_run.py` SHA256 `a5cd6e5eae86918a5ebb788e6249bd65f9def4b52edfa12bf624870e54a1da33` against both complete recovered native runners and the unchanged sibling `verification/verify_run.py` helper SHA256 `cd3671a45ecca9b55542d75353991ad591c61b5f18f2e3cc3a855875a61d7589`.

**No blocking source or recording incompatibility found by static review.** Python AST parsing succeeds. No JAX import, build, training, evaluation, or numerical check was executed in this review; native CPU/runtime checks remain separate evidence.

- Native RNOT builders and trainers are called without overrides. SO3 reads full `ExperimentConfig`; SE3 records its three separate configuration dataclasses. The actual RNOT training seed is taken from the constructed training configuration.
- RCPM uses the native selected gamma and seed, default 68 components / 5 transforms, and the main's explicit 5000 iterations / batch 256 / logging interval 500. Its unchanged default learning rate is .001. The smaller lower-level defaults are not accidentally used.
- Evaluation recreates the native seed 12345 without an offset and the exact five sequential `key, subkey = split(key)` calls, with batch size 1024. Each unchanged evaluator receives the corresponding key. No key is consumed for recording or checkpointing.
- Raw metrics are copied to Python floats. SO3 records KL/ESS/ESS ratio; SE3 also preserves the returned native mean residual. No additional solver diagnostic or replacement metric is called. Mean and population-standard-deviation/sqrt(5) match the native reductions; sample SE is separately marked supplemental.
- The shared checkpoint helper accesses fields present in both native RNOT dictionaries (`state`, `key`, `psi.phi.landmarks`) and both RCPM dictionaries (`params`, `key`). Its retained-key/optimizer-state limitation notes remain accurate. Serialization compatibility still depends on the separate native build/checkpoint test.
- Raw batches are persisted atomically after each evaluation, and errors are recorded with the phase and traceback. Nonfinite floats are represented safely in JSON while their explicit nonfinite flags survive. No failed/nonfinite row is silently excluded.
- Training time preserves the native return boundary and separately records synchronization time. The fresh-process gamma/cache-history difference is stated. Source and helper hashes are checked; the native environment settings and full installed package versions are retained.

## Metadata and packaging notes sent to the parent

1. The parent applied the metadata refinement: RCPM now records `flow.n_transforms` and `dict(flow.single_transform_cfg)`, including the existing initialization and potential settings. Removing exactly these two added lines recovers the originally reviewed SHA256 `5dc2a41de52446651e68d32c0c8f8cc1d7ce5befa2ef1f199229007617c5548e`; no other wrapper change occurred. The post-refinement temporary and committed copies match byte-for-byte (commit `3ae0f9d07d22fdc9d1c0a4ae84c5a77be1a273fd`). AST parsing still succeeds.
2. Deployment must include the unchanged sibling `verify_run.py` and the source manifest at the wrapper's default `verification/provenance/recovered_liegroup_source_manifest.json`, or explicitly supply `--source-manifest`. The temporary manifest currently lives at `/private/tmp/mlgh-liegroups-20260908/source_manifest.json`.

No wrapper or experiment source was edited by this reviewer. See `compatibility.md` and `static_compatibility.json` for the complete native-runner provenance audit.
