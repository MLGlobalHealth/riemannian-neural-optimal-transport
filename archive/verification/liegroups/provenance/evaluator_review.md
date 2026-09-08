# Independent native Lie-group checkpoint evaluator review

**No blocking restoration or recording issue identified.** Reviewed frozen evaluator SHA256 `dae24f203584329953951d5d28749d32ae85dfb89ac59dcfc3de1aedf95e9c0e` and adjacent tests SHA256 `adf04fc92a2fa3682177d176f5ff10cf3fcd381a55e7d442c35dfca8531b5fb0`. Both parse successfully. No evaluator, test, experiment, or finalMLGH source was modified. This reviewer did not import JAX, initialize a native model, train, or evaluate it.

The implementation follows the native restoration requirements documented in `checkpoint_restoration_guidance.md`:

- RNOT uses the correct SO3/SE3 native builder, checks the saved configuration and actual constructed density/geometry, replaces checkpoint parameters and landmark embedding, clones psi, and constructs a fresh ArgminSolver with the unchanged solver configuration. This prevents an old compiled solver from referencing fresh-builder landmarks. Native evaluators do not use the retained trainer/loss references.
- RCPM restores the entire saved Flax variables dictionary and uses the native gamma1 flow. Parameter paths/shapes/dtypes are checked against the builder, checkpoint arrays are checked byte-for-byte after transfer, and mixed float32/float64 values are preserved. The unbound native flow applies the supplied variables without a separate stale parameter closure.
- RNOT state.params is cross-checked against the checkpoint's top-level params. Parameter counts/dtypes/nonfinite counts and landmark metadata are verified. Only fields needed by the native evaluation path are used; this is not a training continuation.
- All five recorded uint32 subkeys are replayed verbatim after independently checking the native seed12345 schedule. There is no old-table +1000 offset. SO3 calls compute_kl_ours, SE3 calls evaluate (including only its existing native mean residual), and RCPM calls compute_kl_rcpm. No replacement metric or supplemental diagnostic solve is present.
- Source, helper, wrapper, checkpoint and original-result hashes are retained and rechecked. Both original and replay package inventories must contain all 80 pinned versions. Precision/PRNG/backend guards are explicit. Host and build differences remain recorded.
- Replay records are labeled evaluation-only and independent_training_run=false. Raw metrics, original metrics, differences, native batch means/SE, nonfinite status and failures remain separate from training-cohort evidence. The existing training comparator ignores this record type.

Independent standard-library-only checks accepted the four actual completed source records and their 80-pin original environments. Ten subsequently downloaded RCPM gamma<1 records were correctly rejected as outside the authorized four-model scope. See `evaluator_review_input_checks.json`. Previously verified all four source checkpoint file hashes in `checkpoint_restoration_inputs.json`.

The six supplied tests were read and cover the four model selections, changed protocol/environment rejection, fresh RNOT solver and checkpoint embedding, intact RCPM variables, payload/state/config/dtype drift, and hash/source tampering. Their author reports all six passing; this review did not repeat that test run.

Native runtime round-trips are still required to empirically confirm restoration on the original host. The planned bounded original-host and other-host replays provide that evidence; no claim of successful numerical replay is made here.
