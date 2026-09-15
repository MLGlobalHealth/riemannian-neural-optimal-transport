# Independent Lie-group comparator review

**No blocking issue identified in the reviewed comparator/tests or the available real wrapper record.** No comparator, experiment, or finalMLGH source was edited.

Reviewed SHA256:

| File | SHA256 |
|---|---|
| `compare_liegroup_results.py` | `a226103c8f9ddc45caff5c23b04fb0eb36bd83b146b1f278ed2290718af7a434` |
| `test_compare_liegroup_results.py` | `5a57829a984b02bca196bdbba38a14ed9f4c814a87df0e8ce5e6cc4e008bc845` |
| `paper_targets.json` | `47518544284797c88115f025c624a330b133336edc5f4a24ac5bd2cd63d93495` |

## Verification performed

- Read both complete files against frozen wrapper `a5cd6e5eae86918a5ebb788e6249bd65f9def4b52edfa12bf624870e54a1da33`, both native mains, and the Table 5 target schema.
- Ran the six supplied standard-library tests once: all passed. They cover the fourteen slots, batch uncertainty, nonfinite records, failed execution, incomplete batches, summary disagreement, duplicate executions/copies, source/helper/wrapper/host/driver/configuration separation, and portable evidence links.
- Executed the comparator on the available real `SE3_ours_nativeJAX435/result.json`. It produced one valid finite native record and a partial 1/14 cohort with no validation errors. Original raw record content is preserved. Output is retained under `comparator_review_output/`.
- Independently recomputed the real record's KL mean and population-standard-deviation/sqrt(5) from its five native batches. Results agree with the comparator and saved wrapper summary. Its KL/ESS means differ from the printed paper targets, while KL batch SE falls within the paper's printed .01 interval; the comparator reports precisely those distinctions.

## Findings

The nine numeric Table 5 entries and five qualitative SE3 RCPM entries are mapped correctly. ESS is compared as a ratio, not as the absolute effective sample count. The comparator checks consistency with ESS/1024. Numeric comparisons use the published two KL decimals and three ESS decimals, and are explicitly descriptive rounding comparisons rather than equivalence tests.

All aggregation stays within a single trained model's five held-out batches. The native population SE and supplemental sample SE remain distinct; the unknown paper interval construction is not assigned an invented confidence level. Additional runs are retained individually, and duplicate/configuration-variant slots do not silently complete the fourteen-row cohort.

Actual nonfinite metric values and measured nonfinite trained parameters are retained as observed numerical evidence. Import/build errors alone remain execution failures, including error strings containing “NaN”. Failure status and phase remain separately available even when a failed run also contains prior observed nonfinite parameters or metrics. A complete nonfinite record can satisfy execution coverage without being labeled a finite numeric match or complete paper reproduction.

Source/helper/wrapper/label and recorded environment determine cohorts. GPU indices/UUIDs are normalized only when the selected host/model/driver/memory identity is known; distinct hosts, drivers, packages, configuration identities, and raw records remain explicit. No cross-record metric mean is created. SE3's additional native mean residual is retained and summary-checked without being substituted for a paper metric.

This review validates recording/comparison behavior, not the algorithms or the eventual fourteen-run numerical results. Only one real completed row was present during the bounded review; the other outcomes are covered by the supplied synthetic schema tests until the cohort finishes.
