# Restored ZIP RCPM: independent five-seed aggregation

Both rows contain the five prescribed seeds in the same nvidia7 JAX 0.4.35 / Flax 0.8.4 environment, label `zip_x64_cuda_jax435`, untouched ZIP source. Per-run means were recomputed from five evaluation batches before aggregation across training seeds.

| Row | KL mean ± ddof0 SE | ESS ratio mean ± ddof0 SE | Paper KL / ESS | Printed mean agreement |
|---|---|---|---|---|
| S2 | 0.0038152133 ± 0.000815 | 0.99641777 ± 0.000422 | 0.0037 / 0.996 | kl: differs; ess_ratio: matches |
| T2 | 0.87990061 ± 0.0286 | 0.61749627 ± 0.0165 | 0.93 / 0.55 | kl: differs; ess_ratio: differs |

The following historical RCPM rows are distinct recorded results; they are shown separately and are not pooled or selected by proximity to the new results.

| Log / row | Historical KL | Historical ESS | New minus historical KL | New minus historical ESS |
|---|---|---|---|---|
| table_random.log:10203 S2 | 0.0037 | 0.996 | 0.00011521333 | 0.00041777157 |
| table_random.log:10205 T2 | 0.931 | 0.548 | -0.051099395 | 0.069496273 |
| table_fps.log:10243 S2 | 0.0037 | 0.996 | 0.00011521333 | 0.00041777157 |
| table_fps.log:10245 T2 | 0.862 | 0.562 | 0.017900605 | 0.055496273 |

The paper-corresponding stored rows are S2 in `table_fps.log` and T2 in `table_random.log`. Full raw seed means, evaluation batches, both SE conventions, source/environment identities, log hashes and rounding checks are preserved in [the JSON report](restored_rcpm_independent_aggregation.json).

Paper uncertainty construction is unspecified; rounded agreement is descriptive. NVIDIA training times are retained in JSON and are not treated as an AMD timing reproduction.
