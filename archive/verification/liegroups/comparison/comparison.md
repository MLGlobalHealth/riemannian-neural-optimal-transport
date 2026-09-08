# Table5 native Lie-group comparison

Each record represents one training seed and five held-out evaluation batches. All means and standard errors below are calculated within that record. Training runs, hosts and configurations are never pooled.

Unique records: 14; identical copied records: 0. Outcome counts: {'complete_finite': 9, 'observed_numerical_nonfinite': 5}.

Native batch SE uses population standard deviation divided by √5. Supplemental sample SE and all raw records/batches are retained in [comparison.json](comparison.json). The paper’s interval construction is unspecified; printed-number agreement is descriptive.

## Source and environment cohorts

| Cohort | Host | Source / helper / wrapper SHA prefixes | Complete unique native rows | Status |
| --- | --- | --- | ---: | --- |
| 5b7cbd20dd5ca329 | nvidia6 | e2747129c7ba / cd3671a45ecc / a5cd6e5eae86 | 14 / 14 | complete_14_native_runs |

Each slot must have exactly one complete native run. Additional executions or configuration variants remain visible and do not silently satisfy completeness. The JSON records full source/helper/wrapper hashes, environment and configuration identities.

## All fourteen Table5 configurations

| Row | Paper KL ± quantity; ESS ratio | Recorded outcomes |
| --- | --- | --- |
| SO3/RNOT | 2.96 ± 0.01; 0.225 | [92a8ee4da7897a73](<../results/SO3_ours_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/1 | 2.86 ± 0.03; 0.391 | [bc20a6097fa5de27](<../results/SO3_rcpm_gamma1_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/0.1 | 3.29 ± 0.10; 0.002 | [353a39ee1503f7e9](<../results/SO3_rcpm_gamma0.1_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/0.05 | 3.54 ± 0.06; 0.006 | [917e19e296d97801](<../results/SO3_rcpm_gamma0.05_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/0.01 | 2.83 ± 0.10; 0.003 | [1cc6da08ef469e97](<../results/SO3_rcpm_gamma0.01_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/0.005 | 3.72 ± 0.10; 0.004 | [457dd462f8bb0a7f](<../results/SO3_rcpm_gamma0.005_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SO3/RCPM/0.001 | 5.90 ± 0.06; 0.004 | [6807bbe31cf87857](<../results/SO3_rcpm_gamma0.001_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SE3/RNOT | 2.50 ± 0.01; 0.683 | [072b769614159236](<../results/SE3_ours_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SE3/RCPM/1 | 14.38 ± 0.09; 0.007 | [5ea08e1f85d4cc38](<../results/SE3_rcpm_gamma1_nativeJAX435/result.json>) (5b7cbd20dd5ca329, complete_finite) |
| SE3/RCPM/0.1 | Numerically unstable; — | [75ebee3589eacc23](<../results/SE3_rcpm_gamma0.1_nativeJAX435/result.json>) (5b7cbd20dd5ca329, observed_numerical_nonfinite) |
| SE3/RCPM/0.05 | Numerically unstable; — | [f915ee2e6c00b8c9](<../results/SE3_rcpm_gamma0.05_nativeJAX435/result.json>) (5b7cbd20dd5ca329, observed_numerical_nonfinite) |
| SE3/RCPM/0.01 | Numerically unstable; — | [d7ebb8759ba79c07](<../results/SE3_rcpm_gamma0.01_nativeJAX435/result.json>) (5b7cbd20dd5ca329, observed_numerical_nonfinite) |
| SE3/RCPM/0.005 | Numerically unstable; — | [0ef728fc7dd4787c](<../results/SE3_rcpm_gamma0.005_nativeJAX435/result.json>) (5b7cbd20dd5ca329, observed_numerical_nonfinite) |
| SE3/RCPM/0.001 | Numerically unstable; — | [23f0a454edfcc4f6](<../results/SE3_rcpm_gamma0.001_nativeJAX435/result.json>) (5b7cbd20dd5ca329, observed_numerical_nonfinite) |

## Individual native outputs

| Record / cohort | Row | KL mean ± batch SE | ESS ratio mean | Printed comparison / outcome |
| --- | --- | ---: | ---: | --- |
| [072b769614159236](<../results/SE3_ours_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RNOT | 2.4051482 ± 0.0093061002 | 0.67014744 | KL mean: differs, ESS mean: differs, KL batch SE: matches |
| [23f0a454edfcc4f6](<../results/SE3_rcpm_gamma0.001_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/0.001 | — ± — | — | observed_nonfinite_instability |
| [0ef728fc7dd4787c](<../results/SE3_rcpm_gamma0.005_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/0.005 | — ± — | — | observed_nonfinite_instability |
| [d7ebb8759ba79c07](<../results/SE3_rcpm_gamma0.01_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/0.01 | — ± — | — | observed_nonfinite_instability |
| [f915ee2e6c00b8c9](<../results/SE3_rcpm_gamma0.05_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/0.05 | — ± — | — | observed_nonfinite_instability |
| [75ebee3589eacc23](<../results/SE3_rcpm_gamma0.1_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/0.1 | — ± — | — | observed_nonfinite_instability |
| [5ea08e1f85d4cc38](<../results/SE3_rcpm_gamma1_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SE3/RCPM/1 | 13.226344 ± 0.044446402 | 0.0072679484 | KL mean: differs, ESS mean: matches, KL batch SE: differs |
| [92a8ee4da7897a73](<../results/SO3_ours_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RNOT | 3.8013232 ± 0.012527669 | 0.78042852 | KL mean: differs, ESS mean: differs, KL batch SE: matches |
| [6807bbe31cf87857](<../results/SO3_rcpm_gamma0.001_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/0.001 | 7.5246892 ± 0.075338178 | 0.0065656253 | KL mean: differs, ESS mean: differs, KL batch SE: differs |
| [457dd462f8bb0a7f](<../results/SO3_rcpm_gamma0.005_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/0.005 | 5.3445539 ± 0.24477719 | 0.0024662507 | KL mean: differs, ESS mean: differs, KL batch SE: differs |
| [1cc6da08ef469e97](<../results/SO3_rcpm_gamma0.01_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/0.01 | 3.8334751 ± 0.069336632 | 0.0041644038 | KL mean: differs, ESS mean: differs, KL batch SE: differs |
| [917e19e296d97801](<../results/SO3_rcpm_gamma0.05_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/0.05 | 4.212878 ± 0.066277227 | 0.0072583483 | KL mean: differs, ESS mean: differs, KL batch SE: differs |
| [353a39ee1503f7e9](<../results/SO3_rcpm_gamma0.1_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/0.1 | 4.1818601 ± 0.084795096 | 0.0093885915 | KL mean: differs, ESS mean: differs, KL batch SE: differs |
| [bc20a6097fa5de27](<../results/SO3_rcpm_gamma1_nativeJAX435/result.json>) / 5b7cbd20dd5ca329 | SO3/RCPM/1 | 2.8200842 ± 0.030677284 | 0.77548345 | KL mean: differs, ESS mean: differs, KL batch SE: matches |

Actual nonfinite evaluation values or measured nonfinite trained parameters establish observed numerical instability. Error messages containing “NaN”, import failures and build failures alone do not. Finite results for the qualitative SE3 rows are retained without inventing an instability threshold.

Full original records, failures, duplicate executions, identical-copy paths, nonfinite values, supplemental sample SE and per-row configuration identities are retained in the JSON. The original inputs are unchanged.
