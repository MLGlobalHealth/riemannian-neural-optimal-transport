# Experiment coverage

This release packages the supplied experiment library, recovered paper runners, explicit experiment settings, plotting code, and the available numerical references. It preserves the original optimization and metric implementations. A runnable recipe does not establish exact agreement with every published number.

| Paper content | Included workflow | Reproduction status |
|---|---|---|
| Tables 1–2, RNOT and RCPM | Five training seeds per row; RNOT FPS/random and RCPM gamma 1 on the sphere and torus | Tested on NVIDIA. All four RNOT KL means match the paper at printed precision using recovered settings; some ESS and RCPM values differ. |
| Tables 1–2, RCNF and Moser | Paper reference values for comparison | Original runners remain unrecovered. |
| Table 3 | Historical ablation driver with an explicit baseline; stored reference values and plotting | Stored values match the published numeric cells at printed precision. The full recovered ablation suite has not been freshly rerun; the native driver includes 21 settings, of which 20 appear in the paper reference. |
| Table 4, semi-dual RCPM | Paper reference values | The semi-dual RCPM runner remains unrecovered. The available RCPM sweep uses its native objective. |
| Table 5, SO(3)/SE(3) | Fourteen supplied native configurations and plotting | All configurations executed. Nine return finite results; the five SE(3) RCPM gamma values below 1 return nonfinite results. Numeric discrepancies remain. |
| Figures 3 and 5 | Native dimension/high-dimension runners, plotters, and historical log-derived numerical references | Full sweeps have not been freshly verified against the published curves. Historical reference values are rounded log outputs. |
| Figure 4 | Recovered landmark-embedding computation and plotter | The full 18-row computation and plotting command executed on NVIDIA. The original notebook kernel environment is unconfirmed; labeled historical values are also provided. |
| Figure 2 | Export source, pushforward and target samples from a saved sphere/torus model; render transport panels | A new figure describes the selected saved model. No model is selected automatically by its appearance or score. |
| Figure 1, continental drift | Two original CSV point clouds, recovered notebook-based training recipe and sample plots | The original saved model is unavailable. A fresh run is supported; the published transported figure, purity and Monge-gap values have not been freshly reproduced. |

## Settings and uncertainty

The main-table RNOT command uses the recovered historical settings explicitly: 128 landmarks, 1000 outer updates, 2500 maximum inner steps, minimum 1000, inner learning rate 0.05, Adam and soft initialization enabled, and line search disabled. The original experiment entry points under `experiments/` retain the supplied defaults, which differ from this recipe.

The supplied dimension and high-dimension sweep recipes retain their own defaults with x64 disabled; their full curves have not been freshly verified. They do not silently inherit the recovered main-table overrides.

SO(3)/SE(3) use their recovered runners' own settings. SO(3) inherits the ZIP defaults, including 256 landmarks and 500 outer updates. SE(3) explicitly specifies 128 landmarks and 200 outer updates. The original SO(3) companion `src/base.py` configuration is still unconfirmed. The SE(3) constructor uses translations in `[-4,4]^3` and `SE3FactorizedCompact`; its older console banner does not describe those actual constructor values.

Tables 1–2 train five independent seeds. Their summaries first average five evaluation batches for each seed, then calculate the row mean and population-standard-deviation divided by the square root of five across seed means. Lie-group and dimension-sweep rows train one model and evaluate five batches; their uncertainty describes evaluation sampling for that model. The ablation driver has its own recorded five-batch key schedule. These different protocols are kept separate.

The paper's rounded uncertainties do not specify a confidence level or an equivalence test. Timings depend on hardware and compilation. Nonfinite results, failed runs and incomplete seed sets remain visible in the outputs and plots.

The continental command uses the recovered notebook's printed 1000-update, 128-landmark, batch-256, 2500-inner-step setup. Appendix F.1 instead states 500 updates, 1024 landmarks, batch 1024 and up to 1000 inner steps. The original checkpoint and unprinted companion defaults are unavailable, so the command is labeled as a fresh recovered notebook recipe. See the [experiment guide](experiments.md#figure-1-continental-drift).

## Observed NVIDIA results

The tested CUDA environment uses Python 3.11.15, JAX 0.4.35, jaxlib 0.4.34, Flax 0.8.4 and Optax 0.2.3. The pinned requirements preserve that installation. Some documentation/testing packages are transitive dependencies of `spherical_kde`; they are not additional project experiment frameworks.

Representative Table 5 RNOT results on RTX 3090:

| Model | Paper KL / ESS ratio | Fresh KL / ESS ratio |
|---|---:|---:|
| SO(3) | 2.96 / 0.225 | 3.801323 / 0.780429 |
| SE(3) | 2.50 / 0.683 | 2.405148 / 0.670147 |

None of the nine Table 5 numeric KL means matches at printed precision in that execution; SE(3) RCPM gamma 1 reproduces its printed ESS ratio. All five SE(3) RCPM gamma values below 1 produce nonfinite outputs, consistent with the paper's qualitative instability description.

Four fixed Lie-group models were evaluated on RTX 3090 and RTX 6000 Ada using identical checkpoints and keys. Original-host replays match exactly; cross-host KL/ESS-ratio differences are at most `6.66e-16`. This does not explain the gaps for those four fixed models. A separate torus RNOT check found repeatable evaluation differences across the same two NVIDIA environments. These observations do not isolate GPU hardware from drivers, compilers or host libraries, test every training configuration on both hosts, or reconstruct the original AMD/ROCm environment.

The detailed run records and source audits are retained in the repository's archival folder. They are excluded from the distributable code archive. The public plotting references identify their origin and precision; they are never presented as newly computed experiments.

The public command package was additionally checked with a full SE(3) RNOT run, the full landmark-embedding computation, and smoke runs for sphere RNOT, RCPM, ablations and continental drift. The SE(3) run exactly reproduces the earlier NVIDIA result above, including its checkpoint bytes. Saved-model transport and offline sample rendering also executed successfully. These bounded checks validate the packaged workflow; they do not constitute a fresh rerun of every table or dimension sweep.
