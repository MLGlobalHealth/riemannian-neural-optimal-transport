# Reproduction report — 8 September 2026

**The recovered configuration reproduces the paper’s displayed means for all four main-table RNOT KL rows, both sphere RNOT ESS rows, and torus FPS ESS on RTX 3090. Reproduction is incomplete:** torus random ESS remains about 0.80 versus the paper’s 0.85, RCPM comparisons retain differences, and several other published experiments lack recovered runners or fresh verification. The ZIP defaults alone do not reproduce several main-table rows. A repeatable evaluation difference between two NVIDIA environments is documented below; the original AMD environment has not been rerun.

The verification concerns reproduction of the published outputs. The ZIP's experiment algorithms and metric implementations remain unchanged. Source, configuration, software, and GPU environment are recorded for each trial.

## Restored main-table configuration

All 30 prescribed experiments completed on `nvidia7` RTX 6000 Ada GPUs: S²/T² RNOT with FPS/random landmarks and S²/T² RCPM with γ=1, each using seeds `12345`, `23456`, `34567`, `45678`, and `56789`. The supported older CUDA environment and recovered settings are described below.

| Experiment | Paper KL | RTX 6000 Ada KL | Paper ESS ratio | RTX 6000 Ada ESS ratio |
|---|---:|---:|---:|---:|
| S² RNOT FPS | 0.03 ± 0.00 | 0.026959 ± 0.001580 | 0.97 ± 0.00 | 0.965099 ± 0.004730 |
| S² RNOT random | 0.04 ± 0.00 | 0.044143 ± 0.002241 | 0.95 ± 0.00 | 0.948114 ± 0.003681 |
| S² RCPM γ=1 | 0.0037 ± 0.0008 | 0.003815 ± 0.000815 | 0.996 ± 0.000 | 0.996418 ± 0.000422 |
| T² RNOT FPS | 0.13 ± 0.01 | 0.127375 ± 0.007099 | 0.93 ± 0.01 | 0.893765 ± 0.028207 |
| T² RNOT random | 0.22 ± 0.04 | 0.222871 ± 0.035244 | 0.85 ± 0.02 | 0.792746 ± 0.033732 |
| T² RCPM γ=1 | 0.93 ± 0.02 | 0.879901 ± 0.028613 | 0.55 ± 0.03 | 0.617496 ± 0.016491 |

Each training seed was evaluated using the original five batches of 1024 points and seed offset +1000. Fresh uncertainties follow the shipped script: population standard deviation across the five seed means divided by √5. The paper calls its ± quantities confidence intervals without specifying their construction or level. Rounded ±0.00 does not mean zero uncertainty, and agreement at printed precision is descriptive rather than a statistical equivalence test. Sample standard errors are also retained in the [machine-readable comparison](comparison/restored_zip_jax435/comparison.json).

The [full comparison](comparison/restored_zip_jax435/comparison.md) links every unchanged per-seed record. No seed was discarded or selected according to its result.

## Measured sensitivity to the NVIDIA execution environment

The same software packages, ZIP source, configuration, and evaluation keys were checked on two hosts: RTX 6000 Ada / driver 570.195.03 on `nvidia7`, and RTX 3090 / driver 550.90.07 on `nvidia6`.

For torus FPS seed `12345`, the complete checkpoint files are byte-for-byte identical: SHA256 `358939c83c3cf560dcb1ac4cc096d15b61a8e8413586f8cb4bcfebe6894e8c8f`. This includes all parameters, landmarks, training state, and retained key. All five evaluation keys agree. The final outputs nevertheless differ:

| T² FPS, seed 12345 | KL | ESS ratio |
|---|---:|---:|
| Historical stored AMD log | 0.1215 | 0.951 |
| Fresh NVIDIA RTX 6000 Ada | 0.120698044 | 0.772387986 |
| Fresh NVIDIA RTX 3090 | 0.121290775 | 0.944883341 |

A separately recorded [native checkpoint replay](provenance/checkpoint_replay_comparison.md) reproduced every original batch on each host exactly: all 15 scalar values (KL, ESS, and ESS ratio across five batches) matched. The restored array fingerprints and effective JAX settings agree across the two replays. This confirms that the observed gap is repeatable during evaluation of the common saved model and keys.

The other three paired checks—torus RNOT random and S²/T² RCPM, all seed `12345`—also have identical checkpoints and evaluation keys across hosts. RCPM outputs agree to numerical precision. Torus random outputs differ: KL/ESS are 0.126726282/0.703857579 on Ada and 0.127321133/0.681479606 on RTX 3090. The [paired checkpoint manifest](provenance/paired_checkpoint_hash_manifest.md) and [training-trace comparison](provenance/t2_seed12345_trajectory_comparison.md) preserve the evidence.

After observing the FPS difference, the four remaining prescribed FPS seeds were declared and run on newly available RTX 3090 GPUs. Its KL/ESS means and shipped standard errors round to the paper's displayed numbers. The remaining four random-landmark seeds were then declared to complete both torus RNOT rows on this second host. Each initial seed is retained unchanged; every prescribed seed is included:

| T² RNOT on RTX 3090 | Paper KL | Fresh KL ± shipped SE | Paper ESS ratio | Fresh ESS ratio ± shipped SE |
|---|---:|---:|---:|---:|
| FPS | 0.13 ± 0.01 | 0.126792 ± 0.007062 | 0.93 ± 0.01 | 0.926916 ± 0.009887 |
| Random | 0.22 ± 0.04 | 0.222998 ± 0.034929 | 0.85 ± 0.02 | 0.796680 ± 0.032788 |

The [complete paired torus comparison](provenance/t2_five_seed_host_comparison.md) verifies that all ten paired FPS/random checkpoint files and their evaluation-key schedules are identical across hosts. The RTX 3090 FPS row reproduces the paper’s displayed means and ± values. The random KL mean rounds to 0.22, but its shipped standard error rounds to 0.03 rather than the paper’s 0.04; random ESS rounds to 0.80 ± 0.03 rather than 0.85 ± 0.02. Thus the second NVIDIA environment does not reproduce every torus metric. Its [12-run comparison](comparison/nvidia6_portability_jax435/comparison.md) also retains the two separate single-seed RCPM checks.

These are separate host groups; their seeds are never pooled. The evidence demonstrates sensitivity during evaluation across the recorded NVIDIA environments. It does not isolate sampling, compiler, GPU, driver, OS/system-library, or Python-build effects, and neither fresh environment is AMD. Both interpreters report Python 3.11.15, but their recorded builds and host system libraries differ. GPU/backend numerical differences must therefore remain part of the reproduction interpretation. The original AMD runtime has not been rerun.

## Source and settings recovered

The `finalMLGH` branch begins with orphan commit `ca8eb58a7f20b7d0c22912892588335d48700ba4`, containing exactly the ZIP's 22 files. Their hashes are in [archive_manifest.json](provenance/archive_manifest.json); the verification additions leave them unchanged.

Historical experiment logs specify substantially different RNOT settings from the ZIP defaults:

| Setting | ZIP defaults | Recovered historical settings |
|---|---:|---:|
| Landmarks | 256 | 128 |
| Outer training steps | 500 | 1000 |
| Maximum / minimum inner steps | 500 / 50 | 2500 / 1000 |
| Inner learning rate | 0.005 | 0.05 |
| Inner Adam | disabled | enabled |
| Soft-argmin initialization | disabled | enabled, γ=0.1 |
| Line search | enabled | absent historically; disabled for the restored ZIP configuration |
| Floating-point configuration | float32 default | x64 enabled by the historical driver |

The restored RNOT trials apply [historical_settings.json](historical_settings.json) with x64 enabled. RCPM uses its native builder and γ=1 with x64 enabled, without configuration overrides. These settings were selected from provenance before the restored results.

Commit `dbf533d86b7b259b42282f0d55c95915c181add6` is a historical source candidate selected from logged settings and repository history. Its exact selected files, including the ablation driver, are preserved under [snapshots/dbf533d](snapshots/dbf533d/). The logs lack an execution source hash, so this is not a proven original execution commit. See [source provenance](provenance/historical_source.md) and the [snapshot manifest](provenance/historical_snapshot_manifest.json).

A controlled S² random trial under the restored environment produced KL 0.039316750 and ESS 0.951389851 from both the historical source and configured ZIP, matching each other to numerical precision and closely matching the historical same-seed AMD log (0.0394/0.951). All 1000 printed losses match between the two fresh sources; 974 match the AMD log at four decimals. The [candidate provenance](provenance/restored_cuda/README.md) retains this comparison and the separately labeled historical-source pilots.

For RCPM, the applicable builder, trainer, evaluator, manifold, and density implementations agree between the ZIP and historical source candidate. Their effective configurations also agree. Software and execution flags remain differences across the recorded environments; see [RCPM recipe provenance](provenance/rcpm_recipe_provenance.md).

## Software environment and rerunning

The restored CUDA trials use Python 3.11.15, JAX 0.4.35, jaxlib 0.4.34, CUDA plugin/PJRT 0.4.35, Flax 0.8.4, and Optax 0.2.3. The complete [80-package freeze](environments/reproduction-cuda.txt) was installed in fresh environments on both NVIDIA hosts, each passing `pip check`. The jaxlib version follows the official JAX 0.4.35 CUDA extra; forcing jaxlib 0.4.35 with that extra was rejected by dependency resolution before any experiment. JAX's supported `CUDA_ROOT` setting points to the installed toolkit; no package source was patched.

[Historical AMD provenance](provenance/amd_environment.md) documents MI300X, ROCm 6.4.0, JAX/ROCm packages 0.4.35, Python 3.10 evidence, and x64 enabled. Flax 0.8.4 and Optax 0.2.3 are dependency-file candidates, not captured runtime versions. The supported CUDA counterpart therefore does not recreate the complete original AMD environment.

Software versions also change seeded execution. [JAX 0.5.0 changed the default Threefry PRNG behavior](https://docs.jax.dev/en/latest/changelog.html#jax-0-5-0-jan-17-2025). A [bounded CPU comparison](provenance/rng_environment_comparison.md) found that 7 of 9 fingerprints differed between JAX 0.4.38 and 0.6.2 defaults; setting `jax_threefry_partitionable=False` in 0.6.2 restored all nine matches. A separate [CPU initialization comparison](provenance/initialization_environment_comparison.md), with PRNG behavior aligned, found exact initial parameter/landmark agreement between the JAX 0.4.35 and 0.4.38 candidates. These checks establish bounded software facts, not full training or AMD equivalence. JAX also documents that [compiler transformations can change exact numerical outputs](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).

[The root README](../README.md) gives the installation and one-command 30-run procedure. It explicitly distinguishes restored settings from the original ZIP entry-point defaults. The launcher requires a fresh output directory, preserves logs/results after failures, and checks the selected interpreter's core versions. GPU timings are recorded descriptively; different hardware and compilation make an AMD/NVIDIA timing match inappropriate as a reproduction criterion.

## Other completed trials retained separately

The initial 30-run ZIP-default suite used JAX 0.6.2 / Flax 0.10.6 with float32 defaults on `nvidia7`. It differs from several published means:

| Experiment | ZIP-default KL ± shipped SE | ZIP-default ESS ratio ± shipped SE |
|---|---:|---:|
| S² RNOT FPS | 0.03968 ± 0.00394 | 0.89982 ± 0.02930 |
| S² RNOT random | 0.06784 ± 0.00687 | 0.76310 ± 0.07315 |
| S² RCPM γ=1 | 0.00851 ± 0.00079 | 0.99601 ± 0.00050 |
| T² RNOT FPS | 0.42591 ± 0.01760 | 0.79339 ± 0.03417 |
| T² RNOT random | 0.47159 ± 0.02335 | 0.70042 ± 0.03899 |
| T² RCPM γ=1 | 0.93351 ± 0.04341 | 0.54508 ± 0.02836 |

[Those records and their comparison](comparison/zip_defaults/comparison.md) are preserved alongside four [ZIP historical-settings trials under JAX 0.6.2](comparison/historical_settings/comparison.md), two [historical-source RNOT trials under JAX 0.4.38](comparison/historical_source_jax438/comparison.md), and ten [historical-source RCPM trials under JAX 0.6.2](comparison/historical_source_jax062/comparison.md). Their sources/configurations/environments differ, so they do not establish a GPU-only effect. The initial JAX 0.6.2 environment was an isolated overlay; a clean installation of that initial trial's exported pins was not separately tested. The restored JAX 0.4.35 CUDA freeze was tested in fresh environments on both hosts. Two historical RCPM initialization failures under JAX 0.4.38 / Flax 0.10.0 are [retained separately](comparison/historical_rcpm_jax438_failure/comparison.md); they failed before training.

## Remaining paper coverage and provenance

| Paper content | Evidence / remaining requirement |
|---|---|
| Tables 1–2, RNOT and RCPM | Complete restored and ZIP-default 30-run suites; separate complete RTX 3090 torus RNOT rows and paired checks. Numerical findings above. |
| Tables 1–2, RCNF and Moser | Original runners/configurations not found in the ZIP, reachable Git history, or inspected server locations. |
| Table 3 ablations | Historical driver recovered. All 79 numeric cells match the included [four stored historical JSONs](provenance/stored_ablation_results/README.md) at displayed precision. This is stored provenance, not a fresh ablation rerun. |
| Table 4, semi-dual RCPM | Original runner/configuration not recovered. |
| Table 5, SO(3)/SE(3) | Geometry components exist; original KL/ESS experiment runners not recovered. |
| Dimension-sweep figures | ZIP drivers exist; full sweeps have not been rerun after the main-table default mismatch was established. |
| Continental drift | Original point-cloud data and runner not recovered. |

Searches covered fetched branches, reachable history, bounded source/output searches, and relevant archive names on `nvidia6`, `nvidia7`, and the configured `dide` host. Related entropic SO(3)/SE(3) W1/W2 experiments belong to a different experiment and do not supply the paper's KL/ESS rows. [Branch/source search details](provenance/rebuttal_search.md) and [archive search details](provenance/rebuttal_archive_search.md) retain the scope.

Targets come from the historical download of the [requested OpenReview PDF](https://openreview.net/pdf?id=ez4oLq7PR3), whose download metadata confirms that URL, and the local camera-ready PDF. Their main Tables 1–3 agree with the [arXiv version](https://arxiv.org/pdf/2602.03566v1). The live OpenReview endpoint returned HTTP 403, so its currently served revision was not retrieved. PDF hashes and extracted targets are in [paper_targets.json](paper_targets.json).

The [training-run manifest](provenance/run_manifest.json) records **90 completed experiments and two initialization failures**, with all completed cohorts and prescribed seeds retained. The separate [evaluation manifest](provenance/evaluation_manifest.json) records **two successful checkpoint replays**. All owned experiment workers, supervisors, and collectors have exited; no jobs remain scheduled. The tooling suite passed 22 tests covering aggregation, source/configuration/environment separation, missing/duplicate/failed runs, portable links, complete job declaration, failure preservation, and checkpoint-evaluation guards. The 22 ZIP files and 24 historical snapshot files were verified against their source manifests; the historical snapshot was also checked against Git. Raw per-run JSONs remain unchanged. Checkpoints and full logs are retained locally in ignored `verification/artifacts/` and isolated server run directories, with hashes in the committed evidence. Checkpoint replays are evaluation-only records and do not add independent training seeds.
