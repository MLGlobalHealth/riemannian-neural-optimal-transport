# Historical AMD environment provenance

This is a read-only audit of committed setup documentation, experiment source, logs, and notebook metadata. It identifies an AMD reconstruction candidate; it does not certify a recovered installation or attribute any measured discrepancy to AMD versus NVIDIA. No software was installed and no experiment was run for this audit.

Source anchor: `dbf533d86b7b259b42282f0d55c95915c181add6` (2026-01-27), the historical source candidate already used for verification. The final-result logs were committed at `a7327bc829837c62d5e41ba222e89ea7b4900d83` (2026-01-26). The GPU setup note was introduced earlier at `41063962082e2914584dbcc26a23a8ac38ef1f1e` (2026-01-12), and remains unchanged at the source anchor. File/line references below refer to that source anchor. Log line numbers normalize carriage returns to newlines.

## Evidence and its limits

| Component | Evidence | What is established |
|---|---|---|
| AMD backend | All ten `experiments/final_results/*.log` files print `Running on: [RocmDevice(id=0)]` at line 2 or 3. | The stored table, sweep, high-dimensional, and ablation runs actually used a ROCm device. This is execution evidence. |
| GPU model | `GPU_SETUP.md:5,17` states eight AMD Instinct MI300X GPUs, architecture `gfx942`. | Contemporaneous setup documentation identifies the hardware; individual table logs do not print model, serial, or GPU driver. |
| ROCm and JAX packages | `GPU_SETUP.md:18,78–81` records ROCm 6.4.0 and `jax==0.4.35`, `jaxlib==0.4.35`, `jax-rocm60-plugin==0.4.35`, `jax-rocm60-pjrt==0.4.35`. | These are documented installed versions, stronger than the generic CPU requirements but not a captured package inventory from each experiment. Exact wheel builds, hashes, and package index are missing. |
| Python and environment | `table_fps.log:58` and the ablation logs' scatter warnings name `anaconda3/envs/rcpms-jax/lib/python3.10/site-packages/...`. Notebook metadata consistently records Python 3.10.19 and kernel display name `rcpms-jax`. `continental_drift_output.ipynb` additionally records Papermill 2.6.0 and a completed 2026-01-21 execution. | Python 3.10 and the environment name have runtime evidence; Python 3.10.19 is evidenced for saved notebooks and is a candidate for tables, whose patch version is not printed. |
| Flax / Optax | `environment.yml:32–33` and `requirements.txt:6–7` pin Flax 0.8.4 and Optax 0.2.3. The GPU setup note says dependencies from `environment.yml` were installed. | Plausible intended pins. No runtime Flax or Optax version, `pip freeze`, or AMD environment export was found. Treat these as unconfirmed, not demonstrated table-runtime versions. |
| CUDA | The environment template mentions installing `jax[cuda]` separately; drivers set both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` if `--gpu` is supplied. | These are portable setup/code paths. They are not evidence that the original logged experiments used CUDA or cuDNN. No original CUDA/cuDNN version is established; the logs affirm ROCm. |
| Device selection | `experiments/run commands.sh:11,13` launches random/FPS tables in `rcpms-jax` with `HIP_VISIBLE_DEVICES=2` / `3`; lines 16–23 similarly assign high-dimensional and ablation runs. | Launch instructions describe one visible GPU per process. Seeing logical device 0 in the logs is consistent with this masking; eight physical GPUs does not imply eight-device distributed training. |
| Library paths | `GPU_SETUP.md:20,30–31,71–72` records `ROCM_PATH=/opt/rocm-6.4.0`, ROCm library additions to `LD_LIBRARY_PATH`, a `~/local/lib/libamd_comgr.so.2` symlink, and a conda activation hook. | Documented host setup. The actual activation-hook content, complete library paths, and symlink target are not committed, so this is not an executable host reconstruction. |

## Precision and compiler configuration

All four historical drivers explicitly call `jax.config.update("jax_enable_x64", True)` before model construction: `run_table.py:32`, `run_ablations.py:32`, `run_experiments.py:37`, and `run_experiments_highD.py:32`. Some stored compiler messages show real `f64` HLO arrays, for example `ablation_T2.log:14109` and `sphere_highD.log:5091`, corroborating double-precision computations. This does **not** prove every parameter or operation was float64: `src/networks.py` supplies no explicit Dense/LayerNorm parameter dtype, compute dtype, or matmul precision, leaving relevant defaults to Flax/JAX.

The source and launch scripts set GPU visibility and `TF_CPP_MIN_LOG_LEVEL=2`. No authored `XLA_FLAGS`, matmul precision override, PRNG implementation override, deterministic/autotuning override, or TF32/fast-math setting was found in the historical experiment source, shell instructions, setup files, or notebook source. Unrecorded inherited shell settings remain unknown. The logs contain XLA GPU autotuning messages; they do not record the selected GEMM algorithms or a complete compiler configuration. Text suggesting `XLA_FLAGS=--xla_dump_to=/tmp/foo` is compiler-generated troubleshooting advice, not evidence that this flag was used. XLA revision, LLVM revision, ROCm component builds, driver/kernel/OS versions, and persistent compilation/autotuning cache state are not captured.

## Why the dependency files cannot be treated as the original GPU lock

The current historical template pins CPU JAX/jaxlib 0.4.28, conflicting with the later GPU note's ROCm stack at 0.4.35. Commit `34d419d6ee31196036c47620f1e6f77aeb6ef277` (2026-01-12) explicitly replaced an earlier platform-specific conda export with a cross-platform requirements template. That older export was marked `osx-arm64`, and contained Python 3.11.14, JAX/jaxlib 0.5.0, jax-metal 0.1.1, Flax 0.10.2, and Optax 0.2.4. It describes a different machine, not an AMD lock. The AMD note introduced later that day is the appropriate historical source for the GPU JAX pins. NumPy, SciPy, Chex, Orbax, and other transitive dependencies are unpinned in the surviving template.

## Provenance-based reconstruction candidate

The most defensible candidate, pending an original environment export or inspection of the AMD machine, is:

```text
source: dbf533d86b7b259b42282f0d55c95915c181add6
hardware: AMD Instinct MI300X (gfx942), one visible GPU per process
ROCm: 6.4.0
Python: 3.10.19               # saved-notebook evidence; table patch version unknown
jax: 0.4.35                  # documented AMD installation
jaxlib: 0.4.35               # documented AMD installation
jax-rocm60-plugin: 0.4.35    # documented AMD installation
jax-rocm60-pjrt: 0.4.35      # documented AMD installation
flax: 0.8.4                 # dependency-file candidate, not runtime-confirmed
optax: 0.2.3                # dependency-file candidate, not runtime-confirmed
jax_enable_x64: true        # historical experiment source
model/solver/training: native historical driver defaults
compiler/PRNG flags: no new overrides; original inherited settings unknown
```

This is a **partially pinned candidate**, not a complete reproducible lock or validated installation recipe. Do not combine it with the conflicting JAX 0.4.28 CPU pins, substitute a newer ROCm wheel family silently, or describe the tentative Flax pin as recovered fact. Before any AMD run, capture the actual resolved package versions and wheel identities plus device/runtime/compiler information. If the original AMD environment remains available, its package inventory, activation hook, and runtime settings would resolve the most important gaps without choosing versions based on resulting metrics.

The completed NVIDIA trials used different accelerator/backend stacks and JAX/Flax versions. They establish results on those recorded NVIDIA environments. They do not isolate a hardware-only effect, because hardware, backend, and library versions changed together. A source- and configuration-matched MI300X/ROCm run using the documented candidate would test the AMD reproduction hypothesis; this audit alone cannot confirm it.

## Evidence hashes

SHA-256 values of exact Git blobs at the source anchor:

| File | SHA-256 |
|---|---|
| `GPU_SETUP.md` | `4f7c3c5964351327813cce6d52074879230f30583460ccf0d7b658e96c825067` |
| `environment.yml` | `a7604033ed20c5c6e7084c517776b8ffc8561c24b88ffbcdc196d722a4ee49f2` |
| `requirements.txt` | `fbda9d2f62dbdec24251b43a537093dee4a998885c1efb6fb1dcdc19553ca75a` |
| `start_jupyter.sh` | `20577ddcc29cdc9f2dc84fde2a771919c5472dfc5f32da5bb292681fae7e7150` |
| `experiments/run commands.sh` | `9a23aa07f71ee4354e53334033cc34572cffec92b491f7481060197b5141d654` |
| `experiments/run_table.py` | `aa70ca3914f7a989e2a1a60439cd56cc249378e02fc2e78c369eef8f77cbcacd` |
| `experiments/run_ablations.py` | `0d13481f1c8f250e8e9c80861bc84ada0e0820a93ec039f60fc64d682f5467ad` |
| `experiments/run_experiments.py` | `c8fd45b8a01f3fd08bc2bf993b0660f9b73726cc84e1f9066e442eb94e8a0005` |
| `experiments/run_experiments_highD.py` | `3d72d55acc3a6b772d328d5011ca82ce8541874574256272792834a06c1cc5b5` |
| `experiments/final_results/table_fps.log` | `da31a46875cd445e2dbe438612033ef52de9264a1c815f87b22c34d325d4fe5d` |
| `experiments/final_results/table_random.log` | `3f625f12817470696d9cd8c8064b1bf13729d9aae2075e57572aa173ec84e8cc` |
| `continental_drift_output.ipynb` | `0fe3541dcc0316388c0dc1bc4e45a75d879e679602d11c646a3c9ca9040a6fb2` |
