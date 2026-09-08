# RNG environment comparison

The installed JAX versions use different default random streams for the same seed. Restoring the older RNG setting in JAX 0.6.2 makes all nine small CPU fingerprints match JAX 0.4.38 bit-for-bit. This establishes a software confound independently of AMD versus NVIDIA.

| Probe | JAX / JAXlib | `jax_threefry_partitionable` | Default PRNG | Backend |
|---|---|---|---|---|
| jax0438_native (nvidia6) | 0.4.38 / 0.4.38 | False | threefry2x32 | CPU |
| jax062_native (nvidia7) | 0.6.2 / 0.6.2 | True | threefry2x32 | CPU |
| jax062_partitionable_false (nvidia7) | 0.6.2 / 0.6.2 | False | threefry2x32 | CPU |

All probes used seed 12345, `PRNGKey`, x64 enabled, explicit float32/float64 shapes `(16, 3)`, and an int32 permutation of 128 items. Two CPU cores and two numerical-library threads were used; no GPU was initialized. Normal and uniform outputs, split/folded keys, raw bits, full values and SHA256 fingerprints are preserved in [the JSON evidence](rng_environment_comparison.json).

| Output | Native 0.4.38 versus native 0.6.2 | Native 0.4.38 versus 0.6.2 with flag False |
|---|---|---|
| bits_uint32 | different | identical |
| fold_in_7 | identical | identical |
| key | identical | identical |
| normal_float32 | different | identical |
| normal_float64 | different | identical |
| permutation_int32 | different | identical |
| split_4 | different | identical |
| uniform_float32 | different | identical |
| uniform_float64 | different | identical |

The repository [AMD environment provenance](amd_environment.md) document JAX 0.4.35 with ROCm on AMD MI300X. Versioned official JAX 0.4.35 source sets `jax_threefry_partitionable=False` and `jax_default_prng_impl=threefry2x32`. [Official versioned source](https://github.com/jax-ml/jax/blob/jax-v0.4.35/jax/_src/config.py#L1028-L1036).

JAX changed the partitionable default to True in 0.5.0; its maintainers explain that this changes seeded random values and document restoring False for previous behavior. [JAX changelog](https://docs.jax.dev/en/latest/changelog.html#jax-0-5-0-jan-17-2025), [maintainer upgrade note](https://github.com/jax-ml/jax/discussions/18480).

No explicit PRNG implementation or partitionable override was found in the scanned ZIP source, verification tooling, or historical snapshot. The ZIP table driver uses `PRNGKey` and repeated key splitting for construction and evaluation, so the observed stream change is relevant to those experiments.

This probe identifies a reproducibility factor and does not measure its effect on final training metrics. It does not establish which flags the original paper processes used, test AMD hardware, or explain every difference in the reruns. Historical setup notes and requirements also differ (0.4.35 versus 0.4.28), so matching the original executed environment still requires provenance evidence. No production code, experimental settings, or recorded results were changed.
