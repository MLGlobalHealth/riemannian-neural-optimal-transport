# Native initialization environment comparison

The supported JAX 0.4.35 / Flax 0.8.4 candidate and nvidia6 JAX 0.4.38 / Flax 0.10.0 produced bit-for-bit identical recorded S2 random initialization on CPU: all 33,409 parameters, all 128 landmarks, the remaining RNG key, and four sampled base/target points. All 12 arrays (33,819 values) were compared from preserved full binary arrays.

| Environment | JAX | JAXlib | Flax | Optax |
|---|---|---|---|---|
| nvidia7 (0435) | 0.4.35 | 0.4.34 | 0.8.4 | 0.2.3 |
| nvidia6 (0438) | 0.4.38 | 0.4.38 | 0.10.0 | 0.2.3 |

Both used untouched historical source `dbf533d86b7b259b42282f0d55c95915c181add6`, its native S2 random builder and seed 12345, x64=True, Threefry partitionable=False, and default matmul precision=None. Source manifests, runner hashes, effective configuration and parameter tree match. The candidate rerun also reproduces the previously recorded `cpu_native_builds.json` fingerprints.

The comparison independently parsed the saved NPZ arrays using the Python standard library, verified every array SHA256 against its JSON record, and compared every byte. The probe used two CPU cores per host and did not train.

Evidence: [comparison JSON](initialization_environment_comparison.json), [candidate record](cpu_initialization_jax0435.json), [candidate full arrays](cpu_initialization_jax0435.npz), [0.4.38 record](cpu_initialization_jax0438.json), [0.4.38 full arrays](cpu_initialization_jax0438.npz), [probe](probe_initialization_comparison.py).

For this controlled CPU build, parameter and landmark initialization do not explain a difference between these two environments. Any observed GPU training divergence still needs to be separated into GPU initialization and subsequent computation: this check establishes neither GPU bitwise identity nor where later execution differs. The four-point sample fingerprints are separate from actual training minibatches. No source, stored experiment result, training loss, gradient, or metric was modified.
