# Native RCPM recipe provenance

The ZIP and historical `dbf533d` sources support the same native RCPM table recipe. All five paired T2 records have identical configuration, target/base specification, geometry, parameter counts/dtypes and evaluation settings. The executions still differ in software, RNG streams and XLA graph settings; both already have x64 enabled.

| Relevant path | ZIP evidence | Historical evidence | Comparison |
|---|---|---|---|
| experiments/run_table.py::build_rcpm_experiment | [source](../../experiments/run_table.py#L284) | [source](../snapshots/dbf533d/experiments/run_table.py#L278) | Identical AST |
| experiments/run_table.py::train_rcpm | [source](../../experiments/run_table.py#L332) | [source](../snapshots/dbf533d/experiments/run_table.py#L326) | Identical AST |
| experiments/run_table.py::compute_kl_rcpm | [source](../../experiments/run_table.py#L367) | [source](../snapshots/dbf533d/experiments/run_table.py#L361) | Identical AST |
| experiments/run_table.py::main | [source](../../experiments/run_table.py#L386) | [source](../snapshots/dbf533d/experiments/run_table.py#L380) | Identical AST |
| src/densities.py::SphereUniform | [source](../../src/densities.py#L149) | [source](../snapshots/dbf533d/src/densities.py#L102) | Identical AST |
| src/densities.py::WrappedNormal | [source](../../src/densities.py#L238) | [source](../snapshots/dbf533d/src/densities.py#L121) | Identical AST |
| src/densities.py::ProductUniformComponents | [source](../../src/densities.py#L309) | [source](../snapshots/dbf533d/src/densities.py#L192) | Identical AST |
| rcpm/densities.py::SphereUniform | [source](../../rcpm/densities.py#L120) | [source](../snapshots/dbf533d/rcpm/densities.py#L115) | Identical AST |
| rcpm/densities.py::WrappedNormal | [source](../../rcpm/densities.py#L160) | [source](../snapshots/dbf533d/rcpm/densities.py#L137) | Identical AST |
| rcpm/densities.py::ProductUniformComponents | [source](../../rcpm/densities.py#L345) | [source](../snapshots/dbf533d/rcpm/densities.py#L322) | Identical AST |
| rcpm/manifolds.py::Manifold | [source](../../rcpm/manifolds.py#L23) | [source](../snapshots/dbf533d/rcpm/manifolds.py#L23) | Identical AST |
| rcpm/manifolds.py::Sphere | [source](../../rcpm/manifolds.py#L66) | [source](../snapshots/dbf533d/rcpm/manifolds.py#L66) | Identical AST |
| rcpm/manifolds.py::Product | [source](../../rcpm/manifolds.py#L468) | [source](../snapshots/dbf533d/rcpm/manifolds.py#L251) | Identical AST |
| src/metrics.py::compute_ess | [source](../../src/metrics.py#L22) | [source](../snapshots/dbf533d/src/metrics.py#L20) | Identical AST |

The full `rcpm/flows.py`, `rcpm/utils.py` and `src/utils.py` files byte-match. RCPM flow initialization uses the `rcpm.densities` implementation, while training/evaluation base and target use `src.densities`; both relevant Sphere/Product paths were checked. Added SO/SE/entropic helpers do not replace those paths.

Recorded T2 settings agree: five transforms, 68 components, gamma 1, uniform alpha initialization, Adam learning rate .001, 5000 steps, batch size 256; Product(S1,S1), component jitter .01, target location [-1,0,-1,0] and scale [.3,.3]. Evaluation uses seed+1000 and five batches of 1024. Both have 1700 parameters with float32/float64 leaves.

Concrete execution differences:

- Historical code enables x64 in the driver; ZIP leaves it to the environment. The restored ZIP cohort explicitly enables it, so effective x64 is True in both paired cohorts. [historical setting](../snapshots/dbf533d/experiments/run_table.py#L32), [ZIP import](../../experiments/run_table.py#L35).
- ZIP disables XLA GPU command buffers/graphs in the driver. The historical-source JAX .6.2 records have XLA_FLAGS unset. Both disable preallocation. [ZIP launch flags](../../experiments/run_table.py#L20).
- Historical GPU selection also sets HIP_VISIBLE_DEVICES; ZIP sets only CUDA_VISIBLE_DEVICES. [historical GPU selection](../snapshots/dbf533d/experiments/run_table.py#L15).
- Restored ZIP uses JAX .4.35 / jaxlib .4.34 / Flax .8.4 / Optax .2.3; historical-source reruns use JAX .6.2 / jaxlib .6.2 / Flax .10.6 / Optax .2.4. All 25 recorded T2 evaluation subkeys differ despite the same prescribed seeds. The earlier [RNG comparison](rng_environment_comparison.md) independently established the partitionable default change.

The separate RNOT builder changes do not affect this native RCPM path. No changed RCPM sampling or metric recipe was found between these two source snapshots. This conclusion concerns reconstructing these executed configurations; original paper logs still lack a complete runtime/source provenance record. No runs or source changes were made.

Full hashes, AST evidence, all five record pairs, actual evaluation keys and environment metadata: [JSON evidence](rcpm_recipe_provenance.json).
