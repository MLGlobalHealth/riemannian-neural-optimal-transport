# Table 5 reproduction protocol

The supplied runners fit the paper’s SO(3) and compact-support SE(3) comparison in method and distribution family. Their exact supplied settings must remain identifiable: the paper does not specify the full Lie-group training recipe or an execution source revision. The native protocol trains once per method/gamma and averages five held-out batches; it is not the earlier five-training-seed main-table protocol.

## Exact camera-ready targets

Table 5 is on PDF page 32; [direct table crop](paper_protocol.png).

| Manifold | Method | KL ± printed quantity | ESS ratio |
| --- | --- | ---: | ---: |
| SO3 | RNOT | 2.96 ± 0.01 | 0.225 |
| SO3 | RCPM γ=1 | 2.86 ± 0.03 | 0.391 |
| SO3 | RCPM γ=0.1 | 3.29 ± 0.10 | 0.002 |
| SO3 | RCPM γ=0.05 | 3.54 ± 0.06 | 0.006 |
| SO3 | RCPM γ=0.01 | 2.83 ± 0.10 | 0.003 |
| SO3 | RCPM γ=0.005 | 3.72 ± 0.10 | 0.004 |
| SO3 | RCPM γ=0.001 | 5.90 ± 0.06 | 0.004 |
| SE3 | RNOT | 2.50 ± 0.01 | 0.683 |
| SE3 | RCPM γ=1 | 14.38 ± 0.09 | 0.007 |
| SE3 | RCPM γ<1 | Numerically unstable | – |

Table 5 does not define its ± construction or specify independent training repetitions. Both supplied scripts compute `np.std(batch_values, ddof=0)/sqrt(5)` over five evaluation batches. ESS in the table is the normalized ratio, not the unnormalized `ess` output. The qualitative instability row supplies no threshold or exact failure type; preserve each gamma’s observed result separately.

## Native runner recipe

- Both: default FPS, 4096 candidate samples for each of base and target; half the landmarks from each distribution. One model per method/gamma, training seed 12345, evaluation seed 12345. Reset the evaluation key for each method/gamma, then split sequentially into five subkeys. Each native evaluation uses 1024 samples. No +1000 seed offset.
- SO3 RNOT: `build_ours_experiment` (lines 125–205) inherits companion `src/base.py`. With the unchanged ZIP this means MLP 128×128, 256 landmarks, 500 outer steps, 500 inner steps, minimum 50, inner LR0.005, no inner Adam, no LogSumExp initialization, line search enabled. RNOT initialization uses `exp_cfg.training.seed` (line 137), not the module’s `TRAINING_SEED` constant.
- SE3 RNOT: `build_experiment` (lines 80–204) explicitly sets MLP 128×128, 128 landmarks, 200 outer steps, 100 inner steps, minimum 500, inner LR0.005, tolerance 1e-6, LogSumExp initialization with γ = 0.01, no inner Adam, and line search enabled. Outer training uses batch 256, LR 0.001 and cosine decay to 0.05 of the initial LR. Preserve these supplied constants; no inferred correction is made.
- RCPM on both: 68 components per transform, 5 transforms, 5000 training iterations, batch 256, Adam LR 0.001, native KL objective. The gamma grid is 1, 0.1, 0.05, 0.01, 0.005, 0.001. The `train_rcpm` helper defaults 500/128 are overridden by native main; record actual 5000/256.
- Total default native scope: 7 trained configurations per manifold, 14 overall. No FPS/random split appears in Table 5. The CLI has no seed or gamma selector; landmark selection affects RNOT only.

## Distribution details and source dependency

SO3 uses uniform rotations and a wrapped-normal target centered at `manifold.zero()` with tangent scale 0.3. The paper describes this family but does not give the exact location and scale in its Lie-group paragraph.

Both SE3 builders use uniform rotations × uniform translations in `[-4,4]^3`. The target is `SE3FactorizedCompact`: rotation centered at 60° about z, translation at `[1,0.5,-0.5]`, rotation scale 0.3, translation scale 0.5, and translation truncation bounds `[-4,4]^3`. This agrees with the distribution family described in G.2, but its numeric box/center/scales are supplied-runner details. **The console banner is stale:** lines 376–377 say `[-2,2]^3` and `SE3WrappedNormal`; those strings do not describe the actual builder objects.

Paper F.2 (page 26) gives 128 landmarks, 1000 outer steps and 2500 inner steps for the sphere/torus experiments, plus RCPM 68 components, batch 256, 5000 steps. It states five evaluation batches of 1024 and AMD MI300X hardware. G.2 (page 31) extends the distribution description to SO3/SE3 without restating an exact training recipe. The differences between the new runners and F.2 therefore warrant provenance labels, not silent replacement of the supplied settings.

## Evidence and bounded source search

[Machine-readable protocol, rounding targets, exact effective configs, line references and source hashes](paper_protocol.json). Camera-ready PDF SHA256: `318386d9f934d70a7758df4f931058b68b5182b4bd60e9996b3c524b569f3e89`. New runner SHA256 values: SO3 `3b5ffa41ee137c99ef05dc3e93369e86d85ee8c6fdbf695b69b26ebff4c6d7d8`; SE3 `c0c7f6609d359ec1636f42dd035c54f3d2dede2eb905240de922eb211b2f108e`.

The existing unpacked ICML LaTeX main/sections and two exact ICML source ZIPs were checked for SO3/SE3, the distinctive Table 5 numbers and output references. They contain no such Lie-group protocol/outputs. This bounded search does not establish that no later source exists. The camera-ready PDF remains the authoritative Table 5 target for this check.

No runner, library, website or mathematical implementation was modified. No training or numerical probe was run for this protocol audit.
