# Stored ablation results supporting Table 3

**Historical stored output; not a fresh rerun.** These four JSON files were extracted byte-for-byte from the original repository at Git revision
`a7327bc829837c62d5e41ba222e89ea7b4900d83` (`a7327bc`, committed 2026-01-26). No reported or published values were changed.

The original paths are `experiments/final_results/ablation_{S2,T2,S10,T10}.json`. Each copied file was checked against its exact Git blob.

| File | SHA-256 |
|---|---|
| [ablation_S2.json](ablation_S2.json) | `52ee1e949323c6a814ad9a1e221b8d59eb549346f2786beb6f8c668502c194f6` |
| [ablation_T2.json](ablation_T2.json) | `ece12011a460332c707ccb6b6fc164238d036369b4c783fdcba700aeb7ca3247` |
| [ablation_S10.json](ablation_S10.json) | `74292677adca1cbf4a1eb4409454cb54050aaf7c35fa4f54b9319b4669f7ca48` |
| [ablation_T10.json](ablation_T10.json) | `8a6e91dc5d680c7463051888396985fe95cab828557266dc687d60541c727bd7` |

[comparison.json](comparison.json) compares all **79 numeric cells** of published Table 3 with these stored KL values at the paper’s displayed precision. All 79 match after rounding. The paper’s one blank cell is excluded from the numeric comparison. The comparison preserves the unrounded stored values and original published targets.

Targets come from [paper_targets.json](../../paper_targets.json); their source PDF identities and revision limitations are recorded there. The comparison JSON includes the target-file hash, exact source revision, stored-file hashes, per-cell values, and rounding rule.

This establishes provenance of the published numbers in previously committed output. It does not establish that a fresh execution reproduces them.
