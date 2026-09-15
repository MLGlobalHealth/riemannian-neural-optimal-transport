# Dependency and release inventory

Read-only audit of `riemannian-neural-ot-finalMLGH` at commit `6e6b7aa04a029aa6fcd3147e9c6b9a7ca2e6fa25` on 2026-09-08. No project source, packages, licenses or GPU jobs were changed. Detailed AST imports and local interpreter metadata are in `import_inventory.json` and `local_python_environments.json` beside this report.

## Direct requirements

The retained `src/`, `rcpm/`, and five native `experiments/` scripts reference nine external distributions. `mpl_toolkits.mplot3d` is supplied by Matplotlib; `jaxlib` is the compiled JAX runtime, not a separately imported application module.

| Distribution | Direct use | Release role |
|---|---|---|
| `jax` | Throughout algorithms and all experiment runners | Core; verified version 0.4.35 |
| `flax` | Networks, embeddings, training state, RCPM flows | Core; verified 0.8.4 |
| `optax` | Training and experiment optimizer construction | Core; verified 0.2.3 |
| `numpy` | Densities, geometry, solvers and experiments | Core; verified NVIDIA freeze 2.4.6 |
| `scipy` | `scipy.stats.gaussian_kde` in densities/empirical modules | Core; verified NVIDIA freeze 1.17.1 |
| `matplotlib` | Imported at module load by both manifold modules, plus plotting helpers | Required by unchanged imports even for the native runners; freeze 3.11.1 |
| `spherical-kde` | Optional import in `src/manifolds.py:14`, unconditional import in `rcpm/manifolds.py:9` | Required by unchanged RCPM module imports; freeze 0.1.2 |
| `cartopy` | Unconditional `cartopy.crs` in `rcpm/manifolds.py:18` | Required by unchanged RCPM module imports; freeze 0.25.0 |
| `pandas` | Top-level `src/empirical.py:5`, and local CSV-data import in `rcpm/densities.py:383` | Empirical-data feature dependency; absent from the verified 80-pin runtime |

The eight distributions excluding pandas cover the imports exercised by the verified native experiment runners. For a release retaining `src.empirical`, either include pandas as a declared dependency or document/install it as an empirical-data extra. The original 80-pin environment should remain unchanged: the tested native suites never imported that optional module. Existing local environments have pandas 3.0.2 with Python 3.11 and pandas 2.3.3 with Python 3.10, but neither is a newly validated native experiment configuration.

The retained files do **not** directly import Hydra, OmegaConf, IPython, setproctitle, scikit-learn, einops, dm-tree, tqdm, PyYAML, wandb, JupyterLab or ipykernel. Some are present in old root/rcpm requirement lists, and some are transitive dependencies of retained libraries. In particular, the current `rcpm/flows.py` uses its own small `instantiate` function rather than importing Hydra. Preserve the vendored requirements as historical upstream material if desired, but do not describe that broad list as the minimal current application requirements.

The RCPM modules use legacy flat imports (`import utils`, `import manifolds`), and native runners explicitly insert both the project root and `rcpm/` on `sys.path`. A companion repository can remain script-oriented. If introducing installation metadata, do not assume that installing `src` and `rcpm` as ordinary packages alone fixes those flat imports; keep the entrypoint/runtime path behavior explicit and test it from an export outside the development tree.

## Exact NVIDIA installation versus a minimal list

The complete freeze currently at `verification/environments/reproduction-cuda.txt` contains 80 resolved package versions, including CUDA runtime wheels, Flax/JAX transitive dependencies, mapping dependencies and documentation/test dependencies pulled in by upstream packages. It is a tested environment record, not a minimal list of directly imported application dependencies.

The saved pip installation report explicitly records `spherical_kde==0.1.2` requiring `cartopy`, `pytest`, `numpy`, `scipy`, `matplotlib`, `pypandoc` and `numpydoc`; the last two account for much of the documentation stack. Do not remove those transitive pins from the exact reproduction freeze while presenting it as the same verified environment.

The successfully resolved installation request was:

```sh
python -m pip install 'jax[cuda12]==0.4.35' 'flax==0.8.4' 'optax==0.2.3' spherical_kde cartopy matplotlib
```

The saved JAX wheel metadata specifies `jaxlib==0.4.34` for its `cuda12` extra, and allows the CUDA12 plugin from 0.4.34 through 0.4.35. The recorded supported resolution is JAX 0.4.35, jaxlib 0.4.34, CUDA12 plugin/PJRT 0.4.35, Flax 0.8.4 and Optax 0.2.3. Preserve those actual versions; forcing jaxlib 0.4.35 would no longer match this verified installation.

For the paper recipe, recommend a fresh Linux Python 3.11 environment and `python -m pip install -r <relocated-exact-CUDA-freeze>`, followed by `python -m pip check`. This exact freeze was installed and used successfully on both NVIDIA servers. A direct-dependency list with only core pins is useful documentation or a convenience install, but is not evidence that its newly resolved transitive packages reproduce the recorded results. The CUDA freeze is not an offline macOS plotting installation.

Keep the existing launch settings: x64 enabled; JAX 0.4.35 default Threefry partitionable false; no preallocation; `CUDA_ROOT` pointing to the selected venv's `lib/python3.11/site-packages/nvidia/cuda_nvcc`; OMP/OpenBLAS/MKL/NumExpr threads 2. The unchanged Lie-group native runners append their own XLA command-buffer/graph flags. The newer companion launcher should preserve those native flags and record its environment. The freeze name's underscore versus hyphen spelling for `spherical_kde` is a normalized package-name difference, not a version difference.

Verified remote environments, last checked after all native/replay jobs exited:

| Host | Python executable | CUDA_ROOT |
|---|---|---|
| nvidia6 | `/data/am1118/mlgh-zip-jax0435-restored-20260908/venv/bin/python` | `/data/am1118/mlgh-zip-jax0435-restored-20260908/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc` |
| nvidia7 | `/data/am1118/mlgh-dbf533d-jax0435-candidate-20260908/venv/bin/python` | `/data/am1118/mlgh-dbf533d-jax0435-candidate-20260908/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc` |

Both contain the same 80 normalized versions, Python 3.11.15, and pass pip check. nvidia6 freeze SHA256 is `59bf165a84bbf82fa6ddeafc4e0472fa25c74a501c4526084c6e8000cb37f559`; nvidia7 freeze SHA256 is `f7b7335f0a864a57487be28384442d962d2519edd54a4c4a21e6382528779171`. The hosts/drivers differ (RTX3090/550.90.07 versus RTX6000Ada/570.195.03); do not pool numerical host results or call this the original AMD environment.

## License and citation files

The untracked `LICENSE` in the original main worktree is byte-identical to `rcpm/LICENSE` in both worktrees: 19,347 bytes, SHA256 `41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0`, titled Attribution-NonCommercial 4.0 International. RCPM source retains Facebook copyright headers. Root can preserve the author repository's existing license bytes and the vendored RCPM license; this audit does not introduce a new license or infer a different grant.

The companion website already has `website/assets/citation.bib` and `website/assets/paper-metadata.json`. The stable citation is arXiv 2602.03566 (2026), title Riemannian Neural Optimal Transport, authors Alessandro Micheli, Yueqi Cao, Anthea Monod and Samir Bhatt. Metadata records OpenReview ID `ez4oLq7PR3`, the camera-ready printed repository `https://github.com/MLGlobalHealth/riemannian-neural-optimal-transport`, and printed ICML/PMLR306 venue metadata. The citation deliberately does not invent proceedings page ranges or a proceedings DOI. Preserve these existing metadata records or a small root citation file when archiving the website.

## Offline local checks without a new installation

Default `python3` is `/Users/alessandromicheli/.pyenv/versions/3.12.8/bin/python3`, with standard library and pip but no NumPy/Matplotlib/JAX/pytest. It is useful for AST parsing, JSON/manifest validation, standard-library unit tests and help/dry-run commands that do not import the training stack.

For offline plot rendering, use `/Users/alessandromicheli/miniforge3/envs/entropic-rnot/bin/python`: Python 3.11.15, NumPy 2.4.4, Matplotlib 3.10.9, pandas 3.0.2. A small in-memory Agg PNG render succeeded in this audit without importing JAX or using the network. Set `MPLBACKEND=Agg`, `MPLCONFIGDIR=/private/tmp/mlgh-paper-release-20260908/mplconfig`, and `XDG_CACHE_HOME=/private/tmp/mlgh-paper-release-20260908/cache` to keep font caches writable and confined to the task directory. Its JAX 0.10.0/Flax 0.12.7 is **not** the paper training runtime; use this environment only for independent NumPy/Matplotlib plotting and lightweight CLI checks.

Alternative local plotting environment: `/Users/alessandromicheli/miniforge3/envs/rcpms-jax/bin/python`, Python 3.10.19, NumPy 2.2.6, Matplotlib 3.10.8, pandas 2.3.3. Its JAX 0.6.2 also differs from the validated stack. Neither local conda environment has Cartopy/spherical-kde. The temporary `/private/tmp/mlgh-page-assets/venv/bin/python` used for paper assets contains PDF/image tools but no NumPy/Matplotlib. No packages were installed or changed for this audit.

## Proposed bounded release validation

1. Build a temporary export containing the intended public files only, with the archive directory excluded. Run all checks from that export with an unrelated working directory and no development-tree PYTHONPATH. AST-parse all retained Python files and validate the small metadata/result fixtures.
2. Exercise the new standard-library CLI help and dry-run/plan modes for the S2/T2 table, high-dimensional and SO3/SE3 selections. Assert they resolve paths within the export, do not import archived verification helpers, and do not launch training. Original native scripts import JAX before their main CLI, so prefer the new lazy-import CLI for offline checks.
3. Render the new plotting CLI from retained small numerical records/fixtures using the local NumPy/Matplotlib environment, including PDF/PNG outputs, nonfinite rows and the distinct training-seed versus evaluation-batch protocols. Check that a companion export does not depend on archived website images, copied historical directories or absolute developer paths. These plot checks validate packaging/presentation, not paper numerical agreement.
4. After root freezes the export/runner, use a fresh isolated nvidia6 release-check directory and the unchanged 80-pin venv for the bounded authorized end-to-end run. A native SE3 RNOT 200-step run was previously about34 seconds of training and has an existing result/checkpoint for exact configuration/key/metric comparison. A clearly labeled two-step S2 smoke can exercise the other command path; its result must not be presented as paper reproduction. Root will choose/freeze exact scope before launch.
5. If those requested checks pass, verify output/checkpoint hashes and source preservation, run the checkpoint plotting entrypoint on its produced artifact if supported, and finish packaging. No full-table retraining or additional parameter sweep is required merely to move files and create the companion entrypoint.

No release-check GPU job has been launched in this audit.
