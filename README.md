# Riemannian Neural Optimal Transport

Code accompanying **Riemannian Neural Optimal Transport**, ICML 2026, by Alessandro Micheli, Yueqi Cao, Anthea Monod and Samir Bhatt.

[Paper](https://openreview.net/pdf?id=ez4oLq7PR3) · [Experiment and figure guide](docs/experiments.md)

RNOT learns continuous transport maps on Riemannian manifolds using distance-to-landmark features and a neural transport potential. This repository contains the experiment library, RCPM baseline, explicit experiment recipes, plotting commands, and available paper reference data.

## Install

The tested experiment environment is **Linux, Python 3.11 and an NVIDIA GPU**. From the repository root:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export CUDA_ROOT="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
```

The requirements pin JAX 0.4.35, jaxlib 0.4.34, Flax 0.8.4 and Optax 0.2.3, including the CUDA toolkit packages used in the tested environment. `CUDA_ROOT` lets this JAX release find that toolkit. The original AMD/ROCm environment is not reconstructed by this installation.

For **CPU plotting only**, install `requirements/plots.txt` in a separate Python 3.11 environment. Continental-drift training additionally uses `requirements/empirical.txt`.

## Run the main experiments

Preview the main-table plan, then run it on a GPU available for your use:

```sh
python -m paper.run tables --gpus 0 --output runs/tables --dry-run
python -m paper.run tables --gpus 0 --output runs/tables
python -m paper.plot tables --input runs/tables --output figures/tables
```

This trains the six available Tables 1–2 rows with five seeds each, using the recovered main-table settings. Use `--gpus 0 1` for two GPUs. Each GPU runs one job at a time. Choose a fresh output directory for every run; results, configurations, checkpoints, evaluation batches and logs are retained.

Run the recovered SO(3)/SE(3) configurations:

```sh
python -m paper.run liegroups --gpus 0 --output runs/liegroups
python -m paper.plot liegroups --input runs/liegroups --output figures/liegroups
```

The [experiment guide](docs/experiments.md) includes ablations, dimension sweeps, embedding computations, continental drift, and checkpoint-based transport plots.

## Generate plots without training

The compact `reference/` data includes paper-reported table values and recovered historical outputs. To render those explicitly labeled references:

```sh
python -m paper.plot tables --reference --output figures/reference-tables
python -m paper.plot sweep --reference --output figures/reference-dimensions
python -m paper.plot highd --reference --output figures/reference-highd
python -m paper.plot ablations --reference --output figures/reference-ablations
python -m paper.plot embeddings --reference --output figures/reference-embeddings
```

Plot commands export **PDF, PNG and the plotted numerical data**. They do not need a system LaTeX installation. Reference plots describe their saved inputs; fresh plots are generated from the run directory you supply.

## What is included

| Path | Purpose |
|---|---|
| `src/`, `rcpm/` | Original library and RCPM implementation |
| `experiments/` | Original supplied experiment entry points |
| `paper/` | Public experiment, checkpoint and plotting commands; explicit recipes |
| `reference/`, `data/` | Numerical plotting references and continental point clouds |
| `requirements/`, `docs/` | Pinned installations and experiment instructions |
| `archive/` | Preserved research history, verification evidence and website; omitted from release ZIPs |

The available runners cover RNOT and RCPM. RCNF/Moser and semi-dual RCPM runners remain unrecovered, and some numerical reproduction differences remain.

## Build the distributable code archive

```sh
python tools/build_release.py --output dist/riemannian-neural-optimal-transport-paper.zip
```

The ZIP includes the runtime code, recipes, plotting data and documentation, with a checksum manifest. It excludes `archive/`, generated outputs, checkpoints from past runs and development tests.

## Citation and license

```bibtex
@inproceedings{micheli2026riemannian,
  title={Riemannian Neural Optimal Transport},
  author={Micheli, Alessandro and Cao, Yueqi and Monod, Anthea and Bhatt, Samir},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=ez4oLq7PR3}
}
```

The existing [CC BY-NC 4.0 license](LICENSE) is retained. The vendored RCPM implementation retains its [original license](rcpm/LICENSE).
