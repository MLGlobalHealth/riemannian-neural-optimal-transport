from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/Users/alessandromicheli/Documents/dev/ml/projects/riemannian-neural-ot-finalMLGH')
original=Path('/Users/alessandromicheli/Documents/dev/ml/projects/riemannian-neural-ot')
stage=Path('/private/tmp/mlgh-paper-release-20260908/stage')
archive=root/'archive'
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='finalMLGH'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(), 'Expected clean worktree before archival'
assert not archive.exists()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(folder): return {str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}
pre={name:inventory(root/name) for name in ('verification','website','src','rcpm','experiments')}
archive.mkdir()
# Activate the new ignore rules before relocating ignored checkpoints.
shutil.copy2(stage/'.gitignore',root/'.gitignore')
for name in ('verification','website'):
    shutil.move(root/name,archive/name)
    assert inventory(archive/name)==pre[name]
for name in ('src','rcpm','experiments'):
    shutil.copytree(root/name,archive/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
    for relative,sha in pre[name].items():
        if '__pycache__' not in relative and not relative.endswith('.pyc') and not relative.endswith('.DS_Store'):
            assert digest(archive/name/relative)==sha
shutil.copy2(root/'README.md',archive/'README.md')
files=[original/x for x in ['Gromov Embeddings E1.ipynb','sphere_figure1.ipynb','torus_figure1.ipynb','continental_drift.ipynb','continental_drift_output.ipynb','continental_drift_data.ipynb','experiments/plot_results.ipynb']]
files += sorted((original/'experiments/final_results').glob('*'))
original_checks={}
for source in files:
    if not source.is_file(): continue
    relative=source.relative_to(original)
    destination=archive/'original'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    assert digest(source)==digest(destination)
    original_checks[str(relative)]={'sha256':digest(destination),'bytes':destination.stat().st_size}
(archive/'INDEX.md').write_text('''# Preserved research material

This folder preserves material outside the public paper experiment workflow. The active code, recipes and figure commands are documented in the repository root README. `tools/build_release.py` and Git source exports exclude this entire archive.

| Path | Contents |
|---|---|
| [README.md](README.md) | Previous branch README, frozen before packaging |
| [verification/](verification/) | Reproduction records, numerical comparisons, historical source snapshots and old audit tools |
| [website/](website/) | Local project page and its assets; publication remains on hold |
| `src/`, `rcpm/`, `experiments/` | Frozen pre-packaging native files, retained for historical hash checks and relative imports |
| [original/](original/) | Original notebooks and experiment outputs supporting the recovered plotting references |
| `release_checks/` | Packaging validation records, added with the release |

The frozen branch state is commit `6e6b7aa04a029aa6fcd3147e9c6b9a7ca2e6fa25`. Historical audit commands should be interpreted in that context; run old relative-path workflows from this directory, not from the new repository root. Historical records retain their original paths, dates and hashes rather than being rewritten as new experiments.

`verification/artifacts/` contains local ignored checkpoints and logs. These files remain on disk but are not added to Git or distributed. The website and previously committed verification records are moved without changing their bytes. The original main worktree is left untouched; supporting notebooks and logs here are copies.
''')
checks={'pre_packaging_commit':'6e6b7aa04a029aa6fcd3147e9c6b9a7ca2e6fa25','moved_unchanged':{name:pre[name] for name in ('verification','website')},'native_before':{name:pre[name] for name in ('src','rcpm','experiments')},'original_copies':original_checks}
Path('/private/tmp/mlgh-paper-release-20260908/archive_move_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'moved_files':{name:len(pre[name]) for name in ('verification','website')},'copied_original_files':len(original_checks),'archive':str(archive)},indent=2))
