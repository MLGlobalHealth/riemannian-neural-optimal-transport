from pathlib import Path
import hashlib,json,shutil
base=Path('/private/tmp/mlgh-paper-release-20260908')
stage=base/'stage'
root=Path('/Users/alessandromicheli/Documents/dev/ml/projects/riemannian-neural-ot-finalMLGH')
assert (root/'archive/verification/REPORT.md').is_file()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for source in stage.rglob('*'):
    if not source.is_file() or source.is_symlink(): continue
    relative=source.relative_to(stage)
    if relative.parts[0]=='tests' or '__pycache__' in relative.parts or source.suffix in ('.pyc','.pyo') or source.name=='.DS_Store': continue
    destination=root/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    if relative.parts[0] in ('src','rcpm','experiments'):
        assert (root/'archive'/relative).is_file() and sha(source)==sha(root/'archive'/relative),relative
    shutil.copy2(source,destination)
    assert sha(source)==sha(destination)
obsolete=root/'rcpm/requirements.txt'
assert sha(obsolete)==sha(root/'archive/rcpm/requirements.txt')
obsolete.unlink()
checks=root/'archive/release_checks'
checks.mkdir()
shutil.copytree(stage/'tests',checks/'tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('archive_move_checks.json','dependency_inventory.md','runner_inventory.md','plot_inventory.md','runtime_packaging_review.md','runner_implementation.md','runner_staging_manifest.json','plot_delivery.json','docs_delivery.json','test_relocation_hashes.json','continental_native_extraction_check.json'):
    source=base/name
    if source.exists(): shutil.copy2(source,checks/name)
shutil.copytree(base/'gpu-validation',checks/'gpu-validation',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('source_export_manifest.json','archive_free_cli_checks.json'):
    shutil.copy2(base/'gpu-validation'/name,checks/name)
# New validation checkpoints are small and retained inside the archive with their JSONs.
# Existing large verification/artifacts remains ignored and unmoved from archive.
summary={'public_files':{str(p.relative_to(root)):sha(p) for directory in ('src','rcpm','experiments','paper','reference','requirements','docs','data','tools') for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts},'native_files_unchanged':23,'old_native_files_archived':24}
(base/'integration_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'root':str(root),'active_files':len(summary['public_files']),'native_files_unchanged':23},indent=2))
