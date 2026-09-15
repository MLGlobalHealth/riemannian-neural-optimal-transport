from pathlib import Path
import json,hashlib,re,datetime
root=Path('/private/tmp/mlgh-paper-release-20260908/gpu-validation')
evidence=root/'evidence'
new_path=evidence/'results/se3_native/results/liegroups_SE3_ours_fps_seed12345/result.json'
old_root=Path('/private/tmp/mlgh-liegroups-20260908/remote_evidence')
old_path=old_root/'results/SE3_ours_nativeJAX435/result.json'
a=json.loads(new_path.read_text());b=json.loads(old_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shared=('configuration','density_configuration','geometry_configuration','landmark_configuration','summary','parameter_count','parameter_dtypes','parameter_nonfinite_count')
checks={key:a[key]==b[key] for key in shared}
checks['all5_evaluation_keys_equal']=[x['prng_key'] for x in a['evaluation_batches']]==[x['prng_key'] for x in b['evaluation_batches']]
checks['all20_native_batch_scalar_values_equal']=[x['metrics'] for x in a['evaluation_batches']]==[x['metrics'] for x in b['evaluation_batches']]
checks['checkpoint_bytes_equal']=a['checkpoint']['sha256']==b['checkpoint']['sha256'] and (new_path.parent/a['checkpoint']['file']).read_bytes()==(old_path.parent/b['checkpoint']['file']).read_bytes()
newlog=evidence/'results/se3_native/logs/liegroups_SE3_ours_fps_seed12345.log'
oldlog=old_root/'logs/SE3_ours_nativeJAX435.log'
pattern=r'\[\s*(\d+)\]\s+loss=([^|\s]+)'
loss_a=re.findall(pattern,newlog.read_text());loss_b=re.findall(pattern,oldlog.read_text())
checks['all200_printed_training_losses_equal']=len(loss_a)==len(loss_b)==200 and loss_a==loss_b
assert all(checks.values()),checks
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'result':'pass','new_record':str(new_path),'new_record_sha256':sha(new_path),'prior_record':str(old_path),'prior_record_sha256':sha(old_path),'checks':checks,'native_metrics':a['summary'],'checkpoint':a['checkpoint'],'training_loss_count':len(loss_a),'runtime':a['timings_seconds'],'source_note':'All23 retained native source/license files have the same hashes; only historical rcpm/requirements.txt is absent from the public core manifest after its archival. No native Python file changed.','metadata_only_additions':{'selection':['suite','dimension','ablation'],'evaluation_config':['key_protocol']}}
(root/'se3_native_parity.json').write_text(json.dumps(result,indent=2)+'\n')
status=json.loads((evidence/'provenance/release_status.json').read_text())
validation={'status':'pass','checks':{name:{k:row.get(k) for k in ('status','pid','gpu','exit_code','started_at','finished_at')} for name,row in status['jobs'].items()},'se3_parity_checks':checks,'archive_free_cli':json.loads((root/'archive_free_cli_checks.json').read_text()),'remote_integrity':json.loads((evidence/'provenance/final_checks.json').read_text()),'file_delivery_verified':len(json.loads((root/'release_delivery.json').read_text())['files']),'figures':{},'source_changes_after_freeze':False,'full_continental_training':False,'additional_training_outside_scope':False}
for p in sorted((root/'plots').rglob('*')):
 if p.is_file():validation['figures'][str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
(root/'validation_summary.json').write_text(json.dumps(validation,indent=2)+'\n')
summary='''# Bounded release validation — passed

The frozen public code completed all seven declared checks on nvidia6 (RTX3090, driver550.90.07), using the unchanged Python3.11.15 / JAX0.4.35 environment. All56 delivered evidence files were fetched and independently checksum-verified. No packaging fix or native algorithm change was needed.

| Check | Scope | Outcome |
|---|---|---|
| SE3 RNOT | Full native200 updates, training seed12345, five1024-point evaluation batches | Passed; exact prior native results and checkpoint |
| S2 RNOT random | Main-table smoke, two updates, seed12345 | Passed |
| S2 RCPM gamma1 | Main-table smoke, two updates, seed12345 | Passed |
| S2 baseline ablation | Smoke, two updates, seed12345 | Passed |
| Continental | Explicit smoke: first256 input rows, eight landmarks, two updates,16-point export | Passed; regular/EMA checkpoint and samples retained |
| Landmark embeddings | Full default nine landmark counts, random and FPS, five diagnostic repetitions | Passed;18 rows rendered |
| S2 checkpoint transport | Restore the smoke checkpoint in a fresh process and export16 samples | Passed; source/config/array guards passed |

SE3 parity is exact: KL **2.4051482424868995**, ESS ratio **0.6701474356065226**. All200 printed training losses, allfive evaluation keys, all20 native batch scalar values, configuration/density/geometry/landmark metadata, and checkpoint bytes match the prior native run. Checkpoint SHA256: `59e80f9a35633e3f853025a928e83cbb7c6781d1cfd97251f0948d47b4faa0ce`. Detailed evidence: [se3_native_parity.json](se3_native_parity.json). This validates the packaging change against that prior run; it is not a new paper-wide numerical reproduction claim.

The full and smoke executions used the archive-free export in `source/`, which excludes tests, caches and all archived development material. All33 exported Python files AST-parse. Standard-library dry-run checks resolve all intended suite declarations from this independent export:30 table jobs,14 Lie-group jobs,126 sweep jobs,12 high-dimensional jobs and84 ablations. These declarations were not launched in full.

The continental run uses a separate pandas3.0.2 overlay; its Linux runtime requirements are already satisfied by the unchanged base. Its historical saved checkpoint is unavailable, so the full continental recipe remains recovered/unverified. The actual smoke solver records Adam=false and line-search=false, and its exported target key is42. Full continental training was not run.

Offline PNG/PDF rendering passed for [continental samples](plots/continental/transport.png), [restored S2 samples](plots/s2_checkpoint/transport.png), and [full embedding output](plots/embeddings/embeddings.png). All three figures were visually inspected: labels and panels render correctly, sample plots are explicitly labeled smoke, and the embedding figure retains its labeled display floor for zero near-collision rates. Renderers consume retained numeric arrays/records and require no archived helper or JAX import in offline mode.

Supervisor2480247 and all11 owned launcher/worker processes have exited; the final15:31:34UTC process check returned no rows for all12 owned PIDs. Every GPU was released with no remaining compute processes. The57 exported source/data/document files and the underlying80-pin freeze were unchanged after execution. GPU2 was excluded throughout.

Raw records, checkpoints, logs, manifests, installation-overlay evidence, launch capacity samples and exit checks are in [evidence/](evidence/). Machine-readable validation and figure checksums are in [validation_summary.json](validation_summary.json). Native source hashes match the prior run; the public source manifest excludes only the archived historical `rcpm/requirements.txt` metadata file.
'''
(root/'VALIDATION.md').write_text(summary)
print(json.dumps({'se3_checks':checks,'delivered_files':validation['file_delivery_verified'],'figure_files':len(validation['figures']),'validation_report':str(root/'VALIDATION.md')},indent=2))
