"""Focused comparison tests using synthetic evidence, never experiment results."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare_results.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
paper=json.loads(Path(__file__).with_name('paper_targets.json').read_text())


def record(seed=12345,kl=.03,host='nvidia6',gpu='0'):
    values={'kl':kl,'ess':993.28,'ess_ratio':.97}
    return {'task_id':str(seed),'status':'complete','label':'zip_defaults','smoke':False,
      'selection':{'suite':'table','manifold':'sphere','dimension':2,'method':'ours','seed':seed,'gamma':None,'landmark_method':'fps'},
      'source':{'sha256':'archive_a'},'runner_sha256':'runner_a','builder':'native_archive_builder',
      'configuration':{'model':{'n_landmarks':256},'training':{'seed':seed,'n_steps':500}},
      'explicit_overrides':{},'evaluation_config':{'batch_size':1024,'n_batches':5,'seed':seed+1000},
      'environment':{'hostname':host,'packages':{'jax':'0.4.33'},'environment':{'CUDA_VISIBLE_DEVICES':gpu,'XLA_FLAGS':'same'},
       'nvidia_smi':'0, H100, GPU-a, 555, 81920 MiB\n1, H100, GPU-b, 555, 81920 MiB','jax_devices':['CudaDevice(id=0)']},
      'evaluation_batches':[{'index':i,'metrics':values.copy(),'metric_nonfinite':{},'residual_diagnostics':{'nonfinite_count':0,'transport_nonfinite_count':0}} for i in range(5)],
      'summary':{metric:{'mean':value} for metric,value in values.items()},
      'parameter_nonfinite_count':0,'timings_seconds':{'training_native_return':10.,'training_synchronized':10.01}}


class CompareTests(unittest.TestCase):
    def report(self,records):
        with tempfile.TemporaryDirectory(prefix='compare-test-') as tmp:
            root=Path(tmp)
            for index,r in enumerate(records):
                path=root/str(index)/'result.json';path.parent.mkdir();path.write_text(json.dumps(r))
            result=c.build_report([root],paper)
            json.dumps(result,allow_nan=False)
            self.assertIn('Experimental verification comparison',c.markdown(result))
            return result

    def test_five_seed_aggregation_and_gpu_indices(self):
        records=[record(seed,kl=.028+i*.001,gpu=str(i%2)) for i,seed in enumerate(c.SHIPPED_SEEDS)]
        report=self.report(records);self.assertEqual(len(report['groups']),1)
        g=report['groups'][0];self.assertEqual(g['status'],'complete_five_seeds')
        self.assertEqual(g['valid_seeds'],c.SHIPPED_SEEDS)
        self.assertAlmostEqual(g['across_seed_statistics']['kl']['mean'],.03)
        self.assertAlmostEqual(g['across_seed_statistics']['kl']['se_ddof0'],.001414213562373095/5**.5)
        self.assertAlmostEqual(g['across_seed_statistics']['kl']['se_ddof1'],.00158113883008419/5**.5)
        self.assertTrue(g['paper_comparison']['both_metric_means_round_as_printed'])

    def test_environment_source_config_and_label_separate(self):
        base=record();variants=[]
        r=deepcopy(base);r['environment']['hostname']='nvidia7';variants.append(r)
        r=deepcopy(base);r['environment']['packages']['jax']='0.5.0';variants.append(r)
        r=deepcopy(base);r['source']['sha256']='archive_b';variants.append(r)
        r=deepcopy(base);r['configuration']['model']['n_landmarks']=128;variants.append(r)
        r=deepcopy(base);r['label']='paper_settings';variants.append(r)
        self.assertEqual(len(self.report([base]+variants)['groups']),6)

    def test_partial_is_explicit(self):
        g=self.report([record()])['groups'][0]
        self.assertEqual(g['status'],'partial_fewer_than_five_seeds')
        self.assertFalse(g['paper_comparison']['complete_five_seed_comparison'])
        self.assertIsNone(g['across_seed_statistics']['kl']['se_ddof1'])

    def test_smoke_with_custom_label_excluded(self):
        r=record();r['smoke']=True;r['label']='looks_like_full'
        report=self.report([r]);self.assertFalse(report['groups']);self.assertEqual(len(report['excluded_runs']),1)

    def test_nonfinite_failure_not_silently_dropped(self):
        records=[record(seed) for seed in c.SHIPPED_SEEDS]
        records[2]['evaluation_batches'][0]['metrics']['kl']='NaN'
        report=self.report(records);g=report['groups'][0]
        self.assertEqual(g['status'],'nonfinite_failure');self.assertEqual(g['independent_valid_seed_count'],4)

    def test_duplicate_seed_not_cherry_picked(self):
        records=[record(seed) for seed in c.SHIPPED_SEEDS]
        r=deepcopy(records[0]);r['started_at']='different_attempt';records.append(r)
        g=self.report(records)['groups'][0]
        self.assertEqual(g['status'],'duplicate_seed_conflict');self.assertEqual(g['independent_valid_seed_count'],4)

    def test_identical_copies_not_duplicate_seeds(self):
        records=[record(seed) for seed in c.SHIPPED_SEEDS];records.append(deepcopy(records[0]))
        report=self.report(records)
        self.assertEqual(report['groups'][0]['status'],'complete_five_seeds')
        self.assertEqual(len(report['duplicate_file_copies']),1)

    def test_override_ineligible_for_default_completeness(self):
        records=[record(seed) for seed in c.SHIPPED_SEEDS]
        for r in records:r['explicit_overrides']={'training':{'n_steps':1000}}
        report=self.report(records);g=report['groups'][0]
        self.assertTrue(g['paper_comparison']['complete_five_seed_comparison'])
        self.assertFalse(g['paper_comparison']['native_default_five_seed_comparison'])
        self.assertTrue(g['paper_comparison']['configured_five_seed_comparison'])
        env=next(iter(report['environment_completeness'].values()))
        self.assertEqual(env['complete_native_table_rows'],[])
        self.assertEqual(env['complete_configured_table_rows'],['S2:RNOT_FPS'])
        self.assertEqual(env['complete_table_rows'],['S2:RNOT_FPS'])
        self.assertIn('S2:RNOT_FPS',env['missing_native_table_rows'])
        self.assertNotIn('S2:RNOT_FPS',env['missing_table_rows'])
        coverage=report['table_row_coverage']
        self.assertEqual(coverage['complete_native_table_rows'],[])
        self.assertEqual(coverage['complete_configured_table_rows'],['S2:RNOT_FPS'])
        row=coverage['rows']['S2:RNOT_FPS']
        self.assertEqual(row['complete_native_group_ids'],[])
        self.assertEqual(row['complete_configured_group_ids'],[g['group_id']])

    def test_partial_configurations_never_pool_for_row_coverage(self):
        records=[record(seed) for seed in c.SHIPPED_SEEDS]
        for index,r in enumerate(records):
            landmarks=128 if index<3 else 256
            r['explicit_overrides']={'model':{'n_landmarks':landmarks}}
            r['configuration']['model']['n_landmarks']=landmarks
        report=self.report(records)
        self.assertEqual(len(report['groups']),2)
        self.assertEqual(sum(g['independent_valid_seed_count'] for g in report['groups']),5)
        self.assertEqual(report['table_row_coverage']['complete_table_rows'],[])
        row=report['table_row_coverage']['rows']['S2:RNOT_FPS']
        self.assertFalse(row['covered_by_complete_five_seed_group'])
        self.assertEqual(row['complete_configured_group_ids'],[])
        self.assertEqual(len(row['groups']),2)
        env=next(iter(report['environment_completeness'].values()))
        self.assertEqual(env['complete_configured_table_rows'],[])

    def test_overall_coverage_keeps_mixed_hosts_labels_and_sources_explicit(self):
        records=[]
        for manifold in ('sphere','torus'):
            for method in ('fps','random','rcpm'):
                for seed in c.SHIPPED_SEEDS:
                    r=record(seed,host='nvidia6' if manifold=='torus' and method!='rcpm' else 'nvidia7')
                    r['selection']['manifold']=manifold
                    if method=='rcpm':
                        r['selection'].update(method='rcpm',landmark_method=None,gamma=1.0)
                        r['label']='native_rcpm'
                        r['source']['sha256']='source_native_rcpm'
                    else:
                        r['selection']['landmark_method']=method
                        r['explicit_overrides']={'model':{'n_landmarks':128}}
                        r['configuration']['model']['n_landmarks']=128
                        r['label']='restored_settings'
                    records.append(r)
        report=self.report(records);coverage=report['table_row_coverage']
        self.assertEqual(len(report['groups']),6)
        self.assertTrue(coverage['all_six_rows_have_complete_five_seed_group'])
        self.assertEqual(coverage['missing_table_rows'],[])
        self.assertEqual(len(coverage['complete_native_table_rows']),2)
        self.assertEqual(len(coverage['complete_configured_table_rows']),4)
        for env in report['environment_completeness'].values():
            self.assertFalse(env['all_six_archive_table_rows_complete'])
            self.assertFalse(env['all_six_table_rows_complete_in_this_environment'])
        by_id={g['group_id']:g for g in report['groups']}
        for row in coverage['rows'].values():
            self.assertEqual(len(row['groups']),1)
            witness=row['groups'][0];group=by_id[witness['group_id']]
            self.assertEqual(witness['valid_seeds'],c.SHIPPED_SEEDS)
            self.assertEqual(witness['hostname'],group['identity']['environment']['hostname'])
            self.assertEqual(witness['label'],group['identity']['label'])
            self.assertEqual(witness['source_sha256'],group['identity']['source_sha256'])
            self.assertEqual(witness['environment'],group['identity']['environment'])
            self.assertIn(witness['environment_id'],report['environment_completeness'])
        self.assertIn('Overall row coverage: 6/6',c.markdown(report))

    def test_rounding_is_printed_precision_not_plusminus_interval(self):
        self.assertTrue(c.rounding_check(.034999,.03,2)['observed_mean_in_rounding_interval'])
        self.assertFalse(c.rounding_check(.035001,.03,2)['observed_mean_in_rounding_interval'])
        self.assertFalse(c.rounding_check(.131,.13,4)['observed_mean_in_rounding_interval'])

    def test_failed_run_retained(self):
        r=record();r['status']='failed';r['error']={'type':'RuntimeError','message':'OOM'}
        report=self.report([r]);g=report['groups'][0]
        self.assertEqual(g['status'],'partial_with_failed_or_unfinished_runs')
        self.assertEqual(g['runs'][0]['errors'][0]['error']['message'],'OOM')

    def test_diagnostic_nonfinite_invalidates_metrics(self):
        r=record();r['evaluation_batches'][0]['residual_diagnostics']['nonfinite_count']=1
        self.assertEqual(self.report([r])['groups'][0]['status'],'nonfinite_failure')

    def test_mismatched_summary_invalid(self):
        r=record();r['summary']['kl']['mean']=0.4
        g=self.report([r])['groups'][0]
        self.assertFalse(g['runs'][0]['valid_for_aggregation'])
        self.assertEqual(g['runs'][0]['errors'][0]['kind'],'summary_inconsistency')

    def test_cli_relative_evidence_survives_bundle_relocation(self):
        with tempfile.TemporaryDirectory(prefix='compare-relocation-') as tmp:
            root=Path(tmp);bundle=root/'original';verification=bundle/'verification'
            results=bundle/'results';output=verification/'comparison'
            verification.mkdir(parents=True)
            for name in ('compare_results.py','paper_targets.json'):
                shutil.copy2(Path(__file__).with_name(name),verification/name)
            r=record();r['environment']['executable']='/recorded/remote/venv/bin/python'
            smoke=deepcopy(r);smoke['smoke']=True
            for name,value in (('run',r),('copy',r),('smoke',smoke)):
                path=results/name/'result.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(value))
            bad=results/'bad'/'result.json';bad.parent.mkdir();bad.write_text('{invalid')
            original_bytes={str(path.relative_to(bundle)):path.read_bytes() for path in results.rglob('result.json')}
            absolute=c.build_report([results],paper)
            absolute_before=deepcopy(absolute)
            relative=c.relative_evidence_paths(absolute,output)
            self.assertEqual(absolute,absolute_before)
            self.assertEqual(relative['groups'][0]['identity'],absolute['groups'][0]['identity'])
            self.assertEqual(relative['groups'][0]['group_id'],absolute['groups'][0]['group_id'])

            subprocess.run([sys.executable,str(verification/'compare_results.py'),
                            '--results-root',str(results),'--output-dir',str(output),'--relative-paths'],
                           check=True,capture_output=True,text=True)
            moved=root/'relocated';shutil.move(str(bundle),moved)
            output=moved/'verification'/'comparison'
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(report['evidence_path_base'],'output_directory')
            self.assertEqual(report['groups'][0]['identity'],absolute['groups'][0]['identity'])
            self.assertEqual(report['groups'][0]['group_id'],absolute['groups'][0]['group_id'])
            self.assertFalse(bundle.exists())
            paths=list(report['input_roots'])
            for group in report['groups']:
                paths.extend(run['path'] for run in group['runs'])
            for key in ('excluded_runs','duplicate_file_copies','read_errors'):
                for entry in report[key]:
                    paths.append(entry['path'])
                    if 'same_record_as' in entry:paths.append(entry['same_record_as'])
            for path in paths:
                self.assertFalse(Path(path).is_absolute())
                self.assertTrue((output/path).exists(),path)
            markdown=(output/'comparison.md').read_text()
            for run in report['groups'][0]['runs']:
                self.assertIn(f"]({run['path']})",markdown)
            self.assertNotIn(str(bundle),markdown)
            for path,contents in original_bytes.items():
                self.assertEqual((moved/path).read_bytes(),contents)


if __name__=='__main__':unittest.main()
