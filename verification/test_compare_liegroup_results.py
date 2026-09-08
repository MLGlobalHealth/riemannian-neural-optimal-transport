"""Synthetic record tests; no JAX import, GPU, model execution or source edits."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('liegroup_compare', Path(__file__).with_name('compare_liegroup_results.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
TARGET_PATH = Path(__file__).with_name('paper_targets.json')
TARGETS = m.load_targets(TARGET_PATH)


def fixture(manifold='SO3', method='ours', gamma=None, kl=2.96, gpu=0):
    values = [kl - .2, kl - .1, kl, kl + .1, kl + .2]
    batches = []
    for i, value in enumerate(values):
        metrics = {'kl': value, 'ess': 512., 'ess_ratio': .5}
        batches.append({'index': i, 'prng_key': [i, i + 100], 'metrics': metrics,
                        'metric_nonfinite': {name: False for name in metrics}})
    r = {'record_type': 'native_liegroup_training_run', 'status': 'complete',
         'label': 'recovered_liegroup_native_cuda_jax435', 'started_at': 'one original execution',
         'selection': {'manifold': manifold, 'method': method, 'gamma': gamma, 'seed': 12345,
                       'landmark_method': 'fps' if method == 'ours' else None},
         'source': {'sha256': 'a' * 64}, 'runner_sha256': 'b' * 64, 'helper_sha256': 'c' * 64,
         'native_runner': {'file': f'experiments/run_{manifold.lower()}_experiment.py', 'sha256': 'd' * 64},
         'configuration': {'model': {'n_landmarks': 256}, 'training': {'seed': 12345}},
         'explicit_configuration_overrides': {},
         'evaluation_config': {'seed': 12345, 'batch_size': 1024, 'n_batches': 5, 'seed_offset': None},
         'environment': {'hostname': 'nvidia7', 'packages': {'jax': '0.4.35'},
             'environment': {'CUDA_VISIBLE_DEVICES': str(gpu), 'HIP_VISIBLE_DEVICES': str(gpu)},
             'jax_devices': ['cuda:0'],
             'nvidia_smi': '0, RTX 6000 Ada, GPU-zero, 570.195.03, 49140 MiB\n1, RTX 6000 Ada, GPU-one, 570.195.03, 49140 MiB'},
         'evaluation_batches': batches, 'parameter_nonfinite_count': 0, 'source_unchanged_after_run': True}
    summarize(r)
    return r


def summarize(record):
    record['summary'] = {}
    for metric in m.METRICS:
        raw = [batch['metrics'][metric] for batch in record['evaluation_batches']]
        if all(math.isfinite(float(x)) for x in raw):
            mean = statistics.mean(raw); se0 = statistics.pstdev(raw) / math.sqrt(len(raw))
            se1 = statistics.stdev(raw) / math.sqrt(len(raw)) if len(raw) > 1 else 'NaN'
        else:
            mean = se0 = se1 = 'NaN'
        record['summary'][metric] = {'mean': mean, 'se_ddof0': se0, 'se_ddof1': se1, 'raw_batches': raw}


def nonfinite(record):
    for batch in record['evaluation_batches']:
        batch['metrics']['kl'] = 'NaN'; batch['metric_nonfinite']['kl'] = True
    summarize(record)
    return record


def write(root, name, record):
    path = root / name / 'result.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, allow_nan=False))
    return path


class LiegroupComparisonTests(unittest.TestCase):
    def test_fourteen_native_rows_use_batch_se_and_preserve_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = []
            for i, row in enumerate(m.ROWS):
                manifold, method, *gamma = row.split('/')
                r = fixture(manifold, 'ours' if method == 'RNOT' else 'rcpm', float(gamma[0]) if gamma else None, gpu=i % 2)
                if TARGETS[row]['kind'] == 'qualitative':
                    r = nonfinite(r)
                paths.append(write(root, str(i), r))
            report = m.compare(paths, TARGETS, root / 'out')
            self.assertEqual(len(report['cohorts']), 1)
            cohort = next(iter(report['cohorts'].values()))
            self.assertEqual(cohort['status'], 'complete_14_native_runs')
            self.assertEqual(cohort['complete_native_row_count'], 14)
            self.assertEqual(report['record_outcome_counts'], {'complete_finite': 9, 'observed_numerical_nonfinite': 5})
            ours = next(x for x in report['records'] if x['row'] == 'SO3/RNOT')
            self.assertAlmostEqual(ours['recomputed_batch_statistics']['kl']['mean'], 2.96)
            self.assertAlmostEqual(ours['recomputed_batch_statistics']['kl']['se_ddof0'], math.sqrt(.02 / 5))
            self.assertEqual(ours['evaluation_batch_count'], 5)
            self.assertTrue(ours['paper_comparison']['kl_mean']['matches_printed_mean'])
            self.assertEqual(report['protocol']['training_repetitions_per_record'], 1)
            self.assertTrue(all(x['paper_comparison']['observed_nonfinite_instability'] for x in report['records']
                                if TARGETS[x['row']]['kind'] == 'qualitative'))

    def test_duplicate_runs_never_average_and_identical_copies_are_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = fixture(kl=2.0); second = fixture(kl=4.0)
            second['started_at'] = 'second independent execution'
            a = write(root, 'first', first); b = write(root, 'second', second); alias = write(root, 'copy', first)
            report = m.compare([a, b, alias], TARGETS, root / 'out')
            self.assertEqual(report['unique_record_count'], 2)
            self.assertEqual(report['identical_record_copy_count'], 1)
            self.assertEqual(sorted(x['recomputed_batch_statistics']['kl']['mean'] for x in report['records']), [2.0, 4.0])
            self.assertTrue(all(not x['paper_comparison']['kl_mean']['matches_printed_mean'] for x in report['records']))
            slot = next(iter(report['cohorts'].values()))['rows']['SO3/RNOT']
            self.assertEqual(slot['status'], 'duplicate_or_configuration_variant')
            self.assertFalse(slot['one_complete_native_record'])
            self.assertEqual(len(next(iter(report['configuration_groups'].values()))), 2)

    def test_source_helper_wrapper_host_driver_and_configuration_stay_distinct(self):
        original = fixture(); variants = [original]
        for field in ('runner_sha256', 'helper_sha256'):
            r = deepcopy(original); r[field] = 'e' * 64; variants.append(r)
        r = deepcopy(original); r['source']['sha256'] = 'f' * 64; variants.append(r)
        r = deepcopy(original); r['environment']['hostname'] = 'nvidia6'; variants.append(r)
        r = deepcopy(original); r['environment']['nvidia_smi'] = r['environment']['nvidia_smi'].replace('570.195.03', '550.90.07'); variants.append(r)
        r = deepcopy(original); r['configuration']['model']['n_landmarks'] = 128; variants.append(r)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = m.compare([write(root, str(i), r) for i, r in enumerate(variants)], TARGETS, root / 'out')
            self.assertEqual(len(report['cohorts']), 6)
            self.assertEqual(len(report['configuration_groups']), 7)
            self.assertTrue(all(c['status'] == 'partial_or_failed' for c in report['cohorts'].values()))
            self.assertEqual(sum(len(c['duplicate_or_variant_rows']) for c in report['cohorts'].values()), 1)

    def test_execution_failures_do_not_become_numerical_matches(self):
        failure = fixture('SE3', 'rcpm', .1)
        failure.update(status='failed', failed_during='building', evaluation_batches=[],
                       error={'type': 'ImportError', 'message': 'NaN mentioned in an import error is not observed numeric output'})
        failure.pop('summary')
        result = m.inspect_record(failure)
        self.assertEqual(result['outcome'], 'execution_failure')
        self.assertFalse(m.paper_comparison(result, TARGETS['SE3/RCPM/0.1'])['observed_nonfinite_instability'])
        failure['parameter_nonfinite_count'] = 12; failure['failed_during'] = 'trained'
        observed = m.inspect_record(failure)
        self.assertEqual(observed['outcome'], 'observed_numerical_nonfinite')
        self.assertFalse(observed['complete_native_record'])
        finite = m.inspect_record(fixture('SE3', 'rcpm', .1))
        self.assertEqual(m.paper_comparison(finite, TARGETS['SE3/RCPM/0.1'])['assessment'], 'finite_output_does_not_establish_instability')

    def test_partial_batch_records_and_mismatched_summary_are_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = []
            for i in range(5):
                r = fixture(); r['started_at'] = str(i); r['evaluation_batches'] = r['evaluation_batches'][:1]
                summarize(r); paths.append(write(root, str(i), r))
            report = m.compare(paths, TARGETS, root / 'out')
            self.assertTrue(all(not x['complete_native_record'] for x in report['records']))
            self.assertTrue(all(x['outcome'] == 'invalid_or_other_protocol' for x in report['records']))
        r = fixture(); r['summary']['kl']['mean'] = 3.0
        inspection = m.inspect_record(r)
        self.assertEqual(inspection['outcome'], 'invalid_or_other_protocol')
        self.assertTrue(any('mean disagrees' in error for error in inspection['errors']))
        r = nonfinite(fixture()); r['summary']['kl']['mean'] = 0.0
        inspection = m.inspect_record(r)
        self.assertFalse(inspection['complete_native_record'])
        self.assertTrue(any('hides nonfinite' in error for error in inspection['errors']))

    def test_cli_relative_links_survive_relocation_without_changing_raw_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root / 'initial bundle'; source = bundle / 'raw evidence'
            record = fixture(); path = write(source, 'one run', record)
            original_bytes = path.read_bytes(); targets = bundle / 'paper_targets.json'; shutil.copy2(TARGET_PATH, targets)
            output = bundle / 'report'
            subprocess.run([sys.executable, '-B', str(Path(m.__file__)), '--results-root', str(source),
                            '--paper-targets', str(targets), '--output-dir', str(output)], check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), original_bytes)
            moved = root / 'relocated bundle'; shutil.move(bundle, moved)
            report = json.loads((moved / 'report/comparison.json').read_text())
            result = report['records'][0]
            self.assertEqual(result['raw_record'], record)
            self.assertFalse(Path(result['evidence_paths'][0]).is_absolute())
            self.assertEqual((moved / 'report' / result['evidence_paths'][0]).read_bytes(), original_bytes)
            for target in re.findall(r'\]\(<([^>]+)>\)', (moved / 'report/comparison.md').read_text()):
                self.assertTrue((moved / 'report' / target).exists())


if __name__ == '__main__':
    unittest.main()
