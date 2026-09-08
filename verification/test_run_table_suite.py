"""Launcher tests use fake CPU subprocesses and never import JAX or use GPUs."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('suite_launcher', Path(__file__).with_name('run_table_suite.py'))
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)


class TableSuiteTests(unittest.TestCase):
    def test_environment_validation_uses_required_versions_without_pinning_patch(self):
        record = {'python_version': [3, 11, 15], 'packages': dict(launcher.PACKAGES)}
        launcher.validate_interpreter(record)
        record['python_version'][2] = 16
        launcher.validate_interpreter(record)
        record['packages']['jax'] = '0.6.2'
        with self.assertRaisesRegex(ValueError, 'jax==0.4.35 required'):
            launcher.validate_interpreter(record)

    def test_declared_thirty_jobs_and_protocol(self):
        plan = launcher.make_plan('/unused/source', '/unused/new-output', ['2', '4'])
        self.assertEqual(len(plan['jobs']), 30)
        self.assertEqual(len({job['job_id'] for job in plan['jobs']}), 30)
        rows = Counter((job['manifold'], job['method'], job['landmark_method']) for job in plan['jobs'])
        self.assertEqual(len(rows), 6)
        self.assertEqual(set(rows.values()), {5})
        for row in rows:
            self.assertEqual([job['seed'] for job in plan['jobs']
                              if (job['manifold'], job['method'], job['landmark_method']) == row], list(launcher.SEEDS))
        for job in plan['jobs']:
            command = job['command_without_gpu']
            self.assertEqual(command[command.index('--source-root') + 1], '/unused/source')
            self.assertNotIn('--smoke', command)
            if job['method'] == 'ours':
                self.assertEqual(job['label'], 'zip_historical_settings_cuda_jax435')
                self.assertEqual(Path(command[command.index('--overrides') + 1]).name, 'historical_settings.json')
                self.assertEqual(command[command.index('--landmark-method') + 1], job['landmark_method'])
            else:
                self.assertEqual(job['label'], 'zip_x64_cuda_jax435')
                self.assertNotIn('--overrides', command)
                self.assertEqual(command[command.index('--gamma') + 1], '1.0')
        for key, value in {'JAX_ENABLE_X64': 'True', 'JAX_THREEFRY_PARTITIONABLE': 'False',
                           'OMP_NUM_THREADS': '2', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}.items():
            self.assertEqual(plan['environment_overrides'][key], value)

    def test_dry_run_needs_no_source_environment_or_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'absent-source'; output = root / 'absent-output'
            completed = subprocess.run([sys.executable, '-B', str(Path(launcher.__file__)), '--dry-run',
                '--source-root', str(source), '--output-dir', str(output), '--gpus', '0', '1',
                '--python', '/absent/experiment/python'], check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(completed.stdout)['jobs']), 30)
            self.assertEqual(list(root.iterdir()), [])

    def test_failure_keeps_active_child_and_pending_jobs_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            overrides = root / 'overrides.json'; overrides.write_text('{}')
            runner = root / 'fake_verifier.py'
            runner.write_text("""import argparse,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--output-dir');p.add_argument('--gpu')
a,_=p.parse_known_args();root=Path(a.output_dir)
assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
assert os.environ['JAX_ENABLE_X64']=='True'
assert os.environ['JAX_THREEFRY_PARTITIONABLE']=='False'
(root/str(a.seed)).write_text('started')
print('preserved child log',a.seed,flush=True)
if a.seed==12345:
 time.sleep(.1)
 raise SystemExit(7)
time.sleep(.3)
(root/(str(a.seed)+'.completed')).write_text('finished normally')
""")
            output = root / 'fresh-output'
            plan = launcher.make_plan(source, output, ['0', '1'], runner=runner, overrides=overrides)
            plan['jobs'] = plan['jobs'][:3]
            unsupported = {'python_version': [3, 11, 15], 'packages': dict(launcher.PACKAGES, jax='0.6.2')}
            with patch.object(launcher, 'inspect_interpreter', return_value=unsupported):
                with self.assertRaisesRegex(ValueError, 'Unsupported experiment environment'):
                    launcher.run_plan(plan)
            self.assertFalse(output.exists())
            environment = {'python_version': [3, 11, 15], 'packages': dict(launcher.PACKAGES)}
            with patch.object(launcher, 'inspect_interpreter', return_value=environment):
                self.assertEqual(launcher.run_plan(plan, poll_seconds=.01), 1)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual([job['status'] for job in status['jobs']], ['failed', 'complete', 'not_started'])
            self.assertEqual([job['exit_code'] for job in status['jobs']], [7, 0, None])
            self.assertTrue((output / 'results' / '23456.completed').exists())
            self.assertFalse((output / 'results' / '34567').exists())
            self.assertIn('preserved child log 12345', Path(status['jobs'][0]['log']).read_text())
            self.assertTrue(all(job['pid'] for job in status['jobs'][:2]))
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['seeds'], list(launcher.SEEDS))
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['interpreter'], environment)
            prior = (output / 'status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                launcher.run_plan(plan)
            self.assertEqual((output / 'status.json').read_bytes(), prior)


if __name__ == '__main__':
    unittest.main()
