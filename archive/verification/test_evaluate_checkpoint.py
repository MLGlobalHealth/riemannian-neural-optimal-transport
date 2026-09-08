"""Metadata and file guards; these tests never import JAX or evaluate a model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('checkpoint_evaluator', Path(__file__).with_name('evaluate_checkpoint.py'))
evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)


def saved_record():
    return {'status': 'complete', 'smoke': False, 'label': 'zip_historical_settings_cuda_jax435',
            'selection': {'suite': 'table', 'method': 'ours', 'seed': 12345},
            'environment': {'jax_enable_x64': True, 'packages': dict(evaluator.EXPECTED_PACKAGES)},
            'evaluation_config': {'n_batches': 5, 'batch_size': 1024, 'seed': 13345},
            'evaluation_batches': [{'index': i, 'prng_key': [i, 2**32 - 1]} for i in range(5)],
            'checkpoint': {'file': 'checkpoint.msgpack', 'sha256': '0' * 64}}


class CheckpointEvaluatorTests(unittest.TestCase):
    def test_only_completed_native_five_batch_supported_recipe_is_accepted(self):
        valid = saved_record()
        evaluator.validate_record(valid)
        variants = []
        for key, value in [('status', 'failed'), ('smoke', True), ('label', 'original_zip')]:
            record = copy.deepcopy(valid); record[key] = value; variants.append(record)
        for key, value in [('suite', 'high_dim'), ('method', 'rcpm')]:
            record = copy.deepcopy(valid); record['selection'][key] = value; variants.append(record)
        for key, value in [('n_batches', 4), ('batch_size', 512), ('seed', 12345)]:
            record = copy.deepcopy(valid); record['evaluation_config'][key] = value; variants.append(record)
        record = copy.deepcopy(valid); record['evaluation_batches'].pop(); variants.append(record)
        record = copy.deepcopy(valid); record['environment']['jax_enable_x64'] = False; variants.append(record)
        record = copy.deepcopy(valid); record['environment']['packages']['jax'] = '0.6.2'; variants.append(record)
        for index, record in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                evaluator.validate_record(record)

    def test_recorded_keys_must_be_ordered_uint32_pairs(self):
        for key in (None, [], [1], [1, 2, 3], [-1, 0], [2**32, 0], [True, 0], [1.0, 0]):
            record = saved_record(); record['evaluation_batches'][0]['prng_key'] = key
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'uint32'):
                evaluator.validate_record(record)
        record = saved_record(); record['evaluation_batches'][1]['index'] = 0
        with self.assertRaisesRegex(ValueError, 'uint32'):
            evaluator.validate_record(record)

    def test_invalid_checkpoint_and_existing_output_fail_before_native_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = root / 'result.json'; checkpoint = root / 'checkpoint.msgpack'
            original.write_text(json.dumps(saved_record())); checkpoint.write_bytes(b'wrong-checkpoint')
            output = root / 'new' / 'evaluation.json'
            command = [sys.executable, '-B', str(Path(evaluator.__file__)), '--platform', 'cpu',
                       '--source-root', str(root / 'absent-source'), '--input-result', str(original),
                       '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Checkpoint checksum does not match', result.stderr)
            self.assertFalse(output.parent.exists())
            output.parent.mkdir(); output.write_text('preserve prior evaluation\n')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Evaluation output must be a new file', result.stderr)
            self.assertEqual(output.read_text(), 'preserve prior evaluation\n')


if __name__ == '__main__':
    unittest.main()
