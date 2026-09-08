"""Record untouched historical S2 random initialization on CPU, without training."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--runner', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
import jax
import numpy as np

start = time.perf_counter()
spec = importlib.util.spec_from_file_location('verification_runner', args.runner)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
module = runner.load_experiment_module(args.source_root, 'table')
assert jax.default_backend() == 'cpu'
assert module.LANDMARK_METHOD == 'random'
selection = SimpleNamespace(method='ours', suite='table', manifold='sphere', dimension=2,
                            gamma=module.RCPM_GAMMA, seed=12345)
exp, configuration, builder = runner.build_experiment(module, selection, {})
params = exp['state'].params
leaves, tree = jax.tree_util.tree_flatten(params)
arrays = {f'parameter_{i}': np.asarray(value) for i, value in enumerate(leaves)}
arrays['remaining_key'] = np.asarray(exp['key'])
arrays['landmarks'] = np.asarray(exp['psi'].phi.landmarks)
kx, ky = jax.random.split(jax.random.PRNGKey(12345))
arrays['base_sample4'] = np.asarray(exp['base'].sample(kx, 4))
arrays['target_sample4'] = np.asarray(exp['target'].sample(ky, 4))

def fingerprint(value):
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'first_values': array.reshape(-1)[:8].tolist(),
            'nonfinite_count': int((~np.isfinite(array)).sum())}

fingerprints = [fingerprint(leaf) for leaf in leaves]
entry = {'status': 'complete', 'builder': builder, 'configuration': runner.json_safe(configuration),
         'parameter_count': sum(int(x.size) for x in leaves), 'parameter_tree': str(tree),
         'parameter_leaves': fingerprints,
         'parameter_fingerprint_sha256': hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()}
entry.update({name: fingerprint(arrays[name]) for name in ('remaining_key', 'landmarks', 'base_sample4', 'target_sample4')})
key = jax.random.PRNGKey(12345)
record = {'label': 'cpu_initialization_provenance', 'status': 'complete',
    'purpose': 'Native S2 random build initialization comparison, seed 12345; no training',
    'source_root': str(args.source_root), 'source_manifest': runner.source_manifest(args.source_root),
    'hostname': socket.gethostname(), 'python': platform.python_version(),
    'executable': sys.executable, 'affinity': sorted(os.sched_getaffinity(0)),
    'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')), None),
    'packages': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'scipy')},
    'environment': {key: os.environ.get(key) for key in ('JAX_PLATFORMS', 'JAX_ENABLE_X64',
        'JAX_THREEFRY_PARTITIONABLE', 'JAX_DEFAULT_PRNG_IMPL', 'JAX_DEFAULT_MATMUL_PRECISION',
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')},
    'jax_config': {key: getattr(jax.config, key) for key in ('jax_enable_x64', 'jax_threefry_partitionable',
        'jax_default_prng_impl', 'jax_default_matmul_precision')},
    'jax_backend': jax.default_backend(), 'jax_devices': [str(x) for x in jax.devices()],
    'prng_fingerprints': {'key': fingerprint(key), 'split3': fingerprint(jax.random.split(key, 3)),
        'normal_float64_4x3': fingerprint(jax.random.normal(key, (4, 3), dtype=jax.numpy.float64)),
        'normal_float32_4x3': fingerprint(jax.random.normal(key, (4, 3), dtype=jax.numpy.float32))},
    'builds': {'ours': entry}, 'seconds': time.perf_counter() - start,
    'runner_sha256': hashlib.sha256(args.runner.read_bytes()).hexdigest(),
    'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
args.output.parent.mkdir(parents=True, exist_ok=True)
array_path = args.output.with_suffix('.npz')
np.savez_compressed(array_path, **arrays)
record['full_arrays'] = {'path': array_path.name, 'sha256': hashlib.sha256(array_path.read_bytes()).hexdigest()}
args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': record['status'], 'output': str(args.output),
                  'parameter_fingerprint_sha256': entry['parameter_fingerprint_sha256'], 'seconds': record['seconds']}))
