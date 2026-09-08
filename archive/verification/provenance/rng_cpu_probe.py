"""Small CPU-only fingerprints of RNG streams, independent of model code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--partitionable', choices=('native', 'false'), default='native')
args = parser.parse_args()
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

flag_names = ('jax_threefry_partitionable', 'jax_default_prng_impl', 'jax_enable_x64',
              'jax_random_seed_offset', 'jax_legacy_prng_key', 'jax_default_matmul_precision',
              'jax_threefry_gpu_kernel_lowering')
native = {name: getattr(jax.config, name, 'unavailable') for name in flag_names}
if args.partitionable == 'false':
    jax.config.update('jax_threefry_partitionable', False)
assert all(device.platform == 'cpu' for device in jax.devices())

def fingerprint(array, expression):
    array = np.asarray(array)
    return {'expression': expression, 'dtype': str(array.dtype), 'shape': list(array.shape),
            'sha256_bytes': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'values': array.tolist()}

key = jax.random.PRNGKey(12345)
outputs = {
    'key': fingerprint(key, 'jax.random.PRNGKey(12345)'),
    'split_4': fingerprint(jax.random.split(key, 4), 'jax.random.split(key, 4)'),
    'fold_in_7': fingerprint(jax.random.fold_in(key, 7), 'jax.random.fold_in(key, 7)'),
    'bits_uint32': fingerprint(jax.random.bits(key, (16, 3), dtype=jnp.uint32),
                              'jax.random.bits(key, (16, 3), dtype=jnp.uint32)'),
    'permutation_int32': fingerprint(jax.random.permutation(key, jnp.arange(128, dtype=jnp.int32)),
                                    'jax.random.permutation(key, jnp.arange(128, dtype=jnp.int32))')}
for dtype in (jnp.float32, jnp.float64):
    for name in ('uniform', 'normal'):
        label = name + '_' + np.dtype(dtype).name
        outputs[label] = fingerprint(getattr(jax.random, name)(key, (16, 3), dtype=dtype),
                                    f'jax.random.{name}(key, (16, 3), dtype=jnp.{np.dtype(dtype).name})')
config_path = Path(jax.__file__).parent / '_src' / 'config.py'
source_lines = config_path.read_text().splitlines()
snippets = {}
for label in ('default_prng_impl =', 'threefry_partitionable ='):
    for index, line in enumerate(source_lines):
        if line.startswith(label):
            snippets[label] = {'line_1_based': index + 1, 'text': '\n'.join(source_lines[index:index+5])}
            break
cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), None)
print(json.dumps({'generated_at': datetime.now(timezone.utc).isoformat(), 'mode': args.partitionable,
    'hostname': socket.gethostname(), 'python': sys.version, 'executable': sys.executable,
    'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__,
    'cpu_model': cpu_model, 'platform': platform.platform(), 'byteorder': sys.byteorder,
    'affinity': sorted(os.sched_getaffinity(0)), 'devices': [str(device) for device in jax.devices()],
    'native_config_before_override': native,
    'effective_config': {name: getattr(jax.config, name, 'unavailable') for name in flag_names},
    'environment': {name: os.environ.get(name) for name in ('JAX_PLATFORMS', 'JAX_ENABLE_X64',
        'JAX_THREEFRY_PARTITIONABLE', 'JAX_DEFAULT_PRNG_IMPL', 'JAX_RANDOM_SEED_OFFSET',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')},
    'installed_config_source': {'path': str(config_path), 'sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                'snippets': snippets},
    'seed': 12345, 'outputs': outputs}, sort_keys=True, indent=2))
