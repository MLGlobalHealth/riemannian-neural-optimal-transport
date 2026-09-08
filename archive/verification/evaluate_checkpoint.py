#!/usr/bin/env python3
"""Replay five native table RNOT evaluation batches from a saved ZIP checkpoint.

Evaluation only: restores saved parameters/landmarks and uses recorded uint32
subkeys. It never trains, runs diagnostics, or creates an independent run.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
EXPECTED_PACKAGES = {'jax': '0.4.35', 'jaxlib': '0.4.34', 'flax': '0.8.4', 'optax': '0.2.3'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_record(saved):
    selection = saved['selection']
    if saved['status'] != 'complete' or saved.get('smoke') or selection['suite'] != 'table' or selection['method'] != 'ours':
        raise ValueError('Requires a completed, non-smoke native table RNOT checkpoint.')
    if saved['label'] != 'zip_historical_settings_cuda_jax435':
        raise ValueError('This evaluator supports the restored ZIP JAX 0.4.35 cohort.')
    if saved['environment']['jax_enable_x64'] is not True:
        raise ValueError('The restored cohort must have x64 enabled.')
    recipe = saved['evaluation_config']
    batches = saved['evaluation_batches']
    if recipe['n_batches'] != 5 or recipe['batch_size'] != 1024 or len(batches) != 5:
        raise ValueError('Requires the original five evaluation batches of 1024 points.')
    if recipe['seed'] != selection['seed'] + 1000:
        raise ValueError('Unexpected original evaluation seed schedule.')
    for index, batch in enumerate(batches):
        key = batch.get('prng_key')
        if batch['index'] != index or not isinstance(key, list) or len(key) != 2 or any(
                not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 2**32 for x in key):
            raise ValueError('Each original batch must contain its exact uint32[2] evaluation subkey.')
    for name, version in EXPECTED_PACKAGES.items():
        if saved['environment']['packages'].get(name) != version:
            raise ValueError(f'Input record is outside the supported {name}=={version} cohort.')


def array_fingerprint(value, np):
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'size': int(array.size), 'nonfinite_count': int((~np.isfinite(array)).sum())}


def parameter_fingerprints(tree, np, prefix='params'):
    if isinstance(tree, dict):
        return {path: value for key in sorted(tree) for path, value in parameter_fingerprints(tree[key], np, prefix + '/' + str(key)).items()}
    return {prefix: array_fingerprint(tree, np)}


def restore_experiment(m, runner, saved, payload):
    cfg = saved['configuration']
    selection = SimpleNamespace(**saved['selection'])
    m.LANDMARK_METHOD = selection.landmark_method
    overrides = {section: cfg[section] for section in ('model', 'solver', 'training')}
    exp = runner.construct_overridden_ours(m, selection, overrides, cfg['manifold_name'], cfg['base_density'], cfg['target_density'])
    if runner.json_safe(exp['cfg']) != cfg:
        raise ValueError('Reconstructed native configuration differs from the recorded configuration.')
    params = m.jax.tree_util.tree_map(m.jnp.asarray, payload['params'])
    landmarks = m.jnp.asarray(payload['landmarks'])
    embedding = m.GromovDistanceEmbedding(manifold=exp['manifold'], landmarks=landmarks)
    psi = exp['psi'].clone(phi=embedding)
    exp['psi'] = psi
    exp['state'] = exp['state'].replace(params=params)
    # Solver closures capture psi; build a fresh solver after restoring landmarks.
    exp['solver'] = m.ArgminSolver(manifold=exp['manifold'], psi_module=psi,
                                  **dataclasses.asdict(exp['cfg'].solver))
    density = {'base_class': type(exp['base']).__name__, 'target_class': type(exp['target']).__name__,
               'target_loc': exp['target'].loc, 'target_scale': exp['target'].scale}
    geometry = {'class': type(exp['manifold']).__name__, 'ambient_dimension': exp['manifold'].D,
        'jitter': getattr(exp['manifold'], 'jitter', None),
        'components': [{'class': type(manifold).__name__, 'ambient_dimension': manifold.D,
                        'jitter': getattr(manifold, 'jitter', None)} for manifold in getattr(exp['manifold'], 'manifolds', [])]}
    if runner.json_safe(density) != saved['density_configuration'] or runner.json_safe(geometry) != saved['geometry_configuration']:
        raise ValueError('Reconstructed data distribution or geometry differs from the original record.')
    before = {'parameters': parameter_fingerprints(payload['params'], m.np),
              'landmarks': array_fingerprint(payload['landmarks'], m.np)}
    restored = {'parameters': parameter_fingerprints(params, m.np),
                'landmarks': array_fingerprint(exp['psi'].phi.landmarks, m.np)}
    if before != restored:
        raise ValueError('Checkpoint arrays changed dtype, shape or bytes during restoration.')
    if parameter_fingerprints(payload['state']['params'], m.np) != before['parameters']:
        raise ValueError('Checkpoint top-level params disagree with saved state.params.')
    if sum(array['size'] for array in restored['parameters'].values()) != saved['parameter_count']:
        raise ValueError('Restored parameter count disagrees with the original record.')
    return exp, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--input-result', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Defaults to the checkpoint next to input-result.')
    parser.add_argument('--expected-checkpoint-sha256', help='Optional additional assertion for a common checkpoint across hosts.')
    parser.add_argument('--output', type=Path, required=True, help='A fresh evaluation-only JSON file.')
    parser.add_argument('--platform', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--gpu', help='Explicit GPU ID for CUDA evaluation.')
    args = parser.parse_args()
    if args.platform == 'cuda' and args.gpu is None:
        parser.error('--gpu is required for CUDA evaluation.')
    if args.output.exists():
        parser.error('Evaluation output must be a new file.')
    saved = json.loads(args.input_result.read_text()); validate_record(saved)
    checkpoint = args.checkpoint or args.input_result.with_name(saved['checkpoint']['file'])
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != saved['checkpoint']['sha256'] or (args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256):
        raise ValueError('Checkpoint checksum does not match the input record or requested common checkpoint.')
    runner_path = Path(__file__).with_name('verify_run.py')
    if sha256(runner_path) != saved['runner_sha256']:
        raise ValueError('The unchanged verifier must match the original runner SHA256.')
    spec = importlib.util.spec_from_file_location('checkpoint_verification_runner', runner_path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    if runner.source_manifest(args.source_root) != saved['source']:
        raise ValueError('Source tree checksum differs from the original experiment.')
    packages = runner.package_versions()
    for name, version in EXPECTED_PACKAGES.items():
        if packages.get(name) != version:
            raise ValueError(f'Evaluation requires {name}=={version}; found {packages.get(name)}.')
    os.environ.update(JAX_PLATFORMS=args.platform, CUDA_VISIBLE_DEVICES=args.gpu if args.platform == 'cuda' else '',
        JAX_ENABLE_X64='True', JAX_THREEFRY_PARTITIONABLE='False', JAX_DEFAULT_PRNG_IMPL='threefry2x32',
        JAX_RANDOM_SEED_OFFSET='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLBACKEND='Agg')
    record = {'schema_version': 1, 'record_type': 'checkpoint_evaluation', 'evaluation_only': True,
        'independent_training_run': False, 'status': 'evaluation_only_building', 'started_at': runner.utc_now(),
        'input_result': str(args.input_result.resolve()), 'input_result_sha256': sha256(args.input_result),
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'source': saved['source'], 'source_root': str(args.source_root.resolve()),
        'runner_sha256': saved['runner_sha256'], 'evaluator_sha256': sha256(Path(__file__)),
        'original_task_id': saved['task_id'], 'original_environment': saved['environment'],
        'configuration': saved['configuration'], 'evaluation_config': saved['evaluation_config'],
        'evaluation_batches': [],
        'interpretation': 'Fixed saved state and recorded evaluation subkeys; identical sampled arrays across hosts are not assumed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    try:
        m = runner.load_experiment_module(args.source_root, 'table')
        from flax import serialization
        payload = serialization.msgpack_restore(checkpoint.read_bytes())
        m.jax.config.update('jax_default_matmul_precision', None)
        record['environment'] = runner.environment()
        record['environment'].update(jax_default_backend=m.jax.default_backend(),
            jax_devices=[str(device) for device in m.jax.devices()],
            jax_config={name: getattr(m.jax.config, name) for name in ('jax_enable_x64', 'jax_threefry_partitionable',
                'jax_default_prng_impl', 'jax_random_seed_offset', 'jax_default_matmul_precision')})
        record['original_host_replay'] = record['environment']['hostname'] == saved['environment']['hostname']
        exp, restored = restore_experiment(m, runner, saved, payload)
        record['restored_arrays'] = restored
        record['status'] = 'evaluation_only_evaluating'; runner.save_json(args.output, record)
        for original in saved['evaluation_batches']:
            key = m.jnp.asarray(original['prng_key'], dtype=m.jnp.uint32)
            start = time.perf_counter()
            metrics = m.compute_kl_ours(exp, key, batch_size=saved['evaluation_config']['batch_size'])
            batch = {'index': original['index'], 'prng_key': original['prng_key'], 'metrics': metrics,
                'evaluation_seconds': time.perf_counter() - start, 'original_metrics': original['metrics'],
                'metric_nonfinite': {name: not math.isfinite(float(value)) for name, value in metrics.items()},
                'comparison_to_original': {name: {'exact_equal': float(value) == float(original['metrics'][name]),
                    'difference': float(value) - float(original['metrics'][name])} for name, value in metrics.items()}}
            record['evaluation_batches'].append(batch); runner.save_json(args.output, record)
            print('EVALUATION_ONLY_BATCH', json.dumps(runner.json_safe(batch)), flush=True)
        record['all_five_batches_exactly_match_original'] = all(check['exact_equal'] for batch in record['evaluation_batches']
            for check in batch['comparison_to_original'].values())
        record['status'] = 'evaluation_only_nonfinite' if any(any(batch['metric_nonfinite'].values())
            for batch in record['evaluation_batches']) else 'evaluation_only_complete'
        record['finished_at'] = runner.utc_now(); runner.save_json(args.output, record)
        return 0 if record['status'] == 'evaluation_only_complete' else 1
    except Exception as error:
        record.update(status='evaluation_only_failed', finished_at=runner.utc_now(),
                      error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        runner.save_json(args.output, record)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
