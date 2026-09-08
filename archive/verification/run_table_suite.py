#!/usr/bin/env python3
"""Run the declared 30 restored ZIP table jobs through the unchanged verifier.

Use the supported experiment Python environment. Each selected GPU runs at most
one child process. A failure stops new jobs while active children finish.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SEEDS = (12345, 23456, 34567, 45678, 56789)
PACKAGES = {'jax': '0.4.35', 'jaxlib': '0.4.34', 'flax': '0.8.4', 'optax': '0.2.3'}
ENVIRONMENT = {
    'JAX_PLATFORMS': 'cuda', 'JAX_ENABLE_X64': 'True',
    'JAX_THREEFRY_PARTITIONABLE': 'False', 'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'MPLBACKEND': 'Agg',
    'PYTHONDONTWRITEBYTECODE': '1',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(source_root, output_dir, gpus, python=sys.executable, runner=None, overrides=None):
    here = Path(__file__).resolve().parent
    runner = Path(runner or here / 'verify_run.py').resolve()
    overrides = Path(overrides or here / 'historical_settings.json').resolve()
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Choose at least one GPU, with no duplicate IDs.')
    jobs = []
    for manifold in ('sphere', 'torus'):
        for method, landmarks in (('ours', 'fps'), ('ours', 'random'), ('rcpm', None)):
            label = 'zip_historical_settings_cuda_jax435' if method == 'ours' else 'zip_x64_cuda_jax435'
            for seed in SEEDS:
                job_id = f'{manifold}_{method}_{landmarks or "gamma1"}_seed{seed}'
                command = [str(python), '-u', str(runner), '--source-root', str(source_root),
                    '--output-dir', str(output_dir / 'results'), '--suite', 'table',
                    '--manifold', manifold, '--dimension', '2', '--method', method,
                    '--seed', str(seed), '--label', label]
                command += ['--landmark-method', landmarks, '--overrides', str(overrides)] if method == 'ours' else ['--gamma', '1.0']
                jobs.append({'job_id': job_id, 'manifold': manifold, 'method': method,
                    'landmark_method': landmarks, 'seed': seed, 'label': label,
                    'command_without_gpu': command, 'log': str(output_dir / 'logs' / (job_id + '.log'))})
    return {'schema_version': 1, 'source_root': str(source_root), 'output_dir': str(output_dir),
        'python': str(python), 'runner': str(runner), 'overrides': str(overrides), 'gpus': list(gpus),
        'environment_overrides': dict(ENVIRONMENT), 'seeds': list(SEEDS), 'jobs': jobs,
        'environment_note': 'Other shell settings, including CUDA_ROOT and XLA_FLAGS, are inherited and recorded when running.'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def inspect_interpreter(python):
    """Read distribution metadata in the selected interpreter, without JAX imports."""
    code = """import importlib.metadata as m,json,sys
versions={}
for name in ('jax','jaxlib','flax','optax'):
 try: versions[name]=m.version(name)
 except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':sys.version,'python_version':list(sys.version_info[:3]),
                  'executable':sys.executable,'packages':versions}))
"""
    try:
        completed = subprocess.run([str(python), '-I', '-c', code], capture_output=True, text=True,
                                   check=True, timeout=30)
        return json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ValueError(f'Cannot inspect experiment interpreter {python}: {error}') from error


def validate_interpreter(record):
    mismatches = []
    if record.get('python_version', [])[:2] != [3, 11]:
        mismatches.append(f"Python 3.11 required; found {record.get('python_version')}")
    for name, expected in PACKAGES.items():
        actual = record.get('packages', {}).get(name)
        if actual != expected:
            mismatches.append(f'{name}=={expected} required; found {actual}')
    if mismatches:
        raise ValueError('Unsupported experiment environment: ' + '; '.join(mismatches))


def run_plan(plan, poll_seconds=0.2):
    output = Path(plan['output_dir'])
    if output.exists():
        raise FileExistsError(f'Output directory must be new: {output}')
    if not Path(plan['source_root']).is_dir():
        raise ValueError('source-root must be the extracted ZIP source directory.')
    for key in ('runner', 'overrides'):
        if not Path(plan[key]).is_file():
            raise ValueError(f'Missing {key}: {plan[key]}')
    interpreter = inspect_interpreter(plan['python'])
    validate_interpreter(interpreter)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'results').mkdir()
    manifest = dict(plan, created_at=utc_now(), interpreter=interpreter,
        runner_sha256=hashlib.sha256(Path(plan['runner']).read_bytes()).hexdigest(),
        overrides_sha256=hashlib.sha256(Path(plan['overrides']).read_bytes()).hexdigest(),
        inherited_environment={key: os.environ.get(key) for key in ('CUDA_ROOT', 'XLA_FLAGS', 'JAX_DEFAULT_PRNG_IMPL',
            'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_RANDOM_SEED_OFFSET', 'LD_LIBRARY_PATH')})
    write_json(output / 'manifest.json', manifest)
    states = [dict(job, status='pending', gpu=None, pid=None, exit_code=None) for job in plan['jobs']]
    status = {'status': 'running', 'started_at': utc_now(), 'stop_reason': None, 'jobs': states}
    write_json(output / 'status.json', status)
    pending, active = deque(range(len(states))), {}

    def stop(signum, frame):
        status['stop_reason'] = status['stop_reason'] or f'signal_{signum}'
        status['status'] = 'waiting_for_active_jobs'

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            # Observe every completed child before deciding whether to start more.
            for gpu, (index, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                states[index].update(status='complete' if code == 0 else 'failed', exit_code=code, finished_at=utc_now())
                del active[gpu]
                if code != 0:
                    status['stop_reason'] = status['stop_reason'] or f'job_failed:{states[index]["job_id"]}'
            if not status['stop_reason']:
                for gpu in plan['gpus']:
                    if status['stop_reason']:
                        break
                    if gpu in active or not pending:
                        continue
                    index = pending.popleft(); entry = states[index]
                    command = entry['command_without_gpu'] + ['--gpu', gpu]
                    env = dict(os.environ, **plan['environment_overrides'], CUDA_VISIBLE_DEVICES=gpu)
                    log = open(entry['log'], 'wb')
                    entry.update(gpu=gpu, command=command, started_at=utc_now())
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   start_new_session=True)
                    except OSError as error:
                        log.write((f'{type(error).__name__}: {error}\n').encode()); log.close()
                        entry.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=utc_now())
                        status['stop_reason'] = f'job_start_failed:{entry["job_id"]}'
                        break
                    entry.update(status='running', pid=process.pid)
                    active[gpu] = (index, process, log)
            if status['stop_reason']:
                status['status'] = 'waiting_for_active_jobs' if active else 'stopped'
                while pending:
                    states[pending.popleft()]['status'] = 'not_started'
            write_json(output / 'status.json', status)
            if active:
                time.sleep(poll_seconds)
        failed = any(entry['status'] == 'failed' for entry in states)
        status.update(status='failed' if failed else 'interrupted' if status['stop_reason'] else 'complete', finished_at=utc_now())
        write_json(output / 'status.json', status)
        return 1 if status['status'] != 'complete' else 0
    finally:
        # Children are never terminated, including after user interruption.
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory that does not already exist.')
    parser.add_argument('--gpus', nargs='+', required=True, help='Explicit GPU IDs or UUIDs, e.g. --gpus 2 3 4.')
    parser.add_argument('--python', default=sys.executable, help='Supported experiment environment Python; defaults to this interpreter.')
    parser.add_argument('--dry-run', action='store_true', help='Print the 30-job plan without importing experiment code, inspecting GPUs, or creating files.')
    args = parser.parse_args()
    try:
        plan = make_plan(args.source_root, args.output_dir, args.gpus, args.python)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return run_plan(plan)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
