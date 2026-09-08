#!/usr/bin/env python3
"""Own only explicitly selected native jobs; preserve failures and all results."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f'Read-only check failed: {args}: {result.stderr}')
    return result.stdout.strip()


def capacity():
    raw = run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
    rows = {}
    for line in raw.splitlines():
        x = [piece.strip() for piece in line.split(',')]
        rows[int(x[0])] = {'uuid': x[1], 'model': x[2], 'driver': x[3], 'used_mib': int(x[4]), 'free_mib': int(x[5]), 'utilization': int(x[6])}
    return {'at': utc(), 'monotonic': time.monotonic(), 'gpus': rows}


def verify_inputs(root, manifest):
    source = root / 'source'
    frozen = json.loads((root / 'provenance/recovered_liegroup_source_manifest.json').read_text())
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    assert actual == frozen['files'] and len(actual) == 24
    assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == manifest['source_sha256']
    for name, digest in manifest['tool_hashes'].items():
        assert hashlib.sha256((root / 'tooling' / name).read_bytes()).hexdigest() == digest, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--phase', choices=['initial', 'queued'], required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / 'provenance/native_cohort_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    lock_file = (root / f'provenance/{args.phase}_supervisor.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / f'provenance/{args.phase}_supervisor_state.json'
    assert not state_path.exists(), 'This phase already has a durable owner record'
    verify_inputs(root, manifest)
    state = {'started_at': utc(), 'pid': os.getpid(), 'phase': args.phase, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'status': 'capacity_checks', 'workers': {}, 'pending': [j['job_id'] for j in manifest['jobs'] if j['phase'] == args.phase], 'events': []}
    save(state_path, state)
    expected_freeze = (root / 'provenance/requirements_frozen.txt').read_text()
    actual_freeze = run([manifest['python'], '-m', 'pip', 'freeze']) + '\n'
    assert actual_freeze == expected_freeze, 'Frozen environment changed'
    save(root / f'provenance/{args.phase}_freeze_verification.json', {'at': utc(), 'freeze_sha256': hashlib.sha256(actual_freeze.encode()).hexdigest(), 'unchanged': True, 'pip_check': run([manifest['python'], '-m', 'pip', 'check'])})
    jobs = {j['job_id']: j for j in manifest['jobs']}
    active = {}
    capacity_history = []
    failed = False
    while state['pending'] or active:
        sample = capacity()
        capacity_history.append(sample)
        # A small, durable monitoring log records occupancy while workers run.
        with (root / f'logs/{args.phase}_capacity.jsonl').open('a') as handle:
            handle.write(json.dumps(sample) + '\n')
        for job_id, owned in list(active.items()):
            result_path = root / jobs[job_id]['output_relative'] / 'result.json'
            native = json.loads(result_path.read_text()) if result_path.exists() else {}
            state['workers'][job_id]['native_status'] = native.get('status')
            state['workers'][job_id]['evaluation_batch_count'] = len(native.get('evaluation_batches', []))
            rc = owned['process'].poll()
            if rc is None:
                continue
            owned['log'].close()
            state['workers'][job_id].update({'exit_code': rc, 'finished_at': utc(), 'outcome': native.get('outcome')})
            active.pop(job_id)
            if rc != 0 or native.get('status') != 'complete' or len(native.get('evaluation_batches', [])) != 5:
                failed = True
                state['events'].append({'at': utc(), 'event': 'failure_queue_paused', 'job_id': job_id, 'exit_code': rc, 'native_error': native.get('error')})
        assigned = {details['gpu'] for details in state['workers'].values() if details.get('exit_code') is None}
        if args.phase == 'queued':
            initial_state = root / 'provenance/initial_supervisor_state.json'
            if initial_state.exists():
                other = json.loads(initial_state.read_text())
                if other.get('status') in ('failed', 'queue_paused_after_failure'):
                    failed = True
                    if not any(e.get('event') == 'initial_phase_failure_queue_paused' for e in state['events']):
                        state['events'].append({'at': utc(), 'event': 'initial_phase_failure_queue_paused'})
        if not failed:
            for job_id in list(state['pending']):
                job = jobs[job_id]
                slots = [job['initial_gpu']] if args.phase == 'initial' else [4, 5, 0, 1, 2, 3]
                chosen = None
                checked_samples = None
                for gpu in slots:
                    if gpu in assigned:
                        continue
                    window = [s for s in capacity_history if sample['monotonic'] - s['monotonic'] <= 45]
                    if not window or sample['monotonic'] - window[0]['monotonic'] < 30:
                        continue
                    if all(s['gpus'][gpu]['used_mib'] <= 2 and s['gpus'][gpu]['free_mib'] >= 16000 and s['gpus'][gpu]['utilization'] <= 5 for s in window):
                        chosen, checked_samples = gpu, window
                        break
                if chosen is None:
                    continue
                verify_inputs(root, manifest)
                output = root / job['output_relative']
                assert not output.exists(), f'Duplicate result directory refused: {output}'
                owner = root / f'provenance/{job_id}_owner.json'
                with owner.open('x') as handle:
                    handle.write(json.dumps({'created_at': utc(), 'supervisor_pid': os.getpid(), 'job_id': job_id}) + '\n')
                env = os.environ.copy()
                for key in list(env):
                    if key.startswith(('JAX_', 'XLA_')) or key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
                        env.pop(key)
                env.update(manifest['environment'])
                env['CUDA_VISIBLE_DEVICES'] = str(chosen)
                env['HIP_VISIBLE_DEVICES'] = str(chosen)
                cpus = manifest['cpu_affinities'][str(chosen)]
                command = ['taskset', '-c', ','.join(map(str, cpus)), manifest['python'], '-u', str(root / 'tooling/verify_liegroup_run.py'), '--source-root', str(root / 'source'), '--source-manifest', str(root / 'provenance/recovered_liegroup_source_manifest.json'), '--output-dir', str(output), '--manifold', job['manifold'], '--method', job['method'], '--gpu', str(chosen)]
                if job['gamma'] is not None:
                    command += ['--gamma', str(job['gamma'])]
                launch = {'declared_at': utc(), 'job_id': job_id, 'gpu': chosen, 'cpu_affinity': cpus, 'command': command, 'environment': {**manifest['environment'], 'CUDA_VISIBLE_DEVICES': str(chosen), 'HIP_VISIBLE_DEVICES': str(chosen)}, 'capacity_window': checked_samples, 'manifest_sha256': state['manifest_sha256'], 'source_sha256': manifest['source_sha256'], 'tool_hashes': manifest['tool_hashes'], 'output': str(output)}
                save(root / f'provenance/{job_id}_launch.json', launch)
                log = (root / f'logs/{job_id}.log').open('x')
                worker = subprocess.Popen(command, cwd=root / 'source', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                launch['worker_pid'] = worker.pid
                launch['started_at'] = utc()
                save(root / f'provenance/{job_id}_launch.json', launch)
                active[job_id] = {'process': worker, 'log': log}
                state['workers'][job_id] = {'pid': worker.pid, 'gpu': chosen, 'cpu_affinity': cpus, 'started_at': launch['started_at'], 'exit_code': None}
                state['pending'].remove(job_id)
                assigned.add(chosen)
                print(json.dumps({'event': 'launched', **state['workers'][job_id], 'job_id': job_id}), flush=True)
        state['status'] = 'queue_paused_after_failure' if failed else 'running' if active else 'capacity_checks'
        state['updated_at'] = utc()
        save(state_path, state)
        if failed and not active:
            break
        if state['pending'] or active:
            time.sleep(5)
    state['status'] = 'failed' if failed else 'complete'
    state['finished_at'] = utc()
    save(state_path, state)
    print(json.dumps({'event': 'supervisor_finished', 'status': state['status']}), flush=True)


if __name__ == '__main__':
    main()
