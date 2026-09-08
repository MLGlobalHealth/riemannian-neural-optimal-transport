#!/usr/bin/env python3
"""Schedule exactly four fixed-checkpoint evaluations on this one host."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f'Preflight command failed: {args}: {result.stderr}')
    return result.stdout.strip()


def snapshot():
    raw = read(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
    gpus = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(',')]
        gpus[int(parts[0])] = {'uuid': parts[1], 'model': parts[2], 'driver': parts[3], 'used_mib': int(parts[4]), 'free_mib': int(parts[5]), 'utilization': int(parts[6])}
    return {'at': utc(), 'monotonic': time.monotonic(), 'gpus': gpus}


def verify(manifest, host):
    root = Path(host['root'])
    for name, digest in manifest['tool_hashes'].items():
        assert sha(root / 'tooling' / name) == digest, name
    native_manifest = json.loads((root / 'provenance/recovered_liegroup_source_manifest.json').read_text())
    actual = {str(p.relative_to(root / 'source')): sha(p) for p in (root / 'source').rglob('*') if p.is_file()}
    assert actual == native_manifest['files'] and len(actual) == 24
    assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == manifest['source_sha256']
    for job in host['jobs']:
        assert sha(Path(job['input_result'])) == job['input_result_sha256']
        assert sha(Path(job['checkpoint'])) == job['checkpoint_sha256']
        assert set(job['cpu_affinity']) <= os.sched_getaffinity(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--host', choices=['nvidia6', 'nvidia7'], required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    host = manifest['hosts'][args.host]
    output_root = Path(host['root']) / 'replay'
    lock = (output_root / 'provenance/supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output_root / 'provenance/supervisor_state.json'
    assert not state_path.exists()
    state = {'host': args.host, 'pid': os.getpid(), 'started_at': utc(), 'status': 'preflight', 'pending': [job['job_id'] for job in host['jobs']], 'workers': {}, 'events': [], 'manifest_sha256': sha(args.manifest)}
    save(state_path, state)
    try:
        verify(manifest, host)
        freeze = read([host['python'], '-m', 'pip', 'freeze']) + '\n'
        assert hashlib.sha256(freeze.encode()).hexdigest() == host['requirements_sha256']
        assert sha(Path(host['requirements'])) == host['requirements_sha256']
        save(output_root / 'provenance/freeze_verification.json', {'at': utc(), 'freeze_sha256': host['requirements_sha256'], 'all80_unchanged': True, 'pip_check': read([host['python'], '-m', 'pip', 'check'])})
        compute = read(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'])
        pids = sorted({line.split(',')[0].strip() for line in compute.splitlines() if line.split(',')[0].strip().isdigit()})
        owners = read(['ps', '-o', 'pid=,ppid=,user=,comm=', '-p', ','.join(pids)]) if pids else ''
        save(output_root / 'provenance/prelaunch_process_owners.json', {'at': utc(), 'gpu_compute_processes': compute, 'owners': owners, 'foreign_processes_untouched': True})
    except BaseException as error:
        state.update(status='guard_failed', error={'type': type(error).__name__, 'message': str(error)}, finished_at=utc())
        save(state_path, state)
        raise
    jobs = {job['job_id']: job for job in host['jobs']}
    active = {}
    history = []
    failed = False
    while state['pending'] or active:
        current = snapshot()
        history.append(current)
        with (output_root / 'logs/capacity.jsonl').open('a') as stream:
            stream.write(json.dumps(current) + '\n')
        for job_id, owned in list(active.items()):
            job = jobs[job_id]
            output = Path(job['output'])
            record = json.loads(output.read_text()) if output.exists() else {}
            state['workers'][job_id]['status'] = record.get('status')
            state['workers'][job_id]['batch_count'] = len(record.get('evaluation_batches', []))
            code = owned['process'].poll()
            if code is None:
                continue
            owned['log'].close()
            active.pop(job_id)
            state['workers'][job_id].update(exit_code=code, finished_at=utc())
            if code != 0 or record.get('status') != 'evaluation_only_complete' or len(record.get('evaluation_batches', [])) != 5:
                failed = True
                state['events'].append({'at': utc(), 'event': 'failure_pauses_unstarted_jobs', 'job_id': job_id, 'exit_code': code, 'native_error': record.get('error')})
        if not failed:
            for job_id in list(state['pending']):
                job = jobs[job_id]
                gpu = job['gpu']
                window = [s for s in history if current['monotonic'] - s['monotonic'] <= 45]
                if not window or current['monotonic'] - window[0]['monotonic'] < 30:
                    continue
                if not all(s['gpus'][gpu]['used_mib'] <= host['capacity_max_used_mib'] and s['gpus'][gpu]['free_mib'] >= 16000 and s['gpus'][gpu]['utilization'] <= 5 for s in window):
                    continue
                verify(manifest, host)
                assert not Path(job['output']).exists()
                environment = os.environ.copy()
                for key in list(environment):
                    if key.startswith(('JAX_', 'XLA_')) or key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
                        environment.pop(key)
                environment.update(host['environment'])
                environment.update(CUDA_VISIBLE_DEVICES=str(gpu), HIP_VISIBLE_DEVICES=str(gpu))
                command = job['command']
                launch = {'predeclared_at': utc(), 'command': command, 'environment': {**host['environment'], 'CUDA_VISIBLE_DEVICES': str(gpu), 'HIP_VISIBLE_DEVICES': str(gpu)}, 'gpu': gpu, 'cpu_affinity': job['cpu_affinity'], 'capacity_window': window, 'input_result_sha256': job['input_result_sha256'], 'checkpoint_sha256': job['checkpoint_sha256'], 'source_sha256': manifest['source_sha256'], 'tool_hashes': manifest['tool_hashes'], 'manifest_sha256': state['manifest_sha256']}
                save(output_root / f'provenance/{job_id}_launch.json', launch)
                log = (output_root / f'logs/{job_id}.log').open('x')
                process = subprocess.Popen(command, env=environment, cwd=Path(host['root']) / 'source', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                launch.update(worker_pid=process.pid, started_at=utc())
                save(output_root / f'provenance/{job_id}_launch.json', launch)
                state['workers'][job_id] = {'pid': process.pid, 'gpu': gpu, 'cpu_affinity': job['cpu_affinity'], 'started_at': launch['started_at'], 'exit_code': None}
                active[job_id] = {'process': process, 'log': log}
                state['pending'].remove(job_id)
                print(json.dumps({'job_id': job_id, **state['workers'][job_id]}), flush=True)
        state.update(status='queue_paused_after_failure' if failed else 'running' if active else 'capacity_checks', updated_at=utc())
        save(state_path, state)
        if failed and not active:
            break
        if state['pending'] or active:
            time.sleep(5)
    verify(manifest, host)
    state.update(status='failed' if failed else 'complete', finished_at=utc())
    save(state_path, state)


if __name__ == '__main__':
    main()
