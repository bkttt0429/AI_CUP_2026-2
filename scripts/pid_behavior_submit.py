"""Serialize one-GPU study jobs and reserve their complete wall-time budgets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
ACTIVE = {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}


def available_seconds(records):
    used = 0
    for state, elapsed, limit in records:
        if state in TERMINAL: used += elapsed
        elif state in ACTIVE: used += max(elapsed, limit)
        else: raise ValueError('unknown scheduler state; refuse new allocation')
    return 3600 - used


def assert_allocation(root, stage):
    job = os.environ.get('SLURM_JOB_ID')
    ledger = json.loads((Path(root)/'jobs.json').read_text())
    if not job or not any(str(x['job']) == job and x['stage'] == stage for x in ledger['jobs']):
        raise RuntimeError('use the budgeted study launcher')
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('study requires exactly one visible GPU')


def submit(root, stage):
    root = Path(root)
    with open(root/'submit.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'jobs.json'
        ledger = json.loads(path.read_text()) if path.exists() else {'jobs': []}
        if ledger.get('pending_submission'):
            raise RuntimeError('previous submission outcome ambiguous; reconcile before retry')
        ids = [str(x['job']) for x in ledger['jobs']]
        records = []
        if ids:
            raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(ids),
                                           '--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw'], text=True)
            found = {}
            for line in raw.splitlines():
                cols = line.split('|')
                if cols[0] in ids:
                    found[cols[0]] = (cols[1].split()[0].rstrip('+'), int(cols[2]), int(cols[3])*60)
            if set(found) != set(ids): raise RuntimeError('scheduler accounting incomplete; retry later')
            records = list(found.values())
            if any(x[0] in ACTIVE for x in records): raise RuntimeError('another study job is still active')
        minutes = {'toy': 5, 'extract': 20, 'train': 30}[stage]
        if available_seconds(records) < minutes*60: raise RuntimeError('insufficient remaining allocation budget')
        scripts = Path(__file__).resolve().parent
        python = '/home/u2507115/BK_AICUP_2026-1/.venv/bin/python'
        script = scripts/('pid_behavior_extract.py' if stage == 'extract' else 'pid_behavior_train.py')
        cmd = ['env', 'AICUP1_ROOT=/home/u2507115/BK_AICUP_2026-1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', python, str(script), '--run-dir', str(root)]
        if stage == 'toy': cmd += ['--toy']
        ledger['pending_submission'] = stage; path.write_text(json.dumps(ledger, indent=2)+'\n')
        args = ['sbatch', '--parsable', '--account=gov113074', '--partition=dev', '--gpus-per-node=1',
                '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=32G', f'--time={minutes}',
                f'--job-name=behavior_{stage}', f'--output={root}/{stage}_%j.log', '--wrap='+shlex.join(cmd)]
        proc = subprocess.run(args, text=True, capture_output=True)
        if proc.returncode:
            ledger.pop('pending_submission'); path.write_text(json.dumps(ledger, indent=2)+'\n')
            raise RuntimeError(proc.stderr)
        job = proc.stdout.strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError('ambiguous job receipt; reconcile pending submission')
        ledger['jobs'].append({'job': job, 'stage': stage, 'limit_seconds': minutes*60})
        ledger.pop('pending_submission'); path.write_text(json.dumps(ledger, indent=2)+'\n')
        print(json.dumps({'job': job, 'stage': stage, 'remaining_seconds_before_reservation': available_seconds(records), 'reserved_seconds': minutes*60}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--stage', choices=('toy', 'extract', 'train'), required=True)
    a = p.parse_args(); submit(a.run_dir, a.stage)
