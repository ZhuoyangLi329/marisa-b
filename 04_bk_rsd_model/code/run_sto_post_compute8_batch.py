"""模块大纲：8核/14GiB/30分钟的测量短批次；两盒单核并行，原数值复用。

剩余不足16分钟停止派发并自然排空；原parent实际暂停后才接管。
每批有独占lock/固定源码；canonical只含完成盒，不生成部分C500。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from sto_post_compute8_common import (PART, ROOT, OUT, read, sha, save, memory_capacity,
    enough_time, remote_pause_guard, target_rows, row_for, all_finished, keys, verified_finished,
    worker_environment)


def main():
    jid = os.environ.get('SLURM_JOB_ID')
    if not jid or os.environ.get('SLURM_CPUS_PER_TASK') != '8' or socket.gethostname().startswith('login'):
        raise ValueError('actual eight-CPU compute-node allocation required')
    cpus = sorted(os.sched_getaffinity(0))[:8]
    if len(cpus) != 8:
        raise ValueError('eight allowed logical CPUs required')
    os.sched_setaffinity(0, {cpus[-1]})
    run = OUT/('job'+jid); run.mkdir()
    for name in ('logs', 'publications', 'staging'):
        (run/name).mkdir()
    lock = (OUT/'COMPUTE_OWNER.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_text = subprocess.run(['scontrol', 'show', 'job', '-o', jid], check=True, text=True,
        stdout=subprocess.PIPE, timeout=20).stdout
    import re
    fields = dict(re.findall(r'\b([A-Za-z][A-Za-z0-9/]*)=([^\s]+)', job_text))
    deadline = (int(os.environ['SLURM_JOB_END_TIME']) if 'SLURM_JOB_END_TIME' in os.environ
                else datetime.fromisoformat(fields['EndTime']).timestamp())
    capacity = memory_capacity(int(os.environ['SLURM_MEM_PER_NODE']))
    state = dict(state='WAITING_FOR_ACTUAL_LOGIN_DELEGATION', job_id=jid, pid=os.getpid(),
        host=socket.gethostname(), allowed_CPUs=cpus, parent_CPU=cpus[-1], worker_limit=capacity,
        actual_CPU_limit=8, memory_MiB=int(os.environ['SLURM_MEM_PER_NODE']),
        workers=[], completed_count=None, started_UTC=datetime.now(timezone.utc).isoformat())
    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat()); save(run/'STATUS.json', state)
    update()
    handoff = OUT/'HANDOFF_READY.json'
    while not handoff.exists():
        if not enough_time(deadline-time.time()):
            update(state='BATCH_DRAINED_BEFORE_DELEGATION_NO_MEASUREMENT', workers=[]); return
        time.sleep(10)
    proof = read(handoff)
    if proof['state'] != 'ORIGINAL_V3_ACTUALLY_STOPPED_INFLIGHT_BOXES_COMPLETED_DELEGATION_READY':
        raise ValueError('actual exclusive delegation not ready')
    reference = proof['original_queue_identity']; remote_pause_guard(reference)
    coordinator_contract = read(OUT/'CONTRACT.json')
    frozen = coordinator_contract['immutable_sources']
    def guard():
        for source, want in frozen.items():
            if sha(source) != want:
                raise ValueError('frozen compute8 source drift: '+source)
        remote_pause_guard(reference)
    guard(); targets = target_rows(); pilot_sha = sha(PART/'code/run_sto_post_rsd_pilot.py')
    completed = all_finished(targets, pilot_sha); pending = [k for k in keys() if k not in completed]
    active = {}; failure = None; stopping = []
    signal.signal(signal.SIGUSR1, lambda *_: stopping.append('Slurm deadline: drain only'))
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('administrative termination: preserve staging'))
    save(run/'CONTRACT.json', dict(immutable_sources=frozen, handoff_sha256=sha(handoff),
        actual_CPU_limit=8, worker_limit=capacity, worker_memory_MiB=6144, shared_memory_MiB=2048,
        covariance_is_of_mean=False, covariance_multiplier=1., volume_Gpc_h3=1.,
        n_covariance=500, n_target_per_truth=100, original_numerical_sources_modified=False,
        output_only_atomic_staging=True, partial_C500_created=False))
    last_guard = 0.
    try:
        while pending or active:
            if time.monotonic()-last_guard >= 60:
                guard(); last_guard = time.monotonic()
            if not enough_time(deadline-time.time()):
                stopping.append('less than16min remaining: no new box')
            for cpu, worker in list(active.items()):
                rc = worker['process'].poll()
                if rc is None:
                    continue
                worker['stream'].close(); del active[cpu]
                _, row = row_for(worker['key'], targets)
                if rc or not verified_finished(worker['key'], row, pilot_sha):
                    failure = dict(key=worker['key'], exit_code=rc)
                else:
                    completed.append(worker['key'])
            while pending and not stopping and not failure and len(active) < capacity:
                used = set(active); cpu = next(c for c in cpus[:-1] if c not in used)
                key = pending.pop(0); source, _ = row_for(key, targets)
                args = [sys.executable, '-B', '-u', str(PART/'code/run_sto_post_compute8_box.py'),
                    '--cosmology', key[0], '--realization', str(key[1])]
                if source:
                    args += ['--source-row', str(source)]
                stream = (run/'logs'/f'{key[0]}_{key[1]:03d}.log').open('x')
                proc = subprocess.Popen(args, cwd=PART, stdout=stream, stderr=subprocess.STDOUT,
                    env=worker_environment(), preexec_fn=lambda cpu=cpu: os.sched_setaffinity(0, {cpu}))
                active[cpu] = dict(key=key, pid=proc.pid, process=proc, stream=stream)
            update(state='DRAINING' if stopping or failure else 'RUNNING',
                completed_count=len(completed), workers=[dict(cpu=c, key=w['key'], pid=w['pid']) for c, w in active.items()],
                seconds_remaining=deadline-time.time())
            if not active and (stopping or failure):
                break
            if active:
                time.sleep(2)
        if failure:
            raise ValueError('scientific worker failed; no automatic retry: '+str(failure))
        update(state='ALL700_COMPLETE_PUBLISHED_NOT_C500_OR_MCMC_RELEASE' if len(completed) == 700
            else 'SHORT_BATCH_DRAINED_INCOMPLETE_SAFE_TO_CONTINUE', workers=[],
            completed_count=len(completed), stop_reasons=stopping, finished_UTC=datetime.now(timezone.utc).isoformat())
    except Exception:
        update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED_PRESERVED',
            error=traceback.format_exc()); raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
