"""模块大纲：CPU shared32；复用原单盒数值入口；700盒同合同断点测量。

新调度器不修改旧调度源码/状态。必须先核实登录节点worker自然完成及parent
退出，已完成盒逐一验hash；source row仍须PASS；不降cov数量/平均cov。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from run_sto_post_target_queue import PART, ROOT, save, sha
from run_sto_post_covariance_queue import source_row

OUT = ROOT / 'compute32_measurement_queue'


def keys():
    return [(c, i) for i in range(100) for c in ('fiducial', 'LC_p', 'LC_m')] + [('fiducial', i) for i in range(100, 500)]


def directory(key):
    c, i = key
    return ROOT / 'pilots' / f'{c}_real{i:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'


def verified_finished(key, row, pilot_sha):
    path = directory(key)
    if not (path / 'DONE.json').exists():
        return False
    done = json.loads((path / 'DONE.json').read_text())
    contract = json.loads((path / 'CONTRACT.json').read_text())
    c, i = key
    if (done['status'] != 'PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION'
        or done['entry_sha256'] != pilot_sha
        or len(done['CPU_affinity']) != 1
        or contract['arguments']['cosmology'] != c
        or contract['arguments']['realization'] != i
        or contract['source_STO_sha256'] != row['source_sha256']
        or contract['source_diagnostics_sha256'] != row['diagnostics_sha256']
        or contract['raw_fof']['source_raw_sha256'] != row['raw_fof']['source_raw_sha256']):
        raise ValueError('completed box source/identity drift: ' + str(key))
    for field, filename in (('contract_sha256', 'CONTRACT.json'),
                            ('measurement_sha256', 'MEASUREMENTS.npz'),
                            ('post_halos_sha256', 'POST_HALOS.npz')):
        if done[field] != sha(path / filename):
            raise ValueError('completed box artifact drift: ' + str(key))
    return True


def migration_gate():
    path = ROOT / 'compute32_migration/MIGRATION.json'
    report = json.loads(path.read_text())
    if (report['state'] != 'LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE'
        or not report['original_parent_processes_confirmed_absent']):
        raise ValueError('original workers/dispatchers have not finished migration')
    for record in report['parents']:
        if sha(ROOT / record['queue'] / 'CONTRACT.json') != record['original_contract_sha256']:
            raise ValueError('old queue contract changed during migration')
        old = json.loads((ROOT / record['queue'] / 'CONTRACT.json').read_text())
        for source, want in old['immutable_sources'].items():
            if sha(source) != want:
                raise ValueError('old frozen numerical source changed: ' + source)
    return path


def worker_command(key, row_path):
    c, i = key
    if i < 100:
        return [sys.executable, '-B', '-u', str(PART / 'code/run_sto_post_target_box.py'),
                '--cosmology', c, '--realization', str(i)]
    return [sys.executable, '-B', '-u', str(PART / 'code/run_sto_post_covariance_box.py'),
            '--realization', str(i), '--source-row', str(row_path)]


def worker_environment():
    # 独立盒仍为单MPI rank，不能让32个子进程继承同一个srun PMI endpoint。
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PMI_', 'PMIX_'))}
    for name in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_STEP_NUM_TASKS', 'SLURM_NTASKS'):
        env.pop(name, None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
    return env


def memory_worker_limit(memory_mib):
    # 登录节点实测VmHWM约5.02GiB/盒；为波动留6GiB/盒和4GiB父进程/页缓存余量。
    limit = min(32, (int(memory_mib) - 4096) // 6144)
    if limit < 1:
        raise ValueError('allocation too small for unchanged measurement producer')
    return limit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise ValueError('measurement dispatcher must run in a compute-node Slurm allocation')
    available = sorted(os.sched_getaffinity(0))
    if args.workers != 32 or len(available) < 32:
        raise ValueError('human-requested32 permitted CPUs required')
    # 即使scheduler分配SMT sibling或额外memory CPUs，也最多用32逻辑核。
    cpus = available[:32]
    memory_mib = int(os.environ['SLURM_MEM_PER_NODE']) if 'SLURM_MEM_PER_NODE' in os.environ else int(os.environ.get('SLURM_MEM_PER_CPU', '1905')) * 32
    maximum_inflight = memory_worker_limit(memory_mib)
    os.sched_setaffinity(0, {cpus[0]})
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    migration = migration_gate()
    target_audit = ROOT / 'VALIDATION_SOURCE_AUDIT.json'
    audit = json.loads(target_audit.read_text())
    if audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':
        raise ValueError('target inputs not verified')
    targets = {(r['cosmology'], r['realization']): r for r in audit['source_rows']}
    if set(targets) != set(keys()[:300]):
        raise ValueError('target keys differ from requested mean100')
    run = OUT / ('job' + os.environ['SLURM_JOB_ID'])
    run.mkdir(exist_ok=False)
    sources = [Path(__file__), PART / 'code/run_sto_post_target_box.py',
        PART / 'code/run_sto_post_covariance_box.py', PART / 'code/run_sto_post_rsd_pilot.py',
        PART / 'code/run_sto_post_covariance_queue.py', PART / 'code/run_sto_post_target_queue.py',
        PART / 'code/post_rsd_blob_catalog.py', PART / 'code/post_rsd_estimator.py',
        PART.parent / '01_reconstruction_methods/common/halo_b000_pipeline.py',
        PART.parent / '01_reconstruction_methods/common/nersc_env.sh',
        ROOT / 'ENSEMBLE_SELECTION.json', target_audit, migration]
    frozen = {str(p): sha(p) for p in sources}
    contract = dict(role='all700 original post measurements; not inference release',
        requested_CPUs=32, active_CPU_pool=cpus, numeric_threads=1, Slurm_job_id=os.environ['SLURM_JOB_ID'],
        allocation_memory_MiB=memory_mib, maximum_inflight_workers=maximum_inflight,
        worker_memory_reservation_MiB=6144, parent_memory_reservation_MiB=4096,
        host=socket.gethostname(), immutable_sources=frozen, n_target_per_truth=100, n_covariance=500,
        covariance_ids=list(range(500)), explicit_target_overlap=True,
        covariance_is_of_mean=False, covariance_multiplier=1., volume_Gpc_h3=1.,
        recon_mesh=256, measurement_mesh=256, nran=30, original_numeric_entry_unchanged=True)
    save(run / 'CONTRACT.json', contract)
    save(OUT / 'CONTRACT.json', contract)
    state = dict(state='PREFLIGHT', pid=os.getpid(), host=socket.gethostname(),
        Slurm_job_id=os.environ['SLURM_JOB_ID'], attempt_directory=str(run),
        started_UTC=datetime.now(timezone.utc).isoformat(), completed=[], workers=[])
    pending = list(keys()); active = {}; error = None; stop = []
    signal.signal(signal.SIGUSR1, lambda *_: stop.append('Slurm deadline; stop dispatch and drain'))
    signal.signal(signal.SIGTERM, lambda *_: stop.append('Slurm termination; preserve current outputs'))
    pilot_sha = frozen[str(PART / 'code/run_sto_post_rsd_pilot.py')]

    def row_for(key):
        c, i = key
        if i < 100:
            return None, targets[key]
        return source_row(i)

    def update():
        state.update(completed_count=len(state['completed']), required=700, pending_count=len(pending),
            target_completed_count=sum(i < 100 for _, i in state['completed']),
            remaining_covariance_completed_count=sum(i >= 100 for _, i in state['completed']),
            workers=[{k: v for k, v in w.items() if k not in ('process', 'stream', 'row')} for w in active.values()],
            updated_UTC=datetime.now(timezone.utc).isoformat())
        save(run / 'STATUS.json', state); save(OUT / 'STATUS.json', state)

    try:
        for key in list(pending):
            row_path, row = row_for(key)
            if row is not None and verified_finished(key, row, pilot_sha):
                state['completed'].append(key); pending.remove(key)
            elif directory(key).exists():
                raise ValueError('incomplete output requires worker inspection; no automatic overwrite: ' + str(key))
        update()
        last_hash_check = 0.
        while pending or active:
            if time.monotonic() - last_hash_check > 60:
                for path, want in frozen.items():
                    if sha(path) != want:
                        raise ValueError('frozen source/input changed: ' + path)
                last_hash_check = time.monotonic()
            for cpu in cpus:
                if cpu in active or error or stop:
                    continue
                if len(active) >= maximum_inflight:
                    break
                ready = None
                for key in pending:
                    path, row = row_for(key)
                    if row is not None:
                        ready = key, path, row; break
                if ready is None:
                    break
                key, path, row = ready; pending.remove(key)
                log = run / f'{key[0]}_real{key[1]:03d}.log'
                stream = log.open('x')
                proc = subprocess.Popen(worker_command(key, path), cwd=PART, stdout=stream, env=worker_environment(),
                    stderr=subprocess.STDOUT, preexec_fn=lambda c=cpu: os.sched_setaffinity(0, {c}))
                active[cpu] = dict(cpu=cpu, pid=proc.pid, key=key, log=str(log),
                    process=proc, stream=stream, row=row, source_row=str(path) if path else str(target_audit),
                    source_row_sha256=sha(path) if path else sha(target_audit))
            state['state'] = 'DRAINING' if stop else ('RUNNING' if active else 'WAITING_FOR_VALIDATED_SOURCE_ROWS')
            update(); time.sleep(2.)
            for cpu, worker in list(active.items()):
                code = worker['process'].poll()
                if code is None:
                    continue
                worker['stream'].close(); del active[cpu]
                if (sha(worker['source_row']) != worker['source_row_sha256']
                    or code or not verified_finished(worker['key'], worker['row'], pilot_sha)):
                    error = f"measurement {worker['key']} failed ({code}); preserve output, no retry"
                else:
                    state['completed'].append(worker['key'])
            update()
            if error and not active:
                raise RuntimeError(error)
            if stop and not active:
                state.update(state='DRAINED_INCOMPLETE', stop_reasons=stop); update(); return
        if len(state['completed']) != 700:
            raise ValueError('required original700 inputs incomplete')
        state['state'] = 'COMPLETE'; update()
    except Exception:
        state.update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED', error=traceback.format_exc())
        update(); raise


if __name__ == '__main__':
    main()
