"""模块大纲：8CPU/14G/30分钟、单路重B投影、原44-bin checkpoint。

为最高bin缓存预留10GiB、parent/临时FFT留4GiB；各数值库单线程。
不足4分钟不派发新bin，当前bin自然完成；部分状态不是完整验收。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import build_post_rsd_ir_matching_probe as work


def resource_workers(cpus,memory_mib):
    if len(cpus) != 8 or len(set(cpus)) != 8 or int(memory_mib) != 14336:
        raise ValueError('exact8 logical CPUs/14GiB; one10GiB heavy worker plus4GiB reserve required')
    return 1


def allocation_deadline(job):
    if 'SLURM_JOB_END_TIME' in os.environ: return int(os.environ['SLURM_JOB_END_TIME'])
    raw = subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
    endings = re.findall(r'\bEndTime=(\S+)',raw)
    if len(endings) != 1: raise ValueError('actual Slurm EndTime missing, do not guess walltime')
    return datetime.fromisoformat(endings[0]).timestamp()


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '8':
        raise ValueError('actual eight-CPU compute allocation required')
    cpus = sorted(os.sched_getaffinity(0))
    resource_workers(cpus,int(os.environ['SLURM_MEM_PER_NODE']))
    stopped = []
    signal.signal(signal.SIGUSR1, lambda *_: stopped.append('deadline warning; drain current bin'))
    os.sched_setaffinity(0, {cpus[-1]})
    _, _, indices, _, _, contract, csha = work.prepare()
    owner = (work.OUT/'COMPUTE_OWNER.lock').open('a'); fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
    job = os.environ['SLURM_JOB_ID']; attempt = work.OUT/('job'+job)
    attempt.mkdir(exist_ok=False)
    entry = work.PART/'code/build_post_rsd_ir_matching_probe.py'
    sources = {str(Path(__file__)):work.digest(Path(__file__)),
        str(work.PART/'code/submit_sto_ir_matching_compute8.slurm'):work.digest(work.PART/'code/submit_sto_ir_matching_compute8.slurm')}
    deadline = allocation_deadline(job)
    pending = [index for index in indices if not work.completed(index,csha)]
    state = dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),job_id=job,
        starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')',1)[1].split()[19],
        allocation_CPUs=cpus,parent_CPU=cpus[-1],worker_limit=1,worker_reserved_MiB=10240,
        parent_and_FFT_reserved_MiB=4096,contract_sha256=csha,queue_sources=sources,
        started_UTC=datetime.now(timezone.utc).isoformat(),completed_bins=[i for i in indices if i not in pending],
        active_worker=None,MCMC_released=False)
    active = None; failure = None
    def update(**values):
        state.update(values,updated_UTC=datetime.now(timezone.utc).isoformat(),seconds_remaining=deadline-time.time())
        work.save(attempt/'STATUS.json',state); work.save(work.OUT/'COMPUTE_STATUS.json',state)
    def guard():
        if work.digest(work.OUT/'CONTRACT.json') != csha: raise ValueError('matching contract drift')
        for path, expected in {**contract['immutable_sources'],**sources}.items():
            if work.digest(path) != expected: raise ValueError('matching/source drift: '+path)
    update(); guard()
    try:
        while pending or active:
            if active:
                proc,index,stream = active; code = proc.poll()
                if code is not None:
                    stream.close(); active = None; update(active_worker=None)
                    if code or not work.completed(index,csha):
                        failure = dict(bin_index=index,exit_code=code); pending.clear()
                    else: state['completed_bins'].append(index)
            if time.time()+240 >= deadline and not stopped: stopped.append('remaining<4min; drain only')
            if stopped: update(state='DRAINING',stop_reasons=stopped)
            if not active and pending and not stopped and not failure:
                guard(); index = pending.pop(0); log = attempt/f'bin{index:03d}.log'; stream = log.open('xb')
                proc = subprocess.Popen([sys.executable,'-B','-u',str(entry),'--bin-index',str(index)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda: (os.sched_setaffinity(0,{cpus[0]}),signal.signal(signal.SIGUSR1,signal.SIG_IGN)))
                active = proc,index,stream
                update(active_worker=dict(bin_index=index,pid=proc.pid,CPU=cpus[0]))
            if not active and (stopped or failure): break
            update()
            if active: time.sleep(2.)
        guard()
        if failure:
            update(state='FAILED_PRESERVED',failure=failure); raise ValueError('matching bin failed; preserve outputs, no numerical retry')
        complete = [index for index in indices if work.completed(index,csha)]
        if len(complete) != 44:
            update(state='DRAINED_PARTIAL_CHECKPOINTS_PRESERVED',completed_bins=complete,remaining_bins=[i for i in indices if i not in complete])
            return
        rows = [work.read(work.OUT/'bins'/f'bin{index:03d}.json') for index in indices]
        report = dict(status='ALL44_SIGMA_ZERO_HARD_TREE_MATCHING_CANDIDATE_DOUBLE_GRID_VERIFIED_NOT_IR_MODEL_RELEASE',
            created_UTC=datetime.now(timezone.utc).isoformat(),contract_sha256=csha,rows=rows,
            bins=44,grid_projections=88,max_double_grid_scaled_error=max(r['double_grid_scaled_error'] for r in rows),
            max_zero_damping_matched_error=max(g['zero_damping_matched_error'] for r in rows for g in r['grids']),
            max_strict_source_matching_remainder_scaled=max(g['strict_source_matching_remainder_scaled'] for r in rows for g in r['grids']),
            original_noIR_or_data_modified=False,production_matching_selected=False,physical_p_calibrated=False,
            STO_noise_closed=False,post_covariance_whitened_error_checked=False,MCMC_released=False)
        target = work.OUT/'AUDIT.json'
        if target.exists(): raise ValueError('preserve existing complete matching report')
        work.save(target,report); update(state='COMPLETE',completed_bins=complete)
    except Exception:
        update(state='FAILED_PRESERVED',error=traceback.format_exc()); raise
    finally:
        owner.close()


if __name__ == '__main__': main()
