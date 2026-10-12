"""模块大纲：不hold/取消原测量；实际全700控制器退出后串行8核matching。

绑定当前真实series PID/starttime/cmd，只等待已验收的真实Slurm终态；
partial短批次可按完成checkpoint接力，数值FAILED不重跑，MCMC门不改变。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from build_post_rsd_ir_matching_probe import ROOT, PART, OUT as MATCH, digest, save, read

OUT = ROOT/'ir_matching_after_measurements_v1'
SERIES = ROOT/'compute8_measurement_series_v3'
DONE_STATE = 'COMPUTE8_MEASUREMENTS_COMPLETE_ORIGINAL_C500_P_P0_CONTINUATIONS_PRESERVED'


def specific_live(identity):
    if identity['host'] != socket.gethostname(): raise ValueError('original specific PID on original host required')
    p = Path('/proc')/str(identity['pid'])
    try: fields = (p/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError: return False
    if fields[19] != identity['starttime_ticks']: return False
    if identity['entry'] not in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode():
        raise ValueError('same PID/starttime command changed; inspect instead of assuming exit')
    return fields[0] != 'Z'


def main(detach):
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
        raise ValueError('known original login26 CPU13 for light control only')
    if detach:
        log = ROOT/'ir_matching_after_measurements_v1.log'; launch = ROOT/'ir_matching_after_measurements_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists(): raise ValueError('inspect existing exact matching continuation, no duplicate')
        s = read(SERIES/'STATUS.json')
        identity = dict(pid=s['pid'],host=s['host'],starttime_ticks=s['starttime_ticks'],entry=str(PART/'code/continue_sto_post_compute8_series_v3.py'))
        if not specific_live(identity): raise ValueError('original measurement series must currently be confirmed live')
        command = [sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            p = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record = dict(pid=p.pid,host=socket.gethostname(),CPU_affinity=[13],command=command,
            launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log))
        with launch.open('x') as stream: json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    if OUT.exists(): raise ValueError('preserve existing controller')
    s = read(SERIES/'STATUS.json')
    identity = dict(pid=s['pid'],host=s['host'],starttime_ticks=s['starttime_ticks'],entry=str(PART/'code/continue_sto_post_compute8_series_v3.py'))
    if not specific_live(identity): raise ValueError('original series PID not actually live at binding')
    files = [Path(__file__),PART/'code/run_sto_ir_matching_compute8.py',
        PART/'code/submit_sto_ir_matching_compute8.slurm',SERIES/'CONTRACT.json',
        ROOT/'compute8_measurement_series_v3.launch.json',MATCH/'CONTRACT.json']
    sources = dict(read(MATCH/'CONTRACT.json')['immutable_sources'])
    sources.update({str(p):digest(p) for p in files})
    OUT.mkdir();owner=(OUT/'OWNER.lock').open('x');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'CONTRACT.json',dict(original_series_identity=identity,immutable_sources=sources,
        requested_compute_CPUs=8,requested_memory='14G',walltime='00:30:00',compute_batches_serial=True,
        original_measurement_scheduler_or_source_modified=False,main_MCMC_released=False))
    state = dict(state='WAITING_FOR_SPECIFIC_ALL700_MEASUREMENT_SERIES_EXIT',pid=os.getpid(),
        host=socket.gethostname(),starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')',1)[1].split()[19],
        original_series_pid=identity['pid'],started_UTC=datetime.now(timezone.utc).isoformat(),current_job_id=None,jobs=[])
    def update(**values):
        state.update(values,updated_UTC=datetime.now(timezone.utc).isoformat());save(OUT/'STATUS.json',state)
    def guard():
        for path,expected in sources.items():
            if digest(path) != expected: raise ValueError('matching continuation source/input drift: '+path)
    def accounting(jid):
        raw = subprocess.check_output(['sacct','-X','-j',str(jid),'-n','-P','-o','JobIDRaw,State,ExitCode'],text=True,timeout=30)
        rows = [line.split('|') for line in raw.splitlines() if line.startswith(str(jid)+'|')]
        return None if len(rows) != 1 else dict(job_id=jid,state=rows[0][1].split()[0],exit_code=rows[0][2])
    update()
    try:
        while True:
            guard();s=read(SERIES/'STATUS.json');live=specific_live(identity)
            update(original_series_actual_live=live,actual_series_state=s['state'],completed_measurements=s.get('completed_count'))
            if not live:
                if s['state'] != DONE_STATE or s.get('completed_count') != 700:
                    raise ValueError('original series exited without its original full700 gate')
                q=read(ROOT/'login_measurement_queue_v3/STATUS.json')
                if q['state'] != 'COMPLETE' or q['completed_count'] != 700 or q['workers']:
                    raise ValueError('original queue full700 terminal bookkeeping required')
                if not s['jobs'] or s['jobs'][-1]['state'] not in ('COMPLETED','TIMEOUT'):
                    raise ValueError('last actual measurement allocation termination evidence required')
                actual = accounting(s['jobs'][-1]['job_id'])
                if actual is None or actual['state'] not in ('COMPLETED','TIMEOUT'):
                    raise ValueError('measurement Slurm handle not terminal; no matching submission')
                break
            time.sleep(60.)
        while True:
            guard();answer=subprocess.check_output(['sbatch','--parsable',str(PART/'code/submit_sto_ir_matching_compute8.slurm')],text=True,timeout=30)
            jid=int(answer.strip().split(';')[0]);update(state='WAITING_FOR_SPECIFIC_MATCHING_COMPUTE8_JOB',current_job_id=jid)
            while True:
                guard();actual=accounting(jid)
                if actual and actual['state'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'): break
                # 观察缺失/PENDING/RUNNING均只等待这个handle，不重新提交。
                update(actual_matching_job_state=None if actual is None else actual['state']);time.sleep(30.)
            state['jobs'].append(actual);update()
            if actual['state'] != 'COMPLETED' or actual['exit_code'] != '0:0':
                raise ValueError('matching batch failed; preserve checkpoints and require inspection')
            batch=read(MATCH/('job'+str(jid))/'STATUS.json')
            if batch['active_worker'] is not None or batch['state'] not in ('COMPLETE','DRAINED_PARTIAL_CHECKPOINTS_PRESERVED'):
                raise ValueError('matching batch did not terminate with completed/drained source state')
            if batch['state'] == 'COMPLETE':
                if not (MATCH/'AUDIT.json').exists(): raise ValueError('all44 actual matching audit missing')
                update(state='ALL44_MATCHING_COMPUTE_AUDIT_COMPLETE_NOT_MCMC_RELEASE',
                    audit_sha256=digest(MATCH/'AUDIT.json'),finished_UTC=datetime.now(timezone.utc).isoformat());return
            update(state='PARTIAL_MATCHING_BATCH_CONFIRMED_TERMINAL_CONTINUING_CHECKPOINTS')
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise
    finally: owner.close()


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    main(ap.parse_args().detach)
