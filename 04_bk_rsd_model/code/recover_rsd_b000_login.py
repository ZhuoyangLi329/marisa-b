"""模块大纲：只延长未收敛链；48k/64k显式预算来源；原八核池内恢复原流水线。

不更改 data/covariance/先验/kernel/诊断门；复用唯一 MCMC 数值入口。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.pop('PYTHONPATH',None);os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import fcntl
import json
import socket
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true');args=parser.parse_args()
    part=Path(__file__).resolve().parents[1];output=part/'outputs/pre_rsd_b000_bonly_mean100_v1'
    launch=json.loads((output/'LOGIN_LAUNCH.json').read_text());pool=launch['affinity']
    if socket.gethostname()!=launch['host'] or len(pool)>8:raise ValueError('wrong login host/core limit')
    os.sched_setaffinity(0,set(pool))
    from rsd_b000_contract import save_json,digest
    logs=output/'logs'
    if args.start:
        if (output/'BUDGET_RECOVERY_LAUNCH.json').exists():raise ValueError('recovery already launched; inspect before retry')
        state=json.loads((output/'LOGIN_STATUS.json').read_text())
        if state['status']!='FAILED' or state['active_workers']:raise ValueError('original chain writers have not stopped')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (logs/f'budget_recovery_{stamp}.out').open('w') as stdout,(logs/f'budget_recovery_{stamp}.err').open('w') as stderr:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                                   stdout=stdout,stderr=stderr,start_new_session=True)
        value={'pid':child.pid,'host':socket.gethostname(),'affinity':pool,'maximum_logical_CPUs':len(pool),
               'threads_per_numeric_library':1,'source_sha256':digest(Path(__file__)),
               'purpose':'explicit budget-only continuation, no changed convergence gate or physical model'}
        save_json(output/'BUDGET_RECOVERY_LAUNCH.json',value);print(json.dumps(value),flush=True);return
    lock=(output/'BUDGET_RECOVERY.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'RUNNING','pid':os.getpid(),'host':socket.gethostname(),'affinity':pool,'maximum_logical_CPUs':len(pool)}
    def status(**extra):
        state.update(extra);state['updated_UTC']=datetime.now(timezone.utc).isoformat();save_json(output/'BUDGET_RECOVERY_STATUS.json',state)
    def bad_cases():
        names=json.loads((output/'CASES.json').read_text());bad=[]
        for index,name in enumerate(names):
            path=output/'cases'/name/'SUMMARY.json'
            if path.exists() and not json.loads(path.read_text())['converged']:bad.append(index)
        return bad
    def extend(indices,budget):
        original=(output/'LOGIN_RUNNER.lock').open('a');fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};pending=list(indices);finished=[]
        ordered=[c for c in (7,8,9,10,11,12,13,6) if c in pool]
        try:
            while pending or active:
                for slot,cpu in enumerate(ordered):
                    if not pending:break
                    if slot in active:continue
                    index=pending.pop(0);names=json.loads((output/'CASES.json').read_text());root=output/'cases'/names[index]
                    initial=root/'INITIAL_BUDGET_SUMMARY.json'
                    if not initial.exists():save_json(initial,json.loads((root/'SUMMARY.json').read_text()))
                    stdout=(logs/f'extension_{index}_max{budget}.out').open('w');stderr=(logs/f'extension_{index}_max{budget}.err').open('w')
                    child=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(part/'code/run_rsd_b000_mcmc.py'),
                                            '--case-index',str(index),'--max-steps',str(budget),'--extend-budget'],
                                            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                    active[slot]=(index,child,stdout,stderr)
                for slot,(index,child,stdout,stderr) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    stdout.close();stderr.close();del active[slot];finished.append(index)
                    if code:raise ValueError('budget worker failed; inspect '+str(index))
                observed=[]
                for slot,(index,child,stdout,stderr) in active.items():
                    try:allowed=sorted(os.sched_getaffinity(child.pid))
                    except ProcessLookupError:allowed=[]
                    if not set(allowed).issubset(pool):raise ValueError('worker escaped approved CPU pool')
                    observed.append({'case_index':index,'pid':child.pid,'actual_allowed_CPUs':allowed})
                status(stage='budget_extension',max_steps=budget,finished_worker_indices=finished,active_workers=observed)
                if pending or active:time.sleep(2)
        finally:fcntl.flock(original,fcntl.LOCK_UN);original.close()
    try:
        for cycle in range(3):
            for budget in (48000,64000):
                bad=bad_cases()
                if not bad:break
                extend(bad,budget)
            if bad_cases():raise ValueError('64k budget exhausted without convergence; keep diagnostics, do not relax gates')
            status(stage='resuming_original_pipeline',active_workers=[])
            code=subprocess.call([sys.executable,'-B','-u',str(part/'code/run_rsd_b000_login.py')])
            if code==0:
                status(status='COMPLETE',stage='complete',active_workers=[]);print('BUDGET_RECOVERY_COMPLETE',flush=True);return
            if not bad_cases():raise ValueError('pipeline failed outside convergence budget; inspect logs')
        raise ValueError('recovery cycle limit reached')
    except BaseException as error:status(status='FAILED',error=str(error));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
