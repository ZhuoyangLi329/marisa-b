"""模块大纲：三组 P0-only 的登录节点连续执行；单核 worker；状态与来源。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
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
from run_rsd_p0only import OUT,PART,parent,digest,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cpus',default='7,8,9');args=parser.parse_args();cpus=[int(c) for c in args.cpus.split(',')]
    if not cpus or len(cpus)>8 or len(cpus)!=len(set(cpus)) or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError('invalid CPU affinity')
    os.sched_setaffinity(0,set(cpus));parent();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'LOGIN_RUNNER.lock').open('a')
    for attempt in range(21):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if args.start or attempt==20:raise
            time.sleep(.25)
    if args.start:
        launchpath=OUT/('LOGIN_RESUME_LAUNCH.json' if args.resume else 'LOGIN_LAUNCH.json')
        if launchpath.exists():raise ValueError('already launched; inspect status before resuming')
        if args.resume:
            previous=json.loads((OUT/'LOGIN_STATUS.json').read_text())
            if previous['status']!='FAILED' or previous.get('active_workers'):raise ValueError('resume requires stopped FAILED controller')
            initial=json.loads((OUT/'LOGIN_LAUNCH.json').read_text())
            snapshot=PART/'work/source_snapshots/run_rsd_p0only_login_initial.py'
            if digest(snapshot)!=initial['runner_sha256']:raise ValueError('initial controller snapshot mismatch')
            save_json(OUT/'resource_recovery/STARTUP_FAILED_STATUS.json',previous)
            save_json(OUT/'resource_recovery/INITIAL_LAUNCH.json',initial)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (logs/f'runner_{stamp}.out').open('w') as stdout,(logs/f'runner_{stamp}.err').open('w') as stderr:
            extra=['--resume'] if args.resume else []
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--cpus',args.cpus,*extra],
                stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        save_json(launchpath,{'pid':child.pid,'host':socket.gethostname(),'affinity':cpus,
            'maximum_logical_CPUs':len(cpus),'numeric_threads':1,'runner_sha256':digest(Path(__file__)),
            'resource_only_resume':args.resume,'launch_UTC':datetime.now(timezone.utc).isoformat()})
        print('P0_ONLY_LAUNCHED',child.pid,socket.gethostname(),cpus,flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();return
    state={'status':'RUNNING','pid':os.getpid(),'host':socket.gethostname(),'affinity':cpus,
           'maximum_logical_CPUs':len(cpus),'numeric_threads':1,'resource_only_resume':args.resume,
           'started_UTC':datetime.now(timezone.utc).isoformat()}
    active={};pending=list(range(3));completed=[]
    def status(**extra):
        state.update(extra,updated_UTC=datetime.now(timezone.utc).isoformat());save_json(OUT/'LOGIN_STATUS.json',state)
    try:
        status(stage='regression',active_workers=[])
        subprocess.run([sys.executable,'-B',str(PART/'code/test_rsd_p0only.py')],check=True)
        while pending or active:
            for cpu in cpus:
                if cpu in active or not pending:continue
                i=pending.pop(0);stdout=(logs/f'case_{i}.out').open('a');stderr=(logs/f'case_{i}.err').open('a')
                child=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(PART/'code/run_rsd_p0only.py'),
                    'sample','--case-index',str(i),'--max-steps','64000'],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                # Popen 返回时 taskset 可能尚未执行；父进程先完成单核绑定再核验，消除启动竞态。
                os.sched_setaffinity(child.pid,{cpu})
                active[cpu]=(i,child,stdout,stderr)
            for cpu,(i,child,stdout,stderr) in list(active.items()):
                code=child.poll()
                if code is None:continue
                stdout.close();stderr.close();del active[cpu]
                if code:raise RuntimeError(f'P0-only case {i} failed with exit {code}')
                completed.append(i)
            observed=[]
            for cpu,(i,child,_,_) in active.items():
                try:allowed=sorted(os.sched_getaffinity(child.pid))
                except ProcessLookupError:allowed=[]
                if not set(allowed).issubset({cpu}):raise ValueError('worker affinity drift')
                observed.append({'case_index':i,'pid':child.pid,'allowed_CPUs':allowed})
            status(stage='MCMC3',active_workers=observed,completed_cases=completed,pending_cases=pending)
            if active or pending:time.sleep(2)
        names=json.loads((OUT/'CASES.json').read_text())
        if not all(json.loads((OUT/'cases'/name/'SUMMARY.json').read_text())['converged'] for name in names):
            raise RuntimeError('MCMC budget reached before original convergence gates passed')
        status(stage='report',active_workers=[])
        subprocess.run([sys.executable,'-B',str(PART/'code/report_rsd_p0only.py')],check=True)
        parent();status(status='COMPLETE',stage='complete',active_workers=[],completed_UTC=datetime.now(timezone.utc).isoformat())
        print('P0_ONLY_COMPLETE',flush=True)
    except BaseException as error:
        status(status='FAILED',error=str(error),active_workers=[]);raise
    finally:
        for _,child,stdout,stderr in active.values():
            child.terminate();child.wait();stdout.close();stderr.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
