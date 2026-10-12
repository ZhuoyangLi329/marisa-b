"""模块大纲：登录节点最多八核心进程树；同一采样器新链；依赖顺序与完成审计。

由用户明确要求登录节点执行。CPU affinity 硬限制八个逻辑 CPU，
各 worker 独占该集合中的一个 CPU；BLAS/OpenMP/NumExpr 线程数均为一。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.pop('PYTHONPATH',None)
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true')
    parser.add_argument('--cancel-pending-slurm',action='store_true')
    args=parser.parse_args()
    # 整个进程树的硬上限先于任何数值模块导入设置。
    cpus=sorted(os.sched_getaffinity(0))[:8]
    if not cpus:raise ValueError('no allowed CPU affinity')
    os.sched_setaffinity(0,set(cpus))
    from rsd_b000_contract import OUT,PART,digest,save_json
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    if args.start:
        if (OUT/'LOGIN_LAUNCH.json').exists():raise ValueError('login runner already launched; inspect state before retry')
        pipeline=json.loads((OUT/'PIPELINE.json').read_text())
        if args.cancel_pending_slurm:
            ids=list(pipeline['jobs'].values())
            listing=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%A|%j|%T'],text=True)
            for row in listing.splitlines():
                job,name,state=row.split('|')
                if job not in ids or not name.startswith('rsd_v1_') or state!='PENDING':
                    raise ValueError('unexpected/non-pending job; inspect before cancelling: '+row)
            subprocess.run(['scancel',*ids],check=True)
            for attempt in range(30):
                remaining=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%A'],text=True)
                if not remaining.strip():break
                time.sleep(1)
            else:raise ValueError('Slurm jobs have not fully stopped; do not start concurrent chain writers')
            pipeline.update({'status':'CANCELLED_FOR_USER_REQUESTED_LOGIN_EXECUTION',
                             'cancelled_slurm_jobs':ids,'execution':'login node, affinity restricted to at most 8 logical CPUs'})
            save_json(OUT/'PIPELINE.json',pipeline)
        elif pipeline['status']!='CANCELLED_FOR_USER_REQUESTED_LOGIN_EXECUTION':
            raise ValueError('cancel pending Slurm writers before login execution')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (logs/f'login_runner_{stamp}.out').open('w') as output,(logs/f'login_runner_{stamp}.err').open('w') as error:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                                   stdin=subprocess.DEVNULL,stdout=output,stderr=error,start_new_session=True)
        value={'pid':child.pid,'host':socket.gethostname(),'launch_UTC':datetime.now(timezone.utc).isoformat(),
               'maximum_logical_CPUs':len(cpus),'affinity':cpus,'threads_per_numeric_library':1,
               'runner_source_sha256':digest(Path(__file__)),'old_Slurm_pipeline':'cancelled before login execution'}
        save_json(OUT/'LOGIN_LAUNCH.json',value);print(json.dumps(value),flush=True);return
    lock=(OUT/'LOGIN_RUNNER.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':cpus,
           'maximum_logical_CPUs':len(cpus),'threads_per_numeric_library':1,'started_UTC':datetime.now(timezone.utc).isoformat()}
    def status(**extra):
        state.update(extra);state['updated_UTC']=datetime.now(timezone.utc).isoformat()
        save_json(OUT/'LOGIN_STATUS.json',state)
    def command(script,*args):return [sys.executable,'-B','-u',str(PART/'code'/script),*args]
    def single(stage,args):
        status(stage=stage,active_workers=[])
        subprocess.run(args,check=True)
    def cases(indices,stage):
        pending=list(indices);active={};finished=[];failures=[]
        while pending or active:
            for slot,cpu in enumerate(cpus):
                if not pending:break
                if slot in active:continue
                index=pending.pop(0)
                output=(logs/f'login_{stage}_{index}.out').open('w')
                error=(logs/f'login_{stage}_{index}.err').open('w')
                child=subprocess.Popen(['taskset','-c',str(cpu),*command('run_rsd_b000_mcmc.py','--case-index',str(index))],
                                       stdin=subprocess.DEVNULL,stdout=output,stderr=error)
                active[slot]=(index,child,output,error)
            for slot,(index,child,output,error) in list(active.items()):
                code=child.poll()
                if code is None:continue
                output.close();error.close();finished.append(index)
                if code:failures.append({'case_index':index,'exit_code':code})
                del active[slot]
            observed=[]
            for slot,item in active.items():
                try:allowed=sorted(os.sched_getaffinity(item[1].pid))
                except ProcessLookupError:allowed=[]
                if not set(allowed).issubset(cpus):raise ValueError('worker escaped eight-core affinity')
                observed.append({'case_index':item[0],'pid':item[1].pid,'CPU':cpus[slot],'actual_allowed_CPUs':allowed})
            status(stage=stage,completed_case_indices=finished,pending_case_indices=pending,
                   active_workers=observed,
                   failures=failures)
            if pending or active:time.sleep(2)
        if failures:raise ValueError('worker failed: '+json.dumps(failures))
        names=json.loads((OUT/'CASES.json').read_text())
        bad=[names[i] for i in finished if not json.loads((OUT/'cases'/names[i]/'SUMMARY.json').read_text())['converged']]
        if bad:raise ValueError('sampler budget reached without convergence: '+json.dumps(bad))
    try:
        parent=json.loads((OUT/'contract.json').read_text())
        if parent['volume_Gpc_h3']!=1 or parent['covariance_multiplier']!=1:raise ValueError('wrong covariance contract')
        cases(range(18),'baseline18')
        single('sensitivity_freeze',command('run_rsd_b000_inference.py','sensitivity'))
        names=json.loads((OUT/'CASES.json').read_text())
        cases(range(18,len(names)),'sensitivity')
        single('report',command('report_rsd_b000.py','--require-complete'))
        single('profile',command('analyze_rsd_b000_results.py'))
        single('completion_audit',command('audit_rsd_b000_completion.py'))
        status(status='COMPLETE',stage='complete',active_workers=[],completed_UTC=datetime.now(timezone.utc).isoformat())
        print('LOGIN_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        status(status='FAILED',error=str(error));raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
