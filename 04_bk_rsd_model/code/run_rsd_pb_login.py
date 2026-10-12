"""模块大纲：登录节点硬限八核；pilot→bulk→联合冻结→18组MCMC→报告/审计。

全部任务自动衔接；不提交 Slurm、不重测 B、不改旧结果、不安装环境。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.pop('PYTHONPATH',None)
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
    parser.add_argument('--cpus',default='7,8,9,10,11,12,13,14')
    parser.add_argument('--resource-recovery',action='store_true')
    parser.add_argument('--measurement-batch-size',type=int,default=100)
    parser.add_argument('--measurement-workers',type=int,default=8);args=parser.parse_args()
    cpus=[int(v) for v in args.cpus.split(',')]
    if not cpus or len(cpus)>8 or len(set(cpus))!=len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError('invalid at-most-eight CPU set')
    if not 1<=args.measurement_workers<=len(cpus) or not 1<=args.measurement_batch_size<=100:
        raise ValueError('invalid measurement resource limits')
    os.sched_setaffinity(0,set(cpus))
    from rsd_pb_common import OUT,PART,digest,save_json
    from run_rsd_pb_inference import parent
    parent();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'LOGIN_RUNNER.lock').open('a')
    for attempt in range(21):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if args.start or attempt==20:raise
            time.sleep(.25) # detached 子进程只等待启动者写完 launch 合同，最多五秒。
    if args.start:
        launchpath=OUT/('LOGIN_RESOURCE_RECOVERY_LAUNCH.json' if args.resource_recovery else 'LOGIN_LAUNCH.json')
        if launchpath.exists():raise ValueError('already launched; inspect live process/status before retry')
        if args.resource_recovery:
            previous=json.loads((OUT/'LOGIN_STATUS.json').read_text());initial=json.loads((OUT/'LOGIN_LAUNCH.json').read_text())
            if previous['status']!='FAILED' or previous.get('active_workers') or not previous.get('failures'):
                raise ValueError('original runner must have stopped with documented worker failure before recovery')
            if any(item['exit_code']!=-9 for item in previous['failures']):raise ValueError('not a resource-only SIGKILL recovery')
            snapshot=PART/'work/source_snapshots/run_rsd_pb_login_batch100.py'
            if digest(snapshot)!=initial['runner_sha256']:raise ValueError('original controller snapshot mismatch')
            save_json(OUT/'resource_recovery/ORIGINAL_FAILED_STATUS.json',previous)
            save_json(OUT/'resource_recovery/ORIGINAL_LAUNCH.json',initial)
            save_json(OUT/'resource_recovery/PREVIOUS_PILOT_AUDIT.json',json.loads((OUT/'PILOT_AUDIT.json').read_text()))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        extra=['--measurement-batch-size',str(args.measurement_batch_size),'--measurement-workers',str(args.measurement_workers)]
        if args.resource_recovery:extra.append('--resource-recovery')
        with (logs/f'runner_{stamp}.out').open('w') as out,(logs/f'runner_{stamp}.err').open('w') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--cpus',args.cpus,*extra],
                                   stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        save_json(launchpath,{'pid':child.pid,'host':socket.gethostname(),'affinity':cpus,
                  'maximum_logical_CPUs':len(cpus),'threads_per_numeric_library':1,
                  'launch_UTC':datetime.now(timezone.utc).isoformat(),'runner_sha256':digest(Path(__file__)),
                  'resource_recovery':args.resource_recovery,'measurement_batch_size':args.measurement_batch_size,
                  'maximum_measurement_workers':args.measurement_workers,'maximum_MCMC_workers':len(cpus),
                  'measurement_model_covariance_sampler_unchanged':True,
                  'pipeline':'pilot P60+fine3 → bulk P800 → freeze full P+B → 18 MCMC → report → completion audit'})
        print('LAUNCHED',child.pid,socket.gethostname(),cpus,flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();return
    state={'status':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':cpus,
           'maximum_logical_CPUs':len(cpus),'threads_per_numeric_library':1,'started_UTC':datetime.now(timezone.utc).isoformat()}
    state.update(resource_recovery=args.resource_recovery,measurement_batch_size=args.measurement_batch_size,
                 maximum_measurement_workers=args.measurement_workers,maximum_MCMC_workers=len(cpus))
    def status(**extra):
        state.update(extra);state['updated_UTC']=datetime.now(timezone.utc).isoformat();save_json(OUT/'LOGIN_STATUS.json',state)
    def command(script,*extra):
        if script=='measure_rsd_p0.py':return ['bash',str(PART/'code/run_rsd_p0_measure.sh'),*map(str,extra)]
        return [sys.executable,'-B','-u',str(PART/'code'/script),*map(str,extra)]
    def single(stage,cmd):
        status(stage=stage,active_workers=[]);subprocess.run(cmd,check=True)
    def pool(stage,tasks,workers=None):
        pending=list(enumerate(tasks));active={};finished=[];failures=[]
        workers=len(cpus) if workers is None else workers
        try:
            while pending or active:
                for slot,cpu in enumerate(cpus[:workers]):
                    if not pending:break
                    if slot in active:continue
                    index,cmd=pending.pop(0);out=(logs/f'{stage}_{index:03d}.out').open('w');err=(logs/f'{stage}_{index:03d}.err').open('w')
                    child=subprocess.Popen(['taskset','-c',str(cpu),*cmd],stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                    active[slot]=(index,child,out,err)
                for slot,(index,child,out,err) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    out.close();err.close();finished.append(index)
                    if code:failures.append({'task':index,'exit_code':code})
                    del active[slot]
                observed=[]
                for slot,(index,child,_,_) in active.items():
                    try:allowed=sorted(os.sched_getaffinity(child.pid))
                    except ProcessLookupError:allowed=[]
                    if not set(allowed).issubset(cpus):raise ValueError('worker escaped CPU ceiling')
                    observed.append({'task':index,'pid':child.pid,'CPU':cpus[slot],'actual_allowed_CPUs':allowed})
                status(stage=stage,completed_tasks=finished,pending_tasks=[i for i,_ in pending],active_workers=observed,failures=failures)
                if failures:
                    pending=[] # 让已启动 worker 安全完成，然后阻断下一阶段。
                if pending or active:time.sleep(2)
        finally:
            for _,child,out,err in active.values():
                child.terminate();child.wait();out.close();err.close()
        if failures:raise ValueError(stage+' failed: '+json.dumps(failures))
    try:
        single('preflight_regression',command('test_rsd_pb.py'))
        if not args.resource_recovery:
            pool('pilot_P60',[command('measure_rsd_p0.py','--pilot','--start',i,'--stop',min(i+8,60)) for i in range(0,60,8)])
            pool('fine_P3',[command('measure_rsd_p0.py','--fine','--meshsize',512,'--start',i,'--stop',i+1) for i in range(3)])
        single('pilot_numeric_audit',command('run_rsd_pb_inference.py','pilot'))
        rows=parent()['samples'];batch=args.measurement_batch_size;tasks=[]
        for i in range(0,800,batch):
            group=rows[i:min(i+batch,800)]
            if all((OUT/'measurements_m256'/f"{r['cosmology']}_real{r['realization']:03d}.npz").exists() for r in group):continue
            tasks.append(command('measure_rsd_p0.py','--start',i,'--stop',min(i+batch,800)))
        pool('bulk_recovery_P800' if args.resource_recovery else 'bulk_P800',tasks,workers=args.measurement_workers)
        single('joint_freeze',command('run_rsd_pb_inference.py','aggregate'))
        single('frozen_likelihood_regression',command('test_rsd_pb.py'))
        pool('MCMC18',[command('run_rsd_pb_mcmc.py','--case-index',i,'--max-steps',64000) for i in range(18)])
        names=json.loads((OUT/'CASES.json').read_text());bad=[]
        for name in names:
            summary=json.loads((OUT/'cases'/name/'SUMMARY.json').read_text())
            if not summary['converged']:bad.append(name)
        single('joint_report',command('report_rsd_pb.py'))
        if bad:raise ValueError('MCMC budget reached without convergence: '+json.dumps(bad))
        single('completion_audit',command('audit_rsd_pb_completion.py'))
        status(status='COMPLETE',stage='complete',active_workers=[],completed_UTC=datetime.now(timezone.utc).isoformat())
        print('P_B_LOGIN_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        status(status='FAILED',error=str(error),active_workers=[]);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
