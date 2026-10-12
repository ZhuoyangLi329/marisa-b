"""模块大纲：登录节点四单核与计算节点两单核共同测量原700盒。

逐盒共享flock由parent和worker共同持有，独立staging验hash后原子发布。
原科学入口及冻结源码不改；原login parent保留，700完成后恢复原C500/p/P0流程。
旧计算调度器停止派发后，当前已运行批次自然排空，排队批次确认hold后取消。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
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
from sto_post_compute8_common import (PART, ROOT, read, sha, save, keys, directory,
    target_rows, row_for, all_finished, original_identity as original_proc_identity, remote_pause_guard,
    accounting, job_fields, command, worker_environment, enough_time, memory_capacity)
from run_sto_post_compute_queue import verified_finished

OUT=ROOT/'simultaneous_measurements_v1'
SUBMIT=PART/'code/submit_sto_post_simultaneous_compute8.slurm'
ENTRY=Path(__file__).resolve()
OLD=ROOT/'compute8_measurement_series_v3'
CPUS=(8,9,10,11)


def stamp(): return datetime.now(timezone.utc).isoformat()


def claim(path):
    """锁文件不删除；失败只关闭本次fd，不能重建inode绕过已存在的锁。"""
    stream=Path(path).open('a+')
    try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: stream.close(); return None
    return stream


def lock_path(key): return OUT/'claims'/f'{key[0]}_{key[1]:03d}.lock'


def own_identity(entry):
    return dict(host=socket.gethostname(),pid=os.getpid(),entry=str(entry),
        starttime_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])


def original_identity(record,*,stopped=False):
    try:fields=(Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    if fields[19]!=record['starttime_ticks'] or fields[0]=='Z':return False
    return original_proc_identity(record,stopped=stopped)


def wait_stopped(record):
    for _ in range(500):
        if original_identity(record,stopped=False):
            fields=(Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='T': return
        else: raise ValueError('specific prior controller disappeared before STOP')
        time.sleep(.01)
    raise ValueError('specific prior controller did not acknowledge STOP')


def guard(contract,remote=False):
    for path,want in contract['immutable_sources'].items():
        if sha(path)!=want: raise ValueError('frozen source changed: '+path)
    ref=contract['original_queue_identity']
    if remote: remote_pause_guard(ref)
    elif not original_identity(ref,stopped=True): raise ValueError('original login parent no longer paused')


def observed_count():
    return sum((directory(key)/'DONE.json').exists() for key in keys())


def verify_job(jid):
    raw,v=job_fields(jid)
    expected=dict(UserId='lzy(112513)',JobName='sto_simultaneous8',Account='desi',QOS='shared',
        NumCPUs='8',**{'CPUs/Task':'8'},TimeLimit='00:30:00',Command=str(SUBMIT))
    if any(v.get(k)!=want for k,want in expected.items()) or not v.get('ReqTRES','').startswith('cpu=8,mem=14G,'):
        raise ValueError('owned simultaneous8 resource/script mismatch: '+raw)
    if not any(x in ('scratch','scratch:1') for x in v.get('Licenses','').split(',')):
        raise ValueError('scratch license missing')
    if command(['scontrol','write','batch_script',str(jid),'-'])!=SUBMIT.read_text():
        raise ValueError('stored batch script differs')
    return v


def cancel_held_pending(jid,verify):
    """hold成功且复查仍PENDING才取消；已RUNNING则让它自然结束。"""
    v=verify(jid)
    if v['JobState']!='PENDING': return False
    result=subprocess.run(['scontrol','hold',str(jid)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        if verify(jid)['JobState']!='PENDING': return False
        raise ValueError('could not hold specific pending job: '+result.stderr)
    if verify(jid)['JobState']!='PENDING': return False
    command(['scancel',str(jid)]); return True


def migrate(update):
    from sto_post_compute8_v3_adapters import verify_job8
    frozen={}
    for rel in ['compute8_measurement_series_v3','physical_p_continuation_login_v3','p0only_continuation_v1']:
        frozen.update(read(ROOT/rel/'CONTRACT.json')['immutable_sources'])
    for path,want in frozen.items():
        if sha(path)!=want:raise ValueError('original frozen source drift before any transition: '+path)
    old=read(OLD/'STATUS.json')
    record={k:old[k] for k in ('pid','host','starttime_ticks')}
    record['entry']=str(PART/'code/continue_sto_post_compute8_series_v3.py')
    if old['state']!='WAITING_FOR_NEXT_SPECIFIC_SHORT8_BATCH':
        raise ValueError('inspect prior controller stage before simultaneous transition')
    if not original_identity(record): raise ValueError('specific prior controller identity missing')
    os.kill(record['pid'],signal.SIGSTOP); wait_stopped(record)
    old=read(OLD/'STATUS.json'); jid=old['current_job_id']; ref=old['original_queue_identity']
    if not original_identity(ref,stopped=True): raise ValueError('original queue must stay paused during transition')
    update(state='DRAINING_OR_CANCELING_PRIOR_SPECIFIC_COMPUTE_BATCH',prior_job_id=jid,
        prior_controller_identity=record,original_queue_identity=ref)
    canceled=cancel_held_pending(jid,verify_job8)
    while True:
        actual,raw=accounting(jid)
        if actual and actual['state'] not in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
            break
        update(prior_job_actual_state=actual,prior_job_observation=raw); time.sleep(15)
    allowed=(canceled and actual['state']=='CANCELLED') or actual['state']=='TIMEOUT' or (
        actual['state']=='COMPLETED' and actual['exit_code']=='0:0')
    if not allowed: raise ValueError('prior job ended unexpectedly: '+str(actual))
    if actual['state']=='COMPLETED':
        batch=read(OLD/f'job{jid}'/'STATUS.json')
        if batch.get('workers') or batch['state'] not in ('SHORT_BATCH_DRAINED_INCOMPLETE_SAFE_TO_CONTINUE',
            'ALL700_COMPLETE_PUBLISHED_NOT_C500_OR_MCMC_RELEASE','BATCH_DRAINED_BEFORE_DELEGATION_NO_MEASUREMENT'):
            raise ValueError('prior compute batch did not drain safely')
    owner=claim(OLD/'COMPUTE_OWNER.lock')
    if owner is None: raise ValueError('prior compute still holds its producer lock')
    owner.close()
    targets=target_rows(); pilot_sha=sha(PART/'code/run_sto_post_rsd_pilot.py')
    completed=all_finished(targets,pilot_sha)
    if any((c,i) not in completed for i in range(100) for c in ('fiducial','LC_p','LC_m')):
        raise ValueError('original target300 must be complete before this continuation')
    # 只结束已暂停的旧compute协调器；原login parent和两条科学continuation仍保留。
    if not original_identity(record,stopped=True): raise ValueError('specific old controller changed')
    os.kill(record['pid'],signal.SIGTERM); os.kill(record['pid'],signal.SIGCONT)
    for _ in range(100):
        if not original_identity(record): break
        time.sleep(.1)
    if original_identity(record): raise ValueError('specific old controller did not terminate')
    report=dict(status='OLD_COMPUTE_TERMINAL_ORIGINAL_LOGIN_PARENT_PRESERVED_SHARED_CLAIMS_READY',
        prior_controller_identity=record,prior_controller_confirmed_absent=True,prior_job=actual,
        pending_job_held_and_cancelled=canceled,original_queue_identity=ref,completed_count=len(completed),
        created_UTC=stamp(),authorization='Human: 登陆节点的别停下。一起算吧。',
        original_numeric_sources_modified=False,original_C500_p_P0_continuations_preserved=True)
    save(OUT/'MIGRATION.json',report)
    for path in [ENTRY,SUBMIT,OUT/'MIGRATION.json']: frozen[str(path)]=sha(path)
    contract=dict(immutable_sources=frozen,original_queue_identity=ref,login_worker_CPUs=list(CPUS),
        login_parent_CPU=12,login_maximum_workers=4,login_task_CPU_union=[7,8,9,10,11,12],
        compute_requested_CPUs=8,compute_memory='14G',compute_walltime='00:30:00',
        numerical_threads=1,shared_per_catalog_flock=True,claim_inherited_by_worker=True,
        original_numerical_sources_modified=False,n_covariance=500,n_target_per_truth=100,
        explicit_target_overlap=True,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.)
    guard(contract);save(OUT/'CONTRACT.json',contract)
    update(state='SIMULTANEOUS_SHARED_CLAIMS_READY',completed_count=len(completed))
    return contract


def atomic_box(args):
    from run_sto_post_compute8_box import OutputOnlyRoot,validate_publication
    key=args.cosmology,args.realization
    if key not in keys() or len(os.sched_getaffinity(0))!=1: raise ValueError('original single-core700 worker required')
    fd=args.claim_fd; want=lock_path(key)
    if os.fstat(fd).st_ino!=want.stat().st_ino or os.fstat(fd).st_dev!=want.stat().st_dev:
        raise ValueError('worker did not inherit its exact catalog lock inode')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Path(args.run).resolve()
    if not run.is_relative_to(OUT.resolve()): raise ValueError('staging must stay in owned simultaneous output')
    final=directory(key); staged_root=run/'staging'/f'{key[0]}_{key[1]:03d}'
    if final.exists() or staged_root.exists(): raise ValueError('preserve existing output or failed attempt')
    staged_root.mkdir(parents=True)
    import run_sto_post_rsd_pilot as pilot
    previous_root=pilot.ROOT; pilot.ROOT=OutputOnlyRoot(ROOT,staged_root)
    try:
        if key[1]<100:
            import run_sto_post_target_box as worker
            sys.argv=[worker.__file__,'--cosmology',key[0],'--realization',str(key[1])]
        else:
            import run_sto_post_covariance_box as worker
            if not args.source_row: raise ValueError('admitted source row required')
            sys.argv=[worker.__file__,'--realization',str(key[1]),'--source-row',args.source_row]
        worker.main()
    finally: pilot.ROOT=previous_root
    staged=staged_root/'pilots'/final.name
    _,row=row_for(key,target_rows());done=validate_publication(staged,final,row,key)
    staged.rename(final)
    save(run/'publications'/f'{key[0]}_{key[1]:03d}.json',dict(
        status='SHARED_CLAIM_ORIGINAL_BYTES_ATOMICALLY_PUBLISHED',key=key,canonical=str(final),
        original_DONE=done,created_UTC=stamp(),original_numeric_sources_modified=False))


def dispatcher(mode):
    contract=read(OUT/'CONTRACT.json'); compute=mode=='compute'
    if compute:
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='8':
            raise ValueError('eight-core compute allocation required')
        cpus=sorted(os.sched_getaffinity(0))
        if len(cpus)!=8 or int(os.environ['SLURM_MEM_PER_NODE'])!=14336: raise ValueError('exact8CPU14G required')
        verify_job(os.environ['SLURM_JOB_ID'])
        run=OUT/f"job{os.environ['SLURM_JOB_ID']}"; capacity=memory_capacity(14336)
        raw,fields=job_fields(os.environ['SLURM_JOB_ID'])
        deadline=int(os.environ['SLURM_JOB_END_TIME']) if 'SLURM_JOB_END_TIME' in os.environ else datetime.fromisoformat(fields['EndTime']).timestamp()
        parent_cpu=cpus[-1]; worker_cpus=cpus[:-1]
    else:
        from run_sto_post_login_queue_v2 import memory_budget
        if socket.gethostname()!='login26' or not set(CPUS+(12,))<=os.sched_getaffinity(0): raise ValueError('original login26 CPU pool required')
        budget=memory_budget(); baseline=int(Path(budget['current_file']).read_text())
        capacity=min(4,(budget['limit_bytes']-baseline-4*1024**3)//(6*1024**3))
        if capacity<1: raise ValueError('insufficient actual login memory headroom')
        run=OUT/'login';deadline=None;parent_cpu=12;worker_cpus=CPUS
    os.sched_setaffinity(0,{parent_cpu});run.mkdir()
    for name in ['logs','staging','publications']: (run/name).mkdir()
    dispatch_lock=claim(run/'DISPATCH.lock')
    if dispatch_lock is None: raise ValueError('same dispatcher already active')
    state=dict(own_identity(ENTRY),state='PREFLIGHT',mode=mode,workers=[],worker_limit=int(capacity),
        parent_CPU=parent_cpu,worker_CPUs=list(worker_cpus),started_UTC=stamp())
    if not compute:state['memory_budget']=budget
    def update(**values):
        state.update(values,updated_UTC=stamp());save(run/'STATUS.json',state)
    update();guard(contract,remote=compute)
    # 计算节点实测跨节点锁语义；协调器全程持有此锁，取得它即拒绝派发。
    probe=claim(OUT/'COORDINATOR.lock')
    if probe is not None:
        probe.close();raise ValueError('coordinator lock not visible across producer processes/nodes')
    update(coordinator_lock_exclusion_confirmed=True)
    targets=target_rows();pilot_sha=sha(PART/'code/run_sto_post_rsd_pilot.py')
    pending=list(keys()) if compute else list(reversed(keys()))
    active={};stop=[];failure=None;last_guard=0.;last_count=0.;count=None
    signal.signal(signal.SIGUSR1,lambda *_:stop.append('deadline drain'))
    signal.signal(signal.SIGTERM,lambda *_:stop.append('requested drain'))
    try:
        while pending or active:
            if time.monotonic()-last_guard>60:
                guard(contract,remote=compute);last_guard=time.monotonic()
            if deadline is not None and not enough_time(deadline-time.time()): stop.append('less than16min remaining')
            for cpu,w in list(active.items()):
                rc=w['process'].poll()
                if rc is None:continue
                w['stream'].close();_,row=row_for(w['key'],targets)
                if rc or sha(w['source_path'])!=w['source_sha256'] or not verified_finished(w['key'],row,pilot_sha):
                    failure=dict(key=w['key'],exit_code=rc)
                w['claim'].close();del active[cpu]
            while not stop and not failure and len(active)<capacity:
                chosen=None
                for key in list(pending):
                    if (directory(key)/'DONE.json').exists():pending.remove(key);continue
                    lease=claim(lock_path(key))
                    if lease is None:continue
                    if directory(key).exists():
                        lease.close()
                        if (directory(key)/'DONE.json').exists():pending.remove(key);continue
                        raise ValueError('partial canonical output must be inspected: '+str(key))
                    source,row=row_for(key,targets)
                    if row is None:lease.close();raise ValueError('original admitted source missing')
                    chosen=key,lease,source;pending.remove(key);break
                if chosen is None:break
                key,lease,source=chosen;cpu=next(c for c in worker_cpus if c not in active)
                argv=[sys.executable,'-B','-u',str(ENTRY),'box','--cosmology',key[0],
                    '--realization',str(key[1]),'--run',str(run),'--claim-fd',str(lease.fileno())]
                if source:argv+=['--source-row',str(source)]
                from run_sto_post_login_queue_v3 import environment_wrapped_command
                stream=(run/'logs'/f'{key[0]}_{key[1]:03d}.log').open('x')
                def before(c=cpu):
                    signal.signal(signal.SIGUSR1,signal.SIG_IGN);os.sched_setaffinity(0,{c})
                proc=subprocess.Popen(environment_wrapped_command(argv),cwd=PART,env=worker_environment(),
                    stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),preexec_fn=before)
                evidence=source or ROOT/'VALIDATION_SOURCE_AUDIT.json'
                active[cpu]=dict(key=key,pid=proc.pid,process=proc,stream=stream,claim=lease,
                    source_path=str(evidence),source_sha256=sha(evidence))
            if time.monotonic()-last_count>10:
                count=observed_count();last_count=time.monotonic()
            update(state='DRAINING' if stop or failure else 'RUNNING',completed_count=count,
                workers=[dict(cpu=c,key=w['key'],pid=w['pid']) for c,w in active.items()],
                seconds_remaining=deadline-time.time() if deadline else None)
            if not active and (stop or failure):break
            if pending or active:time.sleep(2)
        if failure:raise ValueError('scientific worker failed; preserve, no retry: '+str(failure))
        count=observed_count()
        update(state='COMPLETE' if count==700 else 'DRAINED_SAFE_TO_CONTINUE',workers=[],
            completed_count=count,finished_UTC=stamp())
    except Exception:
        update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED_PRESERVED',error=traceback.format_exc());raise
    finally:dispatch_lock.close()


def coordinator():
    if socket.gethostname()!='login26' or not set(CPUS+(12,))<=os.sched_getaffinity(0): raise ValueError('original login26 required')
    os.sched_setaffinity(0,{12});OUT.mkdir();(OUT/'claims').mkdir()
    owner=claim(OUT/'COORDINATOR.lock')
    if owner is None:raise ValueError('another simultaneous coordinator already owns this output')
    state=dict(own_identity(ENTRY),state='PREFLIGHT',current_job_id=None,jobs=[],started_UTC=stamp())
    def update(**values):
        state.update(values,updated_UTC=stamp());save(OUT/'STATUS.json',state)
    update();login=None
    try:
        contract=migrate(update)
        stream=(OUT/'login.log').open('x')
        login=subprocess.Popen([sys.executable,'-B','-u',str(ENTRY),'login'],cwd=PART,env=worker_environment(),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            preexec_fn=lambda:os.sched_setaffinity(0,set(CPUS+(12,))))
        update(state='LOGIN_AND_COMPUTE_SHARED_CLAIMS',login_pid=login.pid)
        jid=None
        while True:
            guard(contract);count=observed_count();update(completed_count=count)
            if login.poll() is not None:
                value=read(OUT/'login/STATUS.json')
                if login.returncode or value['state']!='COMPLETE':raise ValueError('login dispatcher failed; inspect preserved outputs')
            if jid is None and count<700:
                update(state='SUBMITTING_SPECIFIC_SHORT_COMPUTE_JOB',submission_attempt_started_UTC=stamp(),current_job_id=None)
                answer=command(['sbatch','--parsable',str(SUBMIT)])
                jid=int(answer.strip().split(';')[0]);update(current_job_id=jid,state='LOGIN_RUNNING_AND_COMPUTE_QUEUED_OR_RUNNING')
                verify_job(jid)
            if jid is not None:
                actual,raw=accounting(jid)
                if actual is None or actual['state'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
                    v=verify_job(jid);update(actual_job_state=v['JobState'],actual_job_reason=v.get('Reason'))
                    if count==700 and v['JobState']=='PENDING':cancel_held_pending(jid,verify_job)
                    time.sleep(20);continue
                state['jobs'].append(dict(actual,accounting_raw=raw));update()
                canceled=actual['state']=='CANCELLED' and count==700
                if not canceled and not (actual['state']=='TIMEOUT' or actual['state']=='COMPLETED' and actual['exit_code']=='0:0'):
                    raise ValueError('compute worker/job failed; preserve, no numerical retry: '+str(actual))
                if actual['state']=='COMPLETED':
                    batch=read(OUT/f'job{jid}'/'STATUS.json')
                    if batch['state'] not in ('COMPLETE','DRAINED_SAFE_TO_CONTINUE') or batch['workers']:
                        raise ValueError('compute did not drain safely')
                jid=None;update(current_job_id=None)
            if observed_count()==700:
                if login.poll() is None:time.sleep(5);continue
                completed=all_finished(target_rows(),sha(PART/'code/run_sto_post_rsd_pilot.py'))
                if len(completed)!=700:raise ValueError('full original700 hashes required')
                ref=contract['original_queue_identity']
                if not original_identity(ref,stopped=True):raise ValueError('original parent identity changed before restoration')
                os.kill(ref['pid'],signal.SIGCONT)
                update(state='ALL700_VERIFIED_ORIGINAL_LOGIN_RESUMED_FOR_C500_P_P0')
                while original_identity(ref):
                    for path,want in contract['immutable_sources'].items():
                        if sha(path)!=want:raise ValueError('frozen source drift after restoration')
                    time.sleep(15)
                final=read(ROOT/'login_measurement_queue_v3/STATUS.json')
                if final['state']!='COMPLETE' or final['completed_count']!=700 or final['workers']:
                    raise ValueError('original full700 gate failed')
                update(state='COMPLETE_ORIGINAL_C500_P_P0_CONTINUATIONS_PRESERVED',finished_UTC=stamp());return
            time.sleep(5)
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise
    finally:owner.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['coordinator','login','compute','box'])
    ap.add_argument('--detach',action='store_true');ap.add_argument('--cosmology');ap.add_argument('--realization',type=int)
    ap.add_argument('--source-row');ap.add_argument('--run');ap.add_argument('--claim-fd',type=int)
    args=ap.parse_args()
    if args.detach:
        if args.mode!='coordinator':raise ValueError('only coordinator detaches')
        launch=ROOT/'simultaneous_measurements_v1.launch.json';log=ROOT/'simultaneous_measurements_v1.log'
        if OUT.exists() or launch.exists() or log.exists():raise ValueError('existing simultaneous attempt must be inspected')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(ENTRY),'coordinator'],cwd=PART,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=proc.pid,host=socket.gethostname(),created_UTC=stamp(),log=str(log),entry=str(ENTRY))
        with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    if args.mode=='box':atomic_box(args)
    elif args.mode=='coordinator':coordinator()
    else:dispatcher(args.mode)


if __name__=='__main__':main()
