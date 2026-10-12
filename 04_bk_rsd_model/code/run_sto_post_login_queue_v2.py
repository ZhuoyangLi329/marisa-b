"""模块大纲：用户指定login恢复；4单核700盒断点测量，复用原数值/逐盒hash门。

独立新队列不覆盖旧状态；compute作业须确认未运行已取消；全局flock防双派发。
按实际user cgroup上限预留6GiB/worker和4GiB共享余量，最多4个测量worker。
不按盒数平均cov；全部700完成后由独立continuation聚合/校准。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1')
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import run_sto_post_compute_queue as shared

PART,ROOT=shared.PART,shared.ROOT
OUT=ROOT/'login_measurement_queue_v2'
CPUS=(8,9,10,11)
PARENT_CPU=12


def memory_budget():
    lines=Path('/proc/self/cgroup').read_text().splitlines();bases=[]
    for line in lines:
        version,controllers,path=line.split(':',2)
        if version=='0':bases.append((Path('/sys/fs/cgroup')/path.lstrip('/'),'memory.max','memory.current'))
    candidates=[]
    for base,limit,current in bases:
        for path in (base,*base.parents):
            if not str(path).startswith('/sys/fs/cgroup'):break
            try:
                value=(path/limit).read_text().strip()
                if value!='max':candidates.append((int(value),path,current))
            except OSError:pass
    if not candidates:raise ValueError('actual finite user memory cap not discovered')
    cap,path,current=min(candidates,key=lambda x:x[0])
    workers=min(4,(cap-4*1024**3)//(6*1024**3))
    if workers<1:raise ValueError('insufficient actual login memory cap')
    return dict(limit_bytes=cap,cgroup=str(path),current_file=str(path/current),maximum_workers=int(workers))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or not set(CPUS+(PARENT_CPU,))<=os.sched_getaffinity(0):
        raise ValueError('original login26 allowed CPU pool required')
    migration=ROOT/'login_resumption_20261009/MIGRATION.json';proof=json.loads(migration.read_text())
    if not proof.get('compute_job_cancelled_before_running') or not proof.get('old_controller_confirmed_absent'):
        raise ValueError('owned pending compute job and old waiting parent migration required')
    if args.detach:
        log=ROOT/'login_measurement_queue_v2.log';launch=ROOT/'login_measurement_queue_v2.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise RuntimeError('prior login attempt must be inspected, never overwritten')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,command=command,worker_CPUs=CPUS,parent_CPU=PARENT_CPU,
            launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log),detached=True)
        with launch.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(value),flush=True);return
    os.sched_setaffinity(0,{PARENT_CPU});budget=memory_budget()
    OUT.mkdir();shared.OUT.mkdir(exist_ok=True)
    lock=(shared.OUT/'DISPATCH.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_migration=shared.migration_gate()
    admission=ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json'
    if json.loads(admission.read_text())['status']!='ALL500_FIDUCIAL_SOURCES_AND_MEASUREMENT_ADMISSION_INDEPENDENTLY_VERIFIED':
        raise ValueError('full original500 source admission required')
    target_audit=ROOT/'VALIDATION_SOURCE_AUDIT.json';audit=json.loads(target_audit.read_text())
    if audit['status']!='ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':raise ValueError('exact300 source audit required')
    targets={(r['cosmology'],r['realization']):r for r in audit['source_rows']}
    if set(targets)!=set(shared.keys()[:300]):raise ValueError('target identity set differs')
    sources=[Path(__file__),PART/'code/run_sto_post_compute_queue.py',
        PART/'code/run_sto_post_target_box.py',PART/'code/run_sto_post_covariance_box.py',
        PART/'code/run_sto_post_rsd_pilot.py',PART/'code/run_sto_post_target_queue.py',PART/'code/run_sto_post_covariance_queue.py',
        PART/'code/post_rsd_blob_catalog.py',PART/'code/post_rsd_estimator.py',
        PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py',PART.parent/'01_reconstruction_methods/common/nersc_env.sh',
        ROOT/'ENSEMBLE_SELECTION.json',target_audit,admission,migration,old_migration]
    frozen={str(p):shared.sha(p) for p in sources}
    shared.save(OUT/'CONTRACT.json',dict(role='user-directed login resumption of identical700 post measurements',
        immutable_sources=frozen,worker_CPUs=CPUS,parent_CPU=PARENT_CPU,memory_budget=budget,numeric_threads=1,
        n_target_per_truth=100,n_covariance=500,covariance_ids=list(range(500)),explicit_target_overlap=True,
        covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,original_numerical_entry_unchanged=True))
    state=dict(state='PREFLIGHT',host=socket.gethostname(),pid=os.getpid(),
        starttime_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        started_UTC=datetime.now(timezone.utc).isoformat(),completed=[],workers=[],CPU_pool=CPUS,
        parent_CPU=PARENT_CPU,memory_budget=budget)
    pending=list(shared.keys());active={};failed=None;stop=[]
    signal.signal(signal.SIGTERM,lambda *_:stop.append('user/administrative stop: no further dispatch; drain current boxes'))
    def row_for(key):return (None,targets[key]) if key[1]<100 else shared.source_row(key[1])
    def update():
        state.update(completed_count=len(state['completed']),required=700,pending_count=len(pending),
            workers=[{k:v for k,v in row.items() if k not in ('process','stream','row')} for row in active.values()],
            updated_UTC=datetime.now(timezone.utc).isoformat())
        shared.save(OUT/'STATUS.json',state)
    pilot_sha=frozen[str(PART/'code/run_sto_post_rsd_pilot.py')];update()
    try:
        for key in list(pending):
            path,row=row_for(key)
            if row is None:raise ValueError('original admitted source missing: '+str(key))
            if shared.verified_finished(key,row,pilot_sha):state['completed'].append(key);pending.remove(key)
            elif shared.directory(key).exists():raise ValueError('partial old box requires process inspection, not overwrite')
            if len(state['completed'])%10==0:update()
        last_check=0.
        while pending or active:
            if time.monotonic()-last_check>60:
                for path,want in frozen.items():
                    if shared.sha(path)!=want:raise ValueError('frozen login resumption source drift: '+path)
                last_check=time.monotonic()
            for cpu in CPUS:
                if cpu in active or failed or stop or len(active)>=budget['maximum_workers'] or not pending:continue
                key=pending.pop(0);path,row=row_for(key);log=OUT/f'{key[0]}_real{key[1]:03d}.log';stream=log.open('xb')
                proc=subprocess.Popen(shared.worker_command(key,path),cwd=PART,stdout=stream,stderr=subprocess.STDOUT,
                    env=shared.worker_environment(),preexec_fn=lambda cpu=cpu:os.sched_setaffinity(0,{cpu}))
                evidence=path if path else target_audit
                active[cpu]=dict(cpu=cpu,pid=proc.pid,key=key,process=proc,stream=stream,row=row,
                    source_row=str(evidence),source_row_sha256=shared.sha(evidence),log=str(log))
            state['state']='DRAINING' if stop or failed else 'RUNNING';update();time.sleep(2.)
            for cpu,worker in list(active.items()):
                rc=worker['process'].poll()
                if rc is None:continue
                worker['stream'].close();del active[cpu]
                if rc or shared.sha(worker['source_row'])!=worker['source_row_sha256'] or not shared.verified_finished(worker['key'],worker['row'],pilot_sha):
                    failed=dict(key=worker['key'],exit_code=rc)
                else:state['completed'].append(worker['key'])
            update()
            if failed and not active:raise RuntimeError('box failed; preserve outputs: '+str(failed))
            if stop and not active:state.update(state='DRAINED_INCOMPLETE',stop_reasons=stop);update();return
        if len(state['completed'])!=700:raise ValueError('all original700 required')
        state['state']='COMPLETE';update()
    except Exception:
        state.update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED',error=traceback.format_exc())
        update();raise


if __name__=='__main__':main()
