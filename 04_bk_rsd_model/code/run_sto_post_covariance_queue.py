"""模块大纲：Fid100..499正式补测；逐盒来源准入；二核起步、target结束后六核。

复用同一post数值入口；Fid0..99复用target。全部500测齐前不生成协方差。
parent与第一个worker共享CPU12；另保留CPU11给输入审计及一个旧单核reader，
target期间4+2+1+1最多8核，target结束后6+1+1最多8核。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from run_sto_post_target_queue import PART,ROOT,save,sha

OUT=ROOT/'fid100_499_measurement_queue'
BASE_CPUS=(12,13)
EXTRA_CPUS=(7,8,9,10)


def output(index):return ROOT/'pilots'/f'fiducial_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'


def source_row(index):
    # owned审计目录按新到旧；每盒必须有明确PASS，不以prefix数或path presence放行。
    for name in ('fid0_499_raw_source_audit_routed_v1','fid0_499_raw_source_audit_v4','fid0_499_raw_source_audit_v3'):
        path=ROOT/name/'rows'/f'fiducial_real{index:03d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['status']=='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':return path,row
    return None,None


def target_released():
    path=ROOT/'target300_measurement_queue/STATUS.json'
    state=json.loads(path.read_text())
    if state['host']!=socket.gethostname():raise ValueError('CPU lease must stay on original target host')
    if state['state']!='COMPLETE' or state['completed_count']!=300 or state['workers']:return False
    try:os.kill(state['pid'],0)
    except ProcessLookupError:return True
    return False


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if not set(BASE_CPUS+EXTRA_CPUS)<=os.sched_getaffinity(0):raise ValueError('declared CPU pool not permitted')
    if args.detach:
        launch=ROOT/'fid100_499_measurement_queue.launch.json';log=ROOT/'fid100_499_measurement_queue.launch.log'
        if launch.exists() or log.exists() or OUT.exists():raise ValueError('existing queue must be inspected, not restarted')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,command=command,base_CPUs=BASE_CPUS,
            conditional_extra_CPUs=EXTRA_CPUS,log=str(log),new_session=True,created_UTC=datetime.now(timezone.utc).isoformat())
        with launch.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(value),flush=True);return
    os.sched_setaffinity(0,{BASE_CPUS[0]});OUT.mkdir(exist_ok=False)
    selection=ROOT/'ENSEMBLE_SELECTION.json';chosen=json.loads(selection.read_text())
    if chosen['covariance_id_range_inclusive']!=[0,499] or chosen['covariance_is_of_mean']:
        raise ValueError('wrong human-confirmed ensemble')
    paths=(Path(__file__),PART/'code/run_sto_post_covariance_box.py',PART/'code/run_sto_post_target_box.py',
        PART/'code/run_sto_post_rsd_pilot.py',PART/'code/post_rsd_blob_catalog.py',PART/'code/post_rsd_estimator.py',
        PART/'code/run_sto_post_target_queue.py',selection,
        PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py',
        PART.parent/'01_reconstruction_methods/common/nersc_env.sh')
    frozen={str(p):sha(p) for p in paths}
    save(OUT/'CONTRACT.json',dict(role='complete remaining400 Fid500 same-box P/B measurements',
        covariance_ids=list(range(500)),measured_here_ids=list(range(100,500)),reused_target_ids=list(range(100)),
        n_covariance=500,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
        explicit_target_overlap=True,immutable_sources=frozen,base_CPUs=BASE_CPUS,
        conditional_extra_CPUs=EXTRA_CPUS,recon_mesh=256,measurement_mesh=256,nran=30,numeric_threads=1))
    state=dict(state='PREFLIGHT',host=socket.gethostname(),pid=os.getpid(),started_UTC=datetime.now(timezone.utc).isoformat(),
        completed=[],workers=[],pending_count=400,base_CPUs=BASE_CPUS,conditional_extra_CPUs=EXTRA_CPUS)
    active={};pending=list(range(100,500));error=None
    def update():
        state.update(completed_count=len(state['completed']),total_measured_here=400,pending_count=len(pending),
            workers=[{k:v for k,v in r.items() if k not in ('process','stream')} for r in active.values()],
            updated_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT/'STATUS.json',state)
    def finished(index,row):
        directory=output(index);path=directory/'DONE.json'
        if not path.exists():return False
        done=json.loads(path.read_text());contract=json.loads((directory/'CONTRACT.json').read_text())
        if (done['entry_sha256']!=frozen[str(PART/'code/run_sto_post_rsd_pilot.py')]
            or done['measurement_sha256']!=sha(directory/'MEASUREMENTS.npz')
            or done['post_halos_sha256']!=sha(directory/'POST_HALOS.npz')
            or done['contract_sha256']!=sha(directory/'CONTRACT.json')
            or contract['source_STO_sha256']!=row['source_sha256']
            or contract['source_diagnostics_sha256']!=row['diagnostics_sha256']
            or contract['raw_fof']['source_raw_sha256']!=row['raw_fof']['source_raw_sha256']):
            raise ValueError('completed covariance box provenance mismatch')
        return True
    try:
        update()
        for index in list(pending):
            path,row=source_row(index)
            if row is not None and finished(index,row):state['completed'].append(index);pending.remove(index)
            elif output(index).exists():raise ValueError('partial output requires original worker inspection')
        while pending or active:
            for path,want in frozen.items():
                if sha(path)!=want:raise ValueError('frozen measurement source or selection changed: '+path)
            cpus=BASE_CPUS+(EXTRA_CPUS if target_released() else ())
            state['active_CPU_pool']=cpus
            for cpu in cpus:
                if cpu in active or not pending or error:continue
                ready=None
                for index in pending:
                    path,row=source_row(index)
                    if path is not None:ready=(index,path,row);break
                if ready is None:break
                index,path,row=ready;pending.remove(index)
                log=OUT/f'fiducial_real{index:03d}.log';stream=log.open('x')
                command=['bash','-c',
                    'source ../01_reconstruction_methods/common/nersc_env.sh\n'
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
                    'export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1\n'
                    'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"\n'
                    'exec python -B -u code/run_sto_post_covariance_box.py "$@"',
                    'covariance-box','--realization',str(index),'--source-row',str(path)]
                proc=subprocess.Popen(command,cwd=PART,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda c=cpu:os.sched_setaffinity(0,{c}))
                active[cpu]=dict(cpu=cpu,pid=proc.pid,realization=index,log=str(log),source_row=str(path),
                    source_row_sha256=sha(path),process=proc,stream=stream)
            state['state']='RUNNING' if active else 'WAITING_FOR_VALIDATED_SOURCE_ROWS';update();time.sleep(2.)
            for cpu,worker in list(active.items()):
                code=worker['process'].poll()
                if code is None:continue
                worker['stream'].close();del active[cpu]
                path=Path(worker['source_row']);row=json.loads(path.read_text())
                if sha(path)!=worker['source_row_sha256'] or code or not finished(worker['realization'],row):
                    error=f"covariance box {worker['realization']} failed ({code}); preserve output and log"
                else:state['completed'].append(worker['realization'])
            update()
            if error and not active:raise RuntimeError(error)
        if len(state['completed'])!=400:raise ValueError('remaining covariance set incomplete')
        state['state']='COMPLETE';update()
    except Exception:
        state.update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED',error=traceback.format_exc());update();raise


if __name__=='__main__':main()
