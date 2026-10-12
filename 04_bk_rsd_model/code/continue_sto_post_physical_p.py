"""模块大纲：等待确切测量job；完整700/C500门后在login26单核校准物理p。

衔接必要校准，不将它当作39组主MCMC或IR完成。不修改/重提测量作业；观察
超时不是进程终止，继续核对同一job。采样和全部收敛门复用唯一原实现。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'physical_p_continuation'
WAIT_STATES={'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED','RESIZING'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def job_record(output,job_id):
    rows=[line.split('|') for line in output.splitlines() if line.strip()]
    matches=[r for r in rows if r[0]==str(job_id)]
    if len(matches)!=1 or len(matches[0])<3:raise ValueError('known Slurm job not unambiguously observed')
    row=matches[0];return dict(job_id=str(job_id),state=row[1].split()[0].rstrip('+'),exit_code=row[2])


def ensemble_gate(record):
    stats=record['statistics']
    if (record['status']!='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE'
        or len(record['rows'])!=700 or stats['n_covariance']!=500 or stats['n_target_per_truth']!=100
        or record['covariance_is_of_mean'] or record['covariance_multiplier']!=1.
        or record['volume_Gpc_h3']!=1.):raise ValueError('complete700/mean100/C_single500/V1 required')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--measurement-job-id',type=int,required=True)
    ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={7}:
        raise ValueError('single CPU7 on original login26; disjoint from audit CPU11')
    admission=ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json'
    proof=json.loads(admission.read_text())
    if proof['status']!='ALL500_FIDUCIAL_SOURCES_AND_MEASUREMENT_ADMISSION_INDEPENDENTLY_VERIFIED':
        raise ValueError('independent500 admission gate not passed')
    if args.detach:
        launch=ROOT/'physical_p_continuation.launch.json';log=ROOT/'physical_p_continuation.log'
        if launch.exists() or log.exists() or OUT.exists():raise ValueError('previous continuation must be inspected, not replaced')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--measurement-job-id',str(args.measurement_job_id)]
        with log.open('x') as stream:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,host=socket.gethostname(),command=command,CPU_affinity=[7],
            measurement_job_id=args.measurement_job_id,created_UTC=datetime.now(timezone.utc).isoformat(),log=str(log))
        with launch.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(value),flush=True);return
    OUT.mkdir(exist_ok=False)
    sources=[Path(__file__),admission,ROOT/'ENSEMBLE_SELECTION.json',
        PART/'code/run_sto_physical_p_mcmc.py',PART/'code/post_rsd_p_calibration.py',
        PART/'code/post_rsd_p0_model.py',PART/'code/run_rsd_b000_mcmc.py',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    frozen={str(p):sha(p) for p in sources}
    save(OUT/'CONTRACT.json',dict(role='necessary physical-p calibration after complete measurements, not main inference',
        measurement_job_id=args.measurement_job_id,CPU_affinity=[7],numeric_threads=1,
        immutable_sources=frozen,full_goal_reduced=False,post_main_MCMC_total=39,IR_control_still_required=True))
    state=dict(state='WAITING_FOR_SPECIFIC_MEASUREMENT_JOB',host=socket.gethostname(),pid=os.getpid(),
        measurement_job_id=args.measurement_job_id,child_pid=None,started_UTC=datetime.now(timezone.utc).isoformat())
    save(OUT/'STATUS.json',state)
    try:
        while True:
            for path,want in frozen.items():
                if sha(path)!=want:raise ValueError('frozen calibration entry/input drift: '+path)
            try:
                observed=subprocess.run(['sacct','-X','-n','-P','-j',str(args.measurement_job_id),
                    '-o','JobIDRaw,State,ExitCode'],check=True,text=True,stdout=subprocess.PIPE,timeout=20)
                job=job_record(observed.stdout,args.measurement_job_id)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as error:
                state.update(last_observation_uncertain=str(error),updated_UTC=datetime.now(timezone.utc).isoformat())
                save(OUT/'STATUS.json',state);time.sleep(30);continue
            state.update(actual_Slurm_job=job,updated_UTC=datetime.now(timezone.utc).isoformat());save(OUT/'STATUS.json',state)
            if job['state']=='COMPLETED':
                if job['exit_code']!='0:0':raise ValueError('measurement job completed with nonzero exit')
                break
            if job['state'] not in WAIT_STATES:raise ValueError('measurement job terminal/not successful; no auto restart: '+str(job))
            time.sleep(30)
        report_path=ROOT/'ensemble_measurements/AUDIT.json';report=json.loads(report_path.read_text())
        ensemble_gate(report)
        for info in report['ensemble_files'].values():
            if sha(info['path'])!=info['sha256']:raise ValueError('complete ensemble bytes drift')
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        entry=PART/'code/run_sto_physical_p_mcmc.py'
        for stage,command in (('FREEZING_PHYSICAL_P_INPUTS','freeze'),('PHYSICAL_P_CALIBRATION_MCMC','sample')):
            state.update(state=stage,child_pid=None);save(OUT/'STATUS.json',state)
            child=subprocess.Popen([sys.executable,'-B','-u',str(entry),command],cwd=PART,env=env)
            state['child_pid']=child.pid;save(OUT/'STATUS.json',state)
            if child.wait():raise ValueError('physical-p '+command+' failed; preserve diagnostic outputs')
            state['child_pid']=None;save(OUT/'STATUS.json',state)
        path=ROOT/'physical_p_calibration_mcmc/cases/physical_p_GSR2_RSD_fNLfixed100/SUMMARY.json'
        result=json.loads(path.read_text())
        state.update(state='PHYSICAL_P_CALIBRATION_CONVERGED_NOT_MAIN_MCMC_RELEASE' if result['converged']
            else 'PHYSICAL_P_CALIBRATION_NOT_CONVERGED_REQUIRES_ATTENTION',summary_sha256=sha(path),
            post_main_MCMC_released=False,IR_control_done=False,finished_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT/'STATUS.json',state);print(json.dumps(state),flush=True)
    except Exception:
        state.update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc())
        save(OUT/'STATUS.json',state);raise


if __name__=='__main__':main()
