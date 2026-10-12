"""模块大纲：login新队列实际终止后自动聚合700/C500，再做原physical-p MCMC。

只等确切parent PID+starttime；不将状态文件/观察超时当终止，不重启测量。
CPU7单核；原数据、priors、sampler与收敛门不改；不替代39主链或IR验收。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from continue_sto_post_physical_p import ensemble_gate
from run_sto_post_target_queue import PART,ROOT,save,sha

OUT=ROOT/'physical_p_continuation_login_v2'
QUEUE=ROOT/'login_measurement_queue_v2'


def queue_is_live(record):
    path=Path('/proc')/str(record['pid'])
    try:actual=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return False
    if actual!=record['starttime_ticks']:return False
    command=(path/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    if str(PART/'code/run_sto_post_login_queue_v2.py') not in command:raise ValueError('measurement PID command identity differs')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={7}:raise ValueError('original login26 CPU7 required')
    if args.detach:
        log=ROOT/'physical_p_continuation_login_v2.log';launch=ROOT/'physical_p_continuation_login_v2.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise RuntimeError('prior continuation must be inspected, not overwritten')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,CPU_affinity=[7],command=command,
            launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log),detached=True)
        with launch.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(value),flush=True);return
    launch_path=ROOT/'login_measurement_queue_v2.launch.json';launch=json.loads(launch_path.read_text())
    sources=[Path(__file__),PART/'code/run_sto_post_login_queue_v2.py',PART/'code/assemble_sto_post_measurements.py',
        PART/'code/continue_sto_post_physical_p.py',PART/'code/run_sto_physical_p_mcmc.py',
        PART/'code/post_rsd_p_calibration.py',PART/'code/post_rsd_p0_model.py',PART/'code/run_rsd_b000_mcmc.py',
        ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json',ROOT/'ENSEMBLE_SELECTION.json',launch_path,
        ROOT/'login_resumption_20261009/MIGRATION.json',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    frozen={str(path):sha(path) for path in sources};OUT.mkdir()
    save(OUT/'CONTRACT.json',dict(role='exact login measurement -> assembly -> necessary p calibration',
        immutable_sources=frozen,queue_PID=launch['pid'],CPU_affinity=[7],numeric_threads=1,
        main_MCMC_required=39,full_goal_reduced=False,IR_still_required=True))
    state=dict(state='WAITING_FOR_SPECIFIC_LOGIN_MEASUREMENT_PROCESS',pid=os.getpid(),host=socket.gethostname(),
        measurement_pid=launch['pid'],child_pid=None,started_UTC=datetime.now(timezone.utc).isoformat())
    def update(**values):
        state.update(values,updated_UTC=datetime.now(timezone.utc).isoformat());save(OUT/'STATUS.json',state)
    update()
    try:
        while True:
            for path,want in frozen.items():
                if sha(path)!=want:raise ValueError('frozen continuation dependency drift: '+path)
            status_path=QUEUE/'STATUS.json'
            if not status_path.exists():
                if not (Path('/proc')/str(launch['pid'])).exists():raise ValueError('queue exited before initial status; inspect launch log')
                update(queue_observation='initializing; no terminal proof');time.sleep(30);continue
            record=json.loads(status_path.read_text())
            if record['pid']!=launch['pid'] or record['host']!='login26':raise ValueError('exact queue identity differs')
            live=queue_is_live(record)
            update(actual_queue_state=record['state'],measurement_process_live=live,completed_measurements=record['completed_count'])
            if record['state']=='COMPLETE' and record['completed_count']==700 and not record['workers'] and not live:break
            if not live:raise ValueError('measurement process ended without complete700; preserve outputs, no retry')
            time.sleep(60)
        env=dict(os.environ)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        commands=[('ASSEMBLING_ALL700_C500',PART/'code/assemble_sto_post_measurements.py',[]),
            ('FREEZING_PHYSICAL_P_INPUTS',PART/'code/run_sto_physical_p_mcmc.py',['freeze']),
            ('PHYSICAL_P_CALIBRATION_MCMC',PART/'code/run_sto_physical_p_mcmc.py',['sample'])]
        for stage,entry,arguments in commands:
            if stage!='ASSEMBLING_ALL700_C500':
                report=json.loads((ROOT/'ensemble_measurements/AUDIT.json').read_text());ensemble_gate(report)
                for info in report['ensemble_files'].values():
                    if sha(info['path'])!=info['sha256']:raise ValueError('completed ensemble drift')
            update(state=stage,child_pid=None)
            proc=subprocess.Popen([sys.executable,'-B','-u',str(entry)]+arguments,cwd=PART,env=env)
            update(child_pid=proc.pid)
            if proc.wait():raise ValueError('continuation stage failed; no automatic retry: '+stage)
            update(child_pid=None)
        summary=ROOT/'physical_p_calibration_mcmc/cases/physical_p_GSR2_RSD_fNLfixed100/SUMMARY.json'
        value=json.loads(summary.read_text())
        update(state='PHYSICAL_P_CALIBRATION_CONVERGED_NOT_MAIN_MCMC_RELEASE' if value['converged']
            else 'PHYSICAL_P_CALIBRATION_NOT_CONVERGED_REQUIRES_ATTENTION',summary_sha256=sha(summary),
            main_MCMC_released=False,IR_completed=False,finished_UTC=datetime.now(timezone.utc).isoformat())
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise


if __name__=='__main__':main()
