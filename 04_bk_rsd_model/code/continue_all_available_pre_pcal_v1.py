"""模块大纲：全部pre补测结束后自动接续物理p校准，不重复启动或改旧链。

等待实际测量完成与post校准owner退出；CPU13上复用同一校准入口。
只读状态和/proc，不打开活动HDF；失败保留并记录，不把失败当完成。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import socket
import subprocess
import sys
import time
import traceback
import run_all_available_rsd_pcal_v1 as calibration
from post_rsd_owner_quiescence import owner_quiescent
from rsd_b000_contract import digest,save_json

OUT=calibration.OUT

def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()

def run():
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=stamp(),state='WAITING_FOR_ACTUAL_ALL500_PRE_MEASUREMENTS_AND_POST_PCAL_EXIT',
        entry_sha256=digest(__file__),calibration_entry_sha256=digest(calibration.__file__))
    def update(**values):
        state.update(values,updated_UTC=stamp());save_json(OUT/'PRE_AUTOMATIC_CONTINUATION_STATUS.json',state)
    update()
    try:
        while True:
            measurement=read(OUT/'PRE_MEASUREMENT_STATUS.json')
            post=read(OUT/'post_PCAL_STATUS.json')
            if measurement['state']=='FAILED_OR_INTERRUPTED_PRESERVED' or post['state']=='FAILED_OR_INTERRUPTED_PRESERVED':
                raise ValueError('actual upstream failed; preserve it, do not start calibration')
            ready=measurement['state']=='ALL500_PRE_POSITIVE_P0_ROWS_READY'
            if not ready and owner_quiescent(measurement):
                raise ValueError('measurement owner exited without all500 completion')
            post_done=post['state']=='ALL_AVAILABLE_PHYSICAL_P_TRIAL_FINISHED_REVIEW_CHAIN'
            if not post_done and owner_quiescent(post):
                raise ValueError('post calibration owner exited without completed summary')
            update(measured_new_rows=measurement['completed_new_rows'],post_calibration_state=post['state'])
            if ready and owner_quiescent(measurement) and post_done and owner_quiescent(post):break
            time.sleep(10)
        if digest(calibration.__file__)!=state['calibration_entry_sha256']:
            raise ValueError('calibration implementation changed while waiting')
        if (OUT/'pre_PCAL_STATUS.json').exists() or calibration.stage_root('pre').exists():
            raise ValueError('an actual pre calibration already exists; no duplicate execution')
        update(state='RUNNING_ALL500_PRE_PHYSICAL_P_CALIBRATION')
        calibration.pipeline('pre')
        result=read(OUT/'pre_PCAL_STATUS.json')
        update(state='ALL500_PRE_PHYSICAL_P_CALIBRATION_TRIAL_FINISHED_REVIEW_CHAIN',
            converged=result['converged'],calibration_target_count=result['calibration_target_count'])
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:
        raise ValueError('one login26 CPU13 required')
    if args.detach:
        launch=OUT/'PRE_AUTOMATIC_CONTINUATION_LAUNCH.json';log=OUT/'pre_automatic_continuation.log'
        if launch.exists() or log.exists():raise ValueError('inspect actual previous continuation; do not duplicate')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp()))
        print(json.dumps(read(launch)),flush=True)
    else:run()
