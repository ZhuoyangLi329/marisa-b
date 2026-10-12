"""模块大纲：修复摘要保存回调后接续全500 pre物理p校准。

v1包裹了未被唯一采样核心使用的save_json；此版包裹实际shared.save_json。
只替换尚未进入校准的等待进程；测量、post旧链及校准科学设置不改。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import json
import signal
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
def stamp():return calibration.stamp()

def corrected_summary(path,value):
    if Path(path).name!='SUMMARY.json':return value
    meta=read(Path(path).parent/'contract.json')
    return dict(value,stage='pre',calibration_target_count=meta['calibration_target_count'],
        all_available_positive_samples_used=True,calibration_main_target_overlap_ids=list(range(100)),
        strict_independent_calibration_claimed=False,
        gof_note='all-available calibration mean with C_single500; not calibrated GOF',
        actual_shared_summary_callback_fixed=True)

def run():
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],started_UTC=stamp(),
        state='WAITING_FOR_ACTUAL_ALL500_PRE_MEASUREMENTS_AND_POST_PCAL_EXIT',entry_sha256=digest(__file__),
        calibration_entry_sha256=digest(calibration.__file__),actual_summary_save_callback_fixed=True)
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'PRE_AUTOMATIC_CONTINUATION_STATUS.json',state)
    update()
    try:
        while True:
            measured=read(OUT/'PRE_MEASUREMENT_STATUS.json');post=read(OUT/'post_PCAL_STATUS.json')
            if measured['state']=='FAILED_OR_INTERRUPTED_PRESERVED' or post['state']=='FAILED_OR_INTERRUPTED_PRESERVED':
                raise ValueError('actual upstream failure; preserve it')
            ready=measured['state']=='ALL500_PRE_POSITIVE_P0_ROWS_READY'
            if not ready and owner_quiescent(measured):raise ValueError('measurement exited without complete500')
            if owner_quiescent(post) and post['state']!='ALL_AVAILABLE_PHYSICAL_P_TRIAL_FINISHED_REVIEW_CHAIN':
                raise ValueError('post pcal exited without complete result')
            update(measured_new_rows=measured['completed_new_rows'],post_calibration_state=post['state'])
            if ready and owner_quiescent(measured) and owner_quiescent(post):break
            time.sleep(10)
        if (OUT/'pre_PCAL_STATUS.json').exists() or calibration.stage_root('pre').exists():raise ValueError('pre calibration exists; no duplicate')
        if digest(calibration.__file__)!=state['calibration_entry_sha256']:raise ValueError('calibration implementation changed while waiting')
        original_save=calibration.original.shared.save_json;original_freeze=calibration.freeze
        def save(path,value):return original_save(path,corrected_summary(path,value))
        def freeze(stage):
            original_freeze(stage);path=calibration.stage_root(stage)/'contract.json';parent=read(path)
            parent['immutable_inputs'][str(Path(__file__).resolve())]=digest(__file__)
            parent['actual_summary_save_callback_fixed']=True;save_json(path,parent)
        calibration.original.shared.save_json=save;calibration.freeze=freeze
        try:
            update(state='RUNNING_ALL500_PRE_PHYSICAL_P_CALIBRATION')
            calibration.pipeline('pre')
        finally:calibration.original.shared.save_json=original_save;calibration.freeze=original_freeze
        result=read(OUT/'pre_PCAL_STATUS.json')
        update(state='ALL500_PRE_PHYSICAL_P_CALIBRATION_TRIAL_FINISHED_REVIEW_CHAIN',
            converged=result['converged'],calibration_target_count=result['calibration_target_count'])
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise

def replace_waiter_and_launch():
    launch=OUT/'PRE_AUTOMATIC_CONTINUATION_V2_LAUNCH.json';log=OUT/'pre_automatic_continuation_v2.log'
    if launch.exists() or log.exists():raise ValueError('existingv2 continuation must be inspected, not duplicated')
    old=read(OUT/'PRE_AUTOMATIC_CONTINUATION_STATUS.json')
    if old['state']!='WAITING_FOR_ACTUAL_ALL500_PRE_MEASUREMENTS_AND_POST_PCAL_EXIT':
        raise ValueError('old pre calibration already entered work; do not interrupt it')
    proc=Path('/proc')/str(old['pid']);command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'continue_all_available_pre_pcal_v1.py' not in command:raise ValueError('exact old wait-only process not found')
    if (OUT/'pre_PCAL_STATUS.json').exists():raise ValueError('pre calibration already launched')
    snapshot=dict(replaced_PID=old['pid'],old_state=old,old_cmdline=command,
        old_proc_stat=(proc/'stat').read_text(),created_UTC=stamp(),
        reason='fix actual summary save callback before all500pre calibration; no science or measurement change')
    os.kill(old['pid'],signal.SIGTERM)
    deadline=time.monotonic()+10
    while not owner_quiescent(old):
        if time.monotonic()>deadline:raise ValueError('old waiter did not exit; no new launch')
        time.sleep(.1)
    snapshot['old_waiter_actually_exited']=True;save_json(OUT/'PRE_CONTINUATION_V2_MIGRATION.json',snapshot)
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(launch,dict(pid=child.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp()))
    print(json.dumps(read(launch)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--replace-waiter-and-detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('one login26 CPU13 required')
    if args.replace_waiter_and_detach:replace_waiter_and_launch()
    else:run()
