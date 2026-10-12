"""模块大纲：用户要求恢复login测量；只取消本任务未运行job、停止旧等待控制器。

先hold并复核PENDING/无节点/零runtime，避免调度竞态；不取消理论模板job。
校准child必须为空；核对cmdline/starttime后只停止旧等待parent，保留原日志。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'login_resumption_20261009'
JOB=59592166


def job_info():
    raw=subprocess.check_output(['scontrol','show','job','-o',str(JOB)],text=True)
    return raw,dict(x.split('=',1) for x in raw.split() if '=' in x)


def main():
    if socket.gethostname()!='login26':raise ValueError('migration must verify original login26 controller')
    if OUT.exists():raise RuntimeError('preserve prior migration; inspect before retry')
    raw,job=job_info()
    if (job.get('JobId')!=str(JOB) or job.get('JobName')!='sm_post_rsd_cpu32'
        or job.get('WorkDir')!=str(PART) or job.get('JobState')!='PENDING'
        or job.get('RunTime')!='00:00:00' or job.get('NodeList') not in ('','(null)','None')):
        raise ValueError('only exact owned unrun measurement job can be cancelled')
    # 防止从read-only核验到cancel之间被调度；若hold后已运行就停止迁移。
    subprocess.run(['scontrol','hold',str(JOB)],check=True)
    held_raw,held=job_info()
    if held.get('JobState')!='PENDING' or held.get('RunTime')!='00:00:00' or held.get('NodeList') not in ('','(null)','None'):
        raise ValueError('job became active; do not cancel workers')
    state_path=ROOT/'physical_p_continuation/STATUS.json';state=json.loads(state_path.read_text())
    if state['pid']!=728654 or state['host']!='login26' or state['child_pid'] is not None:
        raise ValueError('original calibration controller/child identity differs')
    p=Path('/proc/728654');identity=None
    if p.exists():
        command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode().strip()
        want=str(PART/'code/continue_sto_post_physical_p.py')+' --measurement-job-id '+str(JOB)
        if not command.endswith(want):raise ValueError('original PID was reused or is not waiting controller')
        identity=dict(pid=728654,command=command,starttime_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])
    OUT.mkdir();record=dict(state='VERIFIED_PENDING_JOB_HELD',created_UTC=datetime.now(timezone.utc).isoformat(),
        user_override='resume original post measurements on login nodes, <=8 logical CPU; same scientific contract',
        job_id=JOB,before_scontrol=raw,held_scontrol=held_raw,old_controller_identity=identity,
        old_controller_status=state,theory_job59594709_unchanged=True)
    def save():
        path=OUT/'MIGRATION.json';tmp=path.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    save();subprocess.run(['scancel',str(JOB)],check=True)
    terminal=None
    for _ in range(30):
        raw_account=subprocess.check_output(['sacct','-X','-n','-P','-j',str(JOB),'-o','JobIDRaw,State,ExitCode,ElapsedRaw'],text=True)
        found=[s.split('|') for s in raw_account.splitlines() if s.split('|')[0]==str(JOB)]
        if len(found)==1 and found[0][1].startswith('CANCELLED') and found[0][3]=='0':terminal=found[0];break
        time.sleep(1.)
    if terminal is None:raise ValueError('unrun cancellation not yet proven; no login dispatcher release')
    record['cancelled_job_accounting']=terminal;record['compute_job_cancelled_before_running']=True;save()
    if p.exists():
        now=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
        if identity is None or now!=identity['starttime_ticks']:raise ValueError('old controller PID reuse')
        latest=json.loads(state_path.read_text())
        if latest['child_pid'] is not None:raise ValueError('calibration child appeared; do not interrupt')
        os.kill(728654,signal.SIGTERM)
    for _ in range(30):
        if not p.exists():break
        time.sleep(.1)
    if p.exists():raise ValueError('old controller not confirmed absent; no replacement release')
    record.update(state='COMPUTE_JOB_CANCELLED_BEFORE_RUN_OLD_CALIBRATION_WAIT_PARENT_ABSENT',
        old_controller_confirmed_absent=True,updated_UTC=datetime.now(timezone.utc).isoformat())
    save();print(json.dumps(record),flush=True)


if __name__=='__main__':main()
