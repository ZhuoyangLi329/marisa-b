"""模块大纲：用户改为全部样本重校准p后，终止旧150校准值的P0自动推断。

只针对已知controller的精确PID/starttime/入口，先停派发再中断其当前child。
700测量、C500组装和全部旧链文件保留；不读取活动HDF、不删除数据。
"""
import json
import os
from pathlib import Path
import signal
import socket
import time
from rsd_b000_contract import PART,save_json
from datetime import datetime,timezone

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=PART/'outputs/rsd_all_available_p_recalibration_20261010_v1'


def actual(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    suffix=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=suffix[0],starttime_ticks=suffix[19],
        cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def main():
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:
        raise ValueError('original host and one controller CPU required')
    receipt=OUT/'SUPERSEDED_P150_P0_STOP.json'
    if receipt.exists():raise ValueError('existing stop receipt must be inspected, not repeated')
    pid=1304572;old=actual(pid);actions=[]
    if old:
        if (old['starttime_ticks']!='144870289'
                or str(PART/'code/continue_sto_post_p0only.py') not in old['cmdline']):
            raise ValueError('old controller PID identity mismatch')
        os.kill(pid,signal.SIGSTOP);actions.append(dict(pid=pid,signal='SIGSTOP'))
        state=json.loads((ROOT/'p0only_continuation_v1/STATUS.json').read_text())
        child=actual(state['child_pid']) if state.get('child_pid') else None
        if child:
            if str(PART/'code/run_sto_post_p0only.py') not in child['cmdline']:
                raise ValueError('unexpected child identity; preserve stopped controller for inspection')
            os.kill(child['pid'],signal.SIGINT);actions.append(dict(pid=child['pid'],signal='SIGINT',identity=child))
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                now=actual(child['pid'])
                if now is None or now['state'] in ('Z','X'):break
                time.sleep(.2)
            else:raise ValueError('actual child still live; do not delete/restart')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        actions.append(dict(pid=pid,signal='SIGTERM_THEN_SIGCONT'))
        deadline=time.monotonic()+10
        while actual(pid) is not None and time.monotonic()<deadline:time.sleep(.1)
        if actual(pid) is not None:raise ValueError('actual controller has not exited')
    result=dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        status='OLD150_P0_EXECUTION_SUPERSEDED_AT_USER_DIRECTION_PRODUCTS_PRESERVED',
        direct_user_request='所有p重新校准以后再给我结果',actions=actions,
        controller_terminal=True,old_products_preserved=True,HDF_read=False,
        original700_and_C500_assembly_not_modified=True)
    OUT.mkdir(exist_ok=True);save_json(receipt,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
