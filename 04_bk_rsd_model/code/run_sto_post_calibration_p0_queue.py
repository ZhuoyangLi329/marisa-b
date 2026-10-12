"""模块大纲：独立校准 P0 的三单核小批次队列；保存 provenance；不做推断。

每个 worker 最多5盒后退出，控制 JAX 常驻内存；parent 与第一个 worker 共用
一个逻辑核。全队列最多三个 CPU，另一个 source 队列可用第四核，总数小于8。
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from rsd_b000_contract import PART,digest

OUT=PART/'outputs/sto_marisa_post_rsd_mean100_v1/calibration_holdout_p0'


def save(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cpus',default='7,8,9');args=ap.parse_args()
    cpus=tuple(int(q) for q in args.cpus.split(','))
    if len(cpus)!=3 or len(set(cpus))!=3 or not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('requires exactly three permitted logical CPUs')
    os.sched_setaffinity(0,{cpus[0]})
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise RuntimeError('thread budget drift')
    for name in ('P0_WINDOW_AUDIT.json','POST_P0_WINDOW_AUDIT.json'):
        if json.loads((OUT/name).read_text())['status']!='PASS':raise RuntimeError('actual P window failed')
    contract=json.loads((OUT/'CONTRACT.json').read_text());contract_sha=digest(OUT/'CONTRACT.json')
    controller_sha=digest(Path(__file__))
    work=OUT/'queue';work.mkdir(exist_ok=True)
    status=work/'STATUS.json'
    if status.exists():raise RuntimeError('queue already has status; inspect original execution before recovery')
    state={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'CPU_pool':cpus,
           'parent_affinity':sorted(os.sched_getaffinity(0)),'max_workers':3,'max_boxes_per_worker':5,
           'started_UTC':datetime.now(timezone.utc).isoformat(),'controller_sha256':controller_sha,
           'measurement_contract_sha256':contract_sha,'physical_p_fit_or_MCMC_run':False}
    active={};pending=list(range(100,250,5));start=time.time();error=None
    def guard():
        if digest(OUT/'CONTRACT.json')!=contract_sha or digest(Path(__file__))!=controller_sha:raise RuntimeError('queue contract/source drift')
        for path,sha in contract['immutable_sources'].items():
            if digest(path)!=sha:raise RuntimeError('measurement source drift: '+path)
    def update():
        completed=[]
        for index in range(100,250):
            path=OUT/'rows'/f'LC_p_real{index:03d}'/'DONE.json'
            if not path.exists():continue
            done=json.loads(path.read_text())
            if done['contract_sha256']!=contract_sha or done['measurement_sha256']!=digest(path.parent/'P0.npz'):raise RuntimeError('holdout output drift')
            completed.append(index)
        state.update(completed_count=len(completed),completed_realizations=completed,
                     workers=[{k:v for k,v in value.items() if k not in ('process','stream')} for value in active.values()],
                     remaining_chunks=pending,elapsed_seconds=time.time()-start,updated_UTC=datetime.now(timezone.utc).isoformat())
        save(status,state)
    try:
        guard();update()
        while pending or active:
            guard()
            for cpu in cpus:
                if cpu in active or not pending or error:continue
                first=pending.pop(0);last=min(first+5,250);log=work/f'LC_p_{first:03d}_{last-1:03d}.log'
                if log.exists():raise RuntimeError('old execution log exists; no overwrite')
                stream=log.open('x')
                command=[sys.executable,'-B','-u',str(PART/'code/measure_sto_post_calibration_p0.py'),
                         '--start',str(first),'--stop',str(last)]
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                         preexec_fn=lambda c=cpu:os.sched_setaffinity(0,{c}))
                active[cpu]={'cpu':cpu,'pid':process.pid,'start':first,'stop':last,'log':str(log),'process':process,'stream':stream}
            time.sleep(2.)
            for cpu,value in list(active.items()):
                code=value['process'].poll()
                if code is None:continue
                value['stream'].close();del active[cpu]
                if code:error=f'worker {value["start"]}:{value["stop"]} exited {code}; preserve {value["log"]}'
            update()
            if error and not active:raise RuntimeError(error)
            if not active and not pending:break
        if state['completed_count']!=150:raise RuntimeError('holdout count not 150')
        state['state']='COMPLETE';update();print(json.dumps(state,indent=2),flush=True)
    except Exception:
        # 不杀死已在测量的盒；只停止发新任务，保留各 worker 的 checkpoint/log。
        state.update(state='FAILED',error=traceback.format_exc());update();raise


if __name__=='__main__':main()
