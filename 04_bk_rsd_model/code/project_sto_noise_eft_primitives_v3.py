"""模块大纲：保留v2的fixture绑定失败；先载入旧测试，再显式路由三种零腿。

v2未进入任何真实投影：旧smoke测试在science路由后import了新oracle，却仍
只mock旧模块的谱，造成两个不同谱相比较。本版先绑定原测试函数，保留该
测试原义；另有两个v2测试完整覆盖新三种零腿。原数值门/公式/旧源码不改。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
# 必须在v2科学路由之前绑定旧smoke；不是跳过或更改测试/误差门。
import test_sto_noise_eft_projection_v1
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import project_sto_noise_eft_primitives_v2 as previous

shared=previous.shared;ROOT=previous.ROOT;OUT=ROOT/'noise_eft_primitives_v3'


def pipeline():
    launch=shared.read(ROOT/'noise_eft_primitives_v2.launch.json')
    if Path('/proc/'+str(launch['pid'])).exists():raise ValueError('v2 process must actually be terminal')
    log=ROOT/'noise_eft_primitives_v2.log'
    if 'independent small-lattice projection test required' not in log.read_text():
        raise ValueError('inspect unexpected v2 failure before this specific recovery')
    extra={str(p):shared.digest(p) for p in (Path(__file__),log,ROOT/'noise_eft_primitives_v2.launch.json')}
    original=(previous.OUT,shared.save_json);save_original=shared.save_json
    def save(path,value):
        for p,want in extra.items():
            if shared.digest(p)!=want:raise ValueError('v3 fixture/source provenance changed')
        value=dict(value)
        if Path(path).name=='CONTRACT.json':
            value['immutable_inputs'].update(extra)
            value['v3_fixture_import_order']='original smoke bound before all-zero-leg science route; new2 zero tests also run'
        save_original(path,value)
    previous.OUT=OUT;shared.save_json=save
    try:previous.pipeline()
    finally:previous.OUT,shared.save_json=original


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('login26 CPU14 single core required')
    if args.detach:
        log=ROOT/'noise_eft_primitives_v3.log';launch=ROOT/'noise_eft_primitives_v3.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect exact original v3 before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        shared.save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],
            created_UTC=shared.stamp(),log=str(log)))
        print(json.dumps(shared.read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
