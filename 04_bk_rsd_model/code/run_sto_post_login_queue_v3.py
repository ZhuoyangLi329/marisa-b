"""模块大纲：原NERSC环境恢复后重用v2dispatcher；保留v2启动失败和未改科学源码。

只将每个worker包回原source nersc_env步骤，随后全部数值线程重置1。
v2 dispatcher源码固定SHA并纳入新合同；独立v3输出不覆盖失败记录。
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
import run_sto_post_login_queue_v2 as queue

PART,ROOT=queue.PART,queue.ROOT
OUT=ROOT/'login_measurement_queue_v3'
V2_SHA='182621acd8e30821d9da3322948d9213fdb5d7c096c403cecd87030149fda75e'


def environment_wrapped_command(command):
    script=('set -eo pipefail\nsource ../01_reconstruction_methods/common/nersc_env.sh\n'
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
        'export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1\n'
        'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"\n'
        'exec "$@"')
    return ['bash','-c',script,'source-original-nersc-env']+command


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or not set(queue.CPUS+(queue.PARENT_CPU,))<=os.sched_getaffinity(0):
        raise ValueError('original login26 CPU pool required')
    v2_path=PART/'code/run_sto_post_login_queue_v2.py'
    if queue.shared.sha(v2_path)!=V2_SHA:raise ValueError('preserve exact v2 dispatcher before environment adapter')
    prior=json.loads((ROOT/'login_measurement_queue_v2/STATUS.json').read_text())
    if prior['state']!='FAILED' or prior['workers'] or (Path('/proc')/str(prior['pid'])).exists():
        raise ValueError('v2 process not proved failed with no active workers')
    if args.detach:
        log=ROOT/'login_measurement_queue_v3.log';launch=ROOT/'login_measurement_queue_v3.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise RuntimeError('prior v3 attempt must be inspected')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(host=socket.gethostname(),pid=process.pid,command=command,worker_CPUs=queue.CPUS,
            parent_CPU=queue.PARENT_CPU,launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log),detached=True)
        with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    original_command=queue.shared.worker_command;original_save=queue.shared.save
    def worker_command(key,row):
        if queue.shared.sha(v2_path)!=V2_SHA:raise ValueError('v2 dispatcher source drift')
        return environment_wrapped_command(original_command(key,row))
    def save(path,value):
        if Path(path)==OUT/'CONTRACT.json':
            value['immutable_sources'][str(v2_path)]=V2_SHA
            value['environment_loading']='original nersc_env then explicit single-thread CPU JAX; same numerical workers'
            value['prior_v2_failure_preserved']=True
        return original_save(path,value)
    queue.OUT=OUT;queue.__file__=str(Path(__file__).resolve())
    queue.shared.worker_command=worker_command;queue.shared.save=save
    sys.argv=[str(Path(__file__).resolve())]
    queue.main()


if __name__=='__main__':main()
