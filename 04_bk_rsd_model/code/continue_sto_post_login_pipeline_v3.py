"""模块大纲：重用v2严格PID/700/C500/p校准衔接，仅绑定已恢复环境的v3队列。"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import continue_sto_post_login_pipeline_v2 as pipeline

PART,ROOT=pipeline.PART,pipeline.ROOT
OUT=ROOT/'physical_p_continuation_login_v3'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={7}:raise ValueError('original login26 CPU7 required')
    prior=ROOT/'physical_p_continuation_login_v2/STATUS.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if old['child_pid'] is not None or (Path('/proc')/str(old['pid'])).exists():
            raise ValueError('v2 continuation still owns process/child')
    if args.detach:
        log=ROOT/'physical_p_continuation_login_v3.log';launch=ROOT/'physical_p_continuation_login_v3.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise RuntimeError('prior v3 control must be inspected')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(host=socket.gethostname(),pid=proc.pid,command=command,CPU_affinity=[7],
            launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log),detached=True)
        with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    original_save=pipeline.save
    def save(path,value):
        if Path(path)==OUT/'CONTRACT.json':
            old=str(ROOT/'login_measurement_queue_v2.launch.json');actual=str(ROOT/'login_measurement_queue_v3.launch.json')
            if old in value['immutable_sources']:
                value['immutable_sources'][actual]=value['immutable_sources'].pop(old)
            value['queue_launch_read_route']={old:actual}
            for name in ('continue_sto_post_login_pipeline_v2.py','run_sto_post_login_queue_v3.py'):
                source=PART/'code'/name;value['immutable_sources'][str(source)]=pipeline.sha(source)
        return original_save(path,value)
    # v2逻辑内的固定root文件名通过Path重定向至真实v3 launch；不伪造原记录。
    original_read=Path.read_text
    def read(path,*args,**kwargs):
        if path==ROOT/'login_measurement_queue_v2.launch.json':path=ROOT/'login_measurement_queue_v3.launch.json'
        return original_read(path,*args,**kwargs)
    original_sha=pipeline.sha
    def sha(path):
        if Path(path)==ROOT/'login_measurement_queue_v2.launch.json':path=ROOT/'login_measurement_queue_v3.launch.json'
        return original_sha(path)
    def live(record):
        p=Path('/proc')/str(record['pid'])
        try:start=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:return False
        if start!=record['starttime_ticks']:return False
        if str(PART/'code/run_sto_post_login_queue_v3.py') not in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode():
            raise ValueError('v3 PID identity differs')
        return True
    pipeline.OUT=OUT;pipeline.QUEUE=ROOT/'login_measurement_queue_v3';pipeline.__file__=str(Path(__file__).resolve())
    pipeline.save=save;pipeline.sha=sha;pipeline.queue_is_live=live;Path.read_text=read
    sys.argv=[str(Path(__file__).resolve())]
    try:pipeline.main()
    finally:Path.read_text=original_read


if __name__=='__main__':main()
