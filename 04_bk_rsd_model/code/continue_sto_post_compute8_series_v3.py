"""模块大纲：v3独立会话，原协调器及v2 STOP/信号保护，数值/统计不改。"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from sto_post_compute8_v3_adapters import configure,OUT,verify_job8,base


def main():
    configure();ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first-job',type=int,required=True);ap.add_argument('--detach',action='store_true')
    args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:
        raise ValueError('original login26 single CPU12 required')
    if args.detach:
        log=base.common.ROOT/'compute8_measurement_series_v3.log';launch=base.common.ROOT/'compute8_measurement_series_v3.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect existing v3, do not duplicate')
        verify_job8(args.first_job)
        cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--first-job',str(args.first_job)]
        with log.open('xb') as stream:
            p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=p.pid,host=socket.gethostname(),command=cmd,CPU_affinity=[12],first_job=args.first_job,
            launched_UTC=datetime.now(timezone.utc).isoformat(),log=str(log))
        with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    sys.argv=[__file__,'--first-job',str(args.first_job)]
    base.bind_coordinator().main()


if __name__=='__main__':main()
