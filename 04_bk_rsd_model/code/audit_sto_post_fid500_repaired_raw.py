"""模块大纲：完整Fid500原来源核验；仅246原FoF换为已验收同源恢复副本。

复用v3原reader与real127同payload配方证据；原权重和全部质量/身份门不改。
原v3读取等待仍保留，不覆盖旧日志或声称它已经终止。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import audit_sto_post_fid500_replayed_metadata as v3

original = v3.original
ROOT = v3.ROOT
OUT = ROOT/'fid0_499_raw_source_audit_v4'
REPAIR = ROOT/'input_repair_fid246'


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise ValueError('single-core input audit required')
    report = json.loads(original.read_blob(REPAIR/'AUDIT.json'))
    if report['status'] != 'SAME_REAL246_FOF_FORMAT_RECIPE_AND_STO_ROW_IDENTITY_PASS':
        raise ValueError('restored file identity audit missing')
    old = Path(report['original_raw_path']); fresh = Path(report['consumed_raw_path'])
    if original.sha(original.read_blob(fresh)) != report['raw_sha256']:
        raise ValueError('restored raw file drift')
    if args.detach:
        launch = ROOT/'fid0_499_raw_source_audit_v4.launch.json';log=ROOT/'fid0_499_raw_source_audit_v4.log'
        if OUT.exists() or launch.exists() or log.exists(): raise ValueError('old v4 execution requires inspection')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,CPU_affinity=sorted(os.sched_getaffinity(0)),
            command=command,log=str(log),new_session=True,created_UTC=datetime.now(timezone.utc).isoformat())
        with launch.open('x') as f: json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(value),flush=True);return
    previous_read,previous_save,previous_out=original.read_blob,original.save,v3.OUT
    frozen={str(p):original.sha(previous_read(p)) for p in (Path(__file__),REPAIR/'AUDIT.json')}
    def read(path,**kwargs):
        return previous_read(fresh if Path(path)==old else path,**kwargs)
    def save(path,value):
        if path.name == 'CONTRACT.json': value['immutable_sources'].update(frozen)
        if path.name in ('CONTRACT.json','AUDIT.json'):
            value.update(raw_repair_realization=246,original_raw_path_for_real246=str(old),
                consumed_raw_path_for_real246=str(fresh),raw_repair_audit_sha256=frozen[str(REPAIR/'AUDIT.json')],
                original_raw_FoF_not_overwritten=True,original_STO_weights_unchanged=True,
                original_v3_read_wait_artifacts_preserved=True)
        if path.parent.name=='rows' and path.name=='fiducial_real246.json':
            value.update(raw_original_path=str(old),raw_consumed_path=str(fresh),
                raw_repair_audit_sha256=frozen[str(REPAIR/'AUDIT.json')])
            value['raw_fof']['source_raw']=str(fresh)
            value['raw_fof']['original_source_raw']=str(old)
        return previous_save(path,value)
    original.read_blob,original.save,v3.OUT=read,save,OUT
    sys.argv=[v3.__file__]
    try: v3.main()
    finally: original.read_blob,original.save,v3.OUT=previous_read,previous_save,previous_out


if __name__=='__main__': main()
