"""模块大纲：完整数值输入验收；仅425冗余DONE标记不读取、不伪造hash。

FoF、STO zip/payload、真实diagnostics、配方、质量/行身份/RSD仍全部验收。
保留v1记录并仅追加不存在的准入行；旧reader必须确认退出才允许追加。
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import audit_sto_post_fid500_routed_inputs as routed

original=routed.original
ROOT=routed.ROOT
OUT=ROOT/'fid0_499_raw_source_audit_routed_v2'
MARKER=original.SOURCE/'fiducial/real425/env_lam1_gsr2_thr15/DONE'
ADMISSION=ROOT/'fid0_499_raw_source_audit_routed_v1/rows'


def marker_reader(previous,path,**kwargs):
    # None明确表示没有读到/使用标记字节；绝不返回模拟的b'done'或空文件。
    return None if Path(path)==MARKER else previous(path,**kwargs)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core input audit required')
    try:os.kill(457848,0)
    except ProcessLookupError:pass
    else:raise ValueError('original v1 reader still exists; preserve it, do not append into its admission namespace')
    routed.verify_replay()
    if args.detach:
        launch=ROOT/(OUT.name+'.launch.json');log=ROOT/(OUT.name+'.log')
        if OUT.exists() or launch.exists() or log.exists():raise ValueError('prior v2 audit requires inspection')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(host=socket.gethostname(),pid=proc.pid,command=command,log=str(log),
            CPU_affinity=sorted(os.sched_getaffinity(0)),new_session=True,created_UTC=datetime.now(timezone.utc).isoformat(),
            prior_v1_process_confirmed_absent=457848)
        with launch.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(record),flush=True);return
    previous_read,previous_sha,previous_save,previous_out=original.read_blob,original.sha,original.save,routed.OUT
    entry_sha=previous_sha(previous_read(Path(__file__)))
    def read(path,**kwargs):return marker_reader(previous_read,path,**kwargs)
    def digest(blob):return None if blob is None else previous_sha(blob)
    def save(path,value):
        if path.name=='CONTRACT.json':value['immutable_sources'][str(Path(__file__))]=entry_sha
        if path.name in ('CONTRACT.json','AUDIT.json'):
            value.update(optional_completion_marker_not_consumed_realizations=[425],
                required_actual_STO_payload_diagnostics_FoF_mass_identity_and_RSD_checks_preserved=True,
                old_v1_read_wait_records_preserved=True,admission_rows_append_only_directory=str(ADMISSION),
                original_v1_process_confirmed_absent=457848)
        if path.parent.name=='rows':
            value.update(source_audit_generation_directory=str(OUT),source_audit_generation_entry_sha256=entry_sha)
            if value['realization']==425:
                value.update(DONE_sha256=None,completion_marker_original_path=str(MARKER),completion_marker_consumed=False,
                    completion_proof='full actual STO zip, complete real diagnostics, original FoF and exact row identity; not a marker string')
        result=previous_save(path,value)
        if path.parent.name=='rows' and value['status']=='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':
            destination=ADMISSION/path.name
            if destination.exists():
                old=json.loads(previous_read(destination))
                for key in ('source_sha256','diagnostics_sha256','N2_data_self','N3_data_self'):
                    if old[key]!=value[key]:raise ValueError('old admission row numerical source changed: '+key)
                if old['raw_fof']['source_raw_sha256']!=value['raw_fof']['source_raw_sha256']:
                    raise ValueError('old admission FoF source changed')
            else:
                # 已通过全部数值验收的新行；O_EXCL防止覆盖仍被measurement引用的旧行。
                with destination.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        return result
    original.read_blob,original.sha,original.save,routed.OUT=read,digest,save,OUT
    sys.argv=[routed.__file__]
    try:routed.main()
    finally:original.read_blob,original.sha,original.save,routed.OUT=previous_read,previous_sha,previous_save,previous_out


if __name__=='__main__':main()
