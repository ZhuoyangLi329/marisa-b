"""模块大纲：保留v4质量/行身份/RSD审计；追加已证实同payload的metadata路由。

默认仅real293；每个metadata副本须原producer/现有DM重放，STO整文件SHA
与原缓存相同。不替换原weights，原real127/246历史证据和等待记录均保留。
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import audit_sto_post_fid500_repaired_raw as v4

original=v4.original
ROOT=v4.ROOT
OUT=ROOT/'fid0_499_raw_source_audit_routed_v1'


def verify_replay():
    directory=ROOT/'input_recipe_replay_fid293_v2'
    report=json.loads(original.read_blob(directory/'AUDIT.json'))
    if (report['status']!='EXISTING_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM'
        or report['realization']!=293 or report['cosmology']!='fiducial'
        or not report['all_arrays_exactly_equal'] or not all(report['all_arrays_exactly_equal'].values())):
        raise ValueError('same-payload existing-DM replay not verified')
    contract_path=directory/'CONTRACT.json';contract=json.loads(original.read_blob(contract_path))
    if (original.sha(original.read_blob(contract_path))!=report['contract_sha256']
        or contract['realization']!=293 or contract['cosmology']!='fiducial'):
        raise ValueError('replay contract drift')
    for path,want in contract['immutable_sources'].items():
        if original.sha(original.read_blob(path))!=want:raise ValueError('replay source drift: '+path)
    old=original.SOURCE/'fiducial/real293/env_lam1_gsr2_thr15'
    fresh=Path(report['fresh_diagnostics_path']);expected=report['original_payload_sha256']
    if (expected!=report['regenerated_payload_sha256']
        or original.sha(original.read_blob(old/'post_sto_halo.npz'))!=expected
        or original.sha(original.read_blob(fresh.parent/'post_sto_halo.npz'))!=expected
        or original.sha(original.read_blob(fresh))!=report['fresh_diagnostics_sha256']):
        raise ValueError('original/replayed payload or metadata drift')
    diag=json.loads(original.read_blob(fresh))
    if (any(diag['config'].get(k)!=v for k,v in original.RECIPE.items())
        or diag['component_config']['method']!='nenv' or diag['component_config']['base_weight']!='number'):
        raise ValueError('producer recipe mismatch')
    return old/'diagnostics.json',fresh,directory,report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core audit required')
    old,fresh,replay,report=verify_replay()
    if args.detach:
        launch=ROOT/(OUT.name+'.launch.json');log=ROOT/(OUT.name+'.log')
        if OUT.exists() or launch.exists() or log.exists():raise ValueError('old execution requires inspection')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,command=command,CPU_affinity=sorted(os.sched_getaffinity(0)),
            new_session=True,log=str(log),created_UTC=datetime.now(timezone.utc).isoformat())
        with launch.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(value),flush=True);return
    previous_read,previous_save,previous_out=original.read_blob,original.save,v4.OUT
    frozen={str(p):original.sha(previous_read(p)) for p in (Path(__file__),replay/'AUDIT.json',replay/'CONTRACT.json')}
    def read(path,**kwargs):return previous_read(fresh if Path(path)==old else path,**kwargs)
    def save(path,value):
        if path.name=='CONTRACT.json':value['immutable_sources'].update(frozen)
        if path.name in ('CONTRACT.json','AUDIT.json'):
            value.update(metadata_replay_realizations=[127,293],original_STO_weights_unchanged=True,
                same_recipe_STO_audit_copies_generated=True,new_Nbody_or_DM_generated=False,
                metadata_replay_audit_for_real293=str(replay/'AUDIT.json'),old_v4_read_wait_artifacts_preserved=True)
        if path.parent.name=='rows' and path.name=='fiducial_real293.json':
            value.update(diagnostics_original_path=str(old),diagnostics_original_sha256=None,
                diagnostics_consumed_path=str(fresh),metadata_replay_audit_sha256=frozen[str(replay/'AUDIT.json')],
                diagnostics_provenance='original producer and existing DM; all arrays and entire STO SHA identical')
        return previous_save(path,value)
    original.read_blob,original.save,v4.OUT=read,save,OUT
    sys.argv=[v4.__file__]
    try:v4.main()
    finally:original.read_blob,original.save,v4.OUT=previous_read,previous_save,previous_out


if __name__=='__main__':main()
