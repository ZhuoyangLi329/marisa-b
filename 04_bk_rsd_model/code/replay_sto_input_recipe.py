"""模块大纲：已有DM/原producer重现指定STO配方；只接受逐数组与文件SHA相同。

用于恢复不可读metadata的证据，不替换原weights、不下载DM、不修改旧输入。
数值流程只调用原producer；输出遵守其第一部分outputs约束。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
import replay_sto_fid127_input_recipe as common
STATUS_PATH=None


def main():
    global STATUS_PATH
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--realization',type=int,required=True);ap.add_argument('--detach',action='store_true')
    ap.add_argument('--attempt',type=int,default=1)
    args=ap.parse_args();index=args.realization
    if len(os.sched_getaffinity(0))!=1 or not 0<=index<500:raise ValueError('single-core existing Fid500 only')
    common.check_code()
    if args.attempt<1:raise ValueError('positive preserved attempt number required')
    suffix='' if args.attempt==1 else f'_v{args.attempt}'
    out=common.ROOT/f'input_recipe_replay_fid{index:03d}{suffix}'
    if args.detach:
        launch=out.with_name(out.name+'.launch.json');log=out.with_name(out.name+'.log')
        if out.exists() or launch.exists() or log.exists():raise ValueError('old execution requires inspection')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--realization',str(index),'--attempt',str(args.attempt)]
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(host=socket.gethostname(),pid=proc.pid,CPU_affinity=sorted(os.sched_getaffinity(0)),
            command=command,log=str(log),created_UTC=datetime.now(timezone.utc).isoformat(),new_session=True)
        with launch.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(value),flush=True);return
    out.mkdir(exist_ok=False)
    STATUS_PATH=out/'STATUS.json'
    source=common.SOURCE/'fiducial'/f'real{index:03d}/env_lam1_gsr2_thr15'
    payload=common.read_blob(source/'post_sto_halo.npz');expected=common.sha(payload)
    raw_path=common.adapter.original.raw_path('fiducial',index)
    raw=common.read_blob(raw_path)
    _,positions,raw_meta=common.catalog_from_blobs('fiducial',index,raw,raw_source=raw_path)
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:old={k:np.array(z[k]) for k in z.files}
    identity=common.validate_periodic_identity(old['positions_mpc_h'],positions)
    staged=out/'original_fof_selected_positions/fiducial'/f'real{index:03d}'
    staged.mkdir(parents=True);np.save(staged/'positions_mpc_h.npy',positions)
    # 原 producer 自己读取原DM快照；不猜测环境标签，不复制其数学实现。
    destination=common.PRODUCED_ROOT
    command=[sys.executable,'-B','-u',str(common.PRODUCER),'--cosmology','fiducial','--realization',str(index),
        '--halo-root',str(staged.parents[1]),'--dm-root',str(common.DM),'--method','nenv',
        '--base-weight','number','--env-source','dm','--weight-source','dm','--env-mesh','128',
        '--recon-mesh','128','--gsr','2','--thr','15','--lambda-th','1','--dtype','f4',
        '--result-root',str(destination),'--label','env_lam1_gsr2_thr15']
    paths=(Path(__file__),Path(common.__file__),common.PRODUCER,
        common.PART/'code/prepare_sto_post_validation_sources.py',common.PART/'code/post_rsd_blob_catalog.py')
    frozen={str(p):common.sha(common.read_blob(p)) for p in paths}
    contract=dict(cosmology='fiducial',realization=index,recipe=common.RECIPE,command=command,
        original_STO_path=str(source/'post_sto_halo.npz'),original_STO_payload_sha256=expected,
        original_STO_position_identity=identity,raw_FoF=raw_meta,immutable_sources=frozen,
        old_diagnostics_read_here=False,original_inputs_modified=False,new_DM_downloaded_or_Nbody_generated=False)
    common.save(out/'CONTRACT.json',contract)
    state=dict(state='RUNNING',host=socket.gethostname(),pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        stage='same_recipe_source_replay',started_UTC=datetime.now(timezone.utc).isoformat())
    common.save(out/'STATUS.json',state)
    with (out/'PRODUCER.log').open('x') as stream:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        state['child_pid']=proc.pid;common.save(out/'STATUS.json',state);code=proc.wait()
    if code:raise RuntimeError('original producer failed; preserve partial outputs and log')
    common.check_code()
    for path,want in frozen.items():
        if common.sha(common.read_blob(path))!=want:raise ValueError('source drift during replay')
    fresh=destination/'fiducial'/f'real{index:03d}/env_lam1_gsr2_thr15'
    generated=common.read_blob(fresh/'post_sto_halo.npz');got=common.sha(generated)
    diagnostics=common.read_blob(fresh/'diagnostics.json');diag=json.loads(diagnostics)
    if (any(diag['config'].get(k)!=v for k,v in common.RECIPE.items())
        or diag['component_config']['method']!='nenv' or diag['component_config']['base_weight']!='number'):
        raise ValueError('replay recipe drift')
    with np.load(io.BytesIO(generated),allow_pickle=False) as z:
        equality={k:bool(np.array_equal(old[k],z[k])) for k in old} if set(z.files)==set(old) else {}
    if not equality or not all(equality.values()) or got!=expected:
        common.save(out/'PAYLOAD_MISMATCH.json',dict(arrays=equality,old_sha256=expected,new_sha256=got))
        raise ValueError('different STO payload; must not use replay metadata')
    if common.sha(common.read_blob(source/'post_sto_halo.npz'))!=expected:raise ValueError('original payload drift')
    report=dict(status='EXISTING_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM',
        realization=index,cosmology='fiducial',created_UTC=datetime.now(timezone.utc).isoformat(),
        all_arrays_exactly_equal=equality,original_payload_sha256=expected,regenerated_payload_sha256=got,
        fresh_diagnostics_path=str(fresh/'diagnostics.json'),fresh_diagnostics_sha256=common.sha(diagnostics),
        contract_sha256=common.sha(common.read_blob(out/'CONTRACT.json')),
        original_inputs_modified=False,new_DM_downloaded_or_Nbody_generated=False,main_covariance_or_MCMC_created=False)
    common.save(out/'AUDIT.json',report)
    state.update(state='COMPLETE',stage='same_payload_and_recipe_verified',child_pid=None,
        updated_UTC=datetime.now(timezone.utc).isoformat());common.save(out/'STATUS.json',state)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        if STATUS_PATH is not None and STATUS_PATH.exists():
            state=json.loads(STATUS_PATH.read_text())
            state.update(state='FAILED',error=traceback.format_exc(),updated_UTC=datetime.now(timezone.utc).isoformat())
            common.save(STATUS_PATH,state)
        raise
