"""模块大纲：补齐主验证 LC_m 的同配方 STO 输入；调用唯一生产源；逐盒 FoF 身份验收。

只生成 STO+Marisa 的上游权重，不选择 post 网格、不放行 post ensemble/MCMC。
串行单核，不下载、不更换 IC 或算法；已有完整源复用，部分源拒绝覆盖。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from rsd_b000_contract import PART,digest
from run_rsd_bonly_workflow import catalog,raw_path
from post_rsd_catalog_identity import validate_periodic_identity

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
DEST=PART.parent/'01_reconstruction_methods/outputs/sto_marisa_post_rsd_source'
PRODUCER=PART.parent/'01_reconstruction_methods/sto_recon/run_halo_sto_variants.py'
VALIDATOR=PART.parent/'01_reconstruction_methods/common/entrypoint.py'
PRODUCER_SHA='6d3502015dd341b9e468c78f517fc7ad7ef39fc4bbb22e4d655e34e106ce7be2'
VALIDATOR_SHA='116a0983fbf498b74075278722e0d79c080b68c46ec568134461e67296f956a1'
HALOS=Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions')
DM=Path('/pscratch/sd/l/lzy/quijote-png/dm')
REQUIRED=('env_labels.npy','weights.npy','post_sto_halo.npz','diagnostics.json','spectra.npz','DONE')
RECIPE={'env_source':'dm','weight_source':'dm','gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.,'env_mesh':128,'recon_mesh':128}


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def check_code():
    if digest(PRODUCER)!=PRODUCER_SHA or digest(VALIDATOR)!=VALIDATOR_SHA:raise RuntimeError('upstream production code drift')


def audit_one(index):
    directory=DEST/'LC_m'/f'real{index:03d}/env_lam1_gsr2_thr15'
    if not all((directory/x).exists() for x in REQUIRED):raise ValueError('incomplete source outputs')
    metadata=json.loads((directory/'diagnostics.json').read_text())
    for k,v in RECIPE.items():
        if metadata['config'][k]!=v:raise ValueError('source recipe mismatch: '+k)
    if metadata['component_config']['method']!='nenv' or metadata['component_config']['base_weight']!='number':raise ValueError('wrong mark definition')
    _,positions,rawmeta=catalog('LC_m',index)
    with np.load(directory/'post_sto_halo.npz',allow_pickle=False) as z:
        identity=validate_periodic_identity(z['positions_mpc_h'],positions)
        w=np.asarray(z['weights'],dtype=float)
        if w.shape!=(len(positions),) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('invalid source weights')
    return {'cosmology':'LC_m','realization':index,'source':str(directory),'halo_positions_match_raw_FoF':True,
            'position_identity':identity,
            'raw_sha256':rawmeta['source_raw_sha256'],'source_sha256':digest(directory/'post_sto_halo.npz'),
            'diagnostics_sha256':digest(directory/'diagnostics.json'),'nhalo':len(w),
            'weight_min_max':[float(w.min()),float(w.max())],'weight_mean':float(w.mean()),
            'N2_self_moment':float(1e9*np.sum(w*w)/w.sum()**2),'N3_self_moment':float(1e18*np.sum(w**3)/w.sum()**3)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('source queue must be hard pinned to exactly one logical CPU')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError('numeric threading contract failed: '+key)
    check_code()
    if json.loads((ROOT/'SOURCE_REPRODUCTION_AUDIT.json').read_text())['status']!='STO_SOURCE_REPRODUCTION_PASS':
        raise RuntimeError('source reproduction gate has not passed')
    work=ROOT/'validation_source_queue_v2';work.mkdir(exist_ok=True);(work/'logs').mkdir(exist_ok=True);(work/'rows').mkdir(exist_ok=True)
    contract={'version':2,'cosmology':'LC_m','realizations':list(range(100)),'recipe':RECIPE,
              'method':'nenv','base_weight':'number','dtype':'f4','result_root':str(DEST),
              'producer_sha256':PRODUCER_SHA,'validator_sha256':VALIDATOR_SHA,'controller_sha256':digest(Path(__file__)),
              'identity_validator_sha256':digest(PART/'code/post_rsd_catalog_identity.py'),
              'previous_failed_queue':str(ROOT/'validation_source_queue'),
              'recovery_reason':'raw LC_m015 has exactly one periodic endpoint L, producer wraps to 0; no nonendpoint tolerance added',
              'role':'STO+Marisa primary validation input weights only','independent_Fid500_not_substituted':True,
              'post_ensemble_or_MCMC_released':False}
    contract_path=work/'CONTRACT.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise RuntimeError('source queue contract changed; inspect before resuming')
    else:atomic_json(contract_path,contract)
    inventory=[]
    for index in range(100):
        directory=DEST/'LC_m'/f'real{index:03d}/env_lam1_gsr2_thr15'
        present=[(directory/x).exists() for x in REQUIRED]
        item={'realization':index,'raw_fof':raw_path('LC_m',index).exists(),
              'cached_realspace_positions':(HALOS/'LC_m'/f'real{index:03d}/positions_mpc_h.npy').exists(),
              'DM_snapshot':(DM/'LC_m'/str(index)/'snapdir_003').is_dir(),
              'complete_source':all(present),'partial_source':any(present) and not all(present)}
        inventory.append(item)
    atomic_json(work/'PREFLIGHT.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':inventory})
    bad=[row for row in inventory if not all(row[k] for k in ('raw_fof','cached_realspace_positions','DM_snapshot')) or row['partial_source']]
    if bad:raise RuntimeError('source queue preflight requires inspection: '+json.dumps(bad))
    if not args.run:
        print(json.dumps({'status':'SOURCE_QUEUE_PREFLIGHT_PASS','source_count':100,'existing_complete':sum(r['complete_source'] for r in inventory),
                          'remaining':sum(not r['complete_source'] for r in inventory),'contract_sha256':digest(contract_path)},indent=2));return
    state={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
           'started_UTC':datetime.now(timezone.utc).isoformat(),'completed':[],'total':100,'role':contract['role']}
    status_path=work/'STATUS.json';start=time.time()
    def update(**values):
        state.update(values,updated_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
        atomic_json(status_path,state)
    try:
        for item in inventory:
            index=item['realization'];check_code()
            if digest(Path(__file__))!=contract['controller_sha256']:raise RuntimeError('controller source changed during queue')
            if digest(PART/'code/post_rsd_catalog_identity.py')!=contract['identity_validator_sha256']:raise RuntimeError('identity validator changed during queue')
            update(current_realization=index,stage='validate_existing' if item['complete_source'] else 'produce',child_pid=None)
            if not item['complete_source']:
                log=work/'logs'/f'LC_m_real{index:03d}.log'
                if log.exists():raise RuntimeError('prior incomplete execution log exists; inspect before retry')
                command=[sys.executable,'-B',str(PRODUCER),'--cosmology','LC_m','--realization',str(index),
                    '--method','nenv','--base-weight','number','--env-source','dm','--weight-source','dm',
                    '--env-mesh','128','--recon-mesh','128','--gsr','2','--thr','15','--lambda-th','1',
                    '--dtype','f4','--result-root',str(DEST),'--label','env_lam1_gsr2_thr15','--skip-existing']
                with log.open('w') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    update(child_pid=process.pid,log=str(log))
                    code=process.wait()
                if code:raise RuntimeError(f'LC_m {index} producer exit {code}; log preserved at {log}')
            check_code();row=audit_one(index);row['reused_source']=item['complete_source']
            atomic_json(work/'rows'/f'LC_m_real{index:03d}.json',row)
            state['completed'].append(index);update(child_pid=None,stage='source_verified')
            print(json.dumps({'completed_count':len(state['completed']),'total':100,'realization':index,
                              'source_sha256':row['source_sha256'],'elapsed_seconds':state['elapsed_seconds']}),flush=True)
        update(state='COMPLETE',stage='all_source_weights_verified',child_pid=None)
    except Exception:
        update(state='FAILED',error=traceback.format_exc());raise


if __name__=='__main__':main()
