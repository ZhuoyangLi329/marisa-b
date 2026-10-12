"""模块大纲：主验证三真值300份 STO 源逐行验收；不放行 post ensemble/Cov/MCMC。"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from rsd_b000_contract import PART,digest
from post_rsd_catalog import catalog
from post_rsd_catalog_identity import validate_periodic_identity
from prepare_sto_post_validation_sources import RECIPE,DEST
from inventory_sto_post_inputs import SOURCE,DM


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single CPU audit only')
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
    queue=json.loads((root/'validation_source_queue_v2/STATUS.json').read_text())
    if queue['state']!='COMPLETE' or queue['completed']!=list(range(100)):raise RuntimeError('LC_m weights not complete')
    target=root/'VALIDATION_SOURCE_AUDIT.json'
    if target.exists():raise RuntimeError('source audit already exists; inspect before overwriting')
    start=time.time();rows=[]
    for cosmo in ('fiducial','LC_p','LC_m'):
        for index in range(100):
            directory=SOURCE/cosmo/f'real{index:03d}/env_lam1_gsr2_thr15'
            if not (directory/'post_sto_halo.npz').exists():directory=DEST/cosmo/f'real{index:03d}/env_lam1_gsr2_thr15'
            diag=json.loads((directory/'diagnostics.json').read_text())
            for key,value in RECIPE.items():
                if diag['config'][key]!=value:raise ValueError('recipe differs: '+str(directory)+' '+key)
            if diag['component_config']['method']!='nenv' or diag['component_config']['base_weight']!='number':raise ValueError('wrong STO mark definition')
            rsd,positions,meta=catalog(cosmo,index)
            with np.load(directory/'post_sto_halo.npz',allow_pickle=False) as z:
                identity=validate_periodic_identity(z['positions_mpc_h'],positions);w=np.asarray(z['weights'],dtype=float)
            if w.shape!=(len(positions),) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('invalid weights')
            if cosmo=='LC_m':
                verified=json.loads((root/'validation_source_queue_v2/rows'/f'LC_m_real{index:03d}.json').read_text())
                if verified['source_sha256']!=digest(directory/'post_sto_halo.npz') or verified['raw_sha256']!=meta['source_raw_sha256']:
                    raise ValueError('new LC_m source changed since production')
            rows.append({'cosmology':cosmo,'realization':index,'source':str(directory),
                         'source_sha256':digest(directory/'post_sto_halo.npz'),'diagnostics_sha256':digest(directory/'diagnostics.json'),
                         'raw_fof':meta,'source_position_identity':identity,'nhalo':len(w),
                         'N2_data_self':float(1e9*np.sum(w*w)/w.sum()**2),
                         'N3_data_self':float(1e18*np.sum(w**3)/w.sum()**3),
                         'negative_weights':int(np.count_nonzero(w<0))})
            if index%20==19:print(json.dumps({'audited':len(rows),'total':300,'last':[cosmo,index]}),flush=True)
    covariance=[]
    for index in range(1000,1500):
        candidates=[p/'fiducial'/f'real{index:03d}/env_lam1_gsr2_thr15' for p in (SOURCE,DEST)]
        cache=any(all((p/name).exists() for name in ('post_sto_halo.npz','diagnostics.json','DONE')) for p in candidates)
        dm=DM/'fiducial'/str(index)/'snapdir_003'
        covariance.append({'realization':index,'matching_cache_candidate':cache,'DM_files_present':bool(list(dm.glob('*.hdf5')))})
    out={'status':'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.time()-start,
         'source_counts':{cosmo:sum(r['cosmology']==cosmo for r in rows) for cosmo in ('fiducial','LC_p','LC_m')},
         'source_rows':rows,'recipe':RECIPE,'weights_carried_from_realspace':True,
         'covariance_path_candidates':covariance,
         'missing_covariance_inputs_count':sum(not(r['matching_cache_candidate'] or r['DM_files_present']) for r in covariance),
         'covariance_count_required':500,'covariance_of_mean':False,'volume_Gpc_h3':1,
         'post_ensemble_or_MCMC_released':False,
         'entry_sha256':digest(Path(__file__)),'LC_m_source_queue_contract_sha256':digest(root/'validation_source_queue_v2/CONTRACT.json')}
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('source_rows','covariance_path_candidates')},indent=2),flush=True)


if __name__=='__main__':main()
