"""模块大纲：当前环境重产 fid000 与旧同配方缓存回归；验证 LC_m 新源身份。"""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from rsd_b000_contract import PART,digest
from inventory_sto_post_inputs import SOURCE
from run_rsd_bonly_workflow import catalog


def main():
    new=PART.parent/'01_reconstruction_methods/outputs/sto_marisa_post_rsd_source'
    old=SOURCE/'fiducial/real000/env_lam1_gsr2_thr15/post_sto_halo.npz'
    fresh=new/'fiducial/real000/env_lam1_gsr2_thr15/post_sto_halo.npz'
    checks={}
    with np.load(old,allow_pickle=False) as a,np.load(fresh,allow_pickle=False) as b:
        for key in ('positions_mpc_h','weights','env_labels','component_labels','base_weights','solution_weight_by_component'):
            x,y=np.asarray(a[key]),np.asarray(b[key])
            if x.shape!=y.shape:raise ValueError('source shape mismatch: '+key)
            checks[key]={'exactly_equal':bool(np.array_equal(x,y)),
                         'max_absolute_difference':float(np.max(np.abs(x.astype(float)-y.astype(float))))}
    required=('positions_mpc_h','weights','env_labels','component_labels','base_weights')
    same=all(checks[key]['exactly_equal'] for key in required)
    negative=new/'LC_m/real000/env_lam1_gsr2_thr15/post_sto_halo.npz'
    diag=json.loads((negative.parent/'diagnostics.json').read_text())
    config_expected={'env_source':'dm','weight_source':'dm','env_mesh':128,'recon_mesh':128,
                     'gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.}
    for key,value in config_expected.items():
        if diag['config'][key]!=value:raise ValueError('negative source recipe mismatch: '+key)
    _,positions,meta=catalog('LC_m',0)
    with np.load(negative,allow_pickle=False) as z:
        np.testing.assert_array_equal(positions,z['positions_mpc_h'])
        w=np.asarray(z['weights'],dtype=float)
        if w.shape!=(len(positions),) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('invalid negative weights')
    out={'status':'STO_SOURCE_REPRODUCTION_PASS' if same else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'fiducial_cache_comparison':checks,
         'old_source_sha256':digest(old),'regenerated_source_sha256':digest(fresh),
         'LC_m_source_sha256':digest(negative),'LC_m_diagnostics_sha256':digest(negative.parent/'diagnostics.json'),
         'LC_m_raw_fof_sha256':meta['source_raw_sha256'],'LC_m_halo_rows_exactly_equal':True,
         'LC_m_nhalo':len(w),'LC_m_weight_min_max':[float(w.min()),float(w.max())],
         'new_source_root':str(new),'old_source_modified':False,'covariance_inputs_complete':False,
         'environment_note':'desilike lacks pmesh; successfully used existing cosmodesi production environment; no installation',
         'entry_sha256':digest(Path(__file__))}
    target=PART/'outputs/sto_marisa_post_rsd_mean100_v1/SOURCE_REPRODUCTION_AUDIT.json'
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if not same:raise SystemExit(1)


if __name__=='__main__':main()
