"""模块大纲：已有 STO 缓存配置和 halo 行身份；同一 FoF 的 RSD carry-over 权重。

仅核验各 available cosmology 的两份 pilot 缓存，不生成新的重构 catalog。
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from run_rsd_bonly_workflow import catalog
from rsd_b000_contract import PART,digest
from inventory_sto_post_inputs import SOURCE


def main():
    rows=[]
    expected={'env_source':'dm','weight_source':'dm','env_mesh':128,'recon_mesh':128,
              'gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.}
    for cosmo in ('fiducial','LC_p'):
        for real in (0,19):
            directory=SOURCE/cosmo/f'real{real:03d}/env_lam1_gsr2_thr15'
            path=directory/'post_sto_halo.npz';diag=directory/'diagnostics.json'
            meta=json.loads(diag.read_text())
            for name,value in expected.items():
                if meta['config'][name]!=value:raise ValueError(('STO recipe drift',name,path))
            if meta['component_config']['method']!='nenv':raise ValueError('not N+Env')
            with np.load(path,allow_pickle=False) as z:
                pos=np.array(z['positions_mpc_h']);weights=np.array(z['weights'],dtype=float)
            rsd,realpos,raw_meta=catalog(cosmo,real)
            np.testing.assert_array_equal(pos,realpos)
            if weights.shape!=(len(pos),) or not np.isfinite(weights).all() or weights.sum()<=0:
                raise ValueError('invalid STO weights')
            np.testing.assert_array_equal(rsd[:,:2],pos[:,:2])
            if np.any(rsd<0) or np.any(rsd>=1000):raise ValueError('RSD out of periodic box')
            sums=[float(np.sum(weights**i)) for i in (1,2,3)]
            rows.append({'cosmology':cosmo,'realization':real,'n_halo':len(pos),
                'source_cache':str(path),'source_cache_sha256':digest(path),'diagnostics_sha256':digest(diag),
                'raw_fof_sha256':raw_meta['source_raw_sha256'],'realspace_rows_exact_match':True,
                'RSD_horizontal_coordinates_unchanged':True,'weight_sum_powers_1_2_3':sums,
                'P_self_Mpc_h3':1e9*sums[1]/sums[0]**2,
                'B_all_same_particle_Mpc_h6':1e18*sums[2]/sums[0]**3,
                'negative_weight_count':int(np.count_nonzero(weights<0)),
                'weight_min':float(weights.min()),'weight_max':float(weights.max()),
                'RSD_displacement_rms_Mpc_h':raw_meta['rms_ds_Mpc_h']})
    payload={'status':'PILOT_CACHE_IDENTITY_PASS','created_UTC':datetime.now(timezone.utc).isoformat(),
        'cases_checked':len(rows),'rows':rows,'required_recipe':expected,
        'scope':'four cached sources only; real-space STO weights carried with halo identity into RSD',
        'new_RSD_environment_classification':False,'STO_covariance_inputs_complete':False,
        'entry_sha256':digest(Path(__file__))}
    target=PART/'outputs/sto_marisa_post_rsd_mean100_v1/CACHE_IDENTITY_AUDIT.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
