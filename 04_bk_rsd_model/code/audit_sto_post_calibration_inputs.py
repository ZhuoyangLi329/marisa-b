"""模块大纲：旧 p 校准的确切权重来源；当前权重逐行比较；独立 holdout 输入盘点。

不把目录族同名当作配置相同，不重跑或覆盖旧校准，不替换 mean100/Fid500 主样本。
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from rsd_b000_contract import PART,digest
from inventory_sto_post_inputs import SOURCE
from run_rsd_bonly_workflow import raw_path


def main():
    old=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source')
    parent=PART.parent/'02_reconstruction_on_pk';rows=[]
    for cosmo in ('fiducial','LC_p'):
        a=old/cosmo/'real000/lam1_dm_nenv';b=SOURCE/cosmo/'real000/env_lam1_gsr2_thr15'
        da=json.loads((a/'diagnostics.json').read_text());db=json.loads((b/'diagnostics.json').read_text())
        with np.load(a/'post_sto_halo.npz') as aa,np.load(b/'post_sto_halo.npz') as bb:
            position_equal=bool(np.array_equal(aa['positions_mpc_h'],bb['positions_mpc_h']))
            x=np.asarray(aa['weights'],dtype=float);y=np.asarray(bb['weights'],dtype=float)
            if x.shape!=y.shape:raise ValueError('source row mismatch')
            rows.append({'cosmology':cosmo,'realization':0,'halo_positions_identical':position_equal,
                         'weights_identical':bool(np.array_equal(x,y)),
                         'max_absolute_weight_difference':float(np.max(np.abs(x-y))),
                         'weight_L2_relative_difference':float(np.linalg.norm(x-y)/np.linalg.norm(x)),
                         'historical_gsr':da['config']['gsr_mpc_h'],'current_gsr':db['config']['gsr_mpc_h'],
                         'historical_N2':float(1e9*np.sum(x*x)/x.sum()**2),'current_N2':float(1e9*np.sum(y*y)/y.sum()**2),
                         'historical_source':str(a),'current_source':str(b),
                         'historical_source_sha256':digest(a/'post_sto_halo.npz'),'current_source_sha256':digest(b/'post_sto_halo.npz'),
                         'historical_diagnostics_sha256':digest(a/'diagnostics.json'),'current_diagnostics_sha256':digest(b/'diagnostics.json')})
    historical=[]
    for path in (parent/'results/fnlfixed100_freep_results.json',
                 parent/'work/quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json'):
        result=json.loads(path.read_text());historical.append({'path':str(path),'sha256':digest(path),
           'rows':[{k:row[k] for k in ('method','target','p','b1','Pshot_residual') if k in row}
                   for row in result['rows'] if row.get('method') in ('sto_only','sto_marisa')],
           'calibration_fNL_fixed':100,'target_realizations':[0,99],
           'target_overlaps_current_validation':True,'historical_volume_ratio':42.875,
           'historical_volume_or_covariance_used_in_current_task':False})
    tail=PART.parent/'01_reconstruction_methods/outputs/sto_marisa_v4_lcp250_source'
    holdout=[]
    required={'env_source':'dm','weight_source':'dm','gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.,'env_mesh':128,'recon_mesh':128}
    for i in range(100,250):
        source=tail/'LC_p'/f'real{i:03d}/env_lam1_gsr2_thr15';weights=source/'post_sto_halo.npz';diag=source/'diagnostics.json'
        matching=False
        if diag.exists():
            config=json.loads(diag.read_text())['config'];matching=all(config[k]==v for k,v in required.items())
        holdout.append({'realization':i,'raw_fof_exists':raw_path('LC_p',i).exists(),'weights_exist':weights.exists(),
                        'recipe_matches':matching,'source':str(source)})
    out={'status':'HISTORICAL_P_SOURCE_AUDITED_NOT_ADOPTED','created_UTC':datetime.now(timezone.utc).isoformat(),
         'source_comparisons':rows,'historical_calibrations':historical,
         'historical_physical_p_direct_reuse_approved':False,'physical_p_newly_fitted':False,
         'frozen_algorithm_p_rec':1.2822,'frozen_algorithm_b_rec':2.2280546353635335,
         'holdout_candidates':holdout,'holdout_complete_count':sum(r['raw_fof_exists'] and r['weights_exist'] and r['recipe_matches'] for r in holdout),
         'holdout_role':'potential independent calibration of weighted STO RSD input before Marisa, not an additional reconstruction comparison',
         'main_validation_ids_unchanged':[0,99],'main_covariance_ids_unchanged':[1000,1499],
         'main_covariance_count':500,'main_covariance_of_mean':False,'volume_Gpc_h3':1.,
         'independent_main_covariance_inputs_still_missing':True,'entry_sha256':digest(Path(__file__))}
    dest=PART/'outputs/sto_marisa_post_rsd_mean100_v1/CALIBRATION_INPUT_AUDIT.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='holdout_candidates'},indent=2),flush=True)


if __name__=='__main__':main()
