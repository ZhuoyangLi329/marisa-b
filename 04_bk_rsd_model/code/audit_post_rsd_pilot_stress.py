"""模块大纲：固定网格原点测试及 nested random 密度测试；只记录未白化误差。"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from rsd_b000_contract import PART,CUTS,selected,digest


def load(directory):
    done=json.loads((directory/'DONE.json').read_text());contract=json.loads((directory/'CONTRACT.json').read_text())
    for key,name in (('measurement_sha256','MEASUREMENTS.npz'),('post_halos_sha256','POST_HALOS.npz'),('contract_sha256','CONTRACT.json')):
        if done[key]!=digest(directory/name):raise ValueError('artifact changed: '+str(directory/name))
    with np.load(directory/'MEASUREMENTS.npz') as z:meas={k:np.array(z[k]) for k in z.files}
    with np.load(directory/'POST_HALOS.npz') as z:pos=np.array(z['positions'],dtype=float);w=np.array(z['weights'])
    with np.load(directory/'RANDOM_PROBES.npz') as z:probes={k:np.array(z[k]) for k in z.files}
    return dict(name=directory.name,done=done,contract=contract,meas=meas,pos=pos,weights=w,probes=probes)


def compare(base,test,kind):
    ac=base['contract'];bc=test['contract'];a=base['meas'];b=test['meas']
    for key in ('source_STO_sha256','source_diagnostics_sha256','fixed_reconstruction','random_seed'):
        if ac[key]!=bc[key]:raise ValueError('non-stress contract field changed: '+key)
    np.testing.assert_array_equal(base['weights'],test['weights'])
    np.testing.assert_array_equal(base['probes']['before'],test['probes']['before'])
    for key in ('P_edges','B_edges','P_post_nmodes','B_post_nmodes'):
        np.testing.assert_array_equal(a[key],b[key])
    if kind=='nested_random_density':
        np.testing.assert_array_equal(base['pos'],test['pos'])
        np.testing.assert_array_equal(base['probes']['after'],test['probes']['after'])
        if (ac['arguments']['nran'],bc['arguments']['nran'])!=(30,60):raise ValueError('not the intended random-density test')
    elif kind=='fixed_mesh_origin':
        np.testing.assert_array_equal(a['P_weighted_input'],b['P_weighted_input'])
        if bc['arguments']['grid_origin_fraction']!=.5:raise ValueError('wrong grid-origin shift')
    else:raise ValueError(kind)
    displacement=(test['pos']-base['pos']+500)%1000-500
    psel=a['P_post_k']<=.08+1e-12;rows=[]
    for cut in CUTS:
        ix=selected(cut);entry={'kmax_B':cut,'number_B':len(ix)}
        for key in ('B_post','B_post_nonzero_only_counterfactual'):
            delta=b[key][ix]-a[key][ix]
            entry[key]={'delta_L2_over_baseline_L2':float(np.linalg.norm(delta)/np.linalg.norm(a[key][ix])),
                        'max_abs_delta':float(np.max(np.abs(delta)))}
        rows.append(entry)
    return {'kind':kind,'base':base['name'],'test':test['name'],
            'fixed_reconstruction_and_STO_identical':True,'random_prefix_identical':True,
            'post_halos_identical':bool(np.array_equal(base['pos'],test['pos'])),
            'halo_shift_difference_RMS_Mpc_h':np.sqrt(np.mean(displacement**2,axis=0)).tolist(),
            'coordinate_translation_roundtrip_RMS_Mpc_h':test['done']['input_coordinate_roundtrip_RMS_Mpc_h'],
            'P0_post_kmax08_relative_L2':float(np.linalg.norm((b['P_post']-a['P_post'])[psel])/np.linalg.norm(a['P_post'][psel])),
            'B_cuts':rows,'whitened_error':None,'fNL_bias':None,
            'measurement_sha256':test['done']['measurement_sha256']}


def main():
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1';pilots=root/'pilots'
    base=load(pilots/'fiducial_real000_rec256_meas256_ran30_selfmatched_zeroaudit')
    origin=load(pilots/(base['name']+'_originhalf'))
    nested=load(pilots/'fiducial_real000_rec256_meas256_ran60_selfmatched_zeroaudit_nested')
    out={'status':'PAIRED_STRESS_DIAGNOSTICS_RECORDED_NOT_SCIENTIFIC_ACCEPTANCE',
         'created_UTC':datetime.now(timezone.utc).isoformat(),
         'comparisons':[compare(base,origin,'fixed_mesh_origin'),compare(base,nested,'nested_random_density')],
         'entry_sha256':digest(Path(__file__)),
         'scope':'one fiducial realization; fixed recon/measurement meshes256; no covariance or MCMC',
         'production_mesh_or_nran_chosen':False,'old_results_modified':False}
    path=root/'PILOT_STRESS_AUDIT.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
