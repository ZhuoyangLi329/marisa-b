"""模块大纲：真实 post pilot 身份哈希、三真值、网格与 estimator 对照。

没有 post covariance 时仅报告未白化的数值差异，不给 sigma/fNL 或科学通过结论。
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from rsd_b000_contract import PART,CUTS,reference,selected,digest


def main():
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1';pilots={};ref=reference()
    for directory in sorted((root/'pilots').iterdir()):
        if not (directory/'DONE.json').exists():continue
        done=json.loads((directory/'DONE.json').read_text());contract=json.loads((directory/'CONTRACT.json').read_text())
        for key,name in [('measurement_sha256','MEASUREMENTS.npz'),('post_halos_sha256','POST_HALOS.npz'),('contract_sha256','CONTRACT.json')]:
            if done[key]!=digest(directory/name):raise ValueError('pilot artifact changed: '+str(directory/name))
        with np.load(directory/'MEASUREMENTS.npz') as z:measure={k:np.array(z[k]) for k in z.files}
        with np.load(directory/'POST_HALOS.npz') as z:positions=np.array(z['positions'],dtype=float);weights=np.array(z['weights'])
        np.testing.assert_array_equal(measure['B_edges'],ref['edges'])
        np.testing.assert_allclose(measure['B_post_k'],ref['B000_k'],atol=5e-5,rtol=0.)
        fixed=contract['fixed_reconstruction']
        if (fixed['algorithm_growth_rate']!=0. or fixed['fNL_rec']!=100.
                or fixed['b_rec']!=2.2280546353635335 or fixed['p_rec']!=1.2822):raise ValueError('frozen algorithm changed')
        if len(done['CPU_affinity'])!=1:raise ValueError('pilot is not pinned to one CPU')
        pilots[directory.name]=dict(done=done,contract=contract,measure=measure,positions=positions,weights=weights)
    def compare(first,second):
        a,b=pilots[first],pilots[second]
        if a['contract']['random_seed']!=b['contract']['random_seed']:raise ValueError('random seed differs')
        if a['contract']['source_STO_sha256']!=b['contract']['source_STO_sha256']:raise ValueError('STO source differs')
        np.testing.assert_array_equal(a['weights'],b['weights'])
        np.testing.assert_array_equal(a['measure']['P_weighted_input'],b['measure']['P_weighted_input'])
        displacement=(b['positions']-a['positions']+500)%1000-500
        rows=[]
        for cut in CUTS:
            ix=selected(cut);x=a['measure']['B_post'][ix];y=b['measure']['B_post'][ix];delta=y-x
            rawscale=np.abs(a['measure']['B_post_raw'][ix])+np.abs(a['measure']['B_post_shot'][ix])
            rows.append({'kmax_B':cut,'number_B':len(ix),'delta_B_L2_over_B_L2':float(np.linalg.norm(delta)/np.linalg.norm(x)),
                         'max_abs_delta_B':float(np.max(np.abs(delta))),
                         'max_abs_delta_over_raw_plus_shot':float(np.max(np.abs(delta)/rawscale))})
        p=a['measure']['P_post'];pdelta=b['measure']['P_post']-p
        psel=a['measure']['P_post_k']<=.08+1e-12
        return {'first':first,'second':second,'halo_displacement_difference_RMS_Mpc_h':np.sqrt(np.mean(displacement**2,axis=0)).tolist(),
                'halo_displacement_difference_max_Mpc_h':float(np.max(np.abs(displacement))),
                'post_halos_identical':a['done']['post_halos_sha256']==b['done']['post_halos_sha256'],
                'P0_kmax08_delta_L2_over_P_L2':float(np.linalg.norm(pdelta[psel])/np.linalg.norm(p[psel])),
                'max_Braw_difference':float(np.max(np.abs(b['measure']['B_post_raw']-a['measure']['B_post_raw']))),
                'B_cuts':rows,'whitened_error':None,'post_covariance_used':False}
    template='fiducial_real000_rec{}_meas256_ran30'
    comparisons=[]
    for first,second in [(template.format(125),template.format(256)),(template.format(256),template.format(512)),
                          (template.format(256),template.format(256)+'_selfmatched')]:
        if first in pilots and second in pilots:comparisons.append(compare(first,second))
    completed=[]
    for name,pilot in pilots.items():
        completed.append({'name':name,'arguments':pilot['contract']['arguments'],
                          'host':pilot['done']['host'],'CPU_affinity':pilot['done']['CPU_affinity'],
                          'measurement_sha256':pilot['done']['measurement_sha256'],
                          'post_halos_sha256':pilot['done']['post_halos_sha256'],
                          'entry_sha256':pilot['done']['entry_sha256'],
                          'complete_UTC':pilot['done']['finished_UTC']})
    report={'status':'PILOT_NUMERIC_COMPARISONS_NOT_SCIENTIFIC_ACCEPTANCE','created_UTC':datetime.now(timezone.utc).isoformat(),
            'completed_pilots':completed,'comparisons':comparisons,
            'covariance_inputs_complete':False,'bulk_released':False,'MCMC_released':False,
            'caveats':['one realization per truth, not mean100','mesh change alters a known CIC window as well as alias error',
                       'no post covariance whitening or numerical acceptance yet','estimator change requires matched new pre control if adopted',
                       'weighted factorial identity is not a shift-noise closure'],
            'entry_sha256':digest(Path(__file__))}
    target=root/'PILOT_NUMERICS.json';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
