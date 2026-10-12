"""模块大纲：本轮实际C402的blind优化/真值profile复核；不改MAP、链或priors。

独立least-squares对照原Minuit MAP；真值仅用于后验诊断，绝不用来初始化
MCMC。forecast残差不是mean100的校准GOF，也不是完整STO闭合证据。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import socket
import numpy as np
from scipy.optimize import least_squares
import run_sto_post_latest_cov_pb_v1 as base
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2
from run_sto_eft_pb_preflight_v2 import profile

OUT=base.OUT/'MAP_PROFILE_REVIEW_v1'


def main():
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:raise ValueError('login26 CPU12 only')
    if OUT.exists():raise ValueError('preserve existing optimization review')
    parent=base.guard();OUT.mkdir();inputs={str(base.OUT/'PB_CONTRACT.json'):base.digest(base.OUT/'PB_CONTRACT.json'),
        str(Path(__file__)):base.digest(Path(__file__)),
        str(base.PART/'code/run_sto_eft_pb_preflight_v2.py'):base.digest(base.PART/'code/run_sto_eft_pb_preflight_v2.py')}
    for t in range(3):
        d=base.case_directory(t)
        for q in [d/'likelihood.npz',d/'contract.json']:inputs[str(q)]=base.digest(q)
    base.save_json(OUT/'CONTRACT.json',dict(status='SAME402_PB_BLIND_LS_AND_FIXED_TRUTH_REVIEW_NOT_MCMC_OR_NOISE_PASS',
        immutable_inputs=inputs,simulation_truth_used_for_free_optimization=False,
        simulation_truth_used_for_MCMC_initialization=False,core_priors_changed=False,
        data_covariance_or_original_MAP_changed=False,full_STO_noise_closure_certified=False))
    state=dict(state='RUNNING',pid=os.getpid(),CPU_affinity=[12],started_UTC=base.stamp(),completed=[])
    base.save_json(OUT/'STATUS.json',state);results=[]
    for t,cosmo in enumerate(base.COSMOS):
        d=base.case_directory(t);meta=base.read(d/'contract.json')
        with np.load(d/'likelihood.npz',allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files}
        model=EFTCandidateModelV2(a['coefficients'],a['templates'],a['operators'],a['intrinsic'],
            a['matched_zero'],a['weight_moments'],meta['p_fixed_physical'],base.F_GROWTH)
        fits=[]
        for f in (-200.,0.,200.):
            for b in (1.25,1.93,2.8):
                initial=np.array([f,b,-.25,-.3,0.,0.,0.,1.,b,2*1.686*(b-model.p_halo),1.,0.])
                fit=least_squares(lambda v:model.residual(v,a['target'],a['chol'])[0],initial,
                    bounds=(a['bounds'][:,0],a['bounds'][:,1]),
                    x_scale=np.array([50.,.2,1.,1.,1.,1.,1.,.5,1.,3.,1.,1.]),
                    ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2500)
                fits.append(dict(chi2=float(fit.fun@fit.fun),physical_parameters=fit.x.tolist(),
                    success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev)))
        best=min(fits,key=lambda r:r['chi2'])
        fixed,fixed_report=profile(model,a['target'],a['chol'],meta['truth_fNL'],a['bounds'])
        original=base.read(d/'MAP.json') if (d/'MAP.json').exists() else None
        row=dict(cosmology=cosmo,blind_free_least_squares=best,blind_candidates=fits,
            fixed_truth_profile=fixed_report,
            fixed_truth_minus_free_LS_chi2=fixed_report['chi2_single_box_weighted']-best['chi2'],
            original_MAP_observed=original is not None,
            original_minuit_valid=original.get('minuit_valid') if original else None,
            original_MAP_chi2=original.get('chi2_MAP') if original else None,
            original_MAP_not_overwritten=True,negative_delta_is_optimizer_diagnostic_not_significance=True,
            original_MCMC_initialization_and_priors_unchanged=True)
        if original:row['original_MAP_minus_free_LS_chi2']=original['chi2_MAP']-best['chi2']
        results.append(row);state['completed'].append(cosmo);state['updated_UTC']=base.stamp()
        base.save_json(OUT/'STATUS.json',state);base.save_json(OUT/(cosmo+'_REVIEW.json'),row)
    for q,want in inputs.items():
        if base.digest(q)!=want:raise ValueError('frozen actual review input drift')
    base.save_json(OUT/'REPORT.json',dict(status='ACTUAL402_BLIND_MAP_AND_FIXED_TRUTH_REVIEW_COMPLETE_NOT_NOISE_CERTIFICATION',
        results=results,n_covariance=402,C_single=True,volume_Gpc_h3=1.,
        p_fixed_physical=parent['p_fixed_physical'],p_uncertainty_propagated=False,
        formal_fNL_or_kmax_released=False,full_STO_noise_closure_certified=False,
        MCMC_data_covariance_priors_MAP_or_initialization_changed=False,
        created_UTC=base.stamp(),audit_source_sha256=base.digest(Path(__file__))))
    state.update(state='COMPLETE_NOT_MCMC_OR_FULL_NOISE_PASS',finished_UTC=base.stamp());base.save_json(OUT/'STATUS.json',state)
    print(json.dumps([dict(cosmology=x['cosmology'],blind_LS_chi2=x['blind_free_least_squares']['chi2'],
        fixed_truth_chi2=x['fixed_truth_profile']['chi2_single_box_weighted'],
        original_MAP_minus_free_LS_chi2=x.get('original_MAP_minus_free_LS_chi2')) for x in results]),flush=True)


if __name__=='__main__':main()
