"""模块大纲：完成链核验；真实 fNL 固定 profile；低 k 先验边界；共同残差诊断。

不改已有链、likelihood、数据或先验，不新增物理项；profile 只是 mock 的事后诊断。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from iminuit import Minuit
from rsd_b000_contract import OUT,digest,save_json,CUTS
from run_rsd_b000_mcmc import Case,clean

def profile_truth(case,summary,mapmeta):
    truth=case.meta['truth_fNL'];start=np.array(mapmeta['center'][1:]);names=case.names[1:]
    def objective(*values):return case.objective(truth,*values)
    trials=[]
    for b1 in (start[0],1.25,1.93,2.8):
        initial=start.copy();initial[0]=b1
        fit=Minuit(objective,*initial,name=names);fit.errordef=1
        for name,bounds,error in zip(names,case.bounds[1:],[.1,.2,.2]+[.3]*(case.ndim-4)):
            fit.limits[name]=bounds;fit.errors[name]=error
        fit.migrad(ncall=10000)
        if not fit.valid:fit.simplex(ncall=3000);fit.migrad(ncall=10000)
        trials.append({'chi2':float(fit.fval),'parameters':[truth]+list(fit.values),'minuit_valid':bool(fit.valid)})
    trials=[r for r in trials if np.isfinite(r['chi2']) and np.isfinite(r['parameters']).all()]
    if not trials:raise ValueError('all truth-constrained profile starts failed')
    best=min(trials,key=lambda r:r['chi2']);delta=best['chi2']-mapmeta['chi2_MAP']
    residual=case.residual(best['parameters'])[0];diagonal=case.target.size
    return {'truth_fNL':truth,'chi2_fixed_truth':best['chi2'],'chi2_free_MAP':mapmeta['chi2_MAP'],
            'delta_chi2_fixed_vs_free':delta,'sqrt_delta_chi2_descriptive':float(np.sqrt(max(0.,delta))),
            'free_MAP_potential_optimization_issue':bool(delta<-.001),
            'profile_parameters':best['parameters'],'profile_minuit_valid':best['minuit_valid'],
            'profile_starts':trials,'whitened_residual':residual.tolist(),
            'data_points':diagonal,'retained_parameter_names':case.names,
            'note':'descriptive profile diagnostic at known mock truth; bounded priors and weak noise parameters prevent an exact Wilks significance claim'}

def main():
    result=json.loads((OUT/'RESULTS.json').read_text());names=json.loads((OUT/'CASES.json').read_text())
    if result['status']!='COMPLETE_WITH_DIAGNOSTICS':raise ValueError('wait for complete baseline and sensitivity summaries')
    records={};summaries={}
    for name in names:
        root=OUT/'cases'/name;summary=json.loads((root/'SUMMARY.json').read_text());case=Case(root)
        if summary['likelihood_sha256']!=case.meta['likelihood_sha256']:raise ValueError('summary likelihood drift')
        if not summary['converged']:raise ValueError('unconverged result must not be treated as complete here')
        records[name]=profile_truth(case,summary,json.loads((root/'MAP.json').read_text()));summaries[name]=summary
    common=[]
    for cut in CUTS:
        vectors=[np.array(records[f'{c}_mean100_k{cut:.2f}']['whitened_residual']) for c in ('fiducial','LC_p','LC_m')]
        norm=[np.linalg.norm(v) for v in vectors]
        common.append({'kmax_B':cut,'cosine_fixed_truth_fid_vs_plus':float(vectors[0]@vectors[1]/(norm[0]*norm[1])),
                       'cosine_fixed_truth_fid_vs_minus':float(vectors[0]@vectors[2]/(norm[0]*norm[2])),
                       'fNL_bias_medians':[summaries[f'{c}_mean100_k{cut:.2f}']['quantiles_2p5_16_50_84_97p5'][2][0]-truth
                                           for c,truth in [('fiducial',0),('LC_p',100),('LC_m',-100)]],
                       'note':'matched ICs are correlated, so the three truth tests are not independent significance measurements'})
    boundary={}
    for name,s in summaries.items():
        if 'k0.08' not in name:continue
        masses=dict(zip(s['parameters'],s['boundary_mass_each_parameter']))
        boundary[name]={'mass_in_outer_2pct_each_side':masses,'flat_uniform_reference_mass':.04,
                        'S_noise_boundary_mass':masses['S_stoch'],
                        'note':'R_stoch is weakly constrained; its approx 4% boundary mass alone is not evidence that fNL is prior-dependent. Wide-nuisance fNL shifts are the direct test.'}
    noise=json.loads((OUT/'MODEL_NOISE_AUDIT.json').read_text())
    value={'status':'POSTERIOR_AND_PROFILE_DIAGNOSTICS_COMPLETE','generated_UTC':datetime.now(timezone.utc).isoformat(),
           'source_results_sha256':digest(OUT/'RESULTS.json'),'analysis_source_sha256':digest(Path(__file__)),
           'completed_baseline':result['baseline_cases'],'completed_sensitivity':len(names)-result['baseline_cases'],'all_converged':True,
           'model_parameters':noise['parameter_count'],'extra_noise_not_IR':'R_stoch_anisotropic retained by native-SP lattice audit; IR remains off',
           'truth_profiles':records,'shared_residual_diagnostics':common,'low_cut_boundary_diagnostics':boundary,
           'volume_note':result['volume_note'],'prior_sensitivity':result['prior_sensitivity'],
           'range_candidates':result['range_candidates'],
           'conclusions':['Read the measured posterior, direct wide-prior comparisons and conditional range candidates; do not transfer a legacy-volume conclusion.',
                          'Shared residuals may reflect model mismatch or common sample fluctuation; these diagnostics do not uniquely identify missing IR/FoG/loop/PNG stochastic physics.',
                          'Matched PNG-sign ICs are correlated, not three independent significance measurements.'],
           'scope_unchanged':{'data':True,'covariance':True,'prior':True,'P_likelihood':False,'IR':False,'FoG':False,'loops':False}}
    save_json(OUT/'FINAL_INTERPRETATION.json',clean(value))
    print(json.dumps(clean({'status':value['status'],'all_converged':True,'profile_optimization_flags':
          [n for n,r in records.items() if r['free_MAP_potential_optimization_issue']],
          'shared_residuals':common}),indent=2),flush=True)

if __name__=='__main__':main()
