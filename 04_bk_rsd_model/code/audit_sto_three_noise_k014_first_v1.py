"""模块大纲：只读取已完成case的HDF，给出0.14的独立链复核及pre/post对照。

其他case可继续运行；已完成case必须被parent列入completed_cases且其全部
实际writer退出。未完成case不读取HDF、不冒充结果。此审计不改链/模型/先验。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import socket
import numpy as np
import emcee
import run_sto_three_noise_k014_first_v1 as run
from post_rsd_owner_quiescence import owner_quiescent
from audit_sto_three_noise_prepost_v1 import independent_prediction


def audit(indices):
    run.guard();state=run.read(run.OUT/'STATUS.json');checks=[]
    for index in indices:
        d=run.case_directory(index)
        if d.name not in state['completed_cases']:
            print('NOT_READY_CASE_NOT_COMPLETED_NO_HDF_READ',d.name,flush=True);continue
        progress=list(d.glob('PROGRESS_ensemble_*.json'))
        if len(progress)!=4 or any(not owner_quiescent(run.read(p)) for p in progress):
            print('NOT_READY_ACTUAL_WRITER_LIVE_NO_HDF_READ',d.name,flush=True);continue
        case=run.EndpointCase(d);case.initialize();stored=run.read(d/'SUMMARY.json')
        backends=[emcee.backends.HDFBackend(str(d/f'chain_ensemble_{j}.h5'),
            name=f'ensemble_{j}',read_only=True) for j in range(4)]
        iterations=[int(b.iteration) for b in backends]
        if iterations!=[stored['steps']]*4:raise ValueError('actual completed HDF lengths disagree')
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape!=(4,stored['steps'],32,case.ndim):raise ValueError('actual ensemble dimension mismatch')
        physical=case.physical(chains)
        if (not np.isfinite(physical).all() or np.any(physical<=case.bounds[:,0])
                or np.any(physical>=case.bounds[:,1])):raise ValueError('actual samples outside recorded priors')
        replay,flat=run.base.shared.diagnose(case,chains,accepted,stored['steps'])
        if replay['gates']!=stored['gates'] or replay['converged']!=stored['converged']:
            raise ValueError('original sampling diagnostics disagree')
        np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],
            stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
        np.testing.assert_allclose(replay['fNL_sigma'],stored['fNL_sigma'],rtol=2e-13)
        with np.load(d/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
            pick=np.random.default_rng(7506+index).choice(len(flat),min(30000,len(flat)),replace=False)
            np.testing.assert_array_equal(z['parameter_names'],case.names)
            np.testing.assert_array_equal(z['samples'],flat[pick])
        errors=[]
        for j,backend in enumerate(backends):
            chain=chains[j].reshape(-1,case.ndim);saved=backend.get_log_prob().reshape(-1)
            pick=np.random.default_rng(91900+4*index+j).choice(len(chain),128,replace=False)
            direct=np.stack([independent_prediction(row,case) for row in case.physical(chain[pick])])
            white=np.linalg.solve(case.chol,(direct-case.target).T).T
            logp=-.5*np.sum(white*white,axis=1)
            np.testing.assert_allclose(saved[pick],logp,rtol=2e-12,atol=1e-10)
            errors.append(float(np.max(abs(saved[pick]-logp))))
        q=np.array(replay['quantiles_2p5_16_50_84_97p5'])[:,0]
        row=dict(name=d.name,stage=case.meta['stage'],cosmology=case.meta['cosmology'],
            truth_fNL=case.meta['truth_fNL'],parameters=case.names,noise_parameter_count=3,
            total_parameter_count=case.ndim,steps_each_ensemble=stored['steps'],
            converged=bool(replay['converged']),gates=replay['gates'],
            fNL_quantiles_2p5_16_50_84_97p5=q.tolist(),fNL_sigma=float(replay['fNL_sigma']),
            rank_rhat_max=float(np.max(replay['rank_normalized_split_rhat'])),
            retained_length_over_tau_min=float(np.min(replay['length_over_tau'])),
            boundary_mass_each_parameter=np.array(replay['boundary_mass_each_parameter']).tolist(),
            kmax_P=case.meta['kmax_P'],kmax_B=case.meta['kmax_B'],P_bins=12,B_bins=27,
            n_covariance=case.meta['n_covariance'],p_fixed=case.meta['p_fixed_physical'],
            minuit_valid=stored['minuit_valid'],chi2_MAP=stored['chi2_MAP'],
            gof_pvalue=None,C_single=True,volume_Gpc_h3=1.,IR=False,
            saved_logprob_spot_checks=512,maximum_saved_logprob_spot_error=max(errors),
            summary_sha256=run.digest(d/'SUMMARY.json'),
            likelihood_sha256=run.digest(d/'likelihood.npz'),audit_source_sha256=run.digest(__file__),
            full_STO_noise_closure_certified=False)
        run.save_json(d/'CHAIN_AUDIT.json',row);checks.append(row)
        print('CASE_VERIFIED',d.name,json.dumps(row),flush=True)
        del chains,physical,flat,backends
    comparisons=[]
    for truth in run.base.previous.COSMOS:
        pre=next((r for r in checks if r['stage']=='pre' and r['cosmology']==truth),None)
        post=next((r for r in checks if r['stage']=='post' and r['cosmology']==truth),None)
        if pre and post:
            comparisons.append(dict(truth_fNL=pre['truth_fNL'],cosmology=truth,
                pre_fNL_median=pre['fNL_quantiles_2p5_16_50_84_97p5'][2],pre_fNL_sigma=pre['fNL_sigma'],
                post_fNL_median=post['fNL_quantiles_2p5_16_50_84_97p5'][2],post_fNL_sigma=post['fNL_sigma'],
                post_to_pre_sigma_ratio=post['fNL_sigma']/pre['fNL_sigma'],
                both_sampling_gates_pass=pre['converged'] and post['converged'],
                different_stage_covariance_cohorts_and_p_recorded=True))
    result=dict(created_UTC=run.base.previous.stamp(),status='ACTUAL_COMPLETED_K014_CASES_VERIFIED_DIAGNOSTIC_ONLY',
        requested_indices=indices,verified_cases=checks,comparisons=comparisons,
        all_six_k014_verified=len(checks)==6,all_four_B_cuts_verified=False,
        original_settings_sha256=run.digest(run.OUT/'RUN_SETTINGS.json'),
        full_STO_noise_closure_certified=False,formal_fNL_or_kmax_released=False)
    run.save_json(run.OUT/'COMPLETION_AUDIT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices',nargs='+',type=int,default=list(range(6)))
    args=parser.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:
        raise ValueError('one original login26 CPU12 required')
    if any(i not in range(6) for i in args.indices):raise ValueError('actual endpoint case indices required')
    audit(args.indices)
