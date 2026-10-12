"""模块大纲：七参数六组独立输入/全链原门回放；生成pre/post对照JSON。

输入审计不读HDF；完成审计只在实际owner和全部writer终止后读完整HDF。
旧real-space兼容闭合与指定先验是模型假设，不作full-STO/IR/kmax放行。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import socket
import numpy as np
import emcee
import run_sto_three_noise_prepost_v1 as run
from post_rsd_owner_quiescence import owner_quiescent


def independent_prediction(row, case):
    fnl,b,b2,bk,a,bs,ps=row
    x=(b-2.25)/1.75
    cb=np.array([1,x,2*x*x-1,4*x**3-3*x,8*x**4-8*x*x+1])
    bdet=np.einsum('i,j,k,ijkn->n',cb,[1,b2,bk],[1,fnl,fnl*fnl],case.coefficients)
    bp=2*1.686*(b-case.meta['p_fixed_physical']);g=run.F_GROWTH;t=case.templates
    pdet=(b*b*t[0]+2*b*g*t[1]+g*g*t[2]+2*fnl*bp*(b*t[3]+g*t[4])+fnl*fnl*bp*bp*t[5])
    p,pmu,pmu4,pM,pmuM,pM2,valid=case.primitives;n2,n3=case.weight_moments
    gnoise=b*b*p+2*b*g*pmu+g*g*pmu4
    lnoise=bp*(b*pM+g*pmuM)
    qnoise=bp*bp*pM2
    return np.r_[pdet+ps*n2,bdet+a*n3*valid+bs*n2*(gnoise+fnl*lnoise+fnl*fnl*qnoise)]


def inputs():
    contract=run.guard();checks=[]
    for index,(stage,truth) in enumerate(run.SPECS):
        case=run.ThreeNoiseCase(run.case_directory(index));meta=case.meta
        with np.load(Path(meta['source_case'])/'likelihood.npz',allow_pickle=False) as old:
            for name in ['target','covariance','chol']:
                np.testing.assert_array_equal(getattr(case,name),old[name])
        if (meta['parameters']!=list(run.NAMES) or case.target.shape!=(21,)
                or meta['n_covariance']!=(402 if stage=='post' else 500)
                or meta['n_target']!=100 or meta['covariance_is_of_mean']
                or meta['volume_Gpc_h3']!=1 or meta['covariance_multiplier']!=1
                or not meta['P_B_cross_covariance'] or meta['IR']):
            raise ValueError('actual requested stage/data/parameter contract mismatch')
        np.testing.assert_array_equal(case.bounds[:6],[[-500,500],[.5,4],[-4,6],[-4,4],[-20,20],[-10,10]])
        np.testing.assert_allclose(case.bounds[6]*case.weight_moments[0],[-1e6,1e6],rtol=3e-15)
        alpha=(meta['n_covariance']-23)/(meta['n_covariance']-1)
        np.testing.assert_allclose(case.chol@case.chol.T,case.covariance/alpha,rtol=3e-13,atol=1e-7)
        np.testing.assert_array_equal(case.indices,run.selected(.08))
        rng=np.random.default_rng(33101+index)
        probes=rng.uniform(case.bounds[:,0],case.bounds[:,1],(128,7))
        probes=np.vstack((probes,[0,.5,-4,-4,-20,-10,case.bounds[6,0]],
                          [500,4,6,4,20,10,case.bounds[6,1]],[-500,4,-4,4,0,0,0]))
        actual=case.model.predictions(probes)
        direct=np.stack([independent_prediction(row,case) for row in probes])
        delta=np.linalg.solve(case.chol,(actual-direct).T).T
        scale=np.maximum(1.,np.max(abs(np.linalg.solve(case.chol,direct.T).T),axis=1))
        error=float(np.max(abs(delta)/scale[:,None]))
        if error>1e-9:raise ValueError('actual seven-parameter polynomial/legacy noise response failed1e-9 gate')
        checks.append(dict(name=meta['name'],stage=stage,cosmology=truth,n_covariance=meta['n_covariance'],
            p_fixed_physical=meta['p_fixed_physical'],actual_independent_prediction_probe_count=len(probes),
            maximum_independent_white_scaled_error=error,original_target_C_single_and_Cholesky_bitwise_unchanged=True,
            likelihood_sha256=run.digest(case.directory/'likelihood.npz')))
    result=dict(status='ACTUAL_SIX_LEGACY3_INPUTS_AND_INDEPENDENT_PREDICTIONS_VERIFIED_NOT_PHYSICAL_NOISE',
        created_UTC=run.previous.stamp(),checks=checks,parent_sha256=run.digest(run.OUT/'CONTRACT.json'),
        parameters=list(run.NAMES),noise_parameter_count=3,model_data_or_priors_changed_by_audit=False,HDF_read=False,
        pre_and_post_covariance_count_and_ids_differ=True,pre_and_post_tracer_p_differ=True,
        full_STO_noise_closure_certified=False,audit_source_sha256=run.digest(__file__))
    run.save_json(run.OUT/'INDEPENDENT_INPUT_AUDIT.json',result)
    return result


def complete():
    state=run.read(run.OUT/'STATUS.json')
    if not owner_quiescent(state):
        print('NOT_READY_ACTUAL_SEVEN_PARAMETER_OWNER_LIVE_NO_HDF_READ',flush=True);return
    for progress in (run.OUT/'cases').glob('*/PROGRESS_ensemble_*.json'):
        if not owner_quiescent(run.read(progress)):
            print('NOT_READY_ACTUAL_SEVEN_PARAMETER_WRITER_LIVE_NO_HDF_READ',flush=True);return
    if (state['state']!='SIX_LEGACY3_PREPOST_TRIALS_FINISHED_REVIEW_CONVERGENCE'
            or state['completed_cases']!=[run.case_directory(i).name for i in range(6)]):
        print('NOT_READY_SIX_REAL_TRIALS_NOT_FINISHED_NO_HDF_READ',flush=True);return
    inputs();checks=[]
    for index,(stage,truth) in enumerate(run.SPECS):
        case=run.ThreeNoiseCase(run.case_directory(index));case.initialize()
        stored=run.read(case.directory/'SUMMARY.json')
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}') for j in range(4)]
        iterations=[int(backend.iteration) for backend in backends]
        if iterations!=[stored['steps']]*4:raise ValueError('actual unequal completed ensembles')
        chains=np.stack([backend.get_chain() for backend in backends]);accepted=np.stack([backend.accepted for backend in backends])
        if chains.shape!=(4,stored['steps'],32,7):raise ValueError('actual complete four32walker seven-param chain required')
        physical=case.physical(chains)
        if not np.isfinite(physical).all() or np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):
            raise ValueError('actual samples outside frozen seven priors')
        replay,flat=run.shared.diagnose(case,chains,accepted,stored['steps'])
        if replay['gates']!=stored['gates'] or replay['converged']!=stored['converged']:
            raise ValueError('actual seven-param original sampling gate replay mismatch')
        np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
        np.testing.assert_allclose(replay['fNL_sigma'],stored['fNL_sigma'],rtol=2e-13)
        with np.load(case.directory/'POSTERIOR_THIN.npz',allow_pickle=False) as source:
            np.testing.assert_array_equal(source['parameter_names'],case.names)
            pick=np.random.default_rng(7500+index).choice(len(flat),min(30000,len(flat)),replace=False)
            np.testing.assert_array_equal(source['samples'],flat[pick])
        q=np.array(replay['quantiles_2p5_16_50_84_97p5'])[:,0]
        checks.append(dict(name=case.meta['name'],stage=stage,cosmology=truth,truth_fNL=case.meta['truth_fNL'],
            steps_each_ensemble=stored['steps'],converged=bool(replay['converged']),original_gates=replay['gates'],
            fNL_quantiles_2p5_16_50_84_97p5=q.tolist(),fNL_sigma=float(replay['fNL_sigma']),
            rank_rhat_max=float(np.max(replay['rank_normalized_split_rhat'])),
            retained_length_over_tau_min=float(np.min(replay['length_over_tau'])),
            prior_sensitive=bool(replay['prior_sensitive']),boundary_mass_each_parameter=np.array(replay['boundary_mass_each_parameter']).tolist(),
            n_covariance=case.meta['n_covariance'],p_fixed_physical=case.meta['p_fixed_physical'],
            summary_sha256=run.digest(case.directory/'SUMMARY.json')))
        del chains,physical,flat,backends
    comparisons=[]
    for truth in run.previous.COSMOS:
        pre=next(row for row in checks if row['stage']=='pre' and row['cosmology']==truth)
        post=next(row for row in checks if row['stage']=='post' and row['cosmology']==truth)
        comparisons.append(dict(cosmology=truth,truth_fNL=pre['truth_fNL'],
            pre_fNL_median=pre['fNL_quantiles_2p5_16_50_84_97p5'][2],pre_fNL_sigma=pre['fNL_sigma'],
            post_fNL_median=post['fNL_quantiles_2p5_16_50_84_97p5'][2],post_fNL_sigma=post['fNL_sigma'],
            post_to_pre_sigma_ratio=post['fNL_sigma']/pre['fNL_sigma'],
            both_original_sampling_gates_pass=pre['converged'] and post['converged'],
            difference_is_not_strict_same_covariance_or_same_tracer_p_reconstruction_only_test=True))
    all_converged=all(row['converged'] for row in checks)
    result=dict(status=('SIX_ACTUAL_USER_THREE_NOISE_PREPOST_CHAINS_VERIFIED_NOT_FORMAL_NOISE'
        if all_converged else 'SIX_LEGACY3_TRIALS_FINISHED_ORIGINAL_SAMPLING_GATES_FAILED'),
        created_UTC=run.previous.stamp(),all_six_converged=all_converged,checks=checks,comparisons=comparisons,
        noise_parameter_count=3,total_parameter_count=7,n_target_per_truth=100,C_single=True,volume_Gpc_h3=1.,
        P_bins=12,B_bins=9,kmax_P=.08,kmax_B=.08,P_B_cross_covariance=True,IR=False,
        input_audit_sha256=run.digest(run.OUT/'INDEPENDENT_INPUT_AUDIT.json'),
        old12_products_preserved=True,p_uncertainty_propagated=False,
        diagnostic_only=True,full_STO_noise_closure_certified=False,formal_fNL_or_kmax_released=False,
        audit_source_sha256=run.digest(__file__))
    run.save_json(run.OUT/'MCMC_COMPLETION_AUDIT.json',run.shared.clean(result))
    print(json.dumps(run.shared.clean(result)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--completed',action='store_true');args=parser.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:raise ValueError('original login26 CPU12 audit only')
    if args.completed:complete()
    else:print(json.dumps(inputs()),flush=True)


if __name__=='__main__':main()
