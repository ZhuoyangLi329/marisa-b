"""模块大纲：只复核已完成且当前controller不再写入的P0阶段，保留诊断状态。

post完成后controller等待pre时，明确证明post文件不会再被该程序写入。
不打开当前采样阶段的HDF；回放全链、原门、分位数/标准差和thin RNG。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import numpy as np
import emcee
import run_all_available_p_p0only_v1 as run
from post_rsd_owner_quiescence import owner_quiescent

def complete(stage):
    state=run.read(run.OUT/'P0_RECALIBRATED_STATUS.json');root=run.root(stage)
    names=run.read(root/'CASES.json')
    if not set(names).issubset(state['completed_cases']):raise ValueError('actual stage cases unfinished; no HDF read')
    live=not owner_quiescent(state)
    if live and not (stage=='post' and state['current_stage']=='pre'):
        raise ValueError('actual controller can still write this stage; no HDF read')
    run.guard(stage);checks=[]
    for index,name in enumerate(names):
        case=run.AllPCase(root/'cases'/name);case.initialize();stored=run.read(case.directory/'SUMMARY.json')
        backends=[emcee.backends.HDFBackend(str(case.directory/'chain.h5'),name='ensemble_'+str(i),read_only=True) for i in range(4)]
        if [int(b.iteration) for b in backends]!=[stored['steps']]*4:raise ValueError('actual unequal finished P0 ensembles')
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        physical=case.physical(chains)
        if chains.shape!=(4,stored['steps'],32,3) or not np.isfinite(physical).all():raise ValueError('invalid actual three-parameter chain')
        if np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):raise ValueError('actual posterior outside priors')
        replay,flat=run.shared.diagnose(case,chains,accepted,stored['steps'])
        if replay['gates']!=stored['gates'] or replay['converged']!=stored['converged']:raise ValueError('actual original diagnostic replay differs')
        np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
        np.testing.assert_allclose(replay['fNL_sigma'],stored['fNL_sigma'],rtol=2e-13)
        with np.load(case.directory/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
            pick=np.random.default_rng(5500+index).choice(len(flat),min(30000,len(flat)),replace=False)
            np.testing.assert_array_equal(z['parameter_names'],case.names)
            np.testing.assert_array_equal(z['samples'],flat[pick])
        checks.append(dict(name=name,converged=bool(replay['converged']),steps_each_ensemble=stored['steps'],
            original_gates=replay['gates'],fNL_quantiles=np.asarray(replay['quantiles_2p5_16_50_84_97p5'])[:,0].tolist(),
            fNL_sigma=float(replay['fNL_sigma']),n_covariance=case.meta['n_covariance'],
            p_fixed_physical=case.meta['p_fixed_physical'],physical_p_calibration_actual_count=case.meta['physical_p_calibration_target_count'],
            summary_sha256=run.digest(case.directory/'SUMMARY.json')))
    report=dict(status='ACTUAL_STAGE_THREE_P0_CHAINS_VERIFIED_DIAGNOSTIC_PCAL_PROTOCOL_UNDER_REVIEW',
        created_UTC=run.stamp(),stage=stage,all_three_converged=all(x['converged'] for x in checks),checks=checks,
        actual_stage_completed_and_controller_never_returns_to_this_stage=True,
        controller_live_in_other_stage=live,all_complete_HDF_prior_gate_quantile_sigma_thin_RNG_replay=True,
        C_single=True,volume_Gpc_h3=1.,n_target=100,n_covariance=500,kmax_P=.08,IR=False,
        p_calibration_statistical_precision_decision_pending=True,not_updated_final_prepost_results=True,
        audit_source_sha256=run.digest(__file__))
    run.save_json(root/'COMPLETION_AUDIT.json',report);print(run.shared.clean(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['post','pre'],required=True)
    args=ap.parse_args();complete(args.stage)
