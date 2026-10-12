"""模块大纲：实际校准owner退出后复核全部HDF、p路径及摘要，不读取活动链。

第一列是物理p而非fNL；独立展开Z1平方，复核C_single与显式p注入。
校准后仍可能有边缘化/模型误差，不通过调整p强迫fNL中位数回真值。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import emcee
import run_all_available_rsd_pcal_v1 as run
from post_rsd_owner_quiescence import owner_quiescent
from post_rsd_p0_model import post_p0_linear
from rsd_b000_contract import F_GROWTH,digest,save_json

def direct_p(fnl,b1,p,templates):
    bp=2*1.686*(b1-p);t=np.asarray(templates);g=F_GROWTH
    return (b1*b1*t[0]+2*b1*g*t[1]+g*g*t[2]
        +2*fnl*bp*(b1*t[3]+g*t[4])+fnl*fnl*bp*bp*t[5])

def verify(stage):
    state=run.read(run.OUT/(stage+'_PCAL_STATUS.json'))
    if not owner_quiescent(state):raise ValueError('actual calibration owner active; no HDF read')
    if state['state']!='ALL_AVAILABLE_PHYSICAL_P_TRIAL_FINISHED_REVIEW_CHAIN':
        raise ValueError('actual calibration not finished')
    root=run.stage_root(stage);parent=run.read(root/'contract.json')
    for path,want in parent['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('actual calibration input/source changed: '+path)
    d=root/'cases'/run.case_name(stage);case=run.original.CalibrationCase(d);case.initialize()
    stored=run.read(d/'SUMMARY.json')
    backends=[emcee.backends.HDFBackend(str(d/'chain.h5'),name='ensemble_'+str(i),read_only=True) for i in range(4)]
    if [int(b.iteration) for b in backends]!=[stored['steps']]*4:raise ValueError('actual unequal calibration ensembles')
    chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
    physical=case.physical(chains)
    if chains.shape!=(4,stored['steps'],32,3) or not np.isfinite(physical).all():raise ValueError('invalid actual calibration chains')
    if np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):raise ValueError('actual samples violate priors')
    replay,flat=run.original.shared.diagnose(case,chains,accepted,stored['steps'])
    semantic_gates={('physical_p_ESS' if k=='fNL_ESS' else k):v for k,v in replay['gates'].items()}
    if semantic_gates!=stored['gates'] or replay['converged']!=stored['converged']:
        raise ValueError('actual calibration sampling diagnostics disagree')
    np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
    np.testing.assert_allclose(replay['fNL_sigma'],stored['physical_p_sigma'],rtol=2e-13)
    with np.load(d/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['parameter_names'],case.names)
        pick=np.random.default_rng(5500).choice(len(flat),min(30000,len(flat)),replace=False)
        np.testing.assert_array_equal(z['samples'],flat[pick])
    with np.load(root/'CALIBRATION_VECTOR_MATRIX.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['realizations'],np.arange(case.meta['calibration_target_count']))
        np.testing.assert_array_equal(z['vectors'].mean(axis=0),case.target)
    rng=np.random.default_rng(911);points=rng.uniform(case.bounds[:,0],case.bounds[:,1],(256,3))
    reference=np.array([direct_p(100,b,p,case.templates)+shot for p,b,shot in points])
    actual=case.model.predictions(points)
    scale=np.maximum(1,np.max(abs(np.linalg.solve(case.chol,reference.T).T),axis=1))
    error=float(np.max(abs(np.linalg.solve(case.chol,(actual-reference).T).T)/scale[:,None]))
    if error>1e-11:raise ValueError('physical p does not match independent Z1-square')
    # 相同b1下保持fNL*(b1-p)不变，P0必须不变；与p_rec没有关系。
    f,b,p=100.,2.2,.9;p2=1.4;f2=f*(b-p)/(b-p2)
    np.testing.assert_allclose(post_p0_linear(f,b,case.templates,p_halo=p),
        post_p0_linear(f2,b,case.templates,p_halo=p2),rtol=2e-14,atol=1e-8)
    np.testing.assert_array_equal(post_p0_linear(0,b,case.templates,p_halo=p),
        post_p0_linear(0,b,case.templates,p_halo=p2))
    q=np.asarray(replay['quantiles_2p5_16_50_84_97p5'])[:,0]
    report=dict(status='ACTUAL_ALL_AVAILABLE_PHYSICAL_P_CHAIN_AND_EXPLICIT_MODEL_PATH_VERIFIED',
        created_UTC=run.stamp(),stage=stage,calibration_target_count=case.meta['calibration_target_count'],
        calibration_fNL_fixed=100.,n_covariance=500,C_single=True,covariance_is_of_mean=False,
        volume_Gpc_h3=1.,steps_each_ensemble=stored['steps'],ensembles=4,walkers=32,
        converged=bool(replay['converged']),original_gates=semantic_gates,
        physical_p_quantiles_2p5_16_50_84_97p5=q.tolist(),physical_p_sigma=float(replay['fNL_sigma']),
        MAP_physical_p=float(case.center[0]),posterior_median_p_is_not_MAP_p=True,
        complete_HDF_prior_quantile_sigma_and_thin_RNG_replay=True,
        independent_Z1_square_probe_count=len(points),independent_white_scaled_error=error,
        P0_dependence_is_fNL_times_b1_minus_p=True,P0_at_fNL_zero_is_independent_of_p=True,
        physical_p_is_not_reconstruction_p_rec=True,calibration_mean_overlaps_main_ids0to99=True,
        p_estimator_for_main_fits='posterior median, original convention retained',
        fNL_median_recovery_guaranteed_by_p_calibration=False,
        raw_summary_calibration_target_count=stored['calibration_target_count'],
        raw_summary_count_is_legacy_hardcoded150=stored['calibration_target_count']!=case.meta['calibration_target_count'],
        actual_data_count_verified_from_matrix_and_case_not_raw_summary=True,
        raw_summary_and_chain_preserved=True,
        summary_sha256=digest(d/'SUMMARY.json'),audit_source_sha256=digest(__file__))
    # 原adapter的摘要保存回调写死150；修正独立伴随元数据，不改链或原摘要。
    corrected=dict(stored,stage=stage,calibration_target_count=case.meta['calibration_target_count'],
        all_available_positive_samples_used=True,calibration_ids=case.meta['calibration_ids'],
        calibration_main_target_overlap_ids=list(range(100)),strict_independent_calibration_claimed=False,
        gof_note='all-available mean with C_single500; not calibrated GOF',
        raw_summary_sha256=report['summary_sha256'],metadata_only_correction_no_resampling=True)
    save_json(d/'SUMMARY_METADATA_CORRECTED.json',corrected)
    report['corrected_summary_metadata_sha256']=digest(d/'SUMMARY_METADATA_CORRECTED.json')
    save_json(root/'COMPLETION_AUDIT.json',run.original.shared.clean(report))
    return float(q[2]),report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['pre','post'],required=True)
    args=ap.parse_args();p,report=verify(args.stage);print(run.original.shared.clean(report),flush=True)
