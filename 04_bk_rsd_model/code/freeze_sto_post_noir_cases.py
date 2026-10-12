"""模块大纲：全数据/校准/噪声门后冻结39组主链及同tracer P输入控制。

只读上游真实产物；不构造假noise audit，不用旧pre covariance或p_rec回退。
缺任一门则在创建mcmc_noir目录前退出；本入口不启动采样或宣布IR完成。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest,save_json
import rsd_pb_common as spectral
from rsd_bonly_common import TABLE
from post_rsd_noir_case_inputs import case_specs,assemble_case

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'mcmc_noir'
CAL=ROOT/'physical_p_calibration_mcmc'
CAL_NAME='physical_p_GSR2_RSD_fNLfixed100'
ENSEMBLE=ROOT/'ensemble_measurements'
TEMPLATES=ROOT/'noir_paware_templates'
HOLDOUT=ROOT/'calibration_holdout_p0'
NOISE=ROOT/'POST_WEIGHTED_NOIR_NOISE_AUDIT.json'


def read(path):return json.loads(Path(path).read_text())


def validate_template_audits(completion,independent,coefficient_sha,contract_sha):
    """认实际回放报告及同一系数/合同SHA，不能只认一个PASS字符串。"""
    if (completion.get('status')!='ALL44_DETERMINISTIC_NOIR_P_AWARE_TEMPLATES_COMPLETE_NOT_LIKELIHOOD'
        or independent.get('status')!='ALL44_NOIR_COEFFICIENT_NODE_AND_FINE_GRID_REPLAY_PASS'
        or completion.get('bins')!=44 or independent.get('bins')!=44
        or len(completion.get('rows',[]))!=44 or len(independent.get('rows',[]))!=44
        or independent.get('nodes_replayed')!=660 or independent.get('fine_probes_replayed')!=88
        or completion.get('coefficients_sha256')!=coefficient_sha
        or independent.get('coefficients_sha256')!=coefficient_sha
        or completion.get('contract_sha256')!=contract_sha
        or independent.get('contract_sha256')!=contract_sha):
        raise ValueError('complete independently validated deterministic post coefficients required')


def main():
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core case input freezing required')
    # 必需真实门在任何输出目录创建前核验；没有少盒/旧结果/默认p的替代分支。
    report_path=ENSEMBLE/'AUDIT.json';report=read(report_path)
    if (report['status']!='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE'
        or len(report['rows'])!=700 or report['statistics']['n_covariance']!=500
        or report['statistics']['n_target_per_truth']!=100 or report['covariance_is_of_mean']
        or report['covariance_multiplier']!=1. or report['volume_Gpc_h3']!=1.):
        raise ValueError('complete original700/mean100/C_single500 V1 ensemble required')
    admission=ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json'
    if read(admission)['status']!='ALL500_FIDUCIAL_SOURCES_AND_MEASUREMENT_ADMISSION_INDEPENDENTLY_VERIFIED':
        raise ValueError('source admission gate not passed')
    summary_path=CAL/'cases'/CAL_NAME/'SUMMARY.json';summary=read(summary_path)
    cal_parent=read(CAL/'contract.json')
    for path,want in cal_parent['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('physical-p calibration input/source drift: '+path)
    if (not summary['converged'] or summary['parameters'][0]!='physical_p'
        or summary['calibration_fNL_fixed']!=100. or summary['fNL_fitted']):
        raise ValueError('converged same-tracer physical-p calibration required')
    p_halo=float(summary['quantiles_2p5_16_50_84_97p5'][2][0])
    noise_audit=read(NOISE)
    if (noise_audit['status']!='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED'
        or noise_audit['p_fixed_physical']!=p_halo or not noise_audit['original_binning_preserved']
        or noise_audit.get('physical_parameter_dependence_independently_audited') is not True):
        raise ValueError('actual post noise/estimator audit with identical calibrated p required')
    definition=noise_audit['B_definition']
    if definition not in ('matched','nonzero_diagnostic'):raise ValueError('unknown audited B definition')
    info=report['ensemble_files'][definition];matrix_path=Path(info['path'])
    if digest(matrix_path)!=info['sha256']:raise ValueError('ensemble input drift')
    with np.load(matrix_path,allow_pickle=False) as z:
        target_vectors=np.array(z['target_vectors']);b_k=np.array(z['B_k']);b_indices=np.array(z['B_indices'])
        ensemble=dict(statistics=report['statistics'],blocks=report['block_slices'],
            covariance_single_box=np.array(z['covariance_single_box']),
            targets=dict(zip(('fiducial','LC_p','LC_m'),target_vectors)))
        p_edges=np.array(z['P_edges'])
    completion_path=TEMPLATES/'COMPLETION_AUDIT.json';completion=read(completion_path)
    coefficient_path=TEMPLATES/'DETERMINISTIC_COEFFICIENTS.npz'
    independent_path=TEMPLATES/'INDEPENDENT_COEFFICIENT_AUDIT.json';independent=read(independent_path)
    template_contract_path=TEMPLATES/'CONTRACT.json';template_contract_sha=digest(template_contract_path)
    validate_template_audits(completion,independent,digest(coefficient_path),template_contract_sha)
    with np.load(coefficient_path,allow_pickle=False) as z:
        coefficients=np.array(z['coefficients']);np.testing.assert_array_equal(z['indices'],b_indices)
        if str(z['contract_sha256'])!=template_contract_sha:raise ValueError('coefficient tensor contract drift')
    noise_path=Path(noise_audit['noise_arrays_path'])
    if digest(noise_path)!=noise_audit['noise_arrays_sha256']:raise ValueError('audited post noise arrays drift')
    with np.load(noise_path,allow_pickle=False) as z:
        b_noise=np.array(z['B_noise_by_truth']);p_noise=np.array(z['P_noise_by_truth'])
        b_noise_coefficients=np.array(z['B_noise_coefficients_by_truth']) if 'B_noise_coefficients_by_truth' in z else None
        np.testing.assert_array_equal(z['B_indices'],b_indices)
    if b_noise.shape!=(3,3,44) or p_noise.shape!=(3,12):raise ValueError('all three targets need explicitly audited noise units')
    if b_noise_coefficients is not None:
        if b_noise_coefficients.shape!=(3,3,5,3,3,44):raise ValueError('same parameter-polynomial noise basis for each truth required')
        if noise_audit.get('noise_representation')!='polynomial_plus_constant':raise ValueError('polynomial noise audit/array mismatch')
    elif noise_audit.get('noise_representation')!='constant_basis_independently_justified':
        raise ValueError('constant-only post noise cannot bypass parameter-dependence verification')
    previous=spectral.OUT
    try:spectral.OUT=HOLDOUT;p_templates,nmodes,p_k=spectral.p_lattice(p_edges)
    finally:spectral.OUT=previous
    window_path=HOLDOUT/'P0_WINDOW.npz';window_audit=read(HOLDOUT/'POST_P0_WINDOW_AUDIT.json')
    if window_audit['status']!='PASS' or window_audit['window_sha256']!=digest(window_path):
        raise ValueError('actual P estimator/window audit drift')
    planned=case_specs(include_weighted_input_P_control=True);ready=[]
    for spec in planned:
        index=('fiducial','LC_p','LC_m').index(spec.cosmology)
        arrays,meta=assemble_case(ensemble,spec,b_k,coefficients,p_templates,b_noise[index],p_noise[index],
            p_halo=p_halo,noise_audit=noise_audit,
            B_noise_coefficients=None if b_noise_coefficients is None else b_noise_coefficients[index])
        ready.append((spec,arrays,meta))
    sources=[Path(__file__),PART/'code/post_rsd_noir_case_inputs.py',PART/'code/post_rsd_ensemble.py',
        PART/'code/post_rsd_noir_polynomial.py',PART/'code/post_rsd_noir_likelihood.py',
        PART/'code/run_sto_post_noir_mcmc.py',PART/'code/run_rsd_b000_mcmc.py',
        PART/'code/run_rsd_p0only.py',PART/'code/rsd_pb_common.py',TABLE,
        ROOT/'ENSEMBLE_SELECTION.json',admission,report_path,matrix_path,summary_path,CAL/'contract.json',
        completion_path,coefficient_path,independent_path,template_contract_path,NOISE,noise_path,window_path,
        HOLDOUT/'POST_P0_WINDOW_AUDIT.json',PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    for path in noise_audit['immutable_inputs']:
        if digest(path)!=noise_audit['immutable_inputs'][path]:raise ValueError('noise audit source drift: '+path)
        sources.append(Path(path))
    frozen={str(p):digest(p) for p in sources}
    if OUT.exists():raise ValueError('preserve existing post cases; inspect instead of overwriting')
    parent=dict(status='POST_NOIR_INPUTS_VALIDATED_FOR_MCMC',immutable_inputs=frozen,
        main_case_count=39,common_weighted_input_P_control_count=18,p_fixed_physical=p_halo,
        physical_p_is_not_p_rec=True,B_definition=definition,noise_audit_sha256=digest(NOISE),
        covariance_is_of_mean=False,volume_Gpc_h3=1.,covariance_multiplier=1.,
        strict_independent_mock_recovery=False,full_RSD_IR_control_completed=False,
        Gaussian_or_finite_PNG_noise_caveats=noise_audit.get('remaining_scientific_caveats',[]))
    OUT.mkdir();save_json(OUT/'CONTRACT.json',parent);parent_sha=digest(OUT/'CONTRACT.json')
    for spec,arrays,meta in ready:
        directory=OUT/'cases'/spec.name;directory.mkdir(parents=True)
        np.savez_compressed(directory/'likelihood.npz',**arrays,P_k=p_k,P_nmodes=nmodes)
        meta.update(likelihood_sha256=digest(directory/'likelihood.npz'),parent_contract_sha256=parent_sha,
            physical_p_calibration_summary_sha256=digest(summary_path),noise_audit_sha256=digest(NOISE))
        save_json(directory/'contract.json',meta)
    save_json(OUT/'CASES.json',[s.name for s in planned])
    save_json(OUT/'FREEZE_AUDIT.json',dict(status='POST_NOIR_39_MAIN_AND18_SAME_TRACER_P_CONTROLS_FROZEN_NOT_SAMPLED',
        parent_contract_sha256=parent_sha,main_cases=[s.name for s in planned if s.role=='main'],
        control_cases=[s.name for s in planned if s.role!='main'],MCMC_started=False,IR_completed=False))
    print('FROZEN39_MAIN_PLUS18_SAME_TRACER_P_INPUT_CONTROLS_NOT_SAMPLED',flush=True)


if __name__=='__main__':main()
