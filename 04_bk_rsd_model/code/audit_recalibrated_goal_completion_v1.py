"""模块大纲：按用户完整目标逐项核验，不以阶段/单测通过代替全范围完成。

只在48实际链、完整对照和PDF检查齐全后通过；源/数据/先验/p逐项回放。
校准物理适用性、C_single条件解释和未传播p误差均保留，不强迫fNL无偏。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import run_recalibrated_prepost_noise_comparison_v1 as run
import run_all_available_rsd_pcal_v1 as calibration
from post_rsd_owner_quiescence import owner_quiescent
from rsd_b000_contract import digest,save_json

def read(path):return json.loads(Path(path).read_text())

def main():
    state=read(run.OUT/'STATUS.json')
    if not owner_quiescent(state):raise ValueError('actual48-case controller active; objective not complete')
    if state['state']!='ALL48_FITS_AND_FULL_CHAIN_AUDITS_COMPLETE_REPORT_REQUIRED':raise ValueError('actual full48 execution incomplete')
    settings=run.guard();final=run.OUT/'final_comparison_v1'
    comparison=read(final/'COMPARISON.json');receipt=read(final/'ARTIFACT_RECEIPT.json')
    qa=read(final/'PDF_QA.json')
    if not qa['passed'] or not qa['visual_review_complete']:raise ValueError('actual final PDF QA incomplete')
    if qa['PNG_exported'] or qa['raster_image_files_saved']:
        raise ValueError('PDF-only deliverable preference not respected')
    if digest(final/'COMPARISON.json')!=receipt['JSON_sha256']:raise ValueError('final comparison changed')
    pdf=Path(receipt['PDF'])
    if digest(pdf)!=receipt['PDF_sha256'] or digest(pdf)!=qa['PDF_sha256']:raise ValueError('actual reviewed PDF changed')
    if len(comparison['cases'])!=48 or len(comparison['comparison_rows'])!=12:raise ValueError('final report does not cover full scope')
    if len(comparison['P0_only_controls'])!=6:raise ValueError('updated P0 control comparison missing')
    rinfo=read(calibration.OUT/'ANISOTROPIC_R_INFORMATION_AUDIT.json')
    if len(rinfo['checks'])!=24 or rinfo['active_HDF_read'] or rinfo['model_prior_or_covariance_changed']:
        raise ValueError('R identifiability diagnostic incomplete or changed the frozen experiment')
    checks=[];evidence=[]
    for stage,count in [('pre',500),('post',250)]:
        root=calibration.stage_root(stage);d=root/'cases'/calibration.case_name(stage)
        meta=read(d/'contract.json');proof=read(root/'COMPLETION_AUDIT.json')
        if meta['calibration_target_count']!=count or not proof['converged']:raise ValueError('full available p calibration incomplete')
        with np.load(root/'CALIBRATION_VECTOR_MATRIX.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['realizations'],np.arange(count));vectors=z['vectors'].copy()
        with np.load(d/'likelihood.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['target'],vectors.mean(axis=0))
        corrected=read(d/'SUMMARY_METADATA_CORRECTED.json')
        if corrected['calibration_target_count']!=count or corrected['parameters'][0]!='physical_p':raise ValueError('actual p count/parameter metadata wrong')
        if settings['p_each_tracer'][stage]!=corrected['quantiles_2p5_16_50_84_97p5'][2][0]:raise ValueError('main B/P physical p not actual calibrated value')
        p0root=calibration.OUT/(stage+'_P0_only_all_available_p');p0proof=read(p0root/'COMPLETION_AUDIT.json')
        if not p0proof['all_three_converged'] or len(p0proof['checks'])!=3:raise ValueError('updated P0 controls incomplete')
        for row in p0proof['checks']:
            p=p0root/'cases'/row['name']/'SUMMARY.json'
            if digest(p)!=row['summary_sha256']:raise ValueError('actual P0 summary drift')
        checks.append(dict(requirement=f'all-available {stage} p and updated3 P0 controls',status='proven',
            calibration_count=count,C_single=True,p_fixed=settings['p_each_tracer'][stage]))
        evidence.extend([d/'contract.json',d/'SUMMARY_METADATA_CORRECTED.json',root/'COMPLETION_AUDIT.json',p0root/'COMPLETION_AUDIT.json'])
    process=read(calibration.OUT/'REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json')
    if (not process['same_P_edges'] or process['RSD_realspace_same_window_fgrowth0_max_scaled_error']>1e-12
        or process['realspace_actual_calibration_volume_scaling']!=42.875):raise ValueError('actual real-space/RSD procedure comparison incomplete')
    grid=read(calibration.OUT/'analytic_pcal_p0_grid_v1/AUDIT.json')
    if len(grid['checks'])!=8 or not grid['no_p_adjustment_to_force_recovery']:raise ValueError('independent p/P0 likelihood grid missing')
    sign=read(calibration.OUT/'P_CALIBRATION_SIGN_RANGE_DIAGNOSTIC.json')
    if len(sign['matched100_controls'])!=18 or len(sign['all_available_positive_controls'])!=6 or not sign['main_p_not_changed']:
        raise ValueError('sign/range p diagnostic or no-retuning invariant incomplete')
    template=read(calibration.OUT/'pre_b_paware_templates_v1/COMPLETION_AUDIT.json')
    if template['actual_bin_count']!=27 or not template['old_p_baseline_reproduced'] or template['full_joint_whitened_max_probe_error']>1e-5:
        raise ValueError('new physical-p pre B source not fully verified')
    checks.append(dict(requirement='compare p process against real space, repair actual metadata and inherited pre-p usage, validate explicit p in P/B',
        status='proven_under_existing_user_authorized_Csingle_configuration',
        precision_change_not_silently_adopted=True,physical_unbiasedness_not_certified=True))
    evidence.extend([calibration.OUT/'REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json',calibration.OUT/'analytic_pcal_p0_grid_v1/AUDIT.json',
        calibration.OUT/'P_CALIBRATION_SIGN_RANGE_DIAGNOSTIC.json',calibration.OUT/'pre_b_paware_templates_v1/COMPLETION_AUDIT.json'])
    for noise in (3,4):
        phase=read(run.OUT/f'PHASE_4PLUS{noise}_COMPLETION.json')
        if not phase['all_converged'] or phase['actual_cases']!=24 or len(phase['audits'])!=24:raise ValueError('full24-case required phase incomplete')
        for index in range((noise-3)*24,(noise-2)*24):
            d=run.directory(index);meta=read(d/'contract.json');audit=read(d/'COMPLETION_AUDIT.json')
            if meta['noise_parameter_count']!=noise or meta['total_parameter_count']!=4+noise or not audit['converged']:
                raise ValueError('actual requested parameter count or converged chain missing')
            if not audit['full_HDF_prior_gate_quantile_sigma_and_thin_RNG_replayed'] or audit['stored_logprob_independent_spots']!=512:
                raise ValueError('chain audit coverage weaker than required')
            if digest(d/'SUMMARY.json')!=audit['summary_sha256']:raise ValueError('actual audited summary changed')
            for ensemble,want in enumerate(audit['chain_sha256']):
                if digest(d/f'chain_ensemble_{ensemble}.h5')!=want:raise ValueError('actual audited HDF changed')
            if meta['volume_Gpc_h3']!=1 or meta['covariance_is_of_mean'] or meta['n_covariance']!=500 or meta['n_target']!=100 or meta['IR']:
                raise ValueError('mean100/Csingle500/V1/noIR setting changed')
            evidence.extend([d/'contract.json',d/'COMPLETION_AUDIT.json',d/'SUMMARY.json'])
        checks.append(dict(requirement=f'all24 4+{noise} pre/post, three truths, four B cuts',status='proven',actual_converged_cases=24))
        evidence.append(run.OUT/f'PHASE_4PLUS{noise}_COMPLETION.json')
    for index in range(24):
        left=run.directory(index);right=run.directory(index+24)
        with np.load(left/'likelihood.npz',allow_pickle=False) as a,np.load(right/'likelihood.npz',allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key][:7] if key=='bounds' else b[key])
            np.testing.assert_array_equal(b['bounds'][7],[-10,10])
        old=read(left/'contract.json');new=read(right/'contract.json')
        if new['parameters'][:7]!=old['parameters'] or new['parameters'][7]!='R_stoch_anisotropic' or old['p_fixed_physical']!=new['p_fixed_physical']:
            raise ValueError('4+4 is not exactly the matched7 parameters plus independent anisotropic R')
    first4=min(read(run.directory(i)/'COMPLETION_AUDIT.json')['created_UTC'] for i in range(24,48))
    end3=read(run.OUT/'PHASE_4PLUS3_COMPLETION.json')['created_UTC']
    if not end3<first4:raise ValueError('extra4-noise stage not ordered after all3-noise phase')
    checks.extend([dict(requirement='same data/C/p/signal and original7 priors, exactly one extra R, all4+3 before4+4',status='proven'),
        dict(requirement='full numerical comparison and PDF-only deliverable with visual QA',status='proven',comparison_rows=12,PDF_pages=qa['pages'],PNG_export=False)])
    evidence.extend([run.OUT/'SETTINGS.json',final/'COMPARISON.json',final/'ARTIFACT_RECEIPT.json',final/'PDF_QA.json',
        calibration.OUT/'ANISOTROPIC_R_INFORMATION_AUDIT.json'])
    result=dict(created_UTC=run.stamp(),status='USER_FULL_P_REVIEW_THEN_ALL_4PLUS3_THEN_ALL_4PLUS4_OBJECTIVE_COMPLETION_PROVEN',
        objective='仔细检查p校准与real-space差异并修复；全部4+3后加入独立各向异性噪声重跑4+4',
        requirements=checks,all48_source_and_chain_hashes_current=True,p_calibration_uncertainty_propagated=False,
        conditional_diagnostic_not_formal_noise_or_unbiased_kmax_certification=True,
        centered_R_has_zero_continuum_B000_monopole_and_weak_actual_lattice_response=True,
        no_old_p_or_old12_parameter_chain_substitution=True,original_products_preserved=True,
        source_evidence_sha256={str(p):digest(p) for p in dict.fromkeys(evidence)},auditor_sha256=digest(__file__))
    save_json(final/'GOAL_COMPLETION_AUDIT.json',result);print(json.dumps(result['requirements']),flush=True)

if __name__=='__main__':main()
