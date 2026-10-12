"""模块大纲：对照实空间实际校准产物与RSD物理p过程，独立源/目标/profile检查。

不改主协方差、不打开活动HDF、不选p以强制fNL恢复；目标stage差异明确记录。
实际real-space profile covariance证明旧42.875缩放，而不是由盒数猜体积。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from iminuit import Minuit
from rsd_b000_contract import PART,TABLE,digest,save_json,F_GROWTH
from audit_all_available_rsd_pcal_v1 import direct_p
import run_all_available_rsd_pcal_v1 as calibration

def read(path):return json.loads(Path(path).read_text())
def profile(target,covariance,templates,fnl=100):
    chol=np.linalg.cholesky(covariance/(486/499));fits=[]
    def objective(p,b1,Pshot_residual):
        white=np.linalg.solve(chol,direct_p(fnl,b1,p,templates)+Pshot_residual-target)
        return float(white@white)
    for p in (.5,1.5,2.5):
        for b in (1.25,1.93,2.8):
            fit=Minuit(objective,p,b,0.);fit.errordef=1;fit.tol=1e-5
            for name,bound,error in zip(fit.parameters,calibration.BOUNDS,(.1,.1,1000.)):
                fit.limits[name]=bound;fit.errors[name]=error
            fit.migrad(ncall=12000)
            if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=12000)
            fit.hesse();fits.append(fit)
    valid=[x for x in fits if x.valid and x.covariance is not None and np.isfinite(x.fval)]
    if not valid:raise ValueError('independent diagnostic calibration MAP not valid')
    best=min(valid,key=lambda x:x.fval)
    return dict(physical_p_MAP=float(best.values[0]),b1_MAP=float(best.values[1]),
        Pshot_MAP=float(best.values[2]),chi2_single_box_forecast=float(best.fval),
        minuit_valid=bool(best.valid),MAP_not_posterior_median=True,no_main_p_replaced=True)

def main():
    real=PART.parent/'02_reconstruction_on_pk';root=calibration.OUT
    manifest_path=real/'inputs/p0_finite_local_png_lattice.json';manifest=read(manifest_path)
    theory_path=real/'inputs/p0_finite_local_png_lattice.npz'
    if digest(theory_path)!=manifest['output']['sha256']:raise ValueError('actual real-space theory changed')
    with np.load(theory_path,allow_pickle=False) as z:rt={k:z[k].copy() for k in z.files}
    real_matrix_path=real/'inputs/five_method_p0_matrix.npz'
    with np.load(real_matrix_path,allow_pickle=False) as z:rm={k:z[k].copy() for k in z.files}
    real_profile_path=real/'work/quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_PROFILE_ARRAYS.npz'
    with np.load(real_profile_path,allow_pickle=False) as z:historical_cov=z['sto_marisa__covariance'].copy()
    scale=rm['sto_marisa__covariance_fNL0_fid500_one_box']/historical_cov/(486/499)
    np.testing.assert_allclose(scale,42.875,rtol=3e-14,atol=0)
    pre_lattice=calibration.PRE/'P0_LATTICE.npz'
    with np.load(pre_lattice,allow_pickle=False) as z:pt={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(pt['edges'],rt['pk_edges'])
    postcal=calibration.stage_root('post')/'cases'/calibration.case_name('post')/'likelihood.npz'
    with np.load(postcal,allow_pickle=False) as z:templates=z['templates'].copy()
    report_path=calibration.STO/'ensemble_measurements/AUDIT.json';report=read(report_path)
    matrix_path=Path(report['ensemble_files']['matched']['path'])
    if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:raise ValueError('full C500 matrix changed')
    with np.load(matrix_path,allow_pickle=False) as z:
        matrix=z['matrix'].copy();cov=z['covariance_single_box'].copy()
        keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']]
    pos={key:i for i,key in enumerate(keys)};positive=[pos['LC_p',i] for i in range(100)]
    weighted100=matrix[positive,:12].mean(axis=0);post100=matrix[positive,12:24].mean(axis=0)
    with np.load(calibration.stage_root('post')/'CALIBRATION_VECTOR_MATRIX.npz',allow_pickle=False) as z:
        weighted250=z['vectors'].mean(axis=0);np.testing.assert_array_equal(z['vectors'][:100],matrix[positive,:12])
    delta=post100-weighted100
    # 对同一mode模板，f_g=0必须还原real-space代数；不要求两个历史window不同的资产相同。
    import post_rsd_p0_model as pmodel
    import rsd_pb_common as common
    old=(pmodel.F_GROWTH,common.F_GROWTH)
    try:
        pmodel.F_GROWTH=0.;common.F_GROWTH=0.
        errors=[]
        for p in (.5,1.3,2.5):
            for b in (.7,2.2,3.9):
                for fnl in (-100.,0.,100.):
                    bp=2*1.686*(b-p);t=templates
                    direct=b*b*t[0]+2*fnl*b*bp*t[3]+fnl*fnl*bp*bp*t[5]
                    got=pmodel.post_p0_linear(fnl,b,t,p_halo=p)
                    errors.append(float(np.max(abs(got-direct)/np.maximum(1,abs(direct)))))
        if max(errors)>1e-12:raise ValueError('same-window RSD real-space algebraic limit failed')
    finally:pmodel.F_GROWTH,common.F_GROWTH=old
    profiles={
        'weighted_input_positive100':profile(weighted100,cov[:12,:12],templates),
        'weighted_input_positive250_all_available':profile(weighted250,cov[:12,:12],templates),
        'post_positive100_all_current_post_measurements':profile(post100,cov[12:24,12:24],templates)}
    result=dict(status='ACTUAL_REALSPACE_RSD_PCAL_DIFFERENCES_AND_TARGET_PROFILES_RECORDED_NOT_FINAL_CALIBRATION_RELEASE',
        created_UTC=calibration.stamp(),same_calibration_parameter_order=['p','b1','Pshot_residual'],
        same_bounds=calibration.BOUNDS.tolist(),same_calibration_fNL_fixed=100.,same_P_bins=12,same_P_edges=True,
        realspace_actual_calibration_volume_scaling=float(np.median(scale)),RSD_volume_Gpc_h3=1.,
        RSD_C_single_not_divided_by_calibration_or_covariance_count=True,
        historical_realspace_calibration_target_count=100,RSD_calibration_post_weighted_input_count=250,
        RSD_calibration_pre_count_when_ready=500,
        realspace_calibration_data_stage='sto_marisa reconstructed P0',
        RSD_current_calibration_data_stage='GSR2 weighted RSD input before Marisa shift',
        target_stage_difference_under_diagnosis_not_assumed_error=True,
        positive_same100_post_minus_weighted_single_box_whitened_norm=float(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cov[12:24,12:24]),delta))),
        independent_fixed_fNL_MAP_profiles=profiles,
        realspace_measured_and_theory_modes_match=bool(np.array_equal(rm['pk_nmodes'],rt['pk_nmodes'])),
        realspace_nmodes=rt['pk_nmodes'].tolist(),RSD_nmodes=pt['nmodes'].tolist(),
        realspace_and_RSD_nmodes_equal=bool(np.array_equal(rt['pk_nmodes'],pt['nmodes'])),
        RSD_realspace_same_window_fgrowth0_algebraic_probes=len(errors),
        RSD_realspace_same_window_fgrowth0_max_scaled_error=max(errors),
        RSD_over_realspace_asset_template_ratios={name:(pt['templates'][index]/rt[name]).tolist()
            for name,index in [('PL_lattice',0),('PL_over_M_lattice',3),('Pphi_lattice',5)]},
        realspace_manifest=manifest,
        realspace_asset_preserves_reference_PL_and_PL_over_M_but_Q_from_active_table=True,
        RSD_all_six_spectra_from_same_PL_M_source=True,
        different_tables_interpolation_and_windows_must_not_be_silently_mixed=True,
        RSD_spectrum_table_sha256=digest(TABLE),RSD_spectrum_table_path=str(TABLE),
        main_volume_or_covariance_or_p_values_changed_by_audit=False,HDF_read=False,
        immutable_evidence={str(p):digest(p) for p in [Path(__file__),manifest_path,theory_path,
            real_matrix_path,real_profile_path,pre_lattice,postcal,report_path,matrix_path,TABLE]})
    save_json(root/'REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json',result)
    print(json.dumps({k:result[k] for k in ['status','realspace_actual_calibration_volume_scaling',
        'positive_same100_post_minus_weighted_single_box_whitened_norm','independent_fixed_fNL_MAP_profiles',
        'RSD_realspace_same_window_fgrowth0_max_scaled_error']}),flush=True)

if __name__=='__main__':main()
