"""模块大纲：三真值/全DM/双网格的实际mixed谱独立回放与诊断边界。

直接按mode分组求均值，不调用生产projected_spectra或残差helper；比较同一mode。
只完成实测输入和网格诊断，不用单盒谱发布STO pure-noise/connected/IR PASS。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest,save_json

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'noise_rsd_input_spectra_v1'


def read(p):return json.loads(Path(p).read_text())


def main():
    if len(os.sched_getaffinity(0))!=1:raise ValueError('one logical core required')
    dest=OUT/'GRID_AND_SPECTRA_AUDIT.json'
    if dest.exists():raise ValueError('preserve prior complete actual grid audit')
    status=read(OUT/'STATUS.json')
    if status['state']!='SIX_ACTUAL_INPUT_SPECTRA_ROWS_COMPLETE_NOT_STO_CLOSURE':
        print('NOT_READY_SAME_ORIGINAL_PROCESS_MUST_FINISH');return
    contract=read(OUT/'CONTRACT.json')
    for path,want in contract['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('input-field source drift')
    rows=[];inputs={};maximum_replay=0.;maximum_grid=0.
    for cosmo in ('fiducial','LC_p','LC_m'):
        saved={};proofs={}
        for mesh in (128,256):
            base=OUT/'rows'/f'{cosmo}_real000_m{mesh}';meta=read(base/'REPORT.json');p=base/'RAW_FIELDS_AND_SPECTRA.npz'
            if (meta['status']!='ACTUAL_INPUT_RSD_MIXED_SPECTRA_MEASURED_NOT_STO_CLOSURE'
                or meta['cosmology']!=cosmo or meta['realization']!=0 or meta['meshsize']!=mesh
                or meta['full_DM_count']!=134217728 or digest(p)!=meta['arrays_sha256']):
                raise ValueError('actual matching full-DM row missing')
            with np.load(p,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            f=arrays['normalized_fourier_modes'];labels=arrays['labels'];counts=arrays['counts']
            actual=arrays['raw_mixed_spectra'];flat=actual.reshape(6,6,-1)
            if f.shape[0]!=6 or len(labels)!=f.shape[1] or len(np.unique(arrays['integer_modes'],axis=0))!=len(labels):
                raise ValueError('unique actual Fourier mode/field geometry required')
            diag=np.stack([actual[t,t].real for t in range(6)]);scale=np.sqrt(diag[:,None]*diag[None,:])
            worst=0.
            for cell in range(len(counts)):
                mask=labels==cell
                if mask.sum()!=counts[cell]:raise ValueError('mode counts drift')
                manual=1e9*(f[:,mask]@f[:,mask].conj().T)/mask.sum()
                worst=max(worst,float(np.max(abs(manual-flat[:,:,cell])/scale.reshape(6,6,-1)[:,:,cell])))
            if worst>3e-13:raise ValueError('independent actual raw spectrum replay failed')
            n=actual[3,3].real-abs(actual[3,1])**2/actual[1,1].real
            np.testing.assert_allclose(n,arrays['operational_RSD_halo_matter_residual'],rtol=2e-13,atol=1e-8)
            nreal=actual[2,2].real-abs(actual[2,0])**2/actual[0,0].real
            np.testing.assert_allclose(nreal,arrays['operational_real_halo_matter_residual'],rtol=2e-13,atol=1e-8)
            np.testing.assert_allclose(arrays['N2_data_self']*arrays['halo_mark_self_matrix'][1,0],arrays['N3_data_self'],rtol=5e-15)
            maximum_replay=max(maximum_replay,worst);saved[mesh]=arrays;proofs[mesh]=meta
            inputs[str(p)]=digest(p);inputs[str(base/'REPORT.json')]=digest(base/'REPORT.json')
        for key in ('DM_input_dataset_hashes','full_DM_count','source_sha256','raw_fof','N2_data_self','N3_data_self'):
            if proofs[128][key]!=proofs[256][key]:raise ValueError('two grids must use identical actual DM/halo datasets')
        a=saved[128];b=saved[256];ma=a['integer_modes'];mb=b['integer_modes']
        ia=np.lexsort(ma.T[::-1]);ib=np.lexsort(mb.T[::-1]);np.testing.assert_array_equal(ma[ia],mb[ib])
        for key in ('k_edges','absolute_mu_edges','counts','field_names','halo_mark_self_matrix'):
            np.testing.assert_array_equal(a[key],b[key])
        residual=a['normalized_fourier_modes'][:,ia]-b['normalized_fourier_modes'][:,ib]
        norm=np.sqrt(np.mean(abs(b['normalized_fourier_modes'][:,ib])**2,axis=1))
        rms=np.sqrt(np.mean(abs(residual)**2,axis=1))/norm
        p=b['raw_mixed_spectra'];auto=np.stack([p[t,t].real for t in range(6)]);scale=np.sqrt(auto[:,None]*auto[None,:])
        grid=float(np.max(abs(a['raw_mixed_spectra']-p)/scale));maximum_grid=max(maximum_grid,grid)
        n=b['operational_RSD_halo_matter_residual'];dn=float(np.max(abs(a['operational_RSD_halo_matter_residual']-n)/n))
        ratio=(p[4,1]/p[3,1]).real
        low=(b['k_edges'][1:]<=.08+1e-12)
        rows.append(dict(cosmology=cosmo,realization=0,full_DM_particles_per_mesh=134217728,
            matching_mode_count=len(ia),field_relative_RMS_grid_difference=rms.tolist(),
            max_auto_normalized_mixed_spectrum_grid_difference=grid,
            max_operational_RSD_residual_relative_grid_difference=dn,
            operational_RSD_residual_range_Mpc_h3=[float(n.min()),float(n.max())],
            weight_square_vs_weight_RSD_matter_cross_ratio_low_k_range=[float(ratio[low].min()),float(ratio[low].max())],
            N2_data_self=proofs[256]['N2_data_self'],N3_data_self=proofs[256]['N3_data_self'],
            covariance_created=False,pure_stochastic_power_calibrated=False))
    report=dict(status='ACTUAL3_TRUTHS_FULL_DM_MARKED_FIELDS_AND_GRID_COMPARISON_COMPLETE_NOT_STO_CLOSURE',
        created_UTC=datetime.now(timezone.utc).isoformat(),rows=rows,immutable_inputs=inputs,
        independent_spectrum_replay_tolerance=3e-13,maximum_independent_spectrum_replay_error=maximum_replay,
        maximum_auto_normalized_grid_difference=maximum_grid,
        grid_comparison_is_diagnostic_not_likelihood_error_bound=True,
        source_and_mark_mixed_terms_are_not_replaced_by_single_N2=True,
        pure_noise_definition_and_deterministic_nonlinearity_not_separated=True,
        single_realization_per_truth_not_ensemble_validation=True,
        all44_B_estimator_windows_not_tested_here=True,connected_and_PNG_noise_validation_complete=False,
        finite_shift_or_IR_noise_validation_complete=False,formal_noise_audit_released=False,MCMC_started=False,
        source_sha256=digest(Path(__file__)),CPU_affinity=sorted(os.sched_getaffinity(0)))
    save_json(dest,report);print(json.dumps({k:v for k,v in report.items() if k!='immutable_inputs'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
