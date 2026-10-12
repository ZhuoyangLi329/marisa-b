"""模块大纲：48组真实链/来源一致性回执、完整4+3/4+4对照JSON与单一PDF。

只在真实controller退出且两个24-case阶段全链审计完后生成；不读活动HDF。
PDF完全矢量，不输出PNG；std是后验标准差，p不确定度没有传播。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json
import numpy as np
import run_recalibrated_prepost_noise_comparison_v1 as run
from post_rsd_owner_quiescence import owner_quiescent
from rsd_b000_contract import digest,save_json

def main():
    state=run.read(run.OUT/'STATUS.json')
    if not owner_quiescent(state):raise ValueError('actual comparison controller active; report not ready')
    if state['state']!='ALL48_FITS_AND_FULL_CHAIN_AUDITS_COMPLETE_REPORT_REQUIRED':raise ValueError('all48 actual fits not finished')
    settings=run.guard();dest=run.OUT/'final_comparison_v1'
    if dest.exists():raise ValueError('existing final comparison preserved; never overwrite')
    for noise in (3,4):
        phase=run.read(run.OUT/f'PHASE_4PLUS{noise}_COMPLETION.json')
        if not phase['all_converged'] or phase['actual_cases']!=24:raise ValueError('required24-case phase not converged')
    r_information_path=run.calibration.OUT/'ANISOTROPIC_R_INFORMATION_AUDIT.json'
    r_information=run.read(r_information_path)
    if len(r_information['checks'])!=24 or r_information['model_prior_or_covariance_changed']:
        raise ValueError('R information diagnostic incomplete or changed the comparison')
    process_path=run.calibration.OUT/'REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json'
    process=run.read(process_path)
    process_keys=('same_calibration_parameter_order','same_bounds','same_calibration_fNL_fixed',
        'same_P_bins','same_P_edges','realspace_actual_calibration_volume_scaling','RSD_volume_Gpc_h3',
        'realspace_calibration_data_stage','RSD_current_calibration_data_stage',
        'target_stage_difference_under_diagnosis_not_assumed_error',
        'RSD_realspace_same_window_fgrowth0_max_scaled_error',
        'different_tables_interpolation_and_windows_must_not_be_silently_mixed')
    records=[];sources=[run.OUT/'SETTINGS.json',run.OUT/'RESULTS.json',r_information_path,process_path]
    calibrations={};p0_controls=[]
    for stage in ('pre','post'):
        root=run.calibration.stage_root(stage)
        calibrated=root/'cases'/run.calibration.case_name(stage)/'SUMMARY_METADATA_CORRECTED.json'
        psummary=run.read(calibrated)
        calibrations[stage]=dict(actual_count=psummary['calibration_target_count'],
            physical_p_quantiles_2p5_16_50_84_97p5=np.asarray(psummary['quantiles_2p5_16_50_84_97p5'])[:,0].tolist(),
            physical_p_sigma=psummary['physical_p_sigma'],C_single=True,
            uncertainty_propagated_into_main_fits=False)
        sources.extend([calibrated,root/'COMPLETION_AUDIT.json'])
        p0root=run.calibration.OUT/(stage+'_P0_only_all_available_p')
        proof=run.read(p0root/'COMPLETION_AUDIT.json')
        if not proof['all_three_converged']:raise ValueError('updated P0 controls incomplete')
        sources.append(p0root/'COMPLETION_AUDIT.json')
        for item in proof['checks']:
            source=p0root/'cases'/item['name']/'SUMMARY.json';summary=run.read(source)
            if digest(source)!=item['summary_sha256']:raise ValueError('actual P0 control summary changed')
            p0_controls.append(dict(stage=stage,truth_fNL=summary['truth_fNL'],name=item['name'],
                fNL_median=item['fNL_quantiles'][2],fNL_sigma=item['fNL_sigma'],
                fNL_quantiles_2p5_16_50_84_97p5=item['fNL_quantiles'],
                p_fixed_physical=item['p_fixed_physical'],kmax_P=.08,total_parameters=3,
                summary_sha256=item['summary_sha256']))
            sources.append(source)
    for index,(noise,cut,truth,stage) in enumerate(run.SPECS):
        d=run.directory(index);meta=run.read(d/'contract.json');audit=run.read(d/'COMPLETION_AUDIT.json')
        summary=run.read(d/'SUMMARY.json')
        if digest(d/'SUMMARY.json')!=audit['summary_sha256'] or not audit['converged'] or not summary['converged']:
            raise ValueError('actual completion evidence changed or failed')
        for ensemble,want in enumerate(audit['chain_sha256']):
            if digest(d/f'chain_ensemble_{ensemble}.h5')!=want:raise ValueError('actual audited chain changed after completion')
        q=np.asarray(summary['quantiles_2p5_16_50_84_97p5'])[:,0];sigma=float(summary['fNL_sigma'])
        records.append(dict(index=index,name=d.name,stage=stage,truth_fNL=meta['truth_fNL'],noise_parameters=noise,
            total_parameters=4+noise,kmax_B=cut,kmax_P=.08,p_fixed_physical=meta['p_fixed_physical'],
            fNL_quantiles_2p5_16_50_84_97p5=q.tolist(),fNL_median=float(q[2]),fNL_sigma=sigma,
            median_minus_truth=float(q[2]-meta['truth_fNL']),posterior_pull=float((q[2]-meta['truth_fNL'])/sigma),
            steps_each_ensemble=audit['steps_each_ensemble'],rank_rhat_max=float(max(summary['rank_normalized_split_rhat'])),
            retained_length_over_tau_min=float(min(summary['length_over_tau'])),prior_sensitive=bool(summary['prior_sensitive']),
            boundary_mass_each_parameter=summary['boundary_mass_each_parameter'],n_covariance=500,n_target=100,
            parameters=meta['parameters'],all_parameter_quantiles_2p5_16_50_84_97p5=summary['quantiles_2p5_16_50_84_97p5'],
            R_quantiles_2p5_16_50_84_97p5=None if noise==3 else np.asarray(summary['quantiles_2p5_16_50_84_97p5'])[:,7].tolist(),
            C_single=True,volume_Gpc_h3=1.,summary_sha256=audit['summary_sha256'],
            completion_audit_sha256=digest(d/'COMPLETION_AUDIT.json')))
        sources.extend([d/'contract.json',d/'SUMMARY.json',d/'COMPLETION_AUDIT.json'])
    rows=[]
    for truth in (0,100,-100):
        for cut in (.08,.10,.12,.14):
            row=dict(truth_fNL=truth,kmax_B=cut)
            for stage in ('pre','post'):
                for noise in (3,4):
                    item=next(x for x in records if x['truth_fNL']==truth and x['kmax_B']==cut and x['stage']==stage and x['noise_parameters']==noise)
                    row[f'{stage}_4plus{noise}']=item
                row[stage+'_sigma_4plus4_over_4plus3']=row[f'{stage}_4plus4']['fNL_sigma']/row[f'{stage}_4plus3']['fNL_sigma']
                p0=next(x for x in p0_controls if x['stage']==stage and x['truth_fNL']==truth)
                row[stage+'_P0_only']=p0
                for noise in (3,4):
                    row[f'{stage}_sigma_4plus{noise}_over_P0']=row[f'{stage}_4plus{noise}']['fNL_sigma']/p0['fNL_sigma']
            rows.append(row)
    sources.extend([Path(__file__),run.OUT/'PHASE_4PLUS3_COMPLETION.json',run.OUT/'PHASE_4PLUS4_COMPLETION.json'])
    dest.mkdir()
    result=dict(created_UTC=run.stamp(),status='ALL48_PCAL_UPDATED_MATCHED_NOISE_PARAMETER_COMPARISONS_VERIFIED_CONDITIONAL_DIAGNOSTIC',
        cases=records,comparison_rows=rows,physical_p_calibrations=calibrations,P0_only_controls=p0_controls,
        same_data_C_p_signal_and_original7priors_in_both_noise_versions=True,
        added_parameter_only='R_stoch_anisotropic',new_parameter_changes_P0=False,
        p_each_tracer=settings['p_each_tracer'],p_calibration_precision_kept_Csingle=True,
        p_calibration_uncertainty_propagated=False,mean_data_count=100,C_single=True,n_covariance=500,
        volume_Gpc_h3=1.,IR=False,strict_independent_recovery_claimed=False,
        full_marked_STO_stochastic_closure_certified=False,unbiased_fNL_or_valid_kmax_certified=False,
        prior_sensitive_cases=[x['name'] for x in records if x['prior_sensitive']],
        anisotropic_R_information=r_information,
        p_calibration_process_comparison={key:process[key] for key in process_keys},
        resolved_implementation_issues=['Pre no longer inherits real-space p=1.179.',
            'Pre B coefficients now depend explicitly on the recalibrated physical p, not metadata alone.',
            'Actual calibration stage/count replace the legacy hardcoded150 summary metadata. Historical chains and raw summaries are preserved.'],
        uniform_R_prior_expected_boundary_mass=0.04,
        R_boundary_mass_alone_is_not_fNL_prior_dependence_evidence=True,
        source_sha256={str(p):digest(p) for p in sources})
    save_json(dest/'COMPARISON.json',result)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.unicode_minus':False})
    output=dest/'RSD_PB_4PLUS3_VS_4PLUS4_ALLP.pdf'
    styles={('pre',3):('#2166ac','-','o'),('pre',4):('#2166ac','--','s'),
        ('post',3):('#d95f02','-','o'),('post',4):('#d95f02','--','s')}
    with PdfPages(output,metadata={'Title':'RSD P0+B000: recalibrated p, 4+3 versus 4+4',
            'Subject':'Conditional single-box forecast; legacy stochastic models; no IR; no PNG export'}) as pdf:
        fig,axes=plt.subplots(3,2,figsize=(11.69,8.27));handles=[];labels=[]
        for row,truth in enumerate((0,100,-100)):
            for stage in ('pre','post'):
                for noise in (3,4):
                    items=sorted([x for x in records if x['truth_fNL']==truth and x['stage']==stage and x['noise_parameters']==noise],key=lambda x:x['kmax_B'])
                    color,line,marker=styles[stage,noise];x=[a['kmax_B'] for a in items]
                    h,=axes[row,0].plot(x,[a['median_minus_truth'] for a in items],color=color,linestyle=line,marker=marker)
                    axes[row,1].plot(x,[a['fNL_sigma'] for a in items],color=color,linestyle=line,marker=marker)
                    if row==0:handles.append(h);labels.append(('Pre' if stage=='pre' else 'STO+Marisa post')+f' 4+{noise}')
            axes[row,0].axhline(0,color='.5',linewidth=.8)
            axes[row,0].set_ylabel(r'$\mathrm{median}(f_{\rm NL})-f_{\rm NL}^{\rm input}$')
            axes[row,1].set_ylabel(r'$\sigma(f_{\rm NL})$ (posterior std)')
            for ax in axes[row]:
                ax.set_title(f'Input fNL = {truth:+d}');ax.grid(alpha=.2);ax.set_xticks([.08,.10,.12,.14])
                ax.set_xlabel(r'$k_{\max}^{B}\ [h\,\mathrm{Mpc}^{-1}]$')
        fig.suptitle('Recalibrated physical p: matched 4+3 and 4+4 noise comparison',y=.98,fontsize=13)
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.947),ncol=4,fontsize=9)
        fig.subplots_adjust(top=.86,bottom=.11,left=.095,right=.96,hspace=.55,wspace=.32)
        fig.text(.5,.028,'Mean100 data; unaveraged C_single500; V=1; P0 k_max=.08; B000 two-leg cuts; no IR.\n'
            'Fixed p; calibration uncertainty not propagated. Conditional diagnostics, not full STO-noise or k_max certification.',ha='center',fontsize=8)
        pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(11.69,8.27));ax=fig.add_axes([.045,.17,.91,.66]);ax.axis('off')
        headings=['Input','kB','Pre 4+3\nmedian / std','Post 4+3\nmedian / std','Pre 4+4\nmedian / std','Post 4+4\nmedian / std','Pre\nstd8/std7','Post\nstd8/std7']
        values=[]
        for row in rows:
            items=[row[k] for k in ['pre_4plus3','post_4plus3','pre_4plus4','post_4plus4']]
            values.append([str(row['truth_fNL']),f"{row['kmax_B']:.2f}"]+
                [f"{a['fNL_median']:.2f} / {a['fNL_sigma']:.2f}" for a in items]+
                [f"{row['pre_sigma_4plus4_over_4plus3']:.3f}",f"{row['post_sigma_4plus4_over_4plus3']:.3f}"])
        table=ax.table(cellText=values,colLabels=headings,loc='center',cellLoc='center',
            colWidths=[.06,.06,.165,.165,.165,.165,.11,.11])
        table.auto_set_font_size(False);table.set_fontsize(8.6);table.scale(1,2.0)
        for (r,c),cell in table.get_celld().items():
            if r==0:cell.set_facecolor('#dde9f2');cell.set_text_props(weight='bold')
            elif ((r-1)//4)%2:cell.set_facecolor('#f4f6f8')
        fig.text(.5,.945,'Complete numerical comparison - posterior standard deviations, not symmetric 68% errors',ha='center',fontsize=12)
        fig.text(.5,.87,f"Physical p: pre={settings['p_each_tracer']['pre']:.6f}, post={settings['p_each_tracer']['post']:.6f}; exactly one added R amplitude",ha='center',fontsize=10)
        fig.text(.06,.105,'All 48 fits: four independent ensembles, 32 walkers; original convergence gates and full-chain replay verified.\n'
            'Same targets, signal, full P-B covariance, fixed p and original seven priors between noise versions.\n'
            'Pre C: fid1000-1499; post C: fid0-499. Fid target/post-C overlap and calibration-target overlap accepted.\n'
            'Centered angular R has zero continuum monopole; its R=-10 to +10 lattice response is at most 0.0375 whitened units.\n'
            'R remains near its uniform prior; this alone is not fNL prior sensitivity. Full parameter quantiles are in the JSON.\n'
            'Old-p and older 12-parameter products are preserved separately; this table is not an independent mock-recovery claim.',fontsize=8.6,va='top')
        pdf.savefig(fig);plt.close(fig)
    save_json(dest/'ARTIFACT_RECEIPT.json',dict(created_UTC=run.stamp(),PDF=str(output),PDF_sha256=digest(output),
        JSON_sha256=digest(dest/'COMPARISON.json'),PDF_pages_expected=2,PNG_generated=False,
        actual48_chain_hashes_verified=True,PDF_visual_QA_status_at_creation='pending; authoritative completed QA is in PDF_QA.json'))
    print(json.dumps(dict(status=result['status'],PDF=str(output),JSON=str(dest/'COMPARISON.json'))),flush=True)

if __name__=='__main__':main()
