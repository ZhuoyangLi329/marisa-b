"""模块大纲：九组p=1.2822真实RSD链与上一轮post-RSD p结果的比较。

逐项核对data/C/先验和新p信号轴，保存全部参数及双页PDF。
比较只包含P0-only、4+4 P+B；绘图不生成PNG。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import run_realspace_p_rsd_sensitivity_v1 as run
from post_rsd_owner_quiescence import owner_quiescent
from report_postP_calibrated_rsd_v1 import record

def main():
    c=run.core;state=c.read(run.OUT/'STATUS.json')
    if not owner_quiescent(state) or state['state']!='ALL_NINE_CHAINS_VERIFIED_COMPARISON_REPORT_REQUIRED':
        raise ValueError('等九组实际完成且controller退出')
    settings=c.guard()
    if state['completed_cases']!=settings['case_names']:raise ValueError('九组范围不完整')
    ensemble=c.arrays(c.MATRIX);keys=[tuple(json.loads(str(x))) for x in ensemble['measurement_keys_json']]
    pos={k:i for i,k in enumerate(keys)};matrix=ensemble['matrix'];cov=ensemble['covariance_single_box']
    np.testing.assert_allclose(np.cov(matrix[[pos['fiducial',i] for i in range(500)]],rowvar=False,ddof=1),
        cov,rtol=3e-13,atol=1e-7)
    tensor=c.arrays(c.TENSOR);x=(run.PHYSICAL_P-1.3)/.5
    collapsed=tensor['coefficients'][:,0]+x*tensor['coefficients'][:,1]+(2*x*x-1)*tensor['coefficients'][:,2]
    rows=[];proofs=[];sources=[Path(__file__),run.OUT/'P_CHOICE.json',run.OUT/'SETTINGS.json']
    for index in range(1,10):
        d=run.directory(index);c.verify_sources(index);case=c.case_at(index);meta=case.meta
        proof=c.read(d/'COMPLETION_AUDIT.json')
        if not proof['converged'] or c.digest(d/'SUMMARY.json')!=proof['summary_sha256']:raise ValueError('链证据改变')
        for j,want in enumerate(proof['chain_sha256']):
            if not owner_quiescent(c.read(d/f'PROGRESS_ensemble_{j}.json')):raise ValueError('writer仍活动')
            if c.digest(d/f'chain_ensemble_{j}.h5')!=want:raise ValueError('链已改变')
        src=Path(meta['source_case']);old=c.arrays(src/'likelihood.npz')
        for name in ('target','covariance','chol','templates','bounds'):
            np.testing.assert_array_equal(getattr(case,name),old[name])
        columns=np.arange(12,24);kind,truth,cut=c.SPECS[index]
        if kind=='PB':
            loc=np.array([int(np.flatnonzero(ensemble['B_indices']==i)[0]) for i in case.indices])
            columns=np.r_[columns,24+loc]
            np.testing.assert_allclose(case.coefficients,collapsed[...,loc],rtol=3e-13,atol=1e-7)
            for name in ('primitives','weight_moments','indices'):np.testing.assert_array_equal(getattr(case,name),old[name])
            if len(case.names)!=8 or meta['noise_parameter_count']!=4 or not meta['P_B_cross_covariance']:raise ValueError('参数或cross改变')
        else:np.testing.assert_array_equal(case.noise,old['noise'])
        np.testing.assert_array_equal(case.target,matrix[[pos[truth,i] for i in range(100)]][:,columns].mean(axis=0))
        np.testing.assert_array_equal(case.covariance,cov[np.ix_(columns,columns)])
        if (meta['p_fixed_physical']!=run.PHYSICAL_P or meta['volume_Gpc_h3']!=1 or meta['n_covariance']!=500
            or meta['n_target']!=100 or meta['covariance_multiplier']!=1 or meta['covariance_is_of_mean']
            or meta['IR'] or meta['kmax_P']!=.08):raise ValueError('统计口径改变')
        new=record(d/'SUMMARY.json');oldrecord=record(src/'SUMMARY.json')
        if oldrecord['sha256']!=c.read(src/'COMPLETION_AUDIT.json')['summary_sha256']:raise ValueError('基线摘要改变')
        rows.append(dict(statistic='P0 only' if kind=='P0' else f'P+B kB={cut:.2f}',kmax_B=cut,
            truth_fNL=meta['truth_fNL'],post_RSD_p=oldrecord,realspace_p=new,
            delta_median=new['fNL_median']-oldrecord['fNL_median'],sigma_ratio=new['fNL_sigma']/oldrecord['fNL_sigma']))
        proofs.append(proof);sources += [d/'SUMMARY.json',d/'contract.json',d/'COMPLETION_AUDIT.json',
            d/'INPUT_AUDIT.json',src/'SUMMARY.json']
    gridpath=run.OUT/'analytic_P0_grid_v1/AUDIT.json';grid=c.read(gridpath);sources.append(gridpath)
    result=dict(status='NINE_FIXED_REALSPACE_P_RSD_FITS_AND_BASELINE_COMPARISON_VERIFIED',
        created_UTC=c.stamp(),p_choice=c.read(run.OUT/'P_CHOICE.json'),comparison_rows=rows,
        main_data_mean_count=100,n_covariance=500,C_single=True,volume_Gpc_h3=1.,kmax_P=.08,kmax_B=[.08,.14],
        physics_plus_noise_parameters=[4,4],p_uncertainty_propagated=False,IR=False,
        data_C_priors_windows_noise_model_same_as_previous_RSD=True,
        full_P_B_cross_covariance=True,accepted_fiducial_covariance_target_overlap_ids=list(range(100)),
        realspace_calibration_and_RSD_targets_not_claimed_independent=True,
        full_STO_noise_closure_certified=False,all_nine_original_gates_pass=True,
        independent_P0_grid=grid,chain_audits=proofs,
        sources_sha256={str(p):c.digest(p) for p in dict.fromkeys(sources)})
    dest=run.OUT/'final_comparison_v1';dest.mkdir(exist_ok=False)
    c.save_json(dest/'COMPARISON.json',result);make_pdf(dest,result)
    c.save_json(dest/'NUMERICAL_COMPLETION_AUDIT.json',dict(nine_cases_verified=True,
        original_data_C_priors_windows_and_models_verified=True,fixed_physical_p=run.PHYSICAL_P,
        p_in_P_B_signal_and_noise_independently_verified=True,main_C_single500_V1_mean100=True,
        P0_then_4plus4_PB_two_cuts_complete=True,all_original_convergence_gates_and_full_chain_replays=True,
        comparison_sha256=c.digest(dest/'COMPARISON.json'),pdf_sha256=c.digest(dest/'REALSPACE_P_RSD_COMPARISON.pdf'),
        visual_PDF_QA_required=True))
    print('REALSPACE_P_COMPARISON_READY',dest,flush=True)

def make_pdf(dest,result):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    rows=result['comparison_rows'];pold=result['p_choice']['baseline_RSD_p']
    with PdfPages(dest/'REALSPACE_P_RSD_COMPARISON.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.patch.set_facecolor('white')
        fig.text(.055,.935,'STO+Marisa RSD | real-space p sensitivity',fontsize=21,weight='bold',color='#173d58')
        fig.text(.055,.887,f'Fixed physical p: post-RSD calibration {pold:.6f}  ->  real-space calibrated value 1.282200',fontsize=12)
        fig.text(.055,.845,'V = 1 (Gpc/h)^3 | mean100 data | unaveraged C_single500 | full P-B cross | kP = 0.08 | no IR')
        ax=fig.add_axes([.055,.30,.89,.47]);ax.axis('off')
        cells=[[r['statistic'],f"{r['truth_fNL']:+d}",
            f"{r['post_RSD_p']['fNL_median']:.2f} / {r['post_RSD_p']['fNL_sigma']:.2f}",
            f"{r['realspace_p']['fNL_median']:.2f} / {r['realspace_p']['fNL_sigma']:.2f}",
            f"{r['delta_median']:+.2f}"] for r in rows]
        table=ax.table(cellText=cells,colLabels=['Statistic','True fNL','p = 1.307926','p = 1.282200','Median change'],
            cellLoc='center',colLoc='center',colWidths=[.23,.12,.25,.25,.15],bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(10)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#d5dfe5');cell.set_linewidth(.5)
            if i==0:cell.set_facecolor('#173d58');cell.set_text_props(color='white',weight='bold')
            else:cell.set_facecolor('#edf6f5' if j==3 else ('#f3f6f8' if i%2 else 'white'))
        fig.text(.055,.26,'Entries: fNL posterior median / posterior standard deviation; central 68% intervals are plotted on page 2.',fontsize=9)
        notes=['Only fixed physical p changes; nine main cases use the previous RSD data, covariance, priors and binning.',
            'P0-only keeps its three active parameters; P+B keeps the same four physical plus four noise parameters.',
            'Real-space p is imported from the value actually used in real-space p-aware inference.',
            'p uncertainty is not propagated. Accepted covariance-target overlap is retained.',
            'This is a sensitivity comparison; sampler convergence does not certify unbiased recovery or complete STO noise closure.']
        for i,line in enumerate(notes):fig.text(.055,.213-i*.03,line,fontsize=9,color='#354b58')
        fig.text(.945,.037,'1 / 2',ha='right',fontsize=9,color='#667782');pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11.7,8.3));fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.79,wspace=.3)
        fig.suptitle('RSD fNL constraints with real-space calibrated p',x=.055,y=.94,ha='left',fontsize=19,weight='bold',color='#173d58')
        fig.text(.055,.887,'Dots: posterior medians. Bars: central 68% credible intervals. Dashed lines: simulation truth.',fontsize=10)
        for ax,truth in zip(axes,(0,100,-100)):
            subset=[r for r in rows if r['truth_fNL']==truth]
            for key,color,offset,label in [('post_RSD_p','#959da3',.12,'Post-RSD p = 1.307926'),
                ('realspace_p','#cb5945',-.12,'Real-space p = 1.282200')]:
                q=np.array([r[key]['fNL_quantiles'] for r in subset]);y=np.array([2,1,0])+offset
                ax.errorbar(q[:,2],y,xerr=np.array([q[:,2]-q[:,1],q[:,3]-q[:,2]]),fmt='o',
                    color=color,markersize=5,capsize=3,lw=1.6,label=label)
            ax.axvline(truth,color='#313f4a',ls='--',lw=1);ax.set_title(f'True fNL = {truth:+d}',fontsize=12,pad=15)
            ax.set_yticks([2,1,0],['P0 only','P+B 0.08','P+B 0.14']);ax.set_ylim(-.5,2.5);ax.set_xlabel('fNL')
            ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
            if ax is not axes[0]:ax.set_yticklabels([])
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',
            bbox_to_anchor=(.5,.11),ncol=2,frameon=False)
        fig.text(.055,.072,'Only fixed physical p changed; all remaining RSD inference settings are the same.',fontsize=9)
        fig.text(.945,.037,'2 / 2',ha='right',fontsize=9,color='#667782');pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':main()
