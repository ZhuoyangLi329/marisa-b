"""模块大纲：核验真实post-P校准和九组新拟合，输出4+4专属比较JSON/PDF。

只在全部实际worker/controller结束后读取证据；逐项重建mean100与C_single，
核对原先验/窗口/完整P-B cross，以及新p进入P、B和噪声模型的证据。
pre不重跑，作为明确标记的历史对照；PDF为矢量，不生成PNG。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import run_postP_calibrated_rsd_v1 as run
from post_rsd_owner_quiescence import owner_quiescent

def record(path):
    s=run.read(path); q=np.asarray(s['quantiles_2p5_16_50_84_97p5'])
    if not s['converged'] or not all(s['gates'].values()): raise ValueError('比较来源未收敛')
    return dict(name=s['name'],truth_fNL=s['truth_fNL'],fNL_median=float(q[2,0]),
        fNL_sigma=s['fNL_sigma'],fNL_quantiles=q[:,0].tolist(),parameters=s['parameters'],
        parameter_quantiles=q.tolist(),steps=s['steps'],p=s['p_fixed_physical'],
        rank_rhat_max=float(max(s['rank_normalized_split_rhat'])),
        retained_length_over_tau_min=float(min(s['length_over_tau'])),
        prior_sensitive=s['prior_sensitive'],boundary_mass=s['boundary_mass_each_parameter'],
        source=str(path),sha256=run.digest(path))

def main():
    state=run.read(run.OUT/'STATUS.json')
    if not owner_quiescent(state): raise ValueError('当前controller仍存活，不读活动链')
    if state['state']!='ALL_TEN_CHAINS_VERIFIED_COMPARISON_REPORT_REQUIRED': raise ValueError('十组未全部完成')
    settings=run.guard(); choice=run.read(run.OUT/'P_CHOICE.json'); p=choice['physical_p']
    if state['completed_cases']!=settings['case_names']: raise ValueError('完整范围未完成')
    source=run.arrays(run.MATRIX); keys=[tuple(json.loads(str(x))) for x in source['measurement_keys_json']]
    pos={k:i for i,k in enumerate(keys)}; matrix=source['matrix']; fullcov=source['covariance_single_box']
    fid=matrix[[pos['fiducial',i] for i in range(500)]]
    np.testing.assert_allclose(np.cov(fid,rowvar=False,ddof=1),fullcov,rtol=3e-13,atol=1e-7)
    tensor=run.arrays(run.TENSOR); xp=(p-1.3)/.5
    # 独立展开p的Chebyshev二次轴，不调用collapse函数。
    coeff=(tensor['coefficients'][:,0]+xp*tensor['coefficients'][:,1]+(2*xp*xp-1)*tensor['coefficients'][:,2])
    proofs=[]; rows=[]; hashes={}; main_reports=[]
    for index in range(10):
        run.verify_sources(index); case=run.case_at(index); d=case.directory
        s=run.read(d/'SUMMARY.json'); proof=run.read(d/'COMPLETION_AUDIT.json')
        if run.digest(d/'SUMMARY.json')!=proof['summary_sha256'] or not proof['converged']:
            raise ValueError('摘要与真实全链回放证据不一致')
        for j,want in enumerate(proof['chain_sha256']):
            progress=run.read(d/f'PROGRESS_ensemble_{j}.json')
            if not owner_quiescent(progress): raise ValueError('本轮writer仍存活')
            if run.digest(d/f'chain_ensemble_{j}.h5')!=want: raise ValueError('已审计链改变')
        meta=case.meta
        if (meta['volume_Gpc_h3']!=1 or meta['covariance_multiplier']!=1 or meta['covariance_is_of_mean']
            or meta['n_covariance']!=500 or meta['IR'] or meta['kmax_P']!=.08): raise ValueError('统计口径改变')
        kind,truth,cut=run.SPECS[index]
        columns=np.arange(12,24)
        if kind=='PB':
            loc=np.array([int(np.flatnonzero(source['B_indices']==i)[0]) for i in case.indices])
            columns=np.r_[columns,24+loc]
            np.testing.assert_allclose(case.coefficients,coeff[...,loc],rtol=3e-13,atol=1e-7)
            if len(case.names)!=8 or meta['noise_parameter_count']!=4 or not meta['P_B_cross_covariance']:
                raise ValueError('不是原完整4+4联合似然')
        np.testing.assert_array_equal(case.target,matrix[[pos[truth,i] for i in range(100)]][:,columns].mean(axis=0))
        np.testing.assert_array_equal(case.covariance,fullcov[np.ix_(columns,columns)])
        if index:
            if meta['p_fixed_physical']!=p or meta['n_target']!=100: raise ValueError('未采用新p或mean100')
            old=run.arrays(Path(meta['source_case'])/'likelihood.npz')
            for name in ('target','covariance','chol','templates','bounds'):
                np.testing.assert_array_equal(getattr(case,name),old[name])
            for name in (('noise',) if kind=='P0' else ('indices','weight_moments','primitives')):
                np.testing.assert_array_equal(getattr(case,name),old[name])
            new=record(d/'SUMMARY.json'); oldrecord=record(Path(meta['source_case'])/'SUMMARY.json')
            if kind=='P0':
                prepath=run.OLD/'pre_P0_only_all_available_p/cases'/(truth+'_pre_P0_all_available_p')/'SUMMARY.json'
            else:
                prepath=run.OLD/'pb_4plus3_then4plus4_all_p_v1/cases'/(truth+f'_pre_PB_4plus4_k{cut:.2f}')/'SUMMARY.json'
                oldproof=run.read(Path(meta['source_case'])/'COMPLETION_AUDIT.json')
                if oldrecord['sha256']!=oldproof['summary_sha256']: raise ValueError('旧对照摘要改变')
                preproof=run.read(prepath.parent/'COMPLETION_AUDIT.json')
                if run.digest(prepath)!=preproof['summary_sha256']: raise ValueError('pre对照摘要改变')
            pre=record(prepath)
            row=dict(statistic='P0 only' if kind=='P0' else f'P+B kB={cut:.2f}',kmax_B=cut,
                truth_fNL=meta['truth_fNL'],new_post=new,old_post=oldrecord,pre=pre,
                delta_median_new_minus_old=new['fNL_median']-oldrecord['fNL_median'],
                sigma_ratio_new_over_old=new['fNL_sigma']/oldrecord['fNL_sigma'])
            rows.append(row); main_reports.append(new)
            for r in (pre,oldrecord,new): hashes[r['source']]=r['sha256']
        elif (meta['calibration_target_count']!=100 or meta['p_calibration_block']!='P_post'
              or meta['truth_fNL']!=100): raise ValueError('不是全100个真实post-P正样本校准')
        proofs.append(proof)
        for name in ('contract.json','likelihood.npz','SUMMARY.json','COMPLETION_AUDIT.json','INPUT_AUDIT.json'):
            hashes[str(d/name)]=run.digest(d/name)
    psummary=run.read(run.directory(0)/'SUMMARY.json')
    if p!=psummary['quantiles_2p5_16_50_84_97p5'][2][0]: raise ValueError('固定p不是新后验中位数')
    for r in rows:
        p0=next(x for x in rows if x['kmax_B'] is None and x['truth_fNL']==r['truth_fNL'])
        r['new_post_sigma_over_P0']=r['new_post']['fNL_sigma']/p0['new_post']['fNL_sigma']
    dest=run.OUT/'final_comparison_v1'; dest.mkdir(exist_ok=False)
    gridpath=run.OUT/'analytic_grid_v1/AUDIT.json'
    grid=run.read(gridpath)
    if not grid['inputs_priors_covariance_and_MCMC_unmodified']: raise ValueError('独立P0积分改变输入')
    hashes[str(gridpath)]=run.digest(gridpath)
    result=dict(status='TRUE_POST_P_CALIBRATION_AND_NINE_NEW_FITS_VERIFIED_CONDITIONAL_DIAGNOSTIC',
        created_UTC=run.stamp(),calibration=choice,comparison_rows=rows,
        full_parameter_summaries=main_reports,calibration_full_summary=psummary,
        covariance='unaveraged C_single500, ddof=1, full P-B cross',volume_Gpc_h3=1.,
        main_target='mean of realizations 0..99, separately for fNL=0,+100,-100',
        post_covariance='fiducial 0..499; accepted overlap with fiducial target 0..99',
        calibration_data='all100 positive post-RSD P0 after Marisa shift; fixed fNL=100; overlaps +100 target',
        unchanged_pre='historical control; its original covariance fid1000..1499 and own calibrated p retained',
        p_uncertainty_propagated=False,IR=False,formal_STO_noise_closure_certified=False,
        changes='calibration P0 data stage and resulting fixed physical p only',
        exact_main_data_covariance_prior_window_replay=True,chain_audits=proofs,independent_P0_grid=grid,
        immutable_evidence=hashes)
    run.save_json(dest/'COMPARISON.json',result)
    make_pdf(dest,result)
    run.save_json(dest/'NUMERICAL_COMPLETION_AUDIT.json',dict(created_UTC=run.stamp(),
        ten_chains_verified=True,nine_main_cases=[r['name'] for r in main_reports],
        all_post_P0_positive100_used=True,main_mean100=True,C_single500_unaveraged=True,
        volume_Gpc_h3=1.,kmax_P=.08,kmax_B=[.08,.14],PB_physics_plus_noise=[4,4],IR=False,
        original_main_data_covariance_priors_windows_unchanged=True,p_is_new_post_calibration_median=True,
        accepted_overlap_recorded=True,old_results_preserved=True,pre_not_rerun=True,
        complete_chain_checks_and_original_sampling_gates=True,
        comparison_sha256=run.digest(dest/'COMPARISON.json'),pdf_sha256=run.digest(dest/'POST_P0_RECALIBRATED_RSD_COMPARISON.pdf'),
        visual_PDF_QA_still_required=True))
    print(json.dumps(dict(p=choice,rows=rows),ensure_ascii=False),flush=True)

def make_pdf(dest,result):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    rows=result['comparison_rows']; p=result['calibration']['physical_p']
    with PdfPages(dest/'POST_P0_RECALIBRATED_RSD_COMPARISON.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3)); fig.patch.set_facecolor('white')
        fig.text(.055,.935,'STO+Marisa RSD | post-P0 recalibration',fontsize=21,weight='bold',color='#173d58')
        fig.text(.055,.887,f'Physical p: old input-P0 1.308086  ->  new post-P0 {p:.6f}',fontsize=13)
        pq=result['calibration']['quantiles']
        fig.text(.055,.848,f'p 68% interval: [{pq[1]:.4f}, {pq[3]:.4f}] | 100 positive post-P0 realizations | calibration fNL fixed at +100')
        fig.text(.055,.815,'V = 1 (Gpc/h)^3 | mean100 data | unaveraged C_single500 | kP = 0.08 | no IR')
        ax=fig.add_axes([.055,.305,.89,.455]); ax.axis('off')
        cells=[]
        for r in rows:
            cells.append([r['statistic'],f"{r['truth_fNL']:+d}"]+[
                f"{r[v]['fNL_median']:.2f} / {r[v]['fNL_sigma']:.2f}" for v in ('pre','old_post','new_post')])
        table=ax.table(cellText=cells,colLabels=['Statistic','True fNL','Pre (unchanged)','Post: input-P0 p','Post: post-P0 p'],
            cellLoc='center',colLoc='center',colWidths=[.22,.12,.22,.22,.22],bbox=[0,0,1,1])
        table.auto_set_font_size(False); table.set_fontsize(10)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#d5dfe5'); cell.set_linewidth(.5)
            if i==0: cell.set_facecolor('#173d58'); cell.set_text_props(color='white',weight='bold')
            else: cell.set_facecolor('#edf6f5' if j==4 else ('#f3f6f8' if i%2 else 'white'))
        fig.text(.055,.265,'Table entries: fNL posterior median / posterior standard deviation (not symmetric 68% intervals).',fontsize=9)
        notes=[
            'Only 4+4 P+B results are shown; P0-only has its original three active parameters.',
            'Main post targets, full covariance, priors, reconstruction settings and binning are unchanged.',
            'Post covariance-target and calibration-target overlaps are retained and explicitly accepted.',
            'p uncertainty is not propagated. Convergence does not certify full STO noise closure or unbiased recovery.',
            'Pre retains its own p and original covariance IDs; it is a historical control, not a pure reconstruction-only test.']
        for i,line in enumerate(notes): fig.text(.055,.218-i*.029,line,fontsize=9,color='#354b58')
        fig.text(.945,.037,'1 / 2',ha='right',fontsize=9,color='#667782'); pdf.savefig(fig); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11.7,8.3)); fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.79,wspace=.3)
        fig.suptitle('Conditional fNL constraints after post-P0 calibration',x=.055,y=.94,ha='left',fontsize=19,weight='bold',color='#173d58')
        fig.text(.055,.887,'Dots: posterior medians. Error bars: central 68% credible intervals. Dashed lines: simulation truth.',fontsize=10)
        groups=[('pre','#427eb0',.18,'Pre (unchanged)'),('old_post','#a1a8ad',0,'Post / input-P0 p'),
                ('new_post','#cb5945',-.18,'Post / post-P0 p')]
        for ax,truth in zip(axes,(0,100,-100)):
            subset=[r for r in rows if r['truth_fNL']==truth]
            for group,color,offset,label in groups:
                q=np.array([r[group]['fNL_quantiles'] for r in subset]); y=np.array([2,1,0])+offset
                ax.errorbar(q[:,2],y,xerr=np.array([q[:,2]-q[:,1],q[:,3]-q[:,2]]),fmt='o',
                    color=color,markersize=5,capsize=3,lw=1.6,label=label)
            ax.axvline(truth,color='#313f4a',ls='--',lw=1); ax.set_title(f'True fNL = {truth:+d}',fontsize=12,pad=15)
            ax.set_yticks([2,1,0],['P0 only','P+B 0.08','P+B 0.14']); ax.set_ylim(-.55,2.55)
            ax.set_xlabel('fNL'); ax.grid(axis='x',alpha=.18); ax.spines[['top','right']].set_visible(False)
            if ax is not axes[0]: ax.set_yticklabels([])
        handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',
            bbox_to_anchor=(.5,.11),ncol=3,frameon=False)
        fig.text(.055,.072,'Same post data and C_single; only the calibration field and resulting fixed p changed. 4+4 P+B only.',fontsize=9)
        fig.text(.945,.037,'2 / 2',ha='right',fontsize=9,color='#667782'); pdf.savefig(fig); plt.close(fig)

if __name__=='__main__': main()
