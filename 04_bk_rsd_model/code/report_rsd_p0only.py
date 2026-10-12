"""模块大纲：Pshot 解析边缘化网格核验；三种拟合对照表；仅 PDF 图表。

P0-only 每真值只有一组，表中沿 kmax_B 重复的是同一个固定 kmax_P=.08 参照。
"""
import csv
import json
from datetime import datetime,timezone
import numpy as np
from scipy.integrate import trapezoid,cumulative_trapezoid
from scipy.special import log_ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from run_rsd_p0only import OUT,PB_OUT,PART,COSMOS,POnlyCase,parent,digest,save_json,SAMPLER_SHA


def grid_marginal(case,nf,nb):
    """Gaussian 常数 Pshot 在其原 flat bounds 内解析积分，独立核对采样后验。"""
    f=np.linspace(*case.bounds[0],nf);b=np.linspace(*case.bounds[1],nb)
    ff,bb=np.meshgrid(f,b,indexing='ij');v=np.column_stack((ff.ravel(),bb.ravel(),np.zeros(ff.size)))
    r=case.residual(v);a=float(case.nwhite@case.nwhite);h=r@case.nwhite
    mu=-h/a;lo=(case.bounds[2,0]-mu)*np.sqrt(a);hi=(case.bounds[2,1]-mu)*np.sqrt(a)
    # 反射正尾以避免两个近 1 的 CDF 相减；常数 sqrt(2pi/a) 在归一化中约去。
    reflect=lo>0;lower=np.where(reflect,-hi,lo);upper=np.where(reflect,-lo,hi)
    loghi=log_ndtr(upper);loglo=log_ndtr(lower)
    with np.errstate(divide='ignore',invalid='ignore'):
        logmass=loghi+np.log(-np.expm1(np.minimum(loglo-loghi,0.)))
    loglike=-.5*(np.sum(r*r,axis=1)-h*h/a)+logmass
    probability=np.exp(loglike-np.max(loglike)).reshape(nf,nb)
    pf=trapezoid(probability,b,axis=1);pf/=trapezoid(pf,f)
    cdf=cumulative_trapezoid(pf,f,initial=0);cdf/=cdf[-1]
    mean=float(trapezoid(f*pf,f));sigma=float(np.sqrt(trapezoid((f-mean)**2*pf,f)))
    quantiles=np.interp([.16,.5,.84],cdf,f)
    return {'grid':[nf,nb],'fNL_mean':mean,'fNL_sigma':sigma,'fNL_p16':float(quantiles[0]),
            'fNL_median':float(quantiles[1]),'fNL_p84':float(quantiles[2])}


def collect():
    parent();names=json.loads((OUT/'CASES.json').read_text());rows=[];grid_checks=[]
    with np.load(PB_OUT/'P0_ENSEMBLE_MATRIX.npz') as z:
        metadata=[json.loads(str(m)) for m in z['metadata_json']]
    if len(metadata)!=800 or not all(m['data_has_rsd'] and m['P_B_same_raw_sample'] for m in metadata):
        raise ValueError('P0 must be the same redshift-space sample used in P+B')
    for name in names:
        case=POnlyCase(OUT/'cases'/name);s=json.loads((case.directory/'SUMMARY.json').read_text())
        if not s['converged'] or not all(s['gates'].values()):raise ValueError('unconverged P0-only case '+name)
        run=json.loads((case.directory/'RUN_PROVENANCE.json').read_text())
        if run['p0only_adapter_sha256']!=digest(PART/'code/run_rsd_p0only.py') or run['sampler_sha256']!=SAMPLER_SHA:
            raise ValueError('sampler provenance drift')
        q=np.array(s['quantiles_2p5_16_50_84_97p5'])
        with np.load(case.directory/'POSTERIOR_THIN.npz') as z:thin=np.array(z['samples'])
        row={'name':name,'cosmology':s['cosmology'],'truth_fNL':s['truth_fNL'],'kmax_P':.08,
             'fNL_median':float(q[2,0]),'fNL_sigma':s['fNL_sigma'],'fNL_p16':float(q[1,0]),'fNL_p84':float(q[3,0]),
             'b1_median':float(q[2,1]),'b1_p16':float(q[1,1]),'b1_p84':float(q[3,1]),
             'Pshot_Mpc_h3_median':float(q[2,2]*case.meta['Pshot_unit_Mpc_h3']),
             'fNL_b1_correlation':float(np.corrcoef(thin[:,:2],rowvar=False)[0,1]),
             'truth_in_68pct_interval':bool(q[1,0]<=s['truth_fNL']<=q[3,0]),
             'rhat_max':float(max(s['rank_normalized_split_rhat'])),'fNL_bulk_ESS':s['bulk_ess'][0],
             'fNL_tail_ESS':s['tail_ess'][0],'steps':s['steps'],'converged':s['converged'],
             'boundary_mass_each_parameter':s['boundary_mass_each_parameter'],
             'Hartlap_factor':case.meta['Hartlap_factor'],'likelihood_sha256':case.meta['likelihood_sha256']}
        coarse=grid_marginal(case,301,251);fine=grid_marginal(case,601,501)
        change=abs(coarse['fNL_sigma']/fine['fNL_sigma']-1)
        medmove=abs(coarse['fNL_median']-fine['fNL_median'])/fine['fNL_sigma']
        mcmed=abs(row['fNL_median']-fine['fNL_median'])/fine['fNL_sigma']
        mcwidth=abs(row['fNL_sigma']/fine['fNL_sigma']-1)
        passed=change<.003 and medmove<.002 and mcmed<.06 and mcwidth<.05
        grid_checks.append({'name':name,'coarse':coarse,'fine':fine,'grid_sigma_fraction_change':change,
                            'grid_median_change_over_sigma':medmove,'MCMC_median_error_over_sigma':mcmed,
                            'MCMC_sigma_fraction_error':mcwidth,'status':'PASS' if passed else 'FAIL'})
        rows.append(row)
    save_json(OUT/'POSTERIOR_GRID_AUDIT.json',{'status':'PASS' if all(c['status']=='PASS' for c in grid_checks) else 'FAIL',
        'method':'finite-flat-Pshot analytic Gaussian integral, then normalized 2D fNL/b1 quadrature',
        'uses_truth_to_select_inputs':False,'cases':grid_checks})
    if any(c['status']!='PASS' for c in grid_checks):raise ValueError('independent posterior integration disagrees with MCMC')
    old=json.loads((PB_OUT/'RESULTS.json').read_text());lookup={r['cosmology']:r for r in rows};comparison=[]
    for r in old['case_rows']:
        p=lookup[r['cosmology']]
        comparison.append({'cosmology':r['cosmology'],'truth_fNL':r['truth_fNL'],'kmax_P':.08,'kmax_B':r['kmax_B'],
            'P0_only_fNL_median':p['fNL_median'],'P0_only_fNL_sigma':p['fNL_sigma'],
            'B_only_fNL_median':r['Bonly_fNL_median'],'B_only_fNL_sigma':r['Bonly_fNL_sigma'],
            'PB_fNL_median':r['fNL_median'],'PB_fNL_sigma':r['fNL_sigma'],
            'PB_sigma_over_P0_only':r['fNL_sigma']/p['fNL_sigma'],
            'PB_sigma_over_B_only':r['fNL_sigma_PB_over_B'],
            'sigma_reduction_PB_vs_P0_only_pct':100*(1-r['fNL_sigma']/p['fNL_sigma'])})
    return {'status':'COMPLETE','generated_UTC':datetime.now(timezone.utc).isoformat(),'P0_only_cases':rows,
            'comparison_rows':comparison,'n_new_MCMC_cases':3,'all_new_cases_converged':True,
            'P0_only_same_result_repeated_along_B_cut':True,'kmax_P':.08,'P_bins':12,'data_has_rsd':True,
            'data_mean_count':100,'covariance_sample_count':500,'covariance_is_of_mean':False,
            'covariance_multiplier':1.,'volume_Gpc_h3':1.,'parameters_P0_only':3,'parameters_B_only':7,'parameters_PB':8,
            'P0_data_model_and_shared_priors_match_PB':True,'Hartlap_P0_only':(500-12-2)/499,
            'IR':False,'FoG':False,'loops':False,'scientifically_certified':False,
            'parent_contract_sha256':digest(OUT/'contract.json'),'source_PB_results_sha256':digest(PB_OUT/'RESULTS.json'),
            'interval_convention':'posterior median +/- posterior std; actual central 68pct quantiles stored separately',
            'interpretation':'mean100+Csingle single-box-precision model check; no high-precision validity certification'}


def plot(value):
    path=OUT/'RSD_P0_B_PB_COMPARISON.pdf';qa=[]
    def page(pdf,fig,title):
        fig.suptitle('Pre-reconstruction RSD | '+title,x=.055,y=.972,ha='left',fontsize=14)
        fig.text(.055,.022,'P0 cutoff fixed at 0.08 h/Mpc. Mean100 + Csingle, V=1; no IR/FoG/loops. Error bars: posterior std.',fontsize=8)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];hidden=set()
        for ax in fig.axes:
            for axis in (ax.xaxis,ax.yaxis):
                lo,hi=sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo-1e-10<=tick.get_loc()<=hi+1e-10:hidden.update((id(tick.label1),id(tick.label2)))
        for item in fig.findobj(Text):
            if id(item) in hidden or not item.get_visible() or not item.get_text():continue
            box=item.get_window_extent(renderer)
            if box.width>0 and box.height>0 and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5):
                outside.append(item.get_text())
        if outside:raise ValueError(('PDF text outside page',outside))
        qa.append({'page':len(qa)+1,'label':title,'text_within_page':True});pdf.savefig(fig);plt.close(fig)
    with PdfPages(path) as pdf:
        fig,ax=plt.subplots(figsize=(12,8));ax.axis('off')
        columns=['truth','kmax B','RSD P0-only fNL +/- std','B-only fNL +/- std','P+B fNL +/- std','std(P+B)/std(P0)']
        cells=[[str(r['truth_fNL']),f"{r['kmax_B']:.2f}",
                f"{r['P0_only_fNL_median']:.1f} +/- {r['P0_only_fNL_sigma']:.1f}",
                f"{r['B_only_fNL_median']:.1f} +/- {r['B_only_fNL_sigma']:.1f}",
                f"{r['PB_fNL_median']:.1f} +/- {r['PB_fNL_sigma']:.1f}",f"{r['PB_sigma_over_P0_only']:.3f}"]
               for r in value['comparison_rows']]
        table=ax.table(cellText=cells,colLabels=columns,cellLoc='center',loc='center',colWidths=[.07,.10,.24,.24,.24,.16])
        table.auto_set_font_size(False);table.set_fontsize(9.5);table.scale(.95,1.6)
        fig.text(.055,.89,'Three new P0-only chains; all converged. The same P0-only fit is repeated across six B cutoffs.',fontsize=10)
        fig.text(.055,.075,'Parameters: P0-only 3; B-only 7; P+B 8. P0-only uses C_PP and a 12-bin Hartlap correction.',fontsize=9)
        fig.subplots_adjust(left=.09,right=.91,top=.86,bottom=.12);page(pdf,fig,'P0-only / B-only / P+B comparison')
        fig,axes=plt.subplots(2,3,figsize=(12,8),sharex='col')
        for j,cosmo in enumerate(COSMOS):
            rows=sorted([r for r in value['comparison_rows'] if r['cosmology']==cosmo],key=lambda r:r['kmax_B'])
            x=np.array([r['kmax_B'] for r in rows]);p=rows[0];pm=p['P0_only_fNL_median'];ps=p['P0_only_fNL_sigma']
            axes[0,j].axhspan(pm-ps,pm+ps,color='#729bc5',alpha=.18)
            axes[0,j].axhline(pm,color='#386eaa',ls=':',label='RSD P0-only (fixed)')
            for prefix,color,label,offset in [('B_only','#999999','B-only',-.001),('PB','#b45742','P+B',.001)]:
                axes[0,j].errorbar(x+offset,[r[prefix+'_fNL_median'] for r in rows],
                    yerr=[r[prefix+'_fNL_sigma'] for r in rows],fmt='o-',capsize=3,ms=4,color=color,label=label)
            axes[0,j].axhline(p['truth_fNL'],ls='--',lw=.8,color='black');axes[0,j].set_title('truth fNL = '+str(p['truth_fNL']))
            axes[0,j].set_ylabel('fNL median +/- std');axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.2)
            axes[1,j].plot(x,[r['B_only_fNL_sigma']/ps for r in rows],'o-',color='#999999',label='B-only / P0-only')
            axes[1,j].plot(x,[r['PB_sigma_over_P0_only'] for r in rows],'s-',color='#b45742',label='P+B / P0-only')
            axes[1,j].axhline(1,color='#386eaa',ls=':');axes[1,j].set_xlabel('kmax B [h/Mpc]')
            axes[1,j].set_ylabel('posterior std ratio');axes[1,j].legend(fontsize=8);axes[1,j].grid(alpha=.2)
        fig.subplots_adjust(left=.07,right=.96,top=.88,bottom=.10,wspace=.33,hspace=.3)
        page(pdf,fig,'recovery and information gain')
    save_json(OUT/'PDF_QA.json',{'status':'PASS','pages':qa,'PDF_sha256':digest(path),
              'no_PNG_generated':True,'visual_raster_inspection':False,'note':'PDF-only preference; text bounds and scientific values checked'})


def main():
    value=collect();save_json(OUT/'RESULTS.json',value)
    fields=list(value['comparison_rows'][0])
    with (OUT/'COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(value['comparison_rows'])
    lines=['# RSD P0-only、B-only、P+B 对照','',
        '三种拟合均为 redshift space、pre-reconstruction、z=0.5、LOS=z；P0 的 0 表示单极矩，仍包含 RSD。',
        'P0-only 三组新增 MCMC 全部收敛；kmax_P 固定 0.08 h/Mpc，12 点，三个参数 fNL/b1/Pshot。',
        '每种真值的同一组 P0-only 结果沿六档 kmax_B 重复，未重复计为六个独立检验。',
        '全部使用 mean100 目标、未平均的 Fid500 单盒 covariance、V=1；P0-only 使用 C_PP，Hartlap 按 12 维重新计算。',
        '表中为后验中位数 ± 标准差；实际中央 68% 分位数另存 RESULTS.json。','',
        '| 真值 | kmax_B | RSD P0-only | RSD B-only | RSD P+B | σ(P+B)/σ(P0) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in value['comparison_rows']:
        lines.append(f"| {r['truth_fNL']} | {r['kmax_B']:.2f} | {r['P0_only_fNL_median']:.1f} ± {r['P0_only_fNL_sigma']:.1f} | {r['B_only_fNL_median']:.1f} ± {r['B_only_fNL_sigma']:.1f} | {r['PB_fNL_median']:.1f} ± {r['PB_fNL_sigma']:.1f} | {r['PB_sigma_over_P0_only']:.3f} |")
    lines.extend(['','解析积分 Pshot 并对 fNL/b1 网格积分，与 MCMC 后验独立核对通过。',
        '三个真值共享相位，cutoff 之间也共享数据；这些比较并非独立的显著性检验。',
        '该结果是单盒精度下的模型检查；高 k 残余偏移、joint 宽先验检查尚缺、无 IR/FoG/loop 等限制仍保留。'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');plot(value);parent()
    save_json(OUT/'COMPLETION_AUDIT.json',{'status':'PASS','new_cases_converged':3,'old_B_and_PB_data_models_cases_results_unchanged':True,
        'same_P0_model_data_shared_priors':True,'unscaled_C_PP_recomputed_from_Fid500':True,'Hartlap_dimension':12,
        'posterior_analytic_marginalization_check':'PASS','maximum_Rhat':max(r['rhat_max'] for r in value['P0_only_cases']),
        'upstream_frozen_artifact_reuse_audit':json.loads((OUT/'UPSTREAM_REUSE_AUDIT.json').read_text()),
        'RESULTS_sha256':digest(OUT/'RESULTS.json'),'PDF_sha256':digest(OUT/'RSD_P0_B_PB_COMPARISON.pdf'),
        'no_PNG_generated':not bool(list(OUT.rglob('*.png'))),'scientifically_certified':False})
    print(json.dumps({'status':'COMPLETE','P0_only':value['P0_only_cases']}),flush=True)


if __name__=='__main__':main()
