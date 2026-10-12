"""模块大纲：真实链状态/有效范围候选；CSV/JSON；单份多页 PDF 科学诊断。

缺失或未收敛结果明确标注，不插值补造；PDF 仅保存矢量图，不输出 PNG。
"""
import argparse
import csv
import json
import textwrap
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from rsd_b000_contract import OUT,CUTS,digest,save_json,selected
from run_rsd_b000_inference import predicted

COLORS={'fiducial':'#305f9e','LC_p':'#b85b35','LC_m':'#538251'}
COSMOS=('fiducial','LC_p','LC_m')
TRUTHS={'fiducial':0,'LC_p':100,'LC_m':-100}
TITLE='Pre-reconstruction RSD B000-only validation'

def load(path):return json.loads(path.read_text()) if path.exists() else None

def baseline_names():
    contract=load(OUT/'contract.json') or {}
    if 'baseline_case_names' in contract:return contract['baseline_case_names']
    return [f'{c}_{v}_k{k:.2f}' for c in COSMOS for v in ('mean100','single000')
            for k in (CUTS if v=='mean100' else (.12,))]

def collect():
    names=load(OUT/'CASES.json') or baseline_names();rows=[];reports={}
    for name in names:
        report=load(OUT/'cases'/name/'SUMMARY.json')
        if report is None:
            rows.append({'name':name,'status':'pending'});continue
        reports[name]=report;q=np.array(report['quantiles_2p5_16_50_84_97p5'])
        rows.append({'name':name,'status':'converged' if report['converged'] else 'not_converged',
                     'cosmology':report['cosmology'],'case_type':report['case_type'],
                     'kmax_B':report['kmax_B'],'volume_Gpc_h3':report['volume_Gpc_h3'],
                     'fNL_p16':float(q[1,0]),'fNL_median':float(q[2,0]),'fNL_p84':float(q[3,0]),
                     'fNL_sigma':report['fNL_sigma'],'fNL_pull':report['fNL_pull'],
                     'chi2_MAP':report['chi2_MAP'],'gof_pvalue':report['gof_pvalue'],
                     'prior_sensitive':report['prior_sensitive'],'steps':report['steps'],
                     'rhat_max':max(report['rank_normalized_split_rhat']) if all(v is not None for v in report['rank_normalized_split_rhat']) else None,
                     'fNL_bulk_ESS':report['bulk_ess'][0],'fNL_tail_ESS':report['tail_ess'][0]})
    prior=[]
    for name in load(OUT/'SENSITIVITY_CASES.json') or []:
        wide=reports.get(name);base=reports.get(name.removesuffix('_wide_nuisance'))
        if wide is None or base is None:prior.append({'name':name,'status':'pending'});continue
        qm=np.asarray(wide['quantiles_2p5_16_50_84_97p5']);qb=np.asarray(base['quantiles_2p5_16_50_84_97p5'])
        valid_width=base['fNL_sigma'] is not None and wide['fNL_sigma'] is not None and base['fNL_sigma']>0
        shift=float(abs(qm[2,0]-qb[2,0])/base['fNL_sigma']) if valid_width else None
        ratio=wide['fNL_sigma']/base['fNL_sigma'] if valid_width else None
        passed=valid_width and base['converged'] and wide['converged'] and shift<.1 and abs(ratio-1)<.1
        prior.append({'name':name,'cosmology':base['cosmology'],'kmax_B':base['kmax_B'],
                      'status':'stable' if passed else 'sensitive_or_unconverged',
                      'fNL_median_shift_baseline_sigma':shift,'fNL_sigma_ratio':ratio})
    candidates={}
    for c in COSMOS:
        good=[]
        for k in CUTS:
            r=reports.get(f'{c}_mean100_k{k:.2f}')
            # 平坦弱约束参数天然约 4% boundary mass；不是 fNL 稳健性的否决证据。
            passed=(r is not None and r['converged'] and
                    r['gof_pvalue']>=.05 and abs(r['fNL_pull'])<=1)
            if not passed:break
            good.append(k)
        boundary=max(good) if good else None
        checked=[p for p in prior if p.get('cosmology')==c and p.get('kmax_B')==boundary]
        candidates[c]={'contiguous_baseline_candidate_kmax':boundary,
                       'boundary_prior_test':'not_available' if not checked else checked[0]['status'],
                       'scientifically_certified':False,
                       'reason':'conditional tree/PBS/Gaussian-stochastic baseline; finite-PNG stochastic closure remains provisional'}
    names_baseline=baseline_names();complete=all(n in reports for n in names_baseline)
    sensitivity_done=bool(prior) and all(p['status']!='pending' for p in prior)
    contract=load(OUT/'contract.json') or {}
    measurement_root=Path(contract.get('source_output',str(OUT)))
    value={'status':'COMPLETE_WITH_DIAGNOSTICS' if complete and sensitivity_done else 'IN_PROGRESS',
           'generated_UTC':datetime.now(timezone.utc).isoformat(),'baseline_cases':len(names_baseline),
           'completed_baseline_cases':sum(n in reports for n in names_baseline),
           'converged_baseline_cases':sum(reports[n]['converged'] for n in names_baseline if n in reports),
           'sensitivity_cases':len(prior),'case_rows':rows,'prior_sensitivity':prior,'range_candidates':candidates,
           'scan_counts':{str(k):len(selected(k)) for k in CUTS},'k015_alias_of_k014':True,
           'aliases':load(OUT/'ALIASES.json') or {},
           'measurement_progress':{f'mesh{n}':len(list((measurement_root/f'measurements_m{n}').glob('*.npz'))) for n in (256,512)},
           'covariance_correction_audit':load(OUT/'COVARIANCE_CORRECTION_AUDIT.json'),
           'numerical_audit':load(OUT/'WORKFLOW_AUDIT.json'),
           'legacy_boundary_audit':load(OUT/'LEGACY_BOUNDARY_AUDIT.json'),
           'P_likelihood':False,'P_derived_b1_prior':False,'IR':False,'FoG':False,'loops':False,
           'volume_note':contract.get('mean_likelihood','Read per-case frozen contract for data mean count and covariance volume.'),
           'likelihood_interpretation':'Mean data with single-box covariance is a single-box-precision model test, not the sampling likelihood of the 100-box mean.',
           'GOF_interpretation':'Gaussian/Hartlap chi2.sf is only a forecast-weighted residual indicator here; averaged-target noise is smaller than Csingle, so this is not a calibrated goodness-of-fit p-value.',
           'cut_note':'kmax applies to maximum of the two measured shell-average k values, not closing third leg',
           'candidate_rule':'contiguous converged baseline, descriptive GOF p>=.05, |fNL pull|<=1; use explicit wide-prior test, not flat-parameter boundary mass, for robustness; not certification',
           'prior_rule':'nuisance bounds x2; median shift<.1 baseline sigma and width change<10%, with both chains converged',
           'scientific_limits':['external p=1.179 fixed, not marginalized','fixed fiducial covariance for all PNG signs',
                                'no finite-PNG stochastic closure','no reconstruction, IR, FoG, loops, or higher multipoles'],
           'contract_sha256':digest(OUT/'contract.json')}
    return value,reports

def page():return plt.subplots(figsize=(12,8))

def save_page(pdf,fig,label,qa):
    fig.suptitle(TITLE+' | '+label,fontsize=14,x=.06,ha='left',y=.97)
    fig.text(.06,.025,'B-only; no P-derived prior. All validity statements are conditional.',fontsize=8,color='#444444')
    fig.text(.95,.025,str(len(qa)+1),fontsize=8,ha='right')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.bbox.width,fig.bbox.height;outside=[];not_drawn=set()
    # Locator 保留视区外 Tick Text，但 Axis 不会绘制它们，不能把隐藏 tick 当作溢出。
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-10<=tick.get_loc()<=hi+1e-10:not_drawn.update((id(tick.label1),id(tick.label2)))
    for item in fig.findobj(Text):
        if id(item) in not_drawn or not item.get_visible() or not item.get_text():continue
        bounds=item.get_window_extent(renderer)
        if bounds.width>0 and bounds.height>0 and (bounds.x0<-.5 or bounds.y0<-.5 or bounds.x1>w+.5 or bounds.y1>h+.5):
            outside.append(item.get_text())
    if outside:raise ValueError(('PDF text extends beyond page',label,outside))
    qa.append({'page':len(qa)+1,'label':label,'text_within_page':True})
    pdf.savefig(fig);plt.close(fig)

def summary_page(pdf,value,qa):
    fig,ax=page();ax.axis('off')
    lines=[f"Status: {value['status']}  |  generated: {value['generated_UTC'][:19]} UTC",
           f"Baseline summaries: {value['completed_baseline_cases']}/{value['baseline_cases']}; converged: {value['converged_baseline_cases']}/{value['baseline_cases']}",
           f"Sensitivity cases: {value['sensitivity_cases']}; no missing result is treated as a pass.",
           f"Measured boxes: mesh256={value['measurement_progress']['mesh256']}/800; mesh512={value['measurement_progress']['mesh512']}/3.",
           '', 'Observable and data contract',
           'Sugiyama B000(k1,k2), the same radial edges and 44 certified shell pairs as real space.',
           'The closing third leg is unrestricted; excluded zero-mode terms keep the full-shell denominator.',
           'kmax: .08 / .10 / .12 / .14 / .18 / .20; B points: 9 / 14 / 20 / 27 / 35 / 44.',
           'The .15 cut has exactly the .14 indices and aliases that chain, rather than duplicating it.',
           '500 disjoint fiducial covariance ICs (1000-1499); 100 validation ICs per PNG sign (0-99).',
           value['volume_note'],
           '', 'Model and inference',
           'Pre-reconstruction tree G + fNL L + fNL^2 Q; fixed f=.7625790038 and external p=1.179.',
           'Free b1 and nuisance biases/noise; no P data, P-derived b1 prior, IR, FoG, loops or counterterms.',
           'Exact-lattice source FFT projection and independent native/direct-enumeration tests.',
           'Four independent emcee ensembles, 32 walkers each; resumable HDF; budget-only extensions recorded separately.',
           'Convergence gates: rank Rhat, bulk/tail ESS, length/tau, stationarity, ensemble spread, acceptance.',
           '', 'Scientific interpretation',
           'A fit containing the true fNL once is not a validity certificate. Unconverged fits are excluded.',
           'Mean target + Csingle is a single-box-precision model test, not the sampling likelihood of the mean.',
           'GOF is only a forecast-weighted residual indicator, not a calibrated chi2 p-value for averaged data.',
           'Finite-PNG stochastic closure, p uncertainty and covariance cosmology dependence remain limitations.',
           'Numerical agreement does not validate reconstructed RSD data or full RSD IR resummation.',
           '', 'Contiguous conditional baseline range candidates']
    for c,r in value['range_candidates'].items():
        lines.append(f"{c}: kmax={r['contiguous_baseline_candidate_kmax']}; boundary prior test={r['boundary_prior_test']}")
    noise=load(OUT/'MODEL_NOISE_AUDIT.json')
    if noise:lines.append(f"Noise rank audit: {noise['parameter_count']} parameters; anisotropic shape retained={noise['native_SP_independent_combination_retained']}")
    legacy=value['legacy_boundary_audit']
    if legacy:
        lines+=['','Legacy control: 72 modes transfer at k=.18607143; old high-k values are not interchangeable.',
                f"Unchanged-count bins differ by <= {legacy['unchanged_count_bins_max_control_shift']:.3g} pilot mean100 sigma."]
    pipeline=load(OUT/'PIPELINE.json')
    if pipeline:
        lines+=['',f"Execution provenance: {pipeline.get('status','submitted')} (IDs are not proof of completion)"]
        lines+=textwrap.wrap('; '.join(f'{k}: {v}' for k,v in pipeline['jobs'].items()),130)
    login=load(OUT/'LOGIN_STATUS.json')
    if login:
        lines.append(f"Login runner: {login['host']}; CPU affinity {login['affinity']}; max {login['maximum_logical_CPUs']} logical CPUs.")
    font=min(8.5,.77*8*72/(1.30*(len(lines)+1)))
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=font,linespacing=1.30,transform=ax.transAxes)
    fig.subplots_adjust(left=.06,right=.95,bottom=.09,top=.90)
    save_page(pdf,fig,'contract and status',qa)

def scan_page(pdf,reports,qa):
    fig,axes=plt.subplots(1,3,figsize=(12,8),sharex=True)
    for ax,c in zip(axes,COSMOS):
        ax.axhline(TRUTHS[c],color='black',ls='--',lw=1,label='truth')
        for k in CUTS:
            r=reports.get(f'{c}_mean100_k{k:.2f}')
            if r is None:continue
            q=np.array(r['quantiles_2p5_16_50_84_97p5'])[:,0]
            color=COLORS[c] if r['converged'] else '#999999';marker='s' if r['prior_sensitive'] else 'o'
            ax.errorbar(k,q[2],yerr=[[q[2]-q[1]],[q[3]-q[2]]],fmt=marker,color=color,capsize=4)
            ax.plot([k,k],[q[0],q[4]],color=color,alpha=.4,lw=1)
        ax.set(title=c,xlabel='two-leg kmax [h/Mpc]',ylabel='fNL posterior');ax.grid(alpha=.2)
        ax.set_xticks(CUTS);ax.tick_params(axis='x',rotation=45);ax.legend(loc='best',fontsize=8)
    fig.text(.07,.08,'Intervals: 68% (thick), 95% (thin). Gray = not converged; square = nuisance/prior boundary flag.',fontsize=9)
    fig.tight_layout(rect=(.03,.12,.98,.90));save_page(pdf,fig,'mean100 cutoff scan',qa)

def diagnostics_page(pdf,reports,qa):
    rows=[reports[n] for n in baseline_names() if n in reports]
    labels=['rhat','fNL_ESS','other_ESS','length_tau','half_shift','ensemble_spread','acceptance']
    values=np.array([[bool(r['gates'][k]) for k in labels]+[not r['prior_sensitive']] for r in rows],dtype=float)
    fig,ax=page();image=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='RdYlGn')
    ax.set_yticks(range(len(rows)),[r['name'] for r in rows],fontsize=8)
    ax.set_xticks(range(8),labels+['prior flag absent'],rotation=25,ha='right',fontsize=9)
    for y,r in enumerate(rows):
        for x,v in enumerate(values[y]):ax.text(x,y,'pass' if v else 'fail',ha='center',va='center',fontsize=7)
    ax.set_title('Sampler convergence and prior boundary mass are separate checks',fontsize=10)
    fig.subplots_adjust(left=.25,right=.96,bottom=.18,top=.86)
    save_page(pdf,fig,'independent-chain diagnostics',qa)

def fit_page(pdf,reports,qa):
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex='col')
    for j,c in enumerate(COSMOS):
        name=f'{c}_mean100_k0.12';root=OUT/'cases'/name;r=reports.get(name)
        if r is None:
            axes[0,j].text(.5,.5,'pending',ha='center',transform=axes[0,j].transAxes);continue
        with np.load(root/'likelihood.npz') as z:arrays={k:np.array(z[k]) for k in z.files}
        mapmeta=load(root/'MAP.json');model=predicted(mapmeta['center'],arrays['coefficients'],arrays['noise'])[0]
        sig=np.sqrt(np.diag(arrays['covariance']));x=np.arange(len(sig))
        axes[0,j].errorbar(x,arrays['target']/1e8,yerr=sig/1e8,fmt='.',color=COLORS[c],label='mean100 +/- sigma')
        axes[0,j].plot(x,model/1e8,color='black',lw=1,label='MAP model')
        axes[0,j].set(title=c,ylabel='B000 / 1e8 [(Mpc/h)^6]');axes[0,j].legend(fontsize=7)
        axes[1,j].plot(x,(arrays['target']-model)/sig,'.-',color=COLORS[c])
        axes[1,j].axhline(0,color='black',lw=.8);axes[1,j].set(xlabel='certified bin order at kmax=.12',ylabel='residual / diagonal sigma')
        for ax in axes[:,j]:ax.grid(alpha=.2)
    fig.tight_layout(rect=(.02,.07,.99,.91));save_page(pdf,fig,'B-only fit and residuals',qa)

def noise_page(pdf,qa):
    with np.load(OUT/'B000_ENSEMBLE_MATRIX.npz') as z:
        idx=z['certified_indices'];cov=z['covariance_single_box'][np.ix_(idx,idx)];sg=z['stochastic_geometry']
    corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    fig,axes=plt.subplots(1,2,figsize=(12,8));im=axes[0].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    axes[0].set(title='500-box RSD covariance correlation',xlabel='certified shell pair',ylabel='certified shell pair')
    fig.colorbar(im,ax=axes[0],fraction=.046)
    fraction=(sg[2]-sg[0]/5)/np.maximum(abs(sg[0]),1)
    axes[1].plot(fraction,'.-');axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(title='Discrete mu^4 departure from continuum 1/5',xlabel='certified shell pair',ylabel='(sum P mu^4 - sum P/5) / sum P')
    axes[1].grid(alpha=.2);audit=load(OUT/'MODEL_NOISE_AUDIT.json')
    fig.text(.08,.10,f"Noise audit: {audit['parameter_count']} parameters; retained independent SP shape: {audit['native_SP_independent_combination_retained']}",fontsize=9)
    fig.text(.08,.07,'This Gaussian noise audit is not finite-PNG stochastic closure.',fontsize=9)
    fig.tight_layout(rect=(.02,.16,.98,.91));save_page(pdf,fig,'covariance and noise rank',qa)

def posterior_page(pdf,reports,qa):
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for j,c in enumerate(COSMOS):
        root=OUT/'cases'/f'{c}_mean100_k0.12';path=root/'POSTERIOR_THIN.npz'
        if not path.exists():continue
        with np.load(path) as z:samples=z['samples']
        for i,(parameter,column) in enumerate([('fNL',0),('b1',1)]):
            axes[i,j].hist(samples[:,column],bins=50,density=True,color=COLORS[c],alpha=.7)
            axes[i,j].set(xlabel=parameter,ylabel='density',title=c+' | kmax=.12');axes[i,j].grid(alpha=.2)
        axes[0,j].axvline(TRUTHS[c],color='black',ls='--')
    fig.tight_layout(rect=(.02,.07,.99,.91));save_page(pdf,fig,'selected marginalized posteriors',qa)

def prior_page(pdf,value,qa):
    fig,axes=plt.subplots(1,2,figsize=(12,8));rows=[r for r in value['prior_sensitivity'] if r['status']!='pending' and r['fNL_sigma_ratio'] is not None]
    for j,r in enumerate(rows):
        color=COLORS[r['cosmology']];axes[0].scatter(j,r['fNL_median_shift_baseline_sigma'],color=color)
        axes[1].scatter(j,r['fNL_sigma_ratio'],color=color)
    labels=[r['cosmology']+' '+str(r['kmax_B']) for r in rows]
    for ax in axes:ax.set_xticks(range(len(rows)),labels,rotation=35,ha='right');ax.grid(alpha=.2)
    axes[0].axhline(.1,color='black',ls='--');axes[0].set(title='Median sensitivity',ylabel='|wide - baseline| / baseline sigma')
    axes[1].axhspan(.9,1.1,color='gray',alpha=.15);axes[1].set(title='Width sensitivity',ylabel='wide nuisance sigma / baseline sigma')
    fig.text(.07,.08,'Only nuisance bounds widened x2; fNL, b1, p, data and covariance fixed. Both chains must converge.',fontsize=9)
    fig.tight_layout(rect=(.03,.16,.98,.91));save_page(pdf,fig,'nuisance prior sensitivity',qa)

def trace_page(pdf,reports,qa):
    import h5py
    fig,axes=plt.subplots(3,1,figsize=(12,8))
    for ax,c in zip(axes,COSMOS):
        root=OUT/'cases'/f'{c}_mean100_k0.12';meta=load(root/'MAP.json');path=root/'chain.h5'
        if meta is None or not path.exists():continue
        transform=np.array(meta['transform']);center=np.array(meta['center'])
        with h5py.File(path) as h:
            for ensemble in range(4):
                data=h[f'ensemble_{ensemble}/chain'];stride=max(1,len(data)//500)
                coords=data[::stride];physical=coords@transform.T+center
                ax.plot(np.arange(len(physical))*stride,np.median(physical[:,:,0],axis=1),lw=.7,label=f'ensemble {ensemble}')
        ax.axhline(TRUTHS[c],color='black',ls='--',lw=.7);ax.set(ylabel=c+' fNL',xlabel='MCMC step');ax.legend(ncol=4,fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(.02,.07,.99,.91));save_page(pdf,fig,'independent ensemble traces at .12',qa)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    value,reports=collect()
    if args.require_complete and value['status']!='COMPLETE_WITH_DIAGNOSTICS':raise ValueError('report requested before all chain summaries')
    target=OUT/'RSD_B000_VALIDATION.pdf';tmp=target.with_suffix('.tmp.pdf');qa=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(tmp,metadata={'Title':TITLE,'Subject':'Conditional pre-reconstruction tree validation; B-only'}) as pdf:
        summary_page(pdf,value,qa)
        if reports:
            scan_page(pdf,reports,qa);diagnostics_page(pdf,reports,qa);fit_page(pdf,reports,qa)
            if (OUT/'B000_ENSEMBLE_MATRIX.npz').exists():noise_page(pdf,qa)
            posterior_page(pdf,reports,qa);prior_page(pdf,value,qa);trace_page(pdf,reports,qa)
    tmp.replace(target)
    value['pdf_sha256']=digest(target);value['report_source_sha256']=digest(Path(__file__))
    save_json(OUT/'RESULTS.json',value)
    save_json(OUT/'PDF_QA.json',{'status':'PASS_VECTOR_BOUNDS','pages':qa,'PDF_only':True,
                               'raster_inspection':False,'note':'User explicitly requires no PNG; vector text bounds checked instead'})
    keys=sorted(set().union(*(r.keys() for r in value['case_rows'])))
    with (OUT/'RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(value['case_rows'])
    print(json.dumps({'status':value['status'],'pages':len(qa),'pdf':str(target),'completed':value['completed_baseline_cases']}),flush=True)

if __name__=='__main__':main()
