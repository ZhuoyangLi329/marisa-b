"""模块大纲：联合链真实状态；B-only 配对比较；条件残差；一份 PDF 诊断。

mean100+Csingle 不是真实 mean 的抽样似然；不把宽后验覆盖真值等同于物理模型准确。
"""
import csv
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from iminuit import Minuit
from rsd_pb_common import OUT,B_OUT,digest,save_json
from rsd_b000_contract import CUTS
from run_rsd_pb_inference import predicted
from run_rsd_pb_mcmc import JointCase

COSMOS=('fiducial','LC_p','LC_m');COLORS={'fiducial':'#356795','LC_p':'#bb633d','LC_m':'#468666'}

def decomposition(residual,cov,alpha):
    """P + (B|P) Schur 补，精确加总为完整联合 Hartlap chi2。"""
    a=cov[:12,:12];cross=cov[12:,:12];bb=cov[12:,12:]
    conditional=residual[12:]-cross@np.linalg.solve(a,residual[:12])
    schur=bb-cross@np.linalg.solve(a,cross.T)
    p=float(alpha*residual[:12]@np.linalg.solve(a,residual[:12]))
    b=float(alpha*conditional@np.linalg.solve(schur,conditional))
    full=float(alpha*residual@np.linalg.solve(cov,residual))
    np.testing.assert_allclose(p+b,full,rtol=2e-12,atol=1e-9)
    return {'P_marginal_chi2':p,'B_conditional_on_P_chi2':b,'joint_chi2':full}

def truth_profile(case,center):
    truth=case.meta['truth_fNL'];fit=Minuit(case.objective,*center,name=case.names);fit.errordef=1
    for name,bound,error in zip(case.names,case.bounds,[30,.1,.2,.2]+[.3]*(case.ndim-4)):
        fit.limits[name]=bound;fit.errors[name]=error
    fit.values[0]=truth;fit.fixed[0]=True;fit.migrad(ncall=15000)
    if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=15000)
    return {'truth_fNL':truth,'chi2_at_profiled_truth':float(fit.fval),'valid':bool(fit.valid)}

def collect():
    names=json.loads((OUT/'CASES.json').read_text());rows=[];decompositions=[]
    for name in names:
        path=OUT/'cases'/name/'SUMMARY.json'
        if not path.exists():rows.append({'name':name,'status':'pending'});continue
        summary=json.loads(path.read_text());old=json.loads((B_OUT/'cases'/name/'SUMMARY.json').read_text())
        q=np.array(summary['quantiles_2p5_16_50_84_97p5']);qb=np.array(old['quantiles_2p5_16_50_84_97p5'])
        case=JointCase(OUT/'cases'/name);m=json.loads((case.directory/'MAP.json').read_text());center=np.array(m['center'])
        oldmap_path=B_OUT/'cases'/name/'MAP.json'
        if digest(oldmap_path)!=case.meta['source_B_MAP_sha256']:raise ValueError('B-only MAP source drift')
        oldmap=json.loads(oldmap_path.read_text())
        residual=predicted(center,case.coefficients,case.noise)[0]-case.target
        parts=decomposition(residual,case.covariance,case.meta['Hartlap_factor'])
        np.testing.assert_allclose(parts['joint_chi2'],m['chi2_MAP'],rtol=2e-11,atol=1e-8)
        profile=truth_profile(case,center);profile['delta_chi2_truth']=profile['chi2_at_profiled_truth']-m['chi2_MAP']
        with np.load(case.directory/'POSTERIOR_THIN.npz') as z:thin=np.array(z['samples'])
        with np.load(B_OUT/'cases'/name/'POSTERIOR_THIN.npz') as z:oldthin=np.array(z['samples'])
        rows.append({'name':name,'status':'converged' if summary['converged'] else 'not_converged',
                    'cosmology':summary['cosmology'],'truth_fNL':summary['truth_fNL'],'kmax_B':summary['kmax_B'],
                    'fNL_median':float(q[2,0]),'fNL_p16':float(q[1,0]),'fNL_p84':float(q[3,0]),
                    'fNL_sigma':summary['fNL_sigma'],'fNL_pull':summary['fNL_pull'],
                    'truth_in_68pct_interval':bool(q[1,0]<=summary['truth_fNL']<=q[3,0]),
                    'Bonly_fNL_median':float(qb[2,0]),'Bonly_fNL_sigma':old['fNL_sigma'],
                    'fNL_sigma_PB_over_B':float(summary['fNL_sigma']/old['fNL_sigma']),
                    'b1_median':float(q[2,1]),'b1_p16':float(q[1,1]),'b1_p84':float(q[3,1]),
                    'Bonly_b1_median':float(qb[2,1]),'Bonly_b1_p16':float(qb[1,1]),'Bonly_b1_p84':float(qb[3,1]),
                    'b1_sigma_PB_over_B':float(np.std(thin[:,1],ddof=1)/np.std(oldthin[:,1],ddof=1)),
                    'fNL_b1_correlation_PB':float(np.corrcoef(thin[:,:2],rowvar=False)[0,1]),
                    'fNL_b1_correlation_B':float(np.corrcoef(oldthin[:,:2],rowvar=False)[0,1]),
                    'MAP_fNL':float(center[0]),'MAP_b1':float(center[1]),'chi2_MAP':m['chi2_MAP'],
                    'Bonly_MAP_fNL':float(oldmap['center'][0]),'Bonly_MAP_b1':float(oldmap['center'][1]),
                    'delta_chi2_profile_truth':profile['delta_chi2_truth'],'profile_truth_valid':profile['valid'],
                    'Pshot_residual_Mpc_h3_median':float(q[2,-1]*case.meta['Pshot_unit_Mpc_h3']),
                    'rhat_max':float(max(summary['rank_normalized_split_rhat'])),
                    'fNL_bulk_ESS':summary['bulk_ess'][0],'fNL_tail_ESS':summary['tail_ess'][0],
                    'steps':summary['steps'],'prior_sensitive':summary['prior_sensitive'],**parts})
        decompositions.append({'name':name,'profile_truth':profile,**parts})
    completed=[r for r in rows if r['status']!='pending'];converged=[r for r in rows if r['status']=='converged']
    value={'status':'COMPLETE' if len(converged)==18 else 'INCOMPLETE','generated_UTC':datetime.now(timezone.utc).isoformat(),
           'n_cases':18,'completed':len(completed),'converged':len(converged),'case_rows':rows,'decomposition':decompositions,
           'P_likelihood':True,'P_bins':12,'kmax_P':.08,'B_cuts':list(CUTS),'volume_Gpc_h3':1.,
           'data_mean_count':100,'covariance_sample_count':500,'covariance_multiplier':1.,'covariance_is_of_mean':False,
           'full_P_B_cross_covariance':True,'extra_parameter':'independent Pshot residual only',
           'old_B_model_data_priors_unchanged':True,'IR':False,'FoG':False,'loops':False,
           'scientifically_certified':False,'contract_sha256':digest(OUT/'contract.json'),
           'interpretation':'single-box-precision model/forecast test using mean100 target, not the sampling likelihood of that mean',
           'GOF_note':'No calibrated chi2 p-value or hard validity boundary inferred from mean100 with Csingle.',
           'scientific_limits':['finite-PNG B stochastic closure remains provisional','bphidelta recipe unchanged in this controlled test',
                                'no IR, FoG, loops or counterterms','P0 only, growth rate fixed; no P2 or P4',
                                'PNG truth signs are phase-correlated, not independent tests','closing B third leg unrestricted'],
           'numerical_audit':json.loads((OUT/'PILOT_AUDIT.json').read_text()),
           'joint_covariance_audit':json.loads((OUT/'JOINT_FREEZE_AUDIT.json').read_text())}
    return value

def savepage(pdf,fig,label,qa):
    fig.suptitle('Pre-reconstruction RSD P0 + B000 | '+label,x=.055,y=.975,ha='left',fontsize=14)
    fig.text(.055,.022,'Mean100 data + C_single; V=1. B kernel unchanged; no IR/FoG/loops. Conditional model test.',fontsize=8)
    fig.text(.95,.022,str(len(qa)+1),ha='right',fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.bbox.width,fig.bbox.height;hidden=set();outside=[]
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-10<=tick.get_loc()<=hi+1e-10:hidden.update((id(tick.label1),id(tick.label2)))
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.width>0 and b.height>0 and (b.x0<-.5 or b.y0<-.5 or b.x1>w+.5 or b.y1>h+.5):outside.append(t.get_text())
    if outside:raise ValueError(('PDF text outside page',label,outside))
    qa.append({'page':len(qa)+1,'label':label,'text_within_page':True});pdf.savefig(fig);plt.close(fig)

def plot(value):
    qa=[];valid=[r for r in value['case_rows'] if r['status']!='pending'];output=OUT/'RSD_PB_VALIDATION.pdf'
    with PdfPages(output) as pdf:
        fig,ax=plt.subplots(figsize=(12,8));ax.axis('off')
        lines=[f"Status: {value['status']} | summaries {value['completed']}/18 | converged {value['converged']}/18",
               'P0: same real-space 12-bin edges/density, kmax_P = 0.08 h/Mpc; actual CPU window.',
               'B000: two measured legs, same six cuts 0.08 / 0.10 / 0.12 / 0.14 / 0.18 / 0.20.',
               'Targets: same 100 RSD boxes per truth 0 / +100 / -100; covariance: independent Fid500.',
               'Full matched P-B cross covariance retained. C_single is NOT divided by 100 or 500.',
               'Seven B-only parameters/priors/model unchanged; one independent residual Pshot added.',
               'Sampling: original four independent ensembles x 32 walkers; identical convergence gates.',
               '',
               'Interpretation: a single-box-precision model test, not the likelihood of the mean100.',
               'Wide credible intervals covering the truth are not proof of an unbiased physical model.',
               'P0 can break bias/PNG/noise degeneracy; it cannot repair missing nonlinear velocity physics.',
               '',
               'No IR, FoG, loops, counterterms, or higher multipoles included in this controlled comparison.',
               'Finite-PNG B stochastic closure and PBS recipe are unchanged, explicitly provisional.',
               'Numerical mesh and frozen-input checks precede production, independently of truth recovery.']
        ax.text(.01,.92,'\n\n'.join(lines),va='top',fontsize=11,transform=ax.transAxes)
        fig.subplots_adjust(left=.055,right=.95,top=.91,bottom=.1);savepage(pdf,fig,'contract and interpretation',qa)
        fig,axes=plt.subplots(2,3,figsize=(12,8),sharex='col')
        for j,c in enumerate(COSMOS):
            rows=sorted([r for r in valid if r['cosmology']==c],key=lambda r:r['kmax_B']);x=np.array([r['kmax_B'] for r in rows])
            if not rows:continue
            for field,label,color,offset in [('Bonly_fNL_median','B-only','#888888',-.001),('fNL_median','P0+B000',COLORS[c],.001)]:
                y=np.array([r[field] for r in rows]);sigma=np.array([r['Bonly_fNL_sigma' if field.startswith('Bonly') else 'fNL_sigma'] for r in rows])
                axes[0,j].errorbar(x+offset,y,yerr=sigma,fmt='o-',ms=4,capsize=3,label=label,color=color)
            axes[0,j].axhline(rows[0]['truth_fNL'],ls='--',color='black',lw=.8);axes[0,j].set_title(c);axes[0,j].set_ylabel('fNL median +/- posterior std')
            axes[0,j].legend(fontsize=9);axes[0,j].grid(alpha=.2)
            axes[1,j].plot(x,[r['fNL_sigma_PB_over_B'] for r in rows],'o-',color=COLORS[c],label='fNL width')
            axes[1,j].plot(x,[r['b1_sigma_PB_over_B'] for r in rows],'s--',color='#555555',label='b1 width')
            axes[1,j].axhline(1,ls=':',color='black');axes[1,j].set_ylabel('posterior std(P+B) / std(B)');axes[1,j].set_xlabel('two-leg kmax_B [h/Mpc]')
            axes[1,j].legend(fontsize=9);axes[1,j].grid(alpha=.2)
        fig.subplots_adjust(left=.065,right=.96,top=.88,bottom=.095,wspace=.32,hspace=.3);savepage(pdf,fig,'fNL recovery and width ratios',qa)
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for col,c in enumerate(COSMOS):
            for row,k in enumerate((.10,.20)):
                name=f'{c}_mean100_k{k:.2f}';ax=axes[row,col]
                for root,color,label in ((B_OUT,'#999999','B-only'),(OUT,COLORS[c],'P+B')):
                    path=root/'cases'/name/'POSTERIOR_THIN.npz'
                    if not path.exists():continue
                    with np.load(path) as z:v=np.array(z['samples'])[::30]
                    ax.scatter(v[:,1],v[:,0],s=3,alpha=.25,color=color,label=label,rasterized=False)
                ax.set_title(f'{c}, kmax_B={k:.2f}');ax.set_xlabel('b1');ax.set_ylabel('fNL');ax.legend(fontsize=8,markerscale=3);ax.grid(alpha=.15)
        fig.subplots_adjust(left=.065,right=.96,top=.88,bottom=.095,wspace=.32,hspace=.42);savepage(pdf,fig,'bias-PNG degeneracy',qa)
        fig,axes=plt.subplots(1,3,figsize=(12,8))
        for ax,c in zip(axes,COSMOS):
            rows=sorted([r for r in valid if r['cosmology']==c],key=lambda r:r['kmax_B']);x=[r['kmax_B'] for r in rows]
            ax.plot(x,[r['P_marginal_chi2'] for r in rows],'o-',label='P marginal')
            ax.plot(x,[r['B_conditional_on_P_chi2'] for r in rows],'s-',label='B | P')
            ax.plot(x,[r['delta_chi2_profile_truth'] for r in rows],'^--',label='profile truth - MAP')
            ax.set_title(c);ax.set_xlabel('two-leg kmax_B [h/Mpc]');ax.set_ylabel('Hartlap-weighted chi2 (descriptive)');ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.subplots_adjust(left=.065,right=.96,top=.86,bottom=.16,wspace=.32);savepage(pdf,fig,'full-covariance conditional residuals',qa)
        with np.load(OUT/'P0_ENSEMBLE_MATRIX.npz') as z:cov=np.array(z['covariance_single_box'])
        corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
        fig,ax=plt.subplots(figsize=(12,8));im=ax.pcolormesh(corr,cmap='RdBu_r',vmin=-1,vmax=1,shading='flat')
        ax.axvline(12,color='black',lw=1);ax.axhline(12,color='black',lw=1);ax.set_aspect('equal')
        ax.set_xlabel('P0[12] then certified B000[44]');ax.set_ylabel('P0[12] then certified B000[44]')
        fig.colorbar(im,ax=ax,label='same-realization single-box correlation');fig.subplots_adjust(left=.08,right=.92,top=.88,bottom=.12)
        savepage(pdf,fig,'matched joint covariance',qa)
        fig,ax=plt.subplots(figsize=(12,8));ax.axis('off')
        cols=['sample','kB','fNL','std','pull','b1','std ratio','Rhat','steps']
        cells=[[r['cosmology'],f"{r['kmax_B']:.2f}",f"{r['fNL_median']:.2f}",f"{r['fNL_sigma']:.2f}",f"{r['fNL_pull']:.2f}",
                f"{r['b1_median']:.3f}",f"{r['fNL_sigma_PB_over_B']:.3f}",f"{r['rhat_max']:.5f}",str(r['steps'])] for r in valid]
        if cells:
            table=ax.table(cellText=cells,colLabels=cols,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
        fig.subplots_adjust(left=.055,right=.95,top=.88,bottom=.1);savepage(pdf,fig,'all joint MCMC summaries',qa)
    save_json(OUT/'PDF_QA.json',{'status':'PASS','pages':qa,'no_PNG_generated':True,
              'visual_raster_inspection':False,'note':'User requires PDF only; checked text bounds, scientific labels, frozen values. No PNG generated.',
              'pdf_sha256':digest(output)})

def main():
    value=collect();save_json(OUT/'RESULTS.json',value)
    fields=sorted(set().union(*(r.keys() for r in value['case_rows'])))
    with (OUT/'RESULTS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(value['case_rows'])
    lines=['# RSD P0+B000 controlled test','',f"Status: {value['status']}; converged {value['converged']}/18.",'',
           'P0 uses the real-space 12-bin edges/density, kmax_P=0.08, with its actual shared CPU binner window. B-only target/model/priors are unchanged.',
           'Targets are same 100-box means. Full matched Fid500 single-box covariance is unscaled: V=1, never /100 or /500.',
           'One independent Pshot residual parameter is added; no IR/FoG/loops or bphidelta/stochastic repair.',
           'Mean100 + Csingle is a forecast/model test, not a calibrated likelihood of the mean.',
           'Posterior truth coverage alone is not a physical validity certificate.','',
           '| sample | kmaxB | fNL median ± std | B-only median ± std | std ratio | b1 median | MAP fNL | Δχ² truth |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in value['case_rows']:
        if r['status']=='pending':continue
        lines.append(f"| {r['cosmology']} | {r['kmax_B']:.2f} | {r['fNL_median']:.2f} ± {r['fNL_sigma']:.2f} | {r['Bonly_fNL_median']:.2f} ± {r['Bonly_fNL_sigma']:.2f} | {r['fNL_sigma_PB_over_B']:.3f} | {r['b1_median']:.3f} | {r['MAP_fNL']:.2f} | {r['delta_chi2_profile_truth']:.3f} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');plot(value)
    print(json.dumps({'status':value['status'],'converged':value['converged'],'PDF':str(OUT/'RSD_PB_VALIDATION.pdf')}),flush=True)

if __name__=='__main__':main()
