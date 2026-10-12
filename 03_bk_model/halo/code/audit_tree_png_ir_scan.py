#!/usr/bin/env python3
"""IR 科学检验：独立 Minuit、合成注入、投影收敛、BAO-sensitive bins 消融。

模块大纲：不重用拟合目标函数实现的独立数值检查；每 bin 的 IR 变化；
区分真实 catalogue 恢复与同模型合成注入；只产生 PDF 图。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from iminuit import Minuit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from run_tree_png_ir_kmax_scan import ROOT,OUT,Profile,polynomial,B1,VOLUME,TRUTHS

def validate_native():
    core=ROOT/'marisa_b_core'
    evidence=OUT/'native_validation.json'
    files=[core/'src/marisa_b/tree_png_ir.cpp',core/'src/marisa_b/tree_png_ir.h',
           core/'python/marisa_b/tree_png_ir_native.py',core/'python/marisa_b/tree_png_ir_resummation.py',
           core/'build/tree_png_ir/test_tree_png_ir_native',core/'build/tree_png_ir/libmarisa_tree_png_ir.so',
           core/'tests/python/test_tree_png_ir_resummation.py',core/'tests/python/test_tree_png_ir_binding.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if evidence.exists():
        old=json.loads(evidence.read_text())
        if old['hashes']==hashes:return old
    environment=dict(os.environ,PYTHONPATH=str(core/'python'),PYTHONDONTWRITEBYTECODE='1')
    results=[]
    for command in ([str(core/'build/tree_png_ir/test_tree_png_ir_native')],
                    [sys.executable,'-m','pytest',str(files[-2]),str(files[-1]),'-q']):
        r=subprocess.run(command,text=True,capture_output=True,env=environment)
        results.append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    payload={'hashes':hashes,'runs':results,'pass':all(r['returncode']==0 for r in results)}
    evidence.write_text(json.dumps(payload,indent=2)+'\n')
    return payload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--high-tag',default='convergence')
    args=parser.parse_args()
    inputs=dict(np.load(OUT/'frozen_inputs.npz'))
    design=dict(np.load(OUT/'ir_model_designs_primary.npz'))
    scan=json.loads((OUT/'ir_kmax_scan_primary.json').read_text())
    idx=inputs['certified_indices'];geometry=inputs['geometry']
    cov=np.cov(inputs['data_+0'][:,idx],rowvar=False,ddof=1)
    sigma=np.sqrt(np.diag(cov)/VOLUME/((500-len(idx)-2)/499.))
    mean=inputs['prior_mean'];whitener=inputs['prior_whitener']
    theta=mean[1:];pvec=np.r_[1.,theta]
    fixed=design['current__fixed_order'];strict=design['current__strict_kir0.15']
    factorized=design['current__factorized_kir0.15']
    changes=((strict-fixed)@pvec)[:,idx]
    data={'scope':'pre-reconstruction finite-local-PNG tree only',
          'synthetic_injection':[],'independent_minuit':[],
          'binwise':{'indices':idx.tolist(),'sigma_effective_fit':sigma.tolist(),
                     'delta_G_over_sigma':(changes[0]/sigma).tolist(),
                     'delta_L_over_sigma':(changes[1]/sigma).tolist(),
                     'delta_Q_over_sigma':(changes[2]/sigma).tolist()},
          'closure_differences':{},'bao_sensitive_bin_ablation':[]}
    for truth in (-100.,0.,100.):
        delta=(polynomial(strict-fixed,truth)@pvec)[idx]
        data['binwise'][f'delta_B_fNL{truth:+.0f}_over_sigma']=(delta/sigma).tolist()
        closure=(polynomial(factorized-strict,truth)@pvec)[idx]
        data['closure_differences'][str(truth)]={'max_abs_delta_B_over_sigma':float(np.max(np.abs(closure/sigma)))}
    # 真正的同模型注入采用 nuisance 先验中心，不把这一自洽性测试代替 N-body 恢复。
    for truth in TRUTHS:
        target=(polynomial(strict,truth)@pvec)[idx]
        for name in ('current__strict_kir0.15','current__fixed_order'):
            result=Profile(design[name][:,idx],target,cov,mean,whitener).fit(truth)
            data['synthetic_injection'].append({'fit_model':name,'truth':truth,'fNL':result['fNL'],
                                               'bias':result['fNL']-truth,'truth_delta_chi2':result['truth_delta_chi2']})
    # 直接以协方差逆矩阵建立独立的五维 Minuit objective。
    for variant,cut,truth in [('fixed_order',.10,-50.),('strict_kir0.15',.10,-50.),
                              ('strict_kir0.15',.10,0.),('strict_kir0.15',.12,100.)]:
        selected=idx[np.max(geometry[idx],axis=1)<=cut+1.e-12]
        raw_cov=np.cov(inputs['data_+0'][:,selected],rowvar=False,ddof=1)
        precision=(500-len(selected)-2)/499.*VOLUME*np.linalg.inv(raw_cov)
        target=inputs[f'data_{truth:+.0f}'][:,selected].mean(axis=0)
        d=design['current__'+variant][:,selected]
        row=next(r for r in scan['rows'] if r['bias']=='current' and r['variant']==variant and r['kmax']==cut and r['truth']==truth)
        def objective(fNL,b2,gamma2,ashot,bshot):
            t=np.array([b2,gamma2,ashot,bshot]);residual=polynomial(d,fNL)@np.r_[1.,t]-target
            prior=whitener@(np.r_[B1,t]-mean)
            return float(residual@precision@residual+prior@prior)
        fit=Minuit(objective,truth,*theta);fit.errordef=1;fit.limits['fNL']=(truth-200,truth+200)
        fit.errors=(10.,.1,.1,.1,.1);fit.tol=1.e-7;fit.strategy=2
        fit.migrad(ncall=100000);fit.hesse()
        data['independent_minuit'].append({'variant':variant,'kmax':cut,'truth':truth,
                                          'valid':bool(fit.valid),'fNL':fit.values['fNL'],
                                          'delta_fNL':fit.values['fNL']-row['fNL'],
                                          'delta_objective':fit.fval-row['best_fit']['objective']})
    # 仅由理论和 covariance 选取最 IR-sensitive 的一半 bins，避免看数据后选择。
    # B000 已积分 k3，故这里是 BAO 敏感度消融，不能宣称完全去除了 BAO。
    order=np.argsort(np.abs(changes[0]/sigma));weak=order[:len(order)//2];strong=order[len(order)//2:]
    for subset,positions in [('less_BAO_sensitive',weak),('more_BAO_sensitive',strong)]:
        selected=idx[positions];subcov=cov[np.ix_(positions,positions)]
        for variant in ('fixed_order','strict_kir0.15'):
            for truth in (-100.,0.,100.):
                target=inputs[f'data_{truth:+.0f}'][:,selected].mean(axis=0)
                fit=Profile(design['current__'+variant][:,selected],target,subcov,mean,whitener).fit(truth)
                data['bao_sensitive_bin_ablation'].append({'subset':subset,'indices':selected.tolist(),'variant':variant,
                                                          'truth':truth,'fNL':fit['fNL'],'interval':fit['interval'],
                                                          'truth_delta_chi2':fit['truth_delta_chi2'],
                                                          'nuisance':fit['best_fit']['nuisance'],'data_chi2':fit['best_fit']['data_chi2']})
    highfile=OUT/f'ir_model_designs_{args.high_tag}.npz'
    if highfile.exists():
        high=dict(np.load(highfile));highscan=json.loads((OUT/f'ir_kmax_scan_{args.high_tag}.json').read_text())
        convergence={}
        for variant in ('fixed_order','strict_kir0.15','factorized_kir0.15'):
            key='current__'+variant
            if key not in high:continue
            diff=high[key]-design[key]
            selectedrows=[r for r in highscan['rows'] if r['bias']=='current' and r['variant']==variant]
            ferrors=[];qerrors=[];fscaled=[]
            for row in selectedrows:
                ref=next(r for r in scan['rows'] if r['bias']=='current' and r['variant']==variant and r['kmax']==row['kmax'] and r['truth']==row['truth'])
                ferrors.append(abs(row['fNL']-ref['fNL']));qerrors.append(abs(row['truth_delta_chi2']-ref['truth_delta_chi2']))
                fscaled.append(abs(row['fNL']-ref['fNL'])/(.5*(ref['interval'][1]-ref['interval'][0])))
            convergence[variant]={'max_fNL_change':max(ferrors),'max_truth_delta_chi2_change':max(qerrors),
                                  'max_fNL_change_over_profile_halfwidth':max(fscaled),
                                  'maximum_delta_prediction_over_sigma':max(float(np.max(np.abs((polynomial(diff,t)@pvec)[idx]/sigma))) for t in (-100,0,100)),
                                  'kmax_decision_unchanged':highscan['summary'][key]['largest_contiguous_valid_kmax']==scan['summary'][key]['largest_contiguous_valid_kmax'],
                                  'high_resolution_kmax':highscan['summary'][key]['largest_contiguous_valid_kmax']}
        data['projection_convergence']=convergence
    historical_path=Path('/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_direct_finite_tree_kmax_validity_scan_v42p875/DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.json')
    historic=json.loads(historical_path.read_text())
    fdiffs=[];qdiffs=[]
    for row in scan['rows']:
        if row['variant']!='historical':continue
        old=next(r for r in historic['rows'] if r['method']=='pre' and r['kmax_h_mpc']==row['kmax'] and r['input_fNL']==row['truth'])
        fdiffs.append(abs(row['fNL']-old['map']['fNL']));qdiffs.append(abs(row['truth_delta_chi2']-old['truth_profile']['delta_chi2']))
    data['historical_fit_reproduction']={'max_fNL_change':max(fdiffs),'max_truth_delta_chi2_change':max(qdiffs)}
    validation=validate_native()
    data['native_validation_artifact']=str(OUT/'native_validation.json')
    gates={'native_cpp_and_python_tests':validation['pass'],
           'synthetic_fNL_recovery':all(abs(r['bias'])<1.e-3 for r in data['synthetic_injection'] if 'strict' in r['fit_model']),
           'independent_minuit':all(r['valid'] and abs(r['delta_fNL'])<.02 and abs(r['delta_objective'])<1.e-5 for r in data['independent_minuit']),
           'historical_fit_reproduction':max(fdiffs)<.02 and max(qdiffs)<1.e-5,
           'projection_convergence_available':'projection_convergence' in data,
           'projection_numerical_sensitivity':bool(data.get('projection_convergence')) and all(
               r['kmax_decision_unchanged'] and r['maximum_delta_prediction_over_sigma']<.05
               and r['max_fNL_change_over_profile_halfwidth']<.05
               for r in data.get('projection_convergence',{}).values())}
    data['gates']=gates
    data['status']='pass' if all(gates.values()) else 'incomplete_or_failed'
    (OUT/'IR_SCIENTIFIC_AUDIT.json').write_text(json.dumps(data,indent=2)+'\n')
    with PdfPages(OUT/'ir_binwise_diagnostics.pdf') as pdf:
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for i,label in enumerate((r'$\Delta G/\sigma$',r'$100\Delta L/\sigma$',r'$10^4\Delta Q/\sigma$')):
            axes[i].plot(np.arange(len(idx)),changes[i]/sigma*100**i,'o-',ms=3)
            axes[i].axhline(0,color='gray',lw=.7);axes[i].set_ylabel(label);axes[i].grid(alpha=.2)
        axes[2].set_xlabel('certified B000 bin position (kmax <= 0.20 h/Mpc)')
        fig.suptitle('strict LO IR minus no IR; nuisance prior mean; effective V=42.875 errors')
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        spectrum=dict(np.load(OUT/'spectrum_split_s0.25.npz'))
        selected=(spectrum['k']>=.005)&(spectrum['k']<=.4)
        k=spectrum['k'][selected];w=spectrum['pwiggle'][selected]/spectrum['pnw'][selected]
        fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(k,w,label='linear wiggle / smooth')
        sigma2=scan['template_metadata']['sigma2_by_kir']['0.15']
        ax.plot(k,np.exp(-k*k*sigma2)*w,label='LO damped wiggle / smooth')
        ax.set_xlabel(r'$k\ [h/{\rm Mpc}]$');ax.set_ylabel(r'$P_w/P_{nw}$');ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print(json.dumps({k:v for k,v in data.items() if k not in ('binwise','bao_sensitive_bin_ablation')},indent=2),flush=True)

if __name__=='__main__':main()
