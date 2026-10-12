#!/usr/bin/env python3
"""B-only 五点注入的 IR/no-IR 同合同 kmax 检验。

模块大纲：native 原子项组合；保持 nuisance 先验和协方差；解析 profile
四个线性 nuisance；一维全域搜索 fNL 并求 Delta-chi2=1 区间；PDF/JSON 审计。
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar,brentq
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'halo/outputs/local_png_tree_ir_20260930'
VOLUME=42.875; B1=1.93; DC=1.686
TRUTHS=(-100.,-50.,0.,50.,100.)
CUTS=(.08,.10,.12,.14,.15,.18,.20)
NUISANCE=('b2','gamma2','Ashot_residual','Bshot_residual')
PARTS=('gaussian','linear_coefficient','quadratic_coefficient')

def legacy_design(inputs):
    return np.stack([np.stack([inputs[f'legacy_zero_{part}']]+[
        inputs[f'legacy_{n}_{part}'] for n in NUISANCE],axis=-1) for part in PARTS])

def components(raw,lattice,theta,pbias,nbar,adapter):
    b2,gamma,ashot,bshot=theta
    bn=b2-4*gamma/3
    bp=2*DC*(B1-pbias)
    # 延续当前 finite-tree 的独立 bias 合同：改变 bphi 的 p 不改 universal bphidelta。
    bpd=2*DC*(B1-1)+2*(DC*(bn-8*(B1-1)/21)-B1+1)
    bp2=4*DC*(DC*(bn-8*(B1-1)/21)-2*(B1-1))
    g=B1**3*lattice[:,21]+B1**2*bn*lattice[:,22]+B1**2*gamma*lattice[:,23]
    g+=bshot/nbar*B1**2*lattice[:,17]+ashot/nbar**2*np.ones(120)
    l=B1**3*raw[:,2]+B1**2*bp*(raw[:,3]+raw[:,4])+B1**2*bpd*raw[:,5]
    l+=B1*bn*bp*raw[:,6]+B1*gamma*bp*raw[:,7]
    l+=bshot/nbar*B1*bp*raw[:,18]*adapter
    q=B1**2*bp*raw[:,9]+B1*bp**2*(raw[:,10]+raw[:,11])
    q+=bp**2*bn*raw[:,12]+bp**2*gamma*raw[:,13]+B1*bp*bpd*raw[:,15]+B1**2*bp2*raw[:,16]
    q+=bshot/nbar*bp**2*raw[:,19]*adapter
    return np.stack((g,l,q))

def design_from_raw(raw,lattice,pbias,nbar,adapter):
    base=components(raw,lattice,np.zeros(4),pbias,nbar,adapter)
    return np.stack([base]+[components(raw,lattice,np.eye(4)[i],pbias,nbar,adapter)-base for i in range(4)],axis=-1)

def polynomial(design,fnl):return design[0]+fnl*design[1]+fnl**2*design[2]

class Profile:
    def __init__(self,design,target,covariance,prior_mean,prior_whitener,volume=VOLUME):
        if not np.isfinite(volume) or volume<=0:
            raise ValueError('拟合体积必须是有限正数，单位为 (Gpc/h)^3')
        self.design=design;self.target=target
        self.volume=float(volume)
        n=len(target);self.hartlap=(500-n-2)/499.
        # 只缩放调查体积对应的协方差，不改变输入均值和 nuisance 先验。
        self.chol=np.linalg.cholesky(covariance/self.volume)/np.sqrt(self.hartlap)
        self.white=np.stack([solve_triangular(self.chol,d,lower=True) for d in design])
        self.y=solve_triangular(self.chol,target,lower=True)
        self.prior_matrix=prior_whitener[:,1:]
        self.prior_rhs=prior_whitener@(prior_mean-np.array([B1,0,0,0,0]))
    def evaluate(self,fnl,details=False):
        d=polynomial(self.white,fnl)
        a=np.vstack((d[:,1:],self.prior_matrix));b=np.r_[self.y-d[:,0],self.prior_rhs]
        theta=np.linalg.lstsq(a,b,rcond=None)[0]
        residual=a@theta-b
        q=float(residual@residual)
        if not details:return q
        n=len(self.target);prediction=polynomial(self.design,fnl)@np.r_[1.,theta]
        return {'objective':q,'data_chi2':float(residual[:n]@residual[:n]),
                'prior_chi2':float(residual[n:]@residual[n:]),'nuisance':theta.tolist(),
                'prediction':prediction.tolist()}
    def fit(self,truth):
        lower,upper=truth-200,truth+200
        grid=np.linspace(lower,upper,401);values=np.array([self.evaluate(x) for x in grid])
        local=np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
        candidates=[]
        for i in local:
            result=minimize_scalar(self.evaluate,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1.e-7})
            candidates.append((float(result.fun),float(result.x),bool(result.success)))
        candidates += [(float(values[0]),lower,False),(float(values[-1]),upper,False)]
        qbest,best,valid=min(candidates)
        details=self.evaluate(best,True);fixed=self.evaluate(truth,True)
        bounds=[];censored=[]
        for endpoint in (lower,upper):
            probe=np.linspace(best,endpoint,401)
            diff=np.array([self.evaluate(x)-qbest-1 for x in probe])
            crossings=np.flatnonzero(diff>=0)
            if not len(crossings):bounds.append(endpoint);censored.append(True)
            else:
                i=int(crossings[0]);a,b=sorted((probe[i-1],probe[i]))
                bounds.append(float(brentq(lambda x:self.evaluate(x)-qbest-1,a,b,xtol=1.e-7)));censored.append(False)
        return {'truth':truth,'fNL':best,'interval':bounds,'optimizer_valid':valid,
                'interval_censored':any(censored),'local_minima':len(local),
                'truth_delta_chi2':max(0.,fixed['objective']-qbest),
                'truth_profile':fixed,'best_fit':details}

def summary(rows):
    result={};largest=None;failed=False
    for cut in CUTS:
        local=[r for r in rows if r['kmax']==cut]
        if not local:continue
        fid=next(r for r in local if r['truth']==0)
        dof=fid['ndata']-4
        p=float(chi2.sf(fid['truth_profile']['data_chi2'],dof))
        cover=sum(r['truth_delta_chi2']<=1+1.e-7 for r in local)
        valid=all(r['optimizer_valid'] and not r['interval_censored'] for r in local)
        passed=p>=.05 and cover==5 and valid
        if not passed:failed=True
        if not failed:largest=cut
        result[f'{cut:.2f}']={'gaussian_p':p,'coverage':cover,'max_truth_delta_chi2':max(r['truth_delta_chi2'] for r in local),
                           'all_intervals_valid':valid,'pass':passed,
                           'biases':[r['fNL']-r['truth'] for r in local]}
    return {'by_kmax':result,'largest_contiguous_valid_kmax':largest}

def plot(payload,path):
    colors={'fixed_order':'#666666','strict_kir0.15':'#d95f02','factorized_kir0.15':'#377eb8','strict_zero':'#984ea3'}
    volume=payload['audit']['volume_gpc3_h3']
    with PdfPages(path) as pdf:
        for bias in ('current','legacy'):
            fig,axes=plt.subplots(1,2,figsize=(11,4.6))
            for name,color in colors.items():
                result=payload['summary'].get(bias+'__'+name)
                if result is None:continue
                cuts=[float(x) for x in result['by_kmax']]
                axes[0].plot(cuts,[v['max_truth_delta_chi2'] for v in result['by_kmax'].values()],'-o',color=color,label=name)
                axes[1].plot(cuts,[v['gaussian_p'] for v in result['by_kmax'].values()],'-o',color=color)
            axes[0].axhline(1,color='black',ls='--',lw=.8);axes[1].axhline(.05,color='black',ls='--',lw=.8)
            axes[0].set_yscale('symlog',linthresh=1);axes[1].set_yscale('log')
            axes[0].set_ylabel(r'max profiled $\Delta\chi^2(f_{NL}^{true})$');axes[1].set_ylabel('Gaussian fixed-truth p-value')
            for ax in axes:ax.set_xlabel(r'$k_{max}\ [h/{\rm Mpc}]$');ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.suptitle(f'Pre-reconstruction B-only; {bias} p calibration; V={volume:g} (Gpc/h)^3')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        for cut in (.08,.10,.12,.14,.20):
            fig,ax=plt.subplots(figsize=(7,6))
            for offset,(name,color) in enumerate(colors.items()):
                local=[r for r in payload['rows'] if r['bias']=='current' and r['variant']==name and r['kmax']==cut]
                if not local:continue
                truth=np.array([r['truth'] for r in local]);fit=np.array([r['fNL'] for r in local])
                interval=np.array([r['interval'] for r in local])
                ax.errorbar(truth+(offset-1.5)*2,fit,yerr=np.vstack((fit-interval[:,0],interval[:,1]-fit)),fmt='o',capsize=2,color=color,label=name)
            ax.plot([-120,120],[-120,120],'k--',lw=.8);ax.set_xlabel(r'input $f_{NL}$');ax.set_ylabel(r'recovered $f_{NL}$')
            ax.set_title(f'Current p calibration, B-only, kmax={cut:.2f} h/Mpc\nV={volume:g} (Gpc/h)^3');ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates',default='tree_ir_templates_r4_m80_s0.25_rbao110.0.npz')
    parser.add_argument('--tag',default='primary');parser.add_argument('--variants',nargs='*')
    parser.add_argument('--volume',type=float,default=VOLUME,help='拟合体积，单位 (Gpc/h)^3；默认保持历史结果')
    args=parser.parse_args()
    if not np.isfinite(args.volume) or args.volume<=0:
        parser.error('--volume 必须是有限正数')
    inputs=dict(np.load(OUT/'frozen_inputs.npz'));info=json.loads((OUT/'frozen_inputs.json').read_text())
    raw=dict(np.load(OUT/args.templates));meta=json.loads((OUT/args.templates).with_suffix('.json').read_text())
    indices=inputs['certified_indices'];nbar=info['nbar']
    historical=legacy_design(inputs)
    # 保留旧 finite-PNG stochastic 的固定投影适配器；IR 不得重新定义此分母。
    adapter=np.ones(120)
    adapter[indices]=inputs['legacy_Bshot_residual_gaussian'][indices]*nbar/(B1**2*raw['fixed_order__continuum'][indices,17])
    designs={'historical':historical};rows=[];summaries={}
    variants=[v[0] for v in meta['variants'] if args.variants is None or v[0] in args.variants]
    for name in variants:
        for bias,pbias in (('current',info['p_current']),('legacy',info['p_legacy'])):
            designs[bias+'__'+name]=design_from_raw(raw[name+'__continuum'],raw[name+'__lattice'],pbias,nbar,adapter)
    for key,design in designs.items():
        bias,variant=key.split('__') if '__' in key else ('legacy','historical')
        localrows=[]
        for cut in CUTS:
            idx=indices[np.max(inputs['geometry'][indices],axis=1)<=cut+1.e-12]
            cov=np.cov(inputs['data_+0'][:,idx],rowvar=False,ddof=1)
            for truth in TRUTHS:
                target=inputs[f'data_{truth:+.0f}'][:,idx].mean(axis=0)
                fit=Profile(design[:,idx,:],target,cov,inputs['prior_mean'],inputs['prior_whitener'],volume=args.volume).fit(truth)
                fit.update(bias=bias,variant=variant,kmax=cut,ndata=len(idx),indices=idx.tolist())
                rows.append(fit);localrows.append(fit)
            print(f'{key} kmax={cut:.2f} max Dchi2={max(r["truth_delta_chi2"] for r in localrows[-5:]):.5f}',flush=True)
        summaries[key]=summary(localrows)
    scale=np.maximum(np.abs(historical),1.)
    difference=designs.get('legacy__fixed_order',historical)-historical
    audit={'historical_vs_recomputed_relative_l2_by_order':[float(np.linalg.norm(difference[i,indices])/max(1.,np.linalg.norm(historical[i,indices]))) for i in range(3)],
           'no_IR_Qstochastic_change_max':float(np.max(np.abs(raw['strict_kir0.15__continuum'][indices,19]-raw['fixed_order__continuum'][indices,19]))),
           'statistic':'B000 only','b1_fixed':B1,'volume_gpc3_h3':args.volume,
           'fNL_bounds':'truth +/- 200 (same injection validation convention)',
           'failure_rule':'Gaussian p>=0.05; 5/5 truths within profiled Delta-chi2<=1; valid uncensored intervals',
           'nuisance_names':NUISANCE,'p_current':info['p_current'],'p_legacy':info['p_legacy'],
           'projection':'Gaussian exact-lattice; deterministic PNG finite-box continuum; inherited fixed stochastic projection adapter',
           'kmax_definition':'cut on the two binned B000 legs; closing leg k3 is integrated and may exceed kmax',
           'kmax0p14_0p15_same_indices':True}
    payload={'status':'computed','audit':audit,'template_metadata':meta,'summary':summaries,'rows':rows}
    destination=OUT/f'ir_kmax_scan_{args.tag}.json'
    destination.write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(OUT/f'ir_model_designs_{args.tag}.npz',**designs)
    plot(payload,OUT/f'ir_kmax_scan_{args.tag}.pdf')
    print(json.dumps({'audit':audit,'decisions':{k:v['largest_contiguous_valid_kmax'] for k,v in summaries.items()}},indent=2),flush=True)

if __name__=='__main__':main()
