"""模块大纲：纯 B whitened likelihood；盲多起点 MAP；四独立 ensemble；完整诊断。

复用第二部分已有独立链诊断，不加载其 P 数据/理论/先验。链可恢复，不覆盖旧目录。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import ast
import json
import math
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import emcee
from iminuit import Minuit
from scipy.stats import chi2
from rsd_b000_contract import OUT,PART,digest,save_json

sys.path.insert(0,str(PART.parent/'02_reconstruction_on_pk/code'))
from run_five_method_p0only_mcmc import independent_diagnostics

def compatible_sampler_hashes():
    """旧来源只允许已审计的 budget24k 版本；模型、诊断和采样 kernel 的 AST 必须一致。"""
    current=Path(__file__);old=PART/'work/source_snapshots/run_rsd_b000_mcmc_budget24k.py'
    allowed={digest(current)}
    if old.exists():
        expected='6be2c9d740a6a0a9bcaf79702a01df7ddf0813ee3228605add8f262a1676c135'
        if digest(old)!=expected:raise ValueError('archived sampler source changed')
        trees=[ast.parse(p.read_text()) for p in (old,current)]
        for name in ('Case','diagnose','checkpoint_targets'):
            nodes=[next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name) for t in trees]
            if ast.dump(nodes[0])!=ast.dump(nodes[1]):raise ValueError('not a budget-only compatible change: '+name)
        loops=[]
        for tree in trees:
            runnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            loop=next(n for n in ast.walk(runnode) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                      and [getattr(x,'id',None) for x in n.target.elts]==['ensemble','backend'])
            loops.append(ast.dump(loop))
        if loops[0]!=loops[1]:raise ValueError('proposal/RNG/backend loop changed; not a budget-only continuation')
        allowed.add(expected)
    return allowed

def budget_provenance(directory,provenance,extend_budget=False):
    """显式增加步数预算；原 RUN_PROVENANCE 不覆盖，扩展来源另行累积记录。"""
    path=directory/'RUN_PROVENANCE.json';summary=directory/'SUMMARY.json';allowed=compatible_sampler_hashes()
    if not path.exists():save_json(path,provenance);return False
    previous=json.loads(path.read_text())
    if previous['sampler_sha256'] not in allowed:raise ValueError('unknown sampler source; do not resume')
    if set(previous)!=set(provenance):raise ValueError('sampler provenance schema drift')
    for key in provenance:
        if key not in ('sampler_sha256','max_steps') and previous[key]!=provenance[key]:
            raise ValueError('likelihood/diagnostics/sampler settings changed: '+key)
    if summary.exists():
        report=json.loads(summary.read_text())
        if report['likelihood_sha256']!=provenance['likelihood_sha256']:raise ValueError('summary likelihood drift')
        if report['converged']:return True  # 仅只读复用，不改已收敛的链或初始来源。
    if previous==provenance:return False
    if not extend_budget or provenance['max_steps']<=previous['max_steps']:
        raise ValueError('source/budget differs; require explicit budget-only extension')
    target=directory/'BUDGET_EXTENSION_PROVENANCE.json'
    history=json.loads(target.read_text()) if target.exists() else []
    if history:
        last=history[-1]['continuation']
        if last['sampler_sha256']!=provenance['sampler_sha256'] or last['max_steps']>provenance['max_steps']:
            raise ValueError('continuation source changed or budget shrank')
    event={'initial':previous,'continuation':provenance,'budget_only_AST_verified':True,
           'data_covariance_prior_proposals_diagnostics_unchanged':True}
    if not history or history[-1]!=event:history.append(event);save_json(target,history)
    return False

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.bool_,bool)):return bool(value)
    if isinstance(value,(np.integer,int)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    return value

class Case:
    def __init__(self,directory):
        self.directory=directory;self.meta=json.loads((directory/'contract.json').read_text())
        if self.meta['P_likelihood'] or self.meta['P_derived_b1_prior']:raise ValueError('not a B-only case')
        if digest(directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('likelihood hash mismatch')
        with np.load(directory/'likelihood.npz') as z:
            for name in z.files:setattr(self,name,np.array(z[name]))
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.target.shape!=(self.meta['B_bins'],) or self.noise.shape!=(self.ndim-4,len(self.target)):
            raise ValueError('B-only dimensions inconsistent')
        self.cwhite=np.linalg.solve(self.chol,self.coefficients.reshape(-1,len(self.target)).T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise.T).T;self.twhite=np.linalg.solve(self.chol,self.target)

    def residual(self,physical):
        v=np.atleast_2d(physical);f,b,b2,bk=v[:,:4].T
        cb=np.polynomial.chebyshev.chebvander((b-2.25)/1.75,3)
        nu=np.column_stack((np.ones(len(b)),b2,bk));ff=np.column_stack((np.ones(len(b)),f,f*f))
        features=(cb[:,:,None,None]*nu[:,None,:,None]*ff[:,None,None,:]).reshape(len(b),-1)
        return features@self.cwhite+v[:,4:]@self.nwhite-self.twhite

    def objective(self,*physical):
        r=self.residual(physical)[0];return float(r@r)

    def initialize(self):
        target=self.directory/'MAP.json'
        if target.exists():
            m=json.loads(target.read_text());self.center=np.array(m['center']);self.transform=np.array(m['transform']);return m
        candidates=[]
        # 起点对三种真值完全相同；不把模拟真值用作拟合中心或先验中心。
        for f0 in (-200.,0.,200.):
            for b0 in (1.25,1.93,2.8):
                start=np.array([f0,b0,-.25,-.3]+[0.]*(self.ndim-4))
                fit=Minuit(self.objective,*start,name=self.names);fit.errordef=1
                for name,bound,error in zip(self.names,self.bounds,[30.,.1,.2,.2]+[.3]*(self.ndim-4)):
                    fit.limits[name]=bound;fit.errors[name]=error
                fit.migrad(ncall=12000)
                if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=12000)
                fit.hesse()
                candidates.append((float(fit.fval),np.array(list(fit.values)),fit))
        candidates=[item for item in candidates if np.isfinite(item[0]) and np.isfinite(item[1]).all()]
        if not candidates:raise ValueError('all blind MAP starts failed')
        candidates.sort(key=lambda item:item[0]);score,center,fit=candidates[0]
        matrix=np.array(fit.covariance) if fit.covariance is not None else None
        if matrix is None or not np.isfinite(matrix).all() or np.any(np.diag(matrix)<=0):
            matrix=np.diag(((self.bounds[:,1]-self.bounds[:,0])/20)**2)
        scale=np.sqrt(np.maximum(np.diag(matrix),1e-16));corr=matrix/np.outer(scale,scale)
        eigen,u=np.linalg.eigh((corr+corr.T)/2);corr=(u*np.maximum(eigen,1e-6))@u.T
        transform=np.diag(scale)@np.linalg.cholesky(corr)
        self.center=center;self.transform=transform
        meta={'center':center.tolist(),'transform':transform.tolist(),'chi2_MAP':score,'minuit_valid':bool(fit.valid),
              'minuit_covariance_accurate':bool(fit.fmin.has_accurate_covar),
              'candidates':[{'chi2':x[0],'parameters':x[1].tolist(),'valid':bool(x[2].valid)} for x in candidates],
              'likelihood_sha256':self.meta['likelihood_sha256'],'blind_start_grid':{'fNL':[-200,0,200],'b1':[1.25,1.93,2.8]}}
        save_json(target,meta);return meta

    def physical(self,coordinates):return np.asarray(coordinates)@self.transform.T+self.center
    def coordinates(self,physical):return np.linalg.solve(self.transform,(np.asarray(physical)-self.center).T).T

    def log_prob(self,coordinates):
        physical=self.physical(coordinates)
        valid=np.all((physical>self.bounds[:,0])&(physical<self.bounds[:,1]),axis=1)
        value=np.full(len(physical),-np.inf)
        if np.any(valid):
            r=self.residual(physical[valid]);value[valid]=-.5*np.sum(r*r,axis=1)
        return value

    def initial(self,seed,ensemble,mapmeta):
        rng=np.random.default_rng(seed);best=np.array(mapmeta['center'])
        # 有多个近似模式时，独立 ensembles 从不同已优化候选启动。
        good=[r for r in mapmeta['candidates'] if r['chi2']<=mapmeta['chi2_MAP']+10 and r['valid']]
        if ensemble>=2 and good:best=np.array(good[(ensemble-2)%len(good)]['parameters'])
        center=self.coordinates(best);points=[]
        for tries in range(100000):
            point=center+rng.normal(size=self.ndim)*(.7 if tries<5000 else .15)
            if np.isfinite(self.log_prob(point[None])[0]):points.append(point)
            if len(points)==32:return np.array(points)
        raise ValueError('unable to initialize walkers within frozen priors')

def diagnose(case,chains,accepted,steps):
    retained=case.physical(chains[:,steps//2:]);diagnostics=independent_diagnostics(retained)
    tau=[]
    for values in retained:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tau.append(emcee.autocorr.integrated_time(values,quiet=True))
    ratio=retained.shape[1]/np.max(tau,axis=0)
    flat=retained.reshape(-1,case.ndim);sigma=np.std(flat,axis=0,ddof=1);half=retained.shape[1]//2
    shift=np.abs(np.median(retained[:,:half].reshape(-1,case.ndim),axis=0)-np.median(retained[:,half:].reshape(-1,case.ndim),axis=0))/sigma
    spread=np.ptp(np.median(retained,axis=(1,2)),axis=0)/sigma
    acceptance=float(np.mean(accepted/steps));quantiles=np.quantile(flat,[.025,.16,.5,.84,.975],axis=0)
    width=case.bounds[:,1]-case.bounds[:,0]
    boundary=np.mean((flat<case.bounds[:,0]+.02*width)|(flat>case.bounds[:,1]-.02*width),axis=0)
    gates={'rhat':bool(np.all(diagnostics['rank_normalized_split_rhat']<1.01)),
           'fNL_ESS':bool(diagnostics['bulk_ess'][0]>2000 and diagnostics['tail_ess'][0]>2000),
           'other_ESS':bool(np.all(diagnostics['bulk_ess'][1:]>1000)&np.all(diagnostics['tail_ess'][1:]>1000)),
           'length_tau':bool(np.all(ratio>50)),'half_shift':bool(np.all(shift<.1)),
           'ensemble_spread':bool(np.all(spread<.2)),'acceptance':bool(.1<acceptance<.7)}
    meta={**diagnostics,'gates':gates,'converged':all(gates.values()),'steps':steps,'retained_steps':retained.shape[1],
          'length_over_tau':ratio,'half_median_shift_sigma':shift,'ensemble_median_spread_sigma':spread,
          'acceptance':acceptance,'quantiles_2p5_16_50_84_97p5':quantiles,'boundary_mass_each_parameter':boundary,
          'boundary_mass_note':'prior sensitivity flag, separate from sampler convergence',
          'fNL_pull':float((quantiles[2,0]-case.meta['truth_fNL'])/sigma[0]),'fNL_sigma':float(sigma[0]),
          'prior_sensitive':bool(np.any(boundary>.01))}
    return meta,flat

def checkpoint_targets(iterations,max_steps=24000):
    # 作业中断可能只写完部分 ensembles；先补齐最长链，再比较等长的独立链。
    longest=max(iterations,default=0)
    first=max(6000,int(math.ceil(longest/2000))*2000)
    if first>max_steps:raise ValueError('existing chain exceeds frozen sampler budget')
    return range(first,max_steps+1,2000)

def run(case_index,max_steps=24000,extend_budget=False):
    if max_steps<24000 or max_steps>64000 or max_steps%2000:raise ValueError('budget must be 24000..64000 in 2000-step increments')
    cases=json.loads((OUT/'CASES.json').read_text())
    if case_index>=len(cases):
        print('SKIP unused sensitivity array slot',case_index,flush=True);return
    name=cases[case_index];case=Case(OUT/'cases'/name)
    provenance={'likelihood_sha256':case.meta['likelihood_sha256'],'sampler_sha256':digest(Path(__file__)),
                'diagnostics_sha256':digest(PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py'),
                'emcee_version':emcee.__version__,'ensembles':4,'walkers':32,'max_steps':max_steps}
    summarypath=case.directory/'SUMMARY.json'
    if budget_provenance(case.directory,provenance,extend_budget):
        print('REUSE CONVERGED '+name,flush=True);return
    started=time.time();mapmeta=case.initialize();backends=[]
    for ensemble in range(4):
        backend=emcee.backends.HDFBackend(str(case.directory/'chain.h5'),name=f'ensemble_{ensemble}')
        if not backend.initialized:backend.reset(32,case.ndim)
        backends.append(backend)
    finished=None
    for steps in checkpoint_targets([b.iteration for b in backends],max_steps=max_steps):
        for ensemble,backend in enumerate(backends):
            seed=20261001+100003*case_index+1013*ensemble
            sampler=emcee.EnsembleSampler(32,case.ndim,case.log_prob,vectorize=True,backend=backend,
                          moves=[(emcee.moves.StretchMove(a=2),.8),(emcee.moves.DEMove(),.2)])
            if backend.iteration==0:
                initial=case.initial(seed,ensemble,mapmeta);sampler.random_state=np.random.RandomState(seed+19).get_state()
            else:initial=None
            remaining=max(0,steps-backend.iteration)
            if remaining:sampler.run_mcmc(initial,remaining,progress=False,skip_initial_state_check=True)
        chains=np.stack([b.get_chain() for b in backends]);actual=chains.shape[1]
        accepted=np.stack([b.accepted for b in backends]);report,flat=diagnose(case,chains,accepted,actual)
        dof=len(case.target)-case.ndim
        report.update({'name':name,'cosmology':case.meta['cosmology'],'truth_fNL':case.meta['truth_fNL'],
                       'kmax_B':case.meta['kmax_B'],'volume_Gpc_h3':case.meta['volume_Gpc_h3'],
                       'case_type':case.meta['case_type'],'parameters':case.names,'chi2_MAP':mapmeta['chi2_MAP'],
                       'dof':dof,'gof_pvalue':float(chi2.sf(mapmeta['chi2_MAP'],dof)) if dof>0 else 0.,
                       'gof_note':'descriptive Gaussian chi2 with finite covariance and bounded priors; not exact posterior predictive p-value',
                       'P_likelihood':False,'P_derived_b1_prior':False,'elapsed_seconds':time.time()-started,
                       'likelihood_sha256':case.meta['likelihood_sha256']})
        save_json(summarypath,clean(report));print(name+' '+json.dumps(clean({'steps':actual,'converged':report['converged'],'gates':report['gates'],
                    'fNL_median':report['quantiles_2p5_16_50_84_97p5'][2,0],'rhat':report['rank_normalized_split_rhat']})),flush=True)
        finished=report
        if report['converged']:break
    rng=np.random.default_rng(5500+case_index);pick=rng.choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=np.array(case.names))
    print('COMPLETE '+name+' converged='+str(finished['converged']),flush=True)
    # 未收敛组仍产出明确诊断并允许其他组完成；最终报告不会将其用于有效 kmax 判定。

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case-index',type=int,required=True)
    parser.add_argument('--max-steps',type=int,default=24000);parser.add_argument('--extend-budget',action='store_true')
    args=parser.parse_args();run(args.case_index,args.max_steps,args.extend_budget)
