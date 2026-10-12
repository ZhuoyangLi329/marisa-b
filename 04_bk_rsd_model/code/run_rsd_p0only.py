"""模块大纲：从联合结果冻结三组 P0-only；三参数适配；复用原 MCMC 流程。

使用相同 mean100、Fid500 C_PP、P window/模型/先验，Hartlap 按 12 维计算。
只采样 fNL、b1、Pshot；B 专属且不进入 P 的 flat 参数积分为常数。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from iminuit import Minuit
from rsd_pb_common import OUT as PB_OUT, B_OUT, PART, P_CAL, F_GROWTH, p_prediction, digest, save_json
import run_rsd_b000_mcmc as shared

OUT=PART/'outputs/pre_rsd_p0only_mean100_v1'
COSMOS=('fiducial','LC_p','LC_m')
PARAMETERS=['fNL','b1','Pshot_residual_nbar']
SAMPLER_SHA='018d3d4bb2db0dfdd739b7933472716bdb1c866d2f7d9aa57dca3f463456614d'
LIVE_BINNER='/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/code/jaxpower/main/lib/python3.12/site-packages/jaxpower/mesh2.py'


def parent():
    value=json.loads((OUT/'contract.json').read_text())
    for path,sha in value['immutable_inputs'].items():
        if digest(path)!=sha:raise ValueError('frozen P0-only input/source drift: '+path)
    return value


def freeze():
    if (OUT/'contract.json').exists():
        parent();print('REUSE frozen P0-only inputs',flush=True);return
    # 2026-10-08 共享 jaxpower 已更新；本次只读取冻结测量/window/template，不调用 live binner。
    # 旧 P+B 守卫保持原样；此处逐项审计，并仅将不执行的外部测量源码列为历史依赖。
    pb=json.loads((PB_OUT/'contract.json').read_text());measurement_changes=[]
    for path,want in pb['immutable_inputs'].items():
        got=digest(path)
        if got!=want:
            if path!=LIVE_BINNER:raise ValueError('upstream input/source drift: '+path)
            measurement_changes.append({'path':path,'frozen_sha256':want,'current_sha256':got,
                                        'executed_in_this_P0_only_fit':False})
    upstream=json.loads((PB_OUT/'COMPLETION_AUDIT.json').read_text())
    if digest(PB_OUT/'RESULTS.json')!=upstream['RESULTS_sha256']:raise ValueError('joint results drift')
    if digest(PB_OUT/'P0_LATTICE.npz')!=pb['p_template_sha256']:raise ValueError('frozen P template drift')
    matrix_sha=json.loads((PB_OUT/'JOINT_FREEZE_AUDIT.json').read_text())['P0_ensemble_sha256']
    if digest(PB_OUT/'P0_ENSEMBLE_MATRIX.npz')!=matrix_sha:raise ValueError('frozen P measurements drift')
    upstream_cases={r['name']:r['likelihood_sha256'] for r in upstream['cases']}
    if digest(Path(shared.__file__))!=SAMPLER_SHA:raise ValueError('original sampler changed')
    if pb['p_fixed']!=P_CAL or pb['f_growth_fixed']!=F_GROWTH:raise ValueError('fixed physics drift')
    with np.load(PB_OUT/'P0_ENSEMBLE_MATRIX.npz') as z:
        p=np.array(z['P0']);full_cov=np.array(z['covariance_single_box'])
        metadata=[json.loads(str(m)) for m in z['metadata_json']]
    covmask=np.array([m['role']=='covariance' for m in metadata])
    if covmask.sum()!=500 or {m['realization'] for m in metadata if m['role']=='covariance'}!=set(range(1000,1500)):
        raise ValueError('wrong Fid500 sample')
    covariance=np.cov(p[covmask],rowvar=False,ddof=1)
    np.testing.assert_allclose(covariance,full_cov[:12,:12],rtol=2e-13)
    alpha=(500-12-2)/499
    with np.load(PB_OUT/'P0_LATTICE.npz') as z:templates=np.array(z['templates'])
    sources=[Path(__file__),Path(shared.__file__),PART/'code/rsd_pb_common.py',
             PART/'code/rsd_bonly_common.py',PART/'code/run_rsd_pb_inference.py',
             PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',
             PB_OUT/'contract.json',PB_OUT/'RESULTS.json',PB_OUT/'RSD_PB_VALIDATION.pdf',
             PB_OUT/'P0_ENSEMBLE_MATRIX.npz',PB_OUT/'P0_LATTICE.npz',PB_OUT/'P0_WINDOW.npz',
             PB_OUT/'COMPLETION_AUDIT.json',PB_OUT/'JOINT_FREEZE_AUDIT.json',
             B_OUT/'RESULTS.json',B_OUT/'RSD_B000_VALIDATION.pdf']
    sources.extend(Path(p) for p in pb['immutable_inputs'] if p!=LIVE_BINNER)
    names=[];audits=[]
    for cosmo in COSMOS:
        ref=PB_OUT/'cases'/f'{cosmo}_mean100_k0.08'
        meta=json.loads((ref/'contract.json').read_text())
        with np.load(ref/'likelihood.npz') as z:
            target=np.array(z['target'][:12]);bounds=np.array(z['bounds'])[[0,1,7]]
            noise=np.array(z['noise'][-1,:12])
        mask=np.array([m['role']=='validation' and m['cosmology']==cosmo for m in metadata])
        if mask.sum()!=100 or {m['realization'] for m in np.array(metadata,dtype=object)[mask]}!=set(range(100)):
            raise ValueError('wrong target IC set')
        np.testing.assert_allclose(target,p[mask].mean(axis=0),rtol=1e-13)
        np.testing.assert_array_equal(noise,np.full(12,meta['Pshot_unit_Mpc_h3']))
        # 六档联合拟合的 P 输入与对应先验必须完全一致，P-only 只需每真值一组。
        for cut in pb['B_cuts']:
            case=PB_OUT/'cases'/f'{cosmo}_mean100_k{cut:.2f}'
            cm=json.loads((case/'contract.json').read_text())
            if digest(case/'likelihood.npz')!=cm['likelihood_sha256']:raise ValueError('P+B likelihood drift')
            if cm['likelihood_sha256']!=upstream_cases[case.name]:raise ValueError('joint completion-audit mismatch')
            with np.load(case/'likelihood.npz') as z:
                np.testing.assert_array_equal(target,z['target'][:12])
                np.testing.assert_array_equal(bounds,z['bounds'][[0,1,7]])
                np.testing.assert_array_equal(noise,z['noise'][-1,:12])
                np.testing.assert_allclose(covariance,z['covariance'][:12,:12],rtol=2e-13)
            sources.extend(case/p for p in ('contract.json','likelihood.npz','SUMMARY.json','RUN_PROVENANCE.json'))
        name=f'{cosmo}_mean100_kP0.08';names.append(name);case=OUT/'cases'/name
        case.mkdir(parents=True,exist_ok=True)
        if (case/'chain.h5').exists():raise ValueError('cannot refreeze existing chains')
        np.savez_compressed(case/'likelihood.npz',target=target,covariance=covariance,
                           chol=np.linalg.cholesky(covariance/alpha),templates=templates,
                           noise=noise,bounds=bounds,P_indices=np.arange(12))
        cm={'name':name,'cosmology':cosmo,'truth_fNL':meta['truth_fNL'],'case_type':'P0_only',
            'parameters':PARAMETERS,'bounds':bounds.tolist(),'P_bins':12,'B_bins':0,'data_bins':12,
            'kmax_P':.08,'kmax_B':None,'P_likelihood':True,'B_likelihood':False,
            'P_derived_b1_prior':False,'P_B_cross_covariance':False,'n_target':100,'n_covariance':500,
            'target_realizations':list(range(100)),'covariance_realizations':list(range(1000,1500)),
            'volume_Gpc_h3':1.,'covariance_multiplier':1.,'covariance_is_of_mean':False,'Hartlap_factor':alpha,
            'p_fixed_external':P_CAL,'f_growth_fixed':F_GROWTH,'IR':False,'FoG':False,'loops':False,
            'Pshot_unit_Mpc_h3':meta['Pshot_unit_Mpc_h3'],'Pshot_physical_bounds':[-1e6,1e6],
            'source_PB_case':str(ref),'source_PB_likelihood_sha256':digest(ref/'likelihood.npz'),
            'likelihood_sha256':digest(case/'likelihood.npz'),'scientifically_certified':False}
        save_json(case/'contract.json',cm)
        audits.append({'name':name,'same_P_data_model_bounds_all_six_PB_cases':True,
                       'unconditional_C_PP_not_conditional_covariance':True,'Hartlap_factor':alpha})
    sources=list(dict.fromkeys(sources))
    save_json(OUT/'CASES.json',names)
    save_json(OUT/'contract.json',{'observable':'P0-only; three independent truth cases at kmax_P=.08',
        'immutable_inputs':{str(p):digest(p) for p in sources},'parameters':PARAMETERS,
        'sampler_sha256':SAMPLER_SHA,'data_mean_count':100,'covariance_sample_count':500,
        'covariance_multiplier':1.,'covariance_is_of_mean':False,'volume_Gpc_h3':1.,
        'kmax_P':.08,'P_bins':12,'p_fixed':P_CAL,'f_growth_fixed':F_GROWTH,
        'statistical_interpretation':'mean100 target with single-box forecast precision, not sampling likelihood of mean',
        'maximum_logical_CPUs':8,'scientifically_certified':False})
    save_json(OUT/'FREEZE_AUDIT.json',{'status':'PASS','cases':audits,'n_cases':3,
        'same_raw_P_measurements_as_PB':True,'remeasurement':False,'covariance_ddof':1,
        'Hartlap_data_dimension':12,'B_only_nuisance_flat_prior_integrals_are_constants':True})
    save_json(OUT/'UPSTREAM_REUSE_AUDIT.json',{'status':'PASS_FOR_FROZEN_ARTIFACT_REUSE',
        'measurement_software_changes':measurement_changes,'other_upstream_immutable_inputs_unchanged':True,
        'frozen_P0_ensemble_sha256':matrix_sha,'frozen_P_template_sha256':pb['p_template_sha256'],
        'live_binner_executed':False,'old_PB_parent_guard_modified':False})
    print('FROZEN 3 P0-only cases, Csingle V1',flush=True)


class POnlyCase(shared.Case):
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'contract.json').read_text())
        if self.meta['B_likelihood'] or not self.meta['P_likelihood']:raise ValueError('not P0-only')
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('likelihood drift')
        with np.load(self.directory/'likelihood.npz') as z:
            for name in z.files:setattr(self,name,np.array(z[name]))
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=PARAMETERS or self.target.shape!=(12,) or self.bounds.shape!=(3,2):raise ValueError('P0-only dimension error')
        alpha=(500-12-2)/499
        if self.meta['Hartlap_factor']!=alpha or self.meta['covariance_multiplier']!=1:raise ValueError('wrong covariance scale')
        np.testing.assert_allclose(self.chol@self.chol.T,self.covariance/alpha,rtol=2e-13)
        self.white_templates=np.linalg.solve(self.chol,self.templates.T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise);self.twhite=np.linalg.solve(self.chol,self.target)

    def predictions(self,physical):
        v=np.atleast_2d(physical)
        return p_prediction(v[:,0,None],v[:,1,None],self.templates)+v[:,2,None]*self.noise

    def residual(self,physical):
        v=np.atleast_2d(physical)
        return p_prediction(v[:,0,None],v[:,1,None],self.white_templates)+v[:,2,None]*self.nwhite-self.twhite

    def initialize(self):
        # 原 sampler 的 MAP 初值是 B 专属的 4+N 维；这里只适配真实的三个 P 参数。
        path=self.directory/'MAP.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['likelihood_sha256']!=self.meta['likelihood_sha256']:raise ValueError('MAP input drift')
            self.center=np.array(result['center']);self.transform=np.array(result['transform']);return result
        candidates=[]
        for f0 in (-200.,0.,200.):
            for b0 in (1.25,1.93,2.8):
                fit=Minuit(self.objective,f0,b0,0.,name=self.names);fit.errordef=1;fit.tol=1e-5
                for name,bounds,error in zip(self.names,self.bounds,(30.,.1,.3)):
                    fit.limits[name]=bounds;fit.errors[name]=error
                fit.migrad(ncall=12000)
                if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=12000)
                fit.hesse()
                candidates.append(fit)
        good=[fit for fit in candidates if fit.valid and fit.covariance is not None and np.isfinite(fit.fval)]
        if not good:raise ValueError('all three-parameter blind MAP/HESSE starts failed')
        fit=min(good,key=lambda f:f.fval)
        self.center=np.array(list(fit.values));self.transform=np.linalg.cholesky(np.array(fit.covariance))
        result={'center':self.center.tolist(),'transform':self.transform.tolist(),'chi2_MAP':float(fit.fval),
                'minuit_valid':bool(fit.valid),'minuit_covariance_accurate':bool(fit.fmin.has_accurate_covar),
                'candidates':[{'chi2':float(f.fval),'parameters':list(f.values),'valid':bool(f.valid)} for f in candidates],
                'likelihood_sha256':self.meta['likelihood_sha256'],
                'blind_start_grid':{'fNL':[-200,0,200],'b1':[1.25,1.93,2.8],'Pshot_residual_nbar':[0]}}
        save_json(path,result);return result


def sample(index,max_steps=64000,extend_budget=False):
    parent()
    original=(shared.OUT,shared.Case,shared.budget_provenance,shared.save_json)
    def provenance(directory,value,extend=False):
        value.update(p0only_adapter_sha256=digest(Path(__file__)),parent_contract_sha256=digest(OUT/'contract.json'))
        return original[2](directory,value,extend)
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            value.update(P_likelihood=True,B_likelihood=False,P_bins=12,B_bins=0,kmax_P=.08,kmax_B=None,
                         P_B_cross_covariance=False,covariance_multiplier=1.,covariance_is_of_mean=False,
                         gof_pvalue=None,gof_note='mean100 target with Csingle; no calibrated GOF p-value',
                         p0only_adapter_sha256=digest(Path(__file__)))
        return original[3](path,value)
    shared.OUT=OUT;shared.Case=POnlyCase;shared.budget_provenance=provenance;shared.save_json=save
    try:return shared.run(index,max_steps,extend_budget)
    finally:shared.OUT,shared.Case,shared.budget_provenance,shared.save_json=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','sample'])
    parser.add_argument('--case-index',type=int,default=0);parser.add_argument('--max-steps',type=int,default=64000)
    parser.add_argument('--extend-budget',action='store_true');args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:sample(args.case_index,args.max_steps,args.extend_budget)
