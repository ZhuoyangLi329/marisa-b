"""模块大纲：人明确批准的provisional EFT12 P+B链；复用原四ensemble sampler。

原mean100/C205_single/P-B cross/p校准；核心先验和Pshot范围保持原值。
完整STO闭合尚未认证，禁止正式fNL/kmax结论；原C500/旧8参数不改。
此adapter只替换12参数预测与初始化新noise坐标，proposal/RNG/HDF/收敛门
继续使用唯一原sampler；没有为采样修改任何数据协方差特征值。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
from iminuit import Minuit
from rsd_b000_contract import PART,digest,save_json,F_GROWTH
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2,NAMES_V2
import run_rsd_b000_mcmc as shared

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
PRE=ROOT/'noise_eft_pb_preflight_v2'
OUT=ROOT/'diagnostic_provisional_eft12_pb_20261010_v1'
APPROVAL=ROOT/'PROVISIONAL_PB_APPROVAL_20261010.json'
COSMOS=('fiducial','LC_p','LC_m')
CASES=[c+'_post_PB_kB0.08_kP0.08_provisional_EFT12' for c in COSMOS]


def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def authority():
    a=read(APPROVAL)
    if (a['status']!='DIRECT_HUMAN_AUTHORIZES_PROVISIONAL_PB_WHILE_FULL_STO_NOISE_VALIDATION_CONTINUES'
        or not a['diagnostic_MCMC_authorized'] or a['full_STO_noise_closure_certified']
        or a['formal_fNL_conclusion_authorized'] or not a['continue_noise_validation']):
        raise ValueError('specific direct human diagnostic permission required; no inferred noise certification')
    return a


def parent():
    authority();c=read(OUT/'CONTRACT.json')
    if c['status']!='PROVISIONAL_PB_EFT12_APPROVED_INPUTS_FROZEN' or not c['diagnostic_only']:
        raise ValueError('not the explicitly authorized provisional inference')
    for p,want in c['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('provisional frozen input/source drift: '+p)
    return c


def freeze():
    authority()
    if OUT.exists():raise ValueError('preserve prior diagnostic chain; never refreeze existing likelihood')
    c=read(PRE/'CONTRACT.json');report=read(PRE/'REPORT.json')
    if (report['status']!='ACTUAL_PB21_FIXED_TRUTH_EFT_CANDIDATE_PREFLIGHT_COMPLETE_NOT_FULL_NOISE_GATE'
        or report['candidate_parameter_count']!=12 or report['n_covariance']!=205
        or not report['Pshot_independent_from_data_only_raw_noise'] or report['full_noise_gate_released']):
        raise ValueError('actual independent-Pshot preflight required; not a formal closure PASS')
    for p,want in c['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('actual preflight frozen source drift')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_eft_candidate_model_v2'))
    if not tests.wasSuccessful() or tests.testsRun!=3:raise ValueError('actual independent-Pshot tests required')
    matrix=PRE/'FROZEN_PB205_PREFLIGHT.npz'
    if digest(matrix)!=report['matrix_sha256']:raise ValueError('actual PB21 matrix drift')
    with np.load(matrix,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files}
    ns=len(a['covariance_ids']);cov=a['covariance_single_box'];chol=a['chol'];alpha=(ns-23)/(ns-1)
    if ns!=205 or cov.shape!=(21,21):raise ValueError('original205 single-box PB21 required')
    np.testing.assert_allclose(chol@chol.T,cov/alpha,rtol=3e-13,atol=1e-8)
    if c['covariance_is_of_mean'] or c['covariance_multiplier']!=1 or c['volume_Gpc_h3']!=1:
        raise ValueError('mean-covariance scaling or volume change forbidden')
    p=c['p_fixed_physical'];OUT.mkdir();paths=[APPROVAL,PRE/'CONTRACT.json',PRE/'REPORT.json',matrix,Path(__file__),
        PART/'code/post_rsd_eft_candidate_model_v1.py',PART/'code/post_rsd_eft_candidate_model_v2.py',
        PART/'code/test_post_rsd_eft_candidate_model_v2.py',PART/'code/post_rsd_stochastic_eft_v1.py',
        PART/'code/post_rsd_noir_polynomial.py',PART/'code/post_rsd_noir_likelihood.py',
        PART/'code/post_rsd_p0_model.py',PART/'code/run_rsd_b000_mcmc.py',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    for t,cosmo in enumerate(COSMOS):
        d=OUT/'cases'/CASES[t];d.mkdir(parents=True)
        marks=a['mark_moments'][t*100:(t+1)*100];n2,n3,nhalo=marks.T;nr=1e9/(30*nhalo)
        units=np.array([n2.mean(),n3.mean(),np.mean(n2*n2),nr.mean(),np.mean(n2*nr)])
        target=a['matched_target_rows'][t*100:(t+1)*100].mean(axis=0)
        bounds=np.array(c['candidate_noise_boxes'][t]);np.testing.assert_array_equal(bounds[:4],
            [[-500,500],[.5,4],[-4,6],[-4,4]])
        np.testing.assert_allclose(bounds[-1]*units[0],[-1e6,1e6],rtol=3e-15)
        payload={k:a[k] for k in ('coefficients','templates','operators','intrinsic','matched_zero')}
        model=EFTCandidateModelV2(payload['coefficients'],payload['templates'],payload['operators'],payload['intrinsic'],
            payload['matched_zero'],units,p,F_GROWTH)
        if model.ndim!=12:raise ValueError('explicit12 candidate parameter order required')
        np.savez_compressed(d/'likelihood.npz',target=target,covariance=cov,chol=chol,
            weight_moments=units,bounds=bounds,**payload)
        save_json(d/'contract.json',dict(name=CASES[t],cosmology=cosmo,truth_fNL=(0,100,-100)[t],case_type='provisional_PB_EFT12',
            parameters=list(NAMES_V2),bounds=bounds.tolist(),likelihood_sha256=digest(d/'likelihood.npz'),
            P_likelihood=True,B_likelihood=True,P_derived_b1_prior=False,P_B_cross_covariance=True,
            n_covariance=205,n_target=100,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
            covariance_ddof=1,Hartlap_factor=alpha,kmax_B=.08,kmax_P=.08,P_bins=12,B_bins=9,IR=False,
            p_fixed_physical=p,p_uncertainty_propagated=False,f_growth_fixed=F_GROWTH,
            covariance_ids=a['covariance_ids'].tolist(),target_ids=list(range(100)),
            fid_target_covariance_overlap=c['fid_target_covariance_overlap'],strict_independent_mock_recovery=False,
            diagnostic_only=True,provisional=True,full_STO_noise_closure_certified=False,
            prior_noise_boxes_have_boundary_flags=True,original8_model_preserved=True,
            original_Pshot_physical_bounds=[-1e6,1e6],authority_sha256=digest(APPROVAL)))
        paths.extend((d/'likelihood.npz',d/'contract.json'))
    save_json(OUT/'CASES.json',CASES);paths.append(OUT/'CASES.json')
    save_json(OUT/'CONTRACT.json',dict(status='PROVISIONAL_PB_EFT12_APPROVED_INPUTS_FROZEN',
        immutable_inputs={str(q):digest(q) for q in paths},diagnostic_only=True,provisional=True,
        human_explicitly_approved_before_full_noise_closure=True,full_STO_noise_closure_certified=False,
        formal_fNL_or_kmax_conclusion_released=False,n_covariance=205,n_target_per_truth=100,P_bins=12,B_bins=9,
        p_fixed_physical=p,p_uncertainty_propagated=False,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
        P_B_cross_covariance=True,kmax_B=.08,kmax_P=.08,IR=False,original_C500_pipeline_replaced=False,
        original8_parameter_model_preserved=True,max_steps=64000,ensembles=4,walkers=32,CPU_affinity=[13]))


class ProvisionalCase(shared.Case):
    def __init__(self,d):
        parent();self.directory=Path(d);self.meta=read(self.directory/'contract.json')
        if not self.meta['provisional'] or self.meta['full_STO_noise_closure_certified']:
            raise ValueError('this adapter cannot produce a certified formal inference')
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('case likelihood drift')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            for k in z.files:setattr(self,k,np.array(z[k]))
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=list(NAMES_V2):raise ValueError('candidate12 order changed')
        self.model=EFTCandidateModelV2(self.coefficients,self.templates,self.operators,self.intrinsic,
            self.matched_zero,self.weight_moments,self.meta['p_fixed_physical'],F_GROWTH)

    def residual(self,v):return self.model.residual(v,self.target,self.chol)

    def initialize(self):
        path=self.directory/'MAP.json'
        if path.exists():
            m=read(path);self.center=np.array(m['center']);self.transform=np.array(m['transform']);return m
        candidates=[]
        # 原blind fNL/b1九起点；新增noise仅从同一个Poisson参考点启动，不读取真值profile。
        for f0 in (-200.,0.,200.):
            for b0 in (1.25,1.93,2.8):
                start=np.array([f0,b0,-.25,-.3,0.,0.,0.,1.,b0,2*1.686*(b0-self.meta['p_fixed_physical']),1.,0.])
                fit=Minuit(self.objective,*start,name=self.names);fit.errordef=1
                for name,limit,error in zip(self.names,self.bounds,[30.,.1,.2,.2]+[.3]*8):
                    fit.limits[name]=limit;fit.errors[name]=error
                fit.migrad(ncall=12000)
                if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=12000)
                fit.hesse();candidates.append((float(fit.fval),np.array(list(fit.values)),fit))
        candidates=[r for r in candidates if np.isfinite(r[0]) and np.isfinite(r[1]).all()]
        if not candidates:raise ValueError('all blind MAP starts failed')
        candidates.sort(key=lambda x:x[0]);score,center,fit=candidates[0]
        matrix=np.array(fit.covariance) if fit.covariance is not None else None
        if matrix is None or not np.isfinite(matrix).all() or np.any(np.diag(matrix)<=0):
            matrix=np.diag(((self.bounds[:,1]-self.bounds[:,0])/20)**2)
        # 完全沿用原proposal变换；此矩阵不是科学数据C，不会regularize/floor C_single。
        scale=np.sqrt(np.maximum(np.diag(matrix),1e-16));corr=matrix/np.outer(scale,scale)
        eigen,u=np.linalg.eigh((corr+corr.T)/2);corr=(u*np.maximum(eigen,1e-6))@u.T
        transform=np.diag(scale)@np.linalg.cholesky(corr);self.center=center;self.transform=transform
        m=dict(center=center.tolist(),transform=transform.tolist(),chi2_MAP=score,minuit_valid=bool(fit.valid),
            minuit_covariance_accurate=bool(fit.fmin.has_accurate_covar),likelihood_sha256=self.meta['likelihood_sha256'],
            candidates=[dict(chi2=r[0],parameters=r[1].tolist(),valid=bool(r[2].valid)) for r in candidates],
            blind_start_grid=dict(fNL=[-200,0,200],b1=[1.25,1.93,2.8]),
            noise_start='same Poisson reference for every cosmology; original Pshot=0',
            simulation_truth_used_in_initialization=False,proposal_transform_algorithm_unchanged=True)
        shared.save_json(path,m);return m


def sample(index):
    parent();old=(shared.OUT,shared.Case,shared.budget_provenance,shared.save_json)
    def provenance(d,v,extend=False):
        v.update(provisional_adapter_sha256=digest(__file__),provisional_parent_sha256=digest(OUT/'CONTRACT.json'),
            direct_human_authority_sha256=digest(APPROVAL));return old[2](d,v,extend)
    def save(p,v):
        if Path(p).name=='SUMMARY.json':
            meta=read(Path(p).parent/'contract.json');v.update(P_likelihood=True,B_likelihood=True,P_B_cross_covariance=True,
                provisional=True,diagnostic_only=True,full_STO_noise_closure_certified=False,
                formal_fNL_or_kmax_conclusion_released=False,n_covariance=205,n_target=100,covariance_is_of_mean=False,
                covariance_multiplier=1.,volume_Gpc_h3=1.,IR=False,p_fixed_physical=meta['p_fixed_physical'],
                p_uncertainty_propagated=False,original_Pshot_physical_bounds=[-1e6,1e6],
                gof_note='descriptive only: mean100 weighted by C_single205; model/noise not certified')
        return old[3](p,v)
    shared.OUT=OUT;shared.Case=ProvisionalCase;shared.budget_provenance=provenance;shared.save_json=save
    try:shared.run(index,max_steps=64000)
    finally:shared.OUT,shared.Case,shared.budget_provenance,shared.save_json=old


def pipeline():
    if not OUT.exists():freeze()
    else:parent()
    state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],started_UTC=stamp(),completed_cases=[])
    def update(**v):state.update(v,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update(stage='blind_MAP_and_original_four_ensemble_sampling')
    try:
        for index,name in enumerate(CASES):
            update(case=name,case_index=index);sample(index)
            state['completed_cases'].append(name);update()
        update(state='ALL3_PROVISIONAL_PB_CASES_FINISHED',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED_REQUIRES_EXACT_PROCESS_INSPECTION',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.prepare_only:
        freeze();print('ACTUAL_PROVISIONAL_PB_INPUTS_FROZEN_NOT_SAMPLED',flush=True);return
    if args.detach:
        log=ROOT/'diagnostic_provisional_eft12_pb_20261010_v1.log';launch=ROOT/'diagnostic_provisional_eft12_pb_20261010_v1.launch.json'
        if log.exists() or launch.exists():raise ValueError('inspect original launch/process before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
