"""模块大纲：全样本物理p校准后自动重拟合pre/post各三组P0-only。

数据仍mean100；协方差C_single500/V1；显式注入各tracer的新p中位数。
复用原三参数MAP和唯一MCMC，不改数据、先验或强迫后验恢复真值。
按post/pre顺序自动等待真实校准退出，保存独立来源和新旧p处理证据。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import json
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import run_all_available_rsd_pcal_v1 as calibration
from audit_all_available_rsd_pcal_v1 import verify,direct_p
from post_rsd_owner_quiescence import owner_quiescent
from post_rsd_noir_likelihood import predict_p_rows
from run_rsd_p0only import POnlyCase as OriginalPOnlyCase
import run_rsd_b000_mcmc as shared
from rsd_b000_contract import PART,digest,save_json

OUT=calibration.OUT
COSMOS=('fiducial','LC_p','LC_m')
NAMES=['fNL','b1','Pshot_residual_nbar']

def read(path):return json.loads(Path(path).read_text())
def stamp():return calibration.stamp()
def root(stage):return OUT/(stage+'_P0_only_all_available_p')

def guard(stage):
    meta=read(root(stage)/'CONTRACT.json')
    for path,want in meta['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('current P0 run source/input changed: '+path)
    return meta

class AllPCase(OriginalPOnlyCase):
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=read(self.directory/'contract.json');guard(self.meta['stage'])
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('P0 likelihood changed')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            for name in z.files:setattr(self,name,z[name].copy())
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=NAMES or self.target.shape!=(12,) or self.templates.shape!=(6,12):raise ValueError('actual three-parameter P0 dimensions required')
        np.testing.assert_allclose(self.chol@self.chol.T,self.covariance/(486/499),rtol=3e-13,atol=1e-8)
        self.white_templates=np.linalg.solve(self.chol,self.templates.T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise);self.twhite=np.linalg.solve(self.chol,self.target)

    def predictions(self,physical):
        rows=np.atleast_2d(physical)
        return predict_p_rows(rows[:,:2],self.templates,p_halo=self.meta['p_fixed_physical'])+rows[:,2,None]*self.noise

    def residual(self,physical):
        rows=np.atleast_2d(physical)
        return predict_p_rows(rows[:,:2],self.white_templates,p_halo=self.meta['p_fixed_physical'])+rows[:,2,None]*self.nwhite-self.twhite

def freeze(stage):
    p,proof=verify(stage)
    if not proof['converged']:raise ValueError('physical-p trial not converged; preserve diagnostics')
    dest=root(stage)
    if dest.exists():raise ValueError('actual P0 output already exists; no refreeze or duplicate')
    cal=calibration.stage_root(stage);cased=cal/'cases'/calibration.case_name(stage)
    sources=[Path(__file__),PART/'code/audit_all_available_rsd_pcal_v1.py',
        PART/'code/run_all_available_rsd_pcal_v1.py',PART/'code/run_rsd_p0only.py',
        PART/'code/run_rsd_b000_mcmc.py',PART/'code/post_rsd_p0_model.py',
        PART/'code/post_rsd_noir_likelihood.py',PART/'code/rsd_pb_common.py',
        cal/'contract.json',cal/'COMPLETION_AUDIT.json',cased/'SUMMARY.json',
        cased/'SUMMARY_METADATA_CORRECTED.json',cased/'likelihood.npz',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    with np.load(cased/'likelihood.npz',allow_pickle=False) as z:templates=z['templates'].copy()
    if stage=='post':
        report_path=calibration.STO/'ensemble_measurements/AUDIT.json';report=read(report_path)
        matrix_path=Path(report['ensemble_files']['matched']['path'])
        if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:raise ValueError('full C500 matrix changed')
        with np.load(matrix_path,allow_pickle=False) as z:
            keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']]
            matrix=z['matrix'].copy();covariance=z['covariance_single_box'][12:24,12:24].copy()
            moments=z['data_N2_N3_nhalo'].copy();cov_ids=z['covariance_indices'].copy()
        pos={key:i for i,key in enumerate(keys)}
        cov_rows=np.array([matrix[pos['fiducial',i],12:24] for i in range(500)])
        np.testing.assert_allclose(np.cov(cov_rows,rowvar=False,ddof=1),covariance,rtol=3e-13,atol=1e-10)
        sources.extend([report_path,matrix_path]);cov_ids=list(range(500))
    else:cov_ids=list(range(1000,1500))
    dest.mkdir();names=[]
    for truth,fnl in zip(COSMOS,(0,100,-100)):
        if stage=='post':
            ids=[pos[truth,i] for i in range(100)]
            target=matrix[ids,12:24].mean(axis=0);unit=float(moments[ids,0].mean())
            noise=np.full(12,unit);bounds=np.array([[-500,500],[.5,4],[-1e6/unit,1e6/unit]])
        else:
            old=PART/'outputs/pre_rsd_p0only_mean100_v1/cases'/(truth+'_mean100_kP0.08')
            oldmeta=read(old/'contract.json')
            if digest(old/'likelihood.npz')!=oldmeta['likelihood_sha256']:raise ValueError('original pre P0 source changed')
            with np.load(old/'likelihood.npz',allow_pickle=False) as z:
                target=z['target'].copy();covariance=z['covariance'].copy();noise=z['noise'].copy();bounds=z['bounds'].copy()
                np.testing.assert_array_equal(templates,z['templates'])
            unit=float(noise[0]);sources.extend([old/'contract.json',old/'likelihood.npz'])
        name=truth+'_'+stage+'_P0_all_available_p';names.append(name);d=dest/'cases'/name;d.mkdir(parents=True)
        np.savez_compressed(d/'likelihood.npz',target=target,covariance=covariance,
            chol=np.linalg.cholesky(covariance/(486/499)),templates=templates,noise=noise,bounds=bounds)
        meta=dict(name=name,stage=stage,cosmology=truth,truth_fNL=fnl,case_type='P0_only',
            parameters=NAMES,bounds=bounds.tolist(),P_bins=12,B_bins=0,data_bins=12,
            kmax_P=.08,kmax_B=None,P_likelihood=True,B_likelihood=False,P_derived_b1_prior=False,
            P_B_cross_covariance=False,n_target=100,n_covariance=500,target_ids=list(range(100)),
            covariance_ids=cov_ids,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
            Hartlap_factor=486/499,p_fixed_physical=p,p_uncertainty_propagated=False,
            physical_p_calibration_target_count=proof['calibration_target_count'],
            physical_p_calibration_summary_sha256=proof['summary_sha256'],physical_p_is_not_p_rec=True,
            physical_p_calibration_metadata_sha256=proof['corrected_summary_metadata_sha256'],
            Pshot_unit_Mpc_h3=unit,Pshot_physical_bounds=[-1e6,1e6],IR=False,
            fiducial_target_covariance_overlap_ids=list(range(100)) if stage=='post' else [],
            calibration_main_target_overlap_ids=list(range(100)),independent_mock_recovery_claimed=False,
            likelihood_sha256=digest(d/'likelihood.npz'),diagnostic_only=True)
        save_json(d/'contract.json',meta);sources.extend([d/'contract.json',d/'likelihood.npz'])
    save_json(dest/'CASES.json',names)
    save_json(dest/'CONTRACT.json',dict(created_UTC=stamp(),stage=stage,p_fixed_physical=p,
        immutable_inputs={str(q):digest(q) for q in dict.fromkeys(sources)},
        data_mean_count=100,n_covariance=500,C_single=True,volume_Gpc_h3=1.,IR=False))
    for name in names:
        case=AllPCase(dest/'cases'/name)
        rng=np.random.default_rng(320);rows=rng.uniform(case.bounds[:,0],case.bounds[:,1],(128,3))
        # 以每case自身噪声单位和模板独立回放，确认显式新p实际进入预测。
        direct=np.array([direct_p(f,b,p,case.templates)+s*case.noise for f,b,s in rows])
        np.testing.assert_allclose(case.predictions(rows),direct,rtol=2e-13,atol=1e-7)
        expected=np.linalg.solve(case.chol,(direct-case.target).T).T
        np.testing.assert_allclose(case.residual(rows),expected,rtol=2e-12,atol=1e-9)
    save_json(dest/'P_INJECTION_CHECK.json',dict(verified=True,physical_p=p,independent_probes_per_case=128,
        calibration_summary_sha256=proof['summary_sha256'],physical_p_is_not_p_rec=True))

def sample(stage,index):
    guard(stage);old=(shared.OUT,shared.Case,shared.save_json,shared.budget_provenance)
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            meta=read(Path(path).parent/'contract.json')
            value.update(P_likelihood=True,B_likelihood=False,kmax_P=.08,kmax_B=None,stage=stage,
                p_fixed_physical=meta['p_fixed_physical'],n_covariance=500,n_target=100,
                physical_p_calibration_target_count=meta['physical_p_calibration_target_count'],
                covariance_is_of_mean=False,covariance_multiplier=1.,p_uncertainty_propagated=False,
                gof_pvalue=None,gof_note='mean100 with C_single500 forecast; no calibrated GOF')
        return old[2](path,value)
    def provenance(directory,value,extend=False):
        value.update(all_available_p_P0_adapter_sha256=digest(__file__),parent_sha256=digest(root(stage)/'CONTRACT.json'))
        return old[3](directory,value,extend)
    shared.OUT,shared.Case,shared.save_json,shared.budget_provenance=root(stage),AllPCase,save,provenance
    try:shared.run(index,64000,False)
    finally:shared.OUT,shared.Case,shared.save_json,shared.budget_provenance=old

def pipeline():
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=stamp(),state='WAITING_FOR_ALL_AVAILABLE_P_CALIBRATION',completed_cases=[])
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'P0_RECALIBRATED_STATUS.json',state)
    update()
    try:
        for stage in ('post','pre'):
            update(state='WAITING_FOR_ACTUAL_PHYSICAL_P_CALIBRATION_EXIT',current_stage=stage,current_case=None)
            while True:
                path=OUT/(stage+'_PCAL_STATUS.json')
                if path.exists():
                    cal=read(path)
                    if cal['state']=='FAILED_OR_INTERRUPTED_PRESERVED':raise ValueError('actual physical-p calibration failed')
                    if owner_quiescent(cal):
                        if cal['state']!='ALL_AVAILABLE_PHYSICAL_P_TRIAL_FINISHED_REVIEW_CHAIN':raise ValueError('pcal owner exited without completion')
                        break
                elif stage=='pre':
                    continuation=read(OUT/'PRE_AUTOMATIC_CONTINUATION_STATUS.json')
                    if continuation['state']=='FAILED_OR_INTERRUPTED_PRESERVED' or owner_quiescent(continuation):
                        raise ValueError('pre automatic continuation ended without calibration')
                time.sleep(10)
            freeze(stage)
            for index,name in enumerate(read(root(stage)/'CASES.json')):
                update(state='SAMPLING_RECALIBRATED_P0_ONLY',current_case=name);sample(stage,index)
                state['completed_cases'].append(name);update()
        update(state='SIX_RECALIBRATED_P0_TRIALS_FINISHED_REVIEW_CHAINS',current_case=None,finished_UTC=stamp())
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={7}:raise ValueError('one login26 CPU7 required')
    if args.detach:
        launch=OUT/'P0_RECALIBRATED_LAUNCH.json';log=OUT/'p0_recalibrated.log'
        if launch.exists() or log.exists():raise ValueError('inspect actual existing P0 continuation; do not duplicate')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[7],created_UTC=stamp()))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()
