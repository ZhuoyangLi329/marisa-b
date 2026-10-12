"""模块大纲：冻结当前fid子集的C_single，先校准物理p，再三组post-P0诊断链。

用户仅授权p校准/P0试跑；不创建B噪声PASS，不替换正式Fid500流程。
复用原预测、先验、多起点MAP和四ensemble采样/收敛门。login26 CPU13单核。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import numpy as np
from rsd_b000_contract import PART,digest,save_json,F_GROWTH
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from run_sto_post_covariance_queue import source_row
from post_rsd_p_calibration import PhysicalPCalibrationCase,PARAMETERS,BOUNDS
from run_sto_physical_p_mcmc import CalibrationCase as OriginalCalibrationCase
from run_sto_post_p0only import POnlyCase as OriginalPostPCase
import run_rsd_b000_mcmc as sampler
import rsd_pb_common as spectral
from rsd_bonly_common import TABLE

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'diagnostic_partial_cov_p_mcmc_20261010_v2'
HOLDOUT=ROOT/'calibration_holdout_p0'
CAL_NAME='physical_p_GSR2_RSD_fNLfixed100_partial_cov'
COSMOS=('fiducial','LC_p','LC_m')
NAMES=[CAL_NAME]+[f'{c}_post_P0_mean100_partial_cov' for c in COSMOS]


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def marginal(covariance,n_covariance,indices):
    """只取C子块后构造自己的Hartlap/Cholesky；不缩放成mean covariance。"""
    c=np.asarray(covariance,dtype=float)[np.ix_(indices,indices)]
    alpha=(n_covariance-len(indices)-2)/(n_covariance-1)
    if alpha<=0 or not np.isfinite(c).all():raise ValueError('insufficient samples or invalid covariance')
    return c,np.linalg.cholesky(c/alpha),alpha


class PartialCalibrationModel(PhysicalPCalibrationCase):
    def __init__(self,target,covariance,cholesky,templates,n_covariance):
        self.target=np.asarray(target,dtype=float);self.covariance=np.asarray(covariance,dtype=float)
        self.cholesky=np.asarray(cholesky,dtype=float);self.templates=np.asarray(templates,dtype=float)
        if (self.target.shape!=(12,) or self.covariance.shape!=(12,12)
            or self.cholesky.shape!=(12,12) or self.templates.shape!=(6,12)):
            raise ValueError('original twelve-bin physical-p inputs required')
        alpha=(n_covariance-14)/(n_covariance-1)
        if alpha<=0:raise ValueError('insufficient covariance samples')
        if not all(np.isfinite(a).all() for a in (self.target,self.covariance,self.cholesky,self.templates)):
            raise ValueError('nonfinite partial calibration arrays')
        np.testing.assert_allclose(self.cholesky@self.cholesky.T,self.covariance/alpha,rtol=3e-13,atol=1e-8)
        # predictions/objective等全部继承原实现，只泛化诊断协方差的样本数。


def parent():
    value=read(OUT/'CONTRACT.json')
    if value['status']!='PARTIAL_C_SINGLE_P_DIAGNOSTIC_FROZEN_NOT_FORMAL_C500':raise ValueError('wrong diagnostic contract')
    for path,want in value['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('partial diagnostic frozen input drift: '+path)
    return value


def freeze():
    if OUT.exists():raise ValueError('preserve existing diagnostic run, never refreeze its covariance')
    production=ROOT/'simultaneous_measurements_v3/CONTRACT.json'
    for path,want in read(production)['immutable_sources'].items():
        if digest(path)!=want:raise ValueError('production frozen source drift before diagnostic: '+path)
    ids=[i for i in range(500) if (directory(('fiducial',i))/'DONE.json').exists()]
    if len(ids)<150:raise ValueError('at least150 actual complete fid rows required for this diagnostic')
    snapshot_utc=stamp();audit=read(ROOT/'VALIDATION_SOURCE_AUDIT.json')
    target_sources={(r['cosmology'],r['realization']):r for r in audit['source_rows']}
    needed=list(dict.fromkeys([('fiducial',i) for i in ids]+[(c,i) for c in COSMOS for i in range(100)]))
    rows={};moments={};records=[];fixed=None
    for key in needed:
        d=directory(key);data,meta,done=load_pilot(d)
        source=target_sources[key] if key[1]<100 else source_row(key[1])[1]
        if source is None or done['entry_sha256']!=digest(PART/'code/run_sto_post_rsd_pilot.py'):
            raise ValueError('source or original numerical entry missing')
        if (meta['arguments']['cosmology'],meta['arguments']['realization'])!=key:raise ValueError('catalog identity mismatch')
        for lhs,rhs in [('source_STO_sha256','source_sha256'),('source_diagnostics_sha256','diagnostics_sha256')]:
            if meta[lhs]!=source[rhs]:raise ValueError('admitted source identity changed')
        if meta['raw_fof']['source_raw_sha256']!=source['raw_fof']['source_raw_sha256']:raise ValueError('raw input changed')
        if fixed is None:fixed=meta['fixed_reconstruction']
        if meta['fixed_reconstruction']!=fixed:raise ValueError('mixed reconstruction')
        rows[key]=np.r_[data['P_weighted_input'],data['P_post']]
        moments[key]=source['N2_data_self']
        if rows[key].shape!=(24,) or not np.isfinite(rows[key]).all():raise ValueError('original P geometry required')
        records.append(dict(key=key,contract_sha256=done['contract_sha256'],measurement_sha256=done['measurement_sha256'],
            post_halos_sha256=done['post_halos_sha256']))
    cov_rows=np.array([rows['fiducial',i] for i in ids]);cov=np.cov(cov_rows,rowvar=False,ddof=1)
    targets=np.array([np.mean([rows[c,i] for i in range(100)],axis=0) for c in COSMOS])
    units=np.array([np.mean([moments[c,i] for i in range(100)]) for c in COSMOS])
    proof=read(HOLDOUT/'HOLDOUT_COMPLETION_AUDIT.json');hold_path=HOLDOUT/'CALIBRATION_P0_HOLDOUT.npz'
    if (proof['status']!='INDEPENDENT_GSR2_HOLDOUT_P0_INPUTS_PASS_NO_P_FIT'
        or digest(hold_path)!=proof['aggregate_sha256']):raise ValueError('actual holdout150 input drift')
    with np.load(hold_path,allow_pickle=False) as z:
        hold=np.array(z['P0']);holdids=z['realizations'].tolist();edges=np.array(z['edges'])
    if hold.shape!=(150,12) or holdids!=list(range(100,250)):raise ValueError('original holdout150 required')
    wa=read(HOLDOUT/'POST_P0_WINDOW_AUDIT.json');window=HOLDOUT/'P0_WINDOW.npz'
    if wa['status']!='PASS' or digest(window)!=wa['window_sha256']:raise ValueError('actual production P window changed')
    previous=spectral.OUT
    try:spectral.OUT=HOLDOUT;templates,nmodes,k=spectral.p_lattice(edges)
    finally:spectral.OUT=previous
    with np.load(window,allow_pickle=False) as z:np.testing.assert_array_equal(nmodes,z['nmodes'])
    c,chol,alpha=marginal(cov,len(ids),tuple(range(12)))
    PartialCalibrationModel(hold.mean(axis=0),c,chol,templates,len(ids))
    OUT.mkdir();case=OUT/'cases'/CAL_NAME;case.mkdir(parents=True)
    matrix_path=OUT/'FROZEN_PARTIAL_P_ENSEMBLE.npz'
    np.savez_compressed(matrix_path,covariance_single_box=cov,covariance_rows=cov_rows,
        covariance_ids=ids,target_vectors=targets,N2_units=units,templates=templates,nmodes=nmodes,k=k,edges=edges)
    np.savez_compressed(case/'likelihood.npz',target=hold.mean(axis=0),covariance=c,chol=chol,templates=templates,bounds=BOUNDS)
    meta=dict(name=CAL_NAME,cosmology='LC_p',truth_fNL=100,case_type='physical_p_calibration',kmax_B=None,kmax_P=.08,
        parameters=list(PARAMETERS),bounds=BOUNDS.tolist(),n_covariance=len(ids),n_target=150,
        volume_Gpc_h3=1.,covariance_multiplier=1.,covariance_is_of_mean=False,Hartlap_factor=alpha,
        P_likelihood=True,B_likelihood=False,P_derived_b1_prior=False,IR=False,diagnostic_only=True,
        likelihood_sha256=digest(case/'likelihood.npz'))
    save_json(case/'contract.json',meta);save_json(OUT/'CASES.json',NAMES)
    sources=[Path(__file__),PART/'code/post_rsd_p_calibration.py',PART/'code/post_rsd_p0_model.py',
        PART/'code/run_sto_physical_p_mcmc.py',PART/'code/run_sto_post_p0only.py',PART/'code/run_sto_post_noir_mcmc.py',
        PART/'code/post_rsd_noir_likelihood.py',PART/'code/run_rsd_p0only.py',PART/'code/run_rsd_b000_mcmc.py',
        PART/'code/rsd_pb_common.py',PART/'code/rsd_b000_contract.py',TABLE,
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',
        production,ROOT/'ENSEMBLE_SELECTION.json',ROOT/'VALIDATION_SOURCE_AUDIT.json',matrix_path,
        HOLDOUT/'HOLDOUT_COMPLETION_AUDIT.json',hold_path,window,HOLDOUT/'POST_P0_WINDOW_AUDIT.json',
        OUT/'CASES.json',case/'likelihood.npz',case/'contract.json']
    sources.extend(Path(p) for p in read(production)['immutable_sources'])
    sources=list(dict.fromkeys(sources))
    save_json(OUT/'CONTRACT.json',dict(status='PARTIAL_C_SINGLE_P_DIAGNOSTIC_FROZEN_NOT_FORMAL_C500',
        immutable_inputs={str(p):digest(p) for p in sources},n_covariance=len(ids),covariance_ids=ids,
        covariance_snapshot_UTC=snapshot_utc,covariance_ddof=1,covariance_is_of_mean=False,
        covariance_multiplier=1.,volume_Gpc_h3=1.,fixed_reconstruction=fixed,
        target_ids=list(range(100)),calibration_ids=list(range(100,250)),rows=records,
        strict_independent_mock_recovery=False,fiducial_target_covariance_overlap=sorted(set(ids)&set(range(100))),
        calibration_covariance_possible_shared_phase_ids=sorted(set(ids)&set(range(100,250))),
        B_noise_model_released=False,B_PB_trials_authorized=False,original_formal_C500_pipeline_replaced=False,
        sampler_budget=64000,ensembles=4,walkers=32,CPU_affinity=[13]))
    print('FROZEN_PARTIAL_C_SINGLE',len(ids),snapshot_utc,flush=True)


def load_case(d):
    d=Path(d);meta=read(d/'contract.json')
    if digest(d/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('diagnostic case drift')
    with np.load(d/'likelihood.npz',allow_pickle=False) as z:arrays={k:np.array(z[k]) for k in z.files}
    alpha=(meta['n_covariance']-14)/(meta['n_covariance']-1)
    if meta['Hartlap_factor']!=alpha or meta['covariance_is_of_mean'] or meta['covariance_multiplier']!=1:
        raise ValueError('partial C_single scale drift')
    np.testing.assert_allclose(arrays['chol']@arrays['chol'].T,arrays['covariance']/alpha,rtol=3e-13,atol=1e-8)
    return d,meta,arrays


class CalibrationCase(OriginalCalibrationCase):
    def __init__(self,d):
        self.directory,self.meta,arrays=load_case(d)
        for k,v in arrays.items():setattr(self,k,v)
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        np.testing.assert_array_equal(self.bounds,BOUNDS)
        if self.names!=list(PARAMETERS):raise ValueError('original physical-p parameters required')
        self.model=PartialCalibrationModel(self.target,self.covariance,self.chol,self.templates,self.meta['n_covariance'])


class POnlyCase(OriginalPostPCase):
    def __init__(self,d):
        self.directory,self.meta,arrays=load_case(d)
        for k,v in arrays.items():setattr(self,k,v)
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=['fNL','b1','Pshot_residual_nbar']:raise ValueError('original P0 parameters required')
        self.white_templates=np.linalg.solve(self.chol,self.templates.T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise);self.twhite=np.linalg.solve(self.chol,self.target)


def freeze_P_cases():
    contract=parent();summary=read(OUT/'cases'/CAL_NAME/'SUMMARY.json')
    if not summary['converged'] or summary['parameters'][0]!='physical_p':raise ValueError('diagnostic physical-p calibration must converge')
    p=float(summary['quantiles_2p5_16_50_84_97p5'][2][0])
    with np.load(OUT/'FROZEN_PARTIAL_P_ENSEMBLE.npz',allow_pickle=False) as z:
        cov=z['covariance_single_box'];targets=z['target_vectors'];units=z['N2_units'];templates=z['templates']
    c,chol,alpha=marginal(cov,contract['n_covariance'],tuple(range(12,24)))
    for i,cosmo in enumerate(COSMOS):
        d=OUT/'cases'/NAMES[i+1]
        if d.exists():raise ValueError('never refreeze an existing P0 trial')
        d.mkdir();bounds=np.array([[-500,500],[.5,4],[-1e6/units[i],1e6/units[i]]])
        np.savez_compressed(d/'likelihood.npz',target=targets[i,12:],covariance=c,chol=chol,
            templates=templates,noise=np.full(12,units[i]),bounds=bounds)
        save_json(d/'contract.json',dict(name=NAMES[i+1],cosmology=cosmo,truth_fNL=(0,100,-100)[i],
            case_type='P0_only',parameters=['fNL','b1','Pshot_residual_nbar'],bounds=bounds.tolist(),
            n_covariance=contract['n_covariance'],n_target=100,volume_Gpc_h3=1.,covariance_multiplier=1.,
            covariance_is_of_mean=False,Hartlap_factor=alpha,kmax_P=.08,kmax_B=None,P_bins=12,B_bins=0,
            P_likelihood=True,B_likelihood=False,P_derived_b1_prior=False,IR=False,p_fixed_physical=p,
            physical_p_calibration_summary_sha256=digest(OUT/'cases'/CAL_NAME/'SUMMARY.json'),
            Pshot_unit_Mpc_h3=float(units[i]),diagnostic_only=True,likelihood_sha256=digest(d/'likelihood.npz')))


def sample(index):
    parent();previous=(sampler.OUT,sampler.Case,sampler.budget_provenance,sampler.save_json,sampler.clean)
    def case(d):return CalibrationCase(d) if index==0 else POnlyCase(d)
    def provenance(d,v,extend=False):
        v.update(partial_adapter_sha256=digest(Path(__file__)),partial_parent_sha256=digest(OUT/'CONTRACT.json'))
        return previous[2](d,v,extend)
    def save(p,v):
        if Path(p).name=='SUMMARY.json':
            meta=read(Path(p).parent/'contract.json')
            v.update(diagnostic_only=True,n_covariance=meta['n_covariance'],P_likelihood=True,B_likelihood=False,
                covariance_is_of_mean=False,covariance_multiplier=1.,IR=False,gof_pvalue=None,
                gof_note='partial C_single forecast-weighted mean; not calibrated model adequacy')
            if index==0:v.update(calibration_fNL_fixed=100.,fNL_fitted=False)
            else:v.update(p_fixed_physical=meta['p_fixed_physical'],physical_p_is_not_p_rec=True)
        return previous[3](p,v)
    def clean(v):
        if index==0 and isinstance(v,dict):
            v=dict(v);v.pop('fNL_pull',None)
            for old,new in [('fNL_median','physical_p_median'),('fNL_sigma','physical_p_sigma'),('fNL_ESS','physical_p_ESS')]:
                if old in v:v[new]=v.pop(old)
        return previous[4](v)
    sampler.OUT,sampler.Case,sampler.budget_provenance,sampler.save_json,sampler.clean=OUT,case,provenance,save,clean
    try:return sampler.run(index,64000,False)
    finally:sampler.OUT,sampler.Case,sampler.budget_provenance,sampler.save_json,sampler.clean=previous


def pipeline():
    state=dict(state='FREEZING_PARTIAL_C_SINGLE',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],
        started_UTC=stamp(),child_pid=None,completed_cases=[])
    def update(**values):
        state.update(values,updated_UTC=stamp());save_json(ROOT/'diagnostic_partial_p_pipeline_v2_STATUS.json',state)
    update()
    try:
        freeze()
        for index in range(4):
            if index==1:freeze_P_cases()
            update(state='PHYSICAL_P_CALIBRATION_MCMC' if index==0 else 'POST_P0_DIAGNOSTIC_MCMC',case=NAMES[index])
            sample(index)
            s=read(OUT/'cases'/NAMES[index]/'SUMMARY.json');state['completed_cases'].append(dict(name=NAMES[index],converged=s['converged']))
            update()
        update(state='PARTIAL_P_CALIBRATION_AND_THREE_P0_TRIALS_FINISHED',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single-core required')
    if args.detach:
        log=ROOT/'diagnostic_partial_p_pipeline_v2.log';launch=ROOT/'diagnostic_partial_p_pipeline_v2.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect existing diagnostic; no duplicate launch')
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=p.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True);return
    pipeline()


if __name__=='__main__':main()
