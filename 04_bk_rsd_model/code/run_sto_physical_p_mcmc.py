"""模块大纲：完整C500后冻结物理p校准；原三参数先验；唯一MCMC采样核心。

校准fNL固定100，使用已验收LC_p100..249均值与weighted-input C_single。
四独立ensemble/提案/RNG/HDF/诊断门复用；不将p_rec或旧GSR1 p作为结果。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from iminuit import Minuit
from rsd_b000_contract import PART,digest,save_json
from rsd_bonly_common import TABLE
from post_rsd_p_calibration import make_calibration_arrays,PhysicalPCalibrationCase,PARAMETERS,BOUNDS
import rsd_pb_common as spectral
import run_rsd_b000_mcmc as shared

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'physical_p_calibration_mcmc'
HOLDOUT=ROOT/'calibration_holdout_p0'
NAME='physical_p_GSR2_RSD_fNLfixed100'


def freeze():
    if (OUT/'contract.json').exists():parent();print('REUSE frozen physical-p inputs',flush=True);return
    report_path=ROOT/'ensemble_measurements/AUDIT.json'
    if not report_path.exists():raise ValueError('complete post Fid500 C_single not available; no partial/old C fallback')
    report=json.loads(report_path.read_text())
    if report['status']!='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE':
        raise ValueError('ensemble provenance not verified')
    matrix_path=Path(report['ensemble_files']['matched']['path'])
    if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:raise ValueError('ensemble matrix drift')
    with np.load(matrix_path,allow_pickle=False) as z:
        targets=dict(zip(('fiducial','LC_p','LC_m'),np.array(z['target_vectors'])))
        ensemble=dict(statistics=report['statistics'],blocks=report['block_slices'],
            covariance_single_box=np.array(z['covariance_single_box']),targets=targets)
    proof=json.loads((HOLDOUT/'HOLDOUT_COMPLETION_AUDIT.json').read_text())
    data_path=HOLDOUT/'CALIBRATION_P0_HOLDOUT.npz'
    if proof['status']!='INDEPENDENT_GSR2_HOLDOUT_P0_INPUTS_PASS_NO_P_FIT' or digest(data_path)!=proof['aggregate_sha256']:
        raise ValueError('real heldout150 measurements drift')
    with np.load(data_path,allow_pickle=False) as z:
        data=np.array(z['P0']);ids=tuple(np.asarray(z['realizations']).tolist());edges=np.array(z['edges'])
    arrays=make_calibration_arrays(ensemble,data,holdout_ids=ids)
    window_audit=json.loads((HOLDOUT/'POST_P0_WINDOW_AUDIT.json').read_text())
    if (window_audit['status']!='PASS' or not window_audit['holdout_measurement_nmodes_match']
        or digest(HOLDOUT/'P0_WINDOW.npz')!=window_audit['window_sha256']):
        raise ValueError('actual operator/window mismatch')
    previous=spectral.OUT
    try:
        spectral.OUT=HOLDOUT;templates,nmodes,k=spectral.p_lattice(edges)
    finally:spectral.OUT=previous
    PhysicalPCalibrationCase(arrays['target'],arrays['covariance'],arrays['chol'],templates)
    sources=(Path(__file__),ROOT/'ENSEMBLE_SELECTION.json',report_path,matrix_path,data_path,
        HOLDOUT/'HOLDOUT_COMPLETION_AUDIT.json',HOLDOUT/'P0_WINDOW.npz',HOLDOUT/'POST_P0_WINDOW_AUDIT.json',
        PART/'code/post_rsd_p_calibration.py',PART/'code/post_rsd_p0_model.py',
        PART/'code/post_rsd_ensemble.py',PART/'code/rsd_pb_common.py',Path(shared.__file__),
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',TABLE)
    OUT.mkdir(exist_ok=False);case=OUT/'cases'/NAME;case.mkdir(parents=True)
    np.savez_compressed(case/'likelihood.npz',target=arrays['target'],covariance=arrays['covariance'],
        chol=arrays['chol'],templates=templates,bounds=BOUNDS,k=k,edges=edges,nmodes=nmodes)
    meta=dict(name=NAME,cosmology='LC_p',truth_fNL=100,case_type='physical_p_calibration',
        parameters=list(PARAMETERS),bounds=BOUNDS.tolist(),P_bins=12,B_bins=0,kmax_P=.08,kmax_B=None,
        volume_Gpc_h3=1.,covariance_multiplier=1.,covariance_is_of_mean=False,
        calibration_target_count=150,n_covariance=500,Hartlap_factor=486/499.,
        calibration_statistics=arrays['metadata'],P_likelihood=True,P_derived_b1_prior=False,
        fNL_fitted=False,physical_p_true_value_known=False,p_rec_is_not_physical_p=True,
        likelihood_sha256=digest(case/'likelihood.npz'))
    save_json(case/'contract.json',meta);save_json(OUT/'CASES.json',[NAME])
    save_json(OUT/'contract.json',dict(status='COMPLETE_C500_AND_HELDOUT150_PHYSICAL_P_CALIBRATION_INPUTS_FROZEN',
        immutable_inputs={str(p):digest(p) for p in sources},case_contract_sha256=digest(case/'contract.json'),
        parameters=list(PARAMETERS),calibration_fNL_fixed=100.,main_MCMC_parameter_added=False,
        covariance_is_of_mean=False,volume_Gpc_h3=1.))
    print('PHYSICAL_P_CALIBRATION_FROZEN_NOT_YET_SAMPLED',flush=True)


def parent():
    meta=json.loads((OUT/'contract.json').read_text())
    for path,want in meta['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('calibration source/input drift: '+path)
    if digest(OUT/'cases'/NAME/'contract.json')!=meta['case_contract_sha256']:raise ValueError('calibration case drift')
    return meta


class CalibrationCase(shared.Case):
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'contract.json').read_text())
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('calibration likelihood drift')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            for name in z.files:setattr(self,name,np.array(z[name]))
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=list(PARAMETERS) or self.meta['truth_fNL']!=100:raise ValueError('not physical-p calibration')
        np.testing.assert_array_equal(self.bounds,BOUNDS)
        self.model=PhysicalPCalibrationCase(self.target,self.covariance,self.chol,self.templates)

    def residual(self,physical):
        predicted=np.atleast_2d(self.model.predictions(np.atleast_2d(physical)))
        return np.linalg.solve(self.chol,(predicted-self.target).T).T

    def initialize(self):
        path=self.directory/'MAP.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['likelihood_sha256']!=self.meta['likelihood_sha256']:raise ValueError('MAP input drift')
            self.center=np.array(result['center']);self.transform=np.array(result['transform']);return result
        candidates=[]
        for p in (.5,1.5,2.5):
            for b in (1.25,1.93,2.8):
                fit=Minuit(self.objective,p,b,0.,name=self.names);fit.errordef=1;fit.tol=1e-5
                for name,bound,error in zip(self.names,self.bounds,(.1,.1,1000.)):
                    fit.limits[name]=bound;fit.errors[name]=error
                fit.migrad(ncall=12000)
                if not fit.valid:fit.simplex(ncall=6000);fit.migrad(ncall=12000)
                fit.hesse();candidates.append(fit)
        valid=[f for f in candidates if f.valid and f.covariance is not None and np.isfinite(f.fval)]
        if not valid:raise ValueError('all blind physical-p MAP starts failed')
        best=min(valid,key=lambda f:f.fval);self.center=np.array(list(best.values))
        self.transform=np.linalg.cholesky(np.array(best.covariance))
        result=dict(center=self.center.tolist(),transform=self.transform.tolist(),chi2_MAP=float(best.fval),
            minuit_valid=bool(best.valid),minuit_covariance_accurate=bool(best.fmin.has_accurate_covar),
            candidates=[dict(chi2=float(f.fval),parameters=list(f.values),valid=bool(f.valid)) for f in candidates],
            likelihood_sha256=self.meta['likelihood_sha256'],blind_start_grid=dict(physical_p=[.5,1.5,2.5],b1=[1.25,1.93,2.8]))
        save_json(path,result);return result


def sample(max_steps=64000,extend_budget=False):
    if len(os.sched_getaffinity(0))!=1:raise ValueError('one logical CPU required')
    parent();previous=(shared.OUT,shared.Case,shared.budget_provenance,shared.save_json,shared.clean)
    def provenance(directory,value,extend=False):
        value.update(physical_p_adapter_sha256=digest(Path(__file__)),parent_contract_sha256=digest(OUT/'contract.json'))
        return previous[2](directory,value,extend)
    def clean(value):
        # 数值诊断/阈值不改；校准第一列是p，不能把它报告成fNL后验或truth pull。
        if isinstance(value,dict):
            value=dict(value);value.pop('fNL_pull',None)
            for old,new in (('fNL_median','physical_p_median'),('fNL_sigma','physical_p_sigma'),('fNL_ESS','physical_p_ESS')):
                if old in value:value[new]=value.pop(old)
        return previous[4](value)
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            value.update(P_likelihood=True,B_likelihood=False,calibration_fNL_fixed=100.,fNL_fitted=False,
                physical_p_true_value_known=False,covariance_is_of_mean=False,covariance_multiplier=1.,
                calibration_target_count=150,kmax_P=.08,kmax_B=None,gof_pvalue=None,
                gof_note='heldout150 mean with unaveraged Fid500 single-box forecast covariance')
        return previous[3](path,value)
    shared.OUT,shared.Case,shared.budget_provenance,shared.save_json,shared.clean=OUT,CalibrationCase,provenance,save,clean
    try:return shared.run(0,max_steps,extend_budget)
    finally:shared.OUT,shared.Case,shared.budget_provenance,shared.save_json,shared.clean=previous


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('freeze','sample'))
    ap.add_argument('--max-steps',type=int,default=64000);ap.add_argument('--extend-budget',action='store_true')
    args=ap.parse_args()
    if args.command=='freeze':freeze()
    else:sample(args.max_steps,args.extend_budget)
