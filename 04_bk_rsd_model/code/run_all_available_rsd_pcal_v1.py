"""模块大纲：全部可用+100样本重校准pre/STO的物理p，使用完整C500。

pre用0..499；STO用同配方0..249，不再保留100..249为独立holdout。
复用既有三参数物理p模型、先验及唯一MCMC核心；不增加主fNL拟合参数。
原校准和原链保留，新参数供随后P0-only和固定4+3的P+B重拟合。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import numpy as np
from rsd_b000_contract import PART,digest,save_json
import run_sto_physical_p_mcmc as original
from post_rsd_p_calibration import BOUNDS,PARAMETERS,PhysicalPCalibrationCase
from measure_pre_rsd_pcal_all500_v1 import OUT,PRE,row_path

STO=PART/'outputs/sto_marisa_post_rsd_mean100_v1'


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def stage_root(stage):return OUT/(stage+'_physical_p_all_available')
def case_name(stage):return stage+'_physical_p_RSD_fNLfixed100_all_available'


def freeze(stage):
    root=stage_root(stage);name=case_name(stage)
    if root.exists():raise ValueError('existing calibration run must be inspected, never overwritten')
    sources=[Path(__file__),PART/'code/run_sto_physical_p_mcmc.py',PART/'code/post_rsd_p_calibration.py',
        PART/'code/post_rsd_p0_model.py',PART/'code/run_rsd_b000_mcmc.py',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    if stage=='post':
        report_path=STO/'ensemble_measurements/AUDIT.json';report=read(report_path)
        matrix_path=Path(report['ensemble_files']['matched']['path'])
        if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:
            raise ValueError('actual complete700/C500 matrix changed')
        if report['statistics']['n_covariance']!=500:raise ValueError('complete C500 required')
        with np.load(matrix_path,allow_pickle=False) as z:
            keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']]
            matrix=z['matrix'].copy();covfull=z['covariance_single_box'].copy()
        position={key:i for i,key in enumerate(keys)}
        first=np.array([matrix[position['LC_p',i],:12] for i in range(100)])
        hold_path=STO/'calibration_holdout_p0/CALIBRATION_P0_HOLDOUT.npz'
        hold_audit=STO/'calibration_holdout_p0/HOLDOUT_COMPLETION_AUDIT.json'
        if digest(hold_path)!=read(hold_audit)['aggregate_sha256']:
            raise ValueError('actual150 weighted-input measurements changed')
        with np.load(hold_path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['realizations'],np.arange(100,250));tail=z['P0'].copy()
        vectors=np.vstack((first,tail));ids=list(range(250))
        covariance=covfull[:12,:12]
        cov_ids=report['statistics']['covariance_ids']
        psource=STO/'physical_p_calibration_mcmc/cases/physical_p_GSR2_RSD_fNLfixed100/likelihood.npz'
        with np.load(psource,allow_pickle=False) as z:templates=z['templates'].copy()
        sources.extend([report_path,matrix_path,hold_path,hold_audit,psource])
        tracer='same GSR2 carried-weight RSD input before Marisa shift'
    else:
        settings=read(OUT/'PRE_MEASUREMENT_SETTINGS.json')
        if read(OUT/'PRE_MEASUREMENT_STATUS.json')['state']!='ALL500_PRE_POSITIVE_P0_ROWS_READY':
            raise ValueError('all500 actual pre P0 rows must finish before calibration')
        vectors=[]
        for real in range(500):
            if real<100:
                p=Path(settings['reused_original_rows'][real]['path'])
                if digest(p)!=settings['reused_original_rows'][real]['sha256']:raise ValueError('old pre row changed')
            else:
                p=row_path(real)
                if digest(p)!=read(p.with_suffix('.json'))['sha256']:raise ValueError('new pre row changed')
            with np.load(p,allow_pickle=False) as z:vectors.append(z['P0'].copy())
            sources.append(p)
        vectors=np.array(vectors);ids=list(range(500));cov_ids=list(range(1000,1500))
        matrix_path=PRE/'P0_ENSEMBLE_MATRIX.npz'
        with np.load(matrix_path,allow_pickle=False) as z:covariance=z['covariance_single_box'][:12,:12].copy()
        psource=PRE/'P0_LATTICE.npz'
        with np.load(psource,allow_pickle=False) as z:templates=z['templates'].copy()
        sources.extend([matrix_path,psource,OUT/'PRE_MEASUREMENT_SETTINGS.json',OUT/'PRE_REPLAY_CHECK.json'])
        tracer='unit-weight M>=1e13 halo pre RSD'
    if vectors.shape!=(len(ids),12) or not np.isfinite(vectors).all():raise ValueError('actual complete calibration vectors required')
    target=vectors.mean(axis=0);alpha=(500-12-2)/499;chol=np.linalg.cholesky(covariance/alpha)
    PhysicalPCalibrationCase(target,covariance,chol,templates)
    root.mkdir();d=root/'cases'/name;d.mkdir(parents=True)
    np.savez_compressed(root/'CALIBRATION_VECTOR_MATRIX.npz',vectors=vectors,realizations=ids)
    np.savez_compressed(d/'likelihood.npz',target=target,covariance=covariance,chol=chol,templates=templates,bounds=BOUNDS)
    meta=dict(name=name,stage=stage,cosmology='LC_p',truth_fNL=100,case_type='physical_p_calibration',
        parameters=list(PARAMETERS),bounds=BOUNDS.tolist(),calibration_target_count=len(ids),
        calibration_ids=ids,calibration_fNL_fixed=100.,fNL_fitted=False,
        n_covariance=500,covariance_ids=cov_ids,volume_Gpc_h3=1.,covariance_multiplier=1.,covariance_is_of_mean=False,
        Hartlap_factor=alpha,P_bins=12,B_bins=0,kmax_P=.08,kmax_B=None,P_likelihood=True,B_likelihood=False,
        IR=False,tracer=tracer,calibration_main_target_overlap_ids=list(range(100)),
        strict_independent_calibration_claimed=False,p_rec_is_not_physical_p=True,
        physical_p_true_value_known=False,likelihood_sha256=digest(d/'likelihood.npz'))
    save_json(d/'contract.json',meta);save_json(root/'CASES.json',[name])
    sources.extend([d/'likelihood.npz',root/'CALIBRATION_VECTOR_MATRIX.npz'])
    save_json(root/'contract.json',dict(created_UTC=stamp(),case_contract_sha256=digest(d/'contract.json'),
        stage=stage,calibration_fNL_fixed=100.,calibration_target_count=len(ids),
        immutable_inputs={str(p):digest(p) for p in dict.fromkeys(sources)},
        physical_p_is_not_p_rec=True,all_available_positive_samples_used=True,
        covariance_is_of_mean=False,volume_Gpc_h3=1.))


def sample(stage):
    root=stage_root(stage);name=case_name(stage)
    old=(original.OUT,original.NAME,original.save_json)
    original.OUT=root;original.NAME=name
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            meta=read(Path(path).parent/'contract.json')
            value.update(stage=stage,calibration_target_count=meta['calibration_target_count'],
                all_available_positive_samples_used=True,calibration_main_target_overlap_ids=list(range(100)),
                strict_independent_calibration_claimed=False,
                gof_note='all-available calibration mean with unaveraged C_single500; not calibrated GOF')
        return old[2](path,value)
    original.save_json=save
    try:original.sample(64000,False)
    finally:original.OUT,original.NAME,original.save_json=old


def pipeline(stage):
    state=dict(pid=os.getpid(),host=socket.gethostname(),stage=stage,state='FREEZING_ALL_AVAILABLE_P_CALIBRATION',
        started_UTC=stamp(),CPU_affinity=sorted(os.sched_getaffinity(0)))
    path=OUT/(stage+'_PCAL_STATUS.json')
    def update(**values):state.update(values,updated_UTC=stamp());save_json(path,state)
    OUT.mkdir(exist_ok=True);update()
    try:
        freeze(stage);update(state='SAMPLING_ALL_AVAILABLE_PHYSICAL_P');sample(stage)
        summary=read(stage_root(stage)/'cases'/case_name(stage)/'SUMMARY.json')
        update(state='ALL_AVAILABLE_PHYSICAL_P_TRIAL_FINISHED_REVIEW_CHAIN',converged=summary['converged'],
            calibration_target_count=summary['calibration_target_count'],physical_p_median=summary['quantiles_2p5_16_50_84_97p5'][2][0])
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['pre','post'],required=True)
    ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('one original login26 CPU13 required')
    if args.detach:
        OUT.mkdir(exist_ok=True);launch=OUT/(args.stage+'_PCAL_LAUNCH.json');log=OUT/(args.stage+'_pcal.log')
        if launch.exists() or log.exists():raise ValueError('inspect actual previous run; never duplicate launch')
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--stage',args.stage],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=p.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp()))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline(args.stage)
