"""模块大纲：用真正post-RSD P0校准p，再自动跑P0和两档4+4 P+B。

只改变校准数据阶段与由此得到的固定p；其余数据、C_single500/V1、
先验、窗口、no-IR模型和原采样诊断不变。旧输入/链/摘要保持只读。
复用现有模型、MAP、emcee及诊断，不新增噪声参数或调整后验位置。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import emcee
from rsd_b000_contract import PART, F_GROWTH, digest, save_json
from run_sto_physical_p_mcmc import CalibrationCase
from run_rsd_p0only import POnlyCase
from run_all_available_p_p0only_v1 import AllPCase
from run_sto_three_noise_prepost_v1 import ThreeNoiseCase
from post_rsd_legacy4_model import LegacyFourNoiseModel, NAMES
from post_rsd_p_calibration import BOUNDS, PARAMETERS
from post_rsd_noir_polynomial import collapse_physical_p
from audit_all_available_rsd_pcal_v1 import direct_p
from run_recalibrated_prepost_noise_comparison_v1 import independent as independent_pb
from ensemble_progress_json_safe import advance_json_safe
import run_rsd_b000_mcmc as shared

OUT = PART/'outputs/sto_marisa_postP_calibration_20261011_v1'
OLD = PART/'outputs/rsd_all_available_p_recalibration_20261010_v2'
STO = PART/'outputs/sto_marisa_post_rsd_mean100_v1'
MATRIX = STO/'ensemble_measurements/MATCHED_ENSEMBLE.npz'
TENSOR = STO/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz'
CPUS = (7,12,13,14)
TRUTHS = ('fiducial','LC_p','LC_m')
SPECS = [('cal','LC_p',None)] + [('P0',t,None) for t in TRUTHS] + [
    ('PB',t,k) for k in (.08,.14) for t in TRUTHS]

def read(path): return json.loads(Path(path).read_text())
def stamp(): return datetime.now(timezone.utc).isoformat()
def directory(index):
    kind,truth,cut = SPECS[index]
    name = 'postP_physical_p_fNLfixed100' if kind == 'cal' else (
        truth+'_post_P0_postPcal' if kind == 'P0' else truth+f'_post_PB_4plus4_k{cut:.2f}_postPcal')
    return OUT/'cases'/name

def arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def guard():
    settings = read(OUT/'SETTINGS.json')
    for path,want in settings['immutable_inputs'].items():
        if digest(path) != want: raise ValueError('本轮输入或代码改变，须另开版本: '+path)
    return settings

class FixedPOnlyCase(POnlyCase):
    predictions = AllPCase.predictions
    residual = AllPCase.residual

class FixedPBCase(ThreeNoiseCase):
    def __init__(self,d):
        self.directory = Path(d); self.meta = read(self.directory/'contract.json')
        if digest(self.directory/'likelihood.npz') != self.meta['likelihood_sha256']:
            raise ValueError('P+B似然文件改变')
        for key,value in arrays(self.directory/'likelihood.npz').items(): setattr(self,key,value)
        self.names = self.meta['parameters']; self.ndim = len(self.names)
        if self.names != list(NAMES) or self.ndim != 8: raise ValueError('要求4+4参数')
        self.model = LegacyFourNoiseModel(self.coefficients,self.templates,self.primitives,
            self.weight_moments,self.meta['p_fixed_physical'],F_GROWTH)

def case_at(index):
    return {'cal':CalibrationCase,'P0':FixedPOnlyCase,'PB':FixedPBCase}[SPECS[index][0]](directory(index))

def seed_index(index):
    kind,truth,cut = SPECS[index]
    if kind == 'cal': return 0
    if kind == 'P0': return TRUTHS.index(truth)
    return (30 if cut == .08 else 24)+2*TRUTHS.index(truth)

def thin_seed(index): return (7500 if SPECS[index][0]=='PB' else 5500)+seed_index(index)

def write_case(index,data,meta,sources):
    d=directory(index); d.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(d/'likelihood.npz',**data)
    meta.update(name=d.name,likelihood_sha256=digest(d/'likelihood.npz'),
        seed_case_index=seed_index(index),thin_seed=thin_seed(index),stage='post',
        volume_Gpc_h3=1.,C_single=True,covariance_multiplier=1.,covariance_is_of_mean=False,
        covariance_ddof=1,n_covariance=500,IR=False,p_rec_is_not_physical_p=True,
        p_calibration_data_stage='STO+Marisa post-RSD P0 after Marisa displacement',
        p_calibration_block='P_post',calibration_main_target_overlap_ids=list(range(100)),
        independent_mock_recovery_claimed=False,diagnostic_only=True)
    save_json(d/'contract.json',meta)
    save_json(d/'SOURCES.json',{str(p):digest(p) for p in dict.fromkeys(sources)})

def verify_sources(index):
    for path,want in read(directory(index)/'SOURCES.json').items():
        if digest(path)!=want: raise ValueError('本case来源改变: '+path)

def freeze_calibration():
    if (OUT/'SETTINGS.json').exists(): guard(); return
    OUT.mkdir(exist_ok=True)
    report=read(STO/'ensemble_measurements/AUDIT.json')
    if digest(MATRIX)!=report['ensemble_files']['matched']['sha256']: raise ValueError('原700测量改变')
    if list(report['block_slices']['P_post']) != [12,24]: raise ValueError('post列位置改变')
    z=arrays(MATRIX); keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']]
    positive=[i for i,k in enumerate(keys) if k[0]=='LC_p']
    ids=[int(keys[i][1]) for i in positive]
    if len(set(keys))!=len(keys) or sorted(ids)!=list(range(100)): raise ValueError('实际post样本集合变化，需检查全部可用输入')
    pos={k:i for i,k in enumerate(keys)}
    vectors=z['matrix'][positive,12:24]; target=vectors.mean(axis=0)
    cov=z['covariance_single_box'][12:24,12:24]
    covariance_rows=z['matrix'][[pos['fiducial',i] for i in range(500)],12:24]
    np.testing.assert_allclose(np.cov(covariance_rows,rowvar=False,ddof=1),cov,rtol=3e-13,atol=1e-8)
    original=OLD/'post_P0_only_all_available_p/cases/LC_p_post_P0_all_available_p'
    old=arrays(original/'likelihood.npz')
    np.testing.assert_array_equal(target,old['target']); np.testing.assert_array_equal(cov,old['covariance'])
    data=dict(target=target,covariance=cov,chol=old['chol'],templates=old['templates'],bounds=BOUNDS)
    sources=[MATRIX,STO/'ensemble_measurements/AUDIT.json',original/'likelihood.npz',original/'contract.json']
    write_case(0,data,dict(cosmology='LC_p',truth_fNL=100,case_type='physical_p_calibration',
        parameters=list(PARAMETERS),bounds=BOUNDS.tolist(),P_bins=12,B_bins=0,kmax_P=.08,kmax_B=None,
        P_likelihood=True,B_likelihood=False,P_derived_b1_prior=False,fNL_fitted=False,
        calibration_fNL_fixed=100.,calibration_target_count=len(ids),calibration_ids=ids,
        all_available_positive_post_P0_used=True,physical_p_true_value_known=False,Hartlap_factor=486/499,
        covariance_ids=list(range(500))),sources)
    np.savez_compressed(OUT/'CALIBRATION_VECTOR_MATRIX.npz',vectors=vectors,realizations=ids,
        source_columns=np.arange(12,24))
    # 保存当前实际依赖，不修改旧模块；采样时只核验本轮实际读取的文件。
    source_names=['run_postP_calibrated_rsd_v1.py','run_sto_physical_p_mcmc.py',
        'post_rsd_p_calibration.py','run_rsd_p0only.py','run_all_available_p_p0only_v1.py',
        'run_sto_three_noise_prepost_v1.py','post_rsd_legacy3_model.py','post_rsd_legacy4_model.py',
        'post_rsd_p0_model.py','post_rsd_noir_likelihood.py','post_rsd_noir_polynomial.py',
        'rsd_pb_common.py','rsd_bonly_common.py','rsd_b000_contract.py','run_rsd_b000_mcmc.py',
        'independent_ensemble_backend.py','ensemble_progress_json_safe.py','hdf_lock_retry.py',
        'provisional_prior_safe_initialization.py','audit_all_available_rsd_pcal_v1.py',
        'audit_sto_three_noise_prepost_v1.py','run_recalibrated_prepost_noise_comparison_v1.py']
    sources += [PART/'code'/name for name in source_names]
    sources += [TENSOR,PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',
        directory(0)/'contract.json',directory(0)/'likelihood.npz',OUT/'CALIBRATION_VECTOR_MATRIX.npz']
    save_json(OUT/'SETTINGS.json',dict(created_UTC=stamp(),case_names=[directory(i).name for i in range(10)],
        phases=['post_P0_p_calibration','three_P0_only','six_4plus4_PB'],CPUs=list(CPUS),
        calibration_count=len(ids),calibration_data_stage='P_post after Marisa shift',
        p_estimator='posterior median',p_uncertainty_propagated=False,
        main_mean_target_count=100,C_single500=True,volume_Gpc_h3=1.,P_cut=.08,B_cuts=[.08,.14],IR=False,
        ensembles=4,walkers=32,initial_steps=6000,step_increment=2000,max_steps=64000,
        immutable_inputs={str(p):digest(p) for p in dict.fromkeys(sources)}))
    input_audit(0)

def freeze_main(index,p):
    kind,truth,cut=SPECS[index]
    if directory(index).exists():
        verify_sources(index)
        if case_at(index).meta['p_fixed_physical']!=p: raise ValueError('已有case固定p不同')
        return
    if kind=='P0':
        src=OLD/'post_P0_only_all_available_p/cases'/(truth+'_post_P0_all_available_p')
    else:
        src=OLD/'pb_4plus3_then4plus4_all_p_v1/cases'/(truth+f'_post_PB_4plus4_k{cut:.2f}')
    meta=read(src/'contract.json'); data=arrays(src/'likelihood.npz')
    if digest(src/'likelihood.npz')!=meta['likelihood_sha256']: raise ValueError('旧对照输入改变')
    preserved={k:v.copy() for k,v in data.items()}; oldp=meta['p_fixed_physical']
    sources=[src/'contract.json',src/'likelihood.npz',OUT/'P_CHOICE.json',directory(0)/'SUMMARY.json']
    if kind=='PB':
        tensor=arrays(TENSOR); loc=np.array([int(np.flatnonzero(tensor['indices']==i)[0]) for i in data['indices']])
        # 先还原旧p系数，证明只更新p而没有改源核或分箱。
        np.testing.assert_allclose(collapse_physical_p(tensor['coefficients'],p_halo=oldp)[...,loc],
            data['coefficients'],rtol=2e-13,atol=1e-8)
        data['coefficients']=collapse_physical_p(tensor['coefficients'],p_halo=p)[...,loc]
        sources.append(TENSOR)
        p0case=case_at(1+TRUTHS.index(truth))
        np.testing.assert_array_equal(data['target'][:12],p0case.target)
        np.testing.assert_array_equal(data['covariance'][:12,:12],p0case.covariance)
    for name in data:
        if name!='coefficients': np.testing.assert_array_equal(data[name],preserved[name])
    # 更新来源字段，不能把原weighted-input/250元数据带进新结果。
    for name in list(meta):
        if 'calibration' in name or name.startswith('physical_p_') or name=='source_P0_case': meta.pop(name)
    meta.update(p_fixed_physical=p,p_uncertainty_propagated=False,p_estimator='posterior median',
        calibration_target_count=100,physical_p_calibration_target_count=100,
        p_calibration_actual_positive_count=100,physical_p_calibration_summary_sha256=digest(directory(0)/'SUMMARY.json'),
        source_case=str(src),previous_fixed_p=oldp,case_type='P0_only' if kind=='P0' else 'PB_4plus4',
        data_covariance_priors_windows_unchanged=True,only_changed_calibration_stage_and_fixed_p=True)
    if kind=='PB': meta['source_P0_case']=str(directory(1+TRUTHS.index(truth)))
    write_case(index,data,meta,sources); input_audit(index)

def independent_rows(index,case,rows):
    kind=SPECS[index][0]
    if kind=='PB': return np.array([independent_pb(row,case) for row in rows])
    if kind=='cal': return np.array([direct_p(100,b,p,case.templates)+s for p,b,s in rows])
    return np.array([direct_p(f,b,case.meta['p_fixed_physical'],case.templates)+s*case.noise for f,b,s in rows])

def input_audit(index):
    case=case_at(index); verify_sources(index)
    rng=np.random.default_rng(81100+index)
    rows=rng.uniform(case.bounds[:,0],case.bounds[:,1],(128,case.ndim))
    rows=np.vstack((rows,case.bounds[:,0],case.bounds[:,1],case.bounds.mean(axis=1)))
    direct=independent_rows(index,case,rows)
    expected=np.linalg.solve(case.chol,(direct-case.target).T).T
    actual=case.residual(rows)
    scale=np.maximum(1,np.max(abs(expected),axis=1))
    error=float(np.max(abs(actual-expected)/scale[:,None]))
    if error>1e-9: raise ValueError('独立预测与实际残差不一致')
    np.testing.assert_allclose(case.chol@case.chol.T,case.covariance/case.meta['Hartlap_factor'],rtol=3e-13,atol=1e-7)
    save_json(case.directory/'INPUT_AUDIT.json',dict(probes=len(rows),max_white_scaled_error=error,
        C_single=True,volume_Gpc_h3=1.,calibration_block='P_post',independent_prediction_verified=True))

def worker(index,ensemble,steps):
    if os.sched_getaffinity(0)!={CPUS[ensemble]}: raise ValueError('worker须单核')
    guard(); verify_sources(index); case=case_at(index); case.initialize()
    def progress(n):
        save_json(case.directory/f'PROGRESS_ensemble_{ensemble}.json',dict(pid=os.getpid(),
            actual_writer_saved_steps=int(n),requested_checkpoint_steps=steps,
            CPU_affinity=[CPUS[ensemble]],updated_UTC=stamp()))
    advance_json_safe(case,case.directory/f'chain_ensemble_{ensemble}.h5',ensemble,seed_index(index),steps,progress)

def normalize_report(index,report):
    if index==0:
        report=dict(report); report.pop('fNL_pull',None)
        report['physical_p_sigma']=report.pop('fNL_sigma')
        report['gates']={('physical_p_ESS' if k=='fNL_ESS' else k):v for k,v in report['gates'].items()}
    return shared.clean(report)

def load_chains(case):
    backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),
        name=f'ensemble_{j}',read_only=True) for j in range(4)]
    return backends,np.stack([b.get_chain() for b in backends]),np.stack([b.accepted for b in backends])

def completed_audit(index):
    guard(); verify_sources(index); case=case_at(index); case.initialize(); saved=read(case.directory/'SUMMARY.json')
    # 只由已wait全部worker的controller或确认controller退出后的审计入口调用。
    backends,chains,accepted=load_chains(case)
    if chains.shape!=(4,saved['steps'],32,case.ndim): raise ValueError('四链实际步数不一致')
    physical=case.physical(chains)
    if not np.isfinite(physical).all() or np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):
        raise ValueError('全链样本超出原先验')
    report,flat=shared.diagnose(case,chains,accepted,saved['steps']); report=normalize_report(index,report)
    if report['gates']!=saved['gates'] or not report['converged']: raise ValueError('原收敛标准未通过')
    np.testing.assert_allclose(report['quantiles_2p5_16_50_84_97p5'],saved['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
    sigma_key='physical_p_sigma' if index==0 else 'fNL_sigma'
    np.testing.assert_allclose(report[sigma_key],saved[sigma_key],rtol=2e-13)
    thin=arrays(case.directory/'POSTERIOR_THIN.npz')
    pick=np.random.default_rng(thin_seed(index)).choice(len(flat),min(30000,len(flat)),replace=False)
    np.testing.assert_array_equal(thin['samples'],flat[pick]); np.testing.assert_array_equal(thin['parameter_names'],case.names)
    maximum=0.; rng=np.random.default_rng(85100+index)
    for j,b in enumerate(backends):
        take=rng.choice(chains.shape[1]*32,128,replace=False)
        rows=physical[j].reshape(-1,case.ndim)[take]
        white=np.linalg.solve(case.chol,(independent_rows(index,case,rows)-case.target).T).T
        direct=-.5*np.sum(white**2,axis=1); stored=b.get_log_prob().ravel()[take]
        maximum=max(maximum,float(np.max(abs(direct-stored))))
        np.testing.assert_allclose(direct,stored,rtol=1e-10,atol=1e-8)
    result=dict(name=case.directory.name,converged=True,steps_each_ensemble=saved['steps'],
        complete_HDF_priors_gates_quantiles_sigma_thin_replayed=True,independent_logprob_spots=512,
        max_logprob_error=maximum,summary_sha256=digest(case.directory/'SUMMARY.json'),
        chain_sha256=[digest(b.filename) for b in backends],created_UTC=stamp())
    save_json(case.directory/'COMPLETION_AUDIT.json',result); return result

def sample(index,update):
    guard(); verify_sources(index); case=case_at(index); mapmeta=case.initialize(); started=time.monotonic()
    for steps in range(6000,64001,2000):
        active=[]
        try:
            for j,cpu in enumerate(CPUS):
                stream=(case.directory/f'worker_{j}_to{steps}.log').open('x')
                proc=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    '--worker',str(index),'--ensemble',str(j),'--steps',str(steps)],stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
                active.append((proc,stream))
                update(current_case=case.directory.name,current_case_index=index,
                    requested_checkpoint_steps=steps,worker_pids=[p.pid for p,_ in active])
            while any(p.poll() is None for p,_ in active): time.sleep(.5)
            if any(p.returncode!=0 for p,_ in active): raise ValueError('采样worker失败，保留日志和链')
        finally:
            for proc,stream in active:
                if proc.poll() is None: proc.send_signal(signal.SIGINT); proc.wait(timeout=30)
                stream.close()
        backends,chains,accepted=load_chains(case)
        if chains.shape!=(4,steps,32,case.ndim): raise ValueError('checkpoint四链不等长')
        report,flat=shared.diagnose(case,chains,accepted,steps)
        report.update(case.meta,chi2_MAP=mapmeta['chi2_MAP'],minuit_valid=mapmeta['minuit_valid'],
            elapsed_seconds=time.monotonic()-started,gof_pvalue=None,
            gof_note='mean data with C_single forecast; not a calibrated goodness-of-fit p-value')
        report=normalize_report(index,report); save_json(case.directory/'SUMMARY.json',report)
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        print(case.directory.name,steps,report['converged'],report['gates'],flush=True)
        if report['converged']: break
    pick=np.random.default_rng(thin_seed(index)).choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=case.names)
    return completed_audit(index)

def pipeline():
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=stamp(),state='CALIBRATING_POST_P0',completed_cases=[])
    def update(**values): state.update(values,updated_UTC=stamp()); save_json(OUT/'STATUS.json',state)
    update()
    try:
        freeze_calibration(); audits=[sample(0,update)]
        summary=read(directory(0)/'SUMMARY.json'); p=float(summary['quantiles_2p5_16_50_84_97p5'][2][0])
        save_json(OUT/'P_CHOICE.json',dict(physical_p=p,estimator='posterior median',
            physical_p_sigma=summary['physical_p_sigma'],quantiles=np.asarray(summary['quantiles_2p5_16_50_84_97p5'])[:,0].tolist(),
            MAP_p=read(directory(0)/'MAP.json')['center'][0],calibration_data_stage='post-RSD P0 after Marisa shift',
            calibration_count=100,summary_sha256=digest(directory(0)/'SUMMARY.json'),p_uncertainty_propagated=False))
        state['completed_cases'].append(directory(0).name)
        for index in range(1,10):
            update(state='SAMPLING_P0_ONLY' if index<4 else 'SAMPLING_4PLUS4_PB',physical_p=p)
            freeze_main(index,p); audits.append(sample(index,update)); state['completed_cases'].append(directory(index).name); update()
            if index==3:
                save_json(OUT/'P0_RESULTS.json',dict(p=p,reports=[read(directory(i)/'SUMMARY.json') for i in range(1,4)]))
        save_json(OUT/'RESULTS.json',dict(physical_p=p,calibration=summary,
            reports=[read(directory(i)/'SUMMARY.json') for i in range(1,10)],audits=audits,
            all_ten_original_sampling_gates_verified=True,full_STO_noise_closure_certified=False))
        update(state='ALL_TEN_CHAINS_VERIFIED_COMPARISON_REPORT_REQUIRED',current_case=None,finished_UTC=stamp())
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc()); raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach',action='store_true')
    ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--worker',type=int)
    ap.add_argument('--ensemble',type=int); ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    if socket.gethostname()!='login26': raise ValueError('本轮指定login26')
    if args.worker is not None: worker(args.worker,args.ensemble,args.steps)
    elif args.prepare_only: freeze_calibration(); print('POST_P0_INPUT_PREPARED',flush=True)
    elif args.detach:
        if os.sched_getaffinity(0)!={12}: raise ValueError('controller单核12')
        OUT.mkdir(exist_ok=True)
        if (OUT/'LAUNCH.json').exists() or (OUT/'controller.log').exists(): raise ValueError('已有启动记录，先检查实际进程，不重复启动')
        with (OUT/'controller.log').open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(OUT/'LAUNCH.json',dict(pid=proc.pid,host=socket.gethostname(),CPUs=list(CPUS),created_UTC=stamp()))
        print(json.dumps(read(OUT/'LAUNCH.json')),flush=True)
    else: pipeline()
