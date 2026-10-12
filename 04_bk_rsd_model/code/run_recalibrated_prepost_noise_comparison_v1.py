"""模块大纲：全样本p核验后完整24组4+3，再同数据/先验/p/C的24组4+4。

严格沿用唯一emcee提案/RNG与原诊断；四worker单核，共用CPU7/12/13/14。
先.14，再.08/.10/.12；mean100/C_single500/V1/no-IR，明确cov与cal重叠。
每阶段真实完成后全HDF回放才进入下一阶段，不以旧p/旧12参数结果代替。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
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
import run_sto_three_noise_prepost_v1 as sampler
import run_all_available_rsd_pcal_v1 as calibration
from rsd_b000_contract import PART,F_GROWTH,selected,digest,save_json
from post_rsd_legacy3_model import LegacyThreeNoiseModel,NAMES as THREE_NAMES
from post_rsd_legacy4_model import LegacyFourNoiseModel,NAMES as FOUR_NAMES
from post_rsd_noir_polynomial import collapse_physical_p
from post_rsd_owner_quiescence import owner_quiescent
from rsd_pcal_readonly_recheck import recheck as calibration_recheck
from audit_sto_three_noise_prepost_v1 import independent_prediction as independent_three_prediction

OUT=calibration.OUT/'pb_4plus3_then4plus4_all_p_v1'
CPUS=(7,12,13,14)
CUTS=(.14,.08,.10,.12)
TRUTHS=('fiducial','LC_p','LC_m')
SPECS=[(noise,cut,truth,stage) for noise in (3,4) for cut in CUTS for truth in TRUTHS for stage in ('post','pre')]
TEMPLATES=calibration.OUT/'pre_b_paware_templates_v1'

def read(path):return json.loads(Path(path).read_text())
def stamp():return calibration.stamp()
def case_name(index):
    noise,cut,truth,stage=SPECS[index]
    return f'{truth}_{stage}_PB_4plus{noise}_k{cut:.2f}'
def directory(index):return OUT/'cases'/case_name(index)

def guard():
    value=read(OUT/'SETTINGS.json')
    for p,want in value['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('current chain source/input changed; use a new run: '+p)
    return value

class ComparisonCase(sampler.ThreeNoiseCase):
    def __init__(self,d):
        guard();self.directory=Path(d);self.meta=read(self.directory/'contract.json')
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('actual likelihood changed')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            for name in z.files:setattr(self,name,z[name].copy())
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        model=LegacyThreeNoiseModel if self.meta['noise_parameter_count']==3 else LegacyFourNoiseModel
        self.model=model(self.coefficients,self.templates,self.primitives,self.weight_moments,
            self.meta['p_fixed_physical'],F_GROWTH)
        if self.ndim!=self.model.ndim or self.bounds.shape!=(self.ndim,2):raise ValueError('actual parameter/model dimensions inconsistent')
        np.testing.assert_allclose(self.chol@self.chol.T,self.covariance/self.meta['Hartlap_factor'],rtol=3e-13,atol=1e-7)

def independent(row,case):
    value=independent_three_prediction(np.asarray(row)[:7],case)
    if len(row)==8:
        value[12:]+=row[7]*case.weight_moments[0]*F_GROWTH**2*(case.primitives[2]-case.primitives[0]/5.)
    return value

def freeze():
    if (OUT/'SETTINGS.json').exists():raise ValueError('actual comparison inputs already frozen; do not refreeze')
    p0state=read(calibration.OUT/'P0_RECALIBRATED_STATUS.json')
    if not owner_quiescent(p0state) or p0state['state']!='SIX_RECALIBRATED_P0_TRIALS_FINISHED_REVIEW_CHAINS':
        raise ValueError('all six same-p P0 controls must actually finish before comparison freeze')
    projection=read(TEMPLATES/'COMPLETION_AUDIT.json')
    template_path=TEMPLATES/'PRE_B_PAWARE.npz'
    if digest(template_path)!=projection['sha256']:raise ValueError('actual pre explicit-p template changed')
    OUT.mkdir(exist_ok=True);pvalues={};pcalproof={}
    for stage in ('post','pre'):
        pvalues[stage],pcalproof[stage]=calibration_recheck(stage,OUT/'calibration_rechecks'/stage)
        if not pcalproof[stage]['converged']:raise ValueError('physical-p trial not converged; preserve it')
    with np.load(template_path,allow_pickle=False) as z:
        pre_coeff=collapse_physical_p(z['coefficients'],p_halo=pvalues['pre'])
        pre_primitives=z['primitives'].copy();pre_indices=z['indices'].copy()
    report_path=calibration.STO/'ensemble_measurements/AUDIT.json';report=read(report_path)
    matrix_path=Path(report['ensemble_files']['matched']['path'])
    if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:raise ValueError('actual complete C500 matrix changed')
    with np.load(matrix_path,allow_pickle=False) as z:
        matrix=z['matrix'].copy();fullcov=z['covariance_single_box'].copy();post_indices=z['B_indices'].copy()
        keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']];moments=z['data_N2_N3_nhalo'].copy()
    position={key:i for i,key in enumerate(keys)}
    post_coefficient_path=calibration.STO/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz'
    with np.load(post_coefficient_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],post_indices)
        post_coeff=collapse_physical_p(z['coefficients'],p_halo=pvalues['post'])
    primitive_path=calibration.STO/'noise_eft_primitives_v3/PRIMITIVES.npz'
    with np.load(primitive_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],post_indices);post_primitives=z['intrinsic'].copy()
    # 几何谱矩没有重构/物理p依赖，新增pre生成须复现原共同7矩，避免暗改噪声。
    pre_positions=np.array([int(np.flatnonzero(post_indices==i)[0]) for i in pre_indices])
    np.testing.assert_allclose(pre_primitives,post_primitives[:,pre_positions],rtol=3e-11,atol=1e-7)
    pre_matrix=calibration.PRE/'P0_ENSEMBLE_MATRIX.npz'
    with np.load(pre_matrix,allow_pickle=False) as z:pre_metadata=[json.loads(str(x)) for x in z['metadata_json']]
    sources=[Path(__file__),PART/'code/rsd_pcal_readonly_recheck.py',
        PART/'code/run_sto_three_noise_prepost_v1.py',PART/'code/post_rsd_legacy3_model.py',
        PART/'code/post_rsd_legacy4_model.py',PART/'code/run_rsd_b000_mcmc.py',
        PART/'code/post_rsd_noir_polynomial.py',PART/'code/post_rsd_noir_likelihood.py',
        PART/'code/post_rsd_p0_model.py',PART/'code/rsd_pb_common.py',PART/'code/rsd_bonly_common.py',
        PART/'code/rsd_b000_contract.py',PART/'code/provisional_prior_safe_initialization.py',
        PART/'code/independent_ensemble_backend.py',PART/'code/ensemble_progress_json_safe.py',
        PART/'code/hdf_lock_retry.py',PART/'code/audit_sto_three_noise_prepost_v1.py',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',
        report_path,matrix_path,template_path,TEMPLATES/'COMPLETION_AUDIT.json',
        post_coefficient_path,primitive_path,pre_matrix,calibration.OUT/'REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json']
    for stage in ('post','pre'):
        sources.extend((OUT/'calibration_rechecks'/stage).glob('*.json'))
        sources.extend([calibration.stage_root(stage)/'contract.json',
            calibration.stage_root(stage)/'cases'/calibration.case_name(stage)/'SUMMARY.json'])
    for index,(noise,cut,truth,stage) in enumerate(SPECS):
        chosen=selected(cut)
        if stage=='post':
            local=np.array([int(np.flatnonzero(post_indices==i)[0]) for i in chosen]);columns=np.r_[np.arange(12,24),24+local]
            rows=[position[truth,i] for i in range(100)];target=matrix[rows][:,columns].mean(axis=0)
            covariance=fullcov[np.ix_(columns,columns)];coeff=post_coeff[...,local];primitives=post_primitives[:,local]
            weight=moments[rows,:2].mean(axis=0);cov_ids=list(range(500))
        else:
            old=calibration.PRE/'cases'/(truth+f'_mean100_k{cut:.2f}');meta=read(old/'contract.json')
            if digest(old/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('original pre P+B measurements changed')
            with np.load(old/'likelihood.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['indices'],chosen);target=z['target'].copy();covariance=z['covariance'].copy()
            local=np.array([int(np.flatnonzero(pre_indices==i)[0]) for i in chosen]);coeff=pre_coeff[...,local];primitives=pre_primitives[:,local]
            rows=[x for x in pre_metadata if x['role']=='validation' and x['cosmology']==truth]
            if len(rows)!=100 or {x['realization'] for x in rows}!=set(range(100)):raise ValueError('actual pre target IDs not mean100')
            nbar=np.array([x['nbar_h3_Mpc3'] for x in rows]);weight=np.array([np.mean(1/nbar),np.mean(1/nbar**2)])
            cov_ids=list(range(1000,1500));sources.extend([old/'contract.json',old/'likelihood.npz'])
        p0d=calibration.OUT/(stage+'_P0_only_all_available_p')/'cases'/(truth+'_'+stage+'_P0_all_available_p')
        p0meta=read(p0d/'contract.json')
        with np.load(p0d/'likelihood.npz',allow_pickle=False) as z:
            templates=z['templates'].copy();np.testing.assert_array_equal(z['target'],target[:12])
            np.testing.assert_allclose(z['covariance'],covariance[:12,:12],rtol=3e-13,atol=1e-7)
            np.testing.assert_allclose(z['noise'],np.full(12,weight[0]),rtol=3e-15,atol=1e-10)
        if p0meta['p_fixed_physical']!=pvalues[stage]:raise ValueError('actual same-tracer P0 and P+B physical p differs')
        names=list(THREE_NAMES if noise==3 else FOUR_NAMES)
        bounds=np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-20,20],[-10,10],[-1e6/weight[0],1e6/weight[0]]])
        if noise==4:bounds=np.vstack((bounds,[-10,10]))
        ndata=len(target);alpha=(500-ndata-2)/499;d=directory(index);d.mkdir(parents=True)
        arrays=dict(target=target,covariance=covariance,chol=np.linalg.cholesky(covariance/alpha),
            coefficients=coeff,templates=templates,primitives=primitives,weight_moments=weight,bounds=bounds,indices=chosen)
        if noise==4:
            with np.load(directory(index-24)/'likelihood.npz',allow_pickle=False) as z:
                for name in arrays:
                    if name!='bounds':np.testing.assert_array_equal(arrays[name],z[name])
                np.testing.assert_array_equal(bounds[:7],z['bounds'])
        np.savez_compressed(d/'likelihood.npz',**arrays)
        meta=dict(name=d.name,stage=stage,cosmology=truth,truth_fNL=(0,100,-100)[TRUTHS.index(truth)],
            parameters=names,bounds=bounds.tolist(),noise_parameter_count=noise,total_parameter_count=4+noise,
            n_covariance=500,n_target=100,covariance_ids=cov_ids,target_ids=list(range(100)),
            P_bins=12,B_bins=len(chosen),kmax_P=.08,kmax_B=cut,Hartlap_factor=alpha,
            volume_Gpc_h3=1.,C_single=True,covariance_multiplier=1.,covariance_is_of_mean=False,
            covariance_ddof=1,P_likelihood=True,B_likelihood=True,P_B_cross_covariance=True,IR=False,
            p_fixed_physical=pvalues[stage],p_uncertainty_propagated=False,
            p_calibration_actual_positive_count=pcalproof[stage]['calibration_target_count'],
            p_calibration_covariance_is_of_mean=False,p_calibration_precision=1.,
            source_P0_case=str(p0d),P0_data_covariance_p_noise_unit_same=True,
            calibration_main_target_overlap_ids=list(range(100)),
            fiducial_target_covariance_overlap_ids=list(range(100)) if stage=='post' else [],
            independent_mock_recovery_claimed=False,legacy_noise_source_same_in_both_parameter_versions=True,
            R_extra_shape='N2*f_growth^2*(sum_Pmu4-sum_P/5)' if noise==4 else None,
            R_only_changes_B=True,formal_STO_noise_closure_certified=False,diagnostic_only=True,
            likelihood_sha256=digest(d/'likelihood.npz'),seed_case_index=index)
        save_json(d/'contract.json',meta);sources.extend([d/'contract.json',d/'likelihood.npz',p0d/'contract.json',p0d/'likelihood.npz'])
    sources=list(dict.fromkeys(sources))
    save_json(OUT/'SETTINGS.json',dict(created_UTC=stamp(),objective='p检查修复后全部4+3，再同口径4+4',
        case_names=[case_name(i) for i in range(48)],phase_noise_counts=[3,4],cases_per_phase=24,
        B_cuts_in_execution_order=list(CUTS),P_cut=.08,CPUs=list(CPUS),maximum_login_CPU_union=list(range(7,15)),
        main_C_single=True,volume_Gpc_h3=1.,target_mean_count=100,pre_and_post_covariance_count=500,
        p_each_tracer=pvalues,p_calibration_kept_original_user_authorized_Csingle=True,
        Cmean_calibration_requires_new_explicit_user_reply_and_new_run=True,
        same_data_covariance_signal_original7priors_in_4plus3_and_4plus4=True,
        ensembles=4,walkers=32,initial_steps=6000,steps_increment=2000,max_steps=64000,
        IR=False,physical_STO_closure_certified=False,immutable_inputs={str(p):digest(p) for p in sources}))
    input_audit()

def input_audit():
    guard();checks=[]
    for index in range(48):
        case=ComparisonCase(directory(index));rng=np.random.default_rng(12100+index)
        rows=rng.uniform(case.bounds[:,0],case.bounds[:,1],(128,case.ndim))
        rows=np.vstack((rows,case.bounds[:,0],case.bounds[:,1],case.bounds.mean(axis=1)))
        direct=np.array([independent(row,case) for row in rows]);got=case.model.predictions(rows)
        scale=np.maximum(1,np.max(abs(np.linalg.solve(case.chol,direct.T).T),axis=1))
        error=float(np.max(abs(np.linalg.solve(case.chol,(got-direct).T).T)/scale[:,None]))
        if error>1e-9:raise ValueError('actual expanded model independent prediction exceeds original1e-9 tolerance')
        checks.append(dict(name=case.meta['name'],probes=len(rows),maximum_white_scaled_error=error,
            noise_parameters=case.meta['noise_parameter_count'],n_data=len(case.target),p=case.meta['p_fixed_physical']))
    save_json(OUT/'INPUT_AUDIT.json',dict(created_UTC=stamp(),checks=checks,actual_case_count=48,
        independent_probes=48*131,same_data_covariance_old7priors_and_p_between_versions=True,
        HDF_read=False,formal_noise_certified=False))

def worker(index,ensemble,steps):
    if os.sched_getaffinity(0)!={CPUS[ensemble]}:raise ValueError('one declared logical CPU per ensemble')
    case=ComparisonCase(directory(index));case.initialize()
    def progress(actual):save_json(case.directory/f'PROGRESS_ensemble_{ensemble}.json',dict(
        pid=os.getpid(),actual_writer_saved_steps=int(actual),requested_checkpoint_steps=steps,
        updated_UTC=stamp(),CPU_affinity=[CPUS[ensemble]],writer_owned=True))
    sampler.advance_json_safe(case,case.directory/f'chain_ensemble_{ensemble}.h5',ensemble,index,steps,progress)

def sample(index,update):
    case=ComparisonCase(directory(index));mapmeta=case.initialize();start=time.monotonic()
    for steps in range(6000,64001,2000):
        guard();active=[]
        try:
            for ensemble,cpu in enumerate(CPUS):
                stream=(case.directory/f'worker_{ensemble}_to{steps}.log').open('x')
                child=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    '--worker',str(index),'--ensemble',str(ensemble),'--steps',str(steps)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active.append((child,stream));update(current_case=case.meta['name'],current_case_index=index,
                    requested_checkpoint_steps=steps,worker_pids=[p.pid for p,_ in active])
            while any(p.poll() is None for p,_ in active):time.sleep(.5)
            if any(p.returncode!=0 for p,_ in active):raise ValueError('actual sampler worker failed; preserve its files')
        finally:
            for proc,stream in active:
                if proc.poll() is None:proc.send_signal(signal.SIGINT);proc.wait(timeout=30)
                stream.close()
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}',read_only=True) for j in range(4)]
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape!=(4,steps,32,case.ndim):raise ValueError('actual unequal complete ensembles')
        report,flat=sampler.shared.diagnose(case,chains,accepted,steps)
        report.update(case.meta,name=case.meta['name'],chi2_MAP=mapmeta['chi2_MAP'],minuit_valid=mapmeta['minuit_valid'],
            elapsed_seconds=time.monotonic()-start,gof_pvalue=None,
            gof_note='mean100 weighted by C_single forecast, not calibrated GOF')
        save_json(case.directory/'SUMMARY.json',sampler.shared.clean(report))
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        print(case.meta['name'],steps,report['converged'],report['gates'],flush=True)
        if report['converged']:break
    pick=np.random.default_rng(7500+index).choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=case.names)
    return sampler.shared.clean(report)

def completed_case_audit(index):
    case=ComparisonCase(directory(index));case.initialize();stored=read(case.directory/'SUMMARY.json')
    for p in case.directory.glob('PROGRESS_ensemble_*.json'):
        if not owner_quiescent(read(p)):raise ValueError('actual writer active; do not audit HDF')
    backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}',read_only=True) for j in range(4)]
    if [int(b.iteration) for b in backends]!=[stored['steps']]*4:raise ValueError('actual finished steps disagree')
    chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
    physical=case.physical(chains)
    if np.any(~np.isfinite(physical)) or np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):
        raise ValueError('completed chain violates actual priors')
    replay,flat=sampler.shared.diagnose(case,chains,accepted,stored['steps'])
    if replay['gates']!=stored['gates'] or replay['converged']!=stored['converged']:raise ValueError('actual completed original gates disagree')
    np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
    np.testing.assert_allclose(replay['fNL_sigma'],stored['fNL_sigma'],rtol=2e-13)
    with np.load(case.directory/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
        pick=np.random.default_rng(7500+index).choice(len(flat),min(30000,len(flat)),replace=False)
        np.testing.assert_array_equal(z['parameter_names'],case.names);np.testing.assert_array_equal(z['samples'],flat[pick])
    # 全链统计之外独立抽查每ensemble128个原log_prob，不冒称逐点全链概率审计。
    rng=np.random.default_rng(73120+index);maximum=0.
    for ensemble,b in enumerate(backends):
        lp=b.get_log_prob();pick=rng.choice(chains.shape[1]*32,128,replace=False)
        coordinates=chains[ensemble].reshape(-1,case.ndim)[pick];values=case.physical(coordinates)
        pred=np.array([independent(row,case) for row in values]);white=np.linalg.solve(case.chol,(pred-case.target).T).T
        direct=-.5*np.sum(white*white,axis=1);saved=lp.ravel()[pick]
        maximum=max(maximum,float(np.max(abs(direct-saved))))
        np.testing.assert_allclose(saved,direct,rtol=1e-10,atol=1e-8)
    result=dict(created_UTC=stamp(),name=case.meta['name'],index=index,converged=bool(replay['converged']),
        full_HDF_prior_gate_quantile_sigma_and_thin_RNG_replayed=True,steps_each_ensemble=stored['steps'],
        stored_logprob_independent_spots=512,maximum_logprob_spot_error=maximum,
        summary_sha256=digest(case.directory/'SUMMARY.json'),chain_sha256=[digest(Path(b.filename)) for b in backends])
    save_json(case.directory/'COMPLETION_AUDIT.json',result);return result

def wait_upstream(update):
    while True:
        p0=read(calibration.OUT/'P0_RECALIBRATED_STATUS.json')
        projection=read(TEMPLATES/'STATUS.json') if (TEMPLATES/'STATUS.json').exists() else None
        if p0['state']=='FAILED_OR_INTERRUPTED_PRESERVED':raise ValueError('actual P0 continuation failed')
        if projection and projection['state']=='FAILED_OR_INTERRUPTED_PRESERVED':raise ValueError('actual explicit-p template construction failed')
        p0ready=p0['state']=='SIX_RECALIBRATED_P0_TRIALS_FINISHED_REVIEW_CHAINS' and owner_quiescent(p0)
        bread=projection and projection['state']=='PRE_EXPLICIT_P_B_TEMPLATES_COMPLETE' and owner_quiescent(projection)
        if not p0ready and owner_quiescent(p0):raise ValueError('P0 owner exited without full completion')
        if projection and not bread and owner_quiescent(projection):raise ValueError('B projection owner exited without completion')
        update(state='WAITING_FOR_ACTUAL_ALL_PCAL_P0_AND_PRE_PAWARE_B',P0_state=p0['state'],
            pre_template_state=None if projection is None else projection['state'],
            pre_template_completed_bins=0 if projection is None else projection.get('completed_bins',0))
        if p0ready and bread:return
        time.sleep(10)

def pipeline():
    OUT.mkdir(exist_ok=True);state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[12],
        started_UTC=stamp(),completed_cases=[],state='WAITING_FOR_ACTUAL_ALL_PCAL_P0_AND_PRE_PAWARE_B')
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update()
    try:
        wait_upstream(update);update(state='FREEZING_SAME_CALIBRATED_P_ALL48_CASES');freeze()
        reports=[];audits=[]
        for noise in (3,4):
            update(state=f'SAMPLING_ALL_4PLUS{noise}_PREPOST_CUTS',current_noise_parameter_count=noise)
            for index in range((noise-3)*24,(noise-2)*24):
                reports.append(sample(index,update));audits.append(completed_case_audit(index))
                state['completed_cases'].append(case_name(index));update()
            phase=audits[-24:]
            save_json(OUT/f'PHASE_4PLUS{noise}_COMPLETION.json',dict(created_UTC=stamp(),actual_cases=24,
                all_cases_independently_verified=True,all_converged=all(x['converged'] for x in phase),audits=phase))
            if not all(x['converged'] for x in phase):
                raise ValueError('phase completed but some original convergence gates failed; retain cases for budget-only continuation')
        save_json(OUT/'RESULTS.json',dict(created_UTC=stamp(),reports=reports,all48_converged=True,
            noise_versions=[3,4],all_original_gates_and_full_chain_replays_verified=True,
            mean_data_count=100,C_single500=True,volume_Gpc_h3=1.,IR=False,
            full_physical_STO_noise_closure_certified=False,final_comparative_report_and_PDF_still_required=True))
        update(state='ALL48_FITS_AND_FULL_CHAIN_AUDITS_COMPLETE_REPORT_REQUIRED',current_case=None,finished_UTC=stamp())
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--ensemble',type=int);ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    if socket.gethostname()!='login26':raise ValueError('original login26 CPU lease required')
    if args.worker is not None:worker(args.worker,args.ensemble,args.steps)
    elif args.detach:
        if os.sched_getaffinity(0)!={12}:raise ValueError('one controller CPU12 required')
        OUT.mkdir(exist_ok=True);launch=OUT/'LAUNCH.json';log=OUT/'controller.log'
        if launch.exists() or log.exists():raise ValueError('inspect previous actual owner; never duplicate launch')
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=child.pid,host=socket.gethostname(),created_UTC=stamp(),CPUs=list(CPUS)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()
