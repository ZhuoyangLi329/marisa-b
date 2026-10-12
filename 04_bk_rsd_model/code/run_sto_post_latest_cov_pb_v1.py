"""模块大纲：人要求最新完成fid数量重跑；同快照p/P0/P+B，四ensemble并行。

旧205链只读保留；目标仍mean100，协方差C_single/V1，不追随新DONE动态改变。
原EFT12公式/先验/moves/RNG/收敛门/64k预算不变；各writer独占自己的HDF。
login测量8..11照常；四ensemble用7/12/13/14，任务CPU union仍7..14共8核。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
import emcee
from rsd_b000_contract import PART,digest,save_json,F_GROWTH,selected
import run_rsd_b000_mcmc as shared
import run_sto_post_partial_p_mcmc_v2 as pbase
import run_sto_post_provisional_eft_mcmc_v1 as original
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2,NAMES_V2
from post_rsd_noir_polynomial import collapse_physical_p
from provisional_prior_safe_initialization import bounded_affine_transform,physical_initial_cloud
from independent_ensemble_backend import advance_ensemble
from hdf_lock_retry import open_with_lock_retry
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from run_sto_eft_pb_preflight_v1 import moments

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'diagnostic_latest_cov_pb_20261010_v1'
POUT=OUT/'p_control';BOUT=OUT/'pb_cases'
CPUS=(7,12,13,14);COSMOS=('fiducial','LC_p','LC_m')


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def runtime_sources():
    names=('run_sto_post_latest_cov_pb_v1.py','independent_ensemble_backend.py',
        'test_independent_ensemble_backend.py','hdf_lock_retry.py','test_hdf_lock_retry.py',
        'provisional_prior_safe_initialization.py','test_provisional_prior_safe_initialization.py',
        'post_rsd_eft_candidate_model_v2.py','post_rsd_eft_candidate_model_v1.py',
        'post_rsd_stochastic_eft_v1.py','run_sto_post_provisional_eft_mcmc_v1.py',
        'run_rsd_b000_mcmc.py')
    return {str(PART/'code'/name):digest(PART/'code'/name) for name in names}


def guard():
    c=read(OUT/'PB_CONTRACT.json')
    if (not c['diagnostic_only'] or c['covariance_is_of_mean']
            or c['covariance_multiplier']!=1. or c['volume_Gpc_h3']!=1.):
        raise ValueError('authorized diagnostic C_single/V1 required')
    for path,want in c['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('latest-cohort source/likelihood drift: '+path)
    return c


class LatestPBCase(original.ProvisionalCase):
    def __init__(self,d):
        guard();self.directory=Path(d);self.meta=read(self.directory/'contract.json')
        if digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('latest case drift')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            for key in z.files:setattr(self,key,np.array(z[key]))
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=list(NAMES_V2):raise ValueError('same12 parameter order required')
        self.model=EFTCandidateModelV2(self.coefficients,self.templates,self.operators,self.intrinsic,
            self.matched_zero,self.weight_moments,self.meta['p_fixed_physical'],F_GROWTH)

    def initialize(self):
        m=super().initialize()
        if m.get('prior_safe_initialization'):return m
        save_json(self.directory/'MAP_HESSIAN_PROPOSAL_ORIGINAL.json',m)
        self.transform,conditioning=bounded_affine_transform(np.array(m['transform']),self.bounds)
        m=dict(m,transform=self.transform.tolist(),prior_safe_initialization=True,affine_conditioning=conditioning)
        save_json(self.directory/'MAP.json',m);return m

    def initial(self,seed,ensemble,mapmeta):
        best=np.array(mapmeta['center'])
        good=[r for r in mapmeta['candidates'] if r['chi2']<=mapmeta['chi2_MAP']+10 and r['valid']]
        if ensemble>=2 and good:best=np.array(good[(ensemble-2)%len(good)]['parameters'])
        values=physical_initial_cloud(best,self.transform,self.bounds,seed)
        points=self.coordinates(values)
        if not np.isfinite(self.log_prob(points)).all():raise ValueError('latest initial walkers not finite')
        return points


def freeze_pb():
    pc=pbase.parent();ns=pc['n_covariance'];ids=pc['covariance_ids']
    cal=read(POUT/'cases'/pbase.CAL_NAME/'SUMMARY.json')
    if not cal['converged']:raise ValueError('same-latest-cohort physical p must actually converge')
    p=float(cal['quantiles_2p5_16_50_84_97p5'][2][0])
    target_path=ROOT/'target300_completion_v1/TARGETS_ONLY.npz'
    if digest(target_path)!=read(ROOT/'target300_completion_v1/AUDIT.json')['target_matrix_sha256']:
        raise ValueError('original targets changed')
    with np.load(target_path,allow_pickle=False) as z:
        indices=z['B_indices'].copy();targets=z['matched_rows'].copy();weights=z['data_N2_N3_nhalo'].copy()
    chosen=selected(.08);local=np.array([np.flatnonzero(indices==j)[0] for j in chosen])
    columns=np.r_[np.arange(12,24),24+local]
    rows=[];records=[]
    for real in ids:
        data,meta,done=load_pilot(directory(('fiducial',real)))
        rows.append(np.r_[data['P_weighted_input'],data['P_post'],data['B_post'][indices]])
        records.append(dict(realization=real,measurement_sha256=done['measurement_sha256']))
    full=np.array(rows);covfull=np.cov(full,rowvar=False,ddof=1)
    with np.load(POUT/'FROZEN_PARTIAL_P_ENSEMBLE.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(covfull[:24,:24],z['covariance_single_box'],rtol=3e-13,atol=1e-7)
        np.testing.assert_allclose(targets[:,:24].reshape(3,100,24).mean(axis=1),z['target_vectors'],rtol=3e-13,atol=1e-7)
        templates=z['templates'].copy()
    covariance=covfull[np.ix_(columns,columns)];alpha=(ns-23)/(ns-1)
    chol=np.linalg.cholesky(covariance/alpha)
    deterministic=ROOT/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz'
    operators_path=ROOT/'noise_transport_templates_v1/OPERATOR_COEFFICIENTS.npz'
    primitives=ROOT/'noise_eft_primitives_v3/PRIMITIVES.npz'
    proofs=[ROOT/'noir_paware_templates/INDEPENDENT_COEFFICIENT_AUDIT.json',
        ROOT/'noise_transport_templates_v1/INDEPENDENT_COEFFICIENT_AUDIT.json',ROOT/'noise_eft_primitives_v3/AUDIT.json']
    if digest(deterministic)!=read(proofs[0])['coefficients_sha256']:raise ValueError('deterministic source changed')
    if digest(operators_path)!=read(proofs[1])['coefficients_sha256']:raise ValueError('transport source changed')
    if digest(primitives)!=read(proofs[2])['matrix_sha256']:raise ValueError('intrinsic/zero source changed')
    with np.load(deterministic,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices);coefficients=collapse_physical_p(z['coefficients'],p_halo=p)[...,local]
    with np.load(operators_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices)
        basis=np.array([1.,(p-1.3)/.5,2*((p-1.3)/.5)**2-1])
        operators=np.tensordot(z['coefficients'],basis,axes=(1,0))[...,local]
    with np.load(primitives,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices)
        intrinsic=z['intrinsic'][:,local];zero=z['matched_zero'][:,local]
    source=runtime_sources();source.update(pc['immutable_inputs'])
    paths=[POUT/'CONTRACT.json',POUT/'cases'/pbase.CAL_NAME/'SUMMARY.json',target_path,deterministic,operators_path,primitives]+proofs
    for i,cosmo in enumerate(COSMOS):
        name=cosmo+'_post_PB_kB0.08_kP0.08_provisional_EFT12_latestcov';d=BOUT/name;d.mkdir(parents=True)
        units=moments(weights[i*100:(i+1)*100]);oldcase=original.OUT/'cases'/original.CASES[i]
        oldmeta=read(oldcase/'contract.json');bounds=np.array(oldmeta['bounds'])
        if oldmeta['parameters']!=list(NAMES_V2):raise ValueError('oldEFT12 prior contract changed')
        target=targets[i*100:(i+1)*100][:,columns].mean(axis=0)
        np.savez_compressed(d/'likelihood.npz',target=target,covariance=covariance,chol=chol,bounds=bounds,
            coefficients=coefficients,templates=templates,operators=operators,intrinsic=intrinsic,
            matched_zero=zero,weight_moments=units)
        save_json(d/'contract.json',dict(name=name,cosmology=cosmo,truth_fNL=(0,100,-100)[i],
            parameters=list(NAMES_V2),bounds=bounds.tolist(),likelihood_sha256=digest(d/'likelihood.npz'),
            n_covariance=ns,covariance_ids=ids,n_target=100,target_ids=list(range(100)),
            P_bins=12,B_bins=9,kmax_P=.08,kmax_B=.08,P_likelihood=True,B_likelihood=True,P_B_cross_covariance=True,
            volume_Gpc_h3=1.,covariance_is_of_mean=False,covariance_multiplier=1.,covariance_ddof=1,Hartlap_factor=alpha,
            p_fixed_physical=p,p_uncertainty_propagated=False,IR=False,case_type='provisional_PB_EFT12_latest_cov',
            diagnostic_only=True,provisional=True,full_STO_noise_closure_certified=False,
            fid_target_covariance_overlap=sorted(set(ids)&set(range(100))),strict_independent_mock_recovery=False))
        paths.extend([d/'likelihood.npz',d/'contract.json',oldcase/'contract.json'])
        # 同一最新C的P0目标、P子块以及物理Pshot先验必须逐项一致。
        with np.load(POUT/'cases'/pbase.NAMES[i+1]/'likelihood.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(target[:12],z['target'],rtol=3e-13,atol=1e-7)
            np.testing.assert_allclose(covariance[:12,:12],z['covariance'],rtol=3e-13,atol=1e-7)
            np.testing.assert_array_equal(bounds[:2],z['bounds'][:2])
            np.testing.assert_allclose(bounds[11],z['bounds'][2],rtol=3e-15,atol=0.)
    source.update({str(q):digest(q) for q in paths})
    save_json(OUT/'PB_CONTRACT.json',dict(status='LATEST_COMPLETED_FID_C_SINGLE_PROVISIONAL_PB_FROZEN',
        immutable_inputs=source,n_covariance=ns,covariance_ids=ids,covariance_snapshot_UTC=pc['covariance_snapshot_UTC'],
        diagnostic_only=True,provisional=True,p_fixed_physical=p,p_uncertainty_propagated=False,
        covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,P_B_cross_covariance=True,
        target_count_per_truth=100,P_bins=12,B_bins=9,kmax_P=.08,kmax_B=.08,IR=False,
        ensemble_CPUs=list(CPUS),login_task_CPU_union=list(range(7,15)),max_steps=64000,
        original_moves_RNG_diagnostics_preserved=True,full_STO_noise_closure_certified=False,
        formal_fNL_or_kmax_released=False,old205_chain_preserved=True,old8_parameter_model_preserved=True,
        original700_and_formal_C500_workflow_replaced=False,covariance_records=records))


def case_directory(index):return BOUT/(COSMOS[index]+'_post_PB_kB0.08_kP0.08_provisional_EFT12_latestcov')


def worker(index,ensemble,steps):
    if os.sched_getaffinity(0)!={CPUS[ensemble]}:raise ValueError('single declared ensemble CPU required')
    case=LatestPBCase(case_directory(index));case.initialize()
    path=case.directory/f'chain_ensemble_{ensemble}.h5'
    def progress(actual):save_json(case.directory/f'PROGRESS_ensemble_{ensemble}.json',dict(
        pid=os.getpid(),CPU_affinity=[CPUS[ensemble]],actual_writer_saved_steps=actual,
        requested_checkpoint_steps=steps,updated_UTC=stamp(),writer_owned=True))
    advance_ensemble(case,path,ensemble,index,steps,progress)


def parallel_case(index,update):
    case=LatestPBCase(case_directory(index));mapmeta=case.initialize();start=time.monotonic()
    for steps in range(6000,64001,2000):
        guard();active=[]
        for ensemble,cpu in enumerate(CPUS):
            log=case.directory/f'worker_{ensemble}_to{steps}.log';stream=log.open('x')
            command=['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                '--worker',str(index),'--ensemble',str(ensemble),'--steps',str(steps)]
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active.append((process,stream));update(case_index=index,case=case.meta['name'],
                checkpoint_requested=steps,worker_pids=[p.pid for p,_ in active])
        try:
            while any(p.poll() is None for p,_ in active):time.sleep(.5)
            failed=[p.returncode for p,_ in active if p.returncode!=0]
            if failed:raise ValueError('independent ensemble failed; preserve all files: '+str(failed))
        finally:
            for p,stream in active:
                if p.poll() is None:p.send_signal(signal.SIGINT);p.wait(timeout=30)
                stream.close()
        # 所有writer退出后才能读取HDF；运行中进度只取writer-owned JSON。
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}') for j in range(4)]
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape[1]!=steps:raise ValueError('independent chains not at same original checkpoint')
        report,flat=shared.diagnose(case,chains,accepted,steps)
        report.update(name=case.meta['name'],cosmology=case.meta['cosmology'],truth_fNL=case.meta['truth_fNL'],
            parameters=case.names,chi2_MAP=mapmeta['chi2_MAP'],dof=9,gof_pvalue=None,
            gof_note='mean100 weighted by latest C_single; not calibrated GOF or complete-noise certification',
            elapsed_seconds=time.monotonic()-start,likelihood_sha256=case.meta['likelihood_sha256'],
            diagnostic_only=True,provisional=True,n_covariance=case.meta['n_covariance'],n_target=100,
            covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,P_B_cross_covariance=True,
            p_fixed_physical=case.meta['p_fixed_physical'],p_uncertainty_propagated=False,
            IR=False,kmax_B=.08,kmax_P=.08,P_likelihood=True,B_likelihood=True,
            full_STO_noise_closure_certified=False,formal_fNL_or_kmax_released=False)
        save_json(case.directory/'SUMMARY.json',shared.clean(report))
        print(case.meta['name'],json.dumps(shared.clean(dict(steps=steps,converged=report['converged'],gates=report['gates']))),flush=True)
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        if report['converged']:break
    rng=np.random.default_rng(5500+index);pick=rng.choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=np.array(case.names))
    return shared.clean(report)


def pipeline():
    if OUT.exists():raise ValueError('preserve existing latest-cohort run; no duplicate refreeze')
    if Path('/proc/1046481').exists():raise ValueError('user-superseded old205 writer must actually exit')
    original.authority()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_independent_ensemble_backend','test_hdf_lock_retry','test_provisional_prior_safe_initialization']))
    if not tests.wasSuccessful() or tests.testsRun!=6:raise ValueError('six native original-kernel/IO/prior-init tests required')
    OUT.mkdir();sources=runtime_sources()
    save_json(OUT/'USER_RESTART.json',dict(direct_human_request='重新跑吧，用最新的realization数目',
        created_UTC=stamp(),old205_writer_confirmed_terminal=True,old205_directory_unchanged=True,
        update_covariance_snapshot=True,update_physical_p_and_P0_comparison=True,
        target_mean_count_per_truth_unchanged=100,C_single=True,volume_Gpc_h3=1.,runtime_sources=sources))
    state=dict(state='PREPARING_LATEST_C_SINGLE',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[12],
        started_UTC=stamp(),completed_p_cases=[],completed_pb_cases=[])
    def update(**v):state.update(v,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update();old_p=(pbase.OUT,pbase.save_json);old_open=emcee.backends.HDFBackend.open
    def opened(self,mode='r'):return open_with_lock_retry(lambda:old_open(self,mode))
    emcee.backends.HDFBackend.open=opened
    def p_save(path,value):
        if Path(path)==POUT/'CONTRACT.json':
            value=dict(value);value['immutable_inputs'].update(sources);value['CPU_affinity']=[12]
            value['new_cohort_requested_by_human']=True
        old_p[1](path,value)
    pbase.OUT=POUT;pbase.save_json=p_save
    try:
        pbase.freeze();pc=pbase.parent();update(state='LATEST_PHYSICAL_P_MCMC',n_covariance=pc['n_covariance'],
            covariance_snapshot_UTC=pc['covariance_snapshot_UTC'],covariance_ids=pc['covariance_ids'])
        pbase.sample(0);state['completed_p_cases'].append(pbase.NAMES[0]);pbase.freeze_P_cases()
        for index in range(1,4):
            update(state='LATEST_P0_COMPARISON_MCMC',case=pbase.NAMES[index]);pbase.sample(index)
            state['completed_p_cases'].append(pbase.NAMES[index])
        update(state='FREEZING_LATEST_PB_SAME_COHORT');freeze_pb()
        update(state='LATEST_PB_PARALLEL_ENSEMBLES',p_fixed_physical=guard()['p_fixed_physical'])
        reports=[]
        for index in range(3):
            reports.append(parallel_case(index,update));state['completed_pb_cases'].append(COSMOS[index]);update()
        save_json(OUT/'RESULTS.json',dict(status='THREE_LATEST_COV_PROVISIONAL_PB_TRIALS_FINISHED_REVIEW_GATES',
            reports=reports,p0_reports=[read(POUT/'cases'/name/'SUMMARY.json') for name in pbase.NAMES[1:]],
            physical_p_calibration=read(POUT/'cases'/pbase.CAL_NAME/'SUMMARY.json'),n_covariance=pc['n_covariance'],
            covariance_ids=pc['covariance_ids'],C_single=True,volume_Gpc_h3=1.,diagnostic_only=True,
            all_pb_cases_converged=all(s['converged'] for s in reports),full_STO_noise_closure_certified=False))
        update(state='LATEST_COV_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE',finished_UTC=stamp())
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED_REQUIRES_EXACT_INSPECTION',error=traceback.format_exc());raise
    finally:pbase.OUT,pbase.save_json=old_p;emcee.backends.HDFBackend.open=old_open


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--ensemble',type=int);ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    if socket.gethostname()!='login26':raise ValueError('same login26 CPU lease required')
    if args.worker is not None:worker(args.worker,args.ensemble,args.steps);return
    if os.sched_getaffinity(0)!={12}:raise ValueError('controller and P calibration on CPU12 only')
    if args.detach:
        log=ROOT/'diagnostic_latest_cov_pb_20261010_v1.log';launch=ROOT/'diagnostic_latest_cov_pb_20261010_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect existing latest restart before relaunch')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=process.pid,host=socket.gethostname(),CPU_affinity=[12],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
