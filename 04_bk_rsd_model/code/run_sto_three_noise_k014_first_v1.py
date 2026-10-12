"""模块大纲：用户要求先看kB=.14；同一4+3模型，优先fiducial post/pre。

只增加B分箱，复用0.08的模型/先验/p/权重矩和原MCMC数值实现。
post仍冻结同402份实际测量，pre仍原500；P固定.08，不加入IR。
科学设置是本次记录而非永久政策；授权修改另开新结果，不混入旧链。
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
import unittest
import numpy as np
import emcee
import run_sto_three_noise_prepost_v1 as base
from rsd_b000_contract import PART,digest,save_json,F_GROWTH,selected
from post_rsd_legacy3_model import LegacyThreeNoiseModel
from post_rsd_noir_polynomial import collapse_physical_p
from post_rsd_owner_quiescence import owner_quiescent
from run_sto_post_compute_queue import directory as measurement_directory
from legacy3_scan_contract import configuration_snapshot
from audit_sto_three_noise_prepost_v1 import independent_prediction

ROOT=base.ROOT
OUT=ROOT/'diagnostic_three_noise_prepost_k014_first_v1'
SPECS=base.SPECS
CPUS=base.CPUS
CUT=.14


def read(path):return json.loads(Path(path).read_text())
def case_directory(index):
    stage,truth=SPECS[index]
    return OUT/'cases'/(truth+'_'+stage+'_PB_legacy3_k014')


def guard():
    settings=read(OUT/'RUN_SETTINGS.json')
    for path,want in settings['immutable_inputs'].items():
        if digest(path)!=want:
            raise ValueError('running likelihood/source changed; use a new version rather than mix chains: '+path)
    return settings


class EndpointCase(base.ThreeNoiseCase):
    def __init__(self,directory):
        guard();self.directory=Path(directory);self.meta=read(self.directory/'contract.json')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as z:
            arrays={name:np.array(z[name]) for name in z.files}
        for name,value in arrays.items():setattr(self,name,value)
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        configuration_snapshot(self.meta,arrays)
        self.model=LegacyThreeNoiseModel(self.coefficients,self.templates,self.primitives,
            self.weight_moments,self.meta['p_fixed_physical'],F_GROWTH)


def freeze():
    if OUT.exists():raise ValueError('existing endpoint run must be inspected; no duplicate/refreeze')
    base.guard();parent=base.previous.guard()
    if not owner_quiescent(read(base.OUT/'STATUS.json')):
        raise ValueError('previous .08 owner must finish before endpoint run')
    if not read(base.OUT/'MCMC_COMPLETION_AUDIT.json')['all_six_converged']:
        raise ValueError('actual .08 results required as unchanged baseline')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_post_rsd_legacy3_model','test_ensemble_progress_json_safe','test_independent_ensemble_backend',
        'test_hdf_lock_retry','test_provisional_prior_safe_initialization','test_legacy3_scan_contract']))
    if not tests.wasSuccessful():raise ValueError('native unchanged-model/sampler tests failed')
    target_path=ROOT/'target300_completion_v1/TARGETS_ONLY.npz'
    coefficient_path=ROOT/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz'
    primitive_path=ROOT/'noise_eft_primitives_v3/PRIMITIVES.npz'
    with np.load(target_path,allow_pickle=False) as z:
        indices=z['B_indices'].copy();targets=z['matched_rows'].copy()
    chosen=selected(CUT);local=np.array([np.flatnonzero(indices==j)[0] for j in chosen])
    columns=np.r_[np.arange(12,24),24+local]
    records={r['realization']:r['measurement_sha256'] for r in parent['covariance_records']}
    rows=[]
    # 不重测，不读取未用于本次似然的halo payload；核验每个实际测量文件的原SHA。
    for position,real in enumerate(parent['covariance_ids']):
        d=measurement_directory(('fiducial',real));p=d/'MEASUREMENTS.npz'
        done=read(d/'DONE.json')
        if done['measurement_sha256']!=records[real] or digest(p)!=records[real]:
            raise ValueError('original frozen402 measurement changed: '+str(real))
        with np.load(p,allow_pickle=False) as z:
            rows.append(np.r_[z['P_weighted_input'],z['P_post'],z['B_post'][indices]])
        if (position+1)%100==0:print('original402 full-PB rows',position+1,flush=True)
    full=np.array(rows);fullcov=np.cov(full,rowvar=False,ddof=1)
    with np.load(coefficient_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices)
        coefficients=collapse_physical_p(z['coefficients'],p_halo=parent['p_fixed_physical'])
    with np.load(primitive_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices);primitives=z['intrinsic'].copy()
    OUT.mkdir()
    fullpath=OUT/'ORIGINAL402_FULL_PB_SNAPSHOT.npz'
    np.savez_compressed(fullpath,rows=full,covariance_single_box=fullcov,
        covariance_ids=parent['covariance_ids'],B_indices=indices)
    paths=[Path(__file__),PART/'code/legacy3_scan_contract.py',PART/'code/test_legacy3_scan_contract.py',
        base.OUT/'CONTRACT.json',base.OUT/'MCMC_COMPLETION_AUDIT.json',fullpath,
        base.previous.OUT/'PB_CONTRACT.json',target_path,coefficient_path,primitive_path]
    paths.extend(Path(path) for path in base.guard()['immutable_inputs'])
    paths.extend(Path(path) for path in parent['immutable_inputs'])
    for index,(stage,truth) in enumerate(SPECS):
        baseline_dir=base.case_directory(index);meta=read(baseline_dir/'contract.json')
        with np.load(baseline_dir/'likelihood.npz',allow_pickle=False) as z:
            baseline={key:np.array(z[key]) for key in z.files}
        if stage=='post':
            target=targets[(index//2)*100:(index//2+1)*100][:,columns].mean(axis=0)
            covariance=fullcov[np.ix_(columns,columns)]
            tree=coefficients[...,local]
            old_local=np.array([np.flatnonzero(indices==j)[0] for j in baseline['indices']])
            old_columns=np.r_[np.arange(12,24),24+old_local]
            np.testing.assert_array_equal(fullcov[np.ix_(old_columns,old_columns)],baseline['covariance'])
            np.testing.assert_array_equal(coefficients[...,old_local],baseline['coefficients'])
            source_case=str(base.previous.case_directory(index//2))
            source_sha=meta['source_likelihood_sha256']
        else:
            source=base.PRE/'cases'/(truth+'_mean100_k0.14')
            source_meta=read(source/'contract.json')
            with np.load(source/'likelihood.npz',allow_pickle=False) as z:
                target=z['target'].copy();covariance=z['covariance'].copy()
                np.testing.assert_array_equal(z['indices'],chosen)
                tree=np.zeros((5,3,3,len(chosen)));tree[:4]=z['coefficients'][...,12:]
            source_case=str(source);source_sha=digest(source/'likelihood.npz')
            paths.extend([source/'contract.json',source/'likelihood.npz'])
        n_cov=meta['n_covariance'];ndata=len(target);alpha=(n_cov-ndata-2)/(n_cov-1)
        if alpha<=0:raise ValueError('non-positive actual Hartlap factor')
        arrays=dict(target=target,covariance=covariance,chol=np.linalg.cholesky(covariance/alpha),
            coefficients=tree,templates=baseline['templates'],primitives=primitives[:,local],
            weight_moments=baseline['weight_moments'],bounds=baseline['bounds'],indices=chosen)
        old_positions=np.r_[np.arange(12),12+np.array([np.flatnonzero(chosen==j)[0] for j in baseline['indices']])]
        np.testing.assert_array_equal(target[old_positions],baseline['target'])
        np.testing.assert_array_equal(covariance[np.ix_(old_positions,old_positions)],baseline['covariance'])
        np.testing.assert_array_equal(primitives[:,old_local if stage=='post' else np.array([
            np.flatnonzero(indices==j)[0] for j in baseline['indices']])],baseline['primitives'])
        d=case_directory(index);d.mkdir(parents=True)
        np.savez_compressed(d/'likelihood.npz',**arrays)
        meta.update(name=d.name,kmax_B=CUT,B_bins=len(chosen),Hartlap_factor=alpha,
            likelihood_sha256=digest(d/'likelihood.npz'),source_case=source_case,source_likelihood_sha256=source_sha,
            same_model_prior_p_and_weight_moments_as_k008=True,k008_reference_case=str(baseline_dir),
            seed_case_index=6+index)
        meta['bounds']=arrays['bounds'].tolist()
        configuration_snapshot(meta,arrays)
        save_json(d/'contract.json',meta)
        paths.extend([baseline_dir/'contract.json',baseline_dir/'likelihood.npz',d/'likelihood.npz',d/'contract.json'])
    paths=list(dict.fromkeys(paths))
    save_json(OUT/'RUN_SETTINGS.json',dict(created_UTC=base.previous.stamp(),
        direct_user_request='你直接先算一个kmax0.14的吧。我先看看。快点',
        case_names=[case_directory(i).name for i in range(6)],kmax_B=CUT,kmax_P=.08,
        parameters=read(case_directory(0)/'contract.json')['parameters'],noise_parameter_count=3,
        ensembles=4,walkers_each=32,CPUs=list(CPUS),max_steps=64000,steps_increment=2000,
        post_covariance_count=402,pre_covariance_count=500,volume_Gpc_h3=1.,C_single=True,
        covariance_is_of_mean=False,IR=False,settings_are_not_permanent_policy=True,
        immutable_inputs={str(path):digest(path) for path in paths},native_tests_passed=tests.testsRun,
        broader_k010_k012_still_required=True,full_STO_noise_closure_certified=False))
    audit_inputs()


def audit_inputs():
    guard();checks=[]
    for index in range(6):
        case=EndpointCase(case_directory(index))
        rng=np.random.default_rng(84100+index)
        probes=rng.uniform(case.bounds[:,0],case.bounds[:,1],(128,case.ndim))
        probes=np.vstack((probes,case.bounds[:,0],case.bounds[:,1],case.bounds.mean(axis=1)))
        direct=np.stack([independent_prediction(row,case) for row in probes])
        actual=case.model.predictions(probes)
        white=np.linalg.solve(case.chol,(actual-direct).T).T
        scale=np.maximum(1,np.max(abs(np.linalg.solve(case.chol,direct.T).T),axis=1))
        error=float(np.max(abs(white)/scale[:,None]))
        if error>1e-9:raise ValueError('actual expanded-bin independent prediction error')
        checks.append(dict(name=case.meta['name'],n_data=case.model.n,probes=len(probes),
            maximum_independent_white_scaled_error=error,
            settings=configuration_snapshot(case.meta,case.__dict__)))
    save_json(OUT/'INPUT_CHECKS.json',dict(created_UTC=base.previous.stamp(),checks=checks,
        actual_case_count=6,old_k008_data_C_and_signal_subblocks_bitwise_preserved=True,
        original_model_and_priors_reused=True,physical_STO_noise_closure_certified=False,HDF_read=False))


def worker(index,ensemble,steps):
    if os.sched_getaffinity(0)!={CPUS[ensemble]}:raise ValueError('single original CPU per ensemble required')
    case=EndpointCase(case_directory(index));case.initialize()
    def progress(actual):save_json(case.directory/f'PROGRESS_ensemble_{ensemble}.json',dict(
        pid=os.getpid(),actual_writer_saved_steps=int(actual),requested_checkpoint_steps=steps,
        CPU_affinity=[CPUS[ensemble]],updated_UTC=base.previous.stamp(),writer_owned=True))
    base.advance_json_safe(case,case.directory/f'chain_ensemble_{ensemble}.h5',ensemble,6+index,steps,progress)


def sample(index,update):
    case=EndpointCase(case_directory(index));mapmeta=case.initialize();start=time.monotonic()
    for steps in range(6000,64001,2000):
        guard();active=[]
        try:
            for ensemble,cpu in enumerate(CPUS):
                stream=(case.directory/f'worker_{ensemble}_to{steps}.log').open('x')
                process=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    '--worker',str(index),'--ensemble',str(ensemble),'--steps',str(steps)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active.append((process,stream));update(case=case.meta['name'],case_index=index,
                    requested_checkpoint=steps,worker_pids=[p.pid for p,_ in active])
            while any(p.poll() is None for p,_ in active):time.sleep(.5)
            if any(p.returncode!=0 for p,_ in active):raise ValueError('endpoint ensemble writer failed; preserve artifacts')
        finally:
            for process,stream in active:
                if process.poll() is None:process.send_signal(signal.SIGINT);process.wait(timeout=30)
                stream.close()
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}') for j in range(4)]
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape!=(4,steps,32,case.ndim):raise ValueError('actual completed ensemble dimensions disagree')
        report,flat=base.shared.diagnose(case,chains,accepted,steps)
        report.update({key:case.meta[key] for key in ['name','stage','cosmology','truth_fNL','parameters',
            'n_covariance','n_target','P_bins','B_bins','kmax_P','kmax_B','p_fixed_physical']})
        report.update(chi2_MAP=mapmeta['chi2_MAP'],minuit_valid=mapmeta['minuit_valid'],
            elapsed_seconds=time.monotonic()-start,likelihood_sha256=case.meta['likelihood_sha256'],
            C_single=True,volume_Gpc_h3=1.,covariance_is_of_mean=False,P_B_cross_covariance=True,
            noise_parameter_count=3,IR=False,gof_pvalue=None,diagnostic_only=True,
            full_STO_noise_closure_certified=False)
        save_json(case.directory/'SUMMARY.json',base.shared.clean(report))
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        print(case.meta['name'],steps,report['converged'],report['gates'],flush=True)
        if report['converged']:break
    pick=np.random.default_rng(7506+index).choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=np.array(case.names))
    return base.shared.clean(report)


def pipeline():
    guard();state=dict(state='SAMPLING_K014_FIDUCIAL_FIRST',pid=os.getpid(),host=socket.gethostname(),
        started_UTC=base.previous.stamp(),CPU_affinity=[12],completed_cases=[])
    def update(**values):state.update(values,updated_UTC=base.previous.stamp());save_json(OUT/'STATUS.json',state)
    update()
    try:
        reports=[]
        for index in range(6):
            reports.append(sample(index,update));state['completed_cases'].append(case_directory(index).name);update()
        save_json(OUT/'RESULTS.json',dict(reports=reports,all_six_converged=all(r['converged'] for r in reports),
            kmax_B=CUT,kmax_P=.08,noise_parameter_count=3,total_parameter_count=7,
            diagnostic_only=True,broader_k010_k012_complete=False,full_STO_noise_closure_certified=False))
        update(state='SIX_K014_FITS_FINISHED_REVIEW_FULL_CHAINS',finished_UTC=base.previous.stamp())
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--detach',action='store_true');ap.add_argument('--worker',type=int)
    ap.add_argument('--ensemble',type=int);ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    if socket.gethostname()!='login26':raise ValueError('same login26 CPU pool required')
    if args.worker is not None:worker(args.worker,args.ensemble,args.steps);return
    if os.sched_getaffinity(0)!={12}:raise ValueError('one controller CPU12 required')
    if args.prepare:freeze();print('ACTUAL_K014_INPUTS_READY',flush=True);return
    if args.detach:
        guard();log=ROOT/'diagnostic_three_noise_prepost_k014_first_v1.log'
        launch=ROOT/'diagnostic_three_noise_prepost_k014_first_v1.launch.json'
        if log.exists() or launch.exists():raise ValueError('inspect existing owner; no duplicate launch')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=process.pid,host=socket.gethostname(),CPU_affinity=[12],
            created_UTC=base.previous.stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
