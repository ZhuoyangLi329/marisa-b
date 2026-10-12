"""模块大纲：按用户要求旧real-space三噪声，六组pre/post RSD P+B重新MCMC。

保持post C402/mean100/p及pre原C500/mean100/p；同9B+12P、相同物理先验，
分别保留各自完整P-B cross，明确pre/post的不同covariance IDs与tracer p。
旧12/8参数结果只读保留，700继续；本分支是用户指定的legacy闭合诊断。
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
from rsd_b000_contract import PART,digest,save_json,F_GROWTH,selected,reference
import run_rsd_b000_mcmc as shared
import run_sto_post_latest_cov_pb_v1 as previous
from post_rsd_legacy3_model import LegacyThreeNoiseModel,NAMES
from provisional_prior_safe_initialization import bounded_affine_transform,physical_initial_cloud
from ensemble_progress_json_safe import advance_json_safe

ROOT=previous.ROOT
OUT=ROOT/'diagnostic_three_noise_prepost_k008_v1'
PRE=PART/'outputs/pre_rsd_pb_mean100_v1'
CPUS=(7,12,13,14)
SPECS=[(stage,truth) for truth in previous.COSMOS for stage in ('post','pre')]


def read(path):return json.loads(Path(path).read_text())
def case_directory(index):
    stage,truth=SPECS[index]
    return OUT/'cases'/(truth+'_'+stage+'_PB_legacy3_k008')


def guard():
    contract=read(OUT/'CONTRACT.json')
    for path,want in contract['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('three-noise frozen source/input drift: '+path)
    return contract


class ThreeNoiseCase(shared.Case):
    def __init__(self,directory):
        guard();self.directory=Path(directory);self.meta=read(self.directory/'contract.json')
        if self.meta['parameters']!=list(NAMES) or digest(self.directory/'likelihood.npz')!=self.meta['likelihood_sha256']:
            raise ValueError('explicit frozen seven parameters required')
        with np.load(self.directory/'likelihood.npz',allow_pickle=False) as source:
            for name in source.files:setattr(self,name,np.array(source[name]))
        self.names=list(NAMES);self.ndim=7
        self.model=LegacyThreeNoiseModel(self.coefficients,self.templates,self.primitives,
            self.weight_moments,self.meta['p_fixed_physical'],F_GROWTH)

    def residual(self,physical):return self.model.residual(physical,self.target,self.chol)

    def initialize(self):
        meta=super().initialize()
        if not meta.get('prior_safe_initialization'):
            save_json(self.directory/'MAP_HESSIAN_PROPOSAL_ORIGINAL.json',meta)
            self.transform,conditioning=bounded_affine_transform(np.array(meta['transform']),self.bounds)
            meta=dict(meta,transform=self.transform.tolist(),prior_safe_initialization=True,affine_conditioning=conditioning)
            save_json(self.directory/'MAP.json',meta)
        return meta

    def initial(self,seed,ensemble,mapmeta):
        best=np.array(mapmeta['center'])
        good=[row for row in mapmeta['candidates'] if row['valid'] and row['chi2']<=mapmeta['chi2_MAP']+10]
        if ensemble>=2 and good:best=np.array(good[(ensemble-2)%len(good)]['parameters'])
        values=physical_initial_cloud(best,self.transform,self.bounds,seed)
        points=self.coordinates(values)
        if not np.isfinite(self.log_prob(points)).all():raise ValueError('new seven-param walkers outside prior')
        return points


def freeze():
    if OUT.exists():raise ValueError('preserve previous three-noise run, never refreeze')
    stop=read(ROOT/'user_three_noise_redirect_20261010_v1/STOP_RECEIPT.json')
    if stop['status']!='USER_SUPERSEDED_EFT12_AND_AUTO128K_ACTUALLY_TERMINATED_PRODUCTS_PRESERVED':
        raise ValueError('explicit human direction change and actual old process stop required')
    for pid in [1209663,1414034]:
        if Path('/proc/'+str(pid)).exists():raise ValueError('superseded old owner still exists')
    previous.guard()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_post_rsd_legacy3_model','test_ensemble_progress_json_safe','test_independent_ensemble_backend',
        'test_hdf_lock_retry','test_provisional_prior_safe_initialization']))
    if not tests.wasSuccessful() or tests.testsRun!=11:raise ValueError('eleven actual legacy/moves/JSON/lock/init tests required')
    OUT.mkdir()
    paths=[ROOT/'user_three_noise_redirect_20261010_v1/USER_REQUEST.json',
        ROOT/'user_three_noise_redirect_20261010_v1/STOP_RECEIPT.json',previous.OUT/'PB_CONTRACT.json',
        PART.parent/'03_bk_model/envclass_b/code/run_joint_pb_matched250.py',
        PART.parent/'03_bk_model/marisa_b_core/scripts/production/run_pre_recon_halo_png_uplift_v1.py',
        PART.parent/'03_bk_model/marisa_b_core/src/marisa_b/marisa_b_native.cpp',
        PRE/'contract.json',PRE/'P0_LATTICE.npz',PRE/'P0_ENSEMBLE_MATRIX.npz']
    for name in ['run_sto_three_noise_prepost_v1.py','post_rsd_legacy3_model.py','test_post_rsd_legacy3_model.py',
                 'run_rsd_b000_mcmc.py','provisional_prior_safe_initialization.py',
                 'independent_ensemble_backend.py','ensemble_progress_json_safe.py','hdf_lock_retry.py',
                 'post_rsd_noir_polynomial.py','post_rsd_noir_likelihood.py','post_rsd_p0_model.py',
                 'rsd_pb_common.py','rsd_bonly_common.py','rsd_b000_contract.py','post_rsd_fixedrec.py',
                 'tree_png_rsd_native.py','test_ensemble_progress_json_safe.py',
                 'test_independent_ensemble_backend.py','test_hdf_lock_retry.py',
                 'test_provisional_prior_safe_initialization.py']:
        paths.append(PART/'code'/name)
    paths.append(PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py')
    indices=selected(.08)
    with np.load(PRE/'P0_LATTICE.npz',allow_pickle=False) as source:pre_templates=source['templates'].copy()
    with np.load(PRE/'P0_ENSEMBLE_MATRIX.npz',allow_pickle=False) as source:
        metadata=[json.loads(str(value)) for value in source['metadata_json']]
    for index,(stage,truth) in enumerate(SPECS):
        t=previous.COSMOS.index(truth)
        if stage=='post':
            source_directory=previous.case_directory(t)
            with np.load(source_directory/'likelihood.npz',allow_pickle=False) as source:
                arrays={name:np.array(source[name]) for name in source.files}
            source_meta=read(source_directory/'contract.json')
            coefficients=arrays['coefficients'];primitives=arrays['intrinsic'];templates=arrays['templates']
            moments=arrays['weight_moments'][:2];physical_p=source_meta['p_fixed_physical']
        else:
            source_directory=PRE/'cases'/(truth+'_mean100_k0.08')
            source_meta=read(source_directory/'contract.json')
            with np.load(source_directory/'likelihood.npz',allow_pickle=False) as source:
                arrays={name:np.array(source[name]) for name in source.files}
            np.testing.assert_array_equal(arrays['indices'],indices)
            coefficients=np.zeros((5,3,3,9));coefficients[:4]=arrays['coefficients'][...,12:]
            with np.load(previous.case_directory(t)/'likelihood.npz',allow_pickle=False) as source:
                primitives=source['intrinsic'].copy()
            templates=pre_templates
            nbars=np.array([row['nbar_h3_Mpc3'] for row in metadata[t*100:(t+1)*100]])
            moments=np.array([np.mean(1/nbars),np.mean(1/nbars**2)])
            physical_p=read(PRE/'contract.json')['p_fixed']
            np.testing.assert_allclose(moments[0],source_meta['Pshot_unit_Mpc_h3'],rtol=3e-15)
        n_cov=source_meta['n_covariance'];alpha=(n_cov-23)/(n_cov-1)
        np.testing.assert_allclose(arrays['chol']@arrays['chol'].T,arrays['covariance']/alpha,rtol=3e-13,atol=1e-7)
        bounds=np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-20,20],[-10,10],[-1e6/moments[0],1e6/moments[0]]])
        np.testing.assert_array_equal(bounds[:4],arrays['bounds'][:4])
        if arrays['target'].shape!=(21,):raise ValueError('original21-point data required')
        model=LegacyThreeNoiseModel(coefficients,templates,primitives,moments,physical_p,F_GROWTH)
        probes=np.array([[0,1.93,-.25,-.3,0,0,0],[100,2.23,.4,-.4,.3,-.2,1],[-100,1.75,-1,.2,-.3,.5,-1]])
        if not np.isfinite(model.predictions(probes)).all():raise ValueError('nonfinite minimal model')
        directory=case_directory(index);directory.mkdir(parents=True)
        np.savez_compressed(directory/'likelihood.npz',target=arrays['target'],covariance=arrays['covariance'],
            chol=arrays['chol'],coefficients=coefficients,templates=templates,primitives=primitives,
            weight_moments=moments,bounds=bounds,indices=indices)
        paths.extend([source_directory/'contract.json',source_directory/'likelihood.npz',directory/'likelihood.npz'])
        meta=dict(name=directory.name,stage=stage,cosmology=truth,truth_fNL=(0,100,-100)[t],
            parameters=list(NAMES),bounds=bounds.tolist(),n_covariance=n_cov,n_target=100,P_bins=12,B_bins=9,
            volume_Gpc_h3=1.,covariance_is_of_mean=False,covariance_multiplier=1.,covariance_ddof=1,
            P_likelihood=True,B_likelihood=True,P_B_cross_covariance=True,Hartlap_factor=alpha,
            kmax_P=.08,kmax_B=.08,IR=False,p_fixed_physical=physical_p,p_uncertainty_propagated=False,
            source_case=str(source_directory),source_likelihood_sha256=digest(source_directory/'likelihood.npz'),
            likelihood_sha256=digest(directory/'likelihood.npz'),
            covariance_IDs=source_meta.get('covariance_ids',source_meta.get('covariance_realizations')),
            target_IDs=list(range(100)),effective_Pshot_physical_bounds=[-1e6,1e6],
            legacy_noise_closure='A*N3*valid + B*N2*(Z_G^2*P+fNL*bphi*Z_G*P/M+fNL^2*bphi^2*P/M^2); independent Pshot',
            closure_is_explicit_user_requested_phenomenological_assumption=True,
            extra_R_anisotropic_raw_Phi_w2_J21_shift_parameters=False,
            prior_policy='retain current RSD native core bounds and residual amplitude widths; A=old S, B=old A; Pshot physical +/-1e6',
            old_realspace_gaussian_nuisance_prior_not_imported=True,
            diagnostic_only=True,provisional=True,full_STO_noise_closure_certified=False,
            formal_fNL_or_kmax_released=False)
        save_json(directory/'contract.json',meta);paths.append(directory/'contract.json')
    paths=list(dict.fromkeys(paths))
    save_json(OUT/'CONTRACT.json',dict(status='USER_REQUESTED_LEGACY_THREE_NOISE_SIX_PREPOST_CASES_FROZEN',
        created_UTC=previous.stamp(),immutable_inputs={str(path):digest(path) for path in paths},
        case_names=[case_directory(i).name for i in range(6)],parameters=list(NAMES),noise_parameter_count=3,
        total_parameter_count=7,CPUs=list(CPUS),maximum_login_CPU_union=list(range(7,15)),
        ensembles=4,walkers=32,max_steps=64000,steps_increment=2000,
        target_mean_count=100,C_single=True,volume_Gpc_h3=1.,post_covariance_count=402,pre_covariance_count=500,
        pre_and_post_covariance_ids_not_same=True,p_each_tracer_fixed_from_existing_calibration=True,
        IR=False,original_measurements_and12_parameter_products_preserved=True,
        original700_measurements_continue=True,full_STO_noise_closure_certified=False,
        no_automatic_parameter_expansion=True,diagnostic_only=True))


def worker(index,ensemble,steps):
    if os.sched_getaffinity(0)!={CPUS[ensemble]}:raise ValueError('one declared CPU per writer')
    case=ThreeNoiseCase(case_directory(index));case.initialize()
    def progress(actual):save_json(case.directory/f'PROGRESS_ensemble_{ensemble}.json',dict(
        pid=os.getpid(),actual_writer_saved_steps=int(actual),requested_checkpoint_steps=steps,
        updated_UTC=previous.stamp(),CPU_affinity=[CPUS[ensemble]],writer_owned=True))
    advance_json_safe(case,case.directory/f'chain_ensemble_{ensemble}.h5',ensemble,index,steps,progress)


def sample(index,update):
    case=ThreeNoiseCase(case_directory(index));mapmeta=case.initialize();start=time.monotonic()
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
            if any(p.returncode!=0 for p,_ in active):raise ValueError('actual seven-param writer failed; preserve files')
        finally:
            for process,stream in active:
                if process.poll() is None:process.send_signal(signal.SIGINT);process.wait(timeout=30)
                stream.close()
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}') for j in range(4)]
        chains=np.stack([backend.get_chain() for backend in backends]);accepted=np.stack([backend.accepted for backend in backends])
        if chains.shape!=(4,steps,32,7):raise ValueError('actual equal four32walker seven-param chains required')
        report,flat=shared.diagnose(case,chains,accepted,steps)
        report.update(name=case.meta['name'],stage=case.meta['stage'],cosmology=case.meta['cosmology'],
            truth_fNL=case.meta['truth_fNL'],parameters=case.names,n_covariance=case.meta['n_covariance'],
            n_target=100,P_bins=12,B_bins=9,kmax_P=.08,kmax_B=.08,p_fixed_physical=case.meta['p_fixed_physical'],
            chi2_MAP=mapmeta['chi2_MAP'],minuit_valid=mapmeta['minuit_valid'],elapsed_seconds=time.monotonic()-start,
            likelihood_sha256=case.meta['likelihood_sha256'],C_single=True,volume_Gpc_h3=1.,
            covariance_is_of_mean=False,P_B_cross_covariance=True,noise_parameter_count=3,IR=False,
            gof_pvalue=None,gof_note='mean100 data weighted by C_single; not calibrated GOF',
            diagnostic_only=True,provisional=True,full_STO_noise_closure_certified=False)
        save_json(case.directory/'SUMMARY.json',shared.clean(report))
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        print(case.meta['name'],steps,report['converged'],report['gates'],flush=True)
        if report['converged']:break
    pick=np.random.default_rng(7500+index).choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=np.array(case.names))
    return shared.clean(report)


def pipeline():
    freeze();state=dict(state='SAMPLING_SIX_LEGACY3_PREPOST_CASES',pid=os.getpid(),host=socket.gethostname(),
        started_UTC=previous.stamp(),CPU_affinity=[12],completed_cases=[])
    def update(**values):state.update(values,updated_UTC=previous.stamp());save_json(OUT/'STATUS.json',state)
    update()
    try:
        reports=[]
        for index in range(6):
            reports.append(sample(index,update));state['completed_cases'].append(case_directory(index).name);update()
        save_json(OUT/'RESULTS.json',dict(status='SIX_USER_REQUESTED_THREE_NOISE_PREPOST_TRIALS_FINISHED_REVIEW_GATES',
            reports=reports,all_six_converged=all(row['converged'] for row in reports),
            noise_parameter_count=3,total_parameter_count=7,diagnostic_only=True,
            pre_covariance_count=500,post_covariance_count=402,full_STO_noise_closure_certified=False))
        update(state='SIX_LEGACY3_PREPOST_TRIALS_FINISHED_REVIEW_CONVERGENCE',finished_UTC=previous.stamp())
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED_REQUIRES_EXACT_INSPECTION',error=traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--detach',action='store_true')
    parser.add_argument('--worker',type=int);parser.add_argument('--ensemble',type=int);parser.add_argument('--steps',type=int)
    args=parser.parse_args()
    if socket.gethostname()!='login26':raise ValueError('original login26 CPU lease required')
    if args.worker is not None:worker(args.worker,args.ensemble,args.steps);return
    if os.sched_getaffinity(0)!={12}:raise ValueError('controller on original CPU12 only')
    if args.detach:
        log=ROOT/'diagnostic_three_noise_prepost_k008_v1.log';launch=ROOT/'diagnostic_three_noise_prepost_k008_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect exact prior run; no duplicate restart')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=process.pid,host=socket.gethostname(),CPU_affinity=[12],created_UTC=previous.stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
