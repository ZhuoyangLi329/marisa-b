"""模块大纲：真实P+B端到端候选检查；原mean100、C205_single、同盒cross与p校准。

不改原500盒正式流程；固定fNL真值与50/50诊断检验噪声候选，而非用真值
初始化自由fNL MCMC。保存明确的preflight，不能冒称完整STO闭合或链结果。
11参数candidate的新增noise box在看结果前记录；core4与原先验一致。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
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
from scipy.optimize import least_squares
from rsd_b000_contract import PART,digest,save_json,F_GROWTH,reference,selected
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from post_rsd_noir_polynomial import collapse_physical_p
from post_rsd_eft_candidate_model_v1 import EFTCandidateModel,NAMES

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'noise_eft_pb_preflight_v1'
CAL=ROOT/'diagnostic_partial_cov_p_mcmc_20261010_v2'
PROJ=ROOT/'noise_eft_primitives_v3'
COSMOS=('fiducial','LC_p','LC_m')


def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def moments(rows):
    n2,n3,nhalo=np.asarray(rows).T;nr=1e9/(30*nhalo)
    return np.array([n2.mean(),n3.mean(),np.mean(n2*n2),nr.mean(),np.mean(n2*nr)])


def bounds_for(units):
    # 原Pshot residual +/-1e6保持同一物理上界；新branch只加入raw>=0必要条件。
    return np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-10,10],[-10,10],[-20,20],
        [0,1+1e6/units[0]],[-10,10],[-50,50],[-20,20]],dtype=float)


def make_model(arrays,units,p):
    return EFTCandidateModel(arrays['coefficients'],arrays['templates'],arrays['operators'],
        arrays['intrinsic'],arrays['matched_zero'],units,p,F_GROWTH)


def profile_fixed_truth(model,target,chol,truth,bounds):
    def rows(x):return np.r_[truth,x]
    def residual(x):return model.residual(rows(x),target,chol)[0]
    fits=[]
    for b in (1.25,1.93,2.8):
        start=np.array([b,-.25,-.3,0.,0.,0.,1.,b,2*1.686*(b-model.p_halo),1.])
        fit=least_squares(residual,start,bounds=(bounds[1:,0],bounds[1:,1]),
            x_scale=np.array([.2,1.,1.,1.,1.,1.,.5,1.,3.,1.]),
            ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2500)
        fits.append((float(fit.fun@fit.fun),fit))
    fits.sort(key=lambda a:a[0]);score,best=fits[0];physical=rows(best.x)
    singular=np.linalg.svd(best.jac,compute_uv=False)
    rank=int(np.sum(singular>max(singular[0],1.)*1e-10))
    prediction=model.predictions(physical)[0];r=model.residual(physical,target,chol)[0]
    return physical,dict(fNL_fixed_to_simulation_truth=True,truth_fNL=truth,parameters=list(NAMES),
        physical_parameters=physical.tolist(),chi2_single_box_weighted=score,
        residual_whitened=r.tolist(),prediction=prediction.tolist(),optimizer_success=bool(best.success),
        optimizer_message=str(best.message),jacobian_singular_values=singular.tolist(),local_effective_rank=rank,
        possible_exact_unidentified_at_fNL0=['a_phi','bphi_w2'] if truth==0 else [],
        starts=[dict(chi2=x[0],success=bool(x[1].success),nfev=int(x[1].nfev)) for x in fits],
        is_MCMC=False,is_full_noise_closure_gate=False)


def pipeline():
    if OUT.exists():raise ValueError('preserve prior preflight; never refreeze its cohort')
    proof=read(PROJ/'AUDIT.json');primitive=PROJ/'PRIMITIVES.npz'
    if (proof['status']!='ALL44_EFT_INTRINSIC_AND_MATCHED_ZERO_PRIMITIVES_PASS_NOT_STO_CLOSURE'
        or not proof.get('all_three_zero_legs') or proof['bins']!=44 or proof['maximum_scaled_error']>proof['original_scaled_tolerance']
        or digest(primitive)!=proof['matrix_sha256']):raise ValueError('actual all44/all3-zero candidate projection required')
    for p,want in read(PROJ/'CONTRACT.json')['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('candidate projection source drift')
    parent=read(CAL/'CONTRACT.json')
    for p,want in parent['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('original partial-p diagnostic frozen input drift')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_eft_candidate_model_v1'))
    if not tests.wasSuccessful() or tests.testsRun!=3:raise ValueError('actual direct-box mean/joint whitening tests required')
    cp=CAL/'cases/physical_p_GSR2_RSD_fNLfixed100_partial_cov/SUMMARY.json';cal=read(cp)
    if not cal['converged']:raise ValueError('original diagnostic physical p must converge')
    p=float(cal['quantiles_2p5_16_50_84_97p5'][2][0])
    target_path=ROOT/'target300_completion_v1/TARGETS_ONLY.npz'
    audit=read(ROOT/'target300_completion_v1/AUDIT.json')
    if digest(target_path)!=audit['target_matrix_sha256']:raise ValueError('original300 target matrix drift')
    with np.load(target_path,allow_pickle=False) as z:
        indices=z['B_indices'];all_rows=z['matched_rows'];counter_rows=z['nonzero_diagnostic_rows'];weights=z['data_N2_N3_nhalo']
        old_target=z['matched_target_vectors'];k=z['B_k']
    wanted=selected(.08);local=np.array([np.flatnonzero(indices==j)[0] for j in wanted])
    if len(local)!=9 or not np.all(np.max(k[local],axis=1)<=.08+1e-12):raise ValueError('original9 B bins at.08 required')
    columns=np.r_[np.arange(12,24),24+local];ids=parent['covariance_ids'];ns=len(ids)
    if ns!=205 or ns!=parent['n_covariance']:raise ValueError('same frozen205 cohort as P0 comparison required')
    full_cov_rows=[];records=[]
    for real in ids:
        data,meta,done=load_pilot(directory(('fiducial',real)))
        full_cov_rows.append(np.r_[data['P_weighted_input'],data['P_post'],data['B_post'][indices]])
        records.append(dict(realization=real,measurement_sha256=done['measurement_sha256']))
    full_cov_rows=np.array(full_cov_rows);cov_full=np.cov(full_cov_rows,rowvar=False,ddof=1)
    with np.load(CAL/'FROZEN_PARTIAL_P_ENSEMBLE.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(cov_full[:24,:24],z['covariance_single_box'],rtol=2e-13,atol=1e-7)
        np.testing.assert_allclose(old_target[:,:24],z['target_vectors'],rtol=2e-13,atol=1e-7)
        templates=z['templates'].copy()
    covariance=cov_full[np.ix_(columns,columns)];alpha=(ns-len(columns)-2)/(ns-1)
    chol=np.linalg.cholesky(covariance/alpha)
    dpath=ROOT/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz'
    da=read(ROOT/'noir_paware_templates/INDEPENDENT_COEFFICIENT_AUDIT.json')
    if (da['status']!='ALL44_NOIR_COEFFICIENT_NODE_AND_FINE_GRID_REPLAY_PASS'
        or digest(dpath)!=da['coefficients_sha256']):raise ValueError('deterministic all44 replay or source bytes differ')
    with np.load(dpath,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices);bc=collapse_physical_p(z['coefficients'],p_halo=p)[...,local]
    opath=ROOT/'noise_transport_templates_v1/OPERATOR_COEFFICIENTS.npz'
    oa=read(ROOT/'noise_transport_templates_v1/INDEPENDENT_COEFFICIENT_AUDIT.json')
    if digest(opath)!=oa['coefficients_sha256']:raise ValueError('original independently validated operator bytes differ')
    with np.load(opath,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices);c=z['coefficients']
        pc=np.array([1.,(p-1.3)/.5,2*((p-1.3)/.5)**2-1]);ops=np.tensordot(c,pc,axes=(1,0))[...,local]
    with np.load(primitive,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],indices);intrinsic=z['intrinsic'][:,local];zero=z['matched_zero'][:,local]
    arrays=dict(coefficients=bc,templates=templates,operators=ops,intrinsic=intrinsic,matched_zero=zero)
    OUT.mkdir();state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],started_UTC=stamp())
    def update(**v):state.update(v,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update(stage='freeze_exact_PB21_C_single')
    matrix=OUT/'FROZEN_PB205_PREFLIGHT.npz'
    np.savez_compressed(matrix,covariance_single_box=covariance,chol=chol,covariance_rows=full_cov_rows[:,columns],
        covariance_ids=ids,matched_target_rows=all_rows[:,columns],mark_moments=weights,
        original_B_indices=wanted,full68_columns=columns,**arrays)
    sources=[Path(__file__),PART/'code/post_rsd_eft_candidate_model_v1.py',PART/'code/test_post_rsd_eft_candidate_model_v1.py',
        PART/'code/post_rsd_stochastic_eft_v1.py',PART/'code/post_rsd_noir_polynomial.py',PART/'code/post_rsd_noir_likelihood.py',
        matrix,primitive,PROJ/'AUDIT.json',PROJ/'CONTRACT.json',cp,CAL/'CONTRACT.json',target_path,dpath,opath]
    save_json(OUT/'CONTRACT.json',dict(status='FROZEN_PRELIMINARY_EFT_PB_PREFLIGHT_NOT_MCMC_RELEASE',
        immutable_inputs={str(q):digest(q) for q in sources},p_fixed_physical=p,p_uncertainty_propagated=False,
        parameters=list(NAMES),candidate_noise_boxes=[bounds_for(moments(weights[t*100:(t+1)*100])).tolist() for t in range(3)],
        noise_box_note='extra noise boxes declared before inspecting profile; not yet formal posterior priors',
        original_core4_bounds=[[-500,500],[.5,4],[-4,6],[-4,4]],old8_parameter_model_preserved=True,
        n_covariance=ns,covariance_ids=ids,covariance_ddof=1,Hartlap_factor=alpha,
        covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,same_box_P_B_cross_covariance=True,
        P_bins=12,B_bins=9,kmax_P=.08,kmax_B=.08,target_ids=list(range(100)),target_count_per_truth=100,
        fid_target_covariance_overlap=sorted(set(ids)&set(range(100))),strict_independent_mock_recovery=False,
        fit_train_ids=list(range(0,100,2)),diagnostic_holdout_ids=list(range(1,100,2)),
        mean50_diagnostic_uses_same_C_single_not_divided=True,covariance_records=records,
        full500_pipeline_replaced=False,IR=False,full_STO_noise_closure_certified=False,MCMC_released=False))
    results=[];start=time.time()
    try:
        for t,(cosmo,truth) in enumerate(zip(COSMOS,(0.,100.,-100.))):
            update(case=cosmo,stage='fixed_truth_mean100_and_train50_profiles')
            data=all_rows[t*100:(t+1)*100][:,columns];marks=weights[t*100:(t+1)*100]
            units=moments(marks);model=make_model(arrays,units,p);bounds=bounds_for(units)
            physical,full=profile_fixed_truth(model,data.mean(axis=0),chol,truth,bounds)
            train_model=make_model(arrays,moments(marks[::2]),p)
            train,fit=profile_fixed_truth(train_model,data[::2].mean(axis=0),chol,truth,bounds)
            held_model=make_model(arrays,moments(marks[1::2]),p);held=data[1::2].mean(axis=0)
            held_residual=held_model.residual(train,held,chol)[0]
            predicted_zero=model.predictions(physical)[0][12:]-make_model(dict(arrays,
                matched_zero=np.zeros_like(zero)),units,p).predictions(physical)[0][12:]
            observed_zero=(all_rows[t*100:(t+1)*100]-counter_rows[t*100:(t+1)*100]).mean(axis=0)[24+local]
            result=dict(cosmology=cosmo,truth_fNL=truth,actual_target_mean100=full,
                fixed_truth_train50=fit,heldout50_prediction=held_model.predictions(train)[0].tolist(),
                heldout50_whitened_residual=held_residual.tolist(),heldout50_single_box_weighted_norm=float(np.linalg.norm(held_residual)),
                actual_mean100_zero_effect_norm=float(np.linalg.norm(np.linalg.solve(chol,np.r_[np.zeros(12),predicted_zero]))),
                actual_zero_atoms=observed_zero.tolist(),predicted_zero_atoms=predicted_zero.tolist(),
                zero_atom_residual_single_box_norm=float(np.linalg.norm(np.linalg.solve(chol,
                    np.r_[np.zeros(12),predicted_zero-observed_zero]))),
                original_matched_data_definition=True,full_STO_noise_closure_certified=False)
            results.append(result);save_json(OUT/(cosmo+'_PROFILE.json'),result);update(completed_cases=len(results))
        save_json(OUT/'REPORT.json',dict(status='ACTUAL_PB21_FIXED_TRUTH_EFT_CANDIDATE_PREFLIGHT_COMPLETE_NOT_FULL_NOISE_GATE',
            created_UTC=stamp(),results=results,unit_tests_actual_NERSC_passed=3,n_covariance=205,
            C_single=True,volume_Gpc_h3=1.,P_B_cross_covariance=True,p_fixed_physical=p,
            data_vectors_changed=False,prior_core4_changed=False,original500_pipeline_replaced=False,
            matrix_sha256=digest(matrix),contract_sha256=digest(OUT/'CONTRACT.json'),elapsed_seconds=time.time()-start,
            CPU_affinity=[13],peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            caveats=['fixed-truth feasibility is not free-fNL recovery or full stochastic closure',
                '50/50 targets overlap fid covariance identities as explicitly accepted',
                'white leading-noise EFT; omitted derivative/loop effects require model-adequacy review',
                'no IR noise certification, no formal C500 MCMC release'],MCMC_started=False,full_noise_gate_released=False))
        update(state='ACTUAL_PB_PREFLIGHT_COMPLETE_NOT_MCMC_RELEASE',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.detach:
        log=ROOT/'noise_eft_pb_preflight_v1.log';launch=ROOT/'noise_eft_pb_preflight_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('preserve existing preflight; inspect original process')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
