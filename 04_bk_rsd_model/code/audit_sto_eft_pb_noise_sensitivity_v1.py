"""模块大纲：与provisional链并行的真实noise幅度/输入-post P差分检查。

同一C205_single/mean100；raw N0/N2=.5/.7/1/1.5/2只是hypotheses，不是校准。
新增P差分统计用原24维C的线性变换，ddof1/单盒；不改变P+B数据或采样。
不生成正式STO noise PASS；固定fNL真值的profile不是自由fNL posterior。
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
import numpy as np
from scipy.optimize import least_squares
from rsd_b000_contract import PART,digest,save_json,F_GROWTH
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
MCMC=ROOT/'diagnostic_provisional_eft12_pb_20261010_v1'
PARENT_P=ROOT/'diagnostic_partial_cov_p_mcmc_20261010_v2'
OUT=ROOT/'noise_eft_sensitivity_v1'


def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def fixed_raw_profile(model,target,chol,bounds,truth,ratio):
    free=[i for i in range(12) if i not in (0,7)]
    def physical(x):
        row=np.empty(12);row[0]=truth;row[7]=ratio;row[free]=x;return row
    def residual(x):return model.residual(physical(x),target,chol)[0]
    fits=[]
    for b in (1.25,1.93,2.8):
        row=np.array([truth,b,-.25,-.3,0.,0.,0.,ratio,b,2*1.686*(b-model.p_halo),1.,0.])
        fit=least_squares(residual,row[free],bounds=(bounds[free,0],bounds[free,1]),
            x_scale=np.array([.2,1.,1.,1.,1.,1.,1.,3.,1.,1.]),
            ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2500)
        fits.append((float(fit.fun@fit.fun),fit))
    fits.sort(key=lambda x:x[0]);chi2,best=fits[0];row=physical(best.x)
    return dict(Nraw_over_N2_fixed_hypothesis=ratio,truth_fNL_fixed=truth,
        chi2_single_box_weighted=chi2,physical_parameters=row.tolist(),optimizer_success=bool(best.success),
        residual_whitened=model.residual(row,target,chol)[0].tolist(),noise_amplitude_calibrated=False)


def pipeline():
    if OUT.exists():raise ValueError('preserve existing actual noise sensitivity')
    parent=read(MCMC/'CONTRACT.json')
    if not parent['diagnostic_only'] or not parent['human_explicitly_approved_before_full_noise_closure']:
        raise ValueError('authorized provisional cohort required')
    for p,want in parent['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('provisional source drift')
    pcontract=read(PARENT_P/'CONTRACT.json')
    for p,want in pcontract['immutable_inputs'].items():
        if digest(p)!=want:raise ValueError('original P205 source drift')
    with np.load(PARENT_P/'FROZEN_PARTIAL_P_ENSEMBLE.npz',allow_pickle=False) as z:
        p_cov=z['covariance_single_box'];p_targets=z['target_vectors'];ids=z['covariance_ids']
    ns=len(ids)
    if ns!=205:raise ValueError('same frozen205 covariance cohort required')
    transform=np.c_[-np.eye(12),np.eye(12)]
    diff_cov=transform@p_cov@transform.T;alpha=(ns-14)/(ns-1)
    diff_chol=np.linalg.cholesky(diff_cov/alpha)  # 不使用floor/pinv/mean covariance。
    OUT.mkdir();sources=[Path(__file__),MCMC/'CONTRACT.json',PARENT_P/'CONTRACT.json',
        PARENT_P/'FROZEN_PARTIAL_P_ENSEMBLE.npz',PART/'code/post_rsd_eft_candidate_model_v1.py',
        PART/'code/post_rsd_eft_candidate_model_v2.py']
    cases=json.loads((MCMC/'CASES.json').read_text())
    sources.extend(MCMC/'cases'/name/'likelihood.npz' for name in cases)
    save_json(OUT/'CONTRACT.json',dict(status='NOISE_HYPOTHESIS_AND_ACTUAL_P_DIFFERENCE_NOT_FULL_CLOSURE',
        immutable_inputs={str(p):digest(p) for p in sources},hypothesis_Nraw_over_N2=[.5,.7,1.,1.5,2.],
        n_covariance=205,covariance_is_of_mean=False,volume_Gpc_h3=1.,C_single=True,
        main_MCMC_inputs_changed=False,raw_noise_amplitude_is_not_calibrated=True))
    state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[14],started_UTC=stamp(),completed_profiles=0)
    def update(**v):state.update(v,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update();results=[];count=0;start=time.time()
    try:
        for index,name in enumerate(cases):
            d=MCMC/'cases'/name;meta=read(d/'contract.json')
            with np.load(d/'likelihood.npz',allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files}
            model=EFTCandidateModelV2(a['coefficients'],a['templates'],a['operators'],a['intrinsic'],
                a['matched_zero'],a['weight_moments'],meta['p_fixed_physical'],F_GROWTH)
            diff=transform@p_targets[index];profiles=[]
            for ratio in (.5,.7,1.,1.5,2.):
                update(case=name,Nraw_over_N2=ratio)
                profiles.append(fixed_raw_profile(model,a['target'],a['chol'],a['bounds'],meta['truth_fNL'],ratio))
                count+=1;update(completed_profiles=count)
            minimum=min(r['chi2_single_box_weighted'] for r in profiles)
            for r in profiles:r['delta_chi2_within_fixed_raw_hypothesis_grid']=r['chi2_single_box_weighted']-minimum
            results.append(dict(cosmology=meta['cosmology'],profiles=profiles,
                input_to_post_P0_mean100_difference=diff.tolist(),
                difference_single_box_whitened_norm=float(np.linalg.norm(np.linalg.solve(diff_chol,diff))),
                difference_covariance_Hartlap=alpha,
                P_difference_cannot_be_interpreted_as_pure_raw_noise=True))
        save_json(OUT/'REPORT.json',dict(status='ACTUAL_RAW_NOISE_HYPOTHESIS_AND_P_SHIFT_DIAGNOSTICS_COMPLETE_NOT_FULL_STO_CLOSURE',
            created_UTC=stamp(),results=results,n_fixed_truth_profiles=15,
            C_single=True,covariance_is_of_mean=False,volume_Gpc_h3=1.,n_covariance=205,
            covariance_difference_is_linear_transform_of_same24_observables=True,
            actual_raw_noise_calibrated=False,finite_PNG_closure_certified=False,IR_noise_certified=False,
            original_MCMC_data_covariance_priors_changed=False,full_noise_gate_released=False,
            CPU_affinity=[14],elapsed_seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.))
        update(state='NOISE_SENSITIVITY_COMPLETE_NOT_FULL_CLOSURE',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('login26 CPU14 single core required')
    if args.detach:
        log=ROOT/'noise_eft_sensitivity_v1.log';launch=ROOT/'noise_eft_sensitivity_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect exact original sensitivity process before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
