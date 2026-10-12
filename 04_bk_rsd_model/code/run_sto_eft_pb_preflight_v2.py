"""模块大纲：独立Pshot的12参数真实P+B preflight；原v1结果/执行器不改。

复用唯一数据/C205/p校准装配；只显式替换模型、noise bounds和profile维数。
保留全部核心先验、原Pshot物理先验，三真值mean100、50/50、全joint白化。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
import numpy as np
from scipy.optimize import least_squares
import run_sto_eft_pb_preflight_v1 as shared
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2,NAMES_V2

ROOT=shared.ROOT;OUT=ROOT/'noise_eft_pb_preflight_v2'
base_bounds=shared.bounds_for


def bounds(units):
    return np.vstack((base_bounds(units),[-1e6/units[0],1e6/units[0]]))


def profile(model,target,chol,truth,prior_bounds):
    def rows(x):return np.r_[truth,x]
    def residual(x):return model.residual(rows(x),target,chol)[0]
    fits=[]
    for b in (1.25,1.93,2.8):
        start=np.array([b,-.25,-.3,0.,0.,0.,1.,b,2*1.686*(b-model.p_halo),1.,0.])
        fit=least_squares(residual,start,bounds=(prior_bounds[1:,0],prior_bounds[1:,1]),
            x_scale=np.array([.2,1.,1.,1.,1.,1.,.5,1.,3.,1.,1.]),
            ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2500)
        fits.append((float(fit.fun@fit.fun),fit))
    fits.sort(key=lambda a:a[0]);score,best=fits[0];physical=rows(best.x)
    singular=np.linalg.svd(best.jac,compute_uv=False);rank=int(np.sum(singular>max(singular[0],1.)*1e-10))
    return physical,dict(fNL_fixed_to_simulation_truth=True,truth_fNL=truth,parameters=list(NAMES_V2),
        physical_parameters=physical.tolist(),chi2_single_box_weighted=score,
        residual_whitened=model.residual(physical,target,chol)[0].tolist(),prediction=model.predictions(physical)[0].tolist(),
        optimizer_success=bool(best.success),optimizer_message=str(best.message),jacobian_singular_values=singular.tolist(),
        local_effective_rank=rank,possible_exact_unidentified_at_fNL0=['a_phi','bphi_w2'] if truth==0 else [],
        starts=[dict(chi2=x[0],success=bool(x[1].success),nfev=int(x[1].nfev)) for x in fits],
        Pshot_independent_effective_constant_not_raw_N0=True,is_MCMC=False,is_full_noise_closure_gate=False)


def pipeline():
    prior=shared.read(ROOT/'noise_eft_pb_preflight_v1/STATUS.json')
    if prior['state']!='ACTUAL_PB_PREFLIGHT_COMPLETE_NOT_MCMC_RELEASE' or Path('/proc/'+str(prior['pid'])).exists():
        raise ValueError('original v1 preflight must be actually complete/terminal')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_eft_candidate_model_v2'))
    if not tests.wasSuccessful() or tests.testsRun!=3:raise ValueError('three actual independent-Pshot tests required')
    old=(shared.OUT,shared.NAMES,shared.EFTCandidateModel,shared.bounds_for,shared.profile_fixed_truth,shared.save_json)
    extra=[Path(__file__),shared.PART/'code/post_rsd_eft_candidate_model_v2.py',
        shared.PART/'code/test_post_rsd_eft_candidate_model_v2.py',ROOT/'noise_eft_pb_preflight_v1/REPORT.json']
    frozen={str(p):shared.digest(p) for p in extra};save=shared.save_json
    def write(path,value):
        for p,want in frozen.items():
            if shared.digest(p)!=want:raise ValueError('independent-Pshot route/source drift')
        value=dict(value)
        if Path(path).name=='CONTRACT.json':
            value['immutable_inputs'].update(frozen);value.update(
                candidate_parameter_count=12,Pshot_original_physical_bounds=[-1e6,1e6],
                Pshot_independent_from_positive_data_only_raw_noise=True,
                baseline_v1_preserved='noise_eft_pb_preflight_v1',
                correction_reason='original effective Pshot is not a certified pure raw-noise calibration')
        if Path(path).name=='REPORT.json':value.update(unit_tests_actual_NERSC_passed=6,candidate_parameter_count=12,
            Pshot_original_physical_bounds=[-1e6,1e6],Pshot_independent_from_data_only_raw_noise=True)
        save(path,value)
    shared.OUT=OUT;shared.NAMES=NAMES_V2;shared.EFTCandidateModel=EFTCandidateModelV2
    shared.bounds_for=bounds;shared.profile_fixed_truth=profile;shared.save_json=write
    try:shared.pipeline()
    finally:shared.OUT,shared.NAMES,shared.EFTCandidateModel,shared.bounds_for,shared.profile_fixed_truth,shared.save_json=old


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.detach:
        log=ROOT/'noise_eft_pb_preflight_v2.log';launch=ROOT/'noise_eft_pb_preflight_v2.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect exact existing v2 before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        shared.save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=shared.stamp(),log=str(log)))
        print(json.dumps(shared.read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
