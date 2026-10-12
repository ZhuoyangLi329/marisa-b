"""模块大纲：独立重算最新冻结盒数的C_single/targets；终止后回放完整链。

输入审计不读取活动HDF。完成审计须实际owner退出且三case产物齐全；
不把numerical/采样PASS提升为完整STO stochastic closure或正式fNL结论。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import time
import numpy as np
import emcee
from rsd_b000_contract import PART,digest,save_json,selected,reference,F_GROWTH
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2,NAMES_V2
import run_sto_post_latest_cov_pb_v1 as run
import run_rsd_b000_mcmc as sampler

ROOT=run.ROOT;OUT=run.OUT


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def input_audit():
    result_path=OUT/'INDEPENDENT_INPUT_AUDIT.json'
    if result_path.exists():
        existing=read(result_path)
        if existing['parent_sha256']!=digest(OUT/'PB_CONTRACT.json'):
            raise ValueError('previous independent input audit belongs to different cohort')
        for path,want in existing['audited_inputs_sha256'].items():
            if digest(path)!=want:raise ValueError('audited input drift')
        return existing
    c=run.guard();ids=c['covariance_ids'];ns=len(ids)
    if ns!=c['n_covariance'] or ns!=402 or ids!=list(range(194))+list(range(292,500)):
        raise ValueError('actual human-requested402 snapshot required, not live growing cohort')
    if len(set(ids))!=ns or not c['P_B_cross_covariance']:
        raise ValueError('unique same-box covariance identities and cross block required')
    wanted=selected(.08);ref=reference();ix=ref['certified_indices']
    local=np.array([np.flatnonzero(ix==j)[0] for j in wanted])
    if len(wanted)!=9:raise ValueError('original9 B bins required')
    rows=[];records=[];fixed=None
    for real in ids:
        arrays,meta,done=load_pilot(directory(('fiducial',real)))
        if meta['arguments']['realization']!=real or meta['arguments']['cosmology']!='fiducial':
            raise ValueError('covariance identity mismatch')
        if fixed is None:fixed=meta['fixed_reconstruction']
        if meta['fixed_reconstruction']!=fixed:raise ValueError('mixed fixed reconstruction')
        np.testing.assert_array_equal(arrays['B_edges'],ref['edges'])
        rows.append(np.r_[arrays['P_post'],arrays['B_post'][wanted]])
        records.append(dict(realization=real,measurement_sha256=done['measurement_sha256']))
    rows=np.array(rows);raw=np.cov(rows,rowvar=False,ddof=1)
    if raw.shape!=(21,21):raise ValueError('full21 P+B covariance required')
    alpha=(ns-21-2)/(ns-1)
    np.linalg.cholesky(raw)  # 明确拒绝非正定；无floor、pinv或shrinking。
    units=np.sqrt(np.diag(raw));corr=raw/np.outer(units,units)
    with np.load(ROOT/'target300_completion_v1/TARGETS_ONLY.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(ix,z['B_indices'])
        all_targets=z['matched_rows'];marks=z['data_N2_N3_nhalo']
    calpath=run.POUT/'cases'/run.pbase.CAL_NAME/'SUMMARY.json'
    cal=read(calpath);physical_p=float(cal['quantiles_2p5_16_50_84_97p5'][2][0])
    if not cal['converged'] or physical_p!=c['p_fixed_physical']:
        raise ValueError('actual converged latest-cohort calibration required')
    with np.load(ROOT/'calibration_holdout_p0/CALIBRATION_P0_HOLDOUT.npz',allow_pickle=False) as z:
        hold=z['P0'];np.testing.assert_array_equal(z['realizations'],np.arange(100,250))
    with np.load(run.POUT/'cases'/run.pbase.CAL_NAME/'likelihood.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(z['target'],hold.mean(axis=0),rtol=3e-13,atol=1e-7)
    checks=[];audited={str(calpath):digest(calpath)}
    for t,cosmo in enumerate(run.COSMOS):
        d=run.case_directory(t);meta=read(d/'contract.json')
        if (meta['n_covariance']!=ns or meta['n_target']!=100 or meta['covariance_is_of_mean']
                or meta['volume_Gpc_h3']!=1. or meta['covariance_multiplier']!=1.
                or meta['parameters']!=list(NAMES_V2) or meta['IR']
                or meta['kmax_B']!=.08 or meta['kmax_P']!=.08
                or meta['full_STO_noise_closure_certified']):raise ValueError('case science contract mismatch')
        with np.load(d/'likelihood.npz',allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files}
        np.testing.assert_allclose(a['covariance'],raw,rtol=3e-13,atol=1e-7)
        np.testing.assert_allclose(a['chol']@a['chol'].T,raw/alpha,rtol=3e-13,atol=1e-7)
        expected=all_targets[t*100:(t+1)*100][:,np.r_[np.arange(12,24),24+local]].mean(axis=0)
        np.testing.assert_allclose(a['target'],expected,rtol=3e-13,atol=1e-7)
        np.testing.assert_array_equal(a['bounds'][:4],[[-500,500],[.5,4],[-4,6],[-4,4]])
        np.testing.assert_allclose(a['bounds'][11]*a['weight_moments'][0],[-1e6,1e6],rtol=3e-15,atol=1e-8)
        model=EFTCandidateModelV2(a['coefficients'],a['templates'],a['operators'],a['intrinsic'],
            a['matched_zero'],a['weight_moments'],physical_p,F_GROWTH)
        probes=np.array([[0.,1.93,-.25,-.3,0.,0.,0.,1.,1.93,2.,1.,0.],
            [100.,2.23,.4,-.4,.3,-.2,2.,.7,2.8,4.,.8,-1.5],
            [-100.,1.75,-1.,.2,-.3,.5,-2.,1.5,1.9,-3.,-.3,.2],
            [500.,3.8,5.,3.,9.,-8.,18.,8.,-8.,45.,18.,10.]])
        averaged=np.zeros((len(probes),21))
        for n2,n3,nhalo in marks[t*100:(t+1)*100]:
            nr=1e9/(30*nhalo);boxmoments=np.array([n2,n3,n2*n2,nr,n2*nr])
            single=EFTCandidateModelV2(a['coefficients'],a['templates'],a['operators'],a['intrinsic'],
                a['matched_zero'],boxmoments,physical_p,F_GROWTH)
            averaged+=single.predictions(probes)/100.
        predicted=model.predictions(probes)
        difference=np.linalg.solve(a['chol'],(predicted-averaged).T).T
        error=float(np.max(abs(difference)/np.maximum(1.,abs(np.linalg.solve(a['chol'],averaged.T).T))))
        if error>1e-9:raise ValueError('actual individual-box noise moment average failed original1e-9 gate')
        pcase=run.POUT/'cases'/run.pbase.NAMES[t+1]
        with np.load(pcase/'likelihood.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(a['target'][:12],z['target'],rtol=3e-13,atol=1e-7)
            np.testing.assert_allclose(raw[:12,:12],z['covariance'],rtol=3e-13,atol=1e-7)
            np.testing.assert_allclose(z['chol']@z['chol'].T,raw[:12,:12]/((ns-14)/(ns-1)),rtol=3e-13,atol=1e-7)
        checks.append(dict(cosmology=cosmo,actual_target_count=100,individually_averaged_noise_probes=len(probes),
            mean_product_not_product_mean=True,maximum_box_average_whitened_scaled_error=error,
            target_covariance_P_block_matches_P0=True))
        for q in [d/'likelihood.npz',d/'contract.json',pcase/'likelihood.npz',pcase/'SUMMARY.json']:
            audited[str(q)]=digest(q)
    proof=dict(status='INDEPENDENT_C402_MEAN100_PB21_INPUTS_PASS_NOT_PHYSICAL_NOISE_CERTIFICATION',
        created_UTC=stamp(),audit_source_sha256=digest(Path(__file__)),parent_sha256=digest(OUT/'PB_CONTRACT.json'),
        audited_inputs_sha256=audited,n_covariance=ns,covariance_ids=ids,covariance_ddof=1,
        covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,P_B_cross_covariance=True,
        Hartlap_PB=alpha,Hartlap_P0=(ns-14)/(ns-1),p_fixed_physical=physical_p,
        p_uncertainty_propagated=False,checks=checks,covariance_records=records,
        maximum_absolute_P_B_correlation=float(np.max(abs(corr[:12,12:]))),
        correlation_minimum_eigenvalue=float(np.linalg.eigvalsh(corr).min()),
        fid_target_covariance_overlap=list(range(100)),strict_independent_mock_recovery=False,
        full_STO_noise_closure_certified=False,formal_fNL_or_kmax_released=False)
    save_json(result_path,proof);return proof


def completed_audit():
    s=read(OUT/'STATUS.json');proc=Path('/proc')/str(s['pid'])
    if proc.exists() and 'run_sto_post_latest_cov_pb_v1.py' in (proc/'cmdline').read_bytes().decode():
        print('NOT_READY_OWNER_STILL_LIVE_NO_HDF_READ',flush=True);return False
    if (s['state']!='LATEST_COV_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE'
            or s['completed_pb_cases']!=list(run.COSMOS) or not (OUT/'RESULTS.json').exists()):
        print('NOT_READY_ALL3_REAL_CASES_NOT_FINISHED_NO_HDF_READ',flush=True);return False
    proof=input_audit();checks=[];all_converged=True
    for t,cosmo in enumerate(run.COSMOS):
        case=run.LatestPBCase(run.case_directory(t));case.initialize();stored=read(case.directory/'SUMMARY.json')
        backends=[emcee.backends.HDFBackend(str(case.directory/f'chain_ensemble_{j}.h5'),name=f'ensemble_{j}') for j in range(4)]
        iterations=[b.iteration for b in backends]
        if len(set(iterations))!=1 or iterations[0]!=stored['steps']:raise ValueError('actual HDF checkpoint inconsistent')
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape!=(4,stored['steps'],32,12):raise ValueError('actual4x32x12 chains required')
        physical=case.physical(chains)
        if not np.isfinite(physical).all() or np.any(physical<=case.bounds[:,0]) or np.any(physical>=case.bounds[:,1]):
            raise ValueError('actual physical samples violate frozen original12 priors')
        recalculated,flat=sampler.diagnose(case,chains,accepted,stored['steps'])
        if recalculated['gates']!=stored['gates'] or recalculated['converged']!=stored['converged']:
            raise ValueError('original sampling gate replay mismatch')
        np.testing.assert_allclose(recalculated['quantiles_2p5_16_50_84_97p5'],
            stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
        np.testing.assert_allclose(recalculated['fNL_sigma'],stored['fNL_sigma'],rtol=2e-13)
        with np.load(case.directory/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['parameter_names'],case.names)
            pick=np.random.default_rng(5500+t).choice(len(flat),min(30000,len(flat)),replace=False)
            np.testing.assert_array_equal(z['samples'],flat[pick])
        checks.append(dict(cosmology=cosmo,steps_each_ensemble=stored['steps'],actual_ensemble_iterations=iterations,
            converged=bool(recalculated['converged']),original_gates=recalculated['gates'],
            fNL_median=float(recalculated['quantiles_2p5_16_50_84_97p5'][2,0]),fNL_sigma=float(recalculated['fNL_sigma']),
            summary_sha256=digest(case.directory/'SUMMARY.json')))
        all_converged=all_converged and bool(recalculated['converged'])
    result=dict(status=('ALL3_PRELIMINARY_PB_SAMPLING_AND_INPUTS_VERIFIED_NOT_FORMAL_NOISE'
        if all_converged else 'ACTUAL3_PB_TRIALS_FINISHED_BUT_ORIGINAL_SAMPLING_GATES_FAILED'),
        created_UTC=stamp(),all3_converged=all_converged,checks=checks,
        input_audit_sha256=digest(OUT/'INDEPENDENT_INPUT_AUDIT.json'),
        C_single=True,n_covariance=402,n_target_per_truth=100,volume_Gpc_h3=1.,
        p_fixed_physical=proof['p_fixed_physical'],p_uncertainty_propagated=False,
        provisional=True,diagnostic_only=True,full_STO_noise_closure_certified=False,
        formal_fNL_or_kmax_released=False,audit_source_sha256=digest(Path(__file__)))
    save_json(OUT/'MCMC_COMPLETION_AUDIT.json',sampler.clean(result));print(json.dumps(sampler.clean(result)),flush=True)
    return all_converged


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--completed',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:raise ValueError('login26 CPU12 only')
    if args.completed:completed_audit()
    elif (OUT/'PB_CONTRACT.json').exists():print(json.dumps(input_audit()),flush=True)
    else:print('NOT_READY_PB_INPUT_FREEZE_NO_HDF_READ',flush=True)


if __name__=='__main__':main()
