"""模块大纲：冻结.08/train50参数，实际odd50未拟合B分箱的全cross条件检查。

复用原205同盒C_single、原44分箱及唯一EFT12；不重拟合高k，不改正在跑的链。
新增分箱只作shape extrapolation，不认证完整STO closure或更高kmax。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import traceback
import unittest
import numpy as np
from rsd_b000_contract import PART, digest, save_json, F_GROWTH, selected, CUTS
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from post_rsd_noir_polynomial import collapse_physical_p
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2
from run_sto_eft_pb_preflight_v1 import moments
from post_rsd_conditional_validation import conditional_residual

ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
MCMC = ROOT / 'diagnostic_provisional_eft12_pb_20261010_v2'
PRE = ROOT / 'noise_eft_pb_preflight_v2'
OUT = ROOT / 'noise_eft_unfitted_bins_v1'


def read(path): return json.loads(Path(path).read_text())
def stamp(): return datetime.now(timezone.utc).isoformat()


def pipeline():
    if OUT.exists(): raise ValueError('preserve existing heldout-bin diagnostic')
    sources = {}
    for contract in (MCMC/'CONTRACT.json', PRE/'CONTRACT.json'):
        sources[str(contract)] = digest(contract)
        for path, expected in read(contract)['immutable_inputs'].items():
            if digest(path) != expected: raise ValueError('frozen provisional input drift: '+path)
            sources[path] = expected
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_conditional_validation'))
    if not tests.wasSuccessful() or tests.testsRun != 3: raise ValueError('three actual conditional-covariance tests required')
    target_path = ROOT/'target300_completion_v1/TARGETS_ONLY.npz'
    expected = read(ROOT/'target300_completion_v1/AUDIT.json')['target_matrix_sha256']
    if digest(target_path) != expected: raise ValueError('original300 target bytes changed')
    with np.load(target_path, allow_pickle=False) as z:
        indices=z['B_indices'].copy(); target_rows=z['matched_rows'][:,12:].copy()
        marks=z['data_N2_N3_nhalo'].copy(); counter_rows=z['nonzero_diagnostic_rows'][:,12:].copy()
    if len(indices)!=44 or target_rows.shape!=(300,56): raise ValueError('original full12P+44B required')
    c = read(PRE/'CONTRACT.json'); ids=c['covariance_ids']; p=c['p_fixed_physical']
    if ids != list(range(153))+list(range(448,500)) or c['covariance_is_of_mean']:
        raise ValueError('same frozen205 C_single required')
    def local(cut): return np.array([np.flatnonzero(indices==j)[0] for j in selected(cut)])
    fitted=np.r_[np.arange(12),12+local(.08)]
    paths=[Path(__file__), PART/'code/post_rsd_conditional_validation.py',
        PART/'code/test_post_rsd_conditional_validation.py', target_path,
        ROOT/'noir_paware_templates/DETERMINISTIC_COEFFICIENTS.npz',
        ROOT/'noise_transport_templates_v1/OPERATOR_COEFFICIENTS.npz',
        ROOT/'noise_eft_primitives_v3/PRIMITIVES.npz']
    cases=json.loads((MCMC/'CASES.json').read_text())
    paths.extend(PRE/(cosmo+'_PROFILE.json') for cosmo in ('fiducial','LC_p','LC_m'))
    sources.update({str(q):digest(q) for q in paths})
    OUT.mkdir(); state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),
        CPU_affinity=[14],started_UTC=stamp(),completed_covariance_rows=0)
    def update(**v): state.update(v,updated_UTC=stamp()); save_json(OUT/'STATUS.json',state)
    update(stage='same205_full_joint_covariance_replay')
    save_json(OUT/'CONTRACT.json',dict(status='FROZEN_ODD50_UNFITTED_B_SHAPE_CHECK_NOT_MCMC_OR_CLOSURE',
        immutable_inputs=sources, n_covariance=205,covariance_ids=ids,covariance_ddof=1,
        C_single=True,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
        P_B_cross_covariance=True,p_fixed_physical=p,p_uncertainty_propagated=False,
        training_target_ids=list(range(0,100,2)),heldout_target_ids=list(range(1,100,2)),
        train_fNL_fixed_to_truth=True,parameters_frozen_at_kmax_B08=True,
        full_dimension_Hartlap_per_cut=True,high_k_data_not_refitted=True,
        original_MCMC_inputs_changed=False,formal_fNL_or_kmax_released=False,full_STO_noise_closure_certified=False))
    try:
        rows=[]
        for real in ids:
            data, meta, done=load_pilot(directory(('fiducial',real)))
            rows.append(np.r_[data['P_post'],data['B_post'][indices]])
            update(completed_covariance_rows=len(rows))
        covariance=np.cov(np.array(rows),rowvar=False,ddof=1)
        with np.load(PRE/'FROZEN_PB205_PREFLIGHT.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(covariance[np.ix_(fitted,fitted)],z['covariance_single_box'],rtol=3e-13,atol=1e-7)
            templates=z['templates'].copy()
        with np.load(paths[4],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['indices'],indices)
            coefficients=collapse_physical_p(z['coefficients'],p_halo=p)
        with np.load(paths[5],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['indices'],indices)
            basis=np.array([1.,(p-1.3)/.5,2*((p-1.3)/.5)**2-1])
            operators=np.tensordot(z['coefficients'],basis,axes=(1,0))
        with np.load(paths[6],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['indices'],indices)
            intrinsic=z['intrinsic'].copy(); zero=z['matched_zero'].copy()
        results=[]
        for i,cosmo in enumerate(('fiducial','LC_p','LC_m')):
            profile=read(PRE/(cosmo+'_PROFILE.json'))
            v=np.array(profile['fixed_truth_train50']['physical_parameters'])
            weights=marks[i*100:(i+1)*100];units=moments(weights[1::2])
            model=EFTCandidateModelV2(coefficients,templates,operators,intrinsic,zero,units,p,F_GROWTH)
            prediction=model.predictions(v)[0];held=target_rows[i*100:(i+1)*100][1::2].mean(axis=0)
            np.testing.assert_allclose(prediction[fitted],profile['heldout50_prediction'],rtol=3e-12,atol=1e-6)
            zero_model=EFTCandidateModelV2(coefficients,templates,operators,intrinsic,np.zeros_like(zero),units,p,F_GROWTH)
            predicted_zero=prediction-zero_model.predictions(v)[0]
            actual_zero=(target_rows[i*100:(i+1)*100]-counter_rows[i*100:(i+1)*100])[1::2].mean(axis=0)
            scans=[]
            for cut in CUTS[1:]:
                included=np.r_[np.arange(12),12+local(cut)]
                fit_at=np.array([np.flatnonzero(included==j)[0] for j in fitted])
                scans.append(dict(kmax_B_probe=float(cut),B_bin_count=len(local(cut)),
                    forecast_shape=conditional_residual((prediction-held)[included],
                        covariance[np.ix_(included,included)],fit_at,205),
                    zero_atom_shape=conditional_residual((predicted_zero-actual_zero)[included],
                        covariance[np.ix_(included,included)],fit_at,205)))
            results.append(dict(cosmology=cosmo,train50_parameters=v.tolist(),heldout_target_count=50,
                unfitted_B_predictions=prediction[12:].tolist(),heldout50_B=held[12:].tolist(),scans=scans))
            update(stage='heldout_shape_projection',completed_cosmologies=len(results))
        for path, expected in sources.items():
            if digest(path)!=expected:raise ValueError('immutable input drift during heldout replay')
        matrix=OUT/'FULL56_C205_SINGLE.npz'
        np.savez_compressed(matrix,covariance_single_box=covariance,covariance_ids=ids,
            covariance_rows=np.array(rows),original_B_indices=indices,fitted_full56_columns=fitted)
        save_json(OUT/'REPORT.json',dict(status='ODD50_UNFITTED_B_CONDITIONAL_SHAPES_COMPLETE_NOT_FULL_STO_CLOSURE',
            created_UTC=stamp(),results=results,n_covariance=205,C_single=True,covariance_is_of_mean=False,
            covariance_multiplier=1.,volume_Gpc_h3=1.,P_B_cross_covariance=True,
            actual_NERSC_conditional_tests_passed=3,original_MCMC_inputs_changed=False,
            shape_is_descriptive_forecast_not_calibrated_GOF=True,high_k_data_refitted=False,
            raw_noise_calibrated=False,finite_shift_connected_closure_certified=False,IR_noise_certified=False,
            full_noise_gate_released=False,formal_fNL_or_kmax_released=False,matrix_sha256=digest(matrix),
            CPU_affinity=[14],peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.))
        update(state='UNFITTED_B_SHAPES_COMPLETE_NOT_FULL_CLOSURE',finished_UTC=stamp())
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:
        raise ValueError('login26 CPU14 single core required')
    if args.detach:
        log=ROOT/'noise_eft_unfitted_bins_v1.log';launch=ROOT/'noise_eft_unfitted_bins_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect existing original diagnostic first')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__': main()
