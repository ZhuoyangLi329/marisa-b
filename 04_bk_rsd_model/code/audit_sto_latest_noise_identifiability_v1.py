"""模块大纲：实际C402/21bins的条件噪声响应和已结束64k边缘先验比较。

只读取likelihood、JSON与已完成的POSTERIOR_THIN；不读HDF、不调整模型。
不把局部rank、flat posterior或同数据拟合转换成正式物理闭合/删参许可。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import socket
import unittest
import numpy as np
import run_sto_post_latest_cov_pb_v1 as run
from post_rsd_noise_identifiability import noise_response, response_diagnostics, NOISE_NAMES
from post_rsd_owner_quiescence import owner_quiescent

OUT = run.OUT / 'noise_identifiability_v1'


def marginal_uniform_diagnostics(samples, bounds):
    result = []
    for index, name in enumerate(NOISE_NAMES, 4):
        lower, upper = bounds[index]
        values = samples[:, index]
        u = np.sort((values - lower) / (upper - lower))
        distance = max(float(np.max(np.arange(1, len(u) + 1) / len(u) - u)),
                       float(np.max(u - np.arange(len(u)) / len(u))))
        result.append(dict(parameter=name, posterior_std_over_uniform_prior_std=float(
            np.std(values, ddof=1) / ((upper - lower) / np.sqrt(12))),
            empirical_uniform_CDF_distance=distance, formal_KS_pvalue=None,
            correlation_not_treated_as_independent_samples=True))
    return result


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('actual diagnostic on original login26 CPU12 only')
    if OUT.exists():
        raise ValueError('preserve existing numerical diagnosis; never overwrite')
    parent = run.guard()
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_noise_identifiability'))
    if not tests.wasSuccessful() or tests.testsRun != 4:
        raise ValueError('four actual analytic/finite-response/cross-whitening tests required')
    records = []
    for index, truth in enumerate(run.COSMOS):
        directory = run.case_directory(index)
        case = run.LatestPBCase(directory)
        meta = run.read(directory / 'MAP.json')
        summary = run.read(directory / 'SUMMARY.json')
        points = [('blind_MAP', np.array(meta['center'])),
                  ('checkpoint_marginal_medians', np.array(summary['quantiles_2p5_16_50_84_97p5'])[2])]
        conditional = np.array(meta['center'])
        conditional[0] = case.meta['truth_fNL']
        points.append(('same_blind_MAP_other_coordinates_fNL_fixed_truth_not_reoptimized', conditional))
        diagnoses = []
        for name, point in points:
            jac = noise_response(case.model, point)
            proof = response_diagnostics(jac, case.chol, np.diff(case.bounds[4:], axis=1).ravel())
            # 原模型逐noise坐标的有限差响应；raw为二次，其中心secant仍精确。
            errors = []
            for j in range(8):
                h = .01
                if j == 3:
                    h = min(h, max(point[7] / 2, 1e-8))
                plus, minus = point.copy(), point.copy()
                plus[4 + j] += h
                minus[4 + j] -= h
                if minus[7] < 0:
                    raise ValueError('diagnostic secant cannot enter negative raw domain')
                difference = (case.model.predictions(plus)[0] - case.model.predictions(minus)[0]) / (2 * h)
                error = np.linalg.solve(case.chol, difference - jac[:, j])
                denom = max(1., float(np.max(abs(np.linalg.solve(case.chol, jac[:, j])))))
                errors.append(float(np.max(abs(error))) / denom)
            if max(errors) > 1e-9:
                raise ValueError('actual analytical response failed original1e-9 numerical gate')
            diagnoses.append(dict(point_name=name, point=point.tolist(),
                actual_original_prediction_secant_max_white_scaled_error=max(errors), **proof))
        stored = directory / 'POSTERIOR_THIN.npz'
        marginals = None
        if stored.exists():
            for progress in directory.glob('PROGRESS_ensemble_*.json'):
                if not owner_quiescent(run.read(progress)):
                    raise ValueError('only actual ended case thin posterior may be compared')
            with np.load(stored, allow_pickle=False) as source:
                np.testing.assert_array_equal(source['parameter_names'], case.names)
                marginals = marginal_uniform_diagnostics(source['samples'], case.bounds)
        records.append(dict(cosmology=truth, summary_checkpoint_steps=summary['steps'],
            summary_converged=summary['converged'], conditional_diagnoses=diagnoses,
            actual_ended_case_thin_marginals=marginals,
            likelihood_sha256=run.digest(directory / 'likelihood.npz')))
    OUT.mkdir()
    source_names = ['post_rsd_noise_identifiability.py', 'test_post_rsd_noise_identifiability.py',
                    'audit_sto_latest_noise_identifiability_v1.py']
    report = dict(status='ACTUAL402_LOCAL_NOISE_RESPONSE_AND_FINISHED_THIN_DIAGNOSIS_NOT_PARAMETER_SELECTION',
        created_UTC=run.stamp(), records=records, actual_native_tests_passed=4,
        parent_sha256=run.digest(run.OUT / 'PB_CONTRACT.json'),
        sources_sha256={name: run.digest(run.PART / 'code' / name) for name in source_names},
        n_covariance=402, n_target_per_truth=100, P_bins=12, B_bins=9,
        p_fixed_physical=parent['p_fixed_physical'], C_single=True, volume_Gpc_h3=1.,
        data_model_covariance_priors_sampler_changed=False, HDF_read=False,
        local_not_global_identifiability=True, noise_parameters_deleted=False,
        simplified7_parameter_RSD_closure_validated=False,
        full_STO_noise_closure_certified=False, formal_fNL_or_kmax_released=False)
    run.save_json(OUT / 'REPORT.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
