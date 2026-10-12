"""模块大纲：终止后的七参数链，抽查实际保存的log_prob与独立全cross似然。

本验收只读已完成HDF及冻结输入，不重采样、不调整模型/先验/协方差。
完整链形状、物理先验、原收敛门与分位数由原完成审计负责；这里仅补充
每ensemble的首末状态及128个固定随机状态的存储似然抽查，不冒充全链log_prob核验。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import socket
import unittest
import numpy as np
import emcee
import run_sto_three_noise_prepost_v1 as run
from audit_sto_three_noise_prepost_v1 import independent_prediction
from post_rsd_owner_quiescence import owner_quiescent


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('original login26 CPU12 audit only')
    run.guard()
    state = run.read(run.OUT/'STATUS.json')
    if not owner_quiescent(state):
        raise ValueError('actual owner live: no HDF access')
    for path in (run.OUT/'cases').glob('*/PROGRESS_ensemble_*.json'):
        if not owner_quiescent(run.read(path)):
            raise ValueError('actual writer live: no HDF access')
    completed = run.read(run.OUT/'MCMC_COMPLETION_AUDIT.json')
    if not completed['all_six_converged'] or len(completed['checks']) != 6:
        raise ValueError('original six-case complete-chain audit required')
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames([
            'test_post_rsd_legacy3_model', 'test_ensemble_progress_json_safe',
            'test_independent_ensemble_backend', 'test_hdf_lock_retry',
            'test_provisional_prior_safe_initialization']))
    if not tests.wasSuccessful() or tests.testsRun != 11:
        raise ValueError('eleven actual native regression tests required')
    checks = []
    for index in range(6):
        case = run.ThreeNoiseCase(run.case_directory(index))
        case.initialize()
        for ensemble in range(4):
            backend = emcee.backends.HDFBackend(
                str(case.directory/f'chain_ensemble_{ensemble}.h5'),
                name=f'ensemble_{ensemble}', read_only=True)
            chain = backend.get_chain().reshape(-1, 7)
            stored = backend.get_log_prob().reshape(-1)
            rng = np.random.default_rng(88100+4*index+ensemble)
            pick = np.unique(np.r_[0, len(chain)-1,
                rng.choice(len(chain), 128, replace=False)])
            points = chain[pick]
            physical = case.physical(points)
            prediction = np.stack([independent_prediction(row, case) for row in physical])
            white = np.linalg.solve(case.chol, (prediction-case.target).T).T
            independent = -.5*np.sum(white*white, axis=1)
            actual = case.log_prob(points)
            np.testing.assert_allclose(stored[pick], actual, rtol=2e-12, atol=1e-10)
            np.testing.assert_allclose(stored[pick], independent, rtol=2e-12, atol=1e-10)
            checks.append(dict(name=case.meta['name'], ensemble=ensemble,
                saved_steps_each_ensemble=int(backend.iteration), checked_states=len(pick),
                maximum_abs_saved_vs_independent_logprob=float(np.max(abs(stored[pick]-independent))),
                summary_sha256=run.digest(case.directory/'SUMMARY.json')))
            del chain, stored, backend
    report = dict(status='SIX_ACTUAL_LEGACY3_SAVED_LOGPROB_SPOT_CHECKS_AND_11_NATIVE_TESTS_PASS',
        created_UTC=run.previous.stamp(), checks=checks,
        native_regression_tests_passed=11, owner_and_all_known_writers_terminal=True,
        full_chain_logprob_checked=False, completion_audit_sha256=run.digest(run.OUT/'MCMC_COMPLETION_AUDIT.json'),
        model_data_covariance_priors_or_chains_changed=False,
        noise_parameter_count=3, total_parameter_count=7,
        full_STO_noise_closure_certified=False, audit_source_sha256=run.digest(__file__))
    run.save_json(run.OUT/'SAVED_LOGPROB_SPOT_AUDIT.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
