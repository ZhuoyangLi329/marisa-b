"""模块大纲：general leading colored/非Poisson shift响应的独立数值记录。

只验Gaussian-pair/F2均值的leading模型与指定相关点过程的渐近极限。
不校准实际STO N(k)，不将数值通过写为正式noise audit或MCMC放行。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import unittest
from sto_post_compute8_common import PART, ROOT, sha
from test_post_rsd_general_noise_transport import GeneralNoiseTest, clustered_leading_rows


def main():
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single logical CPU for this bounded operator audit')
    target = ROOT/'GENERAL_COLORED_NOISE_TRANSPORT_AUDIT.json'
    if target.exists():
        raise ValueError('preserve prior general noise transport audit')
    start = time.time(); stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GeneralNoiseTest))
    if not tests.wasSuccessful() or tests.testsRun != 6 or tests.skipped:
        raise ValueError('all six explicit point/Wick/shell/cluster tests required: '+stream.getvalue())
    rows = clustered_leading_rows(); large = [r for r in rows if r['nparents'] == 1000000]
    maximum = max(max(r['signal_relative_error'], r['uniform_relative_error']) for r in large)
    tolerance = 1e-4
    files = ['post_rsd_general_noise_transport.py', 'test_post_rsd_general_noise_transport.py',
        'audit_post_rsd_general_noise_transport.py', 'post_rsd_clustered_moments.py',
        'post_rsd_marked_moments.py', 'post_rsd_poisson_projection.py', 'post_rsd_noise_transport.py',
        'post_rsd_fft_projection.py', 'post_rsd_fixedrec.py', 'rsd_b000_fft_projection.py']
    report = dict(status='LEADING_GENERAL_GAUSSIAN_PAIR_COLORED_TRANSPORT_PASS_NOT_ACTUAL_STO_CLOSURE',
        created_UTC=datetime.now(timezone.utc).isoformat(), independent_tests_passed=tests.testsRun,
        test_output=stream.getvalue(), rows=rows, max_large_N_cluster_relative_error=maximum,
        original_cluster_tolerance=tolerance, source_sha256={name:sha(PART/'code'/name) for name in files},
        Gaussian_pair_and_F2_background_independently_verified=True,
        finite_independent_uniform_random_cancellation_leading_order_verified=True,
        actual_particle_N2_is_not_raw_stochastic_Nk=True,
        white_formula='Nraw*Kcross+(Nraw-N2)*Ksame+Nraw*(Nraw-N2)*KNN',
        colored_formula='-sum_pairs(Ti+Tj)*(Di*Nj+Dj*Ni+Ni*Nj)+N2*sum_i Ri*(Di+Ni)',
        shared_exact_FFT_and_original_full_shell_denominator_reused=True,
        third_leg_kmax_added=False, zero_atom_terms_still_require_separate_estimator_choice=True,
        actual_STO_noise_power_calibrated=False, PNG_noise_response_calibrated=False,
        connected_three_or_four_point_stochastic_responses_validated=False,
        finite_shift_or_IR_validated=False, original_physical_p_calibrated=False,
        production_likelihood_or_priors_modified=False, formal_noise_audit_released=False,
        MCMC_released=False, full_goal_reduced=False,
        CPU_affinity=sorted(os.sched_getaffinity(0)), elapsed_seconds=time.time()-start)
    if maximum > tolerance:
        raise ValueError('original leading cluster gate failed')
    with target.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows', 'test_output', 'source_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
