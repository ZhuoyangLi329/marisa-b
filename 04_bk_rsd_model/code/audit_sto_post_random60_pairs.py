"""模块大纲：完整20盒random30/60身份与统计验收；保存配对差，不替代C500。

复用已有pilot验收函数。要求所有固定算法参数、源权重和post halo完全相同。
保留原B及零腿反事实两套诊断，不自行选择正式estimator或增加拟合参数。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import unittest
import numpy as np
from audit_sto_post_fid20_pilots import ROOT, PART, CUTS, COUNTS, FIELDS, sha, load_pilot
from post_rsd_paired_controls import paired_moments, selected_summary


def main():
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single-core paired audit required')
    queue = ROOT / 'pilot_fid20_random60_queue'
    state = json.loads((queue / 'STATUS.json').read_text())
    if state['state'] != 'COMPLETE' or state['completed'] != list(range(20)) or state['workers']:
        raise RuntimeError('all twenty paired controls must complete first')
    out = ROOT / 'pilot_fid20_random60_diagnostics'
    if out.exists(): raise RuntimeError('preserve the existing paired audit')
    tests = unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_paired_controls')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful() or result.testsRun != 3: raise RuntimeError('paired statistics regressions failed')
    contract = json.loads((queue / 'CONTRACT.json').read_text())
    for path, expected in contract['immutable_sources'].items():
        if sha(path) != expected: raise ValueError('queue input drift: ' + path)
    baseline_path = ROOT / 'pilot_fid20_diagnostics/AUDIT.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['status'] != 'FID20_PILOT_MEASUREMENTS_VERIFIED_NOT_MAIN_COVARIANCE':
        raise ValueError('original complete baseline audit required')
    matrix = baseline_path.parent / 'PILOT_MEASUREMENTS.npz'
    if sha(matrix) != baseline['matrix_sha256']: raise ValueError('baseline aggregate drift')
    with np.load(matrix, allow_pickle=False) as z:
        base_arrays = {key: np.array(z[key]) for key in FIELDS}
        ix = z['B_indices'].copy(); b_k = z['B_k'].copy(); p_k = z['P_k'].copy()
        np.testing.assert_array_equal(z['realizations'], np.arange(20))
    rows_by_id = {r['realization']: r for r in baseline['rows']}
    if sorted(rows_by_id) != list(range(20)): raise ValueError('baseline realization set changed')
    rows = []; controls = {key: [] for key in FIELDS}
    for index in range(20):
        tag = f'fiducial_real{index:03d}_rec256_meas256_ran'
        old_dir = ROOT / 'pilots' / (tag + '30_selfmatched_zeroaudit_nested')
        new_dir = ROOT / 'pilots' / (tag + '60_selfmatched_zeroaudit_nested')
        old, om, od = load_pilot(old_dir); new, nm, nd = load_pilot(new_dir)
        prior = rows_by_id[index]
        if sha(old_dir / 'DONE.json') != prior['DONE_sha256']:
            raise ValueError('paired baseline completion changed')
        if od['post_halos_sha256'] != nd['post_halos_sha256']:
            raise ValueError('finite-random control changed halo coordinates or STO weights')
        for key in ('fixed_reconstruction', 'source_STO_sha256', 'source_diagnostics_sha256',
                    'raw_fof', 'random_seed', 'random_seed_namespace_nran',
                    'entry_sha256', 'shared_measure_source_sha256', 'matched_self_adapter_sha256'):
            if om[key] != nm[key]: raise ValueError('control is not paired: ' + key)
        if om['arguments']['nran'] != 30 or nm['arguments']['nran'] != 60:
            raise ValueError('wrong random-density pair')
        if ({k: v for k, v in om['arguments'].items() if k != 'nran'}
                != {k: v for k, v in nm['arguments'].items() if k != 'nran'}):
            raise ValueError('another numerical setting changed')
        for key in ('P_edges', 'B_edges', 'P_post_k', 'P_post_nmodes', 'B_post_k', 'B_post_nmodes'):
            np.testing.assert_array_equal(old[key], new[key])
        for field in FIELDS:
            np.testing.assert_array_equal(old[field], base_arrays[field][index])
            controls[field].append(new[field])
        rows.append({'realization': index, 'baseline_DONE_sha256': sha(old_dir / 'DONE.json'),
            'control_DONE_sha256': sha(new_dir / 'DONE.json'),
            'baseline_measurement_sha256': od['measurement_sha256'],
            'control_measurement_sha256': nd['measurement_sha256'],
            'identical_post_halos_sha256': nd['post_halos_sha256'],
            'source_STO_sha256': nm['source_STO_sha256'], 'raw_fof_sha256': nm['raw_fof']['source_raw_sha256'],
            'random_seed': nm['random_seed']})
    arrays = {'realizations': np.arange(20), 'B_indices': ix, 'B_k': b_k, 'P_k': p_k}
    summaries = {}
    for field in FIELDS:
        control = np.array(controls[field]); arrays['random60_' + field] = control
        arrays['delta_' + field] = control - base_arrays[field]
    for field in ('P_weighted_input', 'P_post', 'B_post', 'B_post_nonzero_only_counterfactual'):
        # B统计只消费正式44-bin集合，避免全registry里未使用的空/退化bin影响诊断。
        selection = ix if field.startswith('B_') else np.arange(len(p_k))
        moments = paired_moments(base_arrays[field][:, selection], np.array(controls[field])[:, selection])
        for name, value in moments.items(): arrays[field + '_' + name] = value
        if field.startswith('B_'):
            summaries[field] = []
            for cut, count in zip(CUTS, COUNTS):
                sub = np.flatnonzero(np.max(b_k[ix], axis=1) <= cut + 1e-12)
                if len(sub) != count: raise ValueError('B fitting point density changed')
                record = selected_summary(moments, sub)
                record['worst_original_registry_bin'] = int(ix[record['worst_mean_delta_bin_index']])
                record['kmax_B'] = cut; summaries[field].append(record)
        else:
            summaries[field] = selected_summary(moments, np.arange(len(p_k)))
            summaries[field]['kmax_P'] = .08
    out.mkdir(); target = out / 'PAIRED_MEASUREMENTS.npz'
    np.savez_compressed(target, **arrays)
    report = {'status': 'ALL20_RANDOM30_60_PAIRS_VERIFIED_DIAGNOSTICS_NOT_MAIN_COVARIANCE',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
        'CPU_affinity': sorted(os.sched_getaffinity(0)), 'paired_count': 20,
        'realizations': list(range(20)), 'rows': rows, 'summaries': summaries,
        'fixed_reconstruction': baseline['fixed_reconstruction'],
        'all20_post_halos_bitwise_identical': True,
        'statistical_scale': 'baseline nran30 single-box std(ddof=1) from20 pilots, not main C500',
        'SE_definition': 'std(control-minus-baseline,ddof=1)/sqrt(20); uncertainty of numerical-control mean only',
        'diagnostic_SE_changes_likelihood_covariance': False,
        'correlated_bins_or_multiple_comparisons_used_for_significance': False,
        'full_covariance_whitening_done': False, 'fNL_shift_measured': False,
        'random_density_convergence_or_primary_MCMC_released': False,
        'main_volume_Gpc_h3': 1., 'main_covariance_required_count': 500,
        'main_covariance_created': False, 'main_covariance_divided_by_N': False,
        'matrix_sha256': sha(target), 'baseline_audit_sha256': sha(baseline_path),
        'queue_contract_sha256': sha(queue / 'CONTRACT.json'),
        'source_sha256': {str(p): sha(p) for p in (Path(__file__),
            PART / 'code/post_rsd_paired_controls.py', PART / 'code/test_post_rsd_paired_controls.py',
            PART / 'code/audit_sto_post_fid20_pilots.py')}}
    with (out / 'AUDIT.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__': main()
