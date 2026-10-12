"""模块大纲：新增统计/P0回归；重放旧800盒及21个P/PB似然数组，旧文件只读。

此处旧pre数据仅验收统计组装算法，不替代待测post或匹配的新pre控制。
不拟合p、不运行MCMC、不选择新的生产样本编号。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
from post_rsd_ensemble import EnsembleSelection, assemble_blocks, marginal_likelihood_arrays


PART = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError('preserve prior statistics audit')
    suite = unittest.TestSuite()
    for name in ('test_post_rsd_ensemble', 'test_post_rsd_p0_model'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 7:
        raise RuntimeError('required seven implementation regressions did not pass')
    out = PART / 'outputs'
    pb = out / 'pre_rsd_pb_mean100_v1'
    old_audit = json.loads((pb / 'JOINT_FREEZE_AUDIT.json').read_text())
    pfile = pb / 'P0_ENSEMBLE_MATRIX.npz'
    bfile = out / 'pre_rsd_b000_bonly_v2/B000_ENSEMBLE_MATRIX.npz'
    if (old_audit['status'] != 'PASS' or sha(pfile) != old_audit['P0_ensemble_sha256']
            or sha(bfile) != old_audit['B_ensemble_sha256']):
        raise RuntimeError('frozen pre matrix provenance mismatch')
    with np.load(pfile, allow_pickle=False) as data:
        p = data['P0'].copy(); old_cov = data['covariance_single_box'].copy()
        pm = [json.loads(str(x)) for x in data['metadata_json']]
        b_indices = data['B_indices'].copy()
    with np.load(bfile, allow_pickle=False) as data:
        b = data['B'].copy(); bm = [json.loads(str(x)) for x in data['metadata_json']]
        np.testing.assert_array_equal(b_indices, data['certified_indices'])
    if p.shape != (800, 12) or b.shape != (800, 120) or len(pm) != 800 or len(bm) != 800:
        raise ValueError('incomplete existing pre ensemble')
    rows = {}
    for pp, bb, mp, mb in zip(p, b, pm, bm):
        for field in ('cosmology', 'realization', 'source_raw_sha256', 'random_seed', 'role'):
            if mp[field] != mb[field]:
                raise ValueError('same-box P/B alignment failed: ' + field)
        key = (mp['cosmology'], mp['realization'])
        if key in rows:
            raise ValueError('duplicate pre measurement key')
        rows[key] = {'P': pp, 'B': bb[b_indices]}
    selection = EnsembleSelection(tuple(range(1000, 1500)))
    stats = assemble_blocks(selection, rows, {'P': 12, 'B': len(b_indices)})
    np.testing.assert_allclose(stats['covariance_single_box'], old_cov, rtol=2e-13)
    cases = []; b_marginals = 0
    input_hashes = {str(pfile): sha(pfile), str(bfile): sha(bfile)}
    for kind, directory in (('PB', pb), ('P', out / 'pre_rsd_p0only_mean100_v1')):
        paths = sorted((directory / 'cases').glob('*/contract.json'))
        count = 0
        for path in paths:
            # 只读已完整缓存的21个P/PB主例；B-only原似然本地缓存不全，
            # 不能把从PB抽出的B边缘检查冒充重放18份独立的B-only结果文件。
            if '_mean100_' not in path.parent.name or '_wide_' in path.parent.name:
                continue
            meta = json.loads(path.read_text())
            likelihood = path.parent / 'likelihood.npz'
            if sha(likelihood) != meta['likelihood_sha256']:
                raise ValueError('old likelihood hash mismatch: ' + str(likelihood))
            with np.load(likelihood, allow_pickle=False) as old:
                if kind == 'P':
                    indices = np.arange(12)
                else:
                    positions = np.array([int(np.flatnonzero(b_indices == i)[0]) for i in old['indices']])
                    indices = 12 + positions
                    if kind == 'PB': indices = np.r_[np.arange(12), indices]
                case = marginal_likelihood_arrays(stats, indices, meta['cosmology'])
                for field in ('target', 'covariance'):
                    np.testing.assert_allclose(case[field], old[field], rtol=3e-13, atol=1e-10)
                if case['Hartlap_factor'] != meta['Hartlap_factor']:
                    raise ValueError('dimension-specific Hartlap drift')
                # macOS/Linux LAPACK可在接近零的L元素出现较大相对差；按每个
                # observable的sigma归一，并直接验证L L^T，不能以元素相对误差冒充统计差异。
                sigma = np.sqrt(np.diag(case['covariance']) / case['Hartlap_factor'])
                chol_error = float(np.max(np.abs(case['chol'] - old['chol']) / sigma[:, None]))
                residual = case['chol'] @ case['chol'].T - case['covariance'] / case['Hartlap_factor']
                chol_residual = float(np.max(np.abs(residual) / np.outer(sigma, sigma)))
                if max(chol_error, chol_residual) > 1e-12:
                    raise ValueError('sigma-scaled Cholesky parity or factorization failed')
                if kind == 'PB':
                    bmarg = marginal_likelihood_arrays(stats, indices[12:], meta['cosmology'])
                    np.testing.assert_allclose(bmarg['target'], old['target'][12:], rtol=3e-13)
                    np.testing.assert_allclose(bmarg['covariance'], old['covariance'][12:, 12:], rtol=3e-13)
                    b_marginals += 1
            input_hashes[str(path)] = sha(path); input_hashes[str(likelihood)] = sha(likelihood)
            cases.append({'kind': kind, 'case': path.parent.name, 'dimension': len(indices),
                          'Hartlap_factor': case['Hartlap_factor'],
                          'sigma_scaled_chol_max_abs_difference': chol_error,
                          'sigma_scaled_chol_factorization_residual': chol_residual}); count += 1
        if count != (3 if kind == 'P' else 18):
            raise ValueError(f'incomplete existing {kind} case set: {count}')
    for path, expected in input_hashes.items():
        if sha(path) != expected:
            raise ValueError('read-only control source changed during audit')
    source_names = ('post_rsd_ensemble.py', 'test_post_rsd_ensemble.py',
                    'post_rsd_p0_model.py', 'test_post_rsd_p0_model.py', Path(__file__).name)
    report = {'status': 'STATISTICS_AND_P0_IMPLEMENTATION_PASS_NO_POST_ENSEMBLE',
              'created_UTC': datetime.now(timezone.utc).isoformat(),
              'unit_tests_passed': result.testsRun, 'pre_control_boxes': 800,
              'pre_likelihoods_replayed': len(cases), 'cases': cases,
              'B_marginals_checked_against_joint_covariance_blocks': b_marginals,
              'saved_B_only_likelihoods_replayed': False,
              'full_same_box_PB_cross_covariance_reproduced': True,
              'P_only_uses_marginal_C_PP_not_conditional_precision': True,
              'volume_Gpc_h3': 1., 'covariance_multiplier': 1., 'covariance_is_of_mean': False,
              'overlap700_and_disjoint800_software_paths_tested': True,
              'new_production_sample_selected': False, 'post_covariance_or_MCMC_created': False,
              'physical_p_calibrated': False, 'old_pre_outputs_modified': False,
              'input_sha256': input_hashes,
              'source_sha256': {name: sha(PART / 'code' / name) for name in source_names},
              'python': sys.executable, 'platform': platform.platform(), 'numpy_version': np.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'input_sha256')}, indent=2))


if __name__ == '__main__': main()
