"""模块大纲：三组主post-P0的独立统计门；不依赖或放行B随机项。

same-shift的线性Z1与weighted-input相同，P0复用原六模板；独立残余
常数Pshot仍取物理[-1e6,1e6]。N2只做参数单位，不假设实际STO为Poisson。
"""
import numpy as np
from post_rsd_ensemble import TRUTHS, marginal_likelihood_arrays
from post_rsd_noir_case_inputs import case_specs, P_NAMES, B_BOUNDS


def calibration_continuation_gate(record, process_live):
    if process_live:
        return False
    if (record['state'] != 'PHYSICAL_P_CALIBRATION_CONVERGED_NOT_MAIN_MCMC_RELEASE'
            or record['child_pid'] is not None or record['main_MCMC_released']
            or not record.get('summary_sha256')):
        raise ValueError('actual p continuation ended without converged calibration; no retry/release')
    return True


def make_p0_case(ensemble, spec, templates, shot_unit, *, p_halo):
    if spec not in case_specs() or spec.observable != 'P0_only' or spec.P_block != 'P_post':
        raise ValueError('only the three original main post-P0 cases are allowed')
    stats = ensemble['statistics']
    if (stats['n_target_per_truth'] != 100 or stats['covariance_ddof'] != 1
            or stats['covariance_ids'] != list(range(500)) or stats['target_ids'] != list(range(100))
            or stats['overlap_policy'] != 'allow_explicit_target_overlap'):
        raise ValueError('original Fid0..499/target0..99 explicit-overlap mean100 required')
    if not np.isscalar(p_halo) or not np.isfinite(p_halo):
        raise ValueError('explicit converged physical-p calibration required')
    if not np.isscalar(shot_unit) or not np.isfinite(shot_unit) or shot_unit <= 0:
        raise ValueError('positive finite parameter normalization unit required')
    t = np.asarray(templates, dtype=float)
    if t.shape != (6, 12) or not np.isfinite(t).all():
        raise ValueError('six actual-window templates and twelve P bins required')
    first, stop = ensemble['blocks']['P_post']
    if stop-first != 12:
        raise ValueError('original twelve post-P bins required')
    arrays = marginal_likelihood_arrays(ensemble, tuple(range(first, stop)), spec.cosmology)
    bounds = np.vstack((B_BOUNDS[:2], np.array([-1e6, 1e6])/shot_unit))
    arrays.update(templates=t.copy(), noise=np.full(12, shot_unit), bounds=bounds,
                  indices=np.arange(first, stop, dtype=int))
    meta = dict(name=spec.name, cosmology=spec.cosmology, truth_fNL=spec.truth_fNL,
        case_type='P0_only', case_role='main', P_block='P_post', parameters=list(P_NAMES),
        bounds=bounds.tolist(), P_bins=12, B_bins=0, data_bins=12, kmax_P=.08, kmax_B=None,
        n_target=100, n_covariance=500, target_realizations=list(range(100)),
        covariance_realizations=list(range(500)), covariance_ddof=1,
        covariance_is_of_mean=False, covariance_multiplier=1., volume_Gpc_h3=1.,
        Hartlap_factor=arrays['Hartlap_factor'], P_likelihood=True, B_likelihood=False,
        P_B_cross_covariance=False, P_derived_b1_prior=False, p_fixed_physical=float(p_halo),
        physical_p_is_not_p_rec=True, Pshot_physical_bounds=[-1e6, 1e6],
        Pshot_unit_Mpc_h3=float(shot_unit), Pshot_unit_is_not_a_Poisson_closure_assumption=True,
        P_noise_model='independent residual constant, same physical prior as original pre-RSD',
        B_noise_model_released=False, strict_independent_mock_recovery=False,
        IR=False, FoG=False, loops=False,
        statistical_interpretation='mean100 target with C_single500 single-box forecast precision')
    return arrays, meta


def verify_ensemble_arrays(report, matched, nonzero):
    """重算真实700行中的Fid500散布及三组均值；B定义不能影响P。"""
    stats = report['statistics']
    if (report['status'] != 'ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE'
            or len(report['rows']) != 700 or stats['n_covariance'] != 500
            or stats['n_target_per_truth'] != 100 or stats['covariance_is_of_mean']
            or stats['covariance_multiplier'] != 1. or stats['volume_Gpc_h3'] != 1.
            or report['covariance_is_of_mean'] or report['covariance_multiplier'] != 1.
            or report['volume_Gpc_h3'] != 1. or report['covariance_ddof'] != 1):
        raise ValueError('actual complete700 and unaveraged C_single500 V1 required')
    import json
    keys = [tuple(json.loads(str(v))) for v in matched['measurement_keys_json']]
    expected = [('fiducial', i) for i in range(500)]
    expected += [(c, i) for c in ('LC_p', 'LC_m') for i in range(100)]
    if keys != expected or len(set(keys)) != 700:
        raise ValueError('original700 ordered measurement identities required')
    rows = [(r['cosmology'], r['realization']) for r in report['rows']]
    if rows != keys:
        raise ValueError('ensemble rows and audited identities differ')
    matrix = matched['matrix']
    if matrix.shape != (700, 68) or not np.isfinite(matrix).all():
        raise ValueError('original P_input12/P_post12/B44 joint rows required')
    if report['block_slices'] != {'P_weighted_input': [0, 12], 'P_post': [12, 24], 'B_post': [24, 68]}:
        raise ValueError('original joint block registry required')
    np.testing.assert_array_equal(matched['covariance_indices'], np.arange(500))
    np.testing.assert_array_equal(matched['measurement_keys_json'], nonzero['measurement_keys_json'])
    for name in ('matrix', 'target_vectors', 'covariance_single_box'):
        # covariance只比较P-P边缘；不要求不同B估计器的交叉块相同。
        a, b = matched[name], nonzero[name]
        if name == 'covariance_single_box':
            a, b = a[:24, :24], b[:24, :24]
        else:
            a, b = a[..., :24], b[..., :24]
        np.testing.assert_array_equal(a, b)
    covariance = np.cov(matrix[:500], rowvar=False, ddof=1)
    np.testing.assert_allclose(matched['covariance_single_box'], covariance, rtol=2e-13, atol=0.)
    positions = {k: i for i, k in enumerate(keys)}
    targets = np.array([matrix[[positions[c, i] for i in range(100)]].mean(axis=0) for c, _ in TRUTHS])
    np.testing.assert_array_equal(matched['target_vectors'], targets)
    return dict(statistics=stats, blocks=report['block_slices'],
        covariance_single_box=matched['covariance_single_box'],
        targets={c: targets[i] for i, (c, _) in enumerate(TRUTHS)}), keys
