"""模块大纲：独立P0统计/参数单位/真值数量/全目标范围及真实退出保护。

随机数组只做接口单测，绝不生成实际科学审计；原39组fixture用于数值等价。
"""
import json
import unittest
import numpy as np
from post_rsd_ensemble import EnsembleSelection, assemble_blocks
from post_rsd_noir_case_inputs import case_specs, assemble_case
from post_rsd_p0only_inputs import make_p0_case, verify_ensemble_arrays, calibration_continuation_gate


class PostP0InputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(202610100043)
        selection = EnsembleSelection(tuple(range(500)), overlap_policy='allow_explicit_target_overlap')
        cls.rows = {key: dict(P_weighted_input=rng.normal(size=12), P_post=rng.normal(size=12),
                    B_post=rng.normal(size=44)) for key in selection.measurement_keys}
        cls.ensemble = assemble_blocks(selection, cls.rows, dict(P_weighted_input=12, P_post=12, B_post=44))
        cls.specs = [s for s in case_specs() if s.observable == 'P0_only']
        cls.templates = rng.normal(size=(6, 12))

    def datasets(self):
        e = self.ensemble
        matched = dict(matrix=e['matrix'].copy(), covariance_single_box=e['covariance_single_box'].copy(),
            target_vectors=np.array([e['targets'][c] for c in ('fiducial', 'LC_p', 'LC_m')]),
            measurement_keys_json=np.array([json.dumps(k) for k in e['measurement_keys']]),
            covariance_indices=e['covariance_indices'].copy())
        nonzero = {k: v.copy() for k, v in matched.items()}
        nonzero['matrix'][:, 24:] += 123.
        nonzero['target_vectors'][:, 24:] += 123.
        nonzero['covariance_single_box'][24:, 24:] *= 2.
        report = dict(status='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE',
            rows=[dict(cosmology=c, realization=i) for c, i in e['measurement_keys']], statistics=e['statistics'],
            block_slices={k: list(v) for k, v in e['blocks'].items()}, covariance_is_of_mean=False,
            covariance_multiplier=1., volume_Gpc_h3=1., covariance_ddof=1)
        return report, matched, nonzero

    def test_three_are_exactly_original39_main_P0_specs(self):
        self.assertEqual(len(self.specs), 3)
        self.assertEqual([s.truth_fNL for s in self.specs], [0, 100, -100])
        for spec in self.specs:
            arrays, meta = make_p0_case(self.ensemble, spec, self.templates, 3336., p_halo=1.417)
            self.assertEqual(meta['B_bins'], 0)
            self.assertFalse(meta['B_noise_model_released'])
            self.assertEqual(meta['Pshot_physical_bounds'], [-1e6, 1e6])
            self.assertEqual(meta['kmax_P'], .08)
            np.testing.assert_array_equal(arrays['indices'], np.arange(12, 24))

    def test_same_original_P0_numerics_without_any_production_B_audit(self):
        # 只用旧组装器的合成fixture比较，不把此fixture写为真实PASS。
        audit = dict(status='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED', p_fixed_physical=1.417,
                     B_definition='matched', original_binning_preserved=True)
        for spec in self.specs:
            new, _ = make_p0_case(self.ensemble, spec, self.templates, 3336., p_halo=1.417)
            old, _ = assemble_case(self.ensemble, spec, np.zeros((44, 2)), np.zeros((5, 3, 3, 3, 44)),
                self.templates, np.zeros((3, 44)), np.full(12, 3336.), p_halo=1.417, noise_audit=audit)
            for name in ('target', 'covariance', 'chol', 'templates', 'noise', 'bounds', 'indices'):
                np.testing.assert_array_equal(new[name], old[name])
            self.assertEqual(new['Hartlap_factor'], old['Hartlap_factor'])

    def test_Csingle_own_marginal_not_average_or_precision_slice(self):
        arrays, meta = make_p0_case(self.ensemble, self.specs[0], self.templates, 3336., p_halo=1.417)
        full = self.ensemble['covariance_single_box']; ix = arrays['indices']
        np.testing.assert_array_equal(arrays['covariance'], full[np.ix_(ix, ix)])
        self.assertFalse(np.allclose(arrays['covariance'], np.linalg.inv(np.linalg.inv(full)[np.ix_(ix, ix)])))
        np.testing.assert_allclose(arrays['chol']@arrays['chol'].T, arrays['covariance']/(486/499.), rtol=3e-13, atol=1e-14)
        self.assertFalse(meta['covariance_is_of_mean'])
        self.assertEqual(meta['target_realizations'], list(range(100)))
        self.assertEqual(meta['covariance_realizations'], list(range(500)))
        self.assertFalse(meta['strict_independent_mock_recovery'])

    def test_shot_unit_reparameterization_preserves_physical_prior_and_model(self):
        a, _ = make_p0_case(self.ensemble, self.specs[0], self.templates, 3336., p_halo=1.417)
        b, _ = make_p0_case(self.ensemble, self.specs[0], self.templates, 987., p_halo=1.417)
        np.testing.assert_array_equal(a['bounds'][:2], b['bounds'][:2])
        np.testing.assert_array_equal(a['bounds'][2]*3336., [-1e6, 1e6])
        np.testing.assert_array_equal(b['bounds'][2]*987., [-1e6, 1e6])
        np.testing.assert_allclose((4321./3336.)*a['noise'], (4321./987.)*b['noise'], rtol=2e-16)

    def test_independently_recomputes_actual_row_contract_and_ignores_B_choice(self):
        report, matched, nonzero = self.datasets()
        ensemble, keys = verify_ensemble_arrays(report, matched, nonzero)
        self.assertEqual(keys, list(self.ensemble['measurement_keys']))
        np.testing.assert_array_equal(ensemble['covariance_single_box'], self.ensemble['covariance_single_box'])

    def test_corrupt_or_averaged_inputs_rejected(self):
        for mode in ('averaged', 'row_identity', 'P_definition', 'target_mean'):
            report, matched, nonzero = self.datasets()
            if mode == 'averaged':
                matched['covariance_single_box'] /= 100.
                nonzero['covariance_single_box'][:24, :24] /= 100.
            elif mode == 'row_identity':
                report['rows'][0]['realization'] = 499
            elif mode == 'P_definition':
                nonzero['matrix'][0, 12] += .01
            else:
                matched['target_vectors'][0, 12] += .01
                nonzero['target_vectors'][0, 12] += .01
            with self.assertRaises((ValueError, AssertionError)):
                verify_ensemble_arrays(report, matched, nonzero)

    def test_B_PB_default_p_and_invalid_units_not_permitted(self):
        for spec in [s for s in case_specs() if s.observable != 'P0_only']:
            with self.assertRaises(ValueError):
                make_p0_case(self.ensemble, spec, self.templates, 3336., p_halo=1.417)
        with self.assertRaises(TypeError):
            make_p0_case(self.ensemble, self.specs[0], self.templates, 3336.)
        for unit in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                make_p0_case(self.ensemble, self.specs[0], self.templates, unit, p_halo=1.417)

    def test_p_continuation_must_really_exit_and_converge(self):
        done = dict(state='PHYSICAL_P_CALIBRATION_CONVERGED_NOT_MAIN_MCMC_RELEASE', child_pid=None,
                    main_MCMC_released=False, summary_sha256='actual_summary_hash_fixture')
        self.assertFalse(calibration_continuation_gate(done, True))
        self.assertTrue(calibration_continuation_gate(done, False))
        for state in ('WAITING_FOR_SPECIFIC_LOGIN_MEASUREMENT_PROCESS', 'PHYSICAL_P_CALIBRATION_MCMC',
                      'PHYSICAL_P_CALIBRATION_NOT_CONVERGED_REQUIRES_ATTENTION', 'FAILED_PRESERVED_REQUIRES_INSPECTION'):
            with self.assertRaises(ValueError):
                calibration_continuation_gate(dict(done, state=state), False)
        with self.assertRaises(ValueError):
            calibration_continuation_gate(dict(done, child_pid=123), False)


if __name__ == '__main__':
    unittest.main()
