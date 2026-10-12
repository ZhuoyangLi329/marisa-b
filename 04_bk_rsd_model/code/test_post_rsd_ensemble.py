"""模块大纲：700/800实体样本；未缩放joint协方差；子块而非条件precision。

只验证统计组装实现，不冒充真实STO样本、生产验收或MCMC结果。
"""
import unittest
import numpy as np
from post_rsd_ensemble import EnsembleSelection, assemble_blocks, marginal_likelihood_arrays


class EnsembleStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.selection = EnsembleSelection(tuple(range(500)),
            overlap_policy='allow_explicit_target_overlap')
        rng = np.random.default_rng(2026100902)
        self.rows = {}
        for key in self.selection.measurement_keys:
            p = rng.normal(size=12)
            self.rows[key] = {'P_post': p, 'B_post': p[:5] * 3 + rng.normal(size=5)}
        self.stats = assemble_blocks(self.selection, self.rows, {'P_post': 12, 'B_post': 5})

    def test_sample_roles_and_explicit_overlap(self):
        self.assertEqual(len(self.selection.measurement_keys), 700)
        self.assertEqual(self.selection.overlap, tuple(range(100)))
        self.assertFalse(self.stats['statistics']['validation_catalogs_disjoint_from_covariance'])
        separate = EnsembleSelection(tuple(range(1000, 1500)))
        self.assertEqual(len(separate.measurement_keys), 800)
        self.assertEqual(separate.overlap, ())
        with self.assertRaises(ValueError): EnsembleSelection(tuple(range(500)))
        with self.assertRaises(ValueError): EnsembleSelection(tuple(range(499)))
        with self.assertRaises(ValueError): EnsembleSelection((0,) * 500)

    def test_single_box_covariance_and_full_PB_cross_block(self):
        vectors = np.array([np.r_[self.rows['fiducial', i]['P_post'], self.rows['fiducial', i]['B_post']]
                            for i in range(500)])
        centered = vectors - vectors.mean(axis=0)
        direct = centered.T @ centered / 499.
        np.testing.assert_allclose(self.stats['covariance_single_box'], direct, rtol=2e-14, atol=1e-14)
        self.assertGreater(np.max(np.abs(direct[:12, 12:])), 2.)
        self.assertFalse(np.allclose(self.stats['covariance_single_box'], direct / 100))
        self.assertFalse(np.allclose(self.stats['covariance_single_box'], direct / 500))
        np.testing.assert_array_equal(self.stats['targets']['fiducial'], vectors[:100].mean(axis=0))

    def test_permuted_storage_does_not_mispair_boxes(self):
        alternate = assemble_blocks(self.selection, dict(reversed(list(self.rows.items()))),
                                    {'P_post': 12, 'B_post': 5})
        np.testing.assert_array_equal(alternate['matrix'], self.stats['matrix'])
        np.testing.assert_array_equal(alternate['covariance_single_box'], self.stats['covariance_single_box'])
        with self.assertRaises(ValueError): assemble_blocks(self.selection, {}, {'P_post': 12})

    def test_marginal_covariance_not_conditional_precision(self):
        full = self.stats['covariance_single_box']
        for indices in (tuple(range(12)), tuple(range(12, 17)), tuple(range(17))):
            case = marginal_likelihood_arrays(self.stats, indices, 'fiducial')
            np.testing.assert_array_equal(case['covariance'], full[np.ix_(indices, indices)])
            np.testing.assert_allclose(case['chol'] @ case['chol'].T,
                                       case['covariance'] / case['Hartlap_factor'], rtol=3e-14, atol=1e-14)
            self.assertEqual(case['Hartlap_factor'], (498 - len(indices)) / 499.)
        conditional = np.linalg.inv(np.linalg.inv(full)[:12, :12])
        self.assertGreater(np.max(np.abs(conditional - full[:12, :12])), .5)

    def test_nonfinite_or_malformed_arrays_are_rejected(self):
        key = self.selection.measurement_keys[0]
        self.rows[key]['P_post'][0] = np.nan
        with self.assertRaises(ValueError): assemble_blocks(self.selection, self.rows, {'P_post': 12})
        with self.assertRaises(ValueError): marginal_likelihood_arrays(self.stats, (-1,), 'fiducial')
        with self.assertRaises(ValueError): marginal_likelihood_arrays(self.stats, (0, 0), 'fiducial')
        with self.assertRaises(ValueError): marginal_likelihood_arrays(self.stats, (0.5,), 'fiducial')


if __name__ == '__main__': unittest.main()
