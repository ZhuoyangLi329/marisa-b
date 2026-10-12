"""模块大纲：mean100标签、两种B的P一致性、signed self矩及缺样本保护。"""
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_target300_summary import TARGET_KEYS, summarize_complete_targets


class TargetSummaryTest(unittest.TestCase):
    def setUp(self):
        self.matched = np.arange(300*68, dtype=float).reshape(300, 68)
        self.nonzero = self.matched.copy(); self.nonzero[:, 24:] += 7.
        self.moments = np.column_stack((1.+np.arange(300)/100, np.arange(300)-150., np.full(300, 20.)))

    def test_exact_three_mean100_vectors_and_no_covariance(self):
        with patch('numpy.cov', side_effect=AssertionError('target audit must not create covariance')):
            arrays, stats = summarize_complete_targets(TARGET_KEYS, self.matched, self.nonzero, self.moments)
        self.assertEqual(len(stats), 3)
        for i in range(3):
            expected = sum(self.matched[i*100+j] for j in range(100))/100.
            np.testing.assert_array_equal(arrays['matched_target_vectors'][i], expected)
        self.assertFalse(any('covariance' in k for k in arrays))
        np.testing.assert_array_equal(arrays['matched_target_vectors'][:, :24],
                                      arrays['nonzero_diagnostic_target_vectors'][:, :24])

    def test_signed_N3_and_mean_square_not_square_mean(self):
        _, stats = summarize_complete_targets(TARGET_KEYS, self.matched, self.nonzero, self.moments)
        self.assertLess(stats[0]['mean_N3'], 0.)
        for i, row in enumerate(stats):
            n2 = self.moments[i*100:(i+1)*100, 0]
            self.assertAlmostEqual(row['mean_N2_squared'], sum(x*x for x in n2)/100., places=12)
            self.assertGreater(row['mean_N2_squared'], row['square_mean_N2'])
            self.assertTrue(row['self_moments_are_not_raw_STO_stochastic_power'])

    def test_missing_duplicate_wrong_identity_and_bool_rejected(self):
        for keys in (TARGET_KEYS[:-1], TARGET_KEYS[:-1]+(TARGET_KEYS[0],), tuple(reversed(TARGET_KEYS)),
                     TARGET_KEYS[:1]+(('fiducial', True),)+TARGET_KEYS[2:]):
            with self.assertRaises(ValueError):
                summarize_complete_targets(keys, self.matched, self.nonzero, self.moments)
        with self.assertRaises(ValueError):
            summarize_complete_targets(TARGET_KEYS, self.matched[:-1], self.nonzero[:-1], self.moments[:-1])

    def test_geometry_P_drift_and_invalid_moments_rejected(self):
        drift = self.nonzero.copy(); drift[5, 0] += 1.
        with self.assertRaises(AssertionError):
            summarize_complete_targets(TARGET_KEYS, self.matched, drift, self.moments)
        for index, value in ((0, 0.), (1, np.nan), (2, 2.3)):
            moments = self.moments.copy(); moments[0, index] = value
            with self.assertRaises(ValueError):
                summarize_complete_targets(TARGET_KEYS, self.matched, self.nonzero, moments)


if __name__ == '__main__':
    unittest.main()
