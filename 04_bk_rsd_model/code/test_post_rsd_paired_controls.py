"""模块大纲：配对差符号；单盒/均值误差区分；零差与错误输入回归。"""
import unittest
import numpy as np
from post_rsd_paired_controls import paired_moments, selected_summary


class PairedControlTest(unittest.TestCase):
    def test_constant_shift_preserves_single_box_scale(self):
        base = np.arange(60., dtype=float).reshape(20, 3)
        delta = np.array([1., -2., 3.])
        result = paired_moments(base, base + delta)
        np.testing.assert_allclose(result['paired_mean_delta'], delta)
        np.testing.assert_allclose(result['baseline_single_box_std'], np.std(base, axis=0, ddof=1))
        np.testing.assert_array_equal(result['diagnostic_SE_of_paired_mean'], 0.)
        summary = selected_summary(result, np.array([2, 0]))
        self.assertEqual(summary['worst_mean_delta_bin_index'], 2)
        self.assertFalse(summary['covariance_whitened'])

    def test_independent_manual_sample_formula_and_pairing(self):
        base = np.array([[2., 5.], [4., 3.], [1., 8.], [7., 4.]])
        delta = np.array([[1., -1.], [-2., 2.], [3., 0.], [0., -3.]])
        result = paired_moments(base, base + delta)
        expected_mean = np.sum(delta, axis=0) / 4
        expected_var = np.sum((delta - expected_mean) ** 2, axis=0) / 3
        np.testing.assert_allclose(result['paired_mean_delta'], expected_mean)
        np.testing.assert_allclose(result['paired_delta_std'] ** 2, expected_var)
        np.testing.assert_allclose(result['diagnostic_SE_of_paired_mean'] ** 2, expected_var / 4)
        reverse = paired_moments(base, (base + delta)[::-1])
        self.assertFalse(np.allclose(reverse['paired_delta_std'], result['paired_delta_std']))

    def test_zero_control_and_input_guards(self):
        base = np.arange(8., dtype=float).reshape(4, 2)
        result = paired_moments(base, base)
        summary = selected_summary(result, np.array([0, 1]))
        self.assertEqual(summary['max_abs_mean_delta_over_single_box_std'], 0.)
        self.assertEqual(summary['max_diagnostic_SE_over_single_box_std'], 0.)
        for a, b in ((base, base[:-1]), (base[:1], base[:1]), (base * np.nan, base),
                     (np.ones_like(base), base)):
            with self.assertRaises(ValueError): paired_moments(a, b)
        for ix in ([], [0, 0], [0., 1.], [-1], [2]):
            with self.assertRaises(ValueError): selected_summary(result, ix)


if __name__ == '__main__': unittest.main()
