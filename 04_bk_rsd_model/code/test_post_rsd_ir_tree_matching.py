"""模块大纲：原tree零阻尼极限、显式hard余项、线性白化及零Sigma上下文。"""
import unittest
import numpy as np
from post_rsd_ir_tree_matching import tree_matched_signal, identity_damping_context


class TreeMatchingTest(unittest.TestCase):
    def test_exact_original_tree_in_zero_damping_limit(self):
        rng = np.random.default_rng(20261010)
        hard = rng.normal(size=(3, 3, 44))*1e9; strict = hard+rng.normal(size=hard.shape)*1e6
        np.testing.assert_array_equal(tree_matched_signal(hard, strict, strict), hard)

    def test_only_damping_change_added_and_higher_order_wiggle_not_resummed(self):
        hard = np.array([3., -7., 11.]); strict = np.array([2.8, -6.9, 10.5])
        damped = np.array([2.3, -6.5, 9.7])
        np.testing.assert_allclose(tree_matched_signal(hard, strict, damped), damped+(hard-strict), atol=1e-15)
        np.testing.assert_array_equal(tree_matched_signal(hard, strict, damped)-hard, damped-strict)

    def test_matching_commutes_with_full_joint_Cholesky_whitening(self):
        rng = np.random.default_rng(2026101024); a = rng.normal(size=(12, 12))
        chol = np.linalg.cholesky(a@a.T+np.eye(12)); h, u, d = rng.normal(size=(3, 12))
        np.testing.assert_allclose(np.linalg.solve(chol, tree_matched_signal(h, u, d)),
            tree_matched_signal(*[np.linalg.solve(chol, x) for x in (h, u, d)]), rtol=5e-14, atol=2e-15)

    def test_identity_is_not_a_reconstruction_off_or_measured_covariance_claim(self):
        h = identity_damping_context(b1=2.2280546353635335, p_halo=1.31, f_growth=.7625790037720044,
                                     reconstruction=None)
        value = h.damping(np.array([[0., 0., 0.], [.01, .02, -.04]]))
        np.testing.assert_array_equal(value, [[1., 0., 0.], [1., 0., 0.]])
        self.assertTrue(h.physical_parameters['numerical_Sigma_zero_control'])
        self.assertFalse(h.physical_parameters['measured_long_covariance'])
        with self.assertRaises(ValueError): tree_matched_signal([1.], [1., 2.], [1.])
        with self.assertRaises(ValueError): tree_matched_signal([np.nan], [1.], [1.])


if __name__ == '__main__':
    unittest.main()
