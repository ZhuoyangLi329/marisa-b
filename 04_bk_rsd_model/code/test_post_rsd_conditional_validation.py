"""模块大纲：独立精度矩阵/显式二维条件公式；失败C不能被数值修补。"""
import unittest
import numpy as np
from post_rsd_conditional_validation import conditional_residual


class ConditionalValidationTests(unittest.TestCase):
    def test_correlated_two_dimensional_formula(self):
        c = np.array([[4., 1.2], [1.2, 9.]])
        r = np.array([2., 5.])
        a = (205 - 2 - 2) / 204
        out = conditional_residual(r, c, np.array([0]), 205)
        expected = (5. - 1.2 / 4. * 2.) / np.sqrt((9. - 1.2**2 / 4.) / a)
        self.assertAlmostEqual(out['conditional_whitened_residual'][0], expected, places=13)
        self.assertNotAlmostEqual(out['conditional_single_box_norm'], 5. / np.sqrt(9. / a), places=5)

    def test_noncontiguous_columns_independent_precision(self):
        rng = np.random.default_rng(541)
        x = rng.normal(size=(7, 7)); c = x @ x.T + np.eye(7); r = rng.normal(size=7)
        fit = np.array([5, 0, 2]); held = np.array([1, 3, 4, 6]); a = (205 - 7 - 2) / 204
        precision = np.linalg.inv(c / a)  # 独立小型测试oracle；production无inv/pinv。
        expected = r[held] + np.linalg.solve(precision[np.ix_(held, held)],
            precision[np.ix_(held, fit)] @ r[fit])
        out = conditional_residual(r, c, fit, 205)
        np.testing.assert_allclose(out['conditional_residual'], expected, rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(out['full_joint_single_box_norm']**2, r @ precision @ r, places=12)

    def test_invalid_or_nonpositive_covariance_rejected(self):
        with self.assertRaises(np.linalg.LinAlgError):
            conditional_residual([1., 2.], np.array([[1., 2.], [2., 1.]]), np.array([0]), 205)
        for cols, count in [(np.array([0, 0]), 205), (np.array([.0]), 205), (np.array([0]), 4)]:
            with self.assertRaises(ValueError):
                conditional_residual([1., 2.], np.eye(2), cols, count)


if __name__ == '__main__':
    unittest.main()
