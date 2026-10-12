"""模块大纲：八解析响应与原候选预测独立secant比较及条件零响应回归。"""
import unittest
import numpy as np
from post_rsd_noise_identifiability import noise_response, response_diagnostics
from test_post_rsd_eft_candidate_model_v2 import CandidateV2Test


class NoiseResponseTests(unittest.TestCase):
    def setUp(self):
        source = CandidateV2Test()
        source.setUp()
        self.model = source.model
        self.row = source.rows[0].copy()

    def test_exact_response_against_original_model_for_all_noise_columns(self):
        for fnl in (0., -100., 100.):
            row = self.row.copy()
            row[0] = fnl
            jac = noise_response(self.model, row)
            for index in range(8):
                minus, plus = row.copy(), row.copy()
                minus[index + 4] -= .125
                plus[index + 4] += .125
                secant = (self.model.predictions(plus)[0] - self.model.predictions(minus)[0]) / .25
                np.testing.assert_allclose(jac[:, index], secant, rtol=2e-10, atol=.03)

    def test_png_noise_directions_exactly_zero_when_fnl_fixed_zero(self):
        row = self.row.copy()
        row[0] = 0.
        jac = noise_response(self.model, row)
        np.testing.assert_array_equal(jac[:, [1, 5]], np.zeros((21, 2)))
        self.assertGreater(np.max(abs(jac[12:, 0])), 0)
        self.assertGreater(np.max(abs(jac[12:, 4])), 0)

    def test_raw_noise_P0_derivative_cancels_independent_Pshot_remains(self):
        jac = noise_response(self.model, self.row)
        np.testing.assert_array_equal(jac[:12, :7], np.zeros((12, 7)))
        np.testing.assert_array_equal(jac[:12, 7], np.full(12, self.model.weight_moments[0]))

    def test_full_cross_whitening_and_invalid_covariance_rejection(self):
        rng = np.random.default_rng(76)
        a = rng.normal(size=(21, 21))
        cov = a @ a.T + np.eye(21)
        chol = np.linalg.cholesky(cov)
        jac = noise_response(self.model, self.row)
        result = response_diagnostics(jac, chol, np.ones(8))
        np.testing.assert_allclose(result['local_derivative_norm_times_full_prior_width'],
                                   np.linalg.norm(np.linalg.solve(chol, jac), axis=0))
        with self.assertRaises(ValueError):
            response_diagnostics(jac, chol.T, np.ones(8))


if __name__ == '__main__':
    unittest.main()
