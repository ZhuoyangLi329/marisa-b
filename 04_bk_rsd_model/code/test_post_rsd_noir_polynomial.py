"""模块大纲：双节点轴的恢复；原有有向post源的非节点回归；显式物理p保护。"""
import unittest
import numpy as np
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_fft_projection import PostFields, integrate_post_design
from post_rsd_noir_polynomial import (B_NODES, P_NODES, fit_noir_design,
    collapse_physical_p, evaluate_noir_design, predict_noir)


class NoIRPolynomialTest(unittest.TestCase):
    def test_tensor_collocation_and_batch_prediction(self):
        rng = np.random.default_rng(2026100905)
        original = rng.normal(size=(5, 3, 3, 3, 7))
        nodes = np.array([[evaluate_noir_design(original, b, p_halo=p) for p in P_NODES] for b in B_NODES])
        fitted = fit_noir_design(nodes)
        np.testing.assert_allclose(fitted, original, rtol=1e-12, atol=4e-15)
        values = np.column_stack((rng.uniform(-500, 500, 13), rng.uniform(.5, 4., 13),
                                  rng.uniform(-4, 6, 13), rng.uniform(-4, 4, 13)))
        p = 1.287
        direct = []
        for f, b, b2, bk in values:
            design = evaluate_noir_design(original, b, p_halo=p)
            direct.append(np.einsum('i,j,ijk->k', [1, b2, bk], [1, f, f*f], design))
        prediction = predict_noir(values, collapse_physical_p(fitted, p_halo=p))
        np.testing.assert_allclose(prediction, direct, rtol=2e-12, atol=2e-9)

    def test_original_oriented_source_at_nonnode_parameters(self):
        rng = np.random.default_rng(2026100906)
        vectors = rng.normal(size=(29, 3, 3)) * .04
        vectors[:, 2] = -vectors[:, :2].sum(axis=1)
        k = np.linalg.norm(vectors, axis=-1)
        power = 9000. / (1. + (k / .07) ** 2)
        transfer = 12000. * (k / .07) ** 2
        reconstruction = FixedReconstruction(10., 2.2280546353635335, 1.2822, 100., 1000./256, 4)
        def direct(b, p):
            return integrate_post_design(PostFields(vectors, power, transfer, b, .7625790037720044,
                p_halo=p, reconstruction=reconstruction))
        nodes = np.array([[direct(b, p) for p in P_NODES] for b in B_NODES])
        coefficients = fit_noir_design(nodes)
        # 数值p坐标不是物理先验；额外测试节点区间外的精确多项式求值。
        for b, p in ((.5, .6), (.71, 1.179), (1.83641, 1.2822), (2.7, 1.51), (4., 2.1)):
            reference = direct(b, p)
            predicted = evaluate_noir_design(coefficients, b, p_halo=p)
            powers = np.array([1., 500., 250000.])[None, :, None]
            scale = np.maximum(1., np.max(np.abs(reference) * powers, axis=(1, 2), keepdims=True))
            np.testing.assert_allclose((predicted - reference) * powers / scale, 0., atol=2e-12)
            values = np.array([[f, b, .43, -.17] for f in (-500., -100., 0., 100., 500.)])
            expected = np.array([np.einsum('i,j,ijk->k', [1., .43, -.17], [1., f, f*f], reference)
                                 for f in values[:, 0]])
            actual = predict_noir(values, collapse_physical_p(coefficients, p_halo=p))
            norm = max(1., float(np.max(np.abs(expected))))
            np.testing.assert_allclose((actual - expected) / norm, 0., atol=2e-12)

    def test_no_implicit_physical_p_or_invalid_axes(self):
        c = np.zeros((5, 3, 3, 3, 2))
        with self.assertRaises(TypeError): collapse_physical_p(c)
        with self.assertRaises(ValueError): collapse_physical_p(c, p_halo=np.nan)
        with self.assertRaises(ValueError): fit_noir_design(c[:4])
        with self.assertRaises(ValueError): predict_noir([[0., 2., 0.]], np.zeros((5, 3, 3, 2)))


if __name__ == '__main__': unittest.main()
