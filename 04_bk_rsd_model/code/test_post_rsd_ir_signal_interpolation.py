"""模块大纲：独立barycentric/非多项式oracle、有限PNG、完整cross白化及保护。"""
import unittest
import numpy as np
from post_rsd_ir_signal_interpolation import bias_nodes, IRSignalInterpolation


def direct_design(b):
    b = np.atleast_1d(b)
    # 显式非四次函数；只认证表示算法，不冒充真实44-bin物理验收。
    base = np.exp(-.27*(b-1.1)**2)*(1+.13*b**4)
    return base[:, None, None, None]*np.arange(1., 28.).reshape(1, 3, 3, 3)


def independent_barycentric(nodes, samples, x):
    # 直接节点积权重，不调用DCT、chebval或生产Clenshaw。
    weights = np.array([1/np.prod(node-np.delete(nodes, i)) for i, node in enumerate(nodes)])
    results = []
    for value in x:
        exact = np.flatnonzero(value == nodes)
        if len(exact):
            results.append(samples[exact[0]])
        else:
            ratio = weights/(value-nodes)
            results.append(np.tensordot(ratio, samples, axes=(0, 0))/ratio.sum())
    return np.asarray(results)


class IRInterpolationTest(unittest.TestCase):
    def test_matches_independent_node_product_interpolation(self):
        for n in (5, 9, 17, 33):
            nodes = bias_nodes(n); samples = direct_design(nodes)
            model = IRSignalInterpolation.from_nodes(samples, p_halo=1.417)
            probes = np.r_[nodes, [.5, .71, 1.41, 2.2280546353635335, 3.83, 4.]]
            expected = independent_barycentric(nodes, samples, probes)
            np.testing.assert_allclose(model.designs_at(probes), expected, rtol=2e-12, atol=2e-12)

    def test_nonquartic_convergence_with_unchanged_domain(self):
        probes = np.linspace(.5, 4., 61)
        errors = []
        for n in (5, 9, 17, 33):
            model = IRSignalInterpolation.from_nodes(direct_design(bias_nodes(n)), p_halo=1.417)
            errors.append(np.max(np.abs(model.designs_at(probes)-direct_design(probes))))
        self.assertGreater(errors[0], .1)
        self.assertLess(errors[2], errors[1]*1e-3)
        self.assertLess(errors[3], 1e-10)

    def test_exact_fnl_quadratic_and_native_bias_affine(self):
        model = IRSignalInterpolation.from_nodes(direct_design(bias_nodes(33)), p_halo=1.417)
        rows = np.array([[f, b, b2, bk] for b in (.5, 1.7, 4.)
            for f in (-500., -100., 0., 17., 500.) for b2, bk in ((-4., 4.), (6., -4.), (0., 0.))])
        expected = []
        for f, b, b2, bk in rows:
            d = direct_design(b)[0]
            expected.append(sum(d[a, t]*([1., b2, bk][a])*([1., f, f*f][t])
                                for a in range(3) for t in range(3)))
        np.testing.assert_allclose(model.predictions(rows), expected, rtol=3e-13, atol=1e-8)
        values = model.predictions([[f, 2.1, .3, -.2] for f in (-3., -1., 1., 3.)])
        np.testing.assert_allclose(values[3]-3*values[2]+3*values[1]-values[0], 0., atol=2e-12)

    def test_full_joint_whitening_not_separate_P_B(self):
        rng = np.random.default_rng(20261010)
        matrix = rng.normal(size=(15, 15)); covariance = matrix@matrix.T+np.eye(15)
        chol = np.linalg.cholesky(covariance)
        model = IRSignalInterpolation.from_nodes(direct_design(bias_nodes(17)), p_halo=1.417)
        white = model.joint_whitened(chol, nP=12)
        rows = np.array([[73., .81, .3, -.1], [-91., 3.7, -2., 1.]])
        joint = np.column_stack((np.zeros((len(rows), 12)), model.predictions(rows)))
        np.testing.assert_allclose(white.predictions(rows), np.linalg.solve(chol, joint.T).T,
                                   rtol=5e-13, atol=1e-8)
        self.assertFalse(np.allclose(chol[12:, 12:], np.linalg.cholesky(covariance[12:, 12:])))

    def test_explicit_parameter_and_no_release_guards(self):
        model = IRSignalInterpolation.from_nodes(direct_design(bias_nodes(9)), p_halo=1.417)
        self.assertFalse(model.metadata['MCMC_released'])
        self.assertFalse(model.metadata['exact_noIR_quartic_assumed'])
        self.assertFalse(model.coefficients.flags.writeable)
        for b in (.499, 4.001, np.nan):
            with self.assertRaises(ValueError): model.designs_at(b)
        for f in (-501., 501., np.nan):
            with self.assertRaises(ValueError): model.predictions([[f, 2., 0., 0.]])
        for n in (True, 1, 3.1):
            with self.assertRaises(ValueError): bias_nodes(n)
        with self.assertRaises(TypeError): IRSignalInterpolation.from_nodes(direct_design(bias_nodes(9)))
        with self.assertRaises(ValueError): IRSignalInterpolation(model.coefficients, np.nan)
        with self.assertRaises(ValueError): model.joint_whitened(np.ones((3, 3)))


if __name__ == '__main__':
    unittest.main()
