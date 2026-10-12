"""模块大纲：独立Gaussian Wick/F2均值、colored shell枚举、非Poisson精确矩。

不把指定cluster模型当成实际STO；source N(k)必须以后另行校准。
"""
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_general_noise_transport import (direct_colored_transport, combine_white_operators,
    GeneralNoisePointFields, GeneralNoiseFFTProjection, integrate_general_noise_transport)
from post_rsd_noise_transport import integrate_noise_transport
from post_rsd_poisson_projection import PoissonShiftFields, integrate_poisson_shift_design
from post_rsd_clustered_moments import ClusteredMarkedMoments
from post_rsd_marked_moments import factorial_shift_response
import rsd_b000_fft_projection as fft_source


def sample(k):
    return 10000./(1+(k/.09)**2), 1000.*(1+(k/.02)**2)


def colored(v):
    k2 = np.sum(v*v, axis=-1)
    mu2 = np.divide(v[..., 2]**2, k2, out=np.zeros_like(k2), where=k2 > 0)
    return 710.*(1+.7*mu2+k2/.04**2)


def powers_and_response(v, p, m, b, fg, ph, rec, fnl):
    k2 = np.sum(v*v, axis=-1); active = k2 > 0
    inv = np.divide(1., k2, out=np.zeros_like(k2), where=active)
    z = b+fg*v[..., 2]**2*inv+fnl*2*1.686*(b-ph)/m
    response = np.zeros_like(k2)
    if rec is not None:
        response[active] = rec.response(v[active], m[active])
    return p*z*z, response


def gaussian_pair_response(v, r, D, N, n2, nrandom):
    """手写三种Wick pairing和F2零模均值，不调用production点算子。

只是Gaussian-pair/tree closure；F2 fluctuation的connected3/4在这里不认证。
"""
    v = np.asarray(v, dtype=int); modes = np.concatenate((v, -v)); rs = np.tile(r, 2)
    lookup = {tuple(k): i for i, k in enumerate(v)}
    lookup.update({tuple(-k): i for i, k in enumerate(v)})
    def covariance(a, b):
        ka, kb = np.asarray(a[2]), np.asarray(b[2])
        if np.any(ka+kb) or not np.any(ka):
            return 0.
        index = lookup[tuple(ka)]
        return D[index]+N[index]+(nrandom if a[0] == b[0] == 'F' else 0.)
    def expect(*items):
        if any(kind == 'F' and power == 1 and mode == (0, 0, 0) for kind, power, mode in items):
            return 0.
        if len(items) == 4:
            return sum(covariance(items[a], items[b])*covariance(items[c], items[d])
                for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)))
        if len(items) == 3:
            double = next(i for i, (_, power, _) in enumerate(items) if power == 2)
            if items[double][2] != (0, 0, 0):
                return 0.
            other = [item for i, item in enumerate(items) if i != double]
            return (n2+nrandom)*covariance(*other)
        raise ValueError('only the specified Gaussian-pair response audit')
    return factorial_shift_response(v, modes, rs, expect)[0]


def clustered_leading_rows():
    v = np.array([[1, 0, 1], [0, 2, -1], [-1, -2, 0]]); r = np.array([.71, .42, .26])
    q = np.concatenate((v, -v, [[1, -1, 2], [-1, 1, -2]])); rq = np.r_[r, r, .18, .18]
    weights = [[.6, 1.9], [1.2, -.3]]
    offsets = [[[0, 0, 0], [.011, .017, -.023]], [[.02, 0, 0], [-.007, .013, .019]]]
    rows = []
    for leg in range(3):
        for nran in (2, 30):
            for n in (1000, 1000000):
                counts = [2*n//5, 3*n//5]
                mom = ClusteredMarkedMoments(counts, weights, offsets, [.016, .035], v[leg], 2*n*nran)
                uniform = ClusteredMarkedMoments(counts, weights, offsets, [0., 0.], v[leg], 2*n*nran)
                exact = factorial_shift_response(v, q, rq, mom.expect)[0]
                pure = factorial_shift_response(v, q, rq, uniform.expect)[0]
                noise = np.array([sum(counts[g]*abs(mom.child_factor(g, 1, tuple(k)))**2
                    for g in range(mom.ngroup)) for k in v])
                D = np.zeros(3)
                D[leg] = abs(sum(counts[g]*mom.amplitudes[g]/2*mom.child_factor(g, 1, tuple(v[leg]))
                    for g in range(mom.ngroup)))**2
                candidate_uniform = direct_colored_transport(v, np.zeros(3), noise, r, mom.N2)
                candidate = direct_colored_transport(v, D, noise, r, mom.N2)-candidate_uniform
                rows.append(dict(signal_leg=leg, nran=nran, nparents=n, data_raw_noise=noise.tolist(),
                    actual_child_self_N2=mom.N2, signal_power=D.tolist(),
                    exact_signal_response=[float((exact-pure).real), float((exact-pure).imag)],
                    candidate_signal_response=float(candidate),
                    signal_relative_error=float(abs(exact-pure-candidate)/max(abs(exact-pure), 1e-30)),
                    uniform_relative_error=float(abs(pure-candidate_uniform)/max(abs(pure), 1e-30))))
    return rows


class GeneralNoiseTest(unittest.TestCase):
    def setUp(self):
        self.rec = FixedReconstruction(10., 2.2280546354, 1.2822, 100.)
        self.fg = .7625790037720044; self.b = 2.11; self.ph = 1.417

    def test_Gaussian_pair_F2_mean_and_finite_random_cancellation_independent(self):
        v = np.array([[1, 0, 1], [0, 2, -1], [-1, -2, 0]])
        r = np.array([.71, .42, .26]); D = np.array([.023, .041, .009]); N = np.array([.007, .011, .004])
        expected = direct_colored_transport(v, D, N, r, .009)
        for nr in (0., .0003, .01):
            full = gaussian_pair_response(v, r, D, N, .009, nr)
            deterministic = gaussian_pair_response(v, r, D, np.zeros(3), 0., 0.)
            np.testing.assert_allclose(full-deterministic, expected, rtol=2e-14, atol=1e-18)

    def test_white_operator_combination_and_Poisson_limit(self):
        rng = np.random.default_rng(202610100205)
        v = rng.normal(size=(19, 3, 3))*.025; v[:, 2] = -v[:, :2].sum(axis=1)
        p, m = sample(np.linalg.norm(v, axis=-1))
        original = PoissonShiftFields(v, p, m, self.b, self.fg, p_halo=self.ph, reconstruction=self.rec)
        ops = integrate_noise_transport(original)
        for nraw in (0., 1800., 3336., 4300.):
            new = GeneralNoisePointFields(v, p, m, self.b, self.fg, p_halo=self.ph,
                reconstruction=self.rec, noise_data_power=nraw, noise_is_data_only=True)
            actual = integrate_general_noise_transport(new, n2_data_self=3336.)
            expected = combine_white_operators(ops, noise_data_raw=nraw, n2_data_self=3336.)
            scale = np.maximum(1., np.max(np.abs(expected), axis=1, keepdims=True))
            np.testing.assert_allclose((actual-expected)/scale, 0., atol=5e-14)
        np.testing.assert_allclose(combine_white_operators(ops, noise_data_raw=3336., n2_data_self=3336.),
            integrate_poisson_shift_design(original, 3336.), rtol=2e-14, atol=1e-8)

    def test_oriented_colored_power_retains_physical_fnl_squared(self):
        rng = np.random.default_rng(202610100206)
        v = rng.normal(size=(31, 3, 3))*.025; v[:, 2] = -v[:, :2].sum(axis=1)
        p, m = sample(np.linalg.norm(v, axis=-1)); noise = colored(v)
        fields = GeneralNoisePointFields(v, p, m, self.b, self.fg, p_halo=self.ph,
            reconstruction=self.rec, noise_data_power=noise, noise_is_data_only=True)
        design = integrate_general_noise_transport(fields, n2_data_self=3336.)
        for fnl in (-499., 0., 100., 473.):
            d, r = powers_and_response(v, p, m, self.b, self.fg, self.ph, self.rec, fnl)
            expected = direct_colored_transport(v, d, noise, r, 3336.)
            actual = np.array([1., fnl, fnl*fnl])@design
            scale = max(1., float(np.max(np.abs(expected))))
            np.testing.assert_allclose((actual-expected)/scale, 0., atol=5e-14)

    def test_exact_shell_projection_full_denominator_and_zero_policy(self):
        n = 32; edges = np.array([0., .02035714285714286, .04107142857142857]); kf = 2*np.pi/1000
        axis = np.rint(np.fft.fftfreq(n)*n).astype(int)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
        norm = np.linalg.norm(grid, axis=-1)*kf
        shells = [grid[(norm >= edges[0]) & (norm < edges[1])], grid[(norm >= edges[1]) & (norm <= edges[2])]]
        with patch.object(fft_source, 'load_spectrum', return_value=sample):
            for pair in ((0, 0), (0, 1), (1, 1)):
                a, b = (shells[i] for i in pair); aa = np.repeat(a, len(b), axis=0); bb = np.tile(b, (len(a), 1))
                v = np.stack((aa, bb, -aa-bb), axis=1)*kf; active = np.linalg.norm(v, axis=-1) > 0
                p = np.zeros(active.shape); m = np.ones(active.shape)
                p[active], m[active] = sample(np.linalg.norm(v[active], axis=-1))
                field = GeneralNoiseFFTProjection(n, edges, pair, self.b, self.fg, p_halo=self.ph,
                    reconstruction=self.rec, noise_data_power=colored, noise_is_data_only=True)
                design = integrate_general_noise_transport(field, n2_data_self=3336.)
                self.assertEqual(field.denominator, len(a)*len(b))
                for fnl in (-171., 0., 243.):
                    d, r = powers_and_response(v, p, m, self.b, self.fg, self.ph, self.rec, fnl)
                    expected = direct_colored_transport(v, d, colored(v), r, 3336.).mean()
                    actual = design@np.array([1., fnl, fnl*fnl])
                    np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=2e-7)

    def test_nonPoisson_colored_signed_cluster_leading_limit(self):
        rows = clustered_leading_rows()
        for row in rows:
            if row['nparents'] == 1000000:
                self.assertLess(max(row['signal_relative_error'], row['uniform_relative_error']), 1e-4)
        for leg in range(3):
            for nran in (2, 30):
                selected = [r for r in rows if r['signal_leg'] == leg and r['nran'] == nran]
                self.assertGreater(selected[0]['signal_relative_error']/selected[1]['signal_relative_error'], 900.)

    def test_actual_noise_must_be_supplied_and_not_random_or_negative_residual(self):
        v = np.array([[[.02, 0, 0], [0, .03, 0], [-.02, -.03, 0]]]); p, m = sample(np.linalg.norm(v, axis=-1))
        kwargs = dict(p_halo=self.ph, reconstruction=self.rec)
        with self.assertRaises(TypeError):
            GeneralNoisePointFields(v, p, m, self.b, self.fg, **kwargs)
        for power, role in ((-1., True), (3000., False)):
            with self.assertRaises(ValueError):
                GeneralNoisePointFields(v, p, m, self.b, self.fg, **kwargs,
                    noise_data_power=power, noise_is_data_only=role)
        with self.assertRaises(ValueError):
            combine_white_operators(np.zeros((3, 3)), noise_data_raw=-1., n2_data_self=3.)


if __name__ == '__main__':
    unittest.main()
