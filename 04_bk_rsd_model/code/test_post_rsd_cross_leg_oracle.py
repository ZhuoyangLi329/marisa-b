"""模块大纲：PN_cross独立卷积对直接mode-pair枚举、原生产投影与零腿保护。

小网格临时谱，不写科学验收产物，不将toy结果替代真实44-bin检查。
"""
from itertools import combinations
import unittest
from unittest.mock import patch
import numpy as np
import rsd_b000_fft_projection as production
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_poisson_projection import PoissonShiftFFTProjection
from post_rsd_noise_transport import integrate_noise_transport
from post_rsd_cross_leg_oracle import cross_leg_convolution


class CrossLegOracleTest(unittest.TestCase):
    def setUp(self):
        self.rec = FixedReconstruction(10., 2.2280546354, 1.2822, 100.)
        self.edges = np.array([0., .008, .014])
        self.sample = lambda k: (10000/(1+(k/.09)**2), 1000*(1+(k/.02)**2))

    def field(self, pair, b=2.11, ph=1.417):
        with patch.object(production, 'load_spectrum', return_value=self.sample):
            return PoissonShiftFFTProjection(16, self.edges, pair, b,
                .7625790037720044, p_halo=ph, reconstruction=self.rec)

    def run_oracle(self, f, rec='default', masks=None):
        mi, mj = (f.masks[s] for s in f.slots[:2]) if masks is None else masks
        return cross_leg_convolution(f.v, f.p, f.m, mi, mj, b1=f.b1,
            f_growth=f.f, p_halo=f.p_halo,
            reconstruction=self.rec if rec == 'default' else rec)

    def direct_pairs(self, f):
        # 真正逐对枚举闭合波矢；不调用生产factor定义、integral或FFT。
        coords = [np.argwhere(f.masks[s]) for s in f.slots[:2]]
        total = np.zeros(3); valid = 0; outside = 0
        for ix in coords[0]:
            for iy in coords[1]:
                x, y = f.v[tuple(ix)], f.v[tuple(iy)]
                z = -x-y
                if any(np.dot(t, t) == 0 for t in (x, y, z)):
                    continue
                iz = tuple((-ix-iy) % f.n)
                if not any(f.masks[s][iz] for s in f.slots[:2]):
                    outside += 1
                vectors = np.stack((x, y, z)); squared = np.sum(vectors*vectors, axis=1)
                power, transfer = self.sample(np.sqrt(squared))
                response = self.rec.response(vectors, transfer)
                z0 = f.b1+f.f*vectors[:, 2]**2/squared
                zp = 2*1.686*(f.b1-f.p_halo)/transfer
                d = np.stack((power*z0*z0, 2*power*z0*zp, power*zp*zp), axis=1)
                for i, j in combinations(range(3), 2):
                    ti = np.dot(vectors[i]+vectors[j], vectors[i])*response[i]/squared[i]
                    tj = np.dot(vectors[i]+vectors[j], vectors[j])*response[j]/squared[j]
                    total -= ti*d[j]+tj*d[i]
                valid += 1
        return total/f.denominator, valid, outside

    def test_direct_pair_enumeration_all_coefficients_and_unrestricted_closing(self):
        for pair in ((0, 0), (0, 1), (1, 1)):
            f = self.field(pair); got, meta = self.run_oracle(f)
            expected, valid, outside = self.direct_pairs(f)
            np.testing.assert_allclose(got, expected, rtol=3e-13, atol=2e-11)
            self.assertGreater(valid, 0)
            if pair != (0, 0): self.assertGreater(outside, 0)
            self.assertEqual(meta['full_denominator'], f.denominator)
            self.assertFalse(meta['weighted_convolution_coefficients_rounded'])

    def test_original_projection_multiple_bias_and_p_and_finite_PNG(self):
        for pair in ((0, 0), (0, 1), (1, 1)):
            for b, ph in ((.63, .9), (2.11, 1.417), (3.83, 1.72)):
                f = self.field(pair, b, ph); got, _ = self.run_oracle(f)
                expected = integrate_noise_transport(f)[0]
                np.testing.assert_allclose(got, expected, rtol=5e-13, atol=2e-10)
                for fnl in (-500., 0., 73., 500.):
                    self.assertAlmostEqual(float(got @ [1, fnl, fnl*fnl]),
                        float(expected @ [1, fnl, fnl*fnl]), delta=2e-9)

    def test_no_production_factor_or_integral_or_transform_access(self):
        f = self.field((0, 1))
        with patch.object(f, 'values', side_effect=AssertionError('production values forbidden')), \
             patch.object(f, 'integral', side_effect=AssertionError('production integral forbidden')), \
             patch.object(f, 'transform', side_effect=AssertionError('production transform forbidden')):
            got, meta = self.run_oracle(f)
        self.assertTrue(np.isfinite(got).all())
        self.assertFalse(meta['production_values_transform_integral_called'])

    def test_full_denominator_includes_source_zero_not_numerator(self):
        f = self.field((0, 0)); got, meta = self.run_oracle(f)
        self.assertTrue(f.masks[0][0, 0, 0])
        self.assertLess(meta['nonzero_input_pairs'], meta['full_denominator'])
        nozero = f.masks[0].copy(); nozero[0, 0, 0] = False
        changed, without = self.run_oracle(f, masks=(nozero, nozero))
        np.testing.assert_allclose(got*meta['full_denominator'],
            changed*without['full_denominator'], rtol=3e-13, atol=2e-9)

    def test_disabled_shift_and_bad_geometry_fail_closed(self):
        f = self.field((0, 1)); got, _ = self.run_oracle(f, rec=None)
        np.testing.assert_array_equal(got, np.zeros(3))
        mi, mj = [f.masks[s].copy() for s in f.slots[:2]]
        mi[1, 0, 0] = False
        with self.assertRaisesRegex(ValueError, 'inversion'):
            self.run_oracle(f, masks=(mi, mj))
        with self.assertRaisesRegex(ValueError, 'wrap'):
            self.run_oracle(f, masks=(np.ones_like(mi), np.ones_like(mj)))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.run_oracle(f, masks=(np.zeros_like(mi), mj))


if __name__ == '__main__':
    unittest.main()
