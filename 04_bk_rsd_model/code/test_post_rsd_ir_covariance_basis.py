"""模块大纲：解析球谐相位/角度；任意bias精确缓存、PSD、pre和显式噪声。

只验收协方差子模块，不代表全44-bin IR-B、STO噪声或fNL恢复通过。
"""
import unittest
import warnings
import numpy as np
from scipy import special
from post_rsd_scipy_harmonic_compat import legacy_sph_harm, install_legacy_sph_harm
from post_rsd_ir_covariance_basis import HarmonicCovarianceBasis
from post_rsd_displacement import GaussianLongDisplacement
from post_rsd_fixedrec import FixedReconstruction


class HarmonicCompatTest(unittest.TestCase):
    def test_analytic_angles_phase_and_negative_azimuth(self):
        polar = np.array([.2, .9, 1.7, 2.8]); azimuth = np.array([-.7, .3, 2.1, 6.1])
        expected = {(0, 0): np.full(4, 1/np.sqrt(4*np.pi)),
            (0, 1): np.sqrt(3/(4*np.pi))*np.cos(polar),
            (1, 1): -np.sqrt(3/(8*np.pi))*np.sin(polar)*np.exp(1j*azimuth),
            (-1, 1): np.sqrt(3/(8*np.pi))*np.sin(polar)*np.exp(-1j*azimuth)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            for (m, ell), value in expected.items():
                np.testing.assert_allclose(legacy_sph_harm(m, ell, azimuth, polar), value, rtol=3e-14, atol=1e-15)

    def test_installed_entry_preserves_real_harmonics_and_out_argument(self):
        info = install_legacy_sph_harm(); self.assertTrue(info['process_only'])
        theta = np.array([.7, 1.9]); phi = np.array([-.7, 1.4])
        out = np.empty(2, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertIs(legacy_sph_harm(2, 4, phi, theta, out=out), out)
            np.testing.assert_allclose(out, special.sph_harm(2, 4, phi, theta), rtol=3e-14, atol=1e-15)


class IRCovarianceBasisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_legacy_sph_harm()
        q = np.geomspace(2*np.pi/1000, .15, 33)
        cls.long = GaussianLongDisplacement(q, 6200*np.exp(-q/.13), 8e6*q*q/(1+7*q), 16, 32)
        ak = np.geomspace(1e-5, 3., 2048); av = 1/(8e6*ak*ak/(1+7*ak))
        cls.rec = FixedReconstruction(10., 2.2280546353635335, 1.2822, 100., 1000/256, 4, alpha_table=(ak, av))
        cls.kwargs = dict(f_growth=.7625790037720044, los=(0, 0, 1), reconstruction=cls.rec, lmax=6)
        cls.basis = HarmonicCovarianceBasis.from_long_modes(cls.long, **cls.kwargs)

    def test_reconstructs_full_bias_and_PBS_corner_domain_without_new_fit_parameters(self):
        from post_rsd_ir_harmonics import HarmonicDisplacement
        for b in (.5, 1.1, 2.2280546353635335, 3.1, 4.):
            for p in (0., 1.3, 3.):
                phi = 2*1.686*(b-p)
                direct = HarmonicDisplacement.from_long_modes(self.long, b1=b, bphi=phi, **self.kwargs)
                np.testing.assert_allclose(self.basis.coefficients_at(b1=b, bphi=phi), direct.coefficients,
                    rtol=5e-12, atol=2e-12)
                self.assertEqual(self.basis.harmonics_at(b1=b, bphi=phi).physical_parameters,
                                 direct.physical_parameters)
        self.assertFalse(self.basis.build_metadata['signal_B_polynomial_interpolation'])
        self.assertFalse(self.basis.build_metadata['MCMC_released'])

    def test_covariance_and_damping_not_fixed_at_probe_bias(self):
        from post_rsd_ir_harmonics import HarmonicDisplacement
        vectors = np.array([[.05, .02, .03], [.007, -.01, .14], [-.12, .09, .01]])
        for b, phi in ((.5, -4.), (2.23, 3.41), (4., 10.)):
            cached = self.basis.harmonics_at(b1=b, bphi=phi)
            direct = HarmonicDisplacement.from_long_modes(self.long, b1=b, bphi=phi, **self.kwargs)
            np.testing.assert_allclose(cached.damping(vectors), direct.damping(vectors), rtol=5e-12, atol=2e-13)
            a = cached.covariance(vectors)
            for fnl in (-500., 0., 500.):
                total = a[:, 0]+fnl*a[:, 1]+fnl*fnl*a[:, 2]
                self.assertGreater(np.linalg.eigvalsh(total).min(), -1e-10)

    def test_pre_limit_and_explicit_noise_offset(self):
        from post_rsd_ir_harmonics import HarmonicDisplacement
        pre = HarmonicCovarianceBasis.from_long_modes(self.long, **dict(self.kwargs, reconstruction=None))
        np.testing.assert_array_equal(pre.coefficients_at(b1=.5, bphi=-4.), pre.coefficients_at(b1=4., bphi=10.))
        noise = np.full(len(self.long.q), 3300.)
        noisy = HarmonicCovarianceBasis.from_long_modes(self.long, shift_noise=noise, **self.kwargs)
        direct = HarmonicDisplacement.from_long_modes(self.long, b1=2.1, bphi=3.2, shift_noise=noise, **self.kwargs)
        np.testing.assert_allclose(noisy.coefficients_at(b1=2.1, bphi=3.2), direct.coefficients, rtol=5e-12, atol=2e-12)
        np.testing.assert_array_equal(noisy.coefficients_at(b1=2.1, bphi=3.2)[:, 1:],
                                     self.basis.coefficients_at(b1=2.1, bphi=3.2)[:, 1:])
        self.assertTrue(noisy.geometry['shift_noise_explicit'])
        self.assertFalse(noisy.build_metadata['stochastic_closure_validated'])
        with self.assertRaises(ValueError):
            HarmonicCovarianceBasis.from_long_modes(self.long, shift_noise=-noise, **self.kwargs)

    def test_basis_arrays_are_immutable_and_invalid_bias_rejected(self):
        self.assertFalse(self.basis.gram_coefficients.flags.writeable)
        with self.assertRaises(ValueError):
            self.basis.coefficients_at(b1=np.nan, bphi=1.)
        with self.assertRaises(ValueError):
            self.basis.gram_coefficients[0, 0, 0, 0] = 0.


if __name__ == '__main__':
    unittest.main()
