"""模块大纲：独立Z1/spectral-t导数、strict PNG mixing、真实窗口型投影和保护。"""
from dataclasses import replace
import unittest
import numpy as np
from post_rsd_ir_p0_projection import ir_p0_modes, project_p0_ir_window
from post_rsd_ir_harmonics import HarmonicDisplacement, reconstruction_signature
from post_rsd_fixedrec import FixedReconstruction, pbs_bias


def independent_coefficients(v, p, pn, m, b, f, ph, damping):
    mu2 = v[:, 2]**2/np.sum(v*v, axis=1); r = b+f*mu2; phi = 2*1.686*(b-ph)
    mn = m*np.sqrt(pn/p)
    # 手写单腿 t²Pn/(tMn)² 源的t导数，不调用生产D/TaggedField。
    base = np.array([pn*r*r, 2*pn*r*phi/mn, pn*(phi/mn)**2])
    eta = (p-pn)/(2*pn); w0 = 2*eta*base[0]; w1 = eta*base[1]
    d0, d1, d2 = damping.T
    return np.array([base[0]+d0*w0, base[1]+d0*(w1+d1*w0),
                     base[2]+d0*(d1*w1+d2*w0)])


class IRP0Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261010)
        self.v = rng.normal(size=(96, 3))*.025
        k = np.linalg.norm(self.v, axis=1)
        self.pn = 6000/(1+(k/.07)**2); self.p = self.pn*(1+.07*np.sin(110*k))
        self.m = 8e6*k*k/(1+7*k); self.b = 2.13; self.ph = 1.31; self.f = .7625790037720044
        self.damping = np.column_stack((rng.uniform(.6, 1., 96), rng.normal(size=96)*1e-4,
                                        rng.normal(size=96)*1e-8))
        ak = np.geomspace(1e-5, 3., 2048); av = 1/(8e6*ak*ak/(1+7*ak))
        self.rec = FixedReconstruction(10., 2.2280546353635335, 1.2822, 100., 1000/256, 4,
                                       alpha_table=(ak, av))

    def mode(self, p=None, pn=None, damping=None):
        return ir_p0_modes(self.v, self.p if p is None else p, self.pn if pn is None else pn,
            self.m, self.b, self.f, p_halo=self.ph, reconstruction=self.rec,
            damping=self.damping if damping is None else damping)

    def test_shared_tagged_source_against_independent_explicit_mixing(self):
        expected = independent_coefficients(self.v, self.p, self.pn, self.m,
                                            self.b, self.f, self.ph, self.damping)
        np.testing.assert_allclose(self.mode(), expected, rtol=5e-14, atol=2e-10)
        for fnl in (-500., -100., 0., 100., 500.):
            np.testing.assert_allclose([1., fnl, fnl*fnl]@self.mode(),
                                       [1., fnl, fnl*fnl]@expected, rtol=5e-14, atol=2e-9)

    def test_no_wiggle_reduces_to_original_exact_linear_source(self):
        c = self.mode(pn=self.p)
        z0 = self.b+self.f*self.v[:, 2]**2/np.sum(self.v*self.v, axis=1)
        zphi = 2*1.686*(self.b-self.ph)/self.m
        for fnl in (-500., 0., 500.):
            np.testing.assert_allclose([1., fnl, fnl*fnl]@c, self.p*(z0+fnl*zphi)**2,
                                       rtol=3e-13, atol=2e-10)

    def test_zero_exponent_strict_wiggle_remainder_is_explicit_not_hidden(self):
        no_damping = np.zeros((96, 3)); no_damping[:, 0] = 1.
        result = self.mode(damping=no_damping)
        r = self.b+self.f*self.v[:, 2]**2/np.sum(self.v*self.v, axis=1)
        phi = 2*1.686*(self.b-self.ph); mn = self.m*np.sqrt(self.pn/self.p)
        exact = np.array([self.p*r*r, 2*self.p*r*phi/self.m, self.p*(phi/self.m)**2])
        remainder = 2*self.pn*r*phi/mn*(np.sqrt(self.p/self.pn)-1)**2/2
        np.testing.assert_allclose(result[[0, 2]], exact[[0, 2]], rtol=5e-14, atol=2e-10)
        np.testing.assert_allclose(result[1]-exact[1], remainder, rtol=3e-10, atol=1e-12)
        self.assertGreater(float(np.max(np.abs(remainder))), 1e-5)

    def test_weighted_window_matches_direct_mode_loop_and_context_guards(self):
        rng = np.random.default_rng(2026101012)
        integer = rng.integers(1, 6, (96, 3)); labels = np.tile(np.arange(12), 8)
        weights = np.where(np.arange(96)%3, 2., 1.); counts = np.bincount(labels, weights=weights, minlength=12)
        window = dict(vectors_integer=integer, labels=labels, weights=weights, nmodes=counts)
        params = dict(b1=self.b, bphi=pbs_bias(self.b, p_halo=self.ph).bphi, f_growth=self.f,
            los=[0., 0., 1.], reconstruction_signature=reconstruction_signature(self.rec))
        class ExplicitDamping(HarmonicDisplacement):
            def damping(inner, vectors, **kwargs): return self.damping
        h = ExplicitDamping(None, 10, 110., params)
        actual = project_p0_ir_window(window, self.p, self.pn, self.m, self.b, self.f,
            p_halo=self.ph, reconstruction=self.rec, harmonics=h)
        values = independent_coefficients(integer*(2*np.pi/1000.), self.p, self.pn, self.m,
                                          self.b, self.f, self.ph, self.damping)
        expected = np.array([[sum(weights[j]*row[j] for j in range(96) if labels[j] == i)/counts[i]
                              for i in range(12)] for row in values])
        np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=2e-10)
        for ph, rec in ((self.ph+.1, self.rec), (self.ph, replace(self.rec, fnl_recon=0.))):
            with self.assertRaises(ValueError):
                project_p0_ir_window(window, self.p, self.pn, self.m, self.b, self.f,
                    p_halo=ph, reconstruction=rec, harmonics=h)

    def test_invalid_spectra_zero_mode_and_uncontrolled_window_rejected(self):
        with self.assertRaises(ValueError): self.mode(pn=np.zeros(96))
        with self.assertRaises(ValueError):
            ir_p0_modes(np.zeros_like(self.v), self.p, self.pn, self.m, self.b, self.f,
                p_halo=self.ph, reconstruction=self.rec, damping=self.damping)


if __name__ == '__main__':
    unittest.main()
