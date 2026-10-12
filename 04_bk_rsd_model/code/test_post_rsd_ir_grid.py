"""模块大纲：只消去不参与积分的格点与对称重复；不以近似平均方向替代有向模型。"""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_projection import IRFFTProjection,integrate_post_design
from post_rsd_ir_grid import GridHarmonics,FastIRFFTProjection
import test_post_rsd_ir_projection as fixture_module
toy_spectrum=fixture_module.toy_spectrum
toy_nowiggle=fixture_module.toy_nowiggle


class IRGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.IRProjectionTest.setUpClass();cls.fixture=fixture_module.IRProjectionTest

    def test_reflection_and_transverse_permutation(self):
        t=self.fixture;rng=np.random.default_rng(2026100906)
        modes=rng.integers(-47,48,size=(1007,3));vectors=modes*(2*np.pi/1000)
        bound=float(np.linalg.norm(vectors,axis=1).max())*1.01
        view=GridHarmonics(t.harm.coefficients,t.harm.lmax,110.,t.harm.physical_parameters,bound)
        expected=t.harm.damping(vectors);actual=view.damping(vectors)
        np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
        self.assertLessEqual(view.statistics['distinct_symmetry_modes'],len(vectors))

    def test_reachable_grid_preserves_same_FFT_result(self):
        t=self.fixture;designs=[]
        with patch('rsd_b000_fft_projection.load_spectrum',return_value=toy_spectrum):
            for engine in (IRFFTProjection,FastIRFFTProjection):
                f=engine(16,[0,.01,.018],(0,1),t.b1,t.f,p_halo=t.ph,reconstruction=t.rec,
                         power_nw=toy_nowiggle,harmonics=t.harm)
                designs.append(integrate_post_design(f))
                self.assertEqual(f.denominator,19*74)
        np.testing.assert_allclose(designs[0],designs[1],rtol=2e-12,atol=2e-7)

    def test_wrong_LOS_rejected(self):
        t=self.fixture;params=dict(t.harm.physical_parameters,los=[1,1,1])
        view=GridHarmonics(t.harm.coefficients,t.harm.lmax,110.,params,.5)
        with self.assertRaises(ValueError):view.damping(np.array([[.1,.2,.3]]))


if __name__=='__main__':unittest.main()
