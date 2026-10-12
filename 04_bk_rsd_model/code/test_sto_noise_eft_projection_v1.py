"""模块大纲：独立小lattice 3个shell pair，原FFT与非循环整数球逐项对照。"""
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_fixedrec import FixedReconstruction
from project_sto_noise_eft_primitives_v1 import (EFTFFT,independent_linear_oracle,
    project_intrinsic_primitives,project_zero_primitives)


class EFTProjectionTest(unittest.TestCase):
    def test_all_primitives_cross_and_diagonal_shells(self):
        edges=np.array([1.,3.,6.])*2*np.pi/1000.
        rec=FixedReconstruction(10.,2.228,1.2822,0.,1000./256,4)
        def spectrum(k):return 2500./(1.+11*k),100.+13000*k*k
        for pair in ((0,0),(0,1),(1,1)):
            with patch('rsd_b000_fft_projection.load_spectrum',return_value=spectrum),patch(
                    'project_sto_noise_eft_primitives_v1.load_spectrum',return_value=spectrum):
                field=EFTFFT(32,edges,pair,2.23,.76258,p_halo=1.3,reconstruction=rec)
                expected,z,counts=independent_linear_oracle(edges,pair,rec)
                np.testing.assert_allclose(project_intrinsic_primitives(field),expected,rtol=3e-14,atol=1e-11)
                np.testing.assert_allclose(project_zero_primitives(field),z,rtol=3e-14,atol=1e-11)
                self.assertEqual(field.denominator,counts['full_denominator'])
                if pair[0]!=pair[1]:np.testing.assert_array_equal(z,np.zeros(14))


if __name__=='__main__':unittest.main()
