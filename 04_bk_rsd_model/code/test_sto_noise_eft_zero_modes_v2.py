"""模块大纲：三种零腿、小lattice完整分母及未用全零组合拒绝的独立检查。"""
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_fixedrec import FixedReconstruction
from project_sto_noise_eft_primitives_v1 import EFTFFT
from post_rsd_stochastic_eft_v1 import project_intrinsic_primitives
from post_rsd_eft_zero_modes_v2 import project_zero_primitives_all_legs,independent_linear_oracle_all_legs


class AllZeroLegModesTest(unittest.TestCase):
    def test_all_three_zero_legs_and_nonzero_sources(self):
        edges=np.array([0.,3.,6.,9.])*2*np.pi/1000.
        rec=FixedReconstruction(10.,2.228,1.2822,0.,1000./256,4)
        def spectrum(k):return 2500./(1.+11*k),100.+13000*k*k
        for pair in ((0,1),(1,0),(1,1),(1,2)):
            with patch('rsd_b000_fft_projection.load_spectrum',return_value=spectrum),patch(
                    'post_rsd_eft_zero_modes_v2.load_spectrum',return_value=spectrum):
                field=EFTFFT(48,edges,pair,2.23,.76258,p_halo=1.3,reconstruction=rec)
                expected,zero,counts=independent_linear_oracle_all_legs(edges,pair,rec)
                np.testing.assert_allclose(project_intrinsic_primitives(field),expected,rtol=4e-14,atol=1e-10)
                np.testing.assert_allclose(project_zero_primitives_all_legs(field),zero,rtol=4e-14,atol=1e-10)
                self.assertEqual(field.denominator,counts['full_denominator'])
                self.assertEqual(counts['nonzero_closing_pairs']+counts['total_zero_atoms'],field.denominator)
                # 另直接枚举所有input mode pairs，独立计数每种零腿。
                left,right=[np.argwhere(field.masks[s]) for s in field.slots[:2]]
                modes=np.rint(np.fft.fftfreq(field.n)*field.n).astype(int)
                kval1=modes[left];kval2=modes[right]
                first0=int(np.all(kval1==0,axis=1).sum())*int(np.any(kval2!=0,axis=1).sum())
                second0=int(np.all(kval2==0,axis=1).sum())*int(np.any(kval1!=0,axis=1).sum())
                closing0=sum(int(np.all(kval2==-k,axis=1).sum()) for k in kval1 if np.any(k))
                self.assertEqual((first0,second0,closing0),(counts['zero_first_leg_count'],
                    counts['zero_second_leg_count'],counts['zero_closing_count']))

    def test_excluded_all_zero_combination_is_not_silently_modeled(self):
        edges=np.array([0.,3.,6.])*2*np.pi/1000.
        rec=FixedReconstruction(10.,2.228,1.2822,0.)
        with self.assertRaises(ValueError):independent_linear_oracle_all_legs(edges,(0,0),rec)


if __name__=='__main__':unittest.main()
