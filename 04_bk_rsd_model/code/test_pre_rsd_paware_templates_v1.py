"""模块大纲：pre精确p多项式、进程内PBS恢复、native源和原FFT重建回归。"""
import unittest
import numpy as np
import rsd_bonly_common as common
from rsd_b000_contract import BNODES,F_GROWTH
from rsd_b000_fft_projection import FFTProjection,integrate_design
from post_rsd_noir_polynomial import P_NODES,evaluate_noir_design
from build_pre_rsd_paware_templates_v1 import fit_pre_paware,physical_p,assign_bias,native_point_audit

class PrePAwareTest(unittest.TestCase):
    def test_exact_cubic_b1_quadratic_p_tensor_and_zero_quartic_axis(self):
        rng=np.random.default_rng(7102);c=np.zeros((5,3,3,3,4));c[:4]=rng.normal(size=(4,3,3,3,4))
        nodes=np.array([[evaluate_noir_design(c,b,p_halo=p) for p in P_NODES] for b in BNODES])
        fit=fit_pre_paware(nodes)
        np.testing.assert_allclose(fit,c,rtol=2e-12,atol=5e-15)
        np.testing.assert_array_equal(fit[4],np.zeros_like(fit[4]))
        for b,p in ((.5,0.),(1.83641,1.179),(2.7,1.2822),(4.,3.)):
            np.testing.assert_allclose(evaluate_noir_design(fit,b,p_halo=p),evaluate_noir_design(c,b,p_halo=p),rtol=2e-12,atol=1e-12)

    def test_PBS_override_restored_even_on_exception(self):
        original=common.P_CAL
        with self.assertRaises(RuntimeError):
            with physical_p(2.37):
                self.assertAlmostEqual(common.bias(2.7).bphi,2*1.686*(2.7-2.37));raise RuntimeError('fixture')
        self.assertEqual(common.P_CAL,original)

    def test_explicit_p_native_oriented_source(self):
        result=native_point_audit();self.assertLess(result['maximum_scaled_error'],2e-10)

    def test_cache_reassignment_matches_fresh_original_FFT(self):
        edges=np.array([.006,.012,.018,.026]);samples=[]
        with physical_p(P_NODES[0]):projector=FFTProjection(32,edges,(1,2),BNODES[0])
        for b in BNODES:
            row=[]
            for p in P_NODES:
                assign_bias(projector,b,p)
                with physical_p(p):row.append(integrate_design(projector))
            samples.append(row)
        coefficient=fit_pre_paware(np.array(samples))
        for b,p in ((.731,.6),(1.83641,1.41421),(4.,2.31)):
            with physical_p(p):
                fresh=FFTProjection(32,edges,(1,2),b);direct=integrate_design(fresh)
                assign_bias(projector,b,p);reused=integrate_design(projector)
            np.testing.assert_array_equal(reused,direct)
            predicted=evaluate_noir_design(coefficient,b,p_halo=p)
            weights=np.array([1,500,250000])[None,:];scale=np.maximum(1,np.max(abs(direct)*weights,axis=1,keepdims=True))
            np.testing.assert_allclose((predicted-direct)*weights/scale,0,atol=2e-10)

if __name__=='__main__':unittest.main()
