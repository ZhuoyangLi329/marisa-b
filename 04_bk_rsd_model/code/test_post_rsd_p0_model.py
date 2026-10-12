"""模块大纲：P0显式p与原pre回归；独立有向Z1平方及输入保护。"""
import unittest
import numpy as np
from post_rsd_p0_model import post_p0_linear
from post_rsd_fixedrec import pbs_bias
from rsd_pb_common import p_prediction
from rsd_b000_contract import P_CAL,F_GROWTH


class PostP0SourceTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2026100907)
        self.mu=rng.uniform(-1,1,(12,71));self.p=rng.uniform(3000,25000,(12,71))
        self.m=rng.uniform(100,30000,(12,71));self.w=rng.uniform(.1,2.,(12,71))
        self.w/=self.w.sum(axis=1)[:,None]
        p=self.p;m=self.m;u=self.mu**2
        self.t=np.asarray([np.sum(self.w*x,axis=1) for x in (p,p*u,p*u*u,p/m,p*u/m,p/m**2)])

    def test_matches_direct_oriented_Z1_square(self):
        for b1 in (.5,2.2280546353635335,4.):
            for ph in (.9,1.179,1.31,1.6):
                for fnl in (-500.,-100.,0.,100.,500.):
                    z=b1+F_GROWTH*self.mu**2+fnl*pbs_bias(b1,p_halo=ph).bphi/self.m
                    direct=np.sum(self.w*self.p*z*z,axis=1)
                    np.testing.assert_allclose(post_p0_linear(fnl,b1,self.t,p_halo=ph),direct,rtol=4e-14,atol=2e-9)

    def test_original_pre_parity_and_explicit_physical_p(self):
        np.testing.assert_array_equal(post_p0_linear(100.,2.2,self.t,p_halo=P_CAL),p_prediction(100.,2.2,self.t))
        with self.assertRaises(TypeError):post_p0_linear(100.,2.2,self.t)
        with self.assertRaises(ValueError):post_p0_linear(np.nan,2.2,self.t,p_halo=1.31)
        with self.assertRaises(ValueError):post_p0_linear(100.,2.2,self.t[:5],p_halo=1.31)


if __name__=='__main__':unittest.main()
