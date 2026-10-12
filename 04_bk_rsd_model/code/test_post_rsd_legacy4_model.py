"""模块大纲：4+4嵌套还原、独立有向腿各向异性及完整joint白化回归。"""
import unittest
import numpy as np
from post_rsd_legacy3_model import LegacyThreeNoiseModel
from post_rsd_legacy4_model import LegacyFourNoiseModel,NAMES

class LegacyFourNoiseTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41104);self.rng=rng;n=9
        self.v=rng.normal(size=(n,31,3,3));self.v[:,:,2]=-self.v[:,:,0]-self.v[:,:,1]
        self.p=rng.uniform(100,30000,(n,31,3));self.w=rng.uniform(.1,2,(n,31));self.w/=self.w.sum(axis=1)[:,None]
        self.mu2=self.v[...,2]**2/np.sum(self.v*self.v,axis=-1)
        primitives=rng.uniform(.1,10,(7,n))
        primitives[0]=np.sum(self.w*np.sum(self.p,axis=-1),axis=1)
        primitives[2]=np.sum(self.w*np.sum(self.p*self.mu2**2,axis=-1),axis=1)
        self.args=(rng.normal(size=(5,3,3,n)),rng.uniform(.1,10,(6,12)),primitives,
            np.array([3000.,1e7]),1.3,.7625790037720044)
        self.old=LegacyThreeNoiseModel(*self.args);self.new=LegacyFourNoiseModel(*self.args)
        self.rows=rng.normal(size=(13,7));self.rows[:,0]*=100;self.rows[:,1]=rng.uniform(.5,4,13)

    def test_zero_R_exactly_restores_all_seven_parameter_predictions(self):
        np.testing.assert_array_equal(self.new.predictions(np.c_[self.rows,np.zeros(13)]),self.old.predictions(self.rows))
        self.assertEqual(len(NAMES),8);self.assertEqual(self.new.ndim,8)

    def test_independent_directed_leg_anisotropic_projection(self):
        direct=3000.*self.args[-1]**2*np.sum(self.w*np.sum(self.p*(self.mu2**2-.2),axis=-1),axis=1)
        np.testing.assert_allclose(self.new.anisotropic_shape,direct,rtol=2e-13,atol=1e-7)
        actual=self.new.predictions(np.c_[self.rows,np.full(13,2.7)])-self.old.predictions(self.rows)
        np.testing.assert_array_equal(actual[:,:12],np.zeros((13,12)))
        np.testing.assert_allclose(actual[:,12:],np.broadcast_to(2.7*direct,(13,9)),rtol=2e-12,atol=1e-4)

    def test_full_joint_residual_not_separate_P_B_whitening(self):
        a=self.rng.normal(size=(21,21));chol=np.linalg.cholesky(a@a.T+np.eye(21))
        target=self.rng.normal(size=21);rows=np.c_[self.rows,np.linspace(-5,5,13)]
        expected=np.linalg.solve(chol,(self.new.predictions(rows)-target).T).T
        np.testing.assert_array_equal(self.new.residual(rows,target,chol),expected)

    def test_no_anisotropic_R_contribution_in_real_space_limit(self):
        model=LegacyFourNoiseModel(*self.args[:-1],0.)
        zero=model.predictions(np.c_[self.rows,np.zeros(13)])
        np.testing.assert_array_equal(model.predictions(np.c_[self.rows,np.ones(13)*10]),zero)

if __name__=='__main__':unittest.main()
