"""模块大纲：旧real-space G/L/Q极限、RSD方向、noise独立性与完整cross白化。"""
import unittest
import numpy as np
from post_rsd_legacy3_model import LegacyThreeNoiseModel, legacy_bshot_shape


class LegacyThreeNoiseTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(81010)
        self.primitive=rng.uniform(.1,2,(7,9))
        self.model=LegacyThreeNoiseModel(np.zeros((5,3,3,9)),np.zeros((6,12)),
            self.primitive,np.array([3300.,1.1e7]),1.2637656599262268,.7625790037720044)
        self.row=np.array([100.,2.23,-.25,-.3,.2,.4,-1.1])

    def test_old_realspace_linear_factor_one_not_Z1_squared_factor_two(self):
        fnl,b,bp=100.,2.2,3.1
        p=self.primitive
        old=b*b*p[0]+fnl*b*bp*p[3]+fnl*fnl*bp*bp*p[5]
        np.testing.assert_array_equal(legacy_bshot_shape(p,fnl,b,bp,0.),old)

    def test_direct_directed_leg_sum_matches_projected_seven_primitives(self):
        rng=np.random.default_rng(64)
        power=rng.uniform(1,10,(9,3));mu=rng.uniform(0,1,(9,3));invM=rng.uniform(.001,.03,(9,3))
        primitive=np.array([power.sum(1),(power*mu).sum(1),(power*mu**2).sum(1),
            (power*invM).sum(1),(power*mu*invM).sum(1),(power*invM**2).sum(1),np.ones(9)])
        b,bp,g,fnl=2.2,3.1,.76,-100.
        z=b+g*mu
        direct=np.sum(power*(z*z+fnl*bp*z*invM+fnl*fnl*bp*bp*invM**2),axis=1)
        np.testing.assert_allclose(legacy_bshot_shape(primitive,fnl,b,bp,g),direct,rtol=3e-15)

    def test_Pshot_and_two_B_amplitudes_separate(self):
        prediction=self.model.predictions(self.row)[0]
        changed=self.row.copy();changed[6]+=1
        difference=self.model.predictions(changed)[0]-prediction
        np.testing.assert_array_equal(difference[12:],np.zeros(9))
        np.testing.assert_allclose(difference[:12],3300.)
        for j in [4,5]:
            changed=self.row.copy();changed[j]+=1
            difference=self.model.predictions(changed)[0]-prediction
            np.testing.assert_array_equal(difference[:12],np.zeros(12))
            self.assertGreater(np.max(abs(difference[12:])),0)

    def test_full_PB_cross_precision_without_block_replacement(self):
        rng=np.random.default_rng(72);a=rng.normal(size=(21,21));c=a@a.T+np.eye(21)
        target=rng.normal(size=21);chol=np.linalg.cholesky(c)
        residual=self.model.residual(self.row,target,chol)[0]
        difference=self.model.predictions(self.row)[0]-target
        np.testing.assert_allclose(residual@residual,difference@np.linalg.solve(c,difference),rtol=3e-14)
        with self.assertRaises(ValueError):self.model.residual(self.row,target,chol.T)


if __name__=='__main__':unittest.main()
