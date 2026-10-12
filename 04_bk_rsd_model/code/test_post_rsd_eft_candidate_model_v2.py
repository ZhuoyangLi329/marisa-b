"""模块大纲：恢复原Pshot；P与raw shift noise解耦、B不改及joint whitening。"""
import unittest
import numpy as np
from post_rsd_eft_candidate_model_v2 import EFTCandidateModelV2
import test_post_rsd_eft_candidate_model_v1 as old_tests


class CandidateV2Test(unittest.TestCase):
    def setUp(self):
        source=old_tests.CandidateModelTest();source.setUp();self.source=source
        m=source.model
        self.model=EFTCandidateModelV2(m.B_coefficients,m.P_templates,m.operators,m.intrinsic_primitives,
            m.zero_primitives,m.weight_moments,m.p_halo,m.f_growth)
        self.rows=np.column_stack((source.rows,[-2.7,.31]))

    def test_original_Pshot_restored_B_prediction_identical(self):
        actual=self.model.predictions(self.rows);old=self.source.model.predictions(self.source.rows)
        np.testing.assert_allclose(actual[:,12:],old[:,12:],rtol=0,atol=0)
        np.testing.assert_allclose(actual[:,:12],np.broadcast_to(
            self.rows[:,11,None]*self.model.weight_moments[0],(2,12)),rtol=3e-15)
        changed=self.rows.copy();changed[:,7]*=1.3
        np.testing.assert_allclose(self.model.predictions(changed)[:,:12],actual[:,:12],rtol=3e-15)
        self.assertGreater(np.max(abs(self.model.predictions(changed)[:,12:]-actual[:,12:])),0.)

    def test_full_joint_residual_still_uses_cross(self):
        rng=np.random.default_rng(91);a=rng.normal(size=(21,21));c=a@a.T+np.eye(21);chol=np.linalg.cholesky(c)
        target=rng.normal(size=21);white=self.model.residual(self.rows,target,chol)
        for i,row in enumerate(self.rows):
            d=self.model.predictions(row)[0]-target
            np.testing.assert_allclose(white[i]@white[i],d@np.linalg.solve(c,d),rtol=3e-14)

    def test_original_negative_effective_Pshot_allowed_raw_negative_forbidden(self):
        self.assertTrue(np.isfinite(self.model.predictions(self.rows)).all())
        negative=self.rows.copy();negative[:,7]=-1.
        with self.assertRaises(ValueError):self.model.predictions(negative)


if __name__=='__main__':unittest.main()
