"""模块大纲：粗C保持single-box、边缘C使用自身维度、校准预测复用原实现。"""
import unittest
import numpy as np
from run_sto_post_partial_p_mcmc import marginal,PartialCalibrationModel
from post_rsd_p_calibration import PhysicalPCalibrationCase


class PartialPTests(unittest.TestCase):
    def test_marginal_is_not_conditional_or_covariance_of_mean(self):
        c=np.array([[4.,1.],[1.,3.]])
        actual,chol,alpha=marginal(c,201,(0,))
        np.testing.assert_array_equal(actual,[[4.]])
        self.assertEqual(alpha,198/200)
        np.testing.assert_allclose(chol@chol.T,actual/alpha)

    def test_dynamic_sample_count_and_no_regularization(self):
        c=np.eye(12);_,chol,alpha=marginal(c,201,tuple(range(12)))
        self.assertEqual(alpha,187/200)
        PartialCalibrationModel(np.zeros(12),c,chol,np.ones((6,12)),201)
        with self.assertRaises(ValueError):marginal(c,14,tuple(range(12)))
        with self.assertRaises(np.linalg.LinAlgError):marginal(np.ones((12,12)),201,tuple(range(12)))

    def test_original_500_case_prediction_parity(self):
        rng=np.random.default_rng(22);a=rng.normal(size=(12,12));c=a@a.T+np.eye(12)
        t=rng.uniform(.01,2.,size=(6,12));target=rng.normal(size=12);chol=np.linalg.cholesky(c/(486/499))
        old=PhysicalPCalibrationCase(target,c,chol,t)
        new=PartialCalibrationModel(target,c,chol,t,500)
        rows=np.array([[.4,1.1,-300.],[1.3,2.2,700.],[2.8,3.8,-1000.]])
        np.testing.assert_array_equal(new.predictions(rows),old.predictions(rows))
        np.testing.assert_array_equal(new.objective_many(rows),old.objective_many(rows))


if __name__=='__main__':unittest.main()
