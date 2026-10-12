"""模块大纲：P0-only 独立似然校验；联合预测 parity；协方差与共享采样核。
"""
import json
import unittest
import numpy as np
import run_rsd_b000_mcmc as shared
from run_rsd_pb_inference import predicted
from run_rsd_p0only import OUT,PB_OUT,POnlyCase,parent,SAMPLER_SHA,digest


class POnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent();cls.names=json.loads((OUT/'CASES.json').read_text())

    def test_same_prediction_prior_and_data_as_joint(self):
        rng=np.random.default_rng(20261008)
        for name in self.names:
            case=POnlyCase(OUT/'cases'/name)
            with np.load(case.meta['source_PB_case']+'/likelihood.npz') as z:
                v=rng.uniform(case.bounds[:,0],case.bounds[:,1],(30,3))
                lift=rng.uniform(z['bounds'][:,0],z['bounds'][:,1],(30,8));lift[:,[0,1,7]]=v
                direct=predicted(lift,z['coefficients'],z['noise'])[:,:12]
                np.testing.assert_allclose(case.predictions(v),direct,rtol=5e-12,atol=1e-7)
                np.testing.assert_array_equal(case.bounds,z['bounds'][[0,1,7]])
                np.testing.assert_array_equal(case.target,z['target'][:12])
                np.testing.assert_allclose(case.covariance,z['covariance'][:12,:12],rtol=2e-13)

    def test_unaveraged_covariance_and_correct_dimension(self):
        with np.load(PB_OUT/'P0_ENSEMBLE_MATRIX.npz') as z:
            mask=np.array([json.loads(str(m))['role']=='covariance' for m in z['metadata_json']])
            expected=np.cov(z['P0'][mask],rowvar=False,ddof=1)
        for name in self.names:
            case=POnlyCase(OUT/'cases'/name);alpha=(500-12-2)/499
            np.testing.assert_allclose(case.covariance,expected,rtol=2e-13)
            self.assertEqual(case.meta['Hartlap_factor'],alpha)
            np.testing.assert_allclose(case.chol@case.chol.T,expected/alpha,rtol=3e-13)
            v=np.array([73.,1.87,.21]);r=case.predictions(v)[0]-case.target
            expected_chi2=alpha*r@np.linalg.solve(expected,r)
            np.testing.assert_allclose(case.objective(*v),expected_chi2,rtol=2e-12)

    def test_shared_sampling_and_prior_support(self):
        self.assertEqual(digest(shared.__file__),SAMPLER_SHA)
        for name in ('objective','physical','coordinates','log_prob','initial'):
            self.assertIs(getattr(POnlyCase,name),getattr(shared.Case,name))
        case=POnlyCase(OUT/'cases'/self.names[0]);case.center=np.zeros(3);case.transform=np.eye(3)
        self.assertFalse(np.isfinite(case.log_prob(np.array([[501,1.9,0]]))[0]))
        self.assertTrue(np.isfinite(case.log_prob(np.array([[0,1.9,0]]))[0]))


if __name__=='__main__':unittest.main(verbosity=2)
