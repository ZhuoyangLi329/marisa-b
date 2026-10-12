"""模块大纲：有界Pshot解析积分与独立数值积分，包含远尾与Gaussian归一化。"""
import unittest
import numpy as np
from scipy.integrate import quad
from rsd_p0_analytic_grid import shot_marginal_loglike,marginal_pdf

class AnalyticGridTest(unittest.TestCase):
    def test_bounded_shot_integral_matches_independent_quadrature(self):
        chol=np.linalg.cholesky([[2.,.3],[.3,1.]]);target=np.array([.4,-.7])
        values=np.array([[.1,.2],[4.,3.],[-7.,-6.],[30.,25.]])
        actual=shot_marginal_loglike(values,target,chol,[-.5,.8])
        for prediction,logp in zip(values,actual):
            def objective(s):
                w=np.linalg.solve(chol,prediction+s-target);return -.5*float(w@w)
            scores=[objective(s) for s in np.linspace(-.5,.8,1001)];maximum=max(scores)
            integral=quad(lambda s:np.exp(objective(s)-maximum),-.5,.8,epsabs=1e-13,epsrel=1e-12)[0]
            self.assertAlmostEqual(logp,np.log(integral)+maximum,places=10)

    def test_far_tail_difference_does_not_round_to_zero(self):
        actual=shot_marginal_loglike(np.array([[1000.,1000.]]),np.zeros(2),np.eye(2),[-.5,.5])
        self.assertTrue(np.isfinite(actual[0]))

    def test_independent_Gaussian_density_quantiles(self):
        x=np.linspace(-8,8,4001);b=np.linspace(-8,8,501)
        logw=-.5*(x[:,None]**2+b[None,:]**2)
        result=marginal_pdf(x,b,logw)
        self.assertAlmostEqual(result['quantiles'][2],0.,places=10)
        self.assertAlmostEqual(result['sigma'],1.,places=8)
        self.assertAlmostEqual(result['quantiles'][1],-.9944578832,places=5)

if __name__=='__main__':unittest.main()
