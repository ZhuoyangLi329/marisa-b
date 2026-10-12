"""模块大纲：独立有向Z1平方；Fid500边缘C；已知物理p注入与统计合同拒绝。"""
import copy
import unittest
import numpy as np
from scipy.optimize import least_squares
from post_rsd_ensemble import EnsembleSelection, assemble_blocks
from post_rsd_p_calibration import make_calibration_arrays, PhysicalPCalibrationCase
from rsd_b000_contract import F_GROWTH


class PhysicalPCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(202610090901)
        self.mu=rng.uniform(-1,1,(12,41)); self.power=rng.uniform(5000,20000,(12,41))
        self.mass=rng.uniform(700,30000,(12,41)); self.weight=rng.uniform(.1,1,(12,41))
        self.weight/=self.weight.sum(axis=1)[:,None]
        p,m,u=self.power,self.mass,self.mu**2
        self.templates=np.array([np.sum(self.weight*a,axis=1) for a in (p,p*u,p*u*u,p/m,p*u/m,p/m**2)])
        selection=EnsembleSelection(tuple(range(500)),overlap_policy='allow_explicit_target_overlap')
        rows={}
        for key in selection.measurement_keys:
            values=rng.normal(size=12)
            rows[key]={'B':rng.normal(size=2),'P_weighted_input':values,'P_post':values+rng.normal(0,.2,12)}
        self.ensemble=assemble_blocks(selection,rows,{'B':2,'P_weighted_input':12,'P_post':12})
        self.holdout=rng.normal(10,1,(150,12))
        self.arrays=make_calibration_arrays(self.ensemble,self.holdout,holdout_ids=range(100,250))
        self.case=PhysicalPCalibrationCase(self.arrays['target'],self.arrays['covariance'],self.arrays['chol'],self.templates)

    def test_independent_oriented_P0_and_priors(self):
        for p,b,shot in ((1.41,2.18,73.),(.4,.8,-100.),(2.7,3.6,200.)):
            z=b+F_GROWTH*self.mu**2+100.*2*1.686*(b-p)/self.mass
            direct=np.sum(self.weight*self.power*z*z,axis=1)+shot
            np.testing.assert_allclose(self.case.predictions((p,b,shot)),direct,rtol=3e-14,atol=1e-9)
        values=np.array([[1.41,2.18,73.],[1.6,2.3,-10.]])
        np.testing.assert_allclose(self.case.log_probability(values),-.5*self.case.objective_many(values))
        self.assertEqual(self.case.log_probability((0.,2.18,73.)),-np.inf)
        self.assertEqual(self.case.log_probability((1.41,np.nan,73.)),-np.inf)

    def test_complete_fid500_marginal_C_not_mean_or_conditional(self):
        C=self.ensemble['covariance_single_box'][2:14,2:14]
        np.testing.assert_array_equal(self.arrays['covariance'],C)
        np.testing.assert_array_equal(self.arrays['target'],self.holdout.mean(axis=0))
        self.assertFalse(np.allclose(C,self.arrays['covariance']/150))
        self.assertFalse(np.allclose(C,np.linalg.inv(np.linalg.inv(self.ensemble['covariance_single_box'])[2:14,2:14])))
        self.assertEqual(self.arrays['Hartlap_factor'],486/499.)
        self.assertEqual(self.arrays['metadata']['potential_shared_phase_ids_with_covariance'],list(range(100,250)))
        self.assertFalse(self.arrays['metadata']['new_main_MCMC_parameter_added'])
        for name,value in (('n_covariance',499),('covariance_is_of_mean',True),('volume_Gpc_h3',100.)):
            bad=copy.deepcopy(self.ensemble);bad['statistics'][name]=value
            with self.assertRaises(ValueError):make_calibration_arrays(bad,self.holdout,holdout_ids=range(100,250))
        with self.assertRaises(ValueError):make_calibration_arrays(self.ensemble,self.holdout[:100],holdout_ids=range(100,200))

    def test_known_physical_p_injection_recovered(self):
        truth=np.array([1.41,2.18,73.])
        z=truth[1]+F_GROWTH*self.mu**2+100.*2*1.686*(truth[1]-truth[0])/self.mass
        target=np.sum(self.weight*self.power*z*z,axis=1)+truth[2]
        case=PhysicalPCalibrationCase(target,self.arrays['covariance'],self.arrays['chol'],self.templates)
        fit=least_squares(lambda a:np.linalg.solve(case.cholesky,case.predictions(a)-target),[1.,2.,0.],
                          x_scale=[1.,1.,1.e4],ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=300)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.x,truth,rtol=2e-8,atol=2e-6)


if __name__=='__main__':unittest.main()
