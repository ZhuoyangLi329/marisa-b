"""模块大纲：用户授权的配置变化被接受；仅实际数组维度错误报错。"""
import unittest
import numpy as np
from legacy3_scan_contract import validate, configuration_snapshot, comparison_differences

NAMES = ['fNL','b1','b2_native','bK2','Ashot_residual','Bshot_residual','Pshot_residual_nbar']
CUTS = (.08,.10,.12,.14)
COUNTS = (9,14,20,27)
BOUNDS = np.array([[-500.,500.],[.5,4.],[-4.,6.],[-4.,4.],[-20.,20.],[-10.,10.]])


class UniformLegacy3Tests(unittest.TestCase):
    def fixture(self, stage='post', cut=.14):
        nb=COUNTS[CUTS.index(cut)];n=12+nb;nc=402 if stage=='post' else 500
        meta=dict(parameters=list(NAMES),stage=stage,kmax_B=cut,B_bins=nb,kmax_P=.08,P_bins=12,
            n_target=100,IR=False,P_B_cross_covariance=True,covariance_is_of_mean=False,
            covariance_multiplier=1,volume_Gpc_h3=1,p_uncertainty_propagated=False,n_covariance=nc)
        arrays=dict(bounds=np.vstack((BOUNDS,[-1e6/3300,1e6/3300])),weight_moments=np.array([3300.,1.1e7]),
            target=np.zeros(n),covariance=np.eye(n),chol=np.eye(n)/np.sqrt((nc-n-2)/(nc-1)),
            coefficients=np.zeros((5,3,3,nb)),primitives=np.zeros((7,nb)),templates=np.zeros((6,12)))
        return meta,arrays

    def test_all_four_cuts_both_stages_same_contract(self):
        for stage in ['pre','post']:
            for cut in CUTS:
                meta,arrays=self.fixture(stage,cut);self.assertGreater(validate(meta,arrays),0)

    def test_authorized_parameter_adjustment_not_policy_rejected(self):
        meta,arrays=self.fixture();meta['parameters'].append('future_response')
        arrays['bounds']=np.vstack((arrays['bounds'],[-2,2]))
        self.assertGreater(validate(meta,arrays),0)
        meta['parameters'][-1]='renamed_future_response'
        self.assertGreater(validate(meta,arrays),0)

    def test_authorized_nuisance_prior_change_is_recorded(self):
        meta,arrays=self.fixture();arrays['bounds'][4,0]=-40
        self.assertGreater(validate(meta,arrays),0)
        self.assertEqual(configuration_snapshot(meta,arrays)['bounds'][4][0],-40)

    def test_authorized_new_cohort_or_scan_range_is_not_policy_rejected(self):
        for key,value in [('n_covariance',500),('kmax_B',.18),('IR',True)]:
            meta,arrays=self.fixture();meta[key]=value
            self.assertGreater(validate(meta,arrays),0)

    def test_differences_are_reported_not_rejected(self):
        meta,arrays=self.fixture();first=configuration_snapshot(meta,arrays)
        arrays['bounds'][4,0]=-40;second=configuration_snapshot(meta,arrays)
        self.assertIn('bounds',comparison_differences(first,second))

    def test_actual_dimension_error_still_diagnosed(self):
        meta,arrays=self.fixture();meta['parameters'].append('not_implemented_in_arrays')
        with self.assertRaises(ValueError):validate(meta,arrays)


if __name__=='__main__':unittest.main()
