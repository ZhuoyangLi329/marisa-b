"""模块大纲：真实job终止状态识别、完整统计门、有符号权重与身份保护。"""
import copy
import unittest
import continue_sto_post_physical_p as follow
import verify_sto_post_fid500_admission as admission


class ContinuationTest(unittest.TestCase):
    def test_only_exact_job_record(self):
        for state in ('PENDING','RUNNING','COMPLETED','CANCELLED by 112513'):
            observed=follow.job_record('59592166|'+state+'|0:0\n59592166.batch|COMPLETED|0:0',59592166)
            self.assertEqual(observed['state'],state.split()[0])
        for output in ('','59592165|COMPLETED|0:0','59592166|COMPLETED|0:0\n59592166|PENDING|0:0'):
            with self.assertRaises(ValueError):follow.job_record(output,59592166)

    def test_complete_data_and_unaveraged_covariance_required(self):
        value=dict(status='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE',
            statistics=dict(n_covariance=500,n_target_per_truth=100),rows=[{}]*700,
            covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.)
        follow.ensemble_gate(value)
        for key,bad in (('rows',[{}]*699),('covariance_is_of_mean',True),('covariance_multiplier',.01),('volume_Gpc_h3',100)):
            changed=copy.deepcopy(value);changed[key]=bad
            with self.assertRaises(ValueError):follow.ensemble_gate(changed)

    def test_signed_third_moment_not_new_positivity_prior(self):
        row=dict(N2_data_self=2.,N3_data_self=-1.,n_halo=10,raw_fof=dict(n_halo=10))
        admission.validate_moments(row)
        for key,bad in (('N2_data_self',0.),('N3_data_self',float('nan')),('n_halo',11)):
            changed=copy.deepcopy(row);changed[key]=bad
            with self.assertRaises(ValueError):admission.validate_moments(changed)

    def test_only_derived_moment_roundoff_with_exact_source_SHA(self):
        a=dict(source='original',source_sha256='same',diagnostics_sha256='same',n_halo=10,
               N2_data_self=3336.7563161008284,N3_data_self=11652462.65745907,
               raw_fof=dict(source_raw_sha256='same'))
        b=copy.deepcopy(a);b['N2_data_self']=3336.7563161008275
        with self.assertRaises(ValueError):admission.science_equal(a,b)
        admission.science_equal(a,b,derived_moment_roundoff=True)
        b['source_sha256']='different'
        with self.assertRaises(ValueError):admission.science_equal(a,b,derived_moment_roundoff=True)
        b=copy.deepcopy(a);b['N2_data_self']*=1.00000001
        with self.assertRaises(ValueError):admission.science_equal(a,b,derived_moment_roundoff=True)


if __name__=='__main__':unittest.main()
