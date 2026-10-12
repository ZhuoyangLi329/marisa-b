"""模块大纲：完整48病例顺序、七/八参数输入与原sampler/RNG布局回归。

短链只作IO/RNG回归，不冒称正式MCMC收敛；不读取任何活动科学链。
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import emcee
import run_recalibrated_prepost_noise_comparison_v1 as run
from rsd_b000_contract import save_json,digest
from post_rsd_legacy3_model import NAMES as THREE_NAMES
from post_rsd_legacy4_model import NAMES as FOUR_NAMES

class RecalibratedComparisonTest(unittest.TestCase):
    def fixture(self,path,noise):
        rng=np.random.default_rng(10371);path.mkdir(parents=True)
        dim=4+noise;n=21
        names=list(THREE_NAMES if noise==3 else FOUR_NAMES)
        bounds=np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-20,20],[-10,10],[-300,300]])
        if noise==4:bounds=np.vstack((bounds,[-10,10]))
        covariance=np.diag(np.full(n,1e12));alpha=(500-n-2)/499
        arrays=dict(target=rng.normal(size=n),covariance=covariance,chol=np.linalg.cholesky(covariance/alpha),
            coefficients=rng.normal(size=(5,3,3,9)),templates=rng.uniform(.1,10,(6,12)),
            primitives=rng.uniform(.1,10,(7,9)),weight_moments=np.array([3000.,1e7]),bounds=bounds,indices=np.arange(9))
        np.savez_compressed(path/'likelihood.npz',**arrays)
        save_json(path/'contract.json',dict(name='fixture',stage='pre',truth_fNL=0,parameters=names,
            noise_parameter_count=noise,p_fixed_physical=1.3,Hartlap_factor=alpha,
            likelihood_sha256=digest(path/'likelihood.npz')))
        center=np.array([0,2.2,.1,-.2]+[0]*noise);transform=np.diag((bounds[:,1]-bounds[:,0])/50)
        save_json(path/'MAP.json',dict(center=center.tolist(),transform=transform.tolist(),chi2_MAP=0.,
            minuit_valid=True,prior_safe_initialization=True,candidates=[]))
        with patch.object(run,'guard',return_value={}):return run.ComparisonCase(path)

    def test_full24_before_extra24_and_all_required_prepost_cuts(self):
        self.assertEqual(len(run.SPECS),48)
        for phase,noise in enumerate((3,4)):
            rows=run.SPECS[phase*24:(phase+1)*24]
            self.assertEqual({x[0] for x in rows},{noise})
            self.assertEqual({x[1] for x in rows},{.08,.10,.12,.14})
            self.assertEqual({x[2] for x in rows},{'fiducial','LC_p','LC_m'})
            self.assertEqual({x[3] for x in rows},{'pre','post'})
            self.assertEqual(len(set(rows)),24);self.assertEqual(rows[0][1],.14)
        for i in range(24):self.assertEqual(run.SPECS[i][1:],run.SPECS[i+24][1:])

    def test_actual_seven_and_eight_parameter_loader_and_independent_prediction(self):
        with tempfile.TemporaryDirectory(prefix='noise-comparison-model-') as folder:
            for noise in (3,4):
                case=self.fixture(Path(folder)/str(noise),noise)
                rng=np.random.default_rng(42);rows=rng.uniform(case.bounds[:,0],case.bounds[:,1],(65,case.ndim))
                expected=np.array([run.independent(row,case) for row in rows]);actual=case.model.predictions(rows)
                np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=1e-5)
                direct=np.linalg.solve(case.chol,(expected-case.target).T).T
                np.testing.assert_allclose(case.residual(rows),direct,rtol=3e-13,atol=1e-10)

    def test_original_single_ensemble_RNG_and_resume_are_unchanged_for_both_dimensions(self):
        with tempfile.TemporaryDirectory(prefix='noise-comparison-rng-') as folder:
            for noise in (3,4):
                case=self.fixture(Path(folder)/str(noise),noise);case.initialize()
                single=case.directory/'continuous.h5';split=case.directory/'resumed.h5'
                run.sampler.advance_json_safe(case,single,1,17,90)
                run.sampler.advance_json_safe(case,split,1,17,31)
                run.sampler.advance_json_safe(case,split,1,17,90)
                a=emcee.backends.HDFBackend(str(single),name='ensemble_1',read_only=True)
                b=emcee.backends.HDFBackend(str(split),name='ensemble_1',read_only=True)
                for actual,expected in ((a.get_chain(),b.get_chain()),(a.get_log_prob(),b.get_log_prob()),(a.accepted,b.accepted)):
                    np.testing.assert_array_equal(actual,expected)
                self.assertEqual(a.get_chain().shape,(90,32,4+noise))
                for x,y in zip(a.random_state,b.random_state):
                    if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
                    else:self.assertEqual(x,y)

if __name__=='__main__':unittest.main()
