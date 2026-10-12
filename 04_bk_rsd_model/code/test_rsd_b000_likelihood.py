"""模块大纲：pure B 维度/先验防护；直接预测与 whitened likelihood；仿射坐标回归。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rsd_b000_contract import BOUNDS,NAMES,digest,save_json
from run_rsd_b000_inference import predicted
from run_rsd_b000_mcmc import Case,checkpoint_targets,diagnose
import emcee

class LikelihoodTest(unittest.TestCase):
    def test_minuit_hdf_sampler_smoke_not_scientific_recovery(self):
        n=9;coef=np.zeros((4,3,3,n));noise=np.zeros((2,n))
        coef[0,0,1,0]=1/30;coef[1,0,0,1]=17.5
        coef[0,1,0,2]=5;coef[0,2,0,3]=5;noise[0,4]=1/.3;noise[1,5]=1/.4
        truth=np.array([80,1.8,-.2,-.3,.1,.2]);target=predicted(truth,coef,noise)[0]
        with tempfile.TemporaryDirectory(prefix='rsd_b000_sampler_QA_') as folder:
            root=Path(folder);np.savez(root/'likelihood.npz',target=target,chol=np.eye(n),coefficients=coef,noise=noise,bounds=BOUNDS)
            meta={'P_likelihood':False,'P_derived_b1_prior':False,'parameters':list(NAMES),'B_bins':n,
                  'truth_fNL':80,'likelihood_sha256':digest(root/'likelihood.npz')};save_json(root/'contract.json',meta)
            case=Case(root);mapmeta=case.initialize();self.assertLess(mapmeta['chi2_MAP'],1e-4)
            np.testing.assert_allclose(case.center,truth,atol=.01)
            chains=[];accepted=[]
            for ensemble in range(4):
                backend=emcee.backends.HDFBackend(str(root/'chain.h5'),name=f'ensemble_{ensemble}');backend.reset(32,6)
                sampler=emcee.EnsembleSampler(32,6,case.log_prob,vectorize=True,backend=backend)
                sampler.random_state=np.random.RandomState(611+ensemble).get_state()
                sampler.run_mcmc(case.initial(333+ensemble,ensemble,mapmeta),80,skip_initial_state_check=True)
                chains.append(backend.get_chain());accepted.append(backend.accepted)
            report,flat=diagnose(case,np.stack(chains),np.stack(accepted),80)
            self.assertFalse(report['converged']);self.assertEqual(flat.shape,(4*40*32,6))
            self.assertTrue(np.isfinite(flat).all());self.assertEqual(backend.iteration,80)

    def test_unequal_ensemble_checkpoint_resume(self):
        self.assertEqual(list(checkpoint_targets([0,0,0,0]))[0],6000)
        self.assertEqual(list(checkpoint_targets([8000,6903,6000,6000]))[0],8000)
        self.assertEqual(list(checkpoint_targets([8050,8000,8000,8000]))[0],10000)
        self.assertEqual(list(checkpoint_targets([24000,23000,22000,22000])),[24000])
        with self.assertRaises(ValueError):checkpoint_targets([24001])

    def test_bonly_prediction_and_whitening(self):
        rng=np.random.default_rng(99);n=20
        coef=rng.normal(size=(4,3,3,n))*np.array([1e8,1e3,.1])[None,None,:,None]
        noise=rng.normal(size=(2,n))*1e7;truth=np.array([100,1.9,-.3,-.2,.1,-.1])
        target=predicted(truth,coef,noise)[0]
        chol=np.eye(n)*1e7
        with tempfile.TemporaryDirectory(prefix='rsd_b000_likelihood_') as folder:
            root=Path(folder);np.savez(root/'likelihood.npz',target=target,chol=chol,coefficients=coef,noise=noise,bounds=BOUNDS)
            meta={'P_likelihood':False,'P_derived_b1_prior':False,'parameters':list(NAMES),'B_bins':n,
                  'likelihood_sha256':digest(root/'likelihood.npz')};save_json(root/'contract.json',meta)
            case=Case(root);self.assertLess(case.objective(*truth),1e-20)
            probes=np.array([truth,truth+np.array([5,.01,.03,0,0,0])])
            np.testing.assert_allclose(case.residual(probes),(predicted(probes,coef,noise)-target)/1e7,rtol=2e-12,atol=1e-12)
            case.center=truth;case.transform=np.diag([10,.1,.1,.1,.2,.2])
            coords=case.coordinates(probes);np.testing.assert_allclose(case.physical(coords),probes)
            self.assertAlmostEqual(case.log_prob(coords)[0],0.,places=14)
            rejected=case.coordinates(np.array([501,1.9,0,0,0,0]))
            self.assertEqual(case.log_prob(rejected[None])[0],-np.inf)
            meta['P_likelihood']=True;save_json(root/'contract.json',meta)
            with self.assertRaises(ValueError):Case(root)

if __name__=='__main__':unittest.main()
