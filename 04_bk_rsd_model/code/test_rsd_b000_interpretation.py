"""模块大纲：隔离合成目标验证固定真值 profile；禁止改已有 likelihood/链。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rsd_b000_contract import NAMES,BOUNDS,save_json,digest
from run_rsd_b000_inference import predicted
from run_rsd_b000_mcmc import Case
from analyze_rsd_b000_results import profile_truth

class ProfileTest(unittest.TestCase):
    def test_fixed_truth_profile_preserves_likelihood(self):
        coef=np.zeros((4,3,3,12));noise=np.zeros((3,12))
        coef[0,0,1,0]=1/30;coef[1,0,0,1]=17.5;coef[0,1,0,2]=5;coef[0,2,0,3]=5
        for i in range(3):noise[i,4+i]=4
        physical=np.array([80,1.8,-.2,-.3,.1,.2,-.1]);data=predicted(physical,coef,noise)[0]
        with tempfile.TemporaryDirectory(prefix='rsd_b000_truth_profile_QA_') as folder:
            root=Path(folder);np.savez(root/'likelihood.npz',target=data,chol=np.eye(12),coefficients=coef,noise=noise,
                                      bounds=np.vstack((BOUNDS,[-10.,10.])))
            original=digest(root/'likelihood.npz')
            save_json(root/'contract.json',{'parameters':list(NAMES)+['R_stoch_anisotropic'],'truth_fNL':0,
                      'P_likelihood':False,'P_derived_b1_prior':False,'B_bins':12,'likelihood_sha256':original})
            case=Case(root);result=profile_truth(case,{}, {'center':physical.tolist(),'chi2_MAP':0.})
            self.assertEqual(result['profile_parameters'][0],0)
            self.assertAlmostEqual(result['delta_chi2_fixed_vs_free'],(80/30)**2,places=4)
            self.assertFalse(result['free_MAP_potential_optimization_issue'])
            self.assertEqual(digest(root/'likelihood.npz'),original);self.assertFalse((root/'chain.h5').exists())
            self.assertFalse((root/'MAP.json').exists())

if __name__=='__main__':unittest.main()
