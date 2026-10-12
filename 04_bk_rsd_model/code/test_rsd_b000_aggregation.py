"""模块大纲：纯合成数据验证 800-row 顺序、体积/Hartlap、21+6 合同及 noise 秩。

隔离临时目录，不读写正式链，不将统计程序测试当作物理模型恢复验收。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_rsd_b000_inference as flow
from rsd_b000_contract import save_json,reference,CUTS,BOUNDS

class AggregateTest(unittest.TestCase):
    def exercise(self,extra_noise):
        ref=reference();idx=ref['certified_indices'];rng=np.random.default_rng(844)
        rows=[{'cosmology':c,'realization':i,'role':role}
              for c,ids,role in [('fiducial',range(100),'validation'),('LC_p',range(100),'validation'),
                                 ('LC_m',range(100),'validation'),('fiducial',range(1000,1500),'covariance')]
              for i in ids]
        data=rng.normal(size=(800,120));phase=rng.normal(size=(600,122))+1j*rng.normal(size=(600,122))
        lookup={(r['cosmology'],r['realization']):j for j,r in enumerate(rows)}
        coef=np.zeros((4,3,3,44));sg=np.stack([np.linspace(1,3,44),np.linspace(1,3,44)/3,
                                            np.linspace(1,3,44)/5,np.ones(44)])
        if extra_noise:sg[2]+=np.sin(np.arange(44))*.05
        def read(c,i,n=256):
            j=lookup[(c,i)];p=i if j<300 else j-200
            return {'B':data[j],'B000_nmodes':ref['B000_nmodes'],'B000_k_edges':ref['B000_k_edges'],
                    'phase_fingerprint':phase[p]}, {'nbar_h3_Mpc3':.0003+(j%13)*1e-7}
        with tempfile.TemporaryDirectory(prefix='rsd_b000_contract_QA_') as directory:
            root=Path(directory);save_json(root/'contract.json',{'samples':rows,'synthetic_QA_only':True})
            save_json(root/'PILOT_AUDIT.json',{'status':'PASS','synthetic_QA_only':True})
            with patch.object(flow,'OUT',root),patch.object(flow,'read_measurement',read),patch.object(flow,'templates',return_value=(coef,sg,np.zeros(44),np.zeros((3,3,44)),np.ones((44,2)))):
                flow.aggregate();names=json.loads((root/'CASES.json').read_text());self.assertEqual(len(names),21)
                audit=json.loads((root/'MODEL_NOISE_AUDIT.json').read_text())
                self.assertEqual(audit['parameter_count'],7 if extra_noise else 6)
                actualcov=np.cov(data[300:],rowvar=False,ddof=1)
                for name in names:
                    case=root/'cases'/name;meta=json.loads((case/'contract.json').read_text())
                    with np.load(case/'likelihood.npz') as z:
                        selected=z['indices'];n=100 if 'mean100' in name else 1
                        np.testing.assert_allclose(z['covariance'],actualcov[np.ix_(selected,selected)])
                        np.testing.assert_allclose(z['target'],data[lookup[(meta['cosmology'],0)]:lookup[(meta['cosmology'],0)]+n].mean(axis=0)[selected])
                        np.testing.assert_allclose(z['chol']@z['chol'].T,z['covariance']/meta['Hartlap_factor'],atol=1e-14)
                        self.assertEqual(z['noise'].shape,(3 if extra_noise else 2,len(selected)))
                    self.assertEqual(meta['volume_Gpc_h3'],1.);self.assertFalse(meta['covariance_is_of_mean']);self.assertFalse(meta['P_likelihood'])
                    self.assertFalse(meta['P_derived_b1_prior']);self.assertEqual(meta['parameters'][1],'b1')
                    if 'mean100' in name:self.assertEqual(meta['B_bins'],[9,14,20,27,35,44][list(CUTS).index(meta['kmax_B'])])
                    save_json(case/'SUMMARY.json',{'converged':True,'gof_pvalue':.5,'fNL_pull':0,'synthetic_QA_only':True})
                aliases=json.loads((root/'ALIASES.json').read_text());self.assertEqual(len(aliases),3)
                flow.sensitivity();allcases=json.loads((root/'CASES.json').read_text());self.assertEqual(len(allcases),27)
                flow.sensitivity();self.assertEqual(json.loads((root/'CASES.json').read_text()),allcases)
                for name in allcases[21:]:
                    with np.load(root/'cases'/name/'likelihood.npz') as z:
                        np.testing.assert_array_equal(z['bounds'][:2],BOUNDS[:2])
                        np.testing.assert_array_equal(z['bounds'][2:6],BOUNDS[2:]*2)

    def test_two_shape_contract_and_volume(self):self.exercise(False)
    def test_independent_anisotropic_noise_is_retained(self):self.exercise(True)

if __name__=='__main__':unittest.main()
