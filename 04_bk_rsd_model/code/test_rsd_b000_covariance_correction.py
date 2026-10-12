"""模块大纲：均值 data/单盒 covariance 防回归；只改统计缩放；旧输出只读。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_rsd_b000_inference as flow
from rsd_b000_contract import CUTS,NAMES,BOUNDS,digest,save_json,selected,reference

class CovarianceCorrectionTest(unittest.TestCase):
    def test_only_covariance_changes_and_legacy_is_immutable(self):
        rng=np.random.default_rng(20261002);data=rng.normal(size=(800,120))
        cov=np.cov(data[300:],rowvar=False,ddof=1);names=list(NAMES)+['R_stoch_anisotropic']
        bounds=np.vstack((BOUNDS,[-10,10]));coef=rng.normal(size=(4,3,3,44));noise=rng.normal(size=(3,44))
        with tempfile.TemporaryDirectory(prefix='rsd_single_cov_QA_') as directory:
            source=Path(directory)/'old';target=Path(directory)/'new';source.mkdir()
            save_json(source/'contract.json',{'source_library_sha256':'synthetic','sampler':{'synthetic':True}})
            save_json(source/'MODEL_NOISE_AUDIT.json',{'parameter_count':7,'native_SP_independent_combination_retained':True})
            np.savez(source/'B000_ENSEMBLE_MATRIX.npz',B=data,covariance_single_box=cov)
            certified=list(reference()['certified_indices'])
            for j,c in enumerate(('fiducial','LC_p','LC_m')):
                for cut in CUTS:
                    name=f'{c}_mean100_k{cut:.2f}';root=source/'cases'/name;root.mkdir(parents=True)
                    idx=selected(cut);take=[certified.index(i) for i in idx];alpha=(500-len(idx)-2)/499
                    np.savez(root/'likelihood.npz',target=data[j*100:(j+1)*100].mean(axis=0)[idx],
                             covariance=cov[np.ix_(idx,idx)]/100,chol=np.linalg.cholesky(cov[np.ix_(idx,idx)]/100/alpha),
                             coefficients=coef[...,take],noise=noise[:,take],bounds=bounds,indices=idx)
                    save_json(root/'contract.json',{'name':name,'cosmology':c,'truth_fNL':[0,100,-100][j],
                              'kmax_B':cut,'volume_Gpc_h3':100,'n_target':100,'n_covariance':500,
                              'P_likelihood':False,'P_derived_b1_prior':False,'parameters':names,'case_type':'baseline',
                              'Hartlap_factor':alpha,'likelihood_sha256':digest(root/'likelihood.npz')})
            hashes={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
            with patch.object(flow,'OUT',target):
                flow.correct_covariance(source)
                import json
                cases=json.loads((target/'CASES.json').read_text());self.assertEqual(len(cases),18)
                for name in cases:
                    with np.load(target/'cases'/name/'likelihood.npz') as z:
                        idx=z['indices'];np.testing.assert_array_equal(z['covariance'],cov[np.ix_(idx,idx)])
                        meta=json.loads((target/'cases'/name/'contract.json').read_text())
                        self.assertEqual(meta['volume_Gpc_h3'],1);self.assertEqual(meta['n_target'],100)
                        self.assertEqual(meta['parameters'],names)
                        np.testing.assert_allclose(z['chol']@z['chol'].T,z['covariance']/meta['Hartlap_factor'],atol=1e-14)
                    self.assertFalse((target/'cases'/name/'chain.h5').exists())
                with self.assertRaises(ValueError):flow.correct_covariance(source)
            self.assertEqual(hashes,{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()})

if __name__=='__main__':unittest.main()
