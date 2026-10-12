"""模块大纲：临时数值矩阵的独立回放/缺节点/NN污染保护；标量闭合独立式。"""
import json
from pathlib import Path
import unittest
import numpy as np
from audit_post_rsd_noise_transport_coefficients import actual_job,evaluate_independent,replay_bin,scalar_oracle,integer_count_oracle
from post_rsd_poisson_projection import PoissonShiftFields,PoissonShiftFFTProjection
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_noise_transport import integrate_noise_transport
from unittest.mock import patch
import rsd_b000_fft_projection as fft_source


class OperatorReplayTest(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'outputs/sto_marisa_post_rsd_mean100_v1/noise_transport_templates_v1/CONTRACT.json'
        self.contract=json.loads(path.read_text());rng=np.random.default_rng(202610092317)
        self.c=rng.normal(size=(3,3,3,3))*1e-5;self.c[:,:,2]=0.;self.c[0,0,2,0]=-.7
        # 测试预言用另一表达式；仅临时矩阵，不生成伪科学验收文件。
        def independent(b,p):
            bc,br=self.contract['coordinate_centers_and_scales']['b1'];pc,pr=self.contract['coordinate_centers_and_scales']['p']
            x,y=(b-bc)/br,(p-pc)/pr
            return np.polynomial.chebyshev.chebval2d(x,y,self.c)
        self.nodes=np.array([[independent(b,p) for p in self.contract['physical_p_nodes']] for b in self.contract['b1_nodes']])
        self.probes=[dict(b1=b,p_halo=p,direct=independent(b,p).tolist(),predicted=independent(b,p).tolist())
            for b,p in self.contract['probes']]

    def test_independent_full_nine_nodes_and_two_probes(self):
        result=replay_bin(self.c,self.nodes,self.probes,self.contract)
        self.assertLess(result['max_node_scaled_error'],1e-14)
        self.assertLess(result['max_fine_probe_scaled_error'],1e-14)

    def test_missing_or_corrupt_node_and_probe_cannot_pass(self):
        with self.assertRaises(ValueError):replay_bin(self.c,self.nodes[:2],self.probes,self.contract)
        with self.assertRaises(ValueError):replay_bin(self.c,self.nodes,self.probes[:1],self.contract)
        changed=self.nodes.copy();changed[1,1,0,2]+=.1
        with self.assertRaises(ValueError):replay_bin(self.c,changed,self.probes,self.contract)

    def test_NN_fnl_or_physical_coordinate_contamination_rejected(self):
        changed=self.c.copy();changed[0,0,2,1]=1e-4
        with self.assertRaises((ValueError,AssertionError)):replay_bin(changed,self.nodes,self.probes,self.contract)
        changed=self.c.copy();changed[1,0,2,0]=.1
        with self.assertRaises((ValueError,AssertionError)):replay_bin(changed,self.nodes,self.probes,self.contract)

    def test_accounting_absence_or_pending_not_completed(self):
        record=actual_job('59594709|PENDING|0:0|\n59594709.batch|COMPLETED|0:0|\n',59594709)
        self.assertEqual(record['state'],'PENDING')
        self.assertEqual(actual_job('59594709|FAILED|1:0|\n',59594709)['state'],'FAILED')
        with self.assertRaises(ValueError):actual_job('',59594709)
        with self.assertRaises(ValueError):actual_job('59594709|COMPLETED|0:0|\n'*2,59594709)

    def test_scalar_closure_oracle_vs_vector_transport_at_points(self):
        rng=np.random.default_rng(202610092318);v=rng.normal(size=(71,3,3))*.03;v[:,2]=-v[:,:2].sum(axis=1)
        k=np.linalg.norm(v,axis=-1);p=10000/(1+(k/.09)**2);m=1000*(1+(k/.02)**2)
        rec=FixedReconstruction(10.,2.2280546354,1.2822,100.)
        for b,ph in ((.63,.9),(2.11,1.417),(3.83,1.72)):
            fields=PoissonShiftFields(v,p,m,b,.7625790037720044,p_halo=ph,reconstruction=rec)
            np.testing.assert_allclose(integrate_noise_transport(fields)[1:],scalar_oracle(fields),rtol=5e-14,atol=2e-11)
            scalar=scalar_oracle(fields)
            for f in (-500.,0.,500.):
                value=np.einsum('afn,f->an',scalar,[1,f,f*f])
                self.assertTrue(np.all(value<=1e-10))

    def test_independent_integer_counts_zero_closing_and_full_denominator(self):
        sample=lambda k:(10000/(1+(k/.09)**2),1000*(1+(k/.02)**2))
        rec=FixedReconstruction(10.,2.2280546354,1.2822,100.)
        edges=np.array([0.,.02035714285714286,.04107142857142857])
        with patch.object(fft_source,'load_spectrum',return_value=sample):
            for pair in ((0,0),(0,1),(1,1)):
                fields=PoissonShiftFFTProjection(32,edges,pair,2.11,.7625790037720044,p_halo=1.417,reconstruction=rec)
                independent,meta=integer_count_oracle(fields)
                expected=integrate_noise_transport(fields)[1:]
                np.testing.assert_allclose(independent,expected,rtol=5e-13,atol=2e-10)
                self.assertEqual(meta['full_denominator'],fields.denominator)
                self.assertLess(meta['maximum_integer_FFT_roundoff'],1e-5)
                if pair[0]==pair[1]:self.assertLess(meta['nonzero_closing_pairs'],meta['nonzero_input_pairs'])


if __name__=='__main__':unittest.main()
