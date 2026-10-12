"""模块大纲：原39组数量/分箱；全P-B交叉块；同tracer P控制；拒绝未审计输入。"""
import unittest
import numpy as np
from post_rsd_ensemble import EnsembleSelection,assemble_blocks
from post_rsd_noir_case_inputs import case_specs,assemble_case,CUTS,COUNTS


class CaseInputsTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(202610092134)
        selection=EnsembleSelection(tuple(range(500)),overlap_policy='allow_explicit_target_overlap')
        rows={}
        for key in selection.measurement_keys:
            base=rng.normal(size=12);latent=rng.normal()
            rows[key]=dict(P_weighted_input=base+.1*latent,P_post=base+.3*rng.normal(size=12),
                           B_post=rng.normal(size=44)+.3*latent)
        self.ensemble=assemble_blocks(selection,rows,dict(P_weighted_input=12,P_post=12,B_post=44))
        maxima=np.concatenate([np.full(n-(COUNTS[i-1] if i else 0),cut) for i,(cut,n) in enumerate(zip(CUTS,COUNTS))])
        self.k=np.column_stack((maxima*.5,maxima));self.c=rng.normal(size=(5,3,3,3,44))
        self.p=rng.normal(size=(6,12));self.bnoise=rng.normal(size=(3,44));self.pnoise=np.full(12,3336.)
        self.audit=dict(status='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED',p_fixed_physical=1.417,
                        B_definition='matched',original_binning_preserved=True)

    def make(self,spec,audit=None):
        return assemble_case(self.ensemble,spec,self.k,self.c,self.p,self.bnoise,self.pnoise,
            p_halo=1.417,noise_audit=self.audit if audit is None else audit)

    def test_exact_original39_and_six_masks(self):
        specs=case_specs();self.assertEqual(len(specs),39);self.assertEqual(len({s.name for s in specs}),39)
        self.assertEqual([sum(s.observable==o for s in specs) for o in ('B_only','P0_only','P_plus_B')],[18,3,18])
        for spec in specs:
            arrays,meta=self.make(spec)
            self.assertFalse(meta['covariance_is_of_mean']);self.assertEqual(meta['n_target'],100)
            self.assertEqual(meta['B_bins'],0 if spec.kmax_B is None else COUNTS[CUTS.index(spec.kmax_B)])
            n=meta['P_bins']+meta['B_bins']
            np.testing.assert_allclose(arrays['chol']@arrays['chol'].T,arrays['covariance']/((500-n-2)/499.),rtol=2e-13,atol=1e-12)

    def test_joint_covariance_not_blockdiagonal_or_precision_slice(self):
        spec=next(s for s in case_specs() if s.observable=='P_plus_B' and s.kmax_B==.12)
        arrays,meta=self.make(spec);ix=arrays['indices'];full=self.ensemble['covariance_single_box']
        np.testing.assert_array_equal(arrays['covariance'],full[np.ix_(ix,ix)])
        self.assertGreater(np.max(np.abs(arrays['covariance'][:12,12:])),1e-3)
        self.assertFalse(np.allclose(arrays['covariance'],np.linalg.inv(np.linalg.inv(full)[np.ix_(ix,ix)])))
        self.assertEqual(meta['Pshot_physical_bounds'],[-1e6,1e6])
        np.testing.assert_array_equal(arrays['bounds'][-1]*meta['Pshot_unit_Mpc_h3'],[-1e6,1e6])

    def test_same_tracer_P_input_control_uses_own_cross_block(self):
        specs=case_specs(include_weighted_input_P_control=True);self.assertEqual(len(specs),57)
        control=next(s for s in specs if s.role!='main' and s.kmax_B==.14)
        arrays,meta=self.make(control);self.assertEqual(meta['P_block'],'P_weighted_input')
        np.testing.assert_array_equal(arrays['indices'][:12],np.arange(12))
        np.testing.assert_array_equal(arrays['covariance'],self.ensemble['covariance_single_box'][np.ix_(arrays['indices'],arrays['indices'])])

    def test_no_legacy_noise_or_default_algorithm_p(self):
        spec=case_specs()[0]
        for key,bad in (('status','legacy pre'),('p_fixed_physical',1.2822),('original_binning_preserved',False)):
            audit=dict(self.audit);audit[key]=bad
            with self.assertRaises(ValueError):self.make(spec,audit)
        with self.assertRaises(TypeError):
            assemble_case(self.ensemble,spec,self.k,self.c,self.p,self.bnoise,self.pnoise,noise_audit=self.audit)

    def test_polynomial_noise_mapping_keeps_original_priors_and_cross_space(self):
        spec=next(s for s in case_specs() if s.observable=='P_plus_B' and s.kmax_B==.12)
        rng=np.random.default_rng(202610092232);nc=rng.normal(size=(3,5,3,3,44))
        arrays,meta=assemble_case(self.ensemble,spec,self.k,self.c,self.p,self.bnoise,self.pnoise,
            p_halo=1.417,noise_audit=self.audit,B_noise_coefficients=nc)
        original,oldmeta=self.make(spec);nB=COUNTS[CUTS.index(.12)]
        self.assertEqual(arrays['noise_coefficients'].shape,(4,5,3,3,12+nB))
        np.testing.assert_array_equal(arrays['noise_coefficients'][:3,...,12:],nc[...,:nB])
        np.testing.assert_array_equal(arrays['noise_coefficients'][...,:12],0.)
        np.testing.assert_array_equal(arrays['noise_coefficients'][3],0.)
        np.testing.assert_array_equal(arrays['bounds'],original['bounds'])
        self.assertEqual(meta['parameters'],oldmeta['parameters'])
        self.assertEqual(meta['noise_representation'],'polynomial_plus_constant')
        with self.assertRaises(ValueError):
            assemble_case(self.ensemble,spec,self.k,self.c,self.p,self.bnoise,self.pnoise,
                p_halo=1.417,noise_audit=self.audit,B_noise_coefficients=nc[:,:4])


if __name__=='__main__':unittest.main()
