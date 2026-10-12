"""模块大纲：在原NERSC采样环境验证Case接口；临时fixture，不是正式拟合。

核对quartic post-B、全P-B白化、物理p、Hartlap/噪声门、校准盲MAP和四组
walker初始化。测试不生成主链，不改共享proposal/RNG/收敛实现。
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import run_sto_post_noir_mcmc as post
import run_sto_physical_p_mcmc as calibration
from post_rsd_p_calibration import PARAMETERS,BOUNDS,PhysicalPCalibrationCase


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class SamplerAdapterTest(unittest.TestCase):
    def make_post(self,root,nP,nB):
        rng=np.random.default_rng(202610091309+nP+nB);n=nP+nB
        directory=root/'case';directory.mkdir()
        coefficients=rng.normal(size=(5,3,3,nB))*1e-4
        templates=rng.uniform(1,2,(6,nP))
        if nB:
            names=['fNL','b1','b2_native','bK2','A_stoch','S_stoch','R_stoch_anisotropic']
            bounds=[[-500,500],[.5,4],[-4,6],[-4,4],[-10,10],[-20,20],[-10,10]]
            if nP:names+=['Pshot_residual_nbar'];bounds+=[[-20,20]]
            noise=rng.normal(size=(len(names)-4,n))
        else:
            names=['fNL','b1','Pshot_residual_nbar'];bounds=[[-500,500],[.5,4],[-20,20]]
            noise=rng.uniform(.5,2,n)
        matrix=rng.normal(size=(n,n));covariance=matrix@matrix.T+np.eye(n)
        alpha=(500-n-2)/499.;chol=np.linalg.cholesky(covariance/alpha)
        np.savez_compressed(directory/'likelihood.npz',target=rng.normal(size=n),covariance=covariance,
            chol=chol,coefficients=coefficients,templates=templates,noise=noise,bounds=bounds)
        write(root/'CONTRACT.json',dict(status='POST_NOIR_INPUTS_VALIDATED_FOR_MCMC',immutable_inputs={},
            scope='temporary numerical interface fixture; not actual scientific admission'))
        meta=dict(parameters=names,bounds=bounds,parent_contract_sha256=post.digest(root/'CONTRACT.json'),
            IR=False,n_covariance=500,n_target=100,volume_Gpc_h3=1,covariance_multiplier=1,
            covariance_is_of_mean=False,P_derived_b1_prior=False,p_fixed_physical=1.417,
            P_bins=nP,B_bins=nB,Hartlap_factor=alpha,P_B_cross_covariance=bool(nP and nB),
            noise_basis_status='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED',
            case_type='P0_only' if not nB else ('P_plus_B' if nP else 'B_only'),
            likelihood_sha256=post.digest(directory/'likelihood.npz'))
        write(directory/'contract.json',meta)
        return directory,meta

    def test_Bonly_and_PB_factories_use_quartic_and_full_joint_covariance(self):
        for nP in (0,12):
            with TemporaryDirectory(prefix='sto_case_fixture_') as tmp:
                root=Path(tmp);directory,meta=self.make_post(root,nP,9)
                with patch.object(post,'OUT',root):case=post.factory(directory)
                self.assertIsInstance(case,post.PostBCase)
                v=np.array([[73.,2.3,.3,-.2]+[.2]*(case.ndim-4)])
                f,b,b2,bk=v[0,:4];x=(b-2.25)/1.75
                # 不调用生产predict_noir：手写T0..T4作为独立接口预言。
                basis=np.array([1,x,2*x*x-1,4*x**3-3*x,8*x**4-8*x*x+1])
                expected=np.zeros(nP+9)
                expected[nP:]=np.einsum('i,j,k,ijkn->n',basis,[1,b2,bk],[1,f,f*f],case.coefficients)
                if nP:
                    growth=.7625790037720044;bp=2*1.686*(b-meta['p_fixed_physical']);t=case.templates
                    expected[:nP]=b*b*t[0]+2*b*growth*t[1]+growth**2*t[2]+2*f*bp*(b*t[3]+growth*t[4])+f*f*bp*bp*t[5]
                expected+=v[0,4:]@case.noise
                np.testing.assert_allclose(case.residual(v)[0],np.linalg.solve(case.chol,expected-case.target),
                    rtol=2e-12,atol=1e-9)

    def test_Ponly_factory_uses_explicit_physical_p(self):
        with TemporaryDirectory(prefix='sto_case_fixture_') as tmp:
            root=Path(tmp);directory,meta=self.make_post(root,12,0)
            with patch.object(post,'OUT',root):case=post.factory(directory)
            self.assertIsInstance(case,post.PostPOnlyCase)
            f,b,shot=123.,2.2,-.3;bp=2*1.686*(b-meta['p_fixed_physical']);g=.7625790037720044;t=case.templates
            expected=b*b*t[0]+2*b*g*t[1]+g*g*t[2]+2*f*bp*(b*t[3]+g*t[4])+f*f*bp*bp*t[5]+shot*case.noise
            np.testing.assert_allclose(case.predictions([[f,b,shot]])[0],expected,rtol=3e-14,atol=1e-9)
            np.testing.assert_allclose(case.residual([[f,b,shot]])[0],np.linalg.solve(case.chol,expected-case.target),
                rtol=3e-14,atol=1e-9)

    def test_factory_preserves_parameter_dependent_post_noise(self):
        for nP in (0,12):
            with TemporaryDirectory(prefix='sto_poly_noise_fixture_') as tmp:
                root=Path(tmp);directory,meta=self.make_post(root,nP,9)
                with np.load(directory/'likelihood.npz') as z:arrays={k:np.array(z[k]) for k in z.files}
                count=len(meta['parameters'])-4;n=nP+9
                nc=np.zeros((count,5,3,3,n));nc[0,1,0,1,nP:]=np.arange(1,10)*.01
                np.savez_compressed(directory/'likelihood.npz',**arrays,noise_coefficients=nc)
                meta['likelihood_sha256']=post.digest(directory/'likelihood.npz');meta['noise_representation']='polynomial_plus_constant'
                write(directory/'contract.json',meta)
                with patch.object(post,'OUT',root):case=post.factory(directory)
                row=np.array([[73.,2.3,.3,-.2]+[.2]*count])
                zero=row.copy();zero[:,4:]=0.
                expected=case.model.predictions(zero)[0]+row[0,4:]@case.noise
                expected[nP:]+=row[0,4]*((2.3-2.25)/1.75)*73*np.arange(1,10)*.01
                np.testing.assert_allclose(case.residual(row)[0],np.linalg.solve(case.chol,expected-case.target),rtol=2e-12,atol=1e-9)

    def test_unaveraged_covariance_and_audited_noise_cannot_be_bypassed(self):
        for key,bad in (('covariance_multiplier',.01),('covariance_is_of_mean',True),
                        ('n_covariance',499),('noise_basis_status','legacy pre'),('P_B_cross_covariance',False)):
            with TemporaryDirectory(prefix='sto_case_fixture_') as tmp:
                root=Path(tmp);directory,meta=self.make_post(root,12,9);meta[key]=bad
                write(directory/'contract.json',meta)
                with patch.object(post,'OUT',root),self.assertRaises(ValueError):post.factory(directory)

    def make_calibration(self,root):
        rng=np.random.default_rng(202610091310);mu=rng.uniform(-1,1,(12,41));p=rng.uniform(5000,20000,(12,41))
        m=rng.uniform(700,30000,(12,41));weight=rng.uniform(.1,1,(12,41));weight/=weight.sum(axis=1)[:,None]
        u=mu*mu;templates=np.array([np.sum(weight*a,axis=1) for a in (p,p*u,p*u*u,p/m,p*u/m,p/m**2)])
        truth=np.array([1.41,2.18,73.]);z=truth[1]+.7625790037720044*u+100*2*1.686*(truth[1]-truth[0])/m
        target=np.sum(weight*p*z*z,axis=1)+truth[2]
        matrix=rng.normal(size=(12,12));covariance=matrix@matrix.T+np.eye(12)
        chol=np.linalg.cholesky(covariance/(486/499.))
        np.savez_compressed(root/'likelihood.npz',target=target,covariance=covariance,chol=chol,
            templates=templates,bounds=BOUNDS)
        write(root/'contract.json',dict(parameters=list(PARAMETERS),truth_fNL=100,
            likelihood_sha256=calibration.digest(root/'likelihood.npz')))
        return calibration.CalibrationCase(root),truth

    def test_calibration_case_loss_prior_and_coordinates(self):
        with TemporaryDirectory(prefix='sto_p_fixture_') as tmp:
            case,truth=self.make_calibration(Path(tmp))
            case.center=truth;case.transform=np.diag([.1,.2,50.])
            parameters=np.array([truth,[1.6,2.3,-10.],[0.,2.18,73.]])
            direct=PhysicalPCalibrationCase(case.target,case.covariance,case.chol,case.templates)
            np.testing.assert_allclose(case.log_prob(case.coordinates(parameters)),direct.log_probability(parameters),
                rtol=3e-13,atol=1e-8)
            np.testing.assert_allclose(case.physical(case.coordinates(parameters)),parameters,rtol=0,atol=2e-14)

    def test_calibration_blind_MAP_and_four_walker_initializations(self):
        with TemporaryDirectory(prefix='sto_p_fixture_') as tmp:
            case,truth=self.make_calibration(Path(tmp));meta=case.initialize()
            np.testing.assert_allclose(meta['center'],truth,rtol=3e-4,atol=2e-2)
            self.assertTrue(meta['minuit_valid']);self.assertEqual(len(meta['candidates']),9)
            for ensemble in range(4):
                first=case.initial(20261009+ensemble,ensemble,meta)
                second=case.initial(20261009+ensemble,ensemble,meta)
                np.testing.assert_array_equal(first,second)
                self.assertEqual(first.shape,(32,3));self.assertTrue(np.isfinite(case.log_prob(first)).all())
            reused=case.initialize();self.assertEqual(meta,reused)


if __name__=='__main__':unittest.main()
