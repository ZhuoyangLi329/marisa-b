"""模块大纲：联合适配回归；P finite-PNG/离散方向；原 B 数值方法不变。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rsd_pb_common import OUT,B_OUT,P_CAL,F_GROWTH,p_lattice,p_coefficients,p_prediction,p_geometry,digest
from run_rsd_pb_inference import predicted
import run_rsd_pb_mcmc as adapter
import run_rsd_b000_mcmc as sampler

class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.edges=p_geometry();cls.t,cls.n,cls.k=p_lattice(cls.edges);cls.c=p_coefficients(cls.t)

    def test_packed_polynomial(self):
        rng=np.random.default_rng(29)
        for f,b in zip(rng.uniform(-500,500,30),rng.uniform(.5,4,30)):
            v=np.array([f,b,-.4,.6,0,0,0,0])
            got=predicted(v,self.c,np.zeros((4,12)))[0]
            np.testing.assert_allclose(got,p_prediction(f,b,self.t),rtol=3e-13,atol=1e-9)

    def test_finite_png_square(self):
        # 从原始每个格点的 Z1² 独立求和，不复用六个压缩模板的预测公式。
        from rsd_b000_contract import load_spectrum
        with np.load(OUT/'P0_WINDOW.npz') as z:
            vectors=np.array(z['vectors_integer'])*(2*np.pi/1000.);labels=np.array(z['labels']);weights=np.array(z['weights'])
        k=np.linalg.norm(vectors,axis=1);mu2=vectors[:,2]**2/k**2
        f=-123.;b=2.03;bp=2*1.686*(b-P_CAL);sp=load_spectrum();direct=[]
        for index in range(12):
            mask=labels==index;p,m=sp(k[mask]);direct.append(np.average((b+F_GROWTH*mu2[mask]+f*bp/m)**2*p,weights=weights[mask]))
        np.testing.assert_allclose(p_prediction(f,b,self.t),direct,rtol=3e-14)

    def test_isotropic_lattice_and_f_zero(self):
        np.testing.assert_allclose(self.t[1,0],self.t[0,0]/3,rtol=3e-14)
        # mu4 不是粗暴设为 1/5；低 k 第一格点 shell 的各向异性确实保留。
        self.assertGreater(abs(self.t[2,0]/self.t[0,0]-.2),1e-3)
        b=1.93;f=81.;bp=2*1.686*(b-P_CAL)
        expected=b*b*self.t[0]+2*f*bp*b*self.t[3]+f*f*bp*bp*self.t[5]
        actual=p_prediction(f,b,self.t)-2*b*F_GROWTH*self.t[1]-F_GROWTH**2*self.t[2]-2*f*bp*F_GROWTH*self.t[4]
        np.testing.assert_allclose(actual,expected,rtol=3e-14)

    def test_unique_sampler_methods(self):
        for name in ('residual','objective','initialize','physical','coordinates','log_prob','initial'):
            self.assertIs(getattr(adapter.JointCase,name),getattr(sampler.Case,name))
        self.assertEqual(digest(Path(sampler.__file__)),'018d3d4bb2db0dfdd739b7933472716bdb1c866d2f7d9aa57dca3f463456614d')

    def test_full_covariance_not_mean_or_block_diagonal(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(500,5));x[:,3]+=x[:,0]
        c=np.cov(x,rowvar=False,ddof=1);alpha=(500-5-2)/499;L=np.linalg.cholesky(c/alpha)
        np.testing.assert_allclose(L@L.T,c/alpha,rtol=1e-14)
        self.assertGreater(c[0,3],.5);self.assertGreater(c[0,0],.5)

    @unittest.skipUnless((OUT/'CASES.json').exists(),'run again after aggregation')
    def test_frozen_joint_cases(self):
        for name in json.loads((OUT/'CASES.json').read_text()):
            case=adapter.JointCase(OUT/'cases'/name);old=sampler.Case(B_OUT/'cases'/name)
            self.assertEqual(case.ndim,old.ndim+1);self.assertEqual(len(case.target),len(old.target)+12)
            np.testing.assert_array_equal(case.target[12:],old.target)
            np.testing.assert_array_equal(case.coefficients[...,12:],old.coefficients)
            np.testing.assert_array_equal(case.noise[:-1,12:],old.noise)
            np.testing.assert_array_equal(case.bounds[:-1],old.bounds)
            np.testing.assert_allclose(case.covariance[12:,12:],old.covariance,rtol=2e-13)
            np.testing.assert_allclose(case.bounds[-1]*case.meta['Pshot_unit_Mpc_h3'],[-1e6,1e6])
            values=np.mean(case.bounds,axis=1);r=case.residual(values)[0]
            unwhite=predicted(values,case.coefficients,case.noise)[0]-case.target
            np.testing.assert_allclose(r,np.linalg.solve(case.chol,unwhite),rtol=2e-11,atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
