"""模块大纲：精确单腿多项式导数、PNG mixing、no-wiggle 与有向对称独立检验。"""
from dataclasses import replace
import unittest
import numpy as np
from numpy.polynomial.polynomial import polymul
from post_rsd_fixedrec import FixedPostRSD,FixedReconstruction,pbs_bias
from tree_png_rsd_native import RsdConfig
from post_rsd_displacement import GaussianLongDisplacement
from post_rsd_ir_oracle import source_leg_wiggles,dress_finite_png,evaluate_ir


class IRSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2026100830)
        cls.v=rng.normal(size=(7,3,3))*.035;cls.v[:,2]=-cls.v[:,:2].sum(axis=1)
        cls.pn=rng.uniform(1000,13000,size=(7,3));cls.m=rng.uniform(1000,22000,size=(7,3))
        cls.p=cls.pn*(1+rng.uniform(-.07,.07,size=(7,3)))
        cls.b=pbs_bias(2.2,.3,-.15,p_halo=1.28);cls.r=RsdConfig(.7625790037720044,(0,0,1))
        ak=np.geomspace(1e-5,3.,2048);av=1/(8e6*ak**2/(1+7*ak))
        cls.c=FixedReconstruction(10.,2.2280546353635335,1.2822,100.,1000/256,4,alpha_table=(ak,av))
        q=np.geomspace(2*np.pi/1000,.15,97);p=6300*np.exp(-q/.13)/(1+(q/.08)**2);m=8e6*q**2/(1+7*q)
        cls.long=GaussianLongDisplacement(q,p,m,24,48)
        cls.engine=FixedPostRSD()

    def test_exact_polynomial_gradient_step_independence(self):
        a=source_leg_wiggles(self.v,self.p,self.pn,self.m,self.b,self.r,self.c,step=.2)
        b=source_leg_wiggles(self.v,self.p,self.pn,self.m,self.b,self.r,self.c,step=.4)
        self.assertTrue(a.physical_potential_unchanged)
        for key in a.wiggles:
            np.testing.assert_allclose(a.wiggles[key],b.wiggles[key],rtol=2e-11,atol=3e-7)
        self.assertLess(max(a.degree_error,b.degree_error),3e-12)

    def test_one_wiggle_has_second_order_remainder(self):
        # 对统一小 wiggle epsilon，full fixed-order - strict linearization 应按 epsilon² 缩放。
        relative=(self.p-self.pn)/self.pn;errors=[]
        # 误差范数的分母必须固定；用每次 full 值会在接近零的 G bin 改变度量。
        fixed_scale=np.maximum(1.,np.abs(self.engine.evaluate(self.v,self.pn,self.m,self.b,self.r,self.c).total))
        for epsilon in (.04,.02):
            p=self.pn*(1+epsilon*relative)
            # 此测试固定 MnW，令 full M 与 full P 同步，不引入 potential 改变。
            m=self.m*np.sqrt(p/self.pn)
            source=source_leg_wiggles(self.v,p,self.pn,m,self.b,self.r,self.c)
            full=self.engine.evaluate(self.v,p,m,self.b,self.r,self.c).total
            linear=source.base.total+source.wiggles['total'].sum(axis=1)
            errors.append(np.linalg.norm((full-linear)/fixed_scale,axis=0))
        ratio=errors[0]/errors[1]
        np.testing.assert_allclose(ratio,4.,rtol=.008)

    def test_gaussian_source_leg_derivative_analytic(self):
        b=replace(self.b,bphi=0.,bphidelta=0.,bphi2=0.)
        source=source_leg_wiggles(self.v,self.p,self.pn,self.m,b,self.r,self.c)
        pairs=source.base.pair_terms[:,:,0]
        for leg in range(3):
            # Gaussian tree 的该腿只在包含它的两个 PiPj pair 中出现。
            expected=(pairs[:,leg]+pairs[:,(leg-1)%3])*(self.p[:,leg]-self.pn[:,leg])/self.pn[:,leg]
            np.testing.assert_allclose(source.wiggles['total'][:,leg,0],expected,rtol=2e-13,atol=3e-7)

    def test_mixing_by_independent_polynomial_convolution(self):
        rng=np.random.default_rng(42);base=rng.normal(size=(4,3));w=rng.normal(size=(4,3,3));d=rng.normal(size=(4,3,3))
        actual=dress_finite_png(base,w,d);expected=base.copy()
        for row in range(4):
            for leg in range(3):expected[row]+=d[row,leg,0]*polymul(w[row,leg],[1,d[row,leg,1],d[row,leg,2]])[:3]
        np.testing.assert_allclose(actual,expected,rtol=5e-15,atol=5e-15)

    def test_no_wiggle_limit_and_oriented_permutation(self):
        v=self.v[:1];p=self.p[:1];pn=self.pn[:1];m=self.m[:1]
        zero=evaluate_ir(v,pn,pn,m,self.b,self.r,self.c,self.long)
        np.testing.assert_array_equal(zero.total,zero.source.base.total)
        one=evaluate_ir(v,p,pn,m,self.b,self.r,self.c,self.long)
        order=[2,0,1]
        two=evaluate_ir(v[:,order],p[:,order],pn[:,order],m[:,order],self.b,self.r,self.c,self.long)
        np.testing.assert_allclose(one.total,two.total,rtol=1e-13,atol=1e-7)
        np.testing.assert_allclose(one.total,one.gravitational_and_bias+one.primordial,rtol=1e-13,atol=1e-7)
        self.assertFalse(one.production_ready)

    def test_reconstruction_alpha_must_be_explicitly_frozen(self):
        with self.assertRaises(ValueError):source_leg_wiggles(self.v,self.p,self.pn,self.m,self.b,self.r,replace(self.c,alpha_table=None))


if __name__=='__main__':unittest.main()
