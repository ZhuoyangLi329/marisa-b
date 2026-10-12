"""模块大纲：独立RSD mapping contraction、任意mark Poisson、PNG响应和零腿测试。

无实际STO认证；原8参数实现不导入、不修改、不设置noise PASS。
"""
from dataclasses import replace
import unittest
import numpy as np
from post_rsd_stochastic_eft_v1 import (MarkedNoiseEFT,intrinsic_factorial,matched_zero_atom,
    EFTPrimitiveFields,project_intrinsic_primitives,intrinsic_from_primitives,zero_from_primitives)
from rsd_b000_fft_projection import Fields


class PointPrimitives(EFTPrimitiveFields,Fields):
    pass


class EFTTest(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[[.03,.01,.02],[-.01,.04,-.01],[-.02,-.05,-.01]],
                         [[.01,-.02,.04],[.04,.01,-.03],[-.05,.01,-.01]]])
        self.p=np.array([[7234.,9113.,5611.],[11451.,8431.,7119.]])
        self.m=np.array([[333.,511.,714.],[312.,501.,623.]])
        self.noise=MarkedNoiseEFT(2413.,3321.,1.29e7,1173.,-913.,2.1e6,2.81,4.21,1.14e7,107.)
        self.b1=2.23;self.bp=3.11;self.f=.7625790037720044

    def test_independent_six_RSD_mapping_placements_and_self(self):
        # 每条signal外腿，两个epsilon nonlinear placements；closure决定速度符号。
        # 此显式contraction不调用候选函数的bracket、mu2 helper或primitive。
        for fnl in (-500.,-100.,0.,73.,500.):
            cross_delta=.5*(self.noise.A_delta+self.noise.n2_self*self.noise.b1_w2)
            cross_phi=.5*(self.noise.A_phi+self.noise.n2_self*self.noise.bphi_w2)
            raw=np.full(2,self.noise.S_factorial+3*self.noise.J21-2*self.noise.n3_self)
            subtract=np.zeros(2)
            for i in range(3):
                k=self.v[:,i];mu=k[:,2]**2/np.sum(k*k,axis=1)
                z=self.b1+self.f*mu+fnl*self.bp/self.m[:,i]
                other=[j for j in range(3) if j!=i]
                for j in other:
                    mapping=-self.f*self.v[:,j,2]*k[:,2]/np.sum(k*k,axis=1)
                    raw += self.p[:,i]*z*(cross_delta+fnl*cross_phi/self.m[:,i]+mapping*self.noise.raw_power)
                z2=self.noise.b1_w2+self.f*mu+fnl*self.noise.bphi_w2/self.m[:,i]
                subtract += self.noise.n2_self*self.p[:,i]*z*z2+self.noise.J21
            expected=raw-subtract+2*self.noise.n3_self
            actual=intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=self.noise)
            np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-7)

    def test_arbitrary_mark_bias_Poisson_factorial_cancels(self):
        # 条件Poisson的标记response可不同于w场；不假定b_w2=b_w。
        poisson=replace(self.noise,raw_power=self.noise.n2_self,A_delta=0.,A_phi=0.,
                        S_factorial=0.,J21=self.noise.n3_self,b1_w2=3.17,bphi_w2=-2.43)
        for fnl in (-500.,0.,500.):
            np.testing.assert_array_equal(intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=poisson),np.zeros(2))

    def test_Gaussian_DESI_and_realspace_limit(self):
        mu=self.v[:,:,2]**2/np.sum(self.v*self.v,axis=2);z=self.b1+self.f*mu
        expected=np.sum(self.p*z*(self.noise.A_delta+self.f*mu*self.noise.P_residual),axis=1)+self.noise.S_factorial
        np.testing.assert_allclose(intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=0,noise=self.noise),expected,rtol=3e-15)
        expected=self.b1*self.noise.A_delta*self.p.sum(axis=1)+self.noise.S_factorial
        np.testing.assert_allclose(intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
            f_growth=0,fnl=0,noise=self.noise),expected,rtol=3e-15)

    def test_independent_PNG_response_survives_zero_bphi(self):
        base=replace(self.noise,A_delta=0.,S_factorial=0.,raw_power=self.noise.n2_self)
        value=intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=0,
            f_growth=self.f,fnl=100,noise=base)
        self.assertGreater(np.max(abs(value)),0.)
        plus=intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=83,noise=base)
        minus=intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=-83,noise=base)
        expected=2*83**2*base.A_phi*self.bp*np.sum(self.p/self.m**2,axis=1)
        np.testing.assert_allclose(plus+minus,expected,rtol=5e-15)

    def test_primitive_representation_versus_direct(self):
        field=PointPrimitives(self.v,self.p,self.m,self.b1,self.f)
        primitives=project_intrinsic_primitives(field)
        for fnl in (-500.,-17.,0.,103.,500.):
            np.testing.assert_allclose(intrinsic_from_primitives(primitives,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=self.noise),
                intrinsic_factorial(self.v,self.p,self.m,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=self.noise),rtol=2e-14)

    def test_zero_atom_independent_mean_covariance_mapping(self):
        # 从F2的均匀背景-(N2+Nr)R*D1和独立random cross=-Nr²重建。
        v=self.v[:,0];p=self.p[:,0];m=self.m[:,0];r=np.array([.37,.23]);fnl=97.
        mu=v[:,2]**2/np.sum(v*v,axis=1);z=self.b1+self.f*mu+fnl*self.bp/m
        zw2=self.noise.b1_w2+self.f*mu+fnl*self.noise.bphi_w2/m
        c21=self.noise.n2_self*zw2*z*p+self.noise.J21
        c21-=(self.noise.n2_self+self.noise.random_power)*r*(p*z*z+self.noise.raw_power)
        c21-=self.noise.random_power**2
        f3=self.noise.n3_self-self.noise.random_power**2
        expected=-2*c21+2*f3
        np.testing.assert_allclose(matched_zero_atom(v,p,m,r,b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=fnl,noise=self.noise),expected,rtol=3e-15)
        unshifted=matched_zero_atom(v,p,m,np.zeros(2),b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=fnl,noise=self.noise)
        # uniform independent random在eta=0完全消失，但shift背景Nr*R项没有理由丢掉。
        no_random=replace(self.noise,random_power=0.)
        np.testing.assert_allclose(unshifted,matched_zero_atom(v,p,m,np.zeros(2),b1=self.b1,bphi=self.bp,
            f_growth=self.f,fnl=fnl,noise=no_random),rtol=3e-15)
        delta=matched_zero_atom(v,p,m,r,b1=self.b1,bphi=self.bp,f_growth=self.f,fnl=fnl,noise=self.noise)-matched_zero_atom(
            v,p,m,r,b1=self.b1,bphi=self.bp,f_growth=self.f,fnl=fnl,noise=no_random)
        np.testing.assert_allclose(delta,2*self.noise.random_power*r*(p*z*z+self.noise.raw_power),rtol=2e-14)

    def test_invalid_physical_inputs_rejected(self):
        with self.assertRaises(ValueError):replace(self.noise,raw_power=-1.)
        with self.assertRaises(ValueError):replace(self.noise,A_phi=np.nan)
        with self.assertRaises(ValueError):intrinsic_factorial(self.v+1,self.p,self.m,
            b1=self.b1,bphi=self.bp,f_growth=self.f,fnl=0,noise=self.noise)
        with self.assertRaises(ValueError):matched_zero_atom(np.zeros((2,3)),self.p[:,0],self.m[:,0],np.ones(2),
            b1=self.b1,bphi=self.bp,f_growth=self.f,fnl=0,noise=self.noise)

    def test_zero_primitive_representation_and_quadratic_fNL(self):
        v=self.v[:,0];p=self.p[:,0];m=self.m[:,0];r=np.array([.37,.23])
        mu=v[:,2]**2/np.sum(v*v,axis=1)
        plain=np.array([p,p*mu,p*mu*mu,p/m,p*mu/m,p/m**2])
        primitive=np.concatenate((plain,plain*r,r[None],np.ones((1,2))))
        for fnl in (-500.,0.,97.,500.):
            np.testing.assert_allclose(zero_from_primitives(primitive,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=self.noise),matched_zero_atom(v,p,m,r,b1=self.b1,bphi=self.bp,
                f_growth=self.f,fnl=fnl,noise=self.noise),rtol=4e-15)


if __name__=='__main__':unittest.main()
