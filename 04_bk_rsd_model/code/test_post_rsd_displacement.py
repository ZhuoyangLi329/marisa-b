"""模块大纲：有解析极限的残余位移 tensor；固定算法 soft 核；PSD 和 PNG 二阶。

只验收长位移子模块；不声称完整 B IR、shift-noise 或模拟 fNL 恢复通过。
"""
from dataclasses import replace
import unittest
import numpy as np
from scipy.special import spherical_jn
from post_rsd_fixedrec import FixedReconstruction,FixedPostRSD,pbs_bias
from tree_png_rsd_native import RsdConfig
from post_rsd_displacement import GaussianLongDisplacement,residual_coefficients,damping_coefficients


class DisplacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q=np.geomspace(2*np.pi/1000,.15,161)
        cls.p=6200*np.exp(-cls.q/0.13)/(1+(cls.q/.08)**2)
        cls.m=8e6*cls.q**2/(1+7*cls.q)
        cls.rule=GaussianLongDisplacement(cls.q,cls.p,cls.m,32,64)
        cls.kw=dict(b1=2.2,bphi=3.4,f_growth=.7625790037720044,los=(0,0,1))
        cls.c=FixedReconstruction(10.,2.2280546353635335,1.2822,100.,1000/256,4)

    def test_pre_rsd_analytic_tensor(self):
        n=np.asarray(self.kw['los']);R=np.eye(3)+self.kw['f_growth']*np.outer(n,n)
        directions=np.array([[0,0,1],[1,0,0],[.3,.4,np.sqrt(.75)]])
        r=110*directions;a=self.rule.relative_covariance(r,**self.kw)
        j0=spherical_jn(0,110*self.q);j2=spherical_jn(2,110*self.q)
        for i,u in enumerate(directions):
            ru=R@u
            integrand=self.p[:,None,None]*((1-j0-j2)[:,None,None]*(R@R.T)/3+j2[:,None,None]*np.outer(ru,ru))
            expected=np.trapz(integrand,self.q,axis=0)/np.pi**2
            np.testing.assert_allclose(a[i,0],expected,rtol=2e-12,atol=2e-12)
        np.testing.assert_array_equal(a[:,1:],0.)

    def test_zero_separation_and_exact_matched_realspace(self):
        a=self.rule.relative_covariance(np.zeros(3),reconstruction=self.c,**self.kw)
        np.testing.assert_array_equal(a,0.)
        c=FixedReconstruction(0.,self.kw['b1'],1.2,0.)
        d0,_=residual_coefficients(np.array([[.03,.02,.01]]),np.array([1300.]),
                reconstruction=c,**dict(self.kw,f_growth=0.))
        np.testing.assert_allclose(d0,0.,atol=1e-14)

    def test_fixed_PNG_polynomial_and_positive_semidefinite(self):
        r=np.array([[30.,-40.,100.],[100.,40.,-10.]])
        a=self.rule.relative_covariance(r,reconstruction=self.c,**self.kw)
        for fnl in (-500.,-100.,0.,100.,500.):
            combined=a[:,0]+fnl*a[:,1]+fnl**2*a[:,2]
            self.assertGreater(float(np.linalg.eigvalsh(combined).min()),-1e-11)
            # 把指定 fNL 吸收到线性 Gaussian Z1 的 b1(q,mu) 不合法；独立比较矢量层即可。
            v=np.array([[.007,.013,.023],[.021,-.045,.035]])
            m=np.array([740.,1900.]);d0,d1=residual_coefficients(v,m,reconstruction=self.c,**self.kw)
            n=np.asarray(self.kw['los']);q2=np.sum(v*v,axis=1)
            z=self.kw['b1']+self.kw['f_growth']*(v@n)**2/q2+fnl*self.kw['bphi']/m
            expected=(v+self.kw['f_growth']*(v@n)[:,None]*n-self.c.response(v,m)[:,None]*z[:,None]*v)/q2[:,None]
            np.testing.assert_allclose(d0+fnl*d1,expected,rtol=2e-14,atol=2e-14)

    def test_cic_permutation_and_los_reversal(self):
        r=np.array([[17.,58.,91.]])
        # 各向异性 Cartesian CIC 仅做网格对称变换，不要求任意旋转不变。
        axes=[1,0,2];a=self.rule.relative_covariance(r,reconstruction=self.c,**self.kw)
        b=self.rule.relative_covariance(r[:,axes],reconstruction=self.c,
                    **dict(self.kw,los=tuple(np.asarray(self.kw['los'])[axes])))
        np.testing.assert_allclose(b,a[:,:,axes][:,:,:,axes],rtol=2e-12,atol=2e-12)
        rev=self.rule.relative_covariance(r,reconstruction=self.c,**dict(self.kw,los=(0,0,-1)))
        np.testing.assert_array_equal(a,rev)

    def test_damping_taylor_and_explicit_noise(self):
        k=np.array([[.05,.04,.02]]);r=110*k/np.linalg.norm(k,axis=1)[:,None]
        a=self.rule.relative_covariance(r,reconstruction=self.c,**self.kw)
        coeff,e=damping_coefficients(k,a)
        eps=.02
        direct=lambda x:np.exp(-(e[:,0]+x*e[:,1]+x*x*e[:,2]))
        np.testing.assert_allclose((direct(eps)-direct(-eps))/(2*eps),coeff[:,0]*coeff[:,1],rtol=2e-8,atol=1e-13)
        np.testing.assert_allclose((direct(eps)+direct(-eps)-2*direct(0))/(2*eps**2),coeff[:,0]*coeff[:,2],rtol=3e-4,atol=5e-13)
        noisy=self.rule.relative_covariance(r,reconstruction=self.c,shift_noise=np.full(len(self.q),3300.),**self.kw)
        np.testing.assert_array_equal(noisy[:,1:],a[:,1:])
        self.assertGreater(np.linalg.eigvalsh(noisy[:,0]-a[:,0]).min(),0.)
        with self.assertRaises(ValueError):self.rule.relative_covariance(r,shift_noise=np.ones(len(self.q)),**self.kw)

    def test_gaussian_soft_kernel_residual_displacement(self):
        engine=FixedPostRSD();b=pbs_bias(2.2,.4,-.2,p_halo=1.28);rsd=RsdConfig(self.kw['f_growth'],(.2,-.3,.9))
        n=np.asarray(rsd.los);n=n/np.linalg.norm(n)
        hard=np.array([.065,.021,.055]);direction=np.array([.3,-.7,.5]);direction/=np.linalg.norm(direction)
        amplitude=3e-6;q=amplitude*direction
        for c in (None,replace(self.c,fnl_recon=0.),self.c):
            values=[]
            for soft in (q,-q):
                v=np.array([soft,hard,-soft-hard]);kk=np.linalg.norm(v,axis=1)
                m=8e6*kk**2/(1+7*kk)
                values.append(engine.evaluate(v,np.ones(3),m,b,rsd,c).z2[0,0,0])
            msoft=8e6*amplitude**2/(1+7*amplitude)
            d0,_=residual_coefficients(q[None],np.array([msoft]),reconstruction=c,
                                      b1=b.b1,bphi=b.bphi,f_growth=rsd.f_growth,los=n)
            zhard=b.b1+rsd.f_growth*(hard@n)**2/(hard@hard)
            expected=.5*np.dot(hard,d0[0])*zhard
            actual=(values[0]-values[1])/2
            np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-8)


if __name__=='__main__':unittest.main()
