"""模块大纲：旧 ABI 回归；完整多项式 oracle；坐标映射独立验算；极限与拒绝门。

这些是确定性 source 测试，不代表离散噪声、catalog 或模拟恢复通过。
"""
import itertools
import unittest
from dataclasses import replace
import numpy as np
from numpy.polynomial.polynomial import polymul
from tree_png_rsd_native import NativeTreeRSD,HaloBiasV1,RsdConfig,ReconstructionConfig,PART
from post_rsd_fixedrec import FixedPostRSD,FixedReconstruction,pbs_bias


class FixedPostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new=FixedPostRSD();cls.old=NativeTreeRSD()
        rng=np.random.default_rng(2026100801)
        cls.v=rng.normal(size=(97,3,3))*.055;cls.v[:,2]=-cls.v[:,:2].sum(axis=1)
        cls.p=rng.uniform(800.,10000.,size=(97,3));cls.m=rng.uniform(900.,17000.,size=(97,3))
        cls.b=HaloBiasV1(2.21,.47,-.28,3.4,1.71,2.63)
        cls.r=RsdConfig(.7625790037720044,(.2,-.3,.9))
        cls.c=FixedReconstruction(10.,2.2280546353635335,1.2822,100.,8.,4)

    def assertScaled(self,actual,expected,tol=3e-12):
        actual=np.asarray(actual);expected=np.asarray(expected)
        scale=np.maximum(1.,np.max(np.abs(expected),axis=0))
        self.assertLess(float(np.max(np.abs(actual-expected)/scale)),tol)

    def test_legacy_abi_is_unchanged(self):
        current=NativeTreeRSD(self.new.path)
        for rec in (None,ReconstructionConfig(10.,2.228,8.,4),
                    ReconstructionConfig(10.,2.228,8.,4,f_nl_recon_ratio=1.,bphi_recon=3.18)):
            expected=self.old.evaluate(self.v,self.p,self.m,self.b,self.r,rec).raw
            actual=current.evaluate(self.v,self.p,self.m,self.b,self.r,rec).raw
            np.testing.assert_array_equal(actual,expected)

    def test_full_polynomial_oracle(self):
        """独立 numpy 完整四次乘积，只在最后截断；不复用 FnlPoly2 或 response。"""
        c=self.c
        pre=self.old.evaluate(self.v,self.p,self.m,self.b,self.r)
        correction=[]
        for v,p,m,z1 in zip(self.v,self.p,self.m,pre.z1):
            k2=np.sum(v*v,axis=1)
            filt=np.exp(-.5*k2*c.smoothing_radius**2)
            filt*=np.prod(np.sinc(v*c.cell_size/(2*np.pi)),axis=1)**c.cic_window_power
            filt/=c.bias_recon+c.fnl_recon*(2*1.686*(c.bias_recon-c.p_recon))/m
            value=np.zeros(3)
            for i,j in ((0,1),(1,2),(2,0)):
                total=v[i]+v[j]
                scalar=-sum(np.dot(total,v[a])/k2[a]*filt[a] for a in (i,j))
                full=polymul(polymul(z1[i,:2],z1[i,:2]),polymul(z1[j,:2],z1[j,:2]))
                value+=p[i]*p[j]*scalar*full[:3]
            correction.append(value)
        post=self.new.evaluate(self.v,self.p,self.m,self.b,self.r,c)
        self.assertScaled(post.total,pre.total+correction)
        np.testing.assert_array_equal(post.z1,pre.z1)
        np.testing.assert_array_equal(post.primordial,pre.primordial)

    def test_coordinate_map_second_order(self):
        """直接 exp[-ik.(x+s)] delta(x) 的偶差分，不调用 kernel 的 shift 表达式。"""
        modes=np.array([[1,2,0],[2,-1,1]],dtype=float);kf=.02
        vectors=np.vstack((modes,-modes.sum(axis=0)))*kf
        m=np.array([2300.,3700.,6100.]);p=np.array([3700.,2300.,1800.])
        r=RsdConfig(.7625790037720044,(0,0,1));c=self.c
        q2=np.sum(vectors[:2]**2,axis=1)
        grid=np.stack(np.meshgrid(*([np.arange(24)*2*np.pi/24]*3),indexing='ij'),axis=-1)
        phases=np.einsum('...a,ia->i...',grid,modes)
        carrier=np.exp(-1j*np.einsum('...a,a->...',grid,modes.sum(axis=0)))
        for fnl in (-100.,0.,100.):
            z1=self.b.b1+r.f_growth*vectors[:2,2]**2/q2+fnl*self.b.bphi/m[:2]
            filt=np.exp(-.5*q2*c.smoothing_radius**2)
            filt*=np.prod(np.sinc(vectors[:2]*c.cell_size/(2*np.pi)),axis=1)**c.cic_window_power
            filt/=c.bias_recon+c.fnl_recon*c.bphi_recon/m[:2]
            delta=np.einsum('i,i...->...',2*z1,np.cos(phases))
            shifts=np.einsum('ia,i...->...a',2*vectors[:2]/q2[:,None]*(filt*z1)[:,None],np.sin(phases))
            kdotshift=np.einsum('...a,a->...',shifts,vectors[:2].sum(axis=0))
            def direct(eps):
                return np.mean(carrier*np.exp(-1j*eps*kdotshift)*(eps*delta))
            def even(eps):return (direct(eps)+direct(-eps))/(2*eps**2)
            measured=(4*even(1e-4)-even(2e-4))/3/2
            pre=self.new.evaluate(vectors,p,m,self.b,r)
            post=self.new.evaluate(vectors,p,m,self.b,r,c)
            expected=(post.z2[0,0]-pre.z2[0,0])@np.array([1.,fnl,fnl**2])
            self.assertAlmostEqual(measured.real,expected,places=9)
            self.assertLess(abs(measured.imag),2e-10)

    def test_fixed_zero_and_no_shift_limits(self):
        zero=replace(self.c,fnl_recon=0.)
        old=ReconstructionConfig(zero.smoothing_radius,zero.bias_recon,zero.cell_size,zero.cic_window_power)
        np.testing.assert_array_equal(self.new.evaluate(self.v,self.p,self.m,self.b,self.r,zero).raw,
                                      self.old.evaluate(self.v,self.p,self.m,self.b,self.r,old).raw)
        off=replace(self.c,smoothing_radius=1e8)
        np.testing.assert_array_equal(self.new.evaluate(self.v,self.p,self.m,self.b,self.r,off).raw,
                                      self.new.evaluate(self.v,self.p,self.m,self.b,self.r).raw)

    def test_symmetries_and_realspace_growth(self):
        for growth in (0.,self.r.f_growth):
            r=replace(self.r,f_growth=growth)
            expected=self.new.evaluate(self.v,self.p,self.m,self.b,r,self.c).total
            for order in itertools.permutations(range(3)):
                ix=list(order)
                self.assertScaled(self.new.evaluate(self.v[:,ix],self.p[:,ix],self.m[:,ix],self.b,r,self.c).total,expected)
            self.assertScaled(self.new.evaluate(-self.v,self.p,self.m,self.b,r,self.c).total,expected)
            self.assertScaled(self.new.evaluate(self.v,self.p,self.m,self.b,replace(r,los=tuple(-np.array(r.los))),self.c).total,expected)
            axes=[2,0,1];rr=replace(r,los=tuple(np.array(r.los)[axes]))
            self.assertScaled(self.new.evaluate(self.v[:,:,axes],self.p,self.m,self.b,rr,self.c).total,expected)

    def test_linked_path_is_not_a_fixed_catalog(self):
        linked=ReconstructionConfig(10.,self.c.bias_recon,8.,4,f_nl_recon_ratio=1.,bphi_recon=self.c.bphi_recon)
        with self.assertRaises(TypeError):self.new.evaluate(self.v,self.p,self.m,self.b,self.r,linked)
        a=self.old.evaluate(self.v,self.p,self.m,self.b,self.r,linked).total
        b=self.new.evaluate(self.v,self.p,self.m,self.b,self.r,self.c).total
        self.assertGreater(float(np.max(np.abs(a[:,0]-b[:,0]))),1.)

    def test_fifth_bias_node_is_required(self):
        def sample(b1):
            return self.new.evaluate(self.v,self.p,self.m,pbs_bias(b1,.4,-.2,p_halo=1.31),self.r,self.c).total
        def interpolated(n,point):
            nodes=2.25+1.75*np.cos((np.arange(n)+.5)*np.pi/n)
            coeff=np.linalg.solve(np.polynomial.chebyshev.chebvander((nodes-2.25)/1.75,n-1),
                                  np.array([sample(b) for b in nodes]).reshape(n,-1))
            return (np.polynomial.chebyshev.chebvander((point-2.25)/1.75,n-1)@coeff).reshape(97,3)
        expected=sample(1.83451)
        self.assertScaled(interpolated(5,1.83451),expected)
        self.assertGreater(float(np.max(np.abs(interpolated(4,1.83451)-expected)/np.maximum(1.,np.abs(expected)))),1e-6)

    def test_denominator_and_settings_rejected(self):
        for kwargs in ({'bias_recon':0.},{'fnl_recon':np.nan},{'cic_window_power':2.5}):
            with self.assertRaises(ValueError):replace(self.c,**kwargs)
        bad=replace(self.c,fnl_recon=-1e9)
        with self.assertRaises(ValueError):bad.response(self.v,self.m)
        with self.assertRaises(ValueError):self.new.evaluate(self.v,self.p,self.m,self.b,self.r,bad)

    def test_separate_reconstruction_alpha_does_not_change_physical_png(self):
        ak=np.geomspace(1e-4,3.,1001);av=2e-4/(1+(ak/.05)**2)
        c=replace(self.c,alpha_table=(ak,av))
        post=self.new.evaluate(self.v,self.p,self.m,self.b,self.r,c)
        pre=self.new.evaluate(self.v,self.p,self.m,self.b,self.r)
        np.testing.assert_array_equal(post.z1,pre.z1)
        np.testing.assert_array_equal(post.primordial,pre.primordial)
        expected=pre.total.copy()
        for row,(v,p,z1) in enumerate(zip(self.v,self.p,pre.z1)):
            q2=np.sum(v*v,axis=1);alpha=np.interp(np.log10(np.sqrt(q2)),np.log10(ak),av)
            response=np.exp(-.5*q2*c.smoothing_radius**2)*np.prod(np.sinc(v*c.cell_size/(2*np.pi)),axis=1)**4
            response/=c.bias_recon+c.fnl_recon*c.bphi_recon*alpha
            for i,j in ((0,1),(1,2),(2,0)):
                factor=-sum(np.dot(v[i]+v[j],v[a])/q2[a]*response[a] for a in (i,j))
                full=polymul(polymul(z1[i,:2],z1[i,:2]),polymul(z1[j,:2],z1[j,:2]))
                expected[row]+=p[i]*p[j]*factor*full[:3]
        self.assertScaled(post.total,expected)
        with self.assertRaises(ValueError):replace(self.c,alpha_table=([.1,.2],[1.,-1.]))
        with self.assertRaises(ValueError):replace(self.c,alpha_table=([.1,.2],[1.,1.])).response(self.v,self.m)


if __name__=='__main__':unittest.main()
