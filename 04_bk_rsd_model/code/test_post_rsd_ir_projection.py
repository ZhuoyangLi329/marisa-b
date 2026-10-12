"""模块大纲：球谐 tensor 对直接角积分；tagged FFT 对逐 mode native IR oracle。

使用明确的人工 wiggle split 作数值测试，不称为真实谱的 no-wiggle 标定。
"""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_displacement import GaussianLongDisplacement,damping_coefficients
from post_rsd_fixedrec import FixedReconstruction,pbs_bias
from tree_png_rsd_native import RsdConfig
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_oracle import source_leg_wiggles,dress_finite_png
from post_rsd_ir_projection import IRPointFields,IRFFTProjection,integrate_post_design


def toy_spectrum(k):
    pn=5400*np.exp(-k/.3)/(1+(k/.07)**2)
    p=pn*(1+.05*np.sin(110*k));m=8e6*k*k/(1+7*k)
    return p,m


def toy_nowiggle(k):return toy_spectrum(k)[0]/(1+.05*np.sin(110*k))


class IRProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b1=2.13;cls.ph=1.31;cls.f=.7625790037720044;cls.b=pbs_bias(cls.b1,.43,-.17,p_halo=cls.ph)
        ak=np.geomspace(1e-5,3.,2048);av=1/(8e6*ak**2/(1+7*ak))
        cls.rec=FixedReconstruction(10.,2.2280546353635335,1.2822,100.,1000/256,4,alpha_table=(ak,av))
        q=np.geomspace(2*np.pi/1000,.15,129);p,m=toy_spectrum(q);pn=toy_nowiggle(q)
        cls.long=GaussianLongDisplacement(q,pn,m*np.sqrt(pn/p),32,64)
        cls.kw=dict(b1=cls.b1,bphi=cls.b.bphi,f_growth=cls.f,los=(0,0,1),reconstruction=cls.rec)
        cls.harm=HarmonicDisplacement.from_long_modes(cls.long,lmax=10,**cls.kw)

    def test_harmonic_tensor_against_independent_direct_angular_kernel(self):
        rng=np.random.default_rng(2026100903);u=rng.normal(size=(37,3));u/=np.linalg.norm(u,axis=1)[:,None]
        direct=self.long.relative_covariance(110*u,**self.kw)
        for ell in (6,8,10):
            h=HarmonicDisplacement.from_long_modes(self.long,lmax=ell,**self.kw)
            error=np.max(np.abs(h.covariance(u)-direct)/np.maximum(1.,np.max(np.abs(direct),axis=0)))
            if ell==10:self.assertLess(float(error),2e-10)
        damping,_=damping_coefficients(.16*u,direct)
        np.testing.assert_allclose(self.harm.damping(.16*u),damping,rtol=3e-9,atol=2e-12)

    def test_tagged_point_source_matches_native_wiggle_derivative(self):
        rng=np.random.default_rng(2026100904);v=rng.normal(size=(13,3,3))*.045;v[:,2]=-v[:,:2].sum(axis=1)
        k=np.linalg.norm(v,axis=-1);p,m=toy_spectrum(k);pn=toy_nowiggle(k);d=self.harm.damping(v)
        fields=IRPointFields(v,p,pn,m,self.b1,self.f,p_halo=self.ph,reconstruction=self.rec,damping=d)
        design=integrate_post_design(fields);actual=(design[0]+self.b.b2*design[1]+self.b.bK2*design[2]).T
        source=source_leg_wiggles(v,p,pn,m,self.b,RsdConfig(self.f),self.rec)
        expected=dress_finite_png(source.base.total,source.wiggles['total'],d)
        scale=np.maximum(1.,np.max(np.abs(expected),axis=0))
        np.testing.assert_allclose((actual-expected)/scale,0.,atol=7e-13)

    def test_fft_full_shell_denominator_against_direct_nodes(self):
        n=16;edges=np.array([0.,.01,.018]);kf=2*np.pi/1000.
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int);grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        k=np.linalg.norm(grid,axis=-1)*kf;a=grid[k<edges[1]];b=grid[(k>=edges[1])&(k<edges[2])]
        aa=np.repeat(a,len(b),axis=0);bb=np.tile(b,(len(a),1));cc=-aa-bb
        valid=(np.sum(aa*aa,axis=1)>0)&(np.sum(bb*bb,axis=1)>0)&(np.sum(cc*cc,axis=1)>0)
        vectors=np.stack((aa[valid],bb[valid],cc[valid]),axis=1)*kf;den=len(a)*len(b)
        total=np.zeros(3)
        for start in range(0,len(vectors),64):
            v=vectors[start:start+64];kk=np.linalg.norm(v,axis=-1);p,m=toy_spectrum(kk);pn=toy_nowiggle(kk)
            source=source_leg_wiggles(v,p,pn,m,self.b,RsdConfig(self.f),self.rec)
            total+=dress_finite_png(source.base.total,source.wiggles['total'],self.harm.damping(v)).sum(axis=0)
        with patch('rsd_b000_fft_projection.load_spectrum',return_value=toy_spectrum):
            fields=IRFFTProjection(n,edges,(0,1),self.b1,self.f,p_halo=self.ph,reconstruction=self.rec,
                                    power_nw=toy_nowiggle,harmonics=self.harm)
        design=integrate_post_design(fields);actual=design[0]+self.b.b2*design[1]+self.b.bK2*design[2]
        np.testing.assert_allclose(actual,total/den,rtol=5e-12,atol=1e-6)
        self.assertEqual(fields.denominator,den)
        self.assertLess(len(vectors),den)
        with self.assertRaises(NotImplementedError):fields.stochastic()

    def test_damping_context_guards(self):
        with self.assertRaises(ValueError):IRFFTProjection(16,[0,.01,.018],(0,1),self.b1+.1,self.f,
            p_halo=self.ph,reconstruction=self.rec,power_nw=toy_nowiggle,harmonics=self.harm)
        with self.assertRaises(ValueError):IRFFTProjection(16,[0,.01,.018],(0,1),self.b1,self.f,
            p_halo=self.ph,reconstruction=replace(self.rec,fnl_recon=0.),power_nw=toy_nowiggle,harmonics=self.harm)


if __name__=='__main__':unittest.main()
