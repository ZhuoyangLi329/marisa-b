"""模块大纲：有向 PN2 候选独立点算、闭合枚举、旧确定性与输入保护。"""
import unittest
from dataclasses import replace
import numpy as np
from post_rsd_fixedrec import FixedReconstruction,pbs_bias
from post_rsd_fft_projection import PostFields,integrate_post_design
from post_rsd_poisson_projection import PoissonShiftFields,PoissonShiftFFTProjection,integrate_poisson_shift_design
from rsd_b000_contract import load_spectrum,F_GROWTH


def independent_oriented(v,p,m,b1,fg,ph,rec,n2,fnl):
    """直接物理式；不读可分离 Field、FFT 或 polynomial degrees。"""
    k2=np.sum(v*v,axis=-1)
    inv2=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    z1=b1+fg*v[...,2]**2*inv2+fnl*pbs_bias(b1,p_halo=ph).bphi/m
    p11=p*z1*z1;result=np.zeros(len(v))
    if rec is None:return result
    r=np.zeros_like(p)
    active=k2>0;r[active]=rec.response(v[active],m[active])
    for i in range(3):
        for j in range(3):
            if i==j:continue
            result-=n2*p11[:,i]*np.sum((v[:,i]+v[:,j])*v[:,j],axis=-1)*inv2[:,j]*r[:,j]
    return np.where(np.all(k2>0,axis=-1),result,0.)


class PoissonProjectionTest(unittest.TestCase):
    def setUp(self):
        k=np.geomspace(1e-4,3.,1001);alpha=2e-4/(1+(k/.05)**2)
        self.rec=FixedReconstruction(10.,2.2280546354,1.2822,100.,1000/256,4,alpha_table=(k,alpha))

    def test_point_polynomial_and_frozen_rec(self):
        rng=np.random.default_rng(2026100903)
        v=rng.normal(size=(83,3,3))*.04;v[:,2]=-v[:,:2].sum(axis=1)
        p,m=load_spectrum()(np.linalg.norm(v,axis=-1));n2=3336.75
        for rec in (None,self.rec,replace(self.rec,fnl_recon=0.)):
            for fg in (0.,F_GROWTH):
                for ph in (1.179,1.31,1.4):
                    fields=PoissonShiftFields(v,p,m,2.11,fg,p_halo=ph,reconstruction=rec)
                    coeff=integrate_poisson_shift_design(fields,n2)
                    for fnl in (-171.,0.,100.,243.):
                        actual=np.array([1.,fnl,fnl*fnl])@coeff
                        expected=independent_oriented(v,p,m,2.11,fg,ph,rec,n2,fnl)
                        scale=max(np.max(np.abs(expected)),1.)
                        np.testing.assert_allclose((actual-expected)/scale,0.,atol=8e-15)

    def test_exact_lattice_zero_legs_and_full_denominator(self):
        edges=np.array([0.,.02035714285714286,.04107142857142857]);n=32;kf=2*np.pi/1000
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
        grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        norm=np.linalg.norm(grid,axis=-1)*kf
        shells=[grid[(norm>=edges[0])&(norm<edges[1])],grid[(norm>=edges[1])&(norm<=edges[2])]]
        for pair in ((0,0),(0,1),(1,1)):
            first,second=(shells[i] for i in pair)
            aa=np.repeat(first,len(second),axis=0);bb=np.tile(second,(len(first),1));cc=-aa-bb
            v=np.stack((aa,bb,cc),axis=1)*kf;active=np.linalg.norm(v,axis=-1)>0
            p=np.zeros(active.shape);m=np.ones(active.shape)
            p[active],m[active]=load_spectrum()(np.linalg.norm(v[active],axis=-1))
            f=PoissonShiftFFTProjection(n,edges,pair,2.11,F_GROWTH,p_halo=1.31,reconstruction=self.rec)
            coeff=integrate_poisson_shift_design(f,3336.75)
            self.assertEqual(f.denominator,len(first)*len(second))
            for fnl in (-100.,0.,100.):
                expected=independent_oriented(v,p,m,2.11,F_GROWTH,1.31,self.rec,3336.75,fnl).mean()
                actual=np.array([1.,fnl,fnl*fnl])@coeff
                np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=1e-6)

    def test_old_deterministic_unchanged_and_guards(self):
        v=np.array([[[.02,.01,0.],[0.,.02,.01],[-.02,-.03,-.01]]])
        p,m=load_spectrum()(np.linalg.norm(v,axis=-1))
        args=(v,p,m,2.11,F_GROWTH);kw=dict(p_halo=1.31,reconstruction=self.rec)
        original=PostFields(*args,**kw);new=PoissonShiftFields(*args,**kw)
        np.testing.assert_array_equal(integrate_post_design(new),integrate_post_design(original))
        np.testing.assert_array_equal(integrate_poisson_shift_design(new,0.),np.zeros((3,1)))
        for bad in (-1.,np.inf,np.nan,np.array([1.])):
            with self.assertRaises(ValueError):integrate_poisson_shift_design(new,bad)
        with self.assertRaises(TypeError):integrate_poisson_shift_design(original,1.)


if __name__=='__main__':unittest.main()
