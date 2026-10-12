"""模块大纲：post 可分离源逐项对 native；独立闭合枚举对 FFT；显式 p 回归。"""
import unittest
from dataclasses import replace
import numpy as np
from post_rsd_fixedrec import FixedPostRSD,FixedReconstruction,pbs_bias
from post_rsd_fft_projection import PostFields,PostFFTProjection,integrate_post_design
from rsd_b000_contract import load_spectrum,F_GROWTH
from tree_png_rsd_native import RsdConfig


class PostProjectionTest(unittest.TestCase):
    def test_point_native_and_pbs_derivatives(self):
        rng=np.random.default_rng(2026100802)
        v=rng.normal(size=(53,3,3))*.055;v[:,2]=-v[:,:2].sum(axis=1)
        p,m=load_spectrum()(np.linalg.norm(v,axis=-1));native=FixedPostRSD()
        rec=FixedReconstruction(10.,2.2280546354,1.2822,100.,8.,4)
        for ph in (1.179,1.2822,1.3395):
            for fg in (0.,F_GROWTH):
                for b1 in (.71,2.03,3.72):
                    fields=PostFields(v,p,m,b1,fg,p_halo=ph,reconstruction=rec)
                    result=integrate_post_design(fields)
                    zero=native.evaluate(v,p,m,pbs_bias(b1,p_halo=ph),RsdConfig(fg),rec).total
                    second=native.evaluate(v,p,m,pbs_bias(b1,1.,p_halo=ph),RsdConfig(fg),rec).total-zero
                    tidal=native.evaluate(v,p,m,pbs_bias(b1,0.,1.,p_halo=ph),RsdConfig(fg),rec).total-zero
                    expected=np.stack((zero,second,tidal)).transpose(0,2,1)
                    scale=np.maximum(1.,np.max(np.abs(expected),axis=-1,keepdims=True))
                    np.testing.assert_allclose((result-expected)/scale,0.,atol=3e-13)

    def test_exact_lattice_fft_matches_native_enumeration(self):
        edges=np.array([0.,.02035714285714286,.04107142857142857]);n=32;kf=2*np.pi/1000.
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
        grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        k=np.linalg.norm(grid,axis=-1)*kf
        first=grid[(k>=edges[0])&(k<edges[1])];second=grid[(k>=edges[1])&(k<=edges[2])]
        aa=np.repeat(first,len(second),axis=0);bb=np.tile(second,(len(first),1));cc=-aa-bb
        valid=(np.sum(aa*aa,axis=1)>0)&(np.sum(bb*bb,axis=1)>0)&(np.sum(cc*cc,axis=1)>0)
        v=np.stack((aa[valid],bb[valid],cc[valid]),axis=1)*kf
        p,m=load_spectrum()(np.linalg.norm(v,axis=-1));native=FixedPostRSD()
        base=FixedReconstruction(10.,2.228,1.2822,100.,8.,4)
        ak=np.geomspace(1e-4,3.,1001);av=2e-4/(1+(ak/.05)**2)
        for rec in (None,base,replace(base,alpha_table=(ak,av))):
            fields=PostFFTProjection(n,edges,(0,1),2.07,F_GROWTH,p_halo=1.31,reconstruction=rec)
            model=integrate_post_design(fields)
            pred=model[0]+.43*model[1]-.17*model[2]
            direct=native.evaluate(v,p,m,pbs_bias(2.07,.43,-.17,p_halo=1.31),RsdConfig(F_GROWTH),rec).total
            expected=direct.sum(axis=0)/(len(first)*len(second))
            np.testing.assert_allclose(pred,expected,rtol=3e-12,atol=1e-6)
            self.assertEqual(fields.denominator,len(first)*len(second))


if __name__=='__main__':unittest.main()
