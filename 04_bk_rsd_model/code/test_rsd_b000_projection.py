"""模块大纲：可分离源与 native 独立逐项检查；FFT 两腿与 brute-force 回归。"""
import unittest
import numpy as np
from rsd_b000_fft_projection import Fields,FFTProjection,integrate_design
from tree_png_rsd_native import NativeTreeRSD,RsdConfig
from rsd_b000_contract import bias,load_spectrum,F_GROWTH

class ProjectionTest(unittest.TestCase):
    def test_factorized_native_random(self):
        rng=np.random.default_rng(88304);v=rng.normal(size=(211,3,3))*.07;v[:,2]=-v[:,:2].sum(axis=1)
        p,m=load_spectrum()(np.linalg.norm(v,axis=-1));native=NativeTreeRSD()
        for fg in (0.,F_GROWTH):
            for b1 in (.61,1.9,3.83):
                design=integrate_design(Fields(v,p,m,b1,fg))
                zero=native.evaluate(v,p,m,bias(b1),RsdConfig(fg)).total
                second=native.evaluate(v,p,m,bias(b1,1),RsdConfig(fg)).total-zero
                tidal=native.evaluate(v,p,m,bias(b1,0,1),RsdConfig(fg)).total-zero
                expected=np.stack((zero,second,tidal)).transpose(0,2,1)
                scale=np.max(np.abs(expected),axis=-1,keepdims=True)
                np.testing.assert_allclose((design-expected)/np.maximum(scale,1.),0.,atol=5e-14)

    def test_fft_matches_direct_modes(self):
        n=32;kf=2*np.pi/1000.;edges=np.array([0.,.02035714285714286,.04107142857142857])
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
        grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        k=np.linalg.norm(grid,axis=-1)*kf
        first=grid[(k>=edges[0])&(k<edges[1])];second=grid[(k>=edges[1])&(k<=edges[2])]
        aa=np.repeat(first,len(second),axis=0);bb=np.tile(second,(len(first),1));cc=-aa-bb
        valid=np.all(np.stack((np.linalg.norm(aa,axis=1),np.linalg.norm(bb,axis=1),np.linalg.norm(cc,axis=1)),axis=1)>0,axis=1)
        vectors=np.stack((aa[valid],bb[valid],cc[valid]),axis=1)*kf
        p,m=load_spectrum()(np.linalg.norm(vectors,axis=-1))
        for fg in (0.,F_GROWTH):
            projected=FFTProjection(n,edges,(0,1),1.91,fg)
            exact=integrate_design(Fields(vectors,p,m,1.91,fg)).sum(axis=-1)/(len(first)*len(second))
            measured=integrate_design(projected)
            np.testing.assert_allclose(measured,exact,rtol=2e-12,atol=1e-7)
            expected=np.mean(np.sum(p,axis=1))*valid.sum()/(len(first)*len(second))
            np.testing.assert_allclose(projected.stochastic()[0],expected,rtol=2e-12)
            np.testing.assert_allclose(projected.stochastic()[1],expected/3,rtol=2e-12)
            self.assertEqual(projected.denominator,len(first)*len(second))

    def test_alias_guard(self):
        with self.assertRaises(ValueError):FFTProjection(32,np.array([0,.04,.07]),(1,1),2.)

if __name__=='__main__':unittest.main()
