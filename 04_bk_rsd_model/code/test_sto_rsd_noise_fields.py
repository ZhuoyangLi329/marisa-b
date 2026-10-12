"""模块大纲：速度单位、mixed spectra、自矩、流式paint与直接低mode检查。"""
import unittest
import numpy as np
from sto_rsd_noise_fields import (gadget_rsd_positions,marked_moments,mode_geometry,
    projected_spectra,projection_residual,StreamingMarkedFourier)


class NoiseFieldsTest(unittest.TestCase):
    def test_particle_velocity_not_FoF_convention_and_periodic_wrap(self):
        a=2/3;hh=100*np.sqrt(.3175/a**3+.6825)
        pos=np.array([[1.,2.,999900.],[2000.,3000.,50.]])
        vel=np.array([[0.,0.,100.],[20.,30.,-300.]])
        got=gadget_rsd_positions(pos,vel,a=a,omega_m=.3175,omega_lambda=.6825)
        expected=pos/1000;expected[:,2]=(expected[:,2]+np.sqrt(a)*vel[:,2]/(a*hh))%1000
        np.testing.assert_allclose(got,expected,rtol=0,atol=2e-13)
        np.testing.assert_array_equal(got[:,:2],pos[:,:2]/1000)
        with self.assertRaises(ValueError):gadget_rsd_positions(pos,vel[:1],a=a,omega_m=.3,omega_lambda=.7)

    def test_true_weight_self_moments_not_nraw(self):
        w=np.array([.5,1.,2.,3.]);n2,n3=marked_moments(w,volume=7.)
        self.assertAlmostEqual(n2,7*sum(x*x for x in w)/sum(w)**2)
        self.assertAlmostEqual(n3,49*sum(x**3 for x in w)/sum(w)**3)
        self.assertNotAlmostEqual(n3,n2*n2)

    def test_mixed_spectra_against_independent_cell_loops(self):
        g=mode_geometry(32,1000.,[.018,.04,.065,.09],[0.,.5,1.])
        rng=np.random.default_rng(84204);f=rng.normal(size=(3,len(g['labels'])))+1j*rng.normal(size=(3,len(g['labels'])))
        got=projected_spectra(f,g,volume=3.)
        expected=np.empty_like(got)
        for i in range(3):
            for j in range(3):
                for k in range(g['nk']):
                    for mu in range(g['nmu']):
                        m=g['labels']==k*g['nmu']+mu
                        expected[i,j,k,mu]=3*np.mean(f[i,m]*f[j,m].conj())
        np.testing.assert_allclose(got,expected,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(got,got.transpose(1,0,2,3).conj(),atol=2e-14)
        noise=projection_residual(got,1,0)
        self.assertTrue(np.all(noise>=0))
        self.assertEqual(sum(g['counts']),len(g['indices']))

    def test_native_streaming_TSC_vs_whole_and_low_mode_DFT(self):
        # 原生pmesh实际调用，不用mock验证自身；local缺MPI时仅NERSC执行。
        from mpi4py import MPI
        rng=np.random.default_rng(84205);x=rng.uniform(0,1000.,(311,3));w=rng.uniform(.2,2.,len(x))
        whole=StreamingMarkedFourier(64,comm=MPI.COMM_WORLD);whole.add(x,w);a=whole.fourier()
        stream=StreamingMarkedFourier(64,comm=MPI.COMM_WORLD)
        for start in range(0,len(x),37):stream.add(x[start:start+37],w[start:start+37])
        b=stream.fourier();np.testing.assert_allclose(a,b,rtol=0,atol=3e-15)
        modes=np.array([[1,0,0],[1,2,-1],[-2,1,1],[0,0,-1]])
        direct=np.exp(-2j*np.pi*x@modes.T/1000.).T@w/w.sum()
        got=np.array([a[tuple(k%64)] for k in modes])
        # TSC/interlacing3仍有有限assignment alias；预定absolute门而非伪称精确DFT。
        np.testing.assert_allclose(got,direct,rtol=0,atol=2e-6)
        for k in modes:np.testing.assert_allclose(a[tuple(k%64)],a[tuple(-k%64)].conj(),rtol=0,atol=3e-15)


if __name__=='__main__':unittest.main()
