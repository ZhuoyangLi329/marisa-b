"""模块大纲：速度单位；wrap/transverse 不变；CIC Fourier 归一化和相位。"""
import unittest
import numpy as np
from export_pre_rsd import redshift_positions
from periodic_rsd_measure import catalog_fourier

class PeriodicRSDTest(unittest.TestCase):
    def test_velocity_units_and_wrap(self):
        p=np.array([[1.,2.,999.],[3.,4.,1.]])
        v=np.array([[0.,0.,100.],[0.,0.,-100.]])
        s,physical,ds,h=redshift_positions(p,v,.5,.3175,1000.)
        np.testing.assert_array_equal(s[:,:2],p[:,:2])
        np.testing.assert_array_equal(physical[:,2],[150.,-150.])
        np.testing.assert_allclose(ds,[225./h,-225./h])
        np.testing.assert_allclose(s[:,2],(p[:,2]+ds)%1000.)
        zero=redshift_positions(p,v*0,.5,.3175,1000.)[0]
        np.testing.assert_array_equal(zero,p)

    def test_uniform_grid_is_zero(self):
        x=np.indices((8,8,8)).reshape(3,-1).T*1000./8
        np.testing.assert_allclose(catalog_fourier(x,8),0.,atol=1e-14)

    def test_translation_phase_and_hermiticity(self):
        n=32;p=np.random.default_rng(1).uniform(0,1000.,size=(800,3))
        original=catalog_fourier(p,n);shifted=catalog_fourier((p+[1000./n,0,0])%1000.,n)
        freq=np.fft.fftfreq(n)[:,None,None]
        np.testing.assert_allclose(shifted,original*np.exp(-2j*np.pi*freq),atol=1e-14)
        neg=(-np.arange(n))%n
        np.testing.assert_allclose(original,np.conj(original[np.ix_(neg,neg,neg)]),atol=1e-14)
        # 使用直接 catalog Fourier sum 独立检查低 k painting，而非只比较两个 meshes。
        fine=catalog_fourier(p,2*n)
        coarse_errors=[];fine_errors=[]
        for mode in ([1,0,0],[1,1,0],[0,0,2]):
            mode=np.array(mode);direct=np.exp(-2j*np.pi*(p@mode)/1000.).mean()
            coarse_error=abs(original[tuple(mode)]-direct)
            self.assertLess(coarse_error,5e-5)
            fine_error=abs(fine[tuple(mode)]-direct)
            self.assertLess(fine_error,1e-5)
            coarse_errors.append(coarse_error);fine_errors.append(fine_error)
        # 单个 mode 的 alias cancellation 不保证固定倍率；比较整体误差范数。
        self.assertLess(np.linalg.norm(fine_errors),np.linalg.norm(coarse_errors))

if __name__=='__main__':unittest.main(verbosity=2)
