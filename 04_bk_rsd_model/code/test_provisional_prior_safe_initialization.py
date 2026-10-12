"""模块大纲：边界MAP/退化affine条件独立测试；不触碰数据C/后验。"""
import unittest
import numpy as np
from provisional_prior_safe_initialization import bounded_affine_transform,physical_initial_cloud


class PriorSafeTest(unittest.TestCase):
    def test_corner_weak_direction_and_reversible_coordinates(self):
        bounds=np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-10,10],[-10,10],[-20,20],
            [0,300],[-10,10],[-50,50],[-20,20],[-300,300]],dtype=float)
        width=bounds[:,1]-bounds[:,0];center=(bounds[:,0]+bounds[:,1])/2
        center[[4,6,8]]=bounds[[4,6,8],1];center[7]=bounds[7,0]
        base=np.diag(width*np.logspace(-12,8,12))
        transform,meta=bounded_affine_transform(base,bounds)
        self.assertGreater(len(meta['changed_rows']),0)
        for seed in (20261001,20262014,20263027,20264040):
            p=physical_initial_cloud(center,transform,bounds,seed)
            self.assertEqual(p.shape,(32,12));self.assertTrue(np.all((p>bounds[:,0])&(p<bounds[:,1])))
            z=np.linalg.solve(transform,(p-center).T).T
            np.testing.assert_allclose(z@transform.T+center,p,rtol=1e-13,atol=1e-12)
            self.assertEqual(np.linalg.matrix_rank(p-p.mean(axis=0)),12)

    def test_seed_reproducible_and_bounds_not_mutated(self):
        bounds=np.array([[-500,500],[.5,4],[0,300]],dtype=float);snapshot=bounds.copy()
        center=np.array([0.,1.93,1.]);transform=np.diag([50.,.175,15.])
        a=physical_initial_cloud(center,transform,bounds,31);b=physical_initial_cloud(center,transform,bounds,31)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(bounds,snapshot)
        with self.assertRaises(ValueError):physical_initial_cloud(center+1000,transform,bounds,31)


if __name__=='__main__':unittest.main()
