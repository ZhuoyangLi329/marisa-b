"""模块大纲：严格端点等价；拒绝微小数值偏移、重排与超出周期输入域。"""
import unittest
import numpy as np
from post_rsd_catalog_identity import validate_periodic_identity


class CatalogIdentityTest(unittest.TestCase):
    def test_exact_and_endpoint_equivalence(self):
        raw=np.array([[12.,1000.,35.],[0.,19.,45.]],dtype='f4');source=raw.copy();source[0,1]=0.
        info=validate_periodic_identity(source,raw)
        self.assertEqual(info['endpoint_equivalent_components'],1)
        self.assertEqual(info['endpoint_indices_first8'],[[0,1]])
        self.assertEqual(raw[0,1],1000.)
        self.assertTrue(validate_periodic_identity(raw,raw)['array_equal_before_periodic_identification'])

    def test_nonendpoint_ulp_is_not_relaxed(self):
        a=np.array([[12.,17.,35.]],dtype='f4');b=a.copy();b[0,0]=np.nextafter(b[0,0],np.float32(np.inf))
        with self.assertRaises(ValueError):validate_periodic_identity(a,b)

    def test_reordering_and_out_of_domain_rejected(self):
        a=np.array([[1.,2.,3.],[4.,5.,6.]])
        with self.assertRaises(ValueError):validate_periodic_identity(a,a[::-1])
        with self.assertRaises(ValueError):validate_periodic_identity(a,a+1000.)
        with self.assertRaises(ValueError):validate_periodic_identity(a,a[:1])


if __name__=='__main__':unittest.main()
