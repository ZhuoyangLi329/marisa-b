"""模块大纲：复现 jaxpower 的 f4 子域舍入；既有 real-space 1e-3 guard 回归。

这里只检验边界算术；生产代码直接调用共享 guard，不复制其实现。
"""
import unittest
import numpy as np

class WrapTest(unittest.TestCase):
    def test_upper_boundary_rounding(self):
        source=np.nextafter(np.float32(1000),np.float32(0))
        self.assertLess(source,1000)
        self.assertEqual(np.floor((source+np.float32(500)-np.float32(500))/np.float32(1000)),1)
        safe=np.float32(1000.-1e-3)
        self.assertEqual(np.floor((safe+np.float32(500)-np.float32(500))/np.float32(1000)),0)

if __name__=='__main__':unittest.main()
