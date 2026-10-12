"""模块大纲：可选缓存 adapter 不改数值回传；异常也恢复原路径。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import post_rsd_catalog as adapter


class CatalogAdapterTest(unittest.TestCase):
    def test_optional_missing_mass_checks_positions_and_restores(self):
        with TemporaryDirectory() as root:
            directory=Path(root)/'LC_p/real100';directory.mkdir(parents=True)
            x=np.array([[0.,1.,2.],[3.,4.,5.]]);np.save(directory/'positions_mpc_h.npy',x)
            value=(x+1,x,{'raw':'untouched'})
            with patch.object(adapter.original,'OLD_POS',Path(root)),patch.object(adapter.original,'catalog',return_value=value) as call:
                result=adapter.catalog('LC_p',100)
                self.assertIs(result[0],value[0]);self.assertIs(result[1],value[1])
                self.assertEqual(adapter.original.OLD_POS,Path(root));call.assert_called_once_with('LC_p',100)
                self.assertTrue(result[2]['raw_FoF_mass_selection_unchanged'])
                self.assertFalse(result[2]['optional_mass_cache_independently_validated'])
            np.save(directory/'positions_mpc_h.npy',x+1.)
            with patch.object(adapter.original,'OLD_POS',Path(root)),patch.object(adapter.original,'catalog',return_value=value):
                with self.assertRaises(ValueError):adapter.catalog('LC_p',100)
                self.assertEqual(adapter.original.OLD_POS,Path(root))

    def test_complete_cache_uses_original_and_failure_restores(self):
        with TemporaryDirectory() as root:
            directory=Path(root)/'LC_p/real100';directory.mkdir(parents=True)
            np.save(directory/'positions_mpc_h.npy',np.zeros((1,3)))
            with patch.object(adapter.original,'OLD_POS',Path(root)),patch.object(adapter.original,'catalog',side_effect=RuntimeError('raw incomplete')):
                with self.assertRaisesRegex(RuntimeError,'raw incomplete'):adapter.catalog('LC_p',100)
                self.assertEqual(adapter.original.OLD_POS,Path(root))
            np.save(directory/'masses_msun_h.npy',np.array([1e13]))
            original_result=(np.zeros((1,3)),np.zeros((1,3)),{})
            with patch.object(adapter.original,'OLD_POS',Path(root)),patch.object(adapter.original,'catalog',return_value=original_result):
                self.assertIs(adapter.catalog('LC_p',100),original_result)


if __name__=='__main__':unittest.main()
