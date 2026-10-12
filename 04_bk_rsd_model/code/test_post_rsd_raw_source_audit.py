"""模块大纲：仅允许省略可选导出；原数值读取不变且异常时恢复配置。"""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import audit_sto_post_fid500_payloads as original
from audit_sto_post_fid500_raw_sources import omit_optional_export, raw_source_audit_context, OUT


class RawSourceAuditTest(unittest.TestCase):
    def test_exact_periodic_endpoint_without_relaxing_out_of_domain(self):
        positions = np.array([[1000., 0., 17.], [11., 1000., 999.9]])
        before = positions.copy()
        result = original.validate_rsd_before_painting(positions)
        self.assertEqual(result['exact_L_endpoint_components'], 2)
        np.testing.assert_array_equal(positions, before)
        for value in (-np.nextafter(0., 1.), np.nextafter(1000., np.inf), np.nan, np.inf):
            bad = positions.copy(); bad[0, 0] = value
            with self.assertRaises(ValueError): original.validate_rsd_before_painting(bad)
        with self.assertRaises(ValueError): original.validate_rsd_before_painting(np.ones((4, 2)))

    def test_only_redundant_exports_can_be_omitted(self):
        root = original.adapter.original.OLD_POS
        with patch('os.open', side_effect=AssertionError('optional export must not be opened')):
            for name in ('positions_mpc_h.npy', 'masses_msun_h.npy'):
                self.assertIsNone(omit_optional_export(root / 'fiducial/real022' / name, max_bytes=128*1024*1024))
            for path in (root / 'fiducial/real022/post_sto_halo.npz', root / 'LC_p/real022/masses_msun_h.npy',
                         root / 'fiducial/real500/masses_msun_h.npy', Path('/not/the/cache/masses_msun_h.npy')):
                with self.assertRaises(ValueError): omit_optional_export(path, max_bytes=128*1024*1024)

    def test_context_restores_and_keeps_numeric_reader(self):
        before = original.OUT, original.optional_blob, original.save
        reader, identity = original.catalog_from_blobs, original.validate_periodic_identity
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with raw_source_audit_context():
                self.assertEqual(original.OUT, OUT)
                self.assertIs(original.catalog_from_blobs, reader)
                self.assertIs(original.validate_periodic_identity, identity)
                raise RuntimeError('synthetic failure')
        self.assertEqual(before, (original.OUT, original.optional_blob, original.save))


if __name__ == '__main__': unittest.main()
