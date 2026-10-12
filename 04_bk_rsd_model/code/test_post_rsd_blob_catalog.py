"""模块大纲：源端无stat读取；原FoF数值路径；可选质量缓存不伪造；全局路径恢复。"""
import hashlib
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_blob_catalog import read_blob, catalog_from_blobs, adapter
from export_pre_rsd import redshift_positions


def npy_blob(value):
    stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); return stream.getvalue()


class BlobCatalogTest(unittest.TestCase):
    def setUp(self):
        n = 3
        self.mass = np.array([900., 1500., 2000.], dtype='f4')
        self.pos = np.array([[10., 20., 30.], [30000., 40000., 999500.], [90000., 70000., 500.]], dtype='f4')
        self.vel = np.array([[1., 2., 3.], [10., 20., 600.], [-5., 0., -700.]], dtype='f4')
        arrays = [np.full(n, 20, dtype='i4'), np.arange(n, dtype='i4') * 20, self.mass,
                  self.pos, self.vel, np.zeros((n, 6), dtype='f4'), np.zeros((n, 6), dtype='f4')]
        self.raw = struct.pack('<iiiQI', n, n, 60, 60, 1) + b''.join(a.tobytes() for a in arrays)

    def test_source_read_never_requests_stat_or_fstat(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'payload'; path.write_bytes(b'known input bytes')
            with patch('os.stat', side_effect=AssertionError('stat requested')), patch('os.fstat', side_effect=AssertionError('fstat requested')):
                self.assertEqual(read_blob(path), b'known input bytes')
                with self.assertRaises(ValueError): read_blob(path, max_bytes=3)

    def test_original_mass_selection_and_RSD_conversion(self):
        original_paths = (adapter.original.RAW_ROOT, adapter.original.OLD_POS)
        positions = self.pos[1:] / np.float32(1000.)
        masses = self.mass[1:].astype(float) * 1e10
        expected, _, _, _ = redshift_positions(positions, self.vel[1:], .5, .3175, 1000.)
        for with_cache in (False, True):
            rsd, real, meta = catalog_from_blobs('fiducial', 65, self.raw, raw_source='/verified/original_fof',
                position_cache_blob=npy_blob(positions) if with_cache else None,
                mass_cache_blob=npy_blob(masses) if with_cache else None)
            np.testing.assert_array_equal(real, positions); np.testing.assert_array_equal(rsd, expected)
            self.assertEqual(meta['source_raw_sha256'], hashlib.sha256(self.raw).hexdigest())
            self.assertEqual(meta['source_raw'], '/verified/original_fof')
            self.assertEqual(meta['n_halo'], 2)
            self.assertEqual(original_paths, (adapter.original.RAW_ROOT, adapter.original.OLD_POS))

    def test_missing_optional_mass_cache_and_error_restore(self):
        original_paths = (adapter.original.RAW_ROOT, adapter.original.OLD_POS)
        with TemporaryDirectory() as root:
            part = Path(root); (part / 'work').mkdir()
            with patch.object(adapter, 'PART', part):
                _, _, meta = catalog_from_blobs('fiducial', 65, self.raw, raw_source='/verified/original_fof',
                    position_cache_blob=npy_blob(self.pos[1:] / np.float32(1000.)))
                self.assertFalse(meta['optional_mass_cache_exists'])
                self.assertFalse(meta['optional_mass_cache_provided'])
                with self.assertRaises((ValueError, struct.error)):
                    catalog_from_blobs('fiducial', 65, self.raw[:-3], raw_source='/verified/original_fof')
        self.assertEqual(original_paths, (adapter.original.RAW_ROOT, adapter.original.OLD_POS))


if __name__ == '__main__': unittest.main()
