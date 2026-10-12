"""模块大纲：新IO适配独立检验；字节身份、拒绝漂移、临时目录清理。"""
import hashlib
import unittest
from pathlib import Path
from run_sto_post_target_box import stage_source


class InputTest(unittest.TestCase):
    def fixture(self):
        source = Path('/original/STO')
        raw = Path('/original/FoF')
        blobs = {source/'post_sto_halo.npz': b'fixture-npz', source/'diagnostics.json': b'{"fixture":1}', raw: b'fixture-fof'}
        row = dict(cosmology='LC_m', realization=99, source=str(source),
                   source_sha256=hashlib.sha256(blobs[source/'post_sto_halo.npz']).hexdigest(),
                   diagnostics_sha256=hashlib.sha256(blobs[source/'diagnostics.json']).hexdigest(),
                   raw_fof=dict(source_raw=str(raw), source_raw_sha256=hashlib.sha256(blobs[raw]).hexdigest()))
        return row, blobs

    def test_same_bytes_and_cleanup(self):
        row, blobs = self.fixture()
        with stage_source(row, blobs.__getitem__) as (root, raw):
            source = root/'LC_m/real099/env_lam1_gsr2_thr15'
            self.assertEqual((source/'post_sto_halo.npz').read_bytes(), b'fixture-npz')
            self.assertEqual((source/'diagnostics.json').read_bytes(), b'{"fixture":1}')
            self.assertEqual(raw, b'fixture-fof')
        self.assertFalse(root.exists())
        self.assertEqual(row['source'], '/original/STO')

    def test_reject_changed_payload(self):
        row, blobs = self.fixture(); blobs[Path(row['source'])/'post_sto_halo.npz'] = b'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            with stage_source(row, blobs.__getitem__): self.fail('must not enter')


if __name__ == '__main__': unittest.main()
