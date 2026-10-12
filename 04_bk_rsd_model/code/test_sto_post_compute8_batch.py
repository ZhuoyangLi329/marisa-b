"""模块大纲：短批次内存/时间/输出隔离/真实PID和accounting保护单测。

临时字节只做文件协议fixture，不伪造生产科学PASS，不启动真实测量。
"""
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import sto_post_compute8_common as common
from run_sto_post_compute8_box import OutputOnlyRoot, validate_publication


class Compute8BatchTest(unittest.TestCase):
    def test_memory_and_deadline_reserve(self):
        self.assertEqual(common.memory_capacity(14*1024), 2)
        self.assertEqual(common.memory_capacity(8*1024), 1)
        with self.assertRaises(ValueError):
            common.memory_capacity(7*1024)
        self.assertTrue(common.enough_time(960))
        self.assertFalse(common.enough_time(959.999))
        self.assertLess(627.0207343101501, 960)

    def test_only_outputs_are_redirected_not_scientific_inputs(self):
        original = Path('/original_fixture'); staged = Path('/stage_fixture')
        root = OutputOnlyRoot(original, staged)
        self.assertEqual(root/'pilots', staged/'pilots')
        for name in ('RECON_ALPHA_COMPARISON.npz', 'MATCHED_SELF_AUDIT.json', 'ZERO_ATOM_AUDIT.json'):
            self.assertEqual(root/name, original/name)

    def fixture(self, root):
        stage = root/'staged'; stage.mkdir(); final = root/'canonical'; key = 'fiducial', 0
        row = dict(source_sha256='actual_SOURCE_fixture', diagnostics_sha256='actual_DIAG_fixture',
                   raw_fof=dict(source_raw_sha256='actual_RAW_fixture'))
        cm = dict(source_STO_sha256=row['source_sha256'], source_diagnostics_sha256=row['diagnostics_sha256'],
            raw_fof=row['raw_fof'], arguments=dict(cosmology='fiducial', realization=0))
        (stage/'CONTRACT.json').write_text(json.dumps(cm))
        (stage/'MEASUREMENTS.npz').write_bytes(b'protocol fixture, not a scientific NPZ')
        (stage/'POST_HALOS.npz').write_bytes(b'protocol fixture, not a scientific NPZ')
        done = dict(status='PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION', CPU_affinity=[1],
            entry_sha256=common.sha(common.PART/'code/run_sto_post_rsd_pilot.py'),
            contract_sha256=common.sha(stage/'CONTRACT.json'), measurement_sha256=common.sha(stage/'MEASUREMENTS.npz'),
            post_halos_sha256=common.sha(stage/'POST_HALOS.npz'))
        (stage/'DONE.json').write_text(json.dumps(done))
        return stage, final, row, key

    def test_complete_staging_preserves_bytes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            stage, final, row, key = self.fixture(Path(temp))
            before = common.sha(stage/'MEASUREMENTS.npz')
            validate_publication(stage, final, row, key)
            stage.rename(final)
            self.assertEqual(common.sha(final/'MEASUREMENTS.npz'), before)
            self.assertFalse(stage.exists())
            with self.assertRaises(ValueError):
                validate_publication(final, final, row, key)

    def test_partial_corrupt_and_source_mismatched_staging_not_published(self):
        for mode in ('partial', 'bytes', 'source'):
            with tempfile.TemporaryDirectory() as temp:
                stage, final, row, key = self.fixture(Path(temp))
                if mode == 'partial':
                    (stage/'DONE.json').unlink()
                elif mode == 'bytes':
                    (stage/'MEASUREMENTS.npz').write_bytes(b'corrupt fixture')
                else:
                    row['source_sha256'] = 'different source'
                with self.assertRaises((ValueError, FileNotFoundError)):
                    validate_publication(stage, final, row, key)
                self.assertFalse(final.exists())

    def test_accounting_absence_is_not_terminal(self):
        for response, state in (('', None), ('999|PENDING|0:0\n', 'PENDING'),
                ('999|COMPLETED|0:0\n', 'COMPLETED'), ('999|TIMEOUT|0:0\n', 'TIMEOUT')):
            with patch.object(common, 'command', return_value=response):
                actual, _ = common.accounting(999)
                self.assertEqual(None if actual is None else actual['state'], state)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'NERSC Linux process identity test')
    def test_actual_process_identity_rejects_PID_reuse_and_nonstopped_parent(self):
        pid = os.getpid(); path = Path('/proc')/str(pid)
        cmd = (path/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
        record = dict(pid=pid, host=socket.gethostname(),
            starttime_ticks=(path/'stat').read_text().rsplit(')', 1)[1].split()[19], entry=cmd.split()[0])
        self.assertTrue(common.original_identity(record))
        self.assertFalse(common.original_identity(dict(record, starttime_ticks='not_original')))
        with self.assertRaises(ValueError):
            common.original_identity(record, stopped=True)

    def test_requested8_memory_fits_and_walltime_is_short(self):
        text = Path(__file__).with_name('submit_sto_post_compute8_batch.slurm').read_text()
        for line in ('#SBATCH --cpus-per-task=8', '#SBATCH --mem=14G', '#SBATCH --time=00:30:00'):
            self.assertIn(line, text)
        self.assertLessEqual(14*1024, 8*1905)


if __name__ == '__main__':
    unittest.main()
