"""模块大纲：逐盒PASS准入、来源路径保护、target活进程不释放CPU。

轻量fixture不运行重构或更改生产输入；用于批量调度前的关键守卫验收。
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import run_sto_post_covariance_box as box
import run_sto_post_covariance_queue as queue


class DispatchTest(unittest.TestCase):
    def test_pass_row_identity_and_owned_path(self):
        with TemporaryDirectory(prefix='sto_cov_dispatch_') as temporary:
            root=Path(temporary);directory=root/'audit/rows';directory.mkdir(parents=True)
            path=directory/'fiducial_real100.json'
            path.write_text(json.dumps(dict(realization=100,status='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS')))
            with patch.object(box,'ROOT',root):
                row,blob=box.read_row(path,100)
                self.assertEqual(row['cosmology'],'fiducial');self.assertEqual(json.loads(blob)['realization'],100)
                with self.assertRaises(ValueError):box.read_row(path,101)
            with patch.object(box,'ROOT',root/'different'):
                with self.assertRaises(ValueError):box.read_row(path,100)

    def test_reject_unverified_source_row(self):
        with TemporaryDirectory(prefix='sto_cov_dispatch_') as temporary:
            root=Path(temporary);directory=root/'audit/rows';directory.mkdir(parents=True)
            path=directory/'fiducial_real293.json'
            path.write_text(json.dumps(dict(realization=293,status='UNVERIFIED')))
            with patch.object(box,'ROOT',root):
                with self.assertRaises(ValueError):box.read_row(path,293)

    def test_live_target_never_releases_CPUs_even_with_complete_state(self):
        with TemporaryDirectory(prefix='sto_cov_dispatch_') as temporary:
            root=Path(temporary);directory=root/'target300_measurement_queue';directory.mkdir()
            state=dict(host=socket.gethostname(),pid=os.getpid(),state='COMPLETE',completed_count=300,workers=[])
            (directory/'STATUS.json').write_text(json.dumps(state))
            with patch.object(queue,'ROOT',root):self.assertFalse(queue.target_released())

    def test_confirmed_terminal_target_releases_CPUs(self):
        child=subprocess.Popen([sys.executable,'-c','pass']);child.wait()
        with TemporaryDirectory(prefix='sto_cov_dispatch_') as temporary:
            root=Path(temporary);directory=root/'target300_measurement_queue';directory.mkdir()
            state=dict(host=socket.gethostname(),pid=child.pid,state='COMPLETE',completed_count=300,workers=[])
            (directory/'STATUS.json').write_text(json.dumps(state))
            with patch.object(queue,'ROOT',root):self.assertTrue(queue.target_released())


if __name__=='__main__':unittest.main()
