"""模块大纲：恢复入口与旧入口都必须等真实PID终止，未知/暂停状态不得放行。"""
import tempfile
import unittest
from pathlib import Path
from post_rsd_owner_quiescence import owner_quiescent


class OwnerQuiescenceTests(unittest.TestCase):
    def test_live_resumed_owner_ignores_finished_status_claim(self):
        with tempfile.TemporaryDirectory(prefix='sto-owner-guard-') as taskdir:
            root=Path(taskdir);proc=root/'123';proc.mkdir()
            for state in ('R','S','T'):
                (proc/'stat').write_text('123 (resumed worker with spaces) '+state+' 1 2 3')
                self.assertFalse(owner_quiescent({'pid':123,'state':'FINISHED'},root))
            (proc/'stat').unlink()
            self.assertFalse(owner_quiescent({'pid':123},root))

    def test_missing_or_authoritatively_terminal(self):
        with tempfile.TemporaryDirectory(prefix='sto-owner-guard-') as taskdir:
            root=Path(taskdir)
            self.assertTrue(owner_quiescent({'pid':123},root))
            proc=root/'123';proc.mkdir()
            for state in ('Z','X'):
                (proc/'stat').write_text('123 (python) '+state+' 1 2 3')
                self.assertTrue(owner_quiescent({'pid':123},root))
            with self.assertRaises(ValueError):owner_quiescent({'pid':True},root)


if __name__=='__main__':unittest.main()
