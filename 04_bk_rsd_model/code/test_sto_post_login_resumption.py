"""模块大纲：actual user memory cap保守预算；不创建科学数据或伪迁移PASS。"""
import unittest
from unittest.mock import patch
from pathlib import Path
from run_sto_post_login_queue_v2 import memory_budget


class LoginBudgetTest(unittest.TestCase):
    def budget(self,cap):
        def read(path,*args,**kwargs):
            name=str(path)
            if name=='/proc/self/cgroup':return '0::/user.slice/user-112513.slice/session-test.scope\n'
            if name=='/sys/fs/cgroup/user.slice/user-112513.slice/memory.max':return str(cap)
            if name.endswith('/memory.max'):return 'max'
            raise FileNotFoundError(name)
        with patch.object(Path,'read_text',read):return memory_budget()

    def test_parent_user_cap_found_not_unlimited_session(self):
        result=self.budget(30*1024**3);self.assertEqual(result['maximum_workers'],4)
        self.assertEqual(result['cgroup'],'/sys/fs/cgroup/user.slice/user-112513.slice')

    def test_decimal_cap_and_small_memory_no_overcommit(self):
        self.assertEqual(self.budget(30_000_000_000)['maximum_workers'],3)
        with self.assertRaises(ValueError):self.budget(8*1024**3)


if __name__=='__main__':unittest.main()
