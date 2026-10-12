"""模块大纲：跨进程锁排他性、子进程继承锁、原700身份唯一性。"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from run_sto_post_simultaneous import claim,keys


class SharedClaimTests(unittest.TestCase):
    def test_700_identities(self):
        identities=keys()
        self.assertEqual(len(identities),700)
        self.assertEqual(len(set(identities)),700)
        self.assertEqual(sum(c=='fiducial' for c,_ in identities),500)
        self.assertEqual(sum(c=='LC_p' for c,_ in identities),100)
        self.assertEqual(sum(c=='LC_m' for c,_ in identities),100)

    def test_independent_process_cannot_claim_same_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.lock';lease=claim(path)
            self.assertIsNotNone(lease)
            code='import fcntl,sys;f=open(sys.argv[1],"a+");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
            result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            lease.close()
            result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True)
            self.assertEqual(result.returncode,0)

    def test_worker_keeps_claim_when_parent_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.lock';lease=claim(path)
            code='import os,sys;os.fstat(int(sys.argv[1]));print("ready",flush=True);sys.stdin.readline()'
            child=subprocess.Popen([sys.executable,'-c',code,str(lease.fileno())],
                pass_fds=(lease.fileno(),),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'ready')
                lease.close()
                self.assertIsNone(claim(path))
                child.stdin.write('finish\n');child.stdin.flush();self.assertEqual(child.wait(timeout=5),0)
                final=claim(path);self.assertIsNotNone(final);final.close()
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
                child.stdin.close();child.stdout.close()


if __name__=='__main__':unittest.main()
