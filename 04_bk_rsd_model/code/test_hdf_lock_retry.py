"""模块大纲：瞬时锁重试、非锁拒绝与真实跨进程HDF锁；不修改科学链。"""
import errno
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py
from hdf_lock_retry import open_with_lock_retry


class HDFLockRetryTests(unittest.TestCase):
    def test_only_failed_open_retried(self):
        calls=[]; events=[]; payload=object()
        def opener():
            calls.append(1)
            if len(calls)<3:raise BlockingIOError(errno.EAGAIN,'unable to lock file')
            return payload
        self.assertIs(open_with_lock_retry(opener,pause=.001,on_retry=events.append),payload)
        self.assertEqual(len(calls),3);self.assertEqual(events,[1,2])

    def test_unrelated_error_or_expired_wait_propagates(self):
        for error in (OSError(5,'IO failure'),OSError(errno.EAGAIN,'network unavailable')):
            calls=[]
            def fail():calls.append(1);raise error
            with self.assertRaises(OSError):open_with_lock_retry(fail,pause=.001)
            self.assertEqual(len(calls),1)
        def busy():raise BlockingIOError(errno.EAGAIN,'unable to lock file')
        with self.assertRaises(BlockingIOError):open_with_lock_retry(busy,maximum_wait=.01,pause=.002)

    def test_actual_cross_process_hdf_lock(self):
        with tempfile.TemporaryDirectory(prefix='sto-hdf-lock-unit-') as taskdir:
            path=Path(taskdir)/'probe.h5'
            with h5py.File(path,'w') as f:f['probe']=[1.]
            program="import h5py,sys,time; f=h5py.File(sys.argv[1],'r+'); print('LOCKED',flush=True); time.sleep(.3); f.close()"
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            child=subprocess.Popen([sys.executable,'-B','-c',program,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            try:
                self.assertEqual(child.stdout.readline().strip(),'LOCKED')
                events=[]
                with open_with_lock_retry(lambda:h5py.File(path,'r+'),maximum_wait=3.,pause=.02,on_retry=events.append) as f:
                    self.assertEqual(f['probe'][0],1.)
                self.assertGreater(len(events),0)
                self.assertEqual(child.wait(timeout=5),0)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
                child.stdout.close();child.stderr.close()


if __name__=='__main__':unittest.main()
