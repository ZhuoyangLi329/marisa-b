"""模块大纲：真实HDF/进度JSON/断点RNG回归；复现并修复第一步int64报错。"""
import json
import tempfile
import unittest
from pathlib import Path
import emcee
import numpy as np
from independent_ensemble_backend import advance_ensemble
from ensemble_progress_json_safe import advance_json_safe
from test_independent_ensemble_backend import ToyCase


class ProgressJSONTests(unittest.TestCase):
    def test_real_progress_json_and_bitwise_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='sto-progress-int64-unit-') as taskdir:
            old=Path(taskdir)/'without-progress.h5';new=Path(taskdir)/'with-progress.h5'
            case=ToyCase();events=[]
            def progress(actual):
                self.assertIs(type(actual),int)
                events.append(json.loads(json.dumps({'actual_writer_saved_steps':actual})))
            advance_ensemble(case,old,0,1,40)
            advance_json_safe(case,new,0,1,1,progress)
            advance_json_safe(case,new,0,1,40,progress)
            a=emcee.backends.HDFBackend(str(old),name='ensemble_0')
            b=emcee.backends.HDFBackend(str(new),name='ensemble_0')
            np.testing.assert_array_equal(a.get_chain(),b.get_chain())
            np.testing.assert_array_equal(a.get_log_prob(),b.get_log_prob())
            np.testing.assert_array_equal(a.accepted,b.accepted)
            np.testing.assert_array_equal(a.random_state[1],b.random_state[1])
            self.assertEqual(events[0]['actual_writer_saved_steps'],1)
            self.assertEqual(events[-1]['actual_writer_saved_steps'],40)


if __name__=='__main__':unittest.main()
