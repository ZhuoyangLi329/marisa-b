"""模块大纲：实际原P采样类/物理p模型/白化残差及PID复用保护单测。

只创建临时接口fixture，不启动生产链，也不将fixture报告为科学验收。
"""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_sto_post_p0only_inputs import PostP0InputsTest
from post_rsd_p0only_inputs import make_p0_case

try:
    import run_sto_post_p0only as runner
    import continue_sto_post_p0only as continuation
    from run_rsd_p0only import POnlyCase as OriginalPOnlyCase
except ImportError:
    runner = None


@unittest.skipIf(runner is None, 'requires existing NERSC iminuit/emcee/h5py environment')
class P0AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PostP0InputsTest.setUpClass()

    def test_reuses_original_MAP_and_checks_full_physical_domain_predictions(self):
        e = PostP0InputsTest.ensemble; spec = PostP0InputsTest.specs[0]
        t = PostP0InputsTest.templates
        self.assertIs(runner.POnlyCase.initialize, OriginalPOnlyCase.initialize)
        rng = np.random.default_rng(202610100101)
        for p in (.5, 1.417, 2.8):
            arrays, meta = make_p0_case(e, spec, t, 3336., p_halo=p)
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); path = directory/'likelihood.npz'
                np.savez_compressed(path, **arrays)
                meta['likelihood_sha256'] = runner.digest(path)
                (directory/'contract.json').write_text(json.dumps(meta))
                case = runner.POnlyCase(directory)
                physical = np.column_stack((rng.uniform(-499., 499., 80), rng.uniform(.501, 3.999, 80),
                    rng.uniform(-.9e6, .9e6, 80)/3336.))
                f, b, shot = physical.T; bp = 2*1.686*(b-p); growth = runner.F_GROWTH
                expected = (b[:, None]**2*t[0] + 2*b[:, None]*growth*t[1] + growth**2*t[2]
                    + 2*f[:, None]*bp[:, None]*(b[:, None]*t[3] + growth*t[4])
                    + f[:, None]**2*bp[:, None]**2*t[5] + shot[:, None]*3336.)
                np.testing.assert_allclose(case.predictions(physical), expected, rtol=2e-13, atol=2e-8)
                np.testing.assert_allclose(case.residual(physical),
                    np.linalg.solve(case.chol, (expected-case.target).T).T, rtol=3e-13, atol=2e-8)

    def test_missing_actual_process_gate_does_not_create_MCMC_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'must_not_create'
            with patch.object(runner, 'OUT', out), patch.object(runner.os, 'sched_getaffinity', return_value={7}), \
                    patch.object(continuation, 'validate_calibration_exit', side_effect=ValueError('actual p parent live')):
                with self.assertRaises(ValueError):
                    runner.freeze()
            self.assertFalse(out.exists())

    def test_exact_process_identity_not_status_or_timeout_and_PID_reuse(self):
        command = (Path('/proc')/str(os.getpid())/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
        entry = command.split()[0]
        record = continuation.identity(os.getpid(), entry)
        self.assertTrue(continuation.specific_process_is_live(record))
        self.assertFalse(continuation.specific_process_is_live(dict(record, starttime_ticks='not_original_starttime')))
        with self.assertRaises(ValueError):
            continuation.specific_process_is_live(dict(record, entry='/never_matching_entry_fixture.py'))
        with self.assertRaises(ValueError):
            continuation.specific_process_is_live(dict(record, host='not_'+socket.gethostname()))


if __name__ == '__main__':
    unittest.main()
