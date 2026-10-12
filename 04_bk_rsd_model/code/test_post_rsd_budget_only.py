"""模块大纲：预算路由、原文件只读、克隆HDF续跑与原RNG逐位比较。"""
import shutil
import tempfile
import unittest
from pathlib import Path
import emcee
import numpy as np
from post_rsd_budget_only import checkpoint_range
from ensemble_progress_json_safe import advance_json_safe
from test_independent_ensemble_backend import ToyCase
from rsd_bonly_common import digest


class BudgetOnlyTests(unittest.TestCase):
    def test_only_original_checkpoint_range_changes(self):
        routed = checkpoint_range(64000, 128000)
        values = list(routed(6000, 64001, 2000))
        self.assertEqual(values[0], 66000)
        self.assertEqual(values[-1], 128000)
        self.assertEqual(len(values), 32)
        self.assertEqual(list(routed(4)), [0, 1, 2, 3])
        self.assertEqual(list(routed(6000, 24001, 2000)), list(range(6000, 24001, 2000)))

    def test_bad_or_nonextending_budgets_rejected(self):
        for previous, maximum in [(True, 128000), (64001, 128000), (64000, 64000),
                                  (64000, 128001), (4000, 128000)]:
            with self.assertRaises(ValueError):
                checkpoint_range(previous, maximum)

    def test_copy_resume_preserves_original_and_full_rng_trajectory(self):
        with tempfile.TemporaryDirectory(prefix='sto-budget-copy-unit-') as taskdir:
            root = Path(taskdir)
            old, extended, direct = [root / name for name in ('original.h5', 'extended.h5', 'direct.h5')]
            case = ToyCase()
            advance_json_safe(case, old, 2, 1, 17)
            before = digest(old)
            shutil.copy2(old, extended)
            advance_json_safe(case, extended, 2, 1, 40)
            advance_json_safe(case, direct, 2, 1, 40)
            self.assertEqual(digest(old), before)
            a = emcee.backends.HDFBackend(str(extended), name='ensemble_2')
            b = emcee.backends.HDFBackend(str(direct), name='ensemble_2')
            np.testing.assert_array_equal(a.get_chain(), b.get_chain())
            np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())
            np.testing.assert_array_equal(a.accepted, b.accepted)
            np.testing.assert_array_equal(a.random_state[1], b.random_state[1])
            self.assertEqual(a.random_state[2:], b.random_state[2:])


if __name__ == '__main__':
    unittest.main()
