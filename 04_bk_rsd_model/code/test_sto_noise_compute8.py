"""模块大纲：8核shared内存/实际作业规格保护，不生成科学PASS。"""
from pathlib import Path
import unittest
from run_sto_noise_transport_compute8 import capacity8


class Compute8Test(unittest.TestCase):
    def test_exact8_and_same_original_memory_budget(self):
        self.assertEqual(capacity8(list(range(8)), 14*1024), 2)
        for allowed in (list(range(7)), list(range(9)), list(range(32))):
            with self.assertRaises(ValueError):
                capacity8(allowed, 14*1024)
        with self.assertRaises(ValueError):
            capacity8(list(range(8)), 7*1024)

    def test_submission_does_not_force_memory_CPU_inflation(self):
        script = Path(__file__).with_name('submit_sto_noise_transport_compute8.slurm').read_text()
        for line in ('#SBATCH --cpus-per-task=8', '#SBATCH --mem=14G', '#SBATCH --time=00:20:00'):
            self.assertIn(line, script)
        self.assertLessEqual(14*1024, 8*1905)
        self.assertNotIn('--cpus-per-task=32', script)
        self.assertIn('run_sto_noise_transport_compute8.py', script)


if __name__ == '__main__':
    unittest.main()
