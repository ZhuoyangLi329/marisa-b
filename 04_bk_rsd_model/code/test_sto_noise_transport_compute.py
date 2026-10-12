"""模块大纲：32核心上限/4GiB独立worker内存预算；噪声模板不是主MCMC。"""
import unittest
from run_sto_noise_transport_compute import capacity


class NoiseComputeTest(unittest.TestCase):
    def test_memory_and_parent_core_budget(self):
        self.assertEqual(capacity(list(range(32)),61440),14)
        self.assertEqual(capacity(list(range(32)),60960),13)
        self.assertEqual(capacity(list(range(8)),61440),7)

    def test_insufficient_allocation_or_CPU_expansion_rejected(self):
        for cpus,mem in (([],61440),(list(range(33)),61440),([0],61440),([0,1],4096)):
            with self.assertRaises(ValueError):capacity(cpus,mem)


if __name__=='__main__':unittest.main()
