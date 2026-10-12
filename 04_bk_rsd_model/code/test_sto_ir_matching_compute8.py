"""模块大纲：重投影内存不超配、固定8核和真实Slurm截止时间保护。"""
import os
import unittest
from unittest.mock import patch
from run_sto_ir_matching_compute8 import resource_workers, allocation_deadline


class MatchingResourceTest(unittest.TestCase):
    def test_one_heavy_worker_and_no_60G_or_16_cpu_fallback(self):
        self.assertEqual(resource_workers(list(range(8)),14336),1)
        for cpus, memory in ((list(range(16)),14336),(list(range(8)),61440),([1]*8,14336)):
            with self.assertRaises(ValueError): resource_workers(cpus,memory)

    def test_deadline_environment_or_specific_job_not_elapsed_guess(self):
        with patch.dict(os.environ,{'SLURM_JOB_END_TIME':'1791600000'}):
            self.assertEqual(allocation_deadline('123'),1791600000)
        with patch.dict(os.environ,{},clear=True), patch('subprocess.check_output',return_value='JobId=123 EndTime=2026-10-10T02:00:00') as tool:
            self.assertGreater(allocation_deadline('123'),0)
            self.assertEqual(tool.call_args.args[0],['scontrol','show','job','-o','123'])
        with patch.dict(os.environ,{},clear=True), patch('subprocess.check_output',return_value='JobId=123'):
            with self.assertRaises(ValueError): allocation_deadline('123')


if __name__ == '__main__': unittest.main()
