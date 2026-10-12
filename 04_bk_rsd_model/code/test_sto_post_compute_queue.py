"""模块大纲：调度迁移不改变样本、身份、原producer或单线程/MPI约束。"""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import run_sto_post_compute_queue as queue
import migrate_sto_post_login_queue as migration
import watch_sto_post_compute_migration as watcher


class ComputeQueueTest(unittest.TestCase):
    def test_exact_original700_and_overlap_reuse(self):
        selected = queue.keys()
        self.assertEqual(len(selected), 700)
        self.assertEqual(len(set(selected)), 700)
        self.assertEqual([i for c, i in selected if c == 'fiducial'], list(range(500)))
        for c in ('LC_p', 'LC_m'):
            self.assertEqual([i for cc, i in selected if cc == c], list(range(100)))

    def test_existing_numeric_entries_not_reimplemented(self):
        for c in ('fiducial', 'LC_p', 'LC_m'):
            cmd = queue.worker_command((c, 17), None)
            self.assertIn(str(queue.PART / 'code/run_sto_post_target_box.py'), cmd)
            self.assertEqual(cmd[-4:], ['--cosmology', c, '--realization', '17'])
        cmd = queue.worker_command(('fiducial', 471), Path('/example/rows/fiducial_real471.json'))
        self.assertIn(str(queue.PART / 'code/run_sto_post_covariance_box.py'), cmd)
        self.assertIn('--source-row', cmd)

    def test_singleton_MPI_and_singlethread_environment(self):
        with patch.dict(os.environ, {'PMI_RANK': '0', 'PMI_SIZE': '1', 'PMIX_SERVER_URI': 'old',
                                    'SLURM_PROCID': '0', 'OMP_NUM_THREADS': '32'}):
            env = queue.worker_environment()
        self.assertFalse(any(k.startswith(('PMI_', 'PMIX_')) for k in env))
        self.assertNotIn('SLURM_PROCID', env)
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['JAX_PLATFORMS'], 'cpu')

    def test_actual_shared_memory_not32_oversubscribed_boxes(self):
        self.assertEqual(queue.memory_worker_limit(1905 * 32), 9)
        self.assertEqual(queue.memory_worker_limit(250000), 32)
        with self.assertRaises(ValueError):
            queue.memory_worker_limit(8000)

    def test_no_finished_artifact_fallback(self):
        with TemporaryDirectory(prefix='sto_cpuqueue_') as tmp:
            root = Path(tmp)
            with patch.object(queue, 'ROOT', root):
                self.assertFalse(queue.verified_finished(('fiducial', 123), {}, 'pilot'))
                directory = queue.directory(('fiducial', 123))
                directory.mkdir(parents=True)
                (directory / 'DONE.json').write_text('{}')
                (directory / 'CONTRACT.json').write_text('{}')
                with self.assertRaises(KeyError):
                    queue.verified_finished(('fiducial', 123), {}, 'pilot')

    def test_no_migration_until_actual_parent_termination(self):
        with TemporaryDirectory(prefix='sto_cpuqueue_') as tmp:
            root = Path(tmp); out = root / 'compute32_migration'; out.mkdir()
            (out / 'MIGRATION.json').write_text(json.dumps(dict(
                state='DISPATCHERS_STOPPED_WORKERS_DRAINING', original_parent_processes_confirmed_absent=False)))
            with patch.object(queue, 'ROOT', root), self.assertRaises(ValueError):
                queue.migration_gate()

    def test_process_starttime_and_missing_PID(self):
        parsed = migration.parse_stat('17 (name with spaces) ' + ' '.join(
            ['T', '12'] + ['0'] * 17 + ['98765'] + ['0'] * 20))
        self.assertEqual(parsed, dict(state='T', ppid=12, starttime=98765))
        if Path('/proc/self/stat').exists():
            current = migration.process(os.getpid())
            self.assertEqual(current['pid'], os.getpid())
            self.assertGreater(current['starttime'], 0)
        self.assertIsNone(migration.process(99999999))

    def test_only_matching_held_job_can_be_released(self):
        value=('JobId=17 JobName=sm_post_rsd_cpu32 Account=desi QOS=shared NumCPUs=32 '
               'CPUs/Task=32 ReqTRES=cpu=32,mem=60G,node=1,billing=32 JobState=PENDING Reason=JobHeldUser Command=' +
               str(queue.PART/'code/submit_sto_post_compute32.slurm'))
        self.assertEqual(watcher.check_job(value, 17)['NumCPUs'], '32')
        packed=value.replace('NumCPUs=32','NumCPUs=33').replace('CPUs/Task=32','CPUs/Task=33')
        self.assertEqual(watcher.check_job(packed,17)['ReqTRES'],'cpu=32,mem=60G,node=1,billing=32')
        for wrong in (value.replace('NumCPUs=32', 'NumCPUs=35'),
                      value.replace('JobHeldUser', 'Licenses'),value.replace('JobId=17','JobId=18')):
            with self.assertRaises(ValueError):watcher.check_job(wrong,17)


if __name__ == '__main__':
    unittest.main()
