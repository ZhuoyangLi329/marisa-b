"""模块大纲：STOP延迟确认、早期USR1不派发、v2路由及不改原数值入口。"""
import signal
import unittest
from unittest.mock import patch
import sto_post_compute8_v2_adapters as adapters


class V2AdapterTest(unittest.TestCase):
    def test_default_one_license_syntax_and_zero_not_dependency(self):
        for value in ('scratch,u2','u2:1,scratch:1','SCRATCH,cfs'):
            self.assertTrue(adapters.has_scratch_license(value))
        for value in ('','u2:1','scratch:0,u2'):
            self.assertFalse(adapters.has_scratch_license(value))

    def test_STOP_ack_waits_for_T_not_signal_return(self):
        with patch.object(adapters.common,'original_identity',return_value=True) as identity:
            with patch.object(adapters,'kernel_state',side_effect=['S','S','T']):
                self.assertTrue(adapters.confirmed_identity({'pid':12345},stopped=True))
        self.assertEqual(identity.call_args.kwargs,{'stopped':True})

    def test_missing_owner_cannot_confirm_STOP(self):
        with patch.object(adapters.common,'original_identity',return_value=False):
            self.assertFalse(adapters.confirmed_identity({'pid':12345},stopped=True))

    def test_early_USR1_blocks_new_box_before_original_main(self):
        import run_sto_post_compute8_batch as original
        previous=(original.enough_time,original.subprocess.Popen,original.OUT,
            adapters.common.OUT,signal.getsignal(signal.SIGUSR1),signal.getsignal(signal.SIGTERM))
        try:
            batch,warnings=adapters.bind_batch()
            self.assertTrue(batch.enough_time(1200))
            signal.raise_signal(signal.SIGUSR1)
            self.assertTrue(warnings)
            self.assertFalse(batch.enough_time(1200))
            self.assertEqual(batch.OUT,adapters.OUT)
        finally:
            original.enough_time,original.subprocess.Popen,original.OUT,adapters.common.OUT=previous[:4]
            signal.signal(signal.SIGUSR1,previous[4]);signal.signal(signal.SIGTERM,previous[5])

    def test_only_control_submission_and_box_bootstrap_rewritten(self):
        part=adapters.common.PART
        self.assertEqual(adapters.rewrite_command(['sbatch',str(part/'code/submit_sto_post_compute8_batch.slurm')]),
            ['sbatch','--chdir='+str(part),'--licenses=scratch,u2',str(adapters.SUBMIT)])
        original=['python',str(part/'code/run_sto_post_compute8_box.py'),'--realization','7']
        self.assertEqual(adapters.rewrite_worker(original),['python',str(adapters.BOX),'--realization','7'])
        untouched=['python',str(part/'code/run_sto_post_target_box.py'),'--realization','7']
        self.assertEqual(adapters.rewrite_worker(untouched),untouched)
        self.assertEqual(adapters.rewrite_command(['sacct','-j','59597886']),['sacct','-j','59597886'])

    def test_worker_preexec_ignores_only_warning_and_preserves_CPU_hook(self):
        previous=signal.getsignal(signal.SIGUSR1);term=signal.getsignal(signal.SIGTERM);calls=[]
        try:
            adapters.protect_worker_preexec(lambda:calls.append('CPU hook'))
            self.assertEqual(signal.getsignal(signal.SIGUSR1),signal.SIG_IGN)
            signal.raise_signal(signal.SIGUSR1)
            self.assertEqual(calls,['CPU hook'])
            self.assertEqual(signal.getsignal(signal.SIGTERM),term)
        finally:
            signal.signal(signal.SIGUSR1,previous)


if __name__=='__main__':unittest.main()
