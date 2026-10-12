"""模块大纲：复现旧保存回调写死150，验证实际shared保存层修复为500。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import run_sto_physical_p_mcmc as original
from continue_all_available_pre_pcal_v2 import corrected_summary
from rsd_b000_contract import save_json

class SummaryCallbackTest(unittest.TestCase):
    def test_actual_old_sample_save_callback_is_corrected_without_sampler_change(self):
        with tempfile.TemporaryDirectory(prefix='test-pcal-save-') as folder:
            root=Path(folder);directory=root/'cases/example';directory.mkdir(parents=True)
            save_json(directory/'contract.json',dict(stage='pre',calibration_target_count=500))
            path=directory/'SUMMARY.json'
            def actual_save(path,value):save_json(path,corrected_summary(path,value))
            def fake_run(*args):
                original.shared.save_json(path,dict(calibration_target_count=9,chi2_MAP=2.5,
                    quantiles_2p5_16_50_84_97p5=[[1.,2.,3.]]))
            with patch.object(original,'OUT',root),patch.object(original,'NAME','example'),\
                    patch.object(original,'parent',return_value={}),\
                    patch.object(original.shared,'save_json',side_effect=actual_save),\
                    patch.object(original.shared,'run',side_effect=fake_run):
                original.sample(64000,False)
            result=json.loads(path.read_text())
            self.assertEqual(result['calibration_target_count'],500)
            self.assertTrue(result['actual_shared_summary_callback_fixed'])
            self.assertEqual(result['chi2_MAP'],2.5)
            self.assertEqual(result['quantiles_2p5_16_50_84_97p5'],[[1.,2.,3.]])

    def test_unrelated_output_is_not_modified(self):
        payload=dict(calibration_target_count=150)
        self.assertIs(corrected_summary(Path('MAP.json'),payload),payload)

if __name__=='__main__':unittest.main()
