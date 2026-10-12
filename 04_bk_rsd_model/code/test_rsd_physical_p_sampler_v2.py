"""模块大纲：通用修正版分别保留pre500/post250样本数、stage和已配置精度。"""
import tempfile
from pathlib import Path
import unittest
from rsd_b000_contract import save_json
from rsd_physical_p_sampler_v2 import corrected_payload

class GeneralPcalSummaryTest(unittest.TestCase):
    def test_pre_and_post_actual_case_counts_and_precision(self):
        with tempfile.TemporaryDirectory(prefix='pcal-metadata-adapter-') as folder:
            for stage,count in [('pre',500),('post',250)]:
                d=Path(folder)/stage;d.mkdir()
                save_json(d/'contract.json',dict(stage=stage,calibration_target_count=count,calibration_ids=list(range(count)),
                    covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.))
                old=dict(calibration_target_count=150,chi2_MAP=1.2,quantiles=[[1,2,3]])
                result=corrected_payload(d/'SUMMARY.json',old)
                self.assertEqual(result['calibration_target_count'],count);self.assertEqual(result['stage'],stage)
                self.assertEqual(result['covariance_multiplier'],1.);self.assertFalse(result['covariance_is_of_mean'])
                self.assertEqual(result['quantiles'],old['quantiles']);self.assertEqual(old['calibration_target_count'],150)

    def test_non_summary_output_is_untouched(self):
        value=dict(p=1.2);self.assertIs(corrected_payload(Path('MAP.json'),value),value)

if __name__=='__main__':unittest.main()
