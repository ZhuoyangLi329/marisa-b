"""模块大纲：实际模板审计schema回归；缺完整测量时不创建主MCMC目录。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import freeze_sto_post_noir_cases as freeze


class FreezeGatesTest(unittest.TestCase):
    def reports(self):
        # 只读既有44-bin真实报告；不制造通过科学验收的噪声/协方差产物。
        root=Path(__file__).resolve().parents[1]/'outputs/sto_marisa_post_rsd_mean100_v1/noir_paware_templates'
        completion=json.loads((root/'COMPLETION_AUDIT.json').read_text())
        independent=json.loads((root/'INDEPENDENT_COEFFICIENT_AUDIT.json').read_text())
        return completion,independent

    def check(self,completion,independent):
        freeze.validate_template_audits(completion,independent,
            completion['coefficients_sha256'],completion['contract_sha256'])

    def test_actual_independent_replay_schema(self):
        self.check(*self.reports())

    def test_status_only_or_old_guessed_status_cannot_pass(self):
        completion,independent=self.reports()
        for status in ('PASS','ALL44_NOIR_COEFFICIENTS_INDEPENDENTLY_VERIFIED'):
            bad=dict(independent,status=status)
            with self.assertRaises(ValueError):self.check(completion,bad)

    def test_replay_counts_and_same_hashes_required(self):
        completion,independent=self.reports()
        for key,bad in (('bins',43),('nodes_replayed',659),('fine_probes_replayed',87),
                        ('rows',independent['rows'][:-1]),('coefficients_sha256','different'),
                        ('contract_sha256','different')):
            changed=dict(independent);changed[key]=bad
            with self.assertRaises(ValueError):self.check(completion,changed)

    def test_missing_ensemble_cannot_create_MCMC_output(self):
        with TemporaryDirectory(prefix='sto_freeze_gate_') as tmp:
            root=Path(tmp);out=root/'mcmc_noir'
            with patch.object(freeze,'ROOT',root),patch.object(freeze,'OUT',out),\
                 patch.object(freeze,'ENSEMBLE',root/'ensemble_measurements'),\
                 patch.object(freeze.os,'sched_getaffinity',return_value={11},create=True):
                with self.assertRaises(FileNotFoundError):freeze.main()
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
