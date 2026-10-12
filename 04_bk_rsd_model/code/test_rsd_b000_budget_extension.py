"""模块大纲：增加链长不改模型/kernel/诊断；来源版本与初始记录防覆盖。"""
import json
import tempfile
import unittest
from pathlib import Path
from rsd_b000_contract import digest,save_json
from run_rsd_b000_mcmc import budget_provenance,compatible_sampler_hashes,checkpoint_targets

class BudgetExtensionTest(unittest.TestCase):
    def test_explicit_budget_extension_preserves_initial_provenance(self):
        hashes=compatible_sampler_hashes();self.assertEqual(len(hashes),2)
        old='6be2c9d740a6a0a9bcaf79702a01df7ddf0813ee3228605add8f262a1676c135'
        new=next(h for h in hashes if h!=old)
        initial={'sampler_sha256':old,'max_steps':24000,'likelihood_sha256':'synthetic',
                 'diagnostics_sha256':'unchanged','emcee_version':'synthetic','ensembles':4,'walkers':32}
        current={**initial,'sampler_sha256':new,'max_steps':48000}
        with tempfile.TemporaryDirectory(prefix='rsd_budget_QA_') as directory:
            root=Path(directory);save_json(root/'RUN_PROVENANCE.json',initial)
            before=digest(root/'RUN_PROVENANCE.json')
            with self.assertRaises(ValueError):budget_provenance(root,current)
            self.assertFalse(budget_provenance(root,current,True))
            self.assertEqual(digest(root/'RUN_PROVENANCE.json'),before)
            history=json.loads((root/'BUDGET_EXTENSION_PROVENANCE.json').read_text())
            self.assertTrue(history[0]['budget_only_AST_verified'])
            with self.assertRaises(ValueError):budget_provenance(root,{**current,'likelihood_sha256':'changed'},True)
            save_json(root/'SUMMARY.json',{'converged':True,'likelihood_sha256':'synthetic'})
            self.assertTrue(budget_provenance(root,{**current,'max_steps':24000}))
        self.assertEqual(list(checkpoint_targets([24000]*4,max_steps=48000))[0],24000)
        self.assertEqual(list(checkpoint_targets([24000]*4,max_steps=48000))[-1],48000)

if __name__=='__main__':unittest.main()
