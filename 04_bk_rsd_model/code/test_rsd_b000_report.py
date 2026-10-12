"""模块大纲：八页矢量边界 QA；临时合成输入不写 PDF/PNG，不作科学恢复声明。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import report_rsd_b000 as report
from rsd_b000_contract import save_json

class FigureSink:
    def savefig(self,figure):pass

class ReportTest(unittest.TestCase):
    def test_all_page_text_bounds_with_synthetic_layout_fixture(self):
        with tempfile.TemporaryDirectory(prefix='rsd_b000_layout_QA_') as directory:
            root=Path(directory);save_json(root/'contract.json',{'synthetic_layout_QA_only':True})
            rows={};rng=np.random.default_rng(101)
            # fixture 不得读取正式输出里的 baseline 列表，保证测试与任务状态隔离。
            fixture_names=[f'{c}_{v}_k{k:.2f}' for c in report.COSMOS for v in ('mean100','single000')
                           for k in (report.CUTS if v=='mean100' else (.12,))]
            for name in fixture_names:
                cosmo=name.split('_mean')[0].split('_single')[0];truth=report.TRUTHS[cosmo]
                median=np.array([truth,1.9,-.2,-.3,.1,.2]);q=np.stack([median+d for d in [-2,-1,0,1,2]])
                meta={'name':name,'cosmology':cosmo,'case_type':'baseline','kmax_B':float(name[-4:]),
                      'volume_Gpc_h3':1 if 'single' in name else 100,'converged':True,
                      'quantiles_2p5_16_50_84_97p5':q.tolist(),'fNL_sigma':1.,'fNL_pull':0.,
                      'chi2_MAP':1.,'gof_pvalue':.5,'prior_sensitive':False,'steps':6000,
                      'rank_normalized_split_rhat':[1.]*6,'bulk_ess':[2500.]*6,'tail_ess':[2500.]*6,
                      'gates':{k:True for k in ['rhat','fNL_ESS','other_ESS','length_tau','half_shift','ensemble_spread','acceptance']}}
                rows[name]=meta;case=root/'cases'/name;case.mkdir(parents=True);save_json(case/'SUMMARY.json',meta)
                if name.endswith('mean100_k0.12'):
                    coef=np.zeros((4,3,3,20));noise=np.zeros((2,20))
                    np.savez(case/'likelihood.npz',target=np.ones(20)*1e8,covariance=np.eye(20)*1e14,coefficients=coef,noise=noise)
                    save_json(case/'MAP.json',{'center':median.tolist(),'transform':np.eye(6).tolist()})
                    np.savez(case/'POSTERIOR_THIN.npz',samples=rng.normal(size=(500,6))+median)
                    with h5py.File(case/'chain.h5','w') as h:
                        for e in range(4):h.create_dataset(f'ensemble_{e}/chain',data=rng.normal(size=(10,32,6)))
            save_json(root/'MODEL_NOISE_AUDIT.json',{'parameter_count':6,'native_SP_independent_combination_retained':False})
            np.savez(root/'B000_ENSEMBLE_MATRIX.npz',certified_indices=np.arange(44),covariance_single_box=np.eye(120),
                     stochastic_geometry=np.stack([np.ones(44)*v for v in [1,1/3,1/5,1]]))
            with patch.object(report,'OUT',root):
                value,actual=report.collect();qa=[];sink=FigureSink()
                self.assertEqual(value['completed_baseline_cases'],21)
                report.summary_page(sink,value,qa);report.scan_page(sink,actual,qa);report.diagnostics_page(sink,actual,qa)
                report.fit_page(sink,actual,qa);report.noise_page(sink,qa);report.posterior_page(sink,actual,qa)
                report.prior_page(sink,value,qa);report.trace_page(sink,actual,qa)
                self.assertEqual(len(qa),8);self.assertTrue(all(v['text_within_page'] for v in qa))

if __name__=='__main__':unittest.main()
