"""模块大纲：后续p校准的通用修正版保存适配，pre/post均从实际case读取样本数。

保留历史v1源码和链；不改唯一三参数模型、emcee或协方差，只修摘要回调。
新校准输入应先在独立输出目录冻结，再调用此入口，不能覆盖旧链。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import run_sto_physical_p_mcmc as original
from rsd_b000_contract import PART,digest

def corrected_payload(path,value):
    if Path(path).name!='SUMMARY.json':return value
    meta=json.loads((Path(path).parent/'contract.json').read_text())
    return dict(value,stage=meta['stage'],calibration_target_count=meta['calibration_target_count'],
        calibration_ids=meta['calibration_ids'],all_available_positive_samples_used=True,
        strict_independent_calibration_claimed=meta.get('strict_independent_calibration_claimed',False),
        covariance_is_of_mean=meta['covariance_is_of_mean'],
        covariance_multiplier=meta['covariance_multiplier'],volume_Gpc_h3=meta['volume_Gpc_h3'],
        actual_shared_summary_callback_fixed=True,
        gof_note='calibration covariance and precision follow the actual case; not automatic forecast GOF')

def sample(root,name,max_steps=64000):
    root=Path(root).resolve();root.relative_to(PART/'outputs')
    d=root/'cases'/name
    if (d/'chain.h5').exists():raise ValueError('existing chain preserved; use its specific audited continuation, not refreeze')
    meta=json.loads((d/'contract.json').read_text())
    if meta['parameters']!=['physical_p','b1','Pshot_residual']:
        raise ValueError('this adapter is only the actual three-parameter physical-p case')
    old=(original.OUT,original.NAME,original.shared.save_json,original.shared.budget_provenance)
    def save(path,value):return old[2](path,corrected_payload(path,value))
    def provenance(directory,value,extend=False):
        value.update(actual_summary_adapter_sha256=digest(__file__))
        return old[3](directory,value,extend)
    original.OUT,original.NAME=root,name
    original.shared.save_json=save;original.shared.budget_provenance=provenance
    try:original.sample(max_steps,False)
    finally:original.OUT,original.NAME,original.shared.save_json,original.shared.budget_provenance=old

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input-directory',required=True)
    ap.add_argument('--case-name',required=True);ap.add_argument('--max-steps',type=int,default=64000)
    args=ap.parse_args();sample(args.input_directory,args.case_name,args.max_steps)
