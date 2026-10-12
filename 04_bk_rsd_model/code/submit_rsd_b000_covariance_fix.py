"""模块大纲：单盒 covariance 修正验收；新链自动依赖；提交来源留档。

不重测，不恢复旧链；提交 ID 不代表收敛，必须等待 SUMMARY 和最终审计。
"""
import json
import subprocess
from datetime import datetime,timezone
from rsd_b000_contract import OUT,PART,LEGACY_OUT,digest,save_json

def main():
    if (OUT/'PIPELINE.json').exists():raise ValueError('pipeline already submitted; never silently duplicate jobs')
    contract=json.loads((OUT/'contract.json').read_text())
    audit=json.loads((OUT/'COVARIANCE_CORRECTION_AUDIT.json').read_text())
    tests=json.loads((OUT/'WORKFLOW_AUDIT.json').read_text())
    if audit['status']!='PASS' or tests['status']!='PASS_NUMERICAL_AND_PROGRAM_REGRESSION':raise ValueError('preflight not passed')
    cases=json.loads((OUT/'CASES.json').read_text())
    if len(cases)!=18 or contract['volume_Gpc_h3']!=1 or contract['covariance_multiplier']!=1:raise ValueError('wrong covariance normalization')
    old_files=[LEGACY_OUT/n for n in ('contract.json','RESULTS.json','RESULTS.csv','RSD_B000_VALIDATION.pdf','B000_ENSEMBLE_MATRIX.npz')]
    for name in cases:
        meta=json.loads((OUT/'cases'/name/'contract.json').read_text())
        if meta['n_target']!=100 or meta['volume_Gpc_h3']!=1 or meta['covariance_is_of_mean']:raise ValueError('case normalization drift')
        old_files.extend(LEGACY_OUT/'cases'/name/n for n in ('contract.json','likelihood.npz','SUMMARY.json','MAP.json','RUN_PROVENANCE.json'))
    frozen={str(p):digest(p) for p in old_files}
    logs=OUT/'logs';logs.mkdir(exist_ok=True);jobs={}
    def submit(key,mode,array=None,dependency=None):
        args=['sbatch','--parsable',f'--job-name=rsd_v1_{key}',f'--output={logs}/{key}_%A_%a.out',f'--error={logs}/{key}_%A_%a.err']
        if array is not None:args.append(f'--array={array}')
        if dependency is not None:args.append(f'--dependency=afterok:{dependency}')
        args.extend([str(PART/'code/rsd_b000_compute_job.sh'),mode])
        value=subprocess.check_output(args,text=True).strip().split(';')[0]
        if not value.isdigit():raise ValueError('ambiguous Slurm submission; inspect before retry')
        jobs[key]=value
        # 每成功一项立即留档；中途失败也不会失去已经提交的作业 ID。
        save_json(OUT/'PIPELINE.json',{'version':contract['version'],'submitted_UTC':datetime.now(timezone.utc).isoformat(),
                  'jobs':jobs,'status':'SUBMITTED_NOT_COMPLETED','baseline_cases':18,'sensitivity_max_cases':6,
                  'data_mean_count':100,'covariance_multiplier':1.,'volume_Gpc_h3':1.,
                  'old_output_immutable_hashes':frozen,'scope':'covariance-only correction; no new physical model'})
        print(key+' '+value,flush=True);return value
    baseline=submit('baseline18','mcmc','0-17')
    sensitivity=submit('sensitivity_freeze','sensitivity',dependency=baseline)
    chains=submit('sensitivity_up_to6','mcmc','18-23',dependency=sensitivity)
    submit('final_report','report',dependency=chains)

if __name__=='__main__':main()
