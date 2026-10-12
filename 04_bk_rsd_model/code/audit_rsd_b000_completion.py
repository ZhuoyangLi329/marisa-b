"""模块大纲：新链收敛/来源核验；未平均 covariance 与旧输出不可变性；完成报告。

只读旧输出与现有链，不重跑模型、不调先验；审计 JSON 只写当前输出目录。
"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from rsd_b000_contract import OUT,PART,digest,save_json
from run_rsd_b000_mcmc import compatible_sampler_hashes

def main():
    parent=json.loads((OUT/'contract.json').read_text());pipeline=json.loads((OUT/'PIPELINE.json').read_text())
    result=json.loads((OUT/'RESULTS.json').read_text());cases=json.loads((OUT/'CASES.json').read_text())
    if result['status']!='COMPLETE_WITH_DIAGNOSTICS':raise ValueError('results not complete')
    if digest(parent['source_ensemble_path'])!=parent['source_ensemble_sha256']:raise ValueError('source ensemble changed')
    for path,expected in pipeline['old_output_immutable_hashes'].items():
        if digest(path)!=expected:raise ValueError('legacy output changed: '+path)
    sampler_hash=digest(PART/'code/run_rsd_b000_mcmc.py')
    allowed_sampler_hashes=compatible_sampler_hashes()
    diag_hash=digest(PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py')
    with np.load(parent['source_ensemble_path']) as z:data=z['B'];cov=z['covariance_single_box']
    np.testing.assert_allclose(cov,np.cov(data[300:],rowvar=False,ddof=1),rtol=2e-13,atol=0)
    records=[]
    for name in cases:
        root=OUT/'cases'/name;meta=json.loads((root/'contract.json').read_text())
        summary=json.loads((root/'SUMMARY.json').read_text());provenance=json.loads((root/'RUN_PROVENANCE.json').read_text())
        if not summary['converged']:raise ValueError('chain did not converge: '+name)
        if meta['volume_Gpc_h3']!=1 or meta['n_target']!=100 or meta['covariance_multiplier']!=1 or meta['covariance_is_of_mean']:
            raise ValueError('wrong data/covariance contract: '+name)
        likelihood=digest(root/'likelihood.npz')
        if likelihood!=meta['likelihood_sha256'] or likelihood!=summary['likelihood_sha256'] or likelihood!=provenance['likelihood_sha256']:
            raise ValueError('likelihood hash drift: '+name)
        if provenance['sampler_sha256'] not in allowed_sampler_hashes or provenance['diagnostics_sha256']!=diag_hash:
            raise ValueError('sampler/diagnostic drift: '+name)
        extension_path=root/'BUDGET_EXTENSION_PROVENANCE.json';extensions=[]
        if extension_path.exists():
            extensions=json.loads(extension_path.read_text());latest=extensions[-1]['continuation']
            if latest['sampler_sha256']!=sampler_hash or latest['likelihood_sha256']!=likelihood or latest['diagnostics_sha256']!=diag_hash:
                raise ValueError('continuation source drift: '+name)
            if summary['steps']>latest['max_steps']:raise ValueError('continuation exceeds recorded budget')
        with np.load(root/'likelihood.npz') as z:
            idx=z['indices'];j=['fiducial','LC_p','LC_m'].index(meta['cosmology'])
            np.testing.assert_array_equal(z['covariance'],cov[np.ix_(idx,idx)])
            np.testing.assert_array_equal(z['target'],data[j*100:(j+1)*100].mean(axis=0)[idx])
            np.testing.assert_allclose(z['chol']@z['chol'].T,z['covariance']/meta['Hartlap_factor'],rtol=2e-13)
        records.append({'name':name,'all_convergence_gates':summary['gates'],'steps':summary['steps'],
                        'likelihood_sha256':likelihood,'single_box_covariance_verified':True,'mean_data_verified':True,
                        'initial_sampler_sha256':provenance['sampler_sha256'],'budget_extensions':extensions})
    if digest(OUT/'RSD_B000_VALIDATION.pdf')!=result['pdf_sha256']:raise ValueError('PDF digest mismatch')
    qa=json.loads((OUT/'PDF_QA.json').read_text())
    if qa['status']!='PASS_VECTOR_BOUNDS' or len(qa['pages'])!=8:raise ValueError('PDF QA failed')
    tests=json.loads((OUT/'WORKFLOW_AUDIT.json').read_text())
    login=json.loads((OUT/'LOGIN_LAUNCH.json').read_text());runtime=json.loads((OUT/'LOGIN_STATUS.json').read_text())
    if login['maximum_logical_CPUs']>8 or runtime['maximum_logical_CPUs']>8 or login['threads_per_numeric_library']!=1:
        raise ValueError('login CPU limit contract violated')
    if login['runner_source_sha256']!=digest(PART/'code/run_rsd_b000_login.py'):raise ValueError('login runner source changed')
    value={'status':'PASS','generated_UTC':datetime.now(timezone.utc).isoformat(),'baseline_cases':result['baseline_cases'],
           'sensitivity_cases':result['sensitivity_cases'],'all_converged':True,'cases':records,
           'tests_run':tests['tests_run'],'old_outputs_immutable_files_verified':len(pipeline['old_output_immutable_hashes']),
           'covariance_multiplier':1.,'volume_Gpc_h3':1.,'data_mean_count':100,'sampler_sha256':sampler_hash,
           'diagnostics_sha256':diag_hash,'pdf_sha256':result['pdf_sha256'],'results_sha256':digest(OUT/'RESULTS.json'),
           'audit_source_sha256':digest(Path(__file__)),'no_new_physical_model':True,'no_PNG':True,
           'login_host':login['host'],'maximum_logical_CPUs':login['maximum_logical_CPUs'],
           'CPU_affinity':login['affinity'],'threads_per_numeric_library':1,'CPU_affinity_guard_source_verified':True}
    save_json(OUT/'COMPLETION_AUDIT.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='cases'},indent=2),flush=True)

if __name__=='__main__':main()
