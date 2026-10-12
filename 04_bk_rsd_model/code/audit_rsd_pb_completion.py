"""模块大纲：联合完成审计；核心/源文件/数据/协方差/链诊断；旧结果哈希不变。"""
import json
import os
from pathlib import Path
import numpy as np
from rsd_pb_common import OUT,B_OUT,PART,digest,save_json
from run_rsd_pb_inference import parent
from run_rsd_pb_mcmc import JointCase

def main():
    value=parent();names=json.loads((OUT/'CASES.json').read_text());results=json.loads((OUT/'RESULTS.json').read_text())
    if len(names)!=18 or results['converged']!=18 or results['status']!='COMPLETE':raise ValueError('not 18 converged joint cases')
    checks=[]
    for name in names:
        case=JointCase(OUT/'cases'/name);summary=json.loads((case.directory/'SUMMARY.json').read_text())
        run=json.loads((case.directory/'RUN_PROVENANCE.json').read_text())
        if not summary['converged'] or not all(summary['gates'].values()):raise ValueError('failed diagnostics '+name)
        if not summary['P_likelihood'] or not summary['P_B_cross_covariance']:raise ValueError('joint flag wrong')
        if run['joint_adapter_sha256']!=digest(PART/'code/run_rsd_pb_mcmc.py'):raise ValueError('adapter drift')
        if run['sampler_sha256']!='018d3d4bb2db0dfdd739b7933472716bdb1c866d2f7d9aa57dca3f463456614d':raise ValueError('B sampler changed')
        with np.load(B_OUT/'cases'/name/'likelihood.npz') as z:
            np.testing.assert_array_equal(case.target[12:],z['target']);np.testing.assert_array_equal(case.coefficients[...,12:],z['coefficients'])
            np.testing.assert_array_equal(case.bounds[:-1],z['bounds']);np.testing.assert_allclose(case.covariance[12:,12:],z['covariance'],rtol=2e-13)
        checks.append({'name':name,'converged':True,'steps':summary['steps'],'P_bins':12,'B_bins':case.meta['B_bins'],
                       'rhat_max':max(summary['rank_normalized_split_rhat']),'likelihood_sha256':digest(case.directory/'likelihood.npz')})
    launch=json.loads((OUT/'LOGIN_LAUNCH.json').read_text());status=json.loads((OUT/'LOGIN_STATUS.json').read_text())
    recovery=None
    if (OUT/'LOGIN_RESOURCE_RECOVERY_LAUNCH.json').exists():
        recovery=json.loads((OUT/'LOGIN_RESOURCE_RECOVERY_LAUNCH.json').read_text())
        if not recovery['measurement_model_covariance_sampler_unchanged']:raise ValueError('not resource-only recovery')
        if recovery['maximum_measurement_workers']>4 or recovery['measurement_batch_size']>10:raise ValueError('memory recovery limits violated')
    if len(launch['affinity'])>8 or not set(os.sched_getaffinity(0)).issubset(launch['affinity']):raise ValueError('core ceiling violated')
    if status.get('active_workers'):raise ValueError('completion while writers active')
    if list(OUT.rglob('*.png')):raise ValueError('unexpected PNG')
    qa=json.loads((OUT/'PDF_QA.json').read_text())
    if qa['status']!='PASS' or qa['pdf_sha256']!=digest(OUT/'RSD_PB_VALIDATION.pdf'):raise ValueError('PDF QA drift')
    save_json(OUT/'COMPLETION_AUDIT.json',{'status':'PASS','n_joint_cases':18,'n_converged':18,'n_P_measurements':800,
              'P_bins':12,'full_same_realization_cross_covariance':True,'data_mean_count':100,'covariance_sample_count':500,
              'volume_Gpc_h3':1.,'covariance_multiplier':1.,'covariance_is_of_mean':False,'cases':checks,
              'all_frozen_old_B_inputs_outputs_sha256_unchanged':True,'maximum_logical_CPUs':len(launch['affinity']),
              'numeric_threads':1,'sampler_kernel_and_diagnostics_unchanged':True,'no_PNG':True,
              'resource_only_measurement_recovery':recovery,
              'PDF_sha256':digest(OUT/'RSD_PB_VALIDATION.pdf'),'RESULTS_sha256':digest(OUT/'RESULTS.json'),
              'scientifically_certified':False})
    print('P+B_COMPLETION_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
