"""模块大纲：实际p/P0先验后验的独立二维积分，以及理想模型中位数偏移诊断。

不打开HDF、不改变p/主C/先验；仅已完成case才比较MCMC摘要。
理想模型检查展示边缘中位数和MAP/真值不同的可能，不能据此调实际p。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from rsd_p0_analytic_grid import posterior_grid
from rsd_b000_contract import digest,save_json
from audit_all_available_rsd_pcal_v1 import direct_p
import run_all_available_rsd_pcal_v1 as calibration

OUT=calibration.OUT/'analytic_pcal_p0_grid_v1'
def read(path):return json.loads(Path(path).read_text())

def compute(d,mode):
    meta=read(d/'contract.json')
    if digest(d/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('actual numerical-grid input changed')
    with np.load(d/'likelihood.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    unit=1. if mode=='p_calibration' else float(arrays['noise'][0])
    args=dict(target=arrays['target'],cholesky=arrays['chol'],templates=arrays['templates'],mode=mode,
        first_bounds=arrays['bounds'][0],b1_bounds=arrays['bounds'][1],shot_bounds=arrays['bounds'][2]*unit,
        fixed_physical_p=meta.get('p_fixed_physical'))
    coarse=posterior_grid(**args,n_first=401,n_b1=501);fine=posterior_grid(**args,n_first=801,n_b1=1001)
    delta=float(np.max(abs(np.asarray(fine['quantiles'])-coarse['quantiles'])))
    # 格点收敛阈值只诊断积分，不修改任何MCMC后验或统计精度。
    tolerance=.0001 if mode=='p_calibration' else .05
    if delta>tolerance:raise ValueError('independent posterior grid needs resolution refinement: '+str(delta))
    np.savez_compressed(OUT/(d.name+'_MARGINAL.npz'),x=fine['x'],pdf=fine['pdf'],cdf=fine['cdf'])
    result=dict(name=d.name,mode=mode,quantiles=fine['quantiles'],sigma=fine['sigma'],mean=fine['mean'],
        grid_resolution=[801,1001],coarse_fine_quantile_max_difference=delta,
        input_sha256=digest(d/'likelihood.npz'),same_covariance_and_bounds_as_MCMC=True)
    return result,args

def main():
    if OUT.exists():raise ValueError('preserve existing analytic grid results; no overwrite')
    OUT.mkdir();checks=[];synthetic=[];sources=[Path(__file__),Path(__file__).with_name('rsd_p0_analytic_grid.py')]
    state=read(calibration.OUT/'P0_RECALIBRATED_STATUS.json')
    for stage in ('pre','post'):
        d=calibration.stage_root(stage)/'cases'/calibration.case_name(stage)
        proof=read(calibration.stage_root(stage)/'COMPLETION_AUDIT.json')
        if not proof['converged']:raise ValueError('actual physical-p chain not converged')
        result,args=compute(d,'p_calibration');stored=read(d/'SUMMARY.json')
        q=np.asarray(stored['quantiles_2p5_16_50_84_97p5'])[:,0]
        result.update(MCMC_quantiles=q.tolist(),MCMC_grid_median_difference=float(q[2]-result['quantiles'][2]),
            calibration_actual_count=proof['calibration_target_count'])
        checks.append(result);sources.extend([d/'contract.json',d/'likelihood.npz',d/'SUMMARY.json'])
        for truth in ('fiducial','LC_p','LC_m'):
            d=calibration.OUT/(stage+'_P0_only_all_available_p')/'cases'/(truth+'_'+stage+'_P0_all_available_p')
            result,args=compute(d,'fNL')
            if d.name in state['completed_cases']:
                stored=read(d/'SUMMARY.json');q=np.asarray(stored['quantiles_2p5_16_50_84_97p5'])[:,0]
                result.update(MCMC_quantiles=q.tolist(),MCMC_grid_median_difference=float(q[2]-result['quantiles'][2]),
                    MCMC_actual_case_completed_at_snapshot=True)
                sources.append(d/'SUMMARY.json')
            else:result['MCMC_actual_case_completed_at_snapshot']=False
            checks.append(result);sources.extend([d/'contract.json',d/'likelihood.npz'])
            if stage=='post' and truth=='LC_p':
                # 同一C/先验/新p，使用精确模型生成理想数据；不是伪造实际观测结果。
                p=args['fixed_physical_p'];b=2.23
                for fnl in (0.,100.,-100.):
                    data=direct_p(fnl,b,p,args['templates'])
                    ideal=posterior_grid(**dict(args,target=data),n_first=801,n_b1=1001)
                    synthetic.append(dict(model_generated_not_real_data=True,truth_fNL=fnl,b1=b,physical_p=p,
                        exact_model_at_true_parameters_has_zero_residual=True,
                        marginal_fNL_median=ideal['quantiles'][2],marginal_fNL_sigma=ideal['sigma'],
                        priors_and_C_single_identical_to_main_post_P0=True))
    report=dict(status='INDEPENDENT_BOUNDED_SHOT_POSTERIOR_GRID_AND_IDEAL_MODEL_MEDIAN_DIAGNOSTIC',
        created_UTC=calibration.stamp(),checks=checks,ideal_model_controls=synthetic,
        actual_p_fNL_data_covariance_or_priors_modified=False,HDF_read=False,
        no_p_adjustment_to_force_recovery=True,calibration_covariance_is_of_mean=False,
        sources_sha256={str(p):digest(p) for p in sources})
    save_json(OUT/'AUDIT.json',report)
    print(json.dumps(dict(status=report['status'],medians=[dict(name=x['name'],grid=x['quantiles'][2],
        MCMC_difference=x.get('MCMC_grid_median_difference')) for x in checks],ideal_model_controls=synthetic)),flush=True)

if __name__=='__main__':main()
