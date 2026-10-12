"""模块大纲：同似然有界Pshot解析积分，独立核对新p校准和P0后验。

不改MCMC、固定p、先验或C；只比较积分和抽样，避免将很小p差别造成的
真实变化与有限MCMC误差混为一谈。只读取已完成case摘要，不打开HDF。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import numpy as np
import run_postP_calibrated_rsd_v1 as run
from rsd_p0_analytic_grid import posterior_grid

def compute(path,mode,dest):
    meta=run.read(path/'contract.json'); data=run.arrays(path/'likelihood.npz')
    if run.digest(path/'likelihood.npz')!=meta['likelihood_sha256']: raise ValueError('输入改变')
    unit=1. if mode=='p_calibration' else float(data['noise'][0])
    args=dict(target=data['target'],cholesky=data['chol'],templates=data['templates'],mode=mode,
        first_bounds=data['bounds'][0],b1_bounds=data['bounds'][1],shot_bounds=data['bounds'][2]*unit,
        fixed_physical_p=meta.get('p_fixed_physical'))
    coarse=posterior_grid(**args,n_first=401,n_b1=501)
    fine=posterior_grid(**args,n_first=801,n_b1=1001)
    delta=float(np.max(abs(np.asarray(fine['quantiles'])-coarse['quantiles'])))
    if delta>(.0001 if mode=='p_calibration' else .05): raise ValueError('原积分分辨率检查失败')
    s=run.read(path/'SUMMARY.json')
    if not s['converged']: raise ValueError('只核对已完成后验')
    median=float(s['quantiles_2p5_16_50_84_97p5'][2][0])
    np.savez_compressed(dest/(path.name+'_MARGINAL.npz'),x=fine['x'],pdf=fine['pdf'],cdf=fine['cdf'])
    return dict(name=path.name,mode=mode,grid_quantiles=fine['quantiles'],grid_sigma=fine['sigma'],
        MCMC_median=median,MCMC_minus_grid_median=median-fine['quantiles'][2],
        coarse_fine_max_quantile_difference=delta,grid_resolution=[801,1001],
        likelihood_sha256=run.digest(path/'likelihood.npz'),summary_sha256=run.digest(path/'SUMMARY.json'))

def main():
    run.guard(); dest=run.OUT/'analytic_grid_v1'; dest.mkdir(exist_ok=False)
    for i in range(4):
        if not run.read(run.directory(i)/'COMPLETION_AUDIT.json')['converged']: raise ValueError('先等P0完成')
    checks=[compute(run.directory(0),'p_calibration',dest)]; pairs=[]
    for i,truth in enumerate(run.TRUTHS,1):
        new=compute(run.directory(i),'fNL',dest)
        old=compute(run.OLD/'post_P0_only_all_available_p/cases'/(truth+'_post_P0_all_available_p'),'fNL',dest)
        checks.extend([new,old]); pairs.append(dict(truth=truth,
            grid_new_minus_old_median=new['grid_quantiles'][2]-old['grid_quantiles'][2],
            grid_new_over_old_sigma=new['grid_sigma']/old['grid_sigma'],
            new_MCMC_minus_grid_median=new['MCMC_minus_grid_median']))
    result=dict(status='SAME_LIKELIHOOD_BOUNDED_SHOT_INDEPENDENT_GRID_COMPLETE',checks=checks,pairs=pairs,
        inputs_priors_covariance_and_MCMC_unmodified=True,HDF_read=False,
        source_sha256={str(p):run.digest(p) for p in [Path(__file__),Path(__file__).with_name('rsd_p0_analytic_grid.py')]})
    run.save_json(dest/'AUDIT.json',result); print(result,flush=True)

if __name__=='__main__': main()
