"""模块大纲：九组采样中的三组P0完成后，独立积分两种固定p的后验。

严格使用相同data/C/先验；避免把有限MCMC差异全归因于p。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import run_realspace_p_rsd_sensitivity_v1 as run
from audit_postP_calibrated_grid_v1 import compute
from pathlib import Path

def main():
    c=run.core;c.guard();dest=run.OUT/'analytic_P0_grid_v1';dest.mkdir(exist_ok=False)
    pairs=[];checks=[]
    for index in range(1,4):
        if not c.read(run.directory(index)/'COMPLETION_AUDIT.json')['converged']:raise ValueError('先等P0完成')
        new=compute(run.directory(index),'fNL',dest)
        old=compute(run.BASE/'cases'/run.BASE_NAMES[index],'fNL',dest)
        checks += [new,old]
        pairs.append(dict(truth_fNL=c.SPECS[index][1],
            grid_new_minus_old_median=new['grid_quantiles'][2]-old['grid_quantiles'][2],
            grid_new_over_old_sigma=new['grid_sigma']/old['grid_sigma'],
            new_MCMC_minus_grid_median=new['MCMC_minus_grid_median']))
    sources=[Path(__file__),Path(__file__).with_name('audit_postP_calibrated_grid_v1.py'),
        Path(__file__).with_name('rsd_p0_analytic_grid.py')]
    result=dict(status='SAME_DATA_C_PRIOR_P0_GRID_COMPARISON_COMPLETE',checks=checks,pairs=pairs,
        fixed_p=run.PHYSICAL_P,baseline_p=c.read(run.OUT/'P_CHOICE.json')['baseline_RSD_p'],
        source_sha256={str(p):c.digest(p) for p in sources},data_C_priors_and_MCMC_unmodified=True,HDF_read=False)
    c.save_json(dest/'AUDIT.json',result);print(result,flush=True)

if __name__=='__main__':main()
