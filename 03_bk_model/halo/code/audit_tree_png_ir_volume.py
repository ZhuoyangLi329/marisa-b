#!/usr/bin/env python3
"""V=1 复算审计与体积对照；只生成 PDF 图。

模块大纲：冻结结果哈希与设计矩阵检查；协方差缩放和默认设置回归；
独立五维 Minuit 交叉检验；逐档位比较和 PDF 汇总。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
from iminuit import Minuit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_tree_png_ir_kmax_scan import OUT, Profile, polynomial, B1, VOLUME

def main():
    old=json.loads((OUT/'ir_kmax_scan_convergence.json').read_text())
    new=json.loads((OUT/'ir_kmax_scan_convergence_v1.json').read_text())
    low=dict(np.load(OUT/'ir_model_designs_convergence.npz'))
    high=dict(np.load(OUT/'ir_model_designs_convergence_v1.npz'))
    inputs=dict(np.load(OUT/'frozen_inputs.npz'))
    expected={
        'ir_kmax_scan_convergence.json':'22b4bd26fdf25dd95cd31c2328c787222617b4331cf757a2407169990a0045ce',
        'ir_kmax_scan_convergence.pdf':'c16e9a7988b9f1fbe150cd9a6d0c77ae90d000c46f77da90b3ac71b7887cf95b',
        'frozen_inputs.npz':'b3599ea7c696c241de0b8cfa9bae710b3a5d073b5259f62ac3bedec8b7a343d8',
    }
    frozen={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in expected}
    data={'volume_old':old['audit']['volume_gpc3_h3'],'volume_new':new['audit']['volume_gpc3_h3'],
          'frozen_hashes':frozen,'minuit_checks':[],
          'number_of_fits':len(new['rows']),
          'censored_rows':[{k:r[k] for k in ('bias','variant','kmax','truth','fNL','interval')}
                           for r in new['rows'] if r['interval_censored']],
          'invalid_optimizer_rows':[{k:r[k] for k in ('bias','variant','kmax','truth')}
                                    for r in new['rows'] if not r['optimizer_valid']],
          'decisions':{key:{'V42.875':old['summary'][key]['largest_contiguous_valid_kmax'],
                            'V1':value['largest_contiguous_valid_kmax']}
                       for key,value in new['summary'].items()}}
    mean=inputs['prior_mean']; whitener=inputs['prior_whitener']
    cases=[('strict_kir0.15',.08,-100.),('strict_kir0.15',.08,100.),
           ('fixed_order',.12,0.),('strict_kir0.15',.12,0.),
           ('fixed_order',.20,0.),('strict_kir0.15',.20,0.),('strict_kir0.15',.20,100.)]
    for variant,cut,truth in cases:
        row=next(r for r in new['rows'] if r['bias']=='current' and r['variant']==variant and r['kmax']==cut and r['truth']==truth)
        idx=np.array(row['indices']); n=len(idx)
        cov=np.cov(inputs['data_+0'][:,idx],rowvar=False,ddof=1)
        precision=(500-n-2)/499.*data['volume_new']*np.linalg.inv(cov)
        target=inputs[f'data_{truth:+.0f}'][:,idx].mean(axis=0)
        design=high['current__'+variant][:,idx]
        # 不复用解析 profile 的白化与最小二乘，独立优化全部五个参数。
        def objective(fNL,b2,gamma2,ashot,bshot):
            theta=np.array([b2,gamma2,ashot,bshot])
            residual=polynomial(design,fNL)@np.r_[1.,theta]-target
            prior=whitener@(np.r_[B1,theta]-mean)
            return float(residual@precision@residual+prior@prior)
        fit=Minuit(objective,truth,*mean[1:]);fit.errordef=1
        fit.limits['fNL']=(truth-200,truth+200)
        fit.errors=(50.,.1,.1,.1,.1);fit.tol=1.e-8;fit.strategy=2
        fit.migrad(ncall=100000);fit.hesse();fit.minos('fNL',ncall=100000)
        error=fit.merrors['fNL']
        interval=np.array([fit.values['fNL']+error.lower,fit.values['fNL']+error.upper])
        data['minuit_checks'].append({'variant':variant,'kmax':cut,'truth':truth,
            'valid':bool(fit.valid),'minos_valid':bool(error.is_valid),
            'delta_fNL':float(fit.values['fNL']-row['fNL']),
            'delta_objective':float(fit.fval-row['best_fit']['objective']),
            'max_interval_endpoint_difference':float(np.max(np.abs(interval-row['interval'])))})
    row=next(r for r in old['rows'] if r['bias']=='current' and r['variant']=='strict_kir0.15' and r['kmax']==.12 and r['truth']==0)
    idx=np.array(row['indices']);cov=np.cov(inputs['data_+0'][:,idx],rowvar=False,ddof=1)
    target=inputs['data_+0'][:,idx].mean(axis=0);design=high['current__strict_kir0.15'][:,idx]
    original=Profile(design,target,cov,mean,whitener)
    changed=Profile(design,target,cov,mean,whitener,volume=1)
    reference=original.fit(0.)
    data['default_volume_regression']={'delta_fNL':reference['fNL']-row['fNL'],
                                      'delta_objective':reference['best_fit']['objective']-row['best_fit']['objective']}
    data['whitened_design_scaling_error']=float(np.max(np.abs(changed.white*np.sqrt(VOLUME)-original.white)))
    data['prior_unchanged']=bool(np.array_equal(original.prior_matrix,changed.prior_matrix) and np.array_equal(original.prior_rhs,changed.prior_rhs))
    rejected=[]
    for invalid in (0.,-1.,np.nan,np.inf):
        try: Profile(design,target,cov,mean,whitener,volume=invalid)
        except ValueError: rejected.append(True)
        else: rejected.append(False)
    gates={
        'frozen_results_and_data_unchanged':frozen==expected,
        'all_designs_unchanged':set(low)==set(high) and all(np.array_equal(low[k],high[k]) for k in low),
        'correct_volume':data['volume_new']==1 and data['volume_old']==VOLUME,
        'whitening_scales_as_sqrt_volume':np.allclose(changed.white*np.sqrt(VOLUME),original.white,rtol=1.e-11,atol=1.e-9),
        'prior_unchanged':data['prior_unchanged'],
        'default_volume_regression':abs(data['default_volume_regression']['delta_fNL'])<1.e-4 and abs(data['default_volume_regression']['delta_objective'])<1.e-8,
        'invalid_volume_rejected':all(rejected),
        'minuit_and_minos_agree':all(r['valid'] and r['minos_valid'] and abs(r['delta_fNL'])<.02 and abs(r['delta_objective'])<1.e-5 and r['max_interval_endpoint_difference']<.05 for r in data['minuit_checks']),
        'all_intervals_uncensored':not data['censored_rows'],
        'all_optimizers_valid':not data['invalid_optimizer_rows'],
    }
    data['gates']={k:bool(v) for k,v in gates.items()}
    data['status']='pass' if all(gates.values()) else 'needs_review'
    data['current_p_truth0_comparison']=[]
    for variant in ('fixed_order','strict_kir0.15','factorized_kir0.15'):
        for row in new['rows']:
            if row['bias']!='current' or row['variant']!=variant or row['truth']!=0:continue
            ref=next(r for r in old['rows'] if r['bias']=='current' and r['variant']==variant and r['truth']==0 and r['kmax']==row['kmax'])
            data['current_p_truth0_comparison'].append({'variant':variant,'kmax':row['kmax'],
                'old_fNL':ref['fNL'],'new_fNL':row['fNL'],'old_interval':ref['interval'],'new_interval':row['interval'],
                'profile_halfwidth_ratio':(row['interval'][1]-row['interval'][0])/(ref['interval'][1]-ref['interval'][0])})
    # 本次变更单独留档，不修改原 V=42.875 的冻结 manifest。
    code=Path(__file__).resolve().parent
    source_files=[code/'run_tree_png_ir_kmax_scan.py',Path(__file__).resolve(),
                  code.parent/'work/tree_ir_v1_20260930_originals/run_tree_png_ir_kmax_scan.py']
    data['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    data['source_changes']={
        'run_tree_png_ir_kmax_scan.py':'新增显式 --volume 与 Profile(volume=...)；保留默认 42.875；PDF 标题读取实际体积；不修改模型或先验。',
        'audit_tree_png_ir_volume.py':'新增体积缩放、旧默认回归、独立 Minuit/MINOS、冻结输入/结果哈希与模型设计矩阵一致性检查及对照 PDF。',
    }
    data['rerun_arguments']='--templates tree_ir_templates_r6_m120_s0.25_rbao110.0.npz --tag convergence_v1 --volume 1 --variants fixed_order strict_kir0.15 factorized_kir0.15'
    (OUT/'IR_VOLUME_V1_AUDIT.json').write_text(json.dumps(data,indent=2)+'\n')
    # 同一真值的区间直接对照：变大的统计误差不等于模型绝对偏差消失。
    fig,axes=plt.subplots(2,2,figsize=(11,8.0))
    for i,(variant,label) in enumerate((('fixed_order','No IR'),('strict_kir0.15','Strict LO IR'))):
        for payload,color,volume in ((old,'#777777','42.875'),(new,'#d95f02','1')):
            result=payload['summary']['current__'+variant]['by_kmax']
            cuts=np.array([float(k) for k in result])
            axes[i,0].plot(cuts,[v['max_truth_delta_chi2'] for v in result.values()],'-o',color=color,label=f'V={volume}')
            rows=sorted([r for r in payload['rows'] if r['bias']=='current' and r['variant']==variant and r['truth']==0],key=lambda r:r['kmax'])
            x=np.array([r['kmax'] for r in rows]); y=np.array([r['fNL'] for r in rows]); ci=np.array([r['interval'] for r in rows])
            axes[i,1].plot(x,y,'-o',color=color,label=f'V={volume}')
            axes[i,1].fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.14)
        axes[i,0].axhline(1,color='black',ls='--',lw=.8)
        axes[i,0].set_yscale('log');axes[i,0].set_ylabel(r'max $\Delta\chi^2(f_{NL}^{true})$')
        axes[i,1].axhline(0,color='black',ls='--',lw=.8);axes[i,1].set_ylabel(r'recovered $f_{NL}$, true $f_{NL}=0$')
        for j in range(2):
            axes[i,j].set_title(label);axes[i,j].set_xlabel(r'$k_{max}$ [h/Mpc]')
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.suptitle('Pre-reconstruction B-only; current p calibration\nSame mean data and nuisance prior; volumes in (Gpc/h)^3; shaded profile intervals: Delta chi2 = 1')
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/'ir_volume_comparison_v1.pdf');plt.close(fig)
    print(json.dumps(data,indent=2),flush=True)

if __name__=='__main__':main()
