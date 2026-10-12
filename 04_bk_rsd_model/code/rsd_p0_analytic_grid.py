"""模块大纲：独立解析积分常数Pshot，核验三参数校准/P0-only边缘后验。

不改变任何MCMC或协方差；严格积分原均匀先验上下限，不另加b1/p先验。
三维后验降至二维积分只用于诊断，不能拿理想模型中位数反调实际p。
"""
import numpy as np
from scipy.special import log_ndtr
from rsd_b000_contract import F_GROWTH

def shot_marginal_loglike(predictions,target,cholesky,shot_bounds):
    values=np.asarray(predictions,dtype=float);target=np.asarray(target,dtype=float)
    chol=np.asarray(cholesky,dtype=float);lo,hi=map(float,shot_bounds)
    if values.ndim!=2 or values.shape[1]!=len(target) or not lo<hi:raise ValueError('finite rows and ordered physical shot limits required')
    one=np.linalg.solve(chol,np.ones(len(target)));a=float(one@one)
    white=np.linalg.solve(chol,(values-target).T).T;b=white@one
    orthogonal=white-(b/a)[:,None]*one
    score=np.sum(orthogonal*orthogonal,axis=1)
    lower=(lo+b/a)*np.sqrt(a);upper=(hi+b/a)*np.sqrt(a)
    # 上尾不能用两个接近1的CDF相减；下尾与跨零使用logCDF差。
    upper_tail=lower>0
    large=np.where(upper_tail,log_ndtr(-lower),log_ndtr(upper))
    small=np.where(upper_tail,log_ndtr(-upper),log_ndtr(lower))
    with np.errstate(divide='ignore',invalid='ignore'):
        log_mass=large+np.log(-np.expm1(small-large))
    return -.5*score+.5*np.log(2*np.pi/a)+log_mass

def marginal_pdf(x,b,logweights):
    weights=np.exp(logweights-np.max(logweights));density=np.trapezoid(weights,b,axis=1)
    norm=float(np.trapezoid(density,x))
    if not np.isfinite(norm) or norm<=0:raise ValueError('posterior grid has no finite normalization')
    pdf=density/norm
    cdf=np.r_[0,np.cumsum((pdf[:-1]+pdf[1:])*.5*np.diff(x))]
    q=np.interp([.025,.16,.5,.84,.975],cdf/cdf[-1],x)
    mean=float(np.trapezoid(pdf*x,x));sigma=float(np.sqrt(np.trapezoid(pdf*(x-mean)**2,x)))
    return dict(quantiles=q.tolist(),mean=mean,sigma=sigma,pdf=pdf,cdf=cdf)

def posterior_grid(target,cholesky,templates,*,mode,first_bounds,b1_bounds,shot_bounds,
                   fixed_physical_p=None,n_first=601,n_b1=701):
    x=np.linspace(*first_bounds,n_first);b=np.linspace(*b1_bounds,n_b1)
    t=np.asarray(templates,dtype=float);logw=np.empty((len(x),len(b)))
    if t.shape!=(6,len(target)) or mode not in ('p_calibration','fNL'):raise ValueError('six matching P templates and recognized grid mode required')
    for start in range(0,len(x),12):
        xx,bb=np.meshgrid(x[start:start+12],b,indexing='ij')
        if mode=='p_calibration':p=xx;fnl=np.full(xx.shape,100.)
        else:
            if fixed_physical_p is None:raise ValueError('explicit calibrated physical p required')
            p=np.full(xx.shape,fixed_physical_p);fnl=xx
        bp=2*1.686*(bb-p);bb=bb.ravel();lam=(fnl*bp).ravel()
        values=(bb[:,None]**2*t[0]+2*bb[:,None]*F_GROWTH*t[1]+F_GROWTH**2*t[2]
            +2*lam[:,None]*(bb[:,None]*t[3]+F_GROWTH*t[4])+lam[:,None]**2*t[5])
        logw[start:start+12]=shot_marginal_loglike(values,target,cholesky,shot_bounds).reshape(xx.shape)
    result=marginal_pdf(x,b,logw);result.update(x=x,b1=b,n_first=n_first,n_b1=n_b1)
    return result
