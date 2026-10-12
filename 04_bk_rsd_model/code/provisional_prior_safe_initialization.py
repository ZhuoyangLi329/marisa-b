"""模块大纲：弱约束/边界MAP的prior-safe初始行者；不修改科学C或目标分布。

只调节可逆affine坐标的condition和起点；sampler moves、RNG seed、先验、
likelihood、burn/ESS/Rhat等全部不改。原Hessian proposal另外保留。
"""
import numpy as np


def bounded_affine_transform(original,bounds):
    t=np.asarray(original,dtype=float);b=np.asarray(bounds,dtype=float);width=b[:,1]-b[:,0]
    if t.shape!=(len(width),len(width)) or not np.isfinite(t).all() or np.any(width<=0):
        raise ValueError('finite square affine map and explicit prior intervals required')
    sigma=np.linalg.norm(t,axis=1)
    if np.any(sigma<=0):raise ValueError('singular initial proposal row')
    capped=np.clip(sigma,width/1000,width/20)
    return (t*(capped/sigma)[:,None]),dict(original_row_sigma=sigma.tolist(),
        adjusted_row_sigma=capped.tolist(),changed_rows=np.flatnonzero(capped!=sigma).tolist(),
        adjustment_is_only_coordinate_conditioning_not_data_covariance=True)


def physical_initial_cloud(center,transform,bounds,seed,walkers=32):
    c=np.asarray(center,dtype=float);t=np.asarray(transform,dtype=float);b=np.asarray(bounds,dtype=float)
    width=b[:,1]-b[:,0]
    if c.shape!=(len(width),) or np.any(c<b[:,0]) or np.any(c>b[:,1]):
        raise ValueError('MAP center must be within unchanged frozen priors')
    rng=np.random.default_rng(seed)
    sigma=np.clip(.2*np.linalg.norm(t,axis=1),width/1000,width/100)
    values=c+rng.normal(size=(walkers,len(c)))*sigma
    # 反射只用于初始点；正式MCMC仍用原strict prior拒绝，不裁剪/反射任何proposal。
    margin=width*1e-10;lo=b[:,0]+margin;hi=b[:,1]-margin;span=hi-lo
    phase=np.mod(values-lo,2*span)
    values=lo+np.where(phase<=span,phase,2*span-phase)
    if not np.all((values>b[:,0])&(values<b[:,1])):raise ValueError('prior-safe initialization failed')
    return values
