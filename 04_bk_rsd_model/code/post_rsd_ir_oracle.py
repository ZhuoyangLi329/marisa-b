"""模块大纲：复用 native source 的逐腿 wiggle 导数；RSD tensor 的 LO 单-wiggle dressing。

这是小批量确定性 oracle，不是 likelihood/生产投影；不补猜测的 stochastic。
strict O(Pw)、fNL<=2、Gaussian long displacements；固定算法 alpha_rec 外部表。
当前 source 对单腿 t=(P/Pnw)^1/2 至多二次，可用有限间隔精确提取导数；
四点 degree gate 随调用检查，避免未来 source 改变后静默使用错误的阶数。
"""
from types import SimpleNamespace
import numpy as np
from post_rsd_fixedrec import FixedPostRSD
from post_rsd_displacement import damping_coefficients


def source_leg_wiggles(vectors,power,power_nw,transfer,bias,rsd,reconstruction=None,*,engine=None,step=.25):
    """BnW 与三个逐腿 directional derivative，维度 [N,3legs,3fnl_coeff]。

    MnW=M sqrt(PnW/P) 固定 P_phi；改变该腿 t 时 PnW->PnW*t²、MnW->MnW*t。
    R_* 必须保持实际 catalog 的算法参数/谱表，不参与此源项 wiggle 导数。
    """
    v=np.asarray(vectors,dtype=float)
    if v.ndim==2:v=v[None]
    if v.ndim!=3 or v.shape[1:]!=(3,3) or not 0<len(v)<=64:raise ValueError('oracle batch must contain 1..64 triangles')
    p,pn,m=(np.array(np.broadcast_to(x,(len(v),3)),dtype=float) for x in (power,power_nw,transfer))
    if not all(np.isfinite(x).all() and np.all(x>0) for x in (p,pn,m)):raise ValueError('invalid physical spectra')
    if not 0<step<.49:raise ValueError('interpolation step must lie in (0,.49)')
    if reconstruction is not None and reconstruction.alpha_table is None:
        raise ValueError('IR source requires explicit frozen alpha_rec; varying physical M must not change catalog shift')
    mn=m*np.sqrt(pn/p);engine=engine or FixedPostRSD()
    base=engine.evaluate(v,pn,mn,bias,rsd,reconstruction)
    components=('total','gravitational_and_bias','primordial')
    wiggles={key:np.empty((len(v),3,3)) for key in components}
    max_degree_error=0.
    for leg in range(3):
        samples=[]
        for t in (1-step,1+step,1+2*step):
            pt=pn.copy();mt=mn.copy();pt[:,leg]*=t*t;mt[:,leg]*=t
            samples.append(engine.evaluate(v,pt,mt,bias,rsd,reconstruction))
        for key in components:
            f0=getattr(base,key);fm,fp,f2=(getattr(sample,key) for sample in samples)
            derivative=(fp-fm)/(2*step)
            quadratic=(fp+fm-2*f0)/(2*step**2)
            predicted=f0+2*step*derivative+(2*step)**2*quadratic
            scale=np.maximum(1.,np.maximum.reduce((np.abs(f0),np.abs(fm),np.abs(fp),np.abs(f2))))
            error=float(np.max(np.abs(f2-predicted)/scale));max_degree_error=max(max_degree_error,error)
            if error>3e-11:raise ValueError('native source is no longer degree<=2 in one-leg t: '+key)
            wiggles[key][:,leg]=derivative*((p[:,leg]-pn[:,leg])/(2*pn[:,leg]))[:,None]
    return SimpleNamespace(base=base,wiggles=wiggles,transfer_nw=mn,degree_error=max_degree_error,
                           physical_potential_unchanged=np.allclose(p/m**2,pn/mn**2,rtol=5e-15,atol=0.))


def dress_finite_png(base,wiggles,damping):
    """纯代数：先有向腿 dressing，再由调用方按原 estimator measure 投影。"""
    base=np.asarray(base,dtype=float);w=np.asarray(wiggles,dtype=float);d=np.asarray(damping,dtype=float)
    if w.shape!=d.shape or w.shape!=(len(base),3,3):raise ValueError('source/damping shape mismatch')
    mixed=w.copy()
    mixed[:,:,1]+=d[:,:,1]*w[:,:,0]
    mixed[:,:,2]+=d[:,:,1]*w[:,:,1]+d[:,:,2]*w[:,:,0]
    return base+np.sum(d[:,:,0,None]*mixed,axis=1)


def evaluate_ir(vectors,power,power_nw,transfer,bias,rsd,reconstruction,long_modes,*,r_bao=110.,shift_noise=None):
    """小型有向三角形 IR oracle，明确不返回可直接用于拟合的 stochastic。"""
    if not np.isfinite(r_bao) or r_bao<=0:raise ValueError('invalid BAO separation')
    v=np.asarray(vectors,dtype=float)
    if v.ndim==2:v=v[None]
    source=source_leg_wiggles(v,power,power_nw,transfer,bias,rsd,reconstruction)
    flat=v.reshape(-1,3);r=r_bao*flat/np.linalg.norm(flat,axis=1)[:,None]
    covariance=long_modes.relative_covariance(r,b1=bias.b1,bphi=bias.bphi,f_growth=rsd.f_growth,
                los=rsd.los,reconstruction=reconstruction,shift_noise=shift_noise)
    damping,exponent=damping_coefficients(flat,covariance);damping=damping.reshape(-1,3,3)
    results={key:dress_finite_png(getattr(source.base,key),source.wiggles[key],damping)
             for key in source.wiggles}
    return SimpleNamespace(**results,source=source,damping=damping,exponent=exponent.reshape(-1,3,3),
             covariance=covariance.reshape(-1,3,3,3,3),
             scope='deterministic LO strict O(Pw), fNL<=2, Gaussian long modes, unprojected oracle',
             stochastic_status='NOT_IMPLEMENTED_OR_VALIDATED_FOR_POST_WEIGHTED_FIELD',production_ready=False)
