"""模块大纲：修正候选v1对outer零腿的错误假设；完整三种zero atom。

原fit44排除(0,0) shell pair，因此没有全三腿为零的项。本模块拒绝该未使用
组合，不更改原registry。physical zero-B公式继续复用v1，只有几何权重修正。
独立oracle从有界整数球做非循环卷积，完整分母含outer零模式。
"""
import numpy as np
from scipy.signal import fftconvolve
from rsd_b000_contract import load_spectrum


def project_zero_primitives_all_legs(fields):
    left,right=(np.asarray(fields.masks[s]) for s in fields.slots[:2])
    left0=int(left[0,0,0]);right0=int(right[0,0,0])
    if left0 and right0:
        raise ValueError('all-zero triangle is excluded by original44 registry; do not silently extrapolate')
    # k1=0、k2=0、k3=0三个互斥情形，非零的对腿都服从同一个zero-B公式。
    multiplicity=((left&right&fields.active).astype(float)
                  +left0*(right&fields.active)+right0*(left&fields.active))
    p=fields.p;mu=fields.mu2;u=fields.u;r=fields.shift_response
    spectra=np.array([p,p*mu,p*mu*mu,p*u,p*mu*u,p*u*u])
    plain=np.sum(spectra*multiplicity,axis=(1,2,3))/fields.denominator
    shifted=np.sum(spectra*(r*multiplicity),axis=(1,2,3))/fields.denominator
    return np.r_[plain,shifted,np.sum(r*multiplicity)/fields.denominator,
                 np.sum(multiplicity)/fields.denominator]


def independent_linear_oracle_all_legs(edges,pair,reconstruction):
    i,j=map(int,pair);kf=2*np.pi/1000.
    extent=int(np.ceil(max(edges[i+1],edges[j+1])/kf))+1
    axis=np.arange(-extent,extent+1)
    v=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)*kf
    k=np.sqrt(np.sum(v*v,axis=-1));active=k>0;canonical=k.copy()
    for edge in edges[1:-1]:
        canonical[np.abs(canonical-edge)<=64*np.finfo(float).eps*max(1.,abs(float(edge)))]=edge
    def shell(index):
        upper=canonical<edges[index+1]
        if index==len(edges)-2:
            upper=canonical<=edges[index+1]+32*np.finfo(float).eps*max(1.,abs(float(edges[index+1])))
        return (canonical>=edges[index])&upper
    left=shell(i);right=shell(j)
    left0=int(left[extent,extent,extent]);right0=int(right[extent,extent,extent])
    if left0 and right0:raise ValueError('original44 must not include both outer zero legs')
    den=int(left.sum())*int(right.sum());a=left&active;b=right&active
    na=int(a.sum());nb=int(b.sum())
    if den<=0:raise ValueError('empty original shell')
    raw=fftconvolve(a.astype(float),b.astype(float),mode='full');pairs=np.rint(raw)
    rounding=float(np.max(np.abs(raw-pairs)))
    if rounding>1e-5 or pairs.min()<0 or int(pairs.sum())!=na*nb:
        raise ValueError('independent integer nonzero linear convolution fails')
    del raw
    axis2=np.arange(-2*extent,2*extent+1)
    vv=np.stack(np.meshgrid(axis2,axis2,axis2,indexing='ij'),axis=-1)*kf
    kk=np.sqrt(np.sum(vv*vv,axis=-1));nz=kk>0
    spectra=np.zeros((6,)+kk.shape);p,m=load_spectrum()(kk[nz])
    mu=vv[...,2][nz]**2/kk[nz]**2;u=1/m
    spectra[:,nz]=np.array([p,p*mu,p*mu*mu,p*u,p*mu*u,p*u*u])
    first=spectra[:,extent:3*extent+1,extent:3*extent+1,extent:3*extent+1]
    multiplicity=a*(nb-b)+b*(na-a)
    first_sum=np.sum(first*multiplicity,axis=(1,2,3))
    center=2*extent;closing0=int(pairs[center,center,center]);pairs[center,center,center]=0.
    closure_sum=np.sum(spectra*pairs,axis=(1,2,3))
    intrinsic=np.r_[(first_sum+closure_sum)/den,(na*nb-closing0)/den]
    zweight=(a&b).astype(float)+left0*b+right0*a
    selected=zweight>0;rs=reconstruction.response(v[selected],np.ones(int(selected.sum())))
    weight=zweight[selected];small=first[:,selected]
    zero=np.r_[(small*weight).sum(axis=1)/den,(small*(weight*rs)).sum(axis=1)/den,
               np.sum(weight*rs)/den,weight.sum()/den]
    nz_zero=closing0+left0*nb+right0*na
    if int(zweight.sum())!=nz_zero or nz_zero+(na*nb-closing0)!=den:
        raise ValueError('all original ordered pairs must be exactly partitioned')
    return intrinsic,zero,dict(full_denominator=den,nonzero_input_pairs=na*nb,
        nonzero_closing_pairs=na*nb-closing0,zero_closing_count=closing0,
        zero_first_leg_count=left0*nb,zero_second_leg_count=right0*na,
        total_zero_atoms=nz_zero,maximum_integer_roundoff=rounding,bounded_input_extent=extent)
