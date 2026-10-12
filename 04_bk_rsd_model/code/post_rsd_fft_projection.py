"""模块大纲：复用原 pre FFT/convolution；显式 tracer p；只增加固定 shift 因子。

旧 rsd_b000_fft_projection.py 不改，保持全部 pre frozen source SHA。
物理 bias family 沿用当前 RSD PBS 约定；这里只将 p 从 pre 常量显式化。
最终投影仍是两腿 registry/full-shell denominator，不是严格三腿 cutoff。
"""
from itertools import product
import numpy as np
from rsd_b000_fft_projection import Field,Fields,FFTProjection,integrate_design,A,D,ONE
from rsd_b000_contract import bias as pre_bias
from post_rsd_fixedrec import FixedReconstruction,pbs_bias


class _PostFactors:
    def configure_post(self,p_halo,reconstruction):
        if not np.isfinite(p_halo):raise ValueError('nonfinite physical p_halo')
        if reconstruction is not None and not isinstance(reconstruction,FixedReconstruction):
            raise TypeError('only fixed reconstruction operator is accepted')
        self.p_halo=float(p_halo);self.reconstruction=reconstruction
        self.z1=pbs_bias(self.b1,p_halo=self.p_halo).bphi*self.u
        self.shift_response=np.zeros_like(self.p)
        if reconstruction is not None:
            self.shift_response[self.active]=reconstruction.response(self.v[self.active],self.m[self.active])

    def degrees(self,field):
        return range(3) if field.root=='DR' else super().degrees(field)

    def values(self,field,degree):
        if field.root=='DR':
            base=Field('D',field.axes,field.inv2,field.invM)
            return super().values(base,degree)*self.shift_response
        return super().values(field,degree)


class PostFields(_PostFactors,Fields):
    def __init__(self,vectors,power,transfer,b1,f_growth,*,p_halo,reconstruction):
        super().__init__(vectors,power,transfer,b1,f_growth)
        self.configure_post(p_halo,reconstruction)


class PostFFTProjection(_PostFactors,FFTProjection):
    def __init__(self,meshsize,edges,pair,b1,f_growth,*,p_halo,reconstruction):
        super().__init__(meshsize,edges,pair,b1,f_growth)
        self.configure_post(p_halo,reconstruction)


def additional_terms(fields):
    """补偿旧 pre 源的 p 常量，并加入 Delta B；不复制已有 Z2/RSD source 表。"""
    old=pre_bias(fields.b1);new=pbs_bias(fields.b1,p_halo=fields.p_halo)
    dbphi=new.bphi-old.bphi;dbphidelta=new.bphidelta-old.bphidelta
    # 同一 PBS family 中 p 只改 bphi 与 bphidelta 的常数；b2 导数和 bphi2 不变。
    if new.bphi2!=old.bphi2:raise ValueError('PBS family mismatch')
    terms=[]
    for a in range(3):
        terms.append((dbphi,1,A.op((a,),invM=True),A.op((a,),inv2=True),ONE))
        terms.append((dbphi,1,A.op((a,),inv2=True),A.op((a,),invM=True),ONE))
    terms.extend(((dbphidelta,1,A.op(invM=True),A,ONE),
                  (dbphidelta,1,A,A.op(invM=True),ONE)))
    if fields.reconstruction is not None:
        # k_i+k_j=-k_ell；原负 shift 因此写成正的 closing-leg vector factor。
        for a in range(3):
            x=Field('DR').op((a,),inv2=True);closing=ONE.op((a,))
            terms.extend(((1.,0,x,D,closing),(1.,0,D,x,closing)))
    return terms


def integrate_post_design(fields):
    result=integrate_design(fields)
    for amplitude,shift,x,y,z in additional_terms(fields):
        if amplitude==0.:continue
        for dx,dy,dz in product(fields.degrees(x),fields.degrees(y),fields.degrees(z)):
            degree=shift+dx+dy+dz
            if degree<=2:
                result[0,degree]+=amplitude*fields.integral(x,dx,y,dy,z,dz)
    return result
