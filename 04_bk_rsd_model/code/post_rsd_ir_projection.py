"""模块大纲：tagged 单-wiggle Fourier 因子；复用唯一 post source/FFT 的 IR 投影。

先对各有向 Fourier leg 作有限-PNG damping mixing，再用原两腿 estimator 投影。
显式只取一个 wiggle，保留 full-shell denominator；不增加第三腿 cut。
当前共享 FFT source 的 LOS 固定 z。此入口不提供未经验证的 stochastic。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_fft_projection import PostFields,PostFFTProjection,integrate_post_design
from post_rsd_fixedrec import pbs_bias
from post_rsd_ir_harmonics import HarmonicDisplacement,reconstruction_signature


@dataclass(frozen=True)
class TaggedField:
    field: object
    wiggle: bool=False
    @property
    def root(self):return self.field.root
    @property
    def axes(self):return self.field.axes


def spectral_t_power(field,degree):
    """P->t²P、M->tM 下的因子幂次；R_* 为外部冻结量。"""
    root=field.root
    if root in ('A','D','DR'):power=2-degree
    elif root=='T':power=1-degree
    elif root in ('P','Pmu2','Pmu4'):power=2
    elif root in ('one','mu2','kz'):power=0
    else:raise ValueError('unknown source factor: '+root)
    return power-int(field.invM)


class _IRFactorMixin:
    def configure_ir(self,power_nw,damping):
        if self.reconstruction is not None and self.reconstruction.alpha_table is None:
            raise ValueError('IR projection needs explicit frozen reconstruction alpha table')
        physical_p=self.p;physical_m=self.m
        pn=np.asarray(np.broadcast_to(power_nw,physical_p.shape),dtype=float)
        if not np.isfinite(pn).all() or np.any(pn[self.active]<=0):raise ValueError('invalid no-wiggle power')
        self.relative_wiggle=np.divide(physical_p-pn,2*pn,out=np.zeros_like(pn),where=self.active)
        self.p=np.where(self.active,pn,0.)
        self.m=physical_m.copy();self.m[self.active]*=np.sqrt(pn[self.active]/physical_p[self.active])
        self.u=np.divide(1.,self.m,out=np.zeros_like(self.m),where=self.active)
        self.configure_post(self.p_halo,self.reconstruction)
        self.ir_damping=np.asarray(damping,dtype=float)
        if self.ir_damping.shape!=self.p.shape+(3,) or not np.isfinite(self.ir_damping).all():raise ValueError('damping shape/value mismatch')
        if np.any(self.ir_damping[...,0]<0) or np.any(self.ir_damping[...,0]>1+1e-13):raise ValueError('invalid Gaussian damping')
        self._zero_ir=np.zeros_like(self.p)
        self._any_wiggle=bool(np.any(self.relative_wiggle))
        self._damping_orders=(True,bool(np.any(self.ir_damping[...,1])),bool(np.any(self.ir_damping[...,2])))

    def degrees(self,field):
        field=field.field if isinstance(field,TaggedField) else field
        return range(3) if field.root in ('A','D','DR','T','P','Pmu2','Pmu4') else super().degrees(field)

    def _bare(self,field,degree):
        if degree not in super().degrees(field):return self._zero_ir
        return super().values(field,degree)

    def values(self,field,degree):
        if not isinstance(field,TaggedField):return self._bare(field,degree)
        original=field.field
        if not field.wiggle:return self._bare(original,degree)
        value=np.zeros_like(self.p)
        for source_degree in range(degree+1):
            if source_degree not in super().degrees(original):continue
            index=degree-source_degree
            # damping 存储 D0,d1,d2，而不是 D0,D0*d1,D0*d2。
            multiplier=1. if index==0 else self.ir_damping[...,index]
            value+=multiplier*spectral_t_power(original,source_degree)*self._bare(original,source_degree)
        return value*self.relative_wiggle*self.ir_damping[...,0]

    def integral(self,x,dx,y,dy,z,dz):
        fields=[x,y,z];degrees=[dx,dy,dz];result=0.
        # 调用原 point/FFT contraction 四次：nw + 每次仅一条腿带 wiggle。
        # 不复制 FFT、parity、cyclic sum、mask 或 normalization 的数值实现。
        for changed in (-1,0,1,2):
            if changed>=0 and not self._any_wiggle:continue
            possible=True
            for i,(field,degree) in enumerate(zip(fields,degrees)):
                bare_degrees=super().degrees(field)
                if i!=changed:
                    if degree not in bare_degrees:possible=False;break
                elif not any(d<=degree and self._damping_orders[degree-d] and spectral_t_power(field,d)!=0 for d in bare_degrees):
                    possible=False;break
            if not possible:continue
            arguments=[]
            for i,(field,degree) in enumerate(zip(fields,degrees)):
                arguments.extend((TaggedField(field,i==changed),degree))
            result+=super().integral(*arguments)
        return result

    def stochastic(self):
        raise NotImplementedError('post weighted stochastic IR has not been validated; legacy pre bases cannot be reused silently')


class IRPointFields(_IRFactorMixin,PostFields):
    def __init__(self,vectors,power,power_nw,transfer,b1,f_growth,*,p_halo,reconstruction,damping):
        super().__init__(vectors,power,transfer,b1,f_growth,p_halo=p_halo,reconstruction=reconstruction)
        self.configure_ir(power_nw,damping)


class IRFFTProjection(_IRFactorMixin,PostFFTProjection):
    def __init__(self,meshsize,edges,pair,b1,f_growth,*,p_halo,reconstruction,power_nw,harmonics):
        if not isinstance(harmonics,HarmonicDisplacement):raise TypeError('explicit parameter-tagged harmonic object required')
        expected={'b1':float(b1),'bphi':pbs_bias(b1,p_halo=p_halo).bphi,'f_growth':float(f_growth)}
        for key,value in expected.items():
            if not np.isclose(harmonics.physical_parameters[key],value,rtol=2e-14,atol=2e-14):raise ValueError('damping parameter mismatch: '+key)
        n=np.asarray(harmonics.physical_parameters['los']);n=n/np.linalg.norm(n)
        if not np.allclose(np.abs(n),[0,0,1],rtol=0,atol=1e-14):raise ValueError('shared projection currently requires LOS=z')
        if harmonics.physical_parameters['reconstruction_signature']!=reconstruction_signature(reconstruction):raise ValueError('damping reconstruction mismatch')
        super().__init__(meshsize,edges,pair,b1,f_growth,p_halo=p_halo,reconstruction=reconstruction)
        pn=power_nw(np.sqrt(self.k2)) if callable(power_nw) else power_nw
        self.configure_ir(pn,harmonics.damping(self.v))
        self.harmonic_metadata=dict(harmonics.physical_parameters,lmax=harmonics.lmax,r_bao=harmonics.r_bao)
