"""模块大纲：独立finite-PNG RSD marked-EFT候选；不替换旧8参数模型。

从epsilon、epsilon_delta*delta、fNL*epsilon_phi*phi及RSD坐标映射推导。
显式分开raw N0、粒子self N2、权重平方场bias及mixed contact；只定义
leading白噪声/tree源与原matched零腿。尚未作实际STO/finite-shift/IR闭合。
此模块无采样器、无先验、无formal PASS；完整验证前不能用于主MCMC。
"""
from dataclasses import dataclass
import numpy as np
from rsd_b000_fft_projection import Field, ONE

PRIMITIVES = ('P', 'Pmu2', 'Pmu4', 'P_over_M', 'Pmu2_over_M', 'P_over_M2', 'valid_fraction')
ZERO_PRIMITIVES = PRIMITIVES[:6]+tuple('R_'+s for s in PRIMITIVES[:6])+('R_only','zero_fraction')


@dataclass(frozen=True)
class MarkedNoiseEFT:
    """全部是物理单位；不把bphi=0变成PNG噪声响应必须为零的假先验。

    A_delta=2*C(eps,eps_delta)-N2*b1_w2 [Mpc/h]^3；
    A_phi=2*C(eps,eps_phi)-N2*bphi_w2 [Mpc/h]^3；
    S_factorial=C(eps,eps,eps)-3*J21+2*N3 [Mpc/h]^6。
    J21=<raw w²-noise,raw w-noise> [Mpc/h]^6；只用于matched零腿。
    """
    raw_power: float
    n2_self: float
    n3_self: float
    A_delta: float
    A_phi: float
    S_factorial: float
    b1_w2: float
    bphi_w2: float
    J21: float
    random_power: float = 0.

    def __post_init__(self):
        if not all(np.isscalar(v) and np.isfinite(v) for v in vars(self).values()):
            raise ValueError('explicit finite physical stochastic coefficients required')
        if min(self.raw_power,self.n2_self,self.n3_self,self.random_power)<0:
            raise ValueError('raw power/self moments/random power must be nonnegative')

    @property
    def P_residual(self):
        return self.raw_power-self.n2_self


def _legs(vectors,power,transfer,b1,bphi,f_growth,fnl):
    v=np.asarray(vectors,dtype=float);p=np.asarray(power,dtype=float);m=np.asarray(transfer,dtype=float)
    if (v.shape[-1:]!=(3,) or p.shape!=v.shape[:-1] or m.shape!=p.shape
        or not all(np.isfinite(a).all() for a in (v,p,m)) or np.any(p<0) or np.any(m<=0)
        or not all(np.isscalar(a) and np.isfinite(a) for a in (b1,bphi,f_growth,fnl))):
        raise ValueError('finite directed leg powers/transfers and scalar physical parameters required')
    k2=np.sum(v*v,axis=-1)
    if np.any(k2<=0):raise ValueError('this source requires nonzero legs; matched zero atom is separate')
    mu2=v[...,2]**2/k2;u=1/m
    z=b1+f_growth*mu2+fnl*bphi*u
    return v,p,u,mu2,z


def intrinsic_factorial(vectors,power,transfer,*,b1,bphi,f_growth,fnl,noise):
    """非零有向闭合三角形；明确A_phi独立于deterministic bphi。

    Bfac=sum_i P_i Z1_i*(A_delta+f*mu_i²*(N0-N2)+fNL*A_phi/M_i)+Sfac。
    fNL二次由同一个Z1自动生成，无盲目Z1²替换或省略交叉项。
    """
    if not isinstance(noise,MarkedNoiseEFT):raise TypeError('explicit marked EFT physical coefficients required')
    v,p,u,mu2,z=_legs(vectors,power,transfer,b1,bphi,f_growth,fnl)
    if v.shape[-2:]!=(3,3) or not np.allclose(v.sum(axis=-2),0.,rtol=0,atol=1e-12):
        raise ValueError('nonzero closed directed triangle required')
    bracket=noise.A_delta+f_growth*mu2*noise.P_residual+fnl*noise.A_phi*u
    return np.sum(p*z*bracket,axis=-1)+noise.S_factorial


def matched_zero_atom(vectors,power,transfer,response,*,b1,bphi,f_growth,fnl,noise):
    """原closing零腿(k,-k,0)的leading matched factorial预测；不改data定义。

    Bzero=-2*(N2*Z_w2*Z_w*P+J21)+2*(N2+Nr)*R*(Z_w²*P+N0)+2*N3。
    random的独立Nr²自项相消；Nr*R的背景权重项必须保留。R不随物理fNL变。
    全shell分母由投影调用方保留，不对零腿追加kmax或重归一化。
    """
    if not isinstance(noise,MarkedNoiseEFT):raise TypeError('explicit marked EFT physical coefficients required')
    _,p,u,mu2,z=_legs(vectors,power,transfer,b1,bphi,f_growth,fnl)
    r=np.asarray(response,dtype=float)
    if r.shape!=p.shape or not np.isfinite(r).all():raise ValueError('explicit finite frozen shift response required')
    zw2=noise.b1_w2+f_growth*mu2+fnl*noise.bphi_w2*u
    return (-2*(noise.n2_self*zw2*z*p+noise.J21)
            +2*(noise.n2_self+noise.random_power)*r*(z*z*p+noise.raw_power)+2*noise.n3_self)


def project_intrinsic_primitives(fields):
    """复用唯一FFT积分/active leg/full shell；无固定physical-p或噪声参数。

    紧凑7个几何谱矩足以表达候选intrinsic的全部G/L/Q。
    不计算connected3/4、有限production位移或IR模板。
    """
    terms=((Field('P'),),(Field('Pmu2'),),(Field('Pmu4'),),
           (Field('P',invM=True),),(Field('Pmu2',invM=True),))
    result=[fields.integral(x[0],0,ONE,0,ONE,0) for x in terms]
    # Field.invM为boolean，只能施加一次；P/M²用两个A定义会引入多余P。
    # 下方adapter只增加此独立primitive，不改原生产Fields类。
    result.append(fields.integral(Field('P_over_M2_eft'),0,ONE,0,ONE,0))
    result.append(fields.integral(ONE,0,ONE,0,ONE,0)/3)
    return np.array(result)


class EFTPrimitiveFields:
    """Mixin只增加独立P/M²谱矩；父类原其他factor定义全部复用。"""
    def values(self,field,degree):
        if field.root=='P_over_M2_eft':
            if degree!=0 or field.axes or field.inv2 or field.invM:
                raise ValueError('P/M² primitive cannot silently add operators/degrees')
            return np.where(self.active,self.p*self.u*self.u,0.)
        return super().values(field,degree)


def intrinsic_from_primitives(primitives,*,b1,bphi,f_growth,fnl,noise):
    a=np.asarray(primitives,dtype=float)
    if a.shape[0]!=7 or not np.isfinite(a).all() or not isinstance(noise,MarkedNoiseEFT):
        raise ValueError('finite seven actual-window primitives and explicit noise required')
    if not all(np.isscalar(x) and np.isfinite(x) for x in (b1,bphi,f_growth,fnl)):
        raise ValueError('finite physical bias/growth/fNL required')
    s0,smu,smu4,sM,smuM,sM2,valid=a
    delta=noise.A_delta*(b1*s0+f_growth*smu+fnl*bphi*sM)
    phi=noise.A_phi*(fnl*(b1*sM+f_growth*smuM)+fnl*fnl*bphi*sM2)
    rsd=noise.P_residual*f_growth*(b1*smu+f_growth*smu4+fnl*bphi*smuM)
    return delta+phi+rsd+noise.S_factorial*valid


def project_zero_primitives(fields):
    """closing=0仅用原两shell交集；sum/原Ni*Nj，没有选择counterfactual data。

    不支持outer零腿，此任务原fit registry中的outer shells都严格非零。
    """
    left,right=(fields.masks[s] for s in fields.slots[:2])
    if np.any((left|right)&~fields.active):
        raise ValueError('outer zero modes need a separate labeled estimator derivation')
    common=left&right
    mu=fields.mu2;u=fields.u;p=fields.p;r=fields.shift_response
    spectra=np.array([p,p*mu,p*mu*mu,p*u,p*mu*u,p*u*u])
    sums=np.sum(spectra[:,common],axis=1)/fields.denominator
    shifted=np.sum((spectra*r)[...,common],axis=1)/fields.denominator
    return np.r_[sums,shifted,np.sum(r[common])/fields.denominator,
                 int(common.sum())/fields.denominator]


def zero_from_primitives(primitives,*,b1,bphi,f_growth,fnl,noise):
    a=np.asarray(primitives,dtype=float)
    if a.shape[0]!=14 or not np.isfinite(a).all() or not isinstance(noise,MarkedNoiseEFT):
        raise ValueError('finite fourteen actual zero-atom primitives and explicit noise required')
    if not all(np.isscalar(x) and np.isfinite(x) for x in (b1,bphi,f_growth,fnl)):
        raise ValueError('finite physical bias/growth/fNL required')
    b2=noise.b1_w2;bp2=noise.bphi_w2;f=f_growth;lam=fnl
    cross=np.array([b1*b2,f*(b1+b2),f*f,lam*(b1*bp2+b2*bphi),
                    lam*f*(bp2+bphi),lam*lam*bp2*bphi])
    signal=np.array([b1*b1,2*b1*f,f*f,2*lam*b1*bphi,2*lam*f*bphi,lam*lam*bphi*bphi])
    return (-2*noise.n2_self*np.tensordot(cross,a[:6],axes=(0,0))
            +2*(noise.n2_self+noise.random_power)*np.tensordot(signal,a[6:12],axes=(0,0))
            +2*(noise.n2_self+noise.random_power)*noise.raw_power*a[12]
            +2*(noise.n3_self-noise.J21)*a[13])
