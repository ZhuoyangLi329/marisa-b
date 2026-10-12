"""模块大纲：有向 marked-Poisson shift 响应；复用原 exact-lattice 投影。

这是独立点过程审计支持的 leading P*N2、一阶 shift 修正，不是完整 STO
非 Poisson stochastic closure。只显式调用时计算，不改变确定性/旧 pre 模型。
N2 是 data 权重的 V sum(w²)/(sum w)²，不含 finite-random 的白噪声。
实际位移从 data 生成；R_* 和 fNL_rec 固定，D=P*Z1² 保留物理 fNL²。
"""
import numpy as np
from rsd_b000_fft_projection import Field,D,ONE
from post_rsd_fft_projection import PostFields,PostFFTProjection


class _PoissonShiftFactors:
    def degrees(self,field):
        return range(1) if field.root=='SR' else super().degrees(field)

    def values(self,field,degree):
        if field.root=='SR':
            if degree!=0:raise ValueError('the catalog shift response is not a physical-fNL polynomial')
            base=Field('one',field.axes,field.inv2,field.invM)
            return super().values(base,0)*self.shift_response
        return super().values(field,degree)


class PoissonShiftFields(_PoissonShiftFactors,PostFields):
    pass


class PoissonShiftFFTProjection(_PoissonShiftFactors,PostFFTProjection):
    pass


def integrate_poisson_shift_design(fields,n2):
    """返回 G,L,Q 系数；不加入任意 nuisance 或自动声称适用于实际 STO。

    Delta B=-N2 sum_{i!=j} P11(ki) [(ki+kj).kj/kj²] R_*(kj)。
    闭合腿 k_l=-(ki+kj) 将负号转为 positive closing-vector contraction。
    两腿 shell 分母和零腿规则完全继承唯一 FFT 引擎。
    """
    if not isinstance(fields,_PoissonShiftFactors):
        raise TypeError('use the explicit PoissonShift fields; legacy stochastic is not a substitute')
    if not np.isscalar(n2) or not np.isfinite(n2) or n2<0:
        raise ValueError('N2 must be an explicit finite nonnegative data shot amplitude')
    result=None
    for axis in range(3):
        response=Field('SR').op((axis,),inv2=True)
        closing=ONE.op((axis,))
        for degree in range(3):
            value=fields.integral(response,0,D,degree,closing,0)
            value+=fields.integral(D,degree,response,0,closing,0)
            if result is None:result=np.zeros((3,)+np.shape(value))
            result[degree]+=float(n2)*value
    return result
