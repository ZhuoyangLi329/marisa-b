"""模块大纲：固定data-only shift的有向PN交叉/PN同腿/NN transport算子。

三个算子不是三个已认证STO nuisance，也不自行选择其幅度或先验。
P11=P_lin Z1²保留物理fNL²；response只含冻结catalog参数。
复用唯一两腿/full-shell/零腿FFT实现，保持旧确定性及Poisson审计源码不变。
"""
import numpy as np
from rsd_b000_fft_projection import Field,D,ONE
from post_rsd_poisson_projection import _PoissonShiftFactors,integrate_poisson_shift_design

NAMES=('PN_cross_leg','PN_same_leg','NN')


def integrate_noise_transport(fields):
    """返回每个算子的G/L/Q；幅度均为1，尚未乘任何实际STO噪声矩。

T_i^{ij}=((ki+kj).ki/ki²) R_*(ki)。
cross=-sum_pairs(T_i D_j+T_j D_i)；same=-sum_pairs(T_i D_i+T_j D_j)。
NN=-sum_pairs(T_i+T_j)，其L/Q严格为0。Gaussian raw transport为
N*(cross+same)+N²*NN；factorial修正必须另外审计，不能当raw直接拟合。
"""
    if not isinstance(fields,_PoissonShiftFactors):raise TypeError('explicit fixed-response fields required')
    cross=integrate_poisson_shift_design(fields,1.)
    same=np.zeros_like(cross);nn=np.zeros_like(cross)
    for axis in range(3):
        closing=ONE.op((axis,));dr=Field('DR').op((axis,),inv2=True)
        sr=Field('SR').op((axis,),inv2=True)
        for degree in range(3):
            same[degree]+=fields.integral(dr,degree,ONE,0,closing,0)
            same[degree]+=fields.integral(ONE,0,dr,degree,closing,0)
        nn[0]+=fields.integral(sr,0,ONE,0,closing,0)
        nn[0]+=fields.integral(ONE,0,sr,0,closing,0)
    return np.stack((cross,same,nn))
