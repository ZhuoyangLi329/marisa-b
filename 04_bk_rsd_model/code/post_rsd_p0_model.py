"""模块大纲：same-shift post的线性P0源；复用pre表达式，显式物理p。

Z1_post=Z1_weighted_input。只补物理b_phi差，不引入p_rec/fNL_rec到拟合Z1。
本模块不处理IR、重构相关噪声或likelihood，不以旧pre的p作为默认校准。
"""
import numpy as np
from rsd_pb_common import p_prediction as pre_prediction
from rsd_b000_contract import P_CAL,F_GROWTH
from post_rsd_fixedrec import pbs_bias


def post_p0_linear(fnl,b1,templates,*,p_halo):
    """返回不含噪声的no-IR线性P0；templates必须来自实际生产P window。"""
    if not all(np.isscalar(x) and np.isfinite(x) for x in (fnl,b1,p_halo)):
        raise ValueError('finite scalar physical parameters are required')
    t=np.asarray(templates,dtype=float)
    if t.ndim!=2 or t.shape[0]!=6 or not np.isfinite(t).all():raise ValueError('six finite P-window templates required')
    value=pre_prediction(fnl,b1,t)
    if p_halo==P_CAL:return value
    old=pbs_bias(b1,p_halo=P_CAL).bphi;new=pbs_bias(b1,p_halo=p_halo).bphi
    return value+2*fnl*(new-old)*(b1*t[3]+F_GROWTH*t[4])+fnl*fnl*(new*new-old*old)*t[5]
