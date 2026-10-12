"""模块大纲：严格保留4+3模型，只增加一个独立RSD各向异性B噪声幅度。

R基为N2*f_g²*sum[P(k_i)*(mu_i^4-1/5)]，使用相同实际B window。
1/5来自连续各向同性平均；离散window的剩余各向异性原样保留。
不按每个cut重新正交化/归一化，不改A/B/Pshot或PNG噪声配方。
这是用户要求的legacy有效模型对照，不是完整marked-STO噪声认证。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_legacy3_model import LegacyThreeNoiseModel,NAMES as THREE_NAMES

NAMES=THREE_NAMES+('R_stoch_anisotropic',)

@dataclass
class LegacyFourNoiseModel(LegacyThreeNoiseModel):
    def __post_init__(self):
        super().__post_init__();self.ndim=8
        self.anisotropic_shape=(self.weight_moments[0]*self.f_growth**2
            *(self.noise_primitives[2]-self.noise_primitives[0]/5.))

    def predictions(self,values):
        rows=np.atleast_2d(np.asarray(values,dtype=float))
        if rows.shape[1:]!=(8,) or not np.isfinite(rows).all():
            raise ValueError('eight finite parameters: unchanged four physics plus exactly four noise amplitudes')
        result=super().predictions(rows[:,:7])
        result[:,12:]+=rows[:,7,None]*self.anisotropic_shape
        return result
