"""模块大纲：保留v1噪声源；恢复原P0有效Pshot独立自由度，明确12参数候选。

v1把tree P残差与data-only raw shift noise硬绑定，真实profile触及Nraw=0。
P0原残差是有效常数nuisance，可含被截断高阶贡献，不是pure epsilon校准。
本版不改B源或数据；Pshot恢复原+/-1e6物理范围，Nraw仍只约束raw>=0。
"""
import numpy as np
from post_rsd_eft_candidate_model_v1 import EFTCandidateModel,NAMES

NAMES_V2=NAMES+('Pshot_residual_nbar',)


class EFTCandidateModelV2(EFTCandidateModel):
    def __post_init__(self):
        super().__post_init__();self.ndim=len(NAMES_V2)

    def predictions(self,values):
        rows=np.atleast_2d(np.asarray(values,dtype=float))
        if rows.shape[1:]!=(12,) or not np.isfinite(rows).all():
            raise ValueError('explicit12 parameter rows required for independent original Pshot')
        result=super().predictions(rows[:,:11])
        n2=self.weight_moments[0]
        result[:,:12]+=n2*(rows[:,11]-(rows[:,7]-1))[:,None]
        return result
