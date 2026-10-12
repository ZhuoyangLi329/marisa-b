"""模块大纲：旧real-space兼容三噪声闭合，明确七参数的RSD有效模型。

旧实际兼容路径的Bshot G/L/Q系数是b1²、b1*bphi、bphi²，L不乘2。
本分支用Z_G=b1+f_g*mu²提升Gaussian/L；Q不改，并按原两腿window投影。
该legacy-compatible现象闭合不是完整marked-STO物理认证；不新增raw、
epsilon_phi、w²-bias、J21、anisotropic-noise或shift-noise自由参数。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_noir_polynomial import predict_noir
from post_rsd_noir_likelihood import predict_p_rows

NAMES = ('fNL', 'b1', 'b2_native', 'bK2', 'Ashot_residual',
         'Bshot_residual', 'Pshot_residual_nbar')


def legacy_bshot_shape(primitives, fnl, b1, bphi, growth):
    """继承旧G+lambda*L+lambda²*Q，不错误替换为完整Z1²的factor2交叉项。"""
    p, pmu, pmu4, pM, pmuM, pM2, valid = np.asarray(primitives, dtype=float)
    return (b1*b1*p + 2*b1*growth*pmu + growth*growth*pmu4
            + fnl*bphi*(b1*pM + growth*pmuM) + fnl*fnl*bphi*bphi*pM2)


@dataclass
class LegacyThreeNoiseModel:
    B_coefficients: np.ndarray
    P_templates: np.ndarray
    noise_primitives: np.ndarray
    weight_moments: np.ndarray
    p_halo: float
    f_growth: float

    def __post_init__(self):
        for name in ('B_coefficients', 'P_templates', 'noise_primitives', 'weight_moments'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.nB = self.B_coefficients.shape[-1]
        if (self.B_coefficients.shape != (5,3,3,self.nB) or self.P_templates.shape != (6,12)
                or self.noise_primitives.shape != (7,self.nB) or self.weight_moments.shape != (2,)
                or np.any(self.weight_moments <= 0)
                or not all(np.isfinite(getattr(self, name)).all() for name in
                    ('B_coefficients','P_templates','noise_primitives','weight_moments'))):
            raise ValueError('explicit quartic tree/12P/seven actual window primitives/actualN2,N3 required')
        if not np.isfinite([self.p_halo,self.f_growth]).all():
            raise ValueError('explicit fixed physical p and growth required')
        self.ndim = 7
        self.n = 12+self.nB

    def predictions(self, values):
        rows = np.atleast_2d(np.asarray(values,dtype=float))
        if rows.shape[1:] != (7,) or not np.isfinite(rows).all():
            raise ValueError('explicit seven finite physical parameters required')
        result = np.zeros((len(rows),self.n))
        n2,n3 = self.weight_moments
        result[:,:12] = predict_p_rows(rows[:,:2],self.P_templates,p_halo=self.p_halo)
        result[:,:12] += rows[:,6,None]*n2
        result[:,12:] = predict_noir(rows[:,:4],self.B_coefficients)
        for index,row in enumerate(rows):
            fnl,b1 = row[:2]
            bphi = 2*1.686*(b1-self.p_halo)
            shape = legacy_bshot_shape(self.noise_primitives,fnl,b1,bphi,self.f_growth)
            result[index,12:] += row[4]*n3*self.noise_primitives[6] + row[5]*n2*shape
        return result

    def residual(self, values, target, cholesky):
        chol=np.asarray(cholesky,dtype=float);target=np.asarray(target,dtype=float)
        if (target.shape!=(self.n,) or chol.shape!=(self.n,self.n)
                or np.any(np.diag(chol)<=0) or np.any(np.triu(chol,1))
                or not np.isfinite(chol).all()):
            raise ValueError('original full-joint positive Cholesky required; no regularization')
        return np.linalg.solve(chol,(self.predictions(values)-target).T).T
