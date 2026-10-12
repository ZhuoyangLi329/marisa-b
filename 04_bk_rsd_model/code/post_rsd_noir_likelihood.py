"""模块大纲：五节点post源的B/P+B似然；显式物理p；完整joint白化。

只作模型表示和线性白化，不选择噪声形状、不决定协方差或先验。
P/B来源仍复用已有实现；噪声基必须由上游实际post审计显式提供。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_noir_polynomial import predict_noir
from post_rsd_p0_model import post_p0_linear


def predict_p_rows(values, templates, *, p_halo):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():
        raise ValueError('finite [fNL,b1] rows required')
    if not np.isscalar(p_halo) or not np.isfinite(p_halo):raise ValueError('explicit finite physical p required')
    return np.array([post_p0_linear(f,b,templates,p_halo=p_halo) for f,b in values])


@dataclass
class PostNoIRLikelihood:
    target: np.ndarray
    cholesky: np.ndarray
    B_coefficients: np.ndarray
    P_templates: np.ndarray
    noise: np.ndarray
    p_halo: float
    noise_coefficients: np.ndarray | None = None

    def __post_init__(self):
        self.target=np.asarray(self.target,dtype=float);self.cholesky=np.asarray(self.cholesky,dtype=float)
        self.B_coefficients=np.asarray(self.B_coefficients,dtype=float)
        self.P_templates=np.asarray(self.P_templates,dtype=float);self.noise=np.asarray(self.noise,dtype=float)
        if (self.B_coefficients.ndim!=4 or self.B_coefficients.shape[:3]!=(5,3,3)
            or self.P_templates.ndim!=2 or self.P_templates.shape[0]!=6):
            raise ValueError('five-node quartic post B and six actual-window P templates required')
        self.nP=self.P_templates.shape[1];self.nB=self.B_coefficients.shape[-1];self.n=self.nP+self.nB
        if self.nP not in (0,12) or not self.nB or self.target.shape!=(self.n,):
            raise ValueError('B-only or original twelve-bin P+B required')
        if self.cholesky.shape!=(self.n,self.n) or self.noise.ndim!=2 or self.noise.shape[1]!=self.n:
            raise ValueError('full joint Cholesky and explicit joint noise basis required')
        if not all(np.isfinite(a).all() for a in (self.target,self.cholesky,self.B_coefficients,self.P_templates,self.noise)):
            raise ValueError('nonfinite likelihood arrays')
        if not np.isscalar(self.p_halo) or not np.isfinite(self.p_halo):raise ValueError('explicit finite physical p required')
        if np.any(np.diag(self.cholesky)<=0) or np.any(np.triu(self.cholesky,1)!=0):
            raise ValueError('positive lower Cholesky required; no regularization')
        self.ndim=4+len(self.noise)
        # 先在完整观测量空间补零，再用完整C_PB的Cholesky白化；禁止分别白化P/B。
        joint_b=np.zeros((5,3,3,self.n));joint_b[...,self.nP:]=self.B_coefficients
        joint_p=np.zeros((6,self.n));joint_p[:,:self.nP]=self.P_templates
        self.cwhite=np.linalg.solve(self.cholesky,joint_b.reshape(45,self.n).T).T.reshape(5,3,3,self.n)
        self.pwhite=np.linalg.solve(self.cholesky,joint_p.T).T
        self.nwhite=np.linalg.solve(self.cholesky,self.noise.T).T
        self.twhite=np.linalg.solve(self.cholesky,self.target)
        self.noise_cwhite=None
        if self.noise_coefficients is not None:
            self.noise_coefficients=np.asarray(self.noise_coefficients,dtype=float)
            expected=(len(self.noise),5,3,3,self.n)
            if self.noise_coefficients.shape!=expected or not np.isfinite(self.noise_coefficients).all():
                raise ValueError('explicit finite [noise,b1:5,nuisance:3,fNL:3,joint bins] tensor required')
            self.noise_cwhite=np.linalg.solve(self.cholesky,
                self.noise_coefficients.reshape(-1,self.n).T).T.reshape(expected)

    def _polynomial_noise(self,rows,coefficients):
        if coefficients is None:return np.zeros((len(rows),self.n))
        # 只改原nuisance的形状表示，不增加参数；先full-joint白化再相乘。
        values=predict_noir(rows[:,:4],coefficients.transpose(1,2,3,0,4))
        values=values.reshape(len(rows),len(self.noise),self.n)
        return np.einsum('ra,ran->rn',rows[:,4:],values)

    def _rows(self,values):
        rows=np.atleast_2d(np.asarray(values,dtype=float))
        if rows.ndim!=2 or rows.shape[1]!=self.ndim or not np.isfinite(rows).all():
            raise ValueError('physical parameters must match explicitly provided noise basis')
        return rows

    def predictions(self,values):
        rows=self._rows(values);result=np.zeros((len(rows),self.n))
        result[:,self.nP:]=predict_noir(rows[:,:4],self.B_coefficients)
        if self.nP:result[:,:self.nP]=predict_p_rows(rows[:,:2],self.P_templates,p_halo=self.p_halo)
        return result+rows[:,4:]@self.noise+self._polynomial_noise(rows,self.noise_coefficients)

    def residual(self,values):
        rows=self._rows(values)
        result=predict_noir(rows[:,:4],self.cwhite)+rows[:,4:]@self.nwhite-self.twhite
        if self.nP:result+=predict_p_rows(rows[:,:2],self.pwhite,p_halo=self.p_halo)
        return result+self._polynomial_noise(rows,self.noise_cwhite)
