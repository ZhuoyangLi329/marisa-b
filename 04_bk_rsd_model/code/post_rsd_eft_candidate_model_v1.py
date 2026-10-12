"""模块大纲：独立11参数EFT候选的P+B预测；精确逐盒权重矩与完整joint白化。

仅用于实际STO候选preflight，不改旧8参数模型、不定义formal prior/放行。
mean data与C_single由调用方给出；对N2²、N2*Nr保留mean(product)。
P residual与B RSD noise/transport使用同一个非负raw N0=u*N2。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_noir_polynomial import predict_noir
from post_rsd_noir_likelihood import predict_p_rows
from post_rsd_stochastic_eft_v1 import MarkedNoiseEFT,intrinsic_from_primitives,zero_from_primitives

NAMES=('fNL','b1','b2_native','bK2','a_delta','a_phi','s_factorial','Nraw_over_N2',
       'b1_w2','bphi_w2','J21_over_N3')


@dataclass
class EFTCandidateModel:
    B_coefficients: np.ndarray
    P_templates: np.ndarray
    operators: np.ndarray
    intrinsic_primitives: np.ndarray
    zero_primitives: np.ndarray
    weight_moments: np.ndarray
    p_halo: float
    f_growth: float

    def __post_init__(self):
        for name in ('B_coefficients','P_templates','operators','intrinsic_primitives','zero_primitives','weight_moments'):
            setattr(self,name,np.asarray(getattr(self,name),dtype=float))
        self.nB=self.B_coefficients.shape[-1]
        if (self.B_coefficients.shape!=(5,3,3,self.nB) or self.P_templates.shape!=(6,12)
            or self.operators.shape!=(3,3,3,self.nB) or self.intrinsic_primitives.shape!=(7,self.nB)
            or self.zero_primitives.shape!=(14,self.nB) or self.weight_moments.shape!=(5,)):
            raise ValueError('original quartic B/12P, explicit operators/primitives and actual5 mark moments required')
        if not all(np.isfinite(getattr(self,n)).all() for n in (
                'B_coefficients','P_templates','operators','intrinsic_primitives','zero_primitives','weight_moments')):
            raise ValueError('nonfinite EFT candidate input')
        if np.any(self.weight_moments<0) or not all(np.isfinite(x) for x in (self.p_halo,self.f_growth)):
            raise ValueError('positive actual self moments and explicit physical p/growth required')
        self.n=self.nB+12;self.ndim=len(NAMES)

    def predictions(self,values):
        rows=np.atleast_2d(np.asarray(values,dtype=float))
        if rows.shape[1:]!=(11,) or not np.isfinite(rows).all() or np.any(rows[:,7]<0):
            raise ValueError('explicit11 physical parameters with nonnegative raw data power required')
        result=np.zeros((len(rows),self.n))
        result[:,:12]=predict_p_rows(rows[:,:2],self.P_templates,p_halo=self.p_halo)
        result[:,12:]=predict_noir(rows[:,:4],self.B_coefficients)
        n2,n3,n2sq,nr,n2nr=self.weight_moments
        for index,row in enumerate(rows):
            fnl,b1,_,_,ad,ap,s,ratio,b2w,bpw,j21=row;bphi=2*1.686*(b1-self.p_halo)
            noise=MarkedNoiseEFT(ratio*n2,n2,n3,ad*b1*n2,ap*2*1.686*n2,s*n2sq,b2w,bpw,j21*n3,nr)
            intrinsic=intrinsic_from_primitives(self.intrinsic_primitives,b1=b1,bphi=bphi,
                f_growth=self.f_growth,fnl=fnl,noise=noise)
            cb=np.array([1.,(b1-2.25)/1.75,2*((b1-2.25)/1.75)**2-1])
            powers=np.array([1.,fnl,fnl*fnl])
            op=np.einsum('i,iafb,f->ab',cb,self.operators,powers)
            transport=ratio*n2*op[0]+(ratio-1)*n2*op[1]+ratio*(ratio-1)*n2sq*op[2]
            zero=zero_from_primitives(self.zero_primitives,b1=b1,bphi=bphi,
                f_growth=self.f_growth,fnl=fnl,noise=noise)
            # 只替换zero项的N0*(N2+Nr)乘积均值，不能把mean(N2)²当mean(N2²)。
            zero+=2*ratio*(n2sq+n2nr-n2*n2-n2*nr)*self.zero_primitives[12]
            result[index,:12]+=(ratio-1)*n2
            result[index,12:]+=intrinsic+transport+zero
        return result

    def residual(self,values,target,cholesky):
        target=np.asarray(target,dtype=float);chol=np.asarray(cholesky,dtype=float)
        if (target.shape!=(self.n,) or chol.shape!=(self.n,self.n) or not np.isfinite(chol).all()
            or np.any(np.diag(chol)<=0) or np.any(np.triu(chol,1)!=0)):
            raise ValueError('own full joint positive Cholesky; no P/B separate whitening or C regularization')
        return np.linalg.solve(chol,(self.predictions(values)-target).T).T
