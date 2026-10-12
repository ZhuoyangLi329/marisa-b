"""模块大纲：固定算法的残余长位移；各向异性相对协方差；有限 PNG 指数系数。

这里只实现 linear-Gaussian 长位移的连续主像，不是完整的 B IR 入口。
保持实际 alpha_rec 冻结，物理长模使用输入的 P_nw/M_nw；不默认添加
1/nbar、random 或 STO 噪声。有限网格 alias 和 PNG 高阶累积量未在此闭合。
"""
from dataclasses import dataclass
import numpy as np


def _los_unit(los):
    n=np.asarray(los,dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)==0:
        raise ValueError('LOS must be a nonzero finite vector')
    return n/np.linalg.norm(n)


def residual_coefficients(vectors,transfer_nw,*,b1,bphi,f_growth,los,reconstruction=None):
    """返回 d0,d1，使 Psi_rec(q)=i[d0(q)+fNL*d1(q)] delta_L(q)。"""
    q=np.asarray(vectors,dtype=float);m=np.asarray(transfer_nw,dtype=float)
    if q.shape[-1]!=3 or not np.isfinite(q).all():raise ValueError('invalid wavevectors')
    q2=np.sum(q*q,axis=-1)
    if np.any(q2<=0) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('zero mode or invalid transfer')
    if not np.isfinite([b1,bphi,f_growth]).all():raise ValueError('invalid physical bias/growth')
    n=_los_unit(los);q_los=q@n
    z0=b1+f_growth*q_los*q_los/q2
    response=0. if reconstruction is None else reconstruction.response(q,m)
    d0=(q+f_growth*q_los[...,None]*n-response[...,None]*z0[...,None]*q)/q2[...,None] if np.ndim(response) else (q+f_growth*q_los[...,None]*n)/q2[...,None]
    d1=-np.asarray(response)[...,None]*bphi/m[...,None]*q/q2[...,None]
    return d0,d1


@dataclass(frozen=True)
class GaussianLongDisplacement:
    """径向正权重梯形 + 球面 Gauss-Legendre/等距 phi；所有参数显式。"""
    q: object
    power_nw: object
    transfer_nw: object
    n_mu: int=24
    n_phi: int=48

    def __post_init__(self):
        q,p,m=(np.asarray(v,dtype=float) for v in (self.q,self.power_nw,self.transfer_nw))
        if (q.ndim!=1 or len(q)<2 or p.shape!=q.shape or m.shape!=q.shape
                or not np.isfinite(q).all() or not np.isfinite(p).all() or not np.isfinite(m).all()
                or np.any(q<=0) or np.any(p<0) or np.any(m<=0) or np.any(np.diff(q)<=0)):
            raise ValueError('invalid long-mode spectrum; require ordered positive q,M and nonnegative P')
        if self.n_mu<4 or self.n_phi<8:raise ValueError('insufficient angular rule')
        for key,value in [('q',q),('power_nw',p),('transfer_nw',m)]:
            object.__setattr__(self,key,value.copy())

    def noise_spectrum(self,shift_noise,reconstruction):
        """所有积分路径复用同一噪声输入校验，不默认添加 shot-noise。"""
        noise=None if shift_noise is None else np.asarray(shift_noise,dtype=float)
        if noise is not None and (noise.shape!=self.q.shape or not np.isfinite(noise).all() or np.any(noise<0)):
            raise ValueError('shift noise must be finite nonnegative input spectrum')
        if noise is not None and reconstruction is None:raise ValueError('shift noise requires an active shift')
        return noise

    def quadrature(self):
        """所有积分路径复用同一径向/角向正权重规则。"""
        mu,wm=np.polynomial.legendre.leggauss(self.n_mu)
        phi=np.arange(self.n_phi)*(2*np.pi/self.n_phi)
        sine=np.sqrt(1-mu*mu)
        directions=np.stack(np.broadcast_arrays(sine[:,None]*np.cos(phi),sine[:,None]*np.sin(phi),mu[:,None]),axis=-1).reshape(-1,3)
        wa=np.repeat(wm,self.n_phi)*(2*np.pi/self.n_phi)
        radial=np.empty_like(self.q)
        radial[0]=(self.q[1]-self.q[0])/2;radial[-1]=(self.q[-1]-self.q[-2])/2
        radial[1:-1]=(self.q[2:]-self.q[:-2])/2
        return directions,wa,radial

    def spectral_tensor(self,iq,directions,*,b1,bphi,f_growth,los,reconstruction=None,noise_power=None):
        """共用唯一 d0/d1 和 spectral outer products；直接积分与 harmonic 压缩一致。"""
        q=self.q[iq];v=q*directions;m=np.full(len(v),self.transfer_nw[iq])
        d0,d1=residual_coefficients(v,m,b1=b1,bphi=bphi,f_growth=f_growth,los=los,reconstruction=reconstruction)
        outer0=d0[:,:,None]*d0[:,None,:]
        outer1=d0[:,:,None]*d1[:,None,:]+d1[:,:,None]*d0[:,None,:]
        outer2=d1[:,:,None]*d1[:,None,:]
        source=self.power_nw[iq]*np.stack((outer0,outer1,outer2),axis=1)
        if noise_power is not None:
            ds=-reconstruction.response(v,m)[:,None]*v/(q*q)
            source[:,0]+=noise_power*ds[:,:,None]*ds[:,None,:]
        return source

    def relative_covariance(self,separations,*,b1,bphi,f_growth,los,reconstruction=None,shift_noise=None):
        """A(r)=A0+fNL*A1+fNL²*A2；输出 (N_r,3_coeff,3,3)。

        shift_noise 可给与 delta_L 无交叉的 data-shift 白/有色噪声功率数组。
        这只是显式 Gaussian 噪声假设，不能当作重构 stochastic closure 的证明。
        实际 runner 的 shift 来自 data，故有限 random 不自动计入该数组。
        """
        r=np.asarray(separations,dtype=float)
        if r.ndim==1:r=r[None]
        if r.ndim!=2 or r.shape[1]!=3 or not np.isfinite(r).all():raise ValueError('invalid separations')
        noise=self.noise_spectrum(shift_noise,reconstruction)
        directions,wa,radial=self.quadrature()
        result=np.zeros((len(r),3,3,3))
        for iq,q in enumerate(self.q):
            v=q*directions
            source=self.spectral_tensor(iq,directions,b1=b1,bphi=bphi,f_growth=f_growth,los=los,
                                        reconstruction=reconstruction,noise_power=None if noise is None else noise[iq])
            # 2*(1-cos(q.r)) = 4*sin(q.r/2)^2，避免极小 r 的相消误差。
            weight=4*np.sin((r@v.T)/2)**2*wa[None,:]
            result+=radial[iq]*q*q/(2*np.pi)**3*np.einsum('ra,acij->rcij',weight,source)
        return result


def damping_coefficients(vectors,covariance_coefficients):
    """逐有向腿返回 exp[-E0]*(1+fNL*d1+fNL²*d2)，不是源项 dressing。"""
    k=np.asarray(vectors,dtype=float);a=np.asarray(covariance_coefficients,dtype=float)
    if k.ndim==1:k=k[None]
    if a.shape!=(len(k),3,3,3) or not np.isfinite(a).all():raise ValueError('covariance shape mismatch')
    exponent=.5*np.einsum('ri,rcij,rj->rc',k,a,k)
    return np.column_stack((np.exp(-exponent[:,0]),-exponent[:,1],
                            .5*exponent[:,1]**2-exponent[:,2])),exponent
