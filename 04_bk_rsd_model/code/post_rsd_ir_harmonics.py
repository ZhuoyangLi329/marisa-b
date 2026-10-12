"""模块大纲：固定 rBAO 的 A_ab(rhat) 球谐压缩；逐 Fourier mode 的各向异性阻尼。

用平面波球谐展开精确处理 q.r 角核，仅 truncation 在 spectral tensor 的 lmax。
复用 GaussianLongDisplacement 的所有 physical/source/积分节点，不平均 LOS 或 CIC。
每个参数点需要自己的 coefficients；本对象不可当作跨 b1 不变的隐式缓存。
"""
from dataclasses import dataclass,asdict
import hashlib
import json
import numpy as np
from scipy.special import sph_harm,spherical_jn


def reconstruction_signature(reconstruction):
    payload=None if reconstruction is None else asdict(reconstruction)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def even_harmonics(directions,lmax):
    """实正交归一球谐；列表含所有偶 l、全部 m，保持 Cartesian 方向信息。"""
    u=np.asarray(directions,dtype=float);norm=np.linalg.norm(u,axis=-1)
    if u.ndim!=2 or u.shape[1]!=3 or np.any(norm==0):raise ValueError('nonzero directions required')
    u=u/norm[:,None];theta=np.arccos(np.clip(u[:,2],-1.,1.));phi=np.arctan2(u[:,1],u[:,0])
    columns=[];degrees=[]
    for ell in range(0,lmax+1,2):
        for m in range(-ell,ell+1):
            yy=sph_harm(abs(m),ell,phi,theta)
            if m==0:column=yy.real
            elif m>0:column=np.sqrt(2)*(-1)**m*yy.real
            else:column=np.sqrt(2)*(-1)**abs(m)*yy.imag
            columns.append(column);degrees.append(ell)
    return np.column_stack(columns),np.asarray(degrees,dtype=int)


@dataclass(frozen=True)
class HarmonicDisplacement:
    coefficients: object
    lmax: int
    r_bao: float
    physical_parameters: dict

    @classmethod
    def from_long_modes(cls,long_modes,*,r_bao=110.,lmax=10,b1,bphi,f_growth,los,reconstruction=None,shift_noise=None):
        if lmax<2 or lmax%2 or long_modes.n_mu<lmax+2 or long_modes.n_phi<2*lmax+2:
            raise ValueError('even lmax and sufficiently dense angular rule required')
        if not np.isfinite(r_bao) or r_bao<0:raise ValueError('invalid BAO separation')
        directions,wa,radial=long_modes.quadrature();y,ell=even_harmonics(directions,lmax)
        weighted_y=y.T*wa[None,:]
        noise=long_modes.noise_spectrum(shift_noise,reconstruction)
        coefficients=np.zeros((len(ell),3,3,3))
        for iq,q in enumerate(long_modes.q):
            source=long_modes.spectral_tensor(iq,directions,b1=b1,bphi=bphi,f_growth=f_growth,los=los,
                    reconstruction=reconstruction,noise_power=None if noise is None else noise[iq])
            moments=(weighted_y@source.reshape(len(directions),-1)).reshape(-1,3,3,3)
            kernel=(ell==0).astype(float)-(-1.)**(ell//2)*spherical_jn(ell,q*r_bao)
            coefficients+=radial[iq]*q*q/np.pi**2*kernel[:,None,None,None]*moments
        params=dict(b1=float(b1),bphi=float(bphi),f_growth=float(f_growth),los=np.asarray(los,dtype=float).tolist(),
                    q_min=float(long_modes.q[0]),q_max=float(long_modes.q[-1]),n_q=len(long_modes.q),
                    n_mu=long_modes.n_mu,n_phi=long_modes.n_phi,shift_noise_explicit=shift_noise is not None,
                    reconstruction_signature=reconstruction_signature(reconstruction),
                    long_spectrum_sha256=hashlib.sha256(b''.join(np.asarray(x,dtype='f8').tobytes() for x in
                        (long_modes.q,long_modes.power_nw,long_modes.transfer_nw))).hexdigest())
        return cls(coefficients,lmax,float(r_bao),params)

    def covariance(self,directions):
        y,_=even_harmonics(directions,self.lmax)
        return (y@np.asarray(self.coefficients).reshape(len(y[0]),-1)).reshape(len(y),3,3,3)

    def damping(self,vectors,*,blocksize=8192):
        """D0,d1,d2；分块计算避免在整块 theory grid 存储每点27个 tensor 值。"""
        v=np.asarray(vectors,dtype=float)
        if v.shape[-1]!=3 or not np.isfinite(v).all():raise ValueError('invalid Fourier vectors')
        shape=v.shape;flat=v.reshape(-1,3);result=np.zeros((len(flat),3));result[:,0]=1.
        for start in range(0,len(flat),blocksize):
            vv=flat[start:start+blocksize];active=np.sum(vv*vv,axis=1)>0
            if not active.any():continue
            x=vv[active];a=self.covariance(x);e=.5*np.einsum('ri,rcij,rj->rc',x,a,x)
            if np.min(e[:,0]) < -1e-11*max(1.,float(np.max(np.abs(e[:,0])))):
                raise ValueError('harmonic approximation gives negative Gaussian exponent; refine before use')
            result[start:start+len(vv)][active]=np.column_stack((np.exp(-np.maximum(e[:,0],0.)),-e[:,1],.5*e[:,1]**2-e[:,2]))
        return result.reshape(shape[:-1]+(3,))
