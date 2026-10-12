"""模块大纲：有向 native G/L/Q；参考类型；Legendre 和显式 mode-list 投影。

连续 B_ell 使用 k1 轴的 Scoccimarro 归一化，不冒称 TriPoSH B_ell1ell2L。
有 CIC 的 post 投影必须提供实际 Cartesian modes，不能丢失绕 LOS 的角度。
"""
from pathlib import Path
import ctypes
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np

PART=Path(__file__).resolve().parents[1]
REFERENCE=PART/'work/reference_package/python/marisa_b/tree_png_rsd_reference.py'
spec=importlib.util.spec_from_file_location('marisa_b_rsd_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=reference
spec.loader.exec_module(reference)
HaloBiasV1=reference.HaloBiasV1
RsdConfig=reference.RsdConfig
ReconstructionConfig=reference.ReconstructionConfig
pre_recon_rsd_ir_covariance=reference.pre_recon_rsd_ir_covariance

def bias_from_eft(b1,b2_eft,gamma2,bphi,bphidelta,bphi2):
    """显式转换 EFT 的 mu12²-1 tidal 约定；PNG 高阶 bias 保持独立。"""
    return HaloBiasV1(b1,b2_eft-4.*gamma2/3.,gamma2,bphi,bphidelta,bphi2)

class NativeTreeRSD:
    def __init__(self,library=None):
        self.path=Path(library or PART/'work/build/libmarisa_tree_rsd.so')
        self.lib=ctypes.CDLL(str(self.path))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.marisa_tree_rsd_batch.argtypes=[ctypes.c_size_t]+[ptr]*6+[ctypes.c_int,ctypes.c_int,ptr]
        self.lib.marisa_tree_rsd_batch.restype=ctypes.c_int
        self.lib.marisa_tree_rsd_last_error.restype=ctypes.c_char_p

    def evaluate(self,vectors,power,transfer,bias,rsd,reconstruction=None):
        v=np.ascontiguousarray(vectors,dtype=float)
        if v.ndim==2:v=v[None]
        if v.ndim!=3 or v.shape[1:]!=(3,3) or not len(v):raise ValueError('vectors must be (N,3,3), N>0')
        p=np.ascontiguousarray(np.broadcast_to(power,(len(v),3)),dtype=float)
        m=np.ascontiguousarray(np.broadcast_to(transfer,(len(v),3)),dtype=float)
        b=np.array([getattr(bias,k) for k in ('b1','b2','bK2','bphi','bphidelta','bphi2')])
        r=np.array([rsd.f_growth,*rsd.los,0.,0.])
        rec=np.array([0.,1.,0.,0.,0.,0.])
        if reconstruction is not None:
            if reconstruction.denominator_mode!='bias_only':
                raise NotImplementedError('v1 native implements same_shift_no_linear_rsd_removal only')
            rec=np.array([reconstruction.smoothing_radius,reconstruction.bias_recon,
                reconstruction.cell_size,reconstruction.cic_window_power,
                reconstruction.bphi_recon,reconstruction.f_nl_recon_ratio],dtype=float)
        out=np.empty((len(v),39))
        status=self.lib.marisa_tree_rsd_batch(len(v),v,p,m,b,r,rec,int(reconstruction is not None),0,out)
        if status:raise ValueError(self.lib.marisa_tree_rsd_last_error().decode())
        return SimpleNamespace(total=out[:,:3],gravitational_and_bias=out[:,3:6],
            primordial=out[:,6:9],pair_terms=out[:,9:18].reshape(-1,3,3),
            z1=out[:,18:27].reshape(-1,3,3),z2=out[:,27:36].reshape(-1,3,3),
            gaussian_stochastic=out[:,36:39],raw=out)

    def project_multipole(self,lengths,power,transfer,bias,rsd,ell=0,n_mu=18,n_phi=20,reconstruction=None):
        if int(ell)!=ell or ell<0 or n_mu<2 or n_phi<2:raise ValueError('invalid quadrature/multipole')
        if reconstruction is not None and reconstruction.cell_size>0 and reconstruction.cic_window_power>0:
            raise ValueError('Cartesian CIC requires exact modes or full SO(3) projection')
        mu,wm=np.polynomial.legendre.leggauss(n_mu)
        ph,wp=np.polynomial.legendre.leggauss(n_phi);ph=np.pi*(ph+1);wp=np.pi*wp
        v=np.array([reference.triangle_vectors_from_shape_orientation(*lengths,x,y,los=rsd.los)
            for x in mu for y in ph])
        lc=np.zeros(ell+1);lc[ell]=1.
        weights=((2*ell+1)/(4*np.pi)*wm*np.polynomial.legendre.legval(mu,lc))[:,None]*wp[None,:]
        return weights.ravel()@self.evaluate(v,power,transfer,bias,rsd,reconstruction).raw

    def project_exact_lattice(self,vectors,power,transfer,bias,rsd,weights=None,
                              reconstruction=None,normalization=None):
        """实际闭合 mode-list 的加权平均；normalization 可保留 estimator 的零模计数。"""
        x=self.evaluate(vectors,power,transfer,bias,rsd,reconstruction)
        w=np.ones(len(x.raw)) if weights is None else np.asarray(weights,dtype=float)
        if w.shape!=(len(x.raw),) or not np.isfinite(w).all() or (w<0).any():raise ValueError('invalid mode weights')
        den=float(w.sum() if normalization is None else normalization)
        if not np.isfinite(den) or den<=0 or den+1e-12<w.sum():raise ValueError('invalid normalization')
        return w@x.raw/den
