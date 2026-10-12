"""模块大纲：冻结 catalog 参数；同位移 post-RSD native ABI；显式 PNG bias 配方。

本入口不修改 pre API、参考包或旧共享库；新库由 build_post_rsd.sh 生成。
只支持 same-shift/bias-only、正定且不触发 clipping 的重构分母。
fNL_rec 是 catalog 运行设置，不是 MCMC 中变化的 fNL。
"""
from dataclasses import dataclass
import ctypes
from types import SimpleNamespace
import numpy as np
from tree_png_rsd_native import NativeTreeRSD, HaloBiasV1, PART


@dataclass(frozen=True)
class FixedReconstruction:
    smoothing_radius: float
    bias_recon: float
    p_recon: float
    fnl_recon: float
    cell_size: float = 0.
    cic_window_power: int = 0
    denominator_floor: float = 1.e-3
    alpha_table: tuple | None = None

    def __post_init__(self):
        values=(self.smoothing_radius,self.bias_recon,self.p_recon,self.fnl_recon,
                self.cell_size,self.denominator_floor,self.cic_window_power)
        if not all(np.isfinite(x) for x in values):
            raise ValueError('nonfinite frozen reconstruction setting')
        if (self.smoothing_radius<0 or self.bias_recon<=0 or self.cell_size<0
                or self.denominator_floor<0 or self.cic_window_power<0
                or int(self.cic_window_power)!=self.cic_window_power):
            raise ValueError('invalid frozen reconstruction setting')
        if self.alpha_table is not None:
            k,a=map(np.asarray,self.alpha_table)
            if (k.ndim!=1 or len(k)<2 or a.shape!=k.shape or not np.isfinite(k).all()
                    or not np.isfinite(a).all() or np.any(k<=0) or np.any(a<=0) or np.any(np.diff(k)<=0)):
                raise ValueError('invalid reconstruction alpha table')
            object.__setattr__(self,'alpha_table',(tuple(k.tolist()),tuple(a.tolist())))

    @property
    def bphi_recon(self):
        return 2.*1.686*(self.bias_recon-self.p_recon)

    def response(self,vectors,transfer_m):
        """W_rec/D_rec；优先实际 alpha 表的 log-k 线性插值，不声称含网格 alias。"""
        v=np.asarray(vectors,dtype=float);m=np.asarray(transfer_m,dtype=float)
        if v.shape[-1]!=3 or not np.isfinite(v).all() or not np.isfinite(m).all() or np.any(m<=0):
            raise ValueError('invalid vector/transfer in frozen shift')
        alpha=1./m
        if self.alpha_table is not None:
            ak,av=map(np.asarray,self.alpha_table);k=np.linalg.norm(v,axis=-1)
            if np.any(k<ak[0]) or np.any(k>ak[-1]):raise ValueError('reconstruction alpha extrapolation is forbidden')
            alpha=np.interp(np.log10(k),np.log10(ak),av)
        denom=self.bias_recon+self.fnl_recon*self.bphi_recon*alpha
        if np.any(~np.isfinite(denom)) or np.any(denom<=self.denominator_floor):
            raise ValueError('fixed reconstruction denominator is nonpositive or below floor')
        w=np.exp(-.5*np.sum(v*v,axis=-1)*self.smoothing_radius**2)
        if self.cell_size>0 and self.cic_window_power>0:
            w*=np.prod(np.sinc(v*self.cell_size/(2*np.pi)),axis=-1)**self.cic_window_power
        return w/denom


def pbs_bias(b1,b2=0.,bK2=0.,*,p_halo):
    """沿用当前 RSD PBS family，但显式给 p；不把 p_rec 当作 p_halo。"""
    dc=1.686;bp=2*dc*(b1-p_halo);lag=b2-8*(b1-1)/21
    return HaloBiasV1(b1,b2,bK2,bp,bp+2*(dc*lag-b1+1),4*dc*(dc*lag-2*(b1-1)))


class FixedPostRSD(NativeTreeRSD):
    def __init__(self,library=None):
        super().__init__(library or PART/'work/build/libmarisa_tree_rsd_fixedrec.so')
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.marisa_tree_rsd_batch_fixed.argtypes=[ctypes.c_size_t]+[ptr]*6+[
            ctypes.c_int,ctypes.c_double,ctypes.c_double,ptr]
        self.lib.marisa_tree_rsd_batch_fixed.restype=ctypes.c_int
        self.lib.marisa_tree_rsd_batch_response.argtypes=[ctypes.c_size_t]+[ptr]*7
        self.lib.marisa_tree_rsd_batch_response.restype=ctypes.c_int

    def evaluate(self,vectors,power,transfer,bias,rsd,reconstruction=None):
        if reconstruction is None:
            return super().evaluate(vectors,power,transfer,bias,rsd)
        if not isinstance(reconstruction,FixedReconstruction):
            raise TypeError('fixed catalog requires FixedReconstruction; linked eta path is not accepted')
        v=np.ascontiguousarray(vectors,dtype=float)
        if v.ndim==2:v=v[None]
        if v.ndim!=3 or v.shape[1:]!=(3,3) or not len(v):
            raise ValueError('vectors must be (N,3,3), N>0')
        p=np.ascontiguousarray(np.broadcast_to(power,(len(v),3)),dtype=float)
        m=np.ascontiguousarray(np.broadcast_to(transfer,(len(v),3)),dtype=float)
        b=np.array([getattr(bias,k) for k in ('b1','b2','bK2','bphi','bphidelta','bphi2')])
        r=np.array([rsd.f_growth,*rsd.los,0.,0.])
        c=reconstruction
        rec=np.array([c.smoothing_radius,c.bias_recon,c.cell_size,c.cic_window_power,c.bphi_recon,0.])
        out=np.empty((len(v),39))
        if c.alpha_table is None:
            status=self.lib.marisa_tree_rsd_batch_fixed(len(v),v,p,m,b,r,rec,0,c.fnl_recon,c.denominator_floor,out)
        else:
            response=np.ascontiguousarray(c.response(v,m))
            status=self.lib.marisa_tree_rsd_batch_response(len(v),v,p,m,b,r,response,out)
        if status:raise ValueError(self.lib.marisa_tree_rsd_last_error().decode())
        return SimpleNamespace(total=out[:,:3],gravitational_and_bias=out[:,3:6],primordial=out[:,6:9],
            pair_terms=out[:,9:18].reshape(-1,3,3),z1=out[:,18:27].reshape(-1,3,3),
            z2=out[:,27:36].reshape(-1,3,3),gaussian_stochastic=out[:,36:39],raw=out,
            stochastic_status='legacy pre raw bases only; post weighted stochastic NOT validated')
