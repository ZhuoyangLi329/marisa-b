"""模块大纲：只计算可达 closing modes；按已验证的 LOS=z/CIC 网格对称折叠阻尼。

不改变 harmonic 函数、source、FFT 或 shell denominator。只去掉不可能参与
当前两腿 pair 的无效求值，使用独立符号反射和 x/y 互换对称减少重复方向。
"""
from dataclasses import dataclass,field
import time
import numpy as np
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_projection import IRFFTProjection


@dataclass(frozen=True)
class GridHarmonics(HarmonicDisplacement):
    k_bound: float
    statistics: dict=field(default_factory=dict,compare=False)

    def damping(self,vectors,*,blocksize=8192):
        start=time.time();v=np.asarray(vectors,dtype=float);flat=v.reshape(-1,3);kf=2*np.pi/1000.
        los=np.asarray(self.physical_parameters['los'],dtype=float);los/=np.linalg.norm(los)
        if not np.allclose(np.abs(los),[0,0,1],rtol=0,atol=1e-14):raise ValueError('symmetry fold only supports LOS=z')
        if self.physical_parameters['n_phi']%4:raise ValueError('angular rule does not preserve transverse mesh symmetry')
        if not np.isfinite(self.k_bound) or self.k_bound<=0:raise ValueError('invalid exact closing support')
        k2=np.sum(flat*flat,axis=1);ids=np.flatnonzero((k2>0)&(k2<=(self.k_bound*(1+2e-14))**2))
        modes=flat[ids]/kf;integer=np.rint(modes)
        if np.max(np.abs(modes-integer),initial=0)>1e-10:raise ValueError('grid optimization requires original integer Fourier modes')
        canonical=np.abs(integer.astype(np.int64));canonical[:,:2]=np.sort(canonical[:,:2],axis=1)
        unique,inverse=np.unique(canonical,axis=0,return_inverse=True)
        values=super().damping(unique.astype(float)*kf,blocksize=blocksize)
        out=np.zeros((len(flat),3));out[:,0]=1.;out[ids]=values[inverse]
        self.statistics.update(total_grid_modes=len(flat),reachable_nonzero_modes=len(ids),
                               distinct_symmetry_modes=len(unique),elapsed_seconds=time.time()-start,
                               exact_support_bound_h_Mpc=self.k_bound,fold='independent sign reflections and x/y swap')
        return out.reshape(v.shape[:-1]+(3,))


class FastIRFFTProjection(IRFFTProjection):
    def __init__(self,meshsize,edges,pair,b1,f_growth,*,p_halo,reconstruction,power_nw,harmonics):
        edges=np.asarray(edges,dtype=float);i,j=map(int,pair)
        view=GridHarmonics(harmonics.coefficients,harmonics.lmax,harmonics.r_bao,
                          harmonics.physical_parameters,float(edges[i+1]+edges[j+1]))
        super().__init__(meshsize,edges,pair,b1,f_growth,p_halo=p_halo,reconstruction=reconstruction,
                         power_nw=power_nw,harmonics=view)
        self.damping_grid_statistics=dict(view.statistics)
