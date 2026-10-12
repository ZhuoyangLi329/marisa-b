"""模块大纲：B-only 冻结合同；共享谱/PBS；严格 lattice 投影；多项式求值。

P_lin 仅是理论输入和自收缩/随机项形状，绝不是 P 数据似然或 b1 先验。
所有新数据、协方差与拟合产物独立于旧单盒 P+B 诊断。
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from tree_png_rsd_native import NativeTreeRSD, HaloBiasV1, RsdConfig

PART = Path(__file__).resolve().parents[1]
OUT = PART/'outputs/pre_rsd_bonly_v1'
sys.path.insert(0, str(PART/'code/vendor'))
import quijote_strict_triangle_fft_estimator as fft
TABLE = Path('/pscratch/sd/l/lzy/reconstruction-png/old_doc_codes/archive_20260615_diagnostics_cleanup/figures/diagnostics/quijote_dm_png_z0p5_marisa_b_local_png_1loop/marisa_b_local_png_1loop_b000_weighted_all120_nmu96_eps1e3_plin_pphi_m_z0.5.dat')
TABLE_HASH = 'c748ba2bdd217581071513fe918d93e5a08b48390451c69af60159d3dc73feeb'
F_GROWTH = .7625790037720044
P_CAL = 1.179
EDGES = np.array([.01, .02, .03, .04, .055, .07, .08])
CUTS = (.04, .055, .07, .08)
NAMES = ('fNL', 'b1', 'b2_native', 'bK2', 'A_stoch', 'S_stoch')
BOUNDS = np.array([[-500., 500.], [.5, 4.], [-4., 6.], [-4., 4.], [-10., 10.], [-20., 20.]])
BNODES = 2.25 + 1.75*np.cos((np.arange(4)+.5)*np.pi/4)
RAW_ROOT = Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5')
OLD_POS = Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)

def bias(b1, b2=0., bK2=0.):
    # 与第三部分和原单盒诊断使用同一个 PBS recipe。
    dc=1.686; bp=2*dc*(b1-P_CAL); lag=b2-8*(b1-1)/21
    return HaloBiasV1(b1,b2,bK2,bp,bp+2*(dc*lag-b1+1),4*dc*(dc*lag-2*(b1-1)))

def load_spectrum():
    if digest(TABLE) != TABLE_HASH: raise ValueError('linear spectrum hash mismatch')
    table=np.loadtxt(TABLE)
    ps=CubicSpline(np.log(table[:,0]),table[:,1]); ms=CubicSpline(np.log(table[:,0]),table[:,3])
    def sample(k):
        if np.min(k)<table[0,0] or np.max(k)>table[-1,0]: raise ValueError('spectrum extrapolation not allowed')
        return ps(np.log(k)), ms(np.log(k))
    return sample

def cut_mask(triangles, cut):
    return np.all(EDGES[1:][triangles] <= cut+1e-12, axis=1)

def prediction(values, coefficients, noise):
    """树源严格至 fNL²；b1 三次、b2/K2 一次，已在独立 native 点验证。"""
    values=np.atleast_2d(values)
    f,b,b2,bk,a,s=values.T
    cb=np.polynomial.chebyshev.chebvander((b-2.25)/1.75,3)
    nu=np.column_stack((np.ones(len(b)),b2,bk))
    ff=np.column_stack((np.ones(len(f)),f,f*f))
    features=(cb[:,:,None,None]*nu[:,None,:,None]*ff[:,None,None,:]).reshape(len(b),-1)
    return features@coefficients.reshape(-1,coefficients.shape[-1]) + np.column_stack((a,s))@noise

def mode_geometry(n=128):
    geometry=fft.build_geometry(n,1000.,EDGES)
    axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    modes=[grid[geometry.shell_index==i] for i in range(geometry.nshell)]
    return geometry,modes

def triangle_blocks(geometry, modes, triangle, batch=24):
    """真正三腿闭合，不使用代表 k，也不对高 k 作 modulo alias。"""
    i,j,k=map(int,triangle); n=geometry.meshsize
    for start in range(0,len(modes[i]),batch):
        aa=modes[i][start:start+batch]; bb=modes[j]
        first=np.repeat(aa,len(bb),axis=0); second=np.tile(bb,(len(aa),1)); third=-first-second
        inside=np.all(np.abs(third)<n//2,axis=1)
        labels=np.full(len(third),-1,dtype=int)
        labels[inside]=geometry.shell_index[tuple((third[inside]%n).T)]
        select=labels==k
        if np.any(select): yield np.stack((first[select],second[select],third[select]),axis=1)*2*np.pi/geometry.boxsize_mpc_h

def project_triangle(geometry,modes,index,bias_values,sample=None):
    """多个 bias 点共享闭合列表；只累加 native 的 G/L/Q 与 Gaussian noise。"""
    sample=load_spectrum() if sample is None else sample
    native=NativeTreeRSD(); total=np.zeros((len(bias_values),6)); count=0
    for vectors in triangle_blocks(geometry,modes,geometry.triangles[index]):
        p,m=sample(np.linalg.norm(vectors,axis=2))
        for a,parameters in enumerate(bias_values):
            result=native.evaluate(np.ascontiguousarray(vectors),p,m,bias(*parameters),RsdConfig(F_GROWTH))
            total[a,:3]+=result.total.sum(axis=0)
            total[a,3:]+=result.gaussian_stochastic.sum(axis=0)
        count+=len(vectors)
    if count != geometry.triangle_counts[index]: raise AssertionError((index,count,geometry.triangle_counts[index]))
    return total/count
