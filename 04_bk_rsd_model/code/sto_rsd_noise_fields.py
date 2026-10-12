"""模块大纲：实际STO输入场的带标记Fourier诊断；不替换production estimator。

流式pmesh TSC和三重interlacing仅用于新的matter/weight-power混合谱检查。
必须做128/256数值比较，不能把matter投影残差自动认作纯随机噪声。
Gadget粒子velocity与FoF GroupVel的存储单位不同，禁止复用FoF转换。
"""
from itertools import product
import numpy as np


def gadget_rsd_positions(coordinates_kpc_h, stored_velocity, *, a, omega_m,
                         omega_lambda, boxsize=1000.):
    """Gadget HDF5: v_phys=sqrt(a)*v_stored; ds=v_phys/(a H_h)。"""
    x=np.asarray(coordinates_kpc_h,dtype=float)/1000.
    v=np.asarray(stored_velocity,dtype=float)
    if (x.ndim!=2 or x.shape[1]!=3 or v.shape!=x.shape
        or not np.isfinite(x).all() or not np.isfinite(v).all()
        or not 0<a<=1 or not np.isfinite(a+omega_m+omega_lambda)
        or omega_m<=0 or omega_lambda<0 or boxsize<=0):
        raise ValueError('explicit finite Gadget coordinates/velocity/cosmology required')
    hh=100.*np.sqrt(omega_m/a**3+omega_lambda)
    result=x.copy();result[:,2]=(x[:,2]+v[:,2]/(np.sqrt(a)*hh))%boxsize
    return result


def marked_moments(weights,volume=1e9):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('finite weights with positive sum required')
    return np.array([volume*np.sum(w*w)/w.sum()**2,
                     volume**2*np.sum(w*w*w)/w.sum()**3])


def mode_geometry(meshsize,boxsize,k_edges,mu_edges):
    """用显式全complex FFT modes；±k各自计数，不能当成独立样本数。"""
    n=int(meshsize);ke=np.asarray(k_edges,dtype=float);me=np.asarray(mu_edges,dtype=float)
    if (n<8 or n%2 or ke.ndim!=1 or np.any(np.diff(ke)<=0) or ke[0]<=0
        or me.ndim!=1 or np.any(np.diff(me)<=0) or me[0]!=0 or me[-1]!=1):
        raise ValueError('even mesh and explicit positive k/absolute-mu bins required')
    axis=np.fft.fftfreq(n)*n
    xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    k=2*np.pi/boxsize*np.linalg.norm(xyz,axis=1)
    mu=np.divide(np.abs(xyz[:,2]),np.linalg.norm(xyz,axis=1),out=np.zeros_like(k),where=k>0)
    shell=np.searchsorted(ke,k,side='right')-1
    mubin=np.minimum(np.searchsorted(me,mu,side='right')-1,len(me)-2)
    valid=(shell>=0)&(shell<len(ke)-1)&np.all(xyz!=-n//2,axis=1)
    indices=np.flatnonzero(valid);labels=shell[valid]*(len(me)-1)+mubin[valid]
    counts=np.bincount(labels,minlength=(len(ke)-1)*(len(me)-1))
    if np.any(counts==0):raise ValueError('empty diagnostic k-mu cell')
    return dict(indices=indices,integer_modes=xyz[valid].astype(int),k=k[valid],mu=mu[valid],
                labels=labels,counts=counts,nk=len(ke)-1,nmu=len(me)-1)


def projected_spectra(fields,geometry,volume=1e9):
    """保存完整mixed spectra；不减matter Poisson shot，也不假设Nraw=N2。"""
    f=np.asarray(fields,dtype=complex);labels=geometry['labels'];counts=geometry['counts']
    if f.ndim!=2 or f.shape[1]!=len(labels) or not np.isfinite(f).all():
        raise ValueError('finite explicit field-by-mode array required')
    result=np.empty((len(f),len(f),len(counts)),dtype=complex)
    for i,j in product(range(len(f)),repeat=2):
        values=volume*f[i]*f[j].conj()
        result[i,j]=(np.bincount(labels,weights=values.real,minlength=len(counts))
            +1j*np.bincount(labels,weights=values.imag,minlength=len(counts)))/counts
    return result.reshape(len(f),len(f),geometry['nk'],geometry['nmu'])


def projection_residual(power,halo_index,matter_index):
    """操作性halo-matter投影残差；有限mode回归/非线性偏差尚未校正。"""
    p=np.asarray(power);mm=p[matter_index,matter_index].real
    if np.any(mm<=0):raise ValueError('positive measured raw matter auto-power required')
    return p[halo_index,halo_index].real-np.abs(p[halo_index,matter_index])**2/mm


class StreamingMarkedFourier:
    def __init__(self,meshsize,*,boxsize=1000.,interlacing=3,comm=None):
        from pmesh.pm import ParticleMesh
        if interlacing!=3:raise ValueError('diagnostic uses explicit TSC/interlacing3')
        self.n=int(meshsize);self.boxsize=float(boxsize);self.offsets=np.arange(3)/3.
        self.pm=ParticleMesh(Nmesh=[self.n]*3,BoxSize=self.boxsize,dtype='f8',resampler='tsc',comm=comm)
        self.meshes=[self.pm.create(type='real',value=0.) for _ in self.offsets]
        self.total=0.;self.count=0

    def add(self,positions,weights=None):
        x=np.asarray(positions,dtype=float)
        w=np.ones(len(x)) if weights is None else np.asarray(weights,dtype=float)
        if x.ndim!=2 or x.shape[1]!=3 or w.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(w).all():
            raise ValueError('finite positions and per-particle marks required')
        for offset,mesh in zip(self.offsets,self.meshes):
            position=(x-offset*self.boxsize/self.n)%self.boxsize
            self.pm.paint(position,mass=w,hold=True,out=mesh)
        self.total+=float(w.sum());self.count+=len(x)

    def fourier(self):
        if self.total<=0:raise ValueError('positive total field weight required')
        freq=np.fft.fftfreq(self.n)
        fx,fy,fz=np.meshgrid(freq,freq,freq,indexing='ij',sparse=True)
        result=np.zeros((self.n,)*3,dtype=complex)
        for offset,mesh in zip(self.offsets,self.meshes):
            arr=np.asarray(mesh.value)
            if arr.shape!=(self.n,)*3:raise ValueError('one-rank diagnostic only')
            if not np.isclose(arr.sum(),self.total,rtol=2e-12,atol=1e-12):raise ValueError('paint mass not conserved')
            result+=np.fft.fftn(arr)/self.total*np.exp(-2j*np.pi*offset*(fx+fy+fz))/3
        result/=(np.sinc(fx)*np.sinc(fy)*np.sinc(fz))**3
        result[0,0,0]=0.
        result[self.n//2,:,:]=0.;result[:,self.n//2,:]=0.;result[:,:,self.n//2]=0.
        return result
