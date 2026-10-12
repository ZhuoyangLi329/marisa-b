"""模块大纲：interlaced CIC；归一化 Fourier field；固定 LOS P0/P2。

严格三边 B0 使用 vendor 中已验证的 FFT estimator；不把它称为旧两腿 B000。
"""
from itertools import product
import numpy as np

def catalog_fourier(position,meshsize=128,boxsize=1000.):
    position=np.asarray(position,dtype=float)
    if position.ndim!=2 or position.shape[1]!=3 or not len(position) or not np.isfinite(position).all():
        raise ValueError('invalid positions')
    n=int(meshsize);u=position/boxsize*n
    frequency=np.fft.fftfreq(n)
    gx,gy,gz=np.meshgrid(frequency,frequency,frequency,indexing='ij',sparse=True)
    total=np.zeros((n,n,n),dtype=complex)
    for offset in (0.,.5):
        coord=(u-offset)%n;lower=np.floor(coord).astype(int);frac=coord-lower
        mesh=np.zeros(n**3)
        for bits in product((0,1),repeat=3):
            ix=(lower+bits)%n
            weight=np.prod(np.where(np.array(bits)[None,:],frac,1.-frac),axis=1)
            np.add.at(mesh,(ix[:,0]*n+ix[:,1])*n+ix[:,2],weight)
        assert abs(mesh.sum()-len(position))<1e-9*len(position)
        total+=np.fft.fftn(mesh.reshape(n,n,n))/len(position)*np.exp(-2j*np.pi*offset*(gx+gy+gz))/2.
    window=(np.sinc(gx)*np.sinc(gy)*np.sinc(gz))**2
    total/=window;total[0,0,0]=0.
    # Nyquist phase 在 interlacing 后不自动 Hermitian；这些平面远离本例 kmax。
    if n%2==0:
        total[n//2,:,:]=0.;total[:,n//2,:]=0.;total[:,:,n//2]=0.
    return total

def power_multipoles(field,edges,nbar,boxsize=1000.):
    n=field.shape[0];axis=2*np.pi*np.fft.fftfreq(n,d=boxsize/n)
    kx,ky,kz=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True)
    k=np.sqrt(kx*kx+ky*ky+kz*kz);mu=np.divide(kz,k,out=np.zeros_like(k),where=k>0)
    raw=boxsize**3*np.abs(field)**2
    result=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        selected=(k>=lo)&(k<hi);count=int(selected.sum())
        if count==0:raise ValueError('empty P shell')
        w2=5.*(.5*(3*mu[selected]**2-1.))
        values=raw[selected]-1./nbar
        result.append([k[selected].mean(),count,values.mean(),np.mean(w2*values),
                       np.sqrt(2*np.mean(raw[selected]**2)/count),
                       np.sqrt(2*np.mean((w2*raw[selected])**2)/count)])
    return np.array(result)
