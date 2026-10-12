"""模块大纲：native Z1/Z2 的可分离源分解；exact lattice 两腿 FFT 投影；独立点验。

此投影不逐一枚举 O(Nmode²) 的高 k mode pairs；闭合由 FFT 精确完成。
分解表同时供独立 point audit 使用，必须逐项回归 native G/L/Q，不能自行换模型。
三腿零模源为零、两腿分母保持完整 shell mode counts；没有第三腿 kmax cut。
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.fft import ifftn
from rsd_b000_contract import F_GROWTH, bias, load_spectrum

@dataclass(frozen=True)
class Field:
    root:str
    axes:tuple=()
    inv2:bool=False
    invM:bool=False
    def op(self,axes=(),inv2=False,invM=False):
        return Field(self.root,tuple(sorted(self.axes+tuple(axes))),self.inv2 or inv2,self.invM or invM)

A=Field('A'); D=Field('D'); T=Field('T'); ONE=Field('one'); MU2=Field('mu2'); KZ=Field('kz')

def source_terms(b1,f_growth=F_GROWTH):
    """每项为 nuisance[base,b2,K2] 权重、fNL 次数偏移、三个 Fourier factors。

    bias recipe 中 bphidelta 的 b2 导数为 2delta_c，bphi2 为 4delta_c²。
    """
    h=bias(b1);f=f_growth;terms=[]
    def add(weight,shift,x,y,z):terms.append((np.array(weight,dtype=float),shift,(x,y,z)))
    add((10*b1/7,1.,-2/3),0,A,A,ONE)
    add((6*f/7,0,0),0,A,A,MU2)
    for a in range(3):
        for x,y in [(A.op((a,),inv2=True),A.op((a,))),(A.op((a,)),A.op((a,),inv2=True))]:
            add((b1,0,0),0,x,y,ONE);add((f,0,0),0,x,y,MU2)
        add((h.bphi,0,0),1,A.op((a,),invM=True),A.op((a,),inv2=True),ONE)
        add((h.bphi,0,0),1,A.op((a,),inv2=True),A.op((a,),invM=True),ONE)
        for b in range(a,3):
            multiplicity=1 if a==b else 2
            x=A.op((a,b),inv2=True)
            add((multiplicity*4*b1/7,0,multiplicity*2),0,x,x,ONE)
            add((multiplicity*8*f/7,0,0),0,x,x,MU2)
    for x,y in [(A.op(invM=True),A),(A,A.op(invM=True))]:
        add((h.bphidelta,2*1.686,0),1,x,y,ONE)
    add((h.bphi2,4*1.686**2,0),2,A.op(invM=True),A.op(invM=True),ONE)
    # Mapping: par=-k3z；完整 finite-PNG Z1² 是 D，不得只保留 Gaussian Z1。
    add((-f,0,0),0,A.op((2,),inv2=True),D,KZ)
    add((-f,0,0),0,D,A.op((2,),inv2=True),KZ)
    add((2,0,0),1,A.op(invM=True),A.op(invM=True),T)
    return terms

class Fields:
    """point 和 FFT 共用唯一 spectral factor 定义。"""
    def __init__(self,vectors,p,m,b1,f_growth):
        self.v=vectors;self.p=p;self.m=m;self.b1=b1;self.f=f_growth
        self.k2=np.sum(vectors*vectors,axis=-1);self.active=self.k2>0
        self.invk2=np.divide(1.,self.k2,out=np.zeros_like(self.k2),where=self.active)
        self.u=np.divide(1.,m,out=np.zeros_like(m),where=self.active)
        self.mu2=vectors[...,2]**2*self.invk2
        self.z0=b1+f_growth*self.mu2;self.z1=bias(b1).bphi*self.u

    def degrees(self,field):return range(3 if field.root=='D' else (2 if field.root in ('A','T') else 1))

    def values(self,field,degree):
        root=field.root
        if root=='A':value=self.p*(self.z0 if degree==0 else self.z1)
        elif root=='D':value=self.p*(self.z0*self.z0 if degree==0 else (2*self.z0*self.z1 if degree==1 else self.z1*self.z1))
        elif root=='T':value=self.m*(self.z0 if degree==0 else self.z1)
        elif root=='one':value=self.active.astype(float)
        elif root=='mu2':value=self.mu2.copy()
        elif root=='kz':value=self.v[...,2].copy()
        elif root=='P':value=self.p.copy()
        elif root=='Pmu2':value=self.p*self.mu2
        elif root=='Pmu4':value=self.p*self.mu2*self.mu2
        else:raise ValueError(root)
        for axis in field.axes:value=value*self.v[...,axis]
        if field.inv2:value=value*self.invk2
        if field.invM:value=value*self.u
        return np.where(self.active,value,0.)

    def integral(self,x,dx,y,dy,z,dz):
        xx=self.values(x,dx);yy=self.values(y,dy);zz=self.values(z,dz)
        return xx[:,0]*yy[:,1]*zz[:,2]+zz[:,0]*xx[:,1]*yy[:,2]+yy[:,0]*zz[:,1]*xx[:,2]

def integrate_design(fields):
    result=None
    for weights,shift,(x,y,z) in source_terms(fields.b1,fields.f):
        for dx,dy,dz in product(fields.degrees(x),fields.degrees(y),fields.degrees(z)):
            degree=shift+dx+dy+dz
            if degree>2:continue
            value=fields.integral(x,dx,y,dy,z,dz)
            if result is None:result=np.zeros((3,3)+np.shape(value))
            result[:,degree]+=weights.reshape((3,)+(1,)*np.ndim(value))*value
    return result

class FFTProjection(Fields):
    def __init__(self,meshsize,edges,pair,b1,f_growth=F_GROWTH):
        n=int(meshsize);kf=2*np.pi/1000.;axis=np.rint(np.fft.fftfreq(n)*n)*kf
        grids=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True)
        k2=sum(g*g for g in grids);k=np.sqrt(k2);active=k>0
        p=np.zeros_like(k);m=np.ones_like(k);p[active],m[active]=load_spectrum()(k[active])
        vectors=np.empty(k.shape+(3,))
        for a,g in enumerate(grids):vectors[...,a]=g
        super().__init__(vectors,p,m,b1,f_growth)
        self.n=n;self.ncell=n**3;self.cache={};self.cache_peak_bytes=0
        from rsd_bonly_common import fft
        labels=fft.assign_shells(k,edges);i,j=map(int,pair)
        self.masks={i:labels==i,j:labels==j};self.slots=(i,j,None)
        self.counts=[int(self.masks[s].sum()) for s in (i,j)];self.denominator=np.prod(self.counts)
        if 2*max(edges[i+1],edges[j+1])>=np.pi*n/1000.:
            raise ValueError('theory FFT would wrap closing leg; increase meshsize')

    def transform(self,field,degree,slot):
        shell=self.slots[slot];key=(field,degree,shell)
        if key not in self.cache:
            value=self.values(field,degree)
            if shell is not None:value*=self.masks[shell]
            transformed=ifftn(value,workers=1,overwrite_x=True)
            parity=(len(field.axes)+(field.root=='kz'))%2
            # 实 even function 的 IFFT 为实；odd function 的 IFFT 为 i*实。
            self.cache[key]=np.ascontiguousarray((transformed.imag if parity else transformed.real)*self.ncell)
            self.cache_peak_bytes=max(self.cache_peak_bytes,sum(a.nbytes for a in self.cache.values()))
        return self.cache[key]

    def integral(self,x,dx,y,dy,z,dz):
        odd=sum((len(field.axes)+(field.root=='kz'))%2 for field in (x,y,z))
        if odd%2:raise ValueError('unexpected parity-odd B000 source')
        sign=-1. if odd%4==2 else 1.
        total=0.
        for fields in (((x,dx),(y,dy),(z,dz)),((z,dz),(x,dx),(y,dy)),((y,dy),(z,dz),(x,dx))):
            arrays=[self.transform(field,degree,slot) for slot,(field,degree) in enumerate(fields)]
            total+=float(np.einsum('ijk,ijk,ijk->',*arrays,optimize=False))/self.ncell
        return sign*total/self.denominator

    def stochastic(self):
        # 返回 sum Plin、sum Plin mu²、sum Plin mu⁴、valid fraction。
        return np.array([self.integral(Field(root),0,ONE,0,ONE,0) for root in ('P','Pmu2','Pmu4')]+[
                         self.integral(ONE,0,ONE,0,ONE,0)/3])
