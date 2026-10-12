"""模块大纲：带标签 marked 点过程的精确矩与有限 random 的 factorial 位移响应。

这是独立 audit 引擎，不是 production likelihood。每个标记族具有固定点数、
固定权重和可不同的正弦强度响应；共享随机相位，random 族均匀且独立。
不把 STO 实际 halo 的排斥/环境响应假定为 Poisson，也不使用 n_eff 代替 F2。
"""
from functools import lru_cache
from itertools import product
import numpy as np
from audit_post_rsd_poisson_response import partitions


def factor(kind,power,mode):return (kind,int(power),tuple(int(x) for x in mode))


class MarkedPointMoments:
    def __init__(self,counts,weights,amplitudes,signal_mode,nrandom):
        counts=np.asarray(counts,dtype=int);weights=np.asarray(weights,dtype=float)
        amplitudes=np.asarray(amplitudes,dtype=float)
        if (counts.ndim!=1 or weights.shape!=counts.shape or amplitudes.shape!=counts.shape
                or np.any(counts<4) or np.any(weights<=0) or not np.isfinite(weights).all()
                or not np.isfinite(amplitudes).all() or np.any(np.abs(amplitudes)>=1) or nrandom<4):
            raise ValueError('invalid marked iid point process')
        self.ndata=int(counts.sum());self.ngroup=len(counts);self.nrandom=int(nrandom)
        total=float(counts@weights)
        self.counts=tuple(counts.tolist()+[self.nrandom])
        self.c=tuple((weights/total).tolist()+[-1/self.nrandom])
        self.amplitudes=tuple(amplitudes.tolist()+[0.])
        self.signal_mode=tuple(int(x) for x in signal_mode)
        self.N2=float(np.sum(counts*(weights/total)**2));self.N3=float(np.sum(counts*(weights/total)**3))
        self.signal_amplitude=float(np.sum(counts*weights*amplitudes)/total)

    def expect(self,*factors):
        return self._expect(tuple(sorted(factors)))

    @lru_cache(None)
    def _expect(self,factors):
        # F1(0) 由归一化精确为零；F2(0) 与 D1(0) 必须保留。
        if any(kind=='F' and power==1 and mode==(0,0,0) for kind,power,mode in factors):return 0.
        if not 1<=len(factors)<=4:raise ValueError('only moments of order one to four are supported')
        result=0.
        for blocks in partitions(list(range(len(factors)))):
            choices=[];phase=0
            for block in blocks:
                wave=tuple(sum(factors[i][2][j] for i in block) for j in range(3))
                degree=sum(factors[i][1] for i in block)
                groups=range(self.ngroup) if any(factors[i][0]=='D' for i in block) else range(self.ngroup+1)
                if wave==(0,0,0):single=[(g,self.c[g]**degree) for g in groups]
                elif wave==self.signal_mode:
                    phase+=1;single=[(g,self.c[g]**degree*self.amplitudes[g]/2) for g in groups if g<self.ngroup]
                elif wave==tuple(-x for x in self.signal_mode):
                    phase-=1;single=[(g,self.c[g]**degree*self.amplitudes[g]/2) for g in groups if g<self.ngroup]
                else:single=[]
                if not single:break
                choices.append(single)
            if len(choices)!=len(blocks) or phase!=0:continue
            for assignment in product(*choices):
                used=[0]*(self.ngroup+1);term=1.
                for group,value in assignment:
                    term*=value*(self.counts[group]-used[group]);used[group]+=1
                result+=term
        return result


def factorial_shift_response(triangle,shift_modes,shift_response,expect):
    """显式区分 D1(shift source)、F1(data-random)、F2(权重平方和)。"""
    v=np.asarray(triangle,dtype=int);qall=np.asarray(shift_modes,dtype=int)
    if v.shape!=(3,3) or np.any(v.sum(axis=0)!=0) or np.any(np.sum(v*v,axis=1)==0):raise ValueError('nonzero closed triangle required')
    if np.any(np.sum(qall*qall,axis=1)==0):raise ValueError('shift zero mode forbidden')
    raw=0.;self_term=0.
    for leg in range(3):
        others=[a for a in range(3) if a!=leg];k=v[leg]
        for q,r in zip(qall,shift_response):
            geom=np.dot(k,q)/np.dot(q,q)*r;src=factor('D',1,q)
            raw-=geom*expect(src,factor('F',1,k-q),*(factor('F',1,v[a]) for a in others))
            self_term+=geom*expect(src,factor('F',2,-k-q),factor('F',1,k))
            self_term-=geom*expect(factor('F',2,-k),src,factor('F',1,k-q))
    return raw-self_term,raw,self_term


class ActualMarkedFields:
    def __init__(self,positions,weights,random_positions):
        self.positions=np.asarray(positions);self.randoms=np.asarray(random_positions)
        self.cd=np.asarray(weights,dtype=float)/np.sum(weights);self.cr=-1/len(self.randoms)

    def field(self,item):
        kind,power,mode=item;k=np.asarray(mode)
        if kind=='F' and power==1 and np.all(k==0):return 0.
        value=np.exp(-2j*np.pi*(self.positions@k))@(self.cd**power)
        if kind=='F':value+=self.cr**power*np.sum(np.exp(-2j*np.pi*(self.randoms@k)))
        return value

    def expect(self,*items):return np.prod([self.field(item) for item in items])
