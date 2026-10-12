"""模块大纲：相关成簇marked点过程的一至四点精确矩，保留真实粒子self标签。

独立parent服从单正弦强度；同parent的child位置/权重相关，故child不是Poisson。
shift仍来自data一阶权重场；F2/F3按child权重幂，而非parent总权重幂扣除。
允许signed weights；这是独立算子审计，不将此cluster模型认作真实STO闭合。
"""
from functools import lru_cache
from itertools import product
import numpy as np
from audit_post_rsd_poisson_response import partitions


class ClusteredMarkedMoments:
    def __init__(self,counts,child_weights,child_offsets,amplitudes,signal_mode,nrandom):
        counts=np.asarray(counts,dtype=int);amplitudes=np.asarray(amplitudes,dtype=float)
        mode=np.asarray(signal_mode)
        if (counts.ndim!=1 or not len(counts) or np.any(counts<1)
            or amplitudes.shape!=counts.shape or not np.isfinite(amplitudes).all()
            or np.any(np.abs(amplitudes)>=1) or mode.shape!=(3,)
            or not np.isfinite(mode).all() or np.any(mode!=np.rint(mode))
            or not np.any(mode) or int(nrandom)!=nrandom or nrandom<1
            or len(child_weights)!=len(counts) or len(child_offsets)!=len(counts)):
            raise ValueError('invalid parent counts/intensities/mode/random count')
        weights=[];offsets=[]
        for w,x in zip(child_weights,child_offsets):
            w=np.asarray(w,dtype=float);x=np.asarray(x,dtype=float)
            if (w.ndim!=1 or not len(w) or x.shape!=(len(w),3)
                or not np.isfinite(w).all() or not np.isfinite(x).all()):
                raise ValueError('finite child weights and 3D offsets required')
            weights.append(w);offsets.append(x)
        total=float(sum(n*w.sum() for n,w in zip(counts,weights)))
        if not np.isfinite(total) or total==0:raise ValueError('nonzero finite signed normalization required')
        self.ngroup=len(counts);self.nrandom=int(nrandom)
        self.counts=tuple(counts.tolist()+[self.nrandom])
        self.c=tuple(tuple((w/total).tolist()) for w in weights)
        self.offsets=tuple(tuple(tuple(a) for a in x) for x in offsets)
        self.amplitudes=tuple(amplitudes.tolist()+[0.])
        self.signal_mode=tuple(mode.astype(int).tolist())
        self.ndata=int(sum(n*len(w) for n,w in zip(counts,weights)))
        self.N2=float(sum(n*np.sum((w/total)**2) for n,w in zip(counts,weights)))
        self.N3=float(sum(n*np.sum((w/total)**3) for n,w in zip(counts,weights)))

    @lru_cache(None)
    def child_factor(self,group,power,mode):
        if group==self.ngroup:return (-1/self.nrandom)**power
        c=np.asarray(self.c[group]);x=np.asarray(self.offsets[group]);k=np.asarray(mode)
        return complex(np.sum(c**power*np.exp(-2j*np.pi*(x@k))))

    def expect(self,*factors):
        return self._expect(tuple(sorted(factors)))

    @lru_cache(None)
    def _expect(self,factors):
        if not 1<=len(factors)<=4:raise ValueError('only moments of order one to four are supported')
        for kind,power,mode in factors:
            if kind not in ('D','F') or power not in (1,2,3) or len(mode)!=3:
                raise ValueError('invalid labeled field')
        if any(kind=='F' and power==1 and mode==(0,0,0) for kind,power,mode in factors):return 0j
        result=0j
        for blocks in partitions(list(range(len(factors)))):
            choices=[];phase=0
            for block in blocks:
                wave=tuple(sum(factors[i][2][j] for i in block) for j in range(3))
                groups=range(self.ngroup) if any(factors[i][0]=='D' for i in block) else range(self.ngroup+1)
                if wave==(0,0,0):sign=0
                elif wave==self.signal_mode:sign=1;phase+=1
                elif wave==tuple(-a for a in self.signal_mode):sign=-1;phase-=1
                else:break
                single=[]
                for g in groups:
                    value=1. if not sign else self.amplitudes[g]/2
                    for i in block:
                        _,power,mode=factors[i];value*=self.child_factor(g,power,mode)
                    if value!=0:single.append((g,value))
                if not single:break
                choices.append(single)
            if len(choices)!=len(blocks) or phase!=0:continue
            for assignment in product(*choices):
                used=[0]*(self.ngroup+1);term=1.+0j
                for group,value in assignment:
                    available=self.counts[group]-used[group]
                    if available<=0:term=0j;break
                    term*=value*available;used[group]+=1
                result+=term
        return result

    def residual_power(self,mode):
        """F1 power扣除实际child与random self；不等同于独立parent shot。"""
        from post_rsd_marked_moments import factor
        k=tuple(int(x) for x in mode)
        if k==(0,0,0):raise ValueError('nonzero residual-power mode required')
        value=self.expect(factor('F',1,k),factor('F',1,tuple(-a for a in k)))
        return value-self.N2-1/self.nrandom
