"""模块大纲：完整一阶marked factorial响应的三/四点连通项分解。

不默认Gaussian、不设F2=N2*F1、不从N2生成真实N(k)。复用唯一标签响应。
保留从raw moments到cumulants的显式输入；仅是一阶shift恒等式，不发布STO闭合。
connected4可能含更高阶确定性trispectrum，不能全部命名为tree stochastic。
"""
from functools import lru_cache
from itertools import product
from math import factorial
import numpy as np
from audit_post_rsd_poisson_response import partitions
from post_rsd_marked_moments import factor,factorial_shift_response


def _items(items):
    if not 1<=len(items)<=4:raise ValueError('explicit moments/cumulants of orders1..4 required')
    result=[]
    for kind,power,mode in items:
        k=np.asarray(mode)
        if (kind not in ('D','F') or power not in (1,2,3) or k.shape!=(3,)
            or not np.isfinite(k).all() or np.any(k!=np.rint(k))):
            raise ValueError('finite labeled D/F weight-power fields and integer modes required')
        result.append((kind,int(power),tuple(k.astype(int))))
    return tuple(sorted(result))


def _finite(value):
    v=complex(value)
    if not np.isfinite(v):raise ValueError('nonfinite measured moment/cumulant')
    return v


class MomentCumulants:
    """Möbius反演；输入必须真实提供一到四点矩，不能缺省connected项为0。"""
    def __init__(self,moment):
        if not callable(moment):raise TypeError('explicit raw-moment callable required')
        self.moment=moment

    def __call__(self,*items):return self._value(_items(items))

    @lru_cache(None)
    def _value(self,items):
        total=0j
        for blocks in partitions(list(range(len(items)))):
            product=complex((-1)**(len(blocks)-1)*factorial(len(blocks)-1))
            for block in blocks:product*=_finite(self.moment(*(items[i] for i in block)))
            total+=product
        return _finite(total)


class CumulantMoments:
    """按集合分割重建raw矩；截断只用于分解，full必须显式调用κ3和κ4。"""
    def __init__(self,cumulant,max_order=4):
        if not callable(cumulant) or max_order not in (2,3,4):
            raise ValueError('explicit cumulant callable and order2/3/4 required')
        self.cumulant=cumulant;self.max_order=max_order

    def __call__(self,*items):return self._value(_items(items))

    @lru_cache(None)
    def _value(self,items):
        total=0j
        for blocks in partitions(list(range(len(items)))):
            if any(len(block)>self.max_order for block in blocks):continue
            term=1.+0j
            for block in blocks:term*=_finite(self.cumulant(*(items[i] for i in block)))
            total+=term
        return _finite(total)


class ConditionalUniformRandomMoments:
    """固定真实data，对有限独立uniform random精确求条件矩，不采样9百万random。

data_field(power,mode)返回每盒 sum((w/sum(w))^power exp(-ikx))。
F_j=data_j+(-1/Nr)^j sum_r exp(-ikxr)；D_j只有data。
仍是实际source的条件矩，尚未分离确定性与stochastic物理。
"""
    def __init__(self,data_field,nrandom):
        n=np.asarray(nrandom,dtype=float)
        if not callable(data_field) or np.any(~np.isfinite(n)) or np.any(n<1) or np.any(n!=np.rint(n)):
            raise ValueError('explicit data fields and positive integer random counts required')
        self.data_field=data_field;self.n=n

    @lru_cache(None)
    def _random(self,items):
        if not items:return np.ones_like(self.n,dtype=complex)
        exponent=sum(item[1] for item in items);result=np.zeros_like(self.n,dtype=complex)
        for blocks in partitions(list(range(len(items)))):
            if any(any(sum(items[i][2][j] for i in block) for j in range(3)) for block in blocks):continue
            term=np.ones_like(self.n,dtype=complex)
            for j in range(len(blocks)):term*=self.n-j
            result+=term
        return result*(-1.)**exponent/self.n**exponent

    def __call__(self,*items):
        labels=_items(items)
        if any(kind=='F' and power==1 and mode==(0,0,0) for kind,power,mode in labels):
            return np.zeros_like(self.n,dtype=complex)
        choices=[(False,True) if kind=='F' else (False,) for kind,_,_ in labels]
        result=np.zeros_like(self.n,dtype=complex)
        for selection in product(*choices):
            term=np.ones_like(self.n,dtype=complex);random=[]
            for item,use_random in zip(labels,selection):
                if use_random:random.append(item)
                else:
                    value=np.asarray(self.data_field(item[1],item[2]),dtype=complex)
                    if not np.isfinite(value).all():raise ValueError('nonfinite actual weighted data field')
                    term*=value
            result+=term*self._random(tuple(random))
        return result


def input_factorial_bispectrum(triangle,moment):
    """raw intrinsic三点减实际F2 mixed self、加2F3；不替代物理signal/noise划分。"""
    v=np.asarray(triangle)
    if (v.shape!=(3,3) or not np.isfinite(v).all() or np.any(v!=np.rint(v))
        or np.any(v.sum(axis=0)) or np.any(np.sum(v*v,axis=1)==0)):
        raise ValueError('explicit nonzero integer closed triangle required')
    raw=_finite(moment(*(factor('F',1,k) for k in v)))
    self_term=sum(_finite(moment(factor('F',2,-k),factor('F',1,k))) for k in v)
    triple=2*_finite(moment(factor('F',3,(0,0,0))))
    return raw-self_term+triple


def decompose_factorial_shift(triangle,shift_modes,shift_response,cumulant):
    """pair/mean、third-connected、fourth-connected三部分与原full exact响应。

返回每部分[corrected,raw,self]；full=pair+third+fourth。要求同一个显式
κ1..4 provider，不引入MCMC参数或假设未给出的实际STOκ3/κ4为零。
"""
    v=np.asarray(triangle);q=np.asarray(shift_modes);r=np.asarray(shift_response,dtype=float)
    if (v.shape!=(3,3) or q.ndim!=2 or q.shape[1]!=3 or r.shape!=(len(q),)
        or not np.isfinite(v).all() or not np.isfinite(q).all() or not np.isfinite(r).all()
        or np.any(v!=np.rint(v)) or np.any(q!=np.rint(q))
        or np.any(v.sum(axis=0)) or np.any(np.sum(v*v,axis=1)==0)
        or np.any(np.sum(q*q,axis=1)==0) or len(set(map(tuple,q)))!=len(q)):
        raise ValueError('nonzero unique integer shift modes and a closed external triangle required')
    versions=np.array([factorial_shift_response(v,q,r,CumulantMoments(cumulant,order))
                       for order in (2,3,4)],dtype=complex)
    if not np.isfinite(versions).all():raise ValueError('nonfinite full marked shift response')
    return dict(pair_and_means=versions[0],third_cumulant=versions[1]-versions[0],
                fourth_cumulant=versions[2]-versions[1],full=versions[2])
