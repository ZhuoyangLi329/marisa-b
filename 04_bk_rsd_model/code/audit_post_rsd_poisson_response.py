"""模块大纲：有限点数 exact occupancy moments；same-shift factorial B 一阶响应。

独立于 native/FFT/tree 源。iid 点集服从随机相位的单正弦强度场，用 set partitions
精确求到四点矩，避免 Monte Carlo 噪声。shift 来自 data；random 用连续极限。
用于判别重构噪声候选公式，不是 STO 非 Poisson 模型的最终验收。
"""
from functools import lru_cache
from itertools import product
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import numpy as np
from rsd_b000_contract import PART,digest


def partitions(items):
    if not items:
        yield ();return
    first,*rest=items
    for blocks in partitions(rest):
        yield ((first,),)+blocks
        for i in range(len(blocks)):
            yield blocks[:i]+((first,)+blocks[i],)+blocks[i+1:]


class ExactPointMoments:
    def __init__(self,npoint,signal_mode,amplitude):
        self.n=int(npoint);self.mode=np.asarray(signal_mode,dtype=int);self.amplitude=float(amplitude)
        if self.n<4 or not 0<=self.amplitude<1:raise ValueError('invalid iid density')

    @lru_cache(None)
    def moment(self,modes):
        """所有参数都是 Fd(k)=N^-1 sum exp(-ik.x)，包括 Fd(0)=1。"""
        result=0.
        for blocks in partitions(list(range(len(modes)))):
            coefficient=1.
            for j in range(len(blocks)):coefficient*=self.n-j
            coefficient/=self.n**len(modes)
            phase=0
            for block in blocks:
                wave=np.sum([modes[j] for j in block],axis=0)
                if np.all(wave==0):continue
                if np.array_equal(wave,self.mode):phase+=1;coefficient*=self.amplitude/2
                elif np.array_equal(wave,-self.mode):phase-=1;coefficient*=self.amplitude/2
                else:coefficient=0.;break
            if phase==0:result+=coefficient
        return result

    def corr(self,*modes,zero_removed=()):
        if any(np.all(np.asarray(modes[j])==0) for j in zero_removed):return 0.
        return self.moment(tuple(tuple(int(q) for q in k) for k in modes))


def response(triangle,shift_modes,shift_response,moments):
    """直接微分 F1F1F1-sum F2(-ki)F1(ki)+2F3，不用对称化 R2。"""
    v=np.asarray(triangle,dtype=int);raw=0.;self_term=0.;nd=1/moments.n
    for leg in range(3):
        i,j=tuple(a for a in range(3) if a!=leg);k=v[leg]
        for q,r in zip(shift_modes,shift_response):
            # Frec'(k)=-sum_q k.q/q² Rq Fd(q) [Fd(k-q)-delta_kq]。
            geom=np.dot(k,q)/np.dot(q,q)*r
            raw-=geom*moments.corr(q,k-q,v[i],v[j],zero_removed=(1,2,3))
            # F2'(k)=Nd Fd'(k)，data 具有均匀背景；不可像 Frec 一样丢其零模。
            self_term+=nd*geom*moments.corr(q,-k-q,k,zero_removed=(2,))
            self_term-=nd*geom*moments.corr(-k,q,k-q,zero_removed=(2,))
    return raw-self_term,raw,self_term


def candidates(v,r,p,nd):
    oriented=0.;symmetric=0.
    for i,j in ((0,1),(1,2),(2,0)):
        total=v[i]+v[j]
        ti=np.dot(total,v[i])/np.dot(v[i],v[i])*r[i]
        tj=np.dot(total,v[j])/np.dot(v[j],v[j])*r[j]
        oriented-=nd*(ti*p[j]+tj*p[i])
        symmetric-=.5*nd*(ti+tj)*(p[i]+p[j])
    return oriented,symmetric


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    v=np.array([[1,0,0],[0,2,0],[-1,-2,0]],dtype=int)
    r=np.array([.71,.42,.26]);q=np.concatenate((v,-v));rq=np.tile(r,2)
    a=.02;rows=[]
    for signal_leg in range(3):
        p=np.zeros(3);p[signal_leg]=(a/2)**2
        candidate=candidates(v,r,p,1.)
        values=[]
        for n in (1000,10000,100000,1000000):
            exact,raw,self_term=response(v,q,rq,ExactPointMoments(n,v[signal_leg],a))
            values.append({'Npoint':n,'derivative':exact,'raw_derivative':raw,'self_derivative':self_term,
                           'N_times_derivative':n*exact})
        rows.append({'signal_leg':signal_leg,'signal_power':p.tolist(),
                     'candidate_N_times_oriented':candidate[0],'candidate_N_times_symmetric_R2':candidate[1],
                     'finite_N':values})
    pure=[]
    for n in (1000,100000):
        value,raw,self_term=response(v,q,rq,ExactPointMoments(n,v[0],0.))
        pure.append({'Npoint':n,'corrected_derivative':value,'raw_derivative':raw,'self_derivative':self_term})
    matched=[]
    for row in rows:
        fine=row['finite_N'][-1]['N_times_derivative']
        scale=max(abs(row['candidate_N_times_oriented']),abs(row['candidate_N_times_symmetric_R2']),
                  max(row['signal_power'])*float(np.max(np.abs(r))),1e-15)
        matched.append(abs(fine-row['candidate_N_times_oriented'])/scale)
    passed=max(matched)<1e-4 and max(abs(x['corrected_derivative']) for x in pure)<1e-14
    out={'status':'EXACT_POINT_RESPONSE_SUPPORTS_ORIENTED_CANDIDATE' if passed else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'triangle':v.tolist(),'shift_response':r.tolist(),
         'rows':rows,'uniform_point_tests':pure,'max_fine_scaled_error':max(matched),
         'scope':'iid fixed-N single sinusoidal intensity, phase averaged exact occupancy, infinitesimal shift, continuous random',
         'comparison_reference':'https://arxiv.org/html/2410.18524v3#S2.SS4 Eq44, differing symmetrized candidate recorded not silently selected',
         'STO_nonpoisson_validated':False,'finite_random_validated':False,'finite_shift_validated':False,
         'production_model_modified':False,'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'entry_sha256':digest(Path(__file__))}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/POISSON_RESPONSE_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
