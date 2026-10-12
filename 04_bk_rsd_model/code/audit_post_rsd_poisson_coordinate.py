"""模块大纲：真实指数坐标位移独立核验 factorial B 的 infinitesimal-response 审计。

均匀参考场用二维周期 quadrature（所有探针波矢 kz=0），不是有限随机样本。
该检查仅定位实际 estimator 的标签/符号，不认证 STO stochastic closure。
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import numpy as np
from rsd_b000_contract import PART,digest
from audit_post_rsd_poisson_response import response


class ActualPointMoments:
    def __init__(self,positions):self.positions=positions;self.n=len(positions)
    def corr(self,*modes,zero_removed=()):
        if any(np.all(np.asarray(modes[j])==0) for j in zero_removed):return 0.
        return np.prod(np.mean(np.exp(-2j*np.pi*(np.asarray(modes)@self.positions.T)),axis=1))


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    rng=np.random.default_rng(2026100901);v=np.array([[1,0,0],[0,2,0],[-1,-2,0]])
    q=np.concatenate((v,-v));rq=np.tile([.71,.42,.26],2);q2=np.sum(q*q,axis=1)
    x,y=np.meshgrid(np.arange(32)/32,np.arange(32)/32,indexing='ij')
    uniform=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    rows=[]
    for n in (8,17,32,79):
        positions=rng.uniform(0,1,size=(n,3));mom=ActualPointMoments(positions)
        fdata=np.mean(np.exp(-2j*np.pi*(q@positions.T)),axis=1)
        multiplier=(-1j/(2*np.pi))*rq[:,None]*q/q2[:,None]*fdata[:,None]
        shift=lambda a:(np.exp(2j*np.pi*(a@q.T))@multiplier).real
        sd=shift(positions);sr=shift(uniform)
        def direct(epsilon):
            fd=np.mean(np.exp(-2j*np.pi*(v@(positions+epsilon*sd).T)),axis=1)
            fr=np.mean(np.exp(-2j*np.pi*(v@(uniform+epsilon*sr).T)),axis=1)
            f=fd-fr
            return np.prod(f)-np.sum(fd.conj()*f)/n+2/n**2
        grad=lambda e:(direct(e)-direct(-e))/(2*e)
        numerical=(4*grad(5e-5)-grad(1e-4))/3
        exact,_,_=response(v,q,rq,mom)
        error=abs(numerical-exact)/max(abs(exact),1e-8)
        rows.append({'Npoint':n,'coordinate_derivative_real_imag':[numerical.real,numerical.imag],
                     'Fourier_identity_derivative_real_imag':[exact.real,exact.imag],'relative_error':error})
    maximum=max(row['relative_error'] for row in rows);passed=maximum<1e-7
    out={'status':'POINT_COORDINATE_FACTORIAL_RESPONSE_PASS' if passed else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':rows,'max_relative_error':maximum,
         'predefined_tolerance':1e-7,'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'scope':'same data-driven shift on data and continuous uniform reference, four point fixtures, infinitesimal response',
         'finite_random_validated':False,'STO_nonpoisson_validated':False,'production_model_modified':False,
         'hashes':{x:digest(PART/'code'/x) for x in ('audit_post_rsd_poisson_coordinate.py','audit_post_rsd_poisson_response.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/POISSON_COORDINATE_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
