"""模块大纲：marked + finite-random 精确点过程与实际坐标导数双重审计。

仅验收 infinitesimal same-shift 和 leading 1/N；不声称 STO 非 Poisson 闭合。
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import time
import numpy as np
from rsd_b000_contract import PART,digest
from audit_post_rsd_poisson_response import candidates
from post_rsd_marked_moments import MarkedPointMoments,ActualMarkedFields,factor,factorial_shift_response


def coordinate_cases():
    rng=np.random.default_rng(2026100902)
    v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]])
    q=np.concatenate((v,-v,[[1,-1,2],[-1,1,-2]]));rq=np.array([.71,.42,.26]*2+[.18,.18])
    rows=[]
    for n,nran in ((8,2),(17,5),(32,30),(79,30)):
        pos=rng.uniform(0,1,(n,3));ran=rng.uniform(0,1,(n*nran,3));weights=rng.uniform(.4,2.3,n)
        fields=ActualMarkedFields(pos,weights,ran)
        src=np.array([fields.field(factor('D',1,ki)) for ki in q]);q2=np.sum(q*q,axis=1)
        multiplier=(-1j/(2*np.pi))*rq[:,None]*q/q2[:,None]*src[:,None]
        sd=(np.exp(2j*np.pi*(pos@q.T))@multiplier).real
        sr=(np.exp(2j*np.pi*(ran@q.T))@multiplier).real
        coeff=np.concatenate((fields.cd,np.full(len(ran),fields.cr)))
        coords=np.concatenate((pos,ran));shift=np.concatenate((sd,sr))
        def direct(epsilon):
            phase=np.exp(-2j*np.pi*(v@(coords+epsilon*shift).T))
            f1=phase@coeff;f2=phase@(coeff**2)
            return np.prod(f1)-np.sum(f2.conj()*f1)+2*np.sum(coeff**3)
        grad=lambda e:(direct(e)-direct(-e))/(2*e)
        measured=(4*grad(5e-5)-grad(1e-4))/3
        exact,_,_=factorial_shift_response(v,q,rq,fields.expect)
        scale=max(abs(exact),1e-8);error=abs(measured-exact)/scale
        rescaled=ActualMarkedFields(pos,7.3*weights,ran)
        check=factorial_shift_response(v,q,rq,rescaled.expect)[0]
        if abs(check-exact)>1e-12*max(1.,abs(exact)):raise ValueError('weight normalization invariance failed')
        rows.append({'ndata':n,'nrandom':len(ran),'coordinate_derivative':[measured.real,measured.imag],
                     'Fourier_derivative':[exact.real,exact.imag],'relative_error':float(error)})
    return rows


def moment_cases():
    triangles=[np.array([[1,0,0],[0,2,0],[-1,-2,0]]),np.array([[1,0,1],[0,2,-1],[-1,-2,0]])]
    r=np.array([.71,.42,.26]);weights=np.array([.6,1.9]);amplitudes=np.array([.016,.035]);rows=[]
    for itri,v in enumerate(triangles):
        q=np.concatenate((v,-v,[[1,-1,2],[-1,1,-2]]));rq=np.concatenate((r,r,[.18,.18]))
        for leg in range(3):
            for nran in (2,30):
                values=[];candidate=None;power=None
                for n in (1000,1000000):
                    counts=np.array([2*n//5,3*n//5]);mom=MarkedPointMoments(counts,weights,amplitudes,v[leg],n*nran)
                    np.testing.assert_allclose(mom.expect(factor('F',2,(0,0,0))),mom.N2+1/(n*nran),rtol=3e-15)
                    np.testing.assert_allclose(mom.expect(factor('D',1,(0,0,0))),1.,rtol=3e-15)
                    np.testing.assert_array_equal(mom.expect(factor('F',1,(0,0,0))),0.)
                    exact,raw,self_term=factorial_shift_response(v,q,rq,mom.expect)
                    power=np.zeros(3);power[leg]=(mom.signal_amplitude/2)**2
                    candidate=candidates(v,r,power,n*mom.N2)[0]
                    values.append({'ndata':n,'nrandom':n*nran,'N2_data':mom.N2,'N3_data':mom.N3,
                                   'derivative':float(exact),'N_times_derivative':float(n*exact),
                                   'raw_derivative':float(raw),'self_derivative':float(self_term)})
                scale=max(abs(candidate),float(np.max(power))*float(np.max(r))*n*mom.N2,1e-15)
                rows.append({'triangle_index':itri,'signal_leg':leg,'nran':nran,'candidate_N_times_derivative':candidate,
                             'fixed_reference_scale':scale,'fine_scaled_error':abs(values[-1]['N_times_derivative']-candidate)/scale,
                             'finite_N':values})
    return rows


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    start=time.time();coordinates=coordinate_cases();moments=moment_cases()
    errors={'coordinate':max(r['relative_error'] for r in coordinates),
            'leading_N':max(r['fine_scaled_error'] for r in moments)}
    passed=bool(errors['coordinate']<1e-7 and errors['leading_N']<1e-4)
    out={'status':'MARKED_FINITE_RANDOM_RESPONSE_TEST_PASS' if passed else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'coordinate_cases':coordinates,'exact_moment_cases':moments,
         'max_errors':errors,'predefined_tolerances':{'coordinate':1e-7,'leading_N':1e-4},
         'scope':'two marked iid populations with different intensity responses; finite independent uniform random; infinitesimal shift',
         'marked_iid_test_validated':passed,'finite_random_test_validated':passed,
         'actual_STO_nonpoisson_validated':False,'finite_shift_validated':False,'production_likelihood_modified':False,
         'initial_execution_note':'numpy.bool_ JSON serialization error; no audit result was written; explicit bool conversion only, thresholds unchanged',
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.time()-start,
         'hashes':{x:digest(PART/'code'/x) for x in ('post_rsd_marked_moments.py','audit_post_rsd_marked_response.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/MARKED_FINITE_RANDOM_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
