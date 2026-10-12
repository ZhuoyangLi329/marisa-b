"""模块大纲：非Poisson相关child的精确噪声响应；独立坐标/标签验收。

只验证指定cluster点过程的算子和候选边界；不代替STO halo实测恢复。
输出另存，不覆盖已有iid/marked审计，不放行production噪声或MCMC。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import unittest
import numpy as np
from post_rsd_clustered_moments import ClusteredMarkedMoments
from post_rsd_marked_moments import factorial_shift_response
from audit_post_rsd_poisson_response import candidates
from test_post_rsd_clustered_moments import clustered_coordinate_cases,ClusteredMomentsTest

PART=Path(__file__).resolve().parents[1]
OUT=PART/'outputs/sto_marisa_post_rsd_mean100_v1/CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json'


def geometry_sum(v,r):
    total=0.
    for i,j in ((0,1),(1,2),(2,0)):
        k=v[i]+v[j]
        total+=np.dot(k,v[i])/np.dot(v[i],v[i])*r[i]
        total+=np.dot(k,v[j])/np.dot(v[j],v[j])*r[j]
    return float(total)


def coincident_cases():
    """独立手写closed-form；h按实际child权重幂，不能按parent扣self。"""
    v=np.array([[1,0,0],[0,2,0],[-1,-2,0]]);r=np.array([.71,.42,.26])
    q=np.concatenate((v,-v));rq=np.tile(r,2);rows=[]
    for weights in ([1.,1.],[.6,1.9],[1.,-.4]):
        h=float(np.sum(np.asarray(weights)**2)/np.sum(weights)**2)
        for nran in (2,30):
            sequence=[]
            for n in (1000,1000000):
                mom=ClusteredMarkedMoments([n],[weights],[[[0,0,0]]*len(weights)],
                    [0.],v[0],n*len(weights)*nran)
                exact,raw,self_term=factorial_shift_response(v,q,rq,mom.expect)
                # 有限N修正来自三个external parent标签互异；随机self精确相消。
                closed=-(1-h)*(1-2/n)*geometry_sum(v,r)/n**2
                error=float(abs(exact-closed)/max(abs(closed),1e-30))
                sequence.append(dict(nparents=n,ndata=mom.ndata,nrandom=mom.nrandom,
                    N2_individual=mom.N2,N3_individual=mom.N3,
                    raw_derivative=float(raw.real),subtracted_self_derivative=float(self_term.real),
                    factorial_derivative=float(exact.real),closed_form=closed,relative_error=error,
                    parent_squared_times_derivative=float((n*n*exact).real),
                    residual_power=float(mom.residual_power((0,2,0)).real)))
            rows.append(dict(weights=weights,signed_weights=bool(np.any(np.asarray(weights)<0)),
                h_child_power2_over_parent_power2=h,nran=nran,finite_N=sequence,
                iid_N2_only_candidate_without_signal=0.,
                leading_parent_squared_response=-(1-h)*geometry_sum(v,r)))
    return rows


def separated_signal_cases():
    v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]]);r=np.array([.71,.42,.26])
    q=np.concatenate((v,-v,[[1,-1,2],[-1,1,-2]]));rq=np.concatenate((r,r,[.18,.18]));rows=[]
    weights=[[.6,1.9],[1.2,-.3]]
    offsets=[[[0,0,0],[.011,.017,-.023]],[[.02,0,0],[-.007,.013,.019]]]
    for leg in range(3):
        for nran in (2,30):
            sequence=[]
            for n in (1000,1000000):
                counts=[2*n//5,3*n//5];ndata=2*n
                mom=ClusteredMarkedMoments(counts,weights,offsets,[.016,.035],v[leg],ndata*nran)
                uniform=ClusteredMarkedMoments(counts,weights,offsets,[0.,0.],v[leg],ndata*nran)
                exact,_,_=factorial_shift_response(v,q,rq,mom.expect)
                pure,_,_=factorial_shift_response(v,q,rq,uniform.expect)
                # 连续parent强度场的child线性响应；不含point self或cluster residual。
                linear=sum(counts[g]*mom.amplitudes[g]/2*mom.child_factor(g,1,tuple(v[leg]))
                           for g in range(mom.ngroup))
                power=np.zeros(3);power[leg]=abs(linear)**2
                iid=candidates(v,r,power,mom.N2)[0]
                difference=exact-pure
                scale=max(abs(n*iid),float(np.max(power))*float(np.max(r))*n*mom.N2,1e-15)
                sequence.append(dict(nparents=n,ndata=ndata,nrandom=mom.nrandom,N2_individual=mom.N2,
                    corrected_response=float(exact.real),uniform_response=float(pure.real),
                    signal_response=float(difference.real),Nparent_times_signal_response=float((n*difference).real),
                    Nparent_times_iid_candidate=float(n*iid),
                    iid_candidate_scaled_difference=float(abs(n*(difference-iid))/scale),
                    residual_powers_uniform=[float(uniform.residual_power(k).real) for k in v],
                    imaginary_response_absolute=float(abs(exact.imag))))
            rows.append(dict(signal_leg=leg,nran=nran,continuous_signal_power=power.tolist(),finite_N=sequence))
    return rows


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('one logical CPU required')
    if OUT.exists():raise RuntimeError('preserve prior correlated-response audit')
    start=time.time()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClusteredMomentsTest))
    if not tests.wasSuccessful():raise RuntimeError('independent moment/coordinate tests failed')
    coordinates=clustered_coordinate_cases();coincident=coincident_cases();separated=separated_signal_cases()
    coordinate_error=max(row['relative_error'] for row in coordinates)
    algebra_error=max(v['relative_error'] for row in coincident for v in row['finite_N'])
    passed=coordinate_error<1e-7 and algebra_error<1e-10
    files=['post_rsd_clustered_moments.py','test_post_rsd_clustered_moments.py',
           'audit_post_rsd_clustered_response.py','post_rsd_marked_moments.py','audit_post_rsd_poisson_response.py']
    report=dict(status='CORRELATED_CLUSTER_OPERATOR_RESPONSE_PASS_NOT_STO_LIKELIHOOD' if passed else 'REQUIRES_INVESTIGATION',
        created_UTC=datetime.now(timezone.utc).isoformat(),scope='specified fixed-parent marked cluster point processes, infinitesimal same-shift, finite uniform random',
        independent_tests_passed=tests.testsRun,coordinates=coordinates,coincident_closed_form_cases=coincident,
        separated_signed_signal_cases=separated,max_coordinate_relative_error=coordinate_error,
        max_coincident_closed_form_relative_error=algebra_error,
        predefined_tolerances=dict(coordinate_relative=1e-7,coincident_closed_form_relative=1e-10),
        max_signal_iid_candidate_scaled_difference=max(row['finite_N'][-1]['iid_candidate_scaled_difference'] for row in separated),
        exact_external_child_self_subtraction_preserved=True,data_only_shift_source=True,
        individual_N2_only_is_universal_nonpoisson_model=False,
        actual_STO_nonpoisson_validated=False,finite_shift_validated=False,
        formal_noise_audit_released=False,production_likelihood_modified=False,
        CPU_affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.time()-start,
        source_sha256={name:hashlib.sha256((PART/'code'/name).read_bytes()).hexdigest() for name in files})
    with OUT.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in report.items() if key not in
        ('coordinates','coincident_closed_form_cases','separated_signed_signal_cases')},indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
