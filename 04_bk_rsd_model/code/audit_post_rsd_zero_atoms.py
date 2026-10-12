"""模块大纲：逐 zero-leg 三角形枚举，验证 estimator 对零模 self-terms 的诊断。

不修改原分箱/denominator，不覆盖 matched 或 legacy 数据。只输出可显式核对的
nonzero-only counterfactual；将其用于正式估计器仍需匹配 pre 与 post covariance。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from mpi4py import MPI
from rsd_b000_contract import PART,digest
from audit_post_rsd_weighted_estimator import shared
from audit_post_rsd_matched_self import all_point_pairs
from post_rsd_estimator import matched_b000_self_contractions


def zero_point_pairs(positions,coefficients,edges):
    n=32;kf=2*np.pi/1000.;axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1);k=np.linalg.norm(grid,axis=-1)*kf
    shells=[grid[(k>=a)&(k<b)] for a,b in zip(edges[:-1],edges[1:])];rows=[]
    for i in range(len(shells)):
        for j in range(i,len(shells)):
            a,b=shells[i],shells[j];aa=np.repeat(a,len(b),axis=0);bb=np.tile(b,(len(a),1));cc=-aa-bb
            triangles=np.stack((aa,bb,cc),axis=1);zero=np.any(np.sum(triangles*triangles,axis=2)==0,axis=1)
            v=triangles[zero];unique,inverse=np.unique(v.reshape(-1,3),axis=0,return_inverse=True)
            f1=np.empty(len(unique),dtype=complex);f2=np.empty_like(f1)
            for start in range(0,len(unique),128):
                phase=np.exp(-1j*(unique[start:start+128]*kf)@positions.T)
                f1[start:start+128]=phase@coefficients;f2[start:start+128]=phase@(coefficients**2)
            f1[np.sum(unique*unique,axis=1)==0]=0.
            first=f1[inverse].reshape(-1,3);second=f2[inverse].reshape(-1,3)
            self_terms=np.sum(second.conj()*first,axis=1)-2*np.sum(coefficients**3)
            raw=float(np.sum(np.prod(first,axis=1)).real)*1e18/len(triangles)
            shot=float(np.sum(self_terms).real)*1e18/len(triangles)
            rows.append({'pair':[i,j],'zero_triangle_count':int(zero.sum()),'full_pair_count':len(triangles),
                         'raw_zero_contribution':raw,'shot_zero_contribution':shot})
    return rows


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('CPU cap')
    rng=np.random.default_rng(2026100820);xd=rng.uniform(10,990,(160,3));xr=rng.uniform(10,990,(480,3))
    xd=(xd-3*np.sin(xd*2*np.pi/1000.)).astype('f4');xr=(xr-3*np.sin(xr*2*np.pi/1000.)).astype('f4')
    w=rng.uniform(.4,2.3,len(xd)).astype('f4');c=np.concatenate((w.astype(float)/np.sum(w,dtype=float),-np.ones(len(xr))/len(xr)))
    positions=np.concatenate((xd,xr)).astype(float);edges=np.array([0.,.020,.039])
    all_oracle=all_point_pairs(positions,c,edges);zeros=zero_point_pairs(positions,c,edges)
    data=shared.make_halo_catalog(xd,'z',MPI.COMM_WORLD,weights=w,redshift=.5)
    shifted=shared.make_halo_catalog(xr,'z',MPI.COMM_WORLD,redshift=.5)
    shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD);record=[]
    with matched_b000_self_contractions(record,matched_paint=True):
        spec=shared.measure_b000(lambda:{'data':data,'shifted':shifted},boxsize=1000.,meshsize=128,
             edges=edges,los='z',basis='sugiyama',mask_edges_policy='default',cache={},mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spec)
    pole=spec.get((0,0,0));k=np.asarray(pole.coords('k'));norm=np.asarray(pole.values('norm'));value=np.asarray(pole.value())
    detected=record[0]['zero_leg_num_shotnoise']/norm;rows=[]
    for full,zero in zip(all_oracle,zeros):
        candidates=[ii for ii,kk in enumerate(k) if [int(np.searchsorted(edges,q,side='right')-1) for q in kk]==zero['pair']]
        if len(candidates)!=1:raise ValueError('pair matching failed')
        i=candidates[0];scale=max(abs(full[x]) for x in ('raw','shot','corrected','all_same'))
        expected=full['corrected']+zero['shot_zero_contribution'];counterfactual=float(value[i]+detected[i])
        expected_fraction=zero['zero_triangle_count']/zero['full_pair_count']
        np.testing.assert_allclose(record[0]['zero_leg_pair_fraction'][i],expected_fraction,rtol=2e-15)
        rows.append({'point':zero,'measured_zero_shot':float(detected[i]),'zero_scaled_error':float(abs(detected[i]-zero['shot_zero_contribution'])/scale),
                     'nonzero_only_expected':expected,'nonzero_only_counterfactual':counterfactual,
                     'nonzero_only_scaled_error':abs(counterfactual-expected)/scale})
    ez=max(row['zero_scaled_error'] for row in rows);ec=max(row['nonzero_only_scaled_error'] for row in rows)
    passed=bool(ez<5e-7 and ec<5e-6)
    out={'status':'ZERO_ATOM_DIAGNOSTIC_TEST_PASS' if passed else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':rows,'max_zero_scaled_error':ez,
         'max_nonzero_counterfactual_scaled_error':ec,'predefined_tolerances':{'zero':5e-7,'corrected':5e-6},
         'all_full_shell_denominators_preserved':True,'baseline_shot_changed':False,
         'production_estimator_choice_finalized':False,'real_catalog_zero_contribution_measured':False,
         'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'hashes':{x:digest(PART/'code'/x) for x in ('post_rsd_estimator.py','audit_post_rsd_zero_atoms.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/ZERO_ATOM_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
