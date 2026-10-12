"""模块大纲：实际 shell windows 与 raw/shot 相同 paint；全小型 registry 的点过程验收。

包括 zero-leg 和 diagonal pairs；仅数值 self-contractions，不是 shift-noise 理论。
保留原 CIC/bin-center 未通过结果，不改其数据或判定门。
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from mpi4py import MPI
from audit_post_rsd_weighted_estimator import shared,PART,digest
from post_rsd_estimator import matched_b000_self_contractions


def all_point_pairs(position,coefficient,edges):
    n=32;axis=np.rint(np.fft.fftfreq(n)*n).astype(int);kf=2*np.pi/1000.
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    k=np.linalg.norm(grid,axis=-1)*kf
    shells=[grid[(k>=lo)&(k<hi)] for lo,hi in zip(edges[:-1],edges[1:])]
    pairs={};union=[]
    for i in range(len(shells)):
        for j in range(i,len(shells)):
            a,b=shells[i],shells[j];aa=np.repeat(a,len(b),axis=0);bb=np.tile(b,(len(a),1));cc=-aa-bb
            if np.max(np.abs(cc))>=n//2:raise RuntimeError('closing-leg alias in point oracle')
            v=np.stack((aa,bb,cc),axis=1);pairs[i,j]=v;union.append(v.reshape(-1,3))
    unique=np.unique(np.concatenate(union),axis=0)
    lookup={tuple(v):i for i,v in enumerate(unique)}
    f1=np.empty(len(unique),dtype=complex);f2=np.empty_like(f1)
    for start in range(0,len(unique),128):
        phase=np.exp(-1j*(unique[start:start+128]*kf)@position.T)
        f1[start:start+128]=phase@coefficient;f2[start:start+128]=phase@(coefficient**2)
    if (0,0,0) in lookup:f1[lookup[0,0,0]]=0.  # raw production field 的 mean removal。
    s3=float(np.sum(coefficient**3));rows=[]
    for (i,j),v in pairs.items():
        ids=np.array([lookup[tuple(q)] for q in v.reshape(-1,3)]).reshape(-1,3)
        p1=f1[ids];p2=f2[ids]
        raw=float(np.mean(np.prod(p1,axis=1)).real)*1e18
        shot=float(np.mean(np.sum(p2.conj()*p1,axis=1)).real-2*s3)*1e18
        rows.append({'pair':[i,j],'raw':raw,'shot':shot,'corrected':raw-shot,
                     'all_same':2*s3*1e18,'ordered_triangles':len(v),
                     'triangles_with_zero_leg':int(np.count_nonzero(np.any(np.sum(v*v,axis=2)==0,axis=1)))})
    return rows


def main():
    start=time.time();rng=np.random.default_rng(2026100820)
    xd=rng.uniform(10,990,size=(160,3));xr=rng.uniform(10,990,size=(480,3))
    xd=(xd-3*np.sin(xd*2*np.pi/1000.)).astype('f4');xr=(xr-3*np.sin(xr*2*np.pi/1000.)).astype('f4')
    w=rng.uniform(.4,2.3,size=len(xd)).astype('f4');c=np.concatenate((w.astype(float)/np.sum(w,dtype=float),-np.ones(len(xr))/len(xr)))
    edges=np.array([0.,.020,.039]);oracle=all_point_pairs(np.concatenate((xd,xr)).astype(float),c,edges)
    data=shared.make_halo_catalog(xd,'z',MPI.COMM_WORLD,weights=w,redshift=.5)
    shifted=shared.make_halo_catalog(xr,'z',MPI.COMM_WORLD,redshift=.5)
    shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD)
    record=[]
    with matched_b000_self_contractions(record,matched_paint=True):
        spectrum=shared.measure_b000(lambda:{'data':data,'shifted':shifted},boxsize=1000.,meshsize=128,
            edges=edges,los='z',basis='sugiyama',mask_edges_policy='default',cache={},mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spectrum)
    pole=spectrum.get((0,0,0));k=np.asarray(pole.coords('k'));norm=np.asarray(pole.values('norm'))
    value=np.asarray(pole.value());shot=np.asarray(pole.values('num_shotnoise'))/norm
    rows=[]
    for truth in oracle:
        pair=truth['pair'];target=[int(np.searchsorted(edges,q,side='right')-1) for q in k[:,0]]
        candidates=[n for n,q in enumerate(k) if [int(np.searchsorted(edges,x,side='right')-1) for x in q]==pair]
        if len(candidates)!=1:raise RuntimeError(('ambiguous pair',pair,k))
        i=candidates[0];scale=max(abs(truth[q]) for q in ('raw','shot','corrected','all_same'))
        errors={name:abs(float(actual)-truth[name])/scale for name,actual in
                [('raw',value[i]+shot[i]),('shot',shot[i]),('corrected',value[i])]}
        rows.append({'point':truth,'measured':{'raw':float(value[i]+shot[i]),'shot':float(shot[i]),'corrected':float(value[i])},
                     'scaled_errors':errors,'bin_center_vs_shell_S113':float(record[0]['difference'][i].real/norm[i])})
    tolerance=5e-6;maximum=max(x for row in rows for x in row['scaled_errors'].values())
    payload={'status':'MATCHED_SELF_CONTRACTION_TEST_PASS' if maximum<tolerance else 'REQUIRES_INVESTIGATION',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':rows,'max_scaled_error':maximum,
        'predefined_tolerance':tolerance,'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'scope':'three bins including diagonal and zero-leg atoms; tiny weighted signed-point fixture',
        'raw_and_shot_paint':'TSC interlacing=3 compensate=True','S113':'actual discrete shell windows',
        'full_production_registry_verified':False,'reconstruction_shift_noise_theory_validated':False,
        'old_shared_installation_modified':False,'old_failed_audits_preserved':True,
        'elapsed_seconds':time.time()-start,'entry_sha256':digest(Path(__file__)),
        'adapter_sha256':digest(PART/'code/post_rsd_estimator.py')}
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1/MATCHED_SELF_AUDIT.json'
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2),flush=True)
    if maximum>=tolerance:raise SystemExit(1)


if __name__=='__main__':main()
