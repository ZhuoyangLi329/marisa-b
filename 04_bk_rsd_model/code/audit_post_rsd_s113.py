"""模块大纲：同一 raw 加权 B000 的 legacy/exact S113 对直接点 self-contraction。

保留最初未通过的 WEIGHTED_ESTIMATOR_AUDIT，不覆盖/放宽它的容差。
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
from audit_post_rsd_weighted_estimator import exact_pair,shared,PART,digest
from post_rsd_estimator import matched_b000_self_contractions


def main():
    start=time.time();rng=np.random.default_rng(2026100820)
    xd=rng.uniform(10,990,size=(160,3));xr=rng.uniform(10,990,size=(480,3))
    xd=(xd-3*np.sin(xd*2*np.pi/1000.)).astype('f4');xr=(xr-3*np.sin(xr*2*np.pi/1000.)).astype('f4')
    weights=rng.uniform(.4,2.3,size=len(xd)).astype('f4')
    c=np.concatenate((weights.astype(float)/np.sum(weights,dtype=float),-np.ones(len(xr))/len(xr)))
    edges=np.array([.008,.020,.039]);oracle=exact_pair(np.concatenate((xd,xr)).astype(float),c,edges)
    data=shared.make_halo_catalog(xd,'z',MPI.COMM_WORLD,weights=weights,redshift=.5)
    shifted=shared.make_halo_catalog(xr,'z',MPI.COMM_WORLD,redshift=.5)
    shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD)
    records=[]
    with matched_b000_self_contractions(records):
        spectrum=shared.measure_b000(lambda:{'data':data,'shifted':shifted},boxsize=1000.,meshsize=128,
            edges=edges,los='z',basis='sugiyama',mask_edges_policy='default',cache={},mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spectrum)
    if len(records)!=1:raise RuntimeError('unexpected S113 call count')
    pole=spectrum.get((0,0,0));k=np.asarray(pole.coords('k'))
    i=int(np.flatnonzero((k[:,0]<edges[1])&(k[:,1]>edges[1]))[0])
    norm=np.asarray(pole.values('norm'));value=np.asarray(pole.value());shot=np.asarray(pole.values('num_shotnoise'))/norm
    correction=records[0]['difference'].real/norm
    scale=max(abs(oracle[q]) for q in ('raw','shot','corrected','all_same'))
    row={'corrected_exact':float(value[i]),'shot_exact':float(shot[i]),'raw':float(value[i]+shot[i]),
         'corrected_legacy':float(value[i]+correction[i]),'shot_legacy':float(shot[i]-correction[i]),
         'S113_difference':float(correction[i]),'exact_error_over_reference_scale':float(abs(value[i]-oracle['corrected'])/scale),
         'legacy_error_over_reference_scale':float(abs(value[i]+correction[i]-oracle['corrected'])/scale)}
    passed=row['exact_error_over_reference_scale']<1e-3
    payload={'status':'EXACT_S113_POINT_TEST_PASS' if passed else 'REQUIRES_INVESTIGATION',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'point_oracle':oracle,'comparison':row,'scope':'one nonzero-leg weighted data/random shell pair',
        'change':'replace radial bin-center j0 projection with actual discrete-shell windows for S113 only',
        'old_shared_installation_modified':False,'original_failed_audit_preserved':True,
        'full_registry_verified':False,'formal_estimator_choice_frozen':False,
        'elapsed_seconds':time.time()-start,'entry_sha256':digest(Path(__file__)),
        'adapter_sha256':digest(PART/'code/post_rsd_estimator.py')}
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1/S113_AUDIT.json'
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
