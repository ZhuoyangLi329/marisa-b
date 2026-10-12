"""模块大纲：实际加权 data/shifted B000 estimator 对点 Fourier/self-contraction oracle。

只核验无零腿的一个离散 off-diagonal shell pair，并比较 64/128 网格；不以此
替代全 registry、生产网格和 reconstruction-induced stochastic 的科学验收。
"""
from pathlib import Path
from datetime import datetime,timezone
import gc
import json
import os
import sys
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from mpi4py import MPI
PART=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/common'))
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
import halo_b000_pipeline as shared
from rsd_b000_contract import digest


def exact_pair(position,coefficient,edges):
    n=32;axis=np.rint(np.fft.fftfreq(n)*n).astype(int);kf=2*np.pi/1000.
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    k=np.linalg.norm(grid,axis=-1)*kf
    a=grid[(k>=edges[0])&(k<edges[1])];b=grid[(k>=edges[1])&(k<edges[2])]
    aa=np.repeat(a,len(b),axis=0);bb=np.tile(b,(len(a),1));cc=-aa-bb
    if np.any(np.sum(cc*cc,axis=1)==0):raise RuntimeError('oracle requires all legs nonzero')
    vectors=np.stack((aa,bb,cc),axis=1)
    unique,inverse=np.unique(vectors.reshape(-1,3),axis=0,return_inverse=True)
    f1=np.empty(len(unique),dtype=complex);f2=np.empty_like(f1)
    for start in range(0,len(unique),128):
        phase=np.exp(-1j*(unique[start:start+128]*kf)@position.T)
        f1[start:start+128]=phase@coefficient;f2[start:start+128]=phase@(coefficient**2)
    first=f1[inverse].reshape(-1,3);second=f2[inverse].reshape(-1,3)
    raw=np.mean(np.prod(first,axis=1));pair=np.mean(np.sum(second.conj()*first,axis=1))
    triple=np.sum(coefficient**3)
    for value in (raw,pair):
        if abs(value.imag)>1e-12:raise RuntimeError('parity-odd monopole in point oracle')
    raw=float(raw.real)*1e18;shot=float(pair.real-2*triple)*1e18
    return dict(raw=raw,shot=shot,corrected=raw-shot,all_same=2*triple*1e18,
                shell_counts=[len(a),len(b)],ordered_triangles=len(vectors),unique_modes=len(unique))


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    t=time.time();rng=np.random.default_rng(2026100820)
    xd=rng.uniform(10,990,size=(160,3));xr=rng.uniform(10,990,size=(480,3))
    # 同一个平滑坐标映射作用于 data 和 random；本测试不拟合或校准该位移。
    xd=(xd-3*np.sin(xd*2*np.pi/1000.)).astype('f4')
    xr=(xr-3*np.sin(xr*2*np.pi/1000.)).astype('f4')
    w=rng.uniform(.4,2.3,size=len(xd)).astype('f4')
    c=np.concatenate((w.astype(float)/np.sum(w,dtype=float),-np.ones(len(xr))/len(xr)))
    pos=np.concatenate((xd,xr)).astype(float)
    edges=np.array([.008,.020,.039])
    oracle=exact_pair(pos,c,edges);print('POINT_ORACLE '+json.dumps(oracle),flush=True)
    data=shared.make_halo_catalog(xd,'z',MPI.COMM_WORLD,weights=w,redshift=.5)
    shifted=shared.make_halo_catalog(xr,'z',MPI.COMM_WORLD,redshift=.5)
    shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD)
    scale=max(abs(oracle[k]) for k in ('raw','shot','corrected','all_same'))
    rows=[]
    for mesh in (64,128):
        tick=time.time()
        spectrum=shared.measure_b000(lambda:{'data':data,'shifted':shifted},boxsize=1000.,meshsize=mesh,
            edges=edges,los='z',basis='sugiyama',mask_edges_policy='default',cache={},mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spectrum);pole=spectrum.get((0,0,0))
        kval=np.asarray(pole.coords('k'));value=np.asarray(pole.value(),dtype=float)
        sn=np.asarray(pole.values('num_shotnoise'),dtype=float)/np.asarray(pole.values('norm'),dtype=float)
        choices=np.flatnonzero((kval[:,0]<edges[1])&(kval[:,1]>edges[1]))
        if len(choices)!=1:raise RuntimeError(('ambiguous off-diagonal pair',kval))
        i=int(choices[0]);raw=float(value[i]+sn[i]);shot=float(sn[i])
        row={'mesh':mesh,'bin_index':i,'k_mean':kval[i].tolist(),'raw':raw,'shot':shot,'corrected':float(value[i]),
             'normalization':float(np.asarray(pole.values('norm'))[i]),
             'raw_error_over_reference_scale':abs(raw-oracle['raw'])/scale,
             'shot_error_over_reference_scale':abs(shot-oracle['shot'])/scale,
             'corrected_error_over_reference_scale':abs(float(value[i])-oracle['corrected'])/scale,
             'elapsed_seconds':time.time()-tick}
        rows.append(row);print('ESTIMATOR '+json.dumps(row),flush=True)
        del spectrum,pole;gc.collect();jax.clear_caches()
    tolerance=1e-3
    passed=max(rows[-1][key] for key in ('raw_error_over_reference_scale','shot_error_over_reference_scale',
                                        'corrected_error_over_reference_scale'))<tolerance
    payload={'status':'WEIGHTED_ESTIMATOR_POINT_COMPARISON_PASS' if passed else 'REQUIRES_INVESTIGATION',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'point_oracle':oracle,'rows':rows,'reference_scale':scale,'fine_mesh_tolerance':tolerance,
        'scope':'one nonzero-leg off-diagonal pair; weighted signed data/random field',
        'raw_paint':'production TSC interlacing=3 compensate=True',
        'shot_operator':'unchanged production compute_fkp3_shotnoise defaults CIC/interlacing=0',
        'full_registry_verified':False,'shift_noise_theory_validated':False,'elapsed_seconds':time.time()-t,
        'source_sha256':digest(Path(__file__)),'shared_measurement_sha256':digest(Path(shared.__file__))}
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1/WEIGHTED_ESTIMATOR_AUDIT.json'
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
