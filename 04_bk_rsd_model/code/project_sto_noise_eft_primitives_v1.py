"""模块大纲：独立EFT候选的原44-window谱矩与matched零腿；双网格/线性卷积回放。

不指定噪声幅度/先验、不启动MCMC；只新增独立候选输出，原700继续。
有限PNG的6个单谱矩及真实zero atom由同一full-shell denominator投影。
oracle用有界整数球和非循环linear convolution，不调用生产factor/integral。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
from scipy.signal import fftconvolve
from rsd_b000_contract import PART,reference,selected,digest,save_json,load_spectrum,F_GROWTH
from post_rsd_ir_inputs import ROOT,ALPHA,frozen_reconstruction
from post_rsd_fft_projection import PostFFTProjection
from post_rsd_stochastic_eft_v1 import (EFTPrimitiveFields,PRIMITIVES,ZERO_PRIMITIVES,
    project_intrinsic_primitives,project_zero_primitives)

OUT=ROOT/'noise_eft_primitives_v1'
TOL=1e-9


class EFTFFT(EFTPrimitiveFields,PostFFTProjection):
    pass


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def independent_linear_oracle(edges,pair,reconstruction):
    """整数有界球直接定义shell；反向partner计数与非循环closing直方图。

    两个非零input shell，不限制第三腿；scalar全部为even，closing反号不改值。
    integer roundoff门仅用于pair multiplicities，不截断任何谱或科学系数。
    """
    i,j=map(int,pair);kf=2*np.pi/1000.
    extent=int(np.ceil(max(edges[i+1],edges[j+1])/kf))+1
    axis=np.arange(-extent,extent+1);v=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)*kf
    k=np.sqrt(np.sum(v*v,axis=-1))
    canonical=k.copy()
    for edge in edges[1:-1]:
        canonical[np.abs(canonical-edge)<=64*np.finfo(float).eps*max(1.,abs(float(edge)))]=edge
    def shell(index):
        upper=(canonical<edges[index+1])
        if index==len(edges)-2:
            upper=canonical<=edges[index+1]+32*np.finfo(float).eps*max(1.,abs(float(edges[index+1])))
        return (canonical>=edges[index])&upper
    left=shell(i);right=shell(j)
    if np.any((left|right)&(k==0)):raise ValueError('original outer shells must exclude zero')
    nleft=int(left.sum());nright=int(right.sum());den=nleft*nright
    if den<=0:raise ValueError('empty original shell')
    raw=fftconvolve(left.astype(float),right.astype(float),mode='full')
    pairs=np.rint(raw)
    rounding=float(np.max(np.abs(raw-pairs)))
    if rounding>1e-5 or pairs.min()<0 or int(pairs.sum())!=den:
        raise ValueError('independent integer linear convolution fails')
    del raw
    axis2=np.arange(-2*extent,2*extent+1)
    vectors=np.stack(np.meshgrid(axis2,axis2,axis2,indexing='ij'),axis=-1)*kf
    kk=np.sqrt(np.sum(vectors*vectors,axis=-1));active=kk>0
    spectrum=np.zeros((6,)+kk.shape);p,m=load_spectrum()(kk[active])
    mu=vectors[...,2][active]**2/kk[active]**2;u=1/m
    spectrum[:,active]=np.array([p,p*mu,p*mu*mu,p*u,p*mu*u,p*u*u])
    # 原input球位于linear convolution较大球的中央切片。
    start=extent;stop=3*extent+1;small=spectrum[:,start:stop,start:stop,start:stop]
    multiplicity=left*(nright-right)+right*(nleft-left)
    first=np.sum(small*multiplicity,axis=(1,2,3))
    center=2*extent;zero_count=int(pairs[center,center,center]);pairs[center,center,center]=0.
    closing=np.sum(spectrum*pairs,axis=(1,2,3))
    intrinsic=np.r_[(first+closing)/den,(den-zero_count)/den]
    common=left&right;pv=small[:,common];r=reconstruction.response(v[common],np.ones(int(common.sum())))
    zero=np.r_[pv.sum(axis=1)/den,(pv*r).sum(axis=1)/den,r.sum()/den,int(common.sum())/den]
    if zero_count!=int(common.sum()):raise ValueError('closing zero multiplicity differs from shell overlap')
    return intrinsic,zero,dict(full_denominator=den,zero_closing_count=zero_count,
        closing_count=den-zero_count,maximum_integer_roundoff=rounding,bounded_input_extent=extent)


def pipeline():
    if OUT.exists():raise ValueError('preserve existing EFT candidate projection')
    production=read(ROOT/'simultaneous_measurements_v3/CONTRACT.json')
    for p,want in production['immutable_sources'].items():
        if digest(p)!=want:raise ValueError('production frozen source drift')
    linked=ROOT/'noise_actual_connected_probes_v1/INDEPENDENT_REPLAY_AUDIT.json'
    if read(linked)['status']!='INDEPENDENT_ACTUAL300_MARKED_FINITE_RANDOM_CONNECTED_PROBE_REPLAY_PASS_NOT_STO_CLOSURE':
        raise ValueError('actual300 independent marked replay required')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_stochastic_eft_v1'))
    if not tests.wasSuccessful() or tests.testsRun!=8:raise ValueError('actual EFT8 unit tests required')
    geometry=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_sto_noise_eft_projection_v1'))
    if not geometry.wasSuccessful() or geometry.testsRun!=1:raise ValueError('independent small-lattice projection test required')
    ref=reference();indices=selected(.20).tolist()
    original=ROOT/'noir_paware_templates/CONTRACT.json';parent=read(original);meshes=parent['meshes']
    if parent['indices']!=indices or len(indices)!=44:raise ValueError('original full44 registry required')
    w=ROOT/'estimator_B_window/B000_WINDOW.npz'
    if digest(w)!=read(ROOT/'estimator_B_window/AUDIT.json')['window_sha256']:
        raise ValueError('actual B window identity changed')
    with np.load(w,allow_pickle=False) as z:counts=z['measured_pair_counts'];np.testing.assert_array_equal(z['fit_indices'],indices)
    sources=[Path(__file__),PART/'code/post_rsd_stochastic_eft_v1.py',PART/'code/test_post_rsd_stochastic_eft_v1.py',
             PART/'code/test_sto_noise_eft_projection_v1.py',
             PART/'code/post_rsd_fft_projection.py',PART/'code/rsd_b000_fft_projection.py',
             ALPHA,original,linked,w]
    OUT.mkdir();save_json(OUT/'CONTRACT.json',dict(status='INDEPENDENT_EFT_TREE_CANDIDATE_PRIMITIVES_NOT_STO_CLOSURE',
        immutable_inputs={str(p):digest(p) for p in sources},indices=indices,meshes=meshes,
        intrinsic_primitives=list(PRIMITIVES),matched_zero_primitives=list(ZERO_PRIMITIVES),
        denominator='unchanged actual full two-shell mode counts',closing_leg_unrestricted=True,
        data_vectors_changed=False,noise_amplitudes_or_priors_selected=False,MCMC_released=False,
        original8_parameter_model_preserved=True,original700_pipeline_preserved=True,scaled_tolerance=TOL))
    state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[14],
        started_UTC=stamp(),completed_bins=0)
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update();start=time.time();rows=[];intrinsic=[];zero=[];recon=frozen_reconstruction(256)
    try:
        for index in indices:
            update(bin=index,stage='dual_grid_and_independent_linear_oracle')
            oracle,zo,meta=independent_linear_oracle(ref['edges'],ref['shell_pairs'][index],recon)
            values=[];zeros=[];errors=[]
            for mesh in meshes[str(index)]:
                used=int(Path('/sys/fs/cgroup/user.slice/user-112513.slice/memory.current').read_text())
                if used>28*1024**3:raise ValueError('preserve failed candidate: no4GiB login memory headroom')
                fields=EFTFFT(mesh,ref['edges'],ref['shell_pairs'][index],2.23,F_GROWTH,
                              p_halo=1.3,reconstruction=recon)
                if fields.denominator!=int(counts[index]) or fields.denominator!=meta['full_denominator']:
                    raise ValueError('original measured full denominator differs')
                value=project_intrinsic_primitives(fields);zz=project_zero_primitives(fields)
                error=max(float(np.max(np.abs(value-oracle)/np.maximum(1.,np.abs(oracle)))),
                          float(np.max(np.abs(zz-zo)/np.maximum(1.,np.abs(zo)))))
                if error>TOL:raise ValueError('predefined independent primitive gate failed')
                errors.append(error);values.append(value);zeros.append(zz)
                del fields;gc.collect()
            intrinsic.append(values[-1]);zero.append(zeros[-1]);rows.append(dict(bin=index,meshes=meshes[str(index)],
                independent_scaled_errors=errors,oracle_count_audit=meta))
            update(completed_bins=len(rows))
        matrix=OUT/'PRIMITIVES.npz';np.savez_compressed(matrix,indices=indices,
            intrinsic=np.array(intrinsic).T,matched_zero=np.array(zero).T)
        save_json(OUT/'AUDIT.json',dict(status='ALL44_EFT_INTRINSIC_AND_MATCHED_ZERO_PRIMITIVES_PASS_NOT_STO_CLOSURE',
            created_UTC=stamp(),bins=44,independent_linear_oracles=44,dual_grid_projections=88,
            unit_tests_actual_NERSC_passed=9,maximum_scaled_error=max(max(r['independent_scaled_errors']) for r in rows),
            original_scaled_tolerance=TOL,rows=rows,matrix_sha256=digest(matrix),contract_sha256=digest(OUT/'CONTRACT.json'),
            CPU_affinity=[14],elapsed_seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            actual_STO_noise_amplitudes_calibrated=False,finite_shift_validated=False,IR_noise_validated=False,
            formal_noise_audit_released=False,MCMC_started=False,original_data_vectors_changed=False))
        update(state='COMPLETE_CANDIDATE_PRIMITIVES_NOT_STO_CLOSURE',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('login26 CPU14 single core required')
    if args.detach:
        log=ROOT/'noise_eft_primitives_v1.log';launch=ROOT/'noise_eft_primitives_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect original candidate before any restart')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
