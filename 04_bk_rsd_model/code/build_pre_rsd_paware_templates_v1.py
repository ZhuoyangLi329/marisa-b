"""模块大纲：原pre tree/FFT生成显式p轴，固定旧数据/分箱/谱和噪声形状。

复用唯一integrate_design；p只在此独立进程内代入原PBS，不改共享源码。
b1三次/p二次是finite-fNL²的精确表示，p轴不是主拟合自由参数。
原固定p系数、独立有向native源与非节点FFT均要回放；不只更新metadata。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import rsd_bonly_common as common
from rsd_b000_contract import PART,BNODES,F_GROWTH,selected,reference,digest,save_json
from rsd_b000_fft_projection import FFTProjection,Fields,integrate_design
from post_rsd_stochastic_eft_v1 import EFTPrimitiveFields,project_intrinsic_primitives
from post_rsd_noir_polynomial import P_NODES,P_CENTER,P_RADIUS,evaluate_noir_design,collapse_physical_p
from tree_png_rsd_native import NativeTreeRSD,RsdConfig
from run_all_available_rsd_pcal_v1 import OUT as BASE,stamp

OUT=BASE/'pre_b_paware_templates_v1'
PRE=PART/'outputs/pre_rsd_pb_mean100_v1'
OLD=PRE/'cases/fiducial_mean100_k0.14/likelihood.npz'
CPUS=(12,14)
PROBES=((.731,.6),(1.83641,1.41421),(4.,2.31))

def read(path):return json.loads(Path(path).read_text())

@contextmanager
def physical_p(p):
    old=common.P_CAL;common.P_CAL=float(p)
    try:yield
    finally:common.P_CAL=old

def fit_pre_paware(node_designs):
    values=np.asarray(node_designs,dtype=float)
    if values.shape[:4]!=(4,3,3,3) or not np.isfinite(values).all():
        raise ValueError('finite [b1:4,p:3,nuisance:3,fNL:3,...] pre design required')
    b,p=np.meshgrid((BNODES-2.25)/1.75,(P_NODES-P_CENTER)/P_RADIUS,indexing='ij')
    v=np.polynomial.chebyshev.chebvander2d(b.ravel(),p.ravel(),(3,2))
    coef=np.linalg.solve(v,values.reshape(12,-1)).reshape(values.shape)
    result=np.zeros((5,)+values.shape[1:]);result[:4]=coef
    return result

class Projection(EFTPrimitiveFields,FFTProjection):pass

def assign_bias(projector,b,p):
    # 无复制模型；Fields/source_terms都调用原PBS，仅清除依赖bias的FFT cache。
    projector.b1=float(b);projector.z0=projector.b1+projector.f*projector.mu2
    with physical_p(p):projector.z1=common.bias(projector.b1).bphi*projector.u
    projector.cache.clear()

def native_point_audit():
    rng=np.random.default_rng(400710);v=rng.normal(size=(29,3,3))*.04;v[:,2]=-v[:,:2].sum(axis=1)
    power,transfer=common.load_spectrum()(np.linalg.norm(v,axis=-1));native=NativeTreeRSD();errors=[]
    for b,p in ((.5,.4),(1.83641,1.179),(2.71,1.51),(4.,2.6)):
        for fg in (0.,F_GROWTH):
            with physical_p(p):
                design=integrate_design(Fields(v,power,transfer,b,fg))
                direct=native.evaluate(v,power,transfer,common.bias(b,.43,-.17),RsdConfig(fg)).total
            expected=np.tensordot([1.,.43,-.17],design,axes=(0,0)).T
            scale=np.maximum(1.,np.max(abs(direct)*[1,500,250000],axis=1))
            error=float(np.max(abs(expected-direct)*[1,500,250000]/scale[:,None]));errors.append(error)
    if max(errors)>2e-10:raise ValueError('explicit-p pre source differs from independent native tree')
    return dict(oriented_modes=29,parameter_pairs=4,growth_values=2,maximum_scaled_error=max(errors))

def setup():
    path=OUT/'SETTINGS.json'
    if path.exists():
        value=read(path)
        for source,want in value['source_sha256'].items():
            if digest(source)!=want:raise ValueError('projection input/source changed during this run: '+source)
        return value
    OUT.mkdir();(OUT/'bins').mkdir()
    native=native_point_audit()
    sources=[Path(__file__),PART/'code/rsd_bonly_common.py',PART/'code/rsd_b000_contract.py',
        PART/'code/rsd_b000_fft_projection.py',PART/'code/post_rsd_stochastic_eft_v1.py',
        PART/'code/post_rsd_noir_polynomial.py',PART/'code/tree_png_rsd_native.py',
        PART/'work/build/libmarisa_tree_rsd.so',PART/'work/reference_package/python/marisa_b/tree_png_rsd_reference.py',
        PART/'outputs/realspace_binning_reference/REALSPACE_BINNING.npz',common.TABLE,OLD]
    value=dict(created_UTC=stamp(),indices=selected(.14).tolist(),meshsize=144,B_cut=.14,
        b1_nodes=BNODES.tolist(),p_nodes=P_NODES.tolist(),b1_degree=3,p_degree=2,
        fixed_f_growth=F_GROWTH,IR=False,finite_fNL_order=2,p_is_not_a_new_MCMC_parameter=True,
        old_fixed_p=1.179,PROBES=[list(x) for x in PROBES],native_source=native,CPUs=list(CPUS),
        source_sha256={str(p):digest(p) for p in sources})
    save_json(path,value);return value

def build(index):
    settings=setup();path=OUT/'bins'/f'bin{index:03d}.npz'
    if path.exists():
        done=read(path.with_suffix('.json'))
        if digest(path)!=done['sha256']:raise ValueError('saved actual bin changed')
        print('REUSE_EXPLICIT_P_BIN',index,flush=True);return
    ref=reference();start=time.monotonic();oldp=common.P_CAL
    with physical_p(P_NODES[0]):projector=Projection(144,ref['edges'],ref['shell_pairs'][index],BNODES[0])
    sampled=[]
    for b in BNODES:
        row=[]
        for p in P_NODES:
            assign_bias(projector,b,p)
            with physical_p(p):row.append(integrate_design(projector))
            print('ACTUAL_PRE_PAWARE_NODE',index,float(b),float(p),time.monotonic()-start,flush=True)
        sampled.append(row)
    coef=fit_pre_paware(np.asarray(sampled));primitives=project_intrinsic_primitives(projector)
    probe_delta=[];probe_errors=[]
    for b,p in PROBES:
        assign_bias(projector,b,p)
        with physical_p(p):actual=integrate_design(projector)
        predicted=evaluate_noir_design(coef,b,p_halo=p)
        powers=np.array([1,500,250000])[None,:]
        scale=np.maximum(1.,np.max(abs(actual)*powers,axis=1,keepdims=True))
        error=float(np.max(abs(predicted-actual)*powers/scale));probe_errors.append(error);probe_delta.append(predicted-actual)
    if max(probe_errors)>2e-10:raise ValueError('actual nonnode explicit-p polynomial failed original tolerance')
    with np.load(OLD,allow_pickle=False) as z:
        local=int(np.flatnonzero(z['indices']==index)[0]);oldcoef=z['coefficients'][...,12+local].copy()
    baseline=collapse_physical_p(coef,p_halo=1.179)[:4]
    powers=np.array([1,500,250000])[None,None,:]
    scale=np.maximum(1,np.max(abs(oldcoef)*powers,axis=(0,2),keepdims=True))
    baseline_error=float(np.max(abs(baseline-oldcoef)*powers/scale))
    if baseline_error>2e-10:raise ValueError('p=1.179 does not reproduce original pre deterministic coefficients')
    if common.P_CAL!=oldp:raise ValueError('temporary explicit p override leaked within this process')
    np.savez_compressed(path,coefficients=coef,primitives=primitives,index=index,
        probe_delta=np.array(probe_delta),probe_errors=probe_errors,baseline_coefficient_delta=baseline-oldcoef,
        counts=projector.counts,settings_sha256=np.array(digest(OUT/'SETTINGS.json')))
    save_json(path.with_suffix('.json'),dict(index=index,sha256=digest(path),finished_UTC=stamp(),
        elapsed_seconds=time.monotonic()-start,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        maximum_probe_error=max(probe_errors),old_p_baseline_error=baseline_error,
        cache_peak_bytes=projector.cache_peak_bytes,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('PRE_PAWARE_BIN_COMPLETE',index,time.monotonic()-start,baseline_error,max(probe_errors),flush=True)
    del projector;gc.collect()

def assemble():
    settings=setup();indices=np.array(settings['indices']);coeff=[];primitives=[];delta=[]
    for index in indices:
        path=OUT/'bins'/f'bin{index:03d}.npz'
        if digest(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('actual p-aware bin changed')
        with np.load(path,allow_pickle=False) as z:
            coeff.append(z['coefficients']);primitives.append(z['primitives']);delta.append(z['probe_delta'])
    coef=np.stack(coeff,axis=-1);primitives=np.stack(primitives,axis=-1);delta=np.stack(delta,axis=-1)
    with np.load(OLD,allow_pickle=False) as z:cov=z['covariance'].copy()
    errors=[]
    for d in delta:
        for f in (-500.,100.,500.):
            for b2,bk in ((-4.,-4.),(6.,4.)):
                vector=np.r_[np.zeros(12),np.einsum('i,j,ijn->n',[1,b2,bk],[1,f,f*f],d)]
                errors.append(float(np.max(abs(np.linalg.solve(np.linalg.cholesky(cov),vector)))))
    if max(errors)>1e-5:raise ValueError('actual full-joint pre covariance template difference exceeds original1e-5 tolerance')
    path=OUT/'PRE_B_PAWARE.npz';np.savez_compressed(path,coefficients=coef,primitives=primitives,indices=indices)
    save_json(OUT/'COMPLETION_AUDIT.json',dict(created_UTC=stamp(),status='PRE_EXPLICIT_PHYSICAL_P_B_TEMPLATE_NUMERICALLY_VERIFIED',
        actual_bin_count=len(indices),B_cut=.14,old_p_baseline_reproduced=True,
        nonnode_FFT_probes_per_bin=len(PROBES),full_joint_whitened_max_probe_error=max(errors),
        p_is_not_a_new_parameter=True,physical_p_calibration_not_replaced=True,
        original_measurements_and_fixed_p_coefficients_not_modified=True,
        model_formula_changed=False,sha256=digest(path),settings_sha256=digest(OUT/'SETTINGS.json')))

def worker(slot):
    if os.sched_getaffinity(0)!={CPUS[slot]}:raise ValueError('one logical CPU per projection worker')
    for index in setup()['indices'][slot::len(CPUS)]:build(index)

def controller():
    settings=setup();state=dict(pid=os.getpid(),host=socket.gethostname(),started_UTC=stamp(),
        state='BUILDING_PRE_EXPLICIT_P_B_TEMPLATES',worker_pids=[])
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    active=[];update()
    try:
        for slot,cpu in enumerate(CPUS):
            stream=(OUT/f'worker_{slot}.log').open('x')
            child=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                '--worker',str(slot)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active.append((child,stream));update(worker_pids=[p.pid for p,_ in active])
        while any(p.poll() is None for p,_ in active):
            update(completed_bins=len(list((OUT/'bins').glob('*.json'))));time.sleep(5)
        if any(p.returncode!=0 for p,_ in active):raise ValueError('actual template worker failed; preserve diagnostics')
        assemble();update(state='PRE_EXPLICIT_P_B_TEMPLATES_COMPLETE',completed_bins=len(settings['indices']),worker_pids=[])
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise
    finally:
        for proc,stream in active:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
            stream.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if args.pilot:build(setup()['indices'][0])
    elif args.worker is not None:worker(args.worker)
    elif args.detach:
        setup();launch=OUT/'LAUNCH.json';log=OUT/'controller.log'
        if launch.exists() or log.exists():raise ValueError('inspect actual existing projection; no duplicate')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),created_UTC=stamp(),CPUs=list(CPUS)))
        print(json.dumps(read(launch)),flush=True)
    else:controller()
