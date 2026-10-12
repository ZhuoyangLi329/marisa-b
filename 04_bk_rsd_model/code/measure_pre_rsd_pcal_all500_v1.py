"""模块大纲：使用全部完备LC_p0..499校准pre的物理p；只补缺少的RSD P0。

原0..99测量只读复用；先重放real000确认当前测量与原window/数据相容。
使用原FoF/RSD catalog、原随机种子与共享measure_pk，不安装软件或改原数据。
四个单线程worker使用CPU8..11，原校准/控制在CPU7/12，登录CPU不超过8核。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1')
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from rsd_b000_contract import PART,digest,save_json,RANDOM_NAMESPACE
from rsd_pb_common import SHARED,p_geometry
from post_rsd_catalog import catalog
from run_rsd_bonly_workflow import raw_path,header_check

OUT=PART/'outputs/rsd_all_available_p_recalibration_20261010_v2'
PRE=PART/'outputs/pre_rsd_pb_mean100_v1'
CPUS=(8,9,10,11)


def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def row_path(real):return OUT/'pre_positive_rows'/f'LC_p_real{real:03d}.npz'


def setup():
    if (OUT/'PRE_MEASUREMENT_SETTINGS.json').exists():return read(OUT/'PRE_MEASUREMENT_SETTINGS.json')
    for real in range(500):header_check(raw_path('LC_p',real))
    _,edges=p_geometry();OUT.mkdir(exist_ok=True)
    (OUT/'pre_positive_rows').mkdir(exist_ok=True)
    with np.load(PRE/'P0_LATTICE.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(edges,z['edges'])
    reused=[]
    for real in range(100):
        p=PRE/'measurements_m256'/f'LC_p_real{real:03d}.npz'
        with np.load(p,allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata_json']))
            if not metadata['data_has_rsd'] or metadata['realization']!=real or metadata['cosmology']!='LC_p':
                raise ValueError('actual pre RSD row identity mismatch')
        reused.append(dict(realization=real,path=str(p),sha256=digest(p)))
    record=dict(created_UTC=stamp(),direct_user_request='所有p重新校准；使用全部可用realization',
        cosmology='LC_p',truth_fNL=100,calibration_ids=list(range(500)),target_count=500,
        reused_original_rows=reused,new_measurement_ids=list(range(100,500)),
        meshsize=256,P_edges=edges.tolist(),kmax_P=.08,LOS='z',z=.5,
        boxsize_Mpc_h=1000.,mass_min_Msun_h=1e13,nran=30,
        random_namespace=RANDOM_NAMESPACE,worker_CPUs=list(CPUS),
        calibration_main_target_overlap_ids=list(range(100)),
        strict_independent_calibration_claimed=False,
        source_sha256={str(p):digest(p) for p in [Path(__file__),SHARED,
            PART/'code/post_rsd_catalog.py',PART/'code/run_rsd_bonly_workflow.py',
            PART/'code/rsd_pb_common.py',PRE/'P0_LATTICE.npz']})
    save_json(OUT/'PRE_MEASUREMENT_SETTINGS.json',record);return record


def measure(real,pilot=False):
    settings=setup();target=OUT/'PRE_REAL000_REPLAY.npz' if pilot else row_path(real)
    if target.exists():
        if not pilot:
            done=read(target.with_suffix('.json'))
            if done['sha256']!=digest(target):raise ValueError('saved measurement changed')
        print('REUSE_PRE_P0',real,flush=True);return
    for path,want in settings['source_sha256'].items():
        if digest(path)!=want:raise ValueError('measurement implementation changed during this run: '+path)
    common=PART.parent/'01_reconstruction_methods/common'
    sys.path.insert(0,str(common));sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
    import jax
    jax.config.update('jax_enable_x64',True)
    from mpi4py import MPI
    import halo_b000_pipeline as shared
    from clustering_statistics.tools import Catalog
    if MPI.COMM_WORLD.size!=1 or len(os.sched_getaffinity(0))!=1:
        raise ValueError('one logical core/rank required')
    if pilot:
        # 直接检查实际生产binner的逐mode标签/权重，不以平均k坐标近似它。
        from jaxpower import MeshAttrs,BinMesh2SpectrumPoles
        attrs=MeshAttrs(meshsize=256,boxsize=np.ones(3,dtype='f4')*1000,
            boxcenter=np.ones(3,dtype='f4')*500,dtype='f4')
        bins=BinMesh2SpectrumPoles(attrs,edges=np.array(settings['P_edges']),ells=(0,))
        coords=attrs.kcoords(kind='index',sparse=True)
        shape=np.broadcast_shapes(*[q.shape for q in coords]);ibin=np.asarray(bins.ibin[0]).reshape(shape)
        mask=(ibin>=1)&(ibin<=12)
        vectors=np.column_stack([np.rint(np.broadcast_to(np.asarray(q),shape)[mask]).astype('i2') for q in coords])
        weights=np.ones(shape,dtype='f8') if bins.wmodes is None else np.asarray(bins.wmodes).reshape(shape)
        with np.load(PRE/'P0_WINDOW.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(vectors,z['vectors_integer'])
            np.testing.assert_array_equal(ibin[mask]-1,z['labels'])
            np.testing.assert_array_equal(weights[mask],z['weights'])
        print('ACTUAL_PRE_MODE_WINDOW_LABELS_AND_WEIGHTS_BITWISE_MATCH',flush=True)
    start=time.monotonic();pos,_,meta=catalog('LC_p',real)
    seed=shared.production_random_seed(RANDOM_NAMESPACE,'LC_p',real,'full',30)
    randoms,_=shared.generate_rank_random_positions(local_data_count=len(pos),boxsize=1000.,
        seed=seed,dtype=np.dtype('f4'),nran=30,mpicomm=MPI.COMM_WORLD)
    wrap=lambda x,label:shared.wrap_positions_to_periodic_box(x,boxsize=1000.,dtype=np.dtype('f4'),
        label=label,mpicomm=MPI.COMM_WORLD)
    data=shared.make_halo_catalog(wrap(pos,'pre pcal RSD'),'z',MPI.COMM_WORLD,redshift=.5)
    random=Catalog({'POSITION':wrap(randoms,'pre pcal random'),
        'INDWEIGHT':np.ones(len(randoms),dtype='f4')},attrs=data.attrs,mpicomm=MPI.COMM_WORLD)
    edges=np.array(settings['P_edges'])
    spectrum=shared.measure_pk(lambda:{'data':data,'shifted':random},boxsize=1000.,meshsize=256,
        edges=edges,los='z',cache={},mpicomm=MPI.COMM_WORLD)
    jax.block_until_ready(spectrum);pole=next(iter(spectrum))
    values=np.asarray(pole.value(),dtype=float);k=np.asarray(pole.coords('k'))
    nmodes=np.asarray(pole.values('nmodes'))
    shot=np.asarray(pole.values('num_shotnoise'))/np.asarray(pole.values('norm'))
    with np.load(PRE/'P0_LATTICE.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(nmodes,z['nmodes'])
        # k平均值是float32展示坐标，不参与模型；模型使用冻结的逐mode模板。
        # 原1e-6检查的失败原样记录，不把坐标求和ULP当作分箱/物理误差。
        coordinate_difference=float(np.max(abs(k-z['k'])/np.maximum(abs(z['k']),1e-30)))
        old_coordinate_check=bool(np.allclose(k,z['k'],rtol=1e-6,atol=1e-10))
    if values.shape!=(12,) or not np.isfinite(values).all():raise ValueError('invalid actual P0 output')
    meta.update(cosmology='LC_p',realization=real,truth_fNL=100,data_has_rsd=True,
        random_seed=seed,meshsize=256,CPU_affinity=sorted(os.sched_getaffinity(0)),
        measurement_entry_sha256=digest(__file__),shared_sha256=digest(SHARED),
        nmodes_exactly_match_original_window=True,
        plotting_k_coordinate_relative_difference=coordinate_difference,
        original_plotting_coordinate_1e6_check_passed=old_coordinate_check,
        model_uses_original_mode_templates_not_plotting_k_coordinate=True)
    np.savez_compressed(target,P0=values,P0_shot=shot,k=k,nmodes=nmodes,edges=edges,
        metadata_json=np.array(json.dumps(meta)))
    done=dict(realization=real,sha256=digest(target),elapsed_seconds=time.monotonic()-start,
        finished_UTC=stamp(),pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),raw=meta)
    if pilot:
        with np.load(PRE/'measurements_m256/LC_p_real000.npz',allow_pickle=False) as old:
            delta=values-old['P0'];np.testing.assert_array_equal(nmodes,old['nmodes'])
        with np.load(PRE/'cases/LC_p_mean100_k0.08/likelihood.npz',allow_pickle=False) as z:
            white=np.linalg.solve(np.linalg.cholesky(z['covariance'][:12,:12]),delta)
        error=float(np.linalg.norm(white))
        done.update(single_box_whitened_replay_error=error,original_data_not_replaced=True)
        if error>=.01:
            save_json(OUT/'PRE_REPLAY_CHECK.json',dict(done,status='NUMERICAL_REPLAY_DIFFERENCE_REQUIRES_DIAGNOSIS'))
            raise ValueError('current pre measurement differs from original by >=0.1 mean100 sigma')
        done['status']='ACTUAL_PRE_RSD_P0_REPLAY_COMPATIBLE'
        save_json(OUT/'PRE_REPLAY_CHECK.json',done)
    else:save_json(target.with_suffix('.json'),done)
    print('PRE_P0_ACTUAL_COMPLETE',real,done['elapsed_seconds'],flush=True)
    del spectrum,data,random,randoms,pos;gc.collect();jax.clear_caches()


def worker(slot):
    if os.sched_getaffinity(0)!={CPUS[slot]}:raise ValueError('configured one CPU per measurement worker')
    for real in range(100+slot,500,4):measure(real)


def controller():
    settings=setup()
    if read(OUT/'PRE_REPLAY_CHECK.json')['status']!='ACTUAL_PRE_RSD_P0_REPLAY_COMPATIBLE':
        raise ValueError('actual old/new measurement compatibility check required')
    state=dict(state='MEASURING_MISSING_PRE_CALIBRATION_P0_ALL500',pid=os.getpid(),host=socket.gethostname(),
        started_UTC=stamp(),CPU_affinity=[12],worker_pids=[],completed_new_rows=0)
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'PRE_MEASUREMENT_STATUS.json',state)
    active=[];update()
    try:
        for slot,cpu in enumerate(CPUS):
            stream=(OUT/f'pre_worker_{slot}.log').open('x')
            proc=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                '--worker',str(slot)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active.append((proc,stream));update(worker_pids=[p.pid for p,_ in active])
        while any(p.poll() is None for p,_ in active):
            update(completed_new_rows=len(list((OUT/'pre_positive_rows').glob('*.json'))));time.sleep(10)
        if any(p.returncode!=0 for p,_ in active):raise ValueError('actual measurement worker failed; preserve outputs')
        if len(list((OUT/'pre_positive_rows').glob('*.json')))!=400:raise ValueError('not all400 missing rows complete')
        update(state='ALL500_PRE_POSITIVE_P0_ROWS_READY',completed_new_rows=400,worker_pids=[])
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise
    finally:
        for proc,stream in active:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
            stream.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if args.pilot:measure(0,pilot=True)
    elif args.worker is not None:worker(args.worker)
    elif args.detach:
        setup();launch=OUT/'PRE_MEASUREMENT_LAUNCH.json';log=OUT/'pre_measurement.log'
        if launch.exists() or log.exists():raise ValueError('inspect existing process; no duplicate launch')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),created_UTC=stamp(),CPU_affinity=[12]))
        print(json.dumps(read(launch)),flush=True)
    else:controller()
