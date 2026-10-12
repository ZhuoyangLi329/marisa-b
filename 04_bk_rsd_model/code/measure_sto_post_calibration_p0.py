"""模块大纲：独立 LC_p100..249 的 carried-STO RSD P0；只准备物理 p 校准输入。

复用原 catalog、权重身份检查、periodic guard 和 measure_pk。此处不重构、
不做额外 STO-only 科学比较、不拟合 p；不替换 mean100 target 或 Fid500 covariance。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
from run_sto_post_rsd_pilot import shared,validate_periodic_identity,spectrum_arrays,save_json
from post_rsd_catalog import catalog
from rsd_b000_contract import PART,digest,RANDOM_NAMESPACE
from rsd_pb_common import p_geometry
import jax
from mpi4py import MPI
from clustering_statistics.tools import Catalog
import jaxpower.mesh2 as mesh2_source

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'calibration_holdout_p0'
SOURCE=PART.parent/'01_reconstruction_methods/outputs/sto_marisa_v4_lcp250_source/LC_p'
RECIPE={'env_source':'dm','weight_source':'dm','gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.,'env_mesh':128,'recon_mesh':128}


def sources():
    paths=[Path(__file__),Path(shared.__file__),Path(mesh2_source.__file__),
           PART/'code/run_sto_post_rsd_pilot.py',PART/'code/run_rsd_bonly_workflow.py',
           PART/'code/post_rsd_catalog_identity.py',PART/'code/rsd_pb_common.py',
           PART/'code/post_rsd_catalog.py',
           PART/'code/build_rsd_p0_window.py']
    return {str(p):digest(p) for p in paths}


def freeze():
    audit=json.loads((ROOT/'CALIBRATION_INPUT_AUDIT.json').read_text())
    if audit['holdout_complete_count']!=150:raise ValueError('independent holdout inputs incomplete')
    _,edges=p_geometry()
    contract={'role':'independent physical p calibration input, weighted tracer before Marisa',
              'cosmology':'LC_p','fNL_truth':100,'realizations':list(range(100,250)),
              'held_out_from_main_validation_and_covariance':True,'z':.5,'LOS':'z','data_has_rsd':True,
              'source_root':str(SOURCE),'recipe':RECIPE,'weights_carried_from_realspace':True,
              'meshsize':256,'nran':30,'random_namespace':RANDOM_NAMESPACE,'P_edges':edges.tolist(),
              'main_validation_ids_unchanged':[0,99],'main_covariance_ids_unchanged':[1000,1499],
              'main_covariance_n':500,'main_covariance_of_mean':False,'main_volume_Gpc_h3':1,
              'no_fit_or_covariance_created':True,'immutable_sources':sources(),
              'input_audit_sha256':digest(ROOT/'CALIBRATION_INPUT_AUDIT.json')}
    contract['prior_failed_attempt']='calibration_holdout_p0_initial_missing_mass_20261009; optional mass cache absent, no P measured'
    OUT.mkdir(exist_ok=True);(OUT/'rows').mkdir(exist_ok=True)
    path=OUT/'CONTRACT.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('holdout contract drift; preserve existing outputs')
    else:save_json(path,contract)
    return contract


def validate_contract():
    contract=json.loads((OUT/'CONTRACT.json').read_text())
    if contract['immutable_sources']!=sources():raise ValueError('holdout measurement source drift')
    if contract['input_audit_sha256']!=digest(ROOT/'CALIBRATION_INPUT_AUDIT.json'):raise ValueError('input audit changed')
    return contract


def measure(index,contract):
    if index not in contract['realizations']:raise ValueError('not an independent holdout index')
    directory=OUT/'rows'/f'LC_p_real{index:03d}';directory.mkdir(exist_ok=True)
    if (directory/'DONE.json').exists():
        done=json.loads((directory/'DONE.json').read_text())
        if done['contract_sha256']!=digest(OUT/'CONTRACT.json') or done['measurement_sha256']!=digest(directory/'P0.npz'):
            raise ValueError('existing holdout checkpoint drift')
        print('REUSE '+directory.name,flush=True);return
    if (directory/'STATUS.json').exists():raise ValueError('incomplete holdout attempt; inspect without overwriting')
    start=time.time();state={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),
                            'affinity':sorted(os.sched_getaffinity(0)),'started_UTC':datetime.now(timezone.utc).isoformat()}
    save_json(directory/'STATUS.json',state)
    source=SOURCE/f'real{index:03d}/env_lam1_gsr2_thr15'
    diag=json.loads((source/'diagnostics.json').read_text())
    for key,value in RECIPE.items():
        if diag['config'][key]!=value:raise ValueError('STO recipe drift: '+key)
    if diag['component_config']['method']!='nenv' or diag['component_config']['base_weight']!='number':
        raise ValueError('wrong weight definition')
    pos,rawpos,rawmeta=catalog('LC_p',index)
    with np.load(source/'post_sto_halo.npz',allow_pickle=False) as z:
        identity=validate_periodic_identity(z['positions_mpc_h'],rawpos);weights=np.array(z['weights'])
    if weights.shape!=(len(pos),) or not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('invalid weights')
    seed=shared.production_random_seed(RANDOM_NAMESPACE,'LC_p',index,'full',30)
    randoms,randommeta=shared.generate_rank_random_positions(local_data_count=len(pos),boxsize=1000.,seed=seed,
                                 dtype=np.dtype('f4'),nran=30,mpicomm=MPI.COMM_WORLD)
    wrap=lambda x,label:shared.wrap_positions_to_periodic_box(x,boxsize=1000.,dtype=np.dtype('f4'),label=label,mpicomm=MPI.COMM_WORLD)
    pos=wrap(pos,'holdout STO RSD');randoms=wrap(randoms,'holdout uniform random')
    data=shared.make_halo_catalog(pos,'z',MPI.COMM_WORLD,weights=weights,redshift=.5)
    random=Catalog({'POSITION':randoms,'INDWEIGHT':np.ones(len(randoms),dtype='f4')},attrs=data.attrs,mpicomm=MPI.COMM_WORLD)
    edges=np.asarray(contract['P_edges'])
    spectrum=shared.measure_pk(lambda:{'data':data,'shifted':random},boxsize=1000.,meshsize=256,
                              edges=edges,los='z',cache={},mpicomm=MPI.COMM_WORLD)
    jax.block_until_ready(spectrum);arrays=spectrum_arrays(spectrum,'P0')
    if arrays['P0'].shape!=(12,) or any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('invalid P0 output')
    with np.load(ROOT/'pilots/fiducial_real000_rec256_meas256_ran30_selfmatched_zeroaudit/MEASUREMENTS.npz') as z:
        np.testing.assert_array_equal(arrays['P0_nmodes'],z['P_weighted_input_nmodes'])
        np.testing.assert_array_equal(edges,z['P_edges'])
    np.savez_compressed(directory/'P0.npz',**arrays,edges=edges)
    w=weights.astype(float)
    done={**state,'state':'COMPLETE','cosmology':'LC_p','realization':index,'raw_fof':rawmeta,
          'source_STO':str(source),'source_STO_sha256':digest(source/'post_sto_halo.npz'),
          'source_diagnostics_sha256':digest(source/'diagnostics.json'),'position_identity':identity,
          'random_seed':seed,'randoms':randommeta,'nhalo':len(pos),'N2_self_moment':float(1e9*np.sum(w*w)/w.sum()**2),
          'N3_self_moment':float(1e18*np.sum(w**3)/w.sum()**3),
          'contract_sha256':digest(OUT/'CONTRACT.json'),'measurement_sha256':digest(directory/'P0.npz'),
          'elapsed_seconds':time.time()-start,'finished_UTC':datetime.now(timezone.utc).isoformat(),
          'physical_p_fitted':False,'scientific_recovery_claimed':False}
    save_json(directory/'DONE.json',done);save_json(directory/'STATUS.json',done)
    print(json.dumps({'holdout':index,'state':'COMPLETE','elapsed_seconds':done['elapsed_seconds'],
                      'measurement_sha256':done['measurement_sha256']}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--start',type=int,default=100);ap.add_argument('--stop',type=int,default=101);args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1 or MPI.COMM_WORLD.size!=1:raise RuntimeError('single-core single-rank only')
    if args.freeze:
        freeze();print('HOLDOUT_P0_CONTRACT_FROZEN_NO_FIT',flush=True);return
    contract=validate_contract()
    for index in range(args.start,args.stop):
        try:
            validate_contract();measure(index,contract);gc.collect();jax.clear_caches()
        except Exception:
            path=OUT/'rows'/f'LC_p_real{index:03d}'/'STATUS.json'
            if path.exists():
                state=json.loads(path.read_text())
                if state.get('pid')==os.getpid() and state.get('state')=='RUNNING':
                    state.update(state='FAILED',error=traceback.format_exc(),updated_UTC=datetime.now(timezone.utc).isoformat())
                    save_json(path,state)
            raise


if __name__=='__main__':main()
