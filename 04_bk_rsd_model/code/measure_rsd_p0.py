"""模块大纲：补测同一 RSD catalog 的 P0；复用共享 estimator/种子/边界 guard。

P 与既有 B 的原始 FoF SHA 必须相同，禁止以 realspace P 冒充 RSD P。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from rsd_pb_common import OUT,PART,LEGACY_OUT,SHARED,digest
from rsd_b000_contract import RANDOM_NAMESPACE
from run_rsd_bonly_workflow import catalog

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=1)
    parser.add_argument('--meshsize',type=int,default=256);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--fine',action='store_true');args=parser.parse_args()
    parent=json.loads((OUT/'contract.json').read_text());rows=parent['samples']
    for source in (SHARED,PART/'code/rsd_pb_common.py',Path(__file__)):
        if digest(source)!=parent['immutable_inputs'][str(source)]:raise ValueError('measurement source drift: '+str(source))
    if args.pilot:rows=[r for r in rows if r['pilot']]
    if args.fine:rows=[r for r in rows if r['role']=='validation' and r['realization']==0]
    rows=rows[args.start:args.stop];edges=np.array(parent['p_edges'])
    common=PART.parent/'01_reconstruction_methods/common';sys.path.insert(0,str(common))
    sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
    import jax
    jax.config.update('jax_enable_x64',True)
    from mpi4py import MPI
    import halo_b000_pipeline as shared
    from clustering_statistics.tools import Catalog
    cache={};out=OUT/f'measurements_m{args.meshsize}';packages=shared.get_package_report()
    for row in rows:
        cosmo=row['cosmology'];real=row['realization'];tag=f'{cosmo}_real{real:03d}'
        path=out/(tag+'.npz')
        if path.exists():
            with np.load(path) as z:
                meta=json.loads(str(z['metadata_json']))
                if meta['parent_sha256']!=digest(OUT/'contract.json') or meta['entry_sha256']!=digest(__file__):raise ValueError('P checkpoint drift')
            print('REUSE '+tag,flush=True);continue
        t=time.time();pos,_,meta=catalog(cosmo,real)
        with np.load(LEGACY_OUT/'measurements_m256'/(tag+'.npz')) as z:bmeta=json.loads(str(z['metadata_json']))
        if meta['source_raw_sha256']!=bmeta['source_raw_sha256']:raise ValueError('P/B raw sample mismatch')
        seed=shared.production_random_seed(RANDOM_NAMESPACE,cosmo,real,'full',30)
        if seed!=bmeta['random_seed']:raise ValueError('P/B random mismatch')
        randoms,_=shared.generate_rank_random_positions(local_data_count=len(pos),boxsize=1000.,seed=seed,
                            dtype=np.dtype('f4'),nran=30,mpicomm=MPI.COMM_WORLD)
        pos=shared.wrap_positions_to_periodic_box(pos,boxsize=1000.,dtype=np.dtype('f4'),label='PB halos',mpicomm=MPI.COMM_WORLD)
        randoms=shared.wrap_positions_to_periodic_box(randoms,boxsize=1000.,dtype=np.dtype('f4'),label='PB randoms',mpicomm=MPI.COMM_WORLD)
        data=shared.make_halo_catalog(pos,'z',MPI.COMM_WORLD,redshift=.5)
        random=Catalog({'POSITION':randoms,'INDWEIGHT':np.ones(len(randoms),dtype='f4')},attrs=data.attrs,mpicomm=MPI.COMM_WORLD)
        spectrum=shared.measure_pk(lambda:{'data':data,'shifted':random},boxsize=1000.,meshsize=args.meshsize,
                                  edges=edges,los='z',cache=cache,mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spectrum);pole=next(iter(spectrum))
        pk=np.asarray(pole.value(),dtype=float);k=np.asarray(pole.coords('k'));nm=np.asarray(pole.values('nmodes'))
        shot=np.asarray(pole.values('num_shotnoise'))/np.asarray(pole.values('norm'))
        if pk.shape!=(12,) or not np.isfinite(pk).all():raise ValueError('invalid P0')
        meta.update(row);meta.update(parent_sha256=digest(OUT/'contract.json'),entry_sha256=digest(__file__),
                    shared_sha256=digest(SHARED),random_seed=seed,meshsize=args.meshsize,data_has_rsd=True,
                    measurement_packages=packages,python_executable=sys.executable,
                    P_B_same_raw_sample=True,elapsed_seconds=time.time()-t,host=os.uname().nodename,
                    allowed_CPUs=sorted(os.sched_getaffinity(0)))
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,P0=pk,P0_shot=shot,k=k,nmodes=nm,edges=edges,
                                                          metadata_json=np.array(json.dumps(meta)))
        tmp.replace(path);print('P0 '+tag+' '+json.dumps(meta),flush=True)

if __name__=='__main__':main()
