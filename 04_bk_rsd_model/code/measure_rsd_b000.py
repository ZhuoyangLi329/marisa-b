"""模块大纲：共用 catalog 读取；复用实空间 TSC/jaxpower 两腿测量；哈希与相位 QC。

不调用 reconstruction/P 测量，不用旧实空间协方差；随机项扣除完全沿用生产 estimator。
"""
import os
os.environ['OMP_NUM_THREADS']=os.environ.get('SLURM_CPUS_PER_TASK','1')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from rsd_b000_contract import *
from run_rsd_bonly_workflow import catalog
from periodic_rsd_measure import catalog_fourier

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=1)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--meshsize',type=int,default=256)
    parser.add_argument('--control-realspace',action='store_true');args=parser.parse_args()
    contract=json.loads((OUT/'contract.json').read_text());rows=contract['samples']
    if args.pilot:rows=[r for r in rows if r['pilot']]
    rows=rows[args.start:args.stop];out=OUT/f'measurements_m{args.meshsize}';out.mkdir(parents=True,exist_ok=True)
    common=PART.parent/'01_reconstruction_methods/common';sys.path.insert(0,str(common))
    # 与第一部分 entrypoint 相同，只满足共享模块导入；本入口不激活/调用 reconstruction。
    sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
    import jax
    jax.config.update('jax_enable_x64',True)
    from mpi4py import MPI
    import halo_b000_pipeline as shared
    from clustering_statistics.tools import Catalog
    shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD)
    ref=reference();cache={}
    axis=np.fft.fftfreq(32)*32;x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
    phase_mask=(x*x+y*y+z*z>=1)&(x*x+y*y+z*z<=9)
    for row in rows:
        cosmo=row['cosmology'];real=row['realization'];tag=f'{cosmo}_real{real:03d}';target=out/(tag+'.npz')
        if target.exists():
            with np.load(target) as saved:
                if saved['B'].shape!=(120,) or not np.isfinite(saved['B']).all():raise ValueError(target)
                meta=json.loads(str(saved['metadata_json']))
                if meta['contract_sha256']!=digest(OUT/'contract.json'):raise ValueError('stale measurement')
                if 'periodic_guard' not in meta:raise ValueError('checkpoint predates shared boundary guard; archive before rerun')
            print('REUSE '+tag,flush=True);continue
        t=time.time();pos,realpos,meta=catalog(cosmo,real)
        seed=shared.production_random_seed(RANDOM_NAMESPACE,cosmo,real,'full',30)
        randompos,randommeta=shared.generate_rank_random_positions(local_data_count=len(pos),boxsize=1000.,seed=seed,
                                                dtype=np.dtype('f4'),nran=30,mpicomm=MPI.COMM_WORLD)
        def guarded(value,label):
            # 原 real-space 生产入口已有同一 guard：wrap + 1e-3 Mpc/h 边界余量。
            # 只做 mod 不够：f4 的 (x+L/2)-L/2 仍会把 nextafter(L,0) 舍入为 L。
            safe=shared.wrap_positions_to_periodic_box(value,boxsize=1000.,dtype=np.dtype('f4'),
                                                       label=label,mpicomm=MPI.COMM_WORLD)
            count=int(np.count_nonzero(np.asarray(value,dtype='f4')!=safe))
            if not np.isfinite(safe).all() or np.any((safe+np.float32(500)-np.float32(500))/np.float32(1000)>=1):
                raise ValueError('shared boundary guard failed device-domain regression')
            return safe,count
        randompos,random_wrap_count=guarded(randompos,'RSD randoms')
        pos,data_wrap_count=guarded(pos,'RSD halos');realpos,real_wrap_count=guarded(realpos,'real-space phase QC')
        def compute(position):
            data=shared.make_halo_catalog(position,'z',MPI.COMM_WORLD,redshift=.5)
            randoms=Catalog({'POSITION':randompos,'INDWEIGHT':np.ones(len(randompos),dtype='f4')},attrs=data.attrs,mpicomm=MPI.COMM_WORLD)
            spectrum=shared.measure_b000(lambda:{'data':data,'shifted':randoms},boxsize=1000.,meshsize=args.meshsize,
                         edges=ref['edges'],los='z',basis='sugiyama',mask_edges_policy='default',cache=cache,mpicomm=MPI.COMM_WORLD)
            jax.block_until_ready(spectrum)
            pole=spectrum.get((0,0,0))
            value=np.asarray(pole.value(),dtype=float)
            shot=np.asarray(pole.values('num_shotnoise'))/np.asarray(pole.values('norm'))
            raw=value+shot
            kval=np.asarray(pole.coords('k'));edges=np.asarray(pole.edges('k'));nmodes=np.asarray(pole.values('nmodes'))
            if value.shape!=(120,) or not np.isfinite(value).all():raise ValueError('invalid B measurement')
            np.testing.assert_allclose(edges,ref['B000_k_edges'],rtol=0,atol=1e-14)
            # 同一个边缘合同，允许旧 GPU 的少量 shell-boundary ulp 差异，但绝不混入 RSD covariance。
            np.testing.assert_allclose(kval,ref['B000_k'],rtol=0,atol=5e-5)
            return {'B':value,'B_raw':raw,'B_shot':shot,'B000_k':kval,'B000_k_edges':edges,'B000_nmodes':nmodes}
        if args.control_realspace and real==0 and cosmo in ('fiducial','LC_p'):
            control=compute(realpos);old=OLD_SAMPLE_ROOT/cosmo/f'real{real:03d}'
            import h5py
            with h5py.File(old/'pre_b000.h5') as h:oldvalue=h['0_0_0/value'][()]
            delta=control['B']-oldvalue
            np.savez_compressed(OUT/f'REALSPACE_MEASUREMENT_CONTROL_{cosmo}.npz',**control,legacy_B=oldvalue,delta=delta)
            print('REALSPACE_CONTROL '+cosmo+' '+json.dumps({'random_seed':seed,'legacy_random_seed':json.loads((old/'metadata.json').read_text())['random_input']['seed'],
                   'max_relative_change':float(np.max(np.abs(delta)/np.maximum(1.,np.abs(oldvalue))))}),flush=True)
        result=compute(pos)
        meta.update(row);meta.update({'fNL_truth':{'fiducial':0,'LC_p':100,'LC_m':-100}[cosmo],
            'meshsize':args.meshsize,'data_has_rsd':True,'P_likelihood':False,'elapsed_seconds':time.time()-t,
            'contract_sha256':digest(OUT/'contract.json'),'random_seed':seed,'nran':30,
            'jax_backend':jax.default_backend(),'shared_measure_module':str(common/'halo_b000_pipeline.py'),
            'periodic_guard':'shared realspace wrap_positions_to_periodic_box: modulo + 1e-3 Mpc/h safety margin',
            'data_boundary_coordinates_wrapped':data_wrap_count,
            'random_boundary_coordinates_wrapped':random_wrap_count,'realspace_boundary_coordinates_wrapped':real_wrap_count,
            'measurement_entry_sha256':digest(Path(__file__)),
            'shared_measure_module_sha256':digest(common/'halo_b000_pipeline.py'),
            'legacy_nmodes_different_bins':int(np.count_nonzero(result['B000_nmodes']!=ref['B000_nmodes']))})
        result['phase_fingerprint']=catalog_fourier(realpos,32)[phase_mask]
        result['metadata_json']=np.array(json.dumps(meta))
        tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,**result);tmp.replace(target)
        print('MEASURE '+tag+' '+json.dumps(meta),flush=True)

if __name__=='__main__':main()
