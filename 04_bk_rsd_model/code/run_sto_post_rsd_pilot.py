"""模块大纲：冻结 STO 缓存身份；RSD carry-over；原 Marisa 重构；原 P/B estimator。

仅生成具名 pilot，不放行正式 ensemble 或 MCMC。读取冻结 alpha 表核对实际
重构 helper 的取值；b_rec/p_rec/fNL_rec 是固定算法输入，物理 p 尚不在此拟合。
全部输出限定第四部分；旧缓存、旧 pre 与共享 production helper 均不修改。
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

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/common'))
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
import jax
jax.config.update('jax_enable_x64',True)
from mpi4py import MPI
from clustering_statistics.tools import Catalog
from marisa_recon_runner_tools import run_marisa_jaxrecon
import halo_b000_pipeline as shared
from run_rsd_bonly_workflow import catalog
from rsd_b000_contract import reference,digest,RANDOM_NAMESPACE
from rsd_pb_common import p_geometry
from inventory_sto_post_inputs import SOURCE
from post_rsd_catalog_identity import validate_periodic_identity
PILOT_STATUS_PATH=None


def save_json(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def spectrum_arrays(spectrum,prefix):
    pole=next(iter(spectrum));value=np.asarray(pole.value(),dtype=float)
    shot=np.asarray(pole.values('num_shotnoise'),dtype=float)/np.asarray(pole.values('norm'),dtype=float)
    return {prefix:value,prefix+'_raw':value+shot,prefix+'_shot':shot,
        prefix+'_k':np.asarray(pole.coords('k')),prefix+'_nmodes':np.asarray(pole.values('nmodes')),
        prefix+'_norm':np.asarray(pole.values('norm'))}


def main():
    global PILOT_STATUS_PATH
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cosmology',choices=('fiducial','LC_p','LC_m'),required=True)
    ap.add_argument('--realization',type=int,default=0)
    ap.add_argument('--recon-mesh',type=int,choices=(125,128,256,512),required=True)
    ap.add_argument('--measurement-mesh',type=int,choices=(256,512),default=256)
    ap.add_argument('--nran',type=int,default=30)
    ap.add_argument('--self-contraction',choices=('legacy','matched'),default='legacy')
    ap.add_argument('--zero-atom-diagnostics',action='store_true',help='write a separate counterfactual, do not alter B_post')
    ap.add_argument('--grid-origin-fraction',type=float,choices=(0.,.5),default=0.,help='translate both inputs by this fraction of a recon cell, then undo translation')
    ap.add_argument('--nested-randoms',action='store_true',help='use the nran=30 seed namespace for a paired random-density test')
    args=ap.parse_args()
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('pilot must run inside at most 8 logical CPUs')
    if MPI.COMM_WORLD.size!=1:raise RuntimeError('pilot audit currently requires one MPI rank')
    for name,expected in (('SOURCE_AUDIT.json','SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED'),
                           ('OPERATOR_AUDIT.json','DISCRETE_OPERATOR_ORACLE_PASS')):
        if json.loads((ROOT/name).read_text())['status']!=expected:raise RuntimeError('prerequisite failed: '+name)
    out=ROOT/'pilots'/f'{args.cosmology}_real{args.realization:03d}_rec{args.recon_mesh}_meas{args.measurement_mesh}_ran{args.nran}'
    if args.self_contraction=='matched':
        out=out.with_name(out.name+'_selfmatched')
        matched_audit=json.loads((ROOT/'MATCHED_SELF_AUDIT.json').read_text())
        if matched_audit['status']!='MATCHED_SELF_CONTRACTION_TEST_PASS':
            raise RuntimeError('matched self-contraction point audit has not passed')
        if matched_audit['adapter_sha256']!=digest(PART/'code/post_rsd_estimator.py'):
            raise RuntimeError('matched estimator adapter changed after point audit')
    if args.zero_atom_diagnostics:
        if args.self_contraction!='matched':raise ValueError('zero-atom diagnostic requires matched self-contractions')
        if json.loads((ROOT/'ZERO_ATOM_AUDIT.json').read_text())['status']!='ZERO_ATOM_DIAGNOSTIC_TEST_PASS':
            raise RuntimeError('zero-atom point audit has not passed')
        out=out.with_name(out.name+'_zeroaudit')
    if args.grid_origin_fraction:out=out.with_name(out.name+'_originhalf')
    if args.nested_randoms:
        if args.nran<30:raise ValueError('nested test requires nran>=30')
        out=out.with_name(out.name+'_nested')
    if (out/'DONE.json').exists():
        done=json.loads((out/'DONE.json').read_text())
        if done['entry_sha256']!=digest(Path(__file__)):raise RuntimeError('completed pilot uses different source; preserve it')
        if done['measurement_sha256']!=digest(out/'MEASUREMENTS.npz'):raise RuntimeError('pilot result changed')
        print('REUSE '+str(out),flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    PILOT_STATUS_PATH=out/'STATUS.json'
    start=time.time();times={}
    status={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
            'start_UTC':datetime.now(timezone.utc).isoformat(),'stage':'input','pilot_only':True}
    def stage(name):
        status.update(stage=name,updated_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
        save_json(out/'STATUS.json',status);print(json.dumps(status),flush=True)
    stage('input')
    source=SOURCE/args.cosmology/f'real{args.realization:03d}/env_lam1_gsr2_thr15'
    if not (source/'post_sto_halo.npz').exists():
        source=PART.parent/'01_reconstruction_methods/outputs/sto_marisa_post_rsd_source'/args.cosmology/f'real{args.realization:03d}/env_lam1_gsr2_thr15'
    meta=json.loads((source/'diagnostics.json').read_text())
    expected={'env_source':'dm','weight_source':'dm','gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.,'env_mesh':128,'recon_mesh':128}
    for key,value in expected.items():
        if meta['config'][key]!=value:raise RuntimeError('STO recipe changed: '+key)
    with np.load(source/'post_sto_halo.npz',allow_pickle=False) as z:
        real_positions=np.array(z['positions_mpc_h']);weights=np.array(z['weights'])
    rsd,raw_positions,raw_meta=catalog(args.cosmology,args.realization)
    identity=validate_periodic_identity(real_positions,raw_positions)
    if weights.shape!=(len(rsd),) or not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('invalid weights')
    with np.load(ROOT/'RECON_ALPHA_COMPARISON.npz') as z:
        ak=np.array(z['alpha_k_2048']);av=np.array(z['alpha_values_2048'])
    fixed={'b_rec':2.2280546353635335,'p_rec':1.2822,'fNL_rec':100.,'smoothing_radius':10.,
           'recon_cellsize':1000./args.recon_mesh,'recon_mesh':args.recon_mesh,'algorithm_growth_rate':0.,
           'data_shift':'disp','random_shift':'disp','physical_growth_rate':.7625790037720044,
           'alpha_table_sha256':digest(ROOT/'RECON_ALPHA_COMPARISON.npz')}
    seed_nran=30 if args.nested_randoms else args.nran
    seed=shared.production_random_seed(RANDOM_NAMESPACE,args.cosmology,args.realization,'full',seed_nran)
    contract={'status':'PILOT_ONLY_NOT_LIKELIHOOD','arguments':vars(args),'fixed_reconstruction':fixed,
        'source_STO':str(source),'source_STO_sha256':digest(source/'post_sto_halo.npz'),
        'source_diagnostics_sha256':digest(source/'diagnostics.json'),'raw_fof':raw_meta,
        'source_position_identity':identity,
        'data_has_rsd':True,'redshift':.5,'LOS':'z','STO_weights_carried_from_realspace':True,
        'random_seed':seed,'entry_sha256':digest(Path(__file__)),
        'random_seed_namespace_nran':seed_nran,
        'grid_origin_translation_fraction':args.grid_origin_fraction,
        'shared_measure_source_sha256':digest(Path(shared.__file__)),'bulk_or_MCMC_released':False}
    if args.self_contraction=='matched':contract['matched_self_adapter_sha256']=digest(PART/'code/post_rsd_estimator.py')
    if (out/'CONTRACT.json').exists():
        if json.loads((out/'CONTRACT.json').read_text())!=contract:
            raise RuntimeError('incomplete pilot contract changed; inspect and preserve failed attempt before retry')
    else:save_json(out/'CONTRACT.json',contract)
    def wrap(pos,label):
        return shared.wrap_positions_to_periodic_box(pos,boxsize=1000.,dtype=np.dtype('f4'),label=label,mpicomm=MPI.COMM_WORLD)
    rsd=wrap(rsd,'STO RSD input');data=shared.make_halo_catalog(rsd,'z',MPI.COMM_WORLD,weights=weights,redshift=.5)
    import jaxpower
    actual_attrs=jaxpower.get_mesh_attrs(rsd,boxsize=np.full(3,1000.,dtype='f4'),
        boxcenter=np.full(3,500.,dtype='f4'),cellsize=fixed['recon_cellsize'],check=True,mpicomm=MPI.COMM_WORLD)
    if not np.all(np.asarray(actual_attrs.meshsize)==args.recon_mesh):raise RuntimeError('actual reconstruction mesh differs from requested')
    actual_mesh={'meshsize':np.asarray(actual_attrs.meshsize).tolist(),'cellsize':(np.asarray(actual_attrs.boxsize)/np.asarray(actual_attrs.meshsize)).tolist(),
                 'fft_backend':actual_attrs.fft_backend,'is_hermitian':bool(actual_attrs.is_hermitian)}
    stage('randoms')
    randompos,randommeta=shared.generate_rank_random_positions(local_data_count=len(rsd),boxsize=1000.,seed=seed,
                                  dtype=np.dtype('f4'),nran=args.nran,mpicomm=MPI.COMM_WORLD)
    randompos=wrap(randompos,'STO pilot randoms')
    offset=np.full(3,args.grid_origin_fraction*fixed['recon_cellsize'],dtype=float)
    data_for_recon=data;random_for_recon=randompos;roundtrip_error=np.zeros(3)
    if args.grid_origin_fraction:
        translated=wrap(rsd.astype(float)+offset,'grid-phase translated data')
        back=wrap(translated.astype(float)-offset,'grid-phase roundtrip data')
        difference=(back.astype(float)-rsd.astype(float)+500)%1000-500
        roundtrip_error=np.sqrt(np.mean(difference*difference,axis=0))
        data_for_recon=shared.make_halo_catalog(translated,'z',MPI.COMM_WORLD,weights=weights,redshift=.5)
        random_for_recon=wrap(randompos.astype(float)+offset,'grid-phase translated randoms')
    stage('reconstruction');tick=time.time()
    with shared.jaxpower_runtime_compat(MPI.COMM_WORLD):
        post,shifted,reconmeta=run_marisa_jaxrecon(data_catalog=data_for_recon,random_positions=random_for_recon,boxsize=1000.,
            growth_rate=0.,bias=fixed['b_rec'],smoothing_radius=10.,recon_cellsize=fixed['recon_cellsize'],
            recon_class='marisa-plane',data_field='disp',random_field='disp',nran=args.nran,
            random_source='pipeline_numpy',los='z',dtype=np.dtype('f4'),mpicomm=MPI.COMM_WORLD,
            fnl_loc=100.,png_p=fixed['p_rec'],png_alpha_redshift=.5,png_alpha_cosmology='quijote')
    times['reconstruction_seconds']=time.time()-tick
    sample=reconmeta['png_bias_model']['sample'];ks=np.asarray(sample['k_hmpc'])
    np.testing.assert_allclose(sample['alpha'],np.interp(np.log10(ks),np.log10(ak),av),rtol=2e-12,atol=0.)
    if args.grid_origin_fraction:
        post=np.asarray(post,dtype=float)-offset;shifted=np.asarray(shifted,dtype=float)-offset
    post=wrap(post,'STO post halos');shifted=wrap(shifted,'STO post shifted randoms')
    np.savez_compressed(out/'POST_HALOS.npz',positions=post,weights=weights)
    # 不保存数百万 random 坐标；种子、算法、输入哈希与小型探针足够定位生产。
    np.savez_compressed(out/'RANDOM_PROBES.npz',before=randompos[:1024],after=shifted[:1024])
    data_post=shared.make_halo_catalog(post,'z',MPI.COMM_WORLD,weights=weights,redshift=.5)
    shifted_cat=Catalog({'POSITION':shifted,'INDWEIGHT':np.ones(len(shifted),dtype='f4')},attrs=data_post.attrs,mpicomm=MPI.COMM_WORLD)
    input_randoms=Catalog({'POSITION':randompos,'INDWEIGHT':np.ones(len(randompos),dtype='f4')},attrs=data.attrs,mpicomm=MPI.COMM_WORLD)
    _,pedges=p_geometry();arrays={};shared.patch_jaxpower_sugiyama_s113_dtype(MPI.COMM_WORLD)
    for label,get_data in [('P_weighted_input',lambda:{'data':data,'shifted':input_randoms}),
                           ('P_post',lambda:{'data':data_post,'shifted':shifted_cat})]:
        stage(label);tick=time.time()
        spectrum=shared.measure_pk(get_data,boxsize=1000.,meshsize=args.measurement_mesh,edges=pedges,los='z',cache={},mpicomm=MPI.COMM_WORLD)
        jax.block_until_ready(spectrum);arrays.update(spectrum_arrays(spectrum,label));del spectrum
        times[label+'_seconds']=time.time()-tick;gc.collect();jax.clear_caches()
    stage('B_post');tick=time.time();ref=reference()
    def measure_B():
        return shared.measure_b000(lambda:{'data':data_post,'shifted':shifted_cat},boxsize=1000.,
            meshsize=args.measurement_mesh,edges=ref['edges'],los='z',basis='sugiyama',mask_edges_policy='default',cache={},mpicomm=MPI.COMM_WORLD)
    self_record=[]
    if args.self_contraction=='matched':
        from post_rsd_estimator import matched_b000_self_contractions
        with matched_b000_self_contractions(self_record,matched_paint=True):spectrum=measure_B()
    else:spectrum=measure_B()
    jax.block_until_ready(spectrum);arrays.update(spectrum_arrays(spectrum,'B_post'))
    if self_record:
        if len(self_record)!=1:raise ValueError('unexpected S113 record count')
        delta=self_record[0]['difference'].real/arrays['B_post_norm']
        arrays['B_post_bincenter_S113_with_TSC_shot']=arrays['B_post']+delta
        arrays['S113_discrete_minus_bincenter']=delta
        if args.zero_atom_diagnostics:
            zero=self_record[0]['zero_leg_num_shotnoise']/arrays['B_post_norm']
            arrays['zero_leg_shot_contribution']=zero
            arrays['zero_leg_pair_fraction']=self_record[0]['zero_leg_pair_fraction']
            arrays['B_post_nonzero_only_counterfactual']=arrays['B_post']+zero
    if arrays['B_post'].shape!=(120,):raise ValueError('B registry dimension changed')
    np.testing.assert_allclose(arrays['B_post_k'],ref['B000_k'],atol=5e-5,rtol=0.)
    for key,value in arrays.items():
        if not np.isfinite(value).all():raise ValueError('nonfinite pilot '+key)
    times['B_post_seconds']=time.time()-tick
    np.savez_compressed(out/'MEASUREMENTS.npz',**arrays,P_edges=pedges,B_edges=ref['edges'])
    report={'status':'PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION','contract_sha256':digest(out/'CONTRACT.json'),
        'entry_sha256':digest(Path(__file__)),'measurement_sha256':digest(out/'MEASUREMENTS.npz'),
        'post_halos_sha256':digest(out/'POST_HALOS.npz'),'reconstruction':reconmeta,'actual_reconstruction_mesh':actual_mesh,'randoms':randommeta,
        'time_seconds':times,'elapsed_seconds':time.time()-start,'host':socket.gethostname(),
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'B_fitting_not_run':True,'covariance_not_assumed':True,
        'self_contraction':args.self_contraction,
        'zero_atom_diagnostics_only':args.zero_atom_diagnostics,
        'grid_origin_translation_mpc_h':offset.tolist(),'input_coordinate_roundtrip_RMS_Mpc_h':roundtrip_error.tolist(),
        'nested_random_seed_namespace_nran':seed_nran,
        'caveats':['pilot only','no post-covariance or inference yet','full registry/shift-noise audit pending'],
        'finished_UTC':datetime.now(timezone.utc).isoformat()}
    save_json(out/'DONE.json',report);status['state']='COMPLETE';stage('complete')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if PILOT_STATUS_PATH is not None and PILOT_STATUS_PATH.exists():
            state=json.loads(PILOT_STATUS_PATH.read_text())
            if state.get('pid')==os.getpid():
                state.update(state='FAILED',error=traceback.format_exc(),updated_UTC=datetime.now(timezone.utc).isoformat())
                save_json(PILOT_STATUS_PATH,state)
        raise
