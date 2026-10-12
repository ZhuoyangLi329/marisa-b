"""模块大纲：同盒DM real/RSD与STO输入权重幂场实测，分片读入与双网格诊断。

第一层真实样本噪声检查：0/+100/-100的real000，mesh128/256，TSC/interlacing3。
输出raw mixed谱及mode，保留halo-matter投影的操作性定义；不发布noise PASS，
不将残差直接当纯epsilon、不减假定Poisson matter shot、不启动B/P+B。
生产重构/测量、正式C500、校准和既有结果均不修改。login26 CPU14单核。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import h5py
try:
    import hdf5plugin  # 仅加载现有filter；不安装缺失依赖。
except ImportError:
    pass
from mpi4py import MPI
from rsd_b000_contract import PART,digest,save_json
from run_rsd_bonly_workflow import catalog
from post_rsd_catalog_identity import validate_periodic_identity
from sto_rsd_noise_fields import (gadget_rsd_positions,marked_moments,mode_geometry,
    projected_spectra,projection_residual,StreamingMarkedFourier)

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'noise_rsd_input_spectra_v1'
CHUNK=1048576
FIELD_NAMES=['matter_real','matter_RSD','halo_w_real','halo_w_RSD','halo_w2_RSD','halo_w3_RSD']
COSMOS=('fiducial','LC_p','LC_m')


def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())


def headers(cosmo,real):
    folder=Path('/pscratch/sd/l/lzy/quijote-png/dm')/cosmo/str(real)/'snapdir_003'
    first=folder/'snap_003.0.hdf5'
    with h5py.File(first,'r') as f:
        h=f['Header'].attrs
        result=dict(a=float(h['Time']),redshift=float(h['Redshift']),boxsize=float(h['BoxSize'])/1000,
            omega_m=float(h['Omega0']),omega_lambda=float(h['OmegaLambda']),h=float(h['HubbleParam']),
            nfiles=int(h['NumFilesPerSnapshot']),nparticles=int(h['NumPart_Total'][1]))
        if int(h['NumPart_Total_HighWord'][1])!=0:raise ValueError('unexpected high-word particle count')
    paths=[folder/f'snap_003.{i}.hdf5' for i in range(result['nfiles'])]
    counts=[]
    for path in paths:
        with h5py.File(path,'r') as f:
            h=f['Header'].attrs;p=f['PartType1/Coordinates'];v=f['PartType1/Velocities']
            if (p.ndim!=2 or p.shape[1]!=3 or v.shape!=p.shape or p.dtype!=np.dtype('float32')
                or v.dtype!=np.dtype('float32') or int(h['NumPart_ThisFile'][1])!=len(p)
                or float(h['Time'])!=result['a'] or float(h['BoxSize'])/1000!=result['boxsize']
                or int(h['NumFilesPerSnapshot'])!=len(paths)
                or int(h['NumPart_Total'][1])!=result['nparticles']):
                raise ValueError('inconsistent DM position/velocity/header shard: '+str(path))
            counts.append(len(p))
    if (sum(counts)!=result['nparticles'] or result['boxsize']!=1000
        or abs(result['redshift']-.5)>1e-6 or abs(result['a']*(1+result['redshift'])-1)>1e-10
        or result['omega_m']!=.3175):raise ValueError('not the actual matching z.5/box1000 DM snapshot')
    return result,paths,counts


def measure(cosmo,meshsize,source,update):
    real=0;start=time.time();name=f'{cosmo}_real000_m{meshsize}'
    destination=OUT/'rows'/name
    if destination.exists():raise ValueError('preserve existing diagnostic row')
    destination.mkdir(parents=True)
    directory=Path(source['source']);sto=directory/'post_sto_halo.npz';diag=directory/'diagnostics.json'
    if digest(sto)!=source['source_sha256'] or digest(diag)!=source['diagnostics_sha256']:
        raise ValueError('same-GSR2 source drift')
    with np.load(sto,allow_pickle=False) as z:weights=z['weights'].astype(float);real_pos=z['positions_mpc_h'].copy()
    rsd,raw_real,raw=catalog(cosmo,real);validate_periodic_identity(real_pos,raw_real)
    if raw['source_raw_sha256']!=source['raw_fof']['source_raw_sha256']:raise ValueError('raw FoF identity drift')
    n2,n3=marked_moments(weights)
    np.testing.assert_allclose([n2,n3],[source['N2_data_self'],source['N3_data_self']],rtol=8*np.finfo(float).eps)
    with np.load(directory/'spectra.npz',allow_pickle=False) as z:kedges=z['k_edges'].copy()
    muedges=np.array([0.,.5,1.]);geometry=mode_geometry(meshsize,1000.,kedges,muedges)
    header,paths,counts=headers(cosmo,real)
    dreal=StreamingMarkedFourier(meshsize,comm=MPI.COMM_WORLD)
    drsd=StreamingMarkedFourier(meshsize,comm=MPI.COMM_WORLD)
    loaded=0;io=[]
    for file_index,(path,count) in enumerate(zip(paths,counts)):
        hp=hashlib.sha256();hv=hashlib.sha256()
        with h5py.File(path,'r') as f:
            for j in range(0,count,CHUNK):
                pos=f['PartType1/Coordinates'][j:j+CHUNK];vel=f['PartType1/Velocities'][j:j+CHUNK]
                hp.update(pos.tobytes());hv.update(vel.tobytes())
                converted=gadget_rsd_positions(pos,vel,a=header['a'],omega_m=header['omega_m'],
                    omega_lambda=header['omega_lambda'])
                dreal.add(np.asarray(pos,dtype=float)/1000.);drsd.add(converted)
                loaded+=len(pos)
        io.append(dict(path=str(path),nparticles=count,coordinates_dataset_sha256=hp.hexdigest(),
            velocity_dataset_sha256=hv.hexdigest()))
        update(state='MEASURING_ACTUAL_RSD_INPUT_NOISE',case=name,stage='DM_streaming_paint',
               DM_file_completed=file_index+1,DM_files_total=len(paths),DM_particles_loaded=loaded)
    if loaded!=header['nparticles'] or dreal.count!=loaded or drsd.count!=loaded:
        raise ValueError('no truncated DM sample permitted')
    fields=[dreal.fourier().ravel()[geometry['indices']],drsd.fourier().ravel()[geometry['indices']]]
    del dreal,drsd;gc.collect()
    for x,w in [(raw_real,weights),(rsd,weights),(rsd,weights**2),(rsd,weights**3)]:
        paint=StreamingMarkedFourier(meshsize,comm=MPI.COMM_WORLD);paint.add(x,w)
        fields.append(paint.fourier().ravel()[geometry['indices']]);del paint;gc.collect()
    fields=np.array(fields);power=projected_spectra(fields,geometry)
    # 实际权重幂场各自按sum(w^r)归一化；mixed self不等于统一的N2。
    moment_self=np.array([[1e9*np.sum(weights**(i+j))/(np.sum(weights**i)*np.sum(weights**j))
                         for j in (1,2,3)] for i in (1,2,3)])
    np.testing.assert_allclose(moment_self[0,0],n2,rtol=3e-15)
    np.testing.assert_allclose(n2*moment_self[1,0],n3,rtol=3e-15)
    nr=projection_residual(power,2,0);ns=projection_residual(power,3,1)
    np.savez_compressed(destination/'RAW_FIELDS_AND_SPECTRA.npz',field_names=np.array(FIELD_NAMES),
        normalized_fourier_modes=fields,integer_modes=geometry['integer_modes'],k=geometry['k'],mu=geometry['mu'],
        k_edges=kedges,absolute_mu_edges=muedges,labels=geometry['labels'],counts=geometry['counts'],
        raw_mixed_spectra=power,halo_mark_self_matrix=moment_self,N2_data_self=n2,N3_data_self=n3,
        operational_real_halo_matter_residual=nr,operational_RSD_halo_matter_residual=ns)
    report=dict(status='ACTUAL_INPUT_RSD_MIXED_SPECTRA_MEASURED_NOT_STO_CLOSURE',cosmology=cosmo,
        realization=real,meshsize=meshsize,assignment='TSC/interlacing3/compensated float64 pmesh, diagnostic only',
        source_sha256=digest(sto),diagnostics_sha256=digest(diag),raw_fof=raw,
        DM_header=header,DM_input_dataset_hashes=io,full_DM_count=loaded,field_names=FIELD_NAMES,
        Gadget_velocity_rule='vphys=sqrt(a)*stored; ds=vphys/(a*H_h); no extra h',
        velocity_source='https://github.com/franciscovillaescusa/Pylians3/blob/master/library/readgadget.py',
        halo_velocity_rule_is_distinct='vphys=GroupVel*(1+z)',
        data_only_shift_noise_source=True,random_noise_added=False,
        N2_data_self=float(n2),N3_data_self=float(n3),
        raw_RSD_residual_range_Mpc_h3=[float(ns.min()),float(ns.max())],
        halo_mark_self_matrix=moment_self.tolist(),
        pure_stochastic_N_k_mu_calibration_complete=False,post_reconstruction_field_measured_here=False,
        connected_stochastic_response_validated=False,PNG_noise_response_validated=False,
        finite_shift_and_IR_noise_validated=False,formal_noise_audit_released=False,MCMC_started=False,
        caveats=['single realization per sign, not an ensemble validation',
            'halo-matter regression residual includes unmodeled deterministic nonlinear/velocity effects',
            'finite-mode regression bias and matter discreteness not removed',
            'nonlinear RSD matter is a diagnostic reference, not automatically the latent linear signal',
            '128/256 mesh comparison still required after all six rows'],
        arrays_sha256=digest(destination/'RAW_FIELDS_AND_SPECTRA.npz'),created_UTC=stamp(),
        elapsed_seconds=time.time()-start,CPU_affinity=sorted(os.sched_getaffinity(0)),
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    save_json(destination/'REPORT.json',report);return report


def pipeline():
    if OUT.exists():raise ValueError('preserve existing diagnostic, no automatic restart')
    proof=read(ROOT/'simultaneous_measurements_v3/CONTRACT.json')
    for path,want in proof['immutable_sources'].items():
        if digest(path)!=want:raise ValueError('production frozen source drift')
    tests=read(ROOT/'NOISE_RSD_FIELDS_TEST_AUDIT.json')
    if tests['status']!='FOUR_INDEPENDENT_INPUT_FIELD_TESTS_PASS_NOT_SCIENTIFIC_CLOSURE':
        raise ValueError('native field tests must actually pass first')
    for path,want in tests['source_sha256'].items():
        if digest(path)!=want:raise ValueError('field code drift after tests')
    audit=read(ROOT/'VALIDATION_SOURCE_AUDIT.json')
    sources={r['cosmology']:r for r in audit['source_rows'] if r['realization']==0}
    if set(sources)!=set(COSMOS):raise ValueError('all three admitted target sources required')
    # 三真值/双网格只检查第一层，不能伪造formal NOISE PASS。
    inputs=[Path(__file__),PART/'code/sto_rsd_noise_fields.py',PART/'code/test_sto_rsd_noise_fields.py',
        ROOT/'NOISE_RSD_FIELDS_TEST_AUDIT.json',ROOT/'VALIDATION_SOURCE_AUDIT.json',
        ROOT/'simultaneous_measurements_v3/CONTRACT.json']
    OUT.mkdir();save_json(OUT/'CONTRACT.json',dict(status='ACTUAL_INPUT_NOISE_MEASUREMENT_NOT_MCMC_RELEASE',
        immutable_inputs={str(p):digest(p) for p in inputs},cosmologies=list(COSMOS),realizations=[0],
        meshes=[128,256],CPU_affinity=[14],chunk_particles=CHUNK,formal_noise_audit_released=False,
        production_sources_untouched=True,target_overlap_explicit=True,pure_noise_definition_not_chosen=True))
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[14],started_UTC=stamp(),
        state='RUNNING',completed=[])
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update()
    try:
        for cosmo in COSMOS:
            for mesh in (128,256):
                update(case=f'{cosmo}_real000_m{mesh}',stage='source_and_header_checks')
                report=measure(cosmo,mesh,sources[cosmo],update)
                state['completed'].append(dict(cosmology=cosmo,meshsize=mesh,arrays_sha256=report['arrays_sha256']))
                update()
        update(state='SIX_ACTUAL_INPUT_SPECTRA_ROWS_COMPLETE_NOT_STO_CLOSURE',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14} or MPI.COMM_WORLD.size!=1:
        raise ValueError('login26 CPU14, one MPI rank required')
    if args.detach:
        log=ROOT/'noise_rsd_input_spectra_v1.log';launch=ROOT/'noise_rsd_input_spectra_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect old process and outputs before any restart')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
