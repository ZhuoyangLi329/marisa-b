"""模块大纲：原300目标的实际marked一到四点诊断；精确particle DFT与finite-random矩。

只读原GSR2权重/FoF，保留data-only固定shift；不生产新IC、不改变主data vector。
3个有向三角形/有限q支撑是缺项探针，不冒称全44-shell、完整shift或纯噪声校准。
输出三/四点连通响应；包含确定性高阶相关，不能全部当tree stochastic。
login26 CPU13单核，原700测量继续，formal噪声/P+B仍不放行。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
from functools import lru_cache
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from rsd_b000_contract import PART,digest,save_json
from run_rsd_bonly_workflow import catalog
from post_rsd_catalog_identity import validate_periodic_identity
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_marked_moments import factor,factorial_shift_response
from post_rsd_marked_cumulants import (MomentCumulants,ConditionalUniformRandomMoments,
    input_factorial_bispectrum,decompose_factorial_shift)

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'noise_actual_connected_probes_v1'
COSMOS=('fiducial','LC_p','LC_m')
TRIANGLES=np.array([[[3,0,1],[0,5,-1],[-3,-5,0]],
                   [[8,0,4],[-3,7,-1],[-5,-7,-3]],
                   [[10,2,1],[-2,9,-3],[-8,-11,2]]])
EXTRA=np.array([[1,-2,3],[18,-4,7],[-23,9,3]])
ALPHA=ROOT/'RECON_ALPHA_COMPARISON.npz'
ALPHA_SHA='d1a846fa8ac71fe5eee626d0020f591d0cdcfdb82a53da97250799d858fcff83'


def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())


def plans():
    if digest(ALPHA)!=ALPHA_SHA:raise ValueError('fixed algorithm alpha table changed')
    with np.load(ALPHA,allow_pickle=False) as z:ak=z['alpha_k_2048'];av=z['alpha_values_2048']
    rec=FixedReconstruction(10.,2.2280546353635335,1.2822,100.,1000./256,4,alpha_table=(ak,av))
    result=[];needed={(0,0,0)}
    def collect(*items):
        for _,_,k in items:needed.add(tuple(k))
        return 0j
    for v in TRIANGLES:
        q=np.concatenate((v,-v,EXTRA,-EXTRA));physical=2*np.pi/1000.*q
        r=rec.response(physical,np.ones(len(q)))
        factorial_shift_response(v,q,r,collect);input_factorial_bispectrum(v,collect)
        result.append(dict(triangle=v,shift_modes=q,response=r))
    return result,np.array(sorted(needed),dtype=int)


def direct_modes(positions,weights,modes):
    """精确particle DFT；正负mode共轭复用；不使用网格assignment或截断粒子。"""
    x=np.asarray(positions,dtype=float);w=np.asarray(weights,dtype=float);c=w/w.sum()
    if x.shape!=(len(w),3) or not np.isfinite(x).all() or not np.isfinite(c).all() or w.sum()<=0:
        raise ValueError('finite actual particles and normalized marks required')
    powers=np.array([c,c*c,c*c*c]);result=np.empty((3,len(modes)),dtype=complex);cache={}
    for j,k in enumerate(modes):
        if not np.any(k):result[:,j]=powers.sum(axis=1);continue
        sign=1 if k[np.flatnonzero(k)[0]]>0 else -1;key=tuple(sign*k)
        if key not in cache:
            phase=np.exp((-2j*np.pi/1000.)*(x@np.asarray(key)))
            cache[key]=powers@phase
        result[:,j]=cache[key] if sign==1 else cache[key].conj()
    return result


def summarize(fields,nrandom,modes,probe_plans):
    lookup={tuple(k):i for i,k in enumerate(modes)};result=[]
    def data(power,mode):return fields[:,power-1,lookup[tuple(mode)]]
    conditional=ConditionalUniformRandomMoments(data,nrandom)
    @lru_cache(None)
    def moment(items):
        # 平均整体平移相位只消去非闭合矩，不增加simulation数量或改变闭合相关。
        if any(sum(item[2][j] for item in items) for j in range(3)):return 0j
        return np.mean(conditional(*items))
    def expect(*items):return moment(tuple(sorted(items)))
    cumulants=MomentCumulants(expect)
    for plan in probe_plans:
        v=plan['triangle'];q=plan['shift_modes'];r=plan['response']
        parts=decompose_factorial_shift(v,q,r,cumulants)
        exact=np.array(factorial_shift_response(v,q,r,expect),dtype=complex)
        np.testing.assert_allclose(parts['full'],exact,rtol=2e-11,atol=2e-18)
        full_rows=np.array(factorial_shift_response(v,q,r,conditional),dtype=complex)
        np.testing.assert_allclose(parts['full'],full_rows.mean(axis=1),rtol=2e-11,atol=2e-18)
        def encode(a):return np.stack((np.asarray(a).real,np.asarray(a).imag),axis=-1).tolist()
        result.append(dict(triangle_integer_modes=v.tolist(),shift_support_integer_modes=q.tolist(),
            shift_response=r.tolist(),physical_units_Mpc_h6_factor=1e18,
            full_raw_self_corrected_response_components=["corrected","raw","self"],
            cumulant_parts_physical={name:encode(value*1e18) for name,value in parts.items()},
            exact_first_order_response_single_box_scatter=encode(np.std(full_rows,axis=1,ddof=1)*1e18),
            intrinsic_factorial_B_physical=encode(input_factorial_bispectrum(v,expect)*1e18),
            connected_parts_are_not_identified_pure_noise=True))
    return result


def pipeline():
    if OUT.exists():raise ValueError('preserve prior actual connected diagnostic')
    production=read(ROOT/'simultaneous_measurements_v3/CONTRACT.json')
    for path,want in production['immutable_sources'].items():
        if digest(path)!=want:raise ValueError('production frozen source drift')
    proof=read(ROOT/'MARKED_CUMULANT_UNIT_AUDIT.json')
    if proof['status']!='NINE_INDEPENDENT_MARKED_CUMULANT_AND_PARTICLE_TESTS_PASS_NOT_STO_CLOSURE':
        raise ValueError('connected/full finite-random unit tests have not actually passed')
    for path,want in proof['source_sha256'].items():
        if digest(path)!=want:raise ValueError('cumulant source drift after tests')
    admission=read(ROOT/'VALIDATION_SOURCE_AUDIT.json')
    source={(r['cosmology'],r['realization']):r for r in admission['source_rows']}
    probe_plans,modes=plans()
    paths=[Path(__file__),PART/'code/post_rsd_marked_cumulants.py',PART/'code/test_post_rsd_marked_cumulants.py',
        PART/'code/post_rsd_marked_moments.py',PART/'code/post_rsd_fixedrec.py',PART/'code/test_sto_connected_particle_fields.py',
        ROOT/'VALIDATION_SOURCE_AUDIT.json',ROOT/'MARKED_CUMULANT_UNIT_AUDIT.json',ALPHA]
    OUT.mkdir();save_json(OUT/'CONTRACT.json',dict(status='ACTUAL_MARKED_CONNECTED_PROBES_NOT_MCMC_RELEASE',
        immutable_inputs={str(p):digest(p) for p in paths},cosmologies=list(COSMOS),realizations=list(range(100)),
        exact_particle_DFT=True,finite_random_counts='30*actual_nhalo, conditionally integrated exactly',
        fixed_reconstruction=dict(b_rec=2.2280546353635335,p_rec=1.2822,fNL_rec=100.,R=10.,mesh=256,
            cic_response_power=4,alpha_sha256=ALPHA_SHA),
        q_support_is_finite_probe_not_full_shift=True,full_shell_estimator_validation=False,
        existing700_measurements_changed=False,formal_noise_audit_released=False,MCMC_started=False))
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],started_UTC=stamp(),state='RUNNING',completed=0)
    def update(**values):state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update();rows=[];records=[];nrandom=[]
    try:
        for cosmo in COSMOS:
            for real in range(100):
                update(case=f'{cosmo}_real{real:03d}',stage='admitted_source_and_exact_DFT')
                entry=source[cosmo,real];d=Path(entry['source'])
                if digest(d/'post_sto_halo.npz')!=entry['source_sha256'] or digest(d/'diagnostics.json')!=entry['diagnostics_sha256']:
                    raise ValueError('same-GSR2 source drift')
                with np.load(d/'post_sto_halo.npz',allow_pickle=False) as z:w=z['weights'].astype(float);x=z['positions_mpc_h'].copy()
                rsd,realpos,raw=catalog(cosmo,real);validate_periodic_identity(x,realpos)
                if raw['source_raw_sha256']!=entry['raw_fof']['source_raw_sha256']:raise ValueError('actual raw FoF identity drift')
                start=time.time();row=direct_modes(rsd,w,modes);zero=np.flatnonzero(np.all(modes==0,axis=1))[0]
                np.testing.assert_allclose(row[0,zero],1.,rtol=0,atol=3e-15)
                np.testing.assert_allclose([1e9*row[1,zero].real,1e18*row[2,zero].real],
                    [entry['N2_data_self'],entry['N3_data_self']],rtol=8*np.finfo(float).eps)
                rows.append(row);nrandom.append(30*len(w));records.append(dict(cosmology=cosmo,realization=real,
                    source_sha256=entry['source_sha256'],diagnostics_sha256=entry['diagnostics_sha256'],
                    raw_fof_sha256=raw['source_raw_sha256'],nhalo=len(w),exact_DFT_seconds=time.time()-start))
                update(completed=len(rows))
        matrix=OUT/'ACTUAL300_PARTICLE_FOURIER_FIELDS.npz'
        np.savez_compressed(matrix,fields=np.array(rows),integer_modes=modes,nrandom=nrandom,
            target_keys_json=np.array([json.dumps([r['cosmology'],r['realization']]) for r in records]))
        update(state='DECOMPOSING_FULL_MARKED_CUMULANTS',stage='actual100_per_truth')
        analysis=[]
        for i,cosmo in enumerate(COSMOS):
            analysis.append(dict(cosmology=cosmo,n_catalogs=100,
                probes=summarize(np.array(rows[i*100:(i+1)*100]),np.array(nrandom[i*100:(i+1)*100]),modes,probe_plans)))
        report=dict(status='ACTUAL300_FINITE_RANDOM_MARKED_CONNECTED_PROBES_COMPLETE_NOT_STO_CLOSURE',
            created_UTC=stamp(),n_catalogs=300,n_per_truth=100,records=records,analysis=analysis,
            field_matrix_sha256=digest(matrix),full_particles_used=True,exact_particle_DFT=True,
            finite_random_conditionally_integrated=True,whole_box_translation_phase_average=True,
            covariance_is_of_mean=False,likelihood_covariance_created=False,volume_Gpc_h3=1.,
            noise_pure_vs_nonlinear_signal_separated=False,all44_B_windows_validated=False,
            full_production_shift_and_IR_noise_validated=False,formal_noise_audit_released=False,MCMC_started=False,
            limitations=['three oriented triangles and finite q support only, not full-shell closure',
                'connected3/4 include deterministic nonlinear/PNG correlations; not pure stochastic templates',
                'same targets used as diagnostic, not an independent mock recovery',
                'fixed response main-image CIC4 does not certify production alias/finite displacement'],
            CPU_affinity=[13],peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save_json(OUT/'REPORT.json',report);update(state='ACTUAL300_CONNECTED_PROBES_COMPLETE_NOT_STO_CLOSURE',finished_UTC=stamp())
    except Exception:update(state='FAILED_PRESERVED_REQUIRES_INSPECTION',error=traceback.format_exc());raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.detach:
        log=ROOT/'noise_actual_connected_probes_v1.log';launch=ROOT/'noise_actual_connected_probes_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect original process before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
