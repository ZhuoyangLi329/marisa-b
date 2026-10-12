"""模块大纲：全部44个生产 B bins 的确定性 IR 双 theory-grid 检查与逐格 checkpoint。

复用同一个 FastIRFFTProjection/source；不是重新实现投影。一个固定数值探针
不代表物理 p 校准或全 b1 参数插值认证。绝不读 pre covariance 代替 post。
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
import numpy as np
from rsd_b000_contract import PART,REFERENCE,reference,selected,CUTS,F_GROWTH,digest
from post_rsd_fixedrec import pbs_bias
from post_rsd_ir_inputs import FrozenIRSpectra,frozen_reconstruction,ROOT,ALPHA_SHA
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_grid import FastIRFFTProjection
from post_rsd_fft_projection import integrate_post_design

OUT=ROOT/'ir_full_registry_audit'


def save(path,value):
    temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def meshes_for_pair(edges,pair):
    # 与原 FFT 的 no-wrap guard 完全相同；strict >，向上取16的整数倍。
    lower=2*max(edges[int(i)+1] for i in pair)*1000/np.pi
    coarse=max(32,16*(math.floor(lower/16)+1))
    return coarse,coarse+16


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan-only',action='store_true');args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('registry audit must be hard pinned to one logical CPU')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError('numeric thread contract failed')
    for name,expected in (('SOURCE_AUDIT.json','SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED'),
                           ('IR_GRID_ACCELERATION_AUDIT.json','ACTUAL_EXACT_GRID_ACCELERATION_PASS')):
        if json.loads((ROOT/name).read_text())['status']!=expected:raise ValueError('prerequisite failed: '+name)
    spectrum=FrozenIRSpectra();recon=frozen_reconstruction(256);ref=reference();indices=selected(.20).tolist()
    if len(indices)!=44:raise ValueError('original registry changed')
    b1=2.2280546353635335;ph=1.31
    paths=[Path(__file__),REFERENCE,PART/'work/build/libmarisa_tree_rsd_fixedrec.so']
    paths += [PART/'code'/x for x in ('post_rsd_ir_grid.py','post_rsd_ir_projection.py','post_rsd_ir_harmonics.py',
                'post_rsd_displacement.py','post_rsd_ir_inputs.py','post_rsd_fft_projection.py',
                'post_rsd_fixedrec.py','rsd_b000_fft_projection.py','rsd_b000_contract.py','rsd_bonly_common.py')]
    contract={'scope':'all44 actual B bins at one diagnostic parameter point, deterministic Gaussian-long LO IR only',
              'indices':indices,'cuts':{str(c):selected(c).tolist() for c in CUTS},
              'edges':ref['edges'].tolist(),'pair_meshes':{str(i):meshes_for_pair(ref['edges'],ref['shell_pairs'][i]) for i in indices},
              'probe_b1':b1,'probe_p_halo':ph,'physical_p_calibrated':False,'f_growth':F_GROWTH,
              'recon_mesh':256,'alpha_sha256':ALPHA_SHA,'spectrum':spectrum.provenance,
              'harmonics':{'nq':1025,'n_mu':48,'n_phi':96,'lmax':12,'r_bao':110.,'qmin':2*np.pi/1000,'k_ir':.15},
              'double_grid_scaled_tolerance':1e-9,'fNL_coefficient_scaling':[1.,500.,250000.],
              'zero_leg_source':0,'denominator':'original full two-shell mode counts','closing_leg_kmax_cut':False,
              'b1_parameter_interpolation_certified':False,'post_covariance_whitening_done':False,
              'immutable_sources':{str(p):digest(p) for p in paths}}
    # JSON 回读会把 tuple 转为 list；冻结前先 canonicalize，确保可重复恢复。
    contract=json.loads(json.dumps(contract))
    OUT.mkdir(exist_ok=True);(OUT/'grids').mkdir(exist_ok=True);(OUT/'bins').mkdir(exist_ok=True)
    cp=OUT/'CONTRACT.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract:raise ValueError('registry audit contract drift; preserve prior checkpoints')
    else:save(cp,contract)
    csha=digest(cp)
    if args.plan_only:
        print(json.dumps({'state':'PLAN_VALIDATED','bins':44,'cuts':{k:len(v) for k,v in contract['cuts'].items()},
                          'meshes':sorted(set(n for pair in contract['pair_meshes'].values() for n in pair)),
                          'contract_sha256':csha}),flush=True);return
    if (OUT/'REGISTRY_AUDIT.json').exists():raise ValueError('completed audit exists; do not overwrite')
    state={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
           'start_UTC':datetime.now(timezone.utc).isoformat(),'contract_sha256':csha,'completed_bins':[]}
    start=time.time()
    def stage(name,**values):
        state.update(stage=name,updated_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,**values)
        save(OUT/'STATUS.json',state);print(json.dumps(state),flush=True)
    def guard():
        if digest(cp)!=csha:raise ValueError('contract changed during audit')
        for p,sha in contract['immutable_sources'].items():
            if digest(p)!=sha:raise ValueError('source changed during registry audit: '+p)
    try:
        stage('harmonic_tensor');guard()
        h=HarmonicDisplacement.from_long_modes(spectrum.long_modes(1025,n_mu=48,n_phi=96),r_bao=110.,lmax=12,
                  b1=b1,bphi=pbs_bias(b1,p_halo=ph).bphi,f_growth=F_GROWTH,los=(0,0,1),reconstruction=recon)
        rows=[]
        for index in indices:
            guard();levels=[]
            for mesh in contract['pair_meshes'][str(index)]:
                path=OUT/'grids'/f'bin{index:03d}_m{mesh:03d}.json'
                if path.exists():
                    result=json.loads(path.read_text())
                    if result['contract_sha256']!=csha:raise ValueError('grid checkpoint contract mismatch')
                else:
                    stage('project',bin_index=index,theory_mesh=mesh);tick=time.time()
                    fields=FastIRFFTProjection(mesh,ref['edges'],ref['shell_pairs'][index],b1,F_GROWTH,
                                p_halo=ph,reconstruction=recon,power_nw=spectrum.nowiggle,harmonics=h)
                    design=integrate_post_design(fields)
                    if design.shape!=(3,3) or not np.isfinite(design).all():raise ValueError('invalid projected basis')
                    result={'index':index,'mesh':mesh,'design':design.tolist(),'counts':fields.counts,
                            'denominator':int(fields.denominator),'cache_peak_bytes':fields.cache_peak_bytes,
                            'damping_grid_statistics':fields.damping_grid_statistics,
                            'elapsed_seconds':time.time()-tick,'contract_sha256':csha}
                    save(path,result);del fields;gc.collect()
                levels.append(result)
            coarse,fine=(np.asarray(r['design']) for r in levels)
            if levels[0]['counts']!=levels[1]['counts'] or levels[0]['denominator']!=levels[1]['denominator']:
                raise ValueError('full-shell mode counts differ between theory grids')
            scale=np.maximum(1.,np.max(np.abs(fine),axis=1,keepdims=True))
            error=float(np.max(np.abs(fine-coarse)/scale))
            powers=np.asarray(contract['fNL_coefficient_scaling'])
            weighted_scale=np.maximum(1.,np.max(np.abs(fine)*powers,axis=1,keepdims=True))
            weighted_error=float(np.max(np.abs(fine-coarse)*powers/weighted_scale))
            row={'index':index,'shell_pair':ref['shell_pairs'][index].tolist(),'meshes':[r['mesh'] for r in levels],
                 'scaled_error':error,'fNL_range_weighted_scaled_error':weighted_error,
                 'denominator':levels[-1]['denominator'],'fine_design':fine.tolist(),
                 'passed':bool(max(error,weighted_error)<contract['double_grid_scaled_tolerance'])}
            save(OUT/'bins'/f'bin{index:03d}.json',row);rows.append(row)
            state['completed_bins'].append(index);stage('bin_verified',bin_index=index)
            if not row['passed']:raise ValueError('predefined double-grid gate failed; no relaxation')
        guard()
        report={'status':'ALL44_DETERMINISTIC_IR_DOUBLE_GRID_PASS_ONE_PARAMETER_POINT',
                'created_UTC':datetime.now(timezone.utc).isoformat(),'contract_sha256':csha,'rows':rows,
                'max_scaled_error':max(r['scaled_error'] for r in rows),
                'max_fNL_range_weighted_scaled_error':max(r['fNL_range_weighted_scaled_error'] for r in rows),
                'full_registry_at_this_probe_verified':True,'b1_parameter_interpolation_verified':False,
                'physical_p_calibrated':False,'post_stochastic_validated':False,'post_covariance_whitening_done':False,
                'IR_effect_or_fNL_recovery_certified':False,'MCMC_released':False,
                'CPU_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.time()-start}
        save(OUT/'REGISTRY_AUDIT.json',report);state['state']='COMPLETE';stage('complete')
    except Exception:
        state.update(state='FAILED',error=traceback.format_exc());stage('failed');raise


if __name__=='__main__':main()
