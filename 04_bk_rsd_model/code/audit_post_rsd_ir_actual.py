"""模块大纲：真实冻结谱的 harmonic/径向收敛与实际低高 B bins 的双 theory-grid 验算。

p_halo=1.31 和 b1=2.228 只是数值探针，不是新的 p 校准或正式 MCMC 参数选择。
仅写第四部分独立审计，不加载数据 likelihood 或使用 pre covariance 冒充 post。
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import gc
import json
import os
import socket
import time
import traceback
import numpy as np
from rsd_b000_contract import PART,reference,digest,F_GROWTH
from post_rsd_fixedrec import pbs_bias
from tree_png_rsd_native import RsdConfig
from post_rsd_ir_inputs import FrozenIRSpectra,frozen_reconstruction,ROOT,ALPHA_SHA
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_oracle import source_leg_wiggles,dress_finite_png
from post_rsd_ir_projection import IRFFTProjection,integrate_post_design


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--indices',type=int,nargs='+',default=[1,99])
    ap.add_argument('--meshes',type=int,nargs='+',default=[144,160]);args=ap.parse_args()
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('task CPU limit exceeded')
    target=ROOT/'IR_ACTUAL_SPECTRUM_AUDIT.json'
    if target.exists():raise FileExistsError('preserve previous actual-spectrum audit before a new revision')
    start=time.time();status_path=ROOT/'IR_ACTUAL_STATUS.json'
    state={'state':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0))}
    def stage(name):
        state.update(stage=name,updated_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
        status_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    stage('frozen_spectra')
    spectrum=FrozenIRSpectra();rec=frozen_reconstruction(256);b1=2.2280546353635335;ph=1.31
    bias=pbs_bias(b1,.43,-.17,p_halo=ph);kw=dict(b1=b1,bphi=bias.bphi,f_growth=F_GROWTH,los=(0,0,1),reconstruction=rec)
    rng=np.random.default_rng(2026100905);v=rng.normal(size=(16,3,3))*.068;v[:,2]=-v[:,:2].sum(axis=1)
    k=np.linalg.norm(v,axis=-1);p,m=spectrum.physical(k);pn=spectrum.nowiggle(k)
    source=source_leg_wiggles(v,p,pn,m,bias,RsdConfig(F_GROWTH),rec)
    hashes={x:digest(PART/'code'/x) for x in ('post_rsd_ir_actual.py',) if (PART/'code'/x).exists()}
    hashes.update({x:digest(PART/'code'/x) for x in ('audit_post_rsd_ir_actual.py','post_rsd_ir_inputs.py','post_rsd_ir_projection.py',
                     'post_rsd_ir_harmonics.py','post_rsd_displacement.py')})
    harmonics={};models={}
    for nq,mu,ell,label in ((257,32,10,'q257_l10'),(513,32,10,'q513_l10'),(1025,32,10,'q1025_l10'),
                             (1025,32,8,'q1025_l8'),(1025,48,12,'q1025_l12_mu48')):
        stage(label);h=HarmonicDisplacement.from_long_modes(spectrum.long_modes(nq,n_mu=mu,n_phi=2*mu),lmax=ell,**kw)
        harmonics[label]=h;models[label]=dress_finite_png(source.base.total,source.wiggles['total'],h.damping(v))
    baseline=models['q1025_l12_mu48'];scale=np.maximum(1.,np.max(np.abs(baseline),axis=0))
    errors={name:float(np.max(np.abs(value-baseline)/scale)) for name,value in models.items()}
    stage('direct_angular_comparison');long=spectrum.long_modes(257,n_mu=32,n_phi=64)
    flat=v.reshape(-1,3);u=flat/np.linalg.norm(flat,axis=1)[:,None]
    direct=long.relative_covariance(110*u,**kw);compressed=harmonics['q257_l10'].covariance(u)
    tensor_error=float(np.max(np.abs(compressed-direct)/np.maximum(1.,np.max(np.abs(direct),axis=0))))
    fine=harmonics['q1025_l12_mu48'];ref=reference();bins=[]
    for index in args.indices:
        if index not in ref['certified_indices']:raise ValueError('not a certified original bin')
        rows=[]
        for mesh in args.meshes:
            stage(f'bin{index:03d}_mesh{mesh}')
            tick=time.time();fields=IRFFTProjection(mesh,ref['edges'],ref['shell_pairs'][index],b1,F_GROWTH,
                p_halo=ph,reconstruction=rec,power_nw=spectrum.nowiggle,harmonics=fine)
            design=integrate_post_design(fields)
            rows.append({'mesh':mesh,'design':design.tolist(),'counts':fields.counts,'denominator':int(fields.denominator),
                         'cache_peak_bytes':fields.cache_peak_bytes,'elapsed_seconds':time.time()-tick})
            del fields;gc.collect()
        first=np.asarray(rows[0]['design']);last=np.asarray(rows[-1]['design'])
        griderror=float(np.max(np.abs(last-first)/np.maximum(1.,np.max(np.abs(last),axis=1,keepdims=True))))
        bins.append({'index':index,'shell_pair':ref['shell_pairs'][index].tolist(),'rows':rows,'double_grid_scaled_error':griderror})
    maximum_grid=max(row['double_grid_scaled_error'] for row in bins)
    tolerances={'tensor':1e-8,'q513_source':2e-6,'double_grid':1e-9}
    passed=tensor_error<tolerances['tensor'] and errors['q513_l10']<tolerances['q513_source'] and maximum_grid<tolerances['double_grid']
    out={'status':'ACTUAL_SPECTRUM_IR_NUMERIC_PROBES_PASS' if passed else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'spectrum':spectrum.provenance,
         'physical_snapshot_relative_errors':spectrum.snapshot_relative_errors,'reconstruction_alpha_sha256':ALPHA_SHA,
         'source_scaled_errors':errors,'harmonic_tensor_vs_direct_scaled_error':tensor_error,'bins':bins,
         'predefined_tolerances':tolerances,'numerical_probe_only':{'b1':b1,'p_halo':ph,'b2':.43,'bK2':-.17},
         'physical_p_calibrated_here':False,'full_production_registry_verified':False,'post_covariance_whitening_done':False,
         'parameter_interpolation_verified':False,'post_stochastic_validated':False,'MCMC_released':False,
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.time()-start,'hashes':hashes}
    for name,sha in hashes.items():
        if digest(PART/'code'/name)!=sha:raise RuntimeError('source changed during actual IR audit: '+name)
    target.write_text(json.dumps(out,indent=2)+'\n');state['state']='COMPLETE' if passed else 'FAILED';stage('complete')
    print(json.dumps(out,indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception:
        path=ROOT/'IR_ACTUAL_STATUS.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('pid')==os.getpid():
                state.update(state='FAILED',error=traceback.format_exc());path.write_text(json.dumps(state,indent=2)+'\n')
        raise
