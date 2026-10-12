"""模块大纲：实际谱上 finite-PNG IR 截断的高阶余项诊断，不更换模型。

比较 strict fNL² dressing 和同一 Gaussian tensor 的未展开指数（后者只为
高阶敏感性参照，非另一套完整理论）。不以 unprojected 误差认证 fNL 偏差。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest,F_GROWTH
from post_rsd_fixedrec import pbs_bias
from post_rsd_ir_inputs import FrozenIRSpectra,frozen_reconstruction,ROOT
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_oracle import source_leg_wiggles,dress_finite_png
from tree_png_rsd_native import RsdConfig


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single CPU diagnostic only')
    target=ROOT/'IR_PNG_TRUNCATION_DIAGNOSTIC.json'
    if target.exists():raise ValueError('preserve prior truncation diagnostic')
    rng=np.random.default_rng(2026100906);kf=2*np.pi/1000
    pairs=[[[1,0,0],[30,0,3]],[[0,0,1],[0,30,3]],[[1,0,0],[0,0,30]],[[0,0,1],[0,0,30]]]
    while len(pairs)<32:
        pair=rng.integers(-25,26,size=(2,3));norm=np.linalg.norm(pair,axis=1)
        if np.all((norm>0)&(norm<32)) and np.any(pair.sum(axis=0)):
            pairs.append(pair.tolist())
    pairs=np.asarray(pairs);v=np.concatenate((pairs,-pairs.sum(axis=1)[:,None,:]),axis=1)*kf
    spectrum=FrozenIRSpectra();recon=frozen_reconstruction(256);k=np.linalg.norm(v,axis=-1)
    p,m=spectrum.physical(k);pn=spectrum.nowiggle(k);long=spectrum.long_modes(513)
    rows=[]
    for b1 in (.5,2.2280546353635335,4.):
        bias=pbs_bias(b1,.43,-.17,p_halo=1.31)
        source=source_leg_wiggles(v,p,pn,m,bias,RsdConfig(F_GROWTH),recon)
        h=HarmonicDisplacement.from_long_modes(long,lmax=10,b1=b1,bphi=bias.bphi,
                        f_growth=F_GROWTH,los=(0,0,1),reconstruction=recon)
        flat=v.reshape(-1,3);tensor=h.covariance(flat)
        e=(.5*np.einsum('ni,ncij,nj->nc',flat,tensor,flat)).reshape(-1,3,3)
        d=h.damping(v);strict=dress_finite_png(source.base.total,source.wiggles['total'],d)
        for fnl in (-500.,-100.,0.,10.,20.,100.,500.):
            powers=np.array([1.,fnl,fnl*fnl]);exponent=e@powers
            if exponent.min() < -1e-10:raise ValueError('negative full physical Gaussian exponent; inspect harmonic truncation')
            exact_d=np.exp(-exponent)
            quadratic_d=d[...,0]*(1+fnl*d[...,1]+fnl*fnl*d[...,2])
            b_strict=strict@powers
            b_exponential=source.base.total@powers+np.sum(exact_d*(source.wiggles['total']@powers),axis=1)
            source_scale=np.maximum(1.,np.sum(np.abs(source.base.total*powers),axis=1)
                                       +np.sum(np.abs(source.wiggles['total']*powers),axis=(1,2)))
            scaled=np.abs(b_exponential-b_strict)/source_scale
            if fnl==0.:np.testing.assert_allclose(scaled,0.,atol=1e-13)
            index=int(np.argmax(scaled))
            rows.append({'b1':b1,'p_halo_probe':1.31,'fNL':fnl,
                         'max_absolute_damping_factor_difference':float(np.max(np.abs(exact_d-quadratic_d))),
                         'minimum_quadratic_damping_factor':float(quadratic_d.min()),
                         'negative_quadratic_damping_leg_count':int(np.count_nonzero(quadratic_d<0)),
                         'max_total_B_difference_over_fixed_source_scale':float(scaled.max()),
                         'total_B_difference_L2_over_source_scale_L2':float(np.linalg.norm(b_exponential-b_strict)/np.linalg.norm(source_scale)),
                         'largest_difference_triangle_index':index,
                         'largest_difference_B_strict':float(b_strict[index]),
                         'largest_difference_B_unexpanded_exponent':float(b_exponential[index])})
    report={'status':'HIGHER_ORDER_IR_PNG_SENSITIVITY_RECORDED_NOT_MODEL_ACCEPTANCE',
            'created_UTC':datetime.now(timezone.utc).isoformat(),'triangles_integer':(v/kf).round().astype(int).tolist(),
            'rows':rows,'fixed_reconstruction_changed':False,'physical_p_calibrated':False,
            'production_model_changed':False,'full_exponential_is_partial_higher_order_reference_only':True,
            'whitened_error':None,'fNL_bias':None,'actual_estimator_projection_test':False,
            'scope':'one no-wiggle split and fixed reconstruction; three b1 probes, 32 closed oriented triangles; Gaussian long modes',
            'spectrum':spectrum.provenance,'CPU_affinity':sorted(os.sched_getaffinity(0)),
            'entry_sha256':digest(Path(__file__))}
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','triangles_integer','spectrum')},indent=2),flush=True)
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
