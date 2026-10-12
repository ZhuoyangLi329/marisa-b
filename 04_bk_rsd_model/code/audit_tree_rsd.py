"""模块大纲：独立参考对照；旧 real-space 回归；方向对称性；投影收敛。"""
from pathlib import Path
from dataclasses import replace
import ctypes
import hashlib
import json
import numpy as np
from tree_png_rsd_native import NativeTreeRSD, HaloBiasV1, RsdConfig, ReconstructionConfig, reference as ref

PART=Path(__file__).resolve().parents[1]

def raw_reference(v,p,m,b,r,rec):
    x=ref.finite_png_halo_bispectrum_rsd(v,p,m,b,r,reconstruction=rec)
    result=[]
    for q in (x.total,x.gravitational_and_bias,x.primordial,*x.pair_terms):
        result.extend((q.G,q.L,q.Q))
    for q in (*x.z1,*x.z2):result.extend((q.c0,q.c1,q.c2))
    s=ref.desi_gaussian_tree_stochastic_bases(v,p,b.b1,r)
    return np.array([*result,s.Bshot_basis,s.Pshot_basis,s.Ashot_basis])

def error(a,b):
    scale=np.maximum(1.,np.max(np.abs(b),axis=0))
    e=float(np.max(np.abs(a-b)/scale))
    assert e<2e-10,e
    return e

def main():
    native=NativeTreeRSD();rng=np.random.default_rng(20261001)
    v=rng.normal(size=(1024,3,3))
    v[:,:2]*=rng.uniform(.015,.14,size=(1024,2,1));v[:,2]=-v[:,0]-v[:,1]
    p=rng.uniform(500.,12000.,size=(len(v),3));m=rng.uniform(500.,20000.,size=(len(v),3))
    b=HaloBiasV1(2.08,.41,-.27,3.25,1.36,2.42);r=RsdConfig(.7625790037720044,(.2,-.3,.9))
    cases={'pre':None,'post_fixed':ReconstructionConfig(10.,2.2280546354),
        'post_adaptive_CIC':ReconstructionConfig(10.,2.2280546354,8.,4,f_nl_recon_ratio=1.,bphi_recon=3.189062231)}
    report={'random_closed_triangles':len(v),'reference_max_scaled_error':{}}
    for name,rec in cases.items():
        expected=np.array([raw_reference(vi,pi,mi,b,r,rec) for vi,pi,mi in zip(v,p,m)])
        actual=native.evaluate(v,p,m,b,r,rec).raw
        report['reference_max_scaled_error'][name]=error(actual,expected)
        report.setdefault('G_L_Q_max_absolute_error',{})[name]=np.max(np.abs(actual[:,:3]-expected[:,:3]),axis=0).tolist()
        error(native.evaluate(-v,p,m,b,r,rec).raw,actual)
        error(native.evaluate(v[:,[1,2,0]],p[:,[1,2,0]],m[:,[1,2,0]],b,r,rec).total,actual[:,:3])
        if rec is None or rec.cell_size==0:
            rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
            rr=replace(r,los=tuple(np.array(r.los)@rotation))
            error(native.evaluate(v@rotation,p,m,b,rr,rec).raw,actual)
    # 专门测试 squeezed cancellation，不让随机样本替代 squeezed 极限。
    squeezed=[]
    for eps in np.geomspace(1e-5,.02,100):
        a=np.array([eps,0.,0.]);bb=np.array([.08,.03,.02]);squeezed.append([a,bb,-a-bb])
    sv=np.array(squeezed)
    report['squeezed_count']=len(sv)
    report['squeezed_error']=error(native.evaluate(sv,p[:100],m[:100],b,r).raw,
        np.array([raw_reference(vi,pi,mi,b,r,None) for vi,pi,mi in zip(sv,p,m)]))
    # 与第三部分既有 C++ source 比较 f_growth=0 的 G/L/Q，而非仅与新 oracle 比较。
    old=PART.parent/'03_bk_model/envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2/build/libpaware_ir.so'
    lib=ctypes.CDLL(str(old));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.paware_source_batch.argtypes=[ctypes.c_size_t]+[ptr]*7
    out=np.empty((len(v),5,27));ba=np.array([b.b1,b.b2,b.bK2,b.bphi,b.bphidelta,b.bphi2])
    cfg=np.array([10.,2.2280546354,8.,4.,3.189062231,1.])
    assert lib.paware_source_batch(len(v),v,p,p,m,ba,cfg,out)==0
    columns=([1,21,22,23],[2,3,4,5,6,7,8,24],[9,10,11,12,13,14,15,16,25,26])
    for pre in (False,True):
        terms=out[:,0].copy()
        if pre:terms[:,[1,8,14,24,25,26]]=0.
        expected=np.stack([terms[:,c].sum(axis=1) for c in columns],axis=1)
        actual=native.evaluate(v,p,m,b,replace(r,f_growth=0.),None if pre else cases['post_adaptive_CIC']).total
        report.setdefault('old_native_real_space_error',{})['pre' if pre else 'post']=error(actual,expected)
    lengths=(.04,.055,.07);p3=p[0];m3=m[0]
    report['multipoles']={}
    monopole_scale=np.maximum(1.,np.abs(native.project_multipole(lengths,p3,m3,b,r,0,26,28)))
    for ell in (0,1,2,4):
        coarse=native.project_multipole(lengths,p3,m3,b,r,ell,18,20)
        fine=native.project_multipole(lengths,p3,m3,b,r,ell,26,28)
        quadrature_error=float(np.max(np.abs(coarse-fine)/monopole_scale))
        assert quadrature_error<2e-10,quadrature_error
        report['multipoles'][ell]={'quadrature_error':quadrature_error,'G_L_Q':fine[:3].tolist()}
        expected=ref.project_scoccimarro_multipole(*lengths,p3,m3,b,r,ell=ell,n_mu=26,n_phi=28)
        assert np.max(np.abs(fine[:3]-np.array([expected.G,expected.L,expected.Q]))/monopole_scale[:3])<2e-10
    rejected=0
    for bad in (v[0]*np.array([1,1,2])[:,None],np.zeros((3,3))):
        try:native.evaluate(bad,p3,m3,b,r)
        except ValueError:rejected+=1
    assert rejected==2
    try:native.project_multipole(lengths,p3,m3,b,r,reconstruction=cases['post_adaptive_CIC'])
    except ValueError:rejected+=1
    assert rejected==3
    # 保留参数真值，验证有限 PNG 的正负响应及代数恢复，不冒充 mock MCMC。
    c=native.evaluate(v,p,m,b,r).total
    injected=100.;y=c@np.array([1.,injected,injected**2])
    grid=np.linspace(-150.,150.,601)
    score=np.array([np.mean((c@np.array([1.,f,f*f])-y)**2) for f in grid])
    assert grid[np.argmin(score)]==injected
    report.update(status='PASS',rejected_invalid_cases=rejected,synthetic_fixed_parameter_fNL_recovered=100.,
        library_sha256=hashlib.sha256(native.path.read_bytes()).hexdigest())
    dest=PART/'outputs/TREE_RSD_NATIVE_AUDIT.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
