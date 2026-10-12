"""模块大纲：RSD P0+B000 独立合同；实际 P 分箱；同一 tree/PNG 系数表示。

不改 B-only 产物；P0 使用完整离散 shell 的 Kaiser + finite-PNG tree。
额外的 Pshot 是独立残余常数，不与未闭合的 B stochastic 强行绑定。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from rsd_b000_contract import PART,LEGACY_OUT,CUTS,BNODES,F_GROWTH,P_CAL,load_spectrum,digest,save_json

OUT=PART/'outputs/pre_rsd_pb_mean100_v1'
B_OUT=PART/'outputs/pre_rsd_b000_bonly_mean100_v1'
P_MATRIX=PART.parent/'03_bk_model/envclass_b/vendor/data/PRE_P0_FNL250_MATRIX.npz'
REAL_P_SAMPLE=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_baseline_b000_pk/full/fiducial/real000/pre_pk.npz')
SHARED=PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py'

def p_geometry():
    with np.load(REAL_P_SAMPLE) as z:
        k=np.array(z['k'][:12]);edges=np.array(z['edges'][:13])
    with np.load(P_MATRIX) as z:
        np.testing.assert_allclose(z['pk_k'],k,rtol=0,atol=1e-12)
    if len(k)!=12 or np.max(k)>.08+1e-12:raise ValueError('not the realspace P0<=.08 geometry')
    return k,edges

def p_lattice(edges,meshsize=64):
    """精确离散六个谱模板；window 只来自原生产 binner，禁止另造 shell mask。"""
    with np.load(OUT/'P0_WINDOW.npz') as z:
        np.testing.assert_array_equal(edges,z['edges']);v=np.array(z['vectors_integer'])*(2*np.pi/1000.)
        labels=np.array(z['labels']);w=np.array(z['weights']);n=np.array(z['nmodes']);kavg=np.array(z['k'])
    k2=np.sum(v*v,axis=1);k=np.sqrt(k2);u=v[:,2]**2/k2;p,m=load_spectrum()(k)
    values=(p,p*u,p*u*u,p/m,p*u/m,p/m**2)
    t=np.array([np.bincount(labels,weights=w*value,minlength=12)/n for value in values])
    return t,n,kavg

def p_prediction(fnl,b1,templates):
    bp=2*1.686*(b1-P_CAL);f=F_GROWTH;t=templates
    return (b1*b1*t[0]+2*b1*f*t[1]+f*f*t[2]
            +fnl*2*bp*(b1*t[3]+f*t[4])+fnl*fnl*bp*bp*t[5])

def p_coefficients(templates):
    values=[]
    for b in BNODES:
        bp=2*1.686*(b-P_CAL);f=F_GROWTH;t=templates
        value=np.zeros((3,3,templates.shape[1]))
        value[0,0]=b*b*t[0]+2*b*f*t[1]+f*f*t[2]
        value[0,1]=2*bp*(b*t[3]+f*t[4]);value[0,2]=bp*bp*t[5]
        values.append(value)
    v=np.polynomial.chebyshev.chebvander((BNODES-2.25)/1.75,3)
    return np.linalg.solve(v,np.asarray(values).reshape(4,-1)).reshape(4,3,3,templates.shape[1])

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'contract.json').exists():
        contract=json.loads((OUT/'contract.json').read_text())
        for path,sha in contract['immutable_inputs'].items():
            if digest(path)!=sha:raise ValueError('input changed: '+path)
        print('REUSE P+B contract',flush=True);return
    _,edges=p_geometry();rows=json.loads((LEGACY_OUT/'contract.json').read_text())['samples']
    window_audit=json.loads((OUT/'P0_WINDOW_AUDIT.json').read_text())
    if window_audit['status']!='PASS' or window_audit['window_sha256']!=digest(OUT/'P0_WINDOW.npz'):
        raise ValueError('production P window not independently audited')
    with np.load(OUT/'P0_WINDOW.npz') as z:window_meta=json.loads(str(z['metadata_json']))
    paths=[B_OUT/'contract.json',B_OUT/'RESULTS.json',B_OUT/'RSD_B000_VALIDATION.pdf',
           LEGACY_OUT/'B000_ENSEMBLE_MATRIX.npz',P_MATRIX,REAL_P_SAMPLE,SHARED,
           PART/'code/run_rsd_b000_mcmc.py',PART/'code/rsd_b000_fft_projection.py',
           PART/'code/rsd_pb_common.py',PART/'code/measure_rsd_p0.py',
           PART/'code/run_rsd_p0_measure.sh',PART.parent/'01_reconstruction_methods/common/nersc_env.sh',
           PART/'code/build_rsd_p0_window.py',OUT/'P0_WINDOW.npz',
           OUT/'P0_WINDOW_AUDIT.json',Path(window_meta['binner_source']),
           PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    for name in json.loads((B_OUT/'contract.json').read_text())['baseline_case_names']:
        paths.extend(B_OUT/'cases'/name/p for p in ('likelihood.npz','contract.json','SUMMARY.json','RUN_PROVENANCE.json'))
    value={'status':'FROZEN_AWAITING_RSD_P0','samples':rows,'n_covariance':500,'n_target_per_sign':100,
           'volume_Gpc_h3':1.,'covariance_multiplier':1.,'covariance_is_of_mean':False,
           'observable':'P0[12] + Sugiyama B000; full same-realization P-B covariance',
           'p_edges':edges.tolist(),'kmax_P':.08,'B_cuts':list(CUTS),'P_likelihood':True,'P_derived_b1_prior':False,
           'P_model':'exact-lattice (b1+f mu^2+fNL bphi/M)^2 PL + independent residual Pshot',
           'P_window':'exported from actual production BinMesh2SpectrumPoles labels/weights; same 12 realspace edges; CPU/GPU ULP counts may differ',
           'B_model':'unchanged frozen B-only coefficients/noise/priors; no bphidelta or stochastic repair in this controlled test',
           'extra_parameter':'Pshot_residual_nbar; flat physical residual Pshot in [-1e6,1e6] (Mpc/h)^3',
           'p_fixed':P_CAL,'f_growth_fixed':F_GROWTH,'IR':False,'FoG':False,'loops':False,
           'measurement_mesh':256,'fine_mesh':512,'shared_measurement':'halo_b000_pipeline.measure_pk',
           'immutable_inputs':{str(p):digest(p) for p in paths},'source_sha256':digest(Path(__file__)),
           'maximum_CPUs':8,'sampler':'reuse unchanged B-only residual/proposal/diagnostic entry via explicit P+B adapter'}
    t,n,k=p_lattice(edges)
    np.savez_compressed(OUT/'P0_LATTICE.npz',templates=t,nmodes=n,k=k,edges=edges,coefficients=p_coefficients(t))
    value['p_template_sha256']=digest(OUT/'P0_LATTICE.npz')
    save_json(OUT/'contract.json',value);(OUT/'measurements_m256').mkdir(exist_ok=True)
    (OUT/'measurements_m512').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    print('FROZEN P0[12]+B; Csingle; V1; 800 matched samples',flush=True)

if __name__=='__main__':freeze()
