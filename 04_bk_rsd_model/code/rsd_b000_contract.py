"""模块大纲：实空间两腿 B000 合同；固定样本/先验；独立 RSD 输出。

扫描起点是 kmax_B=.08，不改变原 radial kmin；旧 strict 草稿只读保留。
"""
from pathlib import Path
import json
import os
import numpy as np
from rsd_bonly_common import PART, TABLE, TABLE_HASH, F_GROWTH, P_CAL, bias, load_spectrum, digest, save_json

LEGACY_OUT=PART/'outputs/pre_rsd_b000_bonly_v2'
# 均值只用于 data vector；目标体积不从 realization 数推断。
OUT=Path(os.environ.get('MARISA_RSD_B000_OUTPUT',str(PART/'outputs/pre_rsd_b000_bonly_mean100_v1')))
REFERENCE=PART/'outputs/realspace_binning_reference/REALSPACE_BINNING.npz'
CUTS=(.08,.10,.12,.14,.18,.20)
ALIASES={'.15':'.14'}
NAMES=('fNL','b1','b2_native','bK2','A_stoch','S_stoch')
BOUNDS=np.array([[-500.,500.],[.5,4.],[-4.,6.],[-4.,4.],[-10.,10.],[-20.,20.]])
BNODES=2.25+1.75*np.cos((np.arange(4)+.5)*np.pi/4)
OLD_SAMPLE_ROOT=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_baseline_b000_pk/full')
RANDOM_NAMESPACE='quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_baseline_b000_pk'

def reference():
    with np.load(REFERENCE) as z:return {k:np.array(z[k]) for k in z.files}

def selected(cut):
    ref=reference();idx=ref['certified_indices']
    return idx[np.max(ref['B000_k'][idx],axis=1)<=cut+1e-12]

def freeze_contract():
    from run_rsd_bonly_workflow import raw_path,header_check
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'contract.json'
    if target.exists():
        data=json.loads(target.read_text())
        if data['reference_sha256']!=digest(REFERENCE):raise ValueError('reference changed after freeze')
        print('REUSE frozen contract',flush=True);return
    rows=[]
    for cosmo,ids,role in [('fiducial',range(100),'validation'),('LC_p',range(100),'validation'),
                           ('LC_m',range(100),'validation'),('fiducial',range(1000,1500),'covariance')]:
        for i in ids:
            p=raw_path(cosmo,i)
            rows.append({'cosmology':cosmo,'realization':i,'role':role,'pilot':role=='validation' and i<20,
                         'raw_path':str(p),'header_nhalo':header_check(p),'raw_size':p.stat().st_size})
    value={'version':'pre_rsd_b000_bonly_mean100_v1','status':'FROZEN_AWAITING_PILOT','samples':rows,
           'observable':'Sugiyama B000(k1,k2); third leg unrestricted; full-shell denominator including zero-mode cells',
           'reference_path':str(REFERENCE),'reference_sha256':digest(REFERENCE),'edges_h_Mpc':reference()['edges'].tolist(),
           'cuts_h_Mpc':list(CUTS),'cut_counts':{str(c):len(selected(c)) for c in CUTS},'k015_alias_of_k014':True,
           'data_stage':'pre_recon','data_has_rsd':True,'redshift':.5,'mass_min_Msun_h':1e13,'boxsize_Mpc_h':1000.,
           'los':[0,0,1],'measurement_meshsize':256,'measurement_resampler':'TSC, interlacing=3, compensate=True',
           'measurement':'reuse part1 halo_b000_pipeline.measure_b000 and shared cosmodesi estimator; P disabled',
           'random_catalog':'unshifted uniform, nran=30; same generator/namespace as realspace; independent cosmo seeds',
           'zero_mode_theory':'all 3 nonzero; excluded terms contribute zero, retain full 2-shell denominator',
           'parameters':list(NAMES),'bounds':BOUNDS.tolist(),'p_external_fixed':P_CAL,'f_growth_fixed':F_GROWTH,
           'IR':False,'FoG':False,'loops':False,'counterterms':False,'P_likelihood':False,'P_derived_b1_prior':False,
           'finite_PNG_stochastic':'not implemented; explicit provisional model systematic',
           'covariance':'500 independent fiducial RSD boxes 1000..1499; full covariance, Hartlap precision',
           'validation':'100 phase-QC matched boxes per sign 0..99; no IC overlap with covariance',
           'mean_likelihood':'100-box mean data vector; C_single without division by target/covariance realization count; V=1',
           'volume_Gpc_h3':1.,'covariance_multiplier':1.,
           'baseline_MCMC':'18 mean100 cases plus 3 single000 cases at kmax=.12; .15 aliases .14',
           'sensitivity_MCMC':'at .08 and provisional boundary per sign, widen nuisance bounds; up to 6 cases',
           'sampler':{'ensembles':4,'walkers':32,'initial_steps':6000,'increment':2000,'max_steps':24000,'burn_fraction':.5},
           'numeric_gates':{'template_max_whitened_error':1e-5,'mesh_shift_mean100_sigma':.1,'phase_correlation_min':.95},
           'spectrum':{'path':str(TABLE),'sha256':TABLE_HASH},'source_library_sha256':digest(PART/'work/build/libmarisa_tree_rsd.so')}
    save_json(target,value)
    print(json.dumps({k:v for k,v in value.items() if k!='samples'},indent=2),flush=True)

if __name__=='__main__':freeze_contract()
