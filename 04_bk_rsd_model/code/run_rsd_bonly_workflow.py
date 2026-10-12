"""模块大纲：只读样本核验；压缩 self-contraction 几何；B 测量；源模板；协方差。

子命令按验收门顺序执行；manifest 冻结后不自动重选样本或修改旧产物。
原始 FoF 和旧位置仅只读；无 power likelihood、P-derived prior 或 P 拟合。
"""
import argparse
import json
import struct
import time
import numpy as np
from rsd_bonly_common import *
from export_pre_rsd import redshift_positions
from periodic_rsd_measure import catalog_fourier
from readfof import FoF_catalog

def raw_path(cosmo,real):
    return RAW_ROOT/cosmo/f'real{real:03d}'/'groups_003/group_tab_003.0'

def header_check(path):
    with path.open('rb') as handle: ng,nt,ni,tid,nf=struct.unpack('<iiiQI',handle.read(24))
    if ng<=0 or ng!=nt or nf!=1 or path.stat().st_size!=24+84*ng:
        raise ValueError(f'incomplete/unsupported FoF: {path}, header={(ng,nt,nf)}, size={path.stat().st_size}')
    return ng

def catalog(cosmo,real):
    source=raw_path(cosmo,real); header_check(source)
    fof=FoF_catalog(str(source.parent.parent),3,read_IDs=False)
    mass=fof.GroupMass.astype(float)*1e10; selected=mass>=1e13
    pos=fof.GroupPos[selected]/np.float32(1000.)
    old=OLD_POS/cosmo/f'real{real:03d}'
    # 不要求旧导出全覆盖；存在时必须逐行完全相等。
    if (old/'positions_mpc_h.npy').exists():
        np.testing.assert_array_equal(pos,np.load(old/'positions_mpc_h.npy'))
        np.testing.assert_array_equal(mass[selected],np.load(old/'masses_msun_h.npy'))
    rsd,vel,ds,hh=redshift_positions(pos,fof.GroupVel[selected],.5,.3175,1000.)
    if not np.isfinite(rsd).all() or not np.isfinite(vel).all(): raise ValueError('nonfinite catalog')
    np.testing.assert_array_equal(rsd[:,:2],pos[:,:2])
    return rsd,pos,{'n_halo':len(pos),'nbar_h3_Mpc3':len(pos)/1e9,'rms_ds_Mpc_h':float(np.sqrt(np.mean(ds*ds))),
                    'source_raw':str(source),'source_raw_sha256':digest(source),'H_h':hh}

def manifest():
    target=OUT/'contract.json'
    if target.exists(): raise FileExistsError('frozen contract exists; inspect/reuse, never overwrite')
    # 100 个跨三 cosmology 的验证 IC；500 个独立 fid IC 建协方差。
    rows=[]
    for cosmo,ids,role in [('fiducial',range(100),'validation'),('LC_p',range(100),'validation'),
                           ('LC_m',range(100),'validation'),('fiducial',range(1000,1500),'covariance')]:
        for real in ids:
            source=raw_path(cosmo,real)
            rows.append({'cosmology':cosmo,'realization':real,'role':role,'pilot':role=='validation' and real<20,
                         'raw_path':str(source),'header_nhalo':header_check(source),'raw_size':source.stat().st_size})
    geometry=fft.build_geometry(64,1000.,EDGES)
    value={'version':'pre_rsd_bonly_v1','data_stage':'pre-recon RSD LOS-z; unit-weight M>=1e13 halos',
           'meshsize':128,'boxsize_mpc_h':1000.,'single_box_volume_Gpc_h3':1.,'edges_h_Mpc':EDGES.tolist(),
           'cuts_h_Mpc':list(CUTS),'triangle_bins':len(geometry.triangles),
           'bins_per_cut':{str(c):int(cut_mask(geometry.triangles,c).sum()) for c in CUTS},
           'samples':rows,'n_covariance':500,'n_validation_per_cosmology':100,'n_pilot_per_cosmology':20,
           'parameters':list(NAMES),'uniform_prior_bounds':BOUNDS.tolist(),'p_external_fixed':P_CAL,'f_growth_fixed':F_GROWTH,
           'pbs_recipe':'same as part3; bphi=2*delta_c*(b1-p); higher responses from shared recipe',
           'P_likelihood':False,'P_derived_b1_prior':False,'P_data_vector':False,
           'IR':False,'FoG':False,'loops':False,'counterterms':False,
           'noise':'two effective B0 residual shapes: exact lattice <sum Plin>/nbar and 1/nbar^2',
           'finite_PNG_stochastic_closure':'not implemented; provisional systematic, not complete stochastic PNG model',
           'point_catalog_self_subtraction':'Braw-<sum Praw>/nbar+2/nbar^2; CIC error tested separately',
           'source_projection':'exact same three-leg Cartesian closed triangle list as estimator',
           'spectrum_path':str(TABLE),'spectrum_sha256':TABLE_HASH,
           'likelihood':'Gaussian Hartlap precision; independently estimated fiducial RSD covariance; no old realspace matrix',
           'mean_data_volume':'100 independent boxes; C_single/100; separate single-box V=1 checks',
           'pairing':'same real index, empirical low-k phase QC required; no unverified fiducial seed claim',
           'pilot_gate':'complete catalogs, phase QC, exact counts, finite B, 128/256 mesh test, native/template test; NOT truth recovery',
           'convergence':{'ensembles':4,'walkers':32,'min_steps':6000,'max_steps':24000,'burn_fraction':.5,
                          'rhat_max':1.01,'fNL_bulk_tail_ESS_min':2000,'other_bulk_tail_ESS_min':1000,'length_over_tau_min':50},
           'status':'frozen; pilot not yet accepted'}
    save_json(target,value); print(json.dumps({k:v for k,v in value.items() if k!='samples'},indent=2),flush=True)

def prepare_geometry(n=128):
    geometry=fft.build_geometry(n,1000.,EDGES)
    selected=geometry.shell_index>=0
    # 原 estimator 的 pair_mode_multiplicity 为唯一 convolution 实现。
    # 把 full grid 压缩为 bins x low-k modes，避免每个 realization 缓存 21 个大 grid。
    weights=np.zeros((len(geometry.triangles),int(selected.sum())))
    for i in range(geometry.nshell):
        for j in range(i,geometry.nshell):
            multiplicity=fft.reverse_fourier_indices(fft.pair_mode_multiplicity(i,j,geometry))[selected]
            labels=geometry.shell_index[selected]
            for row,triangle in enumerate(geometry.triangles):
                for remaining in range(3):
                    other=tuple(sorted(np.delete(triangle,remaining)))
                    if other==(i,j): weights[row]+=multiplicity*(labels==triangle[remaining])/geometry.triangle_counts[row]
    np.testing.assert_allclose(weights.sum(axis=1),3.,rtol=1e-13)
    target=OUT/f'geometry_m{n}.npz'
    np.savez_compressed(target,triangles=geometry.triangles,counts=geometry.triangle_counts,
                        kmean=geometry.k_mean_h_mpc,weights=weights,selected_flat=np.flatnonzero(selected),
                        shell_labels=geometry.shell_index[selected])
    print('GEOMETRY COMPLETE '+str(target),flush=True)

def measure(args):
    contract=json.loads((OUT/'contract.json').read_text()); n=args.meshsize
    geometry=fft.build_geometry(n,1000.,EDGES)
    with np.load(OUT/f'geometry_m{n}.npz') as z:
        weights=z['weights']; indices=z['selected_flat']; labels=z['shell_labels']
        np.testing.assert_array_equal(z['counts'],geometry.triangle_counts)
    rows=contract['samples']
    rows=[r for r in rows if (args.role=='all' or r['role']==args.role) and (not args.pilot or r['pilot'])]
    rows=rows[args.start:args.stop]; output=OUT/f'measurements_m{n}'; output.mkdir(exist_ok=True)
    for row in rows:
        t=time.time();cosmo=row['cosmology'];real=row['realization'];tag=f'{cosmo}_real{real:03d}'
        target=output/(tag+'.npz')
        if target.exists():
            with np.load(target) as saved:
                if saved['B_factorial'].shape!=(len(geometry.triangles),) or not np.isfinite(saved['B_factorial']).all(): raise ValueError(target)
            print('REUSE '+tag,flush=True);continue
        pos,realpos,meta=catalog(cosmo,real); field=catalog_fourier(pos,n)
        filtered=fft.filter_normalized_fourier(field,geometry)
        braw=fft.measure_bispectrum(filtered,filtered,filtered,geometry)
        # 此 Praw 是 B 的同对象收缩，不保存/拟合任何 P 数据。
        pair=geometry.volume_mpc3_h3*weights@(np.abs(field.ravel()[indices])**2)/meta['nbar_h3_Mpc3']
        triple=2/meta['nbar_h3_Mpc3']**2
        b=braw-pair+triple
        if not np.isfinite(b).all(): raise ValueError('nonfinite B')
        # 仅相位/IC QC，不是 P 检验；用低 k real-space 场消除 RSD 模型假设。
        phase=catalog_fourier(realpos,32)
        aa=np.fft.fftfreq(32)*32; xx,yy,zz=np.meshgrid(aa,aa,aa,indexing='ij')
        pm=(xx*xx+yy*yy+zz*zz>=1)&(xx*xx+yy*yy+zz*zz<=9)
        meta.update(row);meta.update({'meshsize':n,'elapsed_seconds':time.time()-t,'fNL_truth':{'fiducial':0,'LC_p':100,'LC_m':-100}[cosmo],
                                    'data_has_rsd':True,'P_likelihood':False,'contract_sha256':digest(OUT/'contract.json')})
        np.savez_compressed(target,B_raw=braw,B_factorial=b,pair_self=pair,triple_self=triple,
                            phase_fingerprint=phase[pm],metadata_json=np.array(json.dumps(meta)))
        print('MEASURE '+tag+' '+json.dumps(meta),flush=True)

def templates(args):
    geometry,modes=mode_geometry(64); sample=load_spectrum()
    target=OUT/'template_bins'; target.mkdir(exist_ok=True)
    params=[(b,0.,0.) for b in BNODES]+[(b,1.,0.) for b in BNODES]+[(b,0.,1.) for b in BNODES]
    # 独立非节点点检验三次插值与 affine b2/K2；不拿训练节点自验。
    rng=np.random.default_rng(90210+args.start)
    probes=np.column_stack((rng.uniform(.5,4.,3),rng.uniform(-2,3,3),rng.uniform(-2,2,3)))
    vander=np.polynomial.chebyshev.chebvander((BNODES-2.25)/1.75,3)
    for index in range(args.start,min(args.stop,len(geometry.triangles))):
        filename=target/f'bin{index:03d}.npz'
        if filename.exists(): print('REUSE '+str(filename),flush=True);continue
        t=time.time(); raw=project_triangle(geometry,modes,index,params+list(probes),sample)
        design=np.stack((raw[:4,:3],raw[4:8,:3]-raw[:4,:3],raw[8:12,:3]-raw[:4,:3]),axis=1)
        coefficients=np.linalg.solve(vander,design.reshape(4,-1)).reshape(4,3,3)
        probe_f=rng.uniform(-500,500,3)
        physical=np.column_stack((probe_f,probes,np.zeros((3,2))))
        predicted=prediction(physical,coefficients[...,None],np.zeros((2,1)))[:,0]
        expected=np.sum(raw[12:,:3]*np.column_stack((np.ones(3),probe_f,probe_f**2)),axis=1)
        rel=np.max(np.abs(predicted-expected)/np.maximum(1.,np.abs(expected)))
        if rel>2e-10: raise AssertionError(('template native mismatch',index,rel))
        np.savez_compressed(filename,coefficients=coefficients,stochastic_nodes=raw[:4,3:],probes=physical,
                            probe_direct=expected,probe_prediction=predicted,relative_error=rel)
        print(f'TEMPLATE bin={index} seconds={time.time()-t:.2f} error={rel:.3g}',flush=True)

def read_measurement(cosmo,real,n=128):
    with np.load(OUT/f'measurements_m{n}'/f'{cosmo}_real{real:03d}.npz') as z:
        return {k:np.array(z[k]) for k in z.files if k!='metadata_json'},json.loads(str(z['metadata_json']))

def pilot():
    fields={c:[] for c in ('fiducial','LC_p','LC_m')}; phase=[];wrong=[]
    for i in range(20):
        rows={c:read_measurement(c,i)[0] for c in fields}
        for c in fields: fields[c].append(rows[c]['B_factorial'])
        for c in ('LC_p','LC_m'):
            a=rows['fiducial']['phase_fingerprint'];b=rows[c]['phase_fingerprint']
            phase.append(float(np.vdot(a,b).real/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real)))
        a=rows['LC_p']['phase_fingerprint'];b=read_measurement('fiducial',(i+1)%20)[0]['phase_fingerprint']
        wrong.append(float(np.vdot(a,b).real/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real)))
    std=np.std(fields['fiducial'],axis=0,ddof=1)
    mesh=[]
    for c in fields:
        coarse=read_measurement(c,0)[0]; fine=read_measurement(c,0,256)[0]
        diff=fine['B_factorial']-coarse['B_factorial']
        mesh.append({'cosmology':c,'max_abs_mesh_delta_over_singlebox_sigma':float(np.max(np.abs(diff)/std)),
                     'max_abs_mesh_delta_over_mean100_sigma':float(np.max(10*np.abs(diff)/std))})
    accepted=min(phase)>.95 and max(row['max_abs_mesh_delta_over_mean100_sigma'] for row in mesh)<.1
    value={'status':'PASS' if accepted else 'FAIL','n_triplets':20,'matched_phase_correlation_min_max':[min(phase),max(phase)],
           'mismatched_control_phase_correlation_min_max':[min(wrong),max(wrong)],'mesh_convergence':mesh,
           'sigma_source':'20 fiducial B scatter; mesh test includes factorial self corrections',
           'gate':'matched phase>.95; max128/256 B shift <0.1 sigma(mean100)',
           'truth_recovery_is_not_a_pilot_gate':True,'bins':len(std)}
    save_json(OUT/'PILOT_AUDIT.json',value);print(json.dumps(value,indent=2),flush=True)
    if not accepted: raise ValueError('pilot gate failed: do not submit bulk/MCMC')

def aggregate():
    if json.loads((OUT/'PILOT_AUDIT.json').read_text())['status']!='PASS': raise ValueError('pilot not accepted')
    geometry,modes=mode_geometry(64)
    coeff=np.stack([np.load(OUT/'template_bins'/f'bin{i:03d}.npz')['coefficients'] for i in range(len(geometry.triangles))],axis=-1)
    maxerr=max(float(np.load(OUT/'template_bins'/f'bin{i:03d}.npz')['relative_error']) for i in range(len(geometry.triangles)))
    with np.load(OUT/'geometry_m128.npz') as z: weights=z['weights'];indices=z['selected_flat']
    axis=2*np.pi*np.fft.fftfreq(128,d=1000./128);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True)
    kval=np.sqrt(x*x+y*y+z*z).ravel()[indices];p,_=load_spectrum()(kval);sumP=weights@p
    samples=[read_measurement('fiducial',i) for i in range(1000,1500)]
    covdata=np.array([r[0]['B_factorial'] for r in samples]);cov=np.cov(covdata,rowvar=False,ddof=1)
    if not np.isfinite(cov).all(): raise ValueError('nonfinite covariance')
    eigen=np.linalg.eigvalsh(cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov))))
    if eigen.min()<=0: raise ValueError('nonpositive covariance; no silent ridge')
    cases=OUT/'cases';cases.mkdir(exist_ok=True); summaries=[]
    for cosmo,truth in [('fiducial',0),('LC_p',100),('LC_m',-100)]:
        validation=[read_measurement(cosmo,i) for i in range(100)]
        data=np.array([r[0]['B_factorial'] for r in validation]);nbars=np.array([r[1]['nbar_h3_Mpc3'] for r in validation])
        for volume in ('mean100','single000'):
            nmean=100 if volume=='mean100' else 1
            chosen=data.mean(axis=0) if nmean==100 else data[0]
            invn=float(np.mean(1/nbars)) if nmean==100 else 1/nbars[0]
            invn2=float(np.mean(1/nbars**2)) if nmean==100 else 1/nbars[0]**2
            noise=np.stack((sumP*invn,np.full(len(sumP),invn2)))
            for cut in (CUTS if nmean==100 else (.07,)):
                mask=cut_mask(geometry.triangles,cut);nd=int(mask.sum());alpha=(500-nd-2)/(500-1)
                if alpha<=0: raise ValueError('invalid Hartlap')
                subcov=cov[np.ix_(mask,mask)]/nmean
                # C_eff=C/Hartlap，正好对应 Hartlap precision，无伪造 likelihood normalisation。
                chol=np.linalg.cholesky(subcov/alpha)
                directory=cases/f'{cosmo}_{volume}_k{cut:.3f}';directory.mkdir(exist_ok=True)
                np.savez_compressed(directory/'likelihood.npz',data=chosen[mask],coefficients=coeff[...,mask],noise=noise[:,mask],
                                    covariance=subcov,chol=chol,triangles=geometry.triangles[mask])
                meta={'cosmology':cosmo,'fNL_truth':truth,'data':'B_factorial only','n_validation':nmean,
                      'volume_Gpc_h3':nmean,'n_covariance':500,'bins':nd,'kmax':cut,'parameters':list(NAMES),
                      'prior_bounds':BOUNDS.tolist(),'Hartlap_factor':alpha,'P_likelihood':False,'P_derived_b1_prior':False,
                      'p_fixed_external':P_CAL,'f_growth_fixed':F_GROWTH,'covariance_ICs':list(range(1000,1500)),
                      'validation_ICs':list(range(nmean)),'contract_sha256':digest(OUT/'contract.json')}
                save_json(directory/'contract.json',meta);summaries.append({k:v for k,v in meta.items() if not k.endswith('_ICs')})
    np.savez_compressed(OUT/'B_COVARIANCE.npz',B_covariance_realizations=covdata,covariance=cov,correlation_eigenvalues=eigen,
                        coefficients=coeff,triangles=geometry.triangles,sum_Plin_lattice=sumP)
    # 两个原始 Gaussian RSD stochastic shapes 在 B0 上近退化，不能当成两个独立 Pshot/Bshot 参数。
    original=np.stack([np.load(OUT/'template_bins'/f'bin{i:03d}.npz')['stochastic_nodes'][1,:2] for i in range(len(sumP))]).T
    whiten=np.linalg.solve(np.linalg.cholesky(cov),original.T);svd=np.linalg.svd(whiten,compute_uv=False)
    cosine=float(np.dot(whiten[:,0],whiten[:,1])/np.linalg.norm(whiten[:,0])/np.linalg.norm(whiten[:,1]))
    value={'status':'READY_FOR_MCMC','n_covariance':500,'n_validation_per_sign':100,'n_mean_cases':12,'n_single_box_cases':3,
           'correlation_eigenvalue_min_max':[float(eigen.min()),float(eigen.max())],'template_max_relative_error':maxerr,
           'gaussian_stochastic_whitened_cosine':cosine,'gaussian_stochastic_singular_values':svd.tolist(),
           'noise_parameterization':'effective sum Plin/nbar + constant/nbar2; discarded lattice SB/SP distinction is provisional systematic',
           'covariance_truth':'fiducial covariance held common across PNG signs; PNG covariance changes not included',
           'cases':summaries}
    save_json(OUT/'AGGREGATION_AUDIT.json',value);print(json.dumps(value,indent=2),flush=True)

def main():
    if (OUT/'SUPERSEDED_BY_REALSPACE_BINNING_REVIEW.json').exists():
        raise RuntimeError('Strict-three-leg plan superseded by user: use realspace binning review, do not run this draft')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['manifest','geometry','measure','templates','pilot','aggregate'])
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=100000)
    parser.add_argument('--meshsize',type=int,default=128);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--role',choices=['all','validation','covariance'],default='all');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.command=='manifest':manifest()
    elif args.command=='geometry':prepare_geometry(args.meshsize)
    elif args.command=='measure':measure(args)
    elif args.command=='templates':templates(args)
    elif args.command=='pilot':pilot()
    elif args.command=='aggregate':aggregate()

if __name__=='__main__':main()
