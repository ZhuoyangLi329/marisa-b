"""模块大纲：pilot 数值门；800份矩阵/协方差；随机形状秩；冻结21组B-only似然。

所有门都是数据完整性/数值精度门，不把 fNL 真值恢复作为筛选输入的条件。
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rsd_b000_contract import *

def read_measurement(cosmo,real,n=256):
    path=OUT/f'measurements_m{n}'/f'{cosmo}_real{real:03d}.npz'
    with np.load(path) as z:
        data={k:np.array(z[k]) for k in z.files if k!='metadata_json'}
        meta=json.loads(str(z['metadata_json']))
    if meta['contract_sha256']!=digest(OUT/'contract.json'):raise ValueError('measurement contract drift')
    if 'periodic_guard' not in meta:raise ValueError('measurement predates shared periodic guard')
    return data,meta

def correlation(a,b):return float(np.vdot(a,b).real/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real))

def templates():
    ref=reference();idx=ref['certified_indices'];coefs=[];noise=[];errors=[];probes=[];counts=[]
    for index in idx:
        with np.load(OUT/'template_bins_m144'/f'bin{index:03d}.npz') as z:
            if str(z['contract_sha256'])!=digest(OUT/'contract.json'):raise ValueError('template contract drift')
            coefs.append(z['coefficients']);noise.append(z['stochastic_geometry']);errors.append(float(z['probe_relative_error']))
            probes.append(z['probe_prediction']-z['probe_direct']);counts.append(z['counts'])
    np.testing.assert_allclose(np.asarray(noise)[:,1],np.asarray(noise)[:,0]/3,rtol=3e-11,atol=1e-6)
    return np.stack(coefs,axis=-1),np.array(noise).T,np.array(errors),np.stack(probes,axis=-1),np.array(counts)

def pilot():
    ref=reference();idx=ref['certified_indices'];covrows=[];phase=[];wrong=[];mode_ref=None
    for i in range(20):
        group={c:read_measurement(c,i)[0] for c in ('fiducial','LC_p','LC_m')}
        covrows.append(group['fiducial']['B'])
        for c,data in group.items():
            if mode_ref is None:mode_ref=data['B000_nmodes']
            np.testing.assert_array_equal(data['B000_nmodes'],mode_ref)
            np.testing.assert_array_equal(data['B000_k_edges'],ref['B000_k_edges'])
            if c!='fiducial':phase.append(correlation(group['fiducial']['phase_fingerprint'],data['phase_fingerprint']))
        wrong.append(correlation(group['LC_p']['phase_fingerprint'],read_measurement('fiducial',(i+1)%20)[0]['phase_fingerprint']))
    sigma=np.std(covrows,axis=0,ddof=1);mesh=[]
    for c in ('fiducial','LC_p','LC_m'):
        coarse=read_measurement(c,0)[0];fine=read_measurement(c,0,512)[0]
        np.testing.assert_array_equal(coarse['B000_nmodes'],fine['B000_nmodes'])
        shift=10*np.abs(fine['B'][idx]-coarse['B'][idx])/sigma[idx]
        mesh.append({'cosmology':c,'max_shift_over_mean100_sigma':float(np.max(shift))})
    coef,noise,errors,probe_delta,counts=templates()
    np.testing.assert_array_equal(np.prod(counts,axis=1),mode_ref[idx])
    if not np.isfinite(errors).all() or errors.max()>2e-10:raise ValueError('template polynomial audit failed')
    theory_mesh=[]
    for position in (0,13,43):
        index=idx[position]
        with np.load(OUT/'template_bins_m160'/f'bin{index:03d}.npz') as z:
            if str(z['contract_sha256'])!=digest(OUT/'contract.json'):raise ValueError('fine template contract drift')
            delta=z['coefficients']-coef[...,position]
        # 极端 prior corner 也检查；实际完整 covariance whitening 在 aggregate 再做。
        f=500.;b=.61;b2=5.;bk=3.
        cb=np.polynomial.chebyshev.chebvander((b-2.25)/1.75,3)[0]
        shift=float(np.einsum('a,b,c,abc',cb,np.array([1,b2,bk]),np.array([1,f,f*f]),delta))
        theory_mesh.append({'bin':int(index),'absolute_shift':shift,'shift_over_mean100_sigma':float(abs(shift)*10/sigma[index])})
    control=np.load(OUT/'REALSPACE_MEASUREMENT_CONTROL_fiducial.npz')
    control_shift=float(np.max(10*np.abs(control['delta'][idx])/sigma[idx]))
    accepted=(min(phase)>.95 and max(r['max_shift_over_mean100_sigma'] for r in mesh)<.1
              and max(r['shift_over_mean100_sigma'] for r in theory_mesh)<1e-5)
    value={'status':'PASS' if accepted else 'FAIL','matched_phase_min_max':[min(phase),max(phase)],
           'mismatched_phase_min_max':[min(wrong),max(wrong)],'n_triplets':20,'B_bins_measured':120,
           'measurement_mesh_convergence':mesh,'theory_mesh_convergence':theory_mesh,
           'realspace_legacy_reproduction_max_shift_mean100_sigma':control_shift,
           'measurement_nmodes_legacy_different_bins':int(np.count_nonzero(mode_ref!=ref['B000_nmodes'])),
           'polynomial_max_relative_error':float(errors.max()),'truth_recovery_gate':False,
           'sigma_note':'pilot fiducial scatter; no old realspace covariance used in likelihood',
           'projection':'exact lattice source decomposition, same two-leg edges/full denominator; independently native audited'}
    save_json(OUT/'PILOT_AUDIT.json',value);print(json.dumps(value,indent=2),flush=True)
    if not accepted:raise ValueError('numeric pilot gate failed; bulk/MCMC must not start')

def predicted(values,coef,noise):
    values=np.atleast_2d(values);f,b,b2,bk=values[:,:4].T
    cb=np.polynomial.chebyshev.chebvander((b-2.25)/1.75,3)
    nu=np.column_stack((np.ones(len(b)),b2,bk));ff=np.column_stack((np.ones(len(f)),f,f*f))
    features=(cb[:,:,None,None]*nu[:,None,:,None]*ff[:,None,None,:]).reshape(len(b),-1)
    return features@coef.reshape(-1,coef.shape[-1])+values[:,4:]@noise

def freeze_case(directory,metadata,data,cov,coef,noise,bounds,names,indices):
    if (directory/'chain.h5').exists():raise ValueError('never refreeze a running/existing chain')
    directory.mkdir(parents=True,exist_ok=True);alpha=(500-len(indices)-2)/499
    chol=np.linalg.cholesky(cov/alpha)
    np.savez_compressed(directory/'likelihood.npz',target=data,covariance=cov,chol=chol,coefficients=coef,noise=noise,
                        bounds=bounds,indices=indices)
    metadata.update({'parameters':names,'bounds':bounds.tolist(),'B_bins':len(indices),'Hartlap_factor':alpha,
                     'P_likelihood':False,'P_derived_b1_prior':False,'likelihood_sha256':digest(directory/'likelihood.npz'),
                     'parent_contract_sha256':digest(OUT/'contract.json'),'p_fixed_external':P_CAL,'f_growth_fixed':F_GROWTH})
    save_json(directory/'contract.json',metadata)

def aggregate():
    if json.loads((OUT/'PILOT_AUDIT.json').read_text())['status']!='PASS':raise ValueError('pilot not accepted')
    ref=reference();idx=ref['certified_indices'];coef,sg,errors,delta,counts=templates()
    contract=json.loads((OUT/'contract.json').read_text());mode_ref=None;rows=[];metadata=[];phases={};all_phases=[]
    for row in contract['samples']:
        data,meta=read_measurement(row['cosmology'],row['realization'])
        if mode_ref is None:mode_ref=data['B000_nmodes']
        np.testing.assert_array_equal(data['B000_nmodes'],mode_ref)
        np.testing.assert_array_equal(data['B000_k_edges'],ref['B000_k_edges'])
        rows.append(data['B']);metadata.append(meta);all_phases.append(data['phase_fingerprint'])
        if row['role']=='validation':phases[(row['cosmology'],row['realization'])]=data['phase_fingerprint']
    phase_min=min(correlation(phases[('fiducial',i)],phases[(c,i)]) for i in range(100) for c in ('LC_p','LC_m'))
    if phase_min<.95:raise ValueError('full validation IC matching failed')
    phase_array=np.asarray(all_phases);norm=np.linalg.norm(phase_array,axis=1)
    if np.any(norm<=0) or not np.isfinite(norm).all():raise ValueError('invalid phase fingerprints')
    unit=phase_array/norm[:,None];cov_phase=unit[300:]
    cross=(cov_phase@unit[:100].conj().T).real
    within=(cov_phase@cov_phase.conj().T).real;np.fill_diagonal(within,0)
    phase_independence={'max_covariance_vs_validation_correlation':float(cross.max()),
                        'max_between_distinct_covariance_correlation':float(within.max()),
                        'note':'empirical duplicate-phase check in addition to disjoint IDs; not formal IC seed metadata'}
    if cross.max()>.95 or within.max()>.95:raise ValueError(('covariance phase duplication',phase_independence))
    data=np.array(rows);cov_full=np.cov(data[300:],rowvar=False,ddof=1);cov=cov_full[np.ix_(idx,idx)]
    corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)));eigen=np.linalg.eigvalsh(corr)
    if not np.isfinite(cov).all() or eigen.min()<=0:raise ValueError('invalid covariance; no automatic ridge')
    # 原生 SP 的 lattice mu^4 部分未必可忽略；在当前 B covariance 下量化后决定是否保留。
    nbar_fid=np.array([m['nbar_h3_Mpc3'] for m in metadata[:100]]);invn=float(np.mean(1/nbar_fid));invn2=float(np.mean(1/nbar_fid**2))
    base=np.stack((sg[0]*invn,sg[3]*invn2));extra=F_GROWTH**2*(sg[2]-sg[0]/5)*invn
    chol=np.linalg.cholesky(cov)
    white=np.linalg.solve(chol,base.T);wextra=np.linalg.solve(chol,extra)
    conversion=np.linalg.lstsq(white,wextra,rcond=None)[0];residual=extra-conversion@base
    maxshift=float(np.max(np.abs(np.linalg.solve(chol,residual))))
    keep_extra=10*maxshift>.1
    names=list(NAMES)+(['R_stoch_anisotropic'] if keep_extra else [])
    bounds=np.vstack((BOUNDS,[-10.,10.])) if keep_extra else BOUNDS.copy()
    noise_audit={'status':'PASS','parameter_count':len(names),'native_SP_independent_combination_retained':bool(keep_extra),
                 'unit_SP_residual_max_whitened_single_box_shift':maxshift,'max_shift_at_prior_amplitude10':10*maxshift,
                 'threshold_for_safe_two_shape_reduction':.1,'conversion_to_effective_A_S':conversion.tolist(),
                 'shapes':'A=sum Plin/nbar; S=valid_fraction/nbar^2; optional R=orthogonal native-SP lattice combination',
                 'finite_PNG_stochastic_closure':'still provisional, not implemented'}
    save_json(OUT/'MODEL_NOISE_AUDIT.json',noise_audit)
    # 第五 b1 点的误差按单盒 covariance 验收；均值不缩小 likelihood covariance。
    f=500.;nu=np.array([1.,6.,4.]);ff=np.array([1.,f,f*f]);template_shift=np.einsum('b,c,bcn->n',nu,ff,delta)
    template_error=float(np.max(np.abs(np.linalg.solve(chol,template_shift))))
    if template_error>1e-5:raise ValueError(('template whitening mismatch',template_error))
    cases=[]
    for j,(cosmo,truth) in enumerate([('fiducial',0),('LC_p',100),('LC_m',-100)]):
        sample=data[j*100:(j+1)*100];nbars=np.array([m['nbar_h3_Mpc3'] for m in metadata[j*100:(j+1)*100]])
        for volume in ('mean100','single000'):
            nmean=100 if volume=='mean100' else 1
            ni=float(np.mean(1/nbars)) if nmean==100 else 1/nbars[0]
            ni2=float(np.mean(1/nbars**2)) if nmean==100 else 1/nbars[0]**2
            noise=np.stack((sg[0]*ni,sg[3]*ni2))
            if keep_extra:noise=np.vstack((noise,F_GROWTH**2*(sg[2]-sg[0]/5)*ni-conversion@noise))
            target=sample.mean(axis=0) if nmean==100 else sample[0]
            for cut in (CUTS if nmean==100 else (.12,)):
                actual=selected(cut);lookup=np.array([list(idx).index(i) for i in actual])
                name=f'{cosmo}_{volume}_k{cut:.2f}';directory=OUT/'cases'/name
                meta={'name':name,'cosmology':cosmo,'truth_fNL':truth,'kmax_B':cut,'volume_Gpc_h3':1.,
                      'covariance_multiplier':1.,'covariance_is_of_mean':False,
                      'target_realizations':list(range(nmean)),'covariance_realizations':list(range(1000,1500)),
                      'n_target':nmean,'n_covariance':500,'noise_audit_sha256':digest(OUT/'MODEL_NOISE_AUDIT.json'),
                      'case_type':'baseline','IR':False,'FoG':False,'loops':False}
                freeze_case(directory,meta,target[actual],cov_full[np.ix_(actual,actual)],
                            coef[...,lookup],noise[:,lookup],bounds,names,actual)
                cases.append(name)
    np.savez_compressed(OUT/'B000_ENSEMBLE_MATRIX.npz',B=data,covariance_single_box=cov_full,
                        metadata_json=np.array([json.dumps(m) for m in metadata]),coefficients=coef,
                        stochastic_geometry=sg,certified_indices=idx,reference_edges=ref['edges'],
                        phase_fingerprints=phase_array,
                        matched_odd_response=(data[100:200]-data[200:300])/200,
                        matched_even_shift=(data[100:200]+data[200:300])/2-data[:100])
    save_json(OUT/'CASES.json',cases)
    np.testing.assert_array_equal(selected(.15),selected(.14))
    aliases={f'{c}_mean100_k0.15':{'source_case':f'{c}_mean100_k0.14','reason':'identical measured shell-average cutoff indices',
                                  'source_likelihood_sha256':digest(OUT/'cases'/f'{c}_mean100_k0.14'/'likelihood.npz')}
             for c in ('fiducial','LC_p','LC_m')}
    save_json(OUT/'ALIASES.json',aliases)
    summary={'status':'READY_FOR_MCMC','n_measurements':800,'n_covariance':500,'n_target_per_sign':100,
             'n_baseline_cases':21,'parameters':names,'phase_matching_min':phase_min,
             'phase_independence':phase_independence,
             'correlation_eigen_min_max':[float(eigen.min()),float(eigen.max())],
             'template_max_whitened_error':template_error,'noise_audit':noise_audit,
             'covariance_PNG_note':'fiducial covariance held common; differences across PNG signs not included in baseline',
             'mode_count_legacy_different_bins':int(np.count_nonzero(mode_ref!=ref['B000_nmodes']))}
    save_json(OUT/'AGGREGATION_AUDIT.json',summary);print(json.dumps(summary,indent=2),flush=True)

def correct_covariance(source=LEGACY_OUT):
    """只修正统计口径：复用不可变旧均值/模型，单盒 covariance，新目录新链。"""
    source=Path(source).resolve()
    if OUT.resolve()==source:raise ValueError('correction must never overwrite legacy output')
    if (OUT/'contract.json').exists():raise ValueError('correction output already frozen; do not overwrite')
    ensemble=source/'B000_ENSEMBLE_MATRIX.npz'
    with np.load(ensemble) as z:
        data=z['B'];cov=z['covariance_single_box']
    if data.shape!=(800,120):raise ValueError('expected immutable 800-by-120 measurement matrix')
    np.testing.assert_allclose(cov,np.cov(data[300:],rowvar=False,ddof=1),rtol=2e-13,atol=0)
    baseline=[f'{c}_mean100_k{k:.2f}' for c in ('fiducial','LC_p','LC_m') for k in CUTS]
    oldcontract=json.loads((source/'contract.json').read_text())
    parent={'version':'pre_rsd_b000_bonly_mean100_v1','status':'FROZEN_COVARIANCE_CORRECTION',
            'source_output':str(source),'source_contract_sha256':digest(source/'contract.json'),
            'source_ensemble_path':str(ensemble),'source_ensemble_sha256':digest(ensemble),
            'source_library_sha256':oldcontract['source_library_sha256'],
            'reference_sha256':digest(REFERENCE),'baseline_case_names':baseline,
            'mean_likelihood':'100-box mean data vector; unaveraged single-box covariance from Fid500; V=1',
            'volume_Gpc_h3':1.,'covariance_multiplier':1.,'n_target':100,'n_covariance':500,
            'covariance_is_of_mean':False,'P_likelihood':False,'IR':False,'FoG':False,'loops':False,
            'p_external_fixed':P_CAL,'f_growth_fixed':F_GROWTH,
            'correction':'only covariance normalization and resulting Hartlap Cholesky/volume metadata; preserve data, model, seven parameters and priors',
            'sampler':oldcontract['sampler'],'cuts_h_Mpc':list(CUTS)}
    noise=json.loads((source/'MODEL_NOISE_AUDIT.json').read_text())
    noise.update({'retained_for_covariance_only_comparison':True,
                  'audit_origin':'legacy numerical rank audit; keep all seven existing parameters, do not prune after changing statistical precision',
                  'likelihood_covariance_multiplier':1.})
    save_json(OUT/'contract.json',parent);save_json(OUT/'MODEL_NOISE_AUDIT.json',noise)
    checks=[]
    for j,cosmo in enumerate(('fiducial','LC_p','LC_m')):
        for cut in CUTS:
            name=f'{cosmo}_mean100_k{cut:.2f}';old=source/'cases'/name
            meta=json.loads((old/'contract.json').read_text())
            if digest(old/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('legacy likelihood changed')
            with np.load(old/'likelihood.npz') as z:arrays={k:np.array(z[k]) for k in z.files}
            idx=arrays['indices'];single=cov[np.ix_(idx,idx)]
            np.testing.assert_array_equal(idx,selected(cut))
            np.testing.assert_array_equal(arrays['target'],data[j*100:(j+1)*100].mean(axis=0)[idx])
            np.testing.assert_allclose(arrays['covariance']*100,single,rtol=2e-13,atol=0)
            meta.update({'volume_Gpc_h3':1.,'covariance_multiplier':1.,'covariance_is_of_mean':False,
                         'source_likelihood_path':str(old/'likelihood.npz'),'source_likelihood_sha256':meta['likelihood_sha256'],
                         'noise_audit_sha256':digest(OUT/'MODEL_NOISE_AUDIT.json')})
            target=OUT/'cases'/name
            freeze_case(target,meta,arrays['target'],single,arrays['coefficients'],arrays['noise'],
                        arrays['bounds'],meta['parameters'],idx)
            with np.load(target/'likelihood.npz') as z:
                for key in ('target','coefficients','noise','bounds','indices'):np.testing.assert_array_equal(z[key],arrays[key])
                np.testing.assert_array_equal(z['covariance'],single)
                np.testing.assert_allclose(z['chol']@z['chol'].T,single/meta['Hartlap_factor'],rtol=2e-13)
            checks.append({'name':name,'target_model_prior_unchanged':True,'covariance_is_unaveraged_single_box':True,
                           'source_likelihood_sha256':meta['source_likelihood_sha256'],'new_likelihood_sha256':digest(target/'likelihood.npz')})
    # 只引用已有 ensemble，不复制历史批量测量或链。
    (OUT/'B000_ENSEMBLE_MATRIX.npz').symlink_to(ensemble)
    save_json(OUT/'CASES.json',baseline)
    save_json(OUT/'ALIASES.json',{f'{c}_mean100_k0.15':{'source_case':f'{c}_mean100_k0.14',
              'reason':'identical two-leg indices; no duplicate MCMC'} for c in ('fiducial','LC_p','LC_m')})
    save_json(OUT/'COVARIANCE_CORRECTION_AUDIT.json',{'status':'PASS','n_baseline':18,'cases':checks,
              'source_ensemble_sha256':digest(ensemble),'old_outputs_modified':False,'data_mean_count':100,
              'covariance_sample_count':500,'covariance_multiplier':1.,'volume_Gpc_h3':1.,
              'covariance_definition':'np.cov(Fid500 single-box B vectors, rowvar=False, ddof=1); never /100 or /500',
              'hartlap_separate_from_volume':True,'new_source_sha256':digest(Path(__file__))})
    print('CORRECTION_FROZEN 18 mean-data single-covariance cases',flush=True)

def sensitivity():
    # 用保守的基线稳定性指标挑边界；非收敛结果不能决定有效性，也不能偷偷当作通过。
    cases=json.loads((OUT/'CASES.json').read_text());new=[]
    if (OUT/'SENSITIVITY_CASES.json').exists():
        previous=json.loads((OUT/'SENSITIVITY_CASES.json').read_text())
        if all(name in cases and (OUT/'cases'/name/'contract.json').exists() for name in previous):
            print('REUSE frozen sensitivity cases '+json.dumps(previous),flush=True);return
        raise ValueError('incomplete sensitivity case manifest; do not overwrite chains')
    for cosmo in ('fiducial','LC_p','LC_m'):
        candidates=[]
        for cut in CUTS:
            name=f'{cosmo}_mean100_k{cut:.2f}'
            report=json.loads((OUT/'cases'/name/'SUMMARY.json').read_text())
            if report['converged'] and report['gof_pvalue']>=.05 and abs(report['fNL_pull'])<=1:
                candidates.append(cut)
        boundary=max(candidates) if candidates else .08
        for cut in sorted({.08,boundary}):
            source=OUT/'cases'/f'{cosmo}_mean100_k{cut:.2f}'
            meta=json.loads((source/'contract.json').read_text());name=meta['name']+'_wide_nuisance'
            with np.load(source/'likelihood.npz') as z:arrays={k:np.array(z[k]) for k in z.files}
            bounds=arrays['bounds'].copy();bounds[2:]*=2
            # b1/fNL 先验保持不变，先单独检测 nuisance 先验而非同时改变 PBS/截断解释。
            meta.update({'name':name,'case_type':'wide_nuisance','baseline_case':source.name,'boundary_selection':'provisional: convergence + GOF + |pull|<=1; not final certification'})
            freeze_case(OUT/'cases'/name,meta,arrays['target'],arrays['covariance'],arrays['coefficients'],arrays['noise'],
                        bounds,meta['parameters'],arrays['indices'])
            new.append(name)
    save_json(OUT/'SENSITIVITY_CASES.json',new);save_json(OUT/'CASES.json',cases+new)
    print('SENSITIVITY_CASES '+json.dumps(new),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['pilot','aggregate','sensitivity','correct_covariance'])
    command=parser.parse_args().command
    {'pilot':pilot,'aggregate':aggregate,'sensitivity':sensitivity,'correct_covariance':correct_covariance}[command]()
