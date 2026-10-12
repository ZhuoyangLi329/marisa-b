"""模块大纲：同样本 P 数值门；完整单盒 P-B 协方差；冻结18组联合似然。

仅补 P 测量，直接引用旧 B 矩阵/目标/模型/先验；不以真值恢复筛选输入。
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rsd_pb_common import OUT,B_OUT,LEGACY_OUT,PART,digest,save_json,p_prediction

def parent():
    value=json.loads((OUT/'contract.json').read_text())
    if value['volume_Gpc_h3']!=1 or value['covariance_multiplier']!=1 or value['covariance_is_of_mean']:
        raise ValueError('not the requested Csingle V1 experiment')
    for path,sha in value['immutable_inputs'].items():
        if digest(path)!=sha:raise ValueError('frozen source/input drift: '+path)
    if digest(OUT/'P0_LATTICE.npz')!=value['p_template_sha256']:raise ValueError('P template drift')
    return value

def read(row,mesh=256):
    path=OUT/f'measurements_m{mesh}'/f"{row['cosmology']}_real{row['realization']:03d}.npz"
    with np.load(path) as z:
        data={k:np.array(z[k]) for k in z.files if k!='metadata_json'};meta=json.loads(str(z['metadata_json']))
    if meta['parent_sha256']!=digest(OUT/'contract.json'):raise ValueError('P measurement contract drift')
    if meta['entry_sha256']!=digest(PART/'code/measure_rsd_p0.py'):raise ValueError('P measurement source drift')
    if (meta['cosmology'],meta['realization'])!=(row['cosmology'],row['realization']):raise ValueError('row ID mismatch')
    return data,meta

def geometry(data):
    with np.load(OUT/'P0_LATTICE.npz') as z:
        np.testing.assert_array_equal(data['edges'],z['edges'])
        np.testing.assert_array_equal(data['nmodes'],z['nmodes'])
        np.testing.assert_allclose(data['k'],z['k'],rtol=1e-6,atol=1e-10)

def pilot():
    value=parent();samples=[r for r in value['samples'] if r['pilot']]
    fid=[]
    for row in samples:
        data,meta=read(row);geometry(data)
        if not meta['P_B_same_raw_sample'] or not meta['data_has_rsd']:raise ValueError('P is not matched RSD data')
        if row['cosmology']=='fiducial':fid.append(data['P0'])
    sigma=np.std(fid,axis=0,ddof=1);checks=[]
    for row in samples:
        if row['realization']!=0:continue
        coarse,_=read(row);fine,_=read(row,512);geometry(fine)
        shift=10*np.abs(fine['P0']-coarse['P0'])/sigma
        checks.append({'cosmology':row['cosmology'],'max_shift_over_mean100_sigma':float(shift.max())})
    passed=max(r['max_shift_over_mean100_sigma'] for r in checks)<.1
    report={'status':'PASS' if passed else 'FAIL','P_bins':12,'pilot_samples':len(samples),'fine_samples':3,
            'same_edges_and_point_count_as_realspace':True,'exact_lattice_nmodes_verified':True,
            'P_window_source':'actual shared production binner labels, including CPU float32 boundary convention',
            'measurement_mesh_convergence':checks,'truth_recovery_gate':False,
            'P_B_same_raw_sample':True,'numeric_source_sha256':digest(Path(__file__))}
    save_json(OUT/'PILOT_AUDIT.json',report);print(json.dumps(report,indent=2),flush=True)
    if not passed:raise ValueError('P numerical gate failed; stop before bulk/MCMC')

def predicted(values,coef,noise):
    # 唯一生产实现来自原 B-only sampler.Case.residual；这里只用于冻结时独立核验。
    values=np.atleast_2d(values);f,b,b2,bk=values[:,:4].T
    cb=np.polynomial.chebyshev.chebvander((b-2.25)/1.75,3)
    nu=np.column_stack((np.ones(len(b)),b2,bk));ff=np.column_stack((np.ones(len(b)),f,f*f))
    features=(cb[:,:,None,None]*nu[:,None,:,None]*ff[:,None,None,:]).reshape(len(b),-1)
    return features@coef.reshape(-1,coef.shape[-1])+values[:,4:]@noise

def aggregate():
    value=parent()
    if json.loads((OUT/'PILOT_AUDIT.json').read_text())['status']!='PASS':raise ValueError('pilot not accepted')
    names=json.loads((B_OUT/'contract.json').read_text())['baseline_case_names']
    if (OUT/'CASES.json').exists():
        if json.loads((OUT/'CASES.json').read_text())!=names:raise ValueError('case manifest drift')
        for name in names:
            case=OUT/'cases'/name;meta=json.loads((case/'contract.json').read_text())
            if digest(case/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('joint likelihood drift')
        print('REUSE 18 frozen P+B cases',flush=True);return
    rows=[];metadata=[];hashes=[]
    with np.load(LEGACY_OUT/'B000_ENSEMBLE_MATRIX.npz') as z:
        b=np.array(z['B']);bmeta=[json.loads(str(m)) for m in z['metadata_json']]
        b_indices=np.array(z['certified_indices']);old_c=np.array(z['covariance_single_box'])
    for row,bm in zip(value['samples'],bmeta):
        data,meta=read(row);geometry(data)
        if (meta['cosmology'],meta['realization'])!=(bm['cosmology'],bm['realization']):raise ValueError('P/B row alignment failed')
        if meta['source_raw_sha256']!=bm['source_raw_sha256'] or meta['random_seed']!=bm['random_seed']:
            raise ValueError('P/B source or seed mismatch')
        rows.append(data['P0']);metadata.append(meta)
        hashes.append(digest(OUT/'measurements_m256'/f"{row['cosmology']}_real{row['realization']:03d}.npz"))
    if len(rows)!=800 or b.shape!=(800,120):raise ValueError('incomplete matched ensemble')
    p=np.array(rows);covmask=np.array([r['role']=='covariance' for r in value['samples']])
    if covmask.sum()!=500 or np.any(covmask[:300]):raise ValueError('covariance/validation overlap')
    joint=np.column_stack((p,b[:,b_indices]));cov=np.cov(joint[covmask],rowvar=False,ddof=1)
    np.testing.assert_allclose(cov[12:,12:],old_c[np.ix_(b_indices,b_indices)],rtol=2e-13)
    corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)));eigen=np.linalg.eigvalsh(corr)
    if not np.isfinite(cov).all() or eigen.min()<=0:raise ValueError('invalid joint covariance; no ridge or block replacement')
    np.savez_compressed(OUT/'P0_ENSEMBLE_MATRIX.npz',P0=p,covariance_single_box=cov,
                       B_indices=b_indices,metadata_json=np.array([json.dumps(m) for m in metadata]),
                       measurement_sha256=np.array(hashes))
    with np.load(OUT/'P0_LATTICE.npz') as z:pcoef=np.array(z['coefficients']);ptemplate=np.array(z['templates'])
    sources={str(PART/'code'/name):digest(PART/'code'/name) for name in
             ('rsd_pb_common.py','run_rsd_pb_inference.py','run_rsd_pb_mcmc.py','run_rsd_b000_mcmc.py')}
    audits=[];rng=np.random.default_rng(20261002)
    for name in names:
        old=B_OUT/'cases'/name;meta=json.loads((old/'contract.json').read_text())
        with np.load(old/'likelihood.npz') as z:a={k:np.array(z[k]) for k in z.files}
        indices=a['indices'];positions=np.array([int(np.flatnonzero(b_indices==i)[0]) for i in indices])
        selected=np.r_[np.arange(12),12+positions];c=cov[np.ix_(selected,selected)]
        mask=np.array([r['role']=='validation' and r['cosmology']==meta['cosmology'] for r in value['samples']])
        if mask.sum()!=100:raise ValueError('wrong target count')
        np.testing.assert_allclose(b[mask][:,indices].mean(axis=0),a['target'],rtol=1e-13)
        np.testing.assert_allclose(c[12:,12:],a['covariance'],rtol=2e-13)
        invn=float(np.mean([1/m['nbar_h3_Mpc3'] for m,r in zip(metadata,mask) if r]))
        data=np.r_[p[mask].mean(axis=0),a['target']]
        coef=np.concatenate((pcoef,a['coefficients']),axis=-1)
        noise=np.zeros((a['noise'].shape[0]+1,len(data)));noise[:-1,12:]=a['noise'];noise[-1,:12]=invn
        bounds=np.vstack((a['bounds'],[-1e6/invn,1e6/invn]));parameters=meta['parameters']+['Pshot_residual_nbar']
        probes=rng.uniform(bounds[:,0],bounds[:,1],size=(20,len(bounds)));joint_pred=predicted(probes,coef,noise)
        np.testing.assert_allclose(joint_pred[:,12:],predicted(probes[:,:-1],a['coefficients'],a['noise']),rtol=2e-12,atol=1e-4)
        direct=np.stack([p_prediction(v[0],v[1],ptemplate)+v[-1]*invn for v in probes])
        np.testing.assert_allclose(joint_pred[:,:12],direct,rtol=3e-12,atol=2e-9)
        alpha=(500-len(data)-2)/499;chol=np.linalg.cholesky(c/alpha)
        case=OUT/'cases'/name;case.mkdir(parents=True,exist_ok=True)
        if (case/'chain.h5').exists():raise ValueError('cannot refreeze a chain')
        np.savez_compressed(case/'likelihood.npz',target=data,covariance=c,chol=chol,coefficients=coef,
                           noise=noise,bounds=bounds,indices=indices,P_indices=np.arange(12))
        meta.update(parameters=parameters,bounds=bounds.tolist(),P_bins=12,data_bins=len(data),kmax_P=.08,
                    P_likelihood=True,P_derived_b1_prior=False,P_B_cross_covariance=True,Hartlap_factor=alpha,
                    covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
                    parent_contract_sha256=digest(OUT/'contract.json'),likelihood_sha256=digest(case/'likelihood.npz'),
                    source_B_likelihood_sha256=digest(old/'likelihood.npz'),source_B_case=str(old),
                    source_B_MAP_sha256=digest(old/'MAP.json'),
                    Pshot_unit_Mpc_h3=invn,Pshot_physical_bounds=[-1e6,1e6],implementation_sources=sources,
                    P_model='exact lattice Kaiser + finite PNG, independent residual Pshot',
                    B_model_unchanged=True,scientifically_certified=False)
        save_json(case/'contract.json',meta)
        audits.append({'name':name,'P_bins':12,'B_bins':len(indices),'joint_bins':len(data),'Hartlap_factor':alpha,
                       'B_target_model_covariance_priors_unchanged':True,'extra_Pshot_parameter':True})
    save_json(OUT/'CASES.json',names)
    save_json(OUT/'JOINT_FREEZE_AUDIT.json',{'status':'PASS','n_cases':18,'n_matched_P_B_samples':800,
              'n_covariance':500,'n_target_per_sign':100,'full_P_B_cross_covariance':True,
              'cross_correlation_max_abs':float(np.abs(corr[:12,12:]).max()),
              'correlation_eigenvalue_min':float(eigen.min()),'covariance_multiplier':1.,'volume_Gpc_h3':1.,
              'covariance_definition':'np.cov(matched Fid500 single-box [P0,B000] vectors, ddof=1); never /100 or /500',
              'P0_ensemble_sha256':digest(OUT/'P0_ENSEMBLE_MATRIX.npz'),'B_ensemble_sha256':digest(LEGACY_OUT/'B000_ENSEMBLE_MATRIX.npz'),
              'old_outputs_modified':False,'cases':audits})
    print('FROZEN 18 P0+B000 cases; full Csingle V1',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['pilot','aggregate'])
    {'pilot':pilot,'aggregate':aggregate}[parser.parse_args().command]()
