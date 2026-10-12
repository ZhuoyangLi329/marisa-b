"""模块大纲：post no-IR Case适配；原MCMC提案/RNG/HDF/诊断唯一实现。

只读已经冻结的新post似然；物理p必须来自校准，不默认p_rec或旧pre p。
五节点B不能调用pre三次残差；IR也不能借用此精确四次表示。
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest
from post_rsd_noir_likelihood import PostNoIRLikelihood,predict_p_rows
import run_rsd_b000_mcmc as shared
from run_rsd_p0only import POnlyCase as OriginalPOnlyCase

OUT=PART/'outputs/sto_marisa_post_rsd_mean100_v1/mcmc_noir'


def load(directory):
    directory=Path(directory);meta=json.loads((directory/'contract.json').read_text())
    parent=json.loads((OUT/'CONTRACT.json').read_text())
    if meta['parent_contract_sha256']!=digest(OUT/'CONTRACT.json'):
        raise ValueError('new post parent contract drift')
    if parent['status']!='POST_NOIR_INPUTS_VALIDATED_FOR_MCMC':
        raise ValueError('post data/covariance/calibration/model gates not released')
    for path,want in parent['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('post frozen input drift: '+path)
    if (meta['IR'] or meta['n_covariance']!=500 or meta['n_target']!=100
        or meta['volume_Gpc_h3']!=1 or meta['covariance_multiplier']!=1 or meta['covariance_is_of_mean']
        or meta['P_derived_b1_prior']):raise ValueError('not the requested no-IR mean100/C_single V1 contract')
    if not np.isscalar(meta['p_fixed_physical']) or not np.isfinite(meta['p_fixed_physical']):
        raise ValueError('explicit calibrated physical p required')
    if digest(directory/'likelihood.npz')!=meta['likelihood_sha256']:raise ValueError('likelihood hash drift')
    with np.load(directory/'likelihood.npz',allow_pickle=False) as z:arrays={k:np.array(z[k]) for k in z.files}
    n=meta['P_bins']+meta['B_bins'];alpha=(500-n-2)/499
    if meta['Hartlap_factor']!=alpha:raise ValueError('Hartlap must use own marginal/joint dimension')
    np.testing.assert_allclose(arrays['chol']@arrays['chol'].T,arrays['covariance']/alpha,rtol=3e-13,atol=1e-8)
    if arrays['bounds'].shape!=(len(meta['parameters']),2):raise ValueError('parameter/bounds mismatch')
    np.testing.assert_array_equal(arrays['bounds'],meta['bounds'])
    return directory,meta,arrays


class PostBCase(shared.Case):
    def __init__(self,directory):
        self.directory,self.meta,arrays=load(directory)
        for name,value in arrays.items():setattr(self,name,value)
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names[:4]!=['fNL','b1','b2_native','bK2'] or not self.meta['B_bins']:
            raise ValueError('post B physical parameter order changed')
        if self.meta['P_bins'] and not self.meta['P_B_cross_covariance']:
            raise ValueError('P+B requires full empirical same-box cross covariance')
        if self.meta['noise_basis_status']!='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED':
            raise ValueError('must not silently reuse unaudited legacy stochastic bases')
        self.model=PostNoIRLikelihood(self.target,self.chol,self.coefficients,self.templates,self.noise,
            self.meta['p_fixed_physical'],noise_coefficients=getattr(self,'noise_coefficients',None))
        if self.model.ndim!=self.ndim:raise ValueError('noise dimension must match prior parameters')

    def residual(self,physical):return self.model.residual(physical)


class PostPOnlyCase(OriginalPOnlyCase):
    def __init__(self,directory):
        self.directory,self.meta,arrays=load(directory)
        for name,value in arrays.items():setattr(self,name,value)
        self.names=self.meta['parameters'];self.ndim=len(self.names)
        if self.names!=['fNL','b1','Pshot_residual_nbar'] or self.meta['P_bins']!=12 or self.meta['B_bins']:
            raise ValueError('original P-only parameter/bin contract required')
        self.white_templates=np.linalg.solve(self.chol,self.templates.T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise);self.twhite=np.linalg.solve(self.chol,self.target)

    def predictions(self,physical):
        rows=np.atleast_2d(physical)
        return predict_p_rows(rows[:,:2],self.templates,p_halo=self.meta['p_fixed_physical'])+rows[:,2,None]*self.noise

    def residual(self,physical):
        rows=np.atleast_2d(physical)
        return predict_p_rows(rows[:,:2],self.white_templates,p_halo=self.meta['p_fixed_physical'])+rows[:,2,None]*self.nwhite-self.twhite


def factory(directory):
    meta=json.loads((Path(directory)/'contract.json').read_text())
    return PostPOnlyCase(directory) if meta['case_type']=='P0_only' else PostBCase(directory)


def run(index,max_steps=64000,extend_budget=False):
    if len(os.sched_getaffinity(0))!=1:raise ValueError('one logical CPU per MCMC required')
    original=shared.OUT,shared.Case,shared.budget_provenance,shared.save_json
    def provenance(directory,value,extend=False):
        value.update(post_adapter_sha256=digest(Path(__file__)),
            post_likelihood_sha256=digest(PART/'code/post_rsd_noir_likelihood.py'),
            post_polynomial_sha256=digest(PART/'code/post_rsd_noir_polynomial.py'))
        return original[2](directory,value,extend)
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            meta=json.loads((Path(path).parent/'contract.json').read_text())
            value.update(data_stage='STO+Marisa post-recon RSD',P_likelihood=bool(meta['P_bins']),
                B_likelihood=bool(meta['B_bins']),P_B_cross_covariance=meta['P_B_cross_covariance'],
                covariance_is_of_mean=False,covariance_multiplier=1.,p_fixed_physical=meta['p_fixed_physical'],
                IR=False,physical_p_is_not_p_rec=True,
                gof_note='descriptive only: mean100 target with unaveraged single-box forecast covariance')
        return original[3](path,value)
    shared.OUT,shared.Case,shared.budget_provenance,shared.save_json=OUT,factory,provenance,save
    try:return shared.run(index,max_steps,extend_budget)
    finally:shared.OUT,shared.Case,shared.budget_provenance,shared.save_json=original


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case-index',type=int,required=True)
    ap.add_argument('--max-steps',type=int,default=64000);ap.add_argument('--extend-budget',action='store_true')
    args=ap.parse_args();run(args.case_index,args.max_steps,args.extend_budget)
