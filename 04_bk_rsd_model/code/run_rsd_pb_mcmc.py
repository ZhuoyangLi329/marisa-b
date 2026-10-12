"""模块大纲：显式联合 Case 适配；复用原采样/诊断唯一流程；记录联合来源。

不编辑 B-only sampler，不修改提案/RNG/HDF/诊断门；只更换输入、合同守卫和输出标签。
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rsd_pb_common import OUT,PART,digest
import run_rsd_b000_mcmc as shared

class JointCase(shared.Case):
    def __init__(self,directory):
        self.directory=directory;self.meta=json.loads((directory/'contract.json').read_text())
        if not self.meta['P_likelihood'] or self.meta['P_derived_b1_prior'] or not self.meta['P_B_cross_covariance']:
            raise ValueError('not an explicit full P+B likelihood')
        if self.meta['volume_Gpc_h3']!=1 or self.meta['covariance_multiplier']!=1 or self.meta['covariance_is_of_mean']:
            raise ValueError('wrong covariance normalization')
        if self.meta['parent_contract_sha256']!=digest(OUT/'contract.json'):raise ValueError('parent contract drift')
        for path,sha in self.meta['implementation_sources'].items():
            if digest(path)!=sha:raise ValueError('joint implementation drift: '+path)
        if digest(directory/'likelihood.npz')!=self.meta['likelihood_sha256']:raise ValueError('likelihood hash mismatch')
        with np.load(directory/'likelihood.npz') as z:
            for name in z.files:setattr(self,name,np.array(z[name]))
        self.names=self.meta['parameters'];self.ndim=len(self.names);n=self.meta['P_bins']+self.meta['B_bins']
        if self.target.shape!=(n,) or self.noise.shape!=(self.ndim-4,n) or self.coefficients.shape!=(4,3,3,n):
            raise ValueError('joint dimensions inconsistent')
        alpha=(self.meta['n_covariance']-n-2)/(self.meta['n_covariance']-1)
        np.testing.assert_allclose(self.chol@self.chol.T,self.covariance/alpha,rtol=3e-13,atol=1e-3)
        if self.meta['Hartlap_factor']!=alpha:raise ValueError('Hartlap must use the joint data dimension')
        self.cwhite=np.linalg.solve(self.chol,self.coefficients.reshape(-1,n).T).T
        self.nwhite=np.linalg.solve(self.chol,self.noise.T).T;self.twhite=np.linalg.solve(self.chol,self.target)

def run(index,max_steps=64000,extend_budget=False):
    original=(shared.OUT,shared.Case,shared.budget_provenance,shared.save_json)
    def provenance(directory,value,extend=False):
        value.update(joint_adapter_sha256=digest(Path(__file__)),joint_common_sha256=digest(PART/'code/rsd_pb_common.py'),
                     joint_inference_sha256=digest(PART/'code/run_rsd_pb_inference.py'))
        return original[2](directory,value,extend)
    def save(path,value):
        if Path(path).name=='SUMMARY.json':
            meta=json.loads((Path(path).parent/'contract.json').read_text())
            value.update(P_likelihood=True,P_derived_b1_prior=False,P_B_cross_covariance=True,P_bins=meta['P_bins'],
                         B_bins=meta['B_bins'],kmax_P=.08,covariance_multiplier=1.,covariance_is_of_mean=False,
                         gof_note='descriptive only: mean100 target with unaveraged single-box covariance; not the sampling likelihood of the mean',
                         joint_adapter_sha256=digest(Path(__file__)))
        return original[3](path,value)
    shared.OUT=OUT;shared.Case=JointCase;shared.budget_provenance=provenance;shared.save_json=save
    try:return shared.run(index,max_steps,extend_budget)
    finally:shared.OUT,shared.Case,shared.budget_provenance,shared.save_json=original

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case-index',type=int,required=True)
    parser.add_argument('--max-steps',type=int,default=64000);parser.add_argument('--extend-budget',action='store_true')
    args=parser.parse_args();run(args.case_index,args.max_steps,args.extend_budget)
