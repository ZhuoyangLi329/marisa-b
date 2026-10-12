"""模块大纲：原39组no-IR病例表；同盒P/B边缘C；显式校准p和审计后的噪声。

只组装已通过上游数值/估计器/噪声门的数组，不创建假审计或默认物理p。
Pshot沿用原RSD的物理[-1e6,1e6]先验；数据mean100，C_single500从不平均。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_ensemble import TRUTHS,marginal_likelihood_arrays
from post_rsd_noir_polynomial import collapse_physical_p

CUTS=(.08,.10,.12,.14,.18,.20)
COUNTS=(9,14,20,27,35,44)
B_NAMES=('fNL','b1','b2_native','bK2','A_stoch','S_stoch','R_stoch_anisotropic')
B_BOUNDS=np.array([[-500,500],[.5,4],[-4,6],[-4,4],[-10,10],[-20,20],[-10,10]],dtype=float)
P_NAMES=('fNL','b1','Pshot_residual_nbar')


@dataclass(frozen=True)
class CaseSpec:
    cosmology:str
    truth_fNL:int
    observable:str
    kmax_B:float|None
    P_block:str='P_post'
    role:str='main'

    @property
    def name(self):
        cut='none' if self.kmax_B is None else f'{self.kmax_B:.2f}'
        return f'{self.cosmology}_{self.role}_{self.observable}_k{cut}'


def case_specs(*,include_weighted_input_P_control=False):
    cases=[]
    for observable in ('B_only','P0_only','P_plus_B'):
        for cosmo,truth in TRUTHS:
            for cut in ((None,) if observable=='P0_only' else CUTS):
                cases.append(CaseSpec(cosmo,truth,observable,cut))
    if include_weighted_input_P_control:
        cases.extend(CaseSpec(c,t,'P_plus_B',cut,'P_weighted_input','same_tracer_pre_P_control')
                     for c,t in TRUTHS for cut in CUTS)
    return tuple(cases)


def assemble_case(ensemble,spec,B_k,paware_coefficients,P_templates,B_noise,P_noise,*,p_halo,noise_audit,
                  B_noise_coefficients=None):
    if not isinstance(spec,CaseSpec) or spec not in case_specs(include_weighted_input_P_control=True):
        raise ValueError('case not in original39 or explicitly labeled same-tracer P-input control')
    if (noise_audit.get('status')!='POST_WEIGHTED_EFFECTIVE_BASIS_AUDITED'
        or noise_audit.get('p_fixed_physical')!=p_halo
        or noise_audit.get('B_definition') not in ('matched','nonzero_diagnostic')
        or noise_audit.get('original_binning_preserved') is not True):
        raise ValueError('upstream post-weighted estimator/noise audit with same calibrated physical p required')
    if not np.isscalar(p_halo) or not np.isfinite(p_halo):raise ValueError('explicit calibrated physical p required')
    triangles=np.asarray(B_k,dtype=float);coeff=np.asarray(paware_coefficients,dtype=float)
    p=np.asarray(P_templates,dtype=float);bnoise=np.asarray(B_noise,dtype=float);pnoise=np.asarray(P_noise,dtype=float)
    if (triangles.shape!=(44,2) or coeff.shape!=(5,3,3,3,44) or p.shape!=(6,12)
        or bnoise.shape!=(3,44) or pnoise.shape!=(12,)
        or not all(np.isfinite(a).all() for a in (triangles,coeff,p,bnoise,pnoise))):
        raise ValueError('original44 B bins/five b1 nodes/three p nodes/three noise shapes/twelve P bins required')
    if np.any(pnoise<=0) or not np.all(pnoise==pnoise[0]):
        raise ValueError('audited positive constant physical Pshot unit required')
    blocks=ensemble['blocks']
    if (blocks['B_post'][1]-blocks['B_post'][0]!=44
        or blocks[spec.P_block][1]-blocks[spec.P_block][0]!=12):raise ValueError('observable block dimension drift')
    bstart=blocks['B_post'][0];pstart,pstop=blocks[spec.P_block]
    if spec.kmax_B is None:mask=np.zeros(44,dtype=bool)
    else:
        mask=np.max(triangles,axis=1)<=spec.kmax_B+1e-12
        if int(mask.sum())!=COUNTS[CUTS.index(spec.kmax_B)]:raise ValueError('original six-cut B density changed')
    nB=int(mask.sum());nP=0 if spec.observable=='B_only' else 12
    indices=tuple(range(pstart,pstop)) if nP else ()
    indices+=tuple((bstart+np.flatnonzero(mask)).tolist())
    arrays=marginal_likelihood_arrays(ensemble,indices,spec.cosmology)
    collapsed=collapse_physical_p(coeff,p_halo=p_halo)
    arrays.update(coefficients=collapsed[...,mask],templates=p if nP else np.empty((6,0)))
    shot_bound=np.array([-1e6,1e6])/pnoise[0]
    if not nB:
        names=P_NAMES;bounds=np.vstack((B_BOUNDS[:2],shot_bound));noise=pnoise.copy()
    else:
        names=B_NAMES;bounds=B_BOUNDS.copy();noise=np.zeros((3,nP+nB));noise[:,nP:]=bnoise[:,mask]
        if nP:
            names+=('Pshot_residual_nbar',);bounds=np.vstack((bounds,shot_bound))
            noise=np.vstack((noise,np.concatenate((pnoise,np.zeros(nB)))))
    arrays.update(noise=noise,bounds=bounds,indices=np.array(indices,dtype=int))
    if B_noise_coefficients is not None:
        nc=np.asarray(B_noise_coefficients,dtype=float)
        if nc.shape!=(3,5,3,3,44) or not np.isfinite(nc).all():
            raise ValueError('post noise coefficient tensor must preserve three original nuisance parameters')
        if nB:
            joint=np.zeros((len(noise),5,3,3,nP+nB));joint[:3,...,nP:]=nc[...,mask]
            arrays['noise_coefficients']=joint
    metadata=dict(name=spec.name,cosmology=spec.cosmology,truth_fNL=spec.truth_fNL,
        case_type=spec.observable,case_role=spec.role,P_block=spec.P_block,
        kmax_B=spec.kmax_B,kmax_P=.08 if nP else None,P_bins=nP,B_bins=nB,
        parameters=list(names),bounds=bounds.tolist(),n_covariance=500,n_target=100,
        volume_Gpc_h3=1.,covariance_multiplier=1.,covariance_is_of_mean=False,
        Hartlap_factor=arrays['Hartlap_factor'],P_B_cross_covariance=bool(nP and nB),
        P_likelihood=bool(nP),P_derived_b1_prior=False,p_fixed_physical=float(p_halo),IR=False,
        B_definition=noise_audit['B_definition'],noise_basis_status=noise_audit['status'],
        noise_representation='polynomial_plus_constant' if nB and B_noise_coefficients is not None else 'constant_basis',
        Pshot_physical_bounds=[-1e6,1e6] if nP else None,
        Pshot_unit_Mpc_h3=float(pnoise[0]) if nP else None,
        zero_atom_policy='explicitly inherited from audited B definition; never silently changed',
        strict_independent_mock_recovery=False,physical_p_is_not_p_rec=True)
    return arrays,metadata
