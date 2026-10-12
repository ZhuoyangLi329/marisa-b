"""Inspect the frozen anisotropic R response, without reading active HDF files.

The centered mu^4 term has zero rotational monopole in the continuum.
This audit quantifies its remaining actual lattice-window response using
the unchanged full joint covariance, not a marginalized/Fisher constraint.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import run_recalibrated_prepost_noise_comparison_v1 as run
from rsd_b000_contract import digest, save_json


def main():
    run.guard()
    mu, weights = np.polynomial.legendre.leggauss(32)
    continuum = float(weights @ (mu**4 - .2) / 2)
    assert abs(continuum) < 1e-14
    rows=[]
    for index in range(24,48):
        case=run.ComparisonCase(run.directory(index))
        response=np.r_[np.zeros(12),case.model.anisotropic_shape]
        white=np.linalg.solve(case.chol,response)
        whitened_prior_half_width_norm=float(np.linalg.norm(white)*10)
        reference=case.weight_moments[0]*case.model.f_growth**2*case.primitives[2]
        relative=float(np.linalg.norm(response[12:])/np.linalg.norm(reference))
        row=case.bounds.mean(axis=1);row[0]=case.meta['truth_fNL'];row[1]=2.2
        left=row.copy();right=row.copy();left[7]=-10;right[7]=10
        delta=(case.model.predictions(right)-case.model.predictions(left))[0]
        np.testing.assert_allclose(delta,20*response,rtol=1e-9,atol=1e-4)
        rows.append(dict(name=case.meta['name'],stage=case.meta['stage'],kmax_B=case.meta['kmax_B'],
            truth_fNL=case.meta['truth_fNL'],R_prior=[-10,10],
            full_joint_white_R10_norm=whitened_prior_half_width_norm,
            full_joint_white_prior_endpoint_separation_norm=2*whitened_prior_half_width_norm,
            centered_to_uncentered_mu4_shape_norm_ratio=relative,
            actual_likelihood_sha256=digest(case.directory/'likelihood.npz')))
    target=run.calibration.OUT/'ANISOTROPIC_R_INFORMATION_AUDIT.json'
    if target.exists():raise ValueError('Preserve previous diagnostic; do not overwrite')
    result=dict(created_UTC=run.stamp(),status='FROZEN_R_B000_INFORMATION_DIAGNOSTIC_NOT_MODEL_CHANGE',
        continuum_isotropic_mean_mu4_minus_one_fifth=continuum,
        continuum_R_B000_projection_zero=True,
        checks=rows,active_HDF_read=False,model_prior_or_covariance_changed=False,
        interpretation='Centered angular R is invisible to the ideal rotational monopole; only its discrete-window leakage enters these B000 fits. A near-uniform R posterior is not evidence of a tightly measured anisotropic stochastic response.',
        likelihood_norm_is_conditional_response_not_marginalized_constraint=True,
        original_products_preserved=True,auditor_sha256=digest(__file__))
    save_json(target,result)
    print(json.dumps(dict(status=result['status'],max_R10_white_norm=max(x['full_joint_white_R10_norm'] for x in rows),
        min_R10_white_norm=min(x['full_joint_white_R10_norm'] for x in rows))),flush=True)


if __name__=='__main__':main()
