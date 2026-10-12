"""模块大纲：GSR2 weighted-input的物理p校准；Fid500 C_single；原三参数先验。

fNL固定为校准真值100；复用现有RSD P0源，不把p_rec作为物理p或新增主MCMC参数。
holdout150只提供校准均值；主mean100和covariance合同保持各自原统计角色。
"""
from dataclasses import dataclass
import numpy as np
from post_rsd_ensemble import marginal_likelihood_arrays
from post_rsd_p0_model import post_p0_linear

PARAMETERS = ('physical_p', 'b1', 'Pshot_residual')
# 与Part2 fnlfixed100_freep calibration相同的三参数范围；体积采用当前V=1。
BOUNDS = np.array([[0., 3.], [.1, 6.], [-1.e5, 1.e5]])
BOUNDS.setflags(write=False)


def make_calibration_arrays(ensemble, holdout_P0, *, holdout_ids, block='P_weighted_input'):
    ids = tuple(holdout_ids)
    data = np.asarray(holdout_P0, dtype=float)
    if ids != tuple(range(100, 250)) or data.shape != (150, 12) or not np.isfinite(data).all():
        raise ValueError('complete validated LC_p100..249 weighted-input P0 required')
    start, stop = ensemble['blocks'][block]
    if stop-start != 12: raise ValueError('original twelve P0 bins required')
    # 唯一统计实现给出边缘C子块与各自Hartlap；校准target另取已验收的holdout均值。
    arrays = marginal_likelihood_arrays(ensemble, tuple(range(start, stop)), 'LC_p')
    arrays['target'] = data.mean(axis=0)
    meta = ensemble['statistics']
    arrays['metadata'] = {'calibration_fNL_fixed': 100., 'calibration_target_count': 150,
        'calibration_ids': list(ids), 'main_target_count': meta['n_target_per_truth'],
        'n_covariance': 500, 'volume_Gpc_h3': 1., 'covariance_multiplier': 1.,
        'covariance_is_of_mean': False, 'covariance_tracer_block': block,
        'main_target_or_covariance_replaced': False,
        'calibration_ids_disjoint_from_main_target_ids': not bool(set(ids)&set(meta['target_ids'])),
        'potential_shared_phase_ids_with_covariance': sorted(set(ids)&set(meta['covariance_ids'])),
        'phase_independence_verified_here': False,
        'physical_p_is_not_algorithm_p_rec': True, 'new_main_MCMC_parameter_added': False}
    return arrays


@dataclass
class PhysicalPCalibrationCase:
    target: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    templates: np.ndarray

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.cholesky = np.asarray(self.cholesky, dtype=float)
        self.templates = np.asarray(self.templates, dtype=float)
        if (self.target.shape != (12,) or self.covariance.shape != (12, 12)
                or self.cholesky.shape != (12, 12) or self.templates.shape != (6, 12)
                or not all(np.isfinite(a).all() for a in (self.target,self.covariance,self.cholesky,self.templates))):
            raise ValueError('twelve P0 bins, full covariance and six finite actual-window templates required')
        np.testing.assert_allclose(self.cholesky@self.cholesky.T, self.covariance/((500-12-2)/499.), rtol=3e-13, atol=1e-10)

    def predictions(self, values):
        values = np.asarray(values, dtype=float); scalar = values.ndim == 1
        rows = np.atleast_2d(values)
        if rows.ndim != 2 or rows.shape[1] != 3 or not np.isfinite(rows).all():
            raise ValueError('finite [physical_p,b1,Pshot] rows required')
        result = np.array([post_p0_linear(100., b, self.templates, p_halo=p)+shot for p,b,shot in rows])
        return result[0] if scalar else result

    def objective_many(self, values):
        predicted = np.atleast_2d(self.predictions(values))
        white = np.linalg.solve(self.cholesky, (predicted-self.target).T).T
        return np.sum(white*white, axis=1)

    def objective(self, values): return float(self.objective_many(values)[0])

    def log_probability(self, values):
        values = np.asarray(values, dtype=float); scalar = values.ndim == 1
        rows = np.atleast_2d(values)
        if rows.ndim != 2 or rows.shape[1] != 3: raise ValueError('three physical parameters required')
        inside = np.isfinite(rows).all(axis=1)&np.all(rows>BOUNDS[:,0],axis=1)&np.all(rows<BOUNDS[:,1],axis=1)
        result = np.full(len(rows), -np.inf)
        if inside.any(): result[inside] = -.5*self.objective_many(rows[inside])
        return float(result[0]) if scalar else result
