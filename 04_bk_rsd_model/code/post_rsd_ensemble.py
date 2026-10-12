"""模块大纲：显式样本角色；同盒多观测量组装；未平均的Fid500协方差。

本模块只负责统计数组，不选择/授权生产样本，不读取或覆盖任何历史结果。
文件来源、权重身份和estimator合同必须由上游测量读取器验收。支持原实空间
重叠方案，但调用者必须显式指定允许重叠；不能把它标成独立验证。
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


TRUTHS = (('fiducial', 0), ('LC_p', 100), ('LC_m', -100))


def _ids(values, count, label):
    values = tuple(values)
    if (len(values) != count or len(set(values)) != count
            or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values)):
        raise ValueError(f'{label} requires {count} unique nonnegative integer IDs')
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class EnsembleSelection:
    covariance_ids: tuple
    target_ids: tuple = tuple(range(100))
    overlap_policy: str = 'disjoint'

    def __post_init__(self):
        object.__setattr__(self, 'covariance_ids', _ids(self.covariance_ids, 500, 'covariance'))
        object.__setattr__(self, 'target_ids', _ids(self.target_ids, 100, 'target'))
        if self.overlap_policy not in ('disjoint', 'allow_explicit_target_overlap'):
            raise ValueError('unknown sample overlap policy')
        if self.overlap and self.overlap_policy == 'disjoint':
            raise ValueError('target/covariance overlap requires an explicit policy')

    @property
    def overlap(self):
        return tuple(sorted(set(self.covariance_ids) & set(self.target_ids)))

    @property
    def measurement_keys(self):
        # 一个实体catalog可以承担两个统计角色，但只对应一行测量。
        candidates = [('fiducial', i) for i in self.covariance_ids]
        candidates += [(cosmo, i) for cosmo, _ in TRUTHS for i in self.target_ids]
        return tuple(dict.fromkeys(candidates))

    def metadata(self):
        return {'n_covariance': 500, 'n_target_per_truth': 100,
                'unique_measurements': len(self.measurement_keys),
                'covariance_ids': list(self.covariance_ids), 'target_ids': list(self.target_ids),
                'overlap_policy': self.overlap_policy,
                'fiducial_target_covariance_exact_overlap': list(self.overlap),
                'PNG_target_covariance_possible_shared_phase_ids': list(self.overlap),
                'validation_catalogs_disjoint_from_covariance': not bool(self.overlap),
                'phase_independence_verified_by_this_module': False,
                'volume_Gpc_h3': 1., 'covariance_multiplier': 1.,
                'covariance_is_of_mean': False, 'covariance_ddof': 1}


def assemble_blocks(selection, rows, block_sizes):
    """组装已验收的同盒观测量，block顺序显式；无缺行回退、缩放或协方差正则化。

    rows[(cosmology, realization)][block]应来自同盒经上游来源验收的测量。
    block可包括P_post、B_post、P_weighted_input或匹配的新pre控制。
    返回full Csingle；P-only/B-only应抽协方差子块后分别做Hartlap修正。
    """
    if not isinstance(selection, EnsembleSelection):
        raise TypeError('explicit EnsembleSelection required')
    if not block_sizes or any(not isinstance(k, str) or not k for k in block_sizes):
        raise ValueError('named observable blocks required')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in block_sizes.values()):
        raise ValueError('positive integer block sizes required')
    keys = selection.measurement_keys
    if set(rows) != set(keys):
        raise ValueError('measurement row keys must exactly match the selected ensemble')
    bounds = {}; offset = 0
    for name, size in block_sizes.items():
        bounds[name] = (offset, offset + int(size)); offset += int(size)
    matrix = np.empty((len(keys), offset), dtype=np.float64)
    for index, key in enumerate(keys):
        for name, (first, stop) in bounds.items():
            values = np.asarray(rows[key][name], dtype=float)
            if values.shape != (stop - first,) or not np.isfinite(values).all():
                raise ValueError(f'invalid {name} measurement for {key}')
            matrix[index, first:stop] = values
    positions = {key: index for index, key in enumerate(keys)}
    cov_indices = np.array([positions['fiducial', i] for i in selection.covariance_ids])
    # 与既有P+B实现相同：同盒joint vectors的样本协方差；绝不除以100/500。
    covariance = np.atleast_2d(np.cov(matrix[cov_indices], rowvar=False, ddof=1))
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise ValueError('invalid covariance; no silent diagonal loading')
    targets = {}
    for cosmo, _ in TRUTHS:
        target_indices = [positions[cosmo, i] for i in selection.target_ids]
        targets[cosmo] = matrix[target_indices].mean(axis=0)
    return {'measurement_keys': keys, 'matrix': matrix, 'blocks': bounds,
            'covariance_indices': cov_indices, 'covariance_single_box': covariance,
            'targets': targets, 'statistics': selection.metadata()}


def marginal_likelihood_arrays(ensemble, indices, cosmology):
    """从C的子块构造边缘似然；不能截取full precision当作P-only/B-only。"""
    meta = ensemble['statistics']
    if (meta['n_covariance'] != 500 or meta['covariance_is_of_mean']
            or meta['covariance_multiplier'] != 1. or meta['volume_Gpc_h3'] != 1.):
        raise ValueError('single-box Fid500 V1 contract required')
    indices = tuple(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)):
        raise ValueError('unique integer observable indices required')
    full = ensemble['covariance_single_box']
    if min(indices) < 0 or max(indices) >= len(full):
        raise ValueError('observable index out of range')
    covariance = full[np.ix_(indices, indices)]
    alpha = (500 - len(indices) - 2) / 499.
    if alpha <= 0:
        raise ValueError('nonpositive Hartlap factor')
    # Cholesky失败直接暴露问题；不放行奇异矩阵或隐式ridge。
    chol = np.linalg.cholesky(covariance / alpha)
    return {'target': ensemble['targets'][cosmology][list(indices)].copy(),
            'covariance': covariance.copy(), 'chol': chol, 'Hartlap_factor': alpha}
