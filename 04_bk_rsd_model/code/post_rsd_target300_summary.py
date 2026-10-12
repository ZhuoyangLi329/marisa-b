"""模块大纲：完整三真值mean100的独立数据汇总；不构造任何协方差。

要求恰好300个目标和原P_input12/P_post12/B44几何，保留两种B定义。
N2/N3是实际粒子self矩，不是STO raw随机功率；mean(N2²)不能写成mean(N2)²。
此数据检查既不决定正式估计器，也不替代全700/Fid500/MCMC门。
"""
from numbers import Integral
import numpy as np

TRUTHS = ('fiducial', 'LC_p', 'LC_m')
TARGET_KEYS = tuple((c, i) for c in TRUTHS for i in range(100))


def summarize_complete_targets(keys, matched, nonzero, moments):
    identities = tuple(tuple(k) for k in keys)
    if any(len(k) != 2 or isinstance(k[1], bool) or not isinstance(k[1], Integral) for k in identities):
        raise ValueError('explicit integer realization identities required')
    if identities != TARGET_KEYS:
        raise ValueError('exact original three truths, each0..99, in canonical target order required')
    a = np.asarray(matched, dtype=float); b = np.asarray(nonzero, dtype=float)
    m = np.asarray(moments, dtype=float)
    if a.shape != (300, 68) or b.shape != a.shape or m.shape != (300, 3):
        raise ValueError('all300 original joint68 vectors and actual N2/N3/nhalo required')
    if not all(np.isfinite(v).all() for v in (a, b, m)):
        raise ValueError('finite actual target measurements required')
    if np.any(m[:, 0] <= 0) or np.any(m[:, 2] <= 0) or np.any(m[:, 2] != np.rint(m[:, 2])):
        raise ValueError('positive self N2 and actual integer halo counts required; signed N3 allowed')
    np.testing.assert_array_equal(a[:, :24], b[:, :24])
    arrays = dict(matched_target_vectors=np.array([a[i*100:(i+1)*100].mean(axis=0) for i in range(3)]),
                  nonzero_diagnostic_target_vectors=np.array([b[i*100:(i+1)*100].mean(axis=0) for i in range(3)]),
                  data_N2_N3_nhalo=m.copy())
    statistics = []
    for i, cosmology in enumerate(TRUTHS):
        values = m[i*100:(i+1)*100]; n2 = float(values[:, 0].mean())
        n2_square = float(np.mean(values[:, 0]**2)); n3 = float(values[:, 1].mean())
        if not np.isfinite([n2, n2_square, n3]).all():
            raise ValueError('weighted self moments overflowed; do not clip')
        statistics.append(dict(cosmology=cosmology, count=100, mean_N2=n2,
            mean_N2_squared=n2_square, square_mean_N2=n2*n2, mean_N3=n3,
            relative_difference_mean_square_vs_square_mean=n2_square/(n2*n2)-1.,
            minimum_N3=float(values[:, 1].min()), maximum_N3=float(values[:, 1].max()),
            self_moments_are_not_raw_STO_stochastic_power=True))
    return arrays, statistics
