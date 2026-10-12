"""模块大纲：固定catalog的no-IR确定性源；b1四次、物理p二次的精确表示。

沿用现有G/L/Q与[1,b2,bK2]约定，只作数值表示。p轴不是新的自由参数，
必须在校准后显式代入；不定义先验，不适用于IR指数的b1依赖。
"""
import numpy as np

B_CENTER, B_RADIUS = 2.25, 1.75
P_CENTER, P_RADIUS = 1.3, .5
B_NODES = B_CENTER + B_RADIUS * np.cos((np.arange(5) + .5) * np.pi / 5)
P_NODES = P_CENTER + P_RADIUS * np.cos((np.arange(3) + .5) * np.pi / 3)
B_NODES.setflags(write=False)
P_NODES.setflags(write=False)


def _finite_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(name + ' must be a finite scalar')
    return float(value)


def _tensor(coefficients):
    value = np.asarray(coefficients, dtype=float)
    if value.ndim < 4 or value.shape[:4] != (5, 3, 3, 3) or not np.isfinite(value).all():
        raise ValueError('finite [b1:5,p:3,nuisance:3,fNL:3,...] tensor required')
    return value


def fit_noir_design(node_designs):
    """输入两个节点轴之后为[1,b2,bK2]、[G,L,Q]、可选观测量轴。"""
    values = _tensor(node_designs)
    bb, pp = np.meshgrid((B_NODES - B_CENTER) / B_RADIUS,
                         (P_NODES - P_CENTER) / P_RADIUS, indexing='ij')
    vander = np.polynomial.chebyshev.chebvander2d(bb.ravel(), pp.ravel(), (4, 2))
    return np.linalg.solve(vander, values.reshape(15, -1)).reshape(values.shape)


def collapse_physical_p(coefficients, *, p_halo):
    """校准p固定后仅保留b1节点；不允许默认取p_rec或某个数值探针。"""
    p = _finite_scalar(p_halo, 'physical p_halo')
    values = _tensor(coefficients)
    basis = np.polynomial.chebyshev.chebvander((p - P_CENTER) / P_RADIUS, 2).ravel()
    return np.tensordot(values, basis, axes=(1, 0))


def evaluate_noir_design(coefficients, b1, *, p_halo):
    b = _finite_scalar(b1, 'physical b1')
    collapsed = collapse_physical_p(coefficients, p_halo=p_halo)
    return np.polynomial.chebyshev.chebval((b - B_CENTER) / B_RADIUS, collapsed)


def predict_noir(values, p_collapsed_coefficients):
    """批量确定性B；values列为[fNL,b1,b2,bK2]，不包括另行建模的随机项。"""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    c = np.asarray(p_collapsed_coefficients, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('finite [fNL,b1,b2,bK2] rows required')
    if c.ndim < 3 or c.shape[:3] != (5, 3, 3) or not np.isfinite(c).all():
        raise ValueError('p-collapsed [b1:5,nuisance:3,fNL:3,...] coefficients required')
    f, b, b2, bk = values.T
    cb = np.polynomial.chebyshev.chebvander((b - B_CENTER) / B_RADIUS, 4)
    nuisance = np.column_stack((np.ones(len(b)), b2, bk))
    powers = np.column_stack((np.ones(len(b)), f, f * f))
    features = (cb[:, :, None, None] * nuisance[:, None, :, None] * powers[:, None, None, :]).reshape(len(b), 45)
    return features @ c.reshape(45, -1)
