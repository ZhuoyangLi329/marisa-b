"""模块大纲：当前EFT12候选的八噪声精确响应，仅用于独立可辨识性诊断。

不修改预测/先验/协方差，不生成新的likelihood，不由局部rank删除参数。
SVD只显示当前白化response的方向与强弱；没有pinv、C正则化或采样器。
"""
import numpy as np

NOISE_NAMES = ('a_delta', 'a_phi', 's_factorial', 'Nraw_over_N2',
               'b1_w2', 'bphi_w2', 'J21_over_N3', 'Pshot_residual_nbar')


def noise_response(model, row):
    """解析d(P,B)/d八noise参数；精确包含mean(N2²)、mean(N2*Nr)。"""
    row = np.asarray(row, dtype=float)
    if row.shape != (12,) or not np.isfinite(row).all() or row[7] < 0:
        raise ValueError('explicit finite EFT12 physical row, nonnegative raw ratio required')
    fnl, b1 = row[:2]
    f = model.f_growth
    bphi = 2 * 1.686 * (b1 - model.p_halo)
    n2, n3, n2sq, nr, n2nr = model.weight_moments
    p, pmu, pmu4, pM, pmuM, pM2, valid = model.intrinsic_primitives
    zero = model.zero_primitives
    cb = np.array([1., (b1 - 2.25) / 1.75, 2 * ((b1 - 2.25) / 1.75)**2 - 1])
    powers = np.array([1., fnl, fnl**2])
    operators = np.einsum('i,iafb,f->ab', cb, model.operators, powers)
    jac = np.zeros((12 + model.nB, 8))
    jac[12:, 0] = b1 * n2 * (b1 * p + f * pmu + fnl * bphi * pM)
    jac[12:, 1] = 2 * 1.686 * n2 * (fnl * (b1 * pM + f * pmuM) + fnl**2 * bphi * pM2)
    jac[12:, 2] = n2sq * valid
    jac[12:, 3] = (n2 * f * (b1 * pmu + f * pmu4 + fnl * bphi * pmuM)
        + n2 * (operators[0] + operators[1]) + (2 * row[7] - 1) * n2sq * operators[2]
        + 2 * (n2sq + n2nr) * zero[12])
    jac[12:, 4] = -2 * n2 * (b1 * zero[0] + f * zero[1] + fnl * bphi * zero[3])
    jac[12:, 5] = -2 * n2 * (fnl * b1 * zero[3] + fnl * f * zero[4] + fnl**2 * bphi * zero[5])
    jac[12:, 6] = -2 * n3 * zero[13]
    jac[:12, 7] = n2
    return jac


def response_diagnostics(jacobian, cholesky, widths):
    jacobian = np.asarray(jacobian, dtype=float)
    widths = np.asarray(widths, dtype=float)
    cholesky = np.asarray(cholesky, dtype=float)
    if (jacobian.ndim != 2 or jacobian.shape[1] != 8 or widths.shape != (8,)
            or np.any(widths <= 0) or cholesky.shape != (len(jacobian), len(jacobian))
            or not all(np.isfinite(x).all() for x in (jacobian, cholesky, widths))
            or np.any(np.diag(cholesky) <= 0) or np.any(np.triu(cholesky, 1))):
        raise ValueError('actual full joint positive Cholesky and eight explicit prior widths required')
    white = np.linalg.solve(cholesky, jacobian)
    scaled = white * widths
    norms = np.linalg.norm(scaled, axis=0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    exact_zero = np.flatnonzero(np.all(jacobian == 0, axis=0))
    nonzero = np.flatnonzero(norms > 0)
    cosine = np.eye(8)
    cosine[np.ix_(nonzero, nonzero)] = (scaled[:, nonzero] / norms[nonzero]).T @ (scaled[:, nonzero] / norms[nonzero])
    cosine[exact_zero, :] = 0
    cosine[:, exact_zero] = 0
    threshold = (singular[0] * 1e-9) if singular[0] else 0
    return dict(local_derivative_norm_times_full_prior_width=norms.tolist(),
        singular_values_local_prior_scaled_noise_response=singular.tolist(),
        exact_zero_conditional_response_parameters=[NOISE_NAMES[j] for j in exact_zero],
        local_response_rank_at_relative_1e9=int(np.count_nonzero(singular > threshold)),
        response_cosine=cosine.tolist(), covariance_modified=False, likelihood_or_priors_modified=False,
        local_not_global_identifiability=True, rank_not_a_parameter_removal_rule=True,
        derivative_times_width_not_finite_full_prior_change_for_nonlinear_raw_ratio=True)
