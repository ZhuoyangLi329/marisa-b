"""模块大纲：未拟合B分箱的条件残差；保留完整同盒cross和C_single。

仅做冻结参数的forecast形状检查，不是新的似然、显著性或噪声闭合认证。
按完整向量维数使用自己的Hartlap；严禁均值缩放、pinv或特征值修补。
"""
import numpy as np


def conditional_residual(residual, covariance_single, fit_columns, n_covariance):
    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance_single, dtype=float)
    columns = np.asarray(fit_columns)
    ndim = len(residual)
    if (residual.ndim != 1 or covariance.shape != (ndim, ndim)
            or not np.isfinite(residual).all() or not np.isfinite(covariance).all()
            or columns.ndim != 1 or columns.dtype.kind not in 'iu'
            or len(columns) == 0 or len(columns) >= ndim
            or len(set(columns.tolist())) != len(columns)
            or np.any(columns < 0) or np.any(columns >= ndim)
            or not isinstance(n_covariance, (int, np.integer))
            or n_covariance <= ndim + 2):
        raise ValueError('finite joint vector, unique integer fitted columns and enough C_single boxes required')
    if not np.allclose(covariance, covariance.T, rtol=1e-13, atol=1e-14):
        raise ValueError('actual symmetric covariance required')
    alpha = (n_covariance - ndim - 2) / (n_covariance - 1)
    joint = covariance / alpha
    joint_chol = np.linalg.cholesky(joint)
    held = np.array([j for j in range(ndim) if j not in set(columns.tolist())])
    ff = joint[np.ix_(columns, columns)]
    hf = joint[np.ix_(held, columns)]
    hh = joint[np.ix_(held, held)]
    fit_chol = np.linalg.cholesky(ff)
    # 三角solve实现C_hf C_ff^-1；不能用pseudo-inverse处理失败候选。
    transformed_cross = np.linalg.solve(fit_chol, hf.T)
    conditional_covariance = hh - transformed_cross.T @ transformed_cross
    conditional_chol = np.linalg.cholesky(conditional_covariance)
    fit_white = np.linalg.solve(fit_chol, residual[columns])
    difference = residual[held] - transformed_cross.T @ fit_white
    held_white = np.linalg.solve(conditional_chol, difference)
    joint_white = np.linalg.solve(joint_chol, residual)
    np.testing.assert_allclose(joint_white @ joint_white,
        fit_white @ fit_white + held_white @ held_white, rtol=3e-12, atol=3e-12)
    return dict(held_columns=held.tolist(), Hartlap_full_dimension=float(alpha),
        full_dimension=ndim, fitted_dimension=len(columns), held_dimension=len(held),
        conditional_residual=difference.tolist(), conditional_whitened_residual=held_white.tolist(),
        conditional_single_box_norm=float(np.linalg.norm(held_white)),
        full_joint_single_box_norm=float(np.linalg.norm(joint_white)),
        fitted_block_norm_with_full_Hartlap=float(np.linalg.norm(fit_white)),
        covariance_is_of_mean=False, covariance_multiplier=1., exact_joint_norm_decomposition=True)
