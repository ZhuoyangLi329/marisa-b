"""MARISA-B 推断使用的无歧义统计工具。

这个模块刻意不提供名为 ``covariance`` 的模糊入口。模拟 realization
矩阵的协方差默认只能通过 :func:`single_realization_covariance` 获得；
只有调用者明确写出 :func:`mean_covariance` 时才会除以 realization 数。
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


PrecisionCorrection = Literal["none", "hartlap", "student_t"]
SUPPORTED_PRECISION_CORRECTIONS = frozenset(
    ("none", "hartlap", "student_t")
)


def _realization_matrix(samples: ArrayLike) -> NDArray[np.float64]:
    """验证并返回 ``(N_realization, N_data)`` 浮点矩阵。

    Parameters
    ----------
    samples
        每一行是一份 realization、每一列是一个观测 bin 的二维数组。

    Returns
    -------
    numpy.ndarray
        连续语义的双精度二维数组，不改变输入内容。
    """

    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            "samples must have shape (n_realizations, n_data)"
        )
    if matrix.shape[0] < 2:
        raise ValueError("at least two realizations are required")
    if matrix.shape[1] < 1:
        raise ValueError("at least one data coordinate is required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("samples must contain only finite values")
    return matrix


def single_realization_covariance(
    samples: ArrayLike,
) -> NDArray[np.float64]:
    """估计一份等效 realization 的样本协方差。

    Parameters
    ----------
    samples
        形状为 ``(N_realization, N_data)`` 的 realization 矩阵。

    Returns
    -------
    numpy.ndarray
        使用 ``ddof=1`` 估计的 ``(N_data, N_data)`` 协方差，绝不除以
        ``N_realization``。
    """

    matrix = _realization_matrix(samples)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    return np.atleast_2d(
        np.asarray(covariance, dtype=np.float64)
    )


def mean_covariance(samples: ArrayLike) -> NDArray[np.float64]:
    """显式估计 realization 均值的协方差。

    Parameters
    ----------
    samples
        形状为 ``(N_realization, N_data)`` 的 realization 矩阵。

    Returns
    -------
    numpy.ndarray
        ``single_realization_covariance(samples) / N_realization``。

    Notes
    -----
    该函数只能被显式调用，不能作为 MARISA-B likelihood 的隐式默认值。
    """

    matrix = _realization_matrix(samples)
    return single_realization_covariance(matrix) / float(matrix.shape[0])


def validate_precision_correction(
    correction: str,
) -> PrecisionCorrection:
    """验证 likelihood 的有限模拟修正标签。

    Parameters
    ----------
    correction
        必须严格为 ``"none"``、``"hartlap"`` 或 ``"student_t"``。

    Returns
    -------
    str
        经过类型收窄的原标签。

    Notes
    -----
    ``student_t`` 表示使用完整有限模拟 likelihood，而不是给精度矩阵
    乘一个常数；因此本函数只验证协议标签，不伪造 Student-t 缩放因子。
    """

    if correction not in SUPPORTED_PRECISION_CORRECTIONS:
        supported = ", ".join(sorted(SUPPORTED_PRECISION_CORRECTIONS))
        raise ValueError(
            f"unsupported precision correction {correction!r}; "
            f"expected one of: {supported}"
        )
    return cast(PrecisionCorrection, correction)


def hartlap_factor(mock_count: int, dimension: int) -> float:
    """返回 Hartlap 逆协方差修正因子。

    Parameters
    ----------
    mock_count
        用来估计协方差的独立 realization 数 ``N``。
    dimension
        应用最终数据 mask 后的 likelihood 维数 ``p``。

    Returns
    -------
    float
        ``(N - p - 2) / (N - 1)``。
    """

    if (
        isinstance(mock_count, bool)
        or not isinstance(mock_count, Integral)
        or int(mock_count) < 2
    ):
        raise ValueError("mock_count must be an integer of at least two")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or int(dimension) < 1
    ):
        raise ValueError("dimension must be a positive integer")
    count = int(mock_count)
    size = int(dimension)
    if count <= size + 2:
        raise ValueError(
            "Hartlap correction requires mock_count > dimension + 2"
        )
    factor = (count - size - 2.0) / (count - 1.0)
    if not 0.0 < factor <= 1.0:
        raise ValueError("invalid Hartlap factor")
    return float(factor)


def weighted_quantile(
    values: ArrayLike,
    probabilities: Iterable[float],
    weights: ArrayLike,
) -> NDArray[np.float64]:
    """用加权经验分布的 midpoint 约定计算唯一版本的分位数。

    Parameters
    ----------
    values
        一维样本值。
    probabilities
        位于闭区间 ``[0, 1]`` 的分位概率。
    weights
        与 ``values`` 同形状的非负有限权重，权重和必须为正。

    Returns
    -------
    numpy.ndarray
        与 ``probabilities`` 顺序一致的分位数。

    Notes
    -----
    排序后第 ``i`` 个正权重点的位置定义为
    ``(cumsum(w)_i - 0.5*w_i) / sum(w)``。这正是生产 Student-t
    importance reweighting 原先预期的 midpoint 经验 CDF 约定。
    """

    sample = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    try:
        quantile_probabilities = np.asarray(
            tuple(probabilities),
            dtype=np.float64,
        )
    except TypeError as error:
        raise ValueError("probabilities must be a finite iterable") from error
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if sample_weights.shape != sample.shape:
        raise ValueError("weights must have the same shape as values")
    if quantile_probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must contain only finite numbers")
    if (
        not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights < 0.0)
    ):
        raise ValueError("weights must be finite and non-negative")
    if (
        not np.all(np.isfinite(quantile_probabilities))
        or np.any(quantile_probabilities < 0.0)
        or np.any(quantile_probabilities > 1.0)
    ):
        raise ValueError("probabilities must lie in [0, 1]")
    positive = sample_weights > 0.0
    if not np.any(positive):
        raise ValueError("weights must have a positive sum")
    sample = sample[positive]
    sample_weights = sample_weights[positive]
    order = np.argsort(sample, kind="mergesort")
    sorted_values = sample[order]
    sorted_weights = sample_weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = (
        cumulative - 0.5 * sorted_weights
    ) / float(cumulative[-1])
    return np.interp(
        quantile_probabilities,
        cumulative,
        sorted_values,
    )
