"""轻量统计语义的 synthetic 回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

import marisa_b.statistics as statistics


def test_single_and_mean_covariance_semantics() -> None:
    samples = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [4.0, 8.0],
            [8.0, 13.0],
        ]
    )
    expected_single = np.cov(samples, rowvar=False, ddof=1)
    single = statistics.single_realization_covariance(samples)
    mean = statistics.mean_covariance(samples)
    np.testing.assert_allclose(single, expected_single)
    np.testing.assert_allclose(mean, single / samples.shape[0])
    assert not hasattr(statistics, "covariance")


def test_covariance_validation_and_single_coordinate_shape() -> None:
    one_coordinate = np.asarray([[1.0], [2.0], [4.0]])
    assert statistics.single_realization_covariance(
        one_coordinate
    ).shape == (1, 1)
    with pytest.raises(ValueError):
        statistics.single_realization_covariance([[1.0, 2.0]])
    with pytest.raises(ValueError):
        statistics.mean_covariance([[1.0], [np.nan]])


def test_hartlap_factor_and_precision_protocol() -> None:
    expected = (500.0 - 27.0 - 2.0) / (500.0 - 1.0)
    assert statistics.hartlap_factor(500, 27) == pytest.approx(expected)
    for label in ("none", "hartlap", "student_t"):
        assert statistics.validate_precision_correction(label) == label
    with pytest.raises(ValueError):
        statistics.validate_precision_correction("mean_covariance")
    with pytest.raises(ValueError):
        statistics.hartlap_factor(29, 27)
    with pytest.raises(ValueError):
        statistics.hartlap_factor(True, 1)


def test_weighted_quantile_midpoint_contract() -> None:
    values = np.asarray([20.0, 0.0, 10.0])
    weights = np.asarray([1.0, 1.0, 2.0])
    probabilities = (0.0, 0.125, 0.5, 0.875, 1.0)
    expected = np.asarray([0.0, 0.0, 10.0, 20.0, 20.0])
    actual = statistics.weighted_quantile(
        values,
        probabilities,
        weights,
    )
    np.testing.assert_allclose(actual, expected)


def test_weighted_quantile_ignores_zero_weight_and_rejects_bad_input() -> None:
    result = statistics.weighted_quantile(
        [0.0, 10.0, 1000.0],
        [0.5],
        [1.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(result, [5.0])
    with pytest.raises(ValueError):
        statistics.weighted_quantile([0.0], [1.1], [1.0])
    with pytest.raises(ValueError):
        statistics.weighted_quantile([0.0], [0.5], [0.0])
