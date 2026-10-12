"""Small NumPy/SciPy fallback for the ArviZ diagnostics used by MARISA-B.

This module intentionally implements only the two ndarray operations needed by
the production MCMC runner: rank/split R-hat and identity effective sample
size.  The formulas follow Vehtari et al. (2021) and Geyer's initial-positive,
initial-monotone autocorrelation sequence, matching the corresponding ArviZ
ndarray calculations without pulling in ArviZ's plotting/xarray dependency
stack.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import next_fast_len
from scipy.stats import norm, rankdata


__version__ = "marisa-b-compat-1"


def _canonical_axes(
    values: np.ndarray,
    chain_axis: int,
    draw_axis: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        array = np.atleast_2d(array)
    chain_axis %= array.ndim
    draw_axis %= array.ndim
    if chain_axis == draw_axis:
        raise ValueError("chain_axis and draw_axis must differ")
    return np.moveaxis(array, (chain_axis, draw_axis), (0, 1))


def _map_parameters(
    values: np.ndarray,
    operation,
) -> float | np.ndarray:
    chains, draws = values.shape[:2]
    parameter_shape = values.shape[2:]
    flattened = values.reshape(chains, draws, -1)
    result = np.asarray(
        [operation(flattened[..., index]) for index in range(flattened.shape[-1])],
        dtype=np.float64,
    ).reshape(parameter_shape)
    return float(result) if not parameter_shape else result


def _split_chains(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _z_scale(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average", nan_policy="omit")
    probabilities = (ranks - 0.375) / (values.size + 0.25)
    return norm.ppf(probabilities).reshape(values.shape)


def _basic_rhat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or not np.all(np.isfinite(values))
    ):
        return math.nan
    draws = values.shape[1]
    chain_variance = np.var(values, axis=1, ddof=1)
    within_variance = float(np.mean(chain_variance))
    if within_variance <= 0.0:
        return math.nan
    between_variance = draws * float(
        np.var(np.mean(values, axis=1), ddof=1)
    )
    return math.sqrt(
        (between_variance / within_variance + draws - 1.0) / draws
    )


def _rhat_one(values: np.ndarray, method: str) -> float:
    if (
        values.shape[0] < 2
        or values.shape[1] < 4
        or not np.all(np.isfinite(values))
    ):
        return math.nan
    if method == "identity":
        return _basic_rhat(values)
    split = _split_chains(values)
    if method == "split":
        return _basic_rhat(split)
    if method == "z_scale":
        return _basic_rhat(_z_scale(split))
    if method == "folded":
        folded = np.abs(split - np.median(split))
        return _basic_rhat(_z_scale(folded))
    if method == "rank":
        location = _basic_rhat(_z_scale(split))
        folded = np.abs(split - np.median(split))
        scale = _basic_rhat(_z_scale(folded))
        return max(location, scale)
    raise TypeError(f"unsupported R-hat method: {method}")


def rhat(
    values: np.ndarray,
    *,
    method: str = "rank",
    chain_axis: int = 0,
    draw_axis: int = 1,
) -> float | np.ndarray:
    """Return an ndarray-compatible R-hat along chain and draw axes."""

    canonical = _canonical_axes(values, chain_axis, draw_axis)
    return _map_parameters(
        canonical,
        lambda parameter: _rhat_one(parameter, method),
    )


def _autocovariance(values: np.ndarray) -> np.ndarray:
    draws = values.shape[1]
    fft_length = next_fast_len(2 * draws)
    centred = values - np.mean(values, axis=1, keepdims=True)
    transformed = np.fft.rfft(centred, n=fft_length, axis=1)
    transformed *= np.conjugate(transformed)
    covariance = np.fft.irfft(transformed, n=fft_length, axis=1)
    return covariance[:, :draws] / draws


def _identity_ess_one(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 4
        or not np.all(np.isfinite(values))
    ):
        return math.nan
    if float(np.max(values) - np.min(values)) < np.finfo(float).resolution:
        return float(values.size)

    chains, draws = values.shape
    autocovariance = _autocovariance(values)
    chain_means = np.mean(values, axis=1)
    mean_variance = float(np.mean(autocovariance[:, 0])) * draws / (
        draws - 1.0
    )
    variance_plus = mean_variance * (draws - 1.0) / draws
    if chains > 1:
        variance_plus += float(np.var(chain_means, ddof=1))
    if variance_plus <= 0.0 or not math.isfinite(variance_plus):
        return math.nan

    rho = np.zeros(draws, dtype=np.float64)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (
        mean_variance - float(np.mean(autocovariance[:, 1]))
    ) / variance_plus
    rho[1] = rho_odd

    lag = 1
    while lag < draws - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (
            mean_variance
            - float(np.mean(autocovariance[:, lag + 1]))
        ) / variance_plus
        rho_odd = 1.0 - (
            mean_variance
            - float(np.mean(autocovariance[:, lag + 2]))
        ) / variance_plus
        if rho_even + rho_odd >= 0.0:
            rho[lag + 1] = rho_even
            rho[lag + 2] = rho_odd
        lag += 2

    maximum_lag = lag - 2
    if rho_even > 0.0:
        rho[maximum_lag + 1] = rho_even
    lag = 1
    while lag <= maximum_lag - 2:
        previous_pair = rho[lag - 1] + rho[lag]
        current_pair = rho[lag + 1] + rho[lag + 2]
        if current_pair > previous_pair:
            rho[lag + 1] = previous_pair / 2.0
            rho[lag + 2] = rho[lag + 1]
        lag += 2

    total = chains * draws
    tau = (
        -1.0
        + 2.0 * float(np.sum(rho[: maximum_lag + 1]))
        + float(np.sum(rho[maximum_lag + 1 : maximum_lag + 2]))
    )
    tau = max(tau, 1.0 / math.log10(total))
    if np.any(np.isnan(rho)):
        return math.nan
    return float(total / tau)


def ess(
    values: np.ndarray,
    *,
    method: str = "identity",
    chain_axis: int = 0,
    draw_axis: int = 1,
) -> float | np.ndarray:
    """Return identity ESS along chain and draw axes."""

    if method != "identity":
        raise TypeError(f"unsupported ESS method: {method}")
    canonical = _canonical_axes(values, chain_axis, draw_axis)
    return _map_parameters(canonical, _identity_ess_one)
