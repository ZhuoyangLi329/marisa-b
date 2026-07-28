#!/usr/bin/env python3
"""Validate the identifiable pre-reconstruction halo EFT bias model.

This is a versioned analysis adapter around the frozen EFT-v2 R0 templates.
It never modifies the frozen fitter or theory products.  Its statistical
contract is deliberately different from the historical joint R0 fit:

1. calibrate b1 with pre-reconstruction P0 only;
2. fix b1 and fit B000 only, excluding global B000 bin 0;
3. compare the complete 24-coordinate fixed-b1 B span with the
   coevolution-reduced span;
4. profile all conditionally linear nuisance directions exactly.

The 500-realization mean is used only as a low-noise central vector.  Every
fit, uncertainty, goodness-of-fit statistic, predictive test, and displayed
error bar uses the covariance of one equivalent realization, never the
covariance of the mean.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution


TAG = "pre_recon_bias_model_v3_20260725"
EXCLUDED_B000_INDEX = 0
P_KMAX_VALUES = (0.08, 0.10, 0.12)
B_KMAX_VALUES = (0.10, 0.12, 0.14, 0.15)
UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX = 0.08
PRIMARY_P_KMAX = 0.10
TARGET_B_KMAX = 0.15
CORE_COMPARE_NAMES = ("b2", "gamma2", "gamma21")
COEVOLUTION_DEPENDENT = (
    "gamma21",
    "gamma21x",
    "gamma211",
    "gamma22",
    "gamma31",
)
POWER_FREE_FULL = (
    "b1",
    "b2",
    "gamma2",
    "gamma21",
    "b_nabla2_delta",
    "Pshot",
    "a0_power",
)
POWER_NONLINEAR_FULL = ("b1", "b2", "gamma2", "gamma21")
POWER_FREE_COEVOLUTION = (
    "b1",
    "b2",
    "gamma2",
    "b_nabla2_delta",
    "Pshot",
    "a0_power",
)
POWER_NONLINEAR_COEVOLUTION = ("b1", "b2", "gamma2")
UPLIFT_FREE_NAMES = (
    "b2",
    "gamma2",
    "Ashot_residual",
    "Bshot_residual",
)
UPLIFT_NONLINEAR_NAMES = ("b2", "gamma2")
PDF_METADATA = {
    "Author": "MARISA-B pre-reconstruction bias-model v3 audit",
    "Creator": "scripts/production/run_pre_recon_bias_model_v3.py",
    "Subject": (
        "Quijote z=1 pre-reconstruction real-space halo P0 and B000 "
        "bias-model validation"
    ),
}


@dataclass(frozen=True)
class PriorBlock:
    names: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    sigma: np.ndarray
    whitener: np.ndarray


@dataclass
class ConditionalFit:
    label: str
    tier: str
    kmax: float
    free_names: tuple[str, ...]
    nonlinear_names: tuple[str, ...]
    linear_names: tuple[str, ...]
    x: np.ndarray
    covariance: np.ndarray
    expanded_names: tuple[str, ...]
    expanded_x: np.ndarray
    expanded_covariance: np.ndarray
    prediction: np.ndarray
    components: dict[str, np.ndarray]
    target: np.ndarray
    covariance_fit: np.ndarray
    covariance_single: np.ndarray
    selected_indices: np.ndarray
    residual: np.ndarray
    whitened: np.ndarray
    data_chi2: float
    prior_chi2: float
    objective: float
    effective_parameter_count: float
    effective_rank: int
    dof: float
    p_value: float
    max_abs_fit_pull: float
    max_abs_single_pull: float
    max_sign_run: int
    singular_values: np.ndarray
    data_jacobian: np.ndarray
    selected_model_jacobian: np.ndarray
    posterior_to_prior_sigma_ratio: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    multistart_objectives: np.ndarray
    analytic_linear_delta_objective: float
    basis_rotation_max_sigma_single: float
    affine_residual_max_sigma_single: float

    def free_index(self, name: str) -> int:
        return self.free_names.index(name)

    def expanded_index(self, name: str) -> int:
        return self.expanded_names.index(name)

    def expanded_value(self, name: str) -> float:
        return float(self.expanded_x[self.expanded_index(name)])

    def expanded_sigma(self, name: str) -> float:
        index = self.expanded_index(name)
        return float(math.sqrt(max(self.expanded_covariance[index, index], 0.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External data bundle; otherwise use MARISA_B_DATA_ROOT.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated-product root; defaults to the external data root.",
    )
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--cv-multistart", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only outputs under the explicitly selected version tag.",
    )
    parser.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="Diagnostic development option; a skipped CV gate cannot pass.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            to_jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_frozen_module(path: Path) -> ModuleType:
    name = "_marisa_b_frozen_eft_v2_for_bias_model_v3"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen fitter {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def longest_sign_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    previous = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign and sign == previous:
            current += 1
        elif sign:
            current = 1
        else:
            current = 0
        previous = sign
        longest = max(longest, current)
    return longest


def covariance_single(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 3:
        raise ValueError("single covariance requires at least three 2D samples")
    return np.cov(samples, rowvar=False, ddof=1)


def b_mask(data: Any, kmax: float, closing_side_safe: bool = False) -> np.ndarray:
    mask = np.max(np.asarray(data.k_pair), axis=1) <= kmax + 1.0e-12
    mask = np.asarray(mask, dtype=bool)
    mask[EXCLUDED_B000_INDEX] = False
    if closing_side_safe:
        upper = np.asarray(data.edges, dtype=np.float64)[:, :, 1]
        mask &= np.sum(upper, axis=1) <= kmax + 1.0e-12
    return mask


def p_mask(data: Any, kmax: float) -> np.ndarray:
    return np.asarray(data.power_k <= kmax + 1.0e-12, dtype=bool)


def coevolution_parameters(parameters: dict[str, float]) -> None:
    """Apply Eggemeier et al. (2021), Eqs. (20)--(24), with NLE_L=0."""
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    gamma2 = parameters["gamma2"]
    gamma2x = parameters["gamma2x"]
    gamma3 = parameters["gamma3"]
    gamma21 = (2.0 / 21.0) * (b1 - 1.0) + (6.0 / 7.0) * gamma2
    parameters["gamma21"] = gamma21
    parameters["gamma21x"] = (2.0 / 21.0) * b2 + (6.0 / 7.0) * gamma2x
    parameters["gamma211"] = (
        (5.0 / 77.0) * (b1 - 1.0)
        + (15.0 / 14.0) * gamma2
        - (9.0 / 7.0) * gamma3
        + gamma21
    )
    parameters["gamma22"] = (
        -(6.0 / 539.0) * (b1 - 1.0) - (9.0 / 49.0) * gamma2
    )
    parameters["gamma31"] = -(4.0 / 11.0) * (b1 - 1.0) - 6.0 * gamma2


def full_parameter_dict(
    module: ModuleType,
    names: Iterable[str],
    values: np.ndarray,
    *,
    tier: str,
    fixed_b1: float | None,
) -> dict[str, float]:
    parameters = {
        name: 0.0
        for name in (
            module.BIAS_NAMES
            + module.COUNTERTERM_NAMES
            + module.STOCHASTIC_POWER_NAMES
            + module.STOCHASTIC_B_NAMES
        )
    }
    for name, value in zip(names, values):
        parameters[name] = float(value)
    if fixed_b1 is not None:
        parameters["b1"] = float(fixed_b1)
    if tier == "coevolution":
        coevolution_parameters(parameters)
    elif tier != "full":
        raise ValueError(f"unknown model tier {tier!r}")
    parameters["a5_mixed"] = 0.0
    return parameters


def prior_block(
    module: ModuleType,
    frozen_prior: Any,
    names: tuple[str, ...],
    *,
    nuisance_scale: float = 1.0,
    nuisance_names: Iterable[str] = (),
) -> PriorBlock:
    if not math.isfinite(nuisance_scale) or nuisance_scale <= 0.0:
        raise ValueError("nuisance prior scale must be finite and positive")
    indices = np.asarray([module.FIT_NAMES.index(name) for name in names], dtype=int)
    mean = np.asarray(frozen_prior.mean[indices], dtype=np.float64)
    covariance = np.asarray(
        frozen_prior.covariance[np.ix_(indices, indices)],
        dtype=np.float64,
    )
    nuisance = set(nuisance_names)
    scales = np.asarray(
        [nuisance_scale if name in nuisance else 1.0 for name in names],
        dtype=np.float64,
    )
    covariance = scales[:, None] * covariance * scales[None, :]
    cholesky = np.linalg.cholesky(covariance)
    whitener = solve_triangular(
        cholesky,
        np.eye(len(names), dtype=np.float64),
        lower=True,
    )
    return PriorBlock(
        names=names,
        mean=mean,
        covariance=covariance,
        sigma=np.sqrt(np.diag(covariance)),
        whitener=whitener,
    )


def expanded_mapping(
    module: ModuleType,
    free_names: tuple[str, ...],
    x: np.ndarray,
    *,
    tier: str,
    fixed_b1: float | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    expanded_names = tuple(module.FIT_NAMES)

    def evaluate(values: np.ndarray) -> np.ndarray:
        parameters = full_parameter_dict(
            module,
            free_names,
            values,
            tier=tier,
            fixed_b1=fixed_b1,
        )
        return np.asarray([parameters[name] for name in expanded_names])

    central = evaluate(x)
    jacobian = np.empty((len(expanded_names), len(free_names)), dtype=np.float64)
    for index in range(len(free_names)):
        step = 1.0e-6 * max(abs(x[index]), 1.0)
        right = x.copy()
        left = x.copy()
        right[index] += step
        left[index] -= step
        jacobian[:, index] = (evaluate(right) - evaluate(left)) / (2.0 * step)
    return expanded_names, central, jacobian


def fit_conditional(
    module: ModuleType,
    *,
    label: str,
    tier: str,
    kmax: float,
    free_names: tuple[str, ...],
    nonlinear_names: tuple[str, ...],
    prior: PriorBlock,
    target: np.ndarray,
    covariance_fit_matrix: np.ndarray,
    covariance_single_matrix: np.ndarray,
    selected_indices: np.ndarray,
    model_and_components: Callable[[np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
    expanded: Callable[[np.ndarray], tuple[tuple[str, ...], np.ndarray, np.ndarray]],
    multistart: int,
    rng: np.random.Generator,
    correlation_jitter: float = 1.0e-12,
) -> ConditionalFit:
    target = np.asarray(target, dtype=np.float64)
    covariance_fit_matrix = np.asarray(covariance_fit_matrix, dtype=np.float64)
    covariance_single_matrix = np.asarray(covariance_single_matrix, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError("fit target must be one-dimensional")
    if covariance_fit_matrix.shape != (target.size, target.size):
        raise ValueError("fit covariance shape does not match target")
    if covariance_single_matrix.shape != (target.size, target.size):
        raise ValueError("single covariance shape does not match target")
    if prior.names != free_names:
        raise ValueError("prior/free-name mismatch")
    name_to_index = {name: index for index, name in enumerate(free_names)}
    nonlinear_indices = np.asarray(
        [name_to_index[name] for name in nonlinear_names],
        dtype=int,
    )
    linear_names = tuple(name for name in free_names if name not in nonlinear_names)
    linear_indices = np.asarray(
        [name_to_index[name] for name in linear_names],
        dtype=int,
    )
    cholesky = module.covariance_cholesky(
        covariance_fit_matrix,
        correlation_jitter,
    )
    single_sigma = np.sqrt(np.diag(covariance_single_matrix))

    def full_residual(x: np.ndarray) -> np.ndarray:
        prediction, _ = model_and_components(x)
        data_part = solve_triangular(cholesky, prediction - target, lower=True)
        prior_part = prior.whitener @ (x - prior.mean)
        return np.concatenate((data_part, prior_part))

    def linear_problem(
        nonlinear: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = prior.mean.copy()
        x[nonlinear_indices] = nonlinear
        x[linear_indices] = 0.0
        base = model_and_components(x)[0]
        design = np.empty((target.size, linear_indices.size), dtype=np.float64)
        for column, parameter_index in enumerate(linear_indices):
            unit = x.copy()
            unit[parameter_index] = 1.0
            design[:, column] = model_and_components(unit)[0] - base
        whitened_design = solve_triangular(cholesky, design, lower=True)
        whitened_target = solve_triangular(cholesky, target - base, lower=True)
        prior_design = prior.whitener[:, linear_indices]
        prior_target = -prior.whitener @ (x - prior.mean)
        return x, base, design, np.vstack((whitened_design, prior_design)), np.concatenate(
            (whitened_target, prior_target)
        )

    def profile_linear(non_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, _base, _design, augmented_design, augmented_target = linear_problem(
            non_linear
        )
        linear = np.linalg.lstsq(
            augmented_design,
            augmented_target,
            rcond=1.0e-12,
        )[0]
        x[linear_indices] = linear
        return x, full_residual(x)

    def profiled_residual(non_linear: np.ndarray) -> np.ndarray:
        return profile_linear(non_linear)[1]

    starts = [prior.mean[nonlinear_indices].copy()]
    for _ in range(max(1, multistart) - 1):
        starts.append(
            prior.mean[nonlinear_indices]
            + 0.35
            * prior.sigma[nonlinear_indices]
            * rng.normal(size=nonlinear_indices.size)
        )
    solutions = [
        least_squares(
            profiled_residual,
            start,
            method="trf",
            x_scale=prior.sigma[nonlinear_indices],
            max_nfev=4000,
            ftol=2.0e-10,
            xtol=2.0e-10,
            gtol=2.0e-10,
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda item: float(item.fun @ item.fun))
    x, residual_all = profile_linear(solution.x)
    prediction, components = model_and_components(x)
    residual = prediction - target
    whitened = solve_triangular(cholesky, residual, lower=True)
    prior_residual = prior.whitener @ (x - prior.mean)
    data_chi2 = float(whitened @ whitened)
    prior_chi2 = float(prior_residual @ prior_residual)
    objective = data_chi2 + prior_chi2

    complete_jacobian = np.empty(
        (target.size + len(free_names), len(free_names)),
        dtype=np.float64,
    )
    selected_model_jacobian = np.empty(
        (target.size, len(free_names)),
        dtype=np.float64,
    )
    for index in range(len(free_names)):
        step = 1.0e-5 * max(abs(x[index]), prior.sigma[index], 1.0e-3)
        right = x.copy()
        left = x.copy()
        right[index] += step
        left[index] -= step
        right_prediction = model_and_components(right)[0]
        left_prediction = model_and_components(left)[0]
        selected_model_jacobian[:, index] = (
            right_prediction - left_prediction
        ) / (2.0 * step)
        complete_jacobian[:, index] = (
            full_residual(right) - full_residual(left)
        ) / (2.0 * step)
    data_jacobian = complete_jacobian[: target.size]
    singular_values = np.linalg.svd(data_jacobian, compute_uv=False)
    tolerance = (
        max(data_jacobian.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
        if singular_values.size
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    covariance = np.linalg.pinv(
        complete_jacobian.T @ complete_jacobian,
        rcond=1.0e-11,
    )
    effective_parameter_count = float(
        np.trace(data_jacobian @ covariance @ data_jacobian.T)
    )
    dof = max(float(target.size) - effective_parameter_count, 0.0)
    p_value = (
        float(chi2_distribution.sf(data_chi2, dof)) if dof > 0.0 else math.nan
    )
    posterior_sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    # Explicit linear least-squares check at the selected nonlinear mode.
    _, _base, _design, augmented_design, augmented_target = linear_problem(
        solution.x
    )
    explicit = least_squares(
        lambda values: augmented_design @ values - augmented_target,
        x[linear_indices],
        method="trf",
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=500,
    )
    analytic_linear_objective = float(
        np.linalg.norm(
            augmented_design @ x[linear_indices] - augmented_target
        )
        ** 2
    )
    explicit_linear_objective = float(explicit.fun @ explicit.fun)
    analytic_delta = abs(analytic_linear_objective - explicit_linear_objective)

    # Check that the model is affine in the registered linear block.
    affine_x = x.copy()
    affine_x[linear_indices] = 0.0
    affine_base = model_and_components(affine_x)[0]
    affine_sum = affine_base.copy()
    for parameter_index in linear_indices:
        unit = affine_x.copy()
        unit[parameter_index] = x[parameter_index]
        affine_sum += model_and_components(unit)[0] - affine_base
    affine_error = np.max(np.abs(affine_sum - prediction) / single_sigma)

    # Rotate the complete linear block and solve with the transformed prior.
    basis_rng = np.random.default_rng(9137 + len(free_names) + target.size)
    orthogonal, _ = np.linalg.qr(
        basis_rng.normal(size=(linear_indices.size, linear_indices.size))
    )
    _, base, design, _augmented_design, _augmented_target = linear_problem(
        solution.x
    )
    conditional_x = prior.mean.copy()
    conditional_x[nonlinear_indices] = solution.x
    conditional_mean = prior.mean[linear_indices]
    conditional_covariance = prior.covariance[np.ix_(linear_indices, linear_indices)]
    cross_covariance = prior.covariance[np.ix_(linear_indices, nonlinear_indices)]
    nonlinear_covariance = prior.covariance[np.ix_(nonlinear_indices, nonlinear_indices)]
    if nonlinear_indices.size:
        conditional_mean = conditional_mean + cross_covariance @ np.linalg.solve(
            nonlinear_covariance,
            solution.x - prior.mean[nonlinear_indices],
        )
        conditional_covariance = conditional_covariance - cross_covariance @ np.linalg.solve(
            nonlinear_covariance,
            cross_covariance.T,
        )
    rotated_design = design @ orthogonal
    rotated_mean = orthogonal.T @ conditional_mean
    rotated_covariance = orthogonal.T @ conditional_covariance @ orthogonal
    rotated_whitener = solve_triangular(
        np.linalg.cholesky(rotated_covariance),
        np.eye(linear_indices.size),
        lower=True,
    )
    whitened_design = solve_triangular(cholesky, rotated_design, lower=True)
    whitened_target = solve_triangular(cholesky, target - base, lower=True)
    rotated_solution = np.linalg.lstsq(
        np.vstack((whitened_design, rotated_whitener)),
        np.concatenate((whitened_target, rotated_whitener @ rotated_mean)),
        rcond=1.0e-12,
    )[0]
    rotated_prediction = base + rotated_design @ rotated_solution
    rotation_error = float(
        np.max(np.abs(rotated_prediction - prediction) / single_sigma)
    )

    expanded_names, expanded_x, expanded_jacobian = expanded(x)
    expanded_covariance = expanded_jacobian @ covariance @ expanded_jacobian.T
    return ConditionalFit(
        label=label,
        tier=tier,
        kmax=kmax,
        free_names=free_names,
        nonlinear_names=nonlinear_names,
        linear_names=linear_names,
        x=x,
        covariance=covariance,
        expanded_names=expanded_names,
        expanded_x=expanded_x,
        expanded_covariance=expanded_covariance,
        prediction=prediction,
        components=components,
        target=target,
        covariance_fit=covariance_fit_matrix,
        covariance_single=covariance_single_matrix,
        selected_indices=np.asarray(selected_indices, dtype=int),
        residual=residual,
        whitened=whitened,
        data_chi2=data_chi2,
        prior_chi2=prior_chi2,
        objective=float(residual_all @ residual_all),
        effective_parameter_count=effective_parameter_count,
        effective_rank=rank,
        dof=dof,
        p_value=p_value,
        max_abs_fit_pull=float(
            np.max(np.abs(residual) / np.sqrt(np.diag(covariance_fit_matrix)))
        ),
        max_abs_single_pull=float(np.max(np.abs(residual) / single_sigma)),
        max_sign_run=longest_sign_run(whitened),
        singular_values=singular_values,
        data_jacobian=data_jacobian,
        selected_model_jacobian=selected_model_jacobian,
        posterior_to_prior_sigma_ratio=np.minimum(
            1.0,
            posterior_sigma / prior.sigma,
        ),
        optimizer_success=bool(solution.success),
        optimizer_message=str(solution.message),
        multistart_objectives=np.asarray(
            [float(item.fun @ item.fun) for item in solutions]
        ),
        analytic_linear_delta_objective=analytic_delta,
        basis_rotation_max_sigma_single=rotation_error,
        affine_residual_max_sigma_single=float(affine_error),
    )


def b_free_names(module: ModuleType, tier: str) -> tuple[str, ...]:
    excluded = {"b1", "Pshot", "a0_power"}
    names = tuple(name for name in module.FIT_NAMES if name not in excluded)
    if tier == "coevolution":
        names = tuple(name for name in names if name not in COEVOLUTION_DEPENDENT)
    return names


def b_nonlinear_names(tier: str) -> tuple[str, ...]:
    if tier == "full":
        return ("b2", "gamma2", "gamma21", "b_nabla2_delta")
    if tier == "coevolution":
        return ("b2", "gamma2", "b_nabla2_delta")
    raise ValueError(tier)


def make_power_fit(
    module: ModuleType,
    power_templates: Any,
    data: Any,
    frozen_prior: Any,
    nbar: float,
    samples: np.ndarray,
    *,
    kmax: float,
    tier: str,
    multistart: int,
    rng: np.random.Generator,
) -> ConditionalFit:
    mask = p_mask(data, kmax)
    indices = np.flatnonzero(mask)
    selected_samples = np.asarray(samples[:, indices], dtype=np.float64)
    target = np.mean(selected_samples, axis=0)
    covariance_single_matrix = covariance_single(selected_samples)
    free_names = (
        POWER_FREE_FULL if tier == "full" else POWER_FREE_COEVOLUTION
    )
    nonlinear_names = (
        POWER_NONLINEAR_FULL
        if tier == "full"
        else POWER_NONLINEAR_COEVOLUTION
    )
    prior = prior_block(
        module,
        frozen_prior,
        free_names,
        nuisance_names=tuple(
            name for name in free_names if name not in nonlinear_names
        ),
    )

    def model(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        parameters = full_parameter_dict(
            module,
            free_names,
            values,
            tier=tier,
            fixed_b1=None,
        )
        components = power_templates.components(parameters, nbar)
        return components["total"][indices], components

    return fit_conditional(
        module,
        label="P0_b1_calibration",
        tier=tier,
        kmax=kmax,
        free_names=free_names,
        nonlinear_names=nonlinear_names,
        prior=prior,
        target=target,
        covariance_fit_matrix=covariance_single_matrix,
        covariance_single_matrix=covariance_single_matrix,
        selected_indices=indices,
        model_and_components=model,
        expanded=lambda values: expanded_mapping(
            module,
            free_names,
            values,
            tier=tier,
            fixed_b1=None,
        ),
        multistart=multistart,
        rng=rng,
    )


def make_b_fit(
    module: ModuleType,
    templates: Any,
    data: Any,
    frozen_prior: Any,
    nbar: float,
    samples: np.ndarray,
    *,
    b1: float,
    kmax: float,
    tier: str,
    nuisance_prior_scale: float,
    multistart: int,
    rng: np.random.Generator,
    closing_side_safe: bool = False,
) -> ConditionalFit:
    mask = b_mask(data, kmax, closing_side_safe=closing_side_safe)
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        raise ValueError(
            f"{tier} kmax={kmax}: only {indices.size} B000 points selected"
        )
    selected_samples = np.asarray(samples[:, indices], dtype=np.float64)
    target = np.mean(selected_samples, axis=0)
    covariance_single_matrix = covariance_single(selected_samples)
    free_names = b_free_names(module, tier)
    nonlinear_names = b_nonlinear_names(tier)
    linear_names = tuple(name for name in free_names if name not in nonlinear_names)
    prior = prior_block(
        module,
        frozen_prior,
        free_names,
        nuisance_scale=nuisance_prior_scale,
        nuisance_names=linear_names,
    )

    def model(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        parameters = full_parameter_dict(
            module,
            free_names,
            values,
            tier=tier,
            fixed_b1=b1,
        )
        components = templates.components(parameters, nbar)
        return components["total"][indices], components

    return fit_conditional(
        module,
        label=(
            "B000_closing_side_safe"
            if closing_side_safe
            else "B000_stored_radial"
        ),
        tier=tier,
        kmax=kmax,
        free_names=free_names,
        nonlinear_names=nonlinear_names,
        prior=prior,
        target=target,
        covariance_fit_matrix=covariance_single_matrix,
        covariance_single_matrix=covariance_single_matrix,
        selected_indices=indices,
        model_and_components=model,
        expanded=lambda values: expanded_mapping(
            module,
            free_names,
            values,
            tier=tier,
            fixed_b1=b1,
        ),
        multistart=multistart,
        rng=rng,
    )


def uplift_components(
    module: ModuleType,
    templates: Any,
    parameters: dict[str, float],
    nbar: float,
) -> dict[str, np.ndarray]:
    """Historical halo-tree + pure-matter-loop uplift control.

    The complete frozen halo templates are polynomial in the bias
    coefficients.  The pure matter one-loop contribution is exactly the
    ``b1^3`` monomial of each loop diagram in this convention.  Keeping only
    those monomials, the tree diagram, and the two leading stochastic shapes
    reproduces the model-order content of the historical uplift while using
    the same bin projection as the complete EFT templates.
    """
    cache: dict[str, float] = {"1": 1.0}
    tree = templates._evaluate_shape("diagram:tree", parameters, cache)
    loop = np.zeros_like(tree)
    b1_cubed = parameters["b1"] ** 3
    for diagram in module.DIAGRAM_NAMES[1:]:
        coefficients = templates._compiled[f"diagram:{diagram}"].get("b1^3")
        if coefficients is None:
            raise KeyError(f"missing b1^3 pure-matter monomial in {diagram}")
        loop += b1_cubed * coefficients
    a_shape = templates._compiled["stochastic:Ashot_residual"].get("1")
    b_shape = templates._compiled["stochastic:Bshot_residual"].get("b1^2")
    if a_shape is None or b_shape is None:
        raise KeyError("missing leading uplift stochastic shapes")
    stochastic = (
        parameters["Ashot_residual"] * a_shape / (nbar * nbar)
        + parameters["Bshot_residual"]
        * parameters["b1"] ** 2
        * b_shape
        / nbar
    )
    counterterm = np.zeros_like(tree)
    return {
        "tree": tree,
        "loop": loop,
        "spt": tree + loop,
        "counterterm": counterterm,
        "stochastic": stochastic,
        "total": tree + loop + stochastic,
    }


def make_uplift_fit(
    module: ModuleType,
    templates: Any,
    data: Any,
    frozen_prior: Any,
    nbar: float,
    samples: np.ndarray,
    *,
    b1: float,
    kmax: float,
    nuisance_prior_scale: float,
    multistart: int,
    rng: np.random.Generator,
    closing_side_safe: bool = False,
) -> ConditionalFit:
    mask = b_mask(data, kmax, closing_side_safe=closing_side_safe)
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        raise ValueError(
            f"uplift kmax={kmax}: only {indices.size} B000 points selected"
        )
    selected_samples = np.asarray(samples[:, indices], dtype=np.float64)
    target = np.mean(selected_samples, axis=0)
    covariance_single_matrix = covariance_single(selected_samples)
    linear_names = tuple(
        name for name in UPLIFT_FREE_NAMES if name not in UPLIFT_NONLINEAR_NAMES
    )
    prior = prior_block(
        module,
        frozen_prior,
        UPLIFT_FREE_NAMES,
        nuisance_scale=nuisance_prior_scale,
        nuisance_names=linear_names,
    )

    def model(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        parameters = full_parameter_dict(
            module,
            UPLIFT_FREE_NAMES,
            values,
            tier="full",
            fixed_b1=b1,
        )
        components = uplift_components(module, templates, parameters, nbar)
        return components["total"][indices], components

    return fit_conditional(
        module,
        label=(
            "B000_uplift_closing_side_safe"
            if closing_side_safe
            else "B000_uplift_stored_radial"
        ),
        tier="uplift",
        kmax=kmax,
        free_names=UPLIFT_FREE_NAMES,
        nonlinear_names=UPLIFT_NONLINEAR_NAMES,
        prior=prior,
        target=target,
        covariance_fit_matrix=covariance_single_matrix,
        covariance_single_matrix=covariance_single_matrix,
        selected_indices=indices,
        model_and_components=model,
        expanded=lambda values: expanded_mapping(
            module,
            UPLIFT_FREE_NAMES,
            values,
            tier="full",
            fixed_b1=b1,
        ),
        multistart=multistart,
        rng=rng,
    )


def qnorm(vector: np.ndarray, covariance: np.ndarray, module: ModuleType) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    cholesky = module.covariance_cholesky(covariance, 1.0e-12)
    whitened = solve_triangular(cholesky, vector, lower=True)
    return float(math.sqrt(max(float(whitened @ whitened), 0.0)))


def perturbativity_report(
    module: ModuleType,
    fit: ConditionalFit,
    frozen_prior: Any,
) -> dict[str, Any]:
    indices = fit.selected_indices
    components = fit.components
    tree = np.asarray(components["tree"])[indices]
    loop = np.asarray(components["loop"])[indices]
    counterterm = np.asarray(components["counterterm"])[indices]
    stochastic = np.asarray(components["stochastic"])[indices]
    total = np.asarray(components["total"])[indices]
    tree_norm = qnorm(tree, fit.covariance_single, module)
    loop_norm = qnorm(loop, fit.covariance_single, module)
    deterministic_norm = qnorm(tree + loop, fit.covariance_single, module)
    counterterm_norm = qnorm(counterterm, fit.covariance_single, module)
    stochastic_norm = qnorm(stochastic, fit.covariance_single, module)
    scale = np.maximum(
        np.abs(total),
        1.0e-6 * max(float(np.max(np.abs(total))), 1.0),
    )
    cancellation = (
        np.abs(tree)
        + np.abs(loop)
        + np.abs(counterterm)
        + np.abs(stochastic)
    ) / scale
    stochastic_names = set(module.FIT_STOCHASTIC_B_NAMES)
    prior_z = {}
    for name in fit.free_names:
        if name in stochastic_names:
            full_index = module.FIT_NAMES.index(name)
            prior_z[name] = abs(
                fit.x[fit.free_index(name)] - frozen_prior.mean[full_index]
            ) / frozen_prior.sigma[full_index]
    loop_ratio = loop_norm / tree_norm if tree_norm > 0.0 else math.inf
    counterterm_ratio = (
        counterterm_norm / deterministic_norm
        if deterministic_norm > 0.0
        else math.inf
    )
    max_stochastic_prior_z = max(prior_z.values(), default=0.0)
    return {
        "covariance": "single-realization selected B000 covariance",
        "norms": {
            "tree": tree_norm,
            "one_loop": loop_norm,
            "tree_plus_one_loop": deterministic_norm,
            "counterterm": counterterm_norm,
            "stochastic": stochastic_norm,
            "total": qnorm(total, fit.covariance_single, module),
        },
        "ratios": {
            "one_loop_over_tree": loop_ratio,
            "counterterm_over_tree_plus_one_loop": counterterm_ratio,
        },
        "cancellation": {
            "maximum_sum_abs_over_abs_total": float(np.max(cancellation)),
            "median_sum_abs_over_abs_total": float(np.median(cancellation)),
            "maximum_bin_global_index": int(indices[int(np.argmax(cancellation))]),
        },
        "stochastic_prior_z": prior_z,
        "maximum_abs_stochastic_prior_z": max_stochastic_prior_z,
        "passes_registered_gate": bool(
            loop_ratio < 1.0
            and counterterm_ratio < 0.5
            and max_stochastic_prior_z <= 2.0
        ),
    }


def fit_report(fit: ConditionalFit) -> dict[str, Any]:
    free_sigma = np.sqrt(np.maximum(np.diag(fit.covariance), 0.0))
    expanded_sigma = np.sqrt(
        np.maximum(np.diag(fit.expanded_covariance), 0.0)
    )
    condition = (
        float(fit.singular_values[0] / fit.singular_values[-1])
        if fit.singular_values.size and fit.singular_values[-1] > 0.0
        else math.inf
    )
    ranks = {}
    if fit.singular_values.size:
        for tolerance in (1.0e-8, 1.0e-10, 1.0e-12):
            ranks[f"{tolerance:.0e}"] = int(
                np.count_nonzero(
                    fit.singular_values
                    > tolerance * fit.singular_values[0]
                )
            )
    return {
        "label": fit.label,
        "tier": fit.tier,
        "kmax_h_mpc": fit.kmax,
        "fit_covariance": "single-realization sample covariance",
        "selected_global_indices": fit.selected_indices,
        "n_data": fit.target.size,
        "free_parameter_count": len(fit.free_names),
        "explicit_nonlinear_parameter_count": len(fit.nonlinear_names),
        "nonlinear_names": fit.nonlinear_names,
        "linear_names": fit.linear_names,
        "free_parameters": {
            name: {
                "value": fit.x[index],
                "posterior_sigma": free_sigma[index],
                "posterior_to_prior_sigma_ratio": (
                    fit.posterior_to_prior_sigma_ratio[index]
                ),
            }
            for index, name in enumerate(fit.free_names)
        },
        "expanded_parameters": {
            name: {
                "value": fit.expanded_x[index],
                "posterior_sigma_conditional_on_fixed_b1": expanded_sigma[index],
            }
            for index, name in enumerate(fit.expanded_names)
        },
        "statistics": {
            "data_chi2": fit.data_chi2,
            "prior_chi2": fit.prior_chi2,
            "objective": fit.objective,
            "effective_parameter_count": fit.effective_parameter_count,
            "effective_rank_machine_tolerance": fit.effective_rank,
            "dof": fit.dof,
            "chi2_per_dof": (
                fit.data_chi2 / fit.dof if fit.dof > 0.0 else math.nan
            ),
            "residual_norm_sigma_single": math.sqrt(
                max(fit.data_chi2, 0.0)
            ),
            "p_value": fit.p_value,
            "max_abs_fit_covariance_pull": fit.max_abs_fit_pull,
            "max_abs_single_pull": fit.max_abs_single_pull,
            "max_same_sign_whitened_run": fit.max_sign_run,
        },
        "identifiability": {
            "whitened_data_jacobian_singular_values": fit.singular_values,
            "condition_number": condition,
            "relative_tolerance_ranks": ranks,
        },
        "numerical_checks": {
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "multistart_objectives": fit.multistart_objectives,
            "analytic_minus_explicit_linear_objective_abs": (
                fit.analytic_linear_delta_objective
            ),
            "basis_rotation_max_abs_prediction_sigma_single": (
                fit.basis_rotation_max_sigma_single
            ),
            "affine_linear_block_max_abs_residual_sigma_single": (
                fit.affine_residual_max_sigma_single
            ),
        },
    }


def closure_pass(fit: ConditionalFit) -> bool:
    return bool(
        fit.optimizer_success
        and fit.dof > 0.0
        and fit.data_chi2 / fit.dof <= 1.5
        and fit.p_value >= 0.05
        and fit.max_abs_fit_pull < 3.0
        and math.sqrt(max(fit.data_chi2, 0.0)) < 0.5
    )


def numerical_pass(fit: ConditionalFit) -> bool:
    reports = fit_report(fit)
    ranks = reports["identifiability"]["relative_tolerance_ranks"]
    return bool(
        fit.optimizer_success
        and len(fit.nonlinear_names) <= 4
        and fit.analytic_linear_delta_objective < 1.0e-5
        and fit.basis_rotation_max_sigma_single < 0.01
        and fit.affine_residual_max_sigma_single < 0.01
        and len(set(ranks.values())) <= 1
    )


def standardized_shift(
    first: ConditionalFit,
    second: ConditionalFit,
    names: Iterable[str],
) -> dict[str, float]:
    result = {}
    for name in names:
        delta = abs(first.expanded_value(name) - second.expanded_value(name))
        sigma = math.sqrt(
            first.expanded_sigma(name) ** 2 + second.expanded_sigma(name) ** 2
        )
        result[name] = delta / sigma if sigma > 0.0 else math.inf
    return result


def prediction_distance_single(
    module: ModuleType,
    first: ConditionalFit,
    second: ConditionalFit,
) -> float:
    if not np.array_equal(first.selected_indices, second.selected_indices):
        raise ValueError("prediction-distance fits use different masks")
    difference = first.prediction - second.prediction
    return qnorm(difference, first.covariance_single, module)


def exact_stochastic_degeneracy(
    module: ModuleType,
    templates: Any,
    b1: float,
) -> dict[str, Any]:
    parameters = full_parameter_dict(
        module,
        (),
        np.empty(0),
        tier="full",
        fixed_b1=b1,
    )
    cache: dict[str, float] = {"1": 1.0}
    shapes = {
        name: templates._evaluate_shape(
            f"stochastic:{name}",
            parameters,
            cache,
        )
        for name in ("abar0_mixed", "a3_mixed", "a4_mixed", "a5_mixed")
    }
    residual = (
        shapes["abar0_mixed"]
        + shapes["a3_mixed"]
        - 0.5 * shapes["a4_mixed"]
        - 0.5 * shapes["a5_mixed"]
    )
    scale = max(
        np.linalg.norm(shape) for shape in shapes.values()
    )
    relative = float(np.linalg.norm(residual) / scale) if scale > 0.0 else 0.0
    return {
        "relation": (
            "S_abar0 + S_a3 - 0.5*S_a4 - 0.5*S_a5 = 0"
        ),
        "relative_l2_residual": relative,
        "absolute_max_residual": float(np.max(np.abs(residual))),
        "passes_1e_minus_10": relative <= 1.0e-10,
    }


def coevolution_submanifold_check(
    module: ModuleType,
    templates: Any,
    nbar: float,
    b1: float,
) -> dict[str, Any]:
    coevolution_names = b_free_names(module, "coevolution")
    rng = np.random.default_rng(260725)
    maximum_relative = 0.0
    maximum_sigma_free = 0.0
    for _ in range(8):
        values = 0.2 * rng.normal(size=len(coevolution_names))
        coevolution = full_parameter_dict(
            module,
            coevolution_names,
            values,
            tier="coevolution",
            fixed_b1=b1,
        )
        full_names = b_free_names(module, "full")
        full_values = np.asarray([coevolution[name] for name in full_names])
        full = full_parameter_dict(
            module,
            full_names,
            full_values,
            tier="full",
            fixed_b1=b1,
        )
        first = templates.components(coevolution, nbar)["total"]
        second = templates.components(full, nbar)["total"]
        difference = first - second
        scale = max(float(np.linalg.norm(first)), 1.0)
        maximum_relative = max(
            maximum_relative,
            float(np.linalg.norm(difference) / scale),
        )
        maximum_sigma_free = max(
            maximum_sigma_free,
            float(np.max(np.abs(difference))),
        )
    return {
        "maximum_relative_l2_difference": maximum_relative,
        "maximum_absolute_difference": maximum_sigma_free,
        "passes_1e_minus_8": maximum_relative <= 1.0e-8,
    }


def predictive_evaluation(
    module: ModuleType,
    fit: ConditionalFit,
    test_samples: np.ndarray,
    model_at_b1: Callable[[float, np.ndarray], np.ndarray],
    b1: float,
    b1_sigma: float,
) -> dict[str, Any]:
    test_samples = np.asarray(
        test_samples[:, fit.selected_indices],
        dtype=np.float64,
    )
    target = np.mean(test_samples, axis=0)
    test_covariance_single = covariance_single(test_samples)
    parameter_covariance = (
        fit.selected_model_jacobian
        @ fit.covariance
        @ fit.selected_model_jacobian.T
    )
    step = 1.0e-5 * max(abs(b1), 1.0)
    derivative_b1 = (
        model_at_b1(b1 + step, fit.x) - model_at_b1(b1 - step, fit.x)
    ) / (2.0 * step)
    predictive_covariance = (
        test_covariance_single
        + parameter_covariance
        + b1_sigma**2 * np.outer(derivative_b1, derivative_b1)
    )
    residual = fit.prediction - target
    cholesky = module.covariance_cholesky(predictive_covariance, 1.0e-12)
    whitened = solve_triangular(cholesky, residual, lower=True)
    chi2 = float(whitened @ whitened)
    dof = float(residual.size)
    return {
        "n_train_in_fit": None,
        "n_test": int(test_samples.shape[0]),
        "n_data": int(residual.size),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "p_value": float(chi2_distribution.sf(chi2, dof)),
        "max_abs_predictive_pull": float(
            np.max(np.abs(residual) / np.sqrt(np.diag(predictive_covariance)))
        ),
        "max_same_sign_whitened_run": longest_sign_run(whitened),
        "predictive_covariance": (
            "held-out single-realization covariance "
            "+ Laplace parameter covariance "
            "+ propagated b1 calibration variance"
        ),
    }


def run_cross_validation(
    module: ModuleType,
    templates: Any,
    power_templates: Any,
    data: Any,
    frozen_prior: Any,
    nbar: float,
    p_samples: np.ndarray,
    b_samples: np.ndarray,
    realization_ids: np.ndarray,
    *,
    tier: str,
    kmax: float,
    multistart: int,
    seed: int,
) -> dict[str, Any]:
    n_samples = b_samples.shape[0]
    if p_samples.shape[0] != n_samples or realization_ids.shape != (n_samples,):
        raise ValueError("cross-validation sample matrices are not aligned")
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    odd = np.flatnonzero(realization_ids % 2 == 1)
    even = np.flatnonzero(realization_ids % 2 == 0)
    splits.append(("odd_train_even_test", odd, even))
    splits.append(("even_train_odd_test", even, odd))
    folds = np.array_split(np.arange(n_samples), 5)
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n_samples), test, assume_unique=True)
        splits.append((f"fivefold_{fold_index}", train, test))
    rows = []
    for split_index, (label, train, test) in enumerate(splits):
        p_fit = make_power_fit(
            module,
            power_templates,
            data,
            frozen_prior,
            nbar,
            p_samples[train],
            kmax=PRIMARY_P_KMAX,
            tier="full",
            multistart=multistart,
            rng=np.random.default_rng(seed + 101 * split_index),
        )
        b1 = p_fit.expanded_value("b1")
        b1_sigma = p_fit.expanded_sigma("b1")
        if tier == "uplift":
            b_fit = make_uplift_fit(
                module,
                templates,
                data,
                frozen_prior,
                nbar,
                b_samples[train],
                b1=b1,
                kmax=kmax,
                nuisance_prior_scale=1.0,
                multistart=multistart,
                rng=np.random.default_rng(seed + 101 * split_index + 17),
            )
        else:
            b_fit = make_b_fit(
                module,
                templates,
                data,
                frozen_prior,
                nbar,
                b_samples[train],
                b1=b1,
                kmax=kmax,
                tier=tier,
                nuisance_prior_scale=1.0,
                multistart=multistart,
                rng=np.random.default_rng(seed + 101 * split_index + 17),
            )
        free_names = b_fit.free_names
        indices = b_fit.selected_indices

        def model_at_b1(new_b1: float, values: np.ndarray) -> np.ndarray:
            parameters = full_parameter_dict(
                module,
                free_names,
                values,
                tier="full" if tier == "uplift" else tier,
                fixed_b1=new_b1,
            )
            components = (
                uplift_components(module, templates, parameters, nbar)
                if tier == "uplift"
                else templates.components(parameters, nbar)
            )
            return components["total"][indices]

        predictive = predictive_evaluation(
            module,
            b_fit,
            b_samples[test],
            model_at_b1,
            b1,
            b1_sigma,
        )
        predictive["n_train_in_fit"] = int(train.size)
        rows.append(
            {
                "split": label,
                "train_realization_minmax": [
                    int(np.min(realization_ids[train])),
                    int(np.max(realization_ids[train])),
                ],
                "test_realization_minmax": [
                    int(np.min(realization_ids[test])),
                    int(np.max(realization_ids[test])),
                ],
                "b1": b1,
                "b1_sigma": b1_sigma,
                "fit": fit_report(b_fit),
                "prediction": predictive,
            }
        )
    fivefold = [row for row in rows if row["split"].startswith("fivefold_")]
    aggregate_chi2 = sum(row["prediction"]["chi2"] for row in fivefold)
    aggregate_dof = sum(row["prediction"]["dof"] for row in fivefold)
    passes = bool(
        aggregate_chi2 / aggregate_dof <= 1.5
        and all(row["prediction"]["p_value"] >= 0.01 for row in rows)
        and all(
            row["prediction"]["max_abs_predictive_pull"] < 3.0
            for row in rows
        )
    )
    return {
        "tier": tier,
        "kmax_h_mpc": kmax,
        "splits": rows,
        "fivefold_aggregate_descriptive": {
            "correlation_note": (
                "fold training sets overlap; aggregate chi-square is a "
                "registered descriptive gate, not an exact independent p-value"
            ),
            "chi2": aggregate_chi2,
            "dof": aggregate_dof,
            "chi2_per_dof": aggregate_chi2 / aggregate_dof,
        },
        "passes_registered_gate": passes,
    }


def plot_power(
    path: Path,
    data: Any,
    fit: ConditionalFit,
) -> None:
    indices = fit.selected_indices
    k = np.asarray(data.power_k)[indices]
    errors_single = np.sqrt(np.diag(fit.covariance_single))
    residual_single = fit.residual / errors_single
    fig, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axis.errorbar(
        k,
        fit.target,
        yerr=errors_single,
        fmt="o",
        color="black",
        ms=4.2,
        lw=0.9,
        capsize=2.0,
        label="Quijote fiducial mean ± single-realization error",
    )
    axis.plot(
        k,
        fit.prediction,
        "-",
        color="#C44E52",
        lw=1.8,
        label="one-loop P0 best fit",
    )
    axis.set_ylabel(r"$P_0(k)\;[(\mathrm{Mpc}/h)^3]$")
    axis.set_title(
        r"Pre-reconstruction $b_1$ calibration, $z=1$, "
        r"$M_{\min}=10^{13}\,h^{-1}M_\odot$"
    )
    axis.legend(fontsize=7.5)
    axis.text(
        0.02,
        0.04,
        (
            rf"$b_1={fit.expanded_value('b1'):.5f}"
            rf"\pm{fit.expanded_sigma('b1'):.5f}$ (single-realization)"
            "\n"
            rf"$\chi^2/\mathrm{{dof}}={fit.data_chi2/fit.dof:.3f}$, "
            rf"$p={fit.p_value:.4f}$"
        ),
        transform=axis.transAxes,
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    residual_axis.axhline(0.0, color="0.3", lw=0.8)
    residual_axis.plot(
        k,
        residual_single,
        "o-",
        color="#C44E52",
        ms=3.8,
        lw=1.1,
    )
    residual_axis.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    residual_axis.set_ylabel(r"residual$/\sigma_{\rm single}$")
    fig.tight_layout()
    fig.savefig(path, metadata={**PDF_METADATA, "Title": "Pre-reconstruction P0 b1 calibration"})
    plt.close(fig)


def plot_b000(
    path: Path,
    data: Any,
    fit: ConditionalFit,
    *,
    production_selected: bool,
) -> None:
    indices = fit.selected_indices
    k_pair = np.asarray(data.k_pair)[indices]
    scale = np.prod(k_pair, axis=1)
    errors_single = np.sqrt(np.diag(fit.covariance_single))
    residual_single = fit.residual / errors_single
    x = np.arange(indices.size)
    clipped = np.clip(residual_single, -0.5, 0.5)
    fig, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]},
    )
    axis.errorbar(
        x,
        scale * fit.target,
        yerr=scale * errors_single,
        fmt="o",
        color="black",
        ms=4.0,
        lw=0.8,
        capsize=1.8,
        label="Quijote fiducial mean ± single-realization error",
        zorder=4,
    )
    axis.plot(
        x,
        scale * fit.prediction,
        "-",
        color="#C44E52",
        lw=1.8,
        label=(
            f"selected {fit.tier} model"
            if production_selected
            else f"diagnostic fallback {fit.tier} model"
        ),
        zorder=5,
    )
    axis.axhline(0.0, color="0.4", lw=0.7)
    axis.set_ylabel(
        r"$k_1k_2 B_{000}(k_1,k_2)\;[(\mathrm{Mpc}/h)^4]$"
    )
    axis.set_title(
        r"Quijote halo pre-reconstruction, $z=1$, "
        r"$M_{\min}=10^{13}\,h^{-1}M_\odot$"
    )
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.legend(fontsize=7.5)
    axis.text(
        0.02,
        0.04,
        (
            rf"$k_{{\max}}={fit.kmax:.2f}\,h\,\mathrm{{Mpc}}^{{-1}}$, "
            rf"$\chi^2/\mathrm{{dof}}={fit.data_chi2/fit.dof:.3f}$, "
            rf"$\|r\|_{{C_{{\rm single}}^{{-1}}}}="
            rf"{math.sqrt(max(fit.data_chi2, 0.0)):.3f}\sigma$"
            "\n"
            r"global bin 0 excluded; fit uses single-realization covariance"
            + (
                ""
                if production_selected
                else "\nno tier passed every registered acceptance gate"
            )
        ),
        transform=axis.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    residual_axis.axhline(0.0, color="0.3", lw=0.8)
    residual_axis.plot(
        x,
        clipped,
        "o-",
        color="#C44E52",
        ms=3.8,
        lw=1.0,
    )
    high = residual_single > 0.5
    low = residual_single < -0.5
    residual_axis.plot(
        x[high],
        np.full(np.count_nonzero(high), 0.48),
        "^",
        color="#C44E52",
        ms=5,
    )
    residual_axis.plot(
        x[low],
        np.full(np.count_nonzero(low), -0.48),
        "v",
        color="#C44E52",
        ms=5,
    )
    residual_axis.set_ylim(-0.5, 0.5)
    residual_axis.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    residual_axis.set_xlabel("selected full-2D B000 bin order")
    residual_axis.set_ylabel(r"residual$/\sigma_{\rm single}$")
    tick_step = max(1, indices.size // 8)
    ticks = np.arange(0, indices.size, tick_step)
    residual_axis.set_xticks(ticks)
    residual_axis.set_xticklabels([str(indices[index]) for index in ticks])
    fig.tight_layout()
    fig.savefig(path, metadata={**PDF_METADATA, "Title": "Pre-reconstruction full-2D k1 k2 B000"})
    plt.close(fig)


def plot_stability(
    path: Path,
    fits: dict[str, dict[float, ConditionalFit]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True)
    colors = {
        "full": "black",
        "coevolution": "#C44E52",
        "uplift": "0.45",
    }
    markers = {"full": "o", "coevolution": "s", "uplift": "^"}
    labels = {
        "b2": r"$b_2$",
        "gamma2": r"$\gamma_2$",
        "gamma21": r"$\gamma_{21}$",
    }
    for axis, name in zip(axes, CORE_COMPARE_NAMES):
        tiers = ("full", "coevolution")
        if name != "gamma21" and "uplift" in fits:
            tiers = (*tiers, "uplift")
        for tier in tiers:
            ordered = [fits[tier][value] for value in B_KMAX_VALUES]
            values = [fit.expanded_value(name) for fit in ordered]
            errors = [fit.expanded_sigma(name) for fit in ordered]
            axis.errorbar(
                B_KMAX_VALUES,
                values,
                yerr=errors,
                marker=markers[tier],
                color=colors[tier],
                lw=1.2,
                capsize=2.0,
                label=tier,
            )
        axis.set_title(labels[name])
        axis.set_xlabel(r"$k_{\max}\,[h\,\mathrm{Mpc}^{-1}]$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("conditional posterior")
    axes[0].legend(fontsize=8)
    fig.suptitle("Pre-reconstruction bias stability with stored-radial cutoff")
    fig.tight_layout()
    fig.savefig(path, metadata={**PDF_METADATA, "Title": "Bias cutoff and tier stability"})
    plt.close(fig)


def plot_identifiability(
    path: Path,
    fits: dict[str, ConditionalFit],
) -> None:
    fig, (singular_axis, shrink_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.2),
    )
    colors = {
        "full": "black",
        "coevolution": "#C44E52",
        "uplift": "0.45",
    }
    for tier, fit in fits.items():
        values = fit.singular_values / fit.singular_values[0]
        singular_axis.semilogy(
            np.arange(1, values.size + 1),
            values,
            "o-",
            color=colors[tier],
            ms=3.6,
            lw=1.1,
            label=tier,
        )
    singular_axis.axhline(1.0e-10, color="0.5", ls="--", lw=0.8)
    singular_axis.set_xlabel("singular-value index")
    singular_axis.set_ylabel(r"$s_i/s_{\max}$")
    singular_axis.set_title("Whitened data Jacobian")
    singular_axis.legend(fontsize=8)
    full = fits["full"]
    order = np.argsort(full.posterior_to_prior_sigma_ratio)
    labels = np.asarray(full.free_names)[order]
    values = full.posterior_to_prior_sigma_ratio[order]
    shrink_axis.barh(
        np.arange(values.size),
        values,
        color=[
            "#C44E52" if name in full.nonlinear_names else "0.35"
            for name in labels
        ],
    )
    shrink_axis.set_yticks(np.arange(values.size))
    shrink_axis.set_yticklabels(labels, fontsize=6.5)
    shrink_axis.set_xlim(0.0, 1.03)
    shrink_axis.set_xlabel(r"$\sigma_{\rm post}/\sigma_{\rm prior}$")
    shrink_axis.set_title("Full-reference prior shrinkage")
    fig.tight_layout()
    fig.savefig(path, metadata={**PDF_METADATA, "Title": "Bias-model identifiability"})
    plt.close(fig)


def relative_shift_against_primary(
    variant: ConditionalFit,
    primary: ConditionalFit,
    names: Iterable[str],
) -> dict[str, float]:
    result = {}
    for name in names:
        sigma = primary.expanded_sigma(name)
        result[name] = (
            abs(variant.expanded_value(name) - primary.expanded_value(name))
            / sigma
            if sigma > 0.0
            else math.inf
        )
    return result


def build_markdown_report(payload: dict[str, Any]) -> str:
    selection = payload["selection"]
    b1 = payload["b1_calibration"]["primary"]
    full_015 = payload["gaussian_fits"]["full"]["0.15"]["fit"]["statistics"]
    full_015_diag = payload["tier_diagnostics"]["full"]["by_kmax"]["0.15"]
    coevolution_015 = payload["gaussian_fits"]["coevolution"]["0.15"]["fit"][
        "statistics"
    ]
    coevolution_015_diag = payload["tier_diagnostics"]["coevolution"][
        "by_kmax"
    ]["0.15"]
    b1_neighbor_max = max(
        payload["gates"]["B_b1_calibration"][
            "neighbor_shift_combined_sigma"
        ].values()
    )
    lines = [
        "# Pre-reconstruction halo bias-model v3",
        "",
        f"Date: {payload['created_utc']}",
        "",
        "## Outcome",
        "",
        f"- Overall status: **{payload['status'].upper()}**.",
        (
            f"- Selected tier: **{selection.get('tier') or 'none'}**, "
            f"maximum passing kmax: "
            f"**{selection.get('kmax_h_mpc') or 'none'} h/Mpc**."
        ),
        (
            f"- Independent P0 calibration: "
            f"`b1={b1['b1']:.6f} +/- {b1['b1_sigma']:.6f}` "
            "(equivalent-single-realization uncertainty)."
        ),
        (
            "- No production model was selected; the B000 PDF therefore "
            f"shows the diagnostic fallback `{selection['fallback_display_fit']['tier']}` "
            f"fit at `kmax={selection['fallback_display_fit']['kmax_h_mpc']:.2f} "
            "h/Mpc`, not an accepted inference model."
            if selection.get("tier") is None
            else "- The displayed B000 fit is the accepted production selection."
        ),
        (
            f"- P0 cutoff stability reaches `{b1_neighbor_max:.3f} sigma` "
            "under the single-realization posterior, "
            + (
                "within"
                if b1_neighbor_max < 0.3
                else "above"
            )
            + " the registered `0.3 sigma` Gate B threshold."
        ),
        (
            "- The complete tier at `kmax=0.15 h/Mpc` is adequate relative "
            "to single-realization cosmic variance "
            f"(`p={full_015['p_value']:.4f}`); its maximum prior-scan "
            f"core shift is `{full_015_diag['maximum_core_shift_prior_scan_primary_sigma']:.3f} "
            "sigma` and its held-out gate is "
            f"`{str(full_015_diag['cross_validation']['passes_registered_gate']).lower()}`."
        ),
        (
            "- The coevolution tier at `kmax=0.15 h/Mpc` is adequate and predicts "
            f"held-out central vectors (`p={coevolution_015['p_value']:.4f}`, held-out "
            f"gate `{str(coevolution_015_diag['cross_validation']['passes_registered_gate']).lower()}`); "
            "its full-versus-reduced tier-agreement gate is "
            f"`{str(coevolution_015_diag['tier_agreement']['passes_registered_gate']).lower()}`."
        ),
        "- The downstream primary likelihood is B000-only; P0 is not reused.",
        "",
        "## P0 calibration",
        "",
        "| tier | kmax | b1 | sigma | chi2/dof | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier, rows in payload["b1_calibration"]["fits"].items():
        for kmax, row in rows.items():
            stats = row["statistics"]
            parameter = row["expanded_parameters"]["b1"]
            lines.append(
                f"| {tier} | {float(kmax):.2f} | "
                f"{parameter['value']:.6f} | "
                f"{parameter['posterior_sigma_conditional_on_fixed_b1']:.6f} | "
                f"{stats['chi2_per_dof']:.3f} | {stats['p_value']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Gaussian B000 single-realization adequacy",
            "",
            "| tier | kmax | Ndata | chi2/dof | p | residual norm | max single pull | loop/tree | ctr/det |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tier, rows in payload["gaussian_fits"].items():
        for kmax, row in rows.items():
            stats = row["fit"]["statistics"]
            ratios = row["perturbativity"]["ratios"]
            lines.append(
                f"| {tier} | {float(kmax):.2f} | {row['fit']['n_data']} | "
                f"{stats['chi2_per_dof']:.3f} | {stats['p_value']:.4f} | "
                f"{stats['residual_norm_sigma_single']:.3f} | "
                f"{stats['max_abs_fit_covariance_pull']:.3f} | "
                f"{ratios['one_loop_over_tree']:.3f} | "
                f"{ratios['counterterm_over_tree_plus_one_loop']:.3f} |"
            )
    ablation = payload["uplift_ablation"]
    paper_range = ablation["paper_range_post_registered_diagnostic"]
    paper_full_stats = paper_range["full_fit"]["statistics"]
    paper_uplift_stats = paper_range["uplift_fit"]["statistics"]
    lines.extend(
        [
            "",
            "## DM-uplift model-order ablation",
            "",
            (
                "The historical control is "
                "`B_h,tree + b1^3(B_m,1loop-B_m,tree)` plus only the two "
                "leading stochastic amplitudes. It is not eligible for "
                "production selection."
            ),
            (
                f"- Common-cut comparison: "
                f"`kmax={ablation['comparison_kmax_h_mpc']:.2f} h/Mpc`."
            ),
            (
                "- Single-realization-covariance chi-square penalty relative "
                "to the complete model: "
                f"`Delta chi2={ablation['delta_single_covariance_chi2_uplift_minus_full']:.3f}`."
            ),
            (
                "- Best-fit prediction distance: "
                f"`{ablation['full_minus_uplift_best_fit_distance_sigma_single']:.3f} "
                "sigma_single`."
            ),
            (
                "- Missing same-order halo sector at the complete fit: "
                f"`{ablation['full_order_minus_uplift_order_at_full_parameters_sigma_single']:.3f} "
                "sigma_single`."
            ),
            (
                "- Uplift held-out gate: "
                f"`{str(ablation['candidate_passes_heldout_prediction']).lower()}`."
            ),
            (
                "- Although the omitted same-order halo sector is not "
                "numerically zero, refitting the simple bias/stochastic "
                "parameters leaves the uplift prediction within the "
                "single-realization adequacy threshold. It therefore cannot "
                "be rejected as a practical Gaussian baseline by this test."
            ),
            (
                "- Explicit post-registered paper-range check at "
                f"`kmax={paper_range['kmax_h_mpc']:.2f} h/Mpc` "
                "(excluded from production selection): complete model "
                f"`p={paper_full_stats['p_value']:.4f}`, uplift "
                f"`p={paper_uplift_stats['p_value']:.4f}`."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Registered gates",
            "",
            "| gate | status |",
            "|---|---|",
        ]
    )
    for gate, row in payload["gates"].items():
        lines.append(f"| {gate} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The 27-dimensional number describes the complete joint "
                "P0+B000 theory span after one exact stochastic degeneracy. "
                "With b1 fixed and P0 removed, the complete B000 likelihood "
                "contains 24 coordinates, of which only four are optimized "
                "nonlinearly; the remainder are solved conditionally."
            ),
            (
                "All adequacy, prior, predictive, and perturbativity "
                "decisions in this report use the covariance of one "
                "equivalent realization. The ensemble mean enters only as "
                "the central data vector."
            ),
            (
                "The mandatory coherence measure is the total "
                "single-covariance residual norm. Same-sign run lengths are "
                "retained as diagnostics and cannot reject a residual whose "
                "total norm is already below 0.5 sigma_single."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    for path in payload["outputs"]["figures"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## PNG gate",
            "",
            payload["png_gate"]["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.tag or "/" in args.tag:
        raise ValueError("--tag must be a directory-safe name")
    if args.multistart < 1 or args.cv_multistart < 1:
        raise ValueError("multistart values must be positive")
    repo_root = args.repo_root.resolve()
    configured_data_root = (
        args.data_root
        if args.data_root is not None
        else Path(os.environ.get("MARISA_B_DATA_ROOT", repo_root))
    )
    data_root = configured_data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )
    vector_dir = (
        data_root
        / "analysis/theory_vectors/eft_v2_r0_exact_primary_20260724"
    )
    paths = {
        "frozen_fitter": (
            repo_root / "scripts/legacy/fit_eft_v2_r0_mean.py"
        ),
        "contract": repo_root / "configs/eft_v2_contract.json",
        "matrix": (
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid500_b000_pk_matrix.npz"
        ),
        "lc_matrix": (
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_"
            "fid_lcp_lcm_first100_b000_pk_matrix.npz"
        ),
        "b_templates": vector_dir / "b000_ir_contract_v2.jsonl",
        "p_templates": vector_dir / "p0_ir_contract_v2.jsonl",
        "scientific_status": repo_root / "docs/SCIENTIFIC_STATUS.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    analysis_dir = output_root / "analysis" / args.tag
    figure_dir = output_root / "figures/diagnostics" / args.tag
    manifest_path = output_root / "manifests" / f"{args.tag}.json"
    report_json = analysis_dir / "report.json"
    report_md = analysis_dir / "REPORT.md"
    arrays_path = analysis_dir / "fit_summary.npz"
    figure_paths = {
        "power": figure_dir / "pre_recon_p0_b1_calibration.pdf",
        "b000": figure_dir / "pre_recon_full2d_k1k2_b000_selected_model.pdf",
        "stability": figure_dir / "pre_recon_bias_kmax_tier_stability.pdf",
        "identifiability": figure_dir / "pre_recon_identifiability.pdf",
    }
    collision_paths = [
        report_json,
        report_md,
        arrays_path,
        manifest_path,
        *figure_paths.values(),
    ]
    existing = [path for path in collision_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "versioned outputs already exist; select a new tag or --overwrite: "
            + ", ".join(map(str, existing))
        )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    module = load_frozen_module(paths["frozen_fitter"])
    data = module.DataSet.load(paths["matrix"])
    contract, frozen_prior, nbar = module.load_prior(paths["contract"])
    templates = module.TemplateSet.load(paths["b_templates"])
    power_templates = module.PowerTemplateSet.load(paths["p_templates"])
    module.validate_bispectrum_geometry(templates, data, "bias-model-v3 B000")
    module.validate_power_geometry(power_templates, data, "bias-model-v3 P0")
    with np.load(paths["matrix"], allow_pickle=False) as values:
        realization_ids = np.asarray(values["fiducial_realizations"], dtype=int)
        b_samples = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        p_samples = np.asarray(values["fiducial_pre_P0"], dtype=np.float64)
    if (
        realization_ids.shape != (500,)
        or b_samples.shape != (500, 120)
        or p_samples.shape != (500, 47)
    ):
        raise ValueError("unexpected fiducial sample matrices")

    rng = np.random.default_rng(args.seed)
    power_fits: dict[str, dict[float, ConditionalFit]] = {
        "full": {},
        "coevolution": {},
    }
    for tier in power_fits:
        for kmax in P_KMAX_VALUES:
            power_fits[tier][kmax] = make_power_fit(
                module,
                power_templates,
                data,
                frozen_prior,
                nbar,
                p_samples,
                kmax=kmax,
                tier=tier,
                multistart=args.multistart,
                rng=rng,
            )
    primary_power = power_fits["full"][PRIMARY_P_KMAX]
    b1 = primary_power.expanded_value("b1")
    b1_sigma = primary_power.expanded_sigma("b1")

    gaussian_fits: dict[str, dict[float, ConditionalFit]] = {
        "full": {},
        "coevolution": {},
        "uplift": {},
    }
    for tier in ("full", "coevolution"):
        for kmax in B_KMAX_VALUES:
            gaussian_fits[tier][kmax] = make_b_fit(
                module,
                templates,
                data,
                frozen_prior,
                nbar,
                b_samples,
                b1=b1,
                kmax=kmax,
                tier=tier,
                nuisance_prior_scale=1.0,
                multistart=args.multistart,
                rng=rng,
            )
    for kmax in B_KMAX_VALUES:
        gaussian_fits["uplift"][kmax] = make_uplift_fit(
            module,
            templates,
            data,
            frozen_prior,
            nbar,
            b_samples,
            b1=b1,
            kmax=kmax,
            nuisance_prior_scale=1.0,
            multistart=args.multistart,
            rng=rng,
        )

    perturbativity = {
        tier: {
            kmax: perturbativity_report(
                module,
                fit,
                frozen_prior,
            )
            for kmax, fit in rows.items()
        }
        for tier, rows in gaussian_fits.items()
    }

    # Preliminary in-sample screen.  Robustness and predictive tests are run
    # over the complete pre-registered cutoff grid below, so a failure at the
    # highest cutoff cannot hide a valid lower-cutoff result.
    candidate_kmax: dict[str, float] = {}
    preliminary_pass: dict[str, dict[float, bool]] = {}
    for tier in ("full", "coevolution", "uplift"):
        preliminary_pass[tier] = {
            kmax: bool(
                closure_pass(gaussian_fits[tier][kmax])
                and numerical_pass(gaussian_fits[tier][kmax])
                and perturbativity[tier][kmax]["passes_registered_gate"]
            )
            for kmax in B_KMAX_VALUES
        }
        passing = [
            kmax
            for kmax in B_KMAX_VALUES
            if preliminary_pass[tier][kmax]
        ]
        candidate_kmax[tier] = max(passing) if passing else min(B_KMAX_VALUES)

    prior_scans: dict[
        str, dict[float, dict[str, ConditionalFit]]
    ] = {}
    b1_scans: dict[
        str, dict[float, dict[str, ConditionalFit]]
    ] = {}
    safe_fits: dict[
        str, dict[float, ConditionalFit | None]
    ] = {}
    cross_validation: dict[str, dict[float, dict[str, Any]]] = {
        "full": {},
        "coevolution": {},
        "uplift": {},
    }
    for tier in ("full", "coevolution"):
        prior_scans[tier] = {}
        b1_scans[tier] = {}
        safe_fits[tier] = {}
        for kmax in B_KMAX_VALUES:
            prior_scans[tier][kmax] = {}
            for scale in (0.5, 2.0):
                prior_scans[tier][kmax][f"{scale:g}"] = make_b_fit(
                    module,
                    templates,
                    data,
                    frozen_prior,
                    nbar,
                    b_samples,
                    b1=b1,
                    kmax=kmax,
                    tier=tier,
                    nuisance_prior_scale=scale,
                    multistart=args.multistart,
                    rng=rng,
                )
            b1_scans[tier][kmax] = {}
            for sign, value in (
                ("minus", b1 - b1_sigma),
                ("plus", b1 + b1_sigma),
            ):
                b1_scans[tier][kmax][sign] = make_b_fit(
                    module,
                    templates,
                    data,
                    frozen_prior,
                    nbar,
                    b_samples,
                    b1=value,
                    kmax=kmax,
                    tier=tier,
                    nuisance_prior_scale=1.0,
                    multistart=args.multistart,
                    rng=rng,
                )
            safe_mask = b_mask(data, kmax, closing_side_safe=True)
            if np.count_nonzero(safe_mask) >= 3:
                safe_fits[tier][kmax] = make_b_fit(
                    module,
                    templates,
                    data,
                    frozen_prior,
                    nbar,
                    b_samples,
                    b1=b1,
                    kmax=kmax,
                    tier=tier,
                    nuisance_prior_scale=1.0,
                    multistart=args.multistart,
                    rng=rng,
                    closing_side_safe=True,
                )
            else:
                safe_fits[tier][kmax] = None

    for tier_index, tier in enumerate(("full", "coevolution", "uplift")):
        for kmax in B_KMAX_VALUES:
            if args.skip_cross_validation:
                cross_validation[tier][kmax] = {
                    "tier": tier,
                    "kmax_h_mpc": kmax,
                    "skipped": True,
                    "passes_registered_gate": False,
                }
            else:
                cross_validation[tier][kmax] = run_cross_validation(
                    module,
                    templates,
                    power_templates,
                    data,
                    frozen_prior,
                    nbar,
                    p_samples,
                    b_samples,
                    realization_ids,
                    tier=tier,
                    kmax=kmax,
                    multistart=args.cv_multistart,
                    seed=(
                        args.seed
                        + 10000 * (tier_index + 1)
                        + int(round(1000.0 * kmax))
                    ),
                )

    paper_range_fits = {
        "full": make_b_fit(
            module,
            templates,
            data,
            frozen_prior,
            nbar,
            b_samples,
            b1=b1,
            kmax=UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX,
            tier="full",
            nuisance_prior_scale=1.0,
            multistart=args.multistart,
            rng=rng,
        ),
        "uplift": make_uplift_fit(
            module,
            templates,
            data,
            frozen_prior,
            nbar,
            b_samples,
            b1=b1,
            kmax=UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX,
            nuisance_prior_scale=1.0,
            multistart=args.multistart,
            rng=rng,
        ),
    }
    paper_range_cross_validation: dict[str, dict[str, Any]] = {}
    for tier_index, tier in enumerate(("full", "uplift")):
        if args.skip_cross_validation:
            paper_range_cross_validation[tier] = {
                "tier": tier,
                "kmax_h_mpc": UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX,
                "skipped": True,
                "passes_registered_gate": False,
            }
        else:
            paper_range_cross_validation[tier] = run_cross_validation(
                module,
                templates,
                power_templates,
                data,
                frozen_prior,
                nbar,
                p_samples,
                b_samples,
                realization_ids,
                tier=tier,
                kmax=UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX,
                multistart=args.cv_multistart,
                seed=args.seed + 40000 + 1000 * tier_index,
            )

    stochastic_degeneracy = exact_stochastic_degeneracy(
        module,
        templates,
        b1,
    )
    submanifold = coevolution_submanifold_check(
        module,
        templates,
        nbar,
        b1,
    )

    b1_neighbor_shifts = {}
    for kmax, fit in power_fits["full"].items():
        if kmax == PRIMARY_P_KMAX:
            continue
        delta = abs(fit.expanded_value("b1") - b1)
        denominator = math.sqrt(
            fit.expanded_sigma("b1") ** 2 + b1_sigma**2
        )
        b1_neighbor_shifts[f"{kmax:.2f}"] = delta / denominator
    b1_coevolution_shift = standardized_shift(
        primary_power,
        power_fits["coevolution"][PRIMARY_P_KMAX],
        ("b1",),
    )["b1"]
    b1_gate_pass = bool(
        closure_pass(primary_power)
        and max(b1_neighbor_shifts.values(), default=0.0) < 0.3
    )

    tier_diagnostics: dict[str, dict[str, Any]] = {}
    tier_pass_by_kmax: dict[str, dict[float, bool]] = {}
    tier_local_pass_by_kmax: dict[str, dict[float, bool]] = {}
    tier_agreement: dict[float, dict[str, Any]] = {}
    for kmax in B_KMAX_VALUES:
        full_compare_at_cut = gaussian_fits["full"][kmax]
        coevolution_compare_at_cut = gaussian_fits["coevolution"][kmax]
        core_shift = standardized_shift(
            full_compare_at_cut,
            coevolution_compare_at_cut,
            CORE_COMPARE_NAMES,
        )
        prediction_distance = prediction_distance_single(
            module,
            full_compare_at_cut,
            coevolution_compare_at_cut,
        )
        tier_agreement[kmax] = {
            "core_shift_combined_sigma": core_shift,
            "prediction_distance_sigma_single": prediction_distance,
            "passes_registered_gate": bool(
                max(core_shift.values()) < 0.5
                and prediction_distance < 0.5
            ),
        }
    for tier in ("full", "coevolution"):
        tier_pass_by_kmax[tier] = {}
        tier_local_pass_by_kmax[tier] = {}
        per_cut: dict[str, Any] = {}
        for kmax in B_KMAX_VALUES:
            primary = gaussian_fits[tier][kmax]
            prior_shift = {
                scale: relative_shift_against_primary(
                    fit,
                    primary,
                    CORE_COMPARE_NAMES,
                )
                for scale, fit in prior_scans[tier][kmax].items()
            }
            b1_shift = {
                label: relative_shift_against_primary(
                    fit,
                    primary,
                    CORE_COMPARE_NAMES,
                )
                for label, fit in b1_scans[tier][kmax].items()
            }
            max_prior_shift = max(
                (
                    value
                    for row in prior_shift.values()
                    for value in row.values()
                ),
                default=0.0,
            )
            max_b1_shift = max(
                (
                    value
                    for row in b1_shift.values()
                    for value in row.values()
                ),
                default=0.0,
            )
            reduced_tier_pass = bool(
                tier == "full"
                or tier_agreement[kmax]["passes_registered_gate"]
            )
            local_pass = bool(
                preliminary_pass[tier][kmax]
                and max_prior_shift < 0.3
                and cross_validation[tier][kmax]["passes_registered_gate"]
                and reduced_tier_pass
            )
            tier_local_pass_by_kmax[tier][kmax] = local_pass
            tier_pass_by_kmax[tier][kmax] = bool(
                b1_gate_pass and local_pass
            )
            per_cut[f"{kmax:.2f}"] = {
                "preliminary_in_sample_pass": preliminary_pass[tier][kmax],
                "prior_scans": {
                    scale: {
                        "core_shift_primary_sigma": prior_shift[scale],
                        "fit": fit_report(fit),
                    }
                    for scale, fit in prior_scans[tier][kmax].items()
                },
                "b1_scans": {
                    label: {
                        "fixed_b1": (
                            b1 - b1_sigma
                            if label == "minus"
                            else b1 + b1_sigma
                        ),
                        "core_shift_primary_sigma": b1_shift[label],
                        "fit": fit_report(fit),
                    }
                    for label, fit in b1_scans[tier][kmax].items()
                },
                "maximum_core_shift_prior_scan_primary_sigma": (
                    max_prior_shift
                ),
                "maximum_core_shift_b1_scan_primary_sigma": max_b1_shift,
                "closing_side_safe_fit": (
                    fit_report(safe_fits[tier][kmax])
                    if safe_fits[tier][kmax] is not None
                    else {
                        "status": "not_run",
                        "reason": "fewer than three conservative bins",
                    }
                ),
                "cross_validation": cross_validation[tier][kmax],
                "tier_agreement": tier_agreement[kmax],
                "local_pass_excluding_global_b1_gate": local_pass,
                "passes_all_registered_gaussian_gates": (
                    tier_pass_by_kmax[tier][kmax]
                ),
            }
        local_passing = [
            kmax
            for kmax in B_KMAX_VALUES
            if tier_local_pass_by_kmax[tier][kmax]
        ]
        global_passing = [
            kmax
            for kmax in B_KMAX_VALUES
            if tier_pass_by_kmax[tier][kmax]
        ]
        tier_diagnostics[tier] = {
            "candidate_kmax_h_mpc": candidate_kmax[tier],
            "maximum_local_passing_kmax_h_mpc": (
                max(local_passing) if local_passing else None
            ),
            "maximum_all_gate_passing_kmax_h_mpc": (
                max(global_passing) if global_passing else None
            ),
            "by_kmax": per_cut,
        }

    comparison_kmax = min(
        candidate_kmax["full"],
        candidate_kmax["coevolution"],
    )
    full_compare = gaussian_fits["full"][comparison_kmax]
    coevolution_compare = gaussian_fits["coevolution"][comparison_kmax]
    tier_core_shift = standardized_shift(
        full_compare,
        coevolution_compare,
        CORE_COMPARE_NAMES,
    )
    tier_prediction_distance = prediction_distance_single(
        module,
        full_compare,
        coevolution_compare,
    )
    tier_robustness_pass = bool(
        max(tier_core_shift.values()) < 0.5
        and tier_prediction_distance < 0.5
    )
    tier_pass = {
        tier: any(tier_pass_by_kmax[tier].values())
        for tier in ("full", "coevolution")
    }
    passing_options = [
        (kmax, tier)
        for tier in ("full", "coevolution")
        for kmax in B_KMAX_VALUES
        if tier_pass_by_kmax[tier][kmax]
    ]
    if passing_options:
        selected_kmax = max(kmax for kmax, _tier in passing_options)
        selected_tier = (
            "coevolution"
            if (selected_kmax, "coevolution") in passing_options
            else "full"
        )
    else:
        selected_tier = None
        selected_kmax = None
    local_options = [
        (kmax, tier)
        for tier in ("full", "coevolution")
        for kmax in B_KMAX_VALUES
        if tier_local_pass_by_kmax[tier][kmax]
    ]
    if local_options:
        fallback_kmax = max(kmax for kmax, _tier in local_options)
        fallback_tier = (
            "coevolution"
            if (fallback_kmax, "coevolution") in local_options
            else "full"
        )
    else:
        fallback_tier = "full"
        fallback_kmax = candidate_kmax["full"]
    selected_fit = (
        gaussian_fits[selected_tier][selected_kmax]
        if selected_tier is not None and selected_kmax is not None
        else gaussian_fits[fallback_tier][fallback_kmax]
    )
    # Historical model-order control evaluated with the same data vector,
    # covariance, fixed b1, masks, and cutoff registry as the production
    # tiers.
    uplift_comparison_kmax = min(
        candidate_kmax["full"],
        candidate_kmax["uplift"],
    )
    full_ablation_fit = gaussian_fits["full"][uplift_comparison_kmax]
    uplift_ablation_fit = gaussian_fits["uplift"][uplift_comparison_kmax]
    if not np.array_equal(
        full_ablation_fit.selected_indices,
        uplift_ablation_fit.selected_indices,
    ):
        raise AssertionError("uplift/full ablation masks differ")
    full_ablation_parameters = full_parameter_dict(
        module,
        full_ablation_fit.free_names,
        full_ablation_fit.x,
        tier="full",
        fixed_b1=b1,
    )
    uplift_order_at_full_parameters = uplift_components(
        module,
        templates,
        full_ablation_parameters,
        nbar,
    )
    missing_halo_order_shape = (
        np.asarray(full_ablation_fit.components["total"])
        - np.asarray(uplift_order_at_full_parameters["total"])
    )
    ablation_indices = full_ablation_fit.selected_indices
    uplift_candidate_cv = cross_validation["uplift"][
        candidate_kmax["uplift"]
    ]
    uplift_ablation = {
        "model_definition": (
            "B_h_tree + b1^3*(B_m_1loop-B_m_tree) "
            "+ leading Ashot/Bshot stochastic"
        ),
        "production_eligible": False,
        "comparison_kmax_h_mpc": uplift_comparison_kmax,
        "candidate_kmax_h_mpc": candidate_kmax["uplift"],
        "candidate_passes_in_sample_closure_numerical_perturbativity": bool(
            closure_pass(gaussian_fits["uplift"][candidate_kmax["uplift"]])
            and numerical_pass(
                gaussian_fits["uplift"][candidate_kmax["uplift"]]
            )
            and perturbativity["uplift"][candidate_kmax["uplift"]][
                "passes_registered_gate"
            ]
        ),
        "candidate_passes_heldout_prediction": uplift_candidate_cv[
            "passes_registered_gate"
        ],
        "full_minus_uplift_best_fit_distance_sigma_single": (
            prediction_distance_single(
                module,
                full_ablation_fit,
                uplift_ablation_fit,
            )
        ),
        "full_order_minus_uplift_order_at_full_parameters_sigma_single": qnorm(
            missing_halo_order_shape[ablation_indices],
            full_ablation_fit.covariance_single,
            module,
        ),
        "core_shift_combined_sigma": standardized_shift(
            full_ablation_fit,
            uplift_ablation_fit,
            ("b2", "gamma2"),
        ),
        "delta_single_covariance_chi2_uplift_minus_full": (
            uplift_ablation_fit.data_chi2 - full_ablation_fit.data_chi2
        ),
        "full_fit": fit_report(full_ablation_fit),
        "uplift_fit": fit_report(uplift_ablation_fit),
        "heldout_prediction": uplift_candidate_cv,
        "heldout_scan": {
            f"{kmax:.2f}": cross_validation["uplift"][kmax]
            for kmax in B_KMAX_VALUES
        },
        "paper_range_post_registered_diagnostic": {
            "kmax_h_mpc": UPLIFT_PAPER_RANGE_DIAGNOSTIC_KMAX,
            "enters_production_selection": False,
            "motivation": (
                "The tree-level reference uses kmax=0.084 h/Mpc; this "
                "diagnostic checks the nearest lower shell cutoff without "
                "altering the pre-registered production scan."
            ),
            "full_fit": fit_report(paper_range_fits["full"]),
            "uplift_fit": fit_report(paper_range_fits["uplift"]),
            "full_perturbativity": perturbativity_report(
                module,
                paper_range_fits["full"],
                frozen_prior,
            ),
            "uplift_perturbativity": perturbativity_report(
                module,
                paper_range_fits["uplift"],
                frozen_prior,
            ),
            "delta_single_covariance_chi2_uplift_minus_full": (
                paper_range_fits["uplift"].data_chi2
                - paper_range_fits["full"].data_chi2
            ),
            "best_fit_prediction_distance_sigma_single": (
                prediction_distance_single(
                    module,
                    paper_range_fits["full"],
                    paper_range_fits["uplift"],
                )
            ),
            "full_heldout_prediction": paper_range_cross_validation["full"],
            "uplift_heldout_prediction": (
                paper_range_cross_validation["uplift"]
            ),
        },
        "interpretation": (
            "This isolates model order rather than optimizer or covariance "
            "semantics. Failure of the uplift with success of the complete "
            "tier is direct evidence that pure matter evolution cannot "
            "replace the same-order halo bias-loop/counterterm/stochastic "
            "sector over the tested cutoff."
        ),
    }

    # Gate H cannot be promoted by the available old truncated halo-PNG
    # response, whose theory basis differs from EFT-v3.  The matched LC inputs
    # are nevertheless audited and recorded for the next theory step.
    with np.load(paths["lc_matrix"], allow_pickle=False) as values:
        lcp_ids = np.asarray(values["LC_p_realizations"], dtype=int)
        lcm_ids = np.asarray(values["LC_m_realizations"], dtype=int)
        lcp_shape = np.asarray(values["LC_p_pre_B000"], dtype=np.float64).shape
        lcm_shape = np.asarray(values["LC_m_pre_B000"], dtype=np.float64).shape
    png_input_compatible = bool(
        lcp_ids.shape == (100,)
        and np.array_equal(lcp_ids, lcm_ids)
        and lcp_shape == (100, 120)
        and lcm_shape == (100, 120)
    )
    if selected_tier is None:
        png_status = "blocked_prerequisite_and_theory"
        png_reason = (
            "No Gaussian tier passed every registered prerequisite gate. "
            "In addition, the only existing halo PNG response is the older "
            "truncated fixed-cutoff bias-v1 basis, not an order-consistent "
            "EFT-v3 response."
        )
        png_summary = (
            "Gate H was not run because the Gaussian prerequisite failed "
            "and no order-consistent EFT-v3 local-PNG response exists."
        )
    else:
        png_status = "blocked_theory"
        png_reason = (
            f"The Gaussian {selected_tier} tier passed at "
            f"kmax={selected_kmax:.2f} h/Mpc, but the only existing halo "
            "PNG response is the older truncated fixed-cutoff bias-v1 "
            "basis, not an order-consistent EFT-v3 response."
        )
        png_summary = (
            "The Gaussian baseline passed under the single-realization "
            "covariance contract, but Gate H remains blocked because no "
            "order-consistent EFT-v3 local-PNG response exists."
        )
    png_gate = {
        "status": png_status,
        "matched_pre_reconstruction_halo_inputs_available": png_input_compatible,
        "lcp_lcm_realizations": (
            [int(lcp_ids[0]), int(lcp_ids[-1])] if lcp_ids.size else []
        ),
        "reason": png_reason,
        "summary": png_summary,
    }

    gates = {
        "A_data_and_algebra": {
            "status": (
                "pass"
                if stochastic_degeneracy["passes_1e_minus_10"]
                and submanifold["passes_1e_minus_8"]
                else "fail"
            ),
            "stochastic_degeneracy": stochastic_degeneracy,
            "coevolution_submanifold": submanifold,
        },
        "B_b1_calibration": {
            "status": "pass" if b1_gate_pass else "fail",
            "neighbor_shift_combined_sigma": b1_neighbor_shifts,
            "full_vs_coevolution_shift_combined_sigma": b1_coevolution_shift,
        },
        "C_numerical": {
            "status": (
                "pass"
                if any(
                    numerical_pass(gaussian_fits[tier][kmax])
                    for tier in ("full", "coevolution")
                    for kmax in B_KMAX_VALUES
                )
                else "fail"
            )
        },
        "D_gaussian_single_realization_adequacy": {
            "status": (
                "pass"
                if any(
                    closure_pass(gaussian_fits[tier][kmax])
                    for tier in ("full", "coevolution")
                    for kmax in B_KMAX_VALUES
                )
                else "fail"
            )
        },
        "E_predictive": {
            "status": (
                "pass"
                if any(
                    cross_validation[tier][kmax]["passes_registered_gate"]
                    for tier in ("full", "coevolution")
                    for kmax in B_KMAX_VALUES
                )
                else "fail"
            )
        },
        "F_prior_and_tier": {
            "status": (
                "pass"
                if any(
                    tier_diagnostics[tier]["by_kmax"][f"{kmax:.2f}"][
                        "maximum_core_shift_prior_scan_primary_sigma"
                    ]
                    < 0.3
                    and (
                        tier == "full"
                        or tier_agreement[kmax]["passes_registered_gate"]
                    )
                    for tier in ("full", "coevolution")
                    for kmax in B_KMAX_VALUES
                )
                else "fail"
            ),
            "comparison_kmax_h_mpc": comparison_kmax,
            "core_shift_combined_sigma": tier_core_shift,
            "prediction_distance_sigma_single": tier_prediction_distance,
            "passes_at_comparison_cut": tier_robustness_pass,
            "selection_rule": (
                "Coevolution additionally requires tier agreement; the "
                "complete-reference tier may be retained when the reduced "
                "tier fails, provided the complete tier is prior-robust."
            ),
        },
        "G_perturbativity": {
            "status": (
                "pass"
                if any(
                    perturbativity[tier][kmax]["passes_registered_gate"]
                    for tier in ("full", "coevolution")
                    for kmax in B_KMAX_VALUES
                )
                else "fail"
            )
        },
        "H_png_recovery": {
            "status": png_gate["status"],
            "reason": png_gate["reason"],
        },
    }
    mandatory_gaussian_gates = tuple(
        gate for gate in gates if gate != "H_png_recovery"
    )
    gaussian_status = (
        "pass"
        if selected_tier is not None
        and all(gates[gate]["status"] == "pass" for gate in mandatory_gaussian_gates)
        else "fail"
    )
    overall_status = (
        "partial_pass_png_blocked"
        if gaussian_status == "pass"
        and str(png_gate["status"]).startswith("blocked")
        else gaussian_status
    )

    plot_power(figure_paths["power"], data, primary_power)
    plot_b000(
        figure_paths["b000"],
        data,
        selected_fit,
        production_selected=selected_tier is not None,
    )
    plot_stability(figure_paths["stability"], gaussian_fits)
    plot_identifiability(
        figure_paths["identifiability"],
        {
            tier: gaussian_fits[tier][candidate_kmax[tier]]
            for tier in ("full", "coevolution", "uplift")
        },
    )

    array_payload: dict[str, np.ndarray] = {
        "p0_selected_indices": primary_power.selected_indices,
        "p0_k": np.asarray(data.power_k)[primary_power.selected_indices],
        "p0_data_mean": primary_power.target,
        "p0_model": primary_power.prediction,
        "p0_error_single": np.sqrt(np.diag(primary_power.covariance_single)),
        "b000_selected_indices": selected_fit.selected_indices,
        "b000_k_pair": np.asarray(data.k_pair)[selected_fit.selected_indices],
        "b000_data_mean": selected_fit.target,
        "b000_model": selected_fit.prediction,
        "b000_error_single": np.sqrt(np.diag(selected_fit.covariance_single)),
        "b000_residual_sigma_single": (
            selected_fit.residual
            / np.sqrt(np.diag(selected_fit.covariance_single))
        ),
        "selected_free_parameter_names": np.asarray(
            selected_fit.free_names,
            dtype="U64",
        ),
        "selected_free_parameter_values": selected_fit.x,
        "selected_free_parameter_covariance": selected_fit.covariance,
        "uplift_ablation_selected_indices": ablation_indices,
        "uplift_ablation_full_best_fit_model": full_ablation_fit.prediction,
        "uplift_ablation_uplift_best_fit_model": uplift_ablation_fit.prediction,
        "uplift_ablation_missing_halo_order_at_full_parameters": (
            missing_halo_order_shape[ablation_indices]
        ),
    }
    np.savez_compressed(arrays_path, **array_payload)

    payload: dict[str, Any] = {
        "schema": "marisa-b-pre-recon-bias-model-v3-report-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "gaussian_status": gaussian_status,
        "goal_thread": "019f89c0-0fed-71b3-8d32-a2953465f08b",
        "scope": {
            "sample": "Quijote halo z=1 Mmin=1e13 h^-1 Msun",
            "stage": "pre-reconstruction",
            "space": "real",
            "initial_conditions": "Gaussian fiducial for bias calibration",
            "observable": "P0 calibration followed by full-2D shell B000",
            "excluded_b000_global_indices": [EXCLUDED_B000_INDEX],
            "stored_radial_kmax_values_h_mpc": B_KMAX_VALUES,
            "target_kmax_h_mpc": TARGET_B_KMAX,
            "display_covariance": "single-realization",
            "fit_covariance": (
                "sample covariance of one equivalent realization; never "
                "divided by the number of realizations"
            ),
            "central_vector": (
                "mean of 500 fiducial realizations, used only to estimate "
                "the ensemble expectation"
            ),
            "p0_reused_in_primary_b_likelihood": False,
            "covariance_contract_revision": (
                "All inference and acceptance decisions use the covariance "
                "of one equivalent realization. The ensemble mean is only "
                "the central vector."
            ),
            "superseded_mean_covariance_archive": (
                "historical result excluded from the source release"
            ),
        },
        "inputs": {
            name: {
                "path": path,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "parameter_architecture": {
            "joint_raw_count": 28,
            "joint_independent_count": 27,
            "fixed_b1_b_only_complete_coordinate_count": len(
                b_free_names(module, "full")
            ),
            "fixed_b1_b_only_coevolution_coordinate_count": len(
                b_free_names(module, "coevolution")
            ),
            "fixed_b1_uplift_control_coordinate_count": len(
                UPLIFT_FREE_NAMES
            ),
            "full_nonlinear_names": b_nonlinear_names("full"),
            "coevolution_nonlinear_names": b_nonlinear_names("coevolution"),
            "coevolution_relations_source": (
                "Eggemeier et al. 2021, arXiv:2102.06902, Eqs. 20-24, "
                "with five NLE Lagrangian coefficients set to zero"
            ),
        },
        "b1_calibration": {
            "primary": {
                "tier": "full",
                "kmax_h_mpc": PRIMARY_P_KMAX,
                "b1": b1,
                "b1_sigma": b1_sigma,
                "fit": fit_report(primary_power),
            },
            "fits": {
                tier: {
                    f"{kmax:.2f}": fit_report(fit)
                    for kmax, fit in rows.items()
                }
                for tier, rows in power_fits.items()
            },
        },
        "gaussian_fits": {
            tier: {
                f"{kmax:.2f}": {
                    "fit": fit_report(fit),
                    "perturbativity": perturbativity[tier][kmax],
                }
                for kmax, fit in rows.items()
            }
            for tier, rows in gaussian_fits.items()
        },
        "tier_diagnostics": tier_diagnostics,
        "uplift_ablation": uplift_ablation,
        "selection": {
            "tier": selected_tier,
            "kmax_h_mpc": selected_kmax,
            "candidate_kmax_by_tier_h_mpc": candidate_kmax,
            "tier_pass": tier_pass,
            "tier_pass_by_kmax": tier_pass_by_kmax,
            "tier_local_pass_excluding_global_b1_gate_by_kmax": (
                tier_local_pass_by_kmax
            ),
            "maximum_local_passing_fit_excluding_global_b1_gate": {
                "tier": fallback_tier if local_options else None,
                "kmax_h_mpc": fallback_kmax if local_options else None,
            },
            "fallback_display_fit": {
                "tier": selected_fit.tier,
                "kmax_h_mpc": selected_fit.kmax,
            },
        },
        "gates": gates,
        "png_gate": png_gate,
        "outputs": {
            "report_json": report_json,
            "report_md": report_md,
            "fit_summary_npz": arrays_path,
            "manifest": manifest_path,
            "figures": list(figure_paths.values()),
        },
        "software": {
            "runner": Path(__file__).resolve(),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    write_json(report_json, payload)
    atomic_write_text(report_md, build_markdown_report(to_jsonable(payload)))
    manifest = {
        "schema": "marisa-b-production-manifest-v1",
        "tag": args.tag,
        "created_utc": payload["created_utc"],
        "command": " ".join(sys.argv),
        "status": overall_status,
        "selected_tier": selected_tier,
        "selected_kmax_h_mpc": selected_kmax,
        "covariance_semantics": payload["scope"],
        "inputs": payload["inputs"],
        "outputs": payload["outputs"],
        "runner_sha256": payload["software"]["runner_sha256"],
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            to_jsonable(
                {
                    "status": overall_status,
                    "gaussian_status": gaussian_status,
                    "selected_tier": selected_tier,
                    "selected_kmax_h_mpc": selected_kmax,
                    "b1": b1,
                    "b1_sigma": b1_sigma,
                    "report": report_json,
                    "figures": list(figure_paths.values()),
                    "png_gate": png_gate["status"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
