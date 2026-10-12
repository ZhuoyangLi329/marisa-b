#!/usr/bin/env python3
"""Regression tests for the EFT-v2 finite-mock fit protocol."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
from scipy.optimize import least_squares, minimize


ROOT = Path(__file__).resolve().parents[2]
FITTER_PATH = ROOT / "scripts/legacy/fit_eft_v2_r0_mean.py"
SELECTED_DR_GENERATOR_PATH = (
    ROOT / "scripts/generate_eft_v2_selected_dr_gate.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name, path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fitter():
    return load_module(
        FITTER_PATH, "eft_v2_r0_fit_statistics_test",
    )


def assert_close(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    label: str,
) -> None:
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        maximum = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{label}: maximum absolute error {maximum}")


def test_contract_schema(fit) -> None:
    contract_path = ROOT / "configs/eft_v2_contract.json"
    schema_path = ROOT / "schemas/eft_v2_contract.schema.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(contract, schema)
    b1 = next(
        row for row in contract["bias_parameters"] if row["id"] == "b1"
    )
    if "fixed" in b1:
        raise AssertionError("contract v2 must not retain a fixed b1")
    if b1["prior"] != {"kind": "normal", "mean": 1.0, "sigma": 2.0}:
        raise AssertionError("unexpected primary b1 prior")
    protocol = contract["fit_protocol"]
    if protocol["b1"]["b_rec"]["likelihood_role"] != "none":
        raise AssertionError("b_rec leaked into the fit likelihood")
    if protocol["covariance_likelihood"]["primary"] != (
        fit.RAW_GAUSSIAN_LIKELIHOOD
    ):
        raise AssertionError("raw Gaussian is not the registered primary")
    parsed = fit.load_fit_protocol(
        contract, SimpleNamespace(mock_count=500),
    )
    if (
        parsed.robustness_likelihoods
        != (
            fit.HARTLAP_GAUSSIAN_LIKELIHOOD,
            fit.SELLENTIN_HEAVENS_LIKELIHOOD,
        )
        or parsed.b1_robustness_prior_sigmas != (1.0, 4.0)
    ):
        raise AssertionError("fit protocol parser changed the frozen checks")

    invalid_cases = []
    fixed_b1 = copy.deepcopy(contract)
    next(
        row for row in fixed_b1["bias_parameters"] if row["id"] == "b1"
    )["fixed"] = 1.0
    invalid_cases.append(fixed_b1)
    promoted_b_rec = copy.deepcopy(contract)
    promoted_b_rec["fit_protocol"]["b1"]["b_rec"][
        "likelihood_role"
    ] = "prior"
    invalid_cases.append(promoted_b_rec)
    missing_robustness = copy.deepcopy(contract)
    missing_robustness["fit_protocol"]["covariance_likelihood"][
        "robustness"
    ] = ["hartlap_gaussian"]
    invalid_cases.append(missing_robustness)
    for invalid in invalid_cases:
        try:
            jsonschema.validate(invalid, schema)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError("contract schema accepted a forbidden mutation")


def test_selected_dr_source_provenance(fit) -> None:
    generator = load_module(
        SELECTED_DR_GENERATOR_PATH,
        "eft_v2_selected_dr_generator_provenance_test",
    )
    if generator.SOURCE_PATHS != fit.SELECTED_DR_SOURCE_PATHS:
        raise AssertionError(
            "selected-DR generator and fitter source manifests differ"
        )
    required = {
        "kernel_primitives_cpp",
        "kernel_primitives_h",
        "parameter_registry_cpp",
        "parameter_registry_h",
        "template_algebra_cpp",
        "template_algebra_h",
        "field_kernel_provider_cpp",
        "field_kernel_provider_h",
        "ir_safe_integrands_cpp",
        "ir_safe_integrands_h",
        "halo_v1_cpp",
        "halo_v1_h",
    }
    missing = required.difference(fit.SELECTED_DR_SOURCE_PATHS)
    if missing:
        raise AssertionError(
            f"selected-DR provenance omits direct dependencies: {sorted(missing)}"
        )
    for relative in fit.SELECTED_DR_SOURCE_PATHS.values():
        if not (ROOT / relative).is_file():
            raise AssertionError(
                f"selected-DR provenance source is missing: {relative}"
            )


def test_bispectrum_loop_range_contract(fit) -> None:
    header = {
        "q_range": [1.0e-4, 30.0],
        "stochastic_qmax": 30.0,
        "hybrid": {
            "factorized_p13": {
                "q_range": [1.0e-4, 30.0],
                "uv_tail_restoration": {
                    "enabled": True,
                    "sampled_kmax": 80.0,
                },
            },
        },
        "stochastic_p13_uv_tail_restoration": {
            "enabled": True,
            "sampled_kmax": 80.0,
        },
    }
    fit.validate_bispectrum_loop_range(header, "valid synthetic header")

    low_qmax = copy.deepcopy(header)
    low_qmax["q_range"][1] = 5.0
    low_qmax["stochastic_qmax"] = 5.0
    low_qmax["hybrid"]["factorized_p13"]["q_range"][1] = 5.0
    fit.validate_bispectrum_loop_range(low_qmax, "valid qmax variant")

    invalid_cases = []
    invalid_q_range = copy.deepcopy(header)
    invalid_q_range["q_range"][0] = 0.0
    invalid_cases.append(invalid_q_range)
    stale_stochastic = copy.deepcopy(header)
    stale_stochastic["stochastic_qmax"] = 20.0
    invalid_cases.append(stale_stochastic)
    stale_factorized = copy.deepcopy(header)
    stale_factorized["hybrid"]["factorized_p13"]["q_range"][1] = 20.0
    invalid_cases.append(stale_factorized)
    short_factorized_tail = copy.deepcopy(header)
    short_factorized_tail["hybrid"]["factorized_p13"][
        "uv_tail_restoration"
    ]["sampled_kmax"] = 20.0
    invalid_cases.append(short_factorized_tail)
    disabled_stochastic_tail = copy.deepcopy(header)
    disabled_stochastic_tail["stochastic_p13_uv_tail_restoration"][
        "enabled"
    ] = False
    invalid_cases.append(disabled_stochastic_tail)
    for invalid in invalid_cases:
        try:
            fit.validate_bispectrum_loop_range(
                invalid, "invalid synthetic header",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "bispectrum loop-range contract accepted a forbidden mutation"
            )

    power_header = {
        "implementation_version": (
            "eft-v2-p0-adaptive-estimator-constrained-radial-v2"
        ),
        "q_range": [1.0e-4, 30.0],
        "p13_uv_tail_restoration": {
            "enabled": True,
            "sampled_kmax": 80.0,
        },
        "shell_nrad": 1,
        "shell_projection": {
            "measure": (
                "integer FFT-lattice radial marginal with authoritative "
                "estimator mode-count boundary allocation"
            ),
            "normalization": "raw shell-mode average",
            "binning": (
                "JAXPower P0 pk_nmodes-constrained lower-inclusive "
                "upper-exclusive baseline"
            ),
            "requested_radial_order": 1,
            "cap_order_to_unique_support": True,
            "box_size_mpc_h": 1000,
            "mesh_size": 256,
            "mode_count_source": "synthetic-mode-counts.txt",
        },
    }
    power_row = {
        "shell_nodes": 1,
        "shell_unique_radius_count": 1,
        "shell_mode_count": 6,
        "shell_expected_mode_count": 6,
        "shell_boundary_mode_adjustment": 0,
    }
    fit.validate_power_shell_metadata(
        SimpleNamespace(header=power_header, bins=[power_row]),
        "valid synthetic power header",
    )
    short_power_tail = copy.deepcopy(power_header)
    short_power_tail["p13_uv_tail_restoration"]["sampled_kmax"] = 20.0
    try:
        fit.validate_power_shell_metadata(
            SimpleNamespace(header=short_power_tail, bins=[power_row]),
            "invalid synthetic power header",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("power-loop contract accepted a short P13 tail")


def test_diagonal_bispectrum_indices(fit) -> None:
    edges = np.zeros((120, 2, 2), dtype=float)
    k_pair = np.zeros((120, 2), dtype=float)
    edges[0] = [[0.0, 0.1], [0.0, 0.1]]
    edges[1] = [[0.0, 0.1], [0.1, 0.2]]
    edges[2] = [[0.1, 0.2], [0.1, 0.2]]
    edges[3] = [[0.2, 0.3], [0.2, 0.3]]
    edges[4:, 0, :] = [0.3, 0.4]
    edges[4:, 1, :] = [0.4, 0.5]
    k_pair[:4] = [
        [0.05, 0.05],
        [0.05, 0.15],
        [0.15, 0.15],
        [0.25, 0.25],
    ]
    mask = np.zeros(120, dtype=bool)
    mask[:4] = [True, True, True, False]
    data = SimpleNamespace(edges=edges, k_pair=k_pair)
    indices = fit.diagonal_bispectrum_indices(data, mask)
    if indices.tolist() != [0, 2]:
        raise AssertionError(
            f"unexpected fitted diagonal indices {indices.tolist()}"
        )


def test_correlation_cholesky(fit) -> None:
    rng = np.random.default_rng(1701)
    raw = rng.normal(size=(9, 9))
    correlation = raw @ raw.T
    normalization = np.sqrt(np.diag(correlation))
    correlation /= np.outer(normalization, normalization)
    scales = np.geomspace(1.0e-7, 1.0e8, 9)
    covariance = correlation * np.outer(scales, scales)
    jitter = 1.0e-12
    cholesky = fit.covariance_cholesky(covariance, jitter)
    expected = covariance + jitter * np.diag(np.diag(covariance))
    assert_close(
        cholesky @ cholesky.T,
        expected,
        rtol=2.0e-12,
        atol=1.0e-20,
        label="correlation-space Cholesky reconstruction",
    )

    residual = rng.normal(size=9) * scales
    rescaling = np.geomspace(0.2, 7.0, 9)
    scaled_covariance = covariance * np.outer(rescaling, rescaling)
    whitened = np.linalg.solve(cholesky, residual)
    scaled_whitened = np.linalg.solve(
        fit.covariance_cholesky(scaled_covariance, jitter),
        residual * rescaling,
    )
    assert_close(
        scaled_whitened,
        whitened,
        rtol=3.0e-12,
        atol=3.0e-12,
        label="observable-rescaling invariance",
    )


def test_finite_mock_objectives(fit) -> None:
    mock_count = 500
    dimension = 47
    t_squared = 41.25
    expected_hartlap = (mock_count - dimension - 2) / (mock_count - 1)
    assert_close(
        fit.hartlap_factor(mock_count, dimension),
        expected_hartlap,
        label="Hartlap factor",
    )
    try:
        fit.hartlap_factor(49, 47)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Hartlap dimension was accepted")

    for likelihood in fit.SUPPORTED_LIKELIHOODS:
        objective = fit.likelihood_data_objective(
            likelihood, t_squared, mock_count, dimension,
        )
        scale = fit.likelihood_residual_scale(
            likelihood, t_squared, mock_count, dimension,
        )
        assert_close(
            scale**2 * t_squared,
            objective,
            rtol=3.0e-15,
            atol=3.0e-15,
            label=f"{likelihood} residual representation",
        )
    expected_sh = mock_count * math.log1p(
        t_squared / (mock_count - 1.0)
    )
    assert_close(
        fit.sellentin_heavens_data_objective(t_squared, mock_count),
        expected_sh,
        label="Sellentin-Heavens objective",
    )


def test_exact_sellentin_heavens_profile(fit) -> None:
    rng = np.random.default_rng(20260724)
    maximum_gradient = 0.0
    maximum_objective_error = 0.0
    maximum_parameter_error = 0.0
    for trial in range(32):
        data_count = int(rng.integers(8, 36))
        parameter_count = int(rng.integers(1, min(8, data_count)))
        prior_count = parameter_count + int(rng.integers(0, 4))
        design = rng.normal(size=(data_count, parameter_count))
        target = rng.normal(size=data_count)
        prior_design = rng.normal(size=(prior_count, parameter_count))
        prior_design[:parameter_count] += np.eye(parameter_count)
        prior_target = rng.normal(size=prior_count)
        mock_count = int(rng.integers(max(50, data_count + 4), 900))
        parameters, weight, t_squared = (
            fit.profile_sellentin_heavens_linear(
                design,
                target,
                prior_design,
                prior_target,
                mock_count,
            )
        )
        data_residual = design @ parameters - target
        prior_residual = prior_design @ parameters - prior_target
        gradient = (
            2.0 * weight * design.T @ data_residual
            + 2.0 * prior_design.T @ prior_residual
        )
        maximum_gradient = max(
            maximum_gradient, float(np.linalg.norm(gradient)),
        )
        assert_close(
            weight,
            mock_count / (mock_count - 1.0 + t_squared),
            rtol=2.0e-13,
            atol=2.0e-13,
            label=f"trial {trial} fixed point",
        )

        def objective(candidate: np.ndarray) -> float:
            candidate_data = design @ candidate - target
            candidate_prior = prior_design @ candidate - prior_target
            return (
                fit.sellentin_heavens_data_objective(
                    float(candidate_data @ candidate_data), mock_count,
                )
                + float(candidate_prior @ candidate_prior)
            )

        direct = minimize(
            objective,
            parameters + 0.05 * rng.normal(size=parameter_count),
            method="BFGS",
            options={"gtol": 1.0e-10, "maxiter": 4000},
        )
        objective_error = abs(float(direct.fun) - objective(parameters))
        parameter_error = float(np.max(np.abs(direct.x - parameters)))
        maximum_objective_error = max(
            maximum_objective_error, objective_error,
        )
        maximum_parameter_error = max(
            maximum_parameter_error, parameter_error,
        )
        if objective_error > 3.0e-8 or parameter_error > 3.0e-5:
            raise AssertionError(
                f"trial {trial} direct/profile mismatch: "
                f"parameter_error={parameter_error}, "
                f"objective_error={objective_error}, "
                f"optimizer={direct.message}"
            )
    if maximum_gradient > 2.0e-8:
        raise AssertionError(
            f"Sellentin-Heavens profile gradient {maximum_gradient}"
        )
    print(
        "Sellentin-Heavens profile:",
        f"max_gradient={maximum_gradient:.3e}",
        f"max_parameter_error={maximum_parameter_error:.3e}",
        f"max_objective_error={maximum_objective_error:.3e}",
    )


def test_nonlinear_sellentin_heavens_profile(fit) -> None:
    rng = np.random.default_rng(413)
    data_count = 24
    linear_count = 5
    mock_count = 500
    design = rng.normal(size=(data_count, linear_count))
    first_shape = rng.normal(size=data_count)
    second_shape = rng.normal(size=data_count)
    target = (
        0.7 * first_shape
        - 0.25 * second_shape
        + design @ rng.normal(size=linear_count)
        + 0.3 * rng.normal(size=data_count)
    )
    prior_sigma = np.asarray([1.5, 1.2] + [1.0] * linear_count)
    prior_mean = np.zeros(2 + linear_count)
    prior_mean[0] = 0.2
    prior_whitener = np.diag(1.0 / prior_sigma)

    def nonlinear_base(non_linear: np.ndarray) -> np.ndarray:
        return (
            np.sin(non_linear[0]) * first_shape
            + non_linear[1] ** 2 * second_shape
        )

    def profile(non_linear: np.ndarray):
        base = nonlinear_base(non_linear)
        parameters = np.zeros(2 + linear_count)
        parameters[:2] = non_linear
        prior_design = prior_whitener[:, 2:]
        prior_target = -prior_whitener @ (parameters - prior_mean)
        linear, weight, t_squared = (
            fit.profile_sellentin_heavens_linear(
                design,
                target - base,
                prior_design,
                prior_target,
                mock_count,
            )
        )
        parameters[2:] = linear
        data_residual = base + design @ linear - target
        scale = fit.likelihood_residual_scale(
            fit.SELLENTIN_HEAVENS_LIKELIHOOD,
            t_squared,
            mock_count,
            data_count,
        )
        residual = np.concatenate((
            scale * data_residual,
            prior_whitener @ (parameters - prior_mean),
        ))
        return parameters, residual, weight, t_squared

    profiled_solution = least_squares(
        lambda non_linear: profile(non_linear)[1],
        np.asarray([0.4, -0.4]),
        method="trf",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=4000,
    )
    parameters, residual, weight, t_squared = profile(
        profiled_solution.x,
    )

    def full_objective(candidate: np.ndarray) -> float:
        data_residual = (
            nonlinear_base(candidate[:2])
            + design @ candidate[2:]
            - target
        )
        prior_residual = prior_whitener @ (candidate - prior_mean)
        return (
            fit.sellentin_heavens_data_objective(
                float(data_residual @ data_residual), mock_count,
            )
            + float(prior_residual @ prior_residual)
        )

    direct = minimize(
        full_objective,
        parameters + 0.02 * rng.normal(size=parameters.size),
        method="BFGS",
        options={"gtol": 1.0e-10, "maxiter": 5000},
    )
    parameter_error = float(np.max(np.abs(direct.x - parameters)))
    objective_error = abs(
        float(direct.fun) - float(residual @ residual)
    )
    fixed_point_error = abs(
        weight - mock_count / (mock_count - 1.0 + t_squared)
    )
    if (
        not profiled_solution.success
        or parameter_error > 3.0e-5
        or objective_error > 2.0e-8
        or fixed_point_error > 2.0e-12
    ):
        raise AssertionError(
            "nonlinear Sellentin-Heavens profile mismatch: "
            f"parameter_error={parameter_error}, "
            f"objective_error={objective_error}, "
            f"fixed_point_error={fixed_point_error}, "
            f"direct_optimizer={direct.message}"
        )
    print(
        "Nonlinear Sellentin-Heavens profile:",
        f"parameter_error={parameter_error:.3e}",
        f"objective_error={objective_error:.3e}",
        f"fixed_point_error={fixed_point_error:.3e}",
    )


def test_b1_prior_replacement(fit) -> None:
    size = len(fit.FIT_NAMES)
    mean = np.arange(size, dtype=np.float64) / 10.0
    sigma = np.linspace(1.0, 3.0, size)
    covariance = np.diag(sigma**2)
    prior = fit.Prior(
        mean=mean,
        sigma=sigma,
        covariance=covariance,
        whitener=np.diag(1.0 / sigma),
    )
    changed = fit.prior_with_b1_sigma(prior, 4.0)
    index = fit.FIT_NAMES.index("b1")
    if changed.mean[index] != prior.mean[index] or changed.sigma[index] != 4.0:
        raise AssertionError("b1 prior replacement changed its mean or width")
    keep = np.arange(size) != index
    assert_close(
        changed.covariance[np.ix_(keep, keep)],
        prior.covariance[np.ix_(keep, keep)],
        label="non-b1 prior covariance",
    )


def test_noiseless_profiled_fit_recovery(fit) -> None:
    rng = np.random.default_rng(8817)
    parameter_count = len(fit.FIT_NAMES)
    b_design = 0.15 * rng.normal(size=(120, parameter_count))
    p_design = 0.15 * rng.normal(size=(47, parameter_count))
    b_quadratic = 0.04 * rng.normal(size=(120, 3))
    p_quadratic = 0.04 * rng.normal(size=(47, 3))
    b_cross = 0.03 * rng.normal(size=120)
    p_cross = 0.03 * rng.normal(size=47)
    b1 = fit.FIT_NAMES.index("b1")
    b2 = fit.FIT_NAMES.index("b2")
    gamma2 = fit.FIT_NAMES.index("gamma2")
    bnabla = fit.FIT_NAMES.index("b_nabla2_delta")
    bshot = fit.FIT_NAMES.index("Bshot_residual")

    class SyntheticTemplates:
        def __init__(
            self,
            design: np.ndarray,
            quadratic: np.ndarray,
            cross: np.ndarray,
        ) -> None:
            self.design = design
            self.quadratic = quadratic
            self.cross = cross

        def components(self, parameters, _number_density):
            values = np.asarray([
                parameters[name] for name in fit.FIT_NAMES
            ])
            total = self.design @ values
            total += self.quadratic @ np.asarray([
                values[b1] ** 2,
                values[b2] * values[gamma2],
                values[bnabla] ** 2,
            ])
            total += self.cross * values[bnabla] * values[bshot]
            return {"total": total}

    bispectrum_templates = SyntheticTemplates(
        b_design, b_quadratic, b_cross,
    )
    power_templates = SyntheticTemplates(
        p_design, p_quadratic, p_cross,
    )
    true_parameters = 0.2 * rng.normal(size=parameter_count)
    true_parameters[b1] = 1.7
    true_parameters[bnabla] = 0.35
    true_parameters[bshot] = -0.4
    parameter_mapping = fit.parameter_dict(true_parameters)
    b_mean = bispectrum_templates.components(
        parameter_mapping, 1.0,
    )["total"]
    p_mean = power_templates.components(
        parameter_mapping, 1.0,
    )["total"]

    b_variance = np.geomspace(0.02, 0.2, 120) ** 2
    p_variance = np.geomspace(0.01, 0.1, 47) ** 2
    joint_variance = np.concatenate((p_variance, b_variance))
    k_pair = np.full((120, 2), 0.2)
    k_pair[:28] = 0.1
    power_k = np.full(47, 0.2)
    power_k[:15] = 0.05
    data = fit.DataSet(
        mean=b_mean,
        covariance_mean=np.diag(b_variance),
        error_mean=np.sqrt(b_variance),
        k_pair=k_pair,
        edges=np.zeros((120, 2, 2)),
        power_mean=p_mean,
        power_error_mean=np.sqrt(p_variance),
        power_k=power_k,
        power_edges=np.linspace(0.0, 0.3, 48),
        power_nmodes=np.ones(47, dtype=int),
        joint_mean_power_then_bispectrum=np.concatenate((p_mean, b_mean)),
        joint_covariance_mean_power_then_bispectrum=np.diag(joint_variance),
        mock_count=500,
    )
    prior_sigma = np.full(parameter_count, 1.5)
    prior = fit.Prior(
        mean=true_parameters.copy(),
        sigma=prior_sigma,
        covariance=np.diag(prior_sigma**2),
        whitener=np.diag(1.0 / prior_sigma),
    )
    maximum_parameter_error = 0.0
    maximum_t_squared = 0.0
    for likelihood in (
        fit.RAW_GAUSSIAN_LIKELIHOOD,
        fit.HARTLAP_GAUSSIAN_LIKELIHOOD,
        fit.SELLENTIN_HEAVENS_LIKELIHOOD,
    ):
        result = fit.fit_one(
            bispectrum_templates,
            power_templates,
            data,
            prior,
            1.0,
            0.15,
            4,
            np.random.default_rng(991),
            likelihood=likelihood,
            mock_count=500,
            correlation_jitter=1.0e-12,
        )
        parameter_error = float(np.max(np.abs(
            result.x - true_parameters
        )))
        maximum_parameter_error = max(
            maximum_parameter_error, parameter_error,
        )
        maximum_t_squared = max(maximum_t_squared, result.chi2)
        json.dumps(result.report(), allow_nan=False, sort_keys=True)
        if (
            not result.optimizer_success
            or parameter_error > 2.0e-8
            or result.chi2 > 2.0e-16
        ):
            raise AssertionError(
                f"{likelihood} noiseless recovery failed: "
                f"parameter_error={parameter_error}, "
                f"T2={result.chi2}, "
                f"optimizer={result.optimizer_message}"
            )
    print(
        "Noiseless profiled recovery:",
        f"max_parameter_error={maximum_parameter_error:.3e}",
        f"max_T2={maximum_t_squared:.3e}",
    )


def main() -> None:
    fit = load_fitter()
    test_contract_schema(fit)
    test_selected_dr_source_provenance(fit)
    test_bispectrum_loop_range_contract(fit)
    test_diagonal_bispectrum_indices(fit)
    test_correlation_cholesky(fit)
    test_finite_mock_objectives(fit)
    test_exact_sellentin_heavens_profile(fit)
    test_nonlinear_sellentin_heavens_profile(fit)
    test_b1_prior_replacement(fit)
    test_noiseless_profiled_fit_recovery(fit)
    print("EFT-v2 fit-statistics tests passed")


if __name__ == "__main__":
    main()
