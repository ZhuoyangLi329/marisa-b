#!/usr/bin/env python3
"""Direct local-PNG fNL recovery profiles for ordinary Quijote halos.

The ten fits use pre/ordinary-standard B000 at z=0.5 and kmax=0.10 h/Mpc
for input fNL = 0, +/-50, +/-100.  The model is the audited six-parameter
Gaussian-plus-linear-PNG model with p=1.12 in

    b_phi = 2 delta_c (b1 - p).

No response gain or fNL^2 term is allowed.  Every fit uses the Fid500
single-realization covariance without division by the number of realizations.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MARISA_B_MAX_THREADS",
):
    os.environ[_name] = "1"

# This runner consumes the authenticated frozen response cache and never
# executes the live analytic binary.  The binary has changed since the cache
# was produced, so use the existing explicit cache-only provenance path while
# retaining all source/input/cache hash checks.
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = ROOT / "codes"
TAG = "quijote_halo_z0p5_local_png_p112_recovery_kmax0p10"
ANALYSIS = ROOT / "analysis" / TAG
FIGURES = ROOT / "figures/diagnostics" / TAG

PROJECTED_RUNNER = CODES / "run_quijote_halo_z0p5_pre_standard_projected_png_audit.py"
PROFILE_RUNNER = CODES / "run_quijote_halo_z0p5_standard_refined_b000_profiles.py"
REFINED_ASSETS = CODES / "quijote_halo_z0p5_standard_refined_assets.py"
INTERVAL_HELPER = CODES / "run_quijote_halo_z0p5_standard_gaussian_repair_profiles.py"

BASE_MATRIX = (
    ROOT
    / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100/"
    "matrices/baseline_pre_standard_matrix.npz"
)
FNL50_MATRIX = (
    ROOT
    / "analysis/quijote_halo_z0p5_standard_png_odd_even_repair/"
    "CENTERED_FNL50_MATRIX.npz"
)
FNL100_MATRIX = (
    ROOT
    / "analysis/quijote_halo_z0p5_standard_png_odd_even_repair/"
    "CENTERED_FNL100_MATRIX.npz"
)

OUTPUT_JSON = ANALYSIS / "RECOVERY_PROFILES.json"
OUTPUT_CSV = ANALYSIS / "RECOVERY_PROFILES.csv"
OUTPUT_ARRAYS = ANALYSIS / "RECOVERY_PROFILE_ARRAYS.npz"
OUTPUT_REPORT = ANALYSIS / "REPORT.md"
OUTPUT_PDF = FIGURES / "quijote_halo_z0p5_local_png_p112_recovery_kmax0p10.pdf"
OUTPUT_BESTFIT_PRE_PDF = FIGURES / "pre_bestfit_B000_vs_measurement_mean.pdf"
OUTPUT_BESTFIT_STANDARD_PDF = FIGURES / "standard_bestfit_B000_vs_measurement_mean.pdf"
ENTRYPOINT = Path(__file__).resolve()

KMAX = 0.10
EXPECTED_INDICES = np.asarray(
    (1, 2, 3, 4, 15, 16, 17, 18, 29, 30, 31, 42, 43, 54),
    dtype=np.int64,
)
INPUT_FNL = (-100.0, -50.0, 0.0, 50.0, 100.0)
P_BPHI = 1.12
DELTA_C = 1.686
# This is a numerical profiling range, not a prior.  The standard-reconstructed
# B000 likelihood has a long positive-fNL/b1 branch with one-box covariance, so
# the historical +/-150 and an initial +/-2000 diagnostic range are too narrow
# to determine whether Delta chi2=1 actually closes.
FNL_BOUNDS = (-20000.0, 20000.0)
FNL_PRIOR_HALF_WIDTH: float | None = None
COVARIANCE_VOLUME = 1.0
METHODS = ("pre", "standard")

PRE_BPHI_LINEAR_KEYS = (
    "bphi_f2",
    "advection",
    "bphi_b2",
    "bphi_bK2",
    "stochastic",
)
STANDARD_BPHI_LINEAR_KEYS = (
    "halo_bphi_f2",
    "halo_bphi_advection",
    "halo_bphi_b2",
    "halo_bphi_bK2",
    "halo_reconstruction",
    "residual_stochastic_linear",
    "physical_K1_matter_loop_response",
)

PDF_METADATA = {
    "Author": "ordinary-halo local-PNG recovery audit",
    "Creator": Path(__file__).name,
    "Subject": (
        "z=0.5 ordinary pre/standard B000; p=1.12; linear fNL; "
        "single-realization covariance"
    ),
}


def fnl_bounds_for_truth(truth: float) -> tuple[float, float]:
    if FNL_PRIOR_HALF_WIDTH is None:
        return FNL_BOUNDS
    width = float(FNL_PRIOR_HALF_WIDTH)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("FNL_PRIOR_HALF_WIDTH must be finite and positive")
    return float(truth - width), float(truth + width)


def covariance_description() -> str:
    if math.isclose(COVARIANCE_VOLUME, 1.0):
        return (
            "Fid500 sample covariance of one 1 (Gpc/h)^3 realization; "
            "not divided by target or covariance realization count"
        )
    return (
        "Fid500 single-box sample covariance divided by "
        f"{COVARIANCE_VOLUME:g} for the target volume; never divided by "
        "the number of target or covariance realizations"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def import_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def input_tag(value: float) -> str:
    if value == 0.0:
        return "fid0"
    return ("p" if value > 0.0 else "m") + str(abs(int(value)))


def validate_inputs(
    baseline: dict[str, np.ndarray],
    fnl50: dict[str, np.ndarray],
    fnl100: dict[str, np.ndarray],
) -> dict[str, Any]:
    required_baseline = {
        "fiducial_pre_B000",
        "fiducial_pre_B000_cov",
        "fiducial_post_B000",
        "fiducial_post_B000_cov",
        "b000_k",
        "b000_k_edges",
        "b000_nmodes",
    }
    required50 = {
        "LC_m50_pre_B000",
        "LC_m50_post_B000",
        "LC_p50_pre_B000",
        "LC_p50_post_B000",
    }
    required100 = {
        "LC_m_pre_B000",
        "LC_m_post_B000",
        "LC_p_pre_B000",
        "LC_p_post_B000",
    }
    for label, values, required in (
        ("baseline", baseline, required_baseline),
        ("fnl50", fnl50, required50),
        ("fnl100", fnl100, required100),
    ):
        missing = required - set(values)
        if missing:
            raise KeyError(f"{label} missing fields {sorted(missing)}")

    geometry: dict[str, Any] = {}
    for label, values in (("fnl50", fnl50), ("fnl100", fnl100)):
        k_maximum = float(np.max(np.abs(baseline["b000_k"] - values["b000_k"])))
        edge_maximum = float(
            np.max(np.abs(baseline["b000_k_edges"] - values["b000_k_edges"]))
        )
        nmodes_equal = bool(
            np.array_equal(baseline["b000_nmodes"], values["b000_nmodes"])
        )
        if k_maximum != 0.0 or edge_maximum != 0.0 or not nmodes_equal:
            raise ValueError(f"{label} B000 geometry does not match Fid500")
        geometry[label] = {
            "maximum_absolute_k_difference": k_maximum,
            "maximum_absolute_edge_difference": edge_maximum,
            "nmodes_exactly_equal": nmodes_equal,
        }

    if baseline["fiducial_pre_B000"].shape != (500, 120):
        raise ValueError("Fid500 pre matrix shape drift")
    if baseline["fiducial_post_B000"].shape != (500, 120):
        raise ValueError("Fid500 post matrix shape drift")
    finite_matrices = {
        "m100_pre": fnl100["LC_m_pre_B000"],
        "m100_standard": fnl100["LC_m_post_B000"],
        "m50_pre": fnl50["LC_m50_pre_B000"],
        "m50_standard": fnl50["LC_m50_post_B000"],
        "p50_pre": fnl50["LC_p50_pre_B000"],
        "p50_standard": fnl50["LC_p50_post_B000"],
        "p100_pre": fnl100["LC_p_pre_B000"],
        "p100_standard": fnl100["LC_p_post_B000"],
    }
    for name, values in finite_matrices.items():
        if values.shape != (100, 120) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} matrix shape/finite-value failure")
    if not all(
        np.all(np.isfinite(baseline[name]))
        for name in ("fiducial_pre_B000", "fiducial_post_B000")
    ):
        raise ValueError("non-finite Fid500 values")

    covariance_reconstruction: dict[str, Any] = {}
    for method in ("pre", "post"):
        samples = np.asarray(
            baseline[f"fiducial_{method}_B000"], dtype=np.float64
        )
        stored = np.asarray(
            baseline[f"fiducial_{method}_B000_cov"], dtype=np.float64
        )
        reconstructed = np.cov(samples, rowvar=False, ddof=1)
        relative_l2 = float(
            np.linalg.norm(reconstructed - stored) / np.linalg.norm(stored)
        )
        diagonal_ratio = np.diag(reconstructed) / np.diag(stored)
        if relative_l2 >= 1.0e-12:
            raise ValueError(f"{method} Fid500 covariance reconstruction failed")
        covariance_reconstruction[method] = {
            "relative_l2": relative_l2,
            "minimum_diagonal_ratio": float(np.min(diagonal_ratio)),
            "maximum_diagonal_ratio": float(np.max(diagonal_ratio)),
            "division_by_realization_count_applied": False,
        }

    return {
        "status": "pass",
        "geometry": geometry,
        "fiducial_realization_count": 500,
        "nonzero_input_realization_count": 100,
        "all_values_finite": True,
        "single_realization_covariance_reconstruction": covariance_reconstruction,
    }


def target_samples(
    method: str,
    truth: float,
    baseline: dict[str, np.ndarray],
    fnl50: dict[str, np.ndarray],
    fnl100: dict[str, np.ndarray],
) -> np.ndarray:
    suffix = "pre_B000" if method == "pre" else "post_B000"
    if truth == 0.0:
        return np.asarray(baseline[f"fiducial_{suffix}"], dtype=np.float64)
    if truth == -50.0:
        return np.asarray(fnl50[f"LC_m50_{suffix}"], dtype=np.float64)
    if truth == 50.0:
        return np.asarray(fnl50[f"LC_p50_{suffix}"], dtype=np.float64)
    if truth == -100.0:
        return np.asarray(fnl100[f"LC_m_{suffix}"], dtype=np.float64)
    if truth == 100.0:
        return np.asarray(fnl100[f"LC_p_{suffix}"], dtype=np.float64)
    raise ValueError(truth)


def p112_linear_evaluator(context: Any, method: str) -> tuple[Any, dict[str, Any]]:
    original = context.evaluate_model
    dependency_keys = (
        PRE_BPHI_LINEAR_KEYS if method == "pre" else STANDARD_BPHI_LINEAR_KEYS
    )

    def evaluate(
        fnl: float, nuisance: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
        _zero, parameters, components = original(0.0, nuisance)
        b1 = float(parameters["b1"])
        bphi_p1 = 2.0 * DELTA_C * (b1 - 1.0)
        bphi_p112 = 2.0 * DELTA_C * (b1 - P_BPHI)
        if abs(bphi_p1) < 1.0e-10:
            raise ValueError("p=1 b_phi is too close to zero for component rescaling")
        ratio = bphi_p112 / bphi_p1

        updated: dict[str, Any] = dict(components)
        bphi_dependent = np.zeros(120, dtype=np.float64)
        used_keys: list[str] = []
        for name in dependency_keys:
            if name not in components:
                if name == "physical_K1_matter_loop_response":
                    continue
                raise KeyError(f"{method} is missing required b_phi component {name}")
            value = np.asarray(components[name], dtype=np.float64)
            if value.shape != (120,):
                raise ValueError(f"{method}:{name} has shape {value.shape}")
            bphi_dependent += value
            updated[name] = ratio * value
            used_keys.append(name)

        old_linear = np.asarray(components["linear_total"], dtype=np.float64)
        new_linear = old_linear + (ratio - 1.0) * bphi_dependent
        gaussian = np.asarray(components["gaussian_total"], dtype=np.float64)
        quadratic_zero = np.zeros_like(new_linear)
        updated["linear_total"] = new_linear
        updated["quadratic_total"] = quadratic_zero
        updated["p_bphi"] = np.asarray(P_BPHI)
        updated["bphi_p1"] = np.asarray(bphi_p1)
        updated["bphi"] = np.asarray(bphi_p112)
        updated["bphi_ratio_p112_over_p1"] = np.asarray(ratio)
        updated["bphi_dependency_keys"] = tuple(used_keys)
        if "physical_png_biases" in components:
            biases = dict(components["physical_png_biases"])
            biases.update(
                {
                    "bphi": bphi_p112,
                    "bphi_universal_p1": bphi_p1,
                    "ratio_bphi": ratio,
                    "bphidelta_policy": "preserve p=1 relation",
                }
            )
            updated["physical_png_biases"] = biases
        if "fixed_brec_linear_total" in components:
            updated["fixed_brec_linear_total"] = new_linear
        prediction = gaussian + float(fnl) * new_linear
        return prediction[context.indices], parameters, updated

    wrapped = replace(
        context,
        reconstruction=f"{context.reconstruction}_linear_p1p12",
        evaluate_model=evaluate,
        fnl_bounds=context.fnl_bounds,
        input_hashes={
            **context.input_hashes,
            "local_png_model": "Gaussian + fNL*linear only",
            "bphi_relation": "2*delta_c*(b1-1.12)",
            "bphidelta_policy": "preserve p=1 relation",
            "response_gain": "forbidden",
        },
    )
    return wrapped, {
        "p": P_BPHI,
        "delta_c": DELTA_C,
        "bphidelta_policy": "preserve existing p=1 relation",
        "dependency_keys": dependency_keys,
        "quadratic_fNL_terms": "forced to zero",
    }


def recovery_context(
    base_context: Any,
    method: str,
    truth: float,
    inputs: np.ndarray,
    fiducial_samples: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    selected = np.asarray(base_context.indices, dtype=np.int64)
    if not np.array_equal(selected, EXPECTED_INDICES):
        raise ValueError(
            f"kmax=0.10 index contract drift: {selected.tolist()}"
        )
    covariance_samples = np.asarray(
        fiducial_samples[:, selected], dtype=np.float64
    )
    covariance_single_box = np.cov(covariance_samples, rowvar=False, ddof=1)
    covariance = covariance_single_box / float(COVARIANCE_VOLUME)
    selected_targets = np.asarray(inputs[:, selected], dtype=np.float64)
    target = np.mean(selected_targets, axis=0)
    fnl_bounds = fnl_bounds_for_truth(truth)
    context = replace(
        base_context,
        reconstruction=(
            f"z0p5_{method}_B000_input_{input_tag(truth)}_"
            f"volume_{COVARIANCE_VOLUME:g}"
        ),
        target=target,
        covariance_single=covariance,
        covariance_mock_count=500,
        samples=covariance_samples,
        realization_ids=np.arange(500, dtype=np.int64),
        fnl_bounds=fnl_bounds,
        input_hashes={
            **base_context.input_hashes,
            "target_input_fNL": f"{truth:g}",
            "target_realization_count": str(inputs.shape[0]),
            "covariance_realization_count": "500",
            "covariance_volume_divisor": f"{COVARIANCE_VOLUME:g}",
            "fNL_flat_prior_lower": f"{fnl_bounds[0]:g}",
            "fNL_flat_prior_upper": f"{fnl_bounds[1]:g}",
        },
    )
    context, theory_contract = p112_linear_evaluator(context, method)
    return context, {
        "input_fNL": truth,
        "target_realization_count": int(inputs.shape[0]),
        "covariance_realization_count": 500,
        "target_volume_gpc3_h3": float(COVARIANCE_VOLUME),
        "covariance": covariance_description(),
        "fNL_flat_prior": fnl_bounds,
        "theory": theory_contract,
    }


def profile_case(
    *,
    audit: Any,
    base: Any,
    assets: Any,
    interval_helper: Any,
    context: Any,
    method: str,
    truth: float,
    context_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record, arrays = audit.profile_context(base, assets, context)
    interval_helper.FNL_BOUNDS = tuple(context.fnl_bounds)
    interval = interval_helper.delta_chi2_interval(context, record["profile"])
    profile_theta = np.asarray(record["profile"]["theta"], dtype=np.float64)
    profile_conditional = context.conditional(profile_theta)
    truth_objective, truth_nonlinear, truth_valid = interval_helper.fixed_fnl_objective(
        context, truth, profile_theta[1:]
    )
    truth_delta_chi2 = float(truth_objective - record["profile"]["objective"])
    fhat = float(record["profile_fNL"])
    half_width = float(interval["half_width"])
    nuisance = np.asarray(profile_conditional.nuisance, dtype=np.float64)
    zero, parameters, components = context.evaluate_model(0.0, nuisance)
    plus, _parameters, _components = context.evaluate_model(100.0, nuisance)
    minus, _parameters, _components = context.evaluate_model(-100.0, nuisance)
    linearity_residual = np.asarray(plus) + np.asarray(minus) - 2.0 * np.asarray(zero)
    linearity_scale = max(
        float(np.max(np.abs(plus))),
        float(np.max(np.abs(minus))),
        float(np.max(np.abs(zero))),
        1.0,
    )
    b1 = float(parameters["b1"])
    bphi = 2.0 * DELTA_C * (b1 - P_BPHI)
    bphi_p1 = 2.0 * DELTA_C * (b1 - 1.0)
    valid_objectives = [
        value
        for value, valid in zip(
            record["profile"]["multistart_objectives"],
            record["profile"]["multistart_valid"],
        )
        if valid
    ]
    multistart_spread = (
        float(max(valid_objectives) - min(valid_objectives))
        if valid_objectives
        else math.inf
    )
    bias = fhat - truth
    interval_closed = bool(
        interval["lower_closed_before_bound"]
        and interval["upper_closed_before_bound"]
    )
    truth_inside = bool(interval["lower"] <= truth <= interval["upper"])
    integrity = {
        "profile_valid": bool(record["profile"]["valid"]),
        "profile_not_at_limit": not bool(record["profile"]["at_limit"]),
        "interval_closed_before_bounds": interval_closed,
        "all_multistarts_valid": bool(all(record["profile"]["multistart_valid"])),
        "multistart_objective_spread_below_1e_minus_5": multistart_spread < 1.0e-5,
        "linearity_relative_residual_below_1e_minus_12": (
            float(np.max(np.abs(linearity_residual))) / linearity_scale < 1.0e-12
        ),
        "quadratic_response_exactly_zero": bool(
            np.all(np.asarray(arrays["quadratic_response"]) == 0.0)
        ),
    }
    numerical_integrity_names = tuple(
        name for name in integrity if name != "interval_closed_before_bounds"
    )
    result = {
        **record,
        "method": method,
        "input_fNL": truth,
        "input_tag": input_tag(truth),
        "context_contract": context_contract,
        "delta_chi2_1_interval": interval,
        "recovery": {
            "fNL_hat": fhat,
            "bias_hat_minus_input": bias,
            "bias_over_profile_half_width": (
                bias / half_width if half_width > 0.0 else None
            ),
            "truth_inside_delta_chi2_1_interval": truth_inside,
            "truth_profile_delta_chi2": truth_delta_chi2,
            "truth_fixed_profile_valid": bool(truth_valid),
            "truth_profiled_nonlinear": dict(
                zip(context.nonlinear_names, np.asarray(truth_nonlinear))
            ),
        },
        "best_fit_png_bias": {
            "p": P_BPHI,
            "delta_c": DELTA_C,
            "b1": b1,
            "bphi": bphi,
            "bphi_p1_reference": bphi_p1,
            "bphi_ratio_p112_over_p1": bphi / bphi_p1,
            "bphidelta_policy": "preserve existing p=1 relation",
        },
        "linearity_audit": {
            "probe_fNL": 100.0,
            "maximum_absolute_B_plus_plus_B_minus_minus_2B0": float(
                np.max(np.abs(linearity_residual))
            ),
            "relative_to_maximum_model_absolute": float(
                np.max(np.abs(linearity_residual)) / linearity_scale
            ),
        },
        "multistart_valid_objective_spread": multistart_spread,
        "integrity_checks": integrity,
        "numerical_integrity_checks": numerical_integrity_names,
        "integrity_pass": bool(
            all(integrity[name] for name in numerical_integrity_names)
        ),
    }
    arrays = {
        **arrays,
        "profile_residual": np.asarray(profile_conditional.prediction)
        - np.asarray(context.target),
        "truth_model": np.asarray(context.conditional((truth, *truth_nonlinear)).prediction),
        "truth_residual": np.asarray(context.conditional((truth, *truth_nonlinear)).prediction)
        - np.asarray(context.target),
        "k_pair": np.asarray(context.k_pair)[context.indices],
    }
    return result, arrays


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "method",
        "input_fNL",
        "target_realizations",
        "fNL_hat",
        "lower",
        "upper",
        "half_width",
        "bias_hat_minus_input",
        "bias_over_half_width",
        "truth_inside_interval",
        "truth_delta_chi2",
        "b1",
        "b2",
        "gamma2",
        "Ashot_residual",
        "Bshot_residual",
        "bphi",
        "profile_data_chi2",
        "profile_prior_chi2",
        "profile_dof",
        "profile_gof_p_value",
        "hartlap",
        "profile_valid",
        "interval_closed",
        "integrity_pass",
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            interval = row["delta_chi2_1_interval"]
            recovery = row["recovery"]
            nuisance = row["profile_nuisance"]["values"]
            writer.writerow(
                {
                    "method": row["method"],
                    "input_fNL": row["input_fNL"],
                    "target_realizations": row["context_contract"][
                        "target_realization_count"
                    ],
                    "fNL_hat": recovery["fNL_hat"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                    "half_width": interval["half_width"],
                    "bias_hat_minus_input": recovery["bias_hat_minus_input"],
                    "bias_over_half_width": recovery["bias_over_profile_half_width"],
                    "truth_inside_interval": recovery[
                        "truth_inside_delta_chi2_1_interval"
                    ],
                    "truth_delta_chi2": recovery["truth_profile_delta_chi2"],
                    "b1": nuisance["b1"],
                    "b2": nuisance["b2"],
                    "gamma2": nuisance["gamma2"],
                    "Ashot_residual": nuisance["Ashot_residual"],
                    "Bshot_residual": nuisance["Bshot_residual"],
                    "bphi": row["best_fit_png_bias"]["bphi"],
                    "profile_data_chi2": row["profile_data_chi2"],
                    "profile_prior_chi2": row["profile_prior_chi2"],
                    "profile_dof": row["profile_dof"],
                    "profile_gof_p_value": row["profile_gof_p_value"],
                    "hartlap": row["context"]["hartlap"],
                    "profile_valid": row["profile"]["valid"],
                    "interval_closed": row["integrity_checks"][
                        "interval_closed_before_bounds"
                    ],
                    "integrity_pass": row["integrity_pass"],
                }
            )
    temporary.replace(OUTPUT_CSV)


def make_pdf(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colours = {"pre": "#4E79A7", "standard": "#E15759"}
    labels = {"pre": "pre reconstruction", "standard": "standard reconstruction"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex="col")
    for column, method in enumerate(METHODS):
        local = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["input_fNL"],
        )
        truth = np.asarray([row["input_fNL"] for row in local])
        fhat = np.asarray([row["recovery"]["fNL_hat"] for row in local])
        lower = np.asarray([row["delta_chi2_1_interval"]["lower"] for row in local])
        upper = np.asarray([row["delta_chi2_1_interval"]["upper"] for row in local])
        half = np.asarray([row["delta_chi2_1_interval"]["half_width"] for row in local])
        axes[0, column].errorbar(
            truth,
            fhat,
            yerr=np.vstack((fhat - lower, upper - fhat)),
            color=colours[method],
            marker="o",
            ms=6,
            lw=1.2,
            capsize=4,
            label=labels[method],
        )
        extent = max(120.0, float(np.max(np.abs(np.concatenate((lower, upper))))))
        diagonal = np.linspace(-extent, extent, 200)
        axes[0, column].plot(diagonal, diagonal, color="0.25", ls="--", lw=1.0)
        axes[0, column].set_xlim(-120.0, 120.0)
        axes[0, column].set_ylabel(r"recovered $\hat f_{\rm NL}$")
        axes[0, column].set_title(labels[method])
        bias = fhat - truth
        axes[1, column].plot(
            truth, bias, color=colours[method], marker="o", ms=6, lw=1.2
        )
        axes[1, column].axhline(0.0, color="0.25", lw=0.9)
        bias_limit = max(10.0, 1.15 * float(np.max(np.abs(bias))))
        axes[1, column].set_ylim(-bias_limit, bias_limit)
        axes[1, column].set_ylabel(r"$\hat f_{\rm NL}-f_{\rm NL}^{\rm input}$")
        axes[1, column].set_xlabel(r"input $f_{\rm NL}$")
        for axis in axes[:, column]:
            axis.grid(alpha=0.20)
        if method == "standard":
            # The one-realization upper intervals reach O(10^4), which makes
            # the central recovery invisible in the full-range panel.  This
            # inset shows only the already-plotted centres, never a second fit.
            inset = axes[0, column].inset_axes((0.10, 0.60, 0.48, 0.33))
            inset.plot(truth, fhat, color=colours[method], marker="o", ms=3.5)
            inset.plot((-120.0, 120.0), (-120.0, 120.0), color="0.25", ls="--", lw=0.7)
            inset.set_xlim(-120.0, 120.0)
            inset.set_ylim(-130.0, 130.0)
            inset.set_title("centres (zoom)", fontsize=7)
            inset.tick_params(labelsize=6)
            inset.grid(alpha=0.18)
    fig.suptitle(
        r"ordinary Quijote halo $B_{000}$ recovery, $z=0.5$, "
        rf"$k_{{\max}}=0.10\,h\,{{\rm Mpc}}^{{-1}}$, $p=1.12$, "
        rf"$V={COVARIANCE_VOLUME:g}\,(h^{{-1}}{{\rm Gpc}})^3$",
        y=0.975,
    )
    fig.text(
        0.5,
        0.025,
        (
            f"Error bars use Fid500 covariance scaled to V={COVARIANCE_VOLUME:g} "
            "(Gpc/h)^3; they are not errors on the ensemble mean."
        ),
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.12, top=0.88, hspace=0.32, wspace=0.28
    )
    fig.savefig(
        OUTPUT_PDF,
        metadata={**PDF_METADATA, "Title": "local-PNG p=1.12 direct recovery profiles"},
    )
    plt.close(fig)


def make_bestfit_pdfs(rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    """Plot diagonal B000 measurement means and the corresponding best fits."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.06,
        }
    )
    colours = {
        -100.0: "#3B4CC0",
        -50.0: "#7B9FF9",
        0.0: "#595959",
        50.0: "#F4987A",
        100.0: "#B40426",
    }
    method_titles = {
        "pre": "Pre reconstruction",
        "standard": r"Standard post reconstruction ($R=10\,h^{-1}{\rm Mpc}$)",
    }
    output_paths = {
        "pre": OUTPUT_BESTFIT_PRE_PDF,
        "standard": OUTPUT_BESTFIT_STANDARD_PDF,
    }

    for method in METHODS:
        local_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["input_fNL"],
        )
        reference_key = f"{method}__{input_tag(0.0)}"
        selected_k = np.asarray(arrays[f"{reference_key}__k_pair"], dtype=np.float64)
        diagonal_positions = np.flatnonzero(
            np.isclose(selected_k[:, 0], selected_k[:, 1], rtol=0.0, atol=5.0e-8)
        )
        if diagonal_positions.size != 4:
            raise ValueError(
                f"expected four diagonal B000 configurations, got {diagonal_positions.size}"
            )
        k = np.mean(selected_k[diagonal_positions], axis=1)
        order = np.argsort(k)
        k = k[order]
        positions = diagonal_positions[order]
        scale = k**2

        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(7.4, 6.9),
            sharex=True,
            gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.08},
        )
        maximum_pull = 0.0
        for row in local_rows:
            truth = float(row["input_fNL"])
            key = f"{method}__{input_tag(truth)}"
            target = np.asarray(arrays[f"{key}__target"], dtype=np.float64)[positions]
            model = np.asarray(arrays[f"{key}__profile_model"], dtype=np.float64)[positions]
            covariance = np.asarray(arrays[f"{key}__covariance"], dtype=np.float64)
            sigma = np.sqrt(np.diag(covariance))[positions]
            pull = (target - model) / sigma
            maximum_pull = max(maximum_pull, float(np.max(np.abs(pull))))
            colour = colours[truth]
            top.errorbar(
                k,
                scale * target,
                yerr=scale * sigma,
                fmt="o",
                color=colour,
                mfc=colour,
                mec="white",
                mew=0.45,
                ms=5.2,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=3,
            )
            top.plot(k, scale * model, color=colour, lw=1.8, zorder=2)
            bottom.plot(k, pull, color=colour, marker="o", ms=4.2, lw=1.35)

        top.set_title(method_titles[method], pad=7)
        top.set_ylabel(r"$k^2 B_{000}(k,k)\ [h^{-4}{\rm Mpc}^{4}]$")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        top.yaxis.set_major_formatter(formatter)
        bottom.axhspan(-1.0, 1.0, color="0.92", zorder=0)
        bottom.axhline(0.0, color="0.35", lw=0.85, zorder=1)
        pull_limit = max(1.5, 0.5 * math.ceil(2.2 * maximum_pull))
        bottom.set_ylim(-pull_limit, pull_limit)
        bottom.set_ylabel(
            rf"$(\overline{{B}}_{{\rm sim}}-B_{{\rm fit}})/\sigma_{{V={COVARIANCE_VOLUME:g}}}$"
        )
        bottom.set_xlabel(r"$k_1=k_2=k\ [h\,{\rm Mpc}^{-1}]$")
        for axis in (top, bottom):
            axis.grid(True, color="0.86", lw=0.65, alpha=0.75)
            axis.tick_params(direction="in", top=True, right=True)
        pad = 0.12 * (float(np.max(k)) - float(np.min(k)))
        bottom.set_xlim(float(np.min(k)) - pad, float(np.max(k)) + pad)

        colour_handles = [
            Line2D(
                [],
                [],
                color=colours[truth],
                marker="o",
                lw=1.6,
                ms=4.5,
                label=rf"input $f_{{\rm NL}}={truth:+.0f}$",
            )
            for truth in INPUT_FNL
        ]
        style_handles = [
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                ls="none",
                ms=4.5,
                label="measurement mean with target-volume error",
            ),
            Line2D([], [], color="black", lw=1.8, label="six-parameter best fit"),
        ]
        first_legend = top.legend(
            handles=colour_handles,
            loc="best",
            frameon=False,
            ncol=2,
            fontsize=8,
        )
        top.add_artist(first_legend)
        top.legend(handles=style_handles, loc="lower right", frameon=False, fontsize=7.5)
        fig.suptitle(
            rf"Ordinary Quijote halo diagonal $B_{{000}}$, $z=0.5$, "
            rf"$k_{{\max}}=0.10\,h\,{{\rm Mpc}}^{{-1}}$, $p=1.12$",
            y=0.985,
            fontsize=13,
        )
        fig.text(
            0.5,
            0.020,
            (
                "Fit uses 14 B000 configurations; four diagonal points are shown.\n"
                f"Error bars use Fid500 covariance scaled to V={COVARIANCE_VOLUME:g} "
                "(Gpc/h)^3, not errors on the measurement mean."
            ),
            ha="center",
            fontsize=7.2,
        )
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.91)
        fig.savefig(
            output_paths[method],
            bbox_inches=None,
            metadata={
                **PDF_METADATA,
                "Title": f"{method} local-PNG best-fit B000 versus measurement means",
            },
        )
        plt.close(fig)


def write_report(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    prior_description = (
        f"input-centred flat prior with half-width {FNL_PRIOR_HALF_WIDTH:g}"
        if FNL_PRIOR_HALF_WIDTH is not None
        else f"wide numerical profiling range {FNL_BOUNDS}"
    )
    lines = [
        "# Ordinary Quijote halo local-PNG direct recovery",
        "",
        f"Created UTC: `{payload['created_utc']}`.",
        "",
        "## Frozen contract",
        "",
        "- ordinary FoF halo, z=0.5, pre and standard reconstruction R=10 Mpc/h;",
        "- B000 only, kmax=0.10 h/Mpc, 14 retained configurations;",
        "- model: Gaussian + fNL times the existing linear local-PNG contribution;",
        "- b_phi=2 delta_c (b1-1.12), with the existing p=1 b_phidelta treatment preserved;",
        "- six fitted parameters: fNL,b1,b2,gamma2,Ashot_residual,Bshot_residual;",
        f"- fNL prior: {prior_description};",
        f"- covariance: {covariance_description()};",
        "- no fNL^2 term, response gain, P(k) prior, EFT, or new bias parameter.",
        "",
        "## Recovery results",
        "",
        "| method | input fNL | recovered fNL (+/-) | flat prior | prior-limited side | bias | truth Delta-chi2 | chi2/dof | p-value |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        recovery = row["recovery"]
        interval = row["delta_chi2_1_interval"]
        plus = float(interval["upper"] - recovery["fNL_hat"])
        minus = float(recovery["fNL_hat"] - interval["lower"])
        limited = []
        if not interval["lower_closed_before_bound"]:
            limited.append("lower")
        if not interval["upper_closed_before_bound"]:
            limited.append("upper")
        limited_label = "+".join(limited) if limited else "none"
        prior = row["context_contract"]["fNL_flat_prior"]
        lines.append(
            "| {method} | {truth:.0f} | {hat:.4f} (+{plus:.4f}/-{minus:.4f}) | "
            "[{prior0:.0f},{prior1:.0f}] | {limited} | {bias:.4f} | {delta:.4f} | "
            "{chi2:.4f}/{dof:d} | {pvalue:.6f} |".format(
                method=row["method"],
                truth=row["input_fNL"],
                hat=recovery["fNL_hat"],
                plus=plus,
                minus=minus,
                prior0=prior[0],
                prior1=prior[1],
                limited=limited_label,
                bias=recovery["bias_hat_minus_input"],
                delta=recovery["truth_profile_delta_chi2"],
                chi2=row["profile_data_chi2"],
                dof=row["profile_dof"],
                pvalue=row["profile_gof_p_value"],
            )
        )
    lines.extend(
        [
            "",
            "## Direct conclusion",
            "",
            (
                "- pre maximum absolute central recovery bias: "
                f"`{payload['summary']['method_maximum_absolute_central_bias']['pre']:.6f}`;"
            ),
            (
                "- standard maximum absolute central recovery bias: "
                f"`{payload['summary']['method_maximum_absolute_central_bias']['standard']:.6f}`;"
            ),
            (
                "- profiles with at least one Delta-chi2=1 side limited by the flat prior: "
                f"`{payload['summary']['prior_limited_profile_count']}/10`;"
            ),
            "",
            "## Integrity",
            "",
            f"- all ten numerical profiles valid: `{payload['summary']['all_integrity_checks_pass']}`;",
            f"- all Delta-chi2=1 sides close before their flat-prior bounds: `{payload['summary']['all_intervals_closed_before_prior_bounds']}`;",
            f"- truth inside every reported profile range: `{payload['summary']['truth_inside_all_intervals']}`;",
            f"- all profile goodness-of-fit p-values above 0.05: `{payload['summary']['all_profile_gof_p_above_0p05']}`;",
            f"- maximum absolute normalized recovery bias: `{payload['summary']['maximum_absolute_bias_over_half_width']:.6f}`.",
            "",
            f"The covariance corresponds to V={COVARIANCE_VOLUME:g} (Gpc/h)^3.  The targets are ensemble means, but no division by the number of target realizations enters the covariance.",
            "",
            "## Figures",
            "",
            f"- `{OUTPUT_PDF.name}`: recovered centres and profile errors;",
            f"- `{OUTPUT_BESTFIT_PRE_PDF.name}`: all five pre measurement means and best fits;",
            f"- `{OUTPUT_BESTFIT_STANDARD_PDF.name}`: all five standard-post measurement means and best fits.",
            "",
        ]
    )
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_REPORT.with_name(f".{OUTPUT_REPORT.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(OUTPUT_REPORT)


def main() -> None:
    for path in (
        PROJECTED_RUNNER,
        PROFILE_RUNNER,
        REFINED_ASSETS,
        INTERVAL_HELPER,
        BASE_MATRIX,
        FNL50_MATRIX,
        FNL100_MATRIX,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_matrix = load_npz(BASE_MATRIX)
    fnl50_matrix = load_npz(FNL50_MATRIX)
    fnl100_matrix = load_npz(FNL100_MATRIX)
    input_validation = validate_inputs(baseline_matrix, fnl50_matrix, fnl100_matrix)

    projected = import_path("_p112_recovery_projected", PROJECTED_RUNNER)
    profiler = import_path("_p112_recovery_profiler", PROFILE_RUNNER)
    refiner = import_path("_p112_recovery_refiner", REFINED_ASSETS)
    interval_helper = import_path("_p112_recovery_interval", INTERVAL_HELPER)
    interval_helper.FNL_BOUNDS = FNL_BOUNDS

    baseline, physical, base, audit, assets, standard_method, physical_cache = (
        projected.load_modules_and_assets()
    )
    audit.VOLUME = float(COVARIANCE_VOLUME)
    assets, refinement = refiner.refined_assets(assets)
    matrix = projected.load_matrix()
    tree = projected.load_pre_tree(matrix)
    dm_tree, dm_total, dm_audit = projected.load_z05_dm_response(matrix)
    bias = json.loads(projected.BIAS_JSON.read_text(encoding="utf-8"))
    growth_ratio = float(base.legacy.configurations()["z0p5"].growth_power_ratio_to_z1)
    nbar = float(bias["nbar_mean"])
    base_contexts = profiler.build_contexts(
        projected,
        physical,
        base,
        assets,
        standard_method,
        physical_cache,
        matrix,
        tree,
        dm_tree,
        dm_total,
        growth_ratio,
        nbar,
        KMAX,
        refinement,
    )

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        fiducial = np.asarray(
            baseline_matrix[
                "fiducial_pre_B000" if method == "pre" else "fiducial_post_B000"
            ],
            dtype=np.float64,
        )
        for truth in INPUT_FNL:
            samples = target_samples(
                method, truth, baseline_matrix, fnl50_matrix, fnl100_matrix
            )
            context, context_contract = recovery_context(
                base_contexts[method], method, truth, samples, fiducial
            )
            row, local_arrays = profile_case(
                audit=audit,
                base=base,
                assets=assets,
                interval_helper=interval_helper,
                context=context,
                method=method,
                truth=truth,
                context_contract=context_contract,
            )
            rows.append(row)
            key = f"{method}__{input_tag(truth)}"
            for name, value in local_arrays.items():
                arrays[f"{key}__{name}"] = np.asarray(value)
            print(
                f"{key}: input={truth:g}, fNL={row['recovery']['fNL_hat']:.6g}, "
                f"interval=[{row['delta_chi2_1_interval']['lower']:.6g},"
                f"{row['delta_chi2_1_interval']['upper']:.6g}], "
                f"bias/half={row['recovery']['bias_over_profile_half_width']:.4f}, "
                f"chi2/dof={row['profile_data_chi2']:.4f}/{row['profile_dof']}",
                flush=True,
            )

    maximum_normalized_bias = max(
        abs(float(row["recovery"]["bias_over_profile_half_width"])) for row in rows
    )
    summary = {
        "all_integrity_checks_pass": bool(all(row["integrity_pass"] for row in rows)),
        "all_intervals_closed_before_prior_bounds": bool(
            all(
                row["integrity_checks"]["interval_closed_before_bounds"]
                for row in rows
            )
        ),
        "prior_limited_profile_count": int(
            sum(
                not row["integrity_checks"]["interval_closed_before_bounds"]
                for row in rows
            )
        ),
        "truth_inside_all_intervals": bool(
            all(
                row["recovery"]["truth_inside_delta_chi2_1_interval"]
                for row in rows
            )
        ),
        "all_profile_gof_p_above_0p05": bool(
            all(float(row["profile_gof_p_value"]) > 0.05 for row in rows)
        ),
        "maximum_absolute_bias_over_half_width": maximum_normalized_bias,
        "method_maximum_absolute_bias_over_half_width": {
            method: max(
                abs(float(row["recovery"]["bias_over_profile_half_width"]))
                for row in rows
                if row["method"] == method
            )
            for method in METHODS
        },
        "method_maximum_absolute_central_bias": {
            method: max(
                abs(float(row["recovery"]["bias_hat_minus_input"]))
                for row in rows
                if row["method"] == method
            )
            for method in METHODS
        },
        "standard_upper_interval_range": (
            min(
                float(row["delta_chi2_1_interval"]["upper"])
                for row in rows
                if row["method"] == "standard"
            ),
            max(
                float(row["delta_chi2_1_interval"]["upper"])
                for row in rows
                if row["method"] == "standard"
            ),
        ),
        "standard_b1_at_upper_range": (
            min(
                float(row["delta_chi2_1_interval"]["b1_at_upper"])
                for row in rows
                if row["method"] == "standard"
            ),
            max(
                float(row["delta_chi2_1_interval"]["b1_at_upper"])
                for row in rows
                if row["method"] == "standard"
            ),
        ),
    }
    payload = {
        "schema": "quijote-z0p5-ordinary-local-png-p112-direct-recovery-v2",
        "created_utc": utc_now(),
        "status": "complete" if summary["all_integrity_checks_pass"] else "integrity_failure",
        "scope": {
            "sample": "ordinary Quijote FoF halo Mmin=1e13",
            "redshift": 0.5,
            "methods": METHODS,
            "standard_reconstruction_R_mpc_h": 10.0,
            "statistic": "B000 only",
            "kmax_h_mpc": KMAX,
            "retained_indices": EXPECTED_INDICES,
            "retained_configuration_count": int(EXPECTED_INDICES.size),
            "input_fNL": INPUT_FNL,
            "fitted_parameters": (
                "fNL",
                "b1",
                "b2",
                "gamma2",
                "Ashot_residual",
                "Bshot_residual",
            ),
            "fNL_flat_prior_half_width": FNL_PRIOR_HALF_WIDTH,
            "fNL_flat_prior_by_input": {
                input_tag(truth): fnl_bounds_for_truth(truth) for truth in INPUT_FNL
            },
            "b1_source": "B000 only; zero P(k) prior precision",
            "target_volume_gpc3_h3": float(COVARIANCE_VOLUME),
            "covariance": covariance_description(),
        },
        "model": {
            "formula": "B_G + fNL * B_PNG_linear",
            "p": P_BPHI,
            "delta_c": DELTA_C,
            "bphi_formula": "2*delta_c*(b1-p)",
            "bphidelta_policy": "preserve existing p=1 relation",
            "fNL2_terms_included": False,
            "response_gain_included": False,
            "new_bias_parameter_included": False,
            "P_likelihood_or_prior_used": False,
            "MCMC_ran": False,
            "frozen_response_cache_only": True,
            "live_analytic_executable_used": False,
        },
        "input_validation": input_validation,
        "refinement": refinement,
        "dm_response_audit": dm_audit,
        "summary": summary,
        "rows": rows,
        "inputs": {
            "runner": artifact(ENTRYPOINT),
            "base_runner": artifact(Path(__file__).resolve()),
            "base_matrix": artifact(BASE_MATRIX),
            "fnl50_matrix": artifact(FNL50_MATRIX),
            "fnl100_matrix": artifact(FNL100_MATRIX),
        },
    }

    write_csv(rows)
    atomic_npz(OUTPUT_ARRAYS, arrays)
    make_pdf(rows)
    make_bestfit_pdfs(rows, arrays)
    payload["outputs"] = {
        "csv": artifact(OUTPUT_CSV),
        "arrays": artifact(OUTPUT_ARRAYS),
        "recovery_pdf": artifact(OUTPUT_PDF),
        "pre_bestfit_pdf": artifact(OUTPUT_BESTFIT_PRE_PDF),
        "standard_bestfit_pdf": artifact(OUTPUT_BESTFIT_STANDARD_PDF),
    }
    write_report(payload)
    payload["outputs"]["report"] = artifact(OUTPUT_REPORT)
    atomic_json(OUTPUT_JSON, payload)
    print(json.dumps(jsonable(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
