#!/usr/bin/env python3
"""Profile direct finite-fNL pre-P0-only and B000-only likelihoods.

There is deliberately no P+B likelihood in this producer.  The primary
B000-only result fixes b1=1.93 to retain the established comparison contract;
a separate free-b1 B000-only diagnostic is saved without a Gaussian b1 prior.
The pre-P0-only likelihood always profiles b1 and a residual constant shot
noise.  Every model evaluation is a direct finite-fNL call at the current
parameters, with no frozen tangent or response-gain parameter.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

for _thread_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MARISA_B_MAX_THREADS",
):
    os.environ[_thread_name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from iminuit import Minuit
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution

from quijote_halo_z0p5_direct_finite_fnl_models import (
    B_COUNT,
    DirectB000Model,
    DirectP0Model,
    KMAX,
    P_COUNT,
    TRUTHS,
    b_fiducial_samples,
    b_geometry,
    b_target_samples,
    direct_model_closure,
    load_runtime,
    p_fiducial_samples,
    p_target_samples,
)


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model")
CODES = ROOT / "marisa_b_core"
TAG = "quijote_halo_z0p5_direct_finite_fnl_single_stat_kmax0p08_v42p875"
ANALYSIS = ROOT / "analysis" / TAG
FIGURES = ROOT / "figures/diagnostics" / TAG
P_THEORY = (
    ROOT
    / "analysis/quijote_halo_z0p5_pre_p0_direct_finite_fnl_kmax0p08_v42p875"
    / "PRE_P0_DIRECT_FINITE_THEORY_LATTICE.npz"
)
P_THEORY_MANIFEST = P_THEORY.with_suffix(".json")
MARISA_THEORY = (
    ROOT
    / "analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1"
    / "theory/marisa_adaptive_finite_tree_bases.npz"
)
MARISA_THEORY_MANIFEST = MARISA_THEORY.with_suffix(".json")

OUTPUT_JSON = ANALYSIS / "DIRECT_FINITE_FNL_SINGLE_STAT_PROFILES.json"
OUTPUT_CSV = ANALYSIS / "DIRECT_FINITE_FNL_SINGLE_STAT_PROFILES.csv"
OUTPUT_NPZ = ANALYSIS / "DIRECT_FINITE_FNL_SINGLE_STAT_ARRAYS.npz"
OUTPUT_RECOVERY = FIGURES / "direct_finite_fNL_single_stat_recovery.pdf"
OUTPUT_PARAMETERS = FIGURES / "direct_finite_fNL_single_stat_all_parameters.pdf"
OUTPUT_BESTFIT = FIGURES / "direct_finite_fNL_single_stat_bestfit.pdf"

METHODS = ("pre", "standard", "marisa")
VOLUME = 42.875
FNL_HALF_WIDTH = 200.0
FIXED_B1 = 1.93
B1_BOUNDS = (0.1, 6.0)
PSHOT_BOUNDS = (-1.0e5, 1.0e5)
P_HARTLAP = (500.0 - P_COUNT - 2.0) / 499.0
B_HARTLAP = (500.0 - B_COUNT - 2.0) / 499.0

P_NAMES = ("fNL", "b1", "Pshot_residual")
B_FIXED_NAMES = (
    "fNL",
    "b2",
    "gamma2",
    "Ashot_residual",
    "Bshot_residual",
)
B_FREE_NAMES = (
    "fNL",
    "b1",
    "b2",
    "gamma2",
    "Ashot_residual",
    "Bshot_residual",
)
METHOD_LABELS = {
    "pre": "pre reconstruction",
    "standard": "standard reconstruction",
    "marisa": "Marisa reconstruction",
}
METHOD_COLOURS = {"pre": "#4E79A7", "standard": "#E15759", "marisa": "#7A4EAB"}
TRUTH_COLOURS = {
    -100.0: "#3B4CC0",
    -50.0: "#7B9FF9",
    0.0: "#595959",
    50.0: "#F4987A",
    100.0: "#B40426",
}
PARAMETER_LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "b2": r"$b_2$",
    "gamma2": r"$\gamma_2$",
    "Pshot_residual": r"$P_{\rm shot,res}$",
    "Ashot_residual": r"$A_{\rm shot}$",
    "Bshot_residual": r"$B_{\rm shot}$",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def load_p_model() -> tuple[DirectP0Model, dict[str, np.ndarray]]:
    if not P_THEORY.is_file() or not P_THEORY_MANIFEST.is_file():
        raise FileNotFoundError(P_THEORY)
    manifest = json.loads(P_THEORY_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError("direct P theory manifest is not pass")
    with np.load(P_THEORY, allow_pickle=False) as handle:
        theory = {name: np.asarray(handle[name]) for name in handle.files}
    model = DirectP0Model(
        pl=np.asarray(theory["PL_lattice"], dtype=np.float64),
        pl_over_m=np.asarray(theory["PL_over_M_lattice"], dtype=np.float64),
        pl_over_m2=np.asarray(theory["PL_over_M2_lattice"], dtype=np.float64),
    )
    return model, theory


def covariance_contract(
    fiducial: np.ndarray, dimension: int, hartlap: float
) -> tuple[np.ndarray, np.ndarray]:
    if fiducial.shape != (500, dimension):
        raise ValueError(f"invalid Fid500 covariance samples: {fiducial.shape}")
    covariance_one = np.cov(fiducial, rowvar=False, ddof=1)
    covariance_fit = covariance_one / VOLUME
    cholesky = np.linalg.cholesky(covariance_fit) / math.sqrt(hartlap)
    return covariance_one, cholesky


def minuit_record(minimizer: Minuit, candidates: list[Minuit]) -> dict[str, Any]:
    return {
        "valid": bool(minimizer.valid),
        "accurate": bool(minimizer.accurate),
        "fval": float(minimizer.fval),
        "edm": float(minimizer.fmin.edm),
        "has_parameters_at_limit": bool(minimizer.fmin.has_parameters_at_limit),
        "has_reached_call_limit": bool(minimizer.fmin.has_reached_call_limit),
        "candidate_fvals": [float(item.fval) for item in candidates],
        "candidate_valid": [bool(item.valid) for item in candidates],
    }


def configure_minuit(minimizer: Minuit, names: tuple[str, ...], truth: float) -> None:
    minimizer.errordef = Minuit.LEAST_SQUARES
    minimizer.limits["fNL"] = (truth - FNL_HALF_WIDTH, truth + FNL_HALF_WIDTH)
    if "b1" in names:
        minimizer.limits["b1"] = B1_BOUNDS
    if "Pshot_residual" in names:
        minimizer.limits["Pshot_residual"] = PSHOT_BOUNDS
    for name, step in {
        "fNL": 8.0,
        "b1": 0.02,
        "b2": 0.08,
        "gamma2": 0.06,
        "Pshot_residual": 100.0,
        "Ashot_residual": 0.1,
        "Bshot_residual": 0.08,
    }.items():
        if name in names:
            minimizer.errors[name] = step
    minimizer.strategy = 1
    minimizer.tol = 1.0e-4


def run_multistart(
    objective: Callable[..., float],
    names: tuple[str, ...],
    starts: list[dict[str, float]],
    truth: float,
) -> tuple[Minuit, list[Minuit]]:
    candidates: list[Minuit] = []
    for start in starts:
        minimizer = Minuit(
            objective,
            *[float(start[name]) for name in names],
            name=names,
        )
        configure_minuit(minimizer, names, truth)
        minimizer.migrad(ncall=100000, iterate=8, use_simplex=True)
        minimizer.hesse()
        candidates.append(minimizer)
    valid = [item for item in candidates if item.valid and math.isfinite(float(item.fval))]
    best = min(valid or candidates, key=lambda item: float(item.fval))
    if not best.valid or not math.isfinite(float(best.fval)):
        raise RuntimeError(f"no valid profile MAP for truth={truth}: {best.fmin}")
    return best, candidates


def run_minos(minimizer: Minuit, names: tuple[str, ...]) -> dict[str, Any]:
    try:
        minimizer.minos(*names, ncall=150000)
    except Exception as error:  # pragma: no cover - recorded fail-closed below
        minos_error = f"{type(error).__name__}: {error}"
    else:
        minos_error = None
    result: dict[str, Any] = {}
    for name in names:
        if name in minimizer.merrors:
            item = minimizer.merrors[name]
            result[name] = {
                "lower": float(minimizer.values[name] + item.lower),
                "upper": float(minimizer.values[name] + item.upper),
                "lower_error": float(item.lower),
                "upper_error": float(item.upper),
                "valid": bool(item.is_valid),
                "lower_valid": bool(item.lower_valid),
                "upper_valid": bool(item.upper_valid),
                "at_lower_limit": bool(item.at_lower_limit),
                "at_upper_limit": bool(item.at_upper_limit),
            }
        else:
            error = float(minimizer.errors[name])
            result[name] = {
                "lower": float(minimizer.values[name] - error),
                "upper": float(minimizer.values[name] + error),
                "lower_error": -error,
                "upper_error": error,
                "valid": False,
                "lower_valid": False,
                "upper_valid": False,
                "at_lower_limit": False,
                "at_upper_limit": False,
            }
    result["exception"] = minos_error
    return result


def generalized_p_start(
    target: np.ndarray, covariance_one: np.ndarray, pl: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack((pl, np.ones(P_COUNT)))
    solved_design = np.linalg.solve(covariance_one, design)
    solved_target = np.linalg.solve(covariance_one, target)
    coefficients = np.linalg.solve(design.T @ solved_design, design.T @ solved_target)
    b1 = float(np.clip(math.sqrt(max(float(coefficients[0]), 0.01)), *B1_BOUNDS))
    pshot = float(np.clip(coefficients[1], *PSHOT_BOUNDS))
    return b1, pshot


def profile_p_case(
    model: DirectP0Model,
    theory: dict[str, np.ndarray],
    target: np.ndarray,
    covariance_one: np.ndarray,
    cholesky: np.ndarray,
    truth: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    def evaluate(mapping: dict[str, float]) -> tuple[float, np.ndarray]:
        prediction = model(mapping["fNL"], mapping["b1"], mapping["Pshot_residual"])
        white = solve_triangular(cholesky, prediction - target, lower=True)
        return float(white @ white), prediction

    def objective(*values: float) -> float:
        return evaluate(dict(zip(P_NAMES, values)))[0]

    b1_start, pshot_start = generalized_p_start(
        target, covariance_one, np.asarray(theory["PL_lattice"], dtype=np.float64)
    )
    starts = [
        {"fNL": fnl, "b1": b1, "Pshot_residual": pshot}
        for fnl, b1, pshot in (
            (truth, b1_start, pshot_start),
            (0.0, 1.93, pshot_start),
            (truth - 50.0, 2.1, 0.0),
            (truth + 50.0, 1.8, 0.0),
        )
    ]
    for start in starts:
        start["fNL"] = float(np.clip(start["fNL"], truth - 199.0, truth + 199.0))
    best, candidates = run_multistart(objective, P_NAMES, starts, truth)
    mapping = {name: float(best.values[name]) for name in P_NAMES}
    data_chi2, prediction = evaluate(mapping)
    minos = run_minos(best, P_NAMES)
    dof = P_COUNT - len(P_NAMES)
    row = {
        "statistic": "pre_P0_only",
        "method": "pre",
        "fit_mode": "free_b1_primary",
        "input_fNL": truth,
        "target_realization_count": 500 if truth == 0.0 else 250,
        "parameter_names": list(P_NAMES),
        "map": mapping,
        "hesse_errors": {name: float(best.errors[name]) for name in P_NAMES},
        "minos": minos,
        "data_chi2": data_chi2,
        "objective": float(best.fval),
        "dof": dof,
        "gof_p_value": float(chi2_distribution.sf(data_chi2, dof)),
        "optimizer": minuit_record(best, candidates),
        "flat_fNL_prior": [truth - FNL_HALF_WIDTH, truth + FNL_HALF_WIDTH],
        "model": {
            "callable": "P0_pre(fNL,b1,Pshot_residual)",
            "density_bias_order": "linear b1 only",
            "finite_png_order": 2,
            "b1_gaussian_prior": False,
            "higher_order_bias": False,
            "frozen_response": False,
        },
    }
    return row, {
        "target": target,
        "prediction": prediction,
        "residual": target - prediction,
        "covariance_single_realization": covariance_one,
    }


def b_prior_chi2(context: Any, mapping: dict[str, float]) -> float:
    values = np.asarray([mapping[name] for name in context.names], dtype=np.float64)
    delta = values - np.asarray(context.prior.mean, dtype=np.float64)
    white = np.asarray(context.prior.whitener, dtype=np.float64) @ delta
    return float(white @ white)


def profile_b_case(
    model: DirectB000Model,
    target: np.ndarray,
    covariance_one: np.ndarray,
    cholesky: np.ndarray,
    truth: float,
    free_b1: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names = B_FREE_NAMES if free_b1 else B_FIXED_NAMES
    context = model.base_context
    prior_mean = {
        name: float(context.prior.mean[context.names.index(name)])
        for name in context.names
    }

    def mapping_from_values(values: tuple[float, ...]) -> dict[str, float]:
        mapping = dict(zip(names, values))
        if not free_b1:
            mapping["b1"] = FIXED_B1
        return {name: float(mapping[name]) for name in ("fNL", *context.names)}

    def evaluate(mapping: dict[str, float]) -> tuple[float, float, np.ndarray]:
        prediction = model(mapping["fNL"], mapping)
        white = solve_triangular(cholesky, prediction - target, lower=True)
        return float(white @ white), b_prior_chi2(context, mapping), prediction

    def objective(*values: float) -> float:
        mapping = mapping_from_values(values)
        data_q, prior_q, _prediction = evaluate(mapping)
        return data_q + prior_q

    starts: list[dict[str, float]] = []
    for fnl in (truth, 0.0, truth - 60.0, truth + 60.0):
        start = {
            "fNL": float(np.clip(fnl, truth - 199.0, truth + 199.0)),
            "b1": FIXED_B1,
            "b2": prior_mean["b2"],
            "gamma2": prior_mean["gamma2"],
            "Ashot_residual": prior_mean["Ashot_residual"],
            "Bshot_residual": prior_mean["Bshot_residual"],
        }
        if free_b1 and len(starts) > 0:
            start["b1"] = (1.6, 2.15, 2.5)[(len(starts) - 1) % 3]
        starts.append({name: start[name] for name in names})
    best, candidates = run_multistart(objective, names, starts, truth)
    mapping = mapping_from_values(tuple(float(best.values[name]) for name in names))
    data_chi2, prior_chi2, prediction = evaluate(mapping)
    minos = run_minos(best, names)
    if not free_b1:
        minos["b1"] = {
            "lower": FIXED_B1,
            "upper": FIXED_B1,
            "lower_error": 0.0,
            "upper_error": 0.0,
            "valid": True,
            "fixed": True,
        }
    dof = B_COUNT - len(names)
    row = {
        "statistic": "B000_only",
        "method": model.method,
        "fit_mode": "free_b1_diagnostic" if free_b1 else "fixed_b1_primary",
        "input_fNL": truth,
        "target_realization_count": 500 if truth == 0.0 else 250,
        "parameter_names": list(names),
        "map": mapping,
        "hesse_errors": {name: float(best.errors[name]) for name in names},
        "minos": minos,
        "data_chi2": data_chi2,
        "prior_chi2": prior_chi2,
        "objective": float(best.fval),
        "dof": dof,
        "gof_p_value_data_only": float(chi2_distribution.sf(data_chi2, dof)),
        "optimizer": minuit_record(best, candidates),
        "flat_fNL_prior": [truth - FNL_HALF_WIDTH, truth + FNL_HALF_WIDTH],
        "model": {
            "callable": f"B000_{model.method}(fNL,b1,b2,gamma2,Ashot,Bshot)",
            "fully_tree_dm": True,
            "finite_png_order": 2,
            "frozen_response": False,
            "empirical_gain": False,
            "matter_loop": False,
            "b1_fixed": None if free_b1 else FIXED_B1,
            "b1_gaussian_prior": False,
        },
    }
    return row, {
        "target": target,
        "prediction": prediction,
        "residual": target - prediction,
        "covariance_single_realization": covariance_one,
        **model.parts(mapping),
    }


def interval_error(row: dict[str, Any], parameter: str) -> tuple[float, float]:
    item = row["minos"][parameter]
    return -float(item["lower_error"]), float(item["upper_error"])


def recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups = sorted({(row["statistic"], row["method"], row["fit_mode"]) for row in rows})
    for statistic, method, fit_mode in groups:
        local = sorted(
            [
                row
                for row in rows
                if row["statistic"] == statistic
                and row["method"] == method
                and row["fit_mode"] == fit_mode
            ],
            key=lambda row: row["input_fNL"],
        )
        truth = np.asarray([row["input_fNL"] for row in local], dtype=np.float64)
        centre = np.asarray([row["map"]["fNL"] for row in local], dtype=np.float64)
        slope, intercept = np.polyfit(truth, centre, 1)
        coverage = []
        for row in local:
            item = row["minos"]["fNL"]
            coverage.append(float(item["lower"]) <= row["input_fNL"] <= float(item["upper"]))
        key = f"{statistic}__{method}__{fit_mode}"
        summary[key] = {
            "centres": centre,
            "biases": centre - truth,
            "slope": float(slope),
            "intercept": float(intercept),
            "mean_absolute_bias": float(np.mean(np.abs(centre - truth))),
            "truth_inside_delta_chi2_1_count": int(sum(coverage)),
            "target_count": len(local),
            "all_optimizer_valid": bool(all(row["optimizer"]["valid"] for row in local)),
            "all_fNL_minos_valid": bool(all(row["minos"]["fNL"]["valid"] for row in local)),
        }
    return summary


def precision_metrics(theory: np.ndarray, measured: np.ndarray, precision: np.ndarray) -> dict[str, float]:
    tpt = float(theory @ precision @ theory)
    mpm = float(measured @ precision @ measured)
    cross = float(theory @ precision @ measured)
    return {
        "measured_on_theory_amplitude": cross / tpt,
        "precision_cosine": cross / math.sqrt(max(tpt * mpm, 1.0e-300)),
        "theory_whitened_norm": math.sqrt(max(tpt, 0.0)),
        "measured_whitened_norm": math.sqrt(max(mpm, 0.0)),
    }


def nuisance_profiled_precision(
    covariance_one: np.ndarray,
    hartlap: float,
    jacobian: np.ndarray,
    prior_precision: np.ndarray,
) -> np.ndarray:
    data_precision = hartlap * VOLUME * np.linalg.inv(covariance_one)
    fisher = jacobian.T @ data_precision @ jacobian + prior_precision
    return (
        data_precision
        - data_precision
        @ jacobian
        @ np.linalg.solve(fisher, jacobian.T @ data_precision)
    )


def odd_even_diagnostics(
    runtime: Any,
    p_model: DirectP0Model,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    p_fid = p_fiducial_samples(runtime)
    p_cov = np.cov(p_fid, rowvar=False, ddof=1)
    p_row = next(
        row
        for row in rows
        if row["statistic"] == "pre_P0_only" and row["input_fNL"] == 0.0
    )
    p_map = p_row["map"]
    p_jacobian = np.column_stack(
        (
            2.0 * float(p_map["b1"]) * np.asarray(p_model.pl, dtype=np.float64),
            np.ones(P_COUNT, dtype=np.float64),
        )
    )
    p_profile_precision = nuisance_profiled_precision(
        p_cov, P_HARTLAP, p_jacobian, np.zeros((2, 2), dtype=np.float64)
    )
    for amplitude in (50.0, 100.0):
        measured_plus = np.mean(p_target_samples(runtime, amplitude), axis=0)
        measured_minus = np.mean(p_target_samples(runtime, -amplitude), axis=0)
        measured_zero = np.mean(p_fid, axis=0)
        measured_odd = (measured_plus - measured_minus) / (2.0 * amplitude)
        measured_even = (measured_plus + measured_minus) / 2.0 - measured_zero
        theory_plus = p_model(amplitude, p_map["b1"], p_map["Pshot_residual"])
        theory_minus = p_model(-amplitude, p_map["b1"], p_map["Pshot_residual"])
        theory_zero = p_model(0.0, p_map["b1"], p_map["Pshot_residual"])
        theory_odd = (theory_plus - theory_minus) / (2.0 * amplitude)
        theory_even = (theory_plus + theory_minus) / 2.0 - theory_zero
        result[f"P0_pm{int(amplitude)}"] = {
            "odd": precision_metrics(theory_odd, measured_odd, p_profile_precision),
            "even": precision_metrics(theory_even, measured_even, p_profile_precision),
        }

    for method in METHODS:
        covariance = np.cov(b_fiducial_samples(runtime, method), rowvar=False, ddof=1)
        model = DirectB000Model(runtime, method)
        row = next(
            item
            for item in rows
            if item["statistic"] == "B000_only"
            and item["method"] == method
            and item["fit_mode"] == "fixed_b1_primary"
            and item["input_fNL"] == 0.0
        )
        mapping = row["map"]
        zero_samples = b_fiducial_samples(runtime, method)
        nuisance_names = B_FIXED_NAMES[1:]
        jacobian_columns = []
        for name in nuisance_names:
            step = 1.0e-4 * max(abs(float(mapping[name])), 1.0)
            plus_mapping = dict(mapping)
            minus_mapping = dict(mapping)
            plus_mapping[name] += step
            minus_mapping[name] -= step
            jacobian_columns.append(
                (model(0.0, plus_mapping) - model(0.0, minus_mapping)) / (2.0 * step)
            )
        jacobian = np.column_stack(jacobian_columns)
        full_prior_precision = (
            np.asarray(model.base_context.prior.whitener, dtype=np.float64).T
            @ np.asarray(model.base_context.prior.whitener, dtype=np.float64)
        )
        positions = np.asarray(
            [model.base_context.names.index(name) for name in nuisance_names],
            dtype=np.int64,
        )
        prior_precision = full_prior_precision[np.ix_(positions, positions)]
        profile_precision = nuisance_profiled_precision(
            covariance, B_HARTLAP, jacobian, prior_precision
        )
        for amplitude in (50.0, 100.0):
            plus_samples = b_target_samples(runtime, method, amplitude)
            minus_samples = b_target_samples(runtime, method, -amplitude)
            measured_odd = (
                np.mean(plus_samples, axis=0) - np.mean(minus_samples, axis=0)
            ) / (2.0 * amplitude)
            measured_even = (
                np.mean(plus_samples, axis=0) + np.mean(minus_samples, axis=0)
            ) / 2.0 - np.mean(zero_samples, axis=0)
            theory_plus = model(amplitude, mapping)
            theory_minus = model(-amplitude, mapping)
            theory_zero = model(0.0, mapping)
            theory_odd = (theory_plus - theory_minus) / (2.0 * amplitude)
            theory_even = (theory_plus + theory_minus) / 2.0 - theory_zero
            result[f"B000_{method}_pm{int(amplitude)}"] = {
                "odd": precision_metrics(theory_odd, measured_odd, profile_precision),
                "even": precision_metrics(theory_even, measured_even, profile_precision),
            }
    return result


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "statistic",
        "method",
        "fit_mode",
        "input_fNL",
        "parameter",
        "centre",
        "plus_error",
        "minus_error",
        "lower",
        "upper",
        "minos_valid",
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ordered = row["parameter_names"]
            for parameter in ordered:
                item = row["minos"][parameter]
                writer.writerow(
                    {
                        "statistic": row["statistic"],
                        "method": row["method"],
                        "fit_mode": row["fit_mode"],
                        "input_fNL": row["input_fNL"],
                        "parameter": parameter,
                        "centre": row["map"][parameter],
                        "plus_error": item["upper_error"],
                        "minus_error": -item["lower_error"],
                        "lower": item["lower"],
                        "upper": item["upper"],
                        "minos_valid": item["valid"],
                    }
                )
    temporary.replace(OUTPUT_CSV)


def plot_recovery(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharex=True, sharey=True)
    diagonal = np.linspace(-115, 115, 200)
    for axis in axes:
        axis.plot(diagonal, diagonal, color="0.35", lw=1.0, ls="--")
        axis.grid(alpha=0.18)
        axis.set_xlabel(r"input $f_{\rm NL}$", fontsize=15)
        axis.tick_params(labelsize=13)
        axis.text(
            0.04,
            0.96,
            r"$V=42.875\,(h^{-1}{\rm Gpc})^3$" + "\n" + r"$k_{\max}=0.08\,h\,{\rm Mpc}^{-1}$",
            transform=axis.transAxes,
            va="top",
            fontsize=11,
        )
    axes[0].set_ylabel(r"recovered $f_{\rm NL}$", fontsize=15)
    axes[0].set_title(r"pre-$P_0$ only", fontsize=15)
    p_rows = sorted(
        [row for row in rows if row["statistic"] == "pre_P0_only"],
        key=lambda row: row["input_fNL"],
    )
    truth = np.asarray([row["input_fNL"] for row in p_rows])
    centre = np.asarray([row["map"]["fNL"] for row in p_rows])
    errors = np.asarray([interval_error(row, "fNL") for row in p_rows]).T
    axes[0].errorbar(
        truth,
        centre,
        yerr=errors,
        color="#4E79A7",
        marker="o",
        capsize=3,
        lw=1.4,
    )
    axes[1].set_title(r"$B_{000}$ only", fontsize=15)
    offsets = {"pre": -2.0, "standard": 0.0, "marisa": 2.0}
    for method in METHODS:
        local = sorted(
            [
                row
                for row in rows
                if row["statistic"] == "B000_only"
                and row["method"] == method
                and row["fit_mode"] == "fixed_b1_primary"
            ],
            key=lambda row: row["input_fNL"],
        )
        truth = np.asarray([row["input_fNL"] for row in local])
        centre = np.asarray([row["map"]["fNL"] for row in local])
        errors = np.asarray([interval_error(row, "fNL") for row in local]).T
        axes[1].errorbar(
            truth + offsets[method],
            centre,
            yerr=errors,
            color=METHOD_COLOURS[method],
            marker="o",
            capsize=3,
            lw=1.3,
            label=METHOD_LABELS[method],
        )
    axes[1].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_RECOVERY,
        metadata={
            "Author": "direct finite-fNL single-stat profiler",
            "Subject": "pre-P0-only and B000-only recovery",
        },
    )
    plt.close(fig)


def plot_parameters(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        OUTPUT_PARAMETERS,
        metadata={"Author": "direct finite-fNL single-stat profiler"},
    ) as pdf:
        groups = [("pre_P0_only", "pre", "free_b1_primary")]
        groups.extend(("B000_only", method, "fixed_b1_primary") for method in METHODS)
        groups.extend(("B000_only", method, "free_b1_diagnostic") for method in METHODS)
        for statistic, method, fit_mode in groups:
            local = sorted(
                [
                    row
                    for row in rows
                    if row["statistic"] == statistic
                    and row["method"] == method
                    and row["fit_mode"] == fit_mode
                ],
                key=lambda row: row["input_fNL"],
            )
            parameters = list(local[0]["parameter_names"])
            columns = 3
            rows_count = int(math.ceil(len(parameters) / columns))
            fig, axes = plt.subplots(
                rows_count, columns, figsize=(12.2, 3.7 * rows_count), squeeze=False
            )
            truth = np.asarray([row["input_fNL"] for row in local])
            for axis, parameter in zip(axes.flat, parameters):
                centres = np.asarray([row["map"][parameter] for row in local])
                errors = np.asarray([interval_error(row, parameter) for row in local]).T
                axis.errorbar(
                    truth,
                    centres,
                    yerr=errors,
                    color=METHOD_COLOURS.get(method, "#4E79A7"),
                    marker="o",
                    capsize=3,
                )
                if parameter == "fNL":
                    axis.plot(truth, truth, color="0.35", ls="--", lw=1.0)
                axis.set_xlabel(r"input $f_{\rm NL}$", fontsize=12)
                axis.set_ylabel(PARAMETER_LABELS[parameter], fontsize=12)
                axis.tick_params(labelsize=10)
                axis.grid(alpha=0.18)
            for axis in axes.flat[len(parameters) :]:
                axis.set_visible(False)
            label = "pre-$P_0$ only" if statistic == "pre_P0_only" else METHOD_LABELS[method]
            fig.text(
                0.5,
                0.99,
                f"{label}; {fit_mode.replace('_', ' ')}",
                ha="center",
                va="top",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)


def plot_bestfit(
    runtime: Any,
    theory: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> None:
    p_k = np.asarray(theory["pk_k"], dtype=np.float64)
    b_pair, _edges = b_geometry(runtime)
    b_scale = b_pair[:, 0] * b_pair[:, 1]
    with PdfPages(
        OUTPUT_BESTFIT,
        metadata={"Author": "direct finite-fNL single-stat profiler"},
    ) as pdf:
        p_cov = np.cov(p_fiducial_samples(runtime), rowvar=False, ddof=1)
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
        for truth in TRUTHS:
            key = f"pre_P0_only__pre__free_b1_primary__{truth:+.0f}"
            target = arrays[f"{key}__target"]
            prediction = arrays[f"{key}__prediction"]
            colour = TRUTH_COLOURS[truth]
            axes[0].errorbar(
                p_k,
                target,
                yerr=np.sqrt(np.diag(p_cov)),
                fmt="o",
                ms=3.0,
                capsize=2,
                color=colour,
                alpha=0.7,
            )
            axes[0].plot(p_k, prediction, color=colour, lw=1.3, label=f"{truth:+.0f}")
            axes[1].plot(
                p_k,
                (target - prediction) / np.sqrt(np.diag(p_cov)),
                color=colour,
                marker="o",
                ms=3.0,
            )
        axes[0].set_ylabel(r"$P_0(k)$", fontsize=13)
        axes[1].set_ylabel(r"$(\bar P_0-P_0^{\rm model})/\sigma_{\rm 1,real}$", fontsize=12)
        axes[1].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$", fontsize=13)
        axes[0].legend(title=r"input $f_{\rm NL}$", frameon=False, ncol=3)
        axes[0].text(
            0.03,
            0.95,
            r"pre-$P_0$ only" + "\n" + r"$V=42.875$, $k_{\max}=0.08$",
            transform=axes[0].transAxes,
            va="top",
        )
        for axis in axes:
            axis.grid(alpha=0.18)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for method in METHODS:
            covariance = np.cov(b_fiducial_samples(runtime, method), rowvar=False, ddof=1)
            sigma = np.sqrt(np.diag(covariance)) * b_scale
            fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
            for truth in TRUTHS:
                key = f"B000_only__{method}__fixed_b1_primary__{truth:+.0f}"
                target = arrays[f"{key}__target"] * b_scale
                prediction = arrays[f"{key}__prediction"] * b_scale
                colour = TRUTH_COLOURS[truth]
                x = np.arange(B_COUNT)
                axes[0].errorbar(
                    x,
                    target,
                    yerr=sigma,
                    fmt="o",
                    ms=3.0,
                    capsize=2,
                    color=colour,
                    alpha=0.65,
                )
                axes[0].plot(x, prediction, color=colour, lw=1.3, label=f"{truth:+.0f}")
                axes[1].plot(
                    x,
                    (target - prediction) / sigma,
                    color=colour,
                    marker="o",
                    ms=3.0,
                )
            labels = [f"{first:.3f},{second:.3f}" for first, second in b_pair]
            axes[1].set_xticks(np.arange(B_COUNT), labels, rotation=35, ha="right")
            axes[0].set_ylabel(r"$k_1k_2 B_{000}$", fontsize=13)
            axes[1].set_ylabel(r"$(\bar B-B^{\rm model})/\sigma_{\rm 1,real}$", fontsize=12)
            axes[1].set_xlabel(r"$(k_1,k_2)\ [h\,{\rm Mpc}^{-1}]$", fontsize=12)
            axes[0].legend(title=r"input $f_{\rm NL}$", frameon=False, ncol=3)
            axes[0].text(
                0.03,
                0.95,
                METHOD_LABELS[method]
                + "\n"
                + r"$B_{000}$ only; $V=42.875$, $k_{\max}=0.08$",
                transform=axes[0].transAxes,
                va="top",
            )
            for axis in axes:
                axis.grid(alpha=0.18)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def row_key(row: dict[str, Any]) -> str:
    return (
        f"{row['statistic']}__{row['method']}__{row['fit_mode']}__"
        f"{row['input_fNL']:+.0f}"
    )


def run(overwrite: bool) -> dict[str, Any]:
    for path in (P_THEORY, P_THEORY_MANIFEST, MARISA_THEORY, MARISA_THEORY_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_JSON.is_file() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {OUTPUT_JSON}; pass --overwrite")
    runtime = load_runtime()
    p_model, theory = load_p_model()
    closure = direct_model_closure(runtime)
    if not closure["pass"]:
        raise ValueError("direct finite B model closure failed")
    b_geometry(runtime)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    p_fid = p_fiducial_samples(runtime)
    p_covariance, p_cholesky = covariance_contract(p_fid, P_COUNT, P_HARTLAP)
    for truth in TRUTHS:
        target = np.mean(p_target_samples(runtime, truth), axis=0)
        row, local = profile_p_case(
            p_model, theory, target, p_covariance, p_cholesky, truth
        )
        rows.append(row)
        key = row_key(row)
        for name, value in local.items():
            arrays[f"{key}__{name}"] = np.asarray(value)
        print(
            f"{key}: fNL={row['map']['fNL']:.6f} "
            f"+{row['minos']['fNL']['upper_error']:.5f}/"
            f"-{(-row['minos']['fNL']['lower_error']):.5f}",
            flush=True,
        )

    for method in METHODS:
        model = DirectB000Model(runtime, method)
        fiducial = b_fiducial_samples(runtime, method)
        covariance, cholesky = covariance_contract(fiducial, B_COUNT, B_HARTLAP)
        for free_b1 in (False, True):
            for truth in TRUTHS:
                target = np.mean(b_target_samples(runtime, method, truth), axis=0)
                row, local = profile_b_case(
                    model, target, covariance, cholesky, truth, free_b1
                )
                rows.append(row)
                key = row_key(row)
                for name, value in local.items():
                    arrays[f"{key}__{name}"] = np.asarray(value)
                print(
                    f"{key}: fNL={row['map']['fNL']:.6f} "
                    f"+{row['minos']['fNL']['upper_error']:.5f}/"
                    f"-{(-row['minos']['fNL']['lower_error']):.5f}",
                    flush=True,
                )

    summary = recovery_summary(rows)
    odd_even = odd_even_diagnostics(runtime, p_model, rows, arrays)
    gates = {
        "direct_model_closure": bool(closure["pass"]),
        "all_optimizer_valid": bool(all(row["optimizer"]["valid"] for row in rows)),
        "all_fNL_minos_valid": bool(all(row["minos"]["fNL"]["valid"] for row in rows)),
        "no_frozen_response": bool(all(not row["model"]["frozen_response"] for row in rows)),
        "no_joint_PplusB_likelihood": bool(
            all(row["statistic"] in {"pre_P0_only", "B000_only"} for row in rows)
        ),
        "row_count_35": len(rows) == 35,
    }
    if not all(gates.values()):
        raise RuntimeError(f"direct finite profile gates failed: {gates}")

    atomic_npz(OUTPUT_NPZ, arrays)
    write_csv(rows)
    plot_recovery(rows)
    plot_parameters(rows)
    plot_bestfit(runtime, theory, rows, arrays)
    payload = {
        "schema": "quijote-halo-z0p5-direct-finite-fnl-single-stat-profiles-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "scope": {
            "sample": "ordinary Quijote FoF halo Mmin=1e13 Msun/h",
            "redshift": 0.5,
            "kmax_h_mpc": KMAX,
            "statistics": ["pre_P0_only", "B000_only"],
            "P0_plus_B000_joint_likelihood": False,
            "B000_methods": list(METHODS),
            "target_volume_gpc3_h3": VOLUME,
            "target_counts": {"fiducial": 500, "nonzero_each": 250},
        },
        "likelihood": {
            "covariance": "Fid500 single-realization sample covariance divided by 42.875",
            "covariance_is_error_on_mean": False,
            "P0_dimension": P_COUNT,
            "P0_hartlap": P_HARTLAP,
            "B000_dimension": B_COUNT,
            "B000_hartlap": B_HARTLAP,
            "fNL_prior": "hard flat [input-200,input+200] for every injection",
            "bestfit_errorbars": "unscaled single-realization scatter",
        },
        "models": {
            "P0": {
                "callable": "P0_pre(fNL,b1,Pshot_residual)",
                "linear_density_bias_only": True,
                "finite_png_order": 2,
                "b1_free_without_gaussian_prior": True,
            },
            "B000": {
                "callable": "B000_method(fNL,b1,b2,gamma2,Ashot,Bshot)",
                "fully_tree_dm": True,
                "finite_png_order": 2,
                "fixed_b1_primary": FIXED_B1,
                "free_b1_diagnostic_without_gaussian_prior": True,
                "Marisa_finite_adaptive_tree": True,
            },
            "frozen_response": False,
            "empirical_response_or_gain": False,
        },
        "direct_model_closure": closure,
        "gates": gates,
        "recovery_summary": summary,
        "odd_even_diagnostics": odd_even,
        "rows": rows,
        "inputs": {
            "runner": artifact(Path(__file__).resolve()),
            "model_module": artifact(
                CODES / "quijote_halo_z0p5_direct_finite_fnl_models.py"
            ),
            "P_theory": artifact(P_THEORY),
            "P_theory_manifest": artifact(P_THEORY_MANIFEST),
            "Marisa_finite_theory": artifact(MARISA_THEORY),
            "Marisa_finite_theory_manifest": artifact(MARISA_THEORY_MANIFEST),
        },
        "outputs": {
            "arrays": str(OUTPUT_NPZ),
            "csv": str(OUTPUT_CSV),
            "recovery_pdf": str(OUTPUT_RECOVERY),
            "parameters_pdf": str(OUTPUT_PARAMETERS),
            "bestfit_pdf": str(OUTPUT_BESTFIT),
        },
    }
    atomic_json(OUTPUT_JSON, payload)
    payload["outputs"] = {
        name: artifact(Path(path)) for name, path in payload["outputs"].items()
    }
    atomic_json(OUTPUT_JSON, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = run(args.overwrite)
    print(json.dumps(payload["recovery_summary"], indent=2, default=jsonable), flush=True)


if __name__ == "__main__":
    main()
