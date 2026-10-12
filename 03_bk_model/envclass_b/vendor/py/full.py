#!/usr/bin/env python3
"""Profile all six parameters in the p=1.12 local-PNG recovery fits.

This is a strict extension of
``run_quijote_halo_z0p5_local_png_p112_recovery_v42p875_priorpm200.py``.
It uses the same ten targets, input-centred flat fNL prior +/-200, target
volume V=42.875 (Gpc/h)^3, kmax=0.10 h/Mpc, and linear-in-fNL theory.  The
full six-dimensional objective is evaluated explicitly and MINOS profiles
each parameter at Delta chi2=1.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MARISA_B_MAX_THREADS",
):
    os.environ[_name] = "1"
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = ROOT / "codes"
TAG = "quijote_halo_z0p5_local_png_p112_recovery_kmax0p10_v42p875_priorpm200"
ANALYSIS = ROOT / "analysis" / TAG
FIGURES = ROOT / "figures/diagnostics" / TAG
RECOVERY_JSON = ANALYSIS / "RECOVERY_PROFILES.json"
RECOVERY_RUNNER = CODES / "run_quijote_halo_z0p5_local_png_p112_recovery_profiles.py"
CONFIG_RUNNER = CODES / "run_quijote_halo_z0p5_local_png_p112_recovery_v42p875_priorpm200.py"

OUTPUT_JSON = ANALYSIS / "FULL_PARAMETER_PROFILES.json"
OUTPUT_CSV = ANALYSIS / "FULL_PARAMETER_PROFILES.csv"
OUTPUT_ARRAYS = ANALYSIS / "FULL_PARAMETER_PROFILE_ARRAYS.npz"
OUTPUT_REPORT = ANALYSIS / "FULL_PARAMETER_REPORT.md"
OUTPUT_PRE_PDF = FIGURES / "pre_full_six_parameter_profiles.pdf"
OUTPUT_STANDARD_PDF = FIGURES / "standard_full_six_parameter_profiles.pdf"
OUTPUT_CORRELATION_PDF = FIGURES / "positive_fNL_parameter_correlations.pdf"

PARAMETERS = (
    "fNL",
    "b1",
    "b2",
    "gamma2",
    "Ashot_residual",
    "Bshot_residual",
)
NUISANCE = PARAMETERS[1:]
SHORT_LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "b2": r"$b_2$",
    "gamma2": r"$\gamma_2$",
    "Ashot_residual": r"$A_{\rm shot}$",
    "Bshot_residual": r"$B_{\rm shot}$",
}
INPUT_FNL = (-100.0, -50.0, 0.0, 50.0, 100.0)
METHODS = ("pre", "standard")
VOLUME = 42.875
PRIOR_HALF_WIDTH = 200.0
B1_BOUNDS = (0.1, 6.0)
MINOS_CL = 0.682689492137

PDF_METADATA = {
    "Author": "ordinary-halo local-PNG full-parameter profiler",
    "Creator": Path(__file__).name,
    "Subject": (
        "six-parameter MINOS profiles; p=1.12; kmax=0.10 h/Mpc; "
        "V=42.875 (Gpc/h)^3; input-centred flat fNL prior +/-200"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def configure_recovery() -> Any:
    recovery = import_path("_p112_full_parameter_recovery", RECOVERY_RUNNER)
    recovery.COVARIANCE_VOLUME = VOLUME
    recovery.FNL_PRIOR_HALF_WIDTH = PRIOR_HALF_WIDTH
    return recovery


def build_runtime(recovery: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    baseline_matrix = recovery.load_npz(recovery.BASE_MATRIX)
    fnl50_matrix = recovery.load_npz(recovery.FNL50_MATRIX)
    fnl100_matrix = recovery.load_npz(recovery.FNL100_MATRIX)
    projected = recovery.import_path("_p112_full_projected", recovery.PROJECTED_RUNNER)
    profiler = recovery.import_path("_p112_full_profiler", recovery.PROFILE_RUNNER)
    refiner = recovery.import_path("_p112_full_refiner", recovery.REFINED_ASSETS)
    baseline, physical, base, audit, assets, standard_method, physical_cache = (
        projected.load_modules_and_assets()
    )
    audit.VOLUME = VOLUME
    assets, refinement = refiner.refined_assets(assets)
    matrix = projected.load_matrix()
    tree = projected.load_pre_tree(matrix)
    dm_tree, dm_total, _dm_audit = projected.load_z05_dm_response(matrix)
    bias = json.loads(projected.BIAS_JSON.read_text(encoding="utf-8"))
    growth_ratio = float(base.legacy.configurations()["z0p5"].growth_power_ratio_to_z1)
    nbar = float(bias["nbar_mean"])
    contexts = profiler.build_contexts(
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
        recovery.KMAX,
        refinement,
    )
    matrices = {
        "baseline": baseline_matrix,
        "fnl50": fnl50_matrix,
        "fnl100": fnl100_matrix,
    }
    return base, assets, contexts, recovery, matrices


def objective_factory(context: Any) -> tuple[Callable[..., float], Callable[[np.ndarray], tuple[float, float, np.ndarray]]]:
    def evaluate(values: np.ndarray) -> tuple[float, float, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        fnl = float(values[0])
        nuisance = values[1:]
        if not context.fnl_bounds[0] <= fnl <= context.fnl_bounds[1]:
            return math.inf, math.inf, np.full(context.target.size, np.nan)
        if not B1_BOUNDS[0] < nuisance[0] < B1_BOUNDS[1]:
            return math.inf, math.inf, np.full(context.target.size, np.nan)
        prediction = np.asarray(context.evaluate_model(fnl, nuisance)[0], dtype=np.float64)
        data_white = solve_triangular(
            context.chol_fit,
            prediction - np.asarray(context.target),
            lower=True,
        )
        prior_white = np.asarray(context.prior.whitener) @ (
            nuisance - np.asarray(context.prior.mean)
        )
        return (
            float(data_white @ data_white),
            float(prior_white @ prior_white),
            prediction,
        )

    def objective(
        fNL: float,
        b1: float,
        b2: float,
        gamma2: float,
        Ashot_residual: float,
        Bshot_residual: float,
    ) -> float:
        data_q, prior_q, _prediction = evaluate(
            np.asarray((fNL, b1, b2, gamma2, Ashot_residual, Bshot_residual))
        )
        return data_q + prior_q

    return objective, evaluate


def minuit_instance(
    objective: Callable[..., float],
    start: np.ndarray,
    fnl_bounds: tuple[float, float],
) -> Minuit:
    m = Minuit(
        objective,
        **{name: float(value) for name, value in zip(PARAMETERS, start)},
    )
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["fNL"] = fnl_bounds
    m.limits["b1"] = B1_BOUNDS
    for name, error in {
        "fNL": 10.0,
        "b1": 0.02,
        "b2": 0.10,
        "gamma2": 0.08,
        "Ashot_residual": 0.10,
        "Bshot_residual": 0.08,
    }.items():
        m.errors[name] = error
    m.strategy = 1
    m.tol = 1.0e-4
    return m


def fit_full(
    context: Any,
    collapsed_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    objective, evaluate = objective_factory(context)
    collapsed_start = np.asarray(
        [collapsed_row["recovery"]["fNL_hat"]]
        + [collapsed_row["profile_nuisance"]["values"][name] for name in NUISANCE],
        dtype=np.float64,
    )
    truth = float(collapsed_row["input_fNL"])
    prior_start = np.asarray(
        [truth, *np.asarray(context.prior.mean, dtype=np.float64)], dtype=np.float64
    )
    zero_start = collapsed_start.copy()
    zero_start[0] = float(np.clip(0.0, *context.fnl_bounds))
    candidates: list[Minuit] = []
    for start in (collapsed_start, prior_start, zero_start):
        m = minuit_instance(objective, start, tuple(context.fnl_bounds))
        m.migrad(ncall=50000, iterate=5, use_simplex=True)
        m.hesse()
        candidates.append(m)
    valid = [m for m in candidates if m.valid and math.isfinite(float(m.fval))]
    best = min(valid or candidates, key=lambda item: float(item.fval))
    if not best.valid or not math.isfinite(float(best.fval)):
        raise RuntimeError(f"full six-parameter Minuit failed: {best.fmin}")

    minos: dict[str, Any] = {}
    for name in PARAMETERS:
        try:
            best.minos(name, cl=MINOS_CL, ncall=120000)
            error = best.merrors[name]
            centre = float(best.values[name])
            minos[name] = {
                "valid": bool(error.is_valid),
                "lower_valid": bool(error.lower_valid),
                "upper_valid": bool(error.upper_valid),
                "at_lower_limit": bool(error.at_lower_limit),
                "at_upper_limit": bool(error.at_upper_limit),
                "lower_error": float(error.lower),
                "upper_error": float(error.upper),
                "lower": centre + float(error.lower),
                "upper": centre + float(error.upper),
                "nfcn": int(error.nfcn),
                "error": None,
            }
        except Exception as error:
            symmetric = float(best.errors[name])
            centre = float(best.values[name])
            minos[name] = {
                "valid": False,
                "lower_valid": False,
                "upper_valid": False,
                "at_lower_limit": False,
                "at_upper_limit": False,
                "lower_error": -symmetric,
                "upper_error": symmetric,
                "lower": centre - symmetric,
                "upper": centre + symmetric,
                "nfcn": 0,
                "error": f"{type(error).__name__}: {error}",
            }

    values = np.asarray([best.values[name] for name in PARAMETERS], dtype=np.float64)
    data_q, prior_q, prediction = evaluate(values)
    covariance = np.asarray(best.covariance, dtype=np.float64)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(sigma, sigma)
    collapsed_objective = float(collapsed_row["profile"]["objective"])
    collapsed_full_data, collapsed_full_prior, _ = evaluate(collapsed_start)
    centre_differences = {
        name: float(best.values[name] - collapsed_start[index])
        for index, name in enumerate(PARAMETERS)
    }
    closure = {
        "full_map_minus_collapsed_map": centre_differences,
        "maximum_absolute_parameter_difference": max(
            abs(value) for value in centre_differences.values()
        ),
        "full_objective_minus_collapsed_objective": float(best.fval - collapsed_objective),
        "collapsed_explicit_objective_minus_collapsed_objective": float(
            collapsed_full_data + collapsed_full_prior - collapsed_objective
        ),
    }
    row = {
        "method": collapsed_row["method"],
        "input_fNL": truth,
        "input_tag": collapsed_row["input_tag"],
        "flat_fNL_prior": tuple(context.fnl_bounds),
        "map": {name: float(best.values[name]) for name in PARAMETERS},
        "hesse_errors": {name: float(best.errors[name]) for name in PARAMETERS},
        "minos": minos,
        "data_chi2": data_q,
        "prior_chi2": prior_q,
        "objective": float(best.fval),
        "dof": int(context.target.size - len(PARAMETERS)),
        "gof_p_value_data_only": float(
            chi2_distribution.sf(data_q, context.target.size - len(PARAMETERS))
        ),
        "optimizer": {
            "valid": bool(best.valid),
            "accurate": bool(best.accurate),
            "edm": float(best.fmin.edm),
            "nfcn": int(best.nfcn),
            "has_accurate_covar": bool(best.fmin.has_accurate_covar),
            "has_posdef_covar": bool(best.fmin.has_posdef_covar),
            "has_parameters_at_limit": bool(best.fmin.has_parameters_at_limit),
            "multistart_objectives": [float(item.fval) for item in candidates],
            "multistart_valid": [bool(item.valid) for item in candidates],
        },
        "collapsed_profile_closure": closure,
        "all_minos_valid": bool(all(item["valid"] for item in minos.values())),
    }
    arrays = {
        "hesse_covariance": covariance,
        "hesse_correlation": correlation,
        "target": np.asarray(context.target),
        "model": prediction,
        "residual": np.asarray(context.target) - prediction,
    }
    return row, arrays


def prior_record(context: Any) -> dict[str, Any]:
    b1_location = context.names.index("b1")
    if not np.all(np.asarray(context.prior.whitener)[b1_location] == 0.0):
        raise ValueError("b1 unexpectedly has Gaussian prior precision")
    gaussian = {}
    for name in context.names:
        location = context.names.index(name)
        if name == "b1":
            continue
        gaussian[name] = {
            "mean": float(context.prior.mean[location]),
            "sigma": float(context.prior.sigma[location]),
        }
    return {
        "fNL": "input-centred flat hard prior +/-200",
        "b1": {"kind": "flat hard", "bounds": B1_BOUNDS, "P_k_prior_used": False},
        "gaussian_nuisance": gaussian,
    }


def write_csv(recovery: Any, rows: list[dict[str, Any]]) -> None:
    fields = (
        "method",
        "input_fNL",
        "parameter",
        "centre",
        "plus_error",
        "minus_error",
        "lower",
        "upper",
        "minos_valid",
        "at_lower_limit",
        "at_upper_limit",
        "hesse_error",
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for parameter in PARAMETERS:
                interval = row["minos"][parameter]
                writer.writerow(
                    {
                        "method": row["method"],
                        "input_fNL": row["input_fNL"],
                        "parameter": parameter,
                        "centre": row["map"][parameter],
                        "plus_error": interval["upper_error"],
                        "minus_error": -interval["lower_error"],
                        "lower": interval["lower"],
                        "upper": interval["upper"],
                        "minos_valid": interval["valid"],
                        "at_lower_limit": interval["at_lower_limit"],
                        "at_upper_limit": interval["at_upper_limit"],
                        "hesse_error": row["hesse_errors"][parameter],
                    }
                )
    temporary.replace(OUTPUT_CSV)


def plot_parameter_summary(rows: list[dict[str, Any]], method: str, output: Path) -> None:
    local = sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: row["input_fNL"],
    )
    colours = {
        -100.0: "#3B4CC0",
        -50.0: "#7B9FF9",
        0.0: "#595959",
        50.0: "#F4987A",
        100.0: "#B40426",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1), constrained_layout=True)
    x = np.asarray([row["input_fNL"] for row in local])
    for axis, parameter in zip(axes.flat, PARAMETERS):
        centre = np.asarray([row["map"][parameter] for row in local])
        minus = np.asarray([-row["minos"][parameter]["lower_error"] for row in local])
        plus = np.asarray([row["minos"][parameter]["upper_error"] for row in local])
        for location, truth in enumerate(x):
            axis.errorbar(
                truth,
                centre[location],
                yerr=np.asarray([[minus[location]], [plus[location]]]),
                fmt="o",
                color=colours[float(truth)],
                ms=5.5,
                capsize=3,
                lw=1.1,
            )
        axis.plot(x, centre, color="0.45", lw=0.9, alpha=0.65)
        if parameter == "fNL":
            axis.plot(x, x, "k--", lw=0.9, label="input")
        elif parameter != "b1":
            axis.axhline(0.0, color="0.4", lw=0.8, ls="--")
        axis.set_title(SHORT_LABELS[parameter])
        axis.set_xlabel(r"input $f_{\rm NL}$")
        axis.set_ylabel(r"profile centre and $\Delta\chi^2=1$ error")
        axis.grid(alpha=0.20)
    handles = [
        Line2D([], [], color=colours[truth], marker="o", ls="none", label=rf"{truth:+.0f}")
        for truth in INPUT_FNL
    ]
    axes[0, 0].legend(handles=handles, title=r"input $f_{\rm NL}$", frameon=False, ncol=2, fontsize=8)
    label = "Pre reconstruction" if method == "pre" else "Standard post reconstruction"
    fig.suptitle(
        rf"{label}: full six-parameter profiles, $k_{{\max}}=0.10\,h\,{{\rm Mpc}}^{{-1}}$, "
        rf"$p=1.12$, $V=42.875\,(h^{{-1}}{{\rm Gpc}})^3$",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.005,
        (
            r"Priors: input-centred flat $f_{\rm NL}\pm200$; flat $b_1\in[0.1,6]$; "
            r"Gaussian $(b_2,\gamma_2,A_{\rm shot},B_{\rm shot})$ widths $(3,2,2,2)$."
        ),
        ha="center",
        fontsize=8,
    )
    fig.savefig(output, metadata={**PDF_METADATA, "Title": f"{method} six-parameter profiles"})
    plt.close(fig)


def plot_correlations(rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    selected = [
        row
        for row in rows
        if row["input_fNL"] in (50.0, 100.0)
    ]
    selected.sort(key=lambda row: (row["method"], row["input_fNL"]))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.8), constrained_layout=True)
    image = None
    for axis, row in zip(axes.flat, selected):
        key = f"{row['method']}__{row['input_tag']}"
        correlation = arrays[f"{key}__hesse_correlation"]
        image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        for i in range(len(PARAMETERS)):
            for j in range(len(PARAMETERS)):
                value = float(correlation[i, j])
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=("white" if abs(value) > 0.55 else "black"),
                )
        labels = [SHORT_LABELS[name] for name in PARAMETERS]
        axis.set_xticks(range(len(PARAMETERS)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(PARAMETERS)), labels)
        axis.set_title(f"{row['method']}, input fNL={row['input_fNL']:+.0f}")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="Hesse correlation")
    fig.suptitle("Positive-input parameter correlations")
    fig.savefig(
        OUTPUT_CORRELATION_PDF,
        metadata={**PDF_METADATA, "Title": "positive-input parameter correlations"},
    )
    plt.close(fig)


def compact_value(row: dict[str, Any], parameter: str) -> str:
    centre = row["map"][parameter]
    interval = row["minos"][parameter]
    return (
        f"{centre:.4g} (+{interval['upper_error']:.3g}/"
        f"-{(-interval['lower_error']):.3g})"
    )


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Full six-parameter local-PNG profiles",
        "",
        f"Created UTC: `{payload['created_utc']}`.",
        "",
        "Priors: input-centred flat fNL +/-200; flat b1 in [0.1,6] with no P(k) prior; Gaussian b2/gamma2/Ashot/Bshot widths 3/2/2/2.",
        "Covariance: Fid500 single-box covariance divided by 42.875; never divided by the number of target realizations.",
        "",
    ]
    for method in METHODS:
        lines.extend(
            [
                f"## {method}",
                "",
                "| input fNL | fNL | b1 | b2 | gamma2 | Ashot | Bshot |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["rows"]:
            if row["method"] != method:
                continue
            lines.append(
                "| {truth:+.0f} | {fNL} | {b1} | {b2} | {gamma2} | {Ashot} | {Bshot} |".format(
                    truth=row["input_fNL"],
                    **{name: compact_value(row, name) for name in PARAMETERS},
                    Ashot=compact_value(row, "Ashot_residual"),
                    Bshot=compact_value(row, "Bshot_residual"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            f"- all MINOS intervals valid: `{payload['summary']['all_minos_valid']}`;",
            f"- all six-parameter optimizers valid: `{payload['summary']['all_optimizers_valid']}`;",
            f"- maximum explicit-vs-collapsed MAP parameter difference: `{payload['summary']['maximum_collapsed_map_parameter_difference']:.6g}`;",
            f"- maximum explicit-vs-collapsed objective difference: `{payload['summary']['maximum_collapsed_objective_difference']:.6g}`.",
            "",
        ]
    )
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_REPORT.with_name(f".{OUTPUT_REPORT.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(OUTPUT_REPORT)


def main() -> None:
    for path in (RECOVERY_JSON, RECOVERY_RUNNER, CONFIG_RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)
    collapsed = json.loads(RECOVERY_JSON.read_text(encoding="utf-8"))
    if collapsed.get("status") != "complete":
        raise ValueError("collapsed recovery result is not complete")
    collapsed_lookup = {
        (row["method"], float(row["input_fNL"])): row for row in collapsed["rows"]
    }

    base, assets, base_contexts, recovery, matrices = build_runtime(configure_recovery())
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    contexts_for_prior: list[Any] = []
    for method in METHODS:
        fiducial = np.asarray(
            matrices["baseline"][
                "fiducial_pre_B000" if method == "pre" else "fiducial_post_B000"
            ],
            dtype=np.float64,
        )
        for truth in INPUT_FNL:
            samples = recovery.target_samples(
                method,
                truth,
                matrices["baseline"],
                matrices["fnl50"],
                matrices["fnl100"],
            )
            context, _contract = recovery.recovery_context(
                base_contexts[method], method, truth, samples, fiducial
            )
            contexts_for_prior.append(context)
            row, local_arrays = fit_full(
                context,
                collapsed_lookup[(method, truth)],
            )
            rows.append(row)
            key = f"{method}__{recovery.input_tag(truth)}"
            for name, value in local_arrays.items():
                arrays[f"{key}__{name}"] = np.asarray(value)
            values = ", ".join(
                f"{name}={row['map'][name]:.5g}"
                for name in PARAMETERS
            )
            print(
                f"{key}: {values}; all_minos={row['all_minos_valid']}; "
                f"closure={row['collapsed_profile_closure']['maximum_absolute_parameter_difference']:.3g}",
                flush=True,
            )

    prior = prior_record(contexts_for_prior[0])
    summary = {
        "all_minos_valid": bool(all(row["all_minos_valid"] for row in rows)),
        "all_optimizers_valid": bool(all(row["optimizer"]["valid"] for row in rows)),
        "maximum_collapsed_map_parameter_difference": max(
            float(row["collapsed_profile_closure"]["maximum_absolute_parameter_difference"])
            for row in rows
        ),
        "maximum_collapsed_objective_difference": max(
            abs(float(row["collapsed_profile_closure"]["full_objective_minus_collapsed_objective"]))
            for row in rows
        ),
    }
    payload = {
        "schema": "quijote-z0p5-local-png-p112-full-six-parameter-profiles-v1",
        "created_utc": utc_now(),
        "status": (
            "complete"
            if summary["all_minos_valid"] and summary["all_optimizers_valid"]
            else "profile_failure"
        ),
        "scope": collapsed["scope"],
        "model": collapsed["model"],
        "priors": prior,
        "summary": summary,
        "rows": rows,
        "inputs": {
            "collapsed_recovery": recovery.artifact(RECOVERY_JSON),
            "runner": recovery.artifact(Path(__file__).resolve()),
            "recovery_runner": recovery.artifact(RECOVERY_RUNNER),
            "configuration_runner": recovery.artifact(CONFIG_RUNNER),
        },
    }

    write_csv(recovery, rows)
    recovery.atomic_npz(OUTPUT_ARRAYS, arrays)
    plot_parameter_summary(rows, "pre", OUTPUT_PRE_PDF)
    plot_parameter_summary(rows, "standard", OUTPUT_STANDARD_PDF)
    plot_correlations(rows, arrays)
    payload["outputs"] = {
        "csv": recovery.artifact(OUTPUT_CSV),
        "arrays": recovery.artifact(OUTPUT_ARRAYS),
        "pre_pdf": recovery.artifact(OUTPUT_PRE_PDF),
        "standard_pdf": recovery.artifact(OUTPUT_STANDARD_PDF),
        "correlation_pdf": recovery.artifact(OUTPUT_CORRELATION_PDF),
    }
    write_report(payload)
    payload["outputs"]["report"] = recovery.artifact(OUTPUT_REPORT)
    recovery.atomic_json(OUTPUT_JSON, payload)
    print(json.dumps(recovery.jsonable(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
