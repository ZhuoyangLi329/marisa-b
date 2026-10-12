#!/usr/bin/env python3
"""Find where fixed-b1 direct finite-tree B000 models fail as kmax grows."""

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

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "MARISA_B_MAX_THREADS",
):
    os.environ[_name] = "1"
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

import quijote_halo_z0p5_direct_finite_fnl_models as low
import quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models as scan
import run_quijote_halo_z0p5_direct_finite_fnl_single_stat_profiles as helpers


ROOT = low.ROOT
CODES = low.CODES
TAG = "quijote_halo_z0p5_direct_finite_tree_kmax_validity_scan_v42p875"
ANALYSIS = ROOT / "analysis" / TAG
FIGURES = ROOT / "figures/diagnostics" / TAG
OUTPUT_JSON = ANALYSIS / "DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.json"
OUTPUT_CSV = ANALYSIS / "DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.csv"
OUTPUT_NPZ = ANALYSIS / "DIRECT_FINITE_TREE_KMAX_VALIDITY_ARRAYS.npz"
OUTPUT_RECOVERY = FIGURES / "direct_finite_tree_kmax_recovery.pdf"
OUTPUT_VALIDITY = FIGURES / "direct_finite_tree_kmax_validity.pdf"
OUTPUT_BESTFIT = FIGURES / "direct_finite_tree_kmax_fiducial_bestfit.pdf"

METHODS = ("pre", "standard", "marisa")
TRUTHS = low.TRUTHS
CUTS = scan.CUTS
VOLUME = 42.875
FIXED_B1 = 1.93
FNL_HALF_WIDTH = 200.0
NAMES = ("fNL", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
NUISANCE_NAMES = NAMES[1:]
METHOD_LABELS = {
    "pre": "pre reconstruction",
    "standard": "standard reconstruction",
    "marisa": "Marisa reconstruction",
}
METHOD_COLOURS = {"pre": "#4E79A7", "standard": "#E15759", "marisa": "#7A4EAB"}
TRUTH_COLOURS = {
    -100.0: "#3B4CC0", -50.0: "#7B9FF9", 0.0: "#595959",
    50.0: "#F4987A", 100.0: "#B40426",
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


def target_samples(
    runtime: scan.ScanRuntime, method: str, truth: float, indices: np.ndarray
) -> np.ndarray:
    matrix = runtime.low_runtime.b_matrices["standard" if method == "pre" else method]
    stage = "pre" if method == "pre" else "post"
    key = low.TRUTH_KEYS[float(truth)]
    count = 500 if truth == 0.0 else 250
    values = np.asarray(matrix[f"{key}_{stage}_B000"], dtype=np.float64)
    if values.shape[0] < count or values.shape[1] != 120:
        raise ValueError(f"invalid target matrix for {method}/{truth}")
    return values[:count, np.asarray(indices, dtype=np.int64)]


def fiducial_samples(
    runtime: scan.ScanRuntime, method: str, indices: np.ndarray
) -> np.ndarray:
    matrix = runtime.low_runtime.b_matrices["standard" if method == "pre" else method]
    stage = "pre" if method == "pre" else "post"
    values = np.asarray(matrix[f"fiducial_{stage}_B000"], dtype=np.float64)
    if values.shape != (500, 120):
        raise ValueError(f"invalid Fid500 matrix for {method}")
    return values[:, np.asarray(indices, dtype=np.int64)]


def prior_chi2(context: Any, mapping: dict[str, float]) -> float:
    values = np.asarray([mapping[name] for name in context.names], dtype=np.float64)
    white = np.asarray(context.prior.whitener, dtype=np.float64) @ (
        values - np.asarray(context.prior.mean, dtype=np.float64)
    )
    return float(white @ white)


def mapping_from_values(values: tuple[float, ...]) -> dict[str, float]:
    supplied = dict(zip(NAMES, values))
    return {
        "fNL": float(supplied["fNL"]), "b1": FIXED_B1,
        "b2": float(supplied["b2"]), "gamma2": float(supplied["gamma2"]),
        "Ashot_residual": float(supplied["Ashot_residual"]),
        "Bshot_residual": float(supplied["Bshot_residual"]),
    }


def fit_fixed_fnl(
    evaluate: Callable[[dict[str, float]], tuple[float, float, np.ndarray]],
    start: dict[str, float],
    fixed_fnl: float,
) -> tuple[Minuit, dict[str, float], float, float, np.ndarray]:
    def objective(*values: float) -> float:
        mapping = {
            "fNL": float(fixed_fnl), "b1": FIXED_B1,
            **{name: float(value) for name, value in zip(NUISANCE_NAMES, values)},
        }
        data_q, prior_q, _prediction = evaluate(mapping)
        return data_q + prior_q

    minimizer = Minuit(
        objective, *[start[name] for name in NUISANCE_NAMES], name=NUISANCE_NAMES
    )
    minimizer.errordef = Minuit.LEAST_SQUARES
    for name, step in {
        "b2": 0.08, "gamma2": 0.06,
        "Ashot_residual": 0.1, "Bshot_residual": 0.08,
    }.items():
        minimizer.errors[name] = step
    minimizer.strategy = 1
    minimizer.tol = 1.0e-4
    minimizer.migrad(ncall=100000, iterate=8, use_simplex=True)
    minimizer.hesse()
    mapping = {
        "fNL": float(fixed_fnl), "b1": FIXED_B1,
        **{name: float(minimizer.values[name]) for name in NUISANCE_NAMES},
    }
    data_q, prior_q, prediction = evaluate(mapping)
    return minimizer, mapping, data_q, prior_q, prediction


def profile_case(
    model: scan.DirectFiniteScanB000Model,
    indices: np.ndarray,
    target: np.ndarray,
    covariance_one: np.ndarray,
    truth: float,
    cut: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ndata = int(indices.size)
    hartlap = (500.0 - ndata - 2.0) / 499.0
    covariance_fit = covariance_one / VOLUME
    cholesky = np.linalg.cholesky(covariance_fit) / math.sqrt(hartlap)
    context = model.base_context
    prior_mean = {
        name: float(context.prior.mean[context.names.index(name)])
        for name in context.names
    }

    def evaluate(mapping: dict[str, float]) -> tuple[float, float, np.ndarray]:
        prediction = model(mapping["fNL"], mapping, indices)
        white = solve_triangular(cholesky, prediction - target, lower=True)
        return float(white @ white), prior_chi2(context, mapping), prediction

    def objective(*values: float) -> float:
        mapping = mapping_from_values(values)
        data_q, prior_q, _prediction = evaluate(mapping)
        return data_q + prior_q

    starts = []
    for f_nl in (truth, 0.0, truth - 60.0, truth + 60.0):
        starts.append(
            {
                "fNL": float(np.clip(f_nl, truth - 199.0, truth + 199.0)),
                "b2": prior_mean["b2"], "gamma2": prior_mean["gamma2"],
                "Ashot_residual": prior_mean["Ashot_residual"],
                "Bshot_residual": prior_mean["Bshot_residual"],
            }
        )
    best, candidates = helpers.run_multistart(objective, NAMES, starts, truth)
    mapping = mapping_from_values(tuple(float(best.values[name]) for name in NAMES))
    data_q, prior_q, prediction = evaluate(mapping)
    minos = helpers.run_minos(best, ("fNL",))
    fixed, fixed_map, fixed_data_q, fixed_prior_q, fixed_prediction = fit_fixed_fnl(
        evaluate, mapping, truth
    )
    truth_delta = float(fixed.fval - best.fval)
    interval = minos["fNL"]
    at_limit = bool(interval["at_lower_limit"] or interval["at_upper_limit"])
    row = {
        "method": model.method,
        "kmax_h_mpc": float(cut),
        "retained_indices": np.asarray(indices, dtype=np.int64).tolist(),
        "ndata": ndata,
        "input_fNL": float(truth),
        "target_realization_count": 500 if truth == 0.0 else 250,
        "map": mapping,
        "hesse_errors": {name: float(best.errors[name]) for name in NAMES},
        "minos": minos,
        "data_chi2": data_q,
        "prior_chi2": prior_q,
        "objective": float(best.fval),
        "dof": ndata - len(NAMES),
        "gof_p_value_data_only": float(
            chi2_distribution.sf(data_q, max(ndata - len(NAMES), 1))
        ),
        "optimizer": helpers.minuit_record(best, candidates),
        "truth_profile": {
            "mapping": fixed_map,
            "data_chi2": fixed_data_q,
            "prior_chi2": fixed_prior_q,
            "objective": float(fixed.fval),
            "delta_chi2": truth_delta,
            "optimizer_valid": bool(fixed.valid),
            "inside_delta_chi2_1": truth_delta <= 1.0 + 1.0e-7,
        },
        "flat_fNL_prior": [truth - FNL_HALF_WIDTH, truth + FNL_HALF_WIDTH],
        "fNL_interval_at_prior_limit": at_limit,
        "model": {
            "callable": f"B000_{model.method}(fNL,b1,b2,gamma2,Ashot,Bshot)",
            "b1_fixed": FIXED_B1, "fully_tree_dm": True,
            "finite_png_order": 2, "frozen_response": False,
            "matter_loop": False, "empirical_gain": False,
        },
        "hartlap": hartlap,
    }
    return row, {
        "target": target, "prediction": prediction,
        "residual": target - prediction,
        "covariance_single_realization": covariance_one,
        "truth_fixed_prediction": fixed_prediction,
    }


def profiled_precision(
    model: scan.DirectFiniteScanB000Model,
    mapping: dict[str, float],
    indices: np.ndarray,
    covariance_one: np.ndarray,
) -> np.ndarray:
    ndata = int(indices.size)
    hartlap = (500.0 - ndata - 2.0) / 499.0
    precision = hartlap * VOLUME * np.linalg.inv(covariance_one)
    columns = []
    for name in NUISANCE_NAMES:
        step = 1.0e-4 * max(abs(mapping[name]), 1.0)
        plus = dict(mapping); minus = dict(mapping)
        plus[name] += step; minus[name] -= step
        columns.append(
            (model(0.0, plus, indices) - model(0.0, minus, indices)) / (2.0 * step)
        )
    jacobian = np.column_stack(columns)
    context = model.base_context
    full_prior = (
        np.asarray(context.prior.whitener).T @ np.asarray(context.prior.whitener)
    )
    positions = np.asarray([context.names.index(name) for name in NUISANCE_NAMES])
    prior = full_prior[np.ix_(positions, positions)]
    fisher = jacobian.T @ precision @ jacobian + prior
    return precision - precision @ jacobian @ np.linalg.solve(fisher, jacobian.T @ precision)


def metrics(theory: np.ndarray, measured: np.ndarray, precision: np.ndarray) -> dict[str, float]:
    tt = float(theory @ precision @ theory)
    mm = float(measured @ precision @ measured)
    tm = float(theory @ precision @ measured)
    return {
        "amplitude": tm / tt,
        "cosine": tm / math.sqrt(max(tt * mm, 1.0e-300)),
        "theory_norm": math.sqrt(max(tt, 0.0)),
        "measured_norm": math.sqrt(max(mm, 0.0)),
    }


def response_diagnostics(
    runtime: scan.ScanRuntime,
    models: dict[str, scan.DirectFiniteScanB000Model],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for method in METHODS:
        model = models[method]
        for cut in CUTS:
            indices = scan.selected_indices(runtime.geometry, cut)
            covariance = np.cov(fiducial_samples(runtime, method, indices), rowvar=False, ddof=1)
            fid_row = next(
                row for row in rows
                if row["method"] == method and row["kmax_h_mpc"] == cut
                and row["input_fNL"] == 0.0
            )
            mapping = fid_row["truth_profile"]["mapping"]
            precision = profiled_precision(model, mapping, indices, covariance)
            local = {}
            zero = np.mean(target_samples(runtime, method, 0.0, indices), axis=0)
            for amplitude in (50.0, 100.0):
                plus = np.mean(target_samples(runtime, method, amplitude, indices), axis=0)
                minus = np.mean(target_samples(runtime, method, -amplitude, indices), axis=0)
                measured_odd = (plus - minus) / (2.0 * amplitude)
                measured_even = (plus + minus) / 2.0 - zero
                theory_plus = model(amplitude, mapping, indices)
                theory_minus = model(-amplitude, mapping, indices)
                theory_zero = model(0.0, mapping, indices)
                theory_odd = (theory_plus - theory_minus) / (2.0 * amplitude)
                theory_even = (theory_plus + theory_minus) / 2.0 - theory_zero
                local[f"pm{int(amplitude)}"] = {
                    "odd": metrics(theory_odd, measured_odd, precision),
                    "even": metrics(theory_even, measured_even, precision),
                }
            result[f"{method}__kmax{cut:.2f}"] = local
    return result


def summarize(
    rows: list[dict[str, Any]], response: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {}
    decision = {}
    for method in METHODS:
        method_rows = {}
        for cut in CUTS:
            local = sorted(
                [r for r in rows if r["method"] == method and r["kmax_h_mpc"] == cut],
                key=lambda r: r["input_fNL"],
            )
            truth = np.asarray([r["input_fNL"] for r in local])
            centres = np.asarray([r["map"]["fNL"] for r in local])
            slope, intercept = np.polyfit(truth, centres, 1)
            coverage = sum(r["truth_profile"]["inside_delta_chi2_1"] for r in local)
            gaussian_row = next(r for r in local if r["input_fNL"] == 0.0)
            gaussian_p = float(
                chi2_distribution.sf(
                    gaussian_row["truth_profile"]["data_chi2"],
                    max(int(gaussian_row["ndata"]) - len(NUISANCE_NAMES), 1),
                )
            )
            all_valid = all(
                r["optimizer"]["valid"] and r["minos"]["fNL"]["valid"]
                and r["truth_profile"]["optimizer_valid"] for r in local
            )
            no_limits = all(not r["fNL_interval_at_prior_limit"] for r in local)
            strict_valid = bool(
                gaussian_p >= 0.05 and coverage == 5 and all_valid and no_limits
            )
            diag = response[f"{method}__kmax{cut:.2f}"]
            odd_support = bool(
                all(
                    0.8 <= diag[tag]["odd"]["amplitude"] <= 1.2
                    and diag[tag]["odd"]["cosine"] >= 0.90
                    for tag in ("pm50", "pm100")
                )
            )
            record = {
                "centres": centres,
                "biases": centres - truth,
                "slope": float(slope), "intercept": float(intercept),
                "mean_absolute_bias": float(np.mean(np.abs(centres - truth))),
                "maximum_truth_delta_chi2": max(r["truth_profile"]["delta_chi2"] for r in local),
                "truth_coverage_count": int(coverage),
                "gaussian_fixed_fNL0_data_chi2": gaussian_row["truth_profile"]["data_chi2"],
                "gaussian_fixed_fNL0_dof": int(gaussian_row["ndata"]) - len(NUISANCE_NAMES),
                "gaussian_fixed_fNL0_p_value": gaussian_p,
                "all_optimizers_and_fNL_intervals_valid": all_valid,
                "no_fNL_prior_limit": no_limits,
                "odd_response_support": odd_support,
                "strict_injection_and_gaussian_valid": strict_valid,
                "response": diag,
            }
            method_rows[f"kmax{cut:.2f}"] = record
        first_failed = next(
            (cut for cut in CUTS if not method_rows[f"kmax{cut:.2f}"]["strict_injection_and_gaussian_valid"]),
            None,
        )
        largest_valid_before_failure = None
        for cut in CUTS:
            if first_failed is not None and cut >= first_failed:
                break
            if method_rows[f"kmax{cut:.2f}"]["strict_injection_and_gaussian_valid"]:
                largest_valid_before_failure = cut
        if first_failed is None:
            largest_valid_before_failure = CUTS[-1]
        summary[method] = method_rows
        decision[method] = {
            "first_failed_kmax_h_mpc": first_failed,
            "largest_strictly_valid_kmax_before_first_failure_h_mpc": largest_valid_before_failure,
            "failure_definition": (
                "first cut where Gaussian fixed-fNL0 p<0.05, or fewer than 5/5 truths "
                "have profiled Delta-chi2<=1, or an optimizer/interval is invalid/censored"
            ),
        }
    return summary, decision


def row_key(row: dict[str, Any]) -> str:
    return f"{row['method']}__kmax{row['kmax_h_mpc']:.2f}__fNL{row['input_fNL']:+.0f}"


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "method", "kmax_h_mpc", "ndata", "input_fNL", "recovered_fNL",
        "plus_error", "minus_error", "truth_delta_chi2", "truth_inside",
        "data_chi2", "prior_chi2", "optimizer_valid", "minos_valid", "at_prior_limit",
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            interval = row["minos"]["fNL"]
            writer.writerow({
                "method": row["method"], "kmax_h_mpc": row["kmax_h_mpc"],
                "ndata": row["ndata"], "input_fNL": row["input_fNL"],
                "recovered_fNL": row["map"]["fNL"],
                "plus_error": interval["upper_error"],
                "minus_error": -interval["lower_error"],
                "truth_delta_chi2": row["truth_profile"]["delta_chi2"],
                "truth_inside": row["truth_profile"]["inside_delta_chi2_1"],
                "data_chi2": row["data_chi2"], "prior_chi2": row["prior_chi2"],
                "optimizer_valid": row["optimizer"]["valid"],
                "minos_valid": interval["valid"],
                "at_prior_limit": row["fNL_interval_at_prior_limit"],
            })
    temporary.replace(OUTPUT_CSV)


def plot_recovery(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_RECOVERY, metadata={"Author": "direct finite-tree kmax scan"}) as pdf:
        for method in METHODS:
            fig, axis = plt.subplots(figsize=(7.6, 6.4))
            diagonal = np.linspace(-125, 125, 200)
            axis.plot(diagonal, diagonal, color="0.3", ls="--", lw=1.0)
            colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(CUTS)))
            for offset, (cut, colour) in enumerate(zip(CUTS, colours)):
                local = sorted(
                    [r for r in rows if r["method"] == method and r["kmax_h_mpc"] == cut],
                    key=lambda r: r["input_fNL"],
                )
                truth = np.asarray([r["input_fNL"] for r in local])
                centre = np.asarray([r["map"]["fNL"] for r in local])
                lower = np.asarray([-r["minos"]["fNL"]["lower_error"] for r in local])
                upper = np.asarray([r["minos"]["fNL"]["upper_error"] for r in local])
                axis.errorbar(
                    truth + (offset - 3) * 0.8, centre,
                    yerr=np.vstack((lower, upper)), color=colour,
                    marker="o", ms=3.5, lw=1.0, capsize=2,
                    label=fr"$k_{{\max}}={cut:.2f}$",
                )
            axis.set_xlabel(r"input $f_{\rm NL}$", fontsize=14)
            axis.set_ylabel(r"recovered $f_{\rm NL}$", fontsize=14)
            axis.set_title(METHOD_LABELS[method], fontsize=15)
            axis.text(
                0.03, 0.97,
                r"$B_{000}$ only, $b_1=1.93$" + "\n" + r"$V=42.875\,(h^{-1}{\rm Gpc})^3$",
                transform=axis.transAxes, va="top", fontsize=11,
            )
            axis.grid(alpha=0.18); axis.legend(frameon=False, ncol=2, fontsize=9)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def plot_validity(summary: dict[str, Any], decision: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 11.4), sharex=True)
    cuts = np.asarray(CUTS)
    for row_index, method in enumerate(METHODS):
        local = [summary[method][f"kmax{cut:.2f}"] for cut in CUTS]
        gaussian_p = np.asarray([x["gaussian_fixed_fNL0_p_value"] for x in local])
        coverage = np.asarray([x["truth_coverage_count"] for x in local])
        max_delta = np.asarray([x["maximum_truth_delta_chi2"] for x in local])
        slopes = np.asarray([x["slope"] for x in local])
        amp50 = np.asarray([x["response"]["pm50"]["odd"]["amplitude"] for x in local])
        amp100 = np.asarray([x["response"]["pm100"]["odd"]["amplitude"] for x in local])
        colour = METHOD_COLOURS[method]
        axes[row_index, 0].plot(cuts, gaussian_p, color=colour, marker="o")
        axes[row_index, 0].axhline(0.05, color="0.3", ls="--", lw=0.9)
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel(f"{METHOD_LABELS[method]}\nGaussian p")
        axes[row_index, 1].plot(cuts, coverage, color=colour, marker="o", label="coverage / 5")
        axes[row_index, 1].plot(cuts, max_delta, color="0.25", marker="s", label=r"max $\Delta\chi^2_{\rm truth}$")
        axes[row_index, 1].axhline(1.0, color="0.3", ls="--", lw=0.9)
        axes[row_index, 1].legend(frameon=False, fontsize=8)
        axes[row_index, 2].plot(cuts, slopes, color=colour, marker="o", label="recovery slope")
        axes[row_index, 2].plot(cuts, amp50, color="#59A14F", marker="^", label=r"odd $\pm50$ amp")
        axes[row_index, 2].plot(cuts, amp100, color="#F28E2B", marker="v", label=r"odd $\pm100$ amp")
        axes[row_index, 2].axhline(1.0, color="0.3", ls="--", lw=0.9)
        axes[row_index, 2].legend(frameon=False, fontsize=8)
        failed = decision[method]["first_failed_kmax_h_mpc"]
        if failed is not None:
            for axis in axes[row_index]:
                axis.axvline(failed, color="#B40426", ls=":", lw=1.2)
        for axis in axes[row_index]:
            axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=12)
    axes[0, 0].set_title("Gaussian backbone")
    axes[0, 1].set_title("injection recovery")
    axes[0, 2].set_title("response amplitude")
    fig.tight_layout()
    fig.savefig(OUTPUT_VALIDITY, metadata={"Author": "direct finite-tree kmax scan"})
    plt.close(fig)


def plot_bestfit(
    runtime: scan.ScanRuntime, rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]
) -> None:
    with PdfPages(OUTPUT_BESTFIT, metadata={"Author": "direct finite-tree kmax scan"}) as pdf:
        for method in METHODS:
            fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)
            colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(CUTS)))
            for cut, colour in zip(CUTS, colours):
                row = next(
                    r for r in rows if r["method"] == method
                    and r["kmax_h_mpc"] == cut and r["input_fNL"] == 0.0
                )
                key = row_key(row); indices = np.asarray(row["retained_indices"])
                target = arrays[f"{key}__target"]
                prediction = arrays[f"{key}__truth_fixed_prediction"]
                covariance = arrays[f"{key}__covariance_single_realization"]
                sigma = np.sqrt(np.diag(covariance))
                kmax_each = np.max(runtime.geometry[indices], axis=1)
                axes[0].plot(kmax_each, target, color=colour, marker="o", ms=3, lw=0.8, alpha=0.55)
                axes[0].plot(kmax_each, prediction, color=colour, lw=1.2, label=f"{cut:.2f}")
                axes[1].plot(kmax_each, (target - prediction) / sigma, color=colour, marker="o", ms=3, lw=0.9)
            axes[0].set_ylabel(r"$B_{000}$")
            axes[1].set_ylabel(r"$(\bar B-B^{\rm model})/\sigma_{\rm 1real}$")
            axes[1].set_xlabel(r"$\max(k_1,k_2)\ [h\,{\rm Mpc}^{-1}]$")
            axes[0].set_title(METHOD_LABELS[method]); axes[0].legend(title=r"$k_{\max}$", frameon=False, ncol=4)
            for axis in axes: axis.grid(alpha=0.18)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def run(overwrite: bool) -> dict[str, Any]:
    if OUTPUT_JSON.is_file() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {OUTPUT_JSON}; pass --overwrite")
    runtime = scan.load_runtime()
    overlap = scan.low_k_overlap(runtime)
    if not overlap["pass"]:
        raise ValueError("high-k model does not reproduce the accepted kmax=0.08 model")
    models = {method: scan.DirectFiniteScanB000Model(runtime, method) for method in METHODS}
    rows = []; arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        model = models[method]
        for cut in CUTS:
            indices = scan.selected_indices(runtime.geometry, cut)
            fiducial = fiducial_samples(runtime, method, indices)
            covariance = np.cov(fiducial, rowvar=False, ddof=1)
            for truth in TRUTHS:
                target = np.mean(target_samples(runtime, method, truth, indices), axis=0)
                row, local = profile_case(model, indices, target, covariance, truth, cut)
                rows.append(row); key = row_key(row)
                for name, value in local.items(): arrays[f"{key}__{name}"] = np.asarray(value)
                print(
                    f"{key}: {row['map']['fNL']:.5f} "
                    f"+{row['minos']['fNL']['upper_error']:.4f}/"
                    f"-{(-row['minos']['fNL']['lower_error']):.4f} "
                    f"dtruth={row['truth_profile']['delta_chi2']:.3f}",
                    flush=True,
                )
    response = response_diagnostics(runtime, models, rows)
    summary, decision = summarize(rows, response)
    integrity = {
        "row_count_105": len(rows) == len(METHODS) * len(CUTS) * len(TRUTHS),
        "low_k_overlap": bool(overlap["pass"]),
        "all_predictions_finite": all(
            np.all(np.isfinite(value)) for name, value in arrays.items() if name.endswith("__prediction")
        ),
        "no_frozen_response": all(not row["model"]["frozen_response"] for row in rows),
        "fixed_b1_1p93": all(row["model"]["b1_fixed"] == FIXED_B1 for row in rows),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"kmax scan integrity failed: {integrity}")
    helpers.atomic_npz(OUTPUT_NPZ, arrays); write_csv(rows)
    plot_recovery(rows); plot_validity(summary, decision); plot_bestfit(runtime, rows, arrays)
    payload = {
        "schema": "quijote-halo-z0p5-direct-finite-tree-kmax-validity-scan-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "methods": list(METHODS), "cuts_h_mpc": list(CUTS),
            "statistic": "B000 only", "b1_fixed": FIXED_B1,
            "fully_tree_dm": True, "finite_png_order": 2,
            "P0_likelihood_used": False, "target_volume_gpc3_h3": VOLUME,
            "target_counts": {"fiducial": 500, "nonzero_each": 250},
        },
        "pre_registered_failure_rule": {
            "gaussian_p_minimum": 0.05,
            "required_truth_coverage": "5/5 at profiled Delta-chi2<=1",
            "optimizer_and_interval_must_be_valid_and_uncensored": True,
            "odd_response_threshold_is_supporting_not_primary": (
                "amplitude in [0.8,1.2] and cosine>=0.90 for both +/-50 and +/-100"
            ),
        },
        "integrity": integrity, "low_k_overlap": overlap,
        "summary": summary, "validity_decision": decision,
        "response_diagnostics": response, "rows": rows,
        "inputs": {
            "runner": artifact(Path(__file__).resolve()),
            "model_module": artifact(CODES / "quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models.py"),
            "pre_high_tree": artifact(scan.PRE_TREE),
            "pre_high_tree_manifest": artifact(scan.PRE_TREE_MANIFEST),
            "marisa_high_finite": artifact(scan.MARISA_FINITE),
            "marisa_high_finite_manifest": artifact(scan.MARISA_FINITE_MANIFEST),
        },
        "outputs": {
            "arrays": str(OUTPUT_NPZ), "csv": str(OUTPUT_CSV),
            "recovery_pdf": str(OUTPUT_RECOVERY), "validity_pdf": str(OUTPUT_VALIDITY),
            "bestfit_pdf": str(OUTPUT_BESTFIT),
        },
    }
    helpers.atomic_json(OUTPUT_JSON, payload)
    payload["outputs"] = {name: artifact(Path(path)) for name, path in payload["outputs"].items()}
    helpers.atomic_json(OUTPUT_JSON, payload)
    print(json.dumps(decision, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); run(args.overwrite)


if __name__ == "__main__":
    main()
