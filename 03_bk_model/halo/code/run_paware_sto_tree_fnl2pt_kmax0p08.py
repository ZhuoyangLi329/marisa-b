#!/usr/bin/env python3
"""p-aware STO-only fully-tree B000 两点扫描（fNL=0/+100，kmax=0.08）。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MARISA_B_MAX_THREADS",
):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
V2_ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
CODES = ROOT / "codes"
TREE10_RUNNER = CODES / "run_quijote_halo_z0p5_fnl250_std_marisa_tree_dm_fixed_b1_profiles.py"
BASE_PROFILE = CODES / "run_quijote_halo_z0p5_marisa_only_p112_fixed_b1_1p93_profiles.py"
FIT_HELPER = CODES / "run_quijote_halo_z0p5_fnl250_std_marisa_fixed_b1_profiles.py"
TREE10_ROOT = (
    ROOT
    / "analysis/quijote_halo_z0p5_fnl250_std_marisa_tree_dm_fixed_b1_1p93_kmax0p10_v42p875"
)
TREE10_PROFILES = TREE10_ROOT / "FULL_AVAILABLE_MATCHED250_TREE_DM_PROFILES.json"
PRE_REFERENCE = (
    ROOT
    / "analysis/quijote_halo_z0p5_local_png_p112_recovery_kmax0p10_v42p875_priorpm200/"
    "FIXED_B1_1P93_PROFILES.json"
)
MATRIX_ROOT = ROOT / "analysis/quijote_halo_z0p5_fnl250_standard_marisa/matrices"
STO_MATRIX = (
    ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/sto_only_matrix.npz"
)
MATRIX_SUMMARY = MATRIX_ROOT / "FNL250_MATRIX_SUMMARY.json"
MATRICES = {
    "standard": MATRIX_ROOT / "STANDARD_FNL250_MATRIX.npz",
    "marisa": MATRIX_ROOT / "MARISA_FNL250_MATRIX.npz",
}

TAG = "quijote_halo_z0p5_sto_tree_dm_paware1p3395_kmax0p08_v42p875"
ANALYSIS = V2_ROOT / "03_bk_model/halo/work" / TAG
FIGURES = ANALYSIS / "figures"
OUTPUT_JSON = ANALYSIS / "FULL_AVAILABLE_MATCHED250_TREE_DM_KMAX0P08_PROFILES.json"
OUTPUT_CSV = ANALYSIS / "FULL_AVAILABLE_MATCHED250_TREE_DM_KMAX0P08_PROFILES.csv"
OUTPUT_ARRAYS = ANALYSIS / "FULL_AVAILABLE_MATCHED250_TREE_DM_KMAX0P08_ARRAYS.npz"
OUTPUT_MANIFEST = ANALYSIS / "TREE_DM_KMAX0P08_MODEL_MANIFEST.json"
OUTPUT_GATE = ANALYSIS / "TREE_DM_KMAX0P08_GAUSSIAN_GATE.json"
OUTPUT_REPORT = ANALYSIS / "REPORT.md"
OUTPUT_RECOVERY = FIGURES / "pre_standard_marisa_tree_dm_kmax0p08_fNL_recovery.pdf"
OUTPUT_PARAMETERS = FIGURES / "tree_dm_kmax0p08_full_available_all_parameter_profiles.pdf"
OUTPUT_COMPARISON = FIGURES / "standard_marisa_tree_dm_kmax0p08_vs_kmax0p10.pdf"
OUTPUT_BESTFIT = {
    (method, target): FIGURES / f"{method}_tree_dm_kmax0p08_bestfit_{target}.pdf"
    for method in ("pre",)
    for target in ("full_available",)
}

METHODS = ("pre",)
TARGETS = ("full_available",)
TRUTHS = (0.0, 100.0)
TRUTH_KEYS = {0.0: "fiducial", 100.0: "LC_p"}
TRUTH_TAGS = {0.0: "fid0", 100.0: "p100"}
ALL_PARAMETERS = ("fNL", "b1", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
FREE_PARAMETERS = ("fNL", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
PRE_TREE_LINEAR_KEYS = (
    "primordial",
    "bphi_f2",
    "advection",
    "bphidelta",
    "bphi_b2",
    "bphi_bK2",
    "stochastic",
)
METHOD_LABELS = {
    "pre": "pre reconstruction",
    "standard": "standard reconstruction",
    "marisa": "Marisa reconstruction",
}
METHOD_COLOURS = {"pre": "#4E79A7", "standard": "#E15759", "marisa": "#7A4EAB"}
METHOD_MARKERS = {"pre": "^", "standard": "s", "marisa": "o"}
TRUTH_COLOURS = {
    -100.0: "#3B4CC0",
    -50.0: "#7B9FF9",
    0.0: "#595959",
    50.0: "#F4987A",
    100.0: "#B40426",
}
PARAMETER_LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$ (fixed)",
    "b2": r"$b_2$",
    "gamma2": r"$\gamma_2$",
    "Ashot_residual": r"$A_{\rm shot}$",
    "Bshot_residual": r"$B_{\rm shot}$",
}
KMAX = 0.08
NDATA = 9
FIXED_B1 = 2.197952896776342
P_RECON = 1.3394637946618602  # p-aware STO tracer（P0 fNL=100 固定校准，2026-09-16）
VOLUME = 42.875
PRIOR_HALF_WIDTH = 200.0
HARTLAP = (500.0 - NDATA - 2.0) / 499.0
GAUSSIAN_GATE_MIN_P = 0.01


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def relative_l2(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(observed - expected) / max(float(np.linalg.norm(expected)), 1.0))


def full_vector(components: dict[str, Any], name: str) -> np.ndarray:
    if name not in components:
        raise KeyError(f"required component is absent: {name}")
    value = np.asarray(components[name], dtype=np.float64)
    if value.shape != (120,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid component {name}: {value.shape}")
    return value


def pre_tree_only_context(context: Any) -> Any:
    original = context.evaluate_model
    selected = np.asarray(context.indices, dtype=np.int64)

    def evaluate(
        fnl: float, nuisance: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
        _old_prediction, parameters, components = original(fnl, nuisance)
        gaussian_tree = full_vector(components, "gaussian_tree")
        gaussian_stochastic = full_vector(components, "gaussian_stochastic")
        gaussian_loop = full_vector(components, "gaussian_matter_loop")
        gaussian_old = full_vector(components, "gaussian_total")
        tree_response = sum(
            (full_vector(components, name) for name in PRE_TREE_LINEAR_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        matter_loop = full_vector(components, "matter_linear_uplift")
        linear_old = full_vector(components, "linear_total")
        if relative_l2(gaussian_old, gaussian_tree + gaussian_stochastic + gaussian_loop) >= 2.0e-13:
            raise ValueError("pre Gaussian decomposition does not close")
        if relative_l2(linear_old, tree_response + matter_loop) >= 2.0e-13:
            raise ValueError("pre PNG decomposition does not close")
        quadratic = full_vector(components, "quadratic_total")
        if np.any(quadratic != 0.0):
            raise ValueError("pre main model quadratic_total is not exactly zero")
        gaussian = gaussian_tree + gaussian_stochastic
        updated = dict(components)
        updated.update(
            {
                "pre_ablation_gaussian_total": gaussian_old,
                "pre_ablation_linear_total": linear_old,
                "removed_gaussian_matter_loop": gaussian_loop,
                "removed_matter_linear_uplift": matter_loop,
                "gaussian_matter_loop": np.zeros(120),
                "matter_linear_uplift": np.zeros(120),
                "gaussian_total": gaussian,
                "fixed_brec_linear_total": tree_response,
                "adaptive_tree": np.zeros(120),
                "adaptive_matter_loop": np.zeros(120),
                "linear_total": tree_response,
            }
        )
        prediction = gaussian + float(fnl) * tree_response
        return prediction[selected], parameters, updated

    return replace(
        context,
        reconstruction=f"{context.reconstruction}_fully_tree_dm",
        evaluate_model=evaluate,
        input_hashes={
            **context.input_hashes,
            "dm_perturbative_order": "tree",
            "gaussian_matter_loop": "removed",
            "matter_linear_uplift": "removed",
            "tree_dm_ablation": "pre component-level exact subtraction and recomposition",
        },
    )


def restrict_context(
    context: Any,
    target_samples: np.ndarray,
    fiducial_samples: np.ndarray,
    geometry: np.ndarray,
) -> Any:
    parent_indices = np.asarray(context.indices, dtype=np.int64)
    retained = np.asarray(
        [index for index in parent_indices if float(np.max(geometry[index])) <= KMAX + 1.0e-12],
        dtype=np.int64,
    )
    if retained.size != NDATA:
        raise ValueError(f"kmax=0.08 requires {NDATA} nonzero-global bins, found {retained.tolist()}")
    positions = []
    for index in retained:
        location = np.flatnonzero(parent_indices == index)
        if location.size != 1:
            raise ValueError(f"parent context does not contain global bin {index}")
        positions.append(int(location[0]))
    positions_array = np.asarray(positions, dtype=np.int64)
    original = context.evaluate_model

    def evaluate_subset(
        fnl: float, nuisance: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
        prediction, parameters, components = original(fnl, nuisance)
        return np.asarray(prediction)[positions_array], parameters, components

    covariance_samples = np.asarray(fiducial_samples[:, retained], dtype=np.float64)
    selected_targets = np.asarray(target_samples[:, retained], dtype=np.float64)
    return replace(
        context,
        reconstruction=f"{context.reconstruction}_kmax0p08",
        kmax=KMAX,
        indices=retained,
        target=np.mean(selected_targets, axis=0),
        covariance_single=np.cov(covariance_samples, rowvar=False, ddof=1) / VOLUME,
        covariance_mock_count=500,
        samples=covariance_samples,
        realization_ids=np.arange(500, dtype=np.int64),
        evaluate_model=evaluate_subset,
        input_hashes={
            **context.input_hashes,
            "requested_kmax_h_mpc": "0.08",
            "retained_global_indices": ",".join(str(int(index)) for index in retained),
            "target_realization_count": str(target_samples.shape[0]),
            "covariance_realization_count": "500",
            "covariance_volume_divisor": "42.875",
        },
    )


def likelihood_contract(
    context: Any,
    covariance_full: np.ndarray,
    truth: float,
    target_count: int,
) -> dict[str, Any]:
    indices = np.asarray(context.indices, dtype=np.int64)
    if indices.size != NDATA:
        raise ValueError("likelihood dimension drift")
    covariance_one = np.asarray(covariance_full, dtype=np.float64)[np.ix_(indices, indices)]
    if relative_l2(np.asarray(context.covariance_single), covariance_one / VOLUME) >= 1.0e-14:
        raise ValueError("covariance scaling drift")
    if not math.isclose(float(context.hartlap), HARTLAP, rel_tol=0.0, abs_tol=2.0e-15):
        raise ValueError("Hartlap drift")
    expected_chol = np.linalg.cholesky(covariance_one / VOLUME) / math.sqrt(HARTLAP)
    if relative_l2(np.asarray(context.chol_fit), expected_chol) >= 1.0e-12:
        raise ValueError("Hartlap Cholesky drift")
    expected_prior = (truth - PRIOR_HALF_WIDTH, truth + PRIOR_HALF_WIDTH)
    if not np.array_equal(np.asarray(context.fnl_bounds), np.asarray(expected_prior)):
        raise ValueError("fNL prior drift")
    return {
        "target_is_measurement_mean": True,
        "target_realization_count": int(target_count),
        "covariance_source": "method/stage-specific Fid500 post/pre B000 single-realization covariance",
        "covariance_is_error_on_mean": False,
        "covariance_volume_divisor": VOLUME,
        "covariance_mock_count": 500,
        "retained_data_dimension": NDATA,
        "retained_global_indices": indices.tolist(),
        "hartlap_formula": "(Nmock-Ndata-2)/(Nmock-1)",
        "hartlap": HARTLAP,
        "flat_fNL_prior": list(expected_prior),
    }


def load_bundle() -> SimpleNamespace:
    for path in (
        TREE10_RUNNER,
        BASE_PROFILE,
        FIT_HELPER,
        TREE10_PROFILES,
        PRE_REFERENCE,
        MATRIX_SUMMARY,
        *MATRICES.values(),
        STO_MATRIX,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    tree10 = import_path("_tree08_tree10", TREE10_RUNNER)
    production = import_path("_tree08_base", BASE_PROFILE)
    helper = import_path("_tree08_fit_helper", FIT_HELPER)
    recovery, contexts, fixed, runtime, _old, adaptive, _reference = production.configure()
    recovery.COVARIANCE_VOLUME = VOLUME
    recovery.FNL_PRIOR_HALF_WIDTH = PRIOR_HALF_WIDTH
    fixed.FIXED_B1 = FIXED_B1
    fixed.FIXED_BPHI = production.FIXED_BPHI
    # p-aware：pre 通道 tracer bphi 的真正入口是 recovery 模块求值时的
    # P_BPHI（bphi = 2*delta_c*(b1 - P_BPHI)）；build 之后统一覆盖，
    # production/fixed 的 FIXED_BPHI 同步按新 p 重算。
    recovery.P_BPHI = P_RECON
    production.FIXED_BPHI = 2.0 * production.DELTA_C * (FIXED_B1 - P_RECON)
    fixed.FIXED_BPHI = production.FIXED_BPHI
    if hasattr(fixed, "P_BPHI"):
        fixed.P_BPHI = P_RECON
    if hasattr(tree10, "P_RECON"):
        tree10.P_RECON = P_RECON
    matrices = {method: load_npz(path) for method, path in MATRICES.items()}
    matrices["sto"] = load_npz(STO_MATRIX)
    if not np.allclose(
        np.asarray(matrices["sto"]["b000_k"], dtype=np.float64),
        np.asarray(matrices["standard"]["B000_k"], dtype=np.float64),
        rtol=0.0, atol=2.0e-6,
    ):
        raise ValueError("sto/standard B000 geometry drift")
    geometry = np.asarray(matrices["standard"]["B000_k"], dtype=np.float64)
    if (
        not np.array_equal(
            matrices["standard"]["B000_k_edges"],
            matrices["marisa"]["B000_k_edges"],
        )
        or not np.array_equal(
            matrices["standard"]["B000_nmodes"],
            matrices["marisa"]["B000_nmodes"],
        )
        or not np.allclose(
            geometry,
            matrices["marisa"]["B000_k"],
            rtol=0.0,
            atol=2.0e-6,
        )
    ):
        raise ValueError("standard/Marisa B000 edge/mode/centroid geometry drift")
    tree10_payload = json.loads(TREE10_PROFILES.read_text(encoding="utf-8"))
    tree10_rows = {
        (str(row["target_selection"]), str(row["method"]), float(row["input_fNL"])): row
        for row in tree10_payload["rows"]
    }
    pre_payload = json.loads(PRE_REFERENCE.read_text(encoding="utf-8"))
    pre_rows = {float(row["input_fNL"]): row for row in pre_payload["rows"] if row["method"] == "pre"}
    if not set(TRUTHS).issubset(pre_rows):
        raise ValueError("pre initialization registry is incomplete")
    return SimpleNamespace(
        tree10=tree10,
        production=production,
        helper=helper,
        recovery=recovery,
        fixed=fixed,
        runtime=runtime,
        contexts=contexts,
        adaptive=adaptive,
        matrices=matrices,
        geometry=geometry,
        tree10_payload=tree10_payload,
        tree10_rows=tree10_rows,
        pre_rows=pre_rows,
    )


def target_samples(matrix: dict[str, np.ndarray], method: str, target: str, truth: float) -> np.ndarray:
    stage = "pre" if method == "pre" else "post"
    samples = np.asarray(matrix[f"{TRUTH_KEYS[truth]}_{stage}_B000"], dtype=np.float64)
    count = 500 if truth == 0.0 else 100
    return samples[:count]


def make_context(
    bundle: SimpleNamespace,
    method: str,
    target: str,
    truth: float,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    # STO-only：pre 结构的 tree 模型，数据用 STO 加权 catalog 的 post B000。
    matrix = bundle.matrices["sto"]
    stage = "post"
    fiducial = np.asarray(matrix[f"fiducial_{stage}_B000"], dtype=np.float64)
    samples = target_samples(matrix, method, target, truth)
    base_context = bundle.contexts["pre" if method == "pre" else "standard"]
    recovery_method = "pre" if method == "pre" else "standard"
    context, contract = bundle.runtime.recovery_context(
        base_context, recovery_method, truth, samples, fiducial
    )
    if method == "pre":
        context = pre_tree_only_context(context)
    else:
        if method == "marisa":
            context = bundle.production.add_adaptive_response(context, bundle.adaptive)
        context = bundle.tree10.tree_only_context(context, method)
    context = restrict_context(context, samples, fiducial, bundle.geometry)
    covariance_key = "fiducial_post_B000_cov"
    covariance_full = np.asarray(matrix[covariance_key], dtype=np.float64)
    likelihood = likelihood_contract(context, covariance_full, truth, samples.shape[0])
    return context, {**contract, "likelihood": likelihood}, samples, fiducial, covariance_full


def reference_row(bundle: SimpleNamespace, method: str, target: str, truth: float) -> dict[str, Any]:
    if method == "pre":
        return bundle.pre_rows[truth]
    return bundle.tree10_rows[(target, method, truth)]


def clean_row(row: dict[str, Any], method: str, truth: float) -> dict[str, Any]:
    row["method"] = method
    row["input_fNL"] = truth
    row["reconstruction"] = (
        {"kind": "pre", "p": P_RECON}
        if method == "pre"
        else {
            "kind": method,
            "R_mpc_h": 10.0,
            "b_rec": 1.9326526728794324,
            "p": P_RECON,
        }
    )
    row["theory"] = {
        "form": (
            "B_pre_G_tree + fNL * R_pre_tree"
            if method == "pre"
            else (
                "B_G_tree + fNL * R_standard_tree"
                if method == "standard"
                else "B_G_tree + fNL * (R_standard_tree + R_adaptive_tree)"
            )
        ),
        "dm_perturbative_order": "tree",
        "gaussian_matter_loop": False,
        "matter_linear_or_one_loop_uplift": False,
        "physical_K1_matter_loop_response": False,
        "adaptive_matter_loop": False,
        "marisa_adaptive_tree": method == "marisa",
        "leading_residual_stochastic": True,
        "fNL_squared": False,
        "response_gain_parameter": False,
        "adaptive_stochastic_png": False,
    }
    row.pop("delta_chi2_vs_free_b1", None)
    row.pop("free_b1_reference", None)
    for key in ("fixed_over_free_half_width", "centre_shift_fixed_minus_free", "free_b1_half_width"):
        row.get("fNL_recovery", {}).pop(key, None)
    return row


def closure_and_gates(bundle: SimpleNamespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    closure_rows: dict[str, Any] = {}
    gates: list[dict[str, Any]] = []
    gate_arrays: dict[str, np.ndarray] = {}
    all_closure = True
    for method in METHODS:
        context, _contract, _samples, _fiducial, _covariance = make_context(
            bundle, method, "full_available", 0.0
        )
        reference = reference_row(bundle, method, "full_available", 0.0)
        nuisance = np.asarray([reference["map"][name] for name in context.names], dtype=np.float64)
        zero, _parameters, components = context.evaluate_model(0.0, nuisance)
        plus, _parameters, _components = context.evaluate_model(1.0, nuisance)
        minus, _parameters, _components = context.evaluate_model(-1.0, nuisance)
        selected = np.asarray(context.indices, dtype=np.int64)
        gaussian = full_vector(components, "gaussian_total")
        gaussian_expected = full_vector(components, "gaussian_tree") + full_vector(
            components, "gaussian_stochastic"
        )
        linear = full_vector(components, "linear_total")
        central = (np.asarray(plus) - np.asarray(minus)) / 2.0
        metrics = {
            "gaussian_tree_plus_stochastic_relative_l2": relative_l2(gaussian, gaussian_expected),
            "central_difference_relative_l2": relative_l2(central, linear[selected]),
            "fNL_linearity_relative_l2": relative_l2(
                np.asarray(plus) + np.asarray(minus), 2.0 * np.asarray(zero)
            ),
            "retained_configuration_count": int(selected.size),
        }
        if method == "pre":
            metrics["active_gaussian_loop_maximum_absolute"] = float(
                np.max(np.abs(full_vector(components, "gaussian_matter_loop")))
            )
            metrics["active_matter_response_loop_maximum_absolute"] = float(
                np.max(np.abs(full_vector(components, "matter_linear_uplift")))
            )
        else:
            metrics["active_gaussian_loop_maximum_absolute"] = float(
                np.max(np.abs(full_vector(components, "gaussian_matter_loop")))
            )
            metrics["active_matter_response_loop_maximum_absolute"] = max(
                float(np.max(np.abs(full_vector(components, "matter_one_loop_uplift")))),
                float(np.max(np.abs(full_vector(components, "physical_K1_matter_loop_response")))),
                float(np.max(np.abs(full_vector(components, "adaptive_matter_loop")))),
            )
        passed = bool(
            metrics["retained_configuration_count"] == NDATA
            and metrics["active_gaussian_loop_maximum_absolute"] == 0.0
            and metrics["active_matter_response_loop_maximum_absolute"] == 0.0
            and all(
                value < 2.0e-13
                for key, value in metrics.items()
                if key.endswith("relative_l2")
            )
        )
        metrics["pass"] = passed
        closure_rows[method] = metrics
        all_closure = all_closure and passed

    manifest = {
        "schema": "quijote-halo-z0p5-fnl250-pre-standard-marisa-tree-dm-kmax0p08-v1",
        "created_utc": utc_now(),
        "status": "pass" if all_closure else "fail",
        "scope": {
            "kmax_h_mpc": KMAX,
            "retained_configuration_count": NDATA,
            "methods": list(METHODS),
            "dm_perturbative_order": "tree in Gaussian and linear PNG response",
            "fixed_b1": FIXED_B1,
            "p": P_RECON,
        },
        "per_method_closure": closure_rows,
        "checks": {
            "all_closures_pass": all_closure,
            "simulation_or_reconstruction_ran": False,
            "old_kmax0p10_outputs_overwritten": False,
        },
        "inputs": {
            "tree10_profiles_initialization_and_comparison": bundle.recovery.artifact(TREE10_PROFILES),
            "pre_reference_initialization": bundle.recovery.artifact(PRE_REFERENCE),
            "matrix_summary": bundle.recovery.artifact(MATRIX_SUMMARY),
            **{f"{method}_matrix": bundle.recovery.artifact(path) for method, path in MATRICES.items()},
            "runner": bundle.recovery.artifact(Path(__file__).resolve()),
        },
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    bundle.recovery.atomic_json(OUTPUT_MANIFEST, manifest)
    if not all_closure:
        raise RuntimeError("kmax0.08 fully-tree closure failed")

    for target in TARGETS:
        for method in METHODS:
            context, _contract, samples, _fiducial, _covariance = make_context(
                bundle, method, target, 0.0
            )
            reference = reference_row(bundle, method, target, 0.0)
            record, local = bundle.helper.fit_gaussian_baseline(bundle.fixed, context, reference)
            record.update(
                {
                    "method": method,
                    "target_selection": target,
                    "target_realization_count": int(samples.shape[0]),
                    "retained_configuration_count": NDATA,
                    "gate_minimum_p_value": GAUSSIAN_GATE_MIN_P,
                }
            )
            record["gate_pass"] = bool(
                record["optimizer"]["valid"]
                and not record["optimizer"]["at_limit"]
                and record["gof_p_value_data_only"] >= GAUSSIAN_GATE_MIN_P
            )
            gates.append(record)
            prefix = f"{target}__{method}__fid0"
            for name, value in local.items():
                gate_arrays[f"{prefix}__{name}"] = np.asarray(value)
    gate_pass = bool(len(gates) == len(METHODS) * len(TARGETS) and all(record["gate_pass"] for record in gates))
    gate_payload = {
        "schema": "quijote-halo-z0p5-fnl250-tree-dm-kmax0p08-gaussian-gate-v1",
        "created_utc": utc_now(),
        "status": "pass" if gate_pass else "fail",
        "threshold": "data-only chi-square p >= 0.01 with fNL=0 and b1=1.93 fixed",
        "records": gates,
        "model_manifest": bundle.recovery.artifact(OUTPUT_MANIFEST),
        "runner": bundle.recovery.artifact(Path(__file__).resolve()),
    }
    bundle.recovery.atomic_json(OUTPUT_GATE, gate_payload)
    if not gate_pass:
        raise PermissionError("one or more kmax0.08 Gaussian baseline gates failed")
    return manifest, gates, gate_arrays


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "target_selection",
        "method",
        "input_fNL",
        "target_realization_count",
        "parameter",
        "centre",
        "plus_error",
        "minus_error",
        "lower",
        "upper",
        "fixed",
        "minos_valid",
        "data_chi2",
        "dof",
    )
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for parameter in ALL_PARAMETERS:
                interval = row["minos"][parameter]
                writer.writerow(
                    {
                        "target_selection": row["target_selection"],
                        "method": row["method"],
                        "input_fNL": row["input_fNL"],
                        "target_realization_count": row["context_contract"]["target_realization_count"],
                        "parameter": parameter,
                        "centre": row["map"][parameter],
                        "plus_error": interval["upper_error"],
                        "minus_error": -interval["lower_error"],
                        "lower": interval["lower"],
                        "upper": interval["upper"],
                        "fixed": parameter == "b1",
                        "minos_valid": interval["valid"],
                        "data_chi2": row["data_chi2"],
                        "dof": row["dof"],
                    }
                )
    temporary.replace(OUTPUT_CSV)


def plot_recovery(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharex=True, sharey=True)
    offsets = {"pre": -2.0, "standard": 0.0, "marisa": 2.0}
    for axis, target in zip(axes, TARGETS):
        for method in METHODS:
            local = sorted(
                (row for row in rows if row["target_selection"] == target and row["method"] == method),
                key=lambda row: row["input_fNL"],
            )
            truth = np.asarray([row["input_fNL"] for row in local])
            centre = np.asarray([row["map"]["fNL"] for row in local])
            minus = np.asarray([-row["minos"]["fNL"]["lower_error"] for row in local])
            plus = np.asarray([row["minos"]["fNL"]["upper_error"] for row in local])
            axis.errorbar(
                truth + offsets[method],
                centre,
                yerr=np.vstack((minus, plus)),
                color=METHOD_COLOURS[method],
                marker=METHOD_MARKERS[method],
                capsize=3,
                lw=1.1,
                label=METHOD_LABELS[method],
            )
        axis.plot((-125, 125), (-125, 125), "k--", lw=0.8)
        axis.set_title(target.replace("_", " "))
        axis.set_xlabel(r"input $f_{\rm NL}$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"recovered $\hat f_{\rm NL}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Fully-tree DM, fixed $b_1=1.93$, $k_{\max}=0.08\ h\,{\rm Mpc}^{-1}$")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_RECOVERY, metadata={"Title": "pre standard Marisa fully-tree kmax0.08 recovery"})
    plt.close(fig)


def plot_parameters(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.3), constrained_layout=True)
    offsets = {"pre": -2.0, "standard": 0.0, "marisa": 2.0}
    for axis, parameter in zip(axes.flat, ALL_PARAMETERS):
        for method in METHODS:
            local = sorted(
                (
                    row
                    for row in rows
                    if row["target_selection"] == "full_available" and row["method"] == method
                ),
                key=lambda row: row["input_fNL"],
            )
            truth = np.asarray([row["input_fNL"] for row in local])
            centre = np.asarray([row["map"][parameter] for row in local])
            minus = np.asarray([-row["minos"][parameter]["lower_error"] for row in local])
            plus = np.asarray([row["minos"][parameter]["upper_error"] for row in local])
            axis.errorbar(
                truth + offsets[method],
                centre,
                yerr=np.vstack((minus, plus)),
                color=METHOD_COLOURS[method],
                marker=METHOD_MARKERS[method],
                capsize=2.5,
                lw=1.0,
                label=METHOD_LABELS[method],
            )
        if parameter == "fNL":
            axis.plot(TRUTHS, TRUTHS, "k--", lw=0.8)
        elif parameter == "b1":
            axis.axhline(FIXED_B1, color="k", ls="--", lw=0.8)
            axis.set_ylim(1.90, 1.96)
        else:
            axis.axhline(0.0, color="0.45", ls="--", lw=0.8)
        axis.set(title=PARAMETER_LABELS[parameter], xlabel=r"input $f_{\rm NL}$")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(r"Fully-tree DM, full available, $k_{\max}=0.08$")
    fig.savefig(OUTPUT_PARAMETERS, metadata={"Title": "fully-tree kmax0.08 all parameters"})
    plt.close(fig)


def plot_bestfit(
    method: str,
    target: str,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> None:
    first = f"{target}__{method}__fid0"
    k_pair = np.asarray(arrays[f"{first}__k_pair"], dtype=np.float64)
    diagonal = np.flatnonzero(np.isclose(k_pair[:, 0], k_pair[:, 1], rtol=0.0, atol=3.0e-7))
    diagonal = diagonal[np.argsort(np.mean(k_pair[diagonal], axis=1))]
    k = np.mean(k_pair[diagonal], axis=1)
    scale = k**2
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.9),
        sharex=True,
        gridspec_kw={"height_ratios": (1.7, 1.0), "hspace": 0.08},
    )
    maximum = 0.0
    local_rows = sorted(
        (row for row in rows if row["method"] == method and row["target_selection"] == target),
        key=lambda row: row["input_fNL"],
    )
    for row in local_rows:
        truth = float(row["input_fNL"])
        prefix = f"{target}__{method}__{TRUTH_TAGS[truth]}"
        measured = arrays[f"{prefix}__target"][diagonal]
        model = arrays[f"{prefix}__profile_model"][diagonal]
        sigma = np.sqrt(np.diag(arrays[f"{prefix}__covariance_single_realization"]))[diagonal]
        pull = (measured - model) / sigma
        maximum = max(maximum, float(np.max(np.abs(pull))))
        top.errorbar(
            k,
            scale * measured,
            yerr=scale * sigma,
            fmt="o",
            color=TRUTH_COLOURS[truth],
            ms=5,
            capsize=2,
        )
        top.plot(k, scale * model, color=TRUTH_COLOURS[truth], lw=1.5)
        bottom.plot(k, pull, color=TRUTH_COLOURS[truth], marker="o", ms=4, lw=1.0)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    top.yaxis.set_major_formatter(formatter)
    top.set_ylabel(r"$k^2B_{000}(k,k)$")
    bottom.axhspan(-1.0, 1.0, color="0.92")
    bottom.axhline(0.0, color="0.35", lw=0.8)
    limit = max(1.5, 0.5 * math.ceil(2.2 * maximum))
    bottom.set(
        ylim=(-limit, limit),
        ylabel=r"$(\bar B-B_{\rm fit})/\sigma_{\rm one\ realization}$",
        xlabel=r"$k_1=k_2=k\ [h\,{\rm Mpc}^{-1}]$",
    )
    for axis in (top, bottom):
        axis.grid(alpha=0.2)
    top.legend(
        [Line2D([], [], color=TRUTH_COLOURS[truth], marker="o", lw=1.4) for truth in TRUTHS],
        [rf"input $f_{{\rm NL}}={truth:+.0f}$" for truth in TRUTHS],
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    fig.suptitle(f"{METHOD_LABELS[method]}, fully-tree DM, {target.replace('_', ' ')}, kmax=0.08")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.92)
    fig.savefig(OUTPUT_BESTFIT[(method, target)], metadata={"Title": f"{method} tree DM kmax0.08 best fit"})
    plt.close(fig)


def plot_kmax_comparison(rows: list[dict[str, Any]], tree10_rows: dict[tuple[str, str, float], dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.5), sharex=True, sharey=True)
    for row_index, method in enumerate(("standard", "marisa")):
        for column, target in enumerate(TARGETS):
            axis = axes[row_index, column]
            local08 = sorted(
                (row for row in rows if row["method"] == method and row["target_selection"] == target),
                key=lambda row: row["input_fNL"],
            )
            local10 = [tree10_rows[(target, method, truth)] for truth in TRUTHS]
            for offset, local, colour, marker, label in (
                (-1.2, local10, "0.45", "o", r"$k_{\max}=0.10$"),
                (1.2, local08, "#009E73", "s", r"$k_{\max}=0.08$"),
            ):
                truth = np.asarray([row["input_fNL"] for row in local])
                centre = np.asarray([row["map"]["fNL"] for row in local])
                minus = np.asarray([-row["minos"]["fNL"]["lower_error"] for row in local])
                plus = np.asarray([row["minos"]["fNL"]["upper_error"] for row in local])
                axis.errorbar(
                    truth + offset,
                    centre,
                    yerr=np.vstack((minus, plus)),
                    color=colour,
                    marker=marker,
                    capsize=3,
                    lw=1.1,
                    label=label,
                )
            axis.plot((-125, 125), (-125, 125), "k--", lw=0.8)
            axis.set_title(f"{method}, {target.replace('_', ' ')}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel(r"input $f_{\rm NL}$")
            if column == 0:
                axis.set_ylabel(r"recovered $\hat f_{\rm NL}$")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_COMPARISON, metadata={"Title": "fully-tree kmax0.08 versus kmax0.10"})
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Fully-tree DM kmax=0.08: pre, standard, and Marisa",
        "",
        "All methods use nine nonzero-global B000 configurations, fixed b1=1.93, truth-centred flat fNL prior +/-200, and their own Fid500 single-realization covariance /42.875 with the nine-dimensional Hartlap factor.",
        "",
        "## Gaussian gates",
        "",
        "| target | method | data chi2/dof | p | pass |",
        "|---|---|---:|---:|---:|",
    ]
    for gate in sorted(gates, key=lambda item: (item["target_selection"], item["method"])):
        lines.append(
            f"| {gate['target_selection']} | {gate['method']} | {gate['data_chi2']:.6g}/{gate['dof']} | "
            f"{gate['gof_p_value_data_only']:.6g} | {gate['gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Profiles",
            "",
            "| target | method | input fNL | fNL | b2 | gamma2 | Ashot | Bshot | chi2/dof |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def fmt(row: dict[str, Any], name: str) -> str:
        interval = row["minos"][name]
        return f"{row['map'][name]:.5g} (+{interval['upper_error']:.4g}/-{(-interval['lower_error']):.4g})"

    for row in sorted(rows, key=lambda item: (item["target_selection"], item["method"], item["input_fNL"])):
        lines.append(
            f"| {row['target_selection']} | {row['method']} | {row['input_fNL']:+.0f} | "
            f"{fmt(row, 'fNL')} | {fmt(row, 'b2')} | {fmt(row, 'gamma2')} | "
            f"{fmt(row, 'Ashot_residual')} | {fmt(row, 'Bshot_residual')} | "
            f"{row['data_chi2']:.6g}/{row['dof']} |"
        )
    lines.extend(["", f"Summary: `{json.dumps(summary, sort_keys=True)}`.", ""])
    temporary = OUTPUT_REPORT.with_name(f".{OUTPUT_REPORT.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(OUTPUT_REPORT)


def assert_pdf_only() -> None:
    raster = [
        path
        for root in (ANALYSIS, FIGURES)
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if raster:
        raise ValueError(f"raster outputs are forbidden: {raster}")


def run_profiles(bundle: SimpleNamespace) -> dict[str, Any]:
    manifest, gates, gate_arrays = closure_and_gates(bundle)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = dict(gate_arrays)
    for target in TARGETS:
        for method in METHODS:
            for truth in TRUTHS:
                context, contract, samples, _fiducial, covariance_full = make_context(
                    bundle, method, target, truth
                )
                reference = reference_row(bundle, method, target, truth)
                row, local = bundle.fixed.fit_case(context, reference)
                row = clean_row(row, method, truth)
                row["target_selection"] = target
                row["context_contract"] = {
                    **contract,
                    "method": method,
                    "target_selection": target,
                    "target_realization_count": int(samples.shape[0]),
                    "tree_dm_kmax0p08_manifest": str(OUTPUT_MANIFEST),
                }
                rows.append(row)
                prefix = f"{target}__{method}__{TRUTH_TAGS[truth]}"
                for name, value in local.items():
                    arrays[f"{prefix}__{name}"] = np.asarray(value)
                selected = np.asarray(context.indices, dtype=np.int64)
                arrays[f"{prefix}__covariance_single_realization"] = covariance_full[
                    np.ix_(selected, selected)
                ]
                arrays[f"{prefix}__covariance_fit_volume42p875"] = np.asarray(
                    context.covariance_single
                )
                nuisance = np.asarray([row["map"][name] for name in context.names])
                _zero, _parameters, components = context.evaluate_model(0.0, nuisance)
                for name in (
                    "gaussian_tree",
                    "gaussian_stochastic",
                    "gaussian_total",
                    "fixed_brec_linear_total",
                    "adaptive_tree",
                    "linear_total",
                    "removed_gaussian_matter_loop",
                ):
                    arrays[f"{prefix}__{name}"] = full_vector(components, name)[selected]
                removed_name = (
                    "removed_matter_linear_uplift"
                    if method == "pre"
                    else "removed_matter_one_loop_uplift"
                )
                arrays[f"{prefix}__removed_matter_response_loop"] = full_vector(
                    components, removed_name
                )[selected]
                print(
                    f"{target} {method} {truth:+.0f}: fNL={row['map']['fNL']:.6g} "
                    f"+{row['minos']['fNL']['upper_error']:.5g}"
                    f"/-{(-row['minos']['fNL']['lower_error']):.5g}",
                    flush=True,
                )

    summary = {
        "row_count": len(rows),
        "all_minos_valid": bool(all(row["all_minos_valid"] for row in rows)),
        "all_optimizers_valid": bool(all(row["optimizer"]["valid"] for row in rows)),
        "truth_inside_count": int(
            sum(row["fNL_recovery"]["truth_inside_delta_chi2_1"] for row in rows)
        ),
        "maximum_absolute_fNL_bias": max(
            abs(float(row["map"]["fNL"] - row["input_fNL"])) for row in rows
        ),
        "gaussian_gate_count": len(gates),
        "all_gaussian_gates_pass": bool(all(gate["gate_pass"] for gate in gates)),
        "minimum_gaussian_p": min(gate["gof_p_value_data_only"] for gate in gates),
    }
    if (
        len(rows) != len(METHODS) * len(TARGETS) * len(TRUTHS)
        or not summary["all_minos_valid"]
        or not summary["all_optimizers_valid"]
        or not summary["all_gaussian_gates_pass"]
    ):
        raise RuntimeError(f"kmax0.08 profile completion contract failed: {summary}")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    bundle.recovery.atomic_npz(OUTPUT_ARRAYS, arrays)
    plot_recovery(rows)
    plot_parameters(rows)
    for method in METHODS:
        for target in TARGETS:
            plot_bestfit(method, target, rows, arrays)
    plot_kmax_comparison(rows, bundle.tree10_rows)
    write_report(rows, gates, summary)
    assert_pdf_only()

    payload = {
        "schema": "quijote-halo-z0p5-fnl250-pre-standard-marisa-fully-tree-dm-kmax0p08-profiles-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "scope": {
            "kmax_h_mpc": KMAX,
            "retained_configuration_count": NDATA,
            "methods": list(METHODS),
            "fixed_b1": FIXED_B1,
            "p": P_RECON,
            "target_selections": list(TARGETS),
            "flat_fNL_prior": "[input-200,input+200]",
            "covariance": (
                "method/stage-specific Fid500 single-realization covariance /42.875 "
                "plus nine-dimensional Hartlap"
            ),
        },
        "model": {
            "dm_perturbative_order": "tree",
            "gaussian_matter_loop": False,
            "linear_PNG_matter_loop": False,
            "physical_K1_matter_loop_response": False,
            "marisa_adaptive_tree": True,
            "marisa_adaptive_matter_loop": False,
            "leading_residual_stochastic": True,
            "linear_fNL_only": True,
            "fNL_squared": False,
            "response_gain_parameter": False,
        },
        "parameters": {"free": list(FREE_PARAMETERS), "fixed": {"b1": FIXED_B1}},
        "summary": summary,
        "gaussian_baselines": gates,
        "rows": rows,
        "inputs": {
            "model_manifest": bundle.recovery.artifact(OUTPUT_MANIFEST),
            "gaussian_gate": bundle.recovery.artifact(OUTPUT_GATE),
            "tree10_profiles_initialization_and_comparison": bundle.recovery.artifact(
                TREE10_PROFILES
            ),
            "pre_reference_initialization": bundle.recovery.artifact(PRE_REFERENCE),
            "matrix_summary": bundle.recovery.artifact(MATRIX_SUMMARY),
            **{f"{method}_matrix": bundle.recovery.artifact(path) for method, path in MATRICES.items()},
            "runner": bundle.recovery.artifact(Path(__file__).resolve()),
        },
    }
    payload["outputs"] = {
        "csv": bundle.recovery.artifact(OUTPUT_CSV),
        "arrays": bundle.recovery.artifact(OUTPUT_ARRAYS),
        "recovery_pdf": bundle.recovery.artifact(OUTPUT_RECOVERY),
        "parameter_pdf": bundle.recovery.artifact(OUTPUT_PARAMETERS),
        "kmax_comparison_pdf": bundle.recovery.artifact(OUTPUT_COMPARISON),
        "bestfit_pdfs": {
            f"{method}_{target}": bundle.recovery.artifact(path)
            for (method, target), path in OUTPUT_BESTFIT.items()
        },
        "report": bundle.recovery.artifact(OUTPUT_REPORT),
    }
    bundle.recovery.atomic_json(OUTPUT_JSON, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "profiles"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_pdf_only()
    bundle = load_bundle()
    manifest, gates, _arrays = closure_and_gates(bundle)
    if args.stage == "preflight":
        print(
            json.dumps(
                {
                    "status": "pass",
                    "closure": manifest["status"],
                    "minimum_gaussian_p": min(gate["gof_p_value_data_only"] for gate in gates),
                },
                sort_keys=True,
            )
        )
        return
    payload = run_profiles(bundle)
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
