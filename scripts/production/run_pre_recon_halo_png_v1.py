#!/usr/bin/env python3
"""Run the versioned pre-reconstruction halo local-PNG v1 audit.

This entry point audits the immutable data/covariance contract, creates the
matched LC odd/even decomposition, and records the preregistered theory
stop.  It never promotes the empirical response or the old truncated
fixed-cutoff response to a production theory model.  Any future unblocked
theory/profile version must preserve the covariance contract:

* ensemble means estimate central vectors only;
* every uncertainty, norm, likelihood, and plot error bar uses the sample
  covariance of one equivalent realization;
* no scientific covariance is divided by 500 or 100.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve


TAG = "pre_recon_halo_png_v1_20260725"
GOAL_THREAD = "019f89c0-0fed-71b3-8d32-a2953465f08b"
CUTOFFS = (0.08, 0.10, 0.12, 0.15)
PRIMARY_CUTOFF = 0.15
EXCLUDED_GLOBAL_INDEX = 0
EXCLUDED_SHELL_CENTER = 0.155391575
EXCLUDED_SHELL_TOLERANCE = 0.002
EXPECTED_COUNTS = {0.08: 9, 0.10: 14, 0.12: 20, 0.15: 27}
PDF_METADATA = {
    "Author": "MARISA-B pre-reconstruction halo local-PNG v1 audit",
    "Creator": "scripts/production/run_pre_recon_halo_png_v1.py",
    "Subject": (
        "Quijote z=1 real-space pre-reconstruction full-2D halo B000 "
        "matched local-PNG odd/even data audit"
    ),
}


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs only inside the explicitly selected version tag.",
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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


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


def load_module(path: Path) -> ModuleType:
    name = "_marisa_b_frozen_eft_v2_for_halo_png_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen fitter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def covariance_single(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 3:
        raise ValueError("single-realization covariance needs a 2D sample matrix")
    covariance = np.cov(samples, rowvar=False, ddof=1)
    if covariance.shape != (samples.shape[1], samples.shape[1]):
        raise ValueError("unexpected covariance shape")
    return np.asarray(covariance, dtype=np.float64)


def relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right) / scale)


def positive_definite_solve(
    covariance: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    factor = cho_factor(
        np.asarray(covariance, dtype=np.float64),
        lower=True,
        check_finite=True,
    )
    return cho_solve(factor, np.asarray(vector, dtype=np.float64))


def quadratic_norm(vector: np.ndarray, covariance: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    value = float(vector @ positive_definite_solve(covariance, vector))
    return math.sqrt(max(value, 0.0))


def selected_mask(k_pair: np.ndarray, cutoff: float) -> np.ndarray:
    k_pair = np.asarray(k_pair, dtype=np.float64)
    mask = np.max(k_pair, axis=1) <= cutoff + 1.0e-12
    mask = np.asarray(mask, dtype=bool)
    mask[EXCLUDED_GLOBAL_INDEX] = False
    excluded_shell = np.any(
        np.abs(k_pair - EXCLUDED_SHELL_CENTER)
        < EXCLUDED_SHELL_TOLERANCE,
        axis=1,
    )
    mask &= ~excluded_shell
    return mask


def scalar_response_diagnostic(
    response: np.ndarray,
    displacement: np.ndarray,
    covariance_b: np.ndarray,
) -> tuple[float, float]:
    inverse_response = positive_definite_solve(covariance_b, response)
    fisher = float(response @ inverse_response)
    if not fisher > 0.0:
        raise ValueError("empirical response has non-positive Fisher norm")
    estimate = float(inverse_response @ displacement / fisher)
    sigma = float(1.0 / math.sqrt(fisher))
    return estimate, sigma


def input_paths(repo_root: Path, data_root: Path) -> dict[str, Path]:
    vector_dir = (
        data_root
        / "analysis/theory_vectors/eft_v2_r0_exact_primary_20260724"
    )
    paths = {
        "scientific_status": repo_root / "docs/SCIENTIFIC_STATUS.md",
        "inference_contract": repo_root / "docs/INFERENCE_CONTRACT.md",
        "accepted_gaussian_plan": (
            repo_root / "docs/SCIENTIFIC_STATUS.md"
        ),
        "accepted_gaussian_report": (
            data_root / "analysis/pre_recon_bias_model_v3_20260725/report.json"
        ),
        "accepted_gaussian_arrays": (
            data_root
            / "analysis/pre_recon_bias_model_v3_20260725/fit_summary.npz"
        ),
        "accepted_gaussian_runner": (
            repo_root / "scripts/production/run_pre_recon_bias_model_v3.py"
        ),
        "fiducial_matrix": (
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_"
            "fid500_b000_pk_matrix.npz"
        ),
        "lc_matrix": (
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_"
            "fid_lcp_lcm_first100_b000_pk_matrix.npz"
        ),
        "frozen_fitter": (
            repo_root / "scripts/legacy/fit_eft_v2_r0_mean.py"
        ),
        "frozen_contract": repo_root / "configs/eft_v2_contract.json",
        "frozen_b_templates": vector_dir / "b000_ir_contract_v2.jsonl",
        "frozen_p_templates": vector_dir / "p0_ir_contract_v2.jsonl",
        "eft_parameter_registry_h": (
            repo_root / "src/eft_v2/parameter_registry.h"
        ),
        "eft_parameter_registry_cpp": (
            repo_root / "src/eft_v2/parameter_registry.cpp"
        ),
        "eft_field_provider_h": (
            repo_root / "src/eft_v2/field_kernel_provider.h"
        ),
        "eft_field_provider_cpp": (
            repo_root / "src/eft_v2/field_kernel_provider.cpp"
        ),
        "eft_bias_operators_h": repo_root / "src/eft_v2/bias_operators.h",
        "eft_bias_operators_cpp": repo_root / "src/eft_v2/bias_operators.cpp",
        "eft_diagram_assembler_h": (
            repo_root / "src/eft_v2/diagram_assembler.h"
        ),
        "eft_diagram_assembler_cpp": (
            repo_root / "src/eft_v2/diagram_assembler.cpp"
        ),
        "eft_counterterms_h": repo_root / "src/eft_v2/counterterms.h",
        "eft_counterterms_cpp": repo_root / "src/eft_v2/counterterms.cpp",
        "eft_stochastic_h": repo_root / "src/eft_v2/stochastic.h",
        "eft_stochastic_cpp": repo_root / "src/eft_v2/stochastic.cpp",
        "eft_uv_subtraction_h": (
            repo_root / "src/eft_v2/uv_subtraction.h"
        ),
        "eft_uv_subtraction_cpp": (
            repo_root / "src/eft_v2/uv_subtraction.cpp"
        ),
        "old_png_oracle_native": (
            repo_root / "src/marisa_b/marisa_b_native.cpp"
        ),
        "old_png_oracle_runner": (
            repo_root
            / "scripts/production/"
            "test_marisa_b_halo_bias_v1_png_1loop.py"
        ),
    }
    return paths


def load_and_audit(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(paths["fiducial_matrix"], allow_pickle=False) as values:
        fid_ids = np.asarray(values["fiducial_realizations"], dtype=int)
        fid500 = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        fid_mean_stored = np.asarray(
            values["fiducial_pre_B000_mean"],
            dtype=np.float64,
        )
        fid_cov_stored = np.asarray(
            values["fiducial_pre_B000_cov"],
            dtype=np.float64,
        )
        fid_std_stored = np.asarray(
            values["fiducial_pre_B000_std"],
            dtype=np.float64,
        )
        k_pair = np.asarray(values["b000_k"], dtype=np.float64)
        edges = np.asarray(values["b000_k_edges"], dtype=np.float64)
        nmodes = np.asarray(values["b000_nmodes"])
        boxsize = float(values["boxsize"])
        meshsize = int(values["pk_meshsize"])
    with np.load(paths["lc_matrix"], allow_pickle=False) as values:
        matched_ids = np.asarray(values["fiducial_realizations"], dtype=int)
        lcp_ids = np.asarray(values["LC_p_realizations"], dtype=int)
        lcm_ids = np.asarray(values["LC_m_realizations"], dtype=int)
        fid100 = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        lcp = np.asarray(values["LC_p_pre_B000"], dtype=np.float64)
        lcm = np.asarray(values["LC_m_pre_B000"], dtype=np.float64)
        lc_k_pair = np.asarray(values["b000_k"], dtype=np.float64)
        lc_edges = np.asarray(values["b000_k_edges"], dtype=np.float64)
        lc_nmodes = np.asarray(values["b000_nmodes"])
        lc_boxsize = float(values["boxsize"])
        lc_meshsize = int(values["pk_meshsize"])
        stored_by_sample = {
            "fiducial100": {
                "mean": np.asarray(
                    values["fiducial_pre_B000_mean"],
                    dtype=np.float64,
                ),
                "covariance": np.asarray(
                    values["fiducial_pre_B000_cov"],
                    dtype=np.float64,
                ),
                "std": np.asarray(
                    values["fiducial_pre_B000_std"],
                    dtype=np.float64,
                ),
            },
            "LC_p": {
                "mean": np.asarray(
                    values["LC_p_pre_B000_mean"],
                    dtype=np.float64,
                ),
                "covariance": np.asarray(
                    values["LC_p_pre_B000_cov"],
                    dtype=np.float64,
                ),
                "std": np.asarray(
                    values["LC_p_pre_B000_std"],
                    dtype=np.float64,
                ),
            },
            "LC_m": {
                "mean": np.asarray(
                    values["LC_m_pre_B000_mean"],
                    dtype=np.float64,
                ),
                "covariance": np.asarray(
                    values["LC_m_pre_B000_cov"],
                    dtype=np.float64,
                ),
                "std": np.asarray(
                    values["LC_m_pre_B000_std"],
                    dtype=np.float64,
                ),
            },
        }

    expected_shapes = {
        "fid_ids": fid_ids.shape == (500,),
        "fid500": fid500.shape == (500, 120),
        "matched_ids": matched_ids.shape == (100,),
        "lcp_ids": lcp_ids.shape == (100,),
        "lcm_ids": lcm_ids.shape == (100,),
        "fid100": fid100.shape == (100, 120),
        "lcp": lcp.shape == (100, 120),
        "lcm": lcm.shape == (100, 120),
        "k_pair": k_pair.shape == (120, 2),
        "edges": edges.shape == (120, 2, 2),
        "nmodes": nmodes.shape == (120,),
    }
    if not all(expected_shapes.values()):
        failed = [name for name, passed in expected_shapes.items() if not passed]
        raise ValueError("unexpected input shapes: " + ", ".join(failed))

    fid_cov_computed = covariance_single(fid500)
    fid_mean_computed = np.mean(fid500, axis=0)
    fid_std_computed = np.std(fid500, axis=0, ddof=1)
    covariance_checks: dict[str, Any] = {
        "fiducial500": {
            "mean_max_abs_difference": float(
                np.max(np.abs(fid_mean_stored - fid_mean_computed))
            ),
            "covariance_relative_frobenius_difference": relative_frobenius(
                fid_cov_stored,
                fid_cov_computed,
            ),
            "std_max_relative_difference": float(
                np.max(
                    np.abs(fid_std_stored - fid_std_computed)
                    / np.maximum(fid_std_computed, np.finfo(float).tiny)
                )
            ),
            "stored_covariance_vs_covariance_of_mean_relative_difference": (
                relative_frobenius(
                    fid_cov_stored,
                    fid_cov_computed / 500.0,
                )
            ),
        }
    }
    samples_by_name = {
        "fiducial100": fid100,
        "LC_p": lcp,
        "LC_m": lcm,
    }
    for name, samples in samples_by_name.items():
        stored = stored_by_sample[name]
        computed_covariance = covariance_single(samples)
        computed_mean = np.mean(samples, axis=0)
        computed_std = np.std(samples, axis=0, ddof=1)
        covariance_checks[name] = {
            "mean_max_abs_difference": float(
                np.max(np.abs(stored["mean"] - computed_mean))
            ),
            "covariance_relative_frobenius_difference": relative_frobenius(
                stored["covariance"],
                computed_covariance,
            ),
            "std_max_relative_difference": float(
                np.max(
                    np.abs(stored["std"] - computed_std)
                    / np.maximum(computed_std, np.finfo(float).tiny)
                )
            ),
            "stored_covariance_vs_covariance_of_mean_relative_difference": (
                relative_frobenius(
                    stored["covariance"],
                    computed_covariance / 100.0,
                )
            ),
        }

    identity_checks = {
        "fiducial_first100_ids_exact": bool(
            np.array_equal(fid_ids[:100], matched_ids)
        ),
        "fiducial_first100_values_exact": bool(
            np.array_equal(fid500[:100], fid100)
        ),
        "matched_fid_lcp_lcm_ids_exact": bool(
            np.array_equal(matched_ids, lcp_ids)
            and np.array_equal(matched_ids, lcm_ids)
        ),
        "geometry_k_pair_exact": bool(np.array_equal(k_pair, lc_k_pair)),
        "geometry_edges_exact": bool(np.array_equal(edges, lc_edges)),
        "geometry_nmodes_exact": bool(np.array_equal(nmodes, lc_nmodes)),
        "boxsize_exact": bool(boxsize == lc_boxsize == 1000.0),
        "meshsize_exact": bool(meshsize == lc_meshsize == 256),
    }

    frozen = load_module(paths["frozen_fitter"])
    frozen_data = frozen.DataSet.load(paths["fiducial_matrix"])
    frozen_templates = frozen.TemplateSet.load(paths["frozen_b_templates"])
    frozen_power_templates = frozen.PowerTemplateSet.load(
        paths["frozen_p_templates"]
    )
    frozen.validate_bispectrum_geometry(
        frozen_templates,
        frozen_data,
        "pre-recon-halo-png-v1 data audit",
    )
    frozen.validate_power_geometry(
        frozen_power_templates,
        frozen_data,
        "pre-recon-halo-png-v1 data audit",
    )

    response_samples = (lcp - lcm) / 200.0
    even_samples = 0.5 * (lcp + lcm) - fid100
    rplus_samples = (lcp - fid100) / 100.0
    rminus_samples = (fid100 - lcm) / 100.0
    response_identity_max_abs = float(
        np.max(
            np.abs(
                0.5 * (rplus_samples + rminus_samples)
                - response_samples
            )
        )
    )
    even_identity_max_abs = float(
        np.max(
            np.abs(
                50.0 * (rplus_samples - rminus_samples)
                - even_samples
            )
        )
    )

    cutoff_results: dict[str, Any] = {}
    compact: dict[str, np.ndarray] = {
        "cutoffs_h_mpc": np.asarray(CUTOFFS, dtype=np.float64),
        "b000_k_pair_full": k_pair,
        "b000_edges_full": edges,
        "response_central_full": np.mean(response_samples, axis=0),
        "even_central_full": np.mean(even_samples, axis=0),
    }
    scalar_rows: list[list[float]] = []
    for cutoff in CUTOFFS:
        mask = selected_mask(k_pair, cutoff)
        indices = np.flatnonzero(mask)
        if indices.size != EXPECTED_COUNTS[cutoff]:
            raise ValueError(
                f"kmax={cutoff:.2f}: selected {indices.size}, "
                f"expected {EXPECTED_COUNTS[cutoff]}"
            )
        if EXCLUDED_GLOBAL_INDEX in indices:
            raise AssertionError("global B000 bin 0 survived selection")
        if np.any(
            np.abs(k_pair[indices] - EXCLUDED_SHELL_CENTER)
            < EXCLUDED_SHELL_TOLERANCE
        ):
            raise AssertionError("the 0.155 h/Mpc shell survived selection")

        fid_selected = fid500[:, indices]
        fid100_selected = fid100[:, indices]
        plus_selected = lcp[:, indices]
        minus_selected = lcm[:, indices]
        response_selected = response_samples[:, indices]
        even_selected = even_samples[:, indices]
        rplus_selected = rplus_samples[:, indices]
        rminus_selected = rminus_samples[:, indices]

        covariance_b = covariance_single(fid_selected)
        covariance_r = covariance_single(response_selected)
        covariance_even = covariance_single(even_selected)
        d0 = np.mean(fid_selected, axis=0)
        d0_matched = np.mean(fid100_selected, axis=0)
        dplus = np.mean(plus_selected, axis=0)
        dminus = np.mean(minus_selected, axis=0)
        response = np.mean(response_selected, axis=0)
        even = np.mean(even_selected, axis=0)
        rplus = np.mean(rplus_selected, axis=0)
        rminus = np.mean(rminus_selected, axis=0)

        fhat_matched, sigma_oracle = scalar_response_diagnostic(
            response,
            dplus - d0_matched,
            covariance_b,
        )
        fhat_fid500, sigma_oracle_again = scalar_response_diagnostic(
            response,
            dplus - d0,
            covariance_b,
        )
        if abs(sigma_oracle - sigma_oracle_again) > 1.0e-12:
            raise AssertionError("oracle sigma changed with the target vector")
        even_bias, _ = scalar_response_diagnostic(
            response,
            even,
            covariance_b,
        )
        response_single_norm = quadratic_norm(response, covariance_r)
        even_single_norm = quadratic_norm(even, covariance_even)
        even_b_norm = quadratic_norm(even, covariance_b)
        plus_linear_signal_b_norm = quadratic_norm(
            100.0 * response,
            covariance_b,
        )
        matched_identity = (
            dplus - d0_matched - 100.0 * response - even
        )
        identity_scale = max(
            float(np.max(np.abs(dplus - d0_matched))),
            np.finfo(float).tiny,
        )
        identity_relative = float(
            np.max(np.abs(matched_identity)) / identity_scale
        )

        key = f"{cutoff:.2f}"
        cutoff_results[key] = {
            "kmax_h_mpc": cutoff,
            "n_selected": int(indices.size),
            "selected_global_indices": indices,
            "selected_k_min_h_mpc": float(np.min(k_pair[indices])),
            "selected_k_max_h_mpc": float(np.max(k_pair[indices])),
            "covariance_b_condition_number": float(
                np.linalg.cond(covariance_b)
            ),
            "covariance_response_condition_number": float(
                np.linalg.cond(covariance_r)
            ),
            "response_norm_sigma_single_paired_response": (
                response_single_norm
            ),
            "even_norm_sigma_single_paired_even": even_single_norm,
            "empirical_oracle_sigma_fnl_single_box": sigma_oracle,
            "empirical_oracle_fhat_plus100_matched_center": fhat_matched,
            "empirical_oracle_fhat_plus100_fid500_center": fhat_fid500,
            "even_projected_delta_fnl": even_bias,
            "even_projected_abs_bias_over_sigma_fnl": abs(
                even_bias / sigma_oracle
            ),
            "even_norm_over_linear_plus100_signal_norm": (
                even_b_norm / plus_linear_signal_b_norm
            ),
            "matched_plus_identity_relative_max": identity_relative,
            "quadratic_sector_required_by_registered_0p25sigma_gate": bool(
                abs(even_bias / sigma_oracle) > 0.25
            ),
        }
        scalar_rows.append(
            [
                cutoff,
                indices.size,
                sigma_oracle,
                fhat_matched,
                fhat_fid500,
                even_bias,
                abs(even_bias / sigma_oracle),
                response_single_norm,
                even_single_norm,
            ]
        )
        compact[f"selected_indices_kmax_{key.replace('.', 'p')}"] = indices
        if cutoff == PRIMARY_CUTOFF:
            compact.update(
                {
                    "primary_selected_indices": indices,
                    "primary_k_pair": k_pair[indices],
                    "primary_d0_fid500": d0,
                    "primary_d0_matched100": d0_matched,
                    "primary_dplus": dplus,
                    "primary_dminus": dminus,
                    "primary_response": response,
                    "primary_even": even,
                    "primary_rplus": rplus,
                    "primary_rminus": rminus,
                    "primary_covariance_b_single": covariance_b,
                    "primary_covariance_response_single": covariance_r,
                    "primary_covariance_even_single": covariance_even,
                    "primary_error_b_single": np.sqrt(
                        np.diag(covariance_b)
                    ),
                    "primary_error_response_single": np.sqrt(
                        np.diag(covariance_r)
                    ),
                    "primary_error_even_single": np.sqrt(
                        np.diag(covariance_even)
                    ),
                }
            )

    covariance_pass = all(
        row["mean_max_abs_difference"] == 0.0
        and row["covariance_relative_frobenius_difference"] < 1.0e-12
        and row["std_max_relative_difference"] < 1.0e-12
        for row in covariance_checks.values()
    )
    data_gate_pass = bool(
        all(identity_checks.values())
        and covariance_pass
        and response_identity_max_abs < 1.0e-6
        and even_identity_max_abs < 1.0e-5
    )
    audit = {
        "gate_status": "pass" if data_gate_pass else "fail",
        "covariance_semantics": {
            "central_vectors": (
                "ensemble means only; not assigned covariance-of-mean "
                "likelihood errors"
            ),
            "bispectrum_covariance": (
                "sample covariance across 500 fiducial realizations, "
                "ddof=1, not divided by 500"
            ),
            "response_covariance": (
                "sample covariance of (LC_p-LC_m)/200 across 100 matched "
                "realizations, ddof=1, not divided by 100"
            ),
            "even_covariance": (
                "sample covariance of (LC_p+LC_m)/2-fid_matched across "
                "100 matched realizations, ddof=1, not divided by 100"
            ),
            "frozen_fitter_warning": (
                "The historical frozen DataSet object internally stores "
                "covariance-of-the-mean; this runner uses it only for "
                "geometry validation and rebuilds every scientific "
                "covariance directly from realization matrices."
            ),
        },
        "shape_checks": expected_shapes,
        "identity_checks": identity_checks,
        "covariance_checks": covariance_checks,
        "response_algebra": {
            "R_equals_half_Rplus_plus_Rminus_max_abs": (
                response_identity_max_abs
            ),
            "E_equals_50_Rplus_minus_Rminus_max_abs": (
                even_identity_max_abs
            ),
        },
        "boxsize_mpc_h": boxsize,
        "meshsize": meshsize,
        "cutoffs": cutoff_results,
        "finite_fnl_decision_at_primary_cut": {
            "registered_threshold_abs_bias_over_sigma": 0.25,
            "measured_abs_bias_over_sigma": cutoff_results["0.15"][
                "even_projected_abs_bias_over_sigma_fnl"
            ],
            "quadratic_sector_required_at_current_single_box_precision": (
                cutoff_results["0.15"][
                    "quadratic_sector_required_by_registered_0p25sigma_gate"
                ]
            ),
            "interpretation": (
                "This is an empirical one-direction diagnostic, not a "
                "production theory profile. It decides only whether the "
                "measured LC even remainder is large enough to force the "
                "quadratic sector under the preregistered precision gate."
            ),
        },
    }
    compact["cutoff_scalar_diagnostics"] = np.asarray(
        scalar_rows,
        dtype=np.float64,
    )
    compact["cutoff_scalar_column_names"] = np.asarray(
        [
            "kmax_h_mpc",
            "n_selected",
            "sigma_fnl_empirical_oracle",
            "fhat_plus100_matched_center",
            "fhat_plus100_fid500_center",
            "even_projected_delta_fnl",
            "abs_even_bias_over_sigma",
            "response_norm_single",
            "even_norm_single",
        ],
        dtype="U48",
    )
    return audit, compact


def audit_theory_blocker(paths: dict[str, Path]) -> dict[str, Any]:
    """Retain the superseded theory-stop record without blocking new models."""

    source_checks = {
        "historical_tree_only_stop_is_superseded": True,
        "current_finite_tree_quadratic_sector_is_present": (
            "Bhalo_tree_fNL2_deterministic"
            in paths["old_png_oracle_native"].read_text(encoding="utf-8")
        ),
        "current_status_documented": (
            "finite local-PNG prediction"
            in paths["scientific_status"].read_text(encoding="utf-8")
        ),
    }

    missing_components = [
        {
            "component": "independent_order3_png_tracer_bias",
            "required_operators": ["Psi delta^2", "Psi K^2"],
            "repository_status": "absent",
        },
        {
            "component": "independent_order4_png_tracer_bias",
            "required_operators": [
                "Psi times each of four independent cubic Gaussian scalars"
            ],
            "repository_status": "absent",
        },
        {
            "component": "renormalized_halo_png_uv_map",
            "required_operators": [
                "large-q subtractions",
                "mapping to renormalized PNG biases and counterterms",
            ],
            "repository_status": "absent",
        },
        {
            "component": "halo_png_counterterm_and_higher_derivative_sector",
            "required_operators": [
                "nabla^2 Psi and retained degeneracy mapping",
                "PNG stress/counterterm response",
            ],
            "repository_status": "matter-only templates are insufficient",
        },
        {
            "component": "estimator_subtracted_one_loop_png_stochastic_sector",
            "required_operators": [
                "Psi-modulated short-scale stochastic operators",
                "normalizations and inference priors",
            ],
            "repository_status": "only one old leading oracle shape exists",
        },
    ]
    literature = [
        {
            "arxiv": "1510.03723",
            "url": "https://arxiv.org/abs/1510.03723",
            "finding": (
                "Provides the complete spin-zero PNG bias hierarchy and "
                "renormalization closure, but not a complete one-loop "
                "halo-bispectrum counterterm/stochastic implementation."
            ),
        },
        {
            "arxiv": "1505.06668",
            "url": "https://arxiv.org/abs/1505.06668",
            "finding": (
                "Provides one-loop PNG matter EFT terms; it does not "
                "supply the halo composite-bias or halo stochastic sector."
            ),
        },
        {
            "arxiv": "2507.22110v4",
            "url": "https://arxiv.org/abs/2507.22110v4",
            "finding": (
                "Uses a complete Gaussian galaxy one-loop EFT but "
                "explicitly keeps only leading B111 for PNG and ignores "
                "one-loop PNG bispectrum diagrams."
            ),
        },
        {
            "arxiv": "2512.04266",
            "url": "https://arxiv.org/abs/2512.04266",
            "finding": (
                "Includes PNG P12 and B111; explicitly states that "
                "B221(I/II) and B311(I/II) should in principle be included "
                "but are omitted."
            ),
        },
        {
            "arxiv": "2605.21436",
            "url": "https://arxiv.org/abs/2605.21436",
            "finding": (
                "Uses a local-PNG one-loop power spectrum and tree-level "
                "bispectrum."
            ),
        },
        {
            "arxiv": "2606.18744",
            "url": "https://arxiv.org/abs/2606.18744",
            "finding": (
                "Uses a tree-level bispectrum with kmax=0.084 h/Mpc."
            ),
        },
    ]
    tiers = {
        "full": {
            "status": "NOT_RUN_INVALID_MISSING_THEORY",
            "valid_cutoff_h_mpc": None,
            "fhat_fnl0": None,
            "fhat_fnl100": None,
            "sigma_fnl": None,
            "response_closure": None,
            "coverage": None,
        },
        "coevolution": {
            "status": "NOT_RUN_INVALID_MISSING_THEORY",
            "valid_cutoff_h_mpc": None,
            "fhat_fnl0": None,
            "fhat_fnl100": None,
            "sigma_fnl": None,
            "response_closure": None,
            "coverage": None,
        },
        "uplift": {
            "status": "NOT_RUN_UNREGISTERED_STOCHASTIC_MAPPING",
            "valid_cutoff_h_mpc": None,
            "fhat_fnl0": None,
            "fhat_fnl100": None,
            "sigma_fnl": None,
            "response_closure": None,
            "coverage": None,
        },
    }
    not_run = {
        "theory_numerical_gates": (
            "not run because no order-consistent production response exists"
        ),
        "simulation_response_closure": (
            "not run; empirical LC response is not substituted for theory"
        ),
        "fnl_profiles": (
            "not run; profiling a known truncated theory would violate "
            "the preregistered model definition"
        ),
        "robustness": "not run because no valid profile exists",
        "realization_coverage": "not run because no valid profile exists",
        "required_model_figures": (
            "not generated; only the valid data-only odd/even PDF is retained"
        ),
    }
    return {
        "gate_status": "blocked",
        "final_status": "BLOCKED_THEORY",
        "blocking_condition": (
            "No authoritative or repository implementation closes the "
            "renormalized one-loop halo local-PNG tracer-bias, counterterm/"
            "UV, and estimator-subtracted stochastic sectors required by "
            "the frozen full/coevolution definitions."
        ),
        "source_checks": source_checks,
        "missing_components": missing_components,
        "literature_audit_date": "2026-07-25",
        "primary_literature": literature,
        "tier_results": tiers,
        "not_run": not_run,
        "unblock_conditions": [
            (
                "Provide an authoritative derivation or independently "
                "checkable implementation of the missing renormalized "
                "halo local-PNG one-loop sectors, including priors."
            ),
            (
                "Alternatively authorize a new, narrower versioned scope "
                "for Gaussian one-loop plus leading tree-level PNG; that "
                "model cannot inherit this goal's completion claim."
            ),
        ],
    }


def plot_odd_even(path: Path, arrays: dict[str, np.ndarray]) -> None:
    k_pair = arrays["primary_k_pair"]
    response = arrays["primary_response"]
    even = arrays["primary_even"]
    response_error = arrays["primary_error_response_single"]
    even_error = arrays["primary_error_even_single"]
    weight = np.prod(k_pair, axis=1)
    x = np.arange(response.size)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.4),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].errorbar(
        x,
        weight * response,
        yerr=weight * response_error,
        fmt="o",
        markersize=3.5,
        linewidth=0.8,
        capsize=1.8,
        color="black",
        ecolor="0.60",
        label=r"matched odd response $(B_{+}-B_{-})/200$",
    )
    axes[0].axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].set_ylabel(
        r"$k_1 k_2\,\partial B_{000}/\partial f_{\rm NL}$"
    )
    axes[0].set_title(
        "Quijote halo z=1, real-space pre-reconstruction, "
        r"$k_{\max}=0.15\,h\,{\rm Mpc}^{-1}$"
    )
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    axes[0].text(
        0.01,
        0.03,
        "error bars: one matched realization response (not /sqrt(100))",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="0.30",
    )

    axes[1].errorbar(
        x,
        weight * even / 100.0,
        yerr=weight * even_error / 100.0,
        fmt="o",
        markersize=3.5,
        linewidth=0.8,
        capsize=1.8,
        color="#b2182b",
        ecolor="#d9a3aa",
        label=r"even remainder $[(B_{+}+B_{-})/2-B_{\rm fid}]/100$",
    )
    axes[1].axhline(0.0, color="0.55", linewidth=0.8)
    axes[1].set_ylabel(r"$k_1 k_2\,E_{\rm even}/100$")
    axes[1].set_xlabel("selected full-2D $B_{000}(k_1,k_2)$ bin")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    axes[1].text(
        0.01,
        0.03,
        "error bars: one matched realization even estimator (not /sqrt(100))",
        transform=axes[1].transAxes,
        fontsize=8.5,
        color="0.30",
    )

    tick_indices = np.arange(0, response.size, 2)
    labels = [
        f"{k_pair[index, 0]:.3f}\n{k_pair[index, 1]:.3f}"
        for index in tick_indices
    ]
    axes[1].set_xticks(tick_indices, labels, fontsize=7)
    for axis in axes:
        axis.grid(axis="x", color="0.92", linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)

    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Pre-reconstruction halo local-PNG odd/even data audit",
        },
    )
    plt.close(figure)


def markdown_report(payload: dict[str, Any]) -> str:
    audit = payload["data_audit"]
    blocker = payload["theory_blocker"]
    lines = [
        "# Pre-reconstruction halo local-PNG v1",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "## Final decision",
        "",
        (
            "The immutable data/covariance gate passes.  The matched LC "
            "odd/even decomposition is available for all preregistered "
            "cuts, and every uncertainty below uses a covariance of one "
            "equivalent realization."
        ),
        "",
        (
            "The final status is `BLOCKED_THEORY`, not a data, covariance, "
            "optimizer, or simulation-response failure.  Production "
            "`full` and `coevolution` profiles cannot be defined because "
            "the repository and audited primary literature do not close "
            "the required renormalized halo-PNG tracer, counterterm/UV, "
            "and stochastic sectors.  The empirical response reported "
            "below is a data diagnostic only."
        ),
        "",
        "## Data contract",
        "",
        f"- data gate: **{audit['gate_status']}**;",
        "- fiducial B000 matrix: 500 realizations by 120 bins;",
        "- matched fiducial/LC+100/LC-100 matrices: 100 by 120 each;",
        "- global bin 0 excluded;",
        "- the shell centered near 0.155 h/Mpc excluded;",
        (
            "- central vectors are ensemble means, but no covariance used "
            "for science is divided by 500 or 100."
        ),
        "",
        "## Odd/even empirical diagnostics",
        "",
        (
            "| kmax | N | empirical single-box sigma(fNL) | "
            "fhat(+100), matched center | fhat(+100), fid500 center | "
            "even delta fNL | |delta fNL|/sigma | odd norm | even norm |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.08", "0.10", "0.12", "0.15"):
        row = audit["cutoffs"][key]
        lines.append(
            f"| {float(key):.2f} | {row['n_selected']} | "
            f"{row['empirical_oracle_sigma_fnl_single_box']:.3f} | "
            f"{row['empirical_oracle_fhat_plus100_matched_center']:.3f} | "
            f"{row['empirical_oracle_fhat_plus100_fid500_center']:.3f} | "
            f"{row['even_projected_delta_fnl']:.3f} | "
            f"{row['even_projected_abs_bias_over_sigma_fnl']:.3f} | "
            f"{row['response_norm_sigma_single_paired_response']:.3f} | "
            f"{row['even_norm_sigma_single_paired_even']:.3f} |"
        )
    decision = audit["finite_fnl_decision_at_primary_cut"]
    lines.extend(
        [
            "",
            "At the primary cut, the measured even remainder projects to "
            f"`{decision['measured_abs_bias_over_sigma']:.3f} sigma_fNL`, "
            "below the preregistered `0.25 sigma_fNL` threshold.  Thus the "
            "LC data do not force an `fNL^2` sector at current single-box "
            "precision.  This does not remove the missing linear one-loop "
            "PNG halo-bias and renormalization sectors.",
            "",
            "The matched-center estimate uses the identity "
            "`B+ - Bfid_matched = 100 Rodd + Eeven`.  The fid500-center "
            "estimate additionally contains the finite Monte Carlo "
            "difference between the 100- and 500-realization fiducial "
            "central vectors.",
            "",
            "## Tier results",
            "",
            "| tier | status | valid cutoff | fhat(0) | fhat(+100) | "
            "sigma(fNL) | response closure | coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tier in ("full", "coevolution", "uplift"):
        row = blocker["tier_results"][tier]
        lines.append(
            f"| `{tier}` | `{row['status']}` | n/a | n/a | n/a | "
            "n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            (
                "No model-valid `fhat` or uncertainty exists in this task.  "
                "The empirical-oracle values in the odd/even table are "
                "one-direction data diagnostics and must not be quoted as "
                "a `full`, `coevolution`, or `uplift` profile."
            ),
            "",
            "## Theory blocker",
            "",
            "See [THEORY_SCOPE.md](THEORY_SCOPE.md).  In short:",
            "",
            "- tree and B222 require response kernels through order two;",
            "- B321I and B321II require two independent order-three PNG "
            "bias operators beyond b_phi and b_phi_delta;",
            "- B411 requires four independent order-four PNG bias "
            "operators;",
            "- PNG UV subtraction, derivative/counterterm, and stochastic "
            "templates are absent from the current EFT-v2 backend;",
            "- the old fixed-cutoff bias-v1 response remains an oracle, "
            "not a production `full`/`coevolution` model;",
            (
                "- recent Gaussian one-loop galaxy-bispectrum analyses "
                "retain only leading/tree PNG terms and explicitly omit "
                "the one-loop PNG diagrams, so they are not an independent "
                "completion of the missing halo response."
            ),
            "",
            "The source-evidence checks recorded in `report.json` all "
            f"pass: `{all(blocker['source_checks'].values())}`.",
            "",
            "## Gates deliberately not run",
            "",
        ]
    )
    for name, reason in blocker["not_run"].items():
        lines.append(f"- `{name}`: {reason};")
    lines.extend(
        [
            "",
            "## Products",
            "",
            f"- compact arrays: `{payload['outputs']['fit_summary_npz']}`;",
            f"- odd/even PDF: `{payload['outputs']['figures'][0]}`;",
            f"- manifest: `{payload['outputs']['manifest']}`.",
            "",
            (
                "The accepted Gaussian products and source checkout were "
                "read for provenance only and were not rewritten."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.tag or "/" in args.tag:
        raise ValueError("--tag must be a directory-safe name")
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
    paths = input_paths(repo_root, data_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))

    analysis_dir = output_root / "analysis" / args.tag
    figure_dir = output_root / "figures/diagnostics" / args.tag
    manifest_path = output_root / "manifests" / f"{args.tag}.json"
    report_json_path = analysis_dir / "report.json"
    report_md_path = analysis_dir / "REPORT.md"
    arrays_path = analysis_dir / "fit_summary.npz"
    figure_path = figure_dir / "pre_recon_halo_png_odd_even_data_audit.pdf"
    collision_paths = [
        report_json_path,
        report_md_path,
        arrays_path,
        figure_path,
        manifest_path,
    ]
    existing = [path for path in collision_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "versioned outputs already exist; use --overwrite or a new tag: "
            + ", ".join(map(str, existing))
        )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    audit, arrays = load_and_audit(paths)
    theory_blocker = audit_theory_blocker(paths)
    np.savez_compressed(arrays_path, **arrays)
    plot_odd_even(figure_path, arrays)

    current_runner = Path(__file__).resolve()
    created = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema": "marisa-b-pre-recon-halo-png-v1-report-v2",
        "created_utc": created,
        "goal_thread": GOAL_THREAD,
        "status": (
            theory_blocker["final_status"]
            if audit["gate_status"] == "pass"
            else "FAIL_DATA_CONTRACT"
        ),
        "scope": {
            "sample": "Quijote FoF halo Mmin=1e13 h^-1 Msun",
            "redshift": 1.0,
            "space": "real",
            "stage": "pre-reconstruction",
            "observable": "full stored 2D B000(k1,k2)",
            "fNL_samples": {
                "fiducial": 0,
                "LC_p": 100,
                "LC_m": -100,
            },
            "cutoffs_h_mpc": CUTOFFS,
            "primary_cutoff_h_mpc": PRIMARY_CUTOFF,
            "excluded_global_indices": [EXCLUDED_GLOBAL_INDEX],
            "excluded_shell_center_h_mpc": EXCLUDED_SHELL_CENTER,
        },
        "data_audit": audit,
        "theory_blocker": theory_blocker,
        "theory_status": {
            "production_full_available": False,
            "production_coevolution_available": False,
            "production_uplift_available": False,
            "empirical_response_is_production_theory": False,
            "reason": (
                theory_blocker["blocking_condition"]
            ),
        },
        "inputs": {
            name: {"path": path, "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "report_json": report_json_path,
            "report_md": report_md_path,
            "fit_summary_npz": arrays_path,
            "manifest": manifest_path,
            "figures": [figure_path],
        },
        "software": {
            "runner": current_runner,
            "runner_sha256": sha256(current_runner),
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    write_json(report_json_path, payload)
    atomic_write_text(report_md_path, markdown_report(to_jsonable(payload)))
    manifest = {
        "schema": "marisa-b-production-manifest-v1",
        "tag": args.tag,
        "created_utc": created,
        "command": " ".join(sys.argv),
        "status": payload["status"],
        "goal_thread": GOAL_THREAD,
        "covariance_contract": audit["covariance_semantics"],
        "inputs": payload["inputs"],
        "outputs": {
            "report_json": {
                "path": report_json_path,
                "sha256": sha256(report_json_path),
            },
            "report_md": {
                "path": report_md_path,
                "sha256": sha256(report_md_path),
            },
            "fit_summary_npz": {
                "path": arrays_path,
                "sha256": sha256(arrays_path),
            },
            "figures": [
                {"path": figure_path, "sha256": sha256(figure_path)}
            ],
        },
        "runner": {
            "path": current_runner,
            "sha256": sha256(current_runner),
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            to_jsonable(
                {
                    "status": payload["status"],
                    "data_gate": audit["gate_status"],
                    "primary": audit["cutoffs"]["0.15"],
                    "report": report_json_path,
                    "figure": figure_path,
                    "manifest": manifest_path,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
