#!/usr/bin/env python3
"""Collapsed MCMC comparison of pre/post-reconstruction halo fNL models.

This entry deliberately reuses the released pre- and post-reconstruction
theory evaluators.  It does not duplicate their physics.  At fixed nonlinear
coordinates it exactly integrates every conditionally-linear nuisance,
including the parameter-dependent log-determinant of the conditional Hessian.

The historical profiler is used only for a joint-MAP centre cross-check.
Profiler/HESSE/MINOS widths are neither produced nor accepted as posterior
uncertainties.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "python"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import emcee
import h5py
import numpy as np
import arviz as az
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, minimize
from scipy.stats import ks_2samp, norm, rankdata, wasserstein_distance

from marisa_b.model import combine_finite_png
from marisa_b.statistics import weighted_quantile
import run_post_recon_halo_finite_png_v1 as post_model
import run_pre_recon_bias_model_v3 as pre_parameter_helpers
import run_pre_recon_halo_png_uplift_v1 as pre_legacy
import run_pre_recon_halo_tree_fnl2_v1 as pre_model
from produce_quijote_halo_marisa_b_bias_v1_gaussian_templates import (
    load_measurement_geometry,
)


TAG = "prepost_recon_halo_fnl_mcmc_v1_20260727"
CUTS = (0.08, 0.10, 0.12, 0.14)
EXPECTED_COUNTS = {0.08: 9, 0.10: 14, 0.12: 20, 0.14: 27}
TIERS = ("coevolution", "full")
RECONSTRUCTIONS = ("pre", "post")
FNL_BOUNDS = (-150.0, 150.0)
WIDE_FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 6.0)
B1_PRIOR_MEAN = 2.7465779027841823
B1_PRIOR_SIGMA = 2.0
EXCLUDED_GLOBAL_BIN = 0
EXCLUDED_SHELL_CENTRE = 0.1553916
EXCLUDED_SHELL_TOLERANCE = 0.002
DEFAULT_STOCHASTIC_MODE = "tied"
DEFAULT_PNG_STOCHASTIC_MODEL = "independent_eq265"
PRIMARY_RESPONSE_MODEL = "finite_halo_tree"
CONTRACT_RELATIVE = Path("configs") / "inference_prepost_example.json"
STATE_RELATIVE = Path("analysis") / TAG / "run_state.json"
CHAINS_RELATIVE = Path("analysis") / TAG / "chains.h5"
SUMMARY_RELATIVE = Path("analysis") / TAG / "summary.json"
RELEASE_RELATIVE = Path("analysis") / TAG / "release.json"
REPORT_RELATIVE = Path("analysis") / TAG / "REPORT.md"
FIGURE_RELATIVE = Path("figures") / "diagnostics" / TAG
FISHER_TAG = "prepost_recon_halo_fnl_fisher_audit_20260728"
FISHER_ANALYSIS_RELATIVE = Path("analysis") / FISHER_TAG
FISHER_SUMMARY_RELATIVE = FISHER_ANALYSIS_RELATIVE / "summary.json"
FISHER_REPORT_RELATIVE = FISHER_ANALYSIS_RELATIVE / "REPORT.md"
FISHER_FIGURE_RELATIVE = (
    Path("figures") / "diagnostics" / FISHER_TAG
)
ADAPTIVE_BREC_FISHER_TAG = (
    "post_recon_halo_local_png_brec_fisher_20260728"
)
ADAPTIVE_BREC_ANALYSIS_RELATIVE = (
    Path("analysis") / ADAPTIVE_BREC_FISHER_TAG
)
ADAPTIVE_BREC_BASIS_RELATIVE = (
    ADAPTIVE_BREC_ANALYSIS_RELATIVE
    / "adaptive_brec_tree_basis.jsonl"
)
ADAPTIVE_BREC_SUMMARY_RELATIVE = (
    ADAPTIVE_BREC_ANALYSIS_RELATIVE / "summary.json"
)
ADAPTIVE_BREC_REPORT_RELATIVE = (
    ADAPTIVE_BREC_ANALYSIS_RELATIVE / "REPORT.md"
)
ADAPTIVE_BREC_FIGURE_RELATIVE = (
    Path("figures") / "diagnostics" / ADAPTIVE_BREC_FISHER_TAG
    / "adaptive_brec_fisher_vs_kmax.pdf"
)
ADAPTIVE_BREC_FULL_TAG = (
    "post_recon_halo_local_png_brec_full_response_20260728"
)
ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE = (
    Path("analysis") / ADAPTIVE_BREC_FULL_TAG
)
ADAPTIVE_BREC_FULL_RESPONSE_RELATIVE = (
    ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE
    / "adaptive_brec_full_response_templates.jsonl"
)
ADAPTIVE_BREC_FULL_MANIFEST_RELATIVE = (
    ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE / "raw_manifest.json"
)
ADAPTIVE_BREC_FULL_SUMMARY_RELATIVE = (
    ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE / "summary.json"
)
ADAPTIVE_BREC_FULL_REPORT_RELATIVE = (
    ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE / "REPORT.md"
)
ADAPTIVE_BREC_FULL_MCMC_CHAINS_RELATIVE = (
    ADAPTIVE_BREC_FULL_ANALYSIS_RELATIVE
    / "adaptive_brec_full_mcmc_chains.h5"
)
ADAPTIVE_BREC_FULL_FIGURE_RELATIVE = (
    Path("figures") / "diagnostics" / ADAPTIVE_BREC_FULL_TAG
)
ADAPTIVE_BREC_FULL_ARCHIVE_RELATIVE = (
    Path("old_doc_codes")
    / "archive_20260728_adaptive_brec_full_response"
)
ADAPTIVE_BREC_FULL_RAW_RELATIVE = (
    ADAPTIVE_BREC_FULL_ARCHIVE_RELATIVE / "raw"
)
ADAPTIVE_BREC_FULL_FAILURE_LOG_RELATIVE = (
    ADAPTIVE_BREC_FULL_ARCHIVE_RELATIVE / "failed_logs"
)
ADAPTIVE_BREC_FINITE_SCHEMA = (
    "marisa-b-eft-v2-post-r1-adaptive-brec-finite-jsonl-v1"
)
ADAPTIVE_BREC_VARIANTS = (
    ("p1", 1.0),
    ("m1", -1.0),
    ("p0p5", 0.5),
    ("m0p5", -0.5),
)
ADAPTIVE_BREC_MCMC_MODELS = (
    "pre",
    "post_fixed_brec",
    "post_adaptive_brec_local",
)
ADAPTIVE_BREC_BPHI_REC = (
    2.0
    * post_model.DELTA_C
    * (post_model.B_REC_H - post_model.P_UNIVERSALITY)
)
ADAPTIVE_BREC_KMIN = 2.0 * math.pi / 1000.0
ADAPTIVE_BREC_POWER_RELATIVE = (
    Path("analysis")
    / "theory_vectors/marisa_b_v0"
    / (
        "marisa_b_pre_gaussian_shellbin_z1_r15_diag15_kmax0p3_"
        "nrad3_nmu12_partial446_20260616_plin_z1.dat"
    )
)
ADAPTIVE_BREC_PNG_TABLE_RELATIVE = (
    Path("analysis")
    / "theory_vectors/marisa_b_v0"
    / (
        "marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_"
        "nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat"
    )
)
ARCHIVE_RELATIVE = (
    Path("log")
    / "prepost_recon_halo_fnl_mcmc_v1_20260727"
)
EXPLICIT_CHAINS_RELATIVE = ARCHIVE_RELATIVE / "explicit_validation.h5"
FITTER_RELATIVE = (
    Path("scripts")
    / "legacy/fit_eft_v2_r0_mean.py"
)
CONTRACT_EFT_RELATIVE = (
    Path("configs")
    / "eft_v2_contract.json"
)
MATRIX_RELATIVE = (
    Path("analysis")
    / "quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
    / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid500_b000_pk_matrix.npz"
)
POST_TEMPLATE_RELATIVE = (
    Path("analysis")
    / post_model.TAG
    / "post_r1_gaussian_templates.jsonl"
)
POST_PNG_RELATIVE = (
    Path("analysis")
    / post_model.TAG
    / "finite_png_templates.npz"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_indices(k_pair: np.ndarray, kmax: float) -> np.ndarray:
    k_pair = np.asarray(k_pair, dtype=np.float64)
    mask = np.max(k_pair, axis=1) <= float(kmax) + 1.0e-12
    mask[EXCLUDED_GLOBAL_BIN] = False
    forbidden_shell = np.any(
        np.isclose(
            k_pair,
            EXCLUDED_SHELL_CENTRE,
            rtol=0.0,
            atol=EXCLUDED_SHELL_TOLERANCE,
        ),
        axis=1,
    )
    mask[forbidden_shell] = False
    indices = np.flatnonzero(mask)
    expected = EXPECTED_COUNTS[float(kmax)]
    if indices.size != expected:
        raise AssertionError(
            f"kmax={kmax:.2f}: selected {indices.size}, expected {expected}"
        )
    return indices


def hartlap_factor(mock_count: int, dimension: int) -> float:
    factor = (float(mock_count) - float(dimension) - 2.0) / (
        float(mock_count) - 1.0
    )
    if not 0.0 < factor <= 1.0:
        raise ValueError(
            f"invalid Hartlap factor for N={mock_count}, p={dimension}"
        )
    return factor


@dataclass(frozen=True)
class ConditionalResult:
    nuisance: np.ndarray
    prediction: np.ndarray
    qmin: float
    logdet_hessian: float
    data_q: float
    prior_q: float
    hessian: np.ndarray
    design: np.ndarray
    affine_error: float


@dataclass
class CollapsedContext:
    reconstruction: str
    tier: str
    kmax: float
    names: tuple[str, ...]
    nonlinear_names: tuple[str, ...]
    prior: Any
    indices: np.ndarray
    target: np.ndarray
    covariance_single: np.ndarray
    covariance_mock_count: int
    k_pair: np.ndarray
    edges: np.ndarray
    realization_ids: np.ndarray
    samples: np.ndarray
    evaluate_model: Callable[
        [float, np.ndarray],
        tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]],
    ]
    input_hashes: dict[str, str]
    fnl_bounds: tuple[float, float] = FNL_BOUNDS

    def __post_init__(self) -> None:
        self.name_to_index = {
            name: index for index, name in enumerate(self.names)
        }
        self.nonlinear_indices = np.asarray(
            [self.name_to_index[name] for name in self.nonlinear_names],
            dtype=int,
        )
        self.linear_names = tuple(
            name for name in self.names if name not in self.nonlinear_names
        )
        self.linear_indices = np.asarray(
            [self.name_to_index[name] for name in self.linear_names],
            dtype=int,
        )
        self.theta_names = ("fNL",) + self.nonlinear_names
        self.hartlap = hartlap_factor(
            self.covariance_mock_count,
            self.indices.size,
        )
        self.chol_single = np.linalg.cholesky(self.covariance_single)
        self.chol_fit = self.chol_single / math.sqrt(self.hartlap)
        if tuple(self.prior.names) != tuple(self.names):
            raise ValueError("prior/name ordering mismatch")
        if self.nonlinear_names[0] != "b1":
            raise ValueError("registered nonlinear ordering must start at b1")
        if self.target.shape != (self.indices.size,):
            raise ValueError("target shape mismatch")
        if self.samples.shape != (
            self.covariance_mock_count,
            self.indices.size,
        ):
            raise ValueError("selected realization matrix shape mismatch")

    @property
    def theta_dimension(self) -> int:
        return 1 + len(self.nonlinear_names)

    def theta_start(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray([0.0]),
                self.prior.mean[self.nonlinear_indices],
            )
        )

    def in_support(self, theta: np.ndarray) -> bool:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.theta_dimension,) or not np.all(
            np.isfinite(theta)
        ):
            return False
        if not self.fnl_bounds[0] <= theta[0] <= self.fnl_bounds[1]:
            return False
        b1_location = 1 + self.nonlinear_names.index("b1")
        if not B1_BOUNDS[0] < theta[b1_location] < B1_BOUNDS[1]:
            return False
        return True

    def _base_and_design(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nuisance = np.asarray(self.prior.mean, dtype=np.float64).copy()
        nuisance[self.nonlinear_indices] = theta[1:]
        nuisance[self.linear_indices] = 0.0
        base = np.asarray(
            self.evaluate_model(float(theta[0]), nuisance)[0],
            dtype=np.float64,
        )
        design = np.empty(
            (self.target.size, self.linear_indices.size),
            dtype=np.float64,
        )
        for column, parameter_index in enumerate(self.linear_indices):
            unit = nuisance.copy()
            unit[parameter_index] = 1.0
            design[:, column] = (
                np.asarray(
                    self.evaluate_model(float(theta[0]), unit)[0],
                    dtype=np.float64,
                )
                - base
            )
        return nuisance, base, design

    def conditional(self, theta: Iterable[float]) -> ConditionalResult:
        theta_array = np.asarray(tuple(theta), dtype=np.float64)
        if not self.in_support(theta_array):
            raise ValueError("theta lies outside hard support")
        nuisance, base, design = self._base_and_design(theta_array)
        data_residual_zero = solve_triangular(
            self.chol_fit,
            base - self.target,
            lower=True,
        )
        data_design = solve_triangular(
            self.chol_fit,
            design,
            lower=True,
        )
        prior_residual_zero = self.prior.whitener @ (
            nuisance - self.prior.mean
        )
        prior_design = self.prior.whitener[:, self.linear_indices]
        residual_zero = np.concatenate(
            (data_residual_zero, prior_residual_zero)
        )
        stacked_design = np.vstack((data_design, prior_design))
        hessian = stacked_design.T @ stacked_design
        sign, logdet = np.linalg.slogdet(hessian)
        if sign <= 0.0 or not math.isfinite(float(logdet)):
            raise np.linalg.LinAlgError(
                "conditional nuisance Hessian is not positive definite"
            )
        solution = np.linalg.solve(
            hessian,
            -(stacked_design.T @ residual_zero),
        )
        nuisance[self.linear_indices] = solution
        prediction = np.asarray(
            self.evaluate_model(float(theta_array[0]), nuisance)[0],
            dtype=np.float64,
        )
        data_residual = solve_triangular(
            self.chol_fit,
            prediction - self.target,
            lower=True,
        )
        prior_residual = self.prior.whitener @ (
            nuisance - self.prior.mean
        )
        data_q = float(data_residual @ data_residual)
        prior_q = float(prior_residual @ prior_residual)
        qmin = data_q + prior_q
        affine_prediction = base + design @ solution
        affine_error = float(
            np.linalg.norm(
                solve_triangular(
                    self.chol_single,
                    prediction - affine_prediction,
                    lower=True,
                )
            )
        )
        return ConditionalResult(
            nuisance=nuisance,
            prediction=prediction,
            qmin=qmin,
            logdet_hessian=float(logdet),
            data_q=data_q,
            prior_q=prior_q,
            hessian=hessian,
            design=design,
            affine_error=affine_error,
        )

    def log_marginal(
        self,
        theta: Iterable[float],
        *,
        include_logdet: bool = True,
    ) -> float:
        theta_array = np.asarray(tuple(theta), dtype=np.float64)
        if not self.in_support(theta_array):
            return -math.inf
        try:
            result = self.conditional(theta_array)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return -math.inf
        objective = result.qmin
        if include_logdet:
            objective += result.logdet_hessian
        return -0.5 * float(objective)

    def profile_objective(self, theta: Iterable[float]) -> float:
        theta_array = np.asarray(tuple(theta), dtype=np.float64)
        if not self.in_support(theta_array):
            return math.inf
        try:
            return self.conditional(theta_array).qmin
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return math.inf

    def log_explicit(self, vector: Iterable[float]) -> float:
        values = np.asarray(tuple(vector), dtype=np.float64)
        if values.shape != (1 + len(self.names),):
            return -math.inf
        fnl = float(values[0])
        nuisance = values[1:]
        theta = np.concatenate(
            (
                np.asarray([fnl]),
                nuisance[self.nonlinear_indices],
            )
        )
        if not self.in_support(theta) or not np.all(np.isfinite(nuisance)):
            return -math.inf
        try:
            prediction = np.asarray(
                self.evaluate_model(fnl, nuisance)[0],
                dtype=np.float64,
            )
            data_residual = solve_triangular(
                self.chol_fit,
                prediction - self.target,
                lower=True,
            )
            prior_residual = self.prior.whitener @ (
                nuisance - self.prior.mean
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return -math.inf
        return -0.5 * float(
            data_residual @ data_residual
            + prior_residual @ prior_residual
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "reconstruction": self.reconstruction,
            "tier": self.tier,
            "kmax_h_mpc": self.kmax,
            "selected_indices": self.indices,
            "n_data": self.indices.size,
            "covariance_mock_count": self.covariance_mock_count,
            "covariance_of_single_realization": True,
            "covariance_divided_by_mock_count": False,
            "hartlap_factor": self.hartlap,
            "names": self.names,
            "nonlinear_names": self.nonlinear_names,
            "linear_names": self.linear_names,
            "theta_names": self.theta_names,
            "fnl_bounds": self.fnl_bounds,
            "b1_bounds": B1_BOUNDS,
            "input_hashes": self.input_hashes,
        }


@dataclass(frozen=True)
class SharedInputs:
    root: Path
    data_root: Path
    output_root: Path
    matrix: Path
    realization_ids: np.ndarray
    k_pair: np.ndarray
    edges: np.ndarray
    pre_samples: np.ndarray
    post_samples: np.ndarray
    matrix_hash: str


def load_shared_inputs(
    root: Path,
    data_root: Path,
    output_root: Path,
) -> SharedInputs:
    matrix = data_root / MATRIX_RELATIVE
    if not matrix.is_file():
        raise FileNotFoundError(matrix)
    with np.load(matrix, allow_pickle=False) as values:
        ids = np.asarray(values["fiducial_realizations"], dtype=int)
        k_pair = np.asarray(values["b000_k"], dtype=np.float64)
        edges = np.asarray(values["b000_k_edges"], dtype=np.float64)
        pre_samples = np.asarray(
            values["fiducial_pre_B000"],
            dtype=np.float64,
        )
        post_samples = np.asarray(
            values["fiducial_post_B000"],
            dtype=np.float64,
        )
        stored_pre_mean = np.asarray(
            values["fiducial_pre_B000_mean"],
            dtype=np.float64,
        )
        stored_post_mean = np.asarray(
            values["fiducial_post_B000_mean"],
            dtype=np.float64,
        )
        stored_pre_cov = np.asarray(
            values["fiducial_pre_B000_cov"],
            dtype=np.float64,
        )
        stored_post_cov = np.asarray(
            values["fiducial_post_B000_cov"],
            dtype=np.float64,
        )
    if (
        ids.shape != (500,)
        or not np.array_equal(ids, np.arange(500))
        or pre_samples.shape != (500, 120)
        or post_samples.shape != (500, 120)
        or k_pair.shape != (120, 2)
        or edges.shape != (120, 2, 2)
    ):
        raise ValueError("unexpected shared fid500 matrix contract")
    checks = (
        np.array_equal(stored_pre_mean, np.mean(pre_samples, axis=0)),
        np.array_equal(stored_post_mean, np.mean(post_samples, axis=0)),
        np.linalg.norm(
            stored_pre_cov
            - np.cov(pre_samples, rowvar=False, ddof=1)
        )
        <= 1.0e-12 * np.linalg.norm(stored_pre_cov),
        np.linalg.norm(
            stored_post_cov
            - np.cov(post_samples, rowvar=False, ddof=1)
        )
        <= 1.0e-12 * np.linalg.norm(stored_post_cov),
    )
    if not all(checks):
        raise ValueError("stored means/covariances do not close")
    return SharedInputs(
        root=root,
        data_root=data_root,
        output_root=output_root,
        matrix=matrix,
        realization_ids=ids,
        k_pair=k_pair,
        edges=edges,
        pre_samples=pre_samples,
        post_samples=post_samples,
        matrix_hash=sha256(matrix),
    )


def build_pre_context(
    shared: SharedInputs,
    *,
    tier: str,
    kmax: float,
    fnl_bounds: tuple[float, float] = FNL_BOUNDS,
) -> CollapsedContext:
    root = shared.root
    paths = pre_model.paths_for(
        root,
        data_root=shared.data_root,
        output_root=shared.output_root,
    )
    pre_legacy.require_inputs(paths, include_theory=True)
    accepted_report_path = (
        shared.data_root
        / "analysis/pre_recon_bias_model_v3_20260725/report.json"
    )
    accepted_report = json.loads(
        accepted_report_path.read_text(encoding="utf-8")
    )
    calibrated_b1 = float(
        accepted_report["b1_calibration"]["primary"]["b1"]
    )
    if not math.isclose(
        calibrated_b1,
        B1_PRIOR_MEAN,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("pre b1 calibration differs from frozen MCMC prior")
    geometry = load_measurement_geometry(paths.fid_matrix)
    frozen, _data, frozen_prior, templates, _power, nbar = (
        pre_legacy.load_frozen(paths)
    )
    tree = pre_legacy.load_tree(paths, geometry, calibrated_b1)
    dm_tree, _ = pre_legacy.load_dm_vector(
        paths.dm_linear,
        geometry,
        component="B000_dBdfNL_local_tree",
    )
    dm_total, _ = pre_legacy.load_dm_vector(
        paths.dm_linear,
        geometry,
        component="B000_dBdfNL_primary",
    )
    dm_quadratic, _ = pre_legacy.load_dm_vector(
        paths.dm_quadratic,
        geometry,
        component="B000_B112II_fNL2_coefficient",
    )
    names, nonlinear, prior = pre_legacy.tier_architecture(
        frozen,
        frozen_prior,
        tier,
        1.0,
        profile_b1=True,
        b1_prior_mean=calibrated_b1,
        b1_prior_sigma=B1_PRIOR_SIGMA,
    )
    indices = selected_indices(shared.k_pair, kmax)
    samples = np.asarray(shared.pre_samples[:, indices], dtype=np.float64)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    target = np.mean(samples, axis=0)

    def evaluate(
        fnl: float,
        nuisance: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        parameters = pre_legacy.parameter_dict(
            frozen,
            names,
            nuisance,
            tier=tier,
            b1=calibrated_b1,
        )
        gaussian = pre_legacy.gaussian_components(
            frozen,
            templates,
            parameters,
            nbar,
            tier,
        )
        response = pre_legacy.response_components(
            tree=tree,
            templates=templates,
            parameters=parameters,
            b1=float(parameters["b1"]),
            p=1.0,
            nbar=nbar,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
            response_model=PRIMARY_RESPONSE_MODEL,
        )
        prediction = combine_finite_png(
            gaussian["total"],
            response["linear_total"],
            response["quadratic_total"],
            fnl,
        )
        return prediction[indices], parameters, response

    return CollapsedContext(
        reconstruction="pre",
        tier=tier,
        kmax=float(kmax),
        names=tuple(names),
        nonlinear_names=tuple(nonlinear),
        prior=prior,
        indices=indices,
        target=target,
        covariance_single=covariance,
        covariance_mock_count=samples.shape[0],
        k_pair=shared.k_pair,
        edges=shared.edges,
        realization_ids=shared.realization_ids,
        samples=samples,
        evaluate_model=evaluate,
        input_hashes={
            "matrix": shared.matrix_hash,
            "runner": sha256(
                root
                / "scripts/production/run_pre_recon_halo_tree_fnl2_v1.py"
            ),
            "legacy_runner": sha256(
                root
                / "scripts/production/run_pre_recon_halo_png_uplift_v1.py"
            ),
            "gaussian_templates": sha256(paths.b_templates),
            "tree_templates": sha256(paths.tree_templates),
            "dm_linear": sha256(paths.dm_linear),
            "dm_quadratic": sha256(paths.dm_quadratic),
            "eft_contract": sha256(paths.frozen_contract),
        },
        fnl_bounds=fnl_bounds,
    )


def build_post_context(
    shared: SharedInputs,
    *,
    tier: str,
    kmax: float,
    fnl_bounds: tuple[float, float] = FNL_BOUNDS,
) -> CollapsedContext:
    root = shared.root
    fitter_path = root / FITTER_RELATIVE
    contract_path = root / CONTRACT_EFT_RELATIVE
    gaussian_path = shared.data_root / POST_TEMPLATE_RELATIVE
    png_path = shared.data_root / POST_PNG_RELATIVE
    for path in (fitter_path, contract_path, gaussian_path, png_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fitter = load_module(
        f"_prepost_mcmc_fitter_{tier}_{str(kmax).replace('.', 'p')}",
        fitter_path,
    )
    data = fitter.DataSet.load(shared.matrix)
    _contract, frozen_prior, nbar = fitter.load_prior(contract_path)
    gaussian_templates = post_model.load_partial_templates(
        fitter,
        data,
        gaussian_path,
    )
    png_templates = post_model.load_png_cache(png_path)
    names = post_model.free_names(
        fitter,
        tier,
        DEFAULT_STOCHASTIC_MODE,
    )
    if DEFAULT_PNG_STOCHASTIC_MODEL == "independent_eq265":
        names = names + (post_model.PNG_STOCHASTIC_NAME,)
    nonlinear = post_model.nonlinear_names(tier)
    prior = post_model.calibrated_prior_block(
        pre_parameter_helpers,
        fitter,
        frozen_prior,
        names,
        nonlinear,
        1.0,
    )
    indices = selected_indices(shared.k_pair, kmax)
    missing = sorted(set(indices) - set(png_templates.indices))
    if missing:
        raise ValueError(f"post PNG cache misses selected bins {missing}")
    samples = np.asarray(shared.post_samples[:, indices], dtype=np.float64)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    target = np.mean(samples, axis=0)

    def evaluate(
        fnl: float,
        nuisance: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        parameters = pre_parameter_helpers.full_parameter_dict(
            fitter,
            names,
            nuisance,
            tier=tier,
            fixed_b1=None,
        )
        gaussian = gaussian_templates.components(
            parameters,
            nbar,
            DEFAULT_STOCHASTIC_MODE,
        )
        response = png_templates.components(
            parameters,
            nbar,
            DEFAULT_STOCHASTIC_MODE,
            DEFAULT_PNG_STOCHASTIC_MODEL,
        )
        prediction = post_model.combine_finite_png_prediction(
            gaussian["total"],
            response,
            fnl,
        )
        return prediction[indices], parameters, response

    return CollapsedContext(
        reconstruction="post",
        tier=tier,
        kmax=float(kmax),
        names=tuple(names),
        nonlinear_names=tuple(nonlinear),
        prior=prior,
        indices=indices,
        target=target,
        covariance_single=covariance,
        covariance_mock_count=samples.shape[0],
        k_pair=shared.k_pair,
        edges=shared.edges,
        realization_ids=shared.realization_ids,
        samples=samples,
        evaluate_model=evaluate,
        input_hashes={
            "matrix": shared.matrix_hash,
            "runner": sha256(
                root
                / "scripts/production/run_post_recon_halo_finite_png_v1.py"
            ),
            "frozen_fitter": sha256(fitter_path),
            "gaussian_templates": sha256(gaussian_path),
            "png_templates": sha256(png_path),
            "eft_contract": sha256(contract_path),
        },
        fnl_bounds=fnl_bounds,
    )


def build_context(
    shared: SharedInputs,
    reconstruction: str,
    tier: str,
    kmax: float,
    *,
    wide_prior: bool = False,
) -> CollapsedContext:
    if reconstruction not in RECONSTRUCTIONS:
        raise ValueError(reconstruction)
    if tier not in TIERS:
        raise ValueError(tier)
    bounds = WIDE_FNL_BOUNDS if wide_prior else FNL_BOUNDS
    builder = build_pre_context if reconstruction == "pre" else build_post_context
    return builder(shared, tier=tier, kmax=kmax, fnl_bounds=bounds)


def minuit_centre(
    context: CollapsedContext,
    *,
    marginal: bool,
    starts: int = 5,
) -> dict[str, Any]:
    objective_name = (
        "collapsed_marginal_map" if marginal else "profile_joint_map_centre"
    )

    def objective(*values: float) -> float:
        theta = np.asarray(values, dtype=np.float64)
        if marginal:
            logp = context.log_marginal(theta)
            return math.inf if not math.isfinite(logp) else -2.0 * logp
        return context.profile_objective(theta)

    base = context.theta_start()
    fnl_starts = (0.0, 100.0, -100.0, 140.0, -140.0)
    minimizers: list[Minuit] = []
    for start_index in range(starts):
        start = base.copy()
        start[0] = fnl_starts[start_index % len(fnl_starts)]
        minimizer = Minuit(
            objective,
            *[float(value) for value in start],
            name=context.theta_names,
        )
        minimizer.errordef = 1.0
        minimizer.tol = 1.0e-4
        minimizer.errors["fNL"] = 20.0
        minimizer.limits["fNL"] = context.fnl_bounds
        minimizer.limits["b1"] = B1_BOUNDS
        for name, index in zip(
            context.nonlinear_names,
            context.nonlinear_indices,
        ):
            if name != "b1":
                minimizer.errors[name] = max(
                    0.05,
                    0.2 * float(context.prior.sigma[index]),
                )
        minimizer.migrad(ncall=30000)
        if not minimizer.valid:
            minimizer.simplex(ncall=10000)
            minimizer.migrad(ncall=50000)
        minimizers.append(minimizer)
    valid = [item for item in minimizers if item.valid]
    selected = min(
        valid if valid else minimizers,
        key=lambda item: float(item.fval),
    )
    theta = np.asarray(
        [selected.values[name] for name in context.theta_names],
        dtype=np.float64,
    )
    return {
        "objective_name": objective_name,
        "theta_names": context.theta_names,
        "theta": theta,
        "objective": float(selected.fval),
        "valid": bool(selected.valid),
        "at_limit": bool(selected.fmin.has_parameters_at_limit),
        "nfcn": int(selected.nfcn),
        "multistart_objectives": [
            float(item.fval) for item in minimizers
        ],
        "multistart_valid": [bool(item.valid) for item in minimizers],
        "uncertainty_policy": (
            "centre_only; Minuit/HESSE/MINOS errors are intentionally absent"
        ),
    }


def validate_context(
    context: CollapsedContext,
    rng: np.random.Generator,
    *,
    random_points: int = 4,
) -> dict[str, Any]:
    rows = []
    maximum_affine = 0.0
    maximum_direct_delta = 0.0
    logdet_spread_values = []
    base = context.theta_start()
    for point_index in range(random_points):
        theta = base.copy()
        theta[0] = float(rng.uniform(-120.0, 120.0))
        theta[1:] += rng.normal(
            0.0,
            0.35 * context.prior.sigma[context.nonlinear_indices],
        )
        theta[1 + context.nonlinear_names.index("b1")] = np.clip(
            theta[1 + context.nonlinear_names.index("b1")],
            0.3,
            5.7,
        )
        conditional = context.conditional(theta)
        maximum_affine = max(maximum_affine, conditional.affine_error)
        logdet_spread_values.append(conditional.logdet_hessian)

        nuisance_zero, base_prediction, design = context._base_and_design(theta)
        initial = conditional.nuisance[context.linear_indices]

        def direct_residual(linear: np.ndarray) -> np.ndarray:
            nuisance = nuisance_zero.copy()
            nuisance[context.linear_indices] = linear
            prediction = context.evaluate_model(float(theta[0]), nuisance)[0]
            data_residual = solve_triangular(
                context.chol_fit,
                prediction - context.target,
                lower=True,
            )
            prior_residual = context.prior.whitener @ (
                nuisance - context.prior.mean
            )
            return np.concatenate((data_residual, prior_residual))

        optimized = least_squares(
            direct_residual,
            initial + rng.normal(0.0, 0.05, size=initial.size),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=2000,
        )
        direct_objective = float(optimized.fun @ optimized.fun)
        direct_delta = abs(direct_objective - conditional.qmin)
        maximum_direct_delta = max(maximum_direct_delta, direct_delta)
        explicit_affine = base_prediction + design @ initial
        rows.append(
            {
                "point": point_index,
                "theta": theta,
                "qmin": conditional.qmin,
                "logdet_hessian": conditional.logdet_hessian,
                "affine_error_sigma_single": conditional.affine_error,
                "explicit_affine_vector_error": float(
                    np.max(
                        np.abs(
                            explicit_affine - conditional.prediction
                        )
                    )
                ),
                "direct_optimizer_success": bool(optimized.success),
                "direct_objective_delta": direct_delta,
            }
        )
    logdet_spread = float(np.ptp(logdet_spread_values))
    profile = minuit_centre(context, marginal=False)
    marginal_map = minuit_centre(context, marginal=True)
    profile_theta = np.asarray(profile["theta"], dtype=np.float64)
    negative_at_profile = context.log_marginal(
        profile_theta,
        include_logdet=False,
    )
    required_at_profile = context.log_marginal(
        profile_theta,
        include_logdet=True,
    )
    return {
        "context": context.metadata(),
        "random_point_checks": rows,
        "maximum_affine_error_sigma_single": maximum_affine,
        "maximum_direct_objective_delta": maximum_direct_delta,
        "logdet_spread_across_random_points": logdet_spread,
        "negative_control": {
            "log_posterior_without_logdet_at_profile_centre": (
                negative_at_profile
            ),
            "log_posterior_with_logdet_at_profile_centre": (
                required_at_profile
            ),
            "absolute_log_posterior_difference": abs(
                required_at_profile - negative_at_profile
            ),
            "parameter_dependent_logdet_detected": logdet_spread > 1.0e-8,
        },
        "profile_centre_only": profile,
        "collapsed_marginal_map": marginal_map,
        "pass": bool(
            maximum_affine < 1.0e-8
            and maximum_direct_delta < 1.0e-7
            and logdet_spread > 1.0e-8
            and profile["valid"]
            and marginal_map["valid"]
        ),
    }


def audit_shared_inputs(
    root: Path,
    shared: SharedInputs,
) -> dict[str, Any]:
    masks = {}
    for cut in CUTS:
        indices = selected_indices(shared.k_pair, cut)
        masks[f"{cut:.2f}"] = {
            "indices": indices,
            "count": indices.size,
            "touches_global_bin_zero": bool(0 in indices),
            "touches_excluded_0p155_shell": bool(
                np.any(
                    np.isclose(
                        shared.k_pair[indices],
                        EXCLUDED_SHELL_CENTRE,
                        rtol=0.0,
                        atol=EXCLUDED_SHELL_TOLERANCE,
                    )
                )
            ),
        }
    required = {
        "run_contract": shared.root / CONTRACT_RELATIVE,
        "matrix": shared.matrix,
        "runner": Path(__file__).resolve(),
        "pre_model": (
            shared.root
            / "scripts/production/run_pre_recon_halo_tree_fnl2_v1.py"
        ),
        "post_model": (
            shared.root
            / "scripts/production/run_post_recon_halo_finite_png_v1.py"
        ),
        "post_gaussian_templates": (
            shared.data_root / POST_TEMPLATE_RELATIVE
        ),
        "post_png_templates": shared.data_root / POST_PNG_RELATIVE,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {
        "schema": "marisa-b-prepost-halo-fnl-mcmc-audit-v1",
        "created_utc": utc_now(),
        "same_500_realization_ids": bool(
            np.array_equal(shared.realization_ids, np.arange(500))
        ),
        "pre_post_same_geometry_by_construction": True,
        "matrix_shape": {
            "pre": shared.pre_samples.shape,
            "post": shared.post_samples.shape,
            "k_pair": shared.k_pair.shape,
            "edges": shared.edges.shape,
        },
        "mask_contract": masks,
        "covariance_contract": {
            "source": "np.cov of the selected 500 individual realizations",
            "covariance_of_single_realization": True,
            "covariance_divided_by_500": False,
            "data_vector": "mean of the same selected realizations",
            "precision": "Hartlap primary",
        },
        "profiler_contract": {
            "use": "joint MAP centre parity only",
            "errors_trusted": False,
            "minos_or_hesse_errors_reported": False,
            "mcmc_width_benchmark": False,
        },
        "source_hashes": {
            name: sha256(path) for name, path in required.items()
        },
        "pass": True,
    }


def benchmark_context(
    context: CollapsedContext,
    *,
    evaluations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    theta = context.theta_start()
    points = []
    for _ in range(evaluations):
        point = theta.copy()
        point[0] = rng.normal(0.0, 50.0)
        point[1:] += rng.normal(
            0.0,
            0.15 * context.prior.sigma[context.nonlinear_indices],
        )
        point[1 + context.nonlinear_names.index("b1")] = np.clip(
            point[1 + context.nonlinear_names.index("b1")],
            0.5,
            5.5,
        )
        points.append(point)
    started = time.perf_counter()
    values = [context.log_marginal(point) for point in points]
    elapsed = time.perf_counter() - started
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("benchmark encountered non-finite log posterior")
    return {
        "context": context.metadata(),
        "evaluations": evaluations,
        "elapsed_seconds": elapsed,
        "milliseconds_per_evaluation": 1000.0 * elapsed / evaluations,
        "serial_evaluations_per_second": evaluations / elapsed,
    }


_ACTIVE_CONTEXT: CollapsedContext | None = None
_ACTIVE_EXPLICIT_CONTEXT: CollapsedContext | None = None


def _global_log_probability(theta: np.ndarray) -> float:
    if _ACTIVE_CONTEXT is None:
        raise RuntimeError("MCMC worker has no active context")
    return _ACTIVE_CONTEXT.log_marginal(theta)


def _global_explicit_log_probability(vector: np.ndarray) -> float:
    if _ACTIVE_EXPLICIT_CONTEXT is None:
        raise RuntimeError("explicit MCMC worker has no active context")
    return _ACTIVE_EXPLICIT_CONTEXT.log_explicit(vector)


def _initial_positions(
    context: CollapsedContext,
    centre: np.ndarray,
    walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scales = np.concatenate(
        (
            np.asarray([8.0]),
            0.03 * context.prior.sigma[context.nonlinear_indices],
        )
    )
    positions = centre + rng.normal(
        0.0,
        scales,
        size=(walkers, context.theta_dimension),
    )
    b1_location = 1 + context.nonlinear_names.index("b1")
    positions[:, 0] = np.clip(
        positions[:, 0],
        context.fnl_bounds[0] + 1.0e-4,
        context.fnl_bounds[1] - 1.0e-4,
    )
    positions[:, b1_location] = np.clip(
        positions[:, b1_location],
        B1_BOUNDS[0] + 1.0e-4,
        B1_BOUNDS[1] - 1.0e-4,
    )
    return positions


def context_key(context: CollapsedContext) -> str:
    return (
        f"{context.reconstruction}_{context.tier}_"
        f"kmax{context.kmax:.2f}".replace(".", "p")
        + (
            "_wide"
            if context.fnl_bounds == WIDE_FNL_BOUNDS
            else ""
        )
    )


def run_mcmc_context(
    context: CollapsedContext,
    *,
    chain_path: Path,
    ensembles: int,
    walkers: int,
    steps: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    if walkers < 2 * context.theta_dimension:
        raise ValueError("emcee needs at least twice ndim walkers")
    marginal_map = minuit_centre(context, marginal=True)
    centre = np.asarray(marginal_map["theta"], dtype=np.float64)
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context
    group_rows = []
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    pool = None
    if workers > 1:
        pool = mp.get_context("fork").Pool(processes=workers)
    try:
        for ensemble in range(ensembles):
            group = f"{context_key(context)}/ensemble_{ensemble}"
            group_exists = False
            if chain_path.is_file():
                with h5py.File(chain_path, "r") as chain_file:
                    group_exists = group in chain_file
            backend = emcee.backends.HDFBackend(
                str(chain_path),
                name=group,
                read_only=False,
            )
            rng = np.random.default_rng(seed + 1009 * ensemble)
            if not group_exists:
                backend.reset(walkers, context.theta_dimension)
                initial = _initial_positions(
                    context,
                    centre,
                    walkers,
                    rng,
                )
            else:
                iteration = int(backend.iteration)
                if (
                    backend.shape
                    != (walkers, context.theta_dimension)
                ):
                    raise ValueError(
                        f"checkpoint shape mismatch for {group}"
                    )
                initial = None if iteration > 0 else _initial_positions(
                    context,
                    centre,
                    walkers,
                    rng,
                )
            remaining = max(0, steps - int(backend.iteration))
            sampler = emcee.EnsembleSampler(
                walkers,
                context.theta_dimension,
                _global_log_probability,
                pool=pool,
                backend=backend,
                moves=[
                    (emcee.moves.StretchMove(a=2.0), 0.8),
                    (emcee.moves.DEMove(), 0.2),
                ],
            )
            if remaining:
                sampler.run_mcmc(
                    initial,
                    remaining,
                    progress=False,
                    skip_initial_state_check=False,
                )
            group_rows.append(
                {
                    "group": group,
                    "iteration": int(backend.iteration),
                    "mean_acceptance_fraction": float(
                        np.mean(sampler.acceptance_fraction)
                    ),
                }
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _ACTIVE_CONTEXT = None
    return {
        "context": context.metadata(),
        "context_key": context_key(context),
        "marginal_map": marginal_map,
        "ensembles": group_rows,
        "walkers": walkers,
        "requested_steps": steps,
        "workers": workers,
        "chain_path": chain_path,
    }


def _initial_explicit_positions(
    context: CollapsedContext,
    centre_theta: np.ndarray,
    walkers: int,
    rng: np.random.Generator,
    collapsed_theta_pool: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.empty((walkers, 1 + len(context.names)))
    theta_scales = np.concatenate(
        (
            np.asarray([5.0]),
            0.02 * context.prior.sigma[context.nonlinear_indices],
        )
    )
    for walker in range(walkers):
        if collapsed_theta_pool is None:
            theta = centre_theta + rng.normal(0.0, theta_scales)
        else:
            theta = np.asarray(
                collapsed_theta_pool[
                    rng.integers(0, collapsed_theta_pool.shape[0])
                ],
                dtype=np.float64,
            ).copy()
        theta[0] = np.clip(
            theta[0],
            context.fnl_bounds[0] + 1.0e-4,
            context.fnl_bounds[1] - 1.0e-4,
        )
        b1_location = 1 + context.nonlinear_names.index("b1")
        theta[b1_location] = np.clip(
            theta[b1_location],
            B1_BOUNDS[0] + 1.0e-4,
            B1_BOUNDS[1] - 1.0e-4,
        )
        conditional = context.conditional(theta)
        nuisance = conditional.nuisance.copy()
        covariance_linear = np.linalg.inv(conditional.hessian)
        nuisance[context.linear_indices] += (
            (0.2 if collapsed_theta_pool is None else 1.0)
            * np.linalg.cholesky(covariance_linear)
            @ rng.normal(size=context.linear_indices.size)
        )
        nuisance[context.nonlinear_indices] = theta[1:]
        positions[walker, 0] = theta[0]
        positions[walker, 1:] = nuisance
    return positions


def run_explicit_validation(
    context: CollapsedContext,
    *,
    explicit_chain_path: Path,
    collapsed_chain_path: Path,
    ensembles: int,
    walkers: int,
    steps: int,
    workers: int,
    seed: int,
    burn_fraction: float,
) -> dict[str, Any]:
    explicit_dimension = 1 + len(context.names)
    if walkers < 2 * explicit_dimension:
        raise ValueError(
            f"explicit chain needs at least {2 * explicit_dimension} walkers"
        )
    profile = minuit_centre(context, marginal=False)
    centre_theta = np.asarray(profile["theta"], dtype=np.float64)
    collapsed_for_initialization, _ = load_context_chains(
        collapsed_chain_path,
        context_key(context),
    )
    collapsed_initial_burn = max(
        1,
        int(burn_fraction * collapsed_for_initialization.shape[1]),
    )
    collapsed_theta_pool = collapsed_for_initialization[
        :,
        collapsed_initial_burn:,
        :,
        :,
    ].reshape(-1, context.theta_dimension)
    explicit_chain_path.parent.mkdir(parents=True, exist_ok=True)
    global _ACTIVE_EXPLICIT_CONTEXT
    _ACTIVE_EXPLICIT_CONTEXT = context
    pool = None
    if workers > 1:
        pool = mp.get_context("fork").Pool(processes=workers)
    group_rows = []
    try:
        for ensemble in range(ensembles):
            group = f"{context_key(context)}/ensemble_{ensemble}"
            group_exists = False
            if explicit_chain_path.is_file():
                with h5py.File(explicit_chain_path, "r") as chain_file:
                    group_exists = group in chain_file
            backend = emcee.backends.HDFBackend(
                str(explicit_chain_path),
                name=group,
                read_only=False,
            )
            rng = np.random.default_rng(seed + 2027 * ensemble)
            if not group_exists:
                backend.reset(walkers, explicit_dimension)
                initial = _initial_explicit_positions(
                    context,
                    centre_theta,
                    walkers,
                    rng,
                    collapsed_theta_pool,
                )
            else:
                if backend.shape != (walkers, explicit_dimension):
                    raise ValueError(
                        f"explicit checkpoint shape mismatch for {group}"
                    )
                initial = (
                    None
                    if backend.iteration > 0
                    else _initial_explicit_positions(
                        context,
                        centre_theta,
                        walkers,
                        rng,
                        collapsed_theta_pool,
                    )
                )
            sampler = emcee.EnsembleSampler(
                walkers,
                explicit_dimension,
                _global_explicit_log_probability,
                pool=pool,
                backend=backend,
                moves=[
                    (emcee.moves.StretchMove(a=2.0), 0.8),
                    (emcee.moves.DEMove(), 0.2),
                ],
            )
            remaining = max(0, steps - int(backend.iteration))
            if remaining:
                sampler.run_mcmc(
                    initial,
                    remaining,
                    progress=False,
                    skip_initial_state_check=False,
                )
            group_rows.append(
                {
                    "group": group,
                    "iteration": int(backend.iteration),
                    "mean_acceptance_fraction": float(
                        np.mean(sampler.acceptance_fraction)
                    ),
                }
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _ACTIVE_EXPLICIT_CONTEXT = None

    explicit_all, explicit_accepted = load_context_chains(
        explicit_chain_path,
        context_key(context),
    )
    collapsed_all, _ = load_context_chains(
        collapsed_chain_path,
        context_key(context),
    )
    explicit_burn = int(burn_fraction * explicit_all.shape[1])
    collapsed_burn = int(burn_fraction * collapsed_all.shape[1])
    explicit_retained = explicit_all[:, explicit_burn:, :, :]
    collapsed_retained = collapsed_all[:, collapsed_burn:, :, :]
    explicit_arviz = np.concatenate(
        [
            explicit_retained[index].transpose(1, 0, 2)
            for index in range(ensembles)
        ],
        axis=0,
    )
    explicit_rhat = np.asarray(
        az.rhat(
            explicit_arviz,
            method="rank",
            chain_axis=0,
            draw_axis=1,
        )
    )
    explicit_fnl = explicit_retained[..., 0].reshape(-1)
    collapsed_fnl = collapsed_retained[..., 0].reshape(-1)
    retained_draw_count = explicit_retained.shape[1]
    drift_block = max(1, retained_draw_count // 3)
    explicit_early_fnl = explicit_retained[
        :,
        :drift_block,
        :,
        0,
    ].reshape(-1)
    explicit_late_fnl = explicit_retained[
        :,
        -drift_block:,
        :,
        0,
    ].reshape(-1)
    explicit_quantiles = np.quantile(
        explicit_fnl,
        [0.16, 0.5, 0.84],
    )
    collapsed_quantiles = np.quantile(
        collapsed_fnl,
        [0.16, 0.5, 0.84],
    )
    combined_scale = math.sqrt(
        float(np.var(explicit_fnl))
        + float(np.var(collapsed_fnl))
    )
    rng_compare = np.random.default_rng(seed + 99173)
    comparison_count = min(
        100000,
        explicit_fnl.size,
        collapsed_fnl.size,
    )
    explicit_compare = rng_compare.choice(
        explicit_fnl,
        size=comparison_count,
        replace=False,
    )
    collapsed_compare = rng_compare.choice(
        collapsed_fnl,
        size=comparison_count,
        replace=False,
    )
    wasserstein = float(
        wasserstein_distance(explicit_compare, collapsed_compare)
    )
    drift_count = min(
        100000,
        explicit_early_fnl.size,
        explicit_late_fnl.size,
    )
    early_compare = rng_compare.choice(
        explicit_early_fnl,
        size=drift_count,
        replace=False,
    )
    late_compare = rng_compare.choice(
        explicit_late_fnl,
        size=drift_count,
        replace=False,
    )
    drift_wasserstein = float(
        wasserstein_distance(early_compare, late_compare)
    )
    drift_median_sigma = float(
        abs(
            np.median(explicit_early_fnl)
            - np.median(explicit_late_fnl)
        )
        / np.std(collapsed_fnl, ddof=1)
    )
    ks = ks_2samp(explicit_compare, collapsed_compare)
    quantile_delta = explicit_quantiles - collapsed_quantiles
    return {
        "context": context.metadata(),
        "explicit_parameter_names": ("fNL",) + context.names,
        "explicit_dimension": explicit_dimension,
        "groups": group_rows,
        "burn_fraction": burn_fraction,
        "explicit_total_retained": explicit_fnl.size,
        "collapsed_total_retained": collapsed_fnl.size,
        "explicit_rank_rhat": explicit_rhat,
        "explicit_fnl_quantiles_16_50_84": explicit_quantiles,
        "collapsed_fnl_quantiles_16_50_84": collapsed_quantiles,
        "fnl_quantile_delta": quantile_delta,
        "median_delta_in_combined_sigma": (
            float(abs(quantile_delta[1]) / combined_scale)
        ),
        "wasserstein_distance_fnl": wasserstein,
        "wasserstein_in_collapsed_sigma": (
            wasserstein / float(np.std(collapsed_fnl, ddof=1))
        ),
        "ks_statistic_diagnostic_only": float(ks.statistic),
        "early_to_late_fnl_median_drift_in_collapsed_sigma": (
            drift_median_sigma
        ),
        "early_to_late_wasserstein_in_collapsed_sigma": (
            drift_wasserstein
            / float(np.std(collapsed_fnl, ddof=1))
        ),
        "mean_acceptance_fraction": (
            float(np.mean(explicit_accepted) / explicit_all.shape[1])
            if explicit_accepted.size
            else math.nan
        ),
        "pass": bool(
            abs(quantile_delta[1]) < 0.1 * combined_scale
            and wasserstein
            < 0.1 * float(np.std(collapsed_fnl, ddof=1))
            and drift_median_sigma < 0.1
            and drift_wasserstein
            < 0.1 * float(np.std(collapsed_fnl, ddof=1))
        ),
        "interpretation": (
            "short direct all-parameter invariance test; no conditional "
            "logdet is used inside the explicit likelihood; the reported "
            "high per-walker Rhat prohibits using this chain for inference"
        ),
        "production_inference_allowed": False,
    }


def shortest_hdi(samples: np.ndarray, probability: float) -> tuple[float, float]:
    values = np.sort(np.asarray(samples, dtype=np.float64))
    if values.ndim != 1 or values.size < 2:
        raise ValueError("HDI needs a one-dimensional non-empty sample")
    count = int(math.floor(probability * values.size))
    count = min(max(count, 1), values.size - 1)
    widths = values[count:] - values[: values.size - count]
    location = int(np.argmin(widths))
    return float(values[location]), float(values[location + count])


def load_context_chains(
    chain_path: Path,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not chain_path.is_file():
        raise FileNotFoundError(chain_path)
    chains = []
    accepted = []
    with h5py.File(chain_path, "r") as chain_file:
        if key not in chain_file:
            raise KeyError(f"missing chain context {key}")
        context_group = chain_file[key]
        ensemble_names = sorted(context_group)
        if len(ensemble_names) != 4:
            raise ValueError(
                f"{key}: expected four ensembles, got {ensemble_names}"
            )
        for name in ensemble_names:
            group = context_group[name]
            chain = np.asarray(group["chain"], dtype=np.float64)
            if chain.ndim != 3:
                raise ValueError(f"{key}/{name}: invalid chain rank")
            chains.append(chain)
            if "accepted" in group:
                accepted.append(np.asarray(group["accepted"], dtype=float))
    shapes = {chain.shape for chain in chains}
    if len(shapes) != 1:
        raise ValueError(f"{key}: ensemble chain shapes differ: {shapes}")
    return np.stack(chains, axis=0), np.stack(accepted, axis=0)


def pooled_rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = rankdata(values, method="average").reshape(values.shape)
    probabilities = (ranks - 0.375) / (values.size + 0.25)
    return norm.ppf(probabilities)


def ensemble_scaled_ess(
    transformed: np.ndarray,
) -> tuple[float, float, float]:
    """ESS for posterior draws while preserving walker coupling.

    The four independent emcee ensembles are the independent chains.  At
    each iteration we average the transformed quantity over the interacting
    walkers.  The spectral ESS of those four ensemble-level time series is
    then rescaled by Var(single draw) / Var(ensemble average), which retains
    same-step walker correlations instead of pretending walkers are
    independent chains.
    """

    transformed = np.asarray(transformed, dtype=np.float64)
    if transformed.ndim != 3:
        raise ValueError("ensemble ESS expects (ensemble,draw,walker)")
    ensemble_series = np.mean(transformed, axis=2)
    ensemble_iteration_ess = float(
        az.ess(
            ensemble_series,
            method="identity",
            chain_axis=0,
            draw_axis=1,
        )
    )
    draw_variance = float(np.var(transformed, ddof=1))
    ensemble_variance = float(np.var(ensemble_series, ddof=1))
    if (
        not math.isfinite(ensemble_iteration_ess)
        or draw_variance <= 0.0
        or ensemble_variance <= 0.0
    ):
        return math.nan, ensemble_iteration_ess, math.nan
    total_draws = transformed.size
    scaled = min(
        float(total_draws),
        ensemble_iteration_ess * draw_variance / ensemble_variance,
    )
    return scaled, ensemble_iteration_ess, draw_variance / ensemble_variance


def independent_ensemble_diagnostics(
    retained: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rank/folded split-Rhat and coupling-aware ESS.

    Walkers inside one affine-invariant ensemble are interacting and are
    therefore never passed to ArviZ as independent chains.  The independent
    units are the four separately initialized ensembles.
    """

    retained = np.asarray(retained, dtype=np.float64)
    if retained.ndim != 4 or retained.shape[0] != 4:
        raise ValueError(
            "independent diagnostics expect four "
            "(draw,walker,parameter) ensembles"
        )
    dimension = retained.shape[-1]
    rank_location_rhat = np.empty(dimension)
    rank_folded_rhat = np.empty(dimension)
    tail_occupancy_rhat = np.empty(dimension)
    bulk_ess = np.empty(dimension)
    tail_ess = np.empty(dimension)
    bulk_ensemble_iteration_ess = np.empty(dimension)
    tail_ensemble_iteration_ess = np.empty(dimension)
    walker_information_multiplier = np.empty(dimension)
    for parameter in range(dimension):
        values = retained[..., parameter]
        rank_values = pooled_rank_normalize(values)
        rank_series = np.mean(rank_values, axis=2)
        folded_values = pooled_rank_normalize(
            np.abs(values - np.median(values))
        )
        folded_series = np.mean(folded_values, axis=2)
        rank_location_rhat[parameter] = float(
            az.rhat(
                rank_series,
                method="split",
                chain_axis=0,
                draw_axis=1,
            )
        )
        rank_folded_rhat[parameter] = float(
            az.rhat(
                folded_series,
                method="split",
                chain_axis=0,
                draw_axis=1,
            )
        )
        quantile_low, quantile_high = np.quantile(
            values,
            (0.05, 0.95),
        )
        lower_indicator = (values <= quantile_low).astype(np.float64)
        upper_indicator = (values >= quantile_high).astype(np.float64)
        tail_fractions = np.stack(
            (
                np.mean(lower_indicator, axis=2),
                np.mean(upper_indicator, axis=2),
            ),
            axis=-1,
        )
        tail_occupancy_rhat[parameter] = float(
            np.max(
                az.rhat(
                    tail_fractions,
                    method="rank",
                    chain_axis=0,
                    draw_axis=1,
                )
            )
        )
        (
            bulk_ess[parameter],
            bulk_ensemble_iteration_ess[parameter],
            walker_information_multiplier[parameter],
        ) = ensemble_scaled_ess(rank_values)
        lower_ess, lower_iteration_ess, _lower_multiplier = (
            ensemble_scaled_ess(lower_indicator)
        )
        upper_ess, upper_iteration_ess, _upper_multiplier = (
            ensemble_scaled_ess(upper_indicator)
        )
        tail_ess[parameter] = min(lower_ess, upper_ess)
        tail_ensemble_iteration_ess[parameter] = min(
            lower_iteration_ess,
            upper_iteration_ess,
        )
    return {
        "rank_location_split_rhat": rank_location_rhat,
        "rank_folded_split_rhat": rank_folded_rhat,
        "rank_normalized_split_rhat": np.maximum(
            rank_location_rhat,
            rank_folded_rhat,
        ),
        "tail_occupancy_rhat_diagnostic": tail_occupancy_rhat,
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "bulk_ensemble_iteration_ess": bulk_ensemble_iteration_ess,
        "tail_ensemble_iteration_ess": tail_ensemble_iteration_ess,
        "walker_information_multiplier": walker_information_multiplier,
    }


def diagnose_context(
    context: CollapsedContext,
    *,
    chain_path: Path,
    burn_fraction: float = 0.5,
) -> dict[str, Any]:
    all_chains, accepted = load_context_chains(
        chain_path,
        context_key(context),
    )
    ensembles, draws, walkers, ndim = all_chains.shape
    if ndim != context.theta_dimension:
        raise ValueError("chain/context dimensionality mismatch")
    burn = int(math.floor(burn_fraction * draws))
    if burn < 1 or draws - burn < 10:
        raise ValueError("chain is too short for diagnostics")
    retained = all_chains[:, burn:, :, :]
    independent = independent_ensemble_diagnostics(retained)
    rank_rhat = independent["rank_normalized_split_rhat"]
    bulk_ess = independent["bulk_ess"]
    tail_ess = independent["tail_ess"]
    # Historical diagnostic retained only to demonstrate why it is not a
    # release gate: walkers within one affine-invariant ensemble interact
    # and cannot be interpreted as independent ArviZ chains.
    legacy_walker_array = np.concatenate(
        [
            retained[ensemble].transpose(1, 0, 2)
            for ensemble in range(ensembles)
        ],
        axis=0,
    )
    legacy_walker_rank_rhat = np.asarray(
        az.rhat(
            legacy_walker_array,
            method="rank",
            chain_axis=0,
            draw_axis=1,
        ),
        dtype=np.float64,
    )
    tau_rows = []
    for ensemble in range(ensembles):
        try:
            tau = np.asarray(
                emcee.autocorr.integrated_time(
                    retained[ensemble],
                    quiet=True,
                ),
                dtype=np.float64,
            )
        except Exception:
            tau = np.full(ndim, math.nan)
        tau_rows.append(tau)
    tau_rows_array = np.asarray(tau_rows)
    tau_max = np.nanmax(tau_rows_array, axis=0)
    flat = retained.reshape(-1, ndim)
    quantile_probabilities = np.asarray(
        [0.025, 0.16, 0.5, 0.84, 0.975]
    )
    quantiles = np.quantile(flat, quantile_probabilities, axis=0)
    hdi68 = np.asarray(
        [shortest_hdi(flat[:, index], 0.68) for index in range(ndim)]
    )
    midpoint = retained.shape[1] // 2
    first_half = retained[:, :midpoint].reshape(-1, ndim)
    second_half = retained[:, midpoint:].reshape(-1, ndim)
    half_median_shift_sigma = np.abs(
        np.median(first_half, axis=0) - np.median(second_half, axis=0)
    ) / np.std(flat, axis=0, ddof=1)
    ensemble_medians = np.median(retained, axis=(1, 2))
    ensemble_median_spread_sigma = np.ptp(
        ensemble_medians,
        axis=0,
    ) / np.std(flat, axis=0, ddof=1)
    fnl = flat[:, 0]
    fnl_width = context.fnl_bounds[1] - context.fnl_bounds[0]
    boundary_width = 0.02 * fnl_width
    fnl_boundary_mass = float(
        np.mean(
            (fnl <= context.fnl_bounds[0] + boundary_width)
            | (fnl >= context.fnl_bounds[1] - boundary_width)
        )
    )
    b1_location = 1 + context.nonlinear_names.index("b1")
    b1 = flat[:, b1_location]
    b1_boundary_mass = float(
        np.mean(
            (b1 <= B1_BOUNDS[0] + 0.02 * (B1_BOUNDS[1] - B1_BOUNDS[0]))
            | (b1 >= B1_BOUNDS[1] - 0.02 * (B1_BOUNDS[1] - B1_BOUNDS[0]))
        )
    )
    retained_draws_per_walker = retained.shape[1]
    length_over_tau = retained_draws_per_walker / tau_max
    profile = minuit_centre(context, marginal=False)
    marginal_map = minuit_centre(context, marginal=True)
    convergence = {
        "independent_ensemble_rank_rhat_below_1p01": bool(
            np.all(rank_rhat < 1.01)
        ),
        "fnl_bulk_ess_above_2000": bool(bulk_ess[0] > 2000.0),
        "fnl_tail_ess_above_2000": bool(tail_ess[0] > 2000.0),
        "other_bulk_ess_above_1000": bool(
            np.all(bulk_ess[1:] > 1000.0)
        ),
        "other_tail_ess_above_1000": bool(
            np.all(tail_ess[1:] > 1000.0)
        ),
        "retained_length_above_50_tau": bool(
            np.all(length_over_tau > 50.0)
        ),
        "half_chain_median_stable_below_0p1_sigma": bool(
            np.all(half_median_shift_sigma < 0.1)
        ),
        "ensemble_median_spread_below_0p2_sigma": bool(
            np.all(ensemble_median_spread_sigma < 0.2)
        ),
        "no_fnl_boundary_sticking": bool(fnl_boundary_mass < 0.05),
        "no_b1_boundary_sticking": bool(b1_boundary_mass < 0.01),
    }
    return {
        "context": context.metadata(),
        "chain_layout": {
            "ensembles": ensembles,
            "walkers_per_ensemble": walkers,
            "draws_per_walker": draws,
            "burn_draws": burn,
            "retained_draws_per_walker": retained_draws_per_walker,
            "total_retained_samples": flat.shape[0],
        },
        "theta_names": context.theta_names,
        "rank_normalized_split_rhat": rank_rhat,
        "rank_location_split_rhat": independent[
            "rank_location_split_rhat"
        ],
        "rank_folded_split_rhat": independent[
            "rank_folded_split_rhat"
        ],
        "tail_occupancy_rhat_diagnostic": independent[
            "tail_occupancy_rhat_diagnostic"
        ],
        "rhat_independence_unit": (
            "four independently initialized emcee ensembles; interacting "
            "walkers are averaged only after pooled rank/fold transforms"
        ),
        "legacy_walker_as_chain_rank_rhat_diagnostic_only": (
            legacy_walker_rank_rhat
        ),
        "bulk_ess": bulk_ess,
        "tail_ess_5_95": tail_ess,
        "ess_method": (
            "spectral ESS of four independent ensemble-average time "
            "series, scaled by pooled single-draw variance over ensemble-"
            "average variance to retain same-step walker coupling"
        ),
        "bulk_ensemble_iteration_ess": independent[
            "bulk_ensemble_iteration_ess"
        ],
        "tail_ensemble_iteration_ess": independent[
            "tail_ensemble_iteration_ess"
        ],
        "walker_information_multiplier": independent[
            "walker_information_multiplier"
        ],
        "integrated_autocorrelation_time_by_ensemble": tau_rows_array,
        "maximum_tau": tau_max,
        "retained_length_over_tau": length_over_tau,
        "half_chain_median_shift_sigma": half_median_shift_sigma,
        "ensemble_medians": ensemble_medians,
        "ensemble_median_spread_sigma": ensemble_median_spread_sigma,
        "mean_acceptance_fraction": (
            float(np.mean(accepted) / draws)
            if accepted.size
            else math.nan
        ),
        "quantile_probabilities": quantile_probabilities,
        "quantiles": quantiles,
        "hdi68": hdi68,
        "fnl_summary": {
            "median": quantiles[2, 0],
            "equal_tail_68": (quantiles[1, 0], quantiles[3, 0]),
            "hdi_68": hdi68[0],
            "equal_tail_95": (quantiles[0, 0], quantiles[4, 0]),
            "standard_deviation": float(np.std(fnl, ddof=1)),
            "boundary_mass_outer_2_percent_each_side": fnl_boundary_mass,
            "truth_zero_inside_95": bool(
                quantiles[0, 0] <= 0.0 <= quantiles[4, 0]
            ),
        },
        "b1_boundary_mass_outer_2_percent_each_side": b1_boundary_mass,
        "profile_centre_only": profile,
        "marginal_map": marginal_map,
        "convergence_checks": convergence,
        "pass": bool(all(convergence.values())),
        "uncertainty_source": (
            "marginalized MCMC quantiles/HDI only; profiler errors prohibited"
        ),
    }


def student_t_reweight(
    context: CollapsedContext,
    *,
    chain_path: Path,
    burn_fraction: float,
    importance_samples: int,
    seed: int,
) -> dict[str, Any]:
    chains, _ = load_context_chains(chain_path, context_key(context))
    burn = int(burn_fraction * chains.shape[1])
    theta_pool = chains[:, burn:, :, :].reshape(
        -1,
        context.theta_dimension,
    )
    if importance_samples < 1000:
        raise ValueError("Student-t reweighting needs at least 1000 samples")
    rng = np.random.default_rng(seed + 773)
    selected = rng.choice(
        theta_pool.shape[0],
        size=importance_samples,
        replace=importance_samples > theta_pool.shape[0],
    )
    theta_samples = theta_pool[selected]
    fnl = theta_samples[:, 0]
    log_weights = np.empty(importance_samples, dtype=np.float64)
    t_squared_values = np.empty(importance_samples, dtype=np.float64)
    identity_errors = np.empty(importance_samples, dtype=np.float64)
    for sample_index, theta in enumerate(theta_samples):
        conditional = context.conditional(theta)
        nuisance = conditional.nuisance.copy()
        covariance_linear = np.linalg.inv(conditional.hessian)
        nuisance[context.linear_indices] += (
            np.linalg.cholesky(covariance_linear)
            @ rng.normal(size=context.linear_indices.size)
        )
        prediction = np.asarray(
            context.evaluate_model(float(theta[0]), nuisance)[0],
            dtype=np.float64,
        )
        residual_single = solve_triangular(
            context.chol_single,
            prediction - context.target,
            lower=True,
        )
        t_squared = float(residual_single @ residual_single)
        gaussian_data_objective = context.hartlap * t_squared
        student_data_objective = float(
            context.covariance_mock_count
            * math.log1p(
                t_squared / (context.covariance_mock_count - 1.0)
            )
        )
        log_weights[sample_index] = -0.5 * (
            student_data_objective - gaussian_data_objective
        )
        t_squared_values[sample_index] = t_squared
        explicit_logp = context.log_explicit(
            np.concatenate(
                (
                    np.asarray([float(theta[0])]),
                    nuisance,
                )
            )
        )
        prior_residual = context.prior.whitener @ (
            nuisance - context.prior.mean
        )
        reconstructed = -0.5 * (
            gaussian_data_objective
            + float(prior_residual @ prior_residual)
        )
        identity_errors[sample_index] = abs(
            explicit_logp - reconstructed
        )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    normalized = weights / np.sum(weights)
    weight_ess = float(1.0 / np.sum(normalized**2))
    probabilities = (0.025, 0.16, 0.5, 0.84, 0.975)
    student_quantiles = weighted_quantile(
        fnl,
        probabilities,
        normalized,
    )
    gaussian_quantiles = np.quantile(fnl, probabilities)
    gaussian_width = 0.5 * (
        gaussian_quantiles[3] - gaussian_quantiles[1]
    )
    student_width = 0.5 * (
        student_quantiles[3] - student_quantiles[1]
    )
    return {
        "context": context.metadata(),
        "method": (
            "exact joint-posterior importance reweighting from "
            "Hartlap-Gaussian to Sellentin-Heavens Student-t; one exact "
            "conditional linear-nuisance draw per collapsed theta sample"
        ),
        "interpretation_limit": (
            "finite-mock likelihood-form sensitivity only; the mean data "
            "vector and covariance estimate use the same 500 realizations, "
            "so this is not an independent-data/covariance coverage test"
        ),
        "importance_samples": importance_samples,
        "weight_effective_sample_size": weight_ess,
        "weight_ess_fraction": weight_ess / importance_samples,
        "maximum_log_weight_range": float(np.ptp(log_weights)),
        "maximum_explicit_likelihood_identity_error": float(
            np.max(identity_errors)
        ),
        "t_squared_range": (
            float(np.min(t_squared_values)),
            float(np.max(t_squared_values)),
        ),
        "probabilities": probabilities,
        "gaussian_hartlap_fnl_quantiles": gaussian_quantiles,
        "student_t_fnl_quantiles": student_quantiles,
        "median_shift": float(
            student_quantiles[2] - gaussian_quantiles[2]
        ),
        "width68_ratio_student_to_gaussian": (
            student_width / gaussian_width
        ),
        "pass": bool(
            weight_ess > 2000.0
            and weight_ess / importance_samples > 0.2
            and np.max(identity_errors) < 1.0e-10
        ),
    }


def retained_chain_samples(
    context: CollapsedContext,
    *,
    chain_path: Path,
    diagnosis: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    chains, _accepted = load_context_chains(
        chain_path,
        context_key(context),
    )
    burn = int(diagnosis["chain_layout"]["burn_draws"])
    if burn != int(diagnosis["chain_layout"]["draws_per_walker"]) - int(
        diagnosis["chain_layout"]["retained_draws_per_walker"]
    ):
        raise ValueError("stored diagnosis burn metadata is inconsistent")
    retained = chains[:, burn:, :, :]
    return retained, retained.reshape(-1, context.theta_dimension)


def histogram_information_gain(
    samples: np.ndarray,
    bounds: tuple[float, float],
    *,
    bins: int = 80,
) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    counts, _edges = np.histogram(values, bins=bins, range=bounds)
    probabilities = (counts.astype(np.float64) + 0.5) / (
        float(np.sum(counts)) + 0.5 * bins
    )
    divergence_nats = float(
        np.sum(probabilities * np.log(probabilities * bins))
    )
    return {
        "method": (
            "80-bin Jeffreys-smoothed discrete KL relative to the "
            "hard-flat fNL prior"
        ),
        "kl_nats": divergence_nats,
        "kl_bits": divergence_nats / math.log(2.0),
    }


def posterior_density(
    samples: np.ndarray,
    bounds: tuple[float, float],
    *,
    bins: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(
        np.asarray(samples, dtype=np.float64),
        bins=bins,
        range=bounds,
        density=True,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density


def posterior_status(
    diagnosis: dict[str, Any],
    bounds: tuple[float, float],
) -> str:
    low, high = bounds
    width = high - low
    hdi = np.asarray(diagnosis["fnl_summary"]["hdi_68"], dtype=np.float64)
    equal95 = np.asarray(
        diagnosis["fnl_summary"]["equal_tail_95"],
        dtype=np.float64,
    )
    touches_hdi = bool(
        hdi[0] <= low + 0.01 * width
        or hdi[1] >= high - 0.01 * width
    )
    broad68 = bool((hdi[1] - hdi[0]) >= 0.60 * width)
    if touches_hdi or broad68:
        return "prior_limited"
    near_95_edge = bool(
        equal95[0] <= low + 0.05 * width
        or equal95[1] >= high - 0.05 * width
    )
    return "prior_influenced" if near_95_edge else "data_informative"


def summarize_posterior(
    context: CollapsedContext,
    *,
    diagnosis: dict[str, Any],
    chain_path: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    retained, flat = retained_chain_samples(
        context,
        chain_path=chain_path,
        diagnosis=diagnosis,
    )
    correlations = np.corrcoef(flat, rowvar=False)
    fnl_correlations = {
        name: float(correlations[0, index])
        for index, name in enumerate(context.theta_names)
        if index > 0
    }
    ordered_degeneracies = sorted(
        fnl_correlations.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    rng = np.random.default_rng(seed)
    density_count = min(200000, flat.shape[0])
    density_indices = rng.choice(
        flat.shape[0],
        size=density_count,
        replace=False,
    )
    corner_count = min(50000, flat.shape[0])
    corner_indices = rng.choice(
        flat.shape[0],
        size=corner_count,
        replace=False,
    )
    fnl_summary = dict(diagnosis["fnl_summary"])
    equal68 = np.asarray(fnl_summary["equal_tail_68"], dtype=np.float64)
    fnl_summary["equal_tail_68_half_width"] = float(
        0.5 * (equal68[1] - equal68[0])
    )
    fnl_summary["posterior_status"] = posterior_status(
        diagnosis,
        context.fnl_bounds,
    )
    fnl_summary["information_gain"] = histogram_information_gain(
        flat[:, 0],
        context.fnl_bounds,
    )
    payload = {
        "context": context.metadata(),
        "converged": bool(diagnosis["pass"]),
        "convergence_checks": diagnosis["convergence_checks"],
        "chain_layout": diagnosis["chain_layout"],
        "maximum_rank_rhat": float(
            np.max(diagnosis["rank_normalized_split_rhat"])
        ),
        "minimum_bulk_ess": float(np.min(diagnosis["bulk_ess"])),
        "minimum_tail_ess": float(np.min(diagnosis["tail_ess_5_95"])),
        "minimum_retained_length_over_tau": float(
            np.min(diagnosis["retained_length_over_tau"])
        ),
        "mean_acceptance_fraction": diagnosis[
            "mean_acceptance_fraction"
        ],
        "marginal_map": diagnosis["marginal_map"],
        "profile_joint_map_centre_only": diagnosis[
            "profile_centre_only"
        ],
        "fnl": fnl_summary,
        "correlation_matrix": correlations,
        "fnl_correlations": fnl_correlations,
        "strongest_fnl_degeneracies": ordered_degeneracies,
        "uncertainty_source": (
            "converged marginalized MCMC quantiles and HDI; no profiler, "
            "HESSE, or MINOS width is used"
        ),
    }
    arrays = {
        "density_fnl": flat[density_indices, 0],
        "corner": flat[corner_indices],
        "trace": retained[:, :: max(1, retained.shape[1] // 500), :, 0],
    }
    return payload, arrays


def bestfit_prediction(
    context: CollapsedContext,
    diagnosis: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    theta = np.asarray(
        diagnosis["profile_centre_only"]["theta"],
        dtype=np.float64,
    )
    conditional = context.conditional(theta)
    residual = conditional.prediction - context.target
    sigma_single = np.sqrt(np.diag(context.covariance_single))
    pull = residual / sigma_single
    whitened_single = solve_triangular(
        context.chol_single,
        residual,
        lower=True,
    )
    full_norm = float(np.linalg.norm(whitened_single))
    payload = {
        "prediction_point": (
            "conditional joint MAP associated with the profiler centre; "
            "the profiler supplies no uncertainty"
        ),
        "fnl_profile_centre": float(theta[0]),
        "theta": theta,
        "conditional_nuisance": conditional.nuisance,
        "max_abs_diagonal_single_realization_pull": float(
            np.max(np.abs(pull))
        ),
        "full_covariance_single_realization_norm": full_norm,
        "acceptance": {
            "max_abs_pull_below_0p5": bool(np.max(np.abs(pull)) < 0.5),
            "full_norm_below_1": bool(full_norm < 1.0),
        },
        "display_errorbar": (
            "sqrt diagonal of the single-realization covariance; never "
            "divided by 500"
        ),
        "display_observable": (
            "k1*k2*B000 for the full 2D vector; this is k^2*B000 on the "
            "diagonal"
        ),
    }
    arrays = {
        "target": context.target,
        "prediction": conditional.prediction,
        "sigma_single": sigma_single,
        "pull": pull,
        "k_pair": context.k_pair[context.indices],
        "indices": context.indices,
    }
    return payload, arrays


def posterior_prediction_band(
    context: CollapsedContext,
    *,
    diagnosis: dict[str, Any],
    chain_path: Path,
    seed: int,
    draws: int = 300,
    theta_pool: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if theta_pool is None:
        _retained, flat = retained_chain_samples(
            context,
            chain_path=chain_path,
            diagnosis=diagnosis,
        )
    else:
        flat = np.asarray(theta_pool, dtype=np.float64)
        if flat.ndim != 2 or flat.shape[1] != context.theta_dimension:
            raise ValueError("posterior prediction theta pool mismatch")
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        flat.shape[0],
        size=min(draws, flat.shape[0]),
        replace=False,
    )
    predictions = np.empty(
        (selected.size, context.target.size),
        dtype=np.float64,
    )
    for row, sample_index in enumerate(selected):
        theta = flat[sample_index]
        conditional = context.conditional(theta)
        nuisance = conditional.nuisance.copy()
        conditional_covariance = np.linalg.inv(conditional.hessian)
        nuisance[context.linear_indices] += (
            np.linalg.cholesky(conditional_covariance)
            @ rng.normal(size=context.linear_indices.size)
        )
        predictions[row] = context.evaluate_model(
            float(theta[0]),
            nuisance,
        )[0]
    return {
        "lower68": np.quantile(predictions, 0.16, axis=0),
        "median": np.quantile(predictions, 0.50, axis=0),
        "upper68": np.quantile(predictions, 0.84, axis=0),
    }


def local_joint_map_estimator(
    context: CollapsedContext,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    theta = np.asarray(
        diagnosis["profile_centre_only"]["theta"],
        dtype=np.float64,
    )
    conditional = context.conditional(theta)
    nuisance = conditional.nuisance
    centre = np.concatenate((theta[:1], nuisance))
    prediction = conditional.prediction
    jacobian = np.empty(
        (context.target.size, centre.size),
        dtype=np.float64,
    )
    parameter_scales = np.concatenate(
        (
            np.asarray([1.0]),
            np.maximum(
                np.asarray(context.prior.sigma, dtype=np.float64),
                np.abs(nuisance),
            ),
        )
    )
    for index in range(centre.size):
        step = 1.0e-4 * max(float(parameter_scales[index]), 1.0)
        if index == 0:
            step = 0.1
        plus = centre.copy()
        minus = centre.copy()
        plus[index] += step
        minus[index] -= step
        plus_prediction = context.evaluate_model(
            float(plus[0]),
            plus[1:],
        )[0]
        minus_prediction = context.evaluate_model(
            float(minus[0]),
            minus[1:],
        )[0]
        jacobian[:, index] = (
            np.asarray(plus_prediction) - np.asarray(minus_prediction)
        ) / (2.0 * step)
    precision_jacobian = np.linalg.solve(
        context.covariance_single,
        jacobian,
    )
    prior_precision = np.zeros((centre.size, centre.size))
    prior_precision[1:, 1:] = (
        context.prior.whitener.T @ context.prior.whitener
    )
    fisher = (
        context.hartlap * jacobian.T @ precision_jacobian
        + prior_precision
    )
    fisher_inverse = np.linalg.pinv(fisher, rcond=1.0e-11)
    response_operator = (
        fisher_inverse
        @ (
            context.hartlap
            * np.linalg.solve(
                context.covariance_single,
                jacobian,
            ).T
        )
    )
    realization_estimates = (
        float(theta[0])
        + (context.samples - context.target) @ response_operator[0]
    )
    return {
        "centre": float(theta[0]),
        "prediction": prediction,
        "jacobian": jacobian,
        "fisher_condition_number": float(np.linalg.cond(fisher)),
        "realization_estimates": realization_estimates,
        "single_realization_scatter": float(
            np.std(realization_estimates, ddof=1)
        ),
        "method": (
            "local Gauss-Newton/Fisher joint-MAP centre response to each "
            "realization around the mean-data joint MAP, retaining the "
            "full nuisance prior; residual-times-model-second-derivative "
            "Hessian terms are not included"
        ),
    }


def paired_centre_diagnostic(
    pre_context: CollapsedContext,
    post_context: CollapsedContext,
    pre_diagnosis: dict[str, Any],
    post_diagnosis: dict[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(
        pre_context.realization_ids,
        post_context.realization_ids,
    ):
        raise ValueError("paired diagnostic needs identical realization IDs")
    pre = local_joint_map_estimator(pre_context, pre_diagnosis)
    post = local_joint_map_estimator(post_context, post_diagnosis)
    pre_estimates = np.asarray(pre.pop("realization_estimates"))
    post_estimates = np.asarray(post.pop("realization_estimates"))
    differences = post_estimates - pre_estimates
    centre_shift = float(post["centre"] - pre["centre"])
    paired_scatter = float(np.std(differences, ddof=1))
    return {
        "method": (
            "paired local Gauss-Newton/Fisher joint-MAP centre response "
            "over the same 500 realizations; this is an empirical "
            "single-realization linearized centre-shift diagnostic, not "
            "an exact realization-by-realization refit and not a profiler "
            "uncertainty"
        ),
        "realization_ids_identical": True,
        "n_paired_realizations": differences.size,
        "pre": pre,
        "post": post,
        "pre_post_estimator_correlation": float(
            np.corrcoef(pre_estimates, post_estimates)[0, 1]
        ),
        "post_minus_pre_centre": centre_shift,
        "paired_single_realization_scatter": paired_scatter,
        "centre_shift_over_paired_single_realization_scatter": (
            centre_shift / paired_scatter
        ),
        "difference_quantiles_16_50_84": np.quantile(
            differences,
            (0.16, 0.50, 0.84),
        ),
        "uncertainty_policy": (
            "no division by 500 and no profiler/HESSE/MINOS width"
        ),
    }


def save_pdf(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": path.stem,
            "Author": "MARISA-B pre/post halo fNL MCMC audit",
            "Subject": (
                "Single-realization covariance; marginalized MCMC "
                "uncertainties; profiler centre only"
            ),
        },
    )
    plt.close(figure)


def plot_fnl_scan(
    figure_path: Path,
    posterior: dict[tuple[str, float], dict[str, Any]],
    arrays: dict[tuple[str, float], dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    colours = {"pre": "#2166ac", "post": "#b2182b"}
    for axis, kmax in zip(axes.flat, CUTS):
        for reconstruction in RECONSTRUCTIONS:
            values = arrays[(reconstruction, kmax)]["density_fnl"]
            x, density = posterior_density(values, FNL_BOUNDS)
            status = posterior[(reconstruction, kmax)]["fnl"][
                "posterior_status"
            ]
            axis.plot(
                x,
                density,
                color=colours[reconstruction],
                lw=2.0,
                label=f"{reconstruction} ({status.replace('_', ' ')})",
            )
            axis.axvline(
                posterior[(reconstruction, kmax)]["fnl"]["median"],
                color=colours[reconstruction],
                lw=1.0,
                alpha=0.75,
            )
        axis.axvline(0.0, color="0.2", ls=":", lw=1.1)
        axis.axhline(
            1.0 / (FNL_BOUNDS[1] - FNL_BOUNDS[0]),
            color="0.65",
            ls="--",
            lw=1.0,
            label="flat-prior density",
        )
        axis.set_title(
            rf"$k_{{\max}}={kmax:.2f}\,h\,{{\rm Mpc}}^{{-1}}$"
        )
        axis.set_ylabel("marginal density")
        axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel(r"$f_{\rm NL}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        "Quijote halo fiducial: marginalized MCMC posterior",
        y=1.055,
    )
    figure.tight_layout()
    save_pdf(figure, figure_path)


def plot_constraints(
    figure_path: Path,
    posterior: dict[tuple[str, float], dict[str, Any]],
    width_ratios: dict[str, Any],
) -> None:
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )
    colours = {"pre": "#2166ac", "post": "#b2182b"}
    offsets = {"pre": -0.0018, "post": 0.0018}
    for reconstruction in RECONSTRUCTIONS:
        medians = []
        lower68 = []
        upper68 = []
        lower95 = []
        upper95 = []
        widths = []
        information = []
        for kmax in CUTS:
            result = posterior[(reconstruction, kmax)]["fnl"]
            median = float(result["median"])
            interval68 = np.asarray(result["equal_tail_68"])
            interval95 = np.asarray(result["equal_tail_95"])
            medians.append(median)
            lower68.append(median - interval68[0])
            upper68.append(interval68[1] - median)
            lower95.append(median - interval95[0])
            upper95.append(interval95[1] - median)
            widths.append(result["equal_tail_68_half_width"])
            information.append(result["information_gain"]["kl_bits"])
        x = np.asarray(CUTS) + offsets[reconstruction]
        top.errorbar(
            x,
            medians,
            yerr=np.asarray((lower95, upper95)),
            color=colours[reconstruction],
            lw=0.9,
            capsize=2,
            alpha=0.42,
        )
        top.errorbar(
            x,
            medians,
            yerr=np.asarray((lower68, upper68)),
            fmt="o-",
            color=colours[reconstruction],
            lw=1.6,
            capsize=3,
            label=f"{reconstruction}: median, 68% / 95%",
        )
        bottom.plot(
            x,
            widths,
            "o-",
            color=colours[reconstruction],
            lw=1.7,
            label=f"{reconstruction} 68% half-width",
        )
        for kmax, xvalue, bits in zip(CUTS, x, information):
            bottom.annotate(
                f"{bits:.2f} bit",
                (xvalue, widths[list(CUTS).index(kmax)]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=colours[reconstruction],
            )
    top.axhline(0.0, color="0.25", ls=":", lw=1.1)
    top.axhspan(
        FNL_BOUNDS[0],
        FNL_BOUNDS[1],
        color="0.7",
        alpha=0.055,
        label="hard prior support",
    )
    top.set_ylabel(r"$f_{\rm NL}$")
    top.set_ylim(-160, 160)
    top.legend(frameon=False, fontsize=8.5, loc="lower left")
    top.grid(alpha=0.18)
    bottom.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    bottom.set_ylabel("68% half-width")
    bottom.grid(alpha=0.18)
    bottom.legend(frameon=False, fontsize=8.5)
    ratio_text = ", ".join(
        f"{float(kmax):.2f}: {row['post_over_pre_width68']:.2f}"
        for kmax, row in sorted(
            width_ratios.items(),
            key=lambda item: float(item[0]),
        )
    )
    figure.suptitle(
        "Marginalized constraints only\n"
        f"post/pre 68%-width ratios = {ratio_text}\n"
        "(post ratios are prior-dominated, not precision gains)",
        fontsize=11,
    )
    figure.tight_layout()
    save_pdf(figure, figure_path)


def plot_corner_pre_post(
    figure_path: Path,
    pre_samples: np.ndarray,
    post_samples: np.ndarray,
    theta_names: tuple[str, ...],
) -> None:
    labels = {
        "fNL": r"$f_{\rm NL}$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "gamma2": r"$\gamma_2$",
        "gamma21": r"$\gamma_{21}$",
        "b_nabla2_delta": r"$b_{\nabla^2\delta}$",
    }
    plot_labels = [labels.get(name, name) for name in theta_names]
    combined = np.vstack((pre_samples, post_samples))
    ranges: list[tuple[float, float]] = []
    for index, name in enumerate(theta_names):
        if name == "fNL":
            ranges.append(FNL_BOUNDS)
        else:
            low, high = np.quantile(combined[:, index], (0.005, 0.995))
            padding = 0.08 * (high - low)
            ranges.append((float(low - padding), float(high + padding)))
    credible_levels = (0.68, 0.95)
    contour_styles = ("--", "-")
    figure = corner.corner(
        pre_samples,
        labels=plot_labels,
        range=ranges,
        bins=42,
        color="#2166ac",
        smooth=1.15,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        no_fill_contours=True,
        levels=credible_levels,
        truths=[0.0] + [None] * (len(theta_names) - 1),
        truth_color="0.25",
        max_n_ticks=4,
        hist_kwargs={"density": True, "lw": 1.6},
        contour_kwargs={
            "linewidths": 1.3,
            "linestyles": contour_styles,
        },
        contourf_kwargs={
            "colors": (
                (0.0, 0.0, 0.0, 0.0),
                (0.859, 0.914, 0.965, 0.55),
                (0.616, 0.788, 0.898, 0.55),
            ),
            "antialiased": True,
        },
    )
    corner.corner(
        post_samples,
        fig=figure,
        range=ranges,
        bins=42,
        color="#b2182b",
        smooth=1.15,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        no_fill_contours=True,
        levels=credible_levels,
        max_n_ticks=4,
        hist_kwargs={"density": True, "lw": 1.6},
        contour_kwargs={
            "linewidths": 1.3,
            "linestyles": contour_styles,
        },
        contourf_kwargs={
            "colors": (
                (0.0, 0.0, 0.0, 0.0),
                (0.961, 0.843, 0.851, 0.55),
                (0.914, 0.608, 0.624, 0.55),
            ),
            "antialiased": True,
        },
    )
    figure.legend(
        handles=[
            Line2D([], [], color="#2166ac", lw=3, label="pre recon"),
            Line2D([], [], color="#b2182b", lw=3, label="post recon"),
            Line2D(
                [],
                [],
                color="0.2",
                lw=1.3,
                ls="-",
                label="68% credible contour",
            ),
            Line2D(
                [],
                [],
                color="0.2",
                lw=1.3,
                ls="--",
                label="95% credible contour",
            ),
            Line2D(
                [],
                [],
                color="0.25",
                lw=1.1,
                ls="-",
                label=r"$f_{\rm NL}=0$ truth",
            ),
        ],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.98, 0.98),
    )
    figure.suptitle(
        r"Coevolution, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$"
        "\nMarginalized MCMC corner plot; profiler widths are not used",
        y=1.01,
    )
    save_pdf(figure, figure_path)


def plot_profile_vs_marginal(
    figure_path: Path,
    posterior: dict[tuple[str, float], dict[str, Any]],
    arrays: dict[tuple[str, float], dict[str, np.ndarray]],
    wide_arrays: dict[tuple[str, float], dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.3, 7.1), sharex=True)
    colours = {"pre": "#2166ac", "post": "#b2182b"}
    for column, kmax in enumerate((0.08, 0.14)):
        for row, reconstruction in enumerate(RECONSTRUCTIONS):
            axis = axes[row, column]
            main_x, main_density = posterior_density(
                arrays[(reconstruction, kmax)]["density_fnl"],
                FNL_BOUNDS,
            )
            wide_x, wide_density = posterior_density(
                wide_arrays[(reconstruction, kmax)]["density_fnl"],
                WIDE_FNL_BOUNDS,
                bins=120,
            )
            record = posterior[(reconstruction, kmax)]
            profile_centre = record[
                "profile_joint_map_centre_only"
            ]["theta"][0]
            axis.plot(
                wide_x,
                wide_density,
                color="0.60",
                lw=1.3,
                label=r"MCMC, wide prior $[-300,300]$",
            )
            axis.plot(
                main_x,
                main_density,
                color=colours[reconstruction],
                lw=2.0,
                label=r"MCMC, primary prior $[-150,150]$",
            )
            axis.axvline(
                profile_centre,
                color="black",
                ls="--",
                lw=1.5,
                label="joint MAP profiler centre",
            )
            axis.axvline(
                record["fnl"]["median"],
                color=colours[reconstruction],
                ls=":",
                lw=1.4,
                label="marginal median",
            )
            axis.axvline(0.0, color="0.25", lw=0.8, alpha=0.6)
            axis.set_title(
                f"{reconstruction}, "
                + rf"$k_{{\max}}={kmax:.2f}$"
            )
            axis.set_ylabel("marginal density")
            axis.grid(alpha=0.16)
    for axis in axes[-1]:
        axis.set_xlabel(r"$f_{\rm NL}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "The profiler contributes a centre only; all widths are MCMC",
        y=1.065,
    )
    figure.tight_layout()
    save_pdf(figure, figure_path)


def plot_posterior_predictive(
    figure_path: Path,
    predictions: dict[str, dict[str, np.ndarray]],
    prediction_summary: dict[str, dict[str, Any]],
    bands: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.2),
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        sharex="col",
    )
    for column, reconstruction in enumerate(RECONSTRUCTIONS):
        values = predictions[reconstruction]
        order = np.lexsort(
            (
                values["k_pair"][:, 0],
                values["k_pair"][:, 1],
                np.max(values["k_pair"], axis=1),
            )
        )
        scale = np.prod(values["k_pair"], axis=1)
        x = np.arange(order.size)
        target = scale * values["target"]
        prediction = scale * values["prediction"]
        errors = scale * values["sigma_single"]
        band = bands[reconstruction]
        axes[0, column].errorbar(
            x,
            target[order],
            yerr=errors[order],
            fmt="o",
            color="black",
            ms=3.5,
            capsize=2,
            lw=0.8,
            label="Quijote mean ± single-realization error",
        )
        axes[0, column].fill_between(
            x,
            (scale * band["lower68"])[order],
            (scale * band["upper68"])[order],
            color="#d73027",
            alpha=0.16,
            lw=0,
            label="marginal posterior model 68%",
        )
        axes[0, column].plot(
            x,
            prediction[order],
            color="#d73027",
            lw=2.0,
            label="joint-MAP model",
        )
        axes[1, column].plot(
            x,
            values["pull"][order],
            "o-",
            color="#d73027",
            ms=3.2,
            lw=1.1,
        )
        axes[1, column].axhline(0.0, color="black", lw=0.8)
        axes[1, column].axhline(0.5, color="0.5", ls=":", lw=0.9)
        axes[1, column].axhline(-0.5, color="0.5", ls=":", lw=0.9)
        axes[1, column].set_ylim(-0.55, 0.55)
        summary = prediction_summary[reconstruction]
        axes[0, column].set_title(
            f"{reconstruction} recon: "
            + rf"$\max|\Delta/\sigma_{{\rm single}}|="
            + f"{summary['max_abs_diagonal_single_realization_pull']:.3f}$, "
            + rf"$\|C_{{\rm single}}^{{-1/2}}\Delta\|="
            + f"{summary['full_covariance_single_realization_norm']:.3f}$"
        )
        pair_labels = [
            f"({values['k_pair'][index, 0]:.2f},"
            f"{values['k_pair'][index, 1]:.2f})"
            for index in order
        ]
        axes[1, column].set_xticks(x[::2])
        axes[1, column].set_xticklabels(
            pair_labels[::2],
            rotation=65,
            ha="right",
            fontsize=7,
        )
        axes[1, column].set_xlabel(
            r"selected full-2D $(k_1,k_2)$ bin "
            r"$[h\,{\rm Mpc}^{-1}]$"
        )
        axes[0, column].grid(alpha=0.14)
        axes[1, column].grid(alpha=0.14)
    axes[0, 0].set_ylabel(
        r"$k_1k_2 B_{000}\ [h^{-4}{\rm Mpc}^{4}]$"
    )
    axes[1, 0].set_ylabel(
        r"$(B_{\rm model}-B_{\rm mean})/\sigma_{\rm single,diag}$"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        r"Coevolution, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$; "
        "covariance is never divided by 500",
        y=1.065,
    )
    figure.tight_layout()
    save_pdf(figure, figure_path)


def plot_full_vs_coevolution(
    figure_path: Path,
    coevolution_arrays: dict[tuple[str, float], dict[str, np.ndarray]],
    full_arrays: dict[tuple[str, float], dict[str, np.ndarray]],
    cross_checks: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for row, reconstruction in enumerate(RECONSTRUCTIONS):
        for column, kmax in enumerate((0.08, 0.14)):
            axis = axes[row, column]
            for tier, values, colour in (
                (
                    "coevolution",
                    coevolution_arrays[(reconstruction, kmax)][
                        "density_fnl"
                    ],
                    "#2166ac",
                ),
                (
                    "full",
                    full_arrays[(reconstruction, kmax)]["density_fnl"],
                    "#b2182b",
                ),
            ):
                x, density = posterior_density(values, FNL_BOUNDS)
                axis.plot(x, density, color=colour, lw=2.0, label=tier)
            row_key = f"{reconstruction}_kmax{kmax:.2f}"
            row_result = cross_checks[row_key]
            axis.set_title(
                f"{reconstruction}, "
                + rf"$k_{{\max}}={kmax:.2f}$; "
                + r"$|\Delta {\rm med}|/\sigma_{\rm comb}="
                + f"{row_result['median_difference_over_combined_sd']:.3f}$"
            )
            axis.axvline(0.0, color="0.25", ls=":", lw=1.0)
            axis.set_ylabel("marginal density")
            axis.grid(alpha=0.16)
    for axis in axes[-1]:
        axis.set_xlabel(r"$f_{\rm NL}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "Halo one-loop tier cross-check (uplift excluded)",
        y=1.055,
    )
    figure.tight_layout()
    save_pdf(figure, figure_path)


def plot_convergence(
    figure_path: Path,
    posterior: dict[tuple[str, float], dict[str, Any]],
    arrays: dict[tuple[str, float], dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    colours = {"pre": "#2166ac", "post": "#b2182b"}
    for reconstruction in RECONSTRUCTIONS:
        rows = [posterior[(reconstruction, kmax)] for kmax in CUTS]
        axes[0, 0].plot(
            CUTS,
            [row["maximum_rank_rhat"] for row in rows],
            "o-",
            color=colours[reconstruction],
            label=reconstruction,
        )
        axes[0, 1].plot(
            CUTS,
            [
                min(row["minimum_bulk_ess"], row["minimum_tail_ess"])
                for row in rows
            ],
            "o-",
            color=colours[reconstruction],
            label=reconstruction,
        )
        axes[1, 0].plot(
            CUTS,
            [row["minimum_retained_length_over_tau"] for row in rows],
            "o-",
            color=colours[reconstruction],
            label=reconstruction,
        )
        trace = arrays[(reconstruction, 0.14)]["trace"]
        for ensemble in range(trace.shape[0]):
            series = np.median(trace[ensemble], axis=1)
            axes[1, 1].plot(
                np.arange(series.size),
                series,
                color=colours[reconstruction],
                alpha=0.34,
                lw=0.8,
            )
    axes[0, 0].axhline(1.01, color="black", ls="--", lw=1.0)
    axes[0, 0].set_title("maximum rank-normalized split R-hat")
    axes[0, 0].set_ylim(0.998, 1.012)
    axes[0, 1].axhline(1000.0, color="0.45", ls=":", lw=1.0)
    axes[0, 1].set_title("minimum bulk/tail ESS over sampled coordinates")
    axes[0, 1].set_yscale("log")
    axes[1, 0].axhline(50.0, color="black", ls="--", lw=1.0)
    axes[1, 0].set_title("minimum retained length / autocorrelation time")
    axes[1, 1].axhline(0.0, color="black", ls=":", lw=0.8)
    axes[1, 1].set_title(
        r"$f_{\rm NL}$ ensemble-median traces, $k_{\max}=0.14$"
    )
    axes[1, 1].set_xlabel("thinned retained draw")
    axes[1, 1].set_ylabel(r"$f_{\rm NL}$")
    for axis in axes.flat:
        axis.grid(alpha=0.16)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    handles = [
        Line2D([], [], color=colours["pre"], marker="o", label="pre"),
        Line2D([], [], color=colours["post"], marker="o", label="post"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle("MCMC convergence diagnostics", y=1.055)
    figure.tight_layout()
    save_pdf(figure, figure_path)


def render_report(summary: dict[str, Any]) -> str:
    rows = []
    primary = summary["posterior"]["coevolution"]
    for reconstruction in RECONSTRUCTIONS:
        for kmax in CUTS:
            row = primary[reconstruction][f"{kmax:.2f}"]
            fnl = row["fnl"]
            profile_record = row["profile_joint_map_centre_only"]
            profile = profile_record["theta"][0]
            profile_label = (
                f"{profile:.2f}†"
                if profile_record["at_limit"]
                else f"{profile:.2f}"
            )
            interval = fnl["equal_tail_68"]
            rows.append(
                "| "
                f"{reconstruction} | {kmax:.2f} | {profile_label} | "
                f"{fnl['median']:.2f} | "
                f"[{interval[0]:.2f}, {interval[1]:.2f}] | "
                f"{fnl['posterior_status'].replace('_', ' ')} | "
                f"{row['maximum_rank_rhat']:.4f} |"
            )
    paired = summary["paired_pre_post_centre_diagnostic"]
    predictive = summary["posterior_predictive"]
    return "\n".join(
        (
            "# Quijote halo pre/post-reconstruction fNL MCMC",
            "",
            "## Frozen statistical interpretation",
            "",
            "The profiler is used only for the joint-MAP centre. No "
            "profiler, HESSE, or MINOS width appears in this report. All "
            "intervals are converged marginalized-MCMC intervals.",
            "",
            "The likelihood uses the covariance of one realization, never "
            "the covariance of the 500-realization mean.",
            "",
            "## Primary coevolution result",
            "",
            "| recon | kmax | profile centre only | MCMC median | "
            "MCMC equal-tail 68% | status | max Rhat |",
            "|---|---:|---:|---:|---:|---|---:|",
            *rows,
            "",
            "† The joint-MAP optimizer reports at least one fitted "
            "coordinate at a configured boundary for this row. The centre "
            "is retained as a flagged diagnostic only.",
            "",
            "The post-reconstruction posterior is prior-limited across the "
            "scan. Its broad interval is a marginalized nuisance-volume "
            "result and must not be replaced by a profiler width.",
            "",
            "## Paired centre diagnostic",
            "",
            f"At kmax=0.14, the post-minus-pre joint-MAP centre shift is "
            f"{paired['post_minus_pre_centre']:.3f}. The matched-500 "
            "linearized single-realization scatter of that centre "
            f"difference is "
            f"{paired['paired_single_realization_scatter']:.3f}, giving "
            f"{paired['centre_shift_over_paired_single_realization_scatter']:.3f} "
            "single-realization scatter units. This is not a profiler error.",
            "",
            "## Posterior-predictive best fits",
            "",
            f"Pre recon has max diagonal single-realization pull "
            f"{predictive['pre']['max_abs_diagonal_single_realization_pull']:.3f} "
            "and full-covariance norm "
            f"{predictive['pre']['full_covariance_single_realization_norm']:.3f}.",
            "",
            f"Post recon has max diagonal single-realization pull "
            f"{predictive['post']['max_abs_diagonal_single_realization_pull']:.3f} "
            "and full-covariance norm "
            f"{predictive['post']['full_covariance_single_realization_norm']:.3f}.",
            "",
            "## Release status",
            "",
            "The statistical MCMC audit is complete. The post-reconstruction "
            "theory remains provisional because the frozen qmax, quadrature, "
            "and 2916-node projector numerical gates have not been completed. "
            "qmax remains paused.",
            "",
        )
    )


def finalize_products(
    root: Path,
    shared: SharedInputs,
    state: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    chain_path = root / CHAINS_RELATIVE
    figure_dir = root / FIGURE_RELATIVE
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary: dict[tuple[str, float], dict[str, Any]] = {}
    primary_arrays: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    primary_contexts: dict[tuple[str, float], CollapsedContext] = {}
    stages = state["stages"]
    for reconstruction in RECONSTRUCTIONS:
        for cut_index, kmax in enumerate(CUTS):
            context = build_context(
                shared,
                reconstruction,
                "coevolution",
                kmax,
            )
            diagnosis_key = f"diagnose/{context_key(context)}"
            if diagnosis_key not in stages:
                raise KeyError(f"missing {diagnosis_key}")
            diagnosis = stages[diagnosis_key]
            if not diagnosis["pass"]:
                raise RuntimeError(f"unconverged production chain {diagnosis_key}")
            result, arrays = summarize_posterior(
                context,
                diagnosis=diagnosis,
                chain_path=chain_path,
                seed=seed + 100 * cut_index + (1 if reconstruction == "post" else 0),
            )
            primary[(reconstruction, kmax)] = result
            primary_arrays[(reconstruction, kmax)] = arrays
            if math.isclose(kmax, 0.14):
                primary_contexts[(reconstruction, kmax)] = context

    full: dict[tuple[str, float], dict[str, Any]] = {}
    full_arrays: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for reconstruction in RECONSTRUCTIONS:
        for cut_index, kmax in enumerate((0.08, 0.14)):
            context = build_context(shared, reconstruction, "full", kmax)
            diagnosis_key = f"diagnose/{context_key(context)}"
            if diagnosis_key not in stages:
                raise KeyError(f"missing {diagnosis_key}")
            diagnosis = stages[diagnosis_key]
            if not diagnosis["pass"]:
                raise RuntimeError(f"unconverged cross-check {diagnosis_key}")
            result, arrays = summarize_posterior(
                context,
                diagnosis=diagnosis,
                chain_path=chain_path,
                seed=seed + 500 + 100 * cut_index
                + (1 if reconstruction == "post" else 0),
            )
            full[(reconstruction, kmax)] = result
            full_arrays[(reconstruction, kmax)] = arrays

    wide: dict[tuple[str, float], dict[str, Any]] = {}
    wide_arrays: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for reconstruction in RECONSTRUCTIONS:
        for cut_index, kmax in enumerate((0.08, 0.14)):
            context = build_context(
                shared,
                reconstruction,
                "coevolution",
                kmax,
                wide_prior=True,
            )
            diagnosis_key = f"diagnose/{context_key(context)}"
            if diagnosis_key not in stages:
                raise KeyError(f"missing {diagnosis_key}")
            diagnosis = stages[diagnosis_key]
            if not diagnosis["pass"]:
                raise RuntimeError(f"unconverged wide-prior chain {diagnosis_key}")
            result, arrays = summarize_posterior(
                context,
                diagnosis=diagnosis,
                chain_path=chain_path,
                seed=seed + 900 + 100 * cut_index
                + (1 if reconstruction == "post" else 0),
            )
            wide[(reconstruction, kmax)] = result
            wide_arrays[(reconstruction, kmax)] = arrays
            main_width = primary[(reconstruction, kmax)]["fnl"][
                "equal_tail_68_half_width"
            ]
            wide_width = result["fnl"]["equal_tail_68_half_width"]
            sensitivity = {
                "wide_bounds": WIDE_FNL_BOUNDS,
                "median_shift_wide_minus_primary": (
                    result["fnl"]["median"]
                    - primary[(reconstruction, kmax)]["fnl"]["median"]
                ),
                "width68_ratio_wide_to_primary": wide_width / main_width,
            }
            primary[(reconstruction, kmax)]["fnl"][
                "wide_prior_sensitivity"
            ] = sensitivity
            if (
                primary[(reconstruction, kmax)]["fnl"][
                    "posterior_status"
                ]
                != "prior_limited"
                and sensitivity["width68_ratio_wide_to_primary"] > 1.10
            ):
                primary[(reconstruction, kmax)]["fnl"][
                    "posterior_status"
                ] = "prior_sensitive"

    width_ratios: dict[str, Any] = {}
    for kmax in CUTS:
        pre_width = primary[("pre", kmax)]["fnl"][
            "equal_tail_68_half_width"
        ]
        post_width = primary[("post", kmax)]["fnl"][
            "equal_tail_68_half_width"
        ]
        width_ratios[f"{kmax:.2f}"] = {
            "pre_width68": pre_width,
            "post_width68": post_width,
            "post_over_pre_width68": post_width / pre_width,
            "interpretation": (
                "descriptive marginalized-width ratio only; post is "
                "prior-limited and this is not a data-only precision ratio"
            ),
        }

    cross_checks: dict[str, Any] = {}
    for reconstruction in RECONSTRUCTIONS:
        for kmax in (0.08, 0.14):
            coevolution = primary[(reconstruction, kmax)]["fnl"]
            full_result = full[(reconstruction, kmax)]["fnl"]
            median_difference = abs(
                coevolution["median"] - full_result["median"]
            )
            combined_sd = math.hypot(
                coevolution["standard_deviation"],
                full_result["standard_deviation"],
            )
            cross_checks[f"{reconstruction}_kmax{kmax:.2f}"] = {
                "absolute_median_difference": median_difference,
                "combined_marginal_sd": combined_sd,
                "median_difference_over_combined_sd": (
                    median_difference / combined_sd
                ),
                "pass_below_0p3": bool(
                    median_difference / combined_sd < 0.3
                ),
            }

    prediction_summary: dict[str, dict[str, Any]] = {}
    prediction_arrays: dict[str, dict[str, np.ndarray]] = {}
    prediction_bands: dict[str, dict[str, np.ndarray]] = {}
    for reconstruction in RECONSTRUCTIONS:
        context = primary_contexts[(reconstruction, 0.14)]
        diagnosis = stages[f"diagnose/{context_key(context)}"]
        prediction_summary[reconstruction], prediction_arrays[
            reconstruction
        ] = bestfit_prediction(context, diagnosis)
        prediction_bands[reconstruction] = posterior_prediction_band(
            context,
            diagnosis=diagnosis,
            chain_path=chain_path,
            seed=seed + (13 if reconstruction == "pre" else 17),
            theta_pool=primary_arrays[(reconstruction, 0.14)]["corner"],
        )

    paired = paired_centre_diagnostic(
        primary_contexts[("pre", 0.14)],
        primary_contexts[("post", 0.14)],
        stages[
            f"diagnose/{context_key(primary_contexts[('pre', 0.14)])}"
        ],
        stages[
            f"diagnose/{context_key(primary_contexts[('post', 0.14)])}"
        ],
    )

    figure_paths = {
        "fnl_posterior_pre_vs_post_kmax_scan": (
            figure_dir / "fnl_posterior_pre_vs_post_kmax_scan.pdf"
        ),
        "fnl_constraint_vs_kmax": (
            figure_dir / "fnl_constraint_vs_kmax.pdf"
        ),
        "corner_pre_vs_post_kmax0p14": (
            figure_dir / "corner_pre_vs_post_kmax0p14.pdf"
        ),
        "profile_vs_marginal_posterior": (
            figure_dir / "profile_vs_marginal_posterior.pdf"
        ),
        "posterior_predictive_b000_pre_post": (
            figure_dir / "posterior_predictive_b000_pre_post.pdf"
        ),
        "full_vs_coevolution_posterior": (
            figure_dir / "full_vs_coevolution_posterior.pdf"
        ),
        "mcmc_convergence_diagnostics": (
            figure_dir / "mcmc_convergence_diagnostics.pdf"
        ),
    }
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    plot_fnl_scan(
        figure_paths["fnl_posterior_pre_vs_post_kmax_scan"],
        primary,
        primary_arrays,
    )
    plot_constraints(
        figure_paths["fnl_constraint_vs_kmax"],
        primary,
        width_ratios,
    )
    plot_corner_pre_post(
        figure_paths["corner_pre_vs_post_kmax0p14"],
        primary_arrays[("pre", 0.14)]["corner"],
        primary_arrays[("post", 0.14)]["corner"],
        tuple(primary[("pre", 0.14)]["context"]["theta_names"]),
    )
    plot_profile_vs_marginal(
        figure_paths["profile_vs_marginal_posterior"],
        primary,
        primary_arrays,
        wide_arrays,
    )
    plot_posterior_predictive(
        figure_paths["posterior_predictive_b000_pre_post"],
        prediction_arrays,
        prediction_summary,
        prediction_bands,
    )
    plot_full_vs_coevolution(
        figure_paths["full_vs_coevolution_posterior"],
        primary_arrays,
        full_arrays,
        cross_checks,
    )
    plot_convergence(
        figure_paths["mcmc_convergence_diagnostics"],
        primary,
        primary_arrays,
    )

    def nested(
        source: dict[tuple[str, float], dict[str, Any]],
        cuts: tuple[float, ...],
    ) -> dict[str, Any]:
        return {
            reconstruction: {
                f"{kmax:.2f}": source[(reconstruction, kmax)]
                for kmax in cuts
            }
            for reconstruction in RECONSTRUCTIONS
        }

    validation_rows = {
        key: value
        for key, value in stages.items()
        if key.startswith("validate/")
    }
    explicit_rows = {
        key: value
        for key, value in stages.items()
        if key.startswith("explicit/")
    }
    student_rows = {
        key: value
        for key, value in stages.items()
        if key.startswith("student_t/")
    }
    statistical_checks = {
        "all_primary_chains_converged": bool(
            all(row["converged"] for row in primary.values())
        ),
        "truth_zero_inside_all_primary_95_intervals": bool(
            all(
                row["fnl"]["truth_zero_inside_95"]
                for row in primary.values()
            )
        ),
        "posterior_predictive_pre_pass": bool(
            all(prediction_summary["pre"]["acceptance"].values())
        ),
        "posterior_predictive_post_pass": bool(
            all(prediction_summary["post"]["acceptance"].values())
        ),
        "full_coevolution_cross_checks_pass": bool(
            all(row["pass_below_0p3"] for row in cross_checks.values())
        ),
        "analytic_validation_pass": bool(
            validation_rows
            and all(row["pass"] for row in validation_rows.values())
        ),
        "explicit_invariance_pass": bool(
            explicit_rows
            and all(row["pass"] for row in explicit_rows.values())
        ),
        "student_t_robustness_pass": bool(
            student_rows
            and all(row["pass"] for row in student_rows.values())
        ),
    }
    independent_reviews = stages.get(
        "independent_reviews",
        {
            "required": 3,
            "completed": 0,
            "all_pass": False,
            "status": "pending",
            "reviews": [],
        },
    )
    summary = {
        "schema": "marisa-b-prepost-halo-fnl-mcmc-summary-v1",
        "created_utc": utc_now(),
        "contract": str(shared.root / CONTRACT_RELATIVE),
        "scientific_scope": {
            "sample": (
                "Quijote halo fiducial fNL=0, z=1, real space, R=15, "
                "Mmin=1e13"
            ),
            "observable": "full two-dimensional B000(k1,k2)",
            "realizations": 500,
            "covariance": (
                "covariance of one realization; never divided by 500"
            ),
            "precision": "Hartlap corrected after each final mask",
            "primary_model": "coevolution",
            "cross_check_model": "full",
            "excluded_model": "uplift",
            "p": 1.0,
            "fnl_prior": FNL_BOUNDS,
            "profiler_policy": (
                "joint MAP centre only; all profiler/HESSE/MINOS widths "
                "are prohibited"
            ),
            "uncertainty_policy": (
                "converged marginalized MCMC quantiles and HDIs only"
            ),
        },
        "posterior": {
            "coevolution": nested(primary, CUTS),
            "full": nested(full, (0.08, 0.14)),
            "coevolution_wide_prior": nested(wide, (0.08, 0.14)),
        },
        "post_over_pre_marginalized_width_ratios": width_ratios,
        "full_vs_coevolution": cross_checks,
        "paired_pre_post_centre_diagnostic": paired,
        "posterior_predictive": prediction_summary,
        "robustness": {
            "explicit_all_parameter_invariance": explicit_rows,
            "student_t": student_rows,
            "analytic_and_logdet_validation": validation_rows,
        },
        "statistical_checks": statistical_checks,
        "statistical_audit_pass": bool(all(statistical_checks.values())),
        "independent_reviews": independent_reviews,
        "scientific_release_gate": {
            "pass": False,
            "status": "provisional_unvalidated_numerical_convergence",
            "reason": (
                "post-reconstruction 256-node theory still lacks the "
                "frozen qmax, quadrature, and 2916-node projector gates"
            ),
            "qmax_status": "paused_by_user",
        },
        "figure_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in figure_paths.items()
        },
        "provenance": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "contract": {
                "path": str(shared.root / CONTRACT_RELATIVE),
                "sha256": sha256(shared.root / CONTRACT_RELATIVE),
            },
            "chains": {
                "path": str(chain_path),
                "sha256": sha256(chain_path),
            },
            "matrix": {
                "path": str(shared.matrix),
                "sha256": shared.matrix_hash,
            },
            "audit": stages.get("audit", {}),
        },
    }
    summary_path = root / SUMMARY_RELATIVE
    atomic_json(summary_path, summary)
    report_path = root / REPORT_RELATIVE
    report_path.write_text(render_report(jsonable(summary)), encoding="utf-8")
    release = {
        "schema": "marisa-b-prepost-halo-fnl-mcmc-release-v1",
        "created_utc": utc_now(),
        "status": "provisional_unvalidated_numerical_convergence",
        "statistical_audit_pass": summary["statistical_audit_pass"],
        "scientific_acceptance": False,
        "profiler_policy": (
            "centre only; no profiler uncertainty is reported or used"
        ),
        "uncertainty_source": "converged marginalized MCMC only",
        "blocking_numerical_gates": [
            "qmax scan (paused by user)",
            "quadrature convergence",
            "2916-node projector convergence",
        ],
        "hashes": {
            "summary": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
            "report": {
                "path": str(report_path),
                "sha256": sha256(report_path),
            },
            "chains": summary["provenance"]["chains"],
            "figures": summary["figure_hashes"],
        },
        "required_independent_reviews": {
            "count": 3,
            "completed": independent_reviews.get("completed", 0),
            "all_pass": independent_reviews.get("all_pass", False),
            "status": independent_reviews.get("status", "pending"),
            "reviews": independent_reviews.get("reviews", []),
        },
    }
    release_path = root / RELEASE_RELATIVE
    atomic_json(release_path, release)
    return {
        "summary_path": summary_path,
        "release_path": release_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
        "statistical_audit_pass": summary["statistical_audit_pass"],
        "scientific_acceptance": False,
        "release_status": release["status"],
        "profiler_policy": release["profiler_policy"],
    }


def b1_calibration_contract(data_root: Path) -> dict[str, Any]:
    report_path = (
        data_root / "analysis/pre_recon_bias_model_v3_20260725/report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration = report["b1_calibration"]
    primary = calibration["primary"]
    fits: dict[str, dict[str, float]] = {}
    for tier, by_cut in calibration["fits"].items():
        fits[tier] = {}
        for cutoff, fit in by_cut.items():
            b1 = fit["free_parameters"]["b1"]
            fits[tier][cutoff] = {
                "b1": float(b1["value"]),
                "b1_sigma": float(b1["posterior_sigma"]),
            }
    return {
        "source": {
            "path": str(report_path),
            "sha256": sha256(report_path),
        },
        "primary": {
            "tier": str(primary["tier"]),
            "kmax_h_mpc": float(primary["kmax_h_mpc"]),
            "b1": float(primary["b1"]),
            "b1_sigma": float(primary["b1_sigma"]),
        },
        "fits": fits,
    }


def context_with_b1_prior(
    context: CollapsedContext,
    *,
    mean: float,
    sigma: float,
) -> CollapsedContext:
    if not math.isfinite(mean):
        raise ValueError("b1 prior mean must be finite")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("b1 prior sigma must be positive and finite")
    location = context.names.index("b1")
    prior_mean = np.asarray(context.prior.mean, dtype=np.float64).copy()
    covariance = np.asarray(
        context.prior.covariance,
        dtype=np.float64,
    ).copy()
    prior_mean[location] = float(mean)
    covariance[location, :] = 0.0
    covariance[:, location] = 0.0
    covariance[location, location] = float(sigma) ** 2
    whitener = solve_triangular(
        np.linalg.cholesky(covariance),
        np.eye(len(context.names)),
        lower=True,
    )
    prior = pre_parameter_helpers.PriorBlock(
        names=context.names,
        mean=prior_mean,
        covariance=covariance,
        sigma=np.sqrt(np.diag(covariance)),
        whitener=whitener,
    )
    return replace(context, prior=prior)


def gaussian_nuisance_map(
    context: CollapsedContext,
) -> tuple[dict[str, Any], ConditionalResult]:
    names = context.nonlinear_names
    base = np.asarray(
        context.prior.mean[context.nonlinear_indices],
        dtype=np.float64,
    )

    def objective(*values: float) -> float:
        theta = np.concatenate(
            (np.asarray([0.0]), np.asarray(values, dtype=np.float64))
        )
        return context.profile_objective(theta)

    offsets = []
    first = np.zeros_like(base)
    offsets.append(first)
    for sign in (1.0, -1.0):
        offset = np.zeros_like(base)
        for name, scale in (
            ("b1", 0.45),
            ("b2", 0.8),
            ("gamma2", -0.55),
            ("b_nabla2_delta", 0.25),
            ("gamma21", 0.35),
        ):
            if name in names:
                offset[names.index(name)] = sign * scale
        offsets.append(offset)
    mixed = np.zeros_like(base)
    for name, scale in (
        ("b1", -0.75),
        ("b2", 1.1),
        ("gamma2", -0.9),
        ("b_nabla2_delta", -0.4),
        ("gamma21", 0.6),
    ):
        if name in names:
            mixed[names.index(name)] = scale
    offsets.extend((mixed, -mixed))

    minimizers: list[Minuit] = []
    for offset in offsets:
        start = base + offset
        b1_location = names.index("b1")
        start[b1_location] = np.clip(
            start[b1_location],
            B1_BOUNDS[0] + 0.05,
            B1_BOUNDS[1] - 0.05,
        )
        minimizer = Minuit(
            objective,
            *[float(value) for value in start],
            name=names,
        )
        minimizer.errordef = 1.0
        minimizer.tol = 1.0e-5
        minimizer.limits["b1"] = B1_BOUNDS
        for name, prior_index in zip(
            names,
            context.nonlinear_indices,
        ):
            minimizer.errors[name] = max(
                0.02,
                0.1 * float(context.prior.sigma[prior_index]),
            )
        minimizer.migrad(ncall=40000)
        if not minimizer.valid:
            minimizer.simplex(ncall=10000)
            minimizer.migrad(ncall=60000)
        minimizers.append(minimizer)
    valid = [item for item in minimizers if item.valid]
    selected = min(
        valid if valid else minimizers,
        key=lambda item: float(item.fval),
    )
    theta = np.concatenate(
        (
            np.asarray([0.0]),
            np.asarray(
                [selected.values[name] for name in names],
                dtype=np.float64,
            ),
        )
    )
    conditional = context.conditional(theta)
    record = {
        "fnl_fixed": 0.0,
        "theta_names": context.theta_names,
        "theta": theta,
        "conditional_nuisance": conditional.nuisance,
        "objective": float(selected.fval),
        "data_q": conditional.data_q,
        "prior_q": conditional.prior_q,
        "valid": bool(selected.valid),
        "at_limit": bool(selected.fmin.has_parameters_at_limit),
        "multistart_objectives": [
            float(item.fval) for item in minimizers
        ],
        "multistart_valid": [bool(item.valid) for item in minimizers],
    }
    return record, conditional


def model_jacobian_at_zero(
    context: CollapsedContext,
    nuisance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if nuisance.shape != (len(context.names),):
        raise ValueError("nuisance vector shape mismatch")

    def model(fnl: float, values: np.ndarray) -> np.ndarray:
        return np.asarray(
            context.evaluate_model(float(fnl), values)[0],
            dtype=np.float64,
        )

    response_steps = (0.5, 1.0, 2.0)
    responses = {
        step: (model(step, nuisance) - model(-step, nuisance))
        / (2.0 * step)
        for step in response_steps
    }
    response = responses[1.0]
    jacobian = np.empty(
        (context.target.size, 1 + len(context.names)),
        dtype=np.float64,
    )
    jacobian[:, 0] = response
    nuisance_steps: dict[str, float] = {}
    for column, name in enumerate(context.names, start=1):
        location = column - 1
        step = max(
            1.0e-5,
            1.0e-4
            * max(
                1.0,
                abs(float(nuisance[location])),
                float(context.prior.sigma[location]),
            ),
        )
        plus = nuisance.copy()
        minus = nuisance.copy()
        plus[location] += step
        minus[location] -= step
        jacobian[:, column] = (
            model(0.0, plus) - model(0.0, minus)
        ) / (2.0 * step)
        nuisance_steps[name] = step
    scale = max(float(np.linalg.norm(response)), 1.0)
    step_closure = max(
        float(np.linalg.norm(responses[step] - response)) / scale
        for step in response_steps
    )
    _prediction, _parameters, components = context.evaluate_model(
        0.0,
        nuisance,
    )
    registered = np.asarray(
        components["linear_total"],
        dtype=np.float64,
    )[context.indices]
    registered_closure = float(
        np.linalg.norm(registered - response) / scale
    )
    return jacobian, {
        "fNL_central_steps": response_steps,
        "maximum_relative_step_closure": step_closure,
        "registered_linear_total_relative_closure": registered_closure,
        "nuisance_steps": nuisance_steps,
    }


def fisher_subset_sigma(
    fisher: np.ndarray,
    scales: np.ndarray,
    nuisance_indices: Iterable[int],
) -> float:
    nuisance = np.asarray(tuple(nuisance_indices), dtype=int)
    keep = np.concatenate((np.asarray([0]), 1 + nuisance))
    local_scales = scales[keep]
    scaled = (
        fisher[np.ix_(keep, keep)]
        * local_scales[:, None]
        * local_scales[None, :]
    )
    covariance_scaled = np.linalg.inv(scaled)
    covariance = (
        covariance_scaled
        * local_scales[:, None]
        * local_scales[None, :]
    )
    variance = float(covariance[0, 0])
    if variance <= 0.0 or not math.isfinite(variance):
        raise np.linalg.LinAlgError("invalid Fisher fNL variance")
    return math.sqrt(variance)


def fisher_from_jacobian(
    model_context: CollapsedContext,
    covariance_context: CollapsedContext,
    jacobian: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if not np.array_equal(
        model_context.indices,
        covariance_context.indices,
    ):
        raise ValueError("Fisher covariance swap has mismatched bins")
    whitened = solve_triangular(
        covariance_context.chol_fit,
        jacobian,
        lower=True,
    )
    fisher = whitened.T @ whitened
    fisher[1:, 1:] += (
        model_context.prior.whitener.T
        @ model_context.prior.whitener
    )
    scales = np.concatenate(
        (
            np.asarray([100.0]),
            np.maximum(
                np.asarray(model_context.prior.sigma, dtype=np.float64),
                1.0e-3,
            ),
        )
    )
    nonlinear = np.asarray(
        [
            model_context.names.index(name)
            for name in model_context.nonlinear_names
        ],
        dtype=int,
    )
    linear = np.asarray(
        [
            index
            for index in range(len(model_context.names))
            if index not in set(nonlinear)
        ],
        dtype=int,
    )
    all_indices = np.arange(len(model_context.names), dtype=int)
    b1_index = model_context.names.index("b1")
    without_b1 = all_indices[all_indices != b1_index]
    fixed_sigma = 1.0 / math.sqrt(float(fisher[0, 0]))
    all_sigma = fisher_subset_sigma(fisher, scales, all_indices)
    nonlinear_sigma = fisher_subset_sigma(fisher, scales, nonlinear)
    linear_sigma = fisher_subset_sigma(fisher, scales, linear)
    b1_fixed_sigma = fisher_subset_sigma(
        fisher,
        scales,
        without_b1,
    )
    scaled_full = fisher * scales[:, None] * scales[None, :]
    eigenvalues = np.linalg.eigvalsh(scaled_full)
    response_white = whitened[:, 0]
    response_norm = float(np.linalg.norm(response_white))
    individual = []
    leave_one_fixed = []
    for nuisance_index, name in enumerate(model_context.names):
        nuisance_white = whitened[:, nuisance_index + 1]
        nuisance_norm = float(np.linalg.norm(nuisance_white))
        cosine = (
            float(response_white @ nuisance_white)
            / (response_norm * nuisance_norm)
            if nuisance_norm > 0.0
            else 0.0
        )
        individual_sigma = fisher_subset_sigma(
            fisher,
            scales,
            (nuisance_index,),
        )
        remaining = all_indices[all_indices != nuisance_index]
        remaining_sigma = fisher_subset_sigma(
            fisher,
            scales,
            remaining,
        )
        individual.append(
            {
                "name": name,
                "data_shape_cosine": cosine,
                "sigma_fNL_with_only_this_nuisance": individual_sigma,
                "degradation_over_fixed": (
                    individual_sigma / fixed_sigma
                ),
            }
        )
        leave_one_fixed.append(
            {
                "name": name,
                "sigma_fNL_when_this_nuisance_is_fixed": remaining_sigma,
                "improvement_relative_to_all_free": (
                    all_sigma - remaining_sigma
                ),
            }
        )
    individual.sort(
        key=lambda row: row["degradation_over_fixed"],
        reverse=True,
    )
    leave_one_fixed.sort(
        key=lambda row: row["improvement_relative_to_all_free"],
        reverse=True,
    )
    return {
        "covariance_reconstruction": covariance_context.reconstruction,
        "hartlap_factor": covariance_context.hartlap,
        "fixed_nuisance_sigma_fNL": fixed_sigma,
        "nonlinear_only_sigma_fNL": nonlinear_sigma,
        "linear_only_sigma_fNL": linear_sigma,
        "all_nuisance_sigma_fNL": all_sigma,
        "b1_fixed_all_other_nuisance_sigma_fNL": b1_fixed_sigma,
        "information_retention_after_all_nuisance": (
            fixed_sigma / all_sigma
        )
        ** 2,
        "degradation_all_over_fixed": all_sigma / fixed_sigma,
        "scaled_fisher_minimum_eigenvalue": float(eigenvalues[0]),
        "scaled_fisher_condition_number": float(
            eigenvalues[-1] / eigenvalues[0]
        ),
        "strongest_individual_degeneracies": individual[:10],
        "largest_leave_one_fixed_improvements": leave_one_fixed[:10],
    }, fisher


def fixed_fisher_sigma(
    response: np.ndarray,
    covariance_context: CollapsedContext,
) -> float:
    whitened = solve_triangular(
        covariance_context.chol_fit,
        np.asarray(response, dtype=np.float64),
        lower=True,
    )
    return 1.0 / float(np.linalg.norm(whitened))


def b1_prior_importance_reweight(
    context: CollapsedContext,
    *,
    chain_path: Path,
    diagnosis: dict[str, Any],
    new_mean: float,
    new_sigma: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    chains, _accepted = load_context_chains(
        chain_path,
        context_key(context),
    )
    burn = int(diagnosis["chain_layout"]["burn_draws"])
    retained = chains[:, burn:, :, :]
    flat = retained.reshape(-1, context.theta_dimension)
    b1_theta_index = 1 + context.nonlinear_names.index("b1")
    old_mean = float(
        context.prior.mean[context.names.index("b1")]
    )
    old_sigma = float(
        context.prior.sigma[context.names.index("b1")]
    )

    def weights_for(samples: np.ndarray) -> np.ndarray:
        b1 = samples[:, b1_theta_index]
        log_weights = (
            -0.5 * ((b1 - new_mean) / new_sigma) ** 2
            + 0.5 * ((b1 - old_mean) / old_sigma) ** 2
        )
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= float(np.sum(weights))
        return weights

    weights = weights_for(flat)
    ess = 1.0 / float(weights @ weights)
    probabilities = (0.025, 0.16, 0.5, 0.84, 0.975)
    fnl_quantiles = weighted_quantile(
        flat[:, 0],
        probabilities,
        weights,
    )
    b1_quantiles = weighted_quantile(
        flat[:, b1_theta_index],
        (0.16, 0.5, 0.84),
        weights,
    )
    ensemble_rows = []
    for ensemble_index in range(retained.shape[0]):
        samples = retained[ensemble_index].reshape(
            -1,
            context.theta_dimension,
        )
        ensemble_weights = weights_for(samples)
        quantiles = weighted_quantile(
            samples[:, 0],
            (0.16, 0.5, 0.84),
            ensemble_weights,
        )
        ensemble_rows.append(
            {
                "ensemble": ensemble_index,
                "effective_sample_size": (
                    1.0 / float(ensemble_weights @ ensemble_weights)
                ),
                "fnl_equal_tail_68": (
                    float(quantiles[0]),
                    float(quantiles[2]),
                ),
                "fnl_median": float(quantiles[1]),
                "fnl_equal_tail_68_half_width": float(
                    0.5 * (quantiles[2] - quantiles[0])
                ),
            }
        )
    weighted_mean = np.sum(weights[:, None] * flat, axis=0)
    centered = flat - weighted_mean
    weighted_covariance = (centered * weights[:, None]).T @ centered
    weighted_correlation = weighted_covariance / np.sqrt(
        np.outer(
            np.diag(weighted_covariance),
            np.diag(weighted_covariance),
        )
    )
    record = {
        "method": (
            "exact prior-ratio importance reweight of the converged "
            "b1-explicit collapsed MCMC chain"
        ),
        "old_prior": {
            "mean": old_mean,
            "sigma": old_sigma,
        },
        "new_power_spectrum_prior": {
            "mean": float(new_mean),
            "sigma": float(new_sigma),
        },
        "sample_count": flat.shape[0],
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / flat.shape[0],
        "probabilities": probabilities,
        "fnl_quantiles": fnl_quantiles,
        "fnl_equal_tail_68_half_width": float(
            0.5 * (fnl_quantiles[3] - fnl_quantiles[1])
        ),
        "b1_equal_tail_68_and_median": b1_quantiles,
        "ensemble_stability": ensemble_rows,
        "weighted_correlation_matrix": weighted_correlation,
        "theta_names": context.theta_names,
    }
    return record, {
        "flat": flat,
        "weights": weights,
        "retained": retained,
    }


def posterior_response_volume_audit(
    context: CollapsedContext,
    *,
    chain_path: Path,
    diagnosis: dict[str, Any],
    seed: int,
    draws: int = 256,
) -> dict[str, Any]:
    _retained, flat = retained_chain_samples(
        context,
        chain_path=chain_path,
        diagnosis=diagnosis,
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        flat.shape[0],
        size=min(draws, flat.shape[0]),
        replace=False,
    )
    fixed_sigmas = []
    b1_values = []
    bphi_values = []
    fnl_values = []
    for sample_index in selected:
        theta = flat[sample_index]
        conditional = context.conditional(theta)
        step = 0.5
        plus = np.asarray(
            context.evaluate_model(
                float(theta[0] + step),
                conditional.nuisance,
            )[0],
            dtype=np.float64,
        )
        minus = np.asarray(
            context.evaluate_model(
                float(theta[0] - step),
                conditional.nuisance,
            )[0],
            dtype=np.float64,
        )
        response = (plus - minus) / (2.0 * step)
        fixed_sigmas.append(
            fixed_fisher_sigma(response, context)
        )
        b1 = float(
            conditional.nuisance[context.names.index("b1")]
        )
        b1_values.append(b1)
        bphi_values.append(2.0 * post_model.DELTA_C * (b1 - 1.0))
        fnl_values.append(float(theta[0]))
    probabilities = (0.16, 0.5, 0.84)
    return {
        "draws": len(selected),
        "sampling": "uniform subsample of retained marginalized MCMC draws",
        "fixed_nuisance_local_sigma_fNL_quantiles": np.quantile(
            fixed_sigmas,
            probabilities,
        ),
        "b1_quantiles": np.quantile(b1_values, probabilities),
        "bphi_quantiles": np.quantile(bphi_values, probabilities),
        "fnl_quantiles": np.quantile(fnl_values, probabilities),
        "correlation_fnl_fixed_sigma": float(
            np.corrcoef(fnl_values, fixed_sigmas)[0, 1]
        ),
        "correlation_b1_fixed_sigma": float(
            np.corrcoef(b1_values, fixed_sigmas)[0, 1]
        ),
    }


def covariance_correlation_metrics(
    context: CollapsedContext,
) -> dict[str, float]:
    diagonal = np.sqrt(np.diag(context.covariance_single))
    correlation = (
        context.covariance_single
        / diagonal[:, None]
        / diagonal[None, :]
    )
    off_diagonal = correlation[
        ~np.eye(correlation.shape[0], dtype=bool)
    ]
    return {
        "median_single_realization_sigma": float(
            np.median(diagonal)
        ),
        "mean_absolute_offdiagonal_correlation": float(
            np.mean(np.abs(off_diagonal))
        ),
        "maximum_absolute_offdiagonal_correlation": float(
            np.max(np.abs(off_diagonal))
        ),
        "condition_number": float(
            np.linalg.cond(context.covariance_single)
        ),
    }


def plot_fisher_decomposition(
    path: Path,
    rows: list[dict[str, Any]],
    covariance_swap: dict[str, float],
    posterior_reweight: dict[str, dict[str, Any]],
) -> None:
    cuts = np.asarray([row["kmax_h_mpc"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    colors = {"pre": "#2166ac", "post": "#b2182b"}

    axes[0, 0].plot(
        cuts,
        [row["pre"]["tight_b1_fisher"]["fixed_nuisance_sigma_fNL"]
         for row in rows],
        "o-",
        color=colors["pre"],
        label="pre, actual response",
    )
    axes[0, 0].plot(
        cuts,
        [row["post"]["tight_b1_fisher"]["fixed_nuisance_sigma_fNL"]
         for row in rows],
        "s-",
        color=colors["post"],
        label="post, actual response",
    )
    axes[0, 0].plot(
        cuts,
        [row["post"]["no_reconstruction_response_fisher"][
            "fixed_nuisance_sigma_fNL"
        ] for row in rows],
        "^-",
        color="black",
        label="post, shift-PNG response removed",
    )
    axes[0, 0].set(
        xlabel=r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$",
        ylabel=r"fixed-nuisance Fisher $\sigma(f_{\rm NL})$",
        title="Response + covariance",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        cuts,
        [row["pre"]["tight_b1_fisher"]["all_nuisance_sigma_fNL"]
         for row in rows],
        "o-",
        color=colors["pre"],
        label="pre, all nuisance",
    )
    axes[0, 1].plot(
        cuts,
        [row["post"]["tight_b1_fisher"]["all_nuisance_sigma_fNL"]
         for row in rows],
        "s-",
        color=colors["post"],
        label="post, all nuisance",
    )
    axes[0, 1].plot(
        cuts,
        [row["post"]["no_reconstruction_response_fisher"][
            "all_nuisance_sigma_fNL"
        ] for row in rows],
        "^-",
        color="black",
        label="post, shift-PNG response removed",
    )
    axes[0, 1].set(
        xlabel=r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$",
        ylabel=r"marginal Fisher $\sigma(f_{\rm NL})$",
        title=r"With $P_0$ calibration of $b_1$",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    swap_labels = (
        "pre R / pre C",
        "pre R / post C",
        "post R / pre C",
        "post R / post C",
        "post no-shift R / post C",
    )
    swap_keys = (
        "pre_response_pre_covariance",
        "pre_response_post_covariance",
        "post_response_pre_covariance",
        "post_response_post_covariance",
        "post_no_reconstruction_response_post_covariance",
    )
    swap_values = [covariance_swap[key] for key in swap_keys]
    axes[1, 0].bar(
        np.arange(len(swap_values)),
        swap_values,
        color=("#2166ac", "#67a9cf", "#ef8a62", "#b2182b", "0.25"),
    )
    axes[1, 0].set_xticks(
        np.arange(len(swap_values)),
        swap_labels,
        rotation=25,
        ha="right",
    )
    axes[1, 0].set(
        ylabel=r"fixed-nuisance Fisher $\sigma(f_{\rm NL})$",
        title=r"$k_{\max}=0.14$ covariance swap",
    )

    x = np.arange(2)
    width = 0.24
    current = [
        posterior_reweight[name]["current_mcmc_equal_tail_68_half_width"]
        for name in ("pre", "post")
    ]
    reweighted = [
        posterior_reweight[name][
            "p0_reweighted"
        ]["fnl_equal_tail_68_half_width"]
        for name in ("pre", "post")
    ]
    fisher = [
        rows[-1][name]["tight_b1_fisher"]["all_nuisance_sigma_fNL"]
        for name in ("pre", "post")
    ]
    axes[1, 1].bar(
        x - width,
        current,
        width,
        label=r"MCMC, broad $\sigma(b_1)=2$",
        color="0.65",
    )
    axes[1, 1].bar(
        x,
        reweighted,
        width,
        label=r"MCMC, $P_0$-reweighted",
        color=("#67a9cf", "#ef8a62"),
    )
    axes[1, 1].bar(
        x + width,
        fisher,
        width,
        label=r"local Fisher, $P_0$ prior",
        color=("#2166ac", "#b2182b"),
    )
    axes[1, 1].set_xticks(x, ("pre", "post"))
    axes[1, 1].set(
        ylabel=r"$68\%$ half-width / Fisher $\sigma(f_{\rm NL})$",
        title=r"$k_{\max}=0.14$ posterior-volume test",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Why post-reconstruction halo fNL constraints are weaker",
        fontsize=14,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": "Pre/post halo fNL Fisher decomposition",
            "Author": "MARISA-B Fisher audit",
            "Subject": (
                "Single-realization covariance; Fisher is a diagnostic, "
                "not a replacement for marginalized MCMC"
            ),
        },
    )
    plt.close(figure)


def plot_response_components(
    path: Path,
    *,
    pre_context: CollapsedContext,
    post_context: CollapsedContext,
    pre_response: np.ndarray,
    post_response: np.ndarray,
    post_reconstruction: np.ndarray,
) -> None:
    if not np.array_equal(pre_context.indices, post_context.indices):
        raise ValueError("response component plot bin mismatch")
    k_pair = pre_context.k_pair[pre_context.indices]
    scale = k_pair[:, 0] * k_pair[:, 1]
    post_no_reconstruction = post_response - post_reconstruction
    x = np.arange(k_pair.shape[0])
    sigma_post = np.sqrt(np.diag(post_context.covariance_single))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    axes[0].plot(
        x,
        scale * pre_response,
        color="#2166ac",
        lw=1.8,
        label="pre total",
    )
    axes[0].plot(
        x,
        scale * post_response,
        color="#b2182b",
        lw=1.8,
        label="post total",
    )
    axes[0].plot(
        x,
        scale * post_no_reconstruction,
        color="black",
        lw=1.5,
        label="post without shift-field PNG response",
    )
    axes[0].plot(
        x,
        scale * post_reconstruction,
        color="#f4a582",
        ls="--",
        lw=1.5,
        label="post shift-field PNG response",
    )
    axes[0].axhline(0.0, color="0.3", lw=0.7)
    axes[0].set_ylabel(
        r"$k_1k_2\,\partial B_{000}/\partial f_{\rm NL}$"
    )
    axes[0].set_title(
        r"Gaussian-MAP local-PNG response, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$"
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    axes[1].plot(
        x,
        post_response / sigma_post,
        color="#b2182b",
        lw=1.8,
        label="post total",
    )
    axes[1].plot(
        x,
        post_no_reconstruction / sigma_post,
        color="black",
        lw=1.5,
        label="post without shift-field PNG response",
    )
    axes[1].plot(
        x,
        post_reconstruction / sigma_post,
        color="#f4a582",
        ls="--",
        lw=1.5,
        label="shift-field PNG response",
    )
    axes[1].axhline(0.0, color="0.3", lw=0.7)
    axes[1].set_ylabel(
        r"response / diagonal $\sigma_{\rm single}$"
    )
    tick_locations = x[::2]
    axes[1].set_xticks(
        tick_locations,
        [
            rf"({a:.3f},{b:.3f})"
            for a, b in k_pair[::2]
        ],
        rotation=55,
        ha="right",
    )
    axes[1].set_xlabel(
        r"full-2D bin $(k_1,k_2)\,[h\,{\rm Mpc}^{-1}]$"
    )
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": "Post-reconstruction PNG response cancellation",
            "Author": "MARISA-B Fisher audit",
            "Subject": "All error normalization uses one realization",
        },
    )
    plt.close(figure)


def plot_b1_prior_volume(
    path: Path,
    posterior_arrays: dict[str, dict[str, np.ndarray]],
    calibration: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    colors = {"pre": "#2166ac", "post": "#b2182b"}
    b1_mean = float(calibration["primary"]["b1"])
    b1_sigma = float(calibration["primary"]["b1_sigma"])
    b1_grid = np.linspace(0.1, 4.2, 500)
    for reconstruction in ("pre", "post"):
        flat = posterior_arrays[reconstruction]["flat"]
        b1 = flat[:, 1]
        axes[0].hist(
            b1,
            bins=90,
            range=(0.1, 4.2),
            density=True,
            histtype="step",
            lw=1.7,
            color=colors[reconstruction],
            label=f"{reconstruction}, broad-prior MCMC",
        )
    axes[0].plot(
        b1_grid,
        norm.pdf(b1_grid, loc=b1_mean, scale=b1_sigma),
        color="black",
        lw=1.5,
        label=r"$P_0$ calibration",
    )
    axes[0].set(
        xlabel=r"$b_1$",
        ylabel="marginal density",
        title=r"Broad $B_{000}$-only prior opens a low-$b_1$ volume",
        xlim=(0.1, 4.2),
    )
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.linspace(FNL_BOUNDS[0], FNL_BOUNDS[1], 51)
    for reconstruction in ("pre", "post"):
        flat = posterior_arrays[reconstruction]["flat"]
        weights = posterior_arrays[reconstruction]["weights"]
        axes[1].hist(
            flat[:, 0],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.3,
            color=colors[reconstruction],
            alpha=0.45,
            label=f"{reconstruction}, broad prior",
        )
        axes[1].hist(
            flat[:, 0],
            bins=bins,
            weights=weights,
            density=True,
            histtype="step",
            lw=1.9,
            color=colors[reconstruction],
            label=f"{reconstruction}, $P_0$-reweighted",
        )
    axes[1].axvline(0.0, color="black", lw=0.8)
    axes[1].set(
        xlabel=r"$f_{\rm NL}$",
        ylabel="marginal density",
        title=r"Exact $b_1$ prior-ratio reweighting",
        xlim=FNL_BOUNDS,
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.suptitle(
        r"$k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$",
        fontsize=13,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": "b1 prior-volume audit",
            "Author": "MARISA-B Fisher audit",
            "Subject": "Converged marginalized MCMC chains",
        },
    )
    plt.close(figure)


def fisher_report_text(summary: dict[str, Any]) -> str:
    row = summary["kmax_scan"][-1]
    pre = row["pre"]["tight_b1_fisher"]
    post = row["post"]["tight_b1_fisher"]
    no_reconstruction = row["post"][
        "no_reconstruction_response_fisher"
    ]
    swap = summary["kmax_0p14_covariance_swap"]
    pre_reweight = summary["posterior_volume"]["pre"]["p0_reweighted"]
    post_reweight = summary["posterior_volume"]["post"]["p0_reweighted"]
    current_pre = summary["posterior_volume"]["pre"][
        "current_mcmc_equal_tail_68_half_width"
    ]
    current_post = summary["posterior_volume"]["post"][
        "current_mcmc_equal_tail_68_half_width"
    ]
    return f"""# Pre/post-reconstruction halo fNL Fisher audit

## Result

The post-reconstruction covariance is not the cause of the weaker constraint.
At `kmax=0.14 h/Mpc`, applying the post covariance to the pre response changes
the fixed-nuisance Fisher width from
`{swap['pre_response_pre_covariance']:.3f}` to
`{swap['pre_response_post_covariance']:.3f}`.  Thus the post covariance alone
would improve the constraint.

The current post finite-PNG response contains a negative shift-field response.
At the same cutoff the actual post fixed-nuisance width is
`{post['fixed_nuisance_sigma_fNL']:.3f}`, whereas removing only that response
gives `{no_reconstruction['fixed_nuisance_sigma_fNL']:.3f}`.  With all
nuisance parameters marginalized and the power-spectrum calibration of `b1`,
the corresponding widths are `{post['all_nuisance_sigma_fNL']:.3f}` and
`{no_reconstruction['all_nuisance_sigma_fNL']:.3f}`.  The actual pre result is
`{pre['all_nuisance_sigma_fNL']:.3f}`.

The original bispectrum-only MCMC used `sigma(b1)=2`.  Its equal-tail 68%
half-widths were `{current_pre:.3f}` (pre) and `{current_post:.3f}` (post).
Exact importance reweighting to the measured power-spectrum calibration gives
`{pre_reweight['fnl_equal_tail_68_half_width']:.3f}` and
`{post_reweight['fnl_equal_tail_68_half_width']:.3f}`.  This removes an
additional post posterior-volume penalty, but post remains weaker because its
response is more degenerate with the Gaussian halo nuisance basis.

## Why this differs from the usual tree-level expectation

[Shirasaki et al. (2021)](https://arxiv.org/abs/2010.04567) forecast the PNG
signal from the primordial `B3` term and assume the linear Kaiser/bias factor
is tightly constrained by the power spectrum.  In that scope `Z1_rec=Z1`, so
the tree-level fNL derivative is reconstruction independent.

The current MARISA-B model is broader.  It also includes the local-PNG halo
bias operators `fNL*b_phi*phi` and `fNL*b_phi_delta*phi*delta`.  Standard
reconstruction estimates its displacement from the halo density itself,

`s(k) = -i k W(k) delta_h(k)/(b_rec k^2)`.

Therefore `delta_h`'s `b_phi` response also changes the displacement.  At
second order,

`Z2_rec = K2 + 0.5*K1_i*K1_j*(S_i+S_j)`,

and differentiating the two `K1` factors generates the registered negative
`halo_reconstruction` response.  The native implementation passes its
finite-difference, product-rule and `R -> infinity` algebra checks.  This
explains why the simpler forecast and the current model need not agree.

## Statistical interpretation

All covariances, error normalizations and Fisher matrices use the covariance
of one 1 `(Gpc/h)^3` realization with the Hartlap precision correction.  The
500-realization mean is used only as the low-noise central data vector; the
covariance is never divided by 500.

The Fisher matrix is a local diagnostic, not a replacement for the converged
marginalized MCMC uncertainty.  It decomposes the result into covariance,
response and nuisance-degeneracy effects.  The importance-reweighted MCMC
provides the nonlinear posterior cross-check.

## Remaining scientific gate

The halo `fNL=+/-100` standard-reconstruction catalogues remain quarantined.
Consequently the response cancellation is algebraically verified and is the
prediction of the current model, but it has not yet passed an empirical halo
post-reconstruction PNG-response validation.  A reliable matched halo PNG
reconstruction response is required before treating the post model as a
production fNL likelihood.

## Outputs

- `fisher_response_covariance_decomposition.pdf`
- `png_response_components_kmax0p14.pdf`
- `b1_prior_volume.pdf`
- the pre/post five-parameter MCMC corner plot remains in the parent MCMC
  diagnostic package
"""


def run_fisher_audit(
    root: Path,
    shared: SharedInputs,
    state: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    calibration = b1_calibration_contract(shared.data_root)
    b1_mean = float(calibration["primary"]["b1"])
    b1_sigma = float(calibration["primary"]["b1_sigma"])
    if not math.isclose(
        b1_mean,
        B1_PRIOR_MEAN,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("power-spectrum b1 centre changed")

    rows: list[dict[str, Any]] = []
    plot_contexts: dict[str, CollapsedContext] = {}
    plot_responses: dict[str, np.ndarray] = {}
    maximum_response_closure = 0.0
    minimum_fisher_eigenvalue = math.inf
    for kmax in CUTS:
        contexts = {
            reconstruction: build_context(
                shared,
                reconstruction,
                "coevolution",
                float(kmax),
            )
            for reconstruction in RECONSTRUCTIONS
        }
        tight_contexts = {
            reconstruction: context_with_b1_prior(
                context,
                mean=b1_mean,
                sigma=b1_sigma,
            )
            for reconstruction, context in contexts.items()
        }
        row: dict[str, Any] = {
            "kmax_h_mpc": float(kmax),
            "n_data": contexts["pre"].target.size,
        }
        response_vectors: dict[str, np.ndarray] = {}
        for reconstruction in RECONSTRUCTIONS:
            weak_map, weak_conditional = gaussian_nuisance_map(
                contexts[reconstruction]
            )
            tight_map, tight_conditional = gaussian_nuisance_map(
                tight_contexts[reconstruction]
            )
            jacobian, derivative_check = model_jacobian_at_zero(
                tight_contexts[reconstruction],
                tight_conditional.nuisance,
            )
            fisher, _matrix = fisher_from_jacobian(
                tight_contexts[reconstruction],
                tight_contexts[reconstruction],
                jacobian,
            )
            maximum_response_closure = max(
                maximum_response_closure,
                derivative_check["maximum_relative_step_closure"],
                derivative_check[
                    "registered_linear_total_relative_closure"
                ],
            )
            minimum_fisher_eigenvalue = min(
                minimum_fisher_eigenvalue,
                fisher["scaled_fisher_minimum_eigenvalue"],
            )
            response_vectors[reconstruction] = jacobian[:, 0].copy()
            row[reconstruction] = {
                "covariance": covariance_correlation_metrics(
                    tight_contexts[reconstruction]
                ),
                "weak_b1_prior_gaussian_map": weak_map,
                "tight_b1_prior_gaussian_map": tight_map,
                "tight_b1_fisher": fisher,
                "derivative_checks": derivative_check,
            }
            if math.isclose(float(kmax), 0.14):
                plot_contexts[reconstruction] = tight_contexts[
                    reconstruction
                ]
                plot_responses[reconstruction] = jacobian[:, 0].copy()

        post_context = tight_contexts["post"]
        _prediction, _parameters, post_components = (
            post_context.evaluate_model(
                0.0,
                np.asarray(
                    row["post"]["tight_b1_prior_gaussian_map"][
                        "conditional_nuisance"
                    ],
                    dtype=np.float64,
                ),
            )
        )
        reconstruction_response = np.asarray(
            post_components["halo_reconstruction"],
            dtype=np.float64,
        )[post_context.indices]
        no_reconstruction_jacobian, _check = model_jacobian_at_zero(
            post_context,
            np.asarray(
                row["post"]["tight_b1_prior_gaussian_map"][
                    "conditional_nuisance"
                ],
                dtype=np.float64,
            ),
        )
        no_reconstruction_jacobian[:, 0] -= reconstruction_response
        no_reconstruction_fisher, _matrix = fisher_from_jacobian(
            post_context,
            post_context,
            no_reconstruction_jacobian,
        )
        row["post"][
            "no_reconstruction_response_fisher"
        ] = no_reconstruction_fisher
        row["post"]["response_components"] = {
            "total_l2": float(
                np.linalg.norm(response_vectors["post"])
            ),
            "reconstruction_l2": float(
                np.linalg.norm(reconstruction_response)
            ),
            "without_reconstruction_l2": float(
                np.linalg.norm(
                    response_vectors["post"]
                    - reconstruction_response
                )
            ),
            "post_covariance_cosine_total_reconstruction": float(
                np.dot(
                    solve_triangular(
                        post_context.chol_fit,
                        response_vectors["post"],
                        lower=True,
                    ),
                    solve_triangular(
                        post_context.chol_fit,
                        reconstruction_response,
                        lower=True,
                    ),
                )
                / (
                    np.linalg.norm(
                        solve_triangular(
                            post_context.chol_fit,
                            response_vectors["post"],
                            lower=True,
                        )
                    )
                    * np.linalg.norm(
                        solve_triangular(
                            post_context.chol_fit,
                            reconstruction_response,
                            lower=True,
                        )
                    )
                )
            ),
        }
        row["covariance_only_same_pre_response_sigma_ratio_post_to_pre"] = (
            fixed_fisher_sigma(
                response_vectors["pre"],
                tight_contexts["post"],
            )
            / fixed_fisher_sigma(
                response_vectors["pre"],
                tight_contexts["pre"],
            )
        )
        row["median_diagonal_sigma_ratio_post_to_pre"] = float(
            np.median(
                np.sqrt(
                    np.diag(
                        tight_contexts["post"].covariance_single
                    )
                    / np.diag(
                        tight_contexts["pre"].covariance_single
                    )
                )
            )
        )
        rows.append(row)
        if math.isclose(float(kmax), 0.14):
            plot_responses["post_reconstruction"] = (
                reconstruction_response
            )

    pre_context = plot_contexts["pre"]
    post_context = plot_contexts["post"]
    covariance_swap = {
        "pre_response_pre_covariance": fixed_fisher_sigma(
            plot_responses["pre"],
            pre_context,
        ),
        "pre_response_post_covariance": fixed_fisher_sigma(
            plot_responses["pre"],
            post_context,
        ),
        "post_response_pre_covariance": fixed_fisher_sigma(
            plot_responses["post"],
            pre_context,
        ),
        "post_response_post_covariance": fixed_fisher_sigma(
            plot_responses["post"],
            post_context,
        ),
        "post_no_reconstruction_response_post_covariance": (
            fixed_fisher_sigma(
                plot_responses["post"]
                - plot_responses["post_reconstruction"],
                post_context,
            )
        ),
    }
    post_whitened_pre = solve_triangular(
        post_context.chol_fit,
        plot_responses["pre"],
        lower=True,
    )
    post_whitened_post = solve_triangular(
        post_context.chol_fit,
        plot_responses["post"],
        lower=True,
    )
    covariance_swap["post_covariance_cosine_pre_post_response"] = float(
        np.dot(post_whitened_pre, post_whitened_post)
        / (
            np.linalg.norm(post_whitened_pre)
            * np.linalg.norm(post_whitened_post)
        )
    )

    chain_path = root / CHAINS_RELATIVE
    posterior_volume: dict[str, Any] = {}
    posterior_arrays: dict[str, dict[str, np.ndarray]] = {}
    for reconstruction in RECONSTRUCTIONS:
        context = build_context(
            shared,
            reconstruction,
            "coevolution",
            0.14,
        )
        diagnosis_key = (
            f"diagnose/{reconstruction}_coevolution_kmax0p14"
        )
        diagnosis = state["stages"][diagnosis_key]
        reweight, arrays = b1_prior_importance_reweight(
            context,
            chain_path=chain_path,
            diagnosis=diagnosis,
            new_mean=b1_mean,
            new_sigma=b1_sigma,
        )
        equal_tail = np.asarray(
            diagnosis["fnl_summary"]["equal_tail_68"],
            dtype=np.float64,
        )
        posterior_volume[reconstruction] = {
            "current_mcmc": diagnosis["fnl_summary"],
            "current_mcmc_equal_tail_68_half_width": float(
                0.5 * (equal_tail[1] - equal_tail[0])
            ),
            "p0_reweighted": reweight,
            "response_over_current_posterior": (
                posterior_response_volume_audit(
                    context,
                    chain_path=chain_path,
                    diagnosis=diagnosis,
                    seed=seed
                    + (0 if reconstruction == "pre" else 1000),
                )
            ),
        }
        posterior_arrays[reconstruction] = arrays

    verification_path = (
        root
        / "analysis/post_recon_halo_finite_png_v1_20260727"
        / "VERIFICATION.json"
    )
    verification = json.loads(
        verification_path.read_text(encoding="utf-8")
    )
    corner_path = (
        root
        / FIGURE_RELATIVE
        / "corner_pre_vs_post_kmax0p14.pdf"
    )
    if not corner_path.is_file():
        raise FileNotFoundError(corner_path)
    checks = {
        "single_realization_covariance_only": bool(
            all(
                context.covariance_mock_count == 500
                for context in plot_contexts.values()
            )
        ),
        "selected_pre_post_bins_identical": bool(
            np.array_equal(pre_context.indices, post_context.indices)
        ),
        "response_central_step_and_registry_closure_below_1e10": bool(
            maximum_response_closure < 1.0e-10
        ),
        "all_scaled_fisher_matrices_positive_definite": bool(
            minimum_fisher_eigenvalue > 0.0
        ),
        "native_post_png_verification_passed": bool(
            verification["passed"]
        ),
        "importance_reweight_ess_above_5000_each": bool(
            all(
                posterior_volume[reconstruction][
                    "p0_reweighted"
                ]["effective_sample_size"]
                > 5000.0
                for reconstruction in RECONSTRUCTIONS
            )
        ),
        "post_covariance_improves_same_pre_response": bool(
            covariance_swap["pre_response_post_covariance"]
            < covariance_swap["pre_response_pre_covariance"]
        ),
        "shift_png_response_ablation_improves_post": bool(
            rows[-1]["post"]["no_reconstruction_response_fisher"][
                "fixed_nuisance_sigma_fNL"
            ]
            < rows[-1]["post"]["tight_b1_fisher"][
                "fixed_nuisance_sigma_fNL"
            ]
        ),
        "post_all_nuisance_fisher_weaker_than_pre": bool(
            rows[-1]["post"]["tight_b1_fisher"][
                "all_nuisance_sigma_fNL"
            ]
            > rows[-1]["pre"]["tight_b1_fisher"][
                "all_nuisance_sigma_fNL"
            ]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Fisher audit checks failed: {failed}")

    figure_dir = root / FISHER_FIGURE_RELATIVE
    figure_paths = {
        "fisher_response_covariance_decomposition": (
            figure_dir
            / "fisher_response_covariance_decomposition.pdf"
        ),
        "png_response_components_kmax0p14": (
            figure_dir / "png_response_components_kmax0p14.pdf"
        ),
        "b1_prior_volume": figure_dir / "b1_prior_volume.pdf",
    }
    plot_fisher_decomposition(
        figure_paths["fisher_response_covariance_decomposition"],
        rows,
        covariance_swap,
        posterior_volume,
    )
    plot_response_components(
        figure_paths["png_response_components_kmax0p14"],
        pre_context=pre_context,
        post_context=post_context,
        pre_response=plot_responses["pre"],
        post_response=plot_responses["post"],
        post_reconstruction=plot_responses["post_reconstruction"],
    )
    plot_b1_prior_volume(
        figure_paths["b1_prior_volume"],
        posterior_arrays,
        calibration,
    )

    summary = {
        "schema": "marisa-b-prepost-halo-fnl-fisher-audit-v1",
        "created_utc": utc_now(),
        "status": (
            "causal_diagnosis_complete; post halo PNG response "
            "empirical validation pending"
        ),
        "scope": {
            "sample": (
                "Quijote z=1 Mmin=1e13 real-space fiducial halos"
            ),
            "tier": "coevolution",
            "cuts_h_mpc": CUTS,
            "excluded_global_bin": EXCLUDED_GLOBAL_BIN,
            "excluded_shell_centre_h_mpc": EXCLUDED_SHELL_CENTRE,
            "covariance": (
                "single-realization sample covariance from 500 fiducial "
                "boxes with Hartlap precision correction"
            ),
            "fisher_role": (
                "local causal diagnostic only; MCMC supplies posterior "
                "uncertainties"
            ),
        },
        "headline": {
            "covariance_is_not_the_failure": True,
            "post_covariance_alone_improves_same_response": True,
            "current_post_shift_field_png_response_cancels_signal": True,
            "post_response_has_stronger_gaussian_nuisance_degeneracy": True,
            "broad_b1_prior_adds_posterior_volume_penalty": True,
            "p0_b1_prior_does_not_fully_reverse_pre_post_ordering": True,
            "empirical_halo_post_png_response_validation_pending": True,
        },
        "b1_calibration": calibration,
        "kmax_scan": rows,
        "kmax_0p14_covariance_swap": covariance_swap,
        "posterior_volume": posterior_volume,
        "literature_scope": {
            "shirasaki_2021": {
                "url": "https://arxiv.org/abs/2010.04567",
                "forecast_scope": (
                    "fNL derivative from primordial B3 with linear "
                    "Kaiser/bias factor tightly constrained by P(k)"
                ),
                "difference_from_current_model": (
                    "current model additionally propagates bphi and "
                    "bphidelta halo-PNG response through the "
                    "halo-derived reconstruction displacement"
                ),
            },
            "biased_tracer_png_model": {
                "url": "https://arxiv.org/abs/2010.14523",
                "relevance": (
                    "tree halo bispectrum includes linear and quadratic "
                    "fNL sectors and demonstrates prior sensitivity"
                ),
            },
        },
        "implementation_checks": {
            "checks": checks,
            "maximum_response_closure": maximum_response_closure,
            "minimum_scaled_fisher_eigenvalue": (
                minimum_fisher_eigenvalue
            ),
            "post_png_verification": {
                "path": str(verification_path),
                "sha256": sha256(verification_path),
                "passed": verification["passed"],
                "test_count": verification["test_count"],
            },
        },
        "scientific_limit": (
            "The fNL=+/-100 halo standard-reconstruction catalogues are "
            "quarantined. The shift-field response cancellation is an "
            "algebraically verified prediction of the current theory, "
            "not yet an empirically validated halo post-PNG response."
        ),
        "outputs": {
            "figures": figure_paths,
            "existing_corner_plot": corner_path,
        },
        "provenance": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256_before_summary_write": sha256(
                    Path(__file__).resolve()
                ),
            },
            "chains": {
                "path": str(chain_path),
                "sha256": sha256(chain_path),
            },
            "matrix": {
                "path": str(shared.matrix),
                "sha256": shared.matrix_hash,
            },
        },
    }
    summary_path = root / FISHER_SUMMARY_RELATIVE
    atomic_json(summary_path, summary)
    report_path = root / FISHER_REPORT_RELATIVE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        fisher_report_text(summary),
        encoding="utf-8",
    )
    figure_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name, path in figure_paths.items()
    }
    summary["outputs"]["figure_hashes"] = figure_hashes
    summary["outputs"]["report"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
    }
    summary["outputs"]["existing_corner_plot"] = {
        "path": str(corner_path),
        "sha256": sha256(corner_path),
    }
    atomic_json(summary_path, summary)
    return {
        "summary_path": summary_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
        "checks": checks,
        "status": summary["status"],
        "headline": summary["headline"],
    }


def load_adaptive_brec_tree_basis(
    path: Path,
    shared: SharedInputs,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 28:
        raise ValueError(
            "adaptive-brec basis must contain one header and 27 bins"
        )
    header = records[0]
    expected_indices = selected_indices(shared.k_pair, 0.14)
    actual_indices = np.asarray(
        [row.get("index") for row in records[1:]],
        dtype=np.int64,
    )
    expected_reconstruction = {
        "R": 15,
        "b_rec_h": post_model.B_REC_H,
        "cell_size": 8,
        "cic_power": 4,
    }
    if (
        header.get("record") != "header"
        or header.get("schema")
        != "marisa-b-post-r1-adaptive-brec-diagnostic-jsonl-v1"
        or header.get("sector") != "adaptive-brec-tree"
        or header.get("reconstruction") != expected_reconstruction
        or not np.array_equal(actual_indices, expected_indices)
    ):
        raise ValueError("adaptive-brec basis contract mismatch")
    basis = np.zeros(shared.k_pair.shape[0], dtype=np.float64)
    for row in records[1:]:
        index = int(row["index"])
        edges = np.asarray(row["edges"], dtype=np.float64)
        if not np.array_equal(
            edges,
            np.asarray(shared.edges[index], dtype=np.float64).reshape(-1),
        ):
            raise ValueError(
                f"adaptive-brec basis edge mismatch in bin {index}"
            )
        value = float(
            row["adaptive_brec_denominator_tree_basis"]
        )
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite adaptive-brec basis in bin {index}"
            )
        basis[index] = value
    return basis, {
        "path": str(path),
        "sha256": sha256(path),
        "header": header,
        "indices": actual_indices,
    }


def compact_fisher_widths(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(result[name])
        for name in (
            "fixed_nuisance_sigma_fNL",
            "all_nuisance_sigma_fNL",
            "b1_fixed_all_other_nuisance_sigma_fNL",
            "nonlinear_only_sigma_fNL",
            "linear_only_sigma_fNL",
            "information_retention_after_all_nuisance",
            "scaled_fisher_minimum_eigenvalue",
            "scaled_fisher_condition_number",
        )
    }


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    return (
        float(np.dot(left, right) / denominator)
        if denominator > 0.0
        else 0.0
    )


def plot_adaptive_brec_fisher(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    cuts = np.asarray(
        [row["kmax_h_mpc"] for row in rows],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        sharex=True,
    )
    specifications = (
        (
            "fixed_nuisance_sigma_fNL",
            r"fixed-nuisance Fisher $\sigma(f_{\rm NL})$",
        ),
        (
            "all_nuisance_sigma_fNL",
            r"marginal Fisher $\sigma(f_{\rm NL})$",
        ),
    )
    curves = (
        ("pre", "pre reconstruction", "black", "--", "o"),
        (
            "post_original",
            r"post, fixed $b_{\rm rec}$",
            "#b2182b",
            "-",
            "s",
        ),
        (
            "post_adaptive_brec",
            r"post, $f_{\rm NL}^{\rm rec}=f_{\rm NL}$",
            "#2166ac",
            "-",
            "D",
        ),
        (
            "post_shift_removed_upper_limit",
            "post, entire shift response removed (optimistic)",
            "#777777",
            ":",
            "^",
        ),
    )
    for axis, (metric, ylabel) in zip(axes, specifications):
        for key, label, color, linestyle, marker in curves:
            axis.plot(
                cuts,
                [row[key][metric] for row in rows],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5.0,
                label=label,
            )
        axis.set(
            xlabel=r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$",
            ylabel=ylabel,
        )
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8.3)
    figure.suptitle(
        "Local-PNG-aware reconstruction denominator: Fisher forecast\n"
        "single-realization covariance; leading deterministic tree "
        "denominator response only",
        fontsize=12.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": (
                "Local-PNG adaptive reconstruction-bias Fisher forecast"
            ),
            "Author": "MARISA-B adaptive-brec diagnostic",
            "Subject": (
                "Single-realization covariance; leading deterministic "
                "tree denominator derivative only"
            ),
        },
    )
    plt.close(figure)


def adaptive_brec_report_text(summary: dict[str, Any]) -> str:
    headline = summary["headline"]
    lines = [
        "# Local-PNG-aware reconstruction-bias Fisher test",
        "",
        "## Result",
        "",
        (
            "At `kmax=0.14 h/Mpc`, synchronizing the leading deterministic "
            "tree reconstruction denominator with "
            "`fNL_rec=fNL` changes the post-reconstruction all-nuisance "
            f"Fisher width from `{headline['post_original_marginal']:.3f}` "
            f"to `{headline['post_adaptive_marginal']:.3f}`."
        ),
        (
            "This is a "
            f"`{100.0*headline['marginal_fractional_improvement']:.1f}%` "
            "improvement over the current fixed-b_rec post response."
        ),
        (
            "The corresponding fixed-nuisance width changes from "
            f"`{headline['post_original_fixed']:.3f}` to "
            f"`{headline['post_adaptive_fixed']:.3f}`."
        ),
        (
            "For comparison, the pre-reconstruction marginal width is "
            f"`{headline['pre_marginal']:.3f}`, while the deliberately "
            "optimistic ablation that removes the entire shift-field "
            f"response gives `{headline['post_shift_removed_marginal']:.3f}`."
        ),
        "",
        "## kmax scan",
        "",
        (
            "| kmax | pre marginal | post fixed-brec | "
            "post adaptive-brec | entire shift removed |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["kmax_scan"]:
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} "
            f"| {row['pre']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_original']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_adaptive_brec']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_shift_removed_upper_limit']['all_nuisance_sigma_fNL']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical contract",
            "",
            (
                "Every Fisher matrix uses the covariance of one "
                "`(Gpc/h)^3` realization estimated from 500 fiducial "
                "boxes, with the same Hartlap precision correction and "
                "the same power-spectrum b1 prior."
            ),
            "",
            "## Scope",
            "",
            (
                "This small test adds the exact shell-projected derivative "
                "of the deterministic Gaussian tree term caused by "
                "`b_rec(k)=b1_rec+fNL_rec*bphi_rec/M(k)` at "
                "`fNL_rec=0`."
            ),
            (
                "It does not yet include the corresponding denominator "
                "derivatives of halo one-loop diagrams, EFT counterterms, "
                "stochastic reconstruction terms, or finite-fNL quadratic "
                "terms."
            ),
            (
                "It is therefore a controlled leading-response forecast, "
                "not a production likelihood or an empirical validation "
                "of a parameter-dependent reconstruction pipeline."
            ),
            "",
            "## Output",
            "",
            "- `adaptive_brec_fisher_vs_kmax.pdf`",
            "",
        ]
    )
    return "\n".join(lines)


def run_adaptive_brec_fisher(
    root: Path,
    shared: SharedInputs,
) -> dict[str, Any]:
    basis, basis_provenance = load_adaptive_brec_tree_basis(
        root / ADAPTIVE_BREC_BASIS_RELATIVE,
        shared,
    )
    calibration = b1_calibration_contract(shared.data_root)
    b1_mean = float(calibration["primary"]["b1"])
    b1_sigma = float(calibration["primary"]["b1_sigma"])
    bphi_recon = (
        2.0
        * post_model.DELTA_C
        * (post_model.B_REC_H - post_model.P_UNIVERSALITY)
    )
    rows: list[dict[str, Any]] = []
    minimum_eigenvalue = math.inf
    for kmax in CUTS:
        contexts = {
            reconstruction: context_with_b1_prior(
                build_context(
                    shared,
                    reconstruction,
                    "coevolution",
                    float(kmax),
                ),
                mean=b1_mean,
                sigma=b1_sigma,
            )
            for reconstruction in RECONSTRUCTIONS
        }
        fisher_results: dict[str, dict[str, Any]] = {}
        jacobians: dict[str, np.ndarray] = {}
        conditionals: dict[str, ConditionalResult] = {}
        for reconstruction, context in contexts.items():
            _map, conditional = gaussian_nuisance_map(context)
            jacobian, _check = model_jacobian_at_zero(
                context,
                conditional.nuisance,
            )
            fisher, _matrix = fisher_from_jacobian(
                context,
                context,
                jacobian,
            )
            fisher_results[reconstruction] = fisher
            jacobians[reconstruction] = jacobian
            conditionals[reconstruction] = conditional
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(fisher["scaled_fisher_minimum_eigenvalue"]),
            )

        post_context = contexts["post"]
        _prediction, parameters, components = (
            post_context.evaluate_model(
                0.0,
                conditionals["post"].nuisance,
            )
        )
        b1_map = float(parameters["b1"])
        correction = (
            b1_map**4
            * bphi_recon
            / post_model.B_REC_H
            * basis[post_context.indices]
        )
        adaptive_jacobian = jacobians["post"].copy()
        adaptive_jacobian[:, 0] += correction
        adaptive_fisher, _matrix = fisher_from_jacobian(
            post_context,
            post_context,
            adaptive_jacobian,
        )
        shift_response = np.asarray(
            components["halo_reconstruction"],
            dtype=np.float64,
        )[post_context.indices]
        no_shift_jacobian = jacobians["post"].copy()
        no_shift_jacobian[:, 0] -= shift_response
        no_shift_fisher, _matrix = fisher_from_jacobian(
            post_context,
            post_context,
            no_shift_jacobian,
        )
        correction_white = solve_triangular(
            post_context.chol_fit,
            correction,
            lower=True,
        )
        shift_white = solve_triangular(
            post_context.chol_fit,
            shift_response,
            lower=True,
        )
        current_white = solve_triangular(
            post_context.chol_fit,
            jacobians["post"][:, 0],
            lower=True,
        )
        rows.append(
            {
                "kmax_h_mpc": float(kmax),
                "n_data": int(post_context.target.size),
                "gaussian_map_b1": b1_map,
                "covariance_contract": {
                    reconstruction: {
                        "mock_count": int(
                            context.covariance_mock_count
                        ),
                        "is_unscaled_sample_covariance": bool(
                            np.array_equal(
                                context.covariance_single,
                                np.cov(
                                    context.samples,
                                    rowvar=False,
                                    ddof=1,
                                ),
                            )
                        ),
                    }
                    for reconstruction, context in contexts.items()
                },
                "pre": compact_fisher_widths(
                    fisher_results["pre"]
                ),
                "post_original": compact_fisher_widths(
                    fisher_results["post"]
                ),
                "post_adaptive_brec": compact_fisher_widths(
                    adaptive_fisher
                ),
                "post_shift_removed_upper_limit": (
                    compact_fisher_widths(no_shift_fisher)
                ),
                "response_geometry": {
                    "correction_over_shift_whitened_norm": float(
                        np.linalg.norm(correction_white)
                        / np.linalg.norm(shift_white)
                    ),
                    "correction_shift_whitened_cosine": (
                        vector_cosine(correction_white, shift_white)
                    ),
                    "correction_current_whitened_cosine": (
                        vector_cosine(correction_white, current_white)
                    ),
                },
            }
        )

    previous_path = (
        root
        / "analysis/prepost_recon_halo_fnl_fisher_audit_20260728"
        / "summary.json"
    )
    previous = json.loads(
        previous_path.read_text(encoding="utf-8")
    )
    regression_errors = []
    for row, old in zip(rows, previous["kmax_scan"]):
        regression_errors.extend(
            (
                abs(
                    row["pre"][name]
                    - float(old["pre"]["tight_b1_fisher"][name])
                ),
                abs(
                    row["post_original"][name]
                    - float(old["post"]["tight_b1_fisher"][name])
                ),
                abs(
                    row["post_shift_removed_upper_limit"][name]
                    - float(
                        old["post"][
                            "no_reconstruction_response_fisher"
                        ][name]
                    )
                ),
            )
            for name in (
                "fixed_nuisance_sigma_fNL",
                "all_nuisance_sigma_fNL",
            )
        )
    maximum_regression_error = max(
        value
        for group in regression_errors
        for value in group
    )
    last = rows[-1]
    original_marginal = last["post_original"][
        "all_nuisance_sigma_fNL"
    ]
    adaptive_marginal = last["post_adaptive_brec"][
        "all_nuisance_sigma_fNL"
    ]
    checks = {
        "basis_has_exact_27_selected_bins": bool(
            basis_provenance["indices"].size == 27
            and np.all(
                np.isfinite(basis[basis_provenance["indices"]])
            )
        ),
        "all_covariances_are_single_realization_500_box": bool(
            all(
                contract["mock_count"] == 500
                and contract["is_unscaled_sample_covariance"]
                for row in rows
                for contract in row["covariance_contract"].values()
            )
            and all(
                row["n_data"] == EXPECTED_COUNTS[
                    row["kmax_h_mpc"]
                ]
                for row in rows
            )
        ),
        "previous_fisher_regression_below_1e10": bool(
            maximum_regression_error < 1.0e-10
        ),
        "all_fisher_matrices_positive_definite": bool(
            minimum_eigenvalue > 0.0
        ),
        "adaptive_brec_improves_post_at_every_cut": bool(
            all(
                row["post_adaptive_brec"][
                    "all_nuisance_sigma_fNL"
                ]
                < row["post_original"][
                    "all_nuisance_sigma_fNL"
                ]
                for row in rows
            )
        ),
        "adaptive_brec_is_not_mislabeled_as_full_shift_removal": bool(
            all(
                row["post_adaptive_brec"][
                    "all_nuisance_sigma_fNL"
                ]
                > row["post_shift_removed_upper_limit"][
                    "all_nuisance_sigma_fNL"
                ]
                for row in rows
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"adaptive-brec Fisher checks failed: {failed}"
        )

    figure_path = root / ADAPTIVE_BREC_FIGURE_RELATIVE
    plot_adaptive_brec_fisher(figure_path, rows)
    summary = {
        "schema": (
            "marisa-b-post-halo-local-png-adaptive-brec-fisher-v1"
        ),
        "created_utc": utc_now(),
        "status": "leading_tree_fisher_forecast_complete",
        "scope": {
            "sample": (
                "Quijote z=1 Mmin=1e13 fNL=0 real-space halos"
            ),
            "tier": "coevolution",
            "cuts_h_mpc": CUTS,
            "covariance": (
                "single-realization sample covariance from 500 "
                "fiducial boxes with Hartlap precision correction"
            ),
            "adaptive_reconstruction_bias": (
                "b_rec(k)=b1_rec+fNL_rec*bphi_rec/M(k), "
                "evaluated along fNL_rec=fNL at fNL=0"
            ),
            "included_new_response": (
                "exact shell-projected deterministic Gaussian tree "
                "reconstruction-denominator derivative"
            ),
            "omitted_new_responses": (
                "halo Gaussian one-loop, EFT counterterm, stochastic, "
                "fixed-Poisson and finite-fNL quadratic denominator "
                "derivatives"
            ),
        },
        "reconstruction_calibration": {
            "b1_rec": post_model.B_REC_H,
            "bphi_rec": bphi_recon,
            "p": post_model.P_UNIVERSALITY,
            "delta_c": post_model.DELTA_C,
        },
        "headline": {
            "kmax_h_mpc": 0.14,
            "pre_marginal": last["pre"][
                "all_nuisance_sigma_fNL"
            ],
            "post_original_fixed": last["post_original"][
                "fixed_nuisance_sigma_fNL"
            ],
            "post_original_marginal": original_marginal,
            "post_adaptive_fixed": last["post_adaptive_brec"][
                "fixed_nuisance_sigma_fNL"
            ],
            "post_adaptive_marginal": adaptive_marginal,
            "post_shift_removed_marginal": (
                last["post_shift_removed_upper_limit"][
                    "all_nuisance_sigma_fNL"
                ]
            ),
            "marginal_fractional_improvement": (
                1.0 - adaptive_marginal / original_marginal
            ),
            "adaptive_post_over_pre_marginal": (
                adaptive_marginal
                / last["pre"]["all_nuisance_sigma_fNL"]
            ),
        },
        "kmax_scan": rows,
        "checks": checks,
        "maximum_previous_fisher_regression_error": (
            maximum_regression_error
        ),
        "minimum_scaled_fisher_eigenvalue": minimum_eigenvalue,
        "scientific_limit": (
            "This is a local leading-tree response forecast. A complete "
            "parameter-dependent post-reconstruction likelihood requires "
            "the same adaptive denominator in all loop, EFT and stochastic "
            "sectors and requires catalog-level reconstruction treatment."
        ),
        "provenance": {
            "basis": basis_provenance,
            "matrix": {
                "path": str(shared.matrix),
                "sha256": shared.matrix_hash,
            },
            "previous_fisher_summary": {
                "path": str(previous_path),
                "sha256": sha256(previous_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256_before_summary_write": sha256(
                    Path(__file__).resolve()
                ),
            },
        },
        "outputs": {
            "figure": {
                "path": str(figure_path),
                "sha256": sha256(figure_path),
            },
        },
    }
    summary_path = root / ADAPTIVE_BREC_SUMMARY_RELATIVE
    report_path = root / ADAPTIVE_BREC_REPORT_RELATIVE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        adaptive_brec_report_text(summary),
        encoding="utf-8",
    )
    summary["outputs"]["report"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
    }
    atomic_json(summary_path, summary)
    return {
        "summary_path": summary_path,
        "report_path": report_path,
        "figure_path": figure_path,
        "checks": checks,
        "headline": summary["headline"],
        "status": summary["status"],
    }


def adaptive_brec_raw_path(
    root: Path,
    variant: str,
    index: int,
) -> Path:
    return (
        root
        / ADAPTIVE_BREC_FULL_RAW_RELATIVE
        / f"fNLrec_{variant}_bin_{index:03d}.jsonl"
    )


def validate_adaptive_brec_raw(
    path: Path,
    *,
    variant_value: float,
    index: int,
    expected_hashes: dict[str, str],
    expected_edges: np.ndarray,
) -> dict[str, Any]:
    header, bins = post_model.read_jsonl(path)
    adaptive = (
        header.get("reconstruction", {}).get(
            "adaptive_local_png_bias"
        )
        if isinstance(header.get("reconstruction"), dict)
        else None
    )
    if (
        header.get("schema") != ADAPTIVE_BREC_FINITE_SCHEMA
        or header.get("production_candidate") is not False
        or header.get("bin_range") != [index, index + 1]
        or header.get("loop_quadrature") != [8, 32, 24]
        or header.get("q_range") != [0.0001, 6.4]
        or header.get("shell_projection", {}).get("radial_order") != 2
        or header.get("shell_projection", {}).get("angular_order") != 8
        or header.get("shell_projection", {}).get(
            "orientation_orders"
        )
        != [2, 2, 2]
        or header.get("source_hashes") != expected_hashes
        or not isinstance(adaptive, dict)
        or not math.isclose(
            float(adaptive.get("fNL_rec", math.nan)),
            variant_value,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(adaptive.get("bphi_rec", math.nan)),
            ADAPTIVE_BREC_BPHI_REC,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or not math.isclose(
            float(adaptive.get("amplitude", math.nan)),
            variant_value * ADAPTIVE_BREC_BPHI_REC,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or not math.isclose(
            float(adaptive.get("kmin", math.nan)),
            ADAPTIVE_BREC_KMIN,
            rel_tol=0.0,
            abs_tol=2.0e-15,
        )
        or len(bins) != 1
        or bins[0].get("index") != index
        or not np.array_equal(
            np.asarray(bins[0].get("edges"), dtype=np.float64),
            np.asarray(expected_edges, dtype=np.float64).reshape(-1),
        )
    ):
        raise ValueError(
            f"{path} violates the adaptive-brec raw contract"
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "index": index,
        "fNL_rec": variant_value,
        "bytes": path.stat().st_size,
    }


def run_adaptive_brec_full_production(
    root: Path,
    shared: SharedInputs,
    *,
    workers: int,
) -> dict[str, Any]:
    if not 1 <= workers <= 28:
        raise ValueError("adaptive-brec production workers must be 1..28")
    driver = (
        shared.root
        / "build/halo_v1/eft_v2_post_r1_template_driver"
    )
    power = shared.data_root / ADAPTIVE_BREC_POWER_RELATIVE
    png_table = (
        shared.data_root / ADAPTIVE_BREC_PNG_TABLE_RELATIVE
    )
    edge_file = shared.root / "configs/eft_v2_b000_edges.txt"
    for path in (driver, power, png_table, edge_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        "linear_power": sha256(power),
        "edge_file": sha256(edge_file),
        "driver_executable": sha256(driver),
        "png_table": sha256(png_table),
    }
    selected = selected_indices(shared.k_pair, 0.14)
    if selected.tolist() != [
        1, 2, 3, 4, 5, 6,
        15, 16, 17, 18, 19, 20,
        29, 30, 31, 32, 33,
        42, 43, 44, 45,
        54, 55, 56,
        65, 66,
        75,
    ]:
        raise AssertionError("adaptive-brec selected-bin contract drift")
    raw_root = root / ADAPTIVE_BREC_FULL_RAW_RELATIVE
    failure_root = (
        root / ADAPTIVE_BREC_FULL_FAILURE_LOG_RELATIVE
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, value, int(index))
        for index in selected
        for variant, value in ADAPTIVE_BREC_VARIANTS
    ]

    def execute(job: tuple[str, float, int]) -> dict[str, Any]:
        variant, value, index = job
        destination = adaptive_brec_raw_path(
            root, variant, index
        )
        if destination.is_file():
            record = validate_adaptive_brec_raw(
                destination,
                variant_value=value,
                index=index,
                expected_hashes=expected_hashes,
                expected_edges=shared.edges[index],
            )
            record.update(
                {
                    "variant": variant,
                    "reused": True,
                    "elapsed_seconds": 0.0,
                }
            )
            return record
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        command = [
            str(driver),
            str(power),
            str(edge_file),
            "7",
            str(index),
            str(index + 1),
            "2",
            "8",
            "2",
            "2",
            "2",
            "8",
            "32",
            "24",
            "0.0001",
            "6.4",
            "30",
            "15",
            f"{post_model.B_REC_H:.17g}",
            "8",
            "fibonacci",
            str(png_table),
            f"{value:.17g}",
            f"{ADAPTIVE_BREC_BPHI_REC:.17g}",
            f"{ADAPTIVE_BREC_KMIN:.17g}",
        ]
        environment = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MARISA_B_MAX_THREADS",
        ):
            environment[name] = "1"
        started = time.monotonic()
        with temporary.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                check=False,
            )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            failure = (
                failure_root
                / f"fNLrec_{variant}_bin_{index:03d}.log"
            )
            failure.write_text(
                completed.stderr,
                encoding="utf-8",
            )
            raise RuntimeError(
                "adaptive-brec driver failed for "
                f"{variant}/bin {index}; log={failure}"
            )
        if completed.stderr:
            raise RuntimeError(
                "adaptive-brec driver emitted unexpected stderr for "
                f"{variant}/bin {index}: {completed.stderr[:400]}"
            )
        temporary.replace(destination)
        record = validate_adaptive_brec_raw(
            destination,
            variant_value=value,
            index=index,
            expected_hashes=expected_hashes,
            expected_edges=shared.edges[index],
        )
        record.update(
            {
                "variant": variant,
                "reused": False,
                "elapsed_seconds": elapsed,
            }
        )
        return record

    records: list[dict[str, Any]] = []
    launched_utc = utc_now()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(execute, job): job
            for job in jobs
        }
        for completed_count, future in enumerate(
            concurrent.futures.as_completed(futures),
            1,
        ):
            record = future.result()
            records.append(record)
            print(
                "adaptive-brec full production "
                f"{completed_count}/{len(jobs)}: "
                f"{record['variant']} bin {record['index']}",
                flush=True,
            )
    records.sort(key=lambda row: (row["index"], row["fNL_rec"]))
    manifest = {
        "schema": "marisa-b-adaptive-brec-raw-manifest-v1",
        "created_utc": utc_now(),
        "launched_utc": launched_utc,
        "status": "complete",
        "worker_limit": workers,
        "threading_contract": (
            "at most workers child processes; one thread per process"
        ),
        "job_count": len(records),
        "selected_indices": selected,
        "finite_difference_variants": ADAPTIVE_BREC_VARIANTS,
        "bphi_rec": ADAPTIVE_BREC_BPHI_REC,
        "kmin": ADAPTIVE_BREC_KMIN,
        "source_hashes": expected_hashes,
        "jobs": records,
    }
    manifest_path = (
        root / ADAPTIVE_BREC_FULL_MANIFEST_RELATIVE
    )
    atomic_json(manifest_path, manifest)
    return {
        "status": "adaptive_brec_full_raw_complete",
        "manifest_path": manifest_path,
        "job_count": len(records),
        "worker_limit": workers,
        "new_job_count": sum(
            not record["reused"] for record in records
        ),
    }


ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS = (
    "diagrams",
    "counterterms",
    "stochastic",
    "stochastic_density_only",
    "stochastic_noisy_shift",
)
ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS = (
    "bshot_bnabla2_cross",
    "bshot_bnabla2_cross_density_only",
    "bshot_bnabla2_cross_noisy_shift",
)


def sparse_linear_combination(
    left: dict[str, float],
    right: dict[str, float],
    left_weight: float,
    right_weight: float,
) -> dict[str, float]:
    result = {}
    for monomial in sorted(set(left) | set(right)):
        value = (
            left_weight * float(left.get(monomial, 0.0))
            + right_weight * float(right.get(monomial, 0.0))
        )
        if value != 0.0:
            result[monomial] = value
    return result


def response_row_linear_combination(
    left: dict[str, Any],
    right: dict[str, Any],
    left_weight: float,
    right_weight: float,
) -> dict[str, Any]:
    left_metadata = {
        key: value
        for key, value in left.items()
        if key not in ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS
        and key not in ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS
        and key not in {
            "fixed_poisson",
            "adaptive_brec_response_diagnostics",
            "adaptive_brec_step_error",
        }
    }
    right_metadata = {
        key: value
        for key, value in right.items()
        if key not in ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS
        and key not in ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS
        and key not in {
            "fixed_poisson",
            "adaptive_brec_response_diagnostics",
            "adaptive_brec_step_error",
        }
    }
    if left_metadata != right_metadata:
        raise ValueError(
            "adaptive-brec finite-difference rows have mismatched geometry"
        )
    result = json.loads(json.dumps(left_metadata))
    for field in ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS:
        if set(left[field]) != set(right[field]):
            raise ValueError(
                f"adaptive-brec polynomial group {field} changed support"
            )
        result[field] = {
            name: sparse_linear_combination(
                left[field][name],
                right[field][name],
                left_weight,
                right_weight,
            )
            for name in left[field]
        }
    for field in ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS:
        result[field] = sparse_linear_combination(
            left[field],
            right[field],
            left_weight,
            right_weight,
        )
    if set(left["fixed_poisson"]) != set(right["fixed_poisson"]):
        raise ValueError("adaptive-brec fixed-Poisson orders changed")
    result["fixed_poisson"] = {}
    for order in left["fixed_poisson"]:
        left_order = left["fixed_poisson"][order]
        right_order = right["fixed_poisson"][order]
        left_maps = left_order["by_inverse_number_density"]
        right_maps = right_order["by_inverse_number_density"]
        if set(left_maps) != set(right_maps):
            raise ValueError(
                "adaptive-brec fixed-Poisson nbar maps changed"
            )
        combined_order = {
            key: value
            for key, value in left_order.items()
            if key != "by_inverse_number_density"
        }
        if combined_order != {
            key: value
            for key, value in right_order.items()
            if key != "by_inverse_number_density"
        }:
            raise ValueError(
                "adaptive-brec fixed-Poisson topology metadata changed"
            )
        combined_order["by_inverse_number_density"] = {
            key: sparse_linear_combination(
                left_maps[key],
                right_maps[key],
                left_weight,
                right_weight,
            )
            for key in left_maps
        }
        result["fixed_poisson"][order] = combined_order
    enforce_adaptive_stochastic_decomposition(result)
    return result


def enforce_adaptive_stochastic_decomposition(
    row: dict[str, Any],
) -> None:
    row["stochastic"] = {
        name: sparse_linear_combination(
            row["stochastic_density_only"][name],
            row["stochastic_noisy_shift"][name],
            1.0,
            1.0,
        )
        for name in row["stochastic_density_only"]
    }
    row["bshot_bnabla2_cross"] = sparse_linear_combination(
        row["bshot_bnabla2_cross_density_only"],
        row["bshot_bnabla2_cross_noisy_shift"],
        1.0,
        1.0,
    )


def adaptive_response_payload(
    row: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        field: row[field]
        for field in ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS
    }
    payload.update(
        {
            field: row[field]
            for field in ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS
        }
    )
    payload["fixed_poisson"] = row["fixed_poisson"]
    return payload


def adaptive_response_coefficients(
    row: dict[str, Any],
) -> np.ndarray:
    values: list[float] = []
    for field in ADAPTIVE_BREC_NESTED_POLYNOMIAL_FIELDS:
        for polynomial in row[field].values():
            values.extend(float(value) for value in polynomial.values())
    for field in ADAPTIVE_BREC_DIRECT_POLYNOMIAL_FIELDS:
        values.extend(float(value) for value in row[field].values())
    for order in row["fixed_poisson"].values():
        for polynomial in order[
            "by_inverse_number_density"
        ].values():
            values.extend(float(value) for value in polynomial.values())
    return np.asarray(values, dtype=np.float64)


def load_adaptive_raw_row(
    root: Path,
    variant: str,
    value: float,
    index: int,
    manifest: dict[str, Any],
    shared: SharedInputs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = adaptive_brec_raw_path(root, variant, index)
    record = next(
        (
            row
            for row in manifest["jobs"]
            if row["variant"] == variant
            and int(row["index"]) == index
        ),
        None,
    )
    if (
        record is None
        or record["sha256"] != sha256(path)
        or not math.isclose(
            float(record["fNL_rec"]),
            value,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(
            f"adaptive-brec manifest mismatch for {variant}/bin {index}"
        )
    header, bins = post_model.read_jsonl(path)
    if len(bins) != 1:
        raise ValueError(f"{path} is not a one-bin raw response")
    if not np.array_equal(
        np.asarray(bins[0]["edges"], dtype=np.float64),
        np.asarray(shared.edges[index], dtype=np.float64).reshape(-1),
    ):
        raise ValueError(f"{path} changed measurement geometry")
    return header, bins[0]


def write_jsonl_records(
    path: Path,
    header: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                jsonable(header),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        for row in rows:
            stream.write(
                json.dumps(
                    jsonable(row),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def run_adaptive_brec_full_aggregate(
    root: Path,
    shared: SharedInputs,
) -> dict[str, Any]:
    manifest_path = (
        root / ADAPTIVE_BREC_FULL_MANIFEST_RELATIVE
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    selected = selected_indices(shared.k_pair, 0.14)
    if (
        manifest.get("status") != "complete"
        or manifest.get("job_count") != 4 * selected.size
        or manifest.get("selected_indices") != selected.tolist()
    ):
        raise ValueError("adaptive-brec raw manifest is incomplete")
    variant_map = dict(ADAPTIVE_BREC_VARIANTS)
    response_rows = []
    maximum_coarse_fine_relative = 0.0
    maximum_richardson_error_relative = 0.0
    reference_header: dict[str, Any] | None = None
    for raw_index in selected:
        index = int(raw_index)
        loaded = {
            variant: load_adaptive_raw_row(
                root,
                variant,
                value,
                index,
                manifest,
                shared,
            )
            for variant, value in ADAPTIVE_BREC_VARIANTS
        }
        headers = [item[0] for item in loaded.values()]
        normalized_headers = []
        for header in headers:
            normalized = json.loads(json.dumps(header))
            adaptive = normalized["reconstruction"].pop(
                "adaptive_local_png_bias"
            )
            normalized["source_hashes"].pop("driver_executable")
            normalized["bin_range"] = [index, index + 1]
            adaptive.pop("fNL_rec")
            adaptive.pop("amplitude")
            normalized["adaptive_static_contract"] = adaptive
            normalized_headers.append(normalized)
        if any(
            header != normalized_headers[0]
            for header in normalized_headers[1:]
        ):
            raise ValueError(
                f"adaptive-brec raw headers drift in bin {index}"
            )
        if reference_header is None:
            reference_header = headers[0]
        coarse = response_row_linear_combination(
            loaded["p1"][1],
            loaded["m1"][1],
            0.5,
            -0.5,
        )
        fine = response_row_linear_combination(
            loaded["p0p5"][1],
            loaded["m0p5"][1],
            1.0,
            -1.0,
        )
        richardson = response_row_linear_combination(
            fine,
            coarse,
            4.0 / 3.0,
            -1.0 / 3.0,
        )
        step_error = response_row_linear_combination(
            richardson,
            fine,
            1.0,
            -1.0,
        )
        coarse_fine = response_row_linear_combination(
            fine,
            coarse,
            1.0,
            -1.0,
        )
        richardson_values = adaptive_response_coefficients(richardson)
        error_values = adaptive_response_coefficients(step_error)
        coarse_fine_values = adaptive_response_coefficients(
            coarse_fine
        )
        denominator = max(
            float(np.linalg.norm(richardson_values)),
            np.finfo(np.float64).tiny,
        )
        coarse_fine_relative = float(
            np.linalg.norm(coarse_fine_values)
            / denominator
        )
        richardson_error_relative = float(
            np.linalg.norm(error_values) / denominator
        )
        maximum_coarse_fine_relative = max(
            maximum_coarse_fine_relative,
            coarse_fine_relative,
        )
        maximum_richardson_error_relative = max(
            maximum_richardson_error_relative,
            richardson_error_relative,
        )
        richardson["adaptive_brec_response_diagnostics"] = {
            "coarse_step": 1.0,
            "fine_step": 0.5,
            "coarse_fine_coefficient_l2_relative": (
                coarse_fine_relative
            ),
            "richardson_minus_fine_coefficient_l2_relative": (
                richardson_error_relative
            ),
            "richardson_minus_fine_coefficient_max_abs": float(
                np.max(np.abs(error_values))
                if error_values.size
                else 0.0
            ),
        }
        richardson["adaptive_brec_step_error"] = (
            adaptive_response_payload(step_error)
        )
        response_rows.append(richardson)
    assert reference_header is not None
    output_header = json.loads(json.dumps(reference_header))
    output_header["schema"] = (
        post_model.ADAPTIVE_BREC_RESPONSE_SCHEMA
    )
    output_header["model"] = (
        "fiducial-zero derivative of the full post-R1 Gaussian "
        "halo one-loop EFT-v2 template with respect to the local-PNG "
        "reconstruction-bias denominator"
    )
    output_header["production_candidate"] = False
    output_header["reconstruction"].pop(
        "adaptive_local_png_bias"
    )
    output_header["bin_range"] = [
        int(selected.min()),
        int(selected.max()) + 1,
    ]
    output_header["bin_indices"] = selected.tolist()
    output_header["adaptive_brec_response"] = {
        "derivative": "dG/dfNL_rec at fNL_rec=0",
        "method": (
            "symmetric finite difference plus Richardson extrapolation"
        ),
        "finite_difference_steps": [1.0, 0.5],
        "fNL_rec_equals_fNL": True,
        "bphi_rec": ADAPTIVE_BREC_BPHI_REC,
        "kmin": ADAPTIVE_BREC_KMIN,
        "finite_box_rule": (
            "adaptive denominator response is zero below 2*pi/L"
        ),
        "included_sectors": [
            "deterministic_tree",
            "B222",
            "B321I",
            "B321II",
            "B411",
            "EFT_counterterms",
            "density_only_stochastic",
            "noisy_shift_stochastic",
            "fixed_Poisson",
        ],
        "raw_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
    }
    output_header["merged_parts"] = [
        {
            "index": int(row["index"]),
            "raw_sha256": {
                variant: next(
                    item["sha256"]
                    for item in manifest["jobs"]
                    if item["variant"] == variant
                    and int(item["index"]) == int(row["index"])
                )
                for variant, _value in ADAPTIVE_BREC_VARIANTS
            },
        }
        for row in response_rows
    ]
    output_path = (
        root / ADAPTIVE_BREC_FULL_RESPONSE_RELATIVE
    )
    write_jsonl_records(
        output_path,
        output_header,
        response_rows,
    )
    fitter = load_module(
        "_adaptive_brec_full_aggregate_fitter",
        shared.root / FITTER_RELATIVE,
    )
    data = fitter.DataSet.load(shared.matrix)
    templates = post_model.load_partial_templates(
        fitter,
        data,
        output_path,
        adaptive_response=True,
    )
    if templates.tied.header["bin_indices"] != selected.tolist():
        raise ValueError("adaptive response loader changed bin ordering")
    aggregate = {
        "schema": "marisa-b-adaptive-brec-aggregate-audit-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "response": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "bin_count": len(response_rows),
        },
        "maximum_coarse_fine_coefficient_l2_relative": (
            maximum_coarse_fine_relative
        ),
        "maximum_richardson_error_coefficient_l2_relative": (
            maximum_richardson_error_relative
        ),
        "formal_loader": "pass",
    }
    aggregate_path = (
        root
        / ADAPTIVE_BREC_FULL_ARCHIVE_RELATIVE
        / "aggregate_audit.json"
    )
    atomic_json(aggregate_path, aggregate)
    return {
        "status": "adaptive_brec_full_response_complete",
        "response_path": output_path,
        "aggregate_audit_path": aggregate_path,
        "bin_count": len(response_rows),
        "maximum_richardson_error_coefficient_l2_relative": (
            maximum_richardson_error_relative
        ),
    }


def evaluate_sparse_response(
    polynomial: dict[str, float],
    parameters: dict[str, float],
) -> float:
    return sum(
        float(coefficient)
        * post_model.PostTemplateSet._monomial_value(
            monomial,
            parameters,
        )
        for monomial, coefficient in polynomial.items()
    )


def evaluate_adaptive_response_payload(
    payload: dict[str, Any],
    module: ModuleType,
    parameters: dict[str, float],
    number_density: float,
) -> dict[str, float]:
    parameters = dict(parameters)
    parameters.setdefault("a5_mixed", 0.0)
    diagrams = {
        name: evaluate_sparse_response(
            payload["diagrams"][name],
            parameters,
        )
        for name in module.DIAGRAM_NAMES
    }
    counterterm = sum(
        float(parameters[name])
        * evaluate_sparse_response(
            payload["counterterms"][name],
            parameters,
        )
        for name in module.COUNTERTERM_NAMES
    )
    stochastic = 0.0
    for name in module.STOCHASTIC_B_NAMES:
        normalization = (
            number_density**2
            if name in {"Ashot_residual", "a1_pure"}
            else number_density
        )
        stochastic += (
            float(parameters[name])
            * evaluate_sparse_response(
                payload["stochastic"][name],
                parameters,
            )
            / normalization
        )
    stochastic += (
        float(parameters["Bshot_residual"])
        * float(parameters["b_nabla2_delta"])
        * evaluate_sparse_response(
            payload["bshot_bnabla2_cross"],
            parameters,
        )
        / number_density
    )
    fixed_poisson = 0.0
    for order in payload["fixed_poisson"].values():
        for key, polynomial in order[
            "by_inverse_number_density"
        ].items():
            inverse_nbar = abs(int(key.rsplit("^", 1)[1]))
            fixed_poisson += (
                evaluate_sparse_response(polynomial, parameters)
                / number_density**inverse_nbar
            )
    loop = sum(
        diagrams[name] for name in module.DIAGRAM_NAMES[1:]
    )
    total = (
        diagrams["tree"]
        + loop
        + counterterm
        + stochastic
        + fixed_poisson
    )
    return {
        "tree": diagrams["tree"],
        "loop": loop,
        "counterterm": counterterm,
        "stochastic": stochastic,
        "fixed_poisson": fixed_poisson,
        "total": total,
    }


def load_adaptive_brec_full_templates(
    shared: SharedInputs,
) -> tuple[Any, ModuleType, float, dict[int, dict[str, Any]], dict[str, Any]]:
    response_path = (
        shared.output_root / ADAPTIVE_BREC_FULL_RESPONSE_RELATIVE
    )
    fitter = load_module(
        "_adaptive_brec_full_fisher_fitter",
        shared.root / FITTER_RELATIVE,
    )
    data = fitter.DataSet.load(shared.matrix)
    _contract, _frozen_prior, number_density = fitter.load_prior(
        shared.root / CONTRACT_EFT_RELATIVE
    )
    templates = post_model.load_partial_templates(
        fitter,
        data,
        response_path,
        adaptive_response=True,
    )
    header, rows = post_model.read_jsonl(response_path)
    by_index = {int(row["index"]): row for row in rows}
    selected = selected_indices(shared.k_pair, 0.14)
    if (
        sorted(by_index) != selected.tolist()
        or not math.isfinite(number_density)
        or number_density <= 0.0
    ):
        raise ValueError(
            "adaptive-brec response cache is incomplete or has invalid nbar"
        )
    return (
        templates,
        fitter,
        float(number_density),
        by_index,
        {
            "path": str(response_path),
            "sha256": sha256(response_path),
            "header": header,
        },
    )


def adaptive_step_error_components(
    rows: dict[int, dict[str, Any]],
    module: ModuleType,
    parameters: dict[str, float],
    number_density: float,
) -> dict[str, np.ndarray]:
    names = (
        "tree",
        "loop",
        "counterterm",
        "stochastic",
        "fixed_poisson",
        "total",
    )
    result = {
        name: np.zeros(120, dtype=np.float64)
        for name in names
    }
    for index, row in rows.items():
        components = evaluate_adaptive_response_payload(
            row["adaptive_brec_step_error"],
            module,
            parameters,
            number_density,
        )
        for name in names:
            result[name][index] = components[name]
    return result


def plot_adaptive_brec_full_fisher(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    cuts = np.asarray(
        [row["kmax_h_mpc"] for row in rows],
        dtype=np.float64,
    )
    curves = (
        ("pre", "pre reconstruction", "black", "--", "o"),
        (
            "post_original",
            r"post, fixed $b_{\rm rec}$",
            "#b2182b",
            "-",
            "s",
        ),
        (
            "post_tree_adaptive",
            "post, tree-only adaptive",
            "#67a9cf",
            "--",
            "D",
        ),
        (
            "post_full_adaptive",
            "post, full adaptive response",
            "#2166ac",
            "-",
            "D",
        ),
        (
            "post_shift_removed_ablation",
            "post, entire shift response removed (ablation)",
            "#777777",
            ":",
            "^",
        ),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(11.6, 4.5), sharex=True
    )
    for axis, metric, ylabel in (
        (
            axes[0],
            "fixed_nuisance_sigma_fNL",
            r"fixed-nuisance Fisher $\sigma(f_{\rm NL})$",
        ),
        (
            axes[1],
            "all_nuisance_sigma_fNL",
            r"all-nuisance Fisher $\sigma(f_{\rm NL})$",
        ),
    ):
        for key, label, color, linestyle, marker in curves:
            axis.plot(
                cuts,
                [row[key][metric] for row in rows],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5.0,
                label=label,
            )
        axis.set(
            xlabel=r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$",
            ylabel=ylabel,
        )
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8.0)
    figure.suptitle(
        "Adaptive reconstruction-bias response through all Gaussian "
        "post-R1 sectors\nsingle-realization covariance from 500 boxes",
        fontsize=12.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": "Full adaptive reconstruction-bias Fisher forecast",
            "Author": "MARISA-B adaptive-brec experiment",
            "Subject": (
                "Single-realization covariance; full Gaussian post-R1 "
                "adaptive denominator response"
            ),
        },
    )
    plt.close(figure)


def plot_adaptive_brec_response_geometry(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    sectors = (
        "tree",
        "loop",
        "counterterm",
        "stochastic",
        "fixed_poisson",
        "total",
    )
    labels = (
        "tree",
        "1-loop",
        "EFT",
        "stochastic",
        "fixed Poisson",
        "sum",
    )
    colors = (
        "#4393c3",
        "#d6604d",
        "#9970ab",
        "#1b7837",
        "#fdb863",
        "#2166ac",
    )
    last = rows[-1]["response_geometry"]
    positions = np.arange(len(sectors))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    axes[0].bar(
        positions,
        [
            last["sectors"][sector]["whitened_norm"]
            for sector in sectors
        ],
        color=colors,
    )
    axes[0].set(
        ylabel=r"$\|C^{-1/2}\Delta R\|$",
        title=r"added response at $k_{\max}=0.14$",
    )
    axes[1].bar(
        positions,
        [
            last["sectors"][sector][
                "cosine_with_original_post_response"
            ]
            for sector in sectors
        ],
        color=colors,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(
        ylabel="whitened cosine",
        title="alignment with old post response",
        ylim=(-1.05, 1.05),
    )
    axes[2].bar(
        positions,
        [
            last["sectors"][sector][
                "nuisance_projected_information_norm"
            ]
            for sector in sectors
        ],
        color=colors,
    )
    axes[2].set(
        ylabel=r"$1/\sigma_{\rm marg}(f_{\rm NL})$",
        title="sector after nuisance projection",
    )
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Geometry of the adaptive denominator correction "
        "(single-realization metric)",
        fontsize=12.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        metadata={
            "Title": "Adaptive reconstruction-bias response geometry",
            "Author": "MARISA-B adaptive-brec experiment",
            "Subject": (
                "Whitened sector norms, cosines, and nuisance projection"
            ),
        },
    )
    plt.close(figure)


def adaptive_brec_full_report_text(
    summary: dict[str, Any],
) -> str:
    headline = summary["headline"]
    direction = (
        "improves"
        if headline["full_vs_tree_fractional_change"] < 0.0
        else "weakens"
    )
    lines = [
        "# Full adaptive reconstruction-bias response test",
        "",
        "## Headline",
        "",
        (
            "At `kmax=0.14 h/Mpc`, propagating the adaptive "
            "reconstruction denominator through every current Gaussian "
            "post-R1 sector changes the all-nuisance Fisher width from "
            f"`{headline['tree_adaptive_marginal']:.3f}` for the "
            "tree-only adaptive approximation to "
            f"`{headline['full_adaptive_marginal']:.3f}`."
        ),
        (
            f"The full response therefore {direction} the tree-only "
            "forecast by "
            f"`{100.0*abs(headline['full_vs_tree_fractional_change']):.2f}%`."
        ),
        (
            "The old fixed-b_rec post width is "
            f"`{headline['post_original_marginal']:.3f}` and the "
            "pre-reconstruction width is "
            f"`{headline['pre_marginal']:.3f}`."
        ),
        "",
        "## kmax scan",
        "",
        (
            "| kmax | pre | post fixed-brec | tree adaptive | "
            "full adaptive | shift-removed ablation |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["kmax_scan"]:
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} "
            f"| {row['pre']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_original']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_tree_adaptive']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_full_adaptive']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_shift_removed_ablation']['all_nuisance_sigma_fNL']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Included derivative",
            "",
            (
                "The added derivative is `dG/dfNL_rec` at "
                "`fNL_rec=0`, with `fNL_rec=fNL` and "
                "`b_rec(q)=b_rec,1+fNL_rec*bphi_rec/M(q)`."
            ),
            (
                "It includes deterministic tree, B222, B321I, B321II, "
                "B411, all current reconstructed EFT counterterms, "
                "density/noisy-shift stochastic terms, and the "
                "estimator-matched fixed-Poisson term."
            ),
            (
                "The fixed-b_rec PNG response itself is unchanged. "
                "Adaptive derivatives internal to an explicitly linear "
                "PNG term or an fNL-squared term vanish from the first "
                "derivative at the fiducial fNL=0 point."
            ),
            "",
            "## Statistical and numerical contract",
            "",
            (
                "All error metrics use the covariance of one "
                "`(Gpc/h)^3` realization estimated from 500 fiducial "
                "boxes, never the covariance of the mean, with the "
                "same Hartlap correction and power-spectrum b1 prior."
            ),
            (
                "The response uses symmetric steps 1 and 0.5 with "
                "Richardson extrapolation; the response-geometry PDF "
                "reports the size and nuisance projection of every "
                "added sector."
            ),
            "",
            "## Scope",
            "",
            (
                "This is an isolated model-level Fisher experiment. "
                "It does not claim that parameter-dependent catalog "
                "reconstruction, its covariance derivative, or a "
                "finite-fNL likelihood has been validated."
            ),
            "",
            "## Outputs",
            "",
            "- `adaptive_brec_full_fisher_vs_kmax.pdf`",
            "- `adaptive_brec_response_geometry.pdf`",
            "",
        ]
    )
    return "\n".join(lines)


def run_adaptive_brec_full_fisher(
    root: Path,
    shared: SharedInputs,
) -> dict[str, Any]:
    (
        response_templates,
        fitter,
        number_density,
        response_rows,
        response_provenance,
    ) = load_adaptive_brec_full_templates(shared)
    tree_basis, tree_basis_provenance = (
        load_adaptive_brec_tree_basis(
            root / ADAPTIVE_BREC_BASIS_RELATIVE,
            shared,
        )
    )
    calibration = b1_calibration_contract(shared.data_root)
    b1_mean = float(calibration["primary"]["b1"])
    b1_sigma = float(calibration["primary"]["b1_sigma"])
    scan = []
    minimum_eigenvalue = math.inf
    regression_errors: list[float] = []
    previous = json.loads(
        (root / ADAPTIVE_BREC_SUMMARY_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    for kmax, previous_row in zip(CUTS, previous["kmax_scan"]):
        contexts = {
            reconstruction: context_with_b1_prior(
                build_context(
                    shared,
                    reconstruction,
                    "coevolution",
                    float(kmax),
                ),
                mean=b1_mean,
                sigma=b1_sigma,
            )
            for reconstruction in RECONSTRUCTIONS
        }
        base_fishers: dict[str, dict[str, Any]] = {}
        jacobians: dict[str, np.ndarray] = {}
        conditionals: dict[str, ConditionalResult] = {}
        for reconstruction, context in contexts.items():
            _map, conditional = gaussian_nuisance_map(context)
            jacobian, _check = model_jacobian_at_zero(
                context,
                conditional.nuisance,
            )
            fisher_result, _matrix = fisher_from_jacobian(
                context, context, jacobian
            )
            base_fishers[reconstruction] = fisher_result
            jacobians[reconstruction] = jacobian
            conditionals[reconstruction] = conditional
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(
                    fisher_result[
                        "scaled_fisher_minimum_eigenvalue"
                    ]
                ),
            )
        post = contexts["post"]
        _prediction, parameters, model_components = (
            post.evaluate_model(
                0.0,
                conditionals["post"].nuisance,
            )
        )
        full = response_templates.components(
            parameters,
            number_density,
            DEFAULT_STOCHASTIC_MODE,
        )
        error = adaptive_step_error_components(
            response_rows,
            fitter,
            parameters,
            number_density,
        )
        indices = post.indices
        b1_map = float(parameters["b1"])
        analytic_tree = (
            b1_map**4
            * ADAPTIVE_BREC_BPHI_REC
            / post_model.B_REC_H
            * tree_basis[indices]
        )
        finite_tree = np.asarray(
            full["tree"], dtype=np.float64
        )[indices]
        full_correction = np.asarray(
            full["total"], dtype=np.float64
        )[indices]
        step_error = np.asarray(
            error["total"], dtype=np.float64
        )[indices]
        tree_jacobian = jacobians["post"].copy()
        tree_jacobian[:, 0] += analytic_tree
        full_jacobian = jacobians["post"].copy()
        full_jacobian[:, 0] += full_correction
        full_error_jacobian = full_jacobian.copy()
        full_error_jacobian[:, 0] += step_error
        shift_response = np.asarray(
            model_components["halo_reconstruction"],
            dtype=np.float64,
        )[indices]
        ablation_jacobian = jacobians["post"].copy()
        ablation_jacobian[:, 0] -= shift_response
        tree_fisher, _ = fisher_from_jacobian(
            post, post, tree_jacobian
        )
        full_fisher, _ = fisher_from_jacobian(
            post, post, full_jacobian
        )
        error_fisher, _ = fisher_from_jacobian(
            post, post, full_error_jacobian
        )
        ablation_fisher, _ = fisher_from_jacobian(
            post, post, ablation_jacobian
        )
        for result in (
            tree_fisher,
            full_fisher,
            error_fisher,
            ablation_fisher,
        ):
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(
                    result["scaled_fisher_minimum_eigenvalue"]
                ),
            )
        old_white = solve_triangular(
            post.chol_fit,
            jacobians["post"][:, 0],
            lower=True,
        )
        tree_white = solve_triangular(
            post.chol_fit, finite_tree, lower=True
        )
        analytic_tree_white = solve_triangular(
            post.chol_fit, analytic_tree, lower=True
        )
        full_white = solve_triangular(
            post.chol_fit, full_correction, lower=True
        )
        error_white = solve_triangular(
            post.chol_fit, step_error, lower=True
        )
        sector_geometry = {}
        for sector in (
            "tree",
            "loop",
            "counterterm",
            "stochastic",
            "fixed_poisson",
            "total",
        ):
            vector = np.asarray(
                full[sector], dtype=np.float64
            )[indices]
            white = solve_triangular(
                post.chol_fit, vector, lower=True
            )
            sector_jacobian = jacobians["post"].copy()
            sector_jacobian[:, 0] = vector
            sector_fisher, _ = fisher_from_jacobian(
                post, post, sector_jacobian
            )
            sector_geometry[sector] = {
                "whitened_norm": float(np.linalg.norm(white)),
                "cosine_with_original_post_response": (
                    vector_cosine(white, old_white)
                ),
                "cosine_with_tree_adaptive_response": (
                    vector_cosine(white, tree_white)
                ),
                "nuisance_projected_information_norm": (
                    1.0
                    / float(
                        sector_fisher[
                            "all_nuisance_sigma_fNL"
                        ]
                    )
                ),
            }
        tree_closure = float(
            np.linalg.norm(tree_white - analytic_tree_white)
            / max(np.linalg.norm(analytic_tree_white), 1.0e-300)
        )
        step_error_ratio = float(
            np.linalg.norm(error_white)
            / max(np.linalg.norm(full_white), 1.0e-300)
        )
        row = {
            "kmax_h_mpc": float(kmax),
            "n_data": int(indices.size),
            "gaussian_map_b1": b1_map,
            "pre": compact_fisher_widths(
                base_fishers["pre"]
            ),
            "post_original": compact_fisher_widths(
                base_fishers["post"]
            ),
            "post_tree_adaptive": compact_fisher_widths(
                tree_fisher
            ),
            "post_full_adaptive": compact_fisher_widths(
                full_fisher
            ),
            "post_full_plus_step_error": compact_fisher_widths(
                error_fisher
            ),
            "post_shift_removed_ablation": (
                compact_fisher_widths(ablation_fisher)
            ),
            "response_geometry": {
                "sectors": sector_geometry,
                "finite_difference_tree_vs_analytic_relative": (
                    tree_closure
                ),
                "richardson_step_error_over_full_correction": (
                    step_error_ratio
                ),
                "full_correction_cosine_with_original_post": (
                    vector_cosine(full_white, old_white)
                ),
                "full_correction_cosine_with_tree": (
                    vector_cosine(full_white, tree_white)
                ),
            },
            "covariance_contract": {
                reconstruction: {
                    "mock_count": int(
                        context.covariance_mock_count
                    ),
                    "is_single_realization_sample_covariance": bool(
                        np.array_equal(
                            context.covariance_single,
                            np.cov(
                                context.samples,
                                rowvar=False,
                                ddof=1,
                            ),
                        )
                    ),
                    "divided_by_500": False,
                }
                for reconstruction, context in contexts.items()
            },
        }
        scan.append(row)
        for current_key, previous_key in (
            ("pre", "pre"),
            ("post_original", "post_original"),
            ("post_tree_adaptive", "post_adaptive_brec"),
            (
                "post_shift_removed_ablation",
                "post_shift_removed_upper_limit",
            ),
        ):
            for metric in (
                "fixed_nuisance_sigma_fNL",
                "all_nuisance_sigma_fNL",
            ):
                regression_errors.append(
                    abs(
                        float(row[current_key][metric])
                        - float(previous_row[previous_key][metric])
                    )
                )
    prior_sensitivity = []
    for scale in (0.5, 1.0, 2.0):
        context = context_with_b1_prior(
            build_context(
                shared, "post", "coevolution", 0.14
            ),
            mean=b1_mean,
            sigma=scale * b1_sigma,
        )
        _map, conditional = gaussian_nuisance_map(context)
        jacobian, _check = model_jacobian_at_zero(
            context, conditional.nuisance
        )
        _prediction, parameters, _components = (
            context.evaluate_model(
                0.0, conditional.nuisance
            )
        )
        correction = np.asarray(
            response_templates.components(
                parameters,
                number_density,
                DEFAULT_STOCHASTIC_MODE,
            )["total"],
            dtype=np.float64,
        )[context.indices]
        old_fisher, _ = fisher_from_jacobian(
            context, context, jacobian
        )
        jacobian[:, 0] += correction
        full_fisher, _ = fisher_from_jacobian(
            context, context, jacobian
        )
        prior_sensitivity.append(
            {
                "b1_prior_sigma": scale * b1_sigma,
                "scale_relative_to_primary": scale,
                "gaussian_map_b1": float(parameters["b1"]),
                "post_original_marginal_sigma_fNL": float(
                    old_fisher["all_nuisance_sigma_fNL"]
                ),
                "post_full_adaptive_marginal_sigma_fNL": float(
                    full_fisher["all_nuisance_sigma_fNL"]
                ),
            }
        )
    last = scan[-1]
    tree_width = float(
        last["post_tree_adaptive"]["all_nuisance_sigma_fNL"]
    )
    full_width = float(
        last["post_full_adaptive"]["all_nuisance_sigma_fNL"]
    )
    maximum_tree_closure = max(
        row["response_geometry"][
            "finite_difference_tree_vs_analytic_relative"
        ]
        for row in scan
    )
    maximum_step_error_ratio = max(
        row["response_geometry"][
            "richardson_step_error_over_full_correction"
        ]
        for row in scan
    )
    maximum_regression_error = max(regression_errors)
    checks = {
        "response_has_exact_27_selected_bins": bool(
            len(response_rows) == 27
        ),
        "all_covariances_are_single_realization_500_box": bool(
            all(
                contract["mock_count"] == 500
                and contract[
                    "is_single_realization_sample_covariance"
                ]
                and contract["divided_by_500"] is False
                for row in scan
                for contract in row[
                    "covariance_contract"
                ].values()
            )
        ),
        "previous_fisher_zero_regression_below_1e10": bool(
            maximum_regression_error < 1.0e-10
        ),
        "tree_finite_difference_closes_analytic_below_1e_minus_4": bool(
            maximum_tree_closure < 1.0e-4
        ),
        "richardson_step_error_below_one_percent": bool(
            maximum_step_error_ratio < 0.01
        ),
        "all_fisher_matrices_positive_definite": bool(
            minimum_eigenvalue > 0.0
        ),
        "all_widths_finite_positive": bool(
            all(
                math.isfinite(float(row[key][metric]))
                and float(row[key][metric]) > 0.0
                for row in scan
                for key in (
                    "pre",
                    "post_original",
                    "post_tree_adaptive",
                    "post_full_adaptive",
                    "post_shift_removed_ablation",
                )
                for metric in (
                    "fixed_nuisance_sigma_fNL",
                    "all_nuisance_sigma_fNL",
                )
            )
        ),
    }
    if not all(checks.values()):
        failed = [
            name for name, passed in checks.items() if not passed
        ]
        raise RuntimeError(
            f"full adaptive-brec Fisher checks failed: {failed}"
        )
    figure_root = (
        root / ADAPTIVE_BREC_FULL_FIGURE_RELATIVE
    )
    fisher_figure = (
        figure_root / "adaptive_brec_full_fisher_vs_kmax.pdf"
    )
    geometry_figure = (
        figure_root / "adaptive_brec_response_geometry.pdf"
    )
    plot_adaptive_brec_full_fisher(fisher_figure, scan)
    plot_adaptive_brec_response_geometry(
        geometry_figure, scan
    )
    summary = {
        "schema": (
            "marisa-b-post-halo-local-png-adaptive-brec-full-fisher-v1"
        ),
        "created_utc": utc_now(),
        "status": "full_adaptive_brec_fisher_complete",
        "scope": {
            "sample": (
                "Quijote z=1 Mmin=1e13 fNL=0 real-space halos"
            ),
            "tier": "coevolution",
            "cuts_h_mpc": CUTS,
            "covariance": (
                "single-realization sample covariance from 500 "
                "fiducial boxes with Hartlap precision correction"
            ),
            "adaptive_response": (
                "dG/dfNL_rec at zero through deterministic one-loop, "
                "EFT, stochastic, and fixed-Poisson sectors"
            ),
        },
        "headline": {
            "kmax_h_mpc": 0.14,
            "pre_marginal": float(
                last["pre"]["all_nuisance_sigma_fNL"]
            ),
            "post_original_marginal": float(
                last["post_original"][
                    "all_nuisance_sigma_fNL"
                ]
            ),
            "tree_adaptive_marginal": tree_width,
            "full_adaptive_marginal": full_width,
            "full_vs_tree_fractional_change": (
                full_width / tree_width - 1.0
            ),
            "full_vs_original_fractional_change": (
                full_width
                / float(
                    last["post_original"][
                        "all_nuisance_sigma_fNL"
                    ]
                )
                - 1.0
            ),
            "full_post_over_pre": (
                full_width
                / float(
                    last["pre"]["all_nuisance_sigma_fNL"]
                )
            ),
            "direction_blind_outcome": (
                "improved"
                if full_width < tree_width
                else "weakened"
            ),
        },
        "kmax_scan": scan,
        "b1_prior_sensitivity_at_kmax_0p14": prior_sensitivity,
        "checks": checks,
        "maximum_previous_fisher_regression_error": (
            maximum_regression_error
        ),
        "maximum_tree_response_relative_closure": (
            maximum_tree_closure
        ),
        "maximum_step_error_over_full_correction": (
            maximum_step_error_ratio
        ),
        "minimum_scaled_fisher_eigenvalue": minimum_eigenvalue,
        "provenance": {
            "response_templates": response_provenance,
            "tree_basis": tree_basis_provenance,
            "matrix": {
                "path": str(shared.matrix),
                "sha256": shared.matrix_hash,
            },
            "previous_tree_fisher": {
                "path": str(
                    root / ADAPTIVE_BREC_SUMMARY_RELATIVE
                ),
                "sha256": sha256(
                    root / ADAPTIVE_BREC_SUMMARY_RELATIVE
                ),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256_before_summary_write": sha256(
                    Path(__file__).resolve()
                ),
            },
        },
        "limitations": [
            (
                "model-level response only; no catalog-level "
                "parameter-dependent reconstruction validation"
            ),
            "fixed covariance; covariance derivative is omitted",
            (
                "local Fisher derivative at fNL=0, not a finite-fNL "
                "likelihood validation"
            ),
        ],
        "outputs": {
            "fisher_figure": {
                "path": str(fisher_figure),
                "sha256": sha256(fisher_figure),
            },
            "response_geometry_figure": {
                "path": str(geometry_figure),
                "sha256": sha256(geometry_figure),
            },
        },
    }
    report_path = (
        root / ADAPTIVE_BREC_FULL_REPORT_RELATIVE
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        adaptive_brec_full_report_text(summary),
        encoding="utf-8",
    )
    summary["outputs"]["report"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
    }
    summary_path = (
        root / ADAPTIVE_BREC_FULL_SUMMARY_RELATIVE
    )
    atomic_json(summary_path, summary)
    return {
        "status": summary["status"],
        "summary_path": summary_path,
        "report_path": report_path,
        "figure_paths": [fisher_figure, geometry_figure],
        "headline": summary["headline"],
        "checks": checks,
    }


def build_adaptive_brec_full_mcmc_context(
    shared: SharedInputs,
    *,
    model: str,
    kmax: float,
    response_bundle: (
        tuple[
            Any,
            ModuleType,
            float,
            dict[int, dict[str, Any]],
            dict[str, Any],
        ]
        | None
    ) = None,
) -> CollapsedContext:
    if model not in ADAPTIVE_BREC_MCMC_MODELS:
        raise ValueError(model)
    calibration = b1_calibration_contract(shared.data_root)
    b1_mean = float(calibration["primary"]["b1"])
    b1_sigma = float(calibration["primary"]["b1_sigma"])
    reconstruction = "pre" if model == "pre" else "post"
    base = context_with_b1_prior(
        build_context(
            shared,
            reconstruction,
            "coevolution",
            float(kmax),
        ),
        mean=b1_mean,
        sigma=b1_sigma,
    )
    labels = {
        "pre": "pre_psprior",
        "post_fixed_brec": "post_fixed_brec_psprior",
        "post_adaptive_brec_local": "post_adaptive_brec_local_psprior",
    }
    prior_contract = {
        "source": calibration["source"],
        "mean": b1_mean,
        "sigma": b1_sigma,
    }
    prior_hash = hashlib.sha256(
        json.dumps(
            prior_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    input_hashes = dict(base.input_hashes)
    input_hashes["power_spectrum_b1_prior_contract"] = prior_hash
    if model != "post_adaptive_brec_local":
        return replace(
            base,
            reconstruction=labels[model],
            input_hashes=input_hashes,
        )
    if response_bundle is None:
        response_bundle = load_adaptive_brec_full_templates(shared)
    (
        response_templates,
        _fitter,
        number_density,
        _response_rows,
        response_provenance,
    ) = response_bundle
    fixed_evaluate = base.evaluate_model

    def evaluate(
        fnl: float,
        nuisance: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        prediction, parameters, components = fixed_evaluate(fnl, nuisance)
        adaptive = response_templates.components(
            parameters,
            number_density,
            DEFAULT_STOCHASTIC_MODE,
        )
        adaptive_total = np.asarray(
            adaptive["total"],
            dtype=np.float64,
        )[base.indices]
        output_components = dict(components)
        output_components["adaptive_gaussian_tree"] = np.asarray(
            adaptive["tree"],
            dtype=np.float64,
        )
        output_components["adaptive_gaussian_loop"] = np.asarray(
            adaptive["loop"],
            dtype=np.float64,
        )
        output_components["adaptive_gaussian_counterterm"] = np.asarray(
            adaptive["counterterm"],
            dtype=np.float64,
        )
        output_components["adaptive_gaussian_stochastic"] = np.asarray(
            adaptive["stochastic"],
            dtype=np.float64,
        )
        output_components["adaptive_gaussian_fixed_poisson"] = np.asarray(
            adaptive["fixed_poisson"],
            dtype=np.float64,
        )
        output_components["adaptive_gaussian_total"] = np.asarray(
            adaptive["total"],
            dtype=np.float64,
        )
        return (
            np.asarray(prediction, dtype=np.float64)
            + float(fnl) * adaptive_total,
            parameters,
            output_components,
        )

    local_contract = {
        "schema": "marisa-b-adaptive-brec-local-response-likelihood-v1",
        "formula": "B_fixed(f,theta)+f*dG(theta)/dfNL_rec|0",
        "fiducial_fNL": 0.0,
        "finite_fNL_adaptive_png_derivatives_included": False,
    }
    input_hashes["adaptive_brec_response"] = str(
        response_provenance["sha256"]
    )
    input_hashes["local_response_likelihood_contract"] = hashlib.sha256(
        json.dumps(
            local_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        base,
        reconstruction=labels[model],
        evaluate_model=evaluate,
        input_hashes=input_hashes,
    )


def adaptive_brec_mcmc_contexts(
    shared: SharedInputs,
) -> dict[tuple[str, float], CollapsedContext]:
    response_bundle = load_adaptive_brec_full_templates(shared)
    return {
        (model, float(kmax)): build_adaptive_brec_full_mcmc_context(
            shared,
            model=model,
            kmax=float(kmax),
            response_bundle=response_bundle,
        )
        for model in ADAPTIVE_BREC_MCMC_MODELS
        for kmax in CUTS
    }


def adaptive_brec_mcmc_chain_contract(
    contexts: dict[tuple[str, float], CollapsedContext],
    *,
    ensembles: int,
    walkers: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "marisa-b-adaptive-brec-full-mcmc-chain-contract-v1",
        "likelihood": (
            "Hartlap-Gaussian with exact conditional marginalization of "
            "the registered linear nuisance block"
        ),
        "target": "Quijote halo fNL=0 mean",
        "covariance": (
            "single-realization sample covariance from 500 fiducial boxes; "
            "never divided by 500"
        ),
        "fNL_hard_flat_prior": FNL_BOUNDS,
        "b1_prior": {
            "mean": float(
                next(iter(contexts.values())).prior.mean[
                    next(iter(contexts.values())).names.index("b1")
                ]
            ),
            "sigma": float(
                next(iter(contexts.values())).prior.sigma[
                    next(iter(contexts.values())).names.index("b1")
                ]
            ),
            "source": "frozen power-spectrum calibration",
        },
        "sampler": {
            "independent_ensembles": int(ensembles),
            "walkers_per_ensemble": int(walkers),
            "seed": int(seed),
            "moves": "0.8*StretchMove(a=2)+0.2*DEMove",
        },
        "contexts": {
            f"{model}/{kmax:.2f}": context.metadata()
            for (model, kmax), context in contexts.items()
        },
    }


def register_adaptive_brec_mcmc_chain_contract(
    chain_path: Path,
    contract: dict[str, Any],
) -> str:
    encoded = json.dumps(
        jsonable(contract),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = "__adaptive_brec_full_mcmc_contract__"
    with h5py.File(chain_path, "a") as chain_file:
        if dataset_name in chain_file:
            stored = chain_file[dataset_name][()]
            if isinstance(stored, bytes):
                stored = stored.decode("utf-8")
            if str(stored) != encoded:
                raise ValueError(
                    "adaptive-brec MCMC checkpoint contract mismatch"
                )
        else:
            dtype = h5py.string_dtype(encoding="utf-8")
            chain_file.create_dataset(
                dataset_name,
                data=encoded,
                dtype=dtype,
            )
        chain_file.attrs[
            "adaptive_brec_full_mcmc_contract_sha256"
        ] = digest
    return digest


def run_adaptive_brec_full_mcmc(
    root: Path,
    shared: SharedInputs,
    *,
    ensembles: int,
    walkers: int,
    steps: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    if ensembles != 4:
        raise ValueError(
            "the registered adaptive-brec MCMC requires four ensembles"
        )
    contexts = adaptive_brec_mcmc_contexts(shared)
    chain_path = root / ADAPTIVE_BREC_FULL_MCMC_CHAINS_RELATIVE
    contract = adaptive_brec_mcmc_chain_contract(
        contexts,
        ensembles=ensembles,
        walkers=walkers,
        seed=seed,
    )
    contract_hash = register_adaptive_brec_mcmc_chain_contract(
        chain_path,
        contract,
    )
    validations = {}
    runs = {}
    for model in ADAPTIVE_BREC_MCMC_MODELS:
        for kmax in CUTS:
            context = contexts[(model, float(kmax))]
            validation = validate_context(
                context,
                np.random.default_rng(
                    seed
                    + 10000 * ADAPTIVE_BREC_MCMC_MODELS.index(model)
                    + int(round(1000.0 * kmax))
                ),
                random_points=2,
            )
            validations[f"{model}/{kmax:.2f}"] = validation
            if not validation["pass"]:
                raise RuntimeError(
                    f"adaptive-brec MCMC validation failed for "
                    f"{model}, kmax={kmax:.2f}"
                )
            runs[f"{model}/{kmax:.2f}"] = run_mcmc_context(
                context,
                chain_path=chain_path,
                ensembles=ensembles,
                walkers=walkers,
                steps=steps,
                workers=workers,
                seed=(
                    seed
                    + 100000 * ADAPTIVE_BREC_MCMC_MODELS.index(model)
                    + int(round(1000.0 * kmax))
                ),
            )
    return {
        "status": "adaptive_brec_full_mcmc_checkpoint_complete",
        "chain_path": chain_path,
        "contract_sha256": contract_hash,
        "requested_steps": int(steps),
        "validations": validations,
        "runs": runs,
    }


def adaptive_brec_locality_audit(
    context: CollapsedContext,
    *,
    chain_path: Path,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    retained, flat = retained_chain_samples(
        context,
        chain_path=chain_path,
        diagnosis=diagnosis,
    )
    fnl = np.asarray(flat[:, 0], dtype=np.float64)
    standard_deviation = float(np.std(fnl, ddof=1))
    full_quantiles = np.quantile(fnl, (0.16, 0.5, 0.84))
    absolute = np.abs(fnl)
    regions = {
        "abs_fNL_le_50": float(np.mean(absolute <= 50.0)),
        "50_lt_abs_fNL_le_100": float(
            np.mean((absolute > 50.0) & (absolute <= 100.0))
        ),
        "100_lt_abs_fNL_le_150": float(
            np.mean((absolute > 100.0) & (absolute <= 150.0))
        ),
    }
    truncated = {}
    for limit in (50.0, 100.0):
        selected = fnl[absolute <= limit]
        if selected.size < 100:
            raise RuntimeError(
                f"too few samples for |fNL|<={limit:g} locality audit"
            )
        quantiles = np.quantile(selected, (0.16, 0.5, 0.84))
        truncated[f"abs_fNL_le_{int(limit)}"] = {
            "retained_samples": int(selected.size),
            "retained_fraction": float(selected.size / fnl.size),
            "equal_tail_68": (quantiles[0], quantiles[2]),
            "median": quantiles[1],
            "median_shift_in_full_posterior_sigma": float(
                (quantiles[1] - full_quantiles[1])
                / max(standard_deviation, 1.0e-300)
            ),
            "width68_ratio_to_full": float(
                (quantiles[2] - quantiles[0])
                / max(
                    full_quantiles[2] - full_quantiles[0],
                    1.0e-300,
                )
            ),
        }
    midpoint = retained.shape[1] // 2
    first = retained[:, :midpoint].reshape(-1, context.theta_dimension)
    second = retained[:, midpoint:].reshape(-1, context.theta_dimension)
    first_fnl = np.quantile(first[:, 0], (0.16, 0.5, 0.84))
    second_fnl = np.quantile(second[:, 0], (0.16, 0.5, 0.84))
    half_interval_shift = np.abs(first_fnl - second_fnl) / max(
        standard_deviation,
        1.0e-300,
    )
    return {
        "posterior_mass_by_locality_region": regions,
        "posterior_mass_sum": float(sum(regions.values())),
        "truncated_posterior_sensitivity": truncated,
        "first_half_fNL_quantiles_16_50_84": first_fnl,
        "second_half_fNL_quantiles_16_50_84": second_fnl,
        "half_chain_fNL_quantile_shift_in_full_sigma": (
            half_interval_shift
        ),
        "maximum_half_chain_fNL_quantile_shift_in_full_sigma": float(
            np.max(half_interval_shift)
        ),
        "interpretation": (
            "local-response extrapolation diagnostic only; truncated "
            "summaries are not replacement inference priors"
        ),
    }


def adaptive_brec_mcmc_convergence_checks(
    diagnosis: dict[str, Any],
    locality: dict[str, Any],
) -> dict[str, bool]:
    rank_rhat = np.asarray(
        diagnosis["rank_normalized_split_rhat"],
        dtype=np.float64,
    )
    bulk_ess = np.asarray(diagnosis["bulk_ess"], dtype=np.float64)
    tail_ess = np.asarray(
        diagnosis["tail_ess_5_95"],
        dtype=np.float64,
    )
    return {
        "exactly_four_independent_ensembles": bool(
            diagnosis["chain_layout"]["ensembles"] == 4
        ),
        "all_rank_normalized_split_rhat_below_1p01": bool(
            np.all(rank_rhat < 1.01)
        ),
        "all_bulk_ess_above_400": bool(np.all(bulk_ess > 400.0)),
        "all_tail_ess_above_400": bool(np.all(tail_ess > 400.0)),
        "fnl_half_chain_median_below_0p1_sigma": bool(
            float(diagnosis["half_chain_median_shift_sigma"][0]) < 0.1
        ),
        "fnl_half_chain_68_interval_stable_below_0p15_sigma": bool(
            locality[
                "maximum_half_chain_fNL_quantile_shift_in_full_sigma"
            ]
            < 0.15
        ),
        "no_fnl_boundary_sticking": bool(
            diagnosis["fnl_summary"][
                "boundary_mass_outer_2_percent_each_side"
            ]
            < 0.05
        ),
        "no_b1_boundary_sticking": bool(
            diagnosis["b1_boundary_mass_outer_2_percent_each_side"]
            < 0.01
        ),
        "finite_reasonable_acceptance_fraction": bool(
            math.isfinite(
                float(diagnosis["mean_acceptance_fraction"])
            )
            and 0.1
            < float(diagnosis["mean_acceptance_fraction"])
            < 0.8
        ),
        "locality_region_partition_closes": bool(
            abs(float(locality["posterior_mass_sum"]) - 1.0)
            < 1.0e-12
        ),
    }


def adaptive_brec_mcmc_fisher_width(
    fisher_summary: dict[str, Any],
    *,
    model: str,
    kmax: float,
) -> float:
    mapping = {
        "pre": "pre",
        "post_fixed_brec": "post_original",
        "post_adaptive_brec_local": "post_full_adaptive",
    }
    rows = [
        row
        for row in fisher_summary["kmax_scan"]
        if math.isclose(
            float(row["kmax_h_mpc"]),
            float(kmax),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    if len(rows) != 1:
        raise ValueError(f"Fisher summary misses kmax={kmax:.2f}")
    return float(
        rows[0][mapping[model]]["all_nuisance_sigma_fNL"]
    )


def plot_adaptive_brec_full_mcmc(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 4.25))
    styles = {
        "pre": ("pre reconstruction", "#222222", "o", -0.0025),
        "post_fixed_brec": (
            "post, fixed $b_{\\rm rec}$",
            "#b2182b",
            "s",
            0.0,
        ),
        "post_adaptive_brec_local": (
            "post, full adaptive local response",
            "#2166ac",
            "^",
            0.0025,
        ),
    }
    for model in ADAPTIVE_BREC_MCMC_MODELS:
        label, colour, marker, offset = styles[model]
        selected = [row for row in rows if row["model"] == model]
        x = np.asarray(
            [row["kmax_h_mpc"] + offset for row in selected],
            dtype=np.float64,
        )
        median = np.asarray(
            [row["fnl"]["median"] for row in selected],
            dtype=np.float64,
        )
        low = np.asarray(
            [row["fnl"]["equal_tail_68"][0] for row in selected],
            dtype=np.float64,
        )
        high = np.asarray(
            [row["fnl"]["equal_tail_68"][1] for row in selected],
            dtype=np.float64,
        )
        axes[0].errorbar(
            x,
            median,
            yerr=np.vstack((median - low, high - median)),
            color=colour,
            marker=marker,
            ms=5.2,
            lw=1.45,
            capsize=3.0,
            label=label,
        )
        axes[1].plot(
            x,
            [
                row["mcmc_width68_over_local_fisher_sigma"]
                for row in selected
            ],
            color=colour,
            marker=marker,
            ms=5.2,
            lw=1.45,
            label=label,
        )
    axes[0].axhline(0.0, color="0.55", lw=1.0, ls="--")
    axes[0].set_ylabel(r"$f_{\rm NL}$ posterior median and 68% interval")
    axes[0].set_title(r"Quijote halo $f_{\rm NL}=0$ mean")
    axes[1].axhline(1.0, color="0.55", lw=1.0, ls="--")
    axes[1].set_ylabel(
        r"MCMC 68% half-width / local Fisher $\sigma(f_{\rm NL})$"
    )
    axes[1].set_title("Non-Gaussianity and prior-shape diagnostic")
    for axis in axes:
        axis.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
        axis.set_xticks(CUTS)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8.7, loc="best")
    figure.suptitle(
        "Power-spectrum $b_1$ prior; single-realization covariance"
        "\nAdaptive curve uses the fiducial-zero local-response likelihood",
        fontsize=11.0,
    )
    figure.tight_layout()
    save_pdf(figure, path)


def plot_adaptive_brec_full_mcmc_corner(
    path: Path,
    samples: dict[str, np.ndarray],
    theta_names: tuple[str, ...],
    summaries: dict[str, dict[str, Any]],
    b1_mean: float,
) -> None:
    labels = {
        "fNL": r"$f_{\rm NL}$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "gamma2": r"$\gamma_2$",
        "gamma21": r"$\gamma_{21}$",
        "b_nabla2_delta": r"$b_{\nabla^2\delta}$",
    }
    plot_labels = [labels.get(name, name) for name in theta_names]
    combined = np.vstack(
        [samples[model] for model in ADAPTIVE_BREC_MCMC_MODELS]
    )
    ranges: list[tuple[float, float]] = []
    for index, name in enumerate(theta_names):
        if name == "fNL":
            ranges.append(FNL_BOUNDS)
            continue
        low, high = np.quantile(combined[:, index], (0.0025, 0.9975))
        span = max(float(high - low), 1.0e-6)
        ranges.append(
            (float(low - 0.08 * span), float(high + 0.08 * span))
        )
    colours = {
        "pre": "#222222",
        "post_fixed_brec": "#b2182b",
        "post_adaptive_brec_local": "#2166ac",
    }
    display = {
        "pre": "pre",
        "post_fixed_brec": "post fixed-$b_{\\rm rec}$",
        "post_adaptive_brec_local": "post adaptive local",
    }
    truths: list[float | None] = []
    for name in theta_names:
        if name == "fNL":
            truths.append(0.0)
        elif name == "b1":
            truths.append(b1_mean)
        else:
            truths.append(None)
    figure = None
    for model in ADAPTIVE_BREC_MCMC_MODELS:
        figure = corner.corner(
            samples[model],
            fig=figure,
            labels=plot_labels,
            range=ranges,
            bins=42,
            color=colours[model],
            smooth=1.05,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            no_fill_contours=True,
            levels=(0.68, 0.95),
            truths=truths if figure is None else None,
            truth_color="0.55",
            max_n_ticks=4,
            hist_kwargs={
                "density": True,
                "lw": 1.45,
                "histtype": "step",
            },
            contour_kwargs={"linewidths": 1.25},
        )
    if figure is None:
        raise RuntimeError("corner plot received no posterior samples")
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colours[model],
                lw=2.3,
                label=display[model],
            )
            for model in ADAPTIVE_BREC_MCMC_MODELS
        ],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.985, 0.985),
    )
    summary_lines = []
    for model in ADAPTIVE_BREC_MCMC_MODELS:
        fnl = summaries[model]["fnl"]
        summary_lines.append(
            f"{display[model]}: "
            f"{fnl['median']:.1f}"
            f"_{{-{fnl['median']-fnl['equal_tail_68'][0]:.1f}}}"
            f"^{{+{fnl['equal_tail_68'][1]-fnl['median']:.1f}}}"
        )
    figure.suptitle(
        r"Coevolution, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$"
        "\n"
        + r"$f_{\rm NL}$ medians and 68% intervals: "
        + "; ".join(summary_lines)
        + "\nMarginalized MCMC; profiler widths are not used",
        y=1.015,
        fontsize=10.5,
    )
    save_pdf(figure, path)


def adaptive_brec_full_mcmc_report_text(
    mcmc: dict[str, Any],
) -> str:
    labels = {
        "pre": "pre",
        "post_fixed_brec": "post fixed-brec",
        "post_adaptive_brec_local": "post adaptive local",
    }
    lines = [
        "",
        "## Fiducial-zero local-response MCMC",
        "",
        (
            "The Quijote halo `fNL=0` mean was sampled with the same "
            "single-realization covariance, Hartlap correction, "
            "coevolution nuisance set, power-spectrum `b1` prior, and "
            "hard-flat `fNL in [-150,150]` prior as the Fisher comparison."
        ),
        "",
        "| kmax | model | fNL median | 68% interval | local Fisher sigma |",
        "|---:|:---|---:|:---|---:|",
    ]
    for row in mcmc["kmax_scan"]:
        fnl = row["fnl"]
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} | {labels[row['model']]} "
            f"| {fnl['median']:.2f} "
            f"| [{fnl['equal_tail_68'][0]:.2f}, "
            f"{fnl['equal_tail_68'][1]:.2f}] "
            f"| {row['local_fisher_sigma_fNL']:.2f} |"
        )
    lines.extend(
        [
            "",
            "### Convergence and locality",
            "",
            (
                "All reported chains pass the registered four-ensemble "
                "split-Rhat, bulk/tail ESS, half-chain stability, "
                "acceptance, and boundary-mass gates."
            ),
            (
                f"The maximum split-Rhat is "
                f"`{mcmc['convergence_headline']['maximum_rank_rhat']:.5f}` "
                f"and the minimum bulk/tail ESS values are "
                f"`{mcmc['convergence_headline']['minimum_bulk_ess']:.0f}`/"
                f"`{mcmc['convergence_headline']['minimum_tail_ess']:.0f}`."
            ),
            (
                "The retained summary records posterior mass in "
                "`|fNL|<=50`, `50<|fNL|<=100`, and "
                "`100<|fNL|<=150`, plus the corresponding truncated "
                "posterior sensitivity checks."
            ),
            "",
            "### Interpretation limit",
            "",
            (
                "The adaptive likelihood is "
                "`B_fixed(f,theta)+f*dG(theta)/dfNL_rec|0`. It is exact "
                "for the derivative used by the fiducial-zero Fisher "
                "analysis, but it is only a local extrapolation away from "
                "`fNL=0`; adaptive derivatives internal to finite-fNL PNG "
                "terms have not been added."
            ),
            "",
            "### Additional outputs",
            "",
            "- `adaptive_brec_full_mcmc_vs_kmax.pdf`",
            "- `adaptive_brec_full_mcmc_corner_kmax0p14.pdf`",
            "- `adaptive_brec_full_mcmc_chains.h5`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_adaptive_brec_full_mcmc(
    root: Path,
    shared: SharedInputs,
    *,
    burn_fraction: float,
    seed: int,
) -> dict[str, Any]:
    summary_path = root / ADAPTIVE_BREC_FULL_SUMMARY_RELATIVE
    chain_path = root / ADAPTIVE_BREC_FULL_MCMC_CHAINS_RELATIVE
    if not summary_path.is_file():
        raise FileNotFoundError(
            "run adaptive-brec-full-fisher before MCMC finalization"
        )
    if not chain_path.is_file():
        raise FileNotFoundError(chain_path)
    fisher_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contexts = adaptive_brec_mcmc_contexts(shared)
    dataset_name = "__adaptive_brec_full_mcmc_contract__"
    with h5py.File(chain_path, "r") as chain_file:
        if dataset_name not in chain_file:
            raise ValueError("MCMC chain file has no registered contract")
        stored_contract_text = chain_file[dataset_name][()]
        if isinstance(stored_contract_text, bytes):
            stored_contract_text = stored_contract_text.decode("utf-8")
        stored_contract = json.loads(str(stored_contract_text))
        stored_contract_hash = hashlib.sha256(
            str(stored_contract_text).encode("utf-8")
        ).hexdigest()
        attributed_hash_value = chain_file.attrs[
            "adaptive_brec_full_mcmc_contract_sha256"
        ]
        if isinstance(attributed_hash_value, bytes):
            attributed_hash_value = attributed_hash_value.decode("utf-8")
        attributed_hash = str(attributed_hash_value)
    if stored_contract_hash != attributed_hash:
        raise ValueError("MCMC chain contract hash does not close")
    current_contexts = {
        f"{model}/{kmax:.2f}": jsonable(context.metadata())
        for (model, kmax), context in contexts.items()
    }
    if stored_contract["contexts"] != current_contexts:
        raise ValueError("MCMC chain context provenance is stale")
    rows = []
    corner_arrays: dict[str, np.ndarray] = {}
    corner_summaries: dict[str, dict[str, Any]] = {}
    failed_contexts = []
    maximum_rank_rhat = 0.0
    minimum_bulk_ess = math.inf
    minimum_tail_ess = math.inf
    for model in ADAPTIVE_BREC_MCMC_MODELS:
        for kmax in CUTS:
            context = contexts[(model, float(kmax))]
            diagnosis = diagnose_context(
                context,
                chain_path=chain_path,
                burn_fraction=burn_fraction,
            )
            locality = adaptive_brec_locality_audit(
                context,
                chain_path=chain_path,
                diagnosis=diagnosis,
            )
            registered_checks = adaptive_brec_mcmc_convergence_checks(
                diagnosis,
                locality,
            )
            posterior, arrays = summarize_posterior(
                context,
                diagnosis=diagnosis,
                chain_path=chain_path,
                seed=(
                    seed
                    + 100000 * ADAPTIVE_BREC_MCMC_MODELS.index(model)
                    + int(round(1000.0 * kmax))
                ),
            )
            posterior["converged"] = bool(
                all(registered_checks.values())
            )
            posterior["registered_convergence_checks"] = registered_checks
            posterior["legacy_stricter_diagnostic_pass"] = bool(
                diagnosis["pass"]
            )
            fisher_width = adaptive_brec_mcmc_fisher_width(
                fisher_summary,
                model=model,
                kmax=float(kmax),
            )
            equal68 = posterior["fnl"]["equal_tail_68"]
            mcmc_width = 0.5 * (
                float(equal68[1]) - float(equal68[0])
            )
            row = {
                "model": model,
                "kmax_h_mpc": float(kmax),
                "n_data": int(context.indices.size),
                "theta_names": context.theta_names,
                "fnl": posterior["fnl"],
                "local_fisher_sigma_fNL": fisher_width,
                "mcmc_equal_tail_68_half_width": mcmc_width,
                "mcmc_width68_over_local_fisher_sigma": (
                    mcmc_width / fisher_width
                ),
                "maximum_rank_rhat": posterior["maximum_rank_rhat"],
                "minimum_bulk_ess": posterior["minimum_bulk_ess"],
                "minimum_tail_ess": posterior["minimum_tail_ess"],
                "mean_acceptance_fraction": posterior[
                    "mean_acceptance_fraction"
                ],
                "registered_convergence_checks": registered_checks,
                "legacy_stricter_diagnostic_pass": bool(
                    diagnosis["pass"]
                ),
                "locality_audit": locality,
                "fnl_correlations": posterior["fnl_correlations"],
                "correlation_matrix": posterior["correlation_matrix"],
                "covariance_contract": {
                    "mock_count": int(context.covariance_mock_count),
                    "single_realization": bool(
                        np.array_equal(
                            context.covariance_single,
                            np.cov(
                                context.samples,
                                rowvar=False,
                                ddof=1,
                            ),
                        )
                    ),
                    "divided_by_500": False,
                    "target_is_fNL0_sample_mean": bool(
                        np.array_equal(
                            context.target,
                            np.mean(context.samples, axis=0),
                        )
                    ),
                },
            }
            rows.append(row)
            if not posterior["converged"]:
                failed_contexts.append(
                    {
                        "model": model,
                        "kmax_h_mpc": float(kmax),
                        "failed_checks": [
                            name
                            for name, passed in registered_checks.items()
                            if not passed
                        ],
                    }
                )
            maximum_rank_rhat = max(
                maximum_rank_rhat,
                float(posterior["maximum_rank_rhat"]),
            )
            minimum_bulk_ess = min(
                minimum_bulk_ess,
                float(posterior["minimum_bulk_ess"]),
            )
            minimum_tail_ess = min(
                minimum_tail_ess,
                float(posterior["minimum_tail_ess"]),
            )
            if math.isclose(kmax, 0.14, abs_tol=1.0e-12):
                corner_arrays[model] = arrays["corner"]
                corner_summaries[model] = posterior
    if failed_contexts:
        return {
            "status": "adaptive_brec_full_mcmc_needs_more_steps",
            "failed_contexts": failed_contexts,
            "maximum_rank_rhat": maximum_rank_rhat,
            "minimum_bulk_ess": minimum_bulk_ess,
            "minimum_tail_ess": minimum_tail_ess,
            "chain_path": chain_path,
            "contract_sha256": stored_contract_hash,
        }
    checks = {
        "all_twelve_contexts_converged": bool(len(rows) == 12),
        "all_covariances_are_single_realization": bool(
            all(
                row["covariance_contract"]["single_realization"]
                and not row["covariance_contract"]["divided_by_500"]
                for row in rows
            )
        ),
        "all_targets_are_fNL0_sample_means": bool(
            all(
                row["covariance_contract"][
                    "target_is_fNL0_sample_mean"
                ]
                for row in rows
            )
        ),
        "hard_fNL_prior_is_minus150_to_plus150": bool(
            all(
                context.fnl_bounds == FNL_BOUNDS
                for context in contexts.values()
            )
        ),
        "power_spectrum_b1_prior_matches_fisher": bool(
            all(
                math.isclose(
                    float(
                        context.prior.sigma[
                            context.names.index("b1")
                        ]
                    ),
                    float(
                        b1_calibration_contract(shared.data_root)[
                            "primary"
                        ]["b1_sigma"]
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                for context in contexts.values()
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "adaptive-brec MCMC statistical contract failed"
        )
    figure_root = root / ADAPTIVE_BREC_FULL_FIGURE_RELATIVE
    mcmc_figure = (
        figure_root / "adaptive_brec_full_mcmc_vs_kmax.pdf"
    )
    corner_figure = (
        figure_root
        / "adaptive_brec_full_mcmc_corner_kmax0p14.pdf"
    )
    plot_adaptive_brec_full_mcmc(mcmc_figure, rows)
    representative = contexts[("pre", 0.14)]
    plot_adaptive_brec_full_mcmc_corner(
        corner_figure,
        corner_arrays,
        representative.theta_names,
        corner_summaries,
        float(
            representative.prior.mean[
                representative.names.index("b1")
            ]
        ),
    )
    mcmc = {
        "schema": "marisa-b-adaptive-brec-full-local-mcmc-v1",
        "created_utc": utc_now(),
        "sample": "Quijote z=1 Mmin=1e13 halo fNL=0 mean",
        "likelihood_model": (
            "B_fixed(fNL,theta)+fNL*dG(theta)/dfNL_rec|0"
        ),
        "finite_fNL_interpretation": (
            "local extrapolation only; not a complete finite-fNL "
            "adaptive-reconstruction likelihood"
        ),
        "kmax_scan": rows,
        "convergence_headline": {
            "maximum_rank_rhat": maximum_rank_rhat,
            "minimum_bulk_ess": minimum_bulk_ess,
            "minimum_tail_ess": minimum_tail_ess,
        },
        "checks": checks,
        "chain_contract": stored_contract,
        "chain_contract_sha256": stored_contract_hash,
    }
    fisher_summary["schema"] = (
        "marisa-b-post-halo-local-png-adaptive-brec-full-"
        "fisher-mcmc-v2"
    )
    fisher_summary["status"] = (
        "full_adaptive_brec_fisher_and_fNL0_mcmc_complete"
    )
    fisher_summary["mcmc"] = mcmc
    fisher_summary["outputs"]["mcmc_constraint_figure"] = {
        "path": str(mcmc_figure),
        "sha256": sha256(mcmc_figure),
    }
    fisher_summary["outputs"]["mcmc_corner_figure"] = {
        "path": str(corner_figure),
        "sha256": sha256(corner_figure),
    }
    fisher_summary["outputs"]["mcmc_chain"] = {
        "path": str(chain_path),
        "sha256": sha256(chain_path),
        "contract_sha256": stored_contract_hash,
    }
    report_path = root / ADAPTIVE_BREC_FULL_REPORT_RELATIVE
    report_path.write_text(
        adaptive_brec_full_report_text(fisher_summary)
        + adaptive_brec_full_mcmc_report_text(mcmc),
        encoding="utf-8",
    )
    fisher_summary["outputs"]["report"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
    }
    atomic_json(summary_path, fisher_summary)
    return {
        "status": fisher_summary["status"],
        "summary_path": summary_path,
        "report_path": report_path,
        "chain_path": chain_path,
        "figure_paths": (mcmc_figure, corner_figure),
        "convergence_headline": mcmc["convergence_headline"],
        "checks": checks,
    }


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": "marisa-b-prepost-halo-fnl-mcmc-state-v1",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "stages": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def update_state(
    path: Path,
    stage: str,
    payload: Any,
) -> None:
    state = read_state(path)
    state["updated_utc"] = utc_now()
    state["stages"][stage] = jsonable(payload)
    atomic_json(path, state)


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
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "validate",
            "benchmark",
            "smoke",
            "production",
            "diagnose",
            "explicit",
            "student-t",
            "fisher-audit",
            "adaptive-brec-fisher",
            "adaptive-brec-full-produce",
            "adaptive-brec-full-aggregate",
            "adaptive-brec-full-fisher",
            "adaptive-brec-full-mcmc",
            "adaptive-brec-full-mcmc-finalize",
            "finalize",
        ),
        required=True,
    )
    parser.add_argument(
        "--reconstruction",
        choices=RECONSTRUCTIONS,
        default="pre",
    )
    parser.add_argument("--tier", choices=TIERS, default="coevolution")
    parser.add_argument("--kmax", type=float, choices=CUTS, default=0.08)
    parser.add_argument("--wide-prior", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--random-points", type=int, default=4)
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--ensembles", type=int, default=4)
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--burn-fraction", type=float, default=0.5)
    parser.add_argument("--importance-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 28:
        raise ValueError("--workers must lie in [1,28]")
    repository_root = args.repo_root.resolve()
    configured_data_root = (
        args.data_root
        if args.data_root is not None
        else Path(os.environ.get("MARISA_B_DATA_ROOT", repository_root))
    )
    data_root = configured_data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )
    root = output_root
    state_path = root / STATE_RELATIVE
    shared = load_shared_inputs(
        repository_root,
        data_root,
        output_root,
    )
    if args.stage == "audit":
        payload = audit_shared_inputs(root, shared)
        update_state(state_path, "audit", payload)
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "fisher-audit":
        payload = run_fisher_audit(
            root,
            shared,
            read_state(state_path),
            seed=args.seed,
        )
        update_state(state_path, "fisher-audit", payload)
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-fisher":
        payload = run_adaptive_brec_fisher(
            root,
            shared,
        )
        update_state(
            state_path,
            "adaptive-brec-fisher",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-full-produce":
        payload = run_adaptive_brec_full_production(
            root,
            shared,
            workers=args.workers,
        )
        update_state(
            state_path,
            "adaptive-brec-full-produce",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-full-aggregate":
        payload = run_adaptive_brec_full_aggregate(
            root,
            shared,
        )
        update_state(
            state_path,
            "adaptive-brec-full-aggregate",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-full-fisher":
        payload = run_adaptive_brec_full_fisher(
            root,
            shared,
        )
        update_state(
            state_path,
            "adaptive-brec-full-fisher",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-full-mcmc":
        payload = run_adaptive_brec_full_mcmc(
            root,
            shared,
            ensembles=args.ensembles,
            walkers=args.walkers,
            steps=args.steps,
            workers=args.workers,
            seed=args.seed,
        )
        update_state(
            state_path,
            "adaptive-brec-full-mcmc",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "adaptive-brec-full-mcmc-finalize":
        payload = finalize_adaptive_brec_full_mcmc(
            root,
            shared,
            burn_fraction=args.burn_fraction,
            seed=args.seed,
        )
        update_state(
            state_path,
            "adaptive-brec-full-mcmc-finalize",
            payload,
        )
        print(json.dumps(jsonable(payload), indent=2))
        return
    if args.stage == "finalize":
        payload = finalize_products(
            root,
            shared,
            read_state(state_path),
            seed=args.seed,
        )
        update_state(state_path, "finalize", payload)
        print(json.dumps(jsonable(payload), indent=2))
        return
    context = build_context(
        shared,
        args.reconstruction,
        args.tier,
        float(args.kmax),
        wide_prior=bool(args.wide_prior),
    )
    suffix = context_key(context)
    if args.stage == "validate":
        payload = validate_context(
            context,
            np.random.default_rng(args.seed),
            random_points=args.random_points,
        )
        update_state(state_path, f"validate/{suffix}", payload)
    elif args.stage == "benchmark":
        payload = benchmark_context(
            context,
            evaluations=args.evaluations,
            rng=np.random.default_rng(args.seed),
        )
        update_state(state_path, f"benchmark/{suffix}", payload)
    elif args.stage in {"smoke", "production"}:
        payload = run_mcmc_context(
            context,
            chain_path=root / CHAINS_RELATIVE,
            ensembles=args.ensembles,
            walkers=args.walkers,
            steps=args.steps,
            workers=args.workers,
            seed=args.seed,
        )
        update_state(state_path, f"{args.stage}/{suffix}", payload)
    elif args.stage == "diagnose":
        payload = diagnose_context(
            context,
            chain_path=root / CHAINS_RELATIVE,
            burn_fraction=args.burn_fraction,
        )
        update_state(state_path, f"diagnose/{suffix}", payload)
    elif args.stage == "explicit":
        payload = run_explicit_validation(
            context,
            explicit_chain_path=root / EXPLICIT_CHAINS_RELATIVE,
            collapsed_chain_path=root / CHAINS_RELATIVE,
            ensembles=args.ensembles,
            walkers=args.walkers,
            steps=args.steps,
            workers=args.workers,
            seed=args.seed,
            burn_fraction=args.burn_fraction,
        )
        update_state(state_path, f"explicit/{suffix}", payload)
    elif args.stage == "student-t":
        payload = student_t_reweight(
            context,
            chain_path=root / CHAINS_RELATIVE,
            burn_fraction=args.burn_fraction,
            importance_samples=args.importance_samples,
            seed=args.seed,
        )
        update_state(state_path, f"student_t/{suffix}", payload)
    else:
        raise AssertionError(args.stage)
    print(json.dumps(jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
