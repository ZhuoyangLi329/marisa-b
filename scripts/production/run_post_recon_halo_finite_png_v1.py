#!/usr/bin/env python3
"""Run the post-reconstruction halo finite-PNG-v1 analysis.

The current entry first enforces the Gaussian gate.  It consumes the
versioned post-R1 EFT templates produced by the formal C++ driver, profiles
``b1`` and every identifiable bispectrum nuisance in the selected tier with
the covariance of one realization, and compares the coevolution and full
deterministic tiers.

The finite-PNG stage is added to the same entry after the Gaussian gate has
passed; a Gaussian failure is intentionally terminal for that stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "python"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import beta as beta_distribution
from scipy.stats import chi2 as chi2_distribution

from marisa_b.model import combine_finite_png


TAG = "post_recon_halo_finite_png_v1_20260727"
CUTS = (0.08, 0.10, 0.12, 0.14)
PRIMARY_TIER = "coevolution"
EXCLUDED_BIN = 0
B1_PRIOR_MEAN = 2.7465779027841823
B1_PRIOR_SIGMA = 2.0
B_REC_H = 2.7340475186190334
DELTA_C = 1.686
P_UNIVERSALITY = 1.0
FIDUCIAL_FNL = 0.0
FNL_PRIOR_HALF_WIDTH = 150.0
FNL_PRIOR_BOUNDS = (
    FIDUCIAL_FNL - FNL_PRIOR_HALF_WIDTH,
    FIDUCIAL_FNL + FNL_PRIOR_HALF_WIDTH,
)
FNL_FIXED_STARTS = (0.0, 100.0, -100.0, 140.0, -140.0)
FNL_RANDOM_START_SIGMA = 75.0
FNL_RANDOM_START_MARGIN = 10.0
SAMPLES = (
    ("fiducial", 0.0, "fiducial_post_B000", 1.95530218e-4),
)
PNG_CACHE_SCHEMA = "marisa-b-post-r1-finite-png-cache-v6"
PNG_JSONL_SCHEMA = "marisa-b-post-r1-finite-png-jsonl-v5"
PNG_JSONL_LEGACY_SCHEMAS = (
    "marisa-b-post-r1-finite-png-jsonl-v1",
    "marisa-b-post-r1-finite-png-jsonl-v2",
    "marisa-b-post-r1-finite-png-jsonl-v3",
    "marisa-b-post-r1-finite-png-jsonl-v4",
)
ADAPTIVE_BREC_RESPONSE_SCHEMA = (
    "marisa-b-eft-v2-post-r1-adaptive-brec-response-jsonl-v1"
)
PNG_TREE_FIELDS = (
    "stochastic_alpha3_basis",
    "dBdfNL_local_tree",
    "dBdfNL_local_primordial",
    "dBdfNL_local_bphi_f2",
    "dBdfNL_local_bphi_advection",
    "dBdfNL_local_bphidelta",
    "dBdfNL_local_bphi_b2",
    "dBdfNL_local_bphi_bK2",
    "dBdfNL_local_bphi_reconstruction",
    "dBdfNL_stochastic_alpha3_basis",
    "Bhalo_tree_fNL2_bphi_B0",
    "Bhalo_tree_fNL2_bphi_sq_advection",
    "Bhalo_tree_fNL2_bphi_sq_F2",
    "Bhalo_tree_fNL2_bphi_sq_b2",
    "Bhalo_tree_fNL2_bphi_sq_bK2",
    "Bhalo_tree_fNL2_bphi_sq_reconstruction",
    "Bhalo_tree_fNL2_bphi_bphidelta",
    "Bhalo_tree_fNL2_bphi2_operator",
    "Bhalo_tree_fNL2_deterministic",
    "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis",
)
MATTER_LINEAR_FIELDS = (
    "tree",
    "B122I",
    "B122II",
    "B113I",
    "B113II",
    "loop",
    "total",
)
MATTER_LINEAR_DIAGRAM_FIELDS = (
    "B122I",
    "B122II",
    "B113I",
    "B113II",
)
MATTER_LINEAR_NUMERICAL_FIELDS = (
    "B122I",
    "B122II",
    "B113I",
    "B113II",
    "loop",
    "total",
)
COEVOLUTION_DEPENDENT = (
    "gamma21",
    "gamma21x",
    "gamma211",
    "gamma22",
    "gamma31",
)
RELEASED_SHIFT_BASE = (
    "Bshot_residual",
    "d2",
    "dG2",
    "dGamma3",
)
RELEASED_SHIFT_PREFIX = "delta_noisy_shift__"
RELEASED_SHIFT_PRIOR_SCALE = 1.0
PNG_STOCHASTIC_NAME = "Bshot_PNG_residual"
PNG_STOCHASTIC_PRIOR_SCALE = 1.0
PRIMARY_PNG_STOCHASTIC_MODEL = "independent_eq265"
PNG_STOCHASTIC_MODELS = (
    "independent_eq265",
    "strict_tied_eq265",
    "legacy_accepted_pre",
)
PNG_TAIL_CANARY_INDICES = (6, 75)
PNG_TAIL_CANARY_QMAX = (10.0, 40.0)
PNG_TAIL_REFERENCE_QMAX = 20.0
PNG_TAIL_FNL_ENVELOPE = 100.0
PNG_NUMERICAL_VECTOR_THRESHOLD_SIGMA = 0.05
PNG_NUMERICAL_BIN_THRESHOLD_SIGMA = 0.1
PDF_METADATA = {
    "Author": "MARISA-B post-reconstruction halo finite-PNG v1",
    "Creator": "scripts/production/run_post_recon_halo_finite_png_v1.py",
    "Subject": (
        "Quijote z=1 R=15 real-space post-reconstruction halo B000"
    ),
}
FINAL_FIGURE_NAMES = (
    "post_gaussian_k2B000_bestfit.pdf",
    "post_full2d_bestfit_fnl0.pdf",
    "post_fnl_profiles_vs_kmax.pdf",
    "post_fnl_recovery_vs_kmax.pdf",
    "post_full_vs_coevolution.pdf",
    "pre_vs_post_fnl_constraints.pdf",
    "post_model_component_audit.pdf",
)


@dataclass(frozen=True)
class Paths:
    repo: Path
    worktree: Path
    data_root: Path
    output_root: Path
    templates: Path
    matrix: Path
    quarantined_halo_png_matrix: Path
    png_cache: Path
    contract: Path
    pre_runner: Path
    frozen_fitter: Path
    analysis: Path
    figures: Path


@dataclass(frozen=True)
class PostTemplateSet:
    tied: Any
    density_only: Any
    noisy_shift: Any
    stochastic_names: tuple[str, ...]
    fixed_poisson_terms: dict[
        int, dict[str, np.ndarray]
    ]

    @staticmethod
    def _monomial_value(
        monomial: str,
        parameters: dict[str, float],
    ) -> float:
        value = 1.0
        if monomial != "1":
            for factor in monomial.split("*"):
                name, power = factor.rsplit("^", 1)
                value *= parameters[name] ** int(power)
        return value

    def fixed_poisson(
        self,
        parameters: dict[str, float],
        number_density: float,
    ) -> np.ndarray:
        result = np.zeros(120, dtype=np.float64)
        cache: dict[str, float] = {"1": 1.0}
        for inverse_nbar, terms in self.fixed_poisson_terms.items():
            for monomial, coefficients in terms.items():
                if monomial not in cache:
                    cache[monomial] = self._monomial_value(
                        monomial,
                        parameters,
                    )
                result += (
                    cache[monomial]
                    * coefficients
                    / number_density**inverse_nbar
                )
        return result

    def _add_fixed_poisson(
        self,
        components: dict[str, np.ndarray],
        parameters: dict[str, float],
        number_density: float,
    ) -> dict[str, np.ndarray]:
        result = {
            name: np.asarray(value).copy()
            for name, value in components.items()
        }
        fixed = self.fixed_poisson(
            parameters,
            number_density,
        )
        result["fixed_poisson"] = fixed
        result["total"] += fixed
        return result

    def components(
        self,
        parameters: dict[str, float],
        number_density: float,
        stochastic_mode: str,
    ) -> dict[str, np.ndarray]:
        # ``a5_mixed`` is the eliminated coordinate of the exact
        # shell-projected stochastic gauge.  Fit-time parameter dictionaries
        # produced by ``full_parameter_dict`` already restore it, whereas
        # serialized/expanded best-fit dictionaries intentionally omit it.
        # Normalize both entry paths here so every downstream component
        # evaluation implements the registered a5_mixed=0 gauge.
        parameters = dict(parameters)
        parameters.setdefault("a5_mixed", 0.0)
        if stochastic_mode == "zero":
            return self._add_fixed_poisson(
                self.density_only.components(
                    parameters,
                    number_density,
                ),
                parameters,
                number_density,
            )
        tied = self.tied.components(
            parameters,
            number_density,
        )
        if stochastic_mode == "tied":
            return self._add_fixed_poisson(
                tied,
                parameters,
                number_density,
            )
        if stochastic_mode != "released":
            raise ValueError(stochastic_mode)
        shift_parameters = dict(parameters)
        for name in self.stochastic_names:
            shift_parameters[name] = 0.0
        for name in RELEASED_SHIFT_BASE:
            shift_parameters[name] = float(
                parameters[
                    RELEASED_SHIFT_PREFIX + name
                ]
            )
        released_shift = self.noisy_shift.components(
            shift_parameters,
            number_density,
        )["stochastic"]
        result = {
            name: np.asarray(value).copy()
            for name, value in tied.items()
        }
        result["stochastic"] += released_shift
        result["total"] += released_shift
        result["released_noisy_shift_delta"] = (
            released_shift
        )
        return self._add_fixed_poisson(
            result,
            parameters,
            number_density,
        )


@dataclass(frozen=True)
class PngTemplateSet:
    indices: np.ndarray
    edges: np.ndarray
    tree: dict[str, np.ndarray]
    fixed_gaussian: dict[str, np.ndarray]
    fixed_linear: dict[str, np.ndarray]
    fixed_quadratic: dict[str, np.ndarray]
    residual_gaussian: dict[str, dict[str, np.ndarray]]
    residual_linear: dict[str, dict[str, np.ndarray]]
    residual_quadratic: dict[str, dict[str, np.ndarray]]
    matter_linear_coefficients: dict[str, np.ndarray]
    matter_b112_coefficients: np.ndarray
    matter_linear_validation_lambdas: np.ndarray
    matter_linear_loop_validation_residuals: np.ndarray
    matter_linear_error_lambdas: np.ndarray
    matter_linear_error_bounds: dict[str, np.ndarray]
    matter_b112_validation_lambdas: np.ndarray
    matter_b112_validation_residuals: np.ndarray
    matter_b112_error_lambdas: np.ndarray
    matter_b112_error_bounds: np.ndarray
    metadata: dict[str, Any]

    @staticmethod
    def _evaluate_polynomial(
        coefficients: np.ndarray,
        value: float,
    ) -> np.ndarray:
        result = np.zeros(coefficients.shape[1], dtype=np.float64)
        for row in coefficients[::-1]:
            result = result * value + row
        return result

    @staticmethod
    def _evaluate_sparse(
        terms: dict[str, np.ndarray],
        parameters: dict[str, float],
    ) -> np.ndarray:
        result = np.zeros(120, dtype=np.float64)
        cache: dict[str, float] = {"1": 1.0}
        for monomial, coefficients in terms.items():
            if monomial not in cache:
                cache[monomial] = PostTemplateSet._monomial_value(
                    monomial,
                    parameters,
                )
            result += cache[monomial] * coefficients
        return result

    def components(
        self,
        parameters: dict[str, float],
        number_density: float,
        stochastic_mode: str,
        png_stochastic_model: str = PRIMARY_PNG_STOCHASTIC_MODEL,
    ) -> dict[str, np.ndarray]:
        if not (
            math.isfinite(number_density)
            and number_density > 0.0
        ):
            raise ValueError("PNG number density must be positive")
        b1 = float(parameters["b1"])
        b2_native = (
            float(parameters["b2"])
            - 4.0 * float(parameters["gamma2"]) / 3.0
        )
        gamma2 = float(parameters["gamma2"])
        bphi = 2.0 * DELTA_C * (b1 - P_UNIVERSALITY)
        b2_lagrangian = (
            b2_native - 8.0 * (b1 - 1.0) / 21.0
        )
        bphidelta = bphi + 2.0 * (
            DELTA_C * b2_lagrangian - b1 + 1.0
        )
        bphi2 = 4.0 * DELTA_C * (
            DELTA_C * b2_lagrangian - 2.0 * (b1 - 1.0)
        )
        if stochastic_mode == "zero":
            residual_map = "density_only"
        elif stochastic_mode in {"tied", "released"}:
            residual_map = "tied"
        else:
            raise ValueError(stochastic_mode)
        if png_stochastic_model not in PNG_STOCHASTIC_MODELS:
            raise ValueError(png_stochastic_model)
        gaussian_bshot = float(parameters["Bshot_residual"])
        png_bshot = float(
            parameters.get(PNG_STOCHASTIC_NAME, 0.0)
        )
        residual_linear_shape = (
            bphi
            * self._evaluate_sparse(
                self.residual_linear[residual_map],
                parameters,
            )
            / number_density
        )
        residual_quadratic_shape = (
            bphi**2
            * self._evaluate_sparse(
                self.residual_quadratic[residual_map],
                parameters,
            )
            / number_density
        )
        if png_stochastic_model == "independent_eq265":
            alpha3_linear = (
                gaussian_bshot * residual_linear_shape
            )
            alpha3png_linear = (
                png_bshot * residual_linear_shape
            )
            alpha3png_quadratic = (
                png_bshot * residual_quadratic_shape
            )
        elif png_stochastic_model == "strict_tied_eq265":
            alpha3_linear = (
                gaussian_bshot * residual_linear_shape
            )
            alpha3png_linear = (
                gaussian_bshot * residual_linear_shape
            )
            alpha3png_quadratic = (
                gaussian_bshot * residual_quadratic_shape
            )
        else:
            # Historical pre-reconstruction compatibility only.  This
            # G+fL+f^2Q closure is not the two-amplitude decomposition of
            # Moradinezhad Dizgah et al. Eq. (2.65).
            alpha3_linear = (
                gaussian_bshot * residual_linear_shape
            )
            alpha3png_linear = np.zeros(
                120, dtype=np.float64
            )
            alpha3png_quadratic = (
                gaussian_bshot * residual_quadratic_shape
            )
        residual_linear = alpha3_linear + alpha3png_linear
        residual_quadratic = alpha3png_quadratic
        released_linear = np.zeros(120, dtype=np.float64)
        released_quadratic = np.zeros(120, dtype=np.float64)
        if stochastic_mode == "released":
            delta_bshot = float(
                parameters[
                    RELEASED_SHIFT_PREFIX + "Bshot_residual"
                ]
            )
            released_linear = (
                delta_bshot
                * bphi
                * self._evaluate_sparse(
                    self.residual_linear["noisy_shift"],
                    parameters,
                )
                / number_density
            )
            released_quadratic = (
                delta_bshot
                * bphi**2
                * self._evaluate_sparse(
                    self.residual_quadratic["noisy_shift"],
                    parameters,
                )
                / number_density
            )

        tree = self.tree
        linear_parts = {
            "halo_primordial": b1**3
            * tree["dBdfNL_local_primordial"],
            "halo_bphi_f2": b1**2
            * bphi
            * tree["dBdfNL_local_bphi_f2"],
            "halo_bphi_advection": b1**2
            * bphi
            * tree["dBdfNL_local_bphi_advection"],
            "halo_bphidelta": b1**2
            * bphidelta
            * tree["dBdfNL_local_bphidelta"],
            "halo_bphi_b2": b1
            * bphi
            * b2_native
            * tree["dBdfNL_local_bphi_b2"],
            "halo_bphi_bK2": b1
            * bphi
            * gamma2
            * tree["dBdfNL_local_bphi_bK2"],
            "halo_reconstruction": b1**3
            * bphi
            * tree["dBdfNL_local_bphi_reconstruction"],
            "residual_stochastic_linear": residual_linear,
            "released_noisy_shift_linear": released_linear,
            "fixed_poisson_linear": (
                bphi
                * self._evaluate_sparse(
                    self.fixed_linear,
                    parameters,
                )
                / number_density
            ),
        }
        lambda_value = b1 / B_REC_H
        matter_loop = self._evaluate_polynomial(
            self.matter_linear_coefficients["loop"],
            lambda_value,
        )
        linear_parts["matter_one_loop_uplift"] = (
            b1**3 * matter_loop
        )
        linear_total = sum(
            linear_parts.values(),
            np.zeros(120, dtype=np.float64),
        )

        quadratic_parts = {
            "halo_bphi_B0": b1**2
            * bphi
            * tree["Bhalo_tree_fNL2_bphi_B0"],
            "halo_bphi_sq_advection": b1
            * bphi**2
            * tree[
                "Bhalo_tree_fNL2_bphi_sq_advection"
            ],
            "halo_bphi_sq_F2": b1
            * bphi**2
            * tree["Bhalo_tree_fNL2_bphi_sq_F2"],
            "halo_bphi_sq_b2": bphi**2
            * b2_native
            * tree["Bhalo_tree_fNL2_bphi_sq_b2"],
            "halo_bphi_sq_bK2": bphi**2
            * gamma2
            * tree["Bhalo_tree_fNL2_bphi_sq_bK2"],
            "halo_bphi_sq_reconstruction": b1**2
            * bphi**2
            * tree[
                "Bhalo_tree_fNL2_bphi_sq_reconstruction"
            ],
            "halo_bphi_bphidelta": b1
            * bphi
            * bphidelta
            * tree[
                "Bhalo_tree_fNL2_bphi_bphidelta"
            ],
            "halo_bphi2_operator": b1**2
            * bphi2
            * tree["Bhalo_tree_fNL2_bphi2_operator"],
            "residual_stochastic_quadratic": residual_quadratic,
            "released_noisy_shift_quadratic": (
                released_quadratic
            ),
            "fixed_poisson_quadratic": (
                bphi**2
                * self._evaluate_sparse(
                    self.fixed_quadratic,
                    parameters,
                )
                / number_density
            ),
            "matter_B112II": b1**3
            * self._evaluate_polynomial(
                self.matter_b112_coefficients,
                lambda_value,
            ),
        }
        quadratic_total = sum(
            quadratic_parts.values(),
            np.zeros(120, dtype=np.float64),
        )
        return {
            **linear_parts,
            **quadratic_parts,
            "residual_stochastic_alpha3_linear": (
                alpha3_linear
            ),
            "residual_stochastic_alpha3PNG_linear": (
                alpha3png_linear
            ),
            "residual_stochastic_alpha3PNG_quadratic": (
                alpha3png_quadratic
            ),
            "linear_total": linear_total,
            "quadratic_total": quadratic_total,
            "bphi": np.asarray(bphi),
            "bphidelta": np.asarray(bphidelta),
            "bphi2": np.asarray(bphi2),
            "b2_native": np.asarray(b2_native),
            "lambda": np.asarray(lambda_value),
        }


def combine_finite_png_prediction(
    gaussian_total: np.ndarray,
    response: dict[str, np.ndarray],
    fnl: float,
) -> np.ndarray:
    """Apply the production finite-fNL recombination in one place."""

    return combine_finite_png(
        gaussian_total,
        response["linear_total"],
        response["quadratic_total"],
        fnl,
    )


def hard_flat_fnl_log_prior(fnl: float) -> float:
    """Return the registered truth-independent hard top-hat log prior."""

    fnl = float(fnl)
    if not math.isfinite(fnl):
        return -math.inf
    lower, upper = FNL_PRIOR_BOUNDS
    return 0.0 if lower <= fnl <= upper else -math.inf


def fnl_prior_contract() -> dict[str, Any]:
    """Validate and serialize the production fNL-prior convention."""

    lower, upper = FNL_PRIOR_BOUNDS
    outside_lower = np.nextafter(lower, -math.inf)
    outside_upper = np.nextafter(upper, math.inf)
    checks = {
        "finite_symmetric_bounds": bool(
            math.isfinite(lower)
            and math.isfinite(upper)
            and math.isclose(
                0.5 * (lower + upper),
                FIDUCIAL_FNL,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                0.5 * (upper - lower),
                FNL_PRIOR_HALF_WIDTH,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ),
        "constant_inside_support": bool(
            all(
                hard_flat_fnl_log_prior(value) == 0.0
                for value in (
                    lower,
                    FIDUCIAL_FNL,
                    upper,
                    -100.0,
                    100.0,
                )
            )
        ),
        "negative_infinity_outside_support": bool(
            all(
                hard_flat_fnl_log_prior(value) == -math.inf
                for value in (
                    outside_lower,
                    outside_upper,
                    -math.inf,
                    math.inf,
                    math.nan,
                )
            )
        ),
        "all_registered_truths_inside_support": bool(
            all(
                hard_flat_fnl_log_prior(truth) == 0.0
                for _sample, truth, _key, _nbar in SAMPLES
            )
        ),
        "all_fixed_starts_strictly_inside_support": bool(
            all(lower < value < upper for value in FNL_FIXED_STARTS)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "registered fNL hard-flat prior contract fails: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "kind": "hard_uniform_top_hat",
        "center": FIDUCIAL_FNL,
        "half_width": FNL_PRIOR_HALF_WIDTH,
        "bounds": [lower, upper],
        "inside_support_log_prior": 0.0,
        "outside_support_log_prior": "-Infinity",
        "sample_truth_independent": True,
        "injection_label_used_to_center_prior": False,
        "checks": checks,
        "pass": True,
    }


@dataclass
class FinitePngProfile:
    tier: str
    sample: str
    truth_fnl: float
    kmax: float
    stochastic_mode: str
    png_stochastic_model: str
    precision_correction: str
    nuisance_prior_scale: float
    indices: np.ndarray
    free_names: tuple[str, ...]
    nonlinear_names: tuple[str, ...]
    nuisance_values: np.ndarray
    expanded_parameters: dict[str, float]
    prediction: np.ndarray
    gaussian_prediction: np.ndarray
    linear_response: np.ndarray
    quadratic_coefficient: np.ndarray
    target: np.ndarray
    covariance_single: np.ndarray
    covariance_mock_count: int
    hartlap_factor: float
    fhat: float
    error_low: float
    error_high: float
    sigma_symmetric: float
    objective: float
    data_chi2: float
    prior_chi2: float
    dof: float
    p_value: float
    max_pull_single: float
    residual_norm_single: float
    minuit_valid: bool
    minuit_accurate: bool
    minuit_at_limit: bool
    minuit_nfcn: int
    profile_x: np.ndarray
    profile_delta: np.ndarray
    profile_curve_kind: str
    nuisance_jacobian: np.ndarray
    multistart_objectives: np.ndarray
    multistart_fnl: np.ndarray
    multistart_valid: np.ndarray
    multistart_at_limit: np.ndarray
    multistart_objective_spread: float
    multistart_agree: bool
    minos_valid: bool
    minos_lower_crossing_valid: bool
    minos_upper_crossing_valid: bool
    minos_at_lower_prior_limit: bool
    minos_at_upper_prior_limit: bool
    profile_global_consistent: bool
    profile_interval_consistency_sigma: float
    affine_reconstruction_error_sigma_single: float
    response_components: dict[str, np.ndarray]


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
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument(
        "--png-cache",
        type=Path,
        help=(
            "Compiled exact-lattice finite-PNG template cache; defaults "
            "to the versioned analysis directory."
        ),
    )
    parser.add_argument(
        "--png-jsonl",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Raw tree-fixed, matter-linear, and matter-b112 JSONL files "
            "to validate and compile into --png-cache."
        ),
    )
    parser.add_argument(
        "--gaussian-reference",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional alternate post-R1 Gaussian JSONL for "
            "covariance-weighted convergence diagnostics."
        ),
    )
    parser.add_argument(
        "--png-reference-cache",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional alternate compiled PNG cache for finite-box or "
            "quadrature robustness profiles."
        ),
    )
    parser.add_argument(
        "--png-tail-canary",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Exactly eight raw v5 matter-linear/matter-b112 JSONL "
            "canaries spanning bins 6/75 and qmax 10/40.  Stage all "
            "compares them with the qmax=20 production cache in the "
            "single-realization covariance metric."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("gaussian", "all"),
        default="gaussian",
    )
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument(
        "--coverage-realizations",
        type=int,
        default=50,
        help=(
            "Number of preregistered fiducial fNL=0 realizations, held "
            "out of the coverage covariance pool, to reprofile with "
            "actual Minuit/MINOS at kmax=0.14."
        ),
    )
    parser.add_argument(
        "--coverage-multistart",
        type=int,
        default=5,
        help="Truth-blind Minuit starts for every coverage realization.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--finalize-reviews",
        action="store_true",
        help=(
            "Do not rerun theory or fits.  Validate REVIEWS.json "
            "against the frozen summary, profiles, figures, sources, "
            "and independent review reports, then atomically promote "
            "a pending scientific pass to the final pass state."
        ),
    )
    parser.add_argument(
        "--generate-verification",
        action="store_true",
        help=(
            "Run the exact frozen native/Python test registry "
            "sequentially, archive hashed logs, generate "
            "VERIFICATION.json, and self-validate it without "
            "running theory or inference."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path, files: list[Path]) -> str:
    """Hash both relative names and bytes of a registered source tree."""

    root = root.resolve()
    resolved = sorted(
        {path.resolve() for path in files},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not resolved:
        raise ValueError("registered source tree is empty")
    digest = hashlib.sha256()
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(
                f"registered source-tree member is missing: {path}"
            )
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def depfile_source_union(
    *,
    repo: Path,
    worktree: Path,
    depfiles: Iterable[Path],
) -> list[Path]:
    """Return and validate the exact source union declared by ``-MMD``.

    The generated dependency files are included in the hash as declarations,
    while every source/header on their primary target rule is resolved and
    included as content.  This closes over halo-v1, ACTio-ReACTio, and the
    explicitly registered Conda GSL headers that are outside ``src/eft_v2``
    but enter the post-R1 driver compilation.
    """

    repo = repo.resolve()
    worktree = worktree.resolve()
    result: set[Path] = {
        (worktree / "Makefile").resolve(),
    }
    for unresolved in depfiles:
        depfile = unresolved.resolve()
        if not depfile.is_file():
            raise FileNotFoundError(
                f"registered compiler depfile is missing: {depfile}"
            )
        result.add(depfile)
        logical = depfile.read_text(encoding="utf-8").replace(
            "\\\n",
            " ",
        )
        primary_rule = logical.splitlines()[0]
        if ":" not in primary_rule:
            raise ValueError(
                f"compiler depfile lacks a primary rule: {depfile}"
            )
        _target, dependency_text = primary_rule.split(":", 1)
        dependencies = shlex.split(dependency_text)
        if not dependencies:
            raise ValueError(
                f"compiler depfile has no source dependencies: {depfile}"
            )
        for token in dependencies:
            source = Path(token)
            if not source.is_absolute():
                source = worktree / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(
                    "registered compiler dependency is missing: "
                    f"{source}"
                )
            allowed_roots = [repo, Path(sys.prefix).resolve(), Path("/usr")]
            gsl_prefix = os.environ.get("GSL_PREFIX", "").strip()
            if gsl_prefix:
                allowed_roots.append(Path(gsl_prefix).expanduser().resolve())
            if not any(
                source.is_relative_to(root)
                for root in allowed_roots
            ):
                raise ValueError(
                    "compiler dependency escapes the repository and "
                    "registered build environment: "
                    f"{source}"
                )
            result.add(source)
    return sorted(
        result,
        key=lambda source: source.as_posix(),
    )


def verification_test_contract(
    paths: Paths,
) -> dict[str, dict[str, Any]]:
    """Return the one authoritative production-test registry."""

    build = paths.worktree / "build/halo_v1"
    power = paths.repo / "tests/data/quijote_fiducial_plin_z1.dat"
    python = Path(sys.executable)
    make_path = shutil.which("make")
    if make_path is None:
        raise FileNotFoundError("make")
    make = Path(make_path)
    binary_tests = {
        "post_r1_png_native": (
            104,
            build / "test_eft_v2_post_r1_png_native",
            (),
            "no-gsl-runtime-dependency",
        ),
        "template_algebra": (
            2107,
            build / "test_eft_v2_template_algebra",
            (),
            "current-runtime",
        ),
        "bias_operators": (
            44641,
            build / "test_eft_v2_bias_operators",
            (),
            "current-runtime",
        ),
        "diagrams": (
            696,
            build / "test_eft_v2_diagrams",
            (power,),
            "current-runtime",
        ),
        "uv_renormalization": (
            749,
            build / "test_eft_v2_uv_renormalization",
            (power,),
            "current-runtime",
        ),
        "counterterms": (
            354,
            build / "test_eft_v2_counterterms",
            (),
            "current-runtime",
        ),
        "stochastic": (
            485,
            build / "test_eft_v2_stochastic",
            (),
            "current-runtime",
        ),
        "ir_resummation": (
            408,
            build / "test_eft_v2_ir_resummation",
            (),
            "current-runtime",
        ),
        "shell_projector": (
            782,
            build / "test_eft_v2_shell_projector",
            (),
            "current-runtime",
        ),
        "tracer_power": (
            330,
            build / "test_eft_v2_tracer_power",
            (),
            "current-runtime",
        ),
        "halo_v1_legacy": (
            25414,
            build / "test_halo_v1",
            (power,),
            "current-runtime",
        ),
    }
    expected: dict[str, dict[str, Any]] = {}
    for label, (
        checks,
        executable,
        arguments,
        runtime_abi,
    ) in binary_tests.items():
        argv = (
            str(executable.resolve()),
            *(str(argument.resolve()) for argument in arguments),
        )
        expected[label] = {
            "checks": checks,
            "executable_path": str(executable.resolve()),
            "executable_sha256": sha256(executable),
            "command": " ".join(argv),
            "runtime_abi": runtime_abi,
            "argv": argv,
            "success_marker": (
                f"checks passed: {checks}",
                f"checks={checks}",
            ),
        }
    fit_statistics = (
        paths.worktree / "tests/legacy/test_fit_eft_v2_r0_mean.py"
    )
    fit_argv = (
        str(python.resolve()),
        str(fit_statistics.resolve()),
    )
    expected["fit_statistics"] = {
        "checks": 10,
        "executable_path": str(python.resolve()),
        "executable_sha256": sha256(python),
        "command": " ".join(fit_argv),
        "runtime_abi": "current-python",
        "argv": fit_argv,
        "success_marker": ("EFT-v2 fit-statistics tests passed",),
    }
    freshness_targets = [
        build / "eft_v2_post_r1_template_driver",
        build / "eft_v2_post_r1_png_template_driver",
        *(row[1] for row in binary_tests.values()),
    ]
    freshness_argv = (
        str(make.resolve()),
        "-C",
        str(paths.worktree.resolve()),
        "-q",
        *(
            str(target.relative_to(paths.worktree))
            for target in freshness_targets
        ),
    )
    expected["build_freshness"] = {
        "checks": 1,
        "executable_path": str(make.resolve()),
        "executable_sha256": sha256(make),
        "command": " ".join(freshness_argv),
        "runtime_abi": "make-q-no-rebuild",
        "argv": freshness_argv,
        "success_marker": (),
    }
    return expected


def verification_source_hashes(paths: Paths) -> dict[str, str]:
    """Hash the complete registered production source/dependency closure."""

    build = paths.worktree / "build/halo_v1"
    expected_sources = {
        "runner": Path(__file__).resolve(),
        "variant_runner": (
            paths.repo
            / "scripts/production/"
            "run_post_recon_halo_finite_png_v1_variants.sh"
        ),
        "frozen_fitter": paths.frozen_fitter,
        "pre_runner": paths.pre_runner,
        "native_cpp": (
            paths.repo / "src/marisa_b/marisa_b_native.cpp"
        ),
        "native_header": (
            paths.repo / "src/marisa_b/marisa_b_native.h"
        ),
        "png_driver_source": (
            paths.worktree
            / "src/eft_v2/post_r1_png_template_driver.cpp"
        ),
        "gaussian_driver_source": (
            paths.worktree
            / "src/eft_v2/post_r1_template_driver.cpp"
        ),
        "png_driver_binary": (
            paths.worktree
            / "build/halo_v1/eft_v2_post_r1_png_template_driver"
        ),
        "gaussian_driver_binary": (
            paths.worktree
            / "build/halo_v1/eft_v2_post_r1_template_driver"
        ),
    }
    eft_v2_sources = [
        path
        for path in (paths.worktree / "src/eft_v2").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}
    ]
    eft_v2_sources.extend(
        (
            paths.worktree / "src/halo_v1/shell_average.cpp",
            paths.worktree / "src/halo_v1/shell_average.h",
            paths.worktree / "Makefile",
        )
    )
    eft_v2_tests = [
        path
        for path in (paths.worktree / "tests/eft_v2").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp", ".py"}
    ]
    eft_v2_tests.append(
        paths.worktree / "tests/test_halo_v1.cpp"
    )
    eft_object_stems = (
        "eft_v2_parameter_registry",
        "eft_v2_template_algebra",
        "eft_v2_kernel_primitives",
        "eft_v2_bias_operators",
        "eft_v2_field_kernel_provider",
        "eft_v2_diagram_assembler",
        "eft_v2_ir_safe_integrands",
        "eft_v2_uv_subtraction",
        "eft_v2_direct_evaluator",
        "eft_v2_fftlog_dr_oracle",
        "eft_v2_counterterms",
        "eft_v2_stochastic",
        "eft_v2_poisson_reconstruction",
        "eft_v2_ir_resummation",
        "eft_v2_lattice_shell_rule",
        "eft_v2_shell_projector",
        "eft_v2_tracer_power",
    )
    gaussian_object_stems = (
        "eft_v2_post_r1_template_driver",
        *eft_object_stems,
        "halo_v1",
        "shell_average",
        "marisa_b_native",
        "Common",
        "PowerSpectrum",
        "array",
        "Quadrature",
    )
    png_object_stems = (
        "eft_v2_post_r1_png_template_driver",
        *eft_object_stems,
        "halo_v1",
        "shell_average",
        "portable_marisa_b_native",
        "Common",
        "PowerSpectrum",
        "array",
        "Quadrature",
    )
    gaussian_dependencies = depfile_source_union(
        repo=paths.repo,
        worktree=paths.worktree,
        depfiles=[
            build / f"{stem}.d"
            for stem in gaussian_object_stems
        ],
    )
    png_dependencies = depfile_source_union(
        repo=paths.repo,
        worktree=paths.worktree,
        depfiles=[
            build / f"{stem}.d"
            for stem in png_object_stems
        ],
    )
    result = {
        label: sha256(source)
        for label, source in expected_sources.items()
    }
    result["eft_v2_compiled_source_tree"] = (
        source_tree_sha256(paths.worktree, eft_v2_sources)
    )
    result["eft_v2_test_source_tree"] = (
        source_tree_sha256(paths.worktree, eft_v2_tests)
    )
    result["gaussian_driver_dependency_tree"] = (
        source_tree_sha256(Path("/"), gaussian_dependencies)
    )
    result["png_driver_dependency_tree"] = (
        source_tree_sha256(Path("/"), png_dependencies)
    )
    return result


def load_verification(paths: Paths) -> dict[str, Any]:
    path = paths.analysis / "VERIFICATION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "marisa-b-post-r1-verification-v1"
        or payload.get("passed") is not True
        or not isinstance(payload.get("tests"), list)
    ):
        raise ValueError(
            "VERIFICATION.json does not certify all production tests"
        )
    expected_tests = verification_test_contract(paths)
    rows = payload["tests"]
    if (
        any(not isinstance(row, dict) for row in rows)
        or len(rows) != len(expected_tests)
        or {row.get("label") for row in rows}
        != set(expected_tests)
    ):
        raise ValueError(
            "VERIFICATION.json does not contain the exact registered "
            "production-test set"
        )
    archive_root = (
        paths.output_root
        / "log/post_recon_halo_finite_png_v1_20260727"
    ).resolve()
    expected_test_fields = {
        "label",
        "passed",
        "checks",
        "command",
        "runtime_abi",
        "executable_path",
        "executable_sha256",
        "log_path",
        "log_sha256",
    }
    test_failures: dict[str, Any] = {}
    for row in rows:
        label = str(row["label"])
        expected = expected_tests[label]
        log_path = Path(str(row.get("log_path", ""))).resolve()
        log_inside_archive = log_path.is_relative_to(archive_root)
        actual_log_sha = (
            sha256(log_path) if log_path.is_file() else None
        )
        log_text = (
            log_path.read_text(encoding="utf-8")
            if log_path.is_file()
            else ""
        )
        log_contract = all(
            marker in log_text
            for marker in (
                f"label={label}\n",
                f"command={expected['command']}\n",
                f"runtime_abi={expected['runtime_abi']}\n",
                "returncode=0\n",
                "success_marker_pass=True\n",
                "--- stdout ---\n",
                "--- stderr ---\n",
            )
        )
        failures = {
            "exact_fields": set(row) == expected_test_fields,
            "passed": row.get("passed") is True,
            "checks": row.get("checks") == expected["checks"],
            "command": row.get("command") == expected["command"],
            "runtime_abi": (
                row.get("runtime_abi") == expected["runtime_abi"]
            ),
            "executable_path": (
                row.get("executable_path")
                == expected["executable_path"]
            ),
            "executable_sha256": (
                row.get("executable_sha256")
                == expected["executable_sha256"]
            ),
            "log_inside_archive": log_inside_archive,
            "log_sha256": (
                actual_log_sha is not None
                and row.get("log_sha256") == actual_log_sha
            ),
            "log_contract": log_contract,
        }
        if not all(failures.values()):
            test_failures[label] = failures
    if test_failures:
        raise ValueError(
            "VERIFICATION.json production-test provenance fails: "
            + json.dumps(test_failures, sort_keys=True)
        )
    expected_hashes = verification_source_hashes(paths)
    recorded = payload.get("source_hashes")
    if (
        not isinstance(recorded, dict)
        or set(recorded) != set(expected_hashes)
    ):
        raise ValueError(
            "VERIFICATION.json has an incomplete source-hash contract"
        )
    mismatches = {
        label: {
            "recorded": recorded[label],
            "actual": expected_hash,
        }
        for label, expected_hash in expected_hashes.items()
        if recorded[label] != expected_hash
    }
    if mismatches:
        raise ValueError(
            "production code changed after verification: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def generate_verification(
    paths: Paths,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Run and hash the complete frozen production-test suite."""

    output = paths.analysis / "VERIFICATION.json"
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} exists; use --overwrite"
        )
    contract = verification_test_contract(paths)
    archive_root = (
        paths.output_root
        / "log/post_recon_halo_finite_png_v1_20260727"
    )
    log_root = archive_root / "verification_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_paths = {
        label: log_root / f"{label}.log"
        for label in contract
    }
    existing_logs = [
        str(path)
        for path in log_paths.values()
        if path.exists()
    ]
    if existing_logs and not overwrite:
        raise FileExistsError(
            "verification logs exist; use --overwrite: "
            + ", ".join(existing_logs)
        )
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    library_roots: dict[str, Path] = {}
    gsl_prefix = os.environ.get("GSL_PREFIX", "").strip()
    if gsl_prefix:
        prefix = Path(gsl_prefix).expanduser().resolve()
        candidates = sorted((prefix / "lib").glob("*-linux-gnu"))
        library_roots["current-runtime"] = (
            candidates[0] if candidates else prefix / "lib"
        )
    rows: list[dict[str, Any]] = []
    failed: dict[str, Any] = {}
    for label, expected in contract.items():
        environment = dict(base_environment)
        library_root = library_roots.get(
            str(expected["runtime_abi"])
        )
        if library_root is not None:
            if not library_root.is_dir():
                raise FileNotFoundError(
                    f"missing runtime ABI directory: {library_root}"
                )
            inherited = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = (
                str(library_root)
                + (f":{inherited}" if inherited else "")
            )
        argv = [str(value) for value in expected["argv"]]
        try:
            completed = subprocess.run(
                argv,
                cwd=paths.repo,
                env=environment,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            returncode: int | None = int(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            returncode = None
            stdout = (
                error.stdout.decode()
                if isinstance(error.stdout, bytes)
                else (error.stdout or "")
            )
            stderr = (
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else (error.stderr or "")
            )
            stderr += "\nverification timeout after 1800 seconds\n"
        combined = stdout + "\n" + stderr
        markers = tuple(expected["success_marker"])
        marker_pass = (
            not markers
            or any(marker in combined for marker in markers)
        )
        passed = bool(returncode == 0 and marker_pass)
        log_text = (
            f"label={label}\n"
            f"command={expected['command']}\n"
            f"runtime_abi={expected['runtime_abi']}\n"
            f"returncode={returncode}\n"
            f"success_marker_pass={marker_pass}\n"
            "--- stdout ---\n"
            f"{stdout}"
            + ("" if stdout.endswith("\n") or not stdout else "\n")
            + "--- stderr ---\n"
            f"{stderr}"
            + ("" if stderr.endswith("\n") or not stderr else "\n")
        )
        log_path = log_paths[label]
        temporary_log = log_path.with_name(
            f".{log_path.name}.tmp-{os.getpid()}"
        )
        temporary_log.write_text(log_text, encoding="utf-8")
        temporary_log.replace(log_path)
        row = {
            "label": label,
            "passed": passed,
            "checks": int(expected["checks"]),
            "command": str(expected["command"]),
            "runtime_abi": str(expected["runtime_abi"]),
            "executable_path": str(expected["executable_path"]),
            "executable_sha256": str(
                expected["executable_sha256"]
            ),
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256(log_path),
        }
        rows.append(row)
        if not passed:
            failed[label] = {
                "returncode": returncode,
                "success_marker_pass": marker_pass,
                "log_path": str(log_path),
            }
    if failed:
        raise RuntimeError(
            "production verification tests failed: "
            + json.dumps(failed, sort_keys=True)
        )
    payload = {
        "schema": "marisa-b-post-r1-verification-v1",
        "passed": True,
        "test_count": len(rows),
        "tests": rows,
        "source_hashes": verification_source_hashes(paths),
        "thread_contract": {
            "tests_run_sequentially": True,
            "blas_and_openmp_threads": 1,
            "maximum_cpu_cores": 1,
        },
    }
    candidate_analysis = (
        archive_root / f".verification_candidate_{os.getpid()}"
    )
    candidate_analysis.mkdir(parents=True, exist_ok=False)
    candidate_path = candidate_analysis / "VERIFICATION.json"
    atomic_json(candidate_path, payload)
    candidate_paths = Paths(
        repo=paths.repo,
        worktree=paths.worktree,
        templates=paths.templates,
        matrix=paths.matrix,
        quarantined_halo_png_matrix=(
            paths.quarantined_halo_png_matrix
        ),
        png_cache=paths.png_cache,
        contract=paths.contract,
        pre_runner=paths.pre_runner,
        frozen_fitter=paths.frozen_fitter,
        analysis=candidate_analysis,
        figures=paths.figures,
    )
    load_verification(candidate_paths)
    paths.analysis.mkdir(parents=True, exist_ok=True)
    candidate_path.replace(output)
    candidate_analysis.rmdir()
    return payload


def write_manifest(
    paths: Paths,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Hash the exact production inputs and deliverables.

    The manifest deliberately excludes itself.  It is rebuilt after both
    the expensive pending-review run and the non-recomputing review
    promotion, because finalization changes ``summary.json`` and
    ``REPORT.md``.  No quarantined halo-PNG catalogue is registered as a
    production input.
    """

    status = str(summary.get("status", ""))
    analysis_artifacts = {
        "analysis:plan": paths.analysis / "PLAN.md",
        "analysis:data_contract": (
            paths.analysis / "DATA_CONTRACT.md"
        ),
        "analysis:summary": paths.analysis / "summary.json",
        "analysis:profiles": paths.analysis / "profiles.npz",
        "analysis:report": paths.analysis / "REPORT.md",
        "analysis:verification": (
            paths.analysis / "VERIFICATION.json"
        ),
        "analysis:gaussian_gate": (
            paths.analysis / "gaussian_gate_diagnostic.json"
        ),
        "analysis:finite_png_cache": paths.png_cache,
    }
    if status == "pass":
        analysis_artifacts["analysis:independent_reviews"] = (
            paths.analysis / "REVIEWS.json"
        )
    production_inputs = {
        "input:fiducial_halo_matrix": paths.matrix,
        "input:gaussian_templates": paths.templates,
        "input:eft_v2_contract": paths.contract,
        "source:analysis_runner": Path(__file__).resolve(),
        "source:numerical_variant_runner": (
            paths.repo
            / "scripts/production/"
            "run_post_recon_halo_finite_png_v1_variants.sh"
        ),
        "source:pre_runner_helpers": paths.pre_runner,
        "source:frozen_eft_v2_fitter": paths.frozen_fitter,
    }

    def register_summary_input(
        label: str,
        row: dict[str, Any],
    ) -> None:
        raw_path = row.get("path")
        recorded_sha = row.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(
            recorded_sha,
            str,
        ):
            raise ValueError(
                f"summary input {label} lacks path/SHA256"
            )
        candidate = Path(raw_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"summary input {label} is missing: {candidate}"
            )
        actual_sha = sha256(candidate)
        if actual_sha != recorded_sha:
            raise ValueError(
                f"summary input {label} changed after analysis"
            )
        if label in production_inputs:
            raise ValueError(f"duplicate manifest label {label}")
        production_inputs[label] = candidate

    tail_files = (
        summary.get("inputs", {})
        .get("png_tail_canary", {})
        .get("files")
    )
    if not isinstance(tail_files, list) or len(tail_files) != 8:
        raise ValueError(
            "summary lacks the exact eight-file PNG tail canary"
        )
    tail_labels: set[str] = set()
    for row in tail_files:
        if not isinstance(row, dict):
            raise ValueError("invalid PNG tail-canary summary row")
        label = (
            "input:png_tail_canary:"
            f"{row.get('sector')}:"
            f"bin{row.get('index')}:"
            f"qmax{float(row.get('qmax_h_mpc')):g}"
        )
        if label in tail_labels:
            raise ValueError("duplicate PNG tail-canary manifest row")
        tail_labels.add(label)
        register_summary_input(label, row)

    numerical_variants = (
        summary.get("theory_gates", {})
        .get("numerical_variants", {})
    )
    if not isinstance(numerical_variants, dict):
        raise ValueError("summary lacks registered numerical variants")
    for index, row in enumerate(
        numerical_variants.get("gaussian", [])
    ):
        if not isinstance(row, dict):
            raise ValueError("invalid Gaussian-reference summary row")
        register_summary_input(
            f"input:gaussian_reference:{index:02d}",
            row,
        )
    for index, row in enumerate(
        numerical_variants.get("png", [])
    ):
        if not isinstance(row, dict):
            raise ValueError("invalid PNG-reference summary row")
        category = str(row.get("category", "unclassified"))
        register_summary_input(
            f"input:png_reference:{category}:{index:02d}",
            row,
        )
    pre_comparison = summary.get("pre_vs_post")
    if not isinstance(pre_comparison, dict):
        raise ValueError(
            "summary lacks the pre-reconstruction comparison input"
        )
    register_summary_input(
        "input:pre_reconstruction_comparison_summary",
        pre_comparison,
    )
    paired_pre_fit = (
        summary.get("theory_gates", {})
        .get("paired_pre_post_gaussian_closure", {})
        .get("pre_fit")
    )
    if not isinstance(paired_pre_fit, dict):
        raise ValueError(
            "summary lacks the paired pre/post Gaussian pre-fit input"
        )
    register_summary_input(
        "input:paired_reconstruction_pre_fit",
        paired_pre_fit,
    )

    figures = {
        f"figure:{name}": paths.figures / name
        for name in FINAL_FIGURE_NAMES
    }
    expected_figure_names = set(FINAL_FIGURE_NAMES)
    actual_figure_names = {
        path.name
        for path in paths.figures.glob("*.pdf")
        if path.is_file()
    }
    if actual_figure_names != expected_figure_names:
        raise ValueError(
            "final figure directory does not contain the exact "
            "preregistered PDF set: "
            + json.dumps(
                {
                    "missing": sorted(
                        expected_figure_names - actual_figure_names
                    ),
                    "unexpected": sorted(
                        actual_figure_names - expected_figure_names
                    ),
                },
                sort_keys=True,
            )
        )
    registered = {
        **analysis_artifacts,
        **production_inputs,
        **figures,
    }
    missing = [
        str(path)
        for path in registered.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "manifest inputs are missing: " + ", ".join(missing)
        )
    artifacts = {
        label: {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for label, path in registered.items()
    }
    payload = {
        "schema": "marisa-b-post-r1-production-manifest-v1",
        "model": "post-recon coevolution finite-PNG v1",
        "tag": TAG,
        "result_status": status,
        "manifest_self_excluded": True,
        "validation_scope": "halo_fiducial_fNL0_only",
        "quarantined_halo_png_catalogue_is_production_input": False,
        "dm_catalogue_is_production_input": False,
        "final_figure_names": list(FINAL_FIGURE_NAMES),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    atomic_json(paths.analysis / "MANIFEST.json", payload)
    return payload


def finalize_independent_reviews(paths: Paths) -> dict[str, Any]:
    """Promote a frozen scientific result only after hashed reviews.

    The expensive theory and profiler stages intentionally finish in a
    pending-review state.  This entry is a narrow, non-recomputing second
    phase: it rejects reviews of any stale source or result artifact and
    requires one distinct reviewer for each preregistered audit scope.
    """

    summary_path = paths.analysis / "summary.json"
    review_path = paths.analysis / "REVIEWS.json"
    if not summary_path.is_file() or not review_path.is_file():
        raise FileNotFoundError(
            "review finalization requires summary.json and REVIEWS.json"
        )
    summary_sha_before = sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema")
        != "marisa-b-post-r1-finite-png-analysis-v1"
        or summary.get("status")
        != "scientific_pass_pending_independent_reviews"
        or summary.get("acceptance", {}).get(
            "all_scientific_gates_pass"
        )
        is not True
    ):
        raise ValueError(
            "only a frozen pending result with every scientific gate "
            "passed can be finalized"
        )
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        reviews.get("schema")
        != "marisa-b-post-r1-independent-reviews-v1"
        or reviews.get("reviewed_summary_sha256")
        != summary_sha_before
    ):
        raise ValueError(
            "REVIEWS.json does not target the current pending summary"
        )

    verification = load_verification(paths)
    if reviews.get("source_hashes") != verification["source_hashes"]:
        raise ValueError(
            "independent reviews do not target the verified sources"
        )

    expected_artifacts = {
        "profiles": paths.analysis / "profiles.npz",
        "gaussian_gate": (
            paths.analysis / "gaussian_gate_diagnostic.json"
        ),
        **{
            f"figure:{name}": paths.figures / name
            for name in FINAL_FIGURE_NAMES
        },
    }
    recorded_artifacts = reviews.get("artifact_hashes")
    if (
        not isinstance(recorded_artifacts, dict)
        or set(recorded_artifacts) != set(expected_artifacts)
    ):
        raise ValueError(
            "REVIEWS.json has an incomplete artifact-hash contract"
        )
    artifact_mismatches = {
        label: {
            "recorded": recorded_artifacts.get(label),
            "actual": (
                sha256(path) if path.is_file() else None
            ),
        }
        for label, path in expected_artifacts.items()
        if (
            not path.is_file()
            or recorded_artifacts.get(label) != sha256(path)
        )
    }
    if artifact_mismatches:
        raise ValueError(
            "reviewed artifacts changed after review: "
            + json.dumps(artifact_mismatches, sort_keys=True)
        )

    required_scopes = {
        "finite_png_algebra",
        "uv_eft_numerics",
        "data_statistics_figures",
    }
    rows = reviews.get("reviews")
    if (
        not isinstance(rows, list)
        or len(rows) != len(required_scopes)
        or {
            row.get("scope")
            for row in rows
            if isinstance(row, dict)
        }
        != required_scopes
    ):
        raise ValueError(
            "REVIEWS.json must contain exactly the three registered "
            "independent scopes"
        )
    reviewers: set[str] = set()
    finalized_status: dict[str, Any] = {}
    analysis_root = paths.analysis.resolve()
    for row in rows:
        reviewer = row.get("reviewer")
        report_value = row.get("report")
        if (
            row.get("status") != "pass"
            or not isinstance(reviewer, str)
            or not reviewer.startswith("/root/")
            or reviewer == "/root"
            or reviewer in reviewers
            or not isinstance(row.get("checks"), list)
            or not row["checks"]
            or not all(
                isinstance(check, str) and check.strip()
                for check in row["checks"]
            )
            or not isinstance(row.get("findings"), list)
            or row.get("all_findings_resolved") is not True
            or not isinstance(report_value, str)
        ):
            raise ValueError(
                f"invalid independent review row for {row.get('scope')}"
            )
        reviewers.add(reviewer)
        report_path = Path(report_value).resolve()
        try:
            report_path.relative_to(analysis_root)
        except ValueError as error:
            raise ValueError(
                "independent review reports must live in the versioned "
                "analysis directory"
            ) from error
        if (
            not report_path.is_file()
            or row.get("report_sha256") != sha256(report_path)
        ):
            raise ValueError(
                f"review report changed for {row['scope']}"
            )
        finalized_status[row["scope"]] = {
            "status": "pass",
            "reviewer": reviewer,
            "report": str(report_path),
            "report_sha256": row["report_sha256"],
            "checks": row["checks"],
            "findings": row["findings"],
            "all_findings_resolved": True,
        }

    summary["status"] = "pass"
    summary["review_status"] = finalized_status
    summary["independent_reviews"] = {
        "record": str(review_path.resolve()),
        "record_sha256": sha256(review_path),
        "reviewed_pending_summary_sha256": summary_sha_before,
        "artifact_hashes": recorded_artifacts,
        "source_hashes": reviews["source_hashes"],
    }
    atomic_json(summary_path, summary)
    write_report(paths.analysis / "REPORT.md", summary)
    write_manifest(paths, summary)
    return summary


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            to_jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    # bool is an int subclass in Python, so this branch must precede the
    # integer conversion or JSON schema booleans silently become 0/1.
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


def is_plain_json_int(value: Any) -> bool:
    """Return whether a decoded provenance value is an actual JSON integer.

    ``bool`` is an ``int`` subclass and Python numeric equality also accepts
    values such as ``1.0 == 1``.  Numerical provenance must not silently
    coerce either case, nor strings or fractional values, into a registered
    integration setting.
    """

    return type(value) is int


def is_plain_json_int_list(
    value: Any,
    *,
    length: int | None = None,
) -> bool:
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(is_plain_json_int(item) for item in value)
    )


def is_plain_json_number(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
    )


def deep_change_paths(
    left: Any,
    right: Any,
    *,
    ignored_paths: frozenset[str] = frozenset(),
    prefix: str = "",
) -> tuple[str, ...]:
    """Return fail-closed dotted paths whose JSON-compatible values differ."""
    if prefix in ignored_paths:
        return ()
    if isinstance(left, dict) and isinstance(right, dict):
        changed: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if path in ignored_paths:
                continue
            if key not in left or key not in right:
                changed.append(path)
                continue
            changed.extend(
                deep_change_paths(
                    left[key],
                    right[key],
                    ignored_paths=ignored_paths,
                    prefix=path,
                )
            )
        return tuple(changed)
    # Lists are atomic settings.  This makes a quadrature/order vector one
    # auditable registered change instead of silently accepting selected
    # element changes.
    if left != right:
        return (prefix or "<root>",)
    return ()


def build_paths(args: argparse.Namespace) -> Paths:
    repo = args.repo_root.resolve()
    configured_data_root = (
        args.data_root
        if args.data_root is not None
        else Path(os.environ.get("MARISA_B_DATA_ROOT", repo))
    )
    data_root = configured_data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )
    worktree = repo
    templates = args.templates.resolve()
    analysis = output_root / "analysis" / TAG
    png_cache = (
        args.png_cache.resolve()
        if args.png_cache is not None
        else analysis / "finite_png_templates.npz"
    )
    return Paths(
        repo=repo,
        worktree=worktree,
        data_root=data_root,
        output_root=output_root,
        templates=templates,
        matrix=(
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid500_b000_pk_matrix.npz"
        ),
        quarantined_halo_png_matrix=(
            data_root
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid_lcp_lcm_first100_b000_pk_matrix.npz"
        ),
        png_cache=png_cache,
        contract=repo / "configs/eft_v2_contract.json",
        pre_runner=repo / "scripts/production/run_pre_recon_bias_model_v3.py",
        frozen_fitter=repo / "scripts/legacy/fit_eft_v2_r0_mean.py",
        analysis=analysis,
        figures=output_root / "figures/diagnostics" / TAG,
    )


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("record") != "header":
        raise ValueError(f"{path} lacks a unique header")
    bins = rows[1:]
    if any(row.get("record") != "bin" for row in bins):
        raise ValueError(f"{path} contains a non-bin payload")
    indices = [row.get("index") for row in bins]
    if (
        any(not is_plain_json_int(index) for index in indices)
        or any(
            index < 0 or index >= 120
            for index in indices
            if is_plain_json_int(index)
        )
        or len(indices) != len(set(indices))
    ):
        raise ValueError(f"{path} has invalid or duplicate bin indices")
    return rows[0], bins


def _full_vector(
    bins: list[dict[str, Any]],
    extractor: Any,
) -> np.ndarray:
    result = np.zeros(120, dtype=np.float64)
    for row in bins:
        result[int(row["index"])] = float(extractor(row))
    if not np.all(np.isfinite(result)):
        raise ValueError("PNG JSONL produced a non-finite vector")
    return result


def _validate_png_header(
    path: Path,
    header: dict[str, Any],
    *,
    allowed_qmax: tuple[float, ...] = (
        PNG_TAIL_REFERENCE_QMAX,
    ),
) -> None:
    sector = str(header.get("sector"))
    schema = header.get("schema")
    if (
        schema != PNG_JSONL_SCHEMA
        and not (
            sector in {
                "tree-fixed",
                "matter-linear",
                "matter-b112",
            }
            and schema in PNG_JSONL_LEGACY_SCHEMAS
        )
    ):
        raise ValueError(f"{path} has the wrong PNG schema")
    projection = header.get("shell_projection")
    reconstruction = header.get("reconstruction")
    fixed = header.get("fixed_poisson_png")
    integration = header.get("integration")
    orientation_orders = (
        projection.get("orientation_orders")
        if isinstance(projection, dict)
        else None
    )
    projection_orders_are_integers = (
        isinstance(projection, dict)
        and is_plain_json_int(projection.get("radial_order"))
        and is_plain_json_int(projection.get("angular_order"))
        and is_plain_json_int_list(orientation_orders, length=3)
    )
    projection_signature = (
        (
            projection["radial_order"],
            projection["angular_order"],
            tuple(orientation_orders),
        )
        if projection_orders_are_integers
        else None
    )
    if (
        not isinstance(projection, dict)
        or not is_plain_json_number(header.get("lambda"))
        or projection.get("measure")
        != (
            "exact float32 FFT-lattice k1-k2-mu plus "
            "normalized Haar orientation cubature"
        )
        or not is_plain_json_number(projection.get("fft_box_size"))
        or not math.isclose(
            float(projection.get("fft_box_size")),
            1000.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not is_plain_json_int(projection.get("fft_mesh_size"))
        or projection.get("fft_mesh_size") != 256
        or projection_signature
        not in {
            (2, 8, (2, 2, 2)),
            (3, 12, (3, 3, 3)),
        }
        or not isinstance(reconstruction, dict)
        or not all(
            is_plain_json_number(reconstruction.get(key))
            for key in ("R", "b_rec_h", "cell_size")
        )
        or not math.isclose(
            float(reconstruction.get("R")),
            15.0,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not math.isclose(
            float(reconstruction.get("b_rec_h")),
            B_REC_H,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or not math.isclose(
            float(reconstruction.get("cell_size")),
            8.0,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or not is_plain_json_int(reconstruction.get("cic_power"))
        or reconstruction.get("cic_power") != 4
        or not isinstance(fixed, dict)
        or fixed.get("intensity")
        != "conditional_1_plus_delta_h"
        or fixed.get("external_estimator_filter")
        != "at_most_one_density_mark_per_cumulant"
        or fixed.get("finite_K1") != "b1+fNL*bphi/M(k)"
        or fixed.get("bare_PL3") is not False
        or header.get("lambda_definition")
        != "b1_over_b_rec_h_for_matter_uplift"
        or not is_plain_json_int(header.get("radial_stop"))
        or header.get("radial_stop") != 7
        or not isinstance(integration, dict)
    ):
        raise ValueError(f"{path} violates the post-R1 PNG contract")
    if (
        schema == PNG_JSONL_SCHEMA
        and (
            integration.get("png_ir_cutoff_scope")
            != "all_local_primordial_B0_T0_legs"
            or integration.get("matter_linear_multicenter_qmc")
            is not True
        )
    ):
        raise ValueError(
            f"{path} lacks the v5 finite-box/QMC integration scope"
        )
    registered_eps = (0.01, 0.02, 0.03)
    registered_qmin = (1.0e-4, 2.0 * math.pi / 1000.0)
    if (
        not all(
            is_plain_json_number(integration.get(key))
            for key in (
                "epsrel",
                "p13_epsrel",
                "qmin",
                "qmax",
                "png_ir_cutoff",
            )
        )
        or
        not any(
            math.isclose(
                float(integration.get("epsrel", math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            for value in registered_eps
        )
        or not any(
            math.isclose(
                float(integration.get("p13_epsrel", math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            for value in registered_eps
        )
        or not any(
            math.isclose(
                float(integration.get("qmin", math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            for value in registered_qmin
        )
        or not any(
            math.isclose(
                float(integration.get("qmax", math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-13,
            )
            for value in allowed_qmax
        )
        or not math.isclose(
            float(integration.get("png_ir_cutoff", math.nan)),
            2.0 * math.pi / 1000.0,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not is_plain_json_int(
            integration.get("b112ii_qmc_power")
        )
        or integration.get("b112ii_qmc_power")
        not in {11, 12, 13}
        or not is_plain_json_int(
            integration.get("b112ii_qmc_replicates")
        )
        or integration.get("b112ii_qmc_replicates")
        not in {4, 8}
    ):
        raise ValueError(
            f"{path} has unregistered PNG integration settings"
        )
    contract = header.get("input_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("linear_power") != "POWER_TABLE_column_1"
        or contract.get("transfer_M") != "PNG_TABLE_column_3"
    ):
        raise ValueError(f"{path} conflates the P_L and PNG tables")
    if schema == PNG_JSONL_SCHEMA:
        source_hashes = header.get("source_hashes")
        if (
            not isinstance(source_hashes, dict)
            or set(source_hashes)
            != {
                "linear_power",
                "png_table",
                "edge_file",
                "driver_executable",
                "parameter_registry",
            }
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in source_hashes.values()
            )
        ):
            raise ValueError(
                f"{path} lacks complete v5 source hashes"
            )
    if sector == "tree-fixed":
        residual = header.get("residual_stochastic_png")
        if (
            schema
            not in {
                PNG_JSONL_SCHEMA,
                "marisa-b-post-r1-finite-png-jsonl-v2",
            }
            or not isinstance(residual, dict)
            or residual.get("shape_convention")
            != (
                "shared_G_L_Q_for_independent_"
                "alpha3_and_alpha3PNG"
            )
            or residual.get("linear_shape_extraction")
            != "H_plus_minus_H_minus_over_4"
            or residual.get("supports_eq2p65_two_amplitude")
            is not True
            or residual.get("leading_noisy_shift") is not False
        ):
            raise ValueError(
                f"{path} lacks v2 Eq. (2.65) stochastic provenance"
            )


def _validate_png_shard_rows(
    path: Path,
    header: dict[str, Any],
    bins: list[dict[str, Any]],
) -> None:
    """Bind every PNG row to its header and projector provenance."""

    indices = [row["index"] for row in bins]
    recorded_indices = header.get("bin_indices")
    if (
        recorded_indices is not None
        and (
            not is_plain_json_int_list(recorded_indices)
            or recorded_indices != indices
        )
    ):
        raise ValueError(
            f"{path} has mismatched merged-header bin indices"
        )
    recorded_range = header.get("bin_range")
    if (
        not is_plain_json_int_list(recorded_range, length=2)
        or not indices
        or not (
            0
            <= recorded_range[0]
            < recorded_range[1]
            <= 120
        )
        or recorded_range != [min(indices), max(indices) + 1]
        or (
            recorded_indices is None
            and indices
            != list(range(recorded_range[0], recorded_range[1]))
        )
    ):
        raise ValueError(
            f"{path} rows do not close against its header bin range"
        )

    projection = header["shell_projection"]
    expected_invariant_nodes = (
        projection["radial_order"] ** 2
        * projection["angular_order"]
    )
    expected_orientation_nodes = math.prod(
        projection["orientation_orders"]
    )
    expected_shell_nodes = (
        expected_invariant_nodes * expected_orientation_nodes
    )
    sector = str(header.get("sector"))

    def finite_numeric_mapping(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and all(
                isinstance(name, str)
                and name
                and is_plain_json_number(coefficient)
                and math.isfinite(float(coefficient))
                for name, coefficient in value.items()
            )
        )

    for row in bins:
        integer_metadata = {
            name: row.get(name)
            for name in (
                "shell_nodes",
                "invariant_shell_nodes",
                "orientation_nodes",
                "zero_external_leg_pairs",
                "closing_zero_pairs",
            )
        }
        edges = row.get("edges")
        valid_fraction = row.get("valid_pair_fraction")
        if (
            not all(
                is_plain_json_int(value)
                for value in integer_metadata.values()
            )
            or integer_metadata["shell_nodes"]
            != expected_shell_nodes
            or integer_metadata["invariant_shell_nodes"]
            != expected_invariant_nodes
            or integer_metadata["orientation_nodes"]
            != expected_orientation_nodes
            or integer_metadata["zero_external_leg_pairs"] < 0
            or integer_metadata["closing_zero_pairs"] < 0
            or not isinstance(edges, list)
            or len(edges) != 4
            or not all(is_plain_json_number(value) for value in edges)
            or not (
                0.0 <= float(edges[0]) < float(edges[1])
                and 0.0 <= float(edges[2]) < float(edges[3])
            )
            or not is_plain_json_number(valid_fraction)
            or not (0.0 < float(valid_fraction) <= 1.0)
        ):
            raise ValueError(
                f"{path} bin {row['index']} has invalid row geometry"
            )
        if sector == "tree-fixed":
            halo_tree = row.get("halo_tree")
            gate = row.get("residual_stochastic_png_gate")
            fixed = row.get("fixed_poisson_png")
            residual = row.get("residual_stochastic_png")
            gate_numeric_fields = (
                "native_max_abs",
                "native_max_rel",
                "tied_minus_density_max_abs",
                "tied_minus_density_max_rel",
            )
            if (
                not isinstance(halo_tree, dict)
                or any(
                    not is_plain_json_number(halo_tree.get(field))
                    or not math.isfinite(float(halo_tree[field]))
                    for field in PNG_TREE_FIELDS
                )
                or not isinstance(gate, dict)
                or gate.get("supports_eq2p65_two_amplitude") is not True
                or gate.get(
                    "legacy_single_amplitude_G_plus_fL_plus_f2Q_is_eq2p65"
                )
                is not False
                or gate.get("noisy_shift_exactly_zero") is not True
                or any(
                    not is_plain_json_number(gate.get(field))
                    or not math.isfinite(float(gate[field]))
                    or float(gate[field]) < 0.0
                    for field in gate_numeric_fields
                )
                or not isinstance(fixed, dict)
                or any(
                    not finite_numeric_mapping(fixed.get(field))
                    for field in (
                        "gaussian",
                        "linear_unit_bphi",
                        "quadratic_unit_bphi2",
                    )
                )
                or not isinstance(residual, dict)
            ):
                raise ValueError(
                    f"{path} bin {row['index']} has an incomplete "
                    "tree-fixed payload"
                )
            if (
                (
                    float(gate["native_max_abs"]) > 2.0e-9
                    and float(gate["native_max_rel"]) > 2.0e-12
                )
                or (
                    float(gate["tied_minus_density_max_abs"])
                    > 2.0e-9
                    and float(gate["tied_minus_density_max_rel"])
                    > 2.0e-12
                )
            ):
                raise ValueError(
                    f"{path} bin {row['index']} fails the native "
                    "tree-fixed stochastic gate"
                )
            residual_maps = {}
            for map_name in ("density_only", "tied", "noisy_shift"):
                raw_map = residual.get(map_name)
                if not isinstance(raw_map, dict) or any(
                    not finite_numeric_mapping(raw_map.get(field))
                    for field in (
                        "gaussian",
                        "linear_unit_bphi",
                        "quadratic_unit_bphi2",
                    )
                ):
                    raise ValueError(
                        f"{path} bin {row['index']} has an incomplete "
                        f"residual stochastic map: {map_name}"
                    )
                residual_maps[map_name] = raw_map
            for field in (
                "gaussian",
                "linear_unit_bphi",
                "quadratic_unit_bphi2",
            ):
                names = (
                    set(residual_maps["density_only"][field])
                    | set(residual_maps["tied"][field])
                    | set(residual_maps["noisy_shift"][field])
                )
                for name in names:
                    error = abs(
                        residual_maps["tied"][field].get(name, 0.0)
                        - residual_maps["density_only"][field].get(
                            name,
                            0.0,
                        )
                        - residual_maps["noisy_shift"][field].get(
                            name,
                            0.0,
                        )
                    )
                    if error > 2.0e-8:
                        raise ValueError(
                            f"{path} bin {row['index']} fails residual "
                            f"stochastic decomposition for {field}/{name}"
                        )
        elif sector == "matter-linear":
            payload = row.get("matter_linear")
            numerical = (
                payload.get("numerical_error_estimate")
                if isinstance(payload, dict)
                else None
            )
            if (
                not isinstance(payload, dict)
                or not is_plain_json_int(payload.get("neval"))
                or payload.get("neval") <= 0
                or any(
                    not is_plain_json_number(payload.get(field))
                    or not math.isfinite(float(payload[field]))
                    for field in MATTER_LINEAR_FIELDS
                )
                or not isinstance(numerical, dict)
                or any(
                    not is_plain_json_number(numerical.get(field))
                    or not math.isfinite(float(numerical[field]))
                    or float(numerical[field]) < 0.0
                    for field in MATTER_LINEAR_NUMERICAL_FIELDS
                )
            ):
                raise ValueError(
                    f"{path} bin {row['index']} has invalid "
                    "matter-linear numerics"
                )
            diagram_sum = sum(
                float(payload[field])
                for field in MATTER_LINEAR_DIAGRAM_FIELDS
            )
            diagram_error_sum = sum(
                float(numerical[field])
                for field in MATTER_LINEAR_DIAGRAM_FIELDS
            )
            if (
                not math.isclose(
                    float(payload["loop"]),
                    diagram_sum,
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-10,
                )
                or not math.isclose(
                    float(payload["total"]),
                    float(payload["tree"]) + float(payload["loop"]),
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-10,
                )
                or not math.isclose(
                    float(numerical["loop"]),
                    diagram_error_sum,
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-12,
                )
                or not math.isclose(
                    float(numerical["total"]),
                    float(numerical["loop"]),
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-12,
                )
            ):
                raise ValueError(
                    f"{path} bin {row['index']} fails the matter-linear "
                    "component closure"
                )
        elif sector == "matter-b112":
            payload = row.get("matter_b112")
            if (
                not isinstance(payload, dict)
                or not is_plain_json_int(payload.get("neval"))
                or payload.get("neval") <= 0
                or not is_plain_json_number(payload.get("coefficient"))
                or not math.isfinite(float(payload["coefficient"]))
                or not is_plain_json_number(
                    payload.get("weighted_error_bound")
                )
                or not math.isfinite(
                    float(payload["weighted_error_bound"])
                )
                or float(payload["weighted_error_bound"]) < 0.0
            ):
                raise ValueError(
                    f"{path} bin {row['index']} has invalid "
                    "matter-b112 numerics"
                )


def compile_png_cache(
    paths: Iterable[Path],
    output: Path,
) -> dict[str, Any]:
    records: list[
        tuple[Path, dict[str, Any], list[dict[str, Any]]]
    ] = []
    for raw_path in paths:
        path = raw_path.resolve()
        header, bins = read_jsonl(path)
        _validate_png_header(path, header)
        _validate_png_shard_rows(path, header, bins)
        records.append((path, header, bins))
    if not records:
        raise ValueError("PNG cache compilation received no inputs")
    common_header_fields = (
        "input_contract",
        "source_hashes",
        "lambda_definition",
        "shell_projection",
        "reconstruction",
        "integration",
        "fixed_poisson_png",
        "radial_stop",
    )
    reference_common = {
        key: records[0][1].get(key)
        for key in common_header_fields
    }
    for path, header, _bins in records[1:]:
        for key in common_header_fields:
            if header.get(key) != reference_common[key]:
                raise ValueError(
                    "PNG sectors/lambdas do not share one registered "
                    f"{key} contract: {path}"
                )
    grouped: dict[
        tuple[str, float],
        list[tuple[Path, dict[str, Any], list[dict[str, Any]]]],
    ] = {}
    for record in records:
        key = (
            str(record[1].get("sector")),
            float(record[1].get("lambda")),
        )
        grouped.setdefault(key, []).append(record)
    if {key[0] for key in grouped} != {
        "tree-fixed",
        "matter-linear",
        "matter-b112",
    }:
        raise ValueError(
            "PNG cache requires tree-fixed, matter-linear, and "
            "matter-b112 sectors"
        )
    merged_records: list[
        tuple[
            tuple[Path, ...],
            dict[str, Any],
            list[dict[str, Any]],
        ]
    ] = []
    for (_sector, _lambda), parts in grouped.items():
        merged_bins: dict[int, dict[str, Any]] = {}
        reference_header = parts[0][1]
        for path, part_header, bins in parts:
            for key in (
                "schema",
                "sector",
                "lambda",
                "input_contract",
                "lambda_definition",
                "shell_projection",
                "reconstruction",
                "integration",
                "fixed_poisson_png",
                "radial_stop",
            ):
                if part_header.get(key) != reference_header.get(key):
                    raise ValueError(
                        f"inconsistent {key} across PNG shards "
                        f"for {_sector}, lambda={_lambda}: {path}"
                    )
            for row in bins:
                index = int(row["index"])
                if index in merged_bins:
                    raise ValueError(
                        f"duplicate bin {index} across PNG shards "
                        f"for {_sector}, lambda={_lambda}: {path}"
                    )
                merged_bins[index] = row
        merged_records.append(
            (
                tuple(part[0] for part in parts),
                parts[0][1],
                [merged_bins[index] for index in sorted(merged_bins)],
            )
        )
    tree_records = [
        record
        for record in merged_records
        if record[1]["sector"] == "tree-fixed"
    ]
    if len(tree_records) != 1:
        raise ValueError(
            "PNG cache requires exactly one tree-fixed lambda group"
        )
    if not math.isclose(
        float(tree_records[0][1]["lambda"]),
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("tree-fixed PNG group must use lambda=0")

    reference_indices: tuple[int, ...] | None = None
    reference_edges: dict[int, tuple[float, ...]] = {}
    for source_paths, _header, bins in merged_records:
        indices = tuple(sorted(int(row["index"]) for row in bins))
        if reference_indices is None:
            reference_indices = indices
            reference_edges = {
                int(row["index"]): tuple(
                    float(value) for value in row["edges"]
                )
                for row in bins
            }
        if indices != reference_indices:
            raise ValueError(
                f"{source_paths} have a mismatched PNG bin set"
            )
        for row in bins:
            index = int(row["index"])
            if not np.array_equal(
                np.asarray(row["edges"], dtype=np.float64),
                np.asarray(
                    reference_edges[index],
                    dtype=np.float64,
                ),
            ):
                raise ValueError(
                    f"{source_paths} have mismatched bin edges"
                )
    assert reference_indices is not None
    compiled_edges = np.vstack(
        [
            np.asarray(reference_edges[index], dtype=np.float64)
            for index in reference_indices
        ]
    )
    selected_edge_bytes = np.ascontiguousarray(
        compiled_edges
    ).tobytes()

    _tree_paths, tree_header, tree_bins = tree_records[0]
    tree = {
        field: _full_vector(
            tree_bins,
            lambda row, field=field: row["halo_tree"][field],
        )
        for field in PNG_TREE_FIELDS
    }
    residual_gate_rows = [
        row.get("residual_stochastic_png_gate", {})
        for row in tree_bins
    ]
    residual_gate_numeric_fields = (
        "native_max_abs",
        "native_max_rel",
        "tied_minus_density_max_abs",
        "tied_minus_density_max_rel",
    )
    if any(
        not gate.get("supports_eq2p65_two_amplitude", False)
        or gate.get(
            "legacy_single_amplitude_G_plus_fL_plus_f2Q_is_eq2p65",
            True,
        )
        or not gate.get("noisy_shift_exactly_zero", False)
        or any(
            not is_plain_json_number(gate.get(field))
            or not math.isfinite(float(gate.get(field)))
            or float(gate.get(field, math.nan)) < 0.0
            for field in residual_gate_numeric_fields
        )
        for gate in residual_gate_rows
    ):
        raise ValueError(
            "tree-fixed cache lacks the strict Eq. (2.65) "
            "two-amplitude stochastic provenance gate"
        )
    residual_native_gate = {
        "maximum_absolute_error": float(
            max(
                float(gate["native_max_abs"])
                for gate in residual_gate_rows
            )
        ),
        "maximum_relative_error": float(
            max(
                float(gate["native_max_rel"])
                for gate in residual_gate_rows
            )
        ),
        "tied_minus_density_maximum_absolute_error": float(
            max(
                float(gate["tied_minus_density_max_abs"])
                for gate in residual_gate_rows
            )
        ),
        "tied_minus_density_maximum_relative_error": float(
            max(
                float(gate["tied_minus_density_max_rel"])
                for gate in residual_gate_rows
            )
        ),
    }
    if (
        residual_native_gate["maximum_absolute_error"] > 2.0e-9
        and residual_native_gate["maximum_relative_error"] > 2.0e-12
    ) or (
        residual_native_gate[
            "tied_minus_density_maximum_absolute_error"
        ] > 2.0e-9
        and residual_native_gate[
            "tied_minus_density_maximum_relative_error"
        ] > 2.0e-12
    ):
        raise ValueError(
            "tree-fixed residual stochastic native/map gate fails"
        )
    fixed_linear_names = sorted(
        {
            name
            for row in tree_bins
            for name in row["fixed_poisson_png"][
                "linear_unit_bphi"
            ]
        }
    )
    fixed_gaussian_names = sorted(
        {
            name
            for row in tree_bins
            for name in row["fixed_poisson_png"]["gaussian"]
        }
    )
    fixed_quadratic_names = sorted(
        {
            name
            for row in tree_bins
            for name in row["fixed_poisson_png"][
                "quadratic_unit_bphi2"
            ]
        }
    )
    fixed_gaussian = np.vstack(
        [
            _full_vector(
                tree_bins,
                lambda row, name=name: row["fixed_poisson_png"][
                    "gaussian"
                ].get(name, 0.0),
            )
            for name in fixed_gaussian_names
        ]
    )
    fixed_linear = np.vstack(
        [
            _full_vector(
                tree_bins,
                lambda row, name=name: row["fixed_poisson_png"][
                    "linear_unit_bphi"
                ].get(name, 0.0),
            )
            for name in fixed_linear_names
        ]
    )
    fixed_quadratic = np.vstack(
        [
            _full_vector(
                tree_bins,
                lambda row, name=name: row["fixed_poisson_png"][
                    "quadratic_unit_bphi2"
                ].get(name, 0.0),
            )
            for name in fixed_quadratic_names
        ]
    )
    residual_maps = ("density_only", "tied", "noisy_shift")
    residual_fields = (
        ("gaussian", "residual_gaussian"),
        ("linear_unit_bphi", "residual_linear"),
        ("quadratic_unit_bphi2", "residual_quadratic"),
    )
    residual_compiled: dict[
        str, dict[str, tuple[list[str], np.ndarray]]
    ] = {}
    for json_field, cache_field in residual_fields:
        residual_compiled[cache_field] = {}
        for map_name in residual_maps:
            names = sorted(
                {
                    name
                    for row in tree_bins
                    for name in row["residual_stochastic_png"][
                        map_name
                    ][json_field]
                }
            )
            coefficients = (
                np.vstack(
                    [
                        _full_vector(
                            tree_bins,
                            lambda row, name=name: row[
                                "residual_stochastic_png"
                            ][map_name][json_field].get(
                                name,
                                0.0,
                            ),
                        )
                        for name in names
                    ]
                )
                if names
                else np.zeros((0, 120), dtype=np.float64)
            )
            residual_compiled[cache_field][map_name] = (
                names,
                coefficients,
            )
    residual_decomposition_errors: dict[str, float] = {}
    for _json_field, cache_field in residual_fields:
        maps = residual_compiled[cache_field]
        monomials = sorted(
            set(maps["density_only"][0])
            | set(maps["tied"][0])
            | set(maps["noisy_shift"][0])
        )
        maximum_error = 0.0
        for monomial in monomials:
            terms = {}
            for map_name in residual_maps:
                names, coefficients = maps[map_name]
                terms[map_name] = (
                    coefficients[names.index(monomial)]
                    if monomial in names
                    else np.zeros(120, dtype=np.float64)
                )
            maximum_error = max(
                maximum_error,
                float(
                    np.max(
                        np.abs(
                            terms["tied"]
                            - terms["density_only"]
                            - terms["noisy_shift"]
                        )
                    )
                ),
            )
        residual_decomposition_errors[cache_field] = maximum_error
    if max(residual_decomposition_errors.values()) > 2.0e-8:
        raise ValueError(
            "finite-PNG residual stochastic map decomposition fails"
        )

    def values_by_lambda(
        sector: str,
        extractor: Any,
    ) -> dict[float, np.ndarray]:
        result: dict[float, np.ndarray] = {}
        for source_paths, header, bins in merged_records:
            if header["sector"] != sector:
                continue
            value = float(header["lambda"])
            if value in result:
                raise ValueError(
                    f"duplicate merged {sector} lambda={value}: "
                    f"{source_paths}"
                )
            result[value] = extractor(bins)
        return result

    linear_by_field: dict[str, dict[float, np.ndarray]] = {}
    for field in MATTER_LINEAR_FIELDS:
        linear_by_field[field] = values_by_lambda(
            "matter-linear",
            lambda bins, field=field: _full_vector(
                bins,
                lambda row: row["matter_linear"][field],
            ),
        )
    linear_error_by_field: dict[str, dict[float, np.ndarray]] = {}
    for field in MATTER_LINEAR_NUMERICAL_FIELDS:
        linear_error_by_field[field] = values_by_lambda(
            "matter-linear",
            lambda bins, field=field: _full_vector(
                bins,
                lambda row: row["matter_linear"][
                    "numerical_error_estimate"
                ][field],
            ),
        )
    linear_neval_by_lambda = values_by_lambda(
        "matter-linear",
        lambda bins: _full_vector(
            bins,
            lambda row: row["matter_linear"]["neval"],
        ),
    )
    if (
        any(
            set(by_lambda) != set(linear_by_field["loop"])
            for by_lambda in linear_error_by_field.values()
        )
        or set(linear_neval_by_lambda)
        != set(linear_by_field["loop"])
        or any(
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            for by_lambda in linear_error_by_field.values()
            for values in by_lambda.values()
        )
        or any(
            not np.all(np.isfinite(values))
            or np.any(
                values[
                    np.asarray(reference_indices, dtype=np.int64)
                ]
                <= 0.0
            )
            for values in linear_neval_by_lambda.values()
        )
    ):
        raise ValueError(
            "matter-linear QMC diagnostics must be complete, finite, "
            "and nonnegative with positive evaluation counts"
        )
    # This pure-matter PNG response contains B122I/B122II/B113I/B113II,
    # not the Gaussian B222/B321/B411 topologies.  Counting the reconstructed
    # K2 and K3 factors therefore gives an exact degree <=2 dependence on
    # the shift amplitude lambda.  Extra lambda nodes remain independent
    # integration-error holdouts and must never be fit as a cubic.
    canonical_linear = (0.0, 1.0, 2.0)
    if any(
        value not in linear_by_field["loop"]
        for value in canonical_linear
    ):
        raise ValueError(
            "matter-linear cache lacks quadratic nodes lambda=0,1,2"
        )
    linear_vandermonde = np.vander(
        np.asarray(canonical_linear),
        N=3,
        increasing=True,
    )
    matter_linear_coefficients: dict[str, np.ndarray] = {}
    linear_validation: dict[str, dict[str, float]] = {}
    for field, by_lambda in linear_by_field.items():
        sampled = np.vstack([by_lambda[value] for value in canonical_linear])
        coefficients = np.linalg.solve(linear_vandermonde, sampled)
        matter_linear_coefficients[field] = coefficients
        validation: dict[str, float] = {}
        for value, measured in by_lambda.items():
            predicted = PngTemplateSet._evaluate_polynomial(
                coefficients,
                value,
            )
            validation[f"{value:.17g}"] = float(
                np.max(np.abs(predicted - measured))
            )
        linear_validation[field] = validation
    extra_linear_lambdas = np.asarray(
        sorted(
            set(linear_by_field["loop"]) - set(canonical_linear)
        ),
        dtype=np.float64,
    )
    matter_linear_loop_validation_residuals = np.vstack(
        [
            PngTemplateSet._evaluate_polynomial(
                matter_linear_coefficients["loop"],
                float(value),
            )
            - linear_by_field["loop"][float(value)]
            for value in extra_linear_lambdas
        ]
    ) if extra_linear_lambdas.size else np.zeros(
        (0, 120),
        dtype=np.float64,
    )
    linear_error_lambdas = np.asarray(
        sorted(linear_error_by_field["total"]),
        dtype=np.float64,
    )
    matter_linear_error_bounds = {
        field: np.vstack(
            [
                linear_error_by_field[field][float(value)]
                for value in linear_error_lambdas
            ]
        )
        for field in MATTER_LINEAR_NUMERICAL_FIELDS
    }

    b112_by_lambda = values_by_lambda(
        "matter-b112",
        lambda bins: _full_vector(
            bins,
            lambda row: row["matter_b112"]["coefficient"],
        ),
    )
    b112_error_by_lambda = values_by_lambda(
        "matter-b112",
        lambda bins: _full_vector(
            bins,
            lambda row: row["matter_b112"][
                "weighted_error_bound"
            ],
        ),
    )
    canonical_b112 = (0.0, 1.0)
    if any(value not in b112_by_lambda for value in canonical_b112):
        raise ValueError("matter-b112 cache lacks lambda=0,1")
    b112_vandermonde = np.vander(
        np.asarray(canonical_b112),
        N=2,
        increasing=True,
    )
    matter_b112_coefficients = np.linalg.solve(
        b112_vandermonde,
        np.vstack([b112_by_lambda[value] for value in canonical_b112]),
    )
    b112_validation = {}
    for value, measured in b112_by_lambda.items():
        predicted = PngTemplateSet._evaluate_polynomial(
            matter_b112_coefficients,
            value,
        )
        b112_validation[f"{value:.17g}"] = float(
            np.max(np.abs(predicted - measured))
        )
    extra_b112_lambdas = np.asarray(
        sorted(set(b112_by_lambda) - set(canonical_b112)),
        dtype=np.float64,
    )
    matter_b112_validation_residuals = np.vstack(
        [
            PngTemplateSet._evaluate_polynomial(
                matter_b112_coefficients,
                float(value),
            )
            - b112_by_lambda[float(value)]
            for value in extra_b112_lambdas
        ]
    ) if extra_b112_lambdas.size else np.zeros(
        (0, 120),
        dtype=np.float64,
    )
    b112_error_lambdas = np.asarray(
        sorted(b112_error_by_lambda),
        dtype=np.float64,
    )
    b112_error_bounds = np.vstack(
        [
            b112_error_by_lambda[float(value)]
            for value in b112_error_lambdas
        ]
    )
    if (
        not np.all(np.isfinite(b112_error_bounds))
        or np.any(b112_error_bounds < 0.0)
    ):
        raise ValueError(
            "matter-b112 weighted QMC bounds must be finite "
            "and nonnegative"
        )

    metadata = {
        "schema": PNG_CACHE_SCHEMA,
        "all_raw_inputs_current_v5": all(
            header.get("schema") == PNG_JSONL_SCHEMA
            for _path, header, _bins in records
        ),
        "common_raw_contract": reference_common,
        "source_files": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "sector": header["sector"],
                "lambda": float(header["lambda"]),
            }
            for path, header, _bins in records
        ],
        "indices": list(reference_indices),
        "row_geometry": {
            "cache_edges_sha256": hashlib.sha256(
                selected_edge_bytes
            ).hexdigest(),
            "rows_bound_to_header_ranges": True,
            "projector_nodes_bound_to_header": True,
        },
        "tree_header": tree_header,
        "linear_interpolation": {
            "degree": 2,
            "canonical_nodes": list(canonical_linear),
            "max_abs_residual_by_field_and_lambda": linear_validation,
            "weighted_qmc_error_estimate_max_by_field_and_lambda": {
                field: {
                    f"{value:.17g}": float(
                        np.max(
                            by_lambda[value][
                                np.asarray(
                                    reference_indices,
                                    dtype=int,
                                )
                            ]
                        )
                    )
                    for value in sorted(by_lambda)
                }
                for field, by_lambda
                in linear_error_by_field.items()
            },
            "minimum_neval_by_lambda": {
                f"{value:.17g}": int(
                    np.min(
                        vector[
                            np.asarray(
                                reference_indices,
                                dtype=int,
                            )
                        ]
                    )
                )
                for value, vector
                in linear_neval_by_lambda.items()
            },
        },
        "b112_interpolation": {
            "degree": 1,
            "canonical_nodes": list(canonical_b112),
            "max_abs_residual_by_lambda": b112_validation,
            "weighted_qmc_error_bound_max_by_lambda": {
                f"{value:.17g}": float(
                    np.max(
                        error[
                            np.asarray(
                                reference_indices,
                                dtype=int,
                            )
                        ]
                    )
                )
                for value, error in b112_error_by_lambda.items()
            },
        },
        "residual_stochastic_map_decomposition_max_abs": (
            residual_decomposition_errors
        ),
        "residual_stochastic_native_gate": residual_native_gate,
        "png_stochastic_primary": (
            "independent alpha3 and alpha3PNG amplitudes following "
            "Moradinezhad Dizgah et al. Eq. (2.65)"
        ),
        "png_stochastic_legacy_note": (
            "G+fL+f^2Q is retained only as a historical robustness "
            "branch and is not an Eq. (2.65) closure"
        ),
    }
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True),
            dtype=np.str_,
        ),
        "indices": np.asarray(reference_indices, dtype=np.int64),
        "edges": compiled_edges,
        "fixed_gaussian_names": np.asarray(
            fixed_gaussian_names,
            dtype=np.str_,
        ),
        "fixed_gaussian_coefficients": fixed_gaussian,
        "fixed_linear_names": np.asarray(
            fixed_linear_names,
            dtype=np.str_,
        ),
        "fixed_linear_coefficients": fixed_linear,
        "fixed_quadratic_names": np.asarray(
            fixed_quadratic_names,
            dtype=np.str_,
        ),
        "fixed_quadratic_coefficients": fixed_quadratic,
        "matter_b112_coefficients": matter_b112_coefficients,
        "matter_linear_validation_lambdas": (
            extra_linear_lambdas
        ),
        "matter_linear_loop_validation_residuals": (
            matter_linear_loop_validation_residuals
        ),
        "matter_linear_error_lambdas": (
            linear_error_lambdas
        ),
        "matter_b112_validation_lambdas": (
            extra_b112_lambdas
        ),
        "matter_b112_validation_residuals": (
            matter_b112_validation_residuals
        ),
        "matter_b112_error_lambdas": b112_error_lambdas,
        "matter_b112_error_bounds": b112_error_bounds,
    }
    for cache_field, maps in residual_compiled.items():
        for map_name, (names, coefficients) in maps.items():
            arrays[
                f"{cache_field}__{map_name}__names"
            ] = np.asarray(names, dtype=np.str_)
            arrays[
                f"{cache_field}__{map_name}__coefficients"
            ] = coefficients
    arrays.update({f"tree__{name}": value for name, value in tree.items()})
    arrays.update(
        {
            f"matter_linear__{name}": value
            for name, value in matter_linear_coefficients.items()
        }
    )
    arrays.update(
        {
            f"matter_linear_error__{name}": value
            for name, value in matter_linear_error_bounds.items()
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)
    return metadata


def load_png_cache(path: Path) -> PngTemplateSet:
    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata_json"].item()))
        if metadata.get("schema") != PNG_CACHE_SCHEMA:
            raise ValueError(f"{path} has the wrong cache schema")
        raw_indices = values["indices"]
        raw_edges = values["edges"]
        if (
            raw_indices.dtype != np.dtype(np.int64)
            or raw_edges.dtype != np.dtype(np.float64)
        ):
            raise ValueError(
                f"{path} coerces index or edge cache provenance"
            )
        indices = np.asarray(raw_indices, dtype=np.int64)
        edges = np.asarray(raw_edges, dtype=np.float64)
        tree = {
            field: np.asarray(
                values[f"tree__{field}"],
                dtype=np.float64,
            )
            for field in PNG_TREE_FIELDS
        }
        fixed_linear_names = [
            str(value) for value in values["fixed_linear_names"]
        ]
        fixed_gaussian_names = [
            str(value) for value in values["fixed_gaussian_names"]
        ]
        fixed_quadratic_names = [
            str(value) for value in values["fixed_quadratic_names"]
        ]
        fixed_linear_coefficients = np.asarray(
            values["fixed_linear_coefficients"],
            dtype=np.float64,
        )
        fixed_gaussian_coefficients = np.asarray(
            values["fixed_gaussian_coefficients"],
            dtype=np.float64,
        )
        fixed_quadratic_coefficients = np.asarray(
            values["fixed_quadratic_coefficients"],
            dtype=np.float64,
        )
        for label, names, coefficients in (
            (
                "fixed_gaussian",
                fixed_gaussian_names,
                fixed_gaussian_coefficients,
            ),
            (
                "fixed_linear",
                fixed_linear_names,
                fixed_linear_coefficients,
            ),
            (
                "fixed_quadratic",
                fixed_quadratic_names,
                fixed_quadratic_coefficients,
            ),
        ):
            if (
                len(names) != len(set(names))
                or coefficients.shape != (len(names), 120)
                or not np.all(np.isfinite(coefficients))
            ):
                raise ValueError(
                    f"PNG cache has invalid {label} sparse arrays"
                )
        matter_linear_coefficients = {
            field: np.asarray(
                values[f"matter_linear__{field}"],
                dtype=np.float64,
            )
            for field in MATTER_LINEAR_FIELDS
        }
        matter_b112_coefficients = np.asarray(
            values["matter_b112_coefficients"],
            dtype=np.float64,
        )
        matter_linear_validation_lambdas = np.asarray(
            values["matter_linear_validation_lambdas"],
            dtype=np.float64,
        )
        matter_linear_loop_validation_residuals = np.asarray(
            values[
                "matter_linear_loop_validation_residuals"
            ],
            dtype=np.float64,
        )
        matter_linear_error_lambdas = np.asarray(
            values["matter_linear_error_lambdas"],
            dtype=np.float64,
        )
        matter_linear_error_bounds = {
            field: np.asarray(
                values[f"matter_linear_error__{field}"],
                dtype=np.float64,
            )
            for field in MATTER_LINEAR_NUMERICAL_FIELDS
        }
        matter_b112_validation_lambdas = np.asarray(
            values["matter_b112_validation_lambdas"],
            dtype=np.float64,
        )
        matter_b112_validation_residuals = np.asarray(
            values["matter_b112_validation_residuals"],
            dtype=np.float64,
        )
        matter_b112_error_lambdas = np.asarray(
            values["matter_b112_error_lambdas"],
            dtype=np.float64,
        )
        matter_b112_error_bounds = np.asarray(
            values["matter_b112_error_bounds"],
            dtype=np.float64,
        )
        residual_loaded: dict[
            str, dict[str, dict[str, np.ndarray]]
        ] = {}
        for cache_field in (
            "residual_gaussian",
            "residual_linear",
            "residual_quadratic",
        ):
            residual_loaded[cache_field] = {}
            for map_name in (
                "density_only",
                "tied",
                "noisy_shift",
            ):
                names = [
                    str(value)
                    for value in values[
                        f"{cache_field}__{map_name}__names"
                    ]
                ]
                coefficients = np.asarray(
                    values[
                        f"{cache_field}__{map_name}__coefficients"
                    ],
                    dtype=np.float64,
                )
                if (
                    len(names) != len(set(names))
                    or coefficients.shape != (len(names), 120)
                    or not np.all(np.isfinite(coefficients))
                ):
                    raise ValueError(
                        "PNG cache has invalid residual sparse "
                        f"arrays for {cache_field}/{map_name}"
                    )
                residual_loaded[cache_field][map_name] = dict(
                    zip(names, coefficients)
                )
    metadata_indices = metadata.get("indices")
    if not is_plain_json_int_list(metadata_indices):
        raise ValueError("PNG cache metadata indices are not JSON integers")
    expected_indices = np.asarray(metadata_indices, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("PNG cache index metadata does not close")
    if (
        len(indices) != len(set(indices.tolist()))
        or np.any(indices < 0)
        or np.any(indices >= 120)
        or edges.shape != (indices.size, 4)
        or not np.all(np.isfinite(edges))
        or metadata.get("row_geometry", {}).get(
            "rows_bound_to_header_ranges"
        )
        is not True
        or metadata.get("row_geometry", {}).get(
            "projector_nodes_bound_to_header"
        )
        is not True
        or metadata.get("row_geometry", {}).get(
            "cache_edges_sha256"
        )
        != hashlib.sha256(
            np.ascontiguousarray(edges).tobytes()
        ).hexdigest()
        or any(value.shape != (120,) for value in tree.values())
        or any(
            value.shape != (3, 120)
            for value in matter_linear_coefficients.values()
        )
        or matter_b112_coefficients.shape != (2, 120)
        or matter_linear_loop_validation_residuals.shape
        != (matter_linear_validation_lambdas.size, 120)
        or any(
            value.shape
            != (matter_linear_error_lambdas.size, 120)
            for value in matter_linear_error_bounds.values()
        )
        or any(
            np.any(value < 0.0)
            for value in matter_linear_error_bounds.values()
        )
        or matter_b112_validation_residuals.shape
        != (matter_b112_validation_lambdas.size, 120)
        or matter_b112_error_bounds.shape
        != (matter_b112_error_lambdas.size, 120)
        or np.any(matter_b112_error_bounds < 0.0)
        or not all(
            np.all(np.isfinite(value))
            for value in (
                indices,
                *tree.values(),
                *matter_linear_coefficients.values(),
                matter_b112_coefficients,
                matter_linear_validation_lambdas,
                matter_linear_loop_validation_residuals,
                matter_linear_error_lambdas,
                *matter_linear_error_bounds.values(),
                matter_b112_validation_lambdas,
                matter_b112_validation_residuals,
                matter_b112_error_lambdas,
                matter_b112_error_bounds,
            )
        )
    ):
        raise ValueError("PNG cache has invalid array dimensions")
    return PngTemplateSet(
        indices=indices,
        edges=edges,
        tree=tree,
        fixed_gaussian=dict(
            zip(
                fixed_gaussian_names,
                fixed_gaussian_coefficients,
            )
        ),
        fixed_linear=dict(
            zip(fixed_linear_names, fixed_linear_coefficients)
        ),
        fixed_quadratic=dict(
            zip(fixed_quadratic_names, fixed_quadratic_coefficients)
        ),
        residual_gaussian=residual_loaded[
            "residual_gaussian"
        ],
        residual_linear=residual_loaded["residual_linear"],
        residual_quadratic=residual_loaded[
            "residual_quadratic"
        ],
        matter_linear_coefficients=matter_linear_coefficients,
        matter_b112_coefficients=matter_b112_coefficients,
        matter_linear_validation_lambdas=(
            matter_linear_validation_lambdas
        ),
        matter_linear_loop_validation_residuals=(
            matter_linear_loop_validation_residuals
        ),
        matter_linear_error_lambdas=(
            matter_linear_error_lambdas
        ),
        matter_linear_error_bounds=(
            matter_linear_error_bounds
        ),
        matter_b112_validation_lambdas=(
            matter_b112_validation_lambdas
        ),
        matter_b112_validation_residuals=(
            matter_b112_validation_residuals
        ),
        matter_b112_error_lambdas=matter_b112_error_lambdas,
        matter_b112_error_bounds=matter_b112_error_bounds,
        metadata=metadata,
    )


def validate_header(
    header: dict[str, Any],
    *,
    registered_reference: bool = False,
    adaptive_response: bool = False,
) -> None:
    expected_schema = (
        ADAPTIVE_BREC_RESPONSE_SCHEMA
        if adaptive_response
        else "marisa-b-eft-v2-post-r1-jsonl-v4"
    )
    expected_production_candidate = not adaptive_response
    if (
        header.get("schema") != expected_schema
        or header.get("production_candidate")
        is not expected_production_candidate
        or header.get("ir_resummation") is not False
        or header.get("stochastic_status")
        != (
            "estimator_matched_fixed_conditional_P_over_nbar"
            "_plus_released_renormalized_stochastic_PL3_closure"
            "_bare_fixed_PL3_excluded"
        )
    ):
        raise ValueError("templates do not implement registered post-R1")
    if adaptive_response:
        response = header.get("adaptive_brec_response")
        if (
            not isinstance(response, dict)
            or response.get("derivative") != "dG/dfNL_rec at fNL_rec=0"
            or response.get("method")
            != "symmetric finite difference plus Richardson extrapolation"
            or response.get("finite_difference_steps") != [1.0, 0.5]
            or response.get("fNL_rec_equals_fNL") is not True
            or not is_plain_json_number(response.get("bphi_rec"))
            or not is_plain_json_number(response.get("kmin"))
            or not math.isclose(
                float(response.get("kmin", math.nan)),
                2.0 * math.pi / 1000.0,
                rel_tol=0.0,
                abs_tol=2.0e-15,
            )
        ):
            raise ValueError(
                "adaptive-brec response header violates its derivative "
                "contract"
            )
    fixed_poisson = header.get("fixed_poisson")
    if (
        not isinstance(fixed_poisson, dict)
        or fixed_poisson.get("intensity")
        != "conditional_1_plus_delta_h"
        or fixed_poisson.get("external_estimator_filter")
        != "at_most_one_density_mark_per_cumulant"
        or fixed_poisson.get("leading_P_over_nbar") is not True
        or fixed_poisson.get("bare_PL3_in_production") is not False
    ):
        raise ValueError(
            "templates lack the registered fixed-Poisson convention"
        )
    reconstruction = header.get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("templates lack reconstruction metadata")
    expected = {
        "R": 15.0,
        "b_rec_h": 2.7340475186190334,
        "cell_size": 8.0,
    }
    for key, value in expected.items():
        raw_value = reconstruction.get(key)
        if (
            not is_plain_json_number(raw_value)
            or not math.isclose(
                float(raw_value),
                value,
                rel_tol=0.0,
                abs_tol=2.0e-14,
            )
        ):
            raise ValueError(f"template reconstruction {key} mismatch")
    if (
        not is_plain_json_int(reconstruction.get("cic_power"))
        or reconstruction.get("cic_power") != 4
    ):
        raise ValueError("template reconstruction cic_power mismatch")
    projection = header.get("shell_projection")
    orientation_orders = (
        projection.get("orientation_orders")
        if isinstance(projection, dict)
        else None
    )
    projection_orders_are_integers = (
        isinstance(projection, dict)
        and is_plain_json_int(projection.get("radial_order"))
        and is_plain_json_int(projection.get("angular_order"))
        and is_plain_json_int_list(orientation_orders, length=3)
    )
    projection_signature = (
        (
            projection["radial_order"],
            projection["angular_order"],
            tuple(orientation_orders),
        )
        if projection_orders_are_integers
        else None
    )
    if (
        not isinstance(projection, dict)
        or projection.get("measure")
        != (
            "exact float32 FFT-lattice k1-k2-mu plus "
            "normalized Haar orientation cubature"
        )
        or projection.get("normalization")
        != "raw N1*N2 estimator pairs"
        or not is_plain_json_number(projection.get("fft_box_size"))
        or not math.isclose(
            float(projection.get("fft_box_size")),
            1000.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not is_plain_json_int(projection.get("fft_mesh_size"))
        or projection.get("fft_mesh_size") != 256
        or projection.get("conditional_cubic_orientation_exact")
        is not False
        or projection.get("direct_cubic_orbit_validation_required")
        is not True
        or projection_signature
        # The 32-node rule is admitted only as a registered diagnostic
        # reference.  Production must use the same 256-node shell projector
        # as the finite-PNG cache; otherwise the exact cross-driver
        # fixed-Poisson identities compare two different numerical
        # observables.  The 2916-node rule is the refined production
        # reference.
        not in (
            {
                (1, 4, (2, 2, 2)),
                (2, 8, (2, 2, 2)),
                (3, 12, (3, 3, 3)),
            }
            if registered_reference
            else {(2, 8, (2, 2, 2))}
        )
    ):
        raise ValueError("templates violate the production shell contract")
    loop_quadrature = header.get("loop_quadrature")
    allowed_loop_quadratures = (
        ([8, 32, 24], [10, 48, 32])
        if registered_reference
        else ([8, 32, 24],)
    )
    if (
        not is_plain_json_int_list(loop_quadrature, length=3)
        or loop_quadrature not in allowed_loop_quadratures
    ):
        raise ValueError("templates use unregistered loop quadrature")
    if header.get("loop_angular_rule") != "antipodal-fibonacci":
        raise ValueError("templates use unregistered angular rule")
    q_range = header.get("q_range")
    if (
        not isinstance(q_range, list)
        or len(q_range) != 2
        or not all(is_plain_json_number(value) for value in q_range)
        or not math.isclose(
            float(q_range[0]),
            1.0e-4,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not any(
            math.isclose(
                float(q_range[1]),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-13,
            )
            for value in (
                (6.4, 9.6)
                if registered_reference
                else (6.4,)
            )
        )
    ):
        raise ValueError("templates use unregistered loop q range")
    renormalization = header.get("renormalization")
    if (
        not isinstance(renormalization, dict)
        or renormalization.get("status")
        != "renormalized-direct-post-r1-candidate"
        or renormalization.get("uv_subtraction") is not True
        or renormalization.get("subleading_tail_restoration")
        is not True
        or not is_plain_json_number(
            renormalization.get("tail_kmax")
        )
        or not math.isclose(
            float(renormalization.get("tail_kmax", math.nan)),
            30.0,
            rel_tol=0.0,
            abs_tol=1.0e-13,
        )
        or float(renormalization["tail_kmax"])
        < float(q_range[1])
    ):
        raise ValueError(
            "templates violate the registered UV-renormalization contract"
        )
    source_hashes = header.get("source_hashes")
    expected_source_hashes = {
        "linear_power",
        "edge_file",
        "driver_executable",
    }
    if adaptive_response:
        expected_source_hashes.add("png_table")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != expected_source_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
            for value in source_hashes.values()
        )
        or header.get("parameter_registry_sha256")
        != (
            "6a19a025a794ef58f417c9f6c1743f527"
            "a338f77092a4b36095d928d74b41805"
        )
        or not is_plain_json_int(header.get("radial_index_stop"))
        or header.get("radial_index_stop") != 7
    ):
        raise ValueError(
            "templates lack the registered source/registry provenance"
        )


def load_partial_templates(
    module: ModuleType,
    data: Any,
    path: Path,
    *,
    registered_reference: bool = False,
    adaptive_response: bool = False,
) -> PostTemplateSet:
    header, bins = read_jsonl(path)
    validate_header(
        header,
        registered_reference=registered_reference,
        adaptive_response=adaptive_response,
    )
    bin_indices = [row["index"] for row in bins]
    if len(set(bin_indices)) != len(bins):
        raise ValueError("post templates contain duplicate bin indices")
    recorded_indices = header.get("bin_indices")
    if (
        recorded_indices is not None
        and (
            not is_plain_json_int_list(recorded_indices)
            or recorded_indices != bin_indices
        )
    ):
        raise ValueError(
            "post merged-header bin indices do not match its rows"
        )
    recorded_range = header.get("bin_range")
    if (
        not is_plain_json_int_list(recorded_range, length=2)
        or not bin_indices
        or not (
            0
            <= recorded_range[0]
            < recorded_range[1]
            <= 120
        )
        or recorded_range
        != [min(bin_indices), max(bin_indices) + 1]
        or (
            recorded_indices is None
            and bin_indices
            != list(range(recorded_range[0], recorded_range[1]))
        )
    ):
        raise ValueError(
            "post template rows do not close against the header bin range"
        )
    projection = header["shell_projection"]
    expected_radial_order = projection["radial_order"]
    expected_angular_order = projection["angular_order"]
    expected_orientation_nodes = math.prod(
        projection["orientation_orders"]
    )
    expected_invariant_nodes = (
        expected_radial_order**2 * expected_angular_order
    )
    expected_shell_nodes = (
        expected_invariant_nodes * expected_orientation_nodes
    )
    by_index = {row["index"]: row for row in bins}
    required = set()
    for kmax in CUTS:
        mask = np.max(np.asarray(data.k_pair), axis=1) <= kmax + 1.0e-12
        mask[EXCLUDED_BIN] = False
        required.update(np.flatnonzero(mask).tolist())
    missing = sorted(required - set(by_index))
    if missing:
        raise ValueError(
            "post templates miss selected bins: "
            + ",".join(map(str, missing))
        )
    full: list[dict[str, Any]] = []
    for index in range(120):
        if index in by_index:
            row = by_index[index]
            integer_metadata = {
                name: row.get(name)
                for name in (
                    "exact_lattice_radial_order",
                    "exact_lattice_angular_order",
                    "orientation_nodes",
                    "invariant_shell_nodes",
                    "shell_nodes",
                    "total_loop_nodes",
                    "zero_external_leg_pairs",
                    "closing_zero_pairs",
                )
            }
            valid_pair_fraction = row.get(
                "exact_lattice_valid_pair_fraction"
            )
            total_variation = row.get(
                "exact_lattice_total_variation"
            )
            recorded_edges = np.asarray(
                row.get("edges", ()),
                dtype=np.float64,
            )
            expected_edges = np.asarray(
                data.edges[index],
                dtype=np.float64,
            ).reshape(-1)
            if (
                recorded_edges.shape != (4,)
                or not np.array_equal(
                    recorded_edges,
                    expected_edges,
                )
                or
                row.get("haar_oriented_cic_projection") is not True
                or row.get("reconstructed_counterterms") is not True
                or row.get("reconstructed_stochastic") is not True
                or row.get(
                    "reconstructed_stochastic_decomposed"
                )
                is not True
                or row.get(
                    "reconstructed_fixed_poisson"
                )
                is not True
                or row.get(
                    "exact_k1_k2_mu_lattice_measure"
                )
                is not True
                or row.get(
                    "conditional_cubic_orientation_exact"
                )
                is not False
                or row.get("exact_joint_lattice_measure") is not False
                or not all(
                    is_plain_json_int(value)
                    for value in integer_metadata.values()
                )
                or integer_metadata["exact_lattice_radial_order"]
                != expected_radial_order
                or integer_metadata["exact_lattice_angular_order"]
                != expected_angular_order
                or integer_metadata["orientation_nodes"]
                != expected_orientation_nodes
                or integer_metadata["invariant_shell_nodes"]
                != expected_invariant_nodes
                or integer_metadata["shell_nodes"]
                != expected_shell_nodes
                or integer_metadata["total_loop_nodes"] <= 0
                or integer_metadata["zero_external_leg_pairs"] < 0
                or integer_metadata["closing_zero_pairs"] < 0
                or not is_plain_json_number(valid_pair_fraction)
                or not is_plain_json_number(total_variation)
                or not (
                    0.0
                    < float(valid_pair_fraction)
                    <= 1.0
                )
                or not math.isclose(
                    float(total_variation),
                    float(valid_pair_fraction),
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-14,
                )
            ):
                raise ValueError(
                    f"post template bin {index} has the wrong geometry "
                    "or lacks a required sector"
                )
            full.append(row)
            continue
        edge = np.asarray(data.edges[index], dtype=np.float64)
        full.append(
            {
                "record": "bin",
                "index": index,
                "edges": [
                    float(edge[0, 0]),
                    float(edge[0, 1]),
                    float(edge[1, 0]),
                    float(edge[1, 1]),
                ],
                "diagrams": {
                    name: {} for name in module.DIAGRAM_NAMES
                },
                "counterterms": {
                    name: {} for name in module.COUNTERTERM_NAMES
                },
                "stochastic": {
                    name: {} for name in module.STOCHASTIC_B_NAMES
                },
                "stochastic_density_only": {
                    name: {} for name in module.STOCHASTIC_B_NAMES
                },
                "stochastic_noisy_shift": {
                    name: {} for name in module.STOCHASTIC_B_NAMES
                },
                "bshot_bnabla2_cross": {},
                "bshot_bnabla2_cross_density_only": {},
                "bshot_bnabla2_cross_noisy_shift": {},
                "fixed_poisson": {
                    order: {
                        "by_inverse_number_density": {
                            f"nbar^-{power}": {}
                            for power in range(1, 4)
                        },
                        "generated_topologies": 0,
                        "estimator_allowed_topologies": 0,
                        "integration_nodes": 0,
                    }
                    for order in ("tree", "one_loop")
                },
            }
        )

    fixed_poisson_terms: dict[
        int, dict[str, np.ndarray]
    ] = {}
    for inverse_nbar in range(1, 4):
        key = f"nbar^-{inverse_nbar}"
        compiled: dict[str, np.ndarray] = {}
        for bin_index, row in enumerate(full):
            tree = row["fixed_poisson"]["tree"][
                "by_inverse_number_density"
            ][key]
            one_loop = row["fixed_poisson"]["one_loop"][
                "by_inverse_number_density"
            ][key]
            if one_loop:
                raise ValueError(
                    "bare fixed-Poisson PL3 terms must not enter "
                    f"production (bin {row['index']}, {key})"
                )
            for monomial, coefficient in tree.items():
                compiled.setdefault(
                    monomial,
                    np.zeros(120, dtype=np.float64),
                )[bin_index] = float(coefficient)
        fixed_poisson_terms[inverse_nbar] = compiled
    if not fixed_poisson_terms[1]:
        raise ValueError(
            "production templates lack the estimator-matched "
            "fixed P/nbar term"
        )
    if fixed_poisson_terms[2] or fixed_poisson_terms[3]:
        raise ValueError(
            "unexpected leading fixed-Poisson nbar^-2 or nbar^-3 term"
        )

    def variant(
        stochastic_field: str,
        cross_field: str,
    ) -> Any:
        rows = []
        for row in full:
            transformed = dict(row)
            transformed["stochastic"] = row[
                stochastic_field
            ]
            transformed["bshot_bnabla2_cross"] = row[
                cross_field
            ]
            rows.append(transformed)
        return module.TemplateSet(
            path=path,
            header=header,
            bins=rows,
        )

    result = PostTemplateSet(
        tied=variant(
            "stochastic",
            "bshot_bnabla2_cross",
        ),
        density_only=variant(
            "stochastic_density_only",
            "bshot_bnabla2_cross_density_only",
        ),
        noisy_shift=variant(
            "stochastic_noisy_shift",
            "bshot_bnabla2_cross_noisy_shift",
        ),
        stochastic_names=tuple(
            module.STOCHASTIC_B_NAMES
        ),
        fixed_poisson_terms=fixed_poisson_terms,
    )
    # Enforce the exported polynomial identity before any fit consumes it.
    for row in full:
        for name in module.STOCHASTIC_B_NAMES:
            tied = row["stochastic"][name]
            density = row["stochastic_density_only"][name]
            shift = row["stochastic_noisy_shift"][name]
            monomials = set(tied) | set(density) | set(shift)
            for monomial in monomials:
                if not math.isclose(
                    float(tied.get(monomial, 0.0)),
                    float(density.get(monomial, 0.0))
                    + float(shift.get(monomial, 0.0)),
                    rel_tol=2.0e-12,
                    abs_tol=2.0e-8,
                ):
                    raise ValueError(
                        "post stochastic decomposition "
                        f"fails in bin {row['index']}, "
                        f"{name}, {monomial}"
                    )
    return result


def calibrated_prior_block(
    runner: ModuleType,
    module: ModuleType,
    frozen_prior: Any,
    names: tuple[str, ...],
    nonlinear_names: tuple[str, ...],
    nuisance_scale: float,
) -> Any:
    base_names = tuple(
        name for name in names
        if name in module.FIT_NAMES
    )
    block = runner.prior_block(
        module,
        frozen_prior,
        base_names,
        nuisance_scale=nuisance_scale,
        nuisance_names=tuple(
            name for name in base_names
            if name not in nonlinear_names
        ),
    )
    mean = np.zeros(len(names), dtype=np.float64)
    covariance = np.zeros(
        (len(names), len(names)),
        dtype=np.float64,
    )
    base_locations = np.asarray(
        [names.index(name) for name in base_names],
        dtype=int,
    )
    mean[base_locations] = block.mean
    covariance[np.ix_(
        base_locations,
        base_locations,
    )] = block.covariance
    for base_name in RELEASED_SHIFT_BASE:
        released_name = (
            RELEASED_SHIFT_PREFIX + base_name
        )
        if released_name not in names:
            continue
        location = names.index(released_name)
        frozen_location = module.FIT_NAMES.index(
            base_name
        )
        covariance[location, location] = (
            RELEASED_SHIFT_PRIOR_SCALE
            * float(frozen_prior.sigma[frozen_location])
        ) ** 2
    if PNG_STOCHASTIC_NAME in names:
        location = names.index(PNG_STOCHASTIC_NAME)
        frozen_location = module.FIT_NAMES.index(
            "Bshot_residual"
        )
        mean[location] = 0.0
        covariance[location, location] = (
            PNG_STOCHASTIC_PRIOR_SCALE
            * nuisance_scale
            * float(frozen_prior.sigma[frozen_location])
        ) ** 2
    b1_index = names.index("b1")
    mean[b1_index] = B1_PRIOR_MEAN
    covariance[b1_index, :] = 0.0
    covariance[:, b1_index] = 0.0
    covariance[b1_index, b1_index] = B1_PRIOR_SIGMA**2
    whitener = solve_triangular(
        np.linalg.cholesky(covariance),
        np.eye(len(names)),
        lower=True,
    )
    return runner.PriorBlock(
        names=names,
        mean=mean,
        covariance=covariance,
        sigma=np.sqrt(np.diag(covariance)),
        whitener=whitener,
    )


def free_names(
    module: ModuleType,
    tier: str,
    stochastic_mode: str,
) -> tuple[str, ...]:
    excluded = {"Pshot", "a0_power"}
    if tier == "coevolution":
        excluded.update(COEVOLUTION_DEPENDENT)
    elif tier != "full":
        raise ValueError(tier)
    if stochastic_mode not in {
        "zero",
        "tied",
        "released",
    }:
        raise ValueError(stochastic_mode)
    result = tuple(
        name for name in module.FIT_NAMES
        if name not in excluded
    )
    if stochastic_mode == "released":
        result += tuple(
            RELEASED_SHIFT_PREFIX + name
            for name in RELEASED_SHIFT_BASE
        )
    return result


def nonlinear_names(tier: str) -> tuple[str, ...]:
    names = ("b1", "b2", "gamma2", "b_nabla2_delta")
    if tier == "full":
        names = names + ("gamma21",)
    return names


def fit_one(
    *,
    module: ModuleType,
    runner: ModuleType,
    templates: Any,
    data: Any,
    frozen_prior: Any,
    nbar: float,
    samples: np.ndarray,
    tier: str,
    stochastic_mode: str,
    kmax: float,
    nuisance_scale: float,
    multistart: int,
    rng: np.random.Generator,
) -> Any:
    mask = np.max(np.asarray(data.k_pair), axis=1) <= kmax + 1.0e-12
    mask[EXCLUDED_BIN] = False
    indices = np.flatnonzero(mask)
    selected_samples = np.asarray(samples[:, indices], dtype=np.float64)
    target = np.mean(selected_samples, axis=0)
    covariance = runner.covariance_single(selected_samples)
    hartlap_factor = (
        selected_samples.shape[0] - indices.size - 2
    ) / (selected_samples.shape[0] - 1)
    if not (0.0 < hartlap_factor <= 1.0):
        raise ValueError(
            "Gaussian fit has too few covariance realizations for "
            "the registered Hartlap precision correction"
        )
    names = free_names(module, tier, stochastic_mode)
    nonlinear = nonlinear_names(tier)
    prior = calibrated_prior_block(
        runner,
        module,
        frozen_prior,
        names,
        nonlinear,
        nuisance_scale,
    )

    def model(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        parameters = runner.full_parameter_dict(
            module,
            names,
            values,
            tier=tier,
            fixed_b1=None,
        )
        components = templates.components(
            parameters,
            nbar,
            stochastic_mode,
        )
        return components["total"][indices], components

    def expanded(
        values: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        (
            expanded_names,
            expanded_values,
            expanded_jacobian,
        ) = runner.expanded_mapping(
            module,
            names,
            values,
            tier=tier,
            fixed_b1=None,
        )
        released_names = tuple(
            name for name in names
            if name.startswith(
                RELEASED_SHIFT_PREFIX
            )
        )
        if not released_names:
            return (
                expanded_names,
                expanded_values,
                expanded_jacobian,
            )
        rows = np.zeros(
            (len(released_names), len(names)),
            dtype=np.float64,
        )
        released_values = []
        for row, name in enumerate(released_names):
            index = names.index(name)
            rows[row, index] = 1.0
            released_values.append(values[index])
        return (
            expanded_names + released_names,
            np.concatenate(
                (
                    expanded_values,
                    np.asarray(released_values),
                )
            ),
            np.vstack((expanded_jacobian, rows)),
        )

    return runner.fit_conditional(
        module,
        label=f"post_B000_{stochastic_mode}",
        tier=tier,
        kmax=kmax,
        free_names=names,
        nonlinear_names=nonlinear,
        prior=prior,
        target=target,
        covariance_fit_matrix=covariance / hartlap_factor,
        covariance_single_matrix=covariance,
        selected_indices=indices,
        model_and_components=model,
        expanded=expanded,
        multistart=multistart,
        rng=rng,
    )


def fit_summary(fit: Any) -> dict[str, Any]:
    hartlap_factor = float(
        np.median(
            np.diag(fit.covariance_single)
            / np.diag(fit.covariance_fit)
        )
    )
    parameters = {
        name: {
            "value": fit.expanded_value(name),
            "sigma": fit.expanded_sigma(name),
        }
        for name in fit.expanded_names
    }
    return {
        "tier": fit.tier,
        "label": fit.label,
        "kmax_h_mpc": fit.kmax,
        "selected_global_indices": fit.selected_indices,
        "free_names": fit.free_names,
        "nonlinear_names": fit.nonlinear_names,
        "parameters": parameters,
        "statistics": {
            "n_data": int(fit.target.size),
            "data_chi2": fit.data_chi2,
            "prior_chi2": fit.prior_chi2,
            "objective": fit.objective,
            "residual_norm_sigma_single": qnorm(
                fit.residual,
                fit.covariance_single,
            ),
            "max_abs_single_pull": fit.max_abs_single_pull,
            "p_value": fit.p_value,
            "effective_parameter_count": fit.effective_parameter_count,
            "dof": fit.dof,
            "hartlap_factor": hartlap_factor,
        },
        "numerical": {
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "multistart_objectives": fit.multistart_objectives,
            "affine_residual_max_sigma_single": (
                fit.affine_residual_max_sigma_single
            ),
            "basis_rotation_max_sigma_single": (
                fit.basis_rotation_max_sigma_single
            ),
            "analytic_linear_delta_objective": (
                fit.analytic_linear_delta_objective
            ),
        },
    }


def qnorm(vector: np.ndarray, covariance: np.ndarray) -> float:
    whitened = solve_triangular(
        np.linalg.cholesky(covariance),
        np.asarray(vector, dtype=np.float64),
        lower=True,
    )
    return float(np.linalg.norm(whitened))


def gaussian_coevolution_submanifold_audit(
    *,
    module: ModuleType,
    runner: ModuleType,
    templates: PostTemplateSet,
    number_density: float,
) -> dict[str, Any]:
    """Prove that the full tier contains the coevolution tier exactly."""

    coevolution_names = free_names(module, "coevolution", "tied")
    full_names = free_names(module, "full", "tied")
    rng = np.random.default_rng(260727)
    component_names = (
        "tree",
        "loop",
        "counterterm",
        "stochastic",
        "fixed_poisson",
        "total",
    )
    maximum_parameter_error = 0.0
    maximum_component_error = {
        name: 0.0 for name in component_names
    }
    maximum_component_relative_error = {
        name: 0.0 for name in component_names
    }
    sample_rows: list[dict[str, Any]] = []
    for sample in range(8):
        values = 0.2 * rng.normal(size=len(coevolution_names))
        values[coevolution_names.index("b1")] = (
            B1_PRIOR_MEAN + 0.15 * rng.normal()
        )
        coevolution = runner.full_parameter_dict(
            module,
            coevolution_names,
            values,
            tier="coevolution",
            fixed_b1=None,
        )
        full_values = np.asarray(
            [coevolution[name] for name in full_names],
            dtype=np.float64,
        )
        full = runner.full_parameter_dict(
            module,
            full_names,
            full_values,
            tier="full",
            fixed_b1=None,
        )
        parameter_error = max(
            abs(float(coevolution[name]) - float(full[name]))
            for name in module.FIT_NAMES
        )
        maximum_parameter_error = max(
            maximum_parameter_error,
            parameter_error,
        )
        coevolution_components = templates.components(
            coevolution,
            number_density,
            "tied",
        )
        full_components = templates.components(
            full,
            number_density,
            "tied",
        )
        row_errors: dict[str, float] = {}
        for name in component_names:
            first = np.asarray(
                coevolution_components[name],
                dtype=np.float64,
            )
            second = np.asarray(
                full_components[name],
                dtype=np.float64,
            )
            error = float(np.max(np.abs(first - second)))
            scale = float(
                max(
                    np.max(np.abs(first)),
                    np.max(np.abs(second)),
                    np.finfo(float).tiny,
                )
            )
            relative = error / scale
            maximum_component_error[name] = max(
                maximum_component_error[name],
                error,
            )
            maximum_component_relative_error[name] = max(
                maximum_component_relative_error[name],
                relative,
            )
            row_errors[name] = error
        sample_rows.append(
            {
                "sample": sample,
                "b1": float(coevolution["b1"]),
                "maximum_parameter_error": parameter_error,
                "maximum_component_absolute_error": row_errors,
            }
        )
    passed = bool(
        maximum_parameter_error <= 1.0e-14
        and all(
            maximum_component_error[name] <= 2.0e-8
            or maximum_component_relative_error[name] <= 2.0e-13
            for name in component_names
        )
    )
    return {
        "semantics": (
            "eight result-blind coevolution points are embedded in the "
            "full parameter space and every post-reconstruction Gaussian "
            "component is compared before fitting"
        ),
        "sample_count": len(sample_rows),
        "maximum_parameter_error": maximum_parameter_error,
        "maximum_component_absolute_error": (
            maximum_component_error
        ),
        "maximum_component_relative_error": (
            maximum_component_relative_error
        ),
        "samples": sample_rows,
        "pass": passed,
    }


def gaussian_perturbativity_audit(
    templates: PostTemplateSet,
    fits: dict[tuple[str, str, float], Any],
    number_density: float,
) -> dict[str, Any]:
    """Audit the fitted deterministic expansion in the inference metric.

    The one-loop diagrams and renormalized counterterms are perturbative
    corrections to the deterministic tree prediction.  Residual
    stochasticity and the estimator-matched fixed-Poisson term are reported
    in the same single-realization covariance norm, but are not assigned an
    SPT loop order and therefore do not enter this gate.
    """

    ratio_threshold = 1.0
    rows: list[dict[str, Any]] = []
    for kmax in CUTS:
        fit = fits[(PRIMARY_TIER, "tied", kmax)]
        parameters = {
            name: fit.expanded_value(name)
            for name in fit.expanded_names
        }
        components = templates.components(
            parameters,
            number_density,
            "tied",
        )
        indices = np.asarray(
            fit.selected_indices,
            dtype=np.int64,
        )
        covariance = np.asarray(
            fit.covariance_single,
            dtype=np.float64,
        )
        selected = {
            name: np.asarray(components[name], dtype=np.float64)[
                indices
            ]
            for name in (
                "tree",
                "loop",
                "counterterm",
                "stochastic",
                "fixed_poisson",
            )
        }
        selected["renormalized_one_loop_correction"] = (
            selected["loop"] + selected["counterterm"]
        )
        norms = {
            name: qnorm(value, covariance)
            for name, value in selected.items()
        }
        sigma = np.sqrt(np.diag(covariance))
        maximum_pulls = {
            name: float(np.max(np.abs(value) / sigma))
            for name, value in selected.items()
        }
        tree_norm = max(
            norms["tree"],
            np.finfo(float).tiny,
        )
        ratios = {
            "loop_to_tree": norms["loop"] / tree_norm,
            "counterterm_to_tree": (
                norms["counterterm"] / tree_norm
            ),
            "renormalized_one_loop_correction_to_tree": (
                norms["renormalized_one_loop_correction"]
                / tree_norm
            ),
            "stochastic_to_tree_report_only": (
                norms["stochastic"] / tree_norm
            ),
            "fixed_poisson_to_tree_report_only": (
                norms["fixed_poisson"] / tree_norm
            ),
        }
        row_pass = bool(
            ratios["loop_to_tree"] < ratio_threshold
            and ratios["counterterm_to_tree"] < ratio_threshold
            and ratios[
                "renormalized_one_loop_correction_to_tree"
            ]
            < ratio_threshold
        )
        rows.append(
            {
                "kmax_h_mpc": kmax,
                "selected_global_indices": indices,
                "covariance": "single-realization",
                "component_norms_sigma_single": norms,
                "component_max_abs_single_bin_sigma": maximum_pulls,
                "ratios": ratios,
                "pass": row_pass,
            }
        )
    return {
        "semantics": (
            "best-fit coevolution/tied deterministic tree versus "
            "one-loop and counterterm components in the "
            "single-realization covariance metric"
        ),
        "ratio_threshold": ratio_threshold,
        "gated_ratios": [
            "loop_to_tree",
            "counterterm_to_tree",
            "renormalized_one_loop_correction_to_tree",
        ],
        "report_only_components": [
            "stochastic",
            "fixed_poisson",
        ],
        "rows": rows,
        "pass": bool(all(row["pass"] for row in rows)),
    }


def png_cache_numerical_gate(
    templates: PngTemplateSet,
    indices: np.ndarray,
    covariance_single: np.ndarray,
) -> dict[str, Any]:
    """Convert cache interpolation/QMC errors to the inference metric."""

    indices = np.asarray(indices, dtype=np.int64)
    sigma = np.sqrt(np.diag(covariance_single))

    def rows(
        lambdas: np.ndarray,
        residuals: np.ndarray,
        fnl_power: int,
    ) -> list[dict[str, Any]]:
        result = []
        for lambda_value, residual in zip(lambdas, residuals):
            b1 = float(lambda_value) * B_REC_H
            scaled = (
                100.0**fnl_power
                * b1**3
                * np.asarray(residual, dtype=np.float64)[indices]
            )
            result.append(
                {
                    "lambda": float(lambda_value),
                    "b1_corresponding": b1,
                    "absolute_fNL_envelope": 100.0,
                    "vector_norm_sigma_single": qnorm(
                        scaled,
                        covariance_single,
                    ),
                    "max_abs_single_bin_sigma": float(
                        np.max(np.abs(scaled) / sigma)
                    ),
                }
            )
        return result

    linear = rows(
        templates.matter_linear_validation_lambdas,
        templates.matter_linear_loop_validation_residuals,
        1,
    )
    b112_interpolation = rows(
        templates.matter_b112_validation_lambdas,
        templates.matter_b112_validation_residuals,
        2,
    )
    required_holdouts = np.asarray([1.37, 2.2])
    linear_holdouts_complete = bool(
        templates.matter_linear_validation_lambdas.shape == (2,)
        and np.allclose(
            np.sort(templates.matter_linear_validation_lambdas),
            required_holdouts,
            rtol=0.0,
            atol=2.0e-14,
        )
    )
    b112_holdouts_complete = bool(
        templates.matter_b112_validation_lambdas.shape == (2,)
        and np.allclose(
            np.sort(templates.matter_b112_validation_lambdas),
            required_holdouts,
            rtol=0.0,
            atol=2.0e-14,
        )
    )
    error_lambdas = templates.matter_b112_error_lambdas
    zero_locations = np.flatnonzero(
        np.isclose(error_lambdas, 0.0, rtol=0.0, atol=2.0e-14)
    )
    one_locations = np.flatnonzero(
        np.isclose(error_lambdas, 1.0, rtol=0.0, atol=2.0e-14)
    )
    b112_qmc_rows: list[dict[str, Any]] = []
    if zero_locations.size == 1 and one_locations.size == 1:
        error_zero = templates.matter_b112_error_bounds[
            int(zero_locations[0])
        ]
        error_one = templates.matter_b112_error_bounds[
            int(one_locations[0])
        ]
        for b1 in np.linspace(0.1, 6.0, 129):
            lambda_value = b1 / B_REC_H
            propagated = (
                abs(1.0 - lambda_value) * error_zero
                + abs(lambda_value) * error_one
            )
            scaled = (
                100.0**2 * b1**3 * propagated[indices]
            )
            b112_qmc_rows.append(
                {
                    "lambda": float(lambda_value),
                    "b1": float(b1),
                    "absolute_fNL_envelope": 100.0,
                    "coherent_sign_vector_norm_sigma_single": (
                        qnorm(scaled, covariance_single)
                    ),
                    "max_abs_single_bin_sigma": float(
                        np.max(np.abs(scaled) / sigma)
                    ),
                }
            )
    linear_error_lambdas = templates.matter_linear_error_lambdas
    linear_locations = {
        value: np.flatnonzero(
            np.isclose(
                linear_error_lambdas,
                value,
                rtol=0.0,
                atol=2.0e-14,
            )
        )
        for value in (0.0, 1.0, 2.0)
    }
    linear_qmc_rows: list[dict[str, Any]] = []
    if all(
        locations.size == 1
        for locations in linear_locations.values()
    ):
        errors = {
            value: templates.matter_linear_error_bounds["total"][
                int(linear_locations[value][0])
            ]
            for value in (0.0, 1.0, 2.0)
        }
        for b1 in np.linspace(0.1, 6.0, 129):
            lambda_value = b1 / B_REC_H
            weights = {
                0.0: 0.5
                * (lambda_value - 1.0)
                * (lambda_value - 2.0),
                1.0: -lambda_value * (lambda_value - 2.0),
                2.0: 0.5
                * lambda_value
                * (lambda_value - 1.0),
            }
            propagated = sum(
                abs(weights[value]) * errors[value]
                for value in (0.0, 1.0, 2.0)
            )
            scaled = 100.0 * b1**3 * propagated[indices]
            linear_qmc_rows.append(
                {
                    "lambda": float(lambda_value),
                    "b1": float(b1),
                    "absolute_fNL_envelope": 100.0,
                    "coherent_sign_vector_norm_sigma_single": (
                        qnorm(scaled, covariance_single)
                    ),
                    "max_abs_single_bin_sigma": float(
                        np.max(np.abs(scaled) / sigma)
                    ),
                }
            )
    linear_qmc_worst = (
        max(
            linear_qmc_rows,
            key=lambda row: row["max_abs_single_bin_sigma"],
        )
        if linear_qmc_rows
        else None
    )
    qmc_worst = (
        max(
            b112_qmc_rows,
            key=lambda row: row["max_abs_single_bin_sigma"],
        )
        if b112_qmc_rows
        else None
    )
    threshold_pass = bool(
        linear
        and b112_interpolation
        and linear_holdouts_complete
        and b112_holdouts_complete
        and qmc_worst is not None
        and linear_qmc_worst is not None
        and all(
            row["vector_norm_sigma_single"] < 0.05
            and row["max_abs_single_bin_sigma"] < 0.1
            for row in linear + b112_interpolation
        )
        and qmc_worst["max_abs_single_bin_sigma"] < 0.1
        and linear_qmc_worst["max_abs_single_bin_sigma"] < 0.1
    )
    provenance_pass = bool(
        templates.metadata.get("all_raw_inputs_current_v5")
        and templates.metadata.get("linear_interpolation", {}).get(
            "degree"
        )
        == 2
        and templates.metadata.get("linear_interpolation", {}).get(
            "canonical_nodes"
        )
        == [0.0, 1.0, 2.0]
    )
    return {
        "pass": threshold_pass and provenance_pass,
        "thresholds": {
            "vector_norm_sigma_single": 0.05,
            "max_abs_single_bin_sigma": 0.1,
        },
        "current_v5_provenance": provenance_pass,
        "matter_linear_quadratic_holdout": linear,
        "matter_linear_holdouts_complete_1p37_2p2": (
            linear_holdouts_complete
        ),
        "matter_b112_linear_holdout": b112_interpolation,
        "matter_b112_holdouts_complete_1p37_2p2": (
            b112_holdouts_complete
        ),
        "matter_b112_weighted_qmc_propagation": {
            "method": (
                "|1-lambda| e(lambda=0) + "
                "|lambda| e(lambda=1), scanned over b1 in [0.1,6]"
            ),
            "scan_count": len(b112_qmc_rows),
            "worst_per_bin_row": qmc_worst,
        },
        "matter_linear_weighted_qmc_propagation": {
            "method": (
                "absolute quadratic-Lagrange weights at lambda=0,1,2, "
                "scanned over b1 in [0.1,6]"
            ),
            "scan_count": len(linear_qmc_rows),
            "worst_per_bin_row": linear_qmc_worst,
        },
        "qmc_vector_norm_note": (
            "coherent-positive-sign diagnostic only; the rigorous "
            "registered QMC gate is the per-bin 0.1 sigma bound"
        ),
    }


def png_tail_canary_gate(
    *,
    raw_paths: Iterable[Path],
    templates: PngTemplateSet,
    expected_edges: np.ndarray,
    covariance_samples: np.ndarray,
) -> dict[str, Any]:
    """Audit the unintegrated PNG UV tail at two result-blind bins.

    The production cache stops the matter-PNG loop integrals at
    ``qmax=20``.  Replicate-QMC errors only cover that retained domain, so
    they cannot bound the omitted tail.  The registered canary evaluates
    the same lambda=1 observable at qmax=10 and 40 on the most squeezed
    retained bin and the highest-k retained diagonal bin.  Only the
    20-to-40 displacement is gated; 10-to-20 is an explicit convergence
    diagnostic.
    """

    paths = tuple(path.resolve() for path in raw_paths)
    expected_keys = {
        (sector, index, qmax)
        for sector in ("matter-linear", "matter-b112")
        for index in PNG_TAIL_CANARY_INDICES
        for qmax in PNG_TAIL_CANARY_QMAX
    }
    if len(paths) != len(expected_keys) or len(set(paths)) != len(paths):
        raise ValueError(
            "PNG tail canary requires exactly eight distinct JSONL files"
        )
    production_common = templates.metadata.get(
        "common_raw_contract",
    )
    if not isinstance(production_common, dict):
        raise ValueError("PNG cache lacks its common raw contract")
    production_integration = production_common.get("integration")
    if (
        not isinstance(production_integration, dict)
        or not math.isclose(
            float(production_integration.get("qmax", math.nan)),
            PNG_TAIL_REFERENCE_QMAX,
            rel_tol=0.0,
            abs_tol=1.0e-13,
        )
    ):
        raise ValueError("PNG production cache is not the qmax=20 reference")
    if not set(PNG_TAIL_CANARY_INDICES).issubset(
        set(np.asarray(templates.indices, dtype=np.int64).tolist())
    ):
        raise ValueError("PNG production cache lacks a tail-canary bin")

    common_header_fields = (
        "input_contract",
        "source_hashes",
        "lambda_definition",
        "shell_projection",
        "reconstruction",
        "integration",
        "fixed_poisson_png",
        "radial_stop",
    )
    expected_edges = np.asarray(expected_edges, dtype=np.float64)
    if expected_edges.shape != (120, 2, 2):
        raise ValueError("tail-canary edge reference has the wrong shape")

    values: dict[
        tuple[str, int, float],
        dict[str, Any],
    ] = {}
    source_files: list[dict[str, Any]] = []
    for path in paths:
        header, bins = read_jsonl(path)
        _validate_png_header(
            path,
            header,
            allowed_qmax=PNG_TAIL_CANARY_QMAX,
        )
        _validate_png_shard_rows(path, header, bins)
        if header.get("schema") != PNG_JSONL_SCHEMA:
            raise ValueError(
                f"{path} is not a current-v5 tail canary"
            )
        sector = str(header.get("sector"))
        if sector not in {"matter-linear", "matter-b112"}:
            raise ValueError(f"{path} has an invalid tail sector")
        if not math.isclose(
            float(header.get("lambda", math.nan)),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ):
            raise ValueError(f"{path} is not the lambda=1 canary")
        integration = header["integration"]
        raw_qmax = float(integration["qmax"])
        matching_qmax = [
            value
            for value in PNG_TAIL_CANARY_QMAX
            if math.isclose(
                raw_qmax,
                value,
                rel_tol=0.0,
                abs_tol=1.0e-13,
            )
        ]
        if len(matching_qmax) != 1:
            raise ValueError(f"{path} has an invalid tail qmax")
        qmax = matching_qmax[0]
        raw_common = {
            key: header.get(key)
            for key in common_header_fields
        }
        changed = deep_change_paths(
            production_common,
            raw_common,
            ignored_paths=frozenset({"integration.qmax"}),
        )
        if changed:
            raise ValueError(
                f"{path} changes non-qmax production settings: "
                + ", ".join(changed)
            )
        if len(bins) != 1:
            raise ValueError(
                f"{path} must contain exactly one tail-canary bin"
            )
        row = bins[0]
        index = row["index"]
        if index not in PNG_TAIL_CANARY_INDICES:
            raise ValueError(f"{path} has an unregistered canary bin")
        if header.get("bin_range") != [index, index + 1]:
            raise ValueError(
                f"{path} has a mismatched single-bin header range"
            )
        key = (sector, index, qmax)
        if key in values:
            raise ValueError(f"duplicate PNG tail canary {key}")
        if not np.array_equal(
            np.asarray(row.get("edges"), dtype=np.float64),
            expected_edges[index].reshape(-1),
        ):
            raise ValueError(f"{path} has mismatched canary-bin edges")
        if (
            not is_plain_json_int(row.get("shell_nodes"))
            or row.get("shell_nodes") != 256
            or not is_plain_json_int(
                row.get("invariant_shell_nodes")
            )
            or row.get("invariant_shell_nodes") != 32
            or not is_plain_json_int(row.get("orientation_nodes"))
            or row.get("orientation_nodes") != 8
        ):
            raise ValueError(
                f"{path} does not use the production 256-node projector"
            )

        if sector == "matter-linear":
            payload = row.get("matter_linear")
            if not isinstance(payload, dict):
                raise ValueError(f"{path} lacks matter-linear values")
            numerical = payload.get("numerical_error_estimate")
            if not isinstance(numerical, dict):
                raise ValueError(
                    f"{path} lacks matter-linear QMC errors"
                )
            component_values = {
                field: float(payload.get(field, math.nan))
                for field in MATTER_LINEAR_FIELDS
            }
            component_errors = {
                field: float(numerical.get(field, math.nan))
                for field in MATTER_LINEAR_NUMERICAL_FIELDS
            }
            neval = payload.get("neval")
            if (
                not all(
                    math.isfinite(value)
                    for value in component_values.values()
                )
                or not all(
                    math.isfinite(value) and value >= 0.0
                    for value in component_errors.values()
                )
                or neval <= 0
            ):
                raise ValueError(
                    f"{path} has invalid matter-linear numerics"
                )
            diagram_sum = sum(
                component_values[field]
                for field in MATTER_LINEAR_DIAGRAM_FIELDS
            )
            diagram_error_sum = sum(
                component_errors[field]
                for field in MATTER_LINEAR_DIAGRAM_FIELDS
            )
            if (
                not math.isclose(
                    component_values["loop"],
                    diagram_sum,
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-10,
                )
                or not math.isclose(
                    component_values["total"],
                    component_values["tree"]
                    + component_values["loop"],
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-10,
                )
                or not math.isclose(
                    component_errors["loop"],
                    diagram_error_sum,
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-12,
                )
                or not math.isclose(
                    component_errors["total"],
                    component_errors["loop"],
                    rel_tol=2.0e-13,
                    abs_tol=2.0e-12,
                )
            ):
                raise ValueError(
                    f"{path} fails the matter-linear component closure"
                )
            values[key] = {
                "value": component_values["loop"],
                "error_bound": component_errors["loop"],
                "tree": component_values["tree"],
                "components": component_values,
                "component_error_bounds": component_errors,
                "neval": neval,
            }
        else:
            payload = row.get("matter_b112")
            if not isinstance(payload, dict):
                raise ValueError(f"{path} lacks matter-b112 values")
            coefficient = float(
                payload.get("coefficient", math.nan)
            )
            error_bound = float(
                payload.get("weighted_error_bound", math.nan)
            )
            neval = payload.get("neval")
            if (
                not math.isfinite(coefficient)
                or not math.isfinite(error_bound)
                or error_bound < 0.0
                or neval <= 0
            ):
                raise ValueError(
                    f"{path} has invalid matter-b112 numerics"
                )
            values[key] = {
                "value": coefficient,
                "error_bound": error_bound,
                "neval": neval,
            }
        source_files.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "sector": sector,
                "index": index,
                "lambda": 1.0,
                "qmax_h_mpc": qmax,
            }
        )
    if set(values) != expected_keys:
        raise ValueError(
            "PNG tail-canary grid is incomplete: "
            + json.dumps(
                {
                    "missing": sorted(expected_keys - set(values)),
                    "unexpected": sorted(set(values) - expected_keys),
                },
                sort_keys=True,
            )
        )

    indices = np.asarray(PNG_TAIL_CANARY_INDICES, dtype=np.int64)
    covariance_samples = np.asarray(
        covariance_samples,
        dtype=np.float64,
    )
    if (
        covariance_samples.ndim != 2
        or covariance_samples.shape[0] < 4
        or covariance_samples.shape[1] != 120
        or not np.all(np.isfinite(covariance_samples))
    ):
        raise ValueError(
            "tail-canary covariance samples have the wrong contract"
        )
    covariance_single = np.cov(
        covariance_samples[:, indices],
        rowvar=False,
        ddof=1,
    )
    if (
        covariance_single.shape != (2, 2)
        or not np.all(np.isfinite(covariance_single))
    ):
        raise ValueError("tail-canary covariance is invalid")

    def lambda_location(lambdas: np.ndarray) -> int:
        locations = np.flatnonzero(
            np.isclose(
                np.asarray(lambdas, dtype=np.float64),
                1.0,
                rtol=0.0,
                atol=2.0e-14,
            )
        )
        if locations.size != 1:
            raise ValueError("PNG cache lacks one unique lambda=1 error row")
        return int(locations[0])

    linear_error_location = lambda_location(
        templates.matter_linear_error_lambdas
    )
    b112_error_location = lambda_location(
        templates.matter_b112_error_lambdas
    )
    linear_values = {
        PNG_TAIL_REFERENCE_QMAX: (
            PngTemplateSet._evaluate_polynomial(
                templates.matter_linear_coefficients["loop"],
                1.0,
            )[indices]
        )
    }
    linear_errors = {
        PNG_TAIL_REFERENCE_QMAX: (
            templates.matter_linear_error_bounds["loop"][
                linear_error_location,
                indices,
            ]
        )
    }
    linear_tree_values = {
        PNG_TAIL_REFERENCE_QMAX: (
            PngTemplateSet._evaluate_polynomial(
                templates.matter_linear_coefficients["tree"],
                1.0,
            )[indices]
        )
    }
    b112_values = {
        PNG_TAIL_REFERENCE_QMAX: (
            PngTemplateSet._evaluate_polynomial(
                templates.matter_b112_coefficients,
                1.0,
            )[indices]
        )
    }
    b112_errors = {
        PNG_TAIL_REFERENCE_QMAX: (
            templates.matter_b112_error_bounds[
                b112_error_location,
                indices,
            ]
        )
    }
    linear_component_values = {
        field: {
            PNG_TAIL_REFERENCE_QMAX: (
                PngTemplateSet._evaluate_polynomial(
                    templates.matter_linear_coefficients[field],
                    1.0,
                )[indices]
            )
        }
        for field in MATTER_LINEAR_DIAGRAM_FIELDS
    }
    linear_component_errors = {
        field: {
            PNG_TAIL_REFERENCE_QMAX: (
                templates.matter_linear_error_bounds[field][
                    linear_error_location,
                    indices,
                ]
            )
        }
        for field in MATTER_LINEAR_DIAGRAM_FIELDS
    }
    for qmax in PNG_TAIL_CANARY_QMAX:
        linear_values[qmax] = np.asarray(
            [
                values[("matter-linear", int(index), qmax)][
                    "value"
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        linear_errors[qmax] = np.asarray(
            [
                values[("matter-linear", int(index), qmax)][
                    "error_bound"
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        linear_tree_values[qmax] = np.asarray(
            [
                values[("matter-linear", int(index), qmax)][
                    "tree"
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        b112_values[qmax] = np.asarray(
            [
                values[("matter-b112", int(index), qmax)][
                    "value"
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        b112_errors[qmax] = np.asarray(
            [
                values[("matter-b112", int(index), qmax)][
                    "error_bound"
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        for field in MATTER_LINEAR_DIAGRAM_FIELDS:
            linear_component_values[field][qmax] = np.asarray(
                [
                    values[
                        ("matter-linear", int(index), qmax)
                    ]["components"][field]
                    for index in indices
                ],
                dtype=np.float64,
            )
            linear_component_errors[field][qmax] = np.asarray(
                [
                    values[
                        ("matter-linear", int(index), qmax)
                    ]["component_error_bounds"][field]
                    for index in indices
                ],
                dtype=np.float64,
            )

    tree_reference = linear_tree_values[
        PNG_TAIL_REFERENCE_QMAX
    ]
    tree_differences = {
        f"{qmax:g}_minus_20": (
            linear_tree_values[qmax] - tree_reference
        )
        for qmax in PNG_TAIL_CANARY_QMAX
    }
    tree_scale = max(
        float(np.max(np.abs(tree_reference))),
        np.finfo(float).tiny,
    )
    tree_max_abs = max(
        float(np.max(np.abs(value)))
        for value in tree_differences.values()
    )
    tree_invariance_pass = bool(
        tree_max_abs <= 2.0e-10 + 2.0e-13 * tree_scale
    )

    b1 = B_REC_H
    sigma_single = np.sqrt(np.diag(covariance_single))

    def metric_row(
        *,
        label: str,
        difference: np.ndarray,
        error_bound: np.ndarray,
        fnl: float,
    ) -> dict[str, Any]:
        difference = np.asarray(difference, dtype=np.float64)
        error_bound = np.asarray(error_bound, dtype=np.float64)
        ratios = np.divide(
            np.abs(difference),
            error_bound,
            out=np.zeros_like(difference),
            where=error_bound > 0.0,
        )
        if np.any((error_bound == 0.0) & (difference != 0.0)):
            ratios[(error_bound == 0.0) & (difference != 0.0)] = (
                np.finfo(float).max
            )
        return {
            "component": label,
            "fNL": fnl,
            "difference_by_global_bin": dict(
                zip(
                    (str(index) for index in indices),
                    difference,
                )
            ),
            "sum_endpoint_qmc_bounds_by_global_bin": dict(
                zip(
                    (str(index) for index in indices),
                    error_bound,
                )
            ),
            "absolute_difference_over_sum_qmc_bound_by_global_bin": (
                dict(
                    zip(
                        (str(index) for index in indices),
                        ratios,
                    )
                )
            ),
            "vector_norm_sigma_single": qnorm(
                difference,
                covariance_single,
            ),
            "max_abs_single_bin_sigma": float(
                np.max(np.abs(difference) / sigma_single)
            ),
        }

    transition_payload: dict[str, Any] = {}
    for qmax_low, qmax_high in (
        (10.0, PNG_TAIL_REFERENCE_QMAX),
        (PNG_TAIL_REFERENCE_QMAX, 40.0),
    ):
        label = f"qmax_{qmax_low:g}_to_{qmax_high:g}"
        rows: list[dict[str, Any]] = []
        diagram_rows: list[dict[str, Any]] = []
        for fnl in (-PNG_TAIL_FNL_ENVELOPE, PNG_TAIL_FNL_ENVELOPE):
            linear_difference = (
                fnl
                * b1**3
                * (
                    linear_values[qmax_high]
                    - linear_values[qmax_low]
                )
            )
            linear_error = (
                abs(fnl)
                * b1**3
                * (
                    linear_errors[qmax_high]
                    + linear_errors[qmax_low]
                )
            )
            b112_difference = (
                fnl**2
                * b1**3
                * (
                    b112_values[qmax_high]
                    - b112_values[qmax_low]
                )
            )
            b112_error = (
                fnl**2
                * b1**3
                * (
                    b112_errors[qmax_high]
                    + b112_errors[qmax_low]
                )
            )
            rows.extend(
                (
                    metric_row(
                        label="matter-linear one-loop uplift",
                        difference=linear_difference,
                        error_bound=linear_error,
                        fnl=fnl,
                    ),
                    metric_row(
                        label="matter-B112II",
                        difference=b112_difference,
                        error_bound=b112_error,
                        fnl=fnl,
                    ),
                    metric_row(
                        label="combined matter PNG correction",
                        difference=(
                            linear_difference + b112_difference
                        ),
                        error_bound=linear_error + b112_error,
                        fnl=fnl,
                    ),
                )
            )
            for field in MATTER_LINEAR_DIAGRAM_FIELDS:
                diagram_rows.append(
                    metric_row(
                        label=f"matter-linear {field}",
                        difference=(
                            fnl
                            * b1**3
                            * (
                                linear_component_values[field][
                                    qmax_high
                                ]
                                - linear_component_values[field][
                                    qmax_low
                                ]
                            )
                        ),
                        error_bound=(
                            abs(fnl)
                            * b1**3
                            * (
                                linear_component_errors[field][
                                    qmax_high
                                ]
                                + linear_component_errors[field][
                                    qmax_low
                                ]
                            )
                        ),
                        fnl=fnl,
                    )
                )
        transition_payload[label] = {
            "rows": rows,
            "matter_linear_diagram_diagnostics_not_gated": (
                diagram_rows
            ),
            "maximum_vector_norm_sigma_single": max(
                row["vector_norm_sigma_single"] for row in rows
            ),
            "maximum_abs_single_bin_sigma": max(
                row["max_abs_single_bin_sigma"] for row in rows
            ),
        }

    gated_rows = transition_payload[
        "qmax_20_to_40"
    ]["rows"]
    threshold_pass = bool(
        all(
            row["vector_norm_sigma_single"]
            < PNG_NUMERICAL_VECTOR_THRESHOLD_SIGMA
            and row["max_abs_single_bin_sigma"]
            < PNG_NUMERICAL_BIN_THRESHOLD_SIGMA
            for row in gated_rows
        )
    )
    source_files.sort(
        key=lambda row: (
            row["sector"],
            row["index"],
            row["qmax_h_mpc"],
        )
    )
    return {
        "pass": bool(tree_invariance_pass and threshold_pass),
        "semantics": (
            "qmax=20 production versus qmax=40 omitted-tail check at "
            "lambda=1 and fNL=+/-100; qmax=10 is diagnostic only"
        ),
        "single_realization_covariance": True,
        "covariance_divided_by_mock_count": False,
        "global_bin_indices": indices,
        "b1_corresponding_to_lambda_1": b1,
        "absolute_fNL_envelope": PNG_TAIL_FNL_ENVELOPE,
        "thresholds": {
            "vector_norm_sigma_single": (
                PNG_NUMERICAL_VECTOR_THRESHOLD_SIGMA
            ),
            "max_abs_single_bin_sigma": (
                PNG_NUMERICAL_BIN_THRESHOLD_SIGMA
            ),
            "gated_transition": "qmax_20_to_40",
        },
        "tree_qmax_invariance": {
            "maximum_absolute_difference": tree_max_abs,
            "maximum_relative_to_reference": tree_max_abs / tree_scale,
            "absolute_tolerance": 2.0e-10,
            "relative_tolerance": 2.0e-13,
            "pass": tree_invariance_pass,
        },
        "raw_integral_values": {
            "matter_linear_loop": {
                f"{qmax:g}": linear_values[qmax]
                for qmax in (10.0, 20.0, 40.0)
            },
            "matter_linear_diagrams": {
                field: {
                    f"{qmax:g}": linear_component_values[field][
                        qmax
                    ]
                    for qmax in (10.0, 20.0, 40.0)
                }
                for field in MATTER_LINEAR_DIAGRAM_FIELDS
            },
            "matter_b112_coefficient": {
                f"{qmax:g}": b112_values[qmax]
                for qmax in (10.0, 20.0, 40.0)
            },
        },
        "transitions": transition_payload,
        "source_files": source_files,
    }


def profile_finite_png(
    *,
    module: ModuleType,
    runner: ModuleType,
    gaussian_templates: PostTemplateSet,
    png_templates: PngTemplateSet,
    data: Any,
    frozen_prior: Any,
    covariance_samples: np.ndarray,
    target_samples: np.ndarray,
    nbar: float,
    tier: str,
    stochastic_mode: str,
    sample: str,
    truth_fnl: float,
    kmax: float,
    nuisance_prior_scale: float,
    multistart: int,
    rng: np.random.Generator,
    compute_curve: bool,
    png_stochastic_model: str = PRIMARY_PNG_STOCHASTIC_MODEL,
    precision_correction: str = "hartlap",
) -> FinitePngProfile:
    prior_contract = fnl_prior_contract()
    if hard_flat_fnl_log_prior(truth_fnl) != 0.0:
        raise ValueError(
            "registered sample truth lies outside the fixed fiducial "
            "fNL prior support"
        )
    mask = (
        np.max(np.asarray(data.k_pair), axis=1)
        <= kmax + 1.0e-12
    )
    mask[EXCLUDED_BIN] = False
    indices = np.flatnonzero(mask)
    expected = {0.08: 9, 0.10: 14, 0.12: 20, 0.14: 27}
    if indices.size != expected[kmax]:
        raise ValueError(
            f"kmax={kmax:.2f}: {indices.size} bins, "
            f"expected {expected[kmax]}"
        )
    missing = sorted(set(indices) - set(png_templates.indices))
    if missing:
        raise ValueError(
            "PNG templates miss selected bins: "
            + ",".join(map(str, missing))
        )
    covariance = runner.covariance_single(
        np.asarray(covariance_samples[:, indices], dtype=np.float64)
    )
    target = np.mean(
        np.asarray(target_samples[:, indices], dtype=np.float64),
        axis=0,
    )
    covariance_mock_count = int(covariance_samples.shape[0])
    if precision_correction not in {"hartlap", "none"}:
        raise ValueError(precision_correction)
    hartlap_factor = (
        (
            covariance_mock_count
            - indices.size
            - 2
        )
        / (covariance_mock_count - 1)
        if precision_correction == "hartlap"
        else 1.0
    )
    if not (0.0 < hartlap_factor <= 1.0):
        raise ValueError(
            "insufficient covariance realizations for the registered "
            "Hartlap precision correction"
        )
    chol_single = np.linalg.cholesky(covariance)
    chol_fit = chol_single / math.sqrt(hartlap_factor)
    precision = hartlap_factor * np.linalg.inv(covariance)
    if png_stochastic_model not in PNG_STOCHASTIC_MODELS:
        raise ValueError(png_stochastic_model)
    names = free_names(module, tier, stochastic_mode)
    if png_stochastic_model == "independent_eq265":
        names = names + (PNG_STOCHASTIC_NAME,)
    nonlinear = nonlinear_names(tier)
    prior = calibrated_prior_block(
        runner,
        module,
        frozen_prior,
        names,
        nonlinear,
        nuisance_prior_scale,
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    nonlinear_indices = np.asarray(
        [name_to_index[name] for name in nonlinear],
        dtype=int,
    )
    linear_indices = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name not in nonlinear
        ],
        dtype=int,
    )

    def evaluate_model(
        fnl: float,
        nuisance: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
        dict[str, float],
    ]:
        parameters = runner.full_parameter_dict(
            module,
            names,
            nuisance,
            tier=tier,
            fixed_b1=None,
        )
        gaussian = gaussian_templates.components(
            parameters,
            nbar,
            stochastic_mode,
        )
        png = png_templates.components(
            parameters,
            nbar,
            stochastic_mode,
            png_stochastic_model,
        )
        linear = np.asarray(
            png["linear_total"],
            dtype=np.float64,
        )
        quadratic = np.asarray(
            png["quadratic_total"],
            dtype=np.float64,
        )
        prediction = combine_finite_png_prediction(
            gaussian["total"],
            png,
            fnl,
        )
        return (
            prediction[indices],
            np.asarray(gaussian["total"])[indices],
            png,
            parameters,
        )

    def solve_linear(
        fnl: float,
        nonlinear_values: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        nuisance = prior.mean.copy()
        nuisance[nonlinear_indices] = nonlinear_values
        nuisance[linear_indices] = 0.0
        base = evaluate_model(fnl, nuisance)[0]
        design = np.empty(
            (target.size, linear_indices.size),
            dtype=np.float64,
        )
        for column, parameter_index in enumerate(linear_indices):
            unit = nuisance.copy()
            unit[parameter_index] = 1.0
            design[:, column] = (
                evaluate_model(fnl, unit)[0] - base
            )
        whitened_design = solve_triangular(
            chol_fit,
            design,
            lower=True,
        )
        whitened_target = solve_triangular(
            chol_fit,
            target - base,
            lower=True,
        )
        prior_design = prior.whitener[:, linear_indices]
        prior_target = -prior.whitener @ (
            nuisance - prior.mean
        )
        solution = np.linalg.lstsq(
            np.vstack((whitened_design, prior_design)),
            np.concatenate((whitened_target, prior_target)),
            rcond=1.0e-12,
        )[0]
        nuisance[linear_indices] = solution
        prediction = evaluate_model(fnl, nuisance)[0]
        data_residual = solve_triangular(
            chol_fit,
            prediction - target,
            lower=True,
        )
        prior_residual = prior.whitener @ (
            nuisance - prior.mean
        )
        objective = float(
            data_residual @ data_residual
            + prior_residual @ prior_residual
        )
        return nuisance, objective

    objective_names = ("fNL",) + nonlinear

    def objective(*values: float) -> float:
        if hard_flat_fnl_log_prior(float(values[0])) == -math.inf:
            return math.inf
        return solve_linear(
            float(values[0]),
            np.asarray(values[1:], dtype=np.float64),
        )[1]

    def make_minuit(start: np.ndarray) -> Minuit:
        minimizer = Minuit(
            objective,
            *[float(value) for value in start],
            name=objective_names,
        )
        minimizer.errordef = 1.0
        minimizer.tol = 1.0e-4
        minimizer.errors["fNL"] = 30.0
        minimizer.limits["fNL"] = tuple(
            prior_contract["bounds"]
        )
        for name, index in zip(nonlinear, nonlinear_indices):
            minimizer.errors[name] = max(
                0.05,
                0.2 * float(prior.sigma[index]),
            )
            if name == "b1":
                minimizer.limits[name] = (0.1, 6.0)
        minimizer.migrad(ncall=15000)
        if not minimizer.valid:
            minimizer.simplex(ncall=5000)
            minimizer.migrad(ncall=20000)
        minimizer.hesse()
        return minimizer

    # The injection label is reporting metadata only.  In particular it must
    # never seed the optimizer: doing so leaks the answer into the numerical
    # procedure and can hide disconnected minima.  The first starts are a
    # fixed symmetric grid shared by all samples; any remaining starts are
    # drawn around zero from the registered RNG.
    base_start = np.concatenate(
        (
            np.asarray([FIDUCIAL_FNL], dtype=np.float64),
            prior.mean[nonlinear_indices],
        )
    )
    minimizers: list[Minuit] = []
    for start_index in range(multistart):
        start = base_start.copy()
        if start_index < len(FNL_FIXED_STARTS):
            start[0] = FNL_FIXED_STARTS[start_index]
        else:
            lower, upper = FNL_PRIOR_BOUNDS
            start[0] = float(
                np.clip(
                    rng.normal(
                        FIDUCIAL_FNL,
                        FNL_RANDOM_START_SIGMA,
                    ),
                    lower + FNL_RANDOM_START_MARGIN,
                    upper - FNL_RANDOM_START_MARGIN,
                )
            )
        if start_index:
            start[1:] += rng.normal(
                0.0,
                0.25 * prior.sigma[nonlinear_indices],
            )
            b1_location = 1 + nonlinear.index("b1")
            start[b1_location] = np.clip(
                start[b1_location],
                0.2,
                5.8,
            )
        minimizers.append(make_minuit(start))
    valid = [item for item in minimizers if item.valid]
    candidates = valid if valid else minimizers
    minuit = min(candidates, key=lambda item: float(item.fval))
    multistart_objectives = np.asarray(
        [float(item.fval) for item in minimizers],
        dtype=np.float64,
    )
    multistart_fnl = np.asarray(
        [float(item.values["fNL"]) for item in minimizers],
        dtype=np.float64,
    )
    multistart_valid = np.asarray(
        [bool(item.valid) for item in minimizers],
        dtype=bool,
    )
    multistart_at_limit = np.asarray(
        [
            bool(item.fmin.has_parameters_at_limit)
            for item in minimizers
        ],
        dtype=bool,
    )
    finite_valid_objectives = multistart_objectives[
        multistart_valid & np.isfinite(multistart_objectives)
    ]
    multistart_objective_spread = (
        float(np.ptp(finite_valid_objectives))
        if finite_valid_objectives.size
        else math.inf
    )
    multistart_agree = bool(
        np.all(multistart_valid)
        and not np.any(multistart_at_limit)
        and finite_valid_objectives.size == multistart
        and multistart_objective_spread <= 1.0e-3
    )
    fhat = float(minuit.values["fNL"])
    minos_valid = False
    minos_lower_crossing_valid = False
    minos_upper_crossing_valid = False
    minos_at_lower_prior_limit = False
    minos_at_upper_prior_limit = False
    try:
        minuit.minos("fNL", cl=0.682689492137)
        merror = minuit.merrors["fNL"]
        error_low = float(merror.lower)
        error_high = float(merror.upper)
        minos_at_lower_prior_limit = bool(merror.at_lower_limit)
        minos_at_upper_prior_limit = bool(merror.at_upper_limit)
        minos_lower_crossing_valid = bool(
            merror.lower_valid
            and not minos_at_lower_prior_limit
            and not merror.at_lower_max_fcn
            and not merror.lower_new_min
            and math.isfinite(error_low)
            and error_low < 0.0
        )
        minos_upper_crossing_valid = bool(
            merror.upper_valid
            and not minos_at_upper_prior_limit
            and not merror.at_upper_max_fcn
            and not merror.upper_new_min
            and math.isfinite(error_high)
            and error_high > 0.0
        )
        minos_valid = bool(
            merror.is_valid
            and minos_lower_crossing_valid
            and minos_upper_crossing_valid
        )
    except Exception:
        symmetric = float(minuit.errors["fNL"])
        error_low = -symmetric
        error_high = symmetric
    # MINOS may internally rerun MIGRAD.  Always refresh the reported
    # minimum after it returns instead of combining a pre-MINOS centre with
    # post-MINOS nuisance values and interval endpoints.
    fhat = float(minuit.values["fNL"])
    sigma_symmetric = 0.5 * (
        abs(error_low) + abs(error_high)
    )
    nonlinear_best = np.asarray(
        [float(minuit.values[name]) for name in nonlinear],
        dtype=np.float64,
    )
    nuisance, objective_best = solve_linear(
        fhat,
        nonlinear_best,
    )
    objective_consistency = abs(
        objective_best - float(minuit.fval)
    )
    if objective_consistency > 1.0e-7 * (
        1.0 + abs(objective_best)
    ):
        raise RuntimeError(
            "conditional objective does not reproduce the selected "
            f"Minuit minimum: delta={objective_consistency:.6g}"
        )
    (
        prediction,
        gaussian,
        response,
        parameters,
    ) = evaluate_model(fhat, nuisance)
    residual = prediction - target
    whitened = solve_triangular(
        chol_single,
        residual,
        lower=True,
    )
    prior_residual = prior.whitener @ (nuisance - prior.mean)
    data_chi2 = float(
        hartlap_factor * (whitened @ whitened)
    )
    prior_chi2 = float(prior_residual @ prior_residual)
    affine_zero = nuisance.copy()
    affine_zero[linear_indices] = 0.0
    affine_base = evaluate_model(fhat, affine_zero)[0]
    affine_prediction = affine_base.copy()
    for parameter_index in linear_indices:
        unit = affine_zero.copy()
        unit[parameter_index] = 1.0
        affine_prediction += nuisance[parameter_index] * (
            evaluate_model(fhat, unit)[0] - affine_base
        )
    affine_error = qnorm(
        prediction - affine_prediction,
        covariance,
    )
    if affine_error > 1.0e-7:
        raise RuntimeError(
            "conditional linear solve is invalid for finite-PNG: "
            f"{affine_error:.6g} sigma_single"
        )

    nuisance_jacobian = np.empty(
        (target.size, len(names)),
        dtype=np.float64,
    )
    for column in range(len(names)):
        step = 1.0e-5 * max(
            abs(float(nuisance[column])),
            float(prior.sigma[column]),
            1.0e-3,
        )
        right = nuisance.copy()
        left = nuisance.copy()
        right[column] += step
        left[column] -= step
        nuisance_jacobian[:, column] = (
            evaluate_model(fhat, right)[0]
            - evaluate_model(fhat, left)[0]
        ) / (2.0 * step)
    fnl_step = max(1.0e-3, 1.0e-4 * max(abs(fhat), 1.0))
    fnl_tangent = (
        evaluate_model(fhat + fnl_step, nuisance)[0]
        - evaluate_model(fhat - fnl_step, nuisance)[0]
    ) / (2.0 * fnl_step)
    design_all = np.column_stack(
        (fnl_tangent, nuisance_jacobian)
    )
    prior_precision = np.zeros(
        (1 + len(names), 1 + len(names)),
        dtype=np.float64,
    )
    prior_precision[1:, 1:] = np.linalg.inv(prior.covariance)
    fisher = (
        design_all.T @ precision @ design_all
        + prior_precision
    )
    fisher_inverse = np.linalg.pinv(fisher, rcond=1.0e-11)
    effective = float(
        np.trace(
            design_all
            @ fisher_inverse
            @ design_all.T
            @ precision
        )
    )
    dof = max(float(target.size) - effective, 1.0e-6)
    p_value = float(chi2_distribution.sf(data_chi2, dof))

    if compute_curve:
        profile_scale = max(
            sigma_symmetric,
            abs(error_low),
            abs(error_high),
            1.0,
        )
        lo = max(
            FNL_PRIOR_BOUNDS[0],
            fhat - 3.2 * profile_scale,
        )
        hi = min(
            FNL_PRIOR_BOUNDS[1],
            fhat + 3.2 * profile_scale,
        )
        profile_x, profile_y, profile_ok = minuit.mnprofile(
            "fNL",
            size=33,
            bound=(lo, hi),
            subtract_min=True,
        )
        profile_x = np.asarray(profile_x, dtype=np.float64)
        profile_delta = np.asarray(profile_y, dtype=np.float64)
        profile_ok = np.asarray(profile_ok, dtype=bool)
        if (
            profile_x.shape != (33,)
            or profile_delta.shape != (33,)
            or profile_ok.shape != (33,)
            or not np.all(np.isfinite(profile_x))
            or not np.all(np.isfinite(profile_delta))
            or not np.all(profile_ok)
        ):
            raise RuntimeError(
                f"invalid Minuit profile for {tier}/{sample}/"
                f"{kmax:.2f}"
            )
        crossing_candidates: list[float] = []
        for index in range(profile_x.size - 1):
            left_delta = profile_delta[index] - 1.0
            right_delta = profile_delta[index + 1] - 1.0
            if left_delta * right_delta > 0.0:
                continue
            denominator = (
                profile_delta[index + 1]
                - profile_delta[index]
            )
            if abs(denominator) < 1.0e-14:
                crossing = 0.5 * (
                    profile_x[index] + profile_x[index + 1]
                )
            else:
                fraction = (
                    1.0 - profile_delta[index]
                ) / denominator
                crossing = profile_x[index] + fraction * (
                    profile_x[index + 1] - profile_x[index]
                )
            crossing_candidates.append(float(crossing))
        left_crossings = [
            value for value in crossing_candidates
            if value < fhat
        ]
        right_crossings = [
            value for value in crossing_candidates
            if value > fhat
        ]
        if left_crossings and right_crossings:
            left_crossing = max(left_crossings)
            right_crossing = min(right_crossings)
            profile_interval_consistency_sigma = max(
                abs(
                    left_crossing - (fhat + error_low)
                ),
                abs(
                    right_crossing - (fhat + error_high)
                ),
            ) / sigma_symmetric
        else:
            profile_interval_consistency_sigma = math.inf
        nearest = int(np.argmin(np.abs(profile_x - fhat)))
        profile_global_consistent = bool(
            np.min(profile_delta) >= -1.0e-6
            and profile_delta[nearest] <= 0.05
            and left_crossings
            and right_crossings
            and profile_interval_consistency_sigma < 0.15
        )
        profile_curve_kind = "actual_33_point_minuit_mnprofile"
    else:
        profile_x = np.empty(0, dtype=np.float64)
        profile_delta = np.empty(0, dtype=np.float64)
        profile_interval_consistency_sigma = math.nan
        profile_global_consistent = False
        profile_curve_kind = "not_computed_minos_interval_only"
    return FinitePngProfile(
        tier=tier,
        sample=sample,
        truth_fnl=float(truth_fnl),
        kmax=float(kmax),
        stochastic_mode=stochastic_mode,
        png_stochastic_model=png_stochastic_model,
        precision_correction=precision_correction,
        nuisance_prior_scale=float(nuisance_prior_scale),
        indices=indices,
        free_names=names,
        nonlinear_names=nonlinear,
        nuisance_values=nuisance,
        expanded_parameters=parameters,
        prediction=prediction,
        gaussian_prediction=gaussian,
        linear_response=np.asarray(
            response["linear_total"],
            dtype=np.float64,
        )[indices],
        quadratic_coefficient=np.asarray(
            response["quadratic_total"],
            dtype=np.float64,
        )[indices],
        target=target,
        covariance_single=covariance,
        covariance_mock_count=covariance_mock_count,
        hartlap_factor=float(hartlap_factor),
        fhat=fhat,
        error_low=error_low,
        error_high=error_high,
        sigma_symmetric=sigma_symmetric,
        objective=objective_best,
        data_chi2=data_chi2,
        prior_chi2=prior_chi2,
        dof=dof,
        p_value=p_value,
        max_pull_single=float(
            np.max(
                np.abs(residual)
                / np.sqrt(np.diag(covariance))
            )
        ),
        residual_norm_single=float(np.linalg.norm(whitened)),
        minuit_valid=bool(minuit.valid),
        minuit_accurate=bool(minuit.fmin.has_accurate_covar),
        minuit_at_limit=bool(
            minuit.fmin.has_parameters_at_limit
        ),
        minuit_nfcn=int(minuit.nfcn),
        profile_x=profile_x,
        profile_delta=profile_delta,
        profile_curve_kind=profile_curve_kind,
        nuisance_jacobian=nuisance_jacobian,
        multistart_objectives=multistart_objectives,
        multistart_fnl=multistart_fnl,
        multistart_valid=multistart_valid,
        multistart_at_limit=multistart_at_limit,
        multistart_objective_spread=multistart_objective_spread,
        multistart_agree=multistart_agree,
        minos_valid=minos_valid,
        minos_lower_crossing_valid=minos_lower_crossing_valid,
        minos_upper_crossing_valid=minos_upper_crossing_valid,
        minos_at_lower_prior_limit=minos_at_lower_prior_limit,
        minos_at_upper_prior_limit=minos_at_upper_prior_limit,
        profile_global_consistent=profile_global_consistent,
        profile_interval_consistency_sigma=(
            profile_interval_consistency_sigma
        ),
        affine_reconstruction_error_sigma_single=affine_error,
        response_components={
            name: np.asarray(value).copy()
            for name, value in response.items()
            if np.asarray(value).ndim == 1
            and np.asarray(value).shape == (120,)
        },
    )


def finite_profile_row(
    result: FinitePngProfile,
) -> dict[str, Any]:
    interval = [
        result.fhat + result.error_low,
        result.fhat + result.error_high,
    ]
    lower_bound, upper_bound = FNL_PRIOR_BOUNDS
    boundary_tolerance = (
        1.0e-7 * (1.0 + upper_bound - lower_bound)
    )
    interval_at_lower_prior = bool(
        interval[0] <= lower_bound + boundary_tolerance
    )
    interval_at_upper_prior = bool(
        interval[1] >= upper_bound - boundary_tolerance
    )
    lower_crossing_found = bool(
        result.minos_lower_crossing_valid
        and not interval_at_lower_prior
    )
    upper_crossing_found = bool(
        result.minos_upper_crossing_valid
        and not interval_at_upper_prior
    )
    interval_is_data_two_sided = bool(
        lower_crossing_found
        and upper_crossing_found
        and result.minos_valid
    )
    interval_kind = (
        "prior_truncated"
        if interval_at_lower_prior or interval_at_upper_prior
        else (
            "data_two_sided"
            if interval_is_data_two_sided
            else "numerically_unresolved"
        )
    )
    multistart_objective_agree = bool(
        np.all(result.multistart_valid)
        and result.multistart_objectives.size > 0
        and np.all(np.isfinite(result.multistart_objectives))
        and result.multistart_objective_spread <= 1.0e-3
    )
    fnl_best_fit_at_prior_boundary = bool(
        result.fhat <= lower_bound + boundary_tolerance
        or result.fhat >= upper_bound - boundary_tolerance
    )
    return {
        "tier": result.tier,
        "sample": result.sample,
        "truth_fnl": result.truth_fnl,
        "kmax_h_mpc": result.kmax,
        "stochastic_mode": result.stochastic_mode,
        "png_stochastic_model": result.png_stochastic_model,
        "likelihood": {
            "family": "Gaussian",
            "precision_correction": result.precision_correction,
            "covariance_mock_count": result.covariance_mock_count,
            "hartlap_factor": result.hartlap_factor,
            "single_realization_covariance": True,
            "divided_by_mock_count": False,
            "fNL_prior": fnl_prior_contract(),
        },
        "nuisance_prior_scale": result.nuisance_prior_scale,
        "n_data": int(result.indices.size),
        "selected_indices": result.indices,
        "fhat": result.fhat,
        "error_low": result.error_low,
        "error_high": result.error_high,
        "sigma_symmetric": result.sigma_symmetric,
        "support_limited_error_semantics": (
            "MINOS offsets clipped by hard-prior support when the "
            "corresponding crossing_found field is false"
        ),
        "data_error_low": (
            result.error_low if lower_crossing_found else None
        ),
        "data_error_high": (
            result.error_high if upper_crossing_found else None
        ),
        "data_sigma_symmetric": (
            result.sigma_symmetric
            if interval_is_data_two_sided
            else None
        ),
        "interval68": interval,
        "interval68_kind": interval_kind,
        "interval68_lower_crossing_found": lower_crossing_found,
        "interval68_upper_crossing_found": upper_crossing_found,
        "interval68_lower_prior_limited": interval_at_lower_prior,
        "interval68_upper_prior_limited": interval_at_upper_prior,
        "interval68_at_lower_prior_boundary": (
            interval_at_lower_prior
        ),
        "interval68_at_upper_prior_boundary": (
            interval_at_upper_prior
        ),
        "interval68_prior_truncated": bool(
            interval_at_lower_prior or interval_at_upper_prior
        ),
        "interval68_is_data_two_sided": bool(
            interval_is_data_two_sided
        ),
        "truth_in_interval68": (
            interval[0] <= result.truth_fnl <= interval[1]
        ),
        "absolute_bias_over_sigma": (
            abs(result.fhat - result.truth_fnl)
            / result.sigma_symmetric
            if interval_is_data_two_sided
            else None
        ),
        "b1": result.expanded_parameters["b1"],
        "objective": result.objective,
        "data_chi2": result.data_chi2,
        "prior_chi2": result.prior_chi2,
        "dof_effective": result.dof,
        "gof_p_single_covariance": result.p_value,
        "residual_norm_sigma_single": result.residual_norm_single,
        "max_abs_pull_sigma_single": result.max_pull_single,
        "minuit": {
            "valid": result.minuit_valid,
            "accurate_covariance": result.minuit_accurate,
            "minos_valid": result.minos_valid,
            "at_limit": result.minuit_at_limit,
            "has_any_parameter_at_limit": result.minuit_at_limit,
            "fNL_best_fit_at_prior_boundary": (
                fnl_best_fit_at_prior_boundary
            ),
            "minos_lower_crossing_valid": (
                result.minos_lower_crossing_valid
            ),
            "minos_upper_crossing_valid": (
                result.minos_upper_crossing_valid
            ),
            "minos_at_lower_prior_limit": (
                result.minos_at_lower_prior_limit
            ),
            "minos_at_upper_prior_limit": (
                result.minos_at_upper_prior_limit
            ),
            "nfcn": result.minuit_nfcn,
            "multistart_objectives": result.multistart_objectives,
            "multistart_fnl": result.multistart_fnl,
            "multistart_valid": result.multistart_valid,
            "multistart_at_limit": result.multistart_at_limit,
            "multistart_objective_spread": (
                result.multistart_objective_spread
            ),
            "multistart_agree_delta_chi2_1e-3": (
                result.multistart_agree
            ),
            "multistart_objective_agree_delta_chi2_1e-3": (
                multistart_objective_agree
            ),
            "multistart_formal_gate_pass": (
                result.multistart_agree
            ),
            "profile_global_consistent": (
                result.profile_global_consistent
            ),
            "profile_curve_kind": result.profile_curve_kind,
            "profile_interval_consistency_sigma": (
                result.profile_interval_consistency_sigma
            ),
            "affine_reconstruction_error_sigma_single": (
                result.affine_reconstruction_error_sigma_single
            ),
        },
        "parameters": result.expanded_parameters,
    }


def actual_profile_coverage(
    *,
    module: ModuleType,
    runner: ModuleType,
    gaussian_templates: PostTemplateSet,
    png_templates: PngTemplateSet,
    data: Any,
    frozen_prior: Any,
    covariance_samples: np.ndarray,
    covariance_realization_ids: np.ndarray,
    target_samples: np.ndarray,
    target_realization_ids: np.ndarray,
    nbar: float,
    sample: str,
    truth_fnl: float,
    realization_count: int,
    multistart: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reprofile covariance-pool-held-out realizations with Minuit/MINOS.

    Coverage is deliberately evaluated only at the preregistered largest
    cutoff.  The covariance pool is disjoint from every tested realization,
    and is always interpreted as a single-realization covariance.  The word
    ``held-out`` does not claim that these realizations were absent from the
    separate 500-realization central-vector diagnostics.
    """

    target_samples = np.asarray(target_samples, dtype=np.float64)
    target_realization_ids = np.asarray(
        target_realization_ids,
        dtype=np.int64,
    )
    covariance_samples = np.asarray(
        covariance_samples,
        dtype=np.float64,
    )
    covariance_realization_ids = np.asarray(
        covariance_realization_ids,
        dtype=np.int64,
    )
    if target_samples.shape[0] != target_realization_ids.size:
        raise ValueError("coverage target IDs do not match target samples")
    if (
        covariance_samples.shape[0]
        != covariance_realization_ids.size
    ):
        raise ValueError(
            "coverage covariance IDs do not match covariance samples"
        )
    if np.intersect1d(
        target_realization_ids[:realization_count],
        covariance_realization_ids,
    ).size:
        raise ValueError(
            "coverage targets overlap the independent covariance pool"
        )
    if realization_count < 1:
        raise ValueError("coverage realization count must be positive")
    if realization_count > target_samples.shape[0]:
        raise ValueError(
            f"requested {realization_count} coverage realizations from "
            f"only {target_samples.shape[0]} {sample} samples"
        )

    fhat = np.empty(realization_count, dtype=np.float64)
    error_low = np.empty(realization_count, dtype=np.float64)
    error_high = np.empty(realization_count, dtype=np.float64)
    objective = np.empty(realization_count, dtype=np.float64)
    residual_norm = np.empty(realization_count, dtype=np.float64)
    valid = np.empty(realization_count, dtype=bool)
    accurate = np.empty(realization_count, dtype=bool)
    minos_valid = np.empty(realization_count, dtype=bool)
    at_limit = np.empty(realization_count, dtype=bool)
    multistart_agree = np.empty(realization_count, dtype=bool)
    for realization in range(realization_count):
        fit = profile_finite_png(
            module=module,
            runner=runner,
            gaussian_templates=gaussian_templates,
            png_templates=png_templates,
            data=data,
            frozen_prior=frozen_prior,
            covariance_samples=covariance_samples,
            target_samples=target_samples[
                realization : realization + 1
            ],
            nbar=nbar,
            tier=PRIMARY_TIER,
            stochastic_mode="tied",
            sample=(
                f"{sample}_coverage_r"
                f"{int(target_realization_ids[realization])}"
            ),
            truth_fnl=truth_fnl,
            kmax=max(CUTS),
            nuisance_prior_scale=1.0,
            multistart=multistart,
            rng=rng,
            compute_curve=False,
        )
        fhat[realization] = fit.fhat
        error_low[realization] = fit.error_low
        error_high[realization] = fit.error_high
        objective[realization] = fit.objective
        residual_norm[realization] = fit.residual_norm_single
        valid[realization] = fit.minuit_valid
        accurate[realization] = fit.minuit_accurate
        minos_valid[realization] = fit.minos_valid
        at_limit[realization] = fit.minuit_at_limit
        multistart_agree[realization] = fit.multistart_agree
        if (
            (realization + 1) % 10 == 0
            or realization + 1 == realization_count
        ):
            print(
                "coverage "
                f"{sample}: {realization + 1}/{realization_count}",
                flush=True,
            )

    sigma = 0.5 * (np.abs(error_low) + np.abs(error_high))
    pulls = (fhat - truth_fnl) / sigma
    covered = (
        (fhat + error_low <= truth_fnl)
        & (truth_fnl <= fhat + error_high)
        & valid
        & accurate
        & minos_valid
        & ~at_limit
        & multistart_agree
    )
    count = int(np.count_nonzero(covered))
    alpha = 0.05
    lower = (
        0.0
        if count == 0
        else float(
            beta_distribution.ppf(
                alpha / 2.0,
                count,
                realization_count - count + 1,
            )
        )
    )
    upper = (
        1.0
        if count == realization_count
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0,
                count + 1,
                realization_count - count,
            )
        )
    )
    all_numerically_valid = bool(
        np.all(valid)
        and np.all(accurate)
        and np.all(minos_valid)
        and not np.any(at_limit)
        and np.all(multistart_agree)
    )
    return {
        "method": (
            "actual independent per-realization coevolution Minuit "
            "reprofiling with MINOS 68% intervals"
        ),
        "truth_blind_optimizer_starts": True,
        "truth_fnl": truth_fnl,
        "kmax_h_mpc": max(CUTS),
        "n_realizations": realization_count,
        "realization_selection": (
            "first N registered target IDs, preregistered before fits"
        ),
        "covariance": (
            "single-realization covariance from a disjoint fiducial "
            "realization pool; no division by mock count"
        ),
        "covariance_realization_count": int(
            covariance_samples.shape[0]
        ),
        "target_covariance_id_overlap_count": 0,
        "multistart_per_realization": multistart,
        "all_profiles_numerically_valid": all_numerically_valid,
        "mean_fhat": float(np.mean(fhat)),
        "bias": float(np.mean(fhat) - truth_fnl),
        "mean_sigma": float(np.mean(sigma)),
        "pull_mean": float(np.mean(pulls)),
        "pull_std_ddof1": float(np.std(pulls, ddof=1)),
        "coverage68": float(np.mean(covered)),
        "covered_count": count,
        "binomial95_interval": [lower, upper],
        "nominal_0p68_inside_binomial95": (
            lower <= 0.68 <= upper
        ),
    }, {
        "realization_id": target_realization_ids[:realization_count],
        "fhat": fhat,
        "error_low": error_low,
        "error_high": error_high,
        "sigma": sigma,
        "pull": pulls,
        "covered": covered,
        "objective": objective,
        "residual_norm_sigma_single": residual_norm,
        "minuit_valid": valid,
        "minuit_accurate": accurate,
        "minos_valid": minos_valid,
        "minuit_at_limit": at_limit,
        "multistart_agree": multistart_agree,
    }


def plot_gaussian_diagonal(
    path: Path,
    fit: Any,
    k_pair: np.ndarray,
    edges: np.ndarray,
) -> None:
    selected = np.asarray(fit.selected_indices, dtype=int)
    diagonal = selected[
        np.all(
            np.isclose(
                edges[selected, 0, :],
                edges[selected, 1, :],
                rtol=0.0,
                atol=2.0e-14,
            ),
            axis=1,
        )
    ]
    location = {
        int(global_index): local_index
        for local_index, global_index in enumerate(selected)
    }
    local = np.asarray(
        [location[int(index)] for index in diagonal],
        dtype=int,
    )
    k = np.mean(k_pair[diagonal], axis=1)
    target = fit.target[local]
    prediction = fit.prediction[local]
    sigma = np.sqrt(np.diag(fit.covariance_single))[local]
    figure, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.25]},
        constrained_layout=True,
    )
    axis.errorbar(
        k,
        k * k * target,
        yerr=k * k * sigma,
        color="black",
        marker="o",
        linestyle="none",
        capsize=2.5,
        label="Quijote mean; one-box error",
    )
    axis.plot(
        k,
        k * k * prediction,
        color="#c51b1d",
        linewidth=2.0,
        label="post-recon coevolution EFT-v2",
    )
    residual_axis.axhline(0.0, color="0.55", linewidth=0.8)
    residual_axis.plot(
        k,
        (prediction - target) / sigma,
        color="#c51b1d",
        marker="o",
        linewidth=1.4,
    )
    residual_axis.set_ylim(-0.5, 0.5)
    axis.set_ylabel(r"$k^2 B_{000}$")
    residual_axis.set_ylabel(r"$(B_{\rm th}-B_{\rm data})/\sigma_{\rm single}$")
    residual_axis.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    axis.legend(frameon=False)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Post-reconstruction Gaussian diagonal k2 B000",
        },
    )
    plt.close(figure)


def plot_full2d_bestfits(
    path: Path,
    profiles: list[FinitePngProfile],
    k_pair: np.ndarray,
) -> None:
    with PdfPages(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Post-reconstruction finite-PNG full-2D best fits",
        },
    ) as pdf:
        for kmax in CUTS:
            rows = [
                next(
                    result
                    for result in profiles
                    if result.tier == PRIMARY_TIER
                    and result.sample == sample
                    and math.isclose(result.kmax, kmax)
                )
                for sample, _truth, _key, _nbar in SAMPLES
            ]
            n_samples = len(SAMPLES)
            figure = plt.figure(
                figsize=(6.6 * n_samples, 6.4)
            )
            grid = figure.add_gridspec(
                2,
                n_samples,
                height_ratios=(3.0, 1.0),
                left=0.055,
                right=0.99,
                bottom=0.10,
                top=0.88,
                wspace=0.25,
                hspace=0.08,
            )
            figure.suptitle(
                rf"post reconstruction, $k_{{\max}}={kmax:.2f}\,"
                rf"h\,{{\rm Mpc}}^{{-1}}$; "
                f"{rows[0].indices.size} full-2D bins",
                fontsize=12.0,
            )
            for column, fit in enumerate(rows):
                upper = figure.add_subplot(grid[0, column])
                lower = figure.add_subplot(
                    grid[1, column],
                    sharex=upper,
                )
                x = np.arange(fit.indices.size)
                weight = np.prod(k_pair[fit.indices], axis=1)
                sigma = np.sqrt(
                    np.diag(fit.covariance_single)
                )
                upper.errorbar(
                    x,
                    weight * fit.target,
                    yerr=weight * sigma,
                    fmt="o",
                    color="black",
                    ecolor="0.55",
                    capsize=2.0,
                    markersize=3.5,
                    label="Quijote mean; one-box error",
                )
                upper.plot(
                    x,
                    weight * fit.prediction,
                    "-o",
                    color="#c51b1d",
                    linewidth=1.7,
                    markersize=2.7,
                    label="profile best fit",
                )
                upper.set_title(
                    rf"{fit.sample}: $\widehat f_{{\rm NL}}="
                    rf"{fit.fhat:.1f}^{{+{fit.error_high:.1f}}}"
                    rf"_{{{fit.error_low:.1f}}}$"
                )
                upper.set_ylabel(
                    r"$k_1k_2 B_{000}(k_1,k_2)$"
                )
                upper.legend(frameon=False, fontsize=8.0)
                pull = (fit.prediction - fit.target) / sigma
                lower.axhline(0.0, color="0.5", linewidth=0.8)
                lower.plot(
                    x,
                    pull,
                    "o-",
                    color="#c51b1d",
                    linewidth=1.3,
                    markersize=3.0,
                )
                lower.set_ylim(-0.5, 0.5)
                lower.set_xlabel("selected full-2D bin")
                lower.set_ylabel(
                    r"$(B_{\rm th}-B_{\rm data})/"
                    r"\sigma_{\rm single}$"
                )
                for axis in (upper, lower):
                    axis.tick_params(
                        direction="in",
                        top=True,
                        right=True,
                    )
                    axis.grid(alpha=0.18)
            pdf.savefig(figure)
            plt.close(figure)


def plot_fnl_profiles(
    path: Path,
    profiles: list[FinitePngProfile],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(SAMPLES),
        figsize=(6.4 * len(SAMPLES), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CUTS)))
    for axis, (sample, truth, _key, _nbar) in zip(
        axes[0], SAMPLES
    ):
        for color, kmax in zip(colors, CUTS):
            fit = next(
                result
                for result in profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
                and math.isclose(result.kmax, kmax)
            )
            axis.plot(
                fit.profile_x,
                fit.profile_delta,
                color=color,
                linewidth=1.5,
                label=rf"$k_{{\max}}={kmax:.2f}$",
            )
        axis.axhline(
            1.0,
            color="0.5",
            linestyle="--",
            linewidth=0.8,
        )
        axis.axvline(
            truth,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_ylim(0.0, 5.0)
        axis.set_xlim(*FNL_PRIOR_BOUNDS)
        axis.set_xlabel(r"$f_{\rm NL}$")
        axis.set_ylabel(r"$\Delta\chi^2_{\rm prof}$")
        axis.set_title(
            rf"{sample}; hard flat prior "
            rf"$[{FNL_PRIOR_BOUNDS[0]:.0f},"
            rf"{FNL_PRIOR_BOUNDS[1]:.0f}]$"
        )
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=7.8)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Post-reconstruction actual Minuit profiles",
        },
    )
    plt.close(figure)


def _plot_profile_interval_series(
    *,
    axis: Any,
    x_values: np.ndarray,
    rows: list[FinitePngProfile],
    color: str,
    label: str,
) -> None:
    """Draw profile intervals without disguising prior limits as crossings."""

    x_values = np.asarray(x_values, dtype=np.float64)
    if x_values.shape != (len(rows),):
        raise ValueError("profile interval x/row dimensions differ")
    axis.plot(
        x_values,
        [row.fhat for row in rows],
        color=color,
        marker="o",
        linewidth=1.5,
        label=label,
        zorder=3,
    )
    lower_bound, upper_bound = FNL_PRIOR_BOUNDS
    boundary_tolerance = (
        1.0e-7 * (1.0 + upper_bound - lower_bound)
    )
    cap_half_width = 7.0e-4
    existing_labels = set(axis.get_legend_handles_labels()[1])
    prior_marker_label_used = (
        "open triangle: prior-limited" in existing_labels
    )
    unresolved_marker_label_used = (
        "x: unresolved crossing" in existing_labels
    )
    for x_value, row in zip(x_values, rows):
        lower = max(lower_bound, row.fhat + row.error_low)
        upper = min(upper_bound, row.fhat + row.error_high)
        lower_prior_limited = bool(
            lower <= lower_bound + boundary_tolerance
        )
        upper_prior_limited = bool(
            upper >= upper_bound - boundary_tolerance
        )
        lower_crossing = bool(
            row.minos_lower_crossing_valid
            and not lower_prior_limited
        )
        upper_crossing = bool(
            row.minos_upper_crossing_valid
            and not upper_prior_limited
        )
        axis.vlines(
            x_value,
            lower,
            upper,
            color=color,
            linewidth=1.35,
            zorder=2,
        )
        if lower_crossing:
            axis.hlines(
                lower,
                x_value - cap_half_width,
                x_value + cap_half_width,
                color=color,
                linewidth=1.2,
            )
        else:
            marker = "v" if lower_prior_limited else "x"
            marker_label = (
                "open triangle: prior-limited"
                if lower_prior_limited
                and not prior_marker_label_used
                else (
                    "x: unresolved crossing"
                    if (
                        not lower_prior_limited
                        and not unresolved_marker_label_used
                    )
                    else None
                )
            )
            axis.scatter(
                [x_value],
                [lower],
                marker=marker,
                s=28.0,
                facecolors=(
                    "none" if lower_prior_limited else color
                ),
                edgecolors=color,
                linewidths=1.1,
                label=marker_label,
                zorder=4,
            )
            prior_marker_label_used |= lower_prior_limited
            unresolved_marker_label_used |= not lower_prior_limited
        if upper_crossing:
            axis.hlines(
                upper,
                x_value - cap_half_width,
                x_value + cap_half_width,
                color=color,
                linewidth=1.2,
            )
        else:
            marker = "^" if upper_prior_limited else "x"
            marker_label = (
                "open triangle: prior-limited"
                if upper_prior_limited
                and not prior_marker_label_used
                else (
                    "x: unresolved crossing"
                    if (
                        not upper_prior_limited
                        and not unresolved_marker_label_used
                    )
                    else None
                )
            )
            axis.scatter(
                [x_value],
                [upper],
                marker=marker,
                s=28.0,
                facecolors=(
                    "none" if upper_prior_limited else color
                ),
                edgecolors=color,
                linewidths=1.1,
                label=marker_label,
                zorder=4,
            )
            prior_marker_label_used |= upper_prior_limited
            unresolved_marker_label_used |= not upper_prior_limited


def plot_fnl_recovery(
    path: Path,
    profiles: list[FinitePngProfile],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(SAMPLES),
        figsize=(6.4 * len(SAMPLES), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (sample, truth, _key, _nbar) in zip(
        axes[0], SAMPLES
    ):
        rows = [
            next(
                result
                for result in profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
                and math.isclose(result.kmax, kmax)
            )
            for kmax in CUTS
        ]
        _plot_profile_interval_series(
            axis=axis,
            x_values=np.asarray(CUTS, dtype=np.float64),
            rows=rows,
            color="#2166ac",
            label="coevolution",
        )
        axis.axhline(
            truth,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.axhline(
            FNL_PRIOR_BOUNDS[0],
            color="0.75",
            linestyle="--",
            linewidth=0.7,
        )
        axis.axhline(
            FNL_PRIOR_BOUNDS[1],
            color="0.75",
            linestyle="--",
            linewidth=0.7,
        )
        axis.set_xlabel(
            r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$"
        )
        axis.set_ylabel(r"$f_{\rm NL}$")
        axis.set_title(
            rf"{sample}; hard flat prior "
            rf"$[{FNL_PRIOR_BOUNDS[0]:.0f},"
            rf"{FNL_PRIOR_BOUNDS[1]:.0f}]$"
        )
        axis.set_xticks(CUTS)
        axis.set_ylim(
            FNL_PRIOR_BOUNDS[0] - 12.0,
            FNL_PRIOR_BOUNDS[1] + 12.0,
        )
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=7.8)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": (
                "Post-reconstruction fiducial-halo null recovery "
                "versus kmax"
            ),
        },
    )
    plt.close(figure)


def plot_full_vs_coevolution(
    path: Path,
    profiles: list[FinitePngProfile],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(SAMPLES),
        figsize=(6.4 * len(SAMPLES), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (sample, truth, _key, _nbar) in zip(
        axes[0], SAMPLES
    ):
        for tier, color, offset in (
            ("coevolution", "#2166ac", -0.0012),
            ("full", "#b2182b", 0.0012),
        ):
            rows = [
                next(
                    result
                    for result in profiles
                    if result.tier == tier
                    and result.sample == sample
                    and math.isclose(result.kmax, kmax)
                )
                for kmax in CUTS
            ]
            _plot_profile_interval_series(
                axis=axis,
                x_values=np.asarray(CUTS) + offset,
                rows=rows,
                color=color,
                label=tier,
            )
        axis.axhline(
            truth,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.axhline(
            FNL_PRIOR_BOUNDS[0],
            color="0.75",
            linestyle="--",
            linewidth=0.7,
        )
        axis.axhline(
            FNL_PRIOR_BOUNDS[1],
            color="0.75",
            linestyle="--",
            linewidth=0.7,
        )
        axis.set_xticks(CUTS)
        axis.set_xlabel(
            r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$"
        )
        axis.set_ylabel(r"$f_{\rm NL}$")
        axis.set_title(
            rf"{sample}; hard flat prior "
            rf"$[{FNL_PRIOR_BOUNDS[0]:.0f},"
            rf"{FNL_PRIOR_BOUNDS[1]:.0f}]$"
        )
        axis.set_ylim(
            FNL_PRIOR_BOUNDS[0] - 12.0,
            FNL_PRIOR_BOUNDS[1] + 12.0,
        )
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=8.0)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Full versus coevolution finite-PNG profiles",
        },
    )
    plt.close(figure)


def plot_pre_vs_post(
    path: Path,
    profiles: list[FinitePngProfile],
    pre_summary_path: Path,
) -> dict[str, Any]:
    pre_summary = json.loads(
        pre_summary_path.read_text(encoding="utf-8")
    )
    pre_covariance_contract = pre_summary.get(
        "covariance_contract"
    )
    if (
        pre_summary.get("status") != "pass"
        or not isinstance(pre_covariance_contract, dict)
        or pre_covariance_contract.get(
            "division_by_number_of_realizations"
        )
        is not False
        or pre_covariance_contract.get("errorbars")
        != "sqrt(diag(C_single))"
        or pre_covariance_contract.get("likelihood")
        != "fiducial500 single-realization covariance"
        or pre_covariance_contract.get(
            "means_used_only_as_central_vectors"
        )
        is not True
    ):
        raise ValueError(
            "pre-reconstruction comparison input does not prove the "
            "registered single-realization covariance contract"
        )
    pre_rows = [
        {
            **row,
            "canonical_plot_kmax_h_mpc": (
                0.14
                if int(row.get("n_data", -1)) == 27
                and math.isclose(
                    float(row["kmax_h_mpc"]),
                    0.15,
                )
                else float(row["kmax_h_mpc"])
            ),
        }
        for row in pre_summary["profiles"]
        if row["response_model"] == "finite_halo_tree"
        and row["tier"] == "coevolution"
        and math.isclose(float(row["p"]), 1.0)
        and math.isclose(
            float(row["nuisance_prior_scale"]),
            1.0,
        )
    ]
    figure, axes = plt.subplots(
        1,
        len(SAMPLES),
        figsize=(6.4 * len(SAMPLES), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (sample, truth, _key, _nbar) in zip(
        axes[0], SAMPLES
    ):
        before = sorted(
            [
                row for row in pre_rows
                if row["sample"] == sample
            ],
            key=lambda row: float(
                row["canonical_plot_kmax_h_mpc"]
            ),
        )
        after = sorted(
            [
                result for result in profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
            ],
            key=lambda result: result.kmax,
        )
        axis.errorbar(
            [
                row["canonical_plot_kmax_h_mpc"]
                for row in before
            ],
            [row["fhat"] for row in before],
            yerr=[
                [abs(row["error_low"]) for row in before],
                [abs(row["error_high"]) for row in before],
            ],
            color="#b2182b",
            marker="s",
            capsize=2.5,
            label="pre reconstruction",
        )
        axis.errorbar(
            [row.kmax for row in after],
            [row.fhat for row in after],
            yerr=[
                [abs(row.error_low) for row in after],
                [abs(row.error_high) for row in after],
            ],
            color="#2166ac",
            marker="o",
            capsize=2.5,
            label="post reconstruction",
        )
        axis.axhline(
            truth,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_xlabel(
            r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$"
        )
        axis.set_ylabel(r"$f_{\rm NL}$")
        axis.set_title(sample)
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=8.0)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Pre- versus post-reconstruction fNL constraints",
        },
    )
    plt.close(figure)
    return {
        "input": str(pre_summary_path),
        "sha256": sha256(pre_summary_path),
        "selection": (
            "finite_halo_tree, coevolution, p=1, "
            "nuisance_prior_scale=1"
        ),
        "covariance_contract": pre_covariance_contract,
        "canonical_cut_remap": (
            "historical pre kmax=0.15 rows with n_data=27 exclude "
            "the 0.155 shell and are plotted at canonical kmax=0.14"
        ),
    }


def plot_component_audit(
    path: Path,
    profiles: list[FinitePngProfile],
) -> None:
    fit = next(
        result
        for result in profiles
        if result.tier == PRIMARY_TIER
        and result.sample == "fiducial"
        and math.isclose(result.kmax, max(CUTS))
    )
    reference_fnl = 100.0
    linear_names = [
        name
        for name in (
            "halo_primordial",
            "halo_bphi_f2",
            "halo_bphi_advection",
            "halo_bphidelta",
            "halo_bphi_b2",
            "halo_bphi_bK2",
            "halo_reconstruction",
            "residual_stochastic_linear",
            "fixed_poisson_linear",
            "matter_one_loop_uplift",
        )
        if name in fit.response_components
    ]
    quadratic_names = [
        name
        for name in (
            "halo_bphi_B0",
            "halo_bphi_sq_advection",
            "halo_bphi_sq_F2",
            "halo_bphi_sq_b2",
            "halo_bphi_sq_bK2",
            "halo_bphi_sq_reconstruction",
            "halo_bphi_bphidelta",
            "halo_bphi2_operator",
            "residual_stochastic_quadratic",
            "fixed_poisson_quadratic",
            "matter_B112II",
        )
        if name in fit.response_components
    ]
    labels = linear_names + quadratic_names
    norms = []
    for name in labels:
        coefficient = fit.response_components[name][fit.indices]
        amplitude = (
            reference_fnl
            if name in linear_names
            else reference_fnl**2
        )
        norms.append(
            qnorm(
                amplitude * coefficient,
                fit.covariance_single,
            )
        )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 8.0),
        constrained_layout=True,
    )
    colors = [
        "#2166ac" if name in linear_names else "#b2182b"
        for name in labels
    ]
    axes[0].bar(
        np.arange(len(labels)),
        norms,
        color=colors,
    )
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(
        labels,
        rotation=55,
        ha="right",
        fontsize=7.3,
    )
    axes[0].set_ylabel(
        r"component norm at best fit [$\sigma_{\rm single}$]"
    )
    axes[0].set_title(
        r"fiducial nuisance point, theory-only $|f_{\rm NL}|=100$: "
        r"linear (blue), finite-$f_{\rm NL}^2$ (red)"
    )
    sigma = np.sqrt(np.diag(fit.covariance_single))
    x = np.arange(fit.indices.size)
    axes[1].plot(
        x,
        reference_fnl
        * fit.response_components["linear_total"][fit.indices]
        / sigma,
        color="#2166ac",
        linewidth=1.7,
        label=r"$f_{\rm NL}R_1$",
    )
    axes[1].plot(
        x,
        reference_fnl**2
        * fit.response_components["quadratic_total"][fit.indices]
        / sigma,
        color="#b2182b",
        linewidth=1.7,
        label=r"$f_{\rm NL}^2R_2$",
    )
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_xlabel("selected full-2D bin")
    axes[1].set_ylabel(r"signal / $\sigma_{\rm single,bin}$")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.16)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": (
                "Post-reconstruction theory-only model component "
                "audit at reference |fNL|=100"
            ),
        },
    )
    plt.close(figure)


def save_profiles_npz(
    path: Path,
    profiles: list[FinitePngProfile],
    coverage_arrays: dict[str, dict[str, np.ndarray]],
    k_pair: np.ndarray,
) -> None:
    ordered = sorted(
        profiles,
        key=lambda row: (
            row.tier,
            row.sample,
            row.kmax,
            row.nuisance_prior_scale,
            row.stochastic_mode,
            row.png_stochastic_model,
            row.precision_correction,
        ),
    )
    maximum_bins = max(row.indices.size for row in ordered)
    identifiers = np.asarray(
        [
            f"{row.tier}/{row.sample}/{row.kmax:.2f}/"
            f"{row.stochastic_mode}/{row.png_stochastic_model}/"
            f"prior{row.nuisance_prior_scale:g}/"
            f"precision-{row.precision_correction}"
            for row in ordered
        ],
        dtype=np.str_,
    )
    indices = np.full(
        (len(ordered), maximum_bins),
        -1,
        dtype=np.int64,
    )
    vectors = {
        name: np.full(
            (len(ordered), maximum_bins),
            np.nan,
            dtype=np.float64,
        )
        for name in (
            "prediction",
            "gaussian_prediction",
            "target",
            "sigma_single",
            "linear_response",
            "quadratic_coefficient",
        )
    }
    profile_x = np.full(
        (len(ordered), 33),
        np.nan,
        dtype=np.float64,
    )
    profile_delta = np.full_like(profile_x, np.nan)
    profile_length = np.empty(len(ordered), dtype=np.int64)
    profile_curve_kind = np.asarray(
        [row.profile_curve_kind for row in ordered],
        dtype=np.str_,
    )
    component_names = sorted(
        {
            name
            for row in ordered
            for name in row.response_components
        }
    )
    component_vectors = {
        name: np.full(
            (len(ordered), maximum_bins),
            np.nan,
            dtype=np.float64,
        )
        for name in component_names
    }
    centres = np.empty(len(ordered), dtype=np.float64)
    errors = np.empty((len(ordered), 2), dtype=np.float64)
    truths = np.empty(len(ordered), dtype=np.float64)
    kmax = np.empty(len(ordered), dtype=np.float64)
    for row_index, row in enumerate(ordered):
        size = row.indices.size
        indices[row_index, :size] = row.indices
        vectors["prediction"][row_index, :size] = row.prediction
        vectors["gaussian_prediction"][
            row_index, :size
        ] = row.gaussian_prediction
        vectors["target"][row_index, :size] = row.target
        vectors["sigma_single"][row_index, :size] = np.sqrt(
            np.diag(row.covariance_single)
        )
        vectors["linear_response"][
            row_index, :size
        ] = row.linear_response
        vectors["quadratic_coefficient"][
            row_index, :size
        ] = row.quadratic_coefficient
        curve_size = min(33, row.profile_x.size)
        profile_x[row_index, :curve_size] = row.profile_x[:curve_size]
        profile_delta[
            row_index, :curve_size
        ] = row.profile_delta[:curve_size]
        profile_length[row_index] = curve_size
        for name, value in row.response_components.items():
            component_vectors[name][row_index, :size] = np.asarray(
                value,
                dtype=np.float64,
            )[row.indices]
        centres[row_index] = row.fhat
        errors[row_index] = (row.error_low, row.error_high)
        truths[row_index] = row.truth_fnl
        kmax[row_index] = row.kmax
    payload: dict[str, Any] = {
        "schema": np.asarray(
            "marisa-b-post-r1-finite-png-profiles-v1",
            dtype=np.str_,
        ),
        "profile_id": identifiers,
        "indices": indices,
        "profile_x": profile_x,
        "profile_delta_chi2": profile_delta,
        "profile_curve_length": profile_length,
        "profile_curve_kind": profile_curve_kind,
        "profile_curve_is_actual_minuit": np.asarray(
            [
                kind == "actual_33_point_minuit_mnprofile"
                for kind in profile_curve_kind
            ],
            dtype=bool,
        ),
        "b000_k_pair": np.asarray(k_pair, dtype=np.float64),
        "fhat": centres,
        "error_low_high": errors,
        "truth_fnl": truths,
        "kmax": kmax,
        **vectors,
        **{
            f"response_component__{name}": value
            for name, value in component_vectors.items()
        },
    }
    for sample, arrays in coverage_arrays.items():
        for name, value in arrays.items():
            payload[f"coverage__{sample}__{name}"] = np.asarray(value)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def run_gaussian(
    *,
    args: argparse.Namespace,
    paths: Paths,
    module: ModuleType,
    runner: ModuleType,
) -> tuple[dict[str, Any], dict[tuple[str, str, float], Any]]:
    data = module.DataSet.load(paths.matrix)
    contract, frozen_prior, nbar = module.load_prior(paths.contract)
    templates = load_partial_templates(
        module,
        data,
        paths.templates,
    )
    coevolution_submanifold = (
        gaussian_coevolution_submanifold_audit(
            module=module,
            runner=runner,
            templates=templates,
            number_density=nbar,
        )
    )
    if not coevolution_submanifold["pass"]:
        raise ValueError(
            "full and coevolution Gaussian tiers do not agree on the "
            "registered coevolution submanifold"
        )
    with np.load(paths.matrix, allow_pickle=False) as values:
        samples = np.asarray(
            values["fiducial_post_B000"],
            dtype=np.float64,
        )
        stored_covariance = np.asarray(
            values["fiducial_post_B000_cov"],
            dtype=np.float64,
        )
    reproduced = np.cov(samples, rowvar=False, ddof=1)
    covariance_scale = np.sqrt(
        np.outer(
            np.diag(stored_covariance),
            np.diag(stored_covariance),
        )
    )
    covariance_reproduction_error = float(
        np.max(
            np.abs(reproduced - stored_covariance)
            / np.maximum(covariance_scale, np.finfo(float).tiny)
        )
    )
    # The archived aggregate and NumPy use equivalent ddof=1 estimators but
    # a different summation order.  Compare in correlation units so a nearly
    # vanishing off-diagonal entry cannot turn harmless roundoff into a
    # provenance failure.
    if covariance_reproduction_error > 5.0e-15:
        raise ValueError("post covariance is not single-realization covariance")

    rng = np.random.default_rng(args.seed)
    fits: dict[tuple[str, str, float], Any] = {}
    for stochastic_mode in (
        "zero",
        "tied",
        "released",
    ):
        for tier in ("coevolution", "full"):
            for kmax in CUTS:
                fits[(tier, stochastic_mode, kmax)] = fit_one(
                    module=module,
                    runner=runner,
                    templates=templates,
                    data=data,
                    frozen_prior=frozen_prior,
                    nbar=nbar,
                    samples=samples,
                    tier=tier,
                    stochastic_mode=stochastic_mode,
                    kmax=kmax,
                    nuisance_scale=1.0,
                    multistart=args.multistart,
                    rng=rng,
                )
    primary = fits[(PRIMARY_TIER, "tied", max(CUTS))]
    paths.figures.mkdir(parents=True, exist_ok=True)
    plot_gaussian_diagonal(
        paths.figures / "post_gaussian_k2B000_bestfit.pdf",
        primary,
        np.asarray(data.k_pair),
        np.asarray(data.edges),
    )
    header, _bins = read_jsonl(paths.templates)
    def fit_numerics_pass(fit: Any) -> bool:
        return bool(
            fit.optimizer_success
            and float(np.ptp(fit.multistart_objectives)) < 1.0e-6
            and fit.affine_residual_max_sigma_single < 1.0e-7
            and fit.basis_rotation_max_sigma_single < 1.0e-7
            and abs(fit.analytic_linear_delta_objective) < 1.0e-8
        )

    gaussian_optimizer_numerics_pass = all(
        fit_numerics_pass(fit)
        for fit in fits.values()
    )
    perturbativity = gaussian_perturbativity_audit(
        templates,
        fits,
        nbar,
    )
    quantitative_pass = bool(
        gaussian_optimizer_numerics_pass
        and coevolution_submanifold["pass"]
        and perturbativity["pass"]
        and all(
            fit.optimizer_success
            and qnorm(fit.residual, fit.covariance_single) < 0.5
            and fit.max_abs_single_pull < 0.5
            for (tier, mode, _kmax), fit in fits.items()
            if tier == PRIMARY_TIER and mode == "tied"
        )
    )
    gaussian_reference_rows: list[dict[str, Any]] = []
    primary_parameters = {
        name: primary.expanded_value(name)
        for name in primary.expanded_names
    }
    base_prediction = templates.components(
        primary_parameters,
        nbar,
        "tied",
    )["total"][primary.selected_indices]
    for raw_reference in args.gaussian_reference:
        reference = raw_reference.resolve()
        reference_header, _reference_bins = read_jsonl(reference)
        validate_header(
            reference_header,
            registered_reference=True,
        )
        reference_templates = load_partial_templates(
            module,
            data,
            reference,
            registered_reference=True,
        )
        reference_prediction = reference_templates.components(
            primary_parameters,
            nbar,
            "tied",
        )["total"][primary.selected_indices]
        displacement = reference_prediction - base_prediction
        reference_fit = fit_one(
            module=module,
            runner=runner,
            templates=reference_templates,
            data=data,
            frozen_prior=frozen_prior,
            nbar=nbar,
            samples=samples,
            tier=PRIMARY_TIER,
            stochastic_mode="tied",
            kmax=max(CUTS),
            nuisance_scale=1.0,
            multistart=args.multistart,
            rng=rng,
        )
        refit_displacement = (
            np.asarray(reference_fit.prediction)
            - np.asarray(primary.prediction)
        )
        same_projection = (
            reference_header["shell_projection"]
            == header["shell_projection"]
        )
        same_loop_quadrature = (
            reference_header["loop_quadrature"]
            == header["loop_quadrature"]
        )
        same_qmax = math.isclose(
            float(reference_header["q_range"][1]),
            float(header["q_range"][1]),
            rel_tol=0.0,
            abs_tol=1.0e-13,
        )
        changed_header_paths = deep_change_paths(
            header,
            reference_header,
            ignored_paths=frozenset(
                {
                    "bin_range",
                    "bin_indices",
                    "merged_parts",
                }
            ),
        )
        changed_header_path_set = set(changed_header_paths)
        projector_change_paths = {
            "shell_projection.radial_order",
            "shell_projection.angular_order",
            "shell_projection.orientation_orders",
        }
        if (
            not same_qmax
            and same_loop_quadrature
            and same_projection
            and changed_header_path_set == {"q_range"}
            and math.isclose(
                float(reference_header["q_range"][0]),
                float(header["q_range"][0]),
                rel_tol=0.0,
                abs_tol=1.0e-13,
            )
        ):
            category = "qmax_uv"
        elif (
            same_qmax
            and not same_loop_quadrature
            and same_projection
            and changed_header_path_set == {"loop_quadrature"}
        ):
            category = "loop_quadrature"
        elif (
            same_qmax
            and same_loop_quadrature
            and not same_projection
            and bool(changed_header_path_set)
            and changed_header_path_set <= projector_change_paths
        ):
            primary_projection = header["shell_projection"]
            reference_projection = reference_header[
                "shell_projection"
            ]

            def projector_nodes(projection: dict[str, Any]) -> int:
                return (
                    int(projection["radial_order"]) ** 2
                    * int(projection["angular_order"])
                    * math.prod(
                        int(order)
                        for order in projection[
                            "orientation_orders"
                        ]
                    )
                )

            category = (
                "shell_projector"
                if projector_nodes(reference_projection)
                > projector_nodes(primary_projection)
                else "coarse_shell_projector_diagnostic"
            )
        else:
            category = "unregistered_mixed_change"
        gaussian_reference_rows.append(
            {
                "category": category,
                "path": str(reference),
                "sha256": sha256(reference),
                "q_range": reference_header["q_range"],
                "changed_header_paths": list(changed_header_paths),
                "loop_quadrature": (
                    reference_header["loop_quadrature"]
                ),
                "loop_angular_rule": (
                    reference_header["loop_angular_rule"]
                ),
                "fixed_bestfit_displacement_norm_sigma_single": (
                    qnorm(displacement, primary.covariance_single)
                ),
                "max_abs_single_bin_sigma": float(
                    np.max(
                        np.abs(displacement)
                        / np.sqrt(
                            np.diag(primary.covariance_single)
                        )
                    )
                ),
                "refit_prediction_displacement_norm_sigma_single": (
                    qnorm(
                        refit_displacement,
                        primary.covariance_single,
                    )
                ),
                "refit_prediction_max_abs_single_bin_sigma": float(
                    np.max(
                        np.abs(refit_displacement)
                        / np.sqrt(
                            np.diag(primary.covariance_single)
                        )
                    )
                ),
                "refit_residual_norm_sigma_single": float(
                    qnorm(
                        reference_fit.residual,
                        reference_fit.covariance_single,
                    )
                ),
                "refit_max_abs_single_bin_sigma": (
                    reference_fit.max_abs_single_pull
                ),
                "refit_optimizer_success": (
                    reference_fit.optimizer_success
                ),
                "refit_multistart_delta_objective": float(
                    np.ptp(reference_fit.multistart_objectives)
                ),
                "refit_affine_residual_max_sigma_single": (
                    reference_fit.affine_residual_max_sigma_single
                ),
                "refit_basis_rotation_max_sigma_single": (
                    reference_fit.basis_rotation_max_sigma_single
                ),
                "refit_analytic_linear_delta_objective": (
                    reference_fit.analytic_linear_delta_objective
                ),
                "refit_optimizer_numerics_pass": (
                    fit_numerics_pass(reference_fit)
                ),
            }
        )
    categories = {
        row["category"] for row in gaussian_reference_rows
    }
    has_qmax_reference = "qmax_uv" in categories
    has_quadrature_reference = "loop_quadrature" in categories
    has_projector_reference = "shell_projector" in categories
    gaussian_reference_pass = []
    for row in gaussian_reference_rows:
        fixed_gate = (
            row["fixed_bestfit_displacement_norm_sigma_single"]
            < 0.05
            and row["max_abs_single_bin_sigma"] < 0.1
        )
        refit_gate = (
            row[
                "refit_prediction_displacement_norm_sigma_single"
            ]
            < 0.05
            and row[
                "refit_prediction_max_abs_single_bin_sigma"
            ]
            < 0.1
            and row["refit_residual_norm_sigma_single"] < 0.5
            and row["refit_max_abs_single_bin_sigma"] < 0.5
            and row["refit_optimizer_numerics_pass"]
        )
        row["pass"] = bool(
            row["category"] != "unregistered_mixed_change"
            and refit_gate
            and (
                refit_gate
                if row["category"] == "qmax_uv"
                else fixed_gate
            )
        )
        row["required_for_production"] = (
            row["category"]
            != "coarse_shell_projector_diagnostic"
        )
        if row["required_for_production"]:
            gaussian_reference_pass.append(row["pass"])
    gaussian_convergence_pass = bool(
        has_qmax_reference
        and has_quadrature_reference
        and has_projector_reference
        and all(gaussian_reference_pass)
    )
    production_numerics = bool(
        header.get("production_candidate") is True
        and "diagnostic_overlay" not in header
        and gaussian_convergence_pass
    )
    summary = {
        "schema": "marisa-b-post-recon-gaussian-gate-v1",
        "status": (
            "pass"
            if quantitative_pass and production_numerics
            else (
                "quantitative_pass_diagnostic_numerics"
                if quantitative_pass
                else "fail"
            )
        ),
        "covariance": {
            "kind": "single-realization sample covariance",
            "divided_by_mock_count": False,
            "mock_count": int(samples.shape[0]),
            "likelihood_precision_correction": "Hartlap",
            "hartlap_factor_by_n_data": {
                str(count): float(
                    (samples.shape[0] - count - 2)
                    / (samples.shape[0] - 1)
                )
                for count in (9, 14, 20, 27)
            },
            "max_reproduction_error_in_correlation_units": (
                covariance_reproduction_error
            ),
        },
        "b1_prior": {
            "mean": B1_PRIOR_MEAN,
            "sigma": B1_PRIOR_SIGMA,
            "semantics": "broad P0-centred prior; b1 is profiled",
        },
        "optimizer_numerics_gate": {
            "pass": gaussian_optimizer_numerics_pass,
            "multistart_delta_objective_max": 1.0e-6,
            "affine_residual_sigma_single_max": 1.0e-7,
            "basis_rotation_sigma_single_max": 1.0e-7,
            "analytic_linear_delta_objective_max": 1.0e-8,
        },
        "coevolution_submanifold_gate": coevolution_submanifold,
        "perturbativity_gate": perturbativity,
        "reconstruction_bias_contract": {
            "b_rec_h": B_REC_H,
            "status": "fixed_observable_not_nuisance",
            "may_vary_without_reconstructing_catalogue": False,
            "effective_matter_reconstruction_bias": (
                "b_rec_eff=b_rec_h/b1"
            ),
            "template_polynomial_coordinate": (
                "lambda=1/b_rec_eff=b1/b_rec_h"
            ),
            "dynamic_at_every_profile_point": True,
        },
        "production_numerics": {
            "pass": production_numerics,
            "candidate_header_validated": (
                header.get("production_candidate") is True
            ),
            "has_higher_qmax_reference": has_qmax_reference,
            "has_refined_quadrature_reference": (
                has_quadrature_reference
            ),
            "has_refined_shell_projector_reference": (
                has_projector_reference
            ),
            "thresholds": {
                "vector_norm_sigma_single": 0.05,
                "max_abs_single_bin_sigma": 0.1,
            },
            "references": gaussian_reference_rows,
        },
        "stochastic_branches": {
            "zero": (
                "density-only residual stochastic closure; the "
                "physical estimator-matched conditional P/nbar "
                "reconstruction term remains in the baseline"
            ),
            "tied": (
                "the same stochastic directions source density "
                "and reconstruction displacement, in addition to "
                "the fixed conditional P/nbar term"
            ),
            "released": (
                "tied prediction plus four independent noisy-shift "
                "increments with zero-centred inherited-width priors; "
                "the fixed conditional P/nbar term is shared"
            ),
            "released_shift_parameters": [
                RELEASED_SHIFT_PREFIX + name
                for name in RELEASED_SHIFT_BASE
            ],
            "fixed_poisson": (
                "estimator-matched conditional leading P/nbar; "
                "bare fixed PL3 contractions are excluded pending "
                "their own UV renormalization"
            ),
        },
        "inputs": {
            "templates": str(paths.templates),
            "templates_sha256": sha256(paths.templates),
            "matrix": str(paths.matrix),
            "matrix_sha256": sha256(paths.matrix),
            "contract": str(paths.contract),
            "contract_sha256": sha256(paths.contract),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
        },
        "fits": {
            f"{tier}/{mode}/{kmax:.2f}": fit_summary(fit)
            for (tier, mode, kmax), fit in fits.items()
        },
        "finite_png_gate": {
            "opened": quantitative_pass and production_numerics,
            "reason": (
                (
                    "all coevolution/tied single-covariance Gaussian "
                    "fit gates pass with production numerics"
                )
                if quantitative_pass and production_numerics
                else (
                    "Gaussian fit is quantitatively acceptable but "
                    "uses diagnostic, non-production numerics"
                    if quantitative_pass
                    else "one or more Gaussian fit gates fail"
                )
            ),
            "production_numerics": production_numerics,
        },
    }
    return summary, fits


def png_algebra_audit(
    *,
    gaussian_templates: PostTemplateSet,
    png_templates: PngTemplateSet,
    parameters: dict[str, float],
    nbar: float,
) -> dict[str, Any]:
    parameters = dict(parameters)
    parameters["Bshot_residual"] = 0.43
    parameters[PNG_STOCHASTIC_NAME] = -0.27
    tree = png_templates.tree
    def closes(
        absolute_error: float,
        scale: float,
        *,
        absolute_tolerance: float = 2.0e-8,
        relative_tolerance: float = 2.0e-12,
    ) -> bool:
        return bool(
            absolute_error <= absolute_tolerance
            or absolute_error
            <= relative_tolerance * max(scale, np.finfo(float).tiny)
        )

    def vector_scale(*vectors: np.ndarray) -> float:
        return float(
            max(
                (
                    np.max(np.abs(np.asarray(vector)))
                    for vector in vectors
                ),
                default=0.0,
            )
        )
    linear_sum = sum(
        (
            tree[name]
            for name in (
                "dBdfNL_local_primordial",
                "dBdfNL_local_bphi_f2",
                "dBdfNL_local_bphi_advection",
                "dBdfNL_local_bphidelta",
                "dBdfNL_local_bphi_b2",
                "dBdfNL_local_bphi_bK2",
                "dBdfNL_local_bphi_reconstruction",
            )
        ),
        np.zeros(120, dtype=np.float64),
    )
    quadratic_sum = sum(
        (
            tree[name]
            for name in (
                "Bhalo_tree_fNL2_bphi_B0",
                "Bhalo_tree_fNL2_bphi_sq_advection",
                "Bhalo_tree_fNL2_bphi_sq_F2",
                "Bhalo_tree_fNL2_bphi_sq_b2",
                "Bhalo_tree_fNL2_bphi_sq_bK2",
                "Bhalo_tree_fNL2_bphi_sq_reconstruction",
                "Bhalo_tree_fNL2_bphi_bphidelta",
                "Bhalo_tree_fNL2_bphi2_operator",
            )
        ),
        np.zeros(120, dtype=np.float64),
    )
    matter_identity = {
        name: float(
            np.max(
                np.abs(
                    png_templates.matter_linear_coefficients[
                        "total"
                    ][order]
                    - png_templates.matter_linear_coefficients[
                        "tree"
                    ][order]
                    - png_templates.matter_linear_coefficients[
                        "loop"
                    ][order]
                )
            )
        )
        for order, name in enumerate(
            ("lambda0", "lambda1", "lambda2")
        )
    }
    matter_identity_scale = {
        name: vector_scale(
            png_templates.matter_linear_coefficients["total"][
                order
            ],
            png_templates.matter_linear_coefficients["tree"][
                order
            ],
            png_templates.matter_linear_coefficients["loop"][
                order
            ],
        )
        for order, name in enumerate(
            ("lambda0", "lambda1", "lambda2")
        )
    }
    fixed_gaussian_png = (
        PngTemplateSet._evaluate_sparse(
            png_templates.fixed_gaussian,
            parameters,
        )
        / nbar
    )
    fixed_gaussian_eft = gaussian_templates.fixed_poisson(
        parameters,
        nbar,
    )
    gaussian_bshot_probe = dict(parameters)
    for name in gaussian_templates.stochastic_names:
        gaussian_bshot_probe[name] = 0.0
    gaussian_bshot_probe["Bshot_residual"] = 1.0
    gaussian_one_loop_bshot_shape = (
        gaussian_templates.components(
            gaussian_bshot_probe,
            nbar,
            "tied",
        )["stochastic"]
    )
    png_tree_bshot_shape = (
        PngTemplateSet._evaluate_sparse(
            png_templates.residual_gaussian["tied"],
            gaussian_bshot_probe,
        )
        / nbar
    )
    bshot_cross_driver_difference = (
        gaussian_one_loop_bshot_shape - png_tree_bshot_shape
    )
    response = png_templates.components(
        parameters,
        nbar,
        "tied",
        "independent_eq265",
    )
    gaussian = gaussian_templates.components(
        parameters,
        nbar,
        "tied",
    )["total"]
    finite_zero = combine_finite_png_prediction(
        gaussian,
        response,
        0.0,
    )
    # A step of four is still an exact symmetric probe for the registered
    # quadratic polynomial, while avoiding cancellation of the much larger
    # Gaussian baseline in float64.
    finite_step = 4.0
    finite_plus = combine_finite_png_prediction(
        gaussian,
        response,
        finite_step,
    )
    finite_minus = combine_finite_png_prediction(
        gaussian,
        response,
        -finite_step,
    )
    finite_derivative = (
        finite_plus - finite_minus
    ) / (2.0 * finite_step)
    finite_quadratic = (
        finite_plus + finite_minus - 2.0 * finite_zero
    ) / (2.0 * finite_step**2)

    matter_parameters = {
        name: 0.0 for name in parameters
    }
    matter_parameters.update(
        {
            "b1": 1.0,
            "b2": 0.0,
            "gamma2": 0.0,
            "Bshot_residual": 0.0,
            PNG_STOCHASTIC_NAME: 0.0,
        }
    )
    matter_response = png_templates.components(
        matter_parameters,
        nbar,
        "tied",
        "independent_eq265",
    )
    matter_lambda = 1.0 / B_REC_H
    expected_matter_tree = PngTemplateSet._evaluate_polynomial(
        png_templates.matter_linear_coefficients["tree"],
        matter_lambda,
    )
    expected_matter_loop = PngTemplateSet._evaluate_polynomial(
        png_templates.matter_linear_coefficients["loop"],
        matter_lambda,
    )
    expected_matter_total = PngTemplateSet._evaluate_polynomial(
        png_templates.matter_linear_coefficients["total"],
        matter_lambda,
    )
    expected_matter_b112 = PngTemplateSet._evaluate_polynomial(
        png_templates.matter_b112_coefficients,
        matter_lambda,
    )
    matter_limit_errors = {
        "halo_tree_to_matter_tree": float(
            np.max(
                np.abs(
                    matter_response["halo_primordial"]
                    - expected_matter_tree
                )
            )
        ),
        "uplift_to_matter_loop": float(
            np.max(
                np.abs(
                    matter_response["matter_one_loop_uplift"]
                    - expected_matter_loop
                )
            )
        ),
        "linear_total_to_matter_total": float(
            np.max(
                np.abs(
                    matter_response["linear_total"]
                    - expected_matter_total
                )
            )
        ),
        "quadratic_total_to_B112II": float(
            np.max(
                np.abs(
                    matter_response["quadratic_total"]
                    - expected_matter_b112
                )
            )
        ),
    }
    matter_limit_scales = {
        "halo_tree_to_matter_tree": vector_scale(
            matter_response["halo_primordial"],
            expected_matter_tree,
        ),
        "uplift_to_matter_loop": vector_scale(
            matter_response["matter_one_loop_uplift"],
            expected_matter_loop,
        ),
        "linear_total_to_matter_total": vector_scale(
            matter_response["linear_total"],
            expected_matter_total,
        ),
        "quadratic_total_to_B112II": vector_scale(
            matter_response["quadratic_total"],
            expected_matter_b112,
        ),
    }
    dynamic_lambda_rows = []
    for dynamic_b1 in (1.6, 3.1):
        probe = dict(matter_parameters)
        probe["b1"] = dynamic_b1
        dynamic = png_templates.components(
            probe,
            nbar,
            "tied",
            "independent_eq265",
        )
        lambda_value = dynamic_b1 / B_REC_H
        expected_loop = dynamic_b1**3 * (
            PngTemplateSet._evaluate_polynomial(
                png_templates.matter_linear_coefficients["loop"],
                lambda_value,
            )
        )
        expected_b112 = dynamic_b1**3 * (
            PngTemplateSet._evaluate_polynomial(
                png_templates.matter_b112_coefficients,
                lambda_value,
            )
        )
        dynamic_lambda_rows.append(
            {
                "b1": dynamic_b1,
                "expected_lambda": lambda_value,
                "reported_lambda": float(dynamic["lambda"]),
                "lambda_absolute_error": abs(
                    float(dynamic["lambda"]) - lambda_value
                ),
                "matter_loop_maximum_absolute_error": float(
                    np.max(
                        np.abs(
                            dynamic["matter_one_loop_uplift"]
                            - expected_loop
                        )
                    )
                ),
                "matter_loop_scale": vector_scale(
                    dynamic["matter_one_loop_uplift"],
                    expected_loop,
                ),
                "matter_B112II_maximum_absolute_error": float(
                    np.max(
                        np.abs(
                            dynamic["matter_B112II"]
                            - expected_b112
                        )
                    )
                ),
                "matter_B112II_scale": vector_scale(
                    dynamic["matter_B112II"],
                    expected_b112,
                ),
            }
        )
    direct_recombinations = {}
    direct_recombination_scales = {}
    for fnl in (0.0, 100.0, -100.0):
        direct = combine_finite_png_prediction(
            gaussian,
            response,
            fnl,
        )
        recombined = gaussian.copy()
        recombined += fnl * sum(
            (
                value
                for name, value in response.items()
                if name not in {
                    "linear_total",
                    "quadratic_total",
                }
                and name
                in {
                    "halo_primordial",
                    "halo_bphi_f2",
                    "halo_bphi_advection",
                    "halo_bphidelta",
                    "halo_bphi_b2",
                    "halo_bphi_bK2",
                    "halo_reconstruction",
                    "residual_stochastic_linear",
                    "fixed_poisson_linear",
                    "matter_one_loop_uplift",
                }
            ),
            np.zeros(120, dtype=np.float64),
        )
        recombined += fnl**2 * sum(
            (
                value
                for name, value in response.items()
                if name
                in {
                    "halo_bphi_B0",
                    "halo_bphi_sq_advection",
                    "halo_bphi_sq_F2",
                    "halo_bphi_sq_b2",
                    "halo_bphi_sq_bK2",
                    "halo_bphi_sq_reconstruction",
                    "halo_bphi_bphidelta",
                    "halo_bphi2_operator",
                    "residual_stochastic_quadratic",
                    "fixed_poisson_quadratic",
                    "matter_B112II",
                }
            ),
            np.zeros(120, dtype=np.float64),
        )
        direct_recombinations[f"{fnl:g}"] = float(
            np.max(np.abs(direct - recombined))
        )
        direct_recombination_scales[f"{fnl:g}"] = (
            vector_scale(direct, recombined)
        )
    b1 = float(parameters["b1"])
    bphi = 2.0 * DELTA_C * (b1 - P_UNIVERSALITY)
    raw_linear_shape = (
        b1
        * bphi
        * tree["dBdfNL_stochastic_alpha3_basis"]
        / nbar
    )
    raw_quadratic_shape = (
        bphi**2
        * tree[
            "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"
        ]
        / nbar
    )
    residual_map_errors: dict[str, float] = {}
    residual_map_scales: dict[str, float] = {}
    expected_raw_maps = {
        "residual_gaussian": (
            b1**2 * tree["stochastic_alpha3_basis"]
        ),
        "residual_linear": (
            b1 * tree["dBdfNL_stochastic_alpha3_basis"]
        ),
        "residual_quadratic": tree[
            "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"
        ],
    }
    for field, expected_map in expected_raw_maps.items():
        maps = getattr(png_templates, field)
        density = PngTemplateSet._evaluate_sparse(
            maps["density_only"],
            parameters,
        )
        tied = PngTemplateSet._evaluate_sparse(
            maps["tied"],
            parameters,
        )
        noisy = PngTemplateSet._evaluate_sparse(
            maps["noisy_shift"],
            parameters,
        )
        residual_map_errors[
            f"{field}_native_max_abs"
        ] = float(np.max(np.abs(density - expected_map)))
        residual_map_scales[
            f"{field}_native_max_abs"
        ] = vector_scale(density, expected_map)
        residual_map_errors[
            f"{field}_tied_minus_density_max_abs"
        ] = float(np.max(np.abs(tied - density)))
        residual_map_scales[
            f"{field}_tied_minus_density_max_abs"
        ] = vector_scale(tied, density)
        residual_map_errors[
            f"{field}_noisy_shift_max_abs"
        ] = float(np.max(np.abs(noisy)))
        residual_map_scales[
            f"{field}_noisy_shift_max_abs"
        ] = vector_scale(density)
    alpha3 = float(parameters["Bshot_residual"])
    alpha3_png = float(parameters[PNG_STOCHASTIC_NAME])
    stochastic_formula_errors = {
        "independent_linear_max_abs": float(
            np.max(
                np.abs(
                    response[
                        "residual_stochastic_linear"
                    ]
                    - (alpha3 + alpha3_png)
                    * raw_linear_shape
                )
            )
        ),
        "independent_quadratic_max_abs": float(
            np.max(
                np.abs(
                    response[
                        "residual_stochastic_quadratic"
                    ]
                    - alpha3_png * raw_quadratic_shape
                )
            )
        ),
    }
    stochastic_formula_scales = {
        "independent_linear_max_abs": vector_scale(
            response["residual_stochastic_linear"],
            (alpha3 + alpha3_png) * raw_linear_shape,
        ),
        "independent_quadratic_max_abs": vector_scale(
            response["residual_stochastic_quadratic"],
            alpha3_png * raw_quadratic_shape,
        ),
    }
    strict_tied = png_templates.components(
        parameters,
        nbar,
        "tied",
        "strict_tied_eq265",
    )
    legacy = png_templates.components(
        parameters,
        nbar,
        "tied",
        "legacy_accepted_pre",
    )
    stochastic_formula_errors.update(
        {
            "strict_tied_linear_max_abs": float(
                np.max(
                    np.abs(
                        strict_tied[
                            "residual_stochastic_linear"
                        ]
                        - 2.0 * alpha3 * raw_linear_shape
                    )
                )
            ),
            "strict_tied_quadratic_max_abs": float(
                np.max(
                    np.abs(
                        strict_tied[
                            "residual_stochastic_quadratic"
                        ]
                        - alpha3 * raw_quadratic_shape
                    )
                )
            ),
            "legacy_linear_max_abs": float(
                np.max(
                    np.abs(
                        legacy[
                            "residual_stochastic_linear"
                        ]
                        - alpha3 * raw_linear_shape
                    )
                )
            ),
            "legacy_quadratic_max_abs": float(
                np.max(
                    np.abs(
                        legacy[
                            "residual_stochastic_quadratic"
                        ]
                        - alpha3 * raw_quadratic_shape
                    )
                )
            ),
        }
    )
    stochastic_formula_scales.update(
        {
            "strict_tied_linear_max_abs": vector_scale(
                strict_tied["residual_stochastic_linear"],
                2.0 * alpha3 * raw_linear_shape,
            ),
            "strict_tied_quadratic_max_abs": vector_scale(
                strict_tied["residual_stochastic_quadratic"],
                alpha3 * raw_quadratic_shape,
            ),
            "legacy_linear_max_abs": vector_scale(
                legacy["residual_stochastic_linear"],
                alpha3 * raw_linear_shape,
            ),
            "legacy_quadratic_max_abs": vector_scale(
                legacy["residual_stochastic_quadratic"],
                alpha3 * raw_quadratic_shape,
            ),
        }
    )
    values = {
        "raw_halo_linear_component_sum_max_abs": float(
            np.max(
                np.abs(
                    linear_sum
                    - tree["dBdfNL_local_tree"]
                )
            )
        ),
        "raw_halo_quadratic_component_sum_max_abs": float(
            np.max(
                np.abs(
                    quadratic_sum
                    - tree[
                        "Bhalo_tree_fNL2_deterministic"
                    ]
                )
            )
        ),
        "matter_total_minus_tree_minus_loop_by_polynomial_order": (
            matter_identity
        ),
        "fixed_poisson_gaussian_cross_driver_max_abs": float(
            np.max(
                np.abs(
                    fixed_gaussian_png - fixed_gaussian_eft
                )
            )
        ),
        "residual_stochastic_gaussian_cross_driver": {
            "gaussian_one_loop_minus_png_tree_max_abs": float(
                np.max(np.abs(bshot_cross_driver_difference))
            ),
            "gaussian_one_loop_minus_png_tree_max_relative": float(
                np.max(np.abs(bshot_cross_driver_difference))
                / max(
                    vector_scale(
                        gaussian_one_loop_bshot_shape,
                        png_tree_bshot_shape,
                    ),
                    np.finfo(float).tiny,
                )
            ),
            "is_acceptance_gate": False,
            "interpretation": (
                "expected hybrid-order loop remainder: the Gaussian "
                "Bshot_residual direction is halo one-loop, whereas "
                "the finite-PNG L/Q completion is leading halo tree"
            ),
        },
        "finite_fnl_direct_recombination_max_abs": (
            direct_recombinations
        ),
        "finite_fnl_zero_max_abs": float(
            np.max(np.abs(finite_zero - gaussian))
        ),
        "finite_fnl_central_step": finite_step,
        "finite_fnl_central_derivative_max_abs": float(
            np.max(
                np.abs(
                    finite_derivative - response["linear_total"]
                )
            )
        ),
        "finite_fnl_central_quadratic_coefficient_max_abs": float(
            np.max(
                np.abs(
                    finite_quadratic
                    - response["quadratic_total"]
                )
            )
        ),
        "matter_limit_max_abs": matter_limit_errors,
        "dynamic_lambda_rows": dynamic_lambda_rows,
        "residual_stochastic_native_map_max_abs": (
            residual_map_errors
        ),
        "eq265_stochastic_formula_max_abs": (
            stochastic_formula_errors
        ),
        "fNL0_returns_gaussian_max_abs": float(
            np.max(np.abs(finite_zero - gaussian))
        ),
        "matter_uplift_uses_loop_not_total": bool(
            closes(
                matter_limit_errors["uplift_to_matter_loop"],
                matter_limit_scales["uplift_to_matter_loop"],
            )
        ),
        "matter_B112II_added_once": bool(
            closes(
                matter_limit_errors[
                    "quadratic_total_to_B112II"
                ],
                matter_limit_scales[
                    "quadratic_total_to_B112II"
                ],
            )
        ),
        "dynamic_lambda": (
            "exact quadratic matter-linear and exact linear B112II "
            "polynomials in b1/b_rec_h"
        ),
    }
    gate_scales = {
        "raw_halo_linear_component_sum": vector_scale(
            linear_sum,
            tree["dBdfNL_local_tree"],
        ),
        "raw_halo_quadratic_component_sum": vector_scale(
            quadratic_sum,
            tree["Bhalo_tree_fNL2_deterministic"],
        ),
        "fixed_poisson_gaussian_cross_driver": vector_scale(
            fixed_gaussian_png,
            fixed_gaussian_eft,
        ),
        "finite_fnl_zero": vector_scale(
            finite_zero,
            gaussian,
        ),
        "finite_fnl_central_derivative": vector_scale(
            finite_derivative,
            response["linear_total"],
        ),
        "finite_fnl_central_quadratic_coefficient": vector_scale(
            finite_quadratic,
            response["quadratic_total"],
        ),
    }
    gate_errors = {
        "raw_halo_linear_component_sum": values[
            "raw_halo_linear_component_sum_max_abs"
        ],
        "raw_halo_quadratic_component_sum": values[
            "raw_halo_quadratic_component_sum_max_abs"
        ],
        "fixed_poisson_gaussian_cross_driver": values[
            "fixed_poisson_gaussian_cross_driver_max_abs"
        ],
        "finite_fnl_zero": values[
            "finite_fnl_zero_max_abs"
        ],
        "finite_fnl_central_derivative": values[
            "finite_fnl_central_derivative_max_abs"
        ],
        "finite_fnl_central_quadratic_coefficient": values[
            "finite_fnl_central_quadratic_coefficient_max_abs"
        ],
    }
    for name, error in matter_identity.items():
        gate_errors[f"matter_identity_{name}"] = error
        gate_scales[f"matter_identity_{name}"] = (
            matter_identity_scale[name]
        )
    for name, error in direct_recombinations.items():
        gate_errors[f"direct_recombination_{name}"] = error
        gate_scales[f"direct_recombination_{name}"] = (
            direct_recombination_scales[name]
        )
    for name, error in matter_limit_errors.items():
        gate_errors[f"matter_limit_{name}"] = error
        gate_scales[f"matter_limit_{name}"] = (
            matter_limit_scales[name]
        )
    for row_index, row in enumerate(dynamic_lambda_rows):
        lambda_name = f"dynamic_lambda_{row_index}"
        gate_errors[lambda_name] = row[
            "lambda_absolute_error"
        ]
        gate_scales[lambda_name] = max(
            abs(float(row["expected_lambda"])),
            np.finfo(float).tiny,
        )
        loop_name = f"dynamic_matter_loop_{row_index}"
        gate_errors[loop_name] = row[
            "matter_loop_maximum_absolute_error"
        ]
        gate_scales[loop_name] = row["matter_loop_scale"]
        b112_name = f"dynamic_matter_B112II_{row_index}"
        gate_errors[b112_name] = row[
            "matter_B112II_maximum_absolute_error"
        ]
        gate_scales[b112_name] = row["matter_B112II_scale"]
    for name, error in residual_map_errors.items():
        gate_errors[f"residual_map_{name}"] = error
        gate_scales[f"residual_map_{name}"] = (
            residual_map_scales[name]
        )
    for name, error in stochastic_formula_errors.items():
        gate_errors[f"stochastic_formula_{name}"] = error
        gate_scales[f"stochastic_formula_{name}"] = (
            stochastic_formula_scales[name]
        )
    values["identity_gates"] = {
        name: {
            "maximum_absolute_error": error,
            "scale": gate_scales[name],
            "maximum_relative_error": (
                error
                / max(
                    gate_scales[name],
                    np.finfo(float).tiny,
                )
            ),
            "pass": closes(error, gate_scales[name]),
        }
        for name, error in gate_errors.items()
    }
    values["pass"] = bool(
        all(
            item["pass"]
            for item in values["identity_gates"].values()
        )
    )
    return values


def write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    rows = summary["profiles"]
    table = [
        "| kmax | injection | fNL best fit | 68% interval | "
        "|bias|/sigma | residual norm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for kmax in CUTS:
        for sample, truth, _key, _nbar in SAMPLES:
            row = next(
                item
                for item in rows
                if item["tier"] == PRIMARY_TIER
                and item["sample"] == sample
                and math.isclose(
                    item["kmax_h_mpc"],
                    kmax,
                )
            )
            bias_over_sigma = row["absolute_bias_over_sigma"]
            bias_display = (
                f"{bias_over_sigma:.3f}"
                if bias_over_sigma is not None
                else "prior-truncated"
            )
            interval_suffix = (
                ""
                if row["interval68_is_data_two_sided"]
                else " (prior-truncated)"
            )
            table.append(
                f"| {kmax:.2f} | {truth:+.0f} | "
                f"{row['fhat']:.2f} | "
                f"[{row['interval68'][0]:.2f}, "
                f"{row['interval68'][1]:.2f}]"
                f"{interval_suffix} | "
                f"{bias_display} | "
                f"{row['residual_norm_sigma_single']:.3f} |"
            )
    acceptance = summary["acceptance"]
    runner_argv = [
        str(value)
        for value in summary["inputs"]["runner"]["argv"]
    ]
    reproduction_command = shlex.join(
        [sys.executable, *runner_argv]
    )
    review_rows = []
    for scope, status in summary["review_status"].items():
        if isinstance(status, dict):
            review_rows.append(
                f"- `{scope}`: `{status['status']}` by "
                f"`{status['reviewer']}`; report SHA256 "
                f"`{status['report_sha256']}`"
            )
        else:
            review_rows.append(f"- `{scope}`: `{status}`")
    text = f"""# Post-reconstruction halo finite-PNG v1

## Scope

This report validates the real-space, z=1, R=15
`post-recon coevolution finite-PNG v1` model.  Its Gaussian baseline is
the reconstructed renormalized halo one-loop EFT-v2 template.  Its PNG
sector is the complete finite local-PNG halo tree through
`fNL^2` in the deterministic sector, plus the leading stochastic sector
with independent `Bshot_residual` and `Bshot_PNG_residual` amplitudes as
required by Moradinezhad Dizgah et al. Eq. (2.65), the dynamic pure-matter
one-loop uplift, and post `B112II`.  The historical one-amplitude
`G+fL+fNL^2 Q` rule is retained only as a labelled robustness branch
because it is not an Eq. (2.65) closure.  This model is not claimed to be
a complete halo-PNG one-loop model.

The Eq. (2.65) finite completion is applied at leading halo-tree order.
The Gaussian `Bshot_residual` shape in the EFT-v2 baseline additionally
contains halo-loop stochastic pieces, while its PNG `L/Q` partners do
not.  Their cross-driver Gaussian-shape difference is therefore recorded
as an expected hybrid-order loop remainder, not forced to zero.  Missing
PNG one-loop stochastic response is an explicit scope limitation.

All likelihoods, pulls, plotted error bars, and norms use the covariance
of one 1 Gpc/h realization.  Ensemble means are used only as low-noise
central measurements.  The covariance is never divided by the number of
mocks.

The production prior on `fNL` is a hard uniform top-hat centred on the
fiducial value zero with support `[-150, 150]`.  Its log prior is zero
inside the support and minus infinity outside.  This support is fixed
independently of any injection label and is never recentered on a sample
truth.  A profile interval that reaches this boundary is explicitly
prior-truncated rather than interpreted as a data-only two-sided
constraint.

## Main coevolution recovery

{chr(10).join(table)}

The largest validated fiducial-halo cutoff is
`{acceptance['largest_validated_halo_fiducial_kmax_h_mpc']}` h/Mpc.  The target
cutoff 0.14 has status `{acceptance['target_kmax_0p14_status']}`.

## Key gates

- finite-PNG algebra: `{summary['theory_gates']['algebra']['pass']}`
- Minuit/profile validity: `{acceptance['all_main_minuit_valid']}`
- truth recovery at 0.14: `{acceptance['target_kmax_0p14_status']}`
- coverage compatible with 68%: `{acceptance['coverage_gate_pass']}`
- PNG qmax=20 to 40 tail canary:
  `{acceptance['png_qmax_tail_canary_gate_pass']}`
- independent-review final state: `{summary['status']}`
- covariance divided by mock count: `False`

## Independent reviews

{chr(10).join(review_rows)}

## Interpretation

The empirical halo validation is deliberately restricted to the reliable
fiducial fNL=0 catalogue.  The halo LCp/LCm standard-reconstruction
catalogues are quarantined and are not loaded by inference, used for
coverage, shown in figures, or allowed to determine kmax.  Direct
fNL=0,+100,-100 recombinations are theory-only algebra tests.  DM
catalogues are out of scope and are not loaded or compared.  `b1` and every
identifiable bispectrum nuisance in the selected tier are re-profiled at
every cutoff from the fiducial/P0 centre.  Coevolution is the production
tier; full is retained only as a cross-check.  The historical Gaussian
uplift model is not run.

## Reproduction

From the repository root, rerun the registered inference command:

```bash
{reproduction_command}
```

The registered Gaussian JSONL and finite-PNG cache are retained at the exact
paths in `summary.json`.  The finite-PNG cache records the SHA256 of every
exact-lattice raw JSONL input.  `VERIFICATION.json` locks the runner, native
sources, driver sources, and both executables; `REVIEWS.json` locks the
pending summary, compact profile product, Gaussian gate, and seven final
PDFs.  See `summary.json` for all input hashes, numerical settings,
robustness rows, and review status.
"""
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_finite_png(
    *,
    args: argparse.Namespace,
    paths: Paths,
    module: ModuleType,
    runner: ModuleType,
    gaussian_summary: dict[str, Any],
    gaussian_fits: dict[tuple[str, str, float], Any],
) -> dict[str, Any]:
    verification = load_verification(paths)
    data = module.DataSet.load(paths.matrix)
    _contract, frozen_prior, fiducial_nbar = module.load_prior(
        paths.contract
    )
    gaussian_templates = load_partial_templates(
        module,
        data,
        paths.templates,
    )
    png_templates = load_png_cache(paths.png_cache)
    gaussian_header, _gaussian_bins = read_jsonl(paths.templates)
    contract_payload = json.loads(
        paths.contract.read_text(encoding="utf-8")
    )
    png_common = png_templates.metadata.get(
        "common_raw_contract",
        {},
    )
    png_source_hashes = png_common.get("source_hashes", {})
    gaussian_source_hashes = gaussian_header.get(
        "source_hashes",
        {},
    )
    data_edges = np.asarray(data.edges, dtype=np.float64).reshape(
        120, 4
    )
    data_k_pair = np.asarray(data.k_pair, dtype=np.float64)
    required_png_mask = (
        np.max(data_k_pair, axis=1)
        <= max(CUTS) + 1.0e-12
    )
    required_png_mask[EXCLUDED_BIN] = False
    required_png_indices = set(
        np.flatnonzero(required_png_mask).tolist()
    )
    cached_png_indices = set(
        np.asarray(png_templates.indices, dtype=np.int64).tolist()
    )
    theory_provenance_cross_gate = {
        "gaussian_driver_header_verification_match": bool(
            gaussian_source_hashes.get("driver_executable")
            == verification.get("source_hashes", {}).get(
                "gaussian_driver_binary"
            )
        ),
        "png_driver_header_verification_match": bool(
            png_source_hashes.get("driver_executable")
            == verification.get("source_hashes", {}).get(
                "png_driver_binary"
            )
        ),
        "linear_power_gaussian_png_contract_match": bool(
            gaussian_source_hashes.get("linear_power")
            == png_source_hashes.get("linear_power")
            == contract_payload.get("source_hashes", {}).get(
                "linear_power_z1"
            )
        ),
        "parameter_registry_gaussian_png_match": bool(
            gaussian_header.get("parameter_registry_sha256")
            == png_source_hashes.get("parameter_registry")
        ),
        "edge_file_gaussian_png_match": bool(
            gaussian_source_hashes.get("edge_file")
            == png_source_hashes.get("edge_file")
        ),
        "reconstruction_gaussian_png_match": bool(
            gaussian_header.get("reconstruction")
            == png_common.get("reconstruction")
        ),
        "fft_geometry_gaussian_png_match": bool(
            gaussian_header.get("shell_projection", {}).get(
                "fft_box_size"
            )
            == png_common.get("shell_projection", {}).get(
                "fft_box_size"
            )
            and gaussian_header.get("shell_projection", {}).get(
                "fft_mesh_size"
            )
            == png_common.get("shell_projection", {}).get(
                "fft_mesh_size"
            )
        ),
        "shell_projector_gaussian_png_match": bool(
            (
                gaussian_header.get("shell_projection", {}).get(
                    "radial_order"
                ),
                gaussian_header.get("shell_projection", {}).get(
                    "angular_order"
                ),
                gaussian_header.get("shell_projection", {}).get(
                    "orientation_orders"
                ),
            )
            == (
                png_common.get("shell_projection", {}).get(
                    "radial_order"
                ),
                png_common.get("shell_projection", {}).get(
                    "angular_order"
                ),
                png_common.get("shell_projection", {}).get(
                    "orientation_orders"
                ),
            )
        ),
        "png_cache_covers_all_selected_bins": bool(
            required_png_indices <= cached_png_indices
        ),
        "png_cache_edges_match_fiducial_matrix": bool(
            np.array_equal(
                png_templates.edges,
                data_edges[png_templates.indices],
            )
        ),
    }
    if not all(theory_provenance_cross_gate.values()):
        raise ValueError(
            "Gaussian and finite-PNG theory provenance do not close"
        )
    with np.load(paths.matrix, allow_pickle=False) as values:
        fiducial_ids = np.asarray(
            values["fiducial_realizations"],
            dtype=np.int64,
        )
        fiducial_samples = np.asarray(
            values["fiducial_post_B000"],
            dtype=np.float64,
        )
        fiducial_pre = np.asarray(
            values["fiducial_pre_B000"],
            dtype=np.float64,
        )
        k_pair = np.asarray(values["b000_k"], dtype=np.float64)
        fiducial_edges = np.asarray(
            values["b000_k_edges"],
            dtype=np.float64,
        )
        fiducial_nmodes = np.asarray(
            values["b000_nmodes"],
            dtype=np.int64,
        )
    # User-supplied scope correction (2026-07-27): the halo LCp/LCm
    # standard-reconstruction catalogues are known to be problematic.
    # Production inference is fail-closed to the fiducial fNL=0 halo
    # catalogue.  In particular, never parse or load arrays from the
    # quarantined matrix here: only its byte hash is recorded below, without
    # making it a data dependency of any likelihood, coverage result, cutoff,
    # or figure.
    target_by_sample = {"fiducial": fiducial_samples}
    target_ids_by_sample = {"fiducial": fiducial_ids}
    registered_vectors = (fiducial_samples, fiducial_pre)
    data_audit = {
        "validation_scope": "halo_fiducial_fNL0_only",
        "halo_nonzero_png_catalogues": (
            "quarantined_not_loaded_not_profiled"
        ),
        "fiducial_ids_registered_0_to_499": bool(
            np.array_equal(fiducial_ids, np.arange(500))
        ),
        "fiducial_realization_ids_unique": bool(
            fiducial_ids.size == np.unique(fiducial_ids).size
        ),
        "fiducial_B000_shapes": bool(
            fiducial_samples.shape == (500, 120)
            and fiducial_pre.shape == (500, 120)
        ),
        "fiducial_B000_finite": bool(
            all(np.all(np.isfinite(vector)) for vector in registered_vectors)
        ),
        "bin_geometry_shape": bool(k_pair.shape == (120, 2)),
        "bin_edges_shape": bool(fiducial_edges.shape == (120, 2, 2)),
        "bin_nmodes_shape": bool(fiducial_nmodes.shape == (120,)),
        "bin_geometry_finite": bool(np.all(np.isfinite(k_pair))),
        "bin_edges_finite": bool(np.all(np.isfinite(fiducial_edges))),
        "bin_nmodes_positive": bool(np.all(fiducial_nmodes > 0)),
        "fiducial_number_density_matches_eft_contract": bool(
            math.isclose(
                float(fiducial_nbar),
                float(SAMPLES[0][3]),
                rel_tol=2.0e-15,
                abs_tol=0.0,
            )
        ),
        "fiducial_number_density_h3_mpc3": {
            "eft_contract": float(fiducial_nbar),
            "registered_sample": float(SAMPLES[0][3]),
        },
        "halo_data_validated_injections": [0],
        "halo_LCp_minus_LCm_used_as_theory_response": False,
        "quarantined_halo_png_matrix": {
            "path": str(paths.quarantined_halo_png_matrix),
            "sha256": (
                sha256(paths.quarantined_halo_png_matrix)
                if paths.quarantined_halo_png_matrix.is_file()
                else None
            ),
            "arrays_loaded_by_production_inference": False,
            "reason": (
                "halo fNL=+/-100 standard-reconstruction catalogues "
                "are known to be problematic"
            ),
        },
    }
    if not all(
        bool(value)
        for value in data_audit.values()
        if isinstance(value, (bool, np.bool_))
    ):
        raise ValueError("fiducial post-reconstruction data do not close")
    numerical_gate_mask = (
        np.max(k_pair, axis=1) <= max(CUTS) + 1.0e-12
    )
    numerical_gate_mask[EXCLUDED_BIN] = False
    numerical_gate_indices = np.flatnonzero(numerical_gate_mask)
    numerical_gate_covariance = np.cov(
        fiducial_samples[:, numerical_gate_indices],
        rowvar=False,
        ddof=1,
    )
    cache_numerical_gate = png_cache_numerical_gate(
        png_templates,
        numerical_gate_indices,
        numerical_gate_covariance,
    )
    if not cache_numerical_gate["pass"]:
        raise RuntimeError(
            "finite-PNG cache interpolation/provenance gate fails"
        )
    tail_canary_gate = png_tail_canary_gate(
        raw_paths=args.png_tail_canary,
        templates=png_templates,
        expected_edges=fiducial_edges,
        covariance_samples=fiducial_samples,
    )

    rng = np.random.default_rng(args.seed + 17)
    main_profiles: list[FinitePngProfile] = []
    for tier in ("coevolution", "full"):
        for sample, truth, _key, number_density in SAMPLES:
            for kmax in CUTS:
                main_profiles.append(
                    profile_finite_png(
                        module=module,
                        runner=runner,
                        gaussian_templates=gaussian_templates,
                        png_templates=png_templates,
                        data=data,
                        frozen_prior=frozen_prior,
                        covariance_samples=fiducial_samples,
                        target_samples=target_by_sample[sample],
                        nbar=number_density,
                        tier=tier,
                        stochastic_mode="tied",
                        sample=sample,
                        truth_fnl=truth,
                        kmax=kmax,
                        nuisance_prior_scale=1.0,
                        multistart=args.multistart,
                        rng=rng,
                        compute_curve=True,
                    )
                )

    robustness_profiles: list[FinitePngProfile] = []
    for prior_scale in (0.5, 2.0):
        for sample, truth, _key, number_density in SAMPLES:
            robustness_profiles.append(
                profile_finite_png(
                    module=module,
                    runner=runner,
                    gaussian_templates=gaussian_templates,
                    png_templates=png_templates,
                    data=data,
                    frozen_prior=frozen_prior,
                    covariance_samples=fiducial_samples,
                    target_samples=target_by_sample[sample],
                    nbar=number_density,
                    tier=PRIMARY_TIER,
                    stochastic_mode="tied",
                    sample=sample,
                    truth_fnl=truth,
                    kmax=max(CUTS),
                    nuisance_prior_scale=prior_scale,
                    multistart=max(5, args.multistart // 2),
                    rng=rng,
                    compute_curve=False,
                )
            )
    for stochastic_mode in ("zero", "released"):
        for sample, truth, _key, number_density in SAMPLES:
            robustness_profiles.append(
                profile_finite_png(
                    module=module,
                    runner=runner,
                    gaussian_templates=gaussian_templates,
                    png_templates=png_templates,
                    data=data,
                    frozen_prior=frozen_prior,
                    covariance_samples=fiducial_samples,
                    target_samples=target_by_sample[sample],
                    nbar=number_density,
                    tier=PRIMARY_TIER,
                    stochastic_mode=stochastic_mode,
                    sample=sample,
                    truth_fnl=truth,
                    kmax=max(CUTS),
                    nuisance_prior_scale=1.0,
                    multistart=max(5, args.multistart // 2),
                    rng=rng,
                    compute_curve=False,
                )
            )
    for png_stochastic_model in (
        "strict_tied_eq265",
        "legacy_accepted_pre",
    ):
        for sample, truth, _key, number_density in SAMPLES:
            robustness_profiles.append(
                profile_finite_png(
                    module=module,
                    runner=runner,
                    gaussian_templates=gaussian_templates,
                    png_templates=png_templates,
                    data=data,
                    frozen_prior=frozen_prior,
                    covariance_samples=fiducial_samples,
                    target_samples=target_by_sample[sample],
                    nbar=number_density,
                    tier=PRIMARY_TIER,
                    stochastic_mode="tied",
                    sample=sample,
                    truth_fnl=truth,
                    kmax=max(CUTS),
                    nuisance_prior_scale=1.0,
                    multistart=max(5, args.multistart // 2),
                    rng=rng,
                    compute_curve=False,
                    png_stochastic_model=png_stochastic_model,
                )
            )
    for sample, truth, _key, number_density in SAMPLES:
        robustness_profiles.append(
            profile_finite_png(
                module=module,
                runner=runner,
                gaussian_templates=gaussian_templates,
                png_templates=png_templates,
                data=data,
                frozen_prior=frozen_prior,
                covariance_samples=fiducial_samples,
                target_samples=target_by_sample[sample],
                nbar=number_density,
                tier=PRIMARY_TIER,
                stochastic_mode="tied",
                sample=sample + "_uncorrected_precision",
                truth_fnl=truth,
                kmax=max(CUTS),
                nuisance_prior_scale=1.0,
                multistart=max(5, args.multistart // 2),
                rng=rng,
                compute_curve=False,
                precision_correction="none",
            )
        )

    coverage: dict[str, Any] = {}
    coverage_arrays: dict[str, dict[str, np.ndarray]] = {}
    # Preregister fiducial IDs 0--99 as null-coverage targets held out of the
    # covariance pool.  Reserve the disjoint fiducial IDs >=100 solely for
    # that covariance so tested vectors never participate in its estimate.
    coverage_covariance_samples = fiducial_samples[100:]
    coverage_covariance_ids = fiducial_ids[100:]
    for sample, truth, _key, number_density in SAMPLES:
        target_ids = target_ids_by_sample[sample]
        payload, arrays = actual_profile_coverage(
            module=module,
            runner=runner,
            gaussian_templates=gaussian_templates,
            png_templates=png_templates,
            data=data,
            frozen_prior=frozen_prior,
            covariance_samples=coverage_covariance_samples,
            covariance_realization_ids=coverage_covariance_ids,
            target_samples=target_by_sample[sample],
            target_realization_ids=target_ids,
            nbar=number_density,
            sample=sample,
            truth_fnl=truth,
            realization_count=args.coverage_realizations,
            multistart=args.coverage_multistart,
            rng=rng,
        )
        coverage[sample] = payload
        coverage_arrays[sample] = arrays

    representative = next(
        result
        for result in main_profiles
        if result.tier == PRIMARY_TIER
        and result.sample == "fiducial"
        and math.isclose(result.kmax, max(CUTS))
    )
    pre_fit_path = (
        paths.data_root
        / "analysis/pre_recon_bias_model_v3_20260725/fit_summary.npz"
    )
    with np.load(pre_fit_path, allow_pickle=False) as values:
        pre_indices = np.asarray(
            values["b000_selected_indices"],
            dtype=np.int64,
        )
        pre_prediction = np.asarray(
            values["b000_model"],
            dtype=np.float64,
        )
    post_gaussian_fit = gaussian_fits[
        (PRIMARY_TIER, "tied", max(CUTS))
    ]
    post_indices = np.asarray(
        post_gaussian_fit.selected_indices,
        dtype=np.int64,
    )
    if not np.array_equal(pre_indices, post_indices):
        raise ValueError(
            "pre/post Gaussian closure uses mismatched selected bins"
        )
    paired_difference_samples = (
        fiducial_samples[:, post_indices]
        - fiducial_pre[:, post_indices]
    )
    paired_covariance = np.cov(
        paired_difference_samples,
        rowvar=False,
        ddof=1,
    )
    paired_data = np.mean(
        paired_difference_samples,
        axis=0,
    )
    paired_model = (
        np.asarray(post_gaussian_fit.prediction)
        - pre_prediction
    )
    paired_residual = paired_model - paired_data
    paired_reconstruction_closure = {
        "semantics": (
            "difference of independently profiled post and pre "
            "Gaussian production baselines, tested with paired "
            "single-realization Cov(Bpost-Bpre)"
        ),
        "pre_fit": {
            "path": str(pre_fit_path),
            "sha256": sha256(pre_fit_path),
        },
        "selected_indices": post_indices,
        "covariance_divided_by_mock_count": False,
        "residual_norm_sigma_single_paired": qnorm(
            paired_residual,
            paired_covariance,
        ),
        "max_abs_single_bin_sigma_paired": float(
            np.max(
                np.abs(paired_residual)
                / np.sqrt(np.diag(paired_covariance))
            )
        ),
    }
    algebra = png_algebra_audit(
        gaussian_templates=gaussian_templates,
        png_templates=png_templates,
        parameters=representative.expanded_parameters,
        nbar=SAMPLES[0][3],
    )
    numerical_variants: dict[str, Any] = {
        "gaussian": [],
        "png": [],
    }
    representative_indices = representative.indices
    representative_gaussian = gaussian_templates.components(
        representative.expanded_parameters,
        SAMPLES[0][3],
        "tied",
    )["total"][representative_indices]
    for reference_path_raw in args.gaussian_reference:
        reference_path = reference_path_raw.resolve()
        reference_templates = load_partial_templates(
            module,
            data,
            reference_path,
            registered_reference=True,
        )
        reference_vector = reference_templates.components(
            representative.expanded_parameters,
            SAMPLES[0][3],
            "tied",
        )["total"][representative_indices]
        displacement = reference_vector - representative_gaussian
        numerical_variants["gaussian"].append(
            {
                "path": str(reference_path),
                "sha256": sha256(reference_path),
                "fixed_bestfit_displacement_norm_sigma_single": (
                    qnorm(
                        displacement,
                        representative.covariance_single,
                    )
                ),
                "max_abs_single_bin_sigma": float(
                    np.max(
                        np.abs(displacement)
                        / np.sqrt(
                            np.diag(
                                representative.covariance_single
                            )
                        )
                    )
                ),
            }
        )
    for reference_cache_raw in args.png_reference_cache:
        reference_cache = reference_cache_raw.resolve()
        alternate = load_png_cache(reference_cache)
        base_integration = png_templates.metadata[
            "common_raw_contract"
        ]["integration"]
        alternate_integration = alternate.metadata[
            "common_raw_contract"
        ]["integration"]
        changed_contract_paths = deep_change_paths(
            png_templates.metadata["common_raw_contract"],
            alternate.metadata["common_raw_contract"],
        )
        changed_contract_path_set = set(changed_contract_paths)
        changed_integration_keys = sorted(
            key
            for key in set(base_integration) | set(alternate_integration)
            if base_integration.get(key)
            != alternate_integration.get(key)
        )
        refined_allowed_paths = {
            "integration.epsrel",
            "integration.p13_epsrel",
            "integration.b112ii_qmc_power",
            "integration.b112ii_qmc_replicates",
        }
        refined_required_paths = {
            "integration.b112ii_qmc_power",
            "integration.b112ii_qmc_replicates",
        }
        if changed_contract_path_set == {"integration.qmin"}:
            variant_category = "finite_box_qmin"
        elif (
            refined_required_paths
            <= changed_contract_path_set
            <= refined_allowed_paths
        ):
            variant_category = "refined_quadrature"
        else:
            variant_category = "unregistered_mixed_change"
        variant_rows = []
        for sample, truth, _key, number_density in SAMPLES:
            baseline_fit = next(
                result
                for result in main_profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
                and math.isclose(result.kmax, max(CUTS))
            )
            baseline_response = png_templates.components(
                baseline_fit.expanded_parameters,
                number_density,
                baseline_fit.stochastic_mode,
                baseline_fit.png_stochastic_model,
            )
            alternate_response = alternate.components(
                baseline_fit.expanded_parameters,
                number_density,
                baseline_fit.stochastic_mode,
                baseline_fit.png_stochastic_model,
            )
            linear_response_difference = (
                alternate_response["linear_total"]
                - baseline_response["linear_total"]
            )
            quadratic_response_difference = (
                alternate_response["quadratic_total"]
                - baseline_response["quadratic_total"]
            )
            bestfit_displacement = (
                baseline_fit.fhat
                * linear_response_difference
                + baseline_fit.fhat**2
                * quadratic_response_difference
            )[baseline_fit.indices]
            theory_only_envelope = []
            for reference_fnl in (
                -PNG_TAIL_FNL_ENVELOPE,
                PNG_TAIL_FNL_ENVELOPE,
            ):
                envelope_displacement = (
                    reference_fnl * linear_response_difference
                    + reference_fnl**2
                    * quadratic_response_difference
                )[baseline_fit.indices]
                theory_only_envelope.append(
                    {
                        "fNL": reference_fnl,
                        "fixed_baseline_nuisance_and_b1": True,
                        "vector_norm_sigma_single": qnorm(
                            envelope_displacement,
                            baseline_fit.covariance_single,
                        ),
                        "max_abs_single_bin_sigma": float(
                            np.max(
                                np.abs(envelope_displacement)
                                / np.sqrt(
                                    np.diag(
                                        baseline_fit.covariance_single
                                    )
                                )
                            )
                        ),
                    }
                )
            alternate_fit = profile_finite_png(
                module=module,
                runner=runner,
                gaussian_templates=gaussian_templates,
                png_templates=alternate,
                data=data,
                frozen_prior=frozen_prior,
                covariance_samples=fiducial_samples,
                target_samples=target_by_sample[sample],
                nbar=number_density,
                tier=PRIMARY_TIER,
                stochastic_mode="tied",
                sample=sample,
                truth_fnl=truth,
                kmax=max(CUTS),
                nuisance_prior_scale=1.0,
                multistart=max(5, args.multistart // 2),
                rng=rng,
                compute_curve=False,
                png_stochastic_model=(
                    baseline_fit.png_stochastic_model
                ),
            )
            variant_rows.append(
                {
                    "sample": sample,
                    "fixed_bestfit_displacement_norm_sigma_single": (
                        qnorm(
                            bestfit_displacement,
                            baseline_fit.covariance_single,
                        )
                    ),
                    "max_abs_single_bin_sigma": float(
                        np.max(
                            np.abs(bestfit_displacement)
                            / np.sqrt(
                                np.diag(
                                    baseline_fit.covariance_single
                                )
                            )
                        )
                    ),
                    "theory_only_fNL_plus_minus_100_envelope": (
                        theory_only_envelope
                    ),
                    "baseline_fhat": baseline_fit.fhat,
                    "alternate_fhat": alternate_fit.fhat,
                    "centre_shift_over_baseline_sigma": (
                        abs(
                            alternate_fit.fhat
                            - baseline_fit.fhat
                        )
                        / baseline_fit.sigma_symmetric
                    ),
                    "alternate_profile": finite_profile_row(
                        alternate_fit
                    ),
                    "alternate_profile_numerically_valid": bool(
                        alternate_fit.minuit_valid
                        and alternate_fit.minuit_accurate
                        and alternate_fit.minos_valid
                        and not alternate_fit.minuit_at_limit
                        and alternate_fit.multistart_agree
                    ),
                }
            )
        numerical_variants["png"].append(
            {
                "category": variant_category,
                "changed_common_contract_paths": list(
                    changed_contract_paths
                ),
                "changed_integration_keys": changed_integration_keys,
                "active_qmc_refinement_keys": sorted(
                    refined_required_paths
                    & changed_contract_path_set
                ),
                "inactive_under_multicenter_qmc_changes": (
                    ["integration.p13_epsrel"]
                    if (
                        "integration.p13_epsrel"
                        in changed_contract_path_set
                        and base_integration.get(
                            "matter_linear_multicenter_qmc"
                        )
                        is True
                        and alternate_integration.get(
                            "matter_linear_multicenter_qmc"
                        )
                        is True
                    )
                    else []
                ),
                "path": str(reference_cache),
                "sha256": sha256(reference_cache),
                "metadata": alternate.metadata,
                "cache_numerical_gate": png_cache_numerical_gate(
                    alternate,
                    representative.indices,
                    representative.covariance_single,
                ),
                "rows": variant_rows,
            }
        )

    per_cut: dict[str, Any] = {}
    common_passing: list[float] = []
    for kmax in CUTS:
        rows = [
            result
            for result in main_profiles
            if result.tier == PRIMARY_TIER
            and math.isclose(result.kmax, kmax)
        ]
        row_pass = {
            result.sample: bool(
                abs(result.fhat - result.truth_fnl)
                < 0.5 * result.sigma_symmetric
                and result.fhat + result.error_low
                <= result.truth_fnl
                <= result.fhat + result.error_high
                and result.minuit_valid
                and result.minuit_accurate
                and result.minos_valid
                and not result.minuit_at_limit
                and result.multistart_agree
                and result.profile_global_consistent
                and result.residual_norm_single < 0.5
                and result.max_pull_single < 0.5
            )
            for result in rows
        }
        per_cut[f"{kmax:.2f}"] = row_pass
        if all(row_pass.values()):
            common_passing.append(kmax)
    adjacent = {}
    for sample, _truth, _key, _nbar in SAMPLES:
        rows = sorted(
            [
                result
                for result in main_profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
            ],
            key=lambda result: result.kmax,
        )
        adjacent[sample] = [
            {
                "cuts": [left.kmax, right.kmax],
                "centre_shift_over_combined_sigma": (
                    abs(right.fhat - left.fhat)
                    / math.sqrt(
                        right.sigma_symmetric**2
                        + left.sigma_symmetric**2
                    )
                ),
                "pass_below_0p5": bool(
                    abs(right.fhat - left.fhat)
                    / math.sqrt(
                        right.sigma_symmetric**2
                        + left.sigma_symmetric**2
                    )
                    < 0.5
                ),
            }
            for left, right in zip(rows[:-1], rows[1:])
        ]
    full_distances = {}
    for sample, _truth, _key, _nbar in SAMPLES:
        full_distances[sample] = {}
        for kmax in CUTS:
            coevolution = next(
                result
                for result in main_profiles
                if result.tier == "coevolution"
                and result.sample == sample
                and math.isclose(result.kmax, kmax)
            )
            full = next(
                result
                for result in main_profiles
                if result.tier == "full"
                and result.sample == sample
                and math.isclose(result.kmax, kmax)
            )
            distance = (
                abs(full.fhat - coevolution.fhat)
                / math.sqrt(
                    full.sigma_symmetric**2
                    + coevolution.sigma_symmetric**2
                )
            )
            full_distances[sample][f"{kmax:.2f}"] = {
                "distance_combined_sigma": distance,
                "pass_below_0p3": bool(distance < 0.3),
            }
    prior_width_distances: dict[str, Any] = {}
    finite_mock_distances: dict[str, Any] = {}
    for sample, _truth, _key, _nbar in SAMPLES:
        baseline = next(
            result
            for result in main_profiles
            if result.tier == PRIMARY_TIER
            and result.sample == sample
            and math.isclose(result.kmax, max(CUTS))
        )
        prior_width_distances[sample] = {}
        for scale in (0.5, 2.0):
            alternate = next(
                result
                for result in robustness_profiles
                if result.sample == sample
                and math.isclose(
                    result.nuisance_prior_scale,
                    scale,
                )
                and result.stochastic_mode == "tied"
                and result.png_stochastic_model
                == PRIMARY_PNG_STOCHASTIC_MODEL
                and result.precision_correction == "hartlap"
            )
            shift = (
                abs(alternate.fhat - baseline.fhat)
                / baseline.sigma_symmetric
            )
            prior_width_distances[sample][f"{scale:g}"] = {
                "centre_shift_over_primary_sigma": shift,
                "pass_below_0p2": bool(shift < 0.2),
            }
        uncorrected = next(
            result
            for result in robustness_profiles
            if result.sample
            == sample + "_uncorrected_precision"
        )
        precision_shift = (
            abs(uncorrected.fhat - baseline.fhat)
            / baseline.sigma_symmetric
        )
        finite_mock_distances[sample] = {
            "primary": "Hartlap-corrected Gaussian precision",
            "alternate": "uncorrected Gaussian precision",
            "centre_shift_over_primary_sigma": precision_shift,
            "uncertainty_ratio": (
                uncorrected.sigma_symmetric
                / baseline.sigma_symmetric
            ),
            "pass_below_0p1": bool(precision_shift < 0.1),
        }
    png_stochastic_distances: dict[str, Any] = {}
    for sample, _truth, _key, _nbar in SAMPLES:
        baseline = next(
            result
            for result in main_profiles
            if result.tier == PRIMARY_TIER
            and result.sample == sample
            and math.isclose(result.kmax, max(CUTS))
        )
        png_stochastic_distances[sample] = {}
        for model in (
            "strict_tied_eq265",
            "legacy_accepted_pre",
        ):
            alternate = next(
                result
                for result in robustness_profiles
                if result.tier == PRIMARY_TIER
                and result.sample == sample
                and math.isclose(result.kmax, max(CUTS))
                and result.stochastic_mode == "tied"
                and result.png_stochastic_model == model
            )
            png_stochastic_distances[sample][model] = {
                "fhat": alternate.fhat,
                "sigma_symmetric": alternate.sigma_symmetric,
                "centre_shift_over_primary_sigma": (
                    abs(alternate.fhat - baseline.fhat)
                    / baseline.sigma_symmetric
                ),
                "uncertainty_ratio_to_primary": (
                    alternate.sigma_symmetric
                    / baseline.sigma_symmetric
                ),
            }
    all_main_valid = all(
        result.minuit_valid
        and result.minuit_accurate
        and result.minos_valid
        and not result.minuit_at_limit
        and result.multistart_agree
        and result.profile_global_consistent
        for result in main_profiles
    )
    all_robustness_valid = all(
        result.minuit_valid
        and result.minuit_accurate
        and result.minos_valid
        and not result.minuit_at_limit
        and result.multistart_agree
        for result in robustness_profiles
    )
    coverage_pass = all(
        item["nominal_0p68_inside_binomial95"]
        and item["all_profiles_numerically_valid"]
        for item in coverage.values()
    )
    adjacent_pass = all(
        row["pass_below_0p5"]
        for rows in adjacent.values()
        for row in rows
    )
    full_coevolution_pass = all(
        row["pass_below_0p3"]
        for sample_rows in full_distances.values()
        for row in sample_rows.values()
    )
    prior_width_pass = all(
        row["pass_below_0p2"]
        for sample_rows in prior_width_distances.values()
        for row in sample_rows.values()
    )
    finite_mock_pass = all(
        row["pass_below_0p1"]
        for row in finite_mock_distances.values()
    )
    paired_reconstruction_pass = bool(
        paired_reconstruction_closure[
            "residual_norm_sigma_single_paired"
        ]
        < 0.5
        and paired_reconstruction_closure[
            "max_abs_single_bin_sigma_paired"
        ]
        < 0.5
    )
    png_variant_categories = {
        row["category"] for row in numerical_variants["png"]
    }
    png_numerical_variants_pass = bool(
        "finite_box_qmin" in png_variant_categories
        and "refined_quadrature" in png_variant_categories
        and all(
            variant["category"]
            != "unregistered_mixed_change"
            and variant["cache_numerical_gate"]["pass"]
            and all(
                row[
                    "fixed_bestfit_displacement_norm_sigma_single"
                ]
                < 0.05
                and row["max_abs_single_bin_sigma"] < 0.1
                and all(
                    envelope[
                        "vector_norm_sigma_single"
                    ]
                    < PNG_NUMERICAL_VECTOR_THRESHOLD_SIGMA
                    and envelope[
                        "max_abs_single_bin_sigma"
                    ]
                    < PNG_NUMERICAL_BIN_THRESHOLD_SIGMA
                    for envelope in row[
                        "theory_only_fNL_plus_minus_100_envelope"
                    ]
                )
                and row["centre_shift_over_baseline_sigma"] < 0.1
                and row["alternate_profile_numerically_valid"]
                for row in variant["rows"]
            )
            for variant in numerical_variants["png"]
        )
    )
    gaussian_gate_pass = bool(
        gaussian_summary.get("status") == "pass"
        and gaussian_summary.get("finite_png_gate", {}).get(
            "opened"
        )
        is True
    )
    largest = max(common_passing) if common_passing else None
    acceptance = {
        "validation_scope": "halo_fiducial_fNL0_only",
        "per_cut_halo_fiducial_fNL0_pass": per_cut,
        "largest_validated_halo_fiducial_kmax_h_mpc": largest,
        "halo_nonzero_png_catalogue_validation": (
            "not_performed_catalogues_quarantined"
        ),
        "theory_only_finite_fNL_recombination_values": [
            0,
            100,
            -100,
        ],
        "target_kmax_0p14_status": (
            "pass"
            if max(CUTS) in common_passing
            else "fail"
        ),
        "all_main_minuit_valid": all_main_valid,
        "all_robustness_minuit_valid": all_robustness_valid,
        "coverage_gate_pass": coverage_pass,
        "adjacent_cut_stability": adjacent,
        "adjacent_cut_gate_pass": adjacent_pass,
        "full_vs_coevolution_distance_combined_sigma": (
            full_distances
        ),
        "full_vs_coevolution_gate_pass": full_coevolution_pass,
        "prior_width_distance_primary_sigma": (
            prior_width_distances
        ),
        "prior_width_gate_pass": prior_width_pass,
        "finite_mock_precision_robustness": finite_mock_distances,
        "finite_mock_gate_pass": finite_mock_pass,
        "paired_reconstruction_gate_pass": (
            paired_reconstruction_pass
        ),
        "png_numerical_variants_gate_pass": (
            png_numerical_variants_pass
        ),
        "gaussian_production_gate_pass": gaussian_gate_pass,
        "png_cache_numerical_gate_pass": (
            cache_numerical_gate["pass"]
        ),
        "png_qmax_tail_canary_gate_pass": (
            tail_canary_gate["pass"]
        ),
        "png_stochastic_model_robustness": (
            png_stochastic_distances
        ),
    }
    scientific_gates_pass = bool(
        largest == max(CUTS)
        and coverage_pass
        and algebra["pass"]
        and all_main_valid
        and all_robustness_valid
        and adjacent_pass
        and full_coevolution_pass
        and prior_width_pass
        and finite_mock_pass
        and paired_reconstruction_pass
        and png_numerical_variants_pass
        and gaussian_gate_pass
        and cache_numerical_gate["pass"]
        and tail_canary_gate["pass"]
        and all(theory_provenance_cross_gate.values())
    )
    acceptance["all_scientific_gates_pass"] = (
        scientific_gates_pass
    )

    paths.figures.mkdir(parents=True, exist_ok=True)
    plot_full2d_bestfits(
        paths.figures / "post_full2d_bestfit_fnl0.pdf",
        main_profiles,
        k_pair,
    )
    plot_fnl_profiles(
        paths.figures / "post_fnl_profiles_vs_kmax.pdf",
        main_profiles,
    )
    plot_fnl_recovery(
        paths.figures / "post_fnl_recovery_vs_kmax.pdf",
        main_profiles,
    )
    plot_full_vs_coevolution(
        paths.figures / "post_full_vs_coevolution.pdf",
        main_profiles,
    )
    pre_comparison = plot_pre_vs_post(
        paths.figures / "pre_vs_post_fnl_constraints.pdf",
        main_profiles,
        paths.data_root
        / "analysis/pre_recon_halo_tree_fnl2_v1_20260726/summary.json",
    )
    plot_component_audit(
        paths.figures / "post_model_component_audit.pdf",
        main_profiles,
    )
    save_profiles_npz(
        paths.analysis / "profiles.npz",
        main_profiles + robustness_profiles,
        coverage_arrays,
        k_pair,
    )
    final_figures = [
        paths.figures / name for name in FINAL_FIGURE_NAMES
    ]
    missing_figures = [
        str(path) for path in final_figures if not path.is_file()
    ]
    if missing_figures:
        raise RuntimeError(
            "missing preregistered final figures: "
            + ", ".join(missing_figures)
        )
    summary = {
        "schema": "marisa-b-post-r1-finite-png-analysis-v1",
        "status": (
            "scientific_pass_pending_independent_reviews"
            if scientific_gates_pass
            else "completed_with_failed_gates"
        ),
        "model": {
            "name": "post-recon coevolution finite-PNG v1",
            "gaussian": (
                "renormalized halo one-loop EFT-v2; coevolution "
                "production and full cross-check"
            ),
            "png": (
                "complete deterministic finite local-PNG halo tree "
                "through fNL^2, Eq. (2.65) leading stochastic "
                "sector with independently profiled Gaussian and "
                "PNG residual amplitudes, dynamic matter one-loop "
                "uplift, and post B112II"
            ),
            "png_stochastic_primary": {
                "model": PRIMARY_PNG_STOCHASTIC_MODEL,
                "gaussian_amplitude": "Bshot_residual",
                "png_amplitude": PNG_STOCHASTIC_NAME,
                "formula": (
                    "B_Gres*(G+fNL*L) + "
                    "B_PNGres*(fNL*L+fNL^2*Q)"
                ),
                "png_amplitude_prior": (
                    "zero mean; inherited Bshot_residual width; "
                    "independent of the Gaussian amplitude"
                ),
                "strict_tied_and_legacy_branches_profiled": True,
            },
            "claim_excluded": "complete halo-PNG one-loop",
            "png_stochastic_scope": (
                "Eq. (2.65) finite completion at leading halo-tree "
                "order; the Gaussian one-loop stochastic direction "
                "contains additional loop pieces whose PNG one-loop "
                "response is outside this model"
            ),
            "p": P_UNIVERSALITY,
            "b1": "profiled independently from the P0/fiducial start",
            "fNL_prior": fnl_prior_contract(),
            "RSD": False,
        },
        "covariance": {
            "kind": "single-realization fiducial500 sample covariance",
            "divided_by_mock_count": False,
            "ensemble_means_are_centres_only": True,
        },
        "inputs": {
            "gaussian_templates": {
                "path": str(paths.templates),
                "sha256": sha256(paths.templates),
            },
            "png_cache": {
                "path": str(paths.png_cache),
                "sha256": sha256(paths.png_cache),
                "metadata": png_templates.metadata,
            },
            "png_tail_canary": {
                "required_file_count": 8,
                "files": tail_canary_gate["source_files"],
            },
            "fiducial_matrix": {
                "path": str(paths.matrix),
                "sha256": sha256(paths.matrix),
            },
            "quarantined_halo_png_matrix": {
                "path": str(paths.quarantined_halo_png_matrix),
                "sha256": (
                    sha256(paths.quarantined_halo_png_matrix)
                    if paths.quarantined_halo_png_matrix.is_file()
                    else None
                ),
                "used_by_inference": False,
            },
            "eft_v2_contract": {
                "path": str(paths.contract),
                "sha256": sha256(paths.contract),
            },
            "post_data_contract_audit": {
                "path": str(paths.analysis / "DATA_CONTRACT.md"),
                "sha256": sha256(
                    paths.analysis / "DATA_CONTRACT.md"
                ),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
                "argv": [str(value) for value in sys.argv],
                "seed": args.seed,
                "multistart": args.multistart,
                "coverage_realizations": (
                    args.coverage_realizations
                ),
                "coverage_multistart": args.coverage_multistart,
            },
        },
        "data_audit": data_audit,
        "gaussian_gate": gaussian_summary,
        "theory_gates": {
            "algebra": algebra,
            "gaussian_png_provenance_cross_gate": (
                theory_provenance_cross_gate
            ),
            "png_cache_numerics": cache_numerical_gate,
            "png_qmax_tail_canary": tail_canary_gate,
            "paired_pre_post_gaussian_closure": (
                paired_reconstruction_closure
            ),
            "production_verification": verification,
            "numerical_variants": numerical_variants,
        },
        "profiles": [
            finite_profile_row(result)
            for result in main_profiles
        ],
        "robustness_profiles": [
            finite_profile_row(result)
            for result in robustness_profiles
        ],
        "coverage": coverage,
        "acceptance": acceptance,
        "pre_vs_post": pre_comparison,
        "outputs": {
            "profiles": str(paths.analysis / "profiles.npz"),
            "figures": [
                str(path) for path in final_figures
            ],
        },
        "review_status": {
            "finite_png_algebra": "pending final independent review",
            "uv_eft_numerics": "pending final independent review",
            "data_statistics_figures": "pending final independent review",
        },
    }
    atomic_json(paths.analysis / "summary.json", summary)
    write_report(paths.analysis / "REPORT.md", summary)
    write_manifest(paths, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.finalize_reviews and args.generate_verification:
        raise ValueError(
            "--finalize-reviews and --generate-verification are "
            "mutually exclusive"
        )
    if args.multistart < 5:
        raise ValueError("--multistart must be at least 5")
    if args.coverage_multistart < 5:
        raise ValueError("--coverage-multistart must be at least 5")
    if not (20 <= args.coverage_realizations <= 100):
        raise ValueError(
            "--coverage-realizations must be between 20 and 100"
        )
    if args.stage == "all" and len(args.png_tail_canary) != 8:
        raise ValueError(
            "--stage all requires exactly eight --png-tail-canary "
            "JSONL files"
        )
    paths = build_paths(args)
    paths.analysis.mkdir(parents=True, exist_ok=True)
    if args.finalize_reviews:
        finalized = finalize_independent_reviews(paths)
        print(json.dumps(to_jsonable(finalized), sort_keys=True))
        return
    if args.generate_verification:
        verification = generate_verification(
            paths,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                to_jsonable(verification),
                sort_keys=True,
            )
        )
        return
    if args.png_jsonl:
        if paths.png_cache.exists() and not args.overwrite:
            raise FileExistsError(
                f"{paths.png_cache} exists; use --overwrite"
            )
        compile_png_cache(args.png_jsonl, paths.png_cache)
    required = [
        paths.templates,
        paths.matrix,
        paths.contract,
        paths.pre_runner,
        paths.frozen_fitter,
    ]
    if args.stage == "all":
        required.extend(
            (
                paths.png_cache,
                paths.data_root
                / "analysis/pre_recon_bias_model_v3_20260725/fit_summary.npz",
                paths.data_root
                / "analysis/pre_recon_halo_tree_fnl2_v1_20260726/summary.json",
                paths.analysis / "VERIFICATION.json",
            )
        )
        required.extend(
            path.resolve() for path in args.png_tail_canary
        )
    required.extend(
        path.resolve()
        for path in (
            list(args.gaussian_reference)
            + list(args.png_reference_cache)
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required inputs: " + ", ".join(missing)
        )
    output = paths.analysis / "gaussian_gate_diagnostic.json"
    figure = paths.figures / "post_gaussian_k2B000_bestfit.pdf"
    if (output.exists() or figure.exists()) and not args.overwrite:
        raise FileExistsError(
            "diagnostic output exists; use --overwrite"
        )
    module = load_module(
        "_marisa_b_post_frozen_eft_v2",
        paths.frozen_fitter,
    )
    runner = load_module(
        "_marisa_b_post_pre_runner_helpers",
        paths.pre_runner,
    )
    summary, gaussian_fits = run_gaussian(
        args=args,
        paths=paths,
        module=module,
        runner=runner,
    )
    atomic_json(output, summary)
    if args.stage == "all":
        if not summary["finite_png_gate"]["opened"]:
            raise RuntimeError(
                "finite-PNG stage is blocked by the Gaussian gate"
            )
        finite_summary = run_finite_png(
            args=args,
            paths=paths,
            module=module,
            runner=runner,
            gaussian_summary=summary,
            gaussian_fits=gaussian_fits,
        )
        print(
            json.dumps(
                to_jsonable(finite_summary),
                sort_keys=True,
            )
        )
        return
    print(json.dumps(to_jsonable(summary), sort_keys=True))


if __name__ == "__main__":
    main()
