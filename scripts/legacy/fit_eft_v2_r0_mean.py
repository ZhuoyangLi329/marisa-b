#!/usr/bin/env python3
"""Historical R0 fitter using covariance of the ensemble mean.

Current production runners import its template and parameter containers only.
Executing its likelihood is forbidden unless the caller explicitly acknowledges
the legacy mean-covariance protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2 as chi2_distribution


BIAS_NAMES = [
    "b1", "b2", "gamma2", "b3", "gamma2x", "gamma3", "gamma21",
    "gamma21x", "gamma211", "gamma22", "gamma31",
]
COUNTERTERM_NAMES = [
    "b_nabla2_delta", "b_nabla2_delta2", "b_nabla2_G2",
    "b_grad_delta2", "b_grad_t2",
]
STOCHASTIC_B_NAMES = [
    "Ashot_residual", "a1_pure", "Bshot_residual", "d2", "dG2",
    "dGamma3", "abar0_mixed", "a3_mixed", "a4_mixed", "a5_mixed",
]
# The shell-projected COBRA derivative-stochastic shapes obey the exact identity
# S_abar0 + S_a3 - S_a4/2 - S_a5/2 = 0.  We choose a5=0 as a coordinate gauge
# and fit the three observable combinations.  Keeping all four would create an
# artificial flat direction and make posterior errors basis-dependent.
FIT_STOCHASTIC_B_NAMES = [name for name in STOCHASTIC_B_NAMES if name != "a5_mixed"]
STOCHASTIC_POWER_NAMES = ["Pshot", "a0_power"]
FIT_NAMES = BIAS_NAMES + COUNTERTERM_NAMES + STOCHASTIC_POWER_NAMES + FIT_STOCHASTIC_B_NAMES
NONLINEAR_FIT_NAMES = BIAS_NAMES + ["b_nabla2_delta"]
LINEAR_FIT_NAMES = [name for name in FIT_NAMES if name not in NONLINEAR_FIT_NAMES]
DIAGRAM_NAMES = ["tree", "B222", "B321I", "B321II", "B411"]
EXACT_DEGENERACY = {
    "relation": "S_abar0 + S_a3 - 0.5*S_a4 - 0.5*S_a5 = 0",
    "coordinate_choice": "a5_mixed=0",
    "reported_combinations": {
        "abar0_mixed": "abar0_mixed + 2*a5_mixed",
        "a3_mixed": "a3_mixed + 2*a5_mixed",
        "a4_mixed": "a4_mixed - a5_mixed",
    },
    "prior_projection": (
        "The independent raw a_i Gaussian prior is marginalized onto the three "
        "reported combinations, including their induced covariance."
    ),
}
PDF_METADATA_CONTEXT = ""
POWER_KMAX_H_MPC = 0.10
MAX_WHITENED_SIGN_RUN = 5
NEXT_BIN_MIN_P_VALUE = 0.01
NEXT_BIN_MAX_ABS_MEAN_WHITENED = 1.0
RAW_GAUSSIAN_LIKELIHOOD = "raw_gaussian"
HARTLAP_GAUSSIAN_LIKELIHOOD = "hartlap_gaussian"
SELLENTIN_HEAVENS_LIKELIHOOD = "sellentin_heavens"
SUPPORTED_LIKELIHOODS = {
    RAW_GAUSSIAN_LIKELIHOOD,
    HARTLAP_GAUSSIAN_LIKELIHOOD,
    SELLENTIN_HEAVENS_LIKELIHOOD,
}
SELECTED_DR_SOURCE_PATHS = {
    "r0_template_driver_cpp": "src/eft_v2/r0_template_driver.cpp",
    "parameter_registry_cpp": "src/eft_v2/parameter_registry.cpp",
    "parameter_registry_h": "src/eft_v2/parameter_registry.h",
    "template_algebra_cpp": "src/eft_v2/template_algebra.cpp",
    "template_algebra_h": "src/eft_v2/template_algebra.h",
    "kernel_primitives_cpp": "src/eft_v2/kernel_primitives.cpp",
    "kernel_primitives_h": "src/eft_v2/kernel_primitives.h",
    "field_kernel_provider_cpp": "src/eft_v2/field_kernel_provider.cpp",
    "field_kernel_provider_h": "src/eft_v2/field_kernel_provider.h",
    "fftlog_dr_oracle_cpp": "src/eft_v2/fftlog_dr_oracle.cpp",
    "fftlog_dr_oracle_h": "src/eft_v2/fftlog_dr_oracle.h",
    "direct_evaluator_cpp": "src/eft_v2/direct_evaluator.cpp",
    "direct_evaluator_h": "src/eft_v2/direct_evaluator.h",
    "ir_safe_integrands_cpp": "src/eft_v2/ir_safe_integrands.cpp",
    "ir_safe_integrands_h": "src/eft_v2/ir_safe_integrands.h",
    "tracer_power_cpp": "src/eft_v2/tracer_power.cpp",
    "tracer_power_h": "src/eft_v2/tracer_power.h",
    "uv_subtraction_cpp": "src/eft_v2/uv_subtraction.cpp",
    "uv_subtraction_h": "src/eft_v2/uv_subtraction.h",
    "bias_operators_cpp": "src/eft_v2/bias_operators.cpp",
    "bias_operators_h": "src/eft_v2/bias_operators.h",
    "diagram_assembler_cpp": "src/eft_v2/diagram_assembler.cpp",
    "diagram_assembler_h": "src/eft_v2/diagram_assembler.h",
    "generated_b222_b321i_tables_h": (
        "src/eft_v2/generated_b222_b321i_tables.h"
    ),
    "generated_b411_table_h": "src/eft_v2/generated_b411_table.h",
    "halo_v1_cpp": "src/halo_v1/halo_v1.cpp",
    "halo_v1_h": "src/halo_v1/halo_v1.h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-legacy-mean-likelihood",
        action="store_true",
        help="Acknowledge that this historical CLI divides covariance by N.",
    )
    parser.add_argument("--ir-templates", type=Path, required=True)
    parser.add_argument("--noir-templates", type=Path, required=True)
    parser.add_argument("--ir-power-templates", type=Path)
    parser.add_argument("--noir-power-templates", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--selected-dr-gate", type=Path,
        help=(
            "Selected-DR v1/v2 certificate; numerical thresholds and current "
            "source provenance are independently revalidated."
        ),
    )
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--regulator", action="append", default=[], metavar="LABEL=PATH",
        help=(
            "Complete 120-bin bispectrum template JSONL used in the regulator "
            "plot and release Gate."
        ),
    )
    parser.add_argument(
        "--regulator-power", action="append", default=[], metavar="LABEL=PATH",
        help=(
            "Complete 47-bin power template paired with a --regulator label. "
            "Required for joint-fit variations that also alter P0."
        ),
    )
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--figure-set",
        choices=("release", "diagonal-k2b000"),
        default="release",
        help=(
            "Generate the five registered release/Gate figures, or only the "
            "user-requested fitted diagonal k^2 B000 comparison."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def require_finite_json(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location}: non-finite floating-point value")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    require_finite_json(value, str(path))
    return value


def read_jsonl(
    path: Path, *, require_full: bool = True, expected_count: int = 120,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = []
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=reject_nonstandard_constant,
                    object_pairs_hook=reject_duplicate_keys,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: top-level JSON record must be an object"
                )
            require_finite_json(row, f"{path}:{line_number}")
            rows.append(row)
    if not rows or rows[0].get("record") != "header":
        raise ValueError(f"{path} must begin with one header record")
    headers = [row for row in rows if row.get("record") == "header"]
    bins = [row for row in rows if row.get("record") == "bin"]
    if len(headers) != 1 or len(headers) + len(bins) != len(rows):
        raise ValueError(f"{path} must contain one header followed only by bin records")
    indices = []
    for row_number, row in enumerate(bins, 1):
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"{path}: bin record {row_number} has a non-integer index")
        if not 0 <= index < expected_count:
            raise ValueError(
                f"{path}: bin index {index} lies outside 0..{expected_count - 1}"
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{path} contains duplicate bin indices")
    if indices != sorted(indices):
        raise ValueError(f"{path} bin records are not in strictly increasing order")
    if require_full and indices != list(range(expected_count)):
        raise ValueError(
            f"{path} does not contain the complete ordered {expected_count}-bin vector"
        )
    if require_full and headers[0].get("bin_range") != [0, expected_count]:
        raise ValueError(
            f"{path} full-vector header bin_range is not [0,{expected_count}]"
        )
    return headers[0], bins


def evaluate_polynomial(polynomial: dict[str, float], parameters: dict[str, float]) -> float:
    total = 0.0
    for monomial, coefficient in polynomial.items():
        value = float(coefficient)
        if monomial != "1":
            for factor in monomial.split("*"):
                name, power = factor.rsplit("^", 1)
                value *= parameters[name] ** int(power)
        total += value
    return total


@dataclass
class TemplateSet:
    path: Path
    header: dict[str, Any]
    bins: list[dict[str, Any]]
    _compiled: dict[str, dict[str, np.ndarray]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = {}
        for name in DIAGRAM_NAMES:
            self._compile_shape(f"diagram:{name}", [row["diagrams"][name] for row in self.bins])
        for name in COUNTERTERM_NAMES:
            self._compile_shape(
                f"counterterm:{name}", [row["counterterms"][name] for row in self.bins],
            )
        for name in STOCHASTIC_B_NAMES:
            self._compile_shape(
                f"stochastic:{name}", [row["stochastic"][name] for row in self.bins],
            )
        self._compile_shape(
            "stochastic:bshot_bnabla2_cross",
            [row["bshot_bnabla2_cross"] for row in self.bins],
        )

    def _compile_shape(
        self, key: str, polynomials: list[dict[str, float]],
    ) -> None:
        terms: dict[str, np.ndarray] = {}
        for bin_index, polynomial in enumerate(polynomials):
            for monomial, coefficient in polynomial.items():
                if monomial not in terms:
                    terms[monomial] = np.zeros(120, dtype=np.float64)
                terms[monomial][bin_index] = float(coefficient)
        self._compiled[key] = terms

    @staticmethod
    def _monomial_value(monomial: str, parameters: dict[str, float]) -> float:
        value = 1.0
        if monomial != "1":
            for factor in monomial.split("*"):
                name, power = factor.rsplit("^", 1)
                value *= parameters[name] ** int(power)
        return value

    def _evaluate_shape(
        self, key: str, parameters: dict[str, float], cache: dict[str, float],
    ) -> np.ndarray:
        result = np.zeros(120, dtype=np.float64)
        for monomial, coefficients in self._compiled[key].items():
            if monomial not in cache:
                cache[monomial] = self._monomial_value(monomial, parameters)
            result += cache[monomial] * coefficients
        return result

    @classmethod
    def load(cls, path: Path) -> "TemplateSet":
        header, bins = read_jsonl(path)
        return cls(path=path, header=header, bins=bins)

    @classmethod
    def load_overlay(cls, path: Path, base: "TemplateSet") -> "TemplateSet":
        header, partial = read_jsonl(path, require_full=False)
        rows = list(base.bins)
        for row in partial:
            rows[int(row["index"])] = row
        return cls(path=path, header=header, bins=rows)

    def components(
        self, parameters: dict[str, float], number_density: float,
    ) -> dict[str, np.ndarray]:
        monomial_cache: dict[str, float] = {"1": 1.0}
        result = {
            name: self._evaluate_shape(
                f"diagram:{name}", parameters, monomial_cache,
            )
            for name in DIAGRAM_NAMES
        }
        result["counterterm"] = np.zeros(120)
        result["stochastic"] = np.zeros(120)
        for name in COUNTERTERM_NAMES:
            shape = self._evaluate_shape(
                f"counterterm:{name}", parameters, monomial_cache,
            )
            result["counterterm"] += parameters[name] * shape
        for name in STOCHASTIC_B_NAMES:
            shape = self._evaluate_shape(
                f"stochastic:{name}", parameters, monomial_cache,
            )
            normalization = (
                number_density**2
                if name in {"Ashot_residual", "a1_pure"}
                else number_density
            )
            result["stochastic"] += parameters[name] * shape / normalization
        cross = self._evaluate_shape(
            "stochastic:bshot_bnabla2_cross", parameters, monomial_cache,
        )
        result["stochastic"] += (
            parameters["Bshot_residual"] * parameters["b_nabla2_delta"]
            * cross / number_density
        )
        result["loop"] = sum(result[name] for name in DIAGRAM_NAMES[1:])
        result["spt"] = result["tree"] + result["loop"]
        result["total"] = result["spt"] + result["counterterm"] + result["stochastic"]
        return result


@dataclass
class PowerTemplateSet:
    path: Path
    header: dict[str, Any]
    bins: list[dict[str, Any]]
    k_effective: np.ndarray = field(init=False)
    _compiled: dict[str, dict[str, np.ndarray]] = field(init=False, repr=False)

    @classmethod
    def load(cls, path: Path) -> "PowerTemplateSet":
        header, bins = read_jsonl(path, expected_count=47)
        return cls(path=path, header=header, bins=bins)

    def __post_init__(self) -> None:
        self.k_effective = np.asarray(
            [row["k_effective"] for row in self.bins], dtype=np.float64,
        )
        self._compiled = {}
        for name in ("tree", "P22", "P13", "counterterm_bnabla2_delta"):
            terms: dict[str, np.ndarray] = {}
            for bin_index, row in enumerate(self.bins):
                for monomial, coefficient in row[name].items():
                    if monomial not in terms:
                        terms[monomial] = np.zeros(47, dtype=np.float64)
                    terms[monomial][bin_index] = float(coefficient)
            self._compiled[name] = terms

    def _evaluate(
        self, name: str, parameters: dict[str, float], cache: dict[str, float],
    ) -> np.ndarray:
        result = np.zeros(47, dtype=np.float64)
        for monomial, coefficients in self._compiled[name].items():
            if monomial not in cache:
                cache[monomial] = TemplateSet._monomial_value(monomial, parameters)
            result += cache[monomial] * coefficients
        return result

    def components(
        self, parameters: dict[str, float], number_density: float,
    ) -> dict[str, np.ndarray]:
        cache: dict[str, float] = {"1": 1.0}
        result = {
            name: self._evaluate(name, parameters, cache)
            for name in ("tree", "P22", "P13")
        }
        shape = self._evaluate("counterterm_bnabla2_delta", parameters, cache)
        result["loop"] = result["P22"] + result["P13"]
        result["spt"] = result["tree"] + result["loop"]
        result["counterterm"] = parameters["b_nabla2_delta"] * shape
        pshot = np.asarray([row["stochastic_pshot"] for row in self.bins])
        a0 = np.asarray([row["stochastic_a0"] for row in self.bins])
        result["stochastic"] = (
            parameters["Pshot"] * pshot + parameters["a0_power"] * a0
        ) / number_density
        result["total"] = result["spt"] + result["counterterm"] + result["stochastic"]
        return result


def comparable_ir_pair_header(header: dict[str, Any]) -> dict[str, Any]:
    result = dict(header)
    for key in ("ir_enabled", "ir_metadata", "merged_parts"):
        result.pop(key, None)
    return result


def bispectrum_shell_radial_order(header: dict[str, Any]) -> int:
    quadrature = header.get("shell_quadrature")
    if (
        not isinstance(quadrature, list)
        or not quadrature
        or not isinstance(quadrature[0], int)
        or isinstance(quadrature[0], bool)
        or quadrature[0] < 1
    ):
        raise ValueError("B000 header has an invalid shell radial order")
    radial_order = quadrature[0]
    projection = header.get("shell_projection")
    if projection is not None:
        if not isinstance(projection, dict):
            raise ValueError("B000 header has an invalid shell_projection")
        if projection.get("cheap_radial_order") != radial_order:
            raise ValueError(
                "B000 shell_projection and shell_quadrature radial orders disagree"
            )
    return radial_order


def regulator_requires_paired_power(
    category: str,
    header: dict[str, Any],
    primary_header: dict[str, Any],
    primary_power: PowerTemplateSet | None,
) -> bool:
    if primary_power is None or category == "fftlog":
        return False
    if category == "shell":
        return (
            bispectrum_shell_radial_order(header)
            != bispectrum_shell_radial_order(primary_header)
        )
    return True


def validate_bispectrum_loop_range(
    header: dict[str, Any],
    label: str,
) -> None:
    q_range = header.get("q_range")
    if (
        not isinstance(q_range, list)
        or len(q_range) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in q_range
        )
        or not 0.0 < float(q_range[0]) < float(q_range[1])
    ):
        raise ValueError(f"{label}: invalid direct-loop q range")
    qmax = float(q_range[1])
    stochastic_qmax = header.get("stochastic_qmax")
    if (
        not isinstance(stochastic_qmax, (int, float))
        or isinstance(stochastic_qmax, bool)
        or not math.isfinite(float(stochastic_qmax))
        or float(stochastic_qmax) != qmax
    ):
        raise ValueError(
            f"{label}: stochastic qmax does not match the direct-loop qmax"
        )
    hybrid = header.get("hybrid")
    factorized = (
        hybrid.get("factorized_p13")
        if isinstance(hybrid, dict) else None
    )
    if (
        not isinstance(factorized, dict)
        or factorized.get("q_range") != q_range
    ):
        raise ValueError(
            f"{label}: factorized-P13 q range does not match the direct loop"
        )
    for sector, restoration in (
        ("factorized-P13", factorized.get("uv_tail_restoration")),
        (
            "mixed-stochastic P13",
            header.get("stochastic_p13_uv_tail_restoration"),
        ),
    ):
        sampled_kmax = (
            restoration.get("sampled_kmax")
            if isinstance(restoration, dict) else None
        )
        if (
            restoration is None
            or restoration.get("enabled") is not True
            or not isinstance(sampled_kmax, (int, float))
            or isinstance(sampled_kmax, bool)
            or not math.isfinite(float(sampled_kmax))
            or float(sampled_kmax) < qmax
        ):
            raise ValueError(
                f"{label}: {sector} UV-tail range does not cover qmax"
            )


def validate_exact_lattice_metadata(
    templates: TemplateSet,
    label: str,
) -> None:
    header = templates.header
    validate_bispectrum_loop_range(header, label)
    projection = header.get("shell_projection")
    quadrature = header.get("shell_quadrature")
    if (
        header.get("implementation_version")
        != "eft-v2-exact-joint-lattice-multilevel-v1"
        or not isinstance(projection, dict)
        or not isinstance(quadrature, list)
        or len(quadrature) != 2
    ):
        raise ValueError(f"{label}: not an exact-lattice multilevel template")
    expected_projection = {
        "measure": "exact joint float32 FFT-lattice k1-k2-mu",
        "normalization": "raw N1*N2 estimator pairs",
        "zero_external_legs_removed": True,
        "closing_zero_atoms_removed": True,
        "multilevel": True,
    }
    for key, expected in expected_projection.items():
        if projection.get(key) != expected:
            raise ValueError(
                f"{label}: shell_projection {key!r} is not the release value"
            )
    radial_order = bispectrum_shell_radial_order(header)
    angular_order = quadrature[1]
    expensive_radial = projection.get("expensive_radial_order")
    interpolation_order = projection.get("expensive_k3_interpolation_order")
    for name, value in (
        ("exact angular order", angular_order),
        ("expensive radial order", expensive_radial),
        ("expensive k3 interpolation order", interpolation_order),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{label}: invalid {name}")
    if projection.get("exact_angular_order") != angular_order:
        raise ValueError(
            f"{label}: shell_projection and shell_quadrature angular orders disagree"
        )
    expected_cheap_nodes = radial_order * radial_order * angular_order
    expected_expensive_nodes = (
        expensive_radial * expensive_radial * interpolation_order
    )
    for row_number, row in enumerate(templates.bins):
        location = f"{label}: bin {row_number}"
        expected_values = {
            "exact_joint_lattice_measure": True,
            "multilevel_external_projection": True,
            "hybrid_factorized_analytic_tadpoles": True,
            "fftlog_analytic_convolutions": False,
            "exact_lattice_radial_order": radial_order,
            "exact_lattice_angular_order": angular_order,
            "expensive_lattice_radial_order": expensive_radial,
            "expensive_k3_interpolation_order": interpolation_order,
            "cheap_shell_nodes": expected_cheap_nodes,
            "expensive_shell_nodes": expected_expensive_nodes,
            "shell_nodes": expected_cheap_nodes + expected_expensive_nodes,
        }
        for key, expected in expected_values.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"{location}: {key} does not match the exact-lattice header"
                )
        for key in (
            "zero_external_leg_pairs",
            "closing_zero_pairs",
            "cached_p13_nodes",
            "total_loop_nodes",
        ):
            value = row.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{location}: invalid {key}")
        if (
            row["cached_p13_nodes"] == 0
            or row["total_loop_nodes"] <= row["cached_p13_nodes"]
        ):
            raise ValueError(
                f"{location}: exact-lattice loop-work accounting is incomplete"
            )
        valid_fraction = row.get("exact_lattice_valid_pair_fraction")
        cheap_variation = row.get("exact_lattice_total_variation")
        expensive_variation = row.get("expensive_total_variation")
        lebesgue = row.get("expensive_maximum_sampled_lebesgue")
        numeric_values = (
            valid_fraction,
            cheap_variation,
            expensive_variation,
            lebesgue,
        )
        if not (
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in numeric_values
            )
            and 0.0 < float(valid_fraction) <= 1.0
            and float(cheap_variation) + 2.0e-10
                >= float(valid_fraction)
            and float(expensive_variation) + 2.0e-10
                >= float(valid_fraction)
            and float(lebesgue) >= 1.0 - 2.0e-12
        ):
            raise ValueError(
                f"{location}: invalid exact-lattice weight diagnostics"
            )


def validate_power_shell_metadata(
    templates: PowerTemplateSet,
    label: str,
) -> None:
    header = templates.header
    q_range = header.get("q_range")
    if (
        not isinstance(q_range, list)
        or len(q_range) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in q_range
        )
        or not 0.0 < float(q_range[0]) < float(q_range[1])
    ):
        raise ValueError(f"{label}: invalid power-loop q range")
    restoration = header.get("p13_uv_tail_restoration")
    sampled_kmax = (
        restoration.get("sampled_kmax")
        if isinstance(restoration, dict) else None
    )
    if (
        not isinstance(restoration, dict)
        or restoration.get("enabled") is not True
        or not isinstance(sampled_kmax, (int, float))
        or isinstance(sampled_kmax, bool)
        or not math.isfinite(float(sampled_kmax))
        or float(sampled_kmax) < float(q_range[1])
    ):
        raise ValueError(f"{label}: P13 UV-tail range does not cover qmax")
    requested = header.get("shell_nrad")
    projection = header.get("shell_projection")
    if (
        header.get("implementation_version")
        != "eft-v2-p0-adaptive-estimator-constrained-radial-v2"
        or not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 1
        or not isinstance(projection, dict)
        or not isinstance(projection.get("mode_count_source"), str)
        or not projection["mode_count_source"]
    ):
        raise ValueError(f"{label}: invalid adaptive P0 shell metadata")
    expected_projection = {
        "measure": (
            "integer FFT-lattice radial marginal with authoritative "
            "estimator mode-count boundary allocation"
        ),
        "normalization": "raw shell-mode average",
        "binning": (
            "JAXPower P0 pk_nmodes-constrained lower-inclusive "
            "upper-exclusive baseline"
        ),
        "requested_radial_order": requested,
        "cap_order_to_unique_support": True,
        "box_size_mpc_h": 1000,
        "mesh_size": 256,
    }
    for key, expected in expected_projection.items():
        if projection.get(key) != expected:
            raise ValueError(
                f"{label}: P0 shell_projection {key!r} is not the release value"
            )
    for row_number, row in enumerate(templates.bins):
        actual = row.get("shell_nodes")
        unique_radii = row.get("shell_unique_radius_count")
        mode_count = row.get("shell_mode_count")
        expected_mode_count = row.get(
            "shell_expected_mode_count"
        )
        boundary_adjustment = row.get(
            "shell_boundary_mode_adjustment"
        )
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or not isinstance(unique_radii, int)
            or isinstance(unique_radii, bool)
            or not isinstance(mode_count, int)
            or isinstance(mode_count, bool)
            or not isinstance(expected_mode_count, int)
            or isinstance(expected_mode_count, bool)
            or not isinstance(boundary_adjustment, int)
            or isinstance(boundary_adjustment, bool)
            or unique_radii < 1
            or mode_count < 1
            or expected_mode_count != mode_count
            or actual != min(requested, unique_radii)
        ):
            raise ValueError(
                f"{label}: bin {row_number} has invalid radial support metadata"
            )


def validate_release_inputs(
    ir: TemplateSet,
    noir: TemplateSet,
    ir_power: PowerTemplateSet | None,
    noir_power: PowerTemplateSet | None,
    contract: dict[str, Any],
    contract_path: Path,
    matrix_path: Path,
) -> None:
    if ir.header.get("schema") != "marisa-b-eft-v2-r0-jsonl-v1":
        raise ValueError("IR bispectrum template has the wrong schema")
    if noir.header.get("schema") != "marisa-b-eft-v2-r0-jsonl-v1":
        raise ValueError("no-IR bispectrum template has the wrong schema")
    if ir.header.get("ir_enabled") is not True:
        raise ValueError("IR bispectrum template is not marked ir_enabled=true")
    if noir.header.get("ir_enabled") is not False:
        raise ValueError("no-IR bispectrum template is not marked ir_enabled=false")
    if comparable_ir_pair_header(ir.header) != comparable_ir_pair_header(noir.header):
        raise ValueError("IR and no-IR bispectrum headers are not a matched pair")
    ir_shell_radial_order = bispectrum_shell_radial_order(ir.header)

    expected_contract = sha256(contract_path)
    expected_linear = str(contract["source_hashes"]["linear_power_z1"])
    expected_matrix = str(contract["source_hashes"]["fiducial500_matrix"])
    if sha256(matrix_path) != expected_matrix:
        raise ValueError("measurement matrix hash does not match the EFT-v2 contract")

    template_sets: list[tuple[str, dict[str, Any]]] = [
        ("IR bispectrum", ir.header),
        ("no-IR bispectrum", noir.header),
    ]
    if (ir_power is None) != (noir_power is None):
        raise ValueError("IR and no-IR power templates must be supplied together")
    if ir_power is not None and noir_power is not None:
        if ir_power.header.get("schema") != "marisa-b-eft-v2-r0-p0-jsonl-v1":
            raise ValueError("IR power template has the wrong schema")
        if noir_power.header.get("schema") != "marisa-b-eft-v2-r0-p0-jsonl-v1":
            raise ValueError("no-IR power template has the wrong schema")
        if ir_power.header.get("ir_enabled") is not True:
            raise ValueError("IR power template is not marked ir_enabled=true")
        if noir_power.header.get("ir_enabled") is not False:
            raise ValueError("no-IR power template is not marked ir_enabled=false")
        if (
            comparable_ir_pair_header(ir_power.header)
            != comparable_ir_pair_header(noir_power.header)
        ):
            raise ValueError("IR and no-IR power headers are not a matched pair")
        template_sets.extend([
            ("IR power", ir_power.header),
            ("no-IR power", noir_power.header),
        ])
        if (
            ir_power.header.get("parameter_registry_sha256")
            != ir.header.get("parameter_registry_sha256")
        ):
            raise ValueError("P0 and B000 parameter registries do not match")
        if ir_power.header.get("q_range") != ir.header.get("q_range"):
            raise ValueError("P0 and B000 q ranges do not match")
        if ir_power.header.get("mu_ren") != ir.header.get("mu_ren"):
            raise ValueError("P0 and B000 subtraction scales do not match")
        if ir_power.header.get("shell_nrad") != ir_shell_radial_order:
            raise ValueError("P0 and B000 radial shell rules do not match")

    for label, header in template_sets:
        source_hashes = header.get("source_hashes")
        if not isinstance(source_hashes, dict):
            raise ValueError(f"{label} template does not contain source hashes")
        if source_hashes.get("eft_v2_contract") != expected_contract:
            raise ValueError(f"{label} contract hash does not match the fit contract")
        if source_hashes.get("linear_power") != expected_linear:
            raise ValueError(f"{label} linear-power hash does not match the contract")


def validate_selected_dr_certificate(
    certificate: dict[str, Any],
    contract: dict[str, Any],
    matrix_path: Path,
) -> dict[str, Any]:
    if certificate.get("schema") not in {
        "marisa-b-eft-v2-r0-selected-dr-gate-v1",
        "marisa-b-eft-v2-r0-selected-dr-gate-v2",
    }:
        raise ValueError("--selected-dr-gate has an unsupported schema")
    cases = certificate.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("--selected-dr-gate does not contain cases")
    labels = [row.get("label") for row in cases if isinstance(row, dict)]
    if (
        len(labels) != len(cases)
        or len(labels) != len(set(labels))
        or set(labels) != {"scalene", "equilateral", "squeezed"}
    ):
        raise ValueError(
            "--selected-dr-gate must contain unique scalene, equilateral, "
            "and squeezed cases"
        )

    gate = contract["gates"]
    expected_thresholds = {
        "relative_total_max": float(gate["direct_dr_relative_total"]),
        "absolute_sigma_mean_max": float(gate["direct_dr_absolute_sigma_mean"]),
        "analytic_frequency_step_sigma_mean_max": float(
            gate["direct_dr_absolute_sigma_mean"]
        ),
        "direct_grid_step_sigma_mean_max": float(
            gate["direct_dr_absolute_sigma_mean"]
        ),
    }
    stored_thresholds = certificate.get("gate")
    threshold_match = bool(
        isinstance(stored_thresholds, dict)
        and all(
            stored_thresholds.get(key) == value
            for key, value in expected_thresholds.items()
        )
    )
    case_checks: dict[str, bool] = {}
    for row in cases:
        label = str(row["label"])
        selected = row.get("selected")
        matter = row.get("matter")
        if not isinstance(selected, dict) or not isinstance(matter, dict):
            case_checks[label] = False
            continue
        case_checks[label] = bool(
            abs(float(selected.get("relative_total", math.inf)))
            <= expected_thresholds["relative_total_max"]
            and abs(float(matter.get("relative_total", math.inf)))
            <= expected_thresholds["relative_total_max"]
            and abs(float(selected.get("delta_sigma_mean", math.inf)))
            <= expected_thresholds["absolute_sigma_mean_max"]
            and abs(float(matter.get("delta_sigma_mean", math.inf)))
            <= expected_thresholds["absolute_sigma_mean_max"]
            and abs(float(row.get(
                "analytic_frequency_step_sigma_mean", math.inf,
            )))
            <= expected_thresholds["analytic_frequency_step_sigma_mean_max"]
            and abs(float(row.get("direct_grid_step_sigma_mean", math.inf)))
            <= expected_thresholds["direct_grid_step_sigma_mean_max"]
            and selected.get("pass") is True
            and matter.get("pass") is True
        )
    numeric_pass = bool(
        certificate.get("status") == "pass"
        and threshold_match
        and all(case_checks.values())
    )

    worktree = Path(__file__).resolve().parent.parent
    required_source_paths = {
        "eft_v2_r0_template_driver_binary": (
            worktree / "build/halo_v1/eft_v2_r0_template_driver"
        ),
        **{
            name: worktree / relative
            for name, relative in SELECTED_DR_SOURCE_PATHS.items()
        },
    }
    current_hashes = {
        key: sha256(path) for key, path in required_source_paths.items()
    }
    current_hashes["linear_power"] = str(
        contract["source_hashes"]["linear_power_z1"]
    )
    input_hashes = certificate.get("input_hashes")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    source_mismatches = {
        key: {
            "certificate": input_hashes.get(key),
            "current": current_hash,
        }
        for key, current_hash in current_hashes.items()
        if input_hashes.get(key) != current_hash
    }
    measurement = certificate.get("measurement_error_scale")
    matrix_hash = sha256(matrix_path)
    matrix_provenance_pass = bool(
        isinstance(measurement, dict)
        and measurement.get("matrix_sha256") == matrix_hash
        and matrix_hash == contract["source_hashes"]["fiducial500_matrix"]
    )
    provenance_pass = bool(
        not source_mismatches and matrix_provenance_pass
    )
    return {
        "numeric_pass": numeric_pass,
        "provenance_pass": provenance_pass,
        "pass": bool(numeric_pass and provenance_pass),
        "threshold_match": threshold_match,
        "case_checks": case_checks,
        "matrix_provenance_pass": matrix_provenance_pass,
        "source_mismatches": source_mismatches,
        "required_source_hashes": current_hashes,
    }


def validate_regulator_inputs(
    label: str,
    category: str,
    templates: TemplateSet,
    power_templates: PowerTemplateSet | None,
    power_path: Path | None,
    primary: TemplateSet,
    primary_power: PowerTemplateSet | None,
    contract: dict[str, Any],
    contract_path: Path,
) -> None:
    header = templates.header
    if header.get("schema") != "marisa-b-eft-v2-r0-jsonl-v1":
        raise ValueError(f"{label}: regulator bispectrum has the wrong schema")
    if header.get("ir_enabled") is not True:
        raise ValueError(f"{label}: regulator bispectrum must be IR enabled")
    source_hashes = header.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError(f"{label}: regulator bispectrum lacks source hashes")
    if source_hashes.get("eft_v2_contract") != sha256(contract_path):
        raise ValueError(f"{label}: regulator bispectrum contract hash mismatch")
    if source_hashes.get("linear_power") != contract["source_hashes"]["linear_power_z1"]:
        raise ValueError(f"{label}: regulator bispectrum linear-power hash mismatch")
    if (
        header.get("parameter_registry_sha256")
        != primary.header.get("parameter_registry_sha256")
    ):
        raise ValueError(f"{label}: regulator bispectrum registry mismatch")
    variant_hybrid = header.get("hybrid")
    primary_hybrid = primary.header.get("hybrid")
    if not isinstance(variant_hybrid, dict) or not isinstance(primary_hybrid, dict):
        raise ValueError(f"{label}: regulator and primary must use the hybrid evaluator")
    bispectrum_shell_radial_order(header)
    bispectrum_shell_radial_order(primary.header)

    paired_power_required = regulator_requires_paired_power(
        category, header, primary.header, primary_power,
    )
    if paired_power_required and power_path is None:
        raise ValueError(
            f"{label}: category {category} requires a complete paired P0 template"
        )
    if power_path is not None:
        if power_templates is None or primary_power is None:
            raise ValueError(f"{label}: paired P0 supplied without primary P0")
        power_header = power_templates.header
        if power_header.get("schema") != "marisa-b-eft-v2-r0-p0-jsonl-v1":
            raise ValueError(f"{label}: regulator power has the wrong schema")
        if power_header.get("ir_enabled") is not True:
            raise ValueError(f"{label}: regulator power must be IR enabled")
        power_source_hashes = power_header.get("source_hashes")
        if not isinstance(power_source_hashes, dict):
            raise ValueError(f"{label}: regulator power lacks source hashes")
        if power_source_hashes.get("eft_v2_contract") != sha256(contract_path):
            raise ValueError(f"{label}: regulator power contract hash mismatch")
        if (
            power_source_hashes.get("linear_power")
            != contract["source_hashes"]["linear_power_z1"]
        ):
            raise ValueError(f"{label}: regulator power linear-power hash mismatch")
        if (
            power_header.get("parameter_registry_sha256")
            != header.get("parameter_registry_sha256")
        ):
            raise ValueError(f"{label}: regulator P0/B000 registry mismatch")
        if power_header.get("q_range") != header.get("q_range"):
            raise ValueError(f"{label}: regulator P0/B000 q-range mismatch")
        if power_header.get("mu_ren") != header.get("mu_ren"):
            raise ValueError(f"{label}: regulator P0/B000 subtraction-scale mismatch")
        if power_header.get("ir_metadata") != header.get("ir_metadata"):
            raise ValueError(f"{label}: regulator P0/B000 IR prescription mismatch")
        if (
            power_header.get("shell_nrad")
            != bispectrum_shell_radial_order(header)
        ):
            raise ValueError(f"{label}: regulator P0/B000 radial shell mismatch")

    def unchanged(*keys: str) -> bool:
        return all(header.get(key) == primary.header.get(key) for key in keys)

    if category == "qmax":
        changed = header.get("q_range") != primary.header.get("q_range")
        consistent = unchanged(
            "mu_ren", "ir_metadata", "shell_quadrature", "shell_projection",
            "loop_quadrature", "stochastic_quadrature",
        )
        power_changed = bool(
            power_path is not None
            and power_templates is not None
            and primary_power is not None
            and power_templates.header.get("q_range")
            != primary_power.header.get("q_range")
        )
    elif category == "mu":
        changed = header.get("mu_ren") != primary.header.get("mu_ren")
        consistent = unchanged(
            "q_range", "ir_metadata", "shell_quadrature", "shell_projection",
            "loop_quadrature", "stochastic_quadrature",
        )
        power_changed = bool(
            power_path is not None
            and power_templates is not None
            and primary_power is not None
            and power_templates.header.get("mu_ren")
            != primary_power.header.get("mu_ren")
        )
    elif category == "ir":
        changed = header.get("ir_metadata") != primary.header.get("ir_metadata")
        consistent = unchanged(
            "q_range", "mu_ren", "shell_quadrature", "shell_projection",
            "loop_quadrature", "stochastic_quadrature",
        )
        power_changed = bool(
            power_path is not None
            and power_templates is not None
            and primary_power is not None
            and power_templates.header.get("ir_metadata")
            != primary_power.header.get("ir_metadata")
        )
    elif category == "integration":
        changed = any([
            header.get("loop_quadrature")
            != primary.header.get("loop_quadrature"),
            header.get("loop_angular_nodes")
            != primary.header.get("loop_angular_nodes"),
            header.get("stochastic_quadrature")
            != primary.header.get("stochastic_quadrature"),
            variant_hybrid.get("factorized_p13", {}).get("quadrature")
            != primary_hybrid.get("factorized_p13", {}).get("quadrature"),
        ])
        consistent = unchanged(
            "q_range", "mu_ren", "ir_metadata",
            "shell_quadrature", "shell_projection",
        )
        power_changed = bool(
            power_path is not None
            and power_templates is not None
            and primary_power is not None
            and power_templates.header.get("loop_quadrature")
            != primary_power.header.get("loop_quadrature")
        )
    elif category == "shell":
        changed = any([
            header.get("shell_quadrature")
            != primary.header.get("shell_quadrature"),
            header.get("shell_projection")
            != primary.header.get("shell_projection"),
        ])
        consistent = unchanged(
            "q_range", "mu_ren", "ir_metadata",
            "loop_quadrature", "stochastic_quadrature",
        )
        power_changed = bool(
            power_path is not None
            and power_templates is not None
            and primary_power is not None
            and power_templates.header.get("shell_nrad")
            != primary_power.header.get("shell_nrad")
        )
    elif category == "fftlog":
        changed = any([
            variant_hybrid.get("b411_fftlog")
            != primary_hybrid.get("b411_fftlog"),
            variant_hybrid.get("b411_regulator")
            != primary_hybrid.get("b411_regulator"),
        ])
        consistent = unchanged(
            "q_range", "mu_ren", "ir_metadata",
            "shell_quadrature", "shell_projection",
            "loop_quadrature", "stochastic_quadrature",
        )
        power_changed = True
    else:
        raise ValueError(f"{label}: unknown regulator category {category!r}")
    if not changed:
        raise ValueError(
            f"{label}: labelled {category} variant does not change that setting"
        )
    if not consistent:
        raise ValueError(
            f"{label}: labelled {category} variant changes an unrelated setting"
        )
    if paired_power_required and not power_changed:
        raise ValueError(
            f"{label}: paired P0 does not implement the labelled {category} change"
        )


@dataclass
class DataSet:
    mean: np.ndarray
    covariance_mean: np.ndarray
    error_mean: np.ndarray
    k_pair: np.ndarray
    edges: np.ndarray
    power_mean: np.ndarray
    power_error_mean: np.ndarray
    power_k: np.ndarray
    power_edges: np.ndarray
    power_nmodes: np.ndarray
    joint_mean_power_then_bispectrum: np.ndarray
    joint_covariance_mean_power_then_bispectrum: np.ndarray
    mock_count: int

    @classmethod
    def load(cls, path: Path) -> "DataSet":
        with np.load(path, allow_pickle=False) as values:
            mean = np.asarray(values["fiducial_pre_B000_mean"], dtype=np.float64)
            covariance_realizations = np.asarray(
                values["fiducial_pre_B000_cov"], dtype=np.float64,
            )
            k_pair = np.asarray(values["b000_k"], dtype=np.float64)
            edges = np.asarray(values["b000_k_edges"], dtype=np.float64)
            bispectrum_samples = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
            power_samples = np.asarray(values["fiducial_pre_P0"], dtype=np.float64)
            power_mean = np.asarray(values["fiducial_pre_P0_mean"], dtype=np.float64)
            power_k = np.asarray(values["pk_k"], dtype=np.float64)
            power_edges = np.asarray(values["pk_edges"], dtype=np.float64)
            power_nmodes = np.asarray(values["pk_nmodes"])
        mock_count = int(bispectrum_samples.shape[0])
        covariance = covariance_realizations / float(mock_count)
        if (
            mean.shape != (120,)
            or covariance_realizations.shape != (120, 120)
            or covariance.shape != (120, 120)
            or k_pair.shape != (120, 2)
            or edges.shape != (120, 2, 2)
            or power_mean.shape != (47,)
            or power_k.shape != (47,)
            or power_edges.shape != (48,)
            or power_nmodes.shape != (47,)
        ):
            raise ValueError("unexpected R0 measurement shapes")
        if (
            mock_count < 3
            or bispectrum_samples.shape != (mock_count, 120)
            or power_samples.shape != (mock_count, 47)
        ):
            raise ValueError("unexpected R0 realization matrix shapes")
        joint_samples = np.column_stack((power_samples, bispectrum_samples))
        joint_covariance = (
            np.cov(joint_samples, rowvar=False, ddof=1) / float(mock_count)
        )
        if (
            not np.array_equal(power_mean, np.mean(power_samples, axis=0))
            or not np.array_equal(mean, np.mean(bispectrum_samples, axis=0))
        ):
            raise ValueError(
                "stored R0 means do not match the realization matrices"
            )
        covariance_difference = (
            joint_covariance[47:, 47:] - covariance
        )
        covariance_scale = float(np.linalg.norm(covariance))
        if (
            covariance_scale <= 0.0
            or float(np.linalg.norm(covariance_difference))
            > 1.0e-12 * covariance_scale
        ):
            raise ValueError(
                "stored B000 covariance does not match the realizations"
            )
        joint_mean = np.concatenate((power_mean, mean))
        return cls(
            mean, covariance, np.sqrt(np.diag(covariance)), k_pair, edges,
            power_mean, np.sqrt(np.diag(joint_covariance[:47, :47])), power_k,
            power_edges, power_nmodes,
            joint_mean, joint_covariance,
            mock_count,
        )

    def mask(self, kmax: float) -> np.ndarray:
        # B_000 in the Sugiyama basis stores two radial modes; the closing side is
        # integrated over and has no independently cuttable estimator bin.
        return np.max(self.k_pair, axis=1) <= kmax + 1.0e-12

    def power_mask(self, kmax: float) -> np.ndarray:
        return self.power_k <= kmax + 1.0e-12


def validate_bispectrum_geometry(
    templates: TemplateSet,
    data: DataSet,
    label: str,
) -> None:
    try:
        template_edges = np.asarray(
            [row["edges"] for row in templates.bins],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid B000 template edges") from error
    expected_edges = np.asarray(data.edges, dtype=np.float64).reshape(120, 4)
    if (
        template_edges.shape != expected_edges.shape
        or not np.allclose(
            template_edges, expected_edges,
            rtol=0.0, atol=2.0e-15,
        )
    ):
        maximum_error = (
            float(np.max(np.abs(template_edges - expected_edges)))
            if template_edges.shape == expected_edges.shape else math.inf
        )
        raise ValueError(
            f"{label}: B000 shell edges do not match the measurement "
            f"(maximum absolute error {maximum_error})"
        )


def validate_power_geometry(
    templates: PowerTemplateSet,
    data: DataSet,
    label: str,
) -> None:
    try:
        template_edges = np.asarray(
            [row["edges"] for row in templates.bins],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid P0 template edges") from error
    expected_edges = np.column_stack(
        (data.power_edges[:-1], data.power_edges[1:])
    )
    if (
        template_edges.shape != expected_edges.shape
        or not np.allclose(
            template_edges, expected_edges,
            rtol=0.0, atol=2.0e-15,
        )
    ):
        raise ValueError(f"{label}: P0 shell edges do not match the measurement")
    template_mode_counts = np.asarray(
        [row.get("shell_mode_count") for row in templates.bins]
    )
    if (
        template_mode_counts.shape != data.power_nmodes.shape
        or not np.array_equal(
            template_mode_counts, data.power_nmodes,
        )
    ):
        raise ValueError(
            f"{label}: P0 shell mode counts do not match the measurement"
        )
    lower = expected_edges[:, 0]
    upper = expected_edges[:, 1]
    if (
        templates.k_effective.shape != data.power_k.shape
        or not np.all(np.isfinite(templates.k_effective))
        or not np.all(templates.k_effective > lower)
        or not np.all(templates.k_effective < upper)
    ):
        raise ValueError(
            f"{label}: P0 effective k is not finite and inside its shell"
        )


@dataclass
class Prior:
    mean: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    whitener: np.ndarray


@dataclass(frozen=True)
class FitProtocol:
    mock_count: int
    primary_likelihood: str
    robustness_likelihoods: tuple[str, ...]
    correlation_jitter: float
    b1_robustness_prior_sigmas: tuple[float, ...]
    b_rec: float


def covariance_cholesky(
    covariance: np.ndarray,
    correlation_jitter: float,
) -> np.ndarray:
    """Cholesky factor after a dimensionless diagonal jitter in correlation space."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("covariance must be a finite square matrix")
    if not math.isfinite(correlation_jitter) or correlation_jitter < 0.0:
        raise ValueError("correlation jitter must be finite and non-negative")
    variance = np.diag(covariance)
    if np.any(variance <= 0.0):
        raise ValueError("covariance diagonal must be positive")
    sigma = np.sqrt(variance)
    correlation = covariance / np.outer(sigma, sigma)
    correlation = 0.5 * (correlation + correlation.T)
    correlation_cholesky = np.linalg.cholesky(
        correlation
        + correlation_jitter * np.eye(covariance.shape[0], dtype=np.float64)
    )
    return sigma[:, None] * correlation_cholesky


def hartlap_factor(mock_count: int, likelihood_dimension: int) -> float:
    if mock_count <= likelihood_dimension + 2:
        raise ValueError(
            "Hartlap correction requires mock_count > likelihood_dimension + 2"
        )
    return float(
        (mock_count - likelihood_dimension - 2.0) / (mock_count - 1.0)
    )


def sellentin_heavens_data_objective(
    t_squared: float,
    mock_count: int,
) -> float:
    if mock_count <= 1:
        raise ValueError("Sellentin-Heavens likelihood requires at least two mocks")
    if not math.isfinite(t_squared) or t_squared < 0.0:
        raise ValueError("T-squared must be finite and non-negative")
    return float(
        mock_count * math.log1p(t_squared / (mock_count - 1.0))
    )


def likelihood_weight(
    likelihood: str,
    t_squared: float,
    mock_count: int,
    likelihood_dimension: int,
) -> float:
    if likelihood == RAW_GAUSSIAN_LIKELIHOOD:
        return 1.0
    if likelihood == HARTLAP_GAUSSIAN_LIKELIHOOD:
        return hartlap_factor(mock_count, likelihood_dimension)
    if likelihood == SELLENTIN_HEAVENS_LIKELIHOOD:
        if mock_count <= 1:
            raise ValueError(
                "Sellentin-Heavens likelihood requires at least two mocks"
            )
        return float(mock_count / (mock_count - 1.0 + t_squared))
    raise ValueError(f"unsupported likelihood {likelihood!r}")


def likelihood_data_objective(
    likelihood: str,
    t_squared: float,
    mock_count: int,
    likelihood_dimension: int,
) -> float:
    if likelihood == SELLENTIN_HEAVENS_LIKELIHOOD:
        return sellentin_heavens_data_objective(t_squared, mock_count)
    return (
        likelihood_weight(
            likelihood, t_squared, mock_count, likelihood_dimension,
        )
        * t_squared
    )


def likelihood_residual_scale(
    likelihood: str,
    t_squared: float,
    mock_count: int,
    likelihood_dimension: int,
) -> float:
    objective = likelihood_data_objective(
        likelihood, t_squared, mock_count, likelihood_dimension,
    )
    if t_squared <= np.finfo(np.float64).eps:
        return math.sqrt(
            likelihood_weight(
                likelihood, 0.0, mock_count, likelihood_dimension,
            )
        )
    return math.sqrt(objective / t_squared)


def solve_weighted_linear(
    design: np.ndarray,
    target: np.ndarray,
    prior_design: np.ndarray,
    prior_target: np.ndarray,
    weight: float,
) -> np.ndarray:
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("linear-profile weight must be finite and non-negative")
    augmented_design = np.vstack((
        math.sqrt(weight) * design,
        prior_design,
    ))
    augmented_target = np.concatenate((
        math.sqrt(weight) * target,
        prior_target,
    ))
    return np.linalg.lstsq(
        augmented_design, augmented_target, rcond=1.0e-12,
    )[0]


def profile_sellentin_heavens_linear(
    design: np.ndarray,
    target: np.ndarray,
    prior_design: np.ndarray,
    prior_target: np.ndarray,
    mock_count: int,
) -> tuple[np.ndarray, float, float]:
    """Exactly profile a linear block under the Sellentin-Heavens likelihood."""
    if mock_count <= 1:
        raise ValueError("Sellentin-Heavens likelihood requires at least two mocks")
    maximum_weight = float(mock_count / (mock_count - 1.0))

    def solve(weight: float) -> np.ndarray:
        return solve_weighted_linear(
            design, target, prior_design, prior_target, weight,
        )

    def fixed_point(weight: float) -> float:
        fitted = solve(weight)
        residual = design @ fitted - target
        t_squared = float(residual @ residual)
        return weight - mock_count / (mock_count - 1.0 + t_squared)

    lower = fixed_point(0.0)
    upper = fixed_point(maximum_weight)
    tolerance = 64.0 * np.finfo(np.float64).eps
    if lower > tolerance or upper < -tolerance:
        raise RuntimeError(
            "failed to bracket the Sellentin-Heavens profile weight: "
            f"lower={lower}, upper={upper}"
        )
    if abs(lower) <= tolerance:
        weight = 0.0
    elif abs(upper) <= tolerance:
        weight = maximum_weight
    else:
        weight = float(brentq(
            fixed_point, 0.0, maximum_weight,
            xtol=1.0e-14, rtol=1.0e-14,
        ))
    fitted = solve(weight)
    residual = design @ fitted - target
    t_squared = float(residual @ residual)
    return fitted, weight, t_squared


def load_prior(contract_path: Path) -> tuple[dict[str, Any], Prior, float]:
    contract = read_json_object(contract_path)
    entries = {
        row["id"]: row
        for sector in ("bias_parameters", "counterterm_parameters", "stochastic_parameters")
        for row in contract[sector]
    }
    mean = np.asarray([entries[name]["prior"]["mean"] for name in FIT_NAMES], dtype=np.float64)
    sigma = np.asarray([entries[name]["prior"]["sigma"] for name in FIT_NAMES], dtype=np.float64)
    covariance = np.diag(sigma**2)
    derivative_names = ["abar0_mixed", "a3_mixed", "a4_mixed"]
    derivative_indices = [FIT_NAMES.index(name) for name in derivative_names]
    raw_derivative_names = derivative_names + ["a5_mixed"]
    raw_covariance = np.diag([
        float(entries[name]["prior"]["sigma"]) ** 2
        for name in raw_derivative_names
    ])
    projection = np.asarray([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, -1.0],
    ])
    projected_covariance = projection @ raw_covariance @ projection.T
    covariance[np.ix_(derivative_indices, derivative_indices)] = projected_covariance
    sigma = np.sqrt(np.diag(covariance))
    whitener = solve_triangular(
        np.linalg.cholesky(covariance), np.eye(len(FIT_NAMES)), lower=True,
    )
    nbar = float(contract["observable"]["number_density_h3_mpc3"])
    return contract, Prior(mean, sigma, covariance, whitener), nbar


def load_fit_protocol(
    contract: dict[str, Any],
    data: DataSet,
) -> FitProtocol:
    if contract.get("schema") != "marisa-b-eft-v2-contract-v2":
        raise ValueError("EFT-v2 fit protocol requires contract schema v2")
    try:
        protocol = contract["fit_protocol"]
        b1 = protocol["b1"]
        covariance = protocol["covariance_likelihood"]
        mock_count = int(covariance["mock_count"])
        primary_likelihood = str(covariance["primary"])
        robustness_likelihoods = tuple(
            str(value) for value in covariance["robustness"]
        )
        correlation_jitter = float(covariance["correlation_jitter"])
        b1_robustness_prior_sigmas = tuple(
            float(value) for value in b1["robustness_prior_sigmas"]
        )
        b_rec = float(b1["b_rec"]["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid EFT-v2 fit_protocol") from error
    if mock_count != data.mock_count:
        raise ValueError(
            f"fit protocol declares {mock_count} mocks but matrix contains "
            f"{data.mock_count}"
        )
    b1_entries = [
        row for row in contract["bias_parameters"] if row.get("id") == "b1"
    ]
    if len(b1_entries) != 1:
        raise ValueError("contract must contain exactly one b1 parameter")
    b1_prior = b1_entries[0].get("prior")
    if (
        not isinstance(b1_prior, dict)
        or b1_prior.get("kind") != "normal"
        or float(b1_prior.get("mean", math.nan)) != 1.0
        or float(b1_prior.get("sigma", math.nan)) != 2.0
        or "fixed" in b1_entries[0]
    ):
        raise ValueError("R0 primary b1 prior must be an unfixed N(1,2^2)")
    if primary_likelihood != RAW_GAUSSIAN_LIKELIHOOD:
        raise ValueError("R0 primary likelihood must be raw_gaussian")
    if (
        robustness_likelihoods
        != (
            HARTLAP_GAUSSIAN_LIKELIHOOD,
            SELLENTIN_HEAVENS_LIKELIHOOD,
        )
    ):
        raise ValueError("R0 requires Hartlap and Sellentin-Heavens checks")
    if correlation_jitter != 1.0e-12:
        raise ValueError("R0 correlation_jitter must equal 1e-12")
    if b1_robustness_prior_sigmas != (1.0, 4.0):
        raise ValueError("R0 b1 robustness prior widths must be one and four")
    if b1.get("role") != "shared_fitted_P0_B000_parameter":
        raise ValueError("b1 must remain a shared fitted P0+B000 parameter")
    if b1["b_rec"].get("likelihood_role") != "none":
        raise ValueError("b_rec must not enter the R0 likelihood")
    if b_rec != 2.7340475186190334:
        raise ValueError("unexpected reconstruction-bias provenance value")
    return FitProtocol(
        mock_count=mock_count,
        primary_likelihood=primary_likelihood,
        robustness_likelihoods=robustness_likelihoods,
        correlation_jitter=correlation_jitter,
        b1_robustness_prior_sigmas=b1_robustness_prior_sigmas,
        b_rec=b_rec,
    )


def prior_with_b1_sigma(prior: Prior, sigma: float) -> Prior:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("b1 prior sigma must be finite and positive")
    index = FIT_NAMES.index("b1")
    covariance = prior.covariance.copy()
    covariance[index, :] = 0.0
    covariance[:, index] = 0.0
    covariance[index, index] = sigma**2
    marginal_sigma = np.sqrt(np.diag(covariance))
    whitener = solve_triangular(
        np.linalg.cholesky(covariance),
        np.eye(len(FIT_NAMES)),
        lower=True,
    )
    return Prior(
        mean=prior.mean.copy(),
        sigma=marginal_sigma,
        covariance=covariance,
        whitener=whitener,
    )


@dataclass
class FitResult:
    kmax: float
    mask: np.ndarray
    likelihood_point_count: int
    likelihood: str
    mock_count: int
    x: np.ndarray
    covariance: np.ndarray
    components: dict[str, np.ndarray]
    power_components: dict[str, np.ndarray] | None
    residual: np.ndarray
    whitened: np.ndarray
    power_residual: np.ndarray | None
    power_whitened: np.ndarray | None
    chi2: float
    data_likelihood_objective: float
    likelihood_weight_at_mode: float
    power_chi2: float
    bispectrum_conditional_chi2: float
    dof: float
    effective_parameter_count: float
    effective_rank: int
    p_value: float | None
    objective: float
    optimizer_success: bool
    optimizer_message: str
    multistart_objectives: np.ndarray
    max_pull: float
    max_bispectrum_pull: float
    max_power_pull: float | None
    max_sign_run: int
    max_bispectrum_sign_run: int
    max_power_sign_run: int | None
    posterior_to_prior_sigma_ratio: np.ndarray
    singular_values: np.ndarray
    model_jacobian: np.ndarray
    power_model_jacobian: np.ndarray | None

    def report(self) -> dict[str, Any]:
        errors = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        rank_dof = self.likelihood_point_count - self.effective_rank
        return {
            "kmax_h_mpc": self.kmax,
            "bin_count": int(np.count_nonzero(self.mask)),
            "joint_likelihood_point_count": self.likelihood_point_count,
            "likelihood": {
                "kind": self.likelihood,
                "mock_count": self.mock_count,
                "raw_hotelling_t_squared": self.chi2,
                "data_objective_minus_2_log_likelihood_up_to_constant": (
                    self.data_likelihood_objective
                ),
                "local_data_weight": self.likelihood_weight_at_mode,
                "frequentist_p_value": self.p_value,
                "gate_role": (
                    "primary"
                    if self.likelihood == RAW_GAUSSIAN_LIKELIHOOD
                    else "diagnostic_only"
                ),
            },
            "parameters": {
                name: {"value": float(self.x[index]), "posterior_sigma": float(errors[index])}
                for index, name in enumerate(FIT_NAMES)
            },
            "chi2": self.chi2,
            "power_chi2": self.power_chi2,
            "bispectrum_chi2_conditional_on_power": self.bispectrum_conditional_chi2,
            "dof": self.dof,
            "chi2_per_dof": self.chi2 / self.dof if self.dof > 0 else math.nan,
            "p_value": self.p_value,
            "optimizer": {
                "success": self.optimizer_success,
                "message": self.optimizer_message,
                "multistart_objectives": [
                    float(value) for value in self.multistart_objectives
                ],
            },
            "rank_based_dof_diagnostic": rank_dof,
            "rank_based_chi2_per_dof_diagnostic": (
                self.chi2 / rank_dof if rank_dof > 0 else math.nan
            ),
            "rank_based_p_value_diagnostic": (
                float(chi2_distribution.sf(self.chi2, rank_dof))
                if (
                    self.likelihood == RAW_GAUSSIAN_LIKELIHOOD
                    and rank_dof > 0
                )
                else None
            ),
            "effective_model_rank": self.effective_rank,
            "whitened_jacobian_condition_number": (
                float(self.singular_values[0] / self.singular_values[-1])
                if self.singular_values.size and self.singular_values[-1] > 0 else math.inf
            ),
            "whitened_jacobian_singular_values": [float(value) for value in self.singular_values],
            "effective_parameter_count_with_gaussian_priors": self.effective_parameter_count,
            "max_abs_pull": self.max_pull,
            "max_abs_bispectrum_pull": self.max_bispectrum_pull,
            "max_abs_power_pull": self.max_power_pull,
            "max_same_sign_whitened_run": self.max_sign_run,
            "max_bispectrum_same_sign_whitened_run": (
                self.max_bispectrum_sign_run
            ),
            "max_power_same_sign_whitened_run": self.max_power_sign_run,
            "posterior_to_marginal_prior_sigma_ratio": {
                name: float(value) for name, value in zip(FIT_NAMES, self.posterior_to_prior_sigma_ratio)
            },
        }


def parameter_dict(x: np.ndarray) -> dict[str, float]:
    result = {
        name: 0.0
        for name in BIAS_NAMES + COUNTERTERM_NAMES
        + STOCHASTIC_POWER_NAMES + STOCHASTIC_B_NAMES
    }
    result.update({name: float(value) for name, value in zip(FIT_NAMES, x)})
    return result


def longest_sign_run(values: np.ndarray) -> int:
    longest = current = 0
    previous = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign != 0 and sign == previous:
            current += 1
        elif sign != 0:
            current = 1
        else:
            current = 0
        previous = sign
        longest = max(longest, current)
    return longest


def regulator_category(label: str) -> str:
    normalized = label.lower().replace("_", "-")
    if "qmax" in normalized:
        return "qmax"
    if normalized.startswith("mu") or "mu-ren" in normalized:
        return "mu"
    if "fftlog" in normalized or "nu=" in normalized:
        return "fftlog"
    if "integration" in normalized or "loop-grid" in normalized:
        return "integration"
    if "shell" in normalized:
        return "shell"
    return "ir"


def regulator_configuration_signature(
    category: str, header: dict[str, Any],
) -> dict[str, Any]:
    hybrid = header.get("hybrid")
    if not isinstance(hybrid, dict):
        hybrid = {}
    if category == "qmax":
        return {"q_range": header.get("q_range")}
    if category == "mu":
        return {"mu_ren": header.get("mu_ren")}
    if category == "ir":
        return {"ir_metadata": header.get("ir_metadata")}
    if category == "integration":
        return {
            "loop_quadrature": header.get("loop_quadrature"),
            "loop_angular_nodes": header.get("loop_angular_nodes"),
            "stochastic_quadrature": header.get("stochastic_quadrature"),
            "factorized_p13_quadrature": hybrid.get(
                "factorized_p13", {},
            ).get("quadrature"),
        }
    if category == "shell":
        return {
            "shell_quadrature": header.get("shell_quadrature"),
            "shell_projection": header.get("shell_projection"),
        }
    if category == "fftlog":
        return {
            "b411_fftlog": hybrid.get("b411_fftlog"),
            "b411_regulator": hybrid.get("b411_regulator"),
        }
    raise ValueError(f"unknown regulator category {category!r}")


def fit_one(
    templates: TemplateSet,
    power_templates: PowerTemplateSet | None,
    data: DataSet,
    prior: Prior,
    number_density: float,
    kmax: float,
    multistart: int,
    rng: np.random.Generator,
    likelihood: str = RAW_GAUSSIAN_LIKELIHOOD,
    mock_count: int | None = None,
    correlation_jitter: float = 1.0e-12,
) -> FitResult:
    if likelihood not in SUPPORTED_LIKELIHOODS:
        raise ValueError(f"unsupported likelihood {likelihood!r}")
    if mock_count is None:
        mock_count = data.mock_count
    if mock_count != data.mock_count:
        raise ValueError(
            f"fit mock_count {mock_count} does not match matrix {data.mock_count}"
        )
    mask = data.mask(kmax)
    indices = np.flatnonzero(mask)
    power_indices = (
        np.flatnonzero(data.power_mask(POWER_KMAX_H_MPC))
        if power_templates else np.empty(0, dtype=int)
    )
    if power_templates:
        selected_indices = np.concatenate((power_indices, 47 + indices))
        covariance = data.joint_covariance_mean_power_then_bispectrum[
            np.ix_(selected_indices, selected_indices)
        ]
        target = data.joint_mean_power_then_bispectrum[selected_indices]
    else:
        covariance = data.covariance_mean[np.ix_(indices, indices)]
        target = data.mean[indices]
    likelihood_point_count = target.size
    cholesky = covariance_cholesky(
        covariance, correlation_jitter,
    )

    name_to_index = {name: index for index, name in enumerate(FIT_NAMES)}
    nonlinear_indices = np.asarray(
        [name_to_index[name] for name in NONLINEAR_FIT_NAMES], dtype=int,
    )
    linear_indices = np.asarray(
        [name_to_index[name] for name in LINEAR_FIT_NAMES], dtype=int,
    )

    def model(
        x: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        parameters = parameter_dict(x)
        components = templates.components(parameters, number_density)
        if power_templates:
            power_components = power_templates.components(parameters, number_density)
            prediction = np.concatenate((
                power_components["total"][power_indices], components["total"][indices],
            ))
        else:
            power_components = None
            prediction = components["total"][indices]
        return prediction, components, power_components

    def full_residual(x: np.ndarray) -> np.ndarray:
        prediction, _, _ = model(x)
        data_part = solve_triangular(cholesky, prediction - target, lower=True)
        t_squared = float(data_part @ data_part)
        scale = likelihood_residual_scale(
            likelihood, t_squared, mock_count, likelihood_point_count,
        )
        return np.concatenate((
            scale * data_part,
            prior.whitener @ (x - prior.mean),
        ))

    def profile_linear(non_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = prior.mean.copy()
        x[nonlinear_indices] = non_linear
        x[linear_indices] = 0.0
        base, _, _ = model(x)
        design = np.empty((likelihood_point_count, linear_indices.size), dtype=np.float64)
        for column, parameter_index in enumerate(linear_indices):
            unit = x.copy()
            unit[parameter_index] = 1.0
            design[:, column] = model(unit)[0] - base
        whitened_design = solve_triangular(cholesky, design, lower=True)
        whitened_target = solve_triangular(cholesky, target - base, lower=True)
        prior_design = prior.whitener[:, linear_indices]
        prior_target = -prior.whitener @ (x - prior.mean)
        if likelihood == SELLENTIN_HEAVENS_LIKELIHOOD:
            linear, _, _ = profile_sellentin_heavens_linear(
                whitened_design,
                whitened_target,
                prior_design,
                prior_target,
                mock_count,
            )
        else:
            linear = solve_weighted_linear(
                whitened_design,
                whitened_target,
                prior_design,
                prior_target,
                likelihood_weight(
                    likelihood, 0.0, mock_count, likelihood_point_count,
                ),
            )
        x[linear_indices] = linear
        return x, full_residual(x)

    def profiled_residual(non_linear: np.ndarray) -> np.ndarray:
        return profile_linear(non_linear)[1]

    starts = [prior.mean[nonlinear_indices].copy()]
    for _ in range(max(1, multistart) - 1):
        starts.append(
            prior.mean[nonlinear_indices]
            + 0.35 * prior.sigma[nonlinear_indices]
            * rng.normal(size=nonlinear_indices.size)
        )
    solutions = [
        least_squares(
            profiled_residual, start, method="trf",
            x_scale=prior.sigma[nonlinear_indices],
            max_nfev=3000, ftol=2.0e-10, xtol=2.0e-10, gtol=2.0e-10,
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda item: float(np.dot(item.fun, item.fun)))
    fitted_x, fitted_residual = profile_linear(solution.x)
    prediction, components, power_components = model(fitted_x)
    joint_raw_residual = prediction - target
    joint_whitened = solve_triangular(cholesky, joint_raw_residual, lower=True)
    raw_residual = components["total"][indices] - data.mean[indices]
    whitened = joint_whitened[power_indices.size:]
    power_raw_residual = (
        power_components["total"][power_indices] - data.power_mean[power_indices]
        if power_components is not None else None
    )
    power_whitened = (
        joint_whitened[:power_indices.size]
        if power_components is not None else None
    )
    data_chi2 = float(np.dot(joint_whitened, joint_whitened))
    data_objective = likelihood_data_objective(
        likelihood, data_chi2, mock_count, likelihood_point_count,
    )
    local_likelihood_weight = likelihood_weight(
        likelihood, data_chi2, mock_count, likelihood_point_count,
    )
    power_chi2 = float(np.dot(
        joint_whitened[:power_indices.size], joint_whitened[:power_indices.size],
    ))
    bispectrum_conditional_chi2 = float(np.dot(whitened, whitened))
    # The profiled optimizer Jacobian only spans nonlinear coordinates.  Build
    # the complete local Jacobian for rank and Laplace-covariance diagnostics.
    complete_jacobian = np.empty(
        (likelihood_point_count + fitted_x.size, fitted_x.size), dtype=np.float64,
    )
    for parameter_index in range(fitted_x.size):
        step = 1.0e-5 * max(
            abs(fitted_x[parameter_index]), prior.sigma[parameter_index], 1.0e-3,
        )
        right = fitted_x.copy()
        left = fitted_x.copy()
        right[parameter_index] += step
        left[parameter_index] -= step
        complete_jacobian[:, parameter_index] = (
            full_residual(right) - full_residual(left)
        ) / (2.0 * step)
    numerical_jacobian = complete_jacobian[:likelihood_point_count]
    singular = np.linalg.svd(numerical_jacobian, compute_uv=False)
    tolerance = max(numerical_jacobian.shape) * np.finfo(float).eps * singular[0] if singular.size else 0.0
    rank = int(np.count_nonzero(singular > tolerance))
    posterior_covariance = np.linalg.pinv(
        complete_jacobian.T @ complete_jacobian, rcond=1.0e-11,
    )
    effective_parameter_count = float(np.trace(
        numerical_jacobian @ posterior_covariance @ numerical_jacobian.T
    ))
    dof = max(0.0, likelihood_point_count - effective_parameter_count)
    posterior_sigma = np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0))
    prior_fraction = np.minimum(1.0, posterior_sigma / prior.sigma)
    bispectrum_pull = raw_residual / data.error_mean[indices]
    power_pull = (
        power_raw_residual / data.power_error_mean[power_indices]
        if power_raw_residual is not None else None
    )
    max_bispectrum_pull = (
        float(np.max(np.abs(bispectrum_pull)))
        if bispectrum_pull.size else math.nan
    )
    max_power_pull = (
        float(np.max(np.abs(power_pull)))
        if power_pull is not None and power_pull.size else None
    )
    max_pull = max(
        [max_bispectrum_pull]
        + ([max_power_pull] if max_power_pull is not None else [])
    )
    bispectrum_sign_run = longest_sign_run(whitened)
    power_sign_run = (
        longest_sign_run(power_whitened)
        if power_whitened is not None else None
    )
    max_sign_run = max(
        [bispectrum_sign_run]
        + ([power_sign_run] if power_sign_run is not None else [])
    )
    model_jacobian = np.empty((120, fitted_x.size), dtype=np.float64)
    power_model_jacobian = (
        np.empty((47, fitted_x.size), dtype=np.float64) if power_templates else None
    )
    for parameter_index in range(fitted_x.size):
        step = 1.0e-5 * max(
            abs(fitted_x[parameter_index]), prior.sigma[parameter_index], 1.0e-3,
        )
        right = fitted_x.copy()
        left = fitted_x.copy()
        right[parameter_index] += step
        left[parameter_index] -= step
        right_total = templates.components(parameter_dict(right), number_density)["total"]
        left_total = templates.components(parameter_dict(left), number_density)["total"]
        model_jacobian[:, parameter_index] = (right_total - left_total) / (2.0 * step)
        if power_templates and power_model_jacobian is not None:
            right_power = power_templates.components(
                parameter_dict(right), number_density,
            )["total"]
            left_power = power_templates.components(
                parameter_dict(left), number_density,
            )["total"]
            power_model_jacobian[:, parameter_index] = (
                right_power - left_power
            ) / (2.0 * step)
    return FitResult(
        kmax=kmax,
        mask=mask,
        likelihood_point_count=likelihood_point_count,
        likelihood=likelihood,
        mock_count=mock_count,
        x=fitted_x,
        covariance=posterior_covariance,
        components=components,
        power_components=power_components,
        residual=raw_residual,
        whitened=whitened,
        power_residual=power_raw_residual,
        power_whitened=power_whitened,
        chi2=data_chi2,
        data_likelihood_objective=data_objective,
        likelihood_weight_at_mode=local_likelihood_weight,
        power_chi2=power_chi2,
        bispectrum_conditional_chi2=bispectrum_conditional_chi2,
        dof=dof,
        effective_parameter_count=effective_parameter_count,
        effective_rank=rank,
        p_value=(
            float(chi2_distribution.sf(data_chi2, dof))
            if likelihood == RAW_GAUSSIAN_LIKELIHOOD and dof > 0
            else None
        ),
        objective=float(np.dot(fitted_residual, fitted_residual)),
        optimizer_success=bool(solution.success),
        optimizer_message=str(solution.message),
        multistart_objectives=np.asarray([
            float(np.dot(item.fun, item.fun)) for item in solutions
        ]),
        max_pull=max_pull,
        max_bispectrum_pull=max_bispectrum_pull,
        max_power_pull=max_power_pull,
        max_sign_run=max_sign_run,
        max_bispectrum_sign_run=bispectrum_sign_run,
        max_power_sign_run=power_sign_run,
        posterior_to_prior_sigma_ratio=prior_fraction,
        singular_values=singular,
        model_jacobian=model_jacobian,
        power_model_jacobian=power_model_jacobian,
    )


def robustness_shift_report(
    fit: FitResult,
    primary: FitResult,
) -> dict[str, Any]:
    primary_sigma = np.sqrt(np.maximum(np.diag(primary.covariance), 0.0))
    safe_sigma = np.where(primary_sigma > 0.0, primary_sigma, math.inf)
    standardized = np.abs(fit.x - primary.x) / safe_sigma
    maximum_index = int(np.argmax(standardized))
    return {
        "gate_role": "diagnostic_only",
        "shift_denominator": "primary raw-Gaussian marginal posterior sigma",
        "maximum_parameter_shift_primary_sigma": float(
            standardized[maximum_index]
        ),
        "maximum_shift_parameter": FIT_NAMES[maximum_index],
        "parameter_shift_primary_sigma": {
            name: float(value)
            for name, value in zip(FIT_NAMES, standardized)
        },
        "fit": fit.report(),
    }


def configure_plotting() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def save_pdf(fig: plt.Figure, path: Path, subject: str) -> None:
    fig.savefig(path, metadata={
        "Title": path.stem,
        "Author": "MARISA-B EFT-v2 R0",
        "Subject": subject + (" | " + PDF_METADATA_CONTEXT if PDF_METADATA_CONTEXT else ""),
        "Keywords": "Quijote; pre-reconstruction; B000; EFT-v2; shell average",
    })


def plot_shell_groups(
    ax: plt.Axes, data: DataSet, values: np.ndarray, *, label: str, **kwargs: Any,
) -> None:
    """Connect only bins sharing the first stored radial shell."""
    first_shell = data.k_pair[:, 0]
    for group_index, value in enumerate(np.unique(first_shell)):
        indices = np.flatnonzero(first_shell == value)
        ax.plot(
            indices, values[indices], label=label if group_index == 0 else None,
            **kwargs,
        )


def plot_components(
    path: Path, data: DataSet, fit: FitResult,
    selected_dr_gate: dict[str, Any] | None,
    selected_dr_validation: dict[str, Any] | None,
) -> None:
    x = np.arange(120)
    c = fit.components
    if fit.power_components is None:
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        power_ax = None
    else:
        fig, (ax, power_ax) = plt.subplots(
            2, 1, figsize=(11.5, 8.0), gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
    ax.errorbar(x, data.mean, yerr=data.error_mean, fmt=".", ms=2.6, lw=0.55,
                color="black", alpha=0.8, label="Quijote pre mean ± covariance-of-mean")
    plot_shell_groups(ax, data, c["tree"], lw=0.75, marker=".", ms=2.0, label="tree")
    plot_shell_groups(ax, data, c["loop"], lw=0.75, marker=".", ms=2.0,
                      label="renormalized loop")
    plot_shell_groups(ax, data, c["counterterm"], lw=0.75, marker=".", ms=2.0,
                      label="counterterm")
    plot_shell_groups(ax, data, c["stochastic"], lw=0.75, marker=".", ms=2.0,
                      label="stochastic")
    plot_shell_groups(ax, data, c["total"], lw=1.2, marker=".", ms=2.5,
                      color="tab:red", label="EFT+IR total")
    outside = np.flatnonzero(~fit.mask)
    ax.plot(outside, c["total"][outside], "x", ms=2.3, color="tab:red", alpha=0.45,
            label=f"total prediction outside fit kmax={fit.kmax:.2f}")
    ax.set_yscale("symlog", linthresh=1.0e7)
    ax.set_xlabel("frozen upper-triangle shell-bin index")
    ax.set_ylabel(r"$B_{000}$ [$(\mathrm{Mpc}/h)^6$]")
    ax.set_title(f"Pre-reconstruction EFT-v2 R0 diagnostic, B fit kmax={fit.kmax:.2f} h/Mpc")
    if selected_dr_gate is not None:
        maximum_selected = max(
            abs(float(row["selected"]["delta_sigma_mean"]))
            for row in selected_dr_gate["cases"]
        )
        ax.text(
            0.01, 0.02,
            (
                "Selected FFTLog/DR S4: "
                + (
                    "CURRENT PASS "
                    if selected_dr_validation is not None
                    and selected_dr_validation["pass"]
                    else "STALE/FAIL "
                )
                + rf"(max $|\Delta|/\sigma_\mathrm{{mean}}={maximum_selected:.4f}$); "
                "full 120-bin release vectors shown"
            ),
            transform=ax.transAxes, fontsize=7.2, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
        )
    ax.legend(ncol=3, fontsize=7.5)
    if power_ax is not None and fit.power_components is not None:
        p = fit.power_components
        power_ax.errorbar(
            data.power_k, data.power_mean, yerr=data.power_error_mean,
            fmt="o", ms=2.8, lw=0.6, color="black", label="Quijote pre P0 mean",
        )
        power_ax.plot(data.power_k, p["tree"], lw=0.9, label="P tree")
        power_ax.plot(data.power_k, p["loop"], lw=0.9, label="P renormalized loop")
        power_ax.plot(data.power_k, p["counterterm"], lw=0.9, label="P counterterm")
        power_ax.plot(data.power_k, p["stochastic"], lw=0.9, label="P stochastic")
        power_ax.plot(data.power_k, p["total"], lw=1.4, color="tab:red", label="P EFT+IR total")
        power_ax.axvline(
            POWER_KMAX_H_MPC, color="0.3", ls="--", lw=0.7,
            label=f"P fit kmax={POWER_KMAX_H_MPC:.2f}",
        )
        power_ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        power_ax.set_ylabel(r"$P_0$ [$(\mathrm{Mpc}/h)^3$]")
        power_ax.legend(ncol=4, fontsize=7.0)
    fig.tight_layout()
    save_pdf(fig, path, "Joint pre P0 and B000 simulation--theory components at the R0 primary kmax")
    plt.close(fig)


def diagonal_bispectrum_indices(
    data: DataSet, fit_mask: np.ndarray,
) -> np.ndarray:
    if fit_mask.shape != (120,):
        raise ValueError("diagonal B000 plot requires a 120-bin fit mask")
    same_shell = np.all(
        np.isclose(
            data.edges[:, 0, :],
            data.edges[:, 1, :],
            rtol=0.0,
            atol=1.0e-14,
        ),
        axis=1,
    )
    indices = np.flatnonzero(same_shell & fit_mask)
    if indices.size == 0:
        raise ValueError("diagonal B000 plot has no fitted same-shell bins")
    effective_k = np.mean(data.k_pair[indices], axis=1)
    return indices[np.argsort(effective_k)]


def plot_diagonal_k2b000(
    path: Path, data: DataSet, fit: FitResult,
) -> None:
    indices = diagonal_bispectrum_indices(data, fit.mask)
    k = np.mean(data.k_pair[indices], axis=1)
    scale = k**2
    measured = data.mean[indices]
    error = data.error_mean[indices]
    components = fit.components
    total = components["total"][indices]
    pull = (total - measured) / error

    fig, (ax, pull_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]},
    )
    ax.errorbar(
        k,
        scale * measured,
        yerr=scale * error,
        fmt="o",
        ms=4.6,
        lw=0.9,
        capsize=2.2,
        color="black",
        label="Quijote halo mean ± covariance-of-mean",
        zorder=5,
    )
    ax.plot(
        k,
        scale * components["tree"][indices],
        "--",
        lw=1.3,
        color="#4C72B0",
        label="tree",
    )
    ax.plot(
        k,
        scale * components["spt"][indices],
        "-.",
        lw=1.35,
        color="#55A868",
        label="tree + renormalized loop",
    )
    ax.plot(
        k,
        scale * total,
        "-",
        marker="^",
        ms=4.0,
        lw=1.8,
        color="#C44E52",
        label="best-fit EFT + NLO IR total",
    )
    ax.axhline(0.0, color="0.35", lw=0.7)
    ax.set_ylabel(
        r"$k^2 B_{000}(k,k)$ "
        r"[$(\mathrm{Mpc}/h)^4$]",
    )
    ax.set_title(
        "Quijote halo pre-reconstruction, "
        r"$z=1$, $M_{\min}=10^{13}\,h^{-1}M_\odot$"
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    p_value = fit.p_value if fit.p_value is not None else math.nan
    ax.text(
        0.02,
        0.03,
        (
            rf"joint $P_0+B_{{000}}$ fit: "
            rf"$k_{{\max}}={fit.kmax:.2f}\,h\,\mathrm{{Mpc}}^{{-1}}$, "
            rf"$p={p_value:.4f}$, "
            rf"$\chi^2/\mathrm{{dof}}={fit.chi2 / fit.dof:.3f}$"
        ),
        transform=ax.transAxes,
        fontsize=7.7,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.88,
        },
    )
    ax.legend(fontsize=7.5, loc="best")

    pull_ax.axhspan(-1.0, 1.0, color="0.85", alpha=0.45)
    pull_ax.axhline(0.0, color="0.25", lw=0.8)
    pull_ax.axhline(3.0, color="0.45", lw=0.7, ls=":")
    pull_ax.axhline(-3.0, color="0.45", lw=0.7, ls=":")
    pull_ax.plot(
        k,
        pull,
        "o-",
        ms=4.0,
        lw=1.1,
        color="#C44E52",
    )
    pull_limit = max(3.6, 1.15 * float(np.max(np.abs(pull))))
    pull_ax.set_ylim(-pull_limit, pull_limit)
    pull_ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    pull_ax.set_ylabel(
        r"$(B_{\rm th}-B_{\rm sim})/\sigma_{\rm mean}$"
    )
    pull_ax.text(
        0.98,
        0.06,
        f"diagonal bins {indices.tolist()}",
        transform=pull_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        color="0.3",
    )
    fig.tight_layout()
    save_pdf(
        fig,
        path,
        (
            "Fitted diagonal k^2 B000 simulation--theory comparison "
            "for the exact contract-v2 R0 diagnostic"
        ),
    )
    plt.close(fig)


def plot_ir_compare(
    path: Path, data: DataSet, ir_fit: FitResult, noir_fit: FitResult,
    noir_at_ir_parameters: dict[str, np.ndarray],
) -> None:
    x = np.arange(120)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.errorbar(x, data.mean, yerr=data.error_mean, fmt=".", ms=2.4, lw=0.5,
                color="black", label="Quijote pre")
    plot_shell_groups(
        ax, data, noir_at_ir_parameters["spt"], lw=0.8, marker=".", ms=2,
        label="SPT fixed order (IR-fit parameters)",
    )
    plot_shell_groups(
        ax, data, noir_at_ir_parameters["total"], lw=0.9, ls="--", marker=".", ms=2,
        label="EFT no IR (IR-fit parameters)",
    )
    plot_shell_groups(
        ax, data, noir_fit.components["total"], lw=1.0, marker=".", ms=2,
        label="EFT no IR (refit)",
    )
    plot_shell_groups(ax, data, ir_fit.components["total"], lw=1.2, marker=".", ms=2,
                      label="EFT + NLO IR")
    ax.set_yscale("symlog", linthresh=1.0e7)
    ax.set_xlabel("frozen upper-triangle shell-bin index")
    ax.set_ylabel(r"$B_{000}$ [$(\mathrm{Mpc}/h)^6$]")
    ax.legend(ncol=3)
    fig.tight_layout()
    save_pdf(fig, path, "Fixed-order, EFT no-IR, and EFT plus NLO-IR comparison")
    plt.close(fig)


def plot_residuals(path: Path, data: DataSet, fit: FitResult) -> None:
    indices = np.flatnonzero(fit.mask)
    pulls = fit.residual / data.error_mean[indices]
    if fit.power_residual is None or fit.power_whitened is None:
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2))
    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].errorbar(indices, fit.residual, yerr=data.error_mean[indices], fmt="o", ms=2.5, lw=0.6)
    axes[0].set_ylabel("theory − simulation")
    axes[0].set_yscale("symlog", linthresh=1.0e6)
    axes[1].axhline(0.0, color="0.2", lw=0.8)
    axes[1].axhspan(-3.0, 3.0, color="tab:green", alpha=0.08)
    axes[1].plot(indices, fit.whitened, "o", ms=2.5, label="whitened residual")
    axes[1].plot(indices, pulls, ".", ms=3.0, alpha=0.55, label="marginal pull")
    axes[1].set_ylabel("standardized residual")
    axes[1].set_xlabel("frozen upper-triangle shell-bin index")
    axes[1].legend(ncol=2)
    if fit.power_residual is not None and fit.power_whitened is not None:
        power_indices = np.flatnonzero(data.power_mask(POWER_KMAX_H_MPC))
        power_pull = (
            fit.power_residual / data.power_error_mean[power_indices]
        )
        axes[2].axhline(0.0, color="0.2", lw=0.8)
        axes[2].axhspan(-3.0, 3.0, color="tab:green", alpha=0.08)
        axes[2].plot(
            data.power_k[power_indices], fit.power_whitened, "o", ms=2.8,
            label="P0 whitened residual",
        )
        axes[2].plot(
            data.power_k[power_indices], power_pull, ".", ms=3.2, alpha=0.6,
            label="P0 marginal pull",
        )
        axes[2].set_ylabel("P0 standardized residual")
        axes[2].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        axes[2].legend(ncol=2)
    fig.tight_layout()
    save_pdf(
        fig, path,
        "Joint P0 and B000 raw, marginal, and covariance-whitened residuals",
    )
    plt.close(fig)


def plot_parameter_stability(path: Path, fits: list[FitResult]) -> None:
    columns = 3
    rows = math.ceil(len(FIT_NAMES) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11.0, 2.15 * rows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_flat, FIT_NAMES):
        index = FIT_NAMES.index(name)
        values = np.asarray([fit.x[index] for fit in fits])
        errors = np.asarray([
            math.sqrt(max(fit.covariance[index, index], 0.0)) for fit in fits
        ])
        ax.errorbar([fit.kmax for fit in fits], values, yerr=errors, fmt="o-", capsize=2)
        ax.set_ylabel(name)
    for ax in axes_flat[len(FIT_NAMES):]:
        ax.set_visible(False)
    for ax in np.asarray(axes)[-1]:
        ax.set_xlabel(r"$k_\mathrm{max}$ [$h/\mathrm{Mpc}$]")
    fig.tight_layout()
    save_pdf(fig, path, "All independent EFT nuisance coordinates versus kmax")
    plt.close(fig)


def plot_regulators(
    path: Path, data: DataSet, primary: FitResult,
    regulator_curves: list[tuple[str, np.ndarray]],
    selected_dr_gate: dict[str, Any] | None,
    selected_dr_validation: dict[str, Any] | None,
) -> None:
    if selected_dr_gate is None:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        oracle_ax = None
    else:
        fig, (ax, oracle_ax) = plt.subplots(
            2, 1, figsize=(10.5, 7.2),
            gridspec_kw={"height_ratios": [1.45, 1.0]},
        )
    ax.axhline(0.0, color="0.2", lw=0.8)
    for label, total in regulator_curves:
        difference = (
            total - primary.components["total"]
        ) / data.error_mean
        plot_shell_groups(
            ax, data, difference, marker=".", ms=2.6, lw=0.6,
            alpha=0.75, label=label,
        )
    if not regulator_curves:
        ax.text(0.5, 0.5, "No regulator variants supplied", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhspan(-0.1, 0.1, color="tab:green", alpha=0.08, label="Gate ±0.10 σmean")
    ax.set_xlabel("full frozen 120-bin shell index")
    ax.set_ylabel(r"$(B_\mathrm{variant}-B_\mathrm{primary})/\sigma_\mathrm{mean}$")
    if regulator_curves:
        ax.legend(ncol=3, fontsize=7.5)
    if oracle_ax is not None and selected_dr_gate is not None:
        labels = [str(row["label"]) for row in selected_dr_gate["cases"]]
        x = np.arange(len(labels), dtype=np.float64)
        selected = np.asarray([
            row["selected"]["delta_sigma_mean"]
            for row in selected_dr_gate["cases"]
        ], dtype=np.float64)
        matter = np.asarray([
            row["matter"]["delta_sigma_mean"]
            for row in selected_dr_gate["cases"]
        ], dtype=np.float64)
        frequency = np.asarray([
            row["analytic_frequency_step_sigma_mean"]
            for row in selected_dr_gate["cases"]
        ], dtype=np.float64)
        direct = np.asarray([
            row["direct_grid_step_sigma_mean"]
            for row in selected_dr_gate["cases"]
        ], dtype=np.float64)
        oracle_ax.axhline(0.0, color="0.2", lw=0.8)
        oracle_ax.axhspan(
            -0.02, 0.02, color="tab:green", alpha=0.10,
            label=r"selected-template Gate $\pm0.02\,\sigma_\mathrm{mean}$",
        )
        oracle_ax.plot(
            x-0.18, selected, "o", ms=5.0,
            label="analytic DR − direct, selected bias",
        )
        oracle_ax.plot(
            x-0.06, matter, "x", ms=5.0,
            label="analytic DR − direct, matter",
        )
        oracle_ax.plot(
            x+0.06, frequency, "s", ms=4.2, fillstyle="none",
            label="last analytic frequency step",
        )
        oracle_ax.plot(
            x+0.18, direct, "d", ms=4.2, fillstyle="none",
            label="last direct-grid step",
        )
        oracle_ax.set_xticks(x, labels)
        oracle_ax.set_ylabel(r"$\Delta/\sigma_\mathrm{mean}$")
        oracle_ax.set_title(
            "Selected-triangle numerical oracle Gate "
            + (
                "[current provenance] "
                if selected_dr_validation is not None
                and selected_dr_validation["pass"]
                else "[stale or failing provenance] "
            )
            + "(nearest Quijote B000 radial-pair error scale)",
            fontsize=9.0,
        )
        oracle_ax.legend(ncol=2, fontsize=7.2)
    fig.tight_layout()
    save_pdf(
        fig, path,
        "Full-bin regulator changes and selected-template FFTLog/DR numerical Gate",
    )
    plt.close(fig)


def main() -> None:
    global PDF_METADATA_CONTEXT
    args = parse_args()
    if not args.allow_legacy_mean_likelihood:
        raise RuntimeError(
            "legacy covariance-of-the-mean likelihood is disabled; "
            "use current production runners with single-realization covariance"
        )
    if args.multistart < 1:
        raise ValueError("--multistart must be positive")
    if (args.ir_power_templates is None) != (args.noir_power_templates is None):
        raise ValueError("IR and no-IR power templates must be supplied together")
    configure_plotting()
    data = DataSet.load(args.matrix)
    contract, prior, nbar = load_prior(args.contract)
    fit_protocol = load_fit_protocol(contract, data)
    selected_dr_gate: dict[str, Any] | None = None
    selected_dr_validation: dict[str, Any] | None = None
    if args.selected_dr_gate is not None:
        selected_dr_gate = read_json_object(args.selected_dr_gate)
        selected_dr_validation = validate_selected_dr_certificate(
            selected_dr_gate, contract, args.matrix,
        )
    ir = TemplateSet.load(args.ir_templates)
    noir = TemplateSet.load(args.noir_templates)
    ir_power = PowerTemplateSet.load(args.ir_power_templates) if args.ir_power_templates else None
    noir_power = (
        PowerTemplateSet.load(args.noir_power_templates)
        if args.noir_power_templates else None
    )
    validate_exact_lattice_metadata(ir, "IR bispectrum")
    validate_exact_lattice_metadata(noir, "no-IR bispectrum")
    validate_bispectrum_geometry(ir, data, "IR bispectrum")
    validate_bispectrum_geometry(noir, data, "no-IR bispectrum")
    if ir_power is not None and noir_power is not None:
        validate_power_shell_metadata(ir_power, "IR power")
        validate_power_shell_metadata(noir_power, "no-IR power")
        validate_power_geometry(ir_power, data, "IR power")
        validate_power_geometry(noir_power, data, "no-IR power")
    validate_release_inputs(
        ir, noir, ir_power, noir_power, contract, args.contract, args.matrix,
    )
    rng = np.random.default_rng(args.seed)
    kmax_values = [float(value) for value in contract["observable"]["kmax_scan_h_mpc"]]
    ir_fits = [
        fit_one(
            ir, ir_power, data, prior, nbar, value, args.multistart, rng,
            likelihood=fit_protocol.primary_likelihood,
            mock_count=fit_protocol.mock_count,
            correlation_jitter=fit_protocol.correlation_jitter,
        )
        for value in kmax_values
    ]
    primary = ir_fits[kmax_values.index(0.15)]
    noir_primary = fit_one(
        noir, noir_power, data, prior, nbar, 0.15, args.multistart, rng,
        likelihood=fit_protocol.primary_likelihood,
        mock_count=fit_protocol.mock_count,
        correlation_jitter=fit_protocol.correlation_jitter,
    )
    robustness_seed = args.seed + 1_000_003
    finite_mock_robustness = {
        likelihood: fit_one(
            ir, ir_power, data, prior, nbar, 0.15, args.multistart,
            np.random.default_rng(robustness_seed),
            likelihood=likelihood,
            mock_count=fit_protocol.mock_count,
            correlation_jitter=fit_protocol.correlation_jitter,
        )
        for likelihood in fit_protocol.robustness_likelihoods
    }
    b1_prior_robustness = {
        sigma: fit_one(
            ir, ir_power, data, prior_with_b1_sigma(prior, sigma),
            nbar, 0.15, args.multistart,
            np.random.default_rng(robustness_seed),
            likelihood=fit_protocol.primary_likelihood,
            mock_count=fit_protocol.mock_count,
            correlation_jitter=fit_protocol.correlation_jitter,
        )
        for sigma in fit_protocol.b1_robustness_prior_sigmas
    }
    metadata_context: dict[str, Any] = {
        "figure_set": args.figure_set,
        "fit_script_sha256": sha256(Path(__file__).resolve()),
        "matrix_sha256": sha256(args.matrix),
        "ir_template_sha256": sha256(args.ir_templates),
        "noir_template_sha256": sha256(args.noir_templates),
        "selected_dr_gate_sha256": (
            sha256(args.selected_dr_gate)
            if args.selected_dr_gate is not None else None
        ),
        "selected_dr_validation": selected_dr_validation,
        "ir_power_template_sha256": (
            sha256(args.ir_power_templates) if args.ir_power_templates else None
        ),
        "noir_power_template_sha256": (
            sha256(args.noir_power_templates) if args.noir_power_templates else None
        ),
        "kmax_h_mpc": primary.kmax,
        "power_kmax_h_mpc": POWER_KMAX_H_MPC if ir_power is not None else None,
        "statistical_gate_conventions": {
            "maximum_whitened_same_sign_run": MAX_WHITENED_SIGN_RUN,
            "next_bin_minimum_p_value": NEXT_BIN_MIN_P_VALUE,
            "next_bin_maximum_abs_mean_whitened_residual": (
                NEXT_BIN_MAX_ABS_MEAN_WHITENED
            ),
            "maximum_pull_scope": (
                "maximum marginal pull across fitted P0 and B000 points"
            ),
            "sign_run_scope": (
                "maximum of the separate P0 and conditional-B000 whitened sequences"
            ),
            "prior_boundary_policy": (
                "all fitted priors are unbounded Gaussian distributions and "
                "the optimizer uses no box bounds"
            ),
        },
        "parameters": {
            name: float(value) for name, value in zip(FIT_NAMES, primary.x)
        },
        "ir": ir.header.get("ir_metadata"),
        "uv": {
            "q_range": ir.header.get("q_range"),
            "mu_ren": ir.header.get("mu_ren"),
            "loop_quadrature": ir.header.get("loop_quadrature"),
            "stochastic_qmax": ir.header.get("stochastic_qmax"),
            "stochastic_quadrature": ir.header.get("stochastic_quadrature"),
        },
        "source_hashes": ir.header.get("source_hashes"),
        "covariance": {
            "estimator": "matched P0+B000 sample covariance of the mean",
            "mock_count": fit_protocol.mock_count,
            "primary_likelihood": fit_protocol.primary_likelihood,
            "correlation_jitter": fit_protocol.correlation_jitter,
        },
        "shell_rule": {
            "bispectrum": ir.header.get("shell_projection"),
            "power": (
                ir_power.header.get("shell_projection")
                if ir_power is not None else None
            ),
            "measurement_ordering": (
                "two stored Sugiyama radial shells; the closing side "
                "is integrated over the exact joint lattice measure"
            ),
        },
        "power_coordinate_comparison": (
            {
                "interpretation": (
                    "display coordinates only; exact estimator geometry "
                    "is locked by edges and mode counts"
                ),
                "maximum_absolute_theory_minus_matrix_k_h_mpc": float(
                    np.max(np.abs(ir_power.k_effective - data.power_k))
                ),
                "rms_theory_minus_matrix_k_h_mpc": float(
                    np.sqrt(np.mean(
                        (ir_power.k_effective - data.power_k) ** 2
                    ))
                ),
            }
            if ir_power is not None else None
        ),
    }
    regulator_power_paths: dict[str, Path] = {}
    for specification in args.regulator_power:
        if "=" not in specification:
            raise ValueError(f"invalid --regulator-power {specification!r}")
        label, filename = specification.split("=", 1)
        if not label or label in regulator_power_paths:
            raise ValueError(f"duplicate or empty regulator-power label {label!r}")
        regulator_power_paths[label] = Path(filename)
    regulator_curves: list[tuple[str, np.ndarray]] = []
    regulator_reports: dict[str, Any] = {}
    seen_regulator_signatures: set[tuple[str, str]] = set()
    for specification in args.regulator:
        if "=" not in specification:
            raise ValueError(f"invalid --regulator {specification!r}")
        label, filename = specification.split("=", 1)
        if not label or label in regulator_reports:
            raise ValueError(f"duplicate or empty regulator label {label!r}")
        templates = TemplateSet.load(Path(filename))
        category = regulator_category(label)
        power_path = regulator_power_paths.get(label)
        variant_power = PowerTemplateSet.load(power_path) if power_path else ir_power
        validate_exact_lattice_metadata(templates, label)
        validate_bispectrum_geometry(templates, data, label)
        if power_path is not None and variant_power is not None:
            validate_power_shell_metadata(
                variant_power, f"{label} power",
            )
            validate_power_geometry(variant_power, data, f"{label} power")
        validate_regulator_inputs(
            label, category, templates, variant_power, power_path,
            ir, ir_power, contract, args.contract,
        )
        configuration_signature = regulator_configuration_signature(
            category, templates.header,
        )
        signature_key = (
            category,
            json.dumps(
                configuration_signature, allow_nan=False,
                separators=(",", ":"), sort_keys=True,
            ),
        )
        if signature_key in seen_regulator_signatures:
            raise ValueError(
                f"{label}: duplicate {category} regulator configuration"
            )
        seen_regulator_signatures.add(signature_key)
        paired_power_required = regulator_requires_paired_power(
            category, templates.header, ir.header, ir_power,
        )
        fit = fit_one(
            templates, variant_power, data, prior, nbar, 0.15,
            args.multistart, rng,
            likelihood=fit_protocol.primary_likelihood,
            mock_count=fit_protocol.mock_count,
            correlation_jitter=fit_protocol.correlation_jitter,
        )
        fixed_components = templates.components(
            parameter_dict(primary.x), nbar,
        )
        regulator_curves.append((label, fixed_components["total"]))
        indices = np.flatnonzero(primary.mask)
        fixed_delta_sigma_full = (
            fixed_components["total"] - primary.components["total"]
        ) / data.error_mean
        refit_delta_sigma_full = (
            fit.components["total"] - primary.components["total"]
        ) / data.error_mean
        fixed_delta_sigma_fit = fixed_delta_sigma_full[indices]
        refit_delta_sigma_fit = refit_delta_sigma_full[indices]
        parameter_shift_primary = np.abs(fit.x - primary.x) / np.sqrt(np.maximum(
            np.diag(primary.covariance), 1.0e-300,
        ))
        parameter_shift_combined = np.abs(fit.x - primary.x) / np.sqrt(np.maximum(
            np.diag(fit.covariance) + np.diag(primary.covariance), 1.0e-300,
        ))
        power_fixed_report: dict[str, Any] | None = None
        if power_path is not None and variant_power is not None and ir_power is not None:
            power_indices = np.flatnonzero(data.power_mask(POWER_KMAX_H_MPC))
            variant_power_fixed = variant_power.components(
                parameter_dict(primary.x), nbar,
            )["total"]
            primary_power_fixed = ir_power.components(
                parameter_dict(primary.x), nbar,
            )["total"]
            power_delta_sigma = (
                variant_power_fixed[power_indices]
                - primary_power_fixed[power_indices]
            ) / data.power_error_mean[power_indices]
            power_fixed_report = {
                "fit_bin_count": int(power_indices.size),
                "max_abs_delta_sigma_mean": float(
                    np.max(np.abs(power_delta_sigma))
                ),
                "rms_delta_sigma_mean": float(
                    np.sqrt(np.mean(power_delta_sigma**2))
                ),
            }
        regulator_reports[label] = {
            "template_sha256": sha256(Path(filename)),
            "category": category,
            "configuration_signature": configuration_signature,
            "complete_bispectrum_bin_count": len(templates.bins),
            "power_template_sha256": (
                sha256(power_path) if power_path is not None else None
            ),
            "paired_power_required": paired_power_required,
            "paired_power_provided": power_path is not None,
            "refit": fit.report(),
            "full_observable_fixed_primary_parameters": {
                "bin_count": 120,
                "max_abs_delta_sigma_mean": float(
                    np.max(np.abs(fixed_delta_sigma_full))
                ),
                "rms_delta_sigma_mean": float(
                    np.sqrt(np.mean(fixed_delta_sigma_full**2))
                ),
            },
            "fit_region_fixed_primary_parameters": {
                "bin_count": int(indices.size),
                "max_abs_delta_sigma_mean": float(
                    np.max(np.abs(fixed_delta_sigma_fit))
                ),
                "rms_delta_sigma_mean": float(
                    np.sqrt(np.mean(fixed_delta_sigma_fit**2))
                ),
            },
            "full_observable_after_refit": {
                "bin_count": 120,
                "max_abs_delta_sigma_mean": float(
                    np.max(np.abs(refit_delta_sigma_full))
                ),
                "rms_delta_sigma_mean": float(
                    np.sqrt(np.mean(refit_delta_sigma_full**2))
                ),
            },
            "fit_region_after_refit": {
                "bin_count": int(indices.size),
                "max_abs_delta_sigma_mean": float(
                    np.max(np.abs(refit_delta_sigma_fit))
                ),
                "rms_delta_sigma_mean": float(
                    np.sqrt(np.mean(refit_delta_sigma_fit**2))
                ),
            },
            "parameter_shift": {
                "maximum_primary_posterior_sigma": float(
                    np.max(parameter_shift_primary)
                ),
                "maximum_primary_posterior_sigma_parameter": FIT_NAMES[
                    int(np.argmax(parameter_shift_primary))
                ],
                "maximum_combined_posterior_sigma": float(
                    np.max(parameter_shift_combined)
                ),
                "maximum_combined_posterior_sigma_parameter": FIT_NAMES[
                    int(np.argmax(parameter_shift_combined))
                ],
            },
            "power_fixed_primary_parameters": power_fixed_report,
        }
    unknown_power_labels = sorted(set(regulator_power_paths) - set(regulator_reports))
    if unknown_power_labels:
        raise ValueError(
            "--regulator-power labels without matching --regulator: "
            + ", ".join(unknown_power_labels)
        )
    metadata_context["regulator_template_sha256"] = {
        label: row["template_sha256"]
        for label, row in sorted(regulator_reports.items())
    }
    metadata_context["regulator_power_template_sha256"] = {
        label: row["power_template_sha256"]
        for label, row in sorted(regulator_reports.items())
        if row["power_template_sha256"] is not None
    }
    PDF_METADATA_CONTEXT = json.dumps(
        metadata_context, allow_nan=False, separators=(",", ":"), sort_keys=True,
    )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_set == "diagonal-k2b000":
        plot_diagonal_k2b000(
            args.figure_dir
            / (
                "quijote_halo_z1_mmin1e13_r15_pre_eft_v2_contract_v2_"
                "exact_diagonal_k2b000_measurement_theory.pdf"
            ),
            data,
            primary,
        )
    else:
        plot_components(
            args.figure_dir / "pre_gaussian_measurement_theory_components.pdf",
            data, primary, selected_dr_gate, selected_dr_validation,
        )
        plot_ir_compare(
            args.figure_dir / "pre_gaussian_noir_ir_compare.pdf",
            data, primary, noir_primary,
            noir.components(parameter_dict(primary.x), nbar),
        )
        plot_residuals(
            args.figure_dir / "pre_gaussian_pull_whitened_residual.pdf",
            data, primary,
        )
        plot_regulators(
            args.figure_dir / "pre_gaussian_regulator_stability.pdf",
            data, primary, regulator_curves, selected_dr_gate,
            selected_dr_validation,
        )
        plot_parameter_stability(
            args.figure_dir / "pre_gaussian_kmax_parameter_stability.pdf",
            ir_fits,
        )

    stability = []
    next_bin_predictions = []
    for left, right in zip(ir_fits[:-1], ir_fits[1:]):
        variance = np.diag(left.covariance) + np.diag(right.covariance)
        standardized = np.abs(right.x - left.x) / np.sqrt(np.maximum(variance, 1.0e-300))
        stability.append({
            "from_kmax": left.kmax,
            "to_kmax": right.kmax,
            "maximum_parameter_shift_sigma": float(np.max(standardized)),
            "parameter": FIT_NAMES[int(np.argmax(standardized))],
        })
        new_mask = right.mask & ~left.mask
        new_indices = np.flatnonzero(new_mask)
        old_indices = np.flatnonzero(left.mask)
        if left.power_components is not None and left.power_model_jacobian is not None:
            power_indices = np.flatnonzero(data.power_mask(POWER_KMAX_H_MPC))
            conditioning_indices = np.concatenate((power_indices, 47 + old_indices))
            heldout_indices = 47 + new_indices
            covariance_source = data.joint_covariance_mean_power_then_bispectrum
            old_residual = np.concatenate((
                left.power_components["total"][power_indices] - data.power_mean[power_indices],
                left.components["total"][old_indices] - data.mean[old_indices],
            ))
            old_jacobian = np.vstack((
                left.power_model_jacobian[power_indices],
                left.model_jacobian[old_indices],
            ))
        else:
            conditioning_indices = old_indices
            heldout_indices = new_indices
            covariance_source = data.covariance_mean
            old_residual = left.components["total"][old_indices] - data.mean[old_indices]
            old_jacobian = left.model_jacobian[old_indices]
        covariance_old = covariance_source[
            np.ix_(conditioning_indices, conditioning_indices)
        ]
        covariance_new_old = covariance_source[
            np.ix_(heldout_indices, conditioning_indices)
        ]
        covariance_new = covariance_source[np.ix_(heldout_indices, heldout_indices)]
        old_cholesky = covariance_cholesky(
            covariance_old,
            fit_protocol.correlation_jitter,
        )
        gain = np.linalg.solve(
            old_cholesky.T,
            solve_triangular(old_cholesky, covariance_new_old.T, lower=True),
        ).T
        new_residual = left.components["total"][new_indices] - data.mean[new_indices]
        conditional_residual = new_residual - gain @ old_residual
        conditional_data_covariance = (
            covariance_new - gain @ covariance_new_old.T
        )
        conditional_jacobian = (
            left.model_jacobian[new_indices] - gain @ old_jacobian
        )
        predictive_covariance = (
            conditional_data_covariance
            + conditional_jacobian @ left.covariance @ conditional_jacobian.T
        )
        predictive_cholesky = covariance_cholesky(
            predictive_covariance,
            fit_protocol.correlation_jitter,
        )
        new_whitened = solve_triangular(
            predictive_cholesky, conditional_residual, lower=True,
        )
        new_chi2 = float(new_whitened @ new_whitened)
        next_bin_predictions.append({
            "from_kmax": left.kmax,
            "to_kmax": right.kmax,
            "new_bin_count": int(new_indices.size),
            "chi2": new_chi2,
            "p_value": float(chi2_distribution.sf(new_chi2, new_indices.size)),
            "conditioning": "held-out bins conditional on fitted-bin covariance plus parameter uncertainty",
            "mean_whitened_residual": float(np.mean(new_whitened)),
            "max_abs_predictive_marginal_pull": float(np.max(np.abs(
                conditional_residual / np.sqrt(np.diag(predictive_covariance))
            ))),
        })
    gates = contract["gates"]
    regulator_class_checks: dict[str, bool] = {}
    regulator_class_counts = {
        "qmax": 0, "mu": 0, "ir": 0, "integration": 0, "shell": 0, "fftlog": 0,
    }
    for label, row in regulator_reports.items():
        category = str(row["category"])
        fixed = row["full_observable_fixed_primary_parameters"]
        refit = row["full_observable_after_refit"]
        numerical_parameter_drift = row["parameter_shift"][
            "maximum_primary_posterior_sigma"
        ]
        paired_power_complete = bool(
            not row["paired_power_required"] or row["paired_power_provided"]
        )
        if category == "qmax":
            passed = (
                fixed["max_abs_delta_sigma_mean"] <= float(gates["regulator_per_bin_sigma_mean"])
                and fixed["rms_delta_sigma_mean"] <= 0.05
            )
        elif category == "mu":
            passed = refit["max_abs_delta_sigma_mean"] <= float(gates["regulator_per_bin_sigma_mean"])
        elif category == "fftlog":
            passed = (
                fixed["max_abs_delta_sigma_mean"]
                <= float(gates["direct_dr_absolute_sigma_mean"])
                and numerical_parameter_drift
                <= float(gates["parameter_numerical_drift_posterior_sigma"])
            )
        elif category == "integration":
            passed = (
                fixed["max_abs_delta_sigma_mean"] <= float(gates["integration_per_bin_sigma_mean"])
                and fixed["rms_delta_sigma_mean"] <= float(gates["integration_rms_sigma_mean"])
                and numerical_parameter_drift
                <= float(gates["parameter_numerical_drift_posterior_sigma"])
            )
        elif category == "shell":
            passed = (
                fixed["max_abs_delta_sigma_mean"] <= float(gates["shell_per_bin_sigma_mean"])
                and numerical_parameter_drift
                <= float(gates["parameter_numerical_drift_posterior_sigma"])
            )
        else:
            passed = fixed["max_abs_delta_sigma_mean"] <= float(gates["ir_per_bin_sigma_mean"])
        passed = bool(passed and paired_power_complete)
        regulator_class_counts[category] += 1
        regulator_class_checks[label] = passed
    regulator_coverage = {
        "qmax": regulator_class_counts["qmax"] >= 5,
        "mu": regulator_class_counts["mu"] >= 3,
        "ir": regulator_class_counts["ir"] >= 6,
        "integration": regulator_class_counts["integration"] >= 1,
        "shell": regulator_class_counts["shell"] >= 1,
        "fftlog": regulator_class_counts["fftlog"] >= 2,
    }
    primary_report = primary.report()
    if primary.p_value is None or any(
        fit.p_value is None for fit in ir_fits
    ):
        raise RuntimeError("R0 Gate fits must use the raw Gaussian likelihood")
    per_cut_statistical_pass = {
        fit.kmax: bool(
            fit.dof > 0
            and fit.p_value >= float(gates["minimum_p_value"])
            and fit.chi2 / fit.dof <= float(gates["maximum_chi2_per_dof"])
            and fit.max_pull <= float(gates["maximum_absolute_pull"])
            and fit.max_sign_run <= MAX_WHITENED_SIGN_RUN
        )
        for fit in ir_fits
    }
    passing_cuts = [value for value, passed in per_cut_statistical_pass.items() if passed]
    k_valid = max(passing_cuts) if passing_cuts else 0.0
    gate_checks = {
        "k_valid_at_least_0p15": bool(k_valid >= float(gates["minimum_k_valid_h_mpc"])),
        "p_value": bool(primary.p_value >= float(gates["minimum_p_value"])),
        "chi2_per_dof": bool(
            primary.dof > 0 and primary.chi2 / primary.dof <= float(gates["maximum_chi2_per_dof"])
        ),
        "max_abs_pull": bool(primary.max_pull <= float(gates["maximum_absolute_pull"])),
        "no_long_whitened_sign_run": bool(
            primary.max_sign_run <= MAX_WHITENED_SIGN_RUN
        ),
        "neighbor_parameter_stability": bool(
            all(row["maximum_parameter_shift_sigma"] <= float(gates["neighbor_parameter_stability_sigma"])
                for row in stability if row["to_kmax"] <= max(k_valid, 0.15))
        ),
        "next_bin_prediction": bool(all(
            row["p_value"] >= NEXT_BIN_MIN_P_VALUE
            and abs(row["mean_whitened_residual"])
            <= NEXT_BIN_MAX_ABS_MEAN_WHITENED
            for row in next_bin_predictions
            if row["to_kmax"] <= max(k_valid, 0.15)
        )),
        "regulator_variant_thresholds": bool(
            regulator_class_checks and all(regulator_class_checks.values())
        ),
        "regulator_scan_coverage": bool(all(regulator_coverage.values())),
        "analytic_fftlog_dr_master_integral_oracle": bool(
            selected_dr_validation is not None
            and selected_dr_validation["pass"]
        ),
        "shared_pre_power_block": bool(ir_power),
        "optimizer_convergence": bool(
            all(fit.optimizer_success for fit in ir_fits) and noir_primary.optimizer_success
        ),
        "no_hard_prior_boundary": True,
    }
    gate_checks["pass"] = bool(all(gate_checks.values()))
    report = {
        "schema": "marisa-b-eft-v2-r0-fit-report-v3",
        "status": "pass" if gate_checks["pass"] else "fail",
        "fit_bin_rule": "max of the two stored Sugiyama radial modes; the closing side is angle-integrated",
        "power_fit_kmax_h_mpc": (
            POWER_KMAX_H_MPC if ir_power is not None else None
        ),
        "statistical_gate_conventions": {
            "maximum_whitened_same_sign_run": MAX_WHITENED_SIGN_RUN,
            "next_bin_minimum_p_value": NEXT_BIN_MIN_P_VALUE,
            "next_bin_maximum_abs_mean_whitened_residual": (
                NEXT_BIN_MAX_ABS_MEAN_WHITENED
            ),
            "maximum_pull_scope": (
                "maximum marginal pull across fitted P0 and B000 points"
            ),
            "sign_run_scope": (
                "maximum of the separate P0 and conditional-B000 whitened sequences"
            ),
            "prior_boundary_policy": (
                "all fitted priors are unbounded Gaussian distributions and "
                "the optimizer uses no box bounds"
            ),
        },
        "fit_protocol": {
            "figure_set": args.figure_set,
            "primary_likelihood": fit_protocol.primary_likelihood,
            "mock_count": fit_protocol.mock_count,
            "covariance_estimator": "matched sample covariance of the mean",
            "correlation_jitter": fit_protocol.correlation_jitter,
            "finite_mock_robustness_gate_role": "diagnostic_only",
            "robustness_multistart_seed": robustness_seed,
            "b1": {
                "treatment": (
                    "shared fitted P0+B000 parameter"
                    if ir_power else "Gaussian external prior"
                ),
                "primary_prior": {
                    "mean": float(prior.mean[FIT_NAMES.index("b1")]),
                    "sigma": float(prior.sigma[FIT_NAMES.index("b1")]),
                },
                "b_rec": fit_protocol.b_rec,
                "b_rec_likelihood_role": "none",
            },
        },
        "b1_treatment": (
            "shared fitted P0+B000 parameter" if ir_power
            else "Gaussian external prior"
        ),
        "number_density": nbar,
        "parameter_names": FIT_NAMES,
        "nonlinear_profiled_parameters": NONLINEAR_FIT_NAMES,
        "conditionally_solved_linear_parameters": LINEAR_FIT_NAMES,
        "exact_degeneracy_reduction": EXACT_DEGENERACY,
        "input_hashes": {
            "fit_script": sha256(Path(__file__).resolve()),
            "ir_templates": sha256(args.ir_templates),
            "noir_templates": sha256(args.noir_templates),
            "matrix": sha256(args.matrix),
            "contract": sha256(args.contract),
            **({
                "ir_power_templates": sha256(args.ir_power_templates),
                "noir_power_templates": sha256(args.noir_power_templates),
            } if args.ir_power_templates else {}),
            **({
                "selected_dr_gate": sha256(args.selected_dr_gate),
            } if args.selected_dr_gate is not None else {}),
        },
        "template_metadata": {
            "ir_bispectrum": ir.header,
            "noir_bispectrum": noir.header,
            **({
                "ir_power": ir_power.header,
                "noir_power": noir_power.header,
            } if ir_power is not None and noir_power is not None else {}),
        },
        "kmax_fits": [fit.report() for fit in ir_fits],
        "no_ir_kmax_0p15": noir_primary.report(),
        "finite_mock_likelihood_robustness": {
            likelihood: robustness_shift_report(fit, primary)
            for likelihood, fit in finite_mock_robustness.items()
        },
        "b1_prior_width_robustness": {
            f"sigma_{sigma:g}": {
                "prior_mean": float(prior.mean[FIT_NAMES.index("b1")]),
                "prior_sigma": sigma,
                **robustness_shift_report(fit, primary),
            }
            for sigma, fit in b1_prior_robustness.items()
        },
        "regulator_variants": regulator_reports,
        "regulator_variant_checks": regulator_class_checks,
        "regulator_scan_coverage": regulator_coverage,
        "regulator_gate_scope": {
            "bispectrum": "all 120 frozen shell bins",
            "fit_region": "reported separately and never substituted for the full-vector Gate",
            "paired_power": (
                "complete 47-bin companion required for every joint-fit "
                "variation except B-only FFTLog settings"
            ),
            "parameter_drift_denominator": "primary-fit marginal posterior sigma",
        },
        "neighbor_stability": stability,
        "next_bin_predictions": next_bin_predictions,
        "k_valid_h_mpc": k_valid,
        "gate_r0": gate_checks,
        "theory_completeness": {
            "analytic_fftlog_dr_master_integral_oracle": {
                "implemented": True,
                "selected_template_gate_pass": bool(
                    selected_dr_validation is not None
                    and selected_dr_validation["pass"]
                ),
                "validation": selected_dr_validation,
                "certificate": (
                    str(args.selected_dr_gate)
                    if args.selected_dr_gate is not None else None
                ),
                "release_limit": (
                    "Selected-template S4 does not replace the final "
                    "120-bin shell-projected regulator and statistical Gates."
                ),
            },
            "shared_pre_power_block": {
                "implemented": bool(ir_power),
                "current_treatment": (
                    "matched-realization P0+B000 covariance with shared b1, low-order bias, "
                    "b_nabla2_delta, Pshot, and a0" if ir_power
                    else "Gaussian external b1 prior"
                ),
            },
        },
        "selected_dr_gate": (
            {
                "schema": selected_dr_gate["schema"],
                "status": selected_dr_gate["status"],
                "case_count": len(selected_dr_gate["cases"]),
                "maximum_absolute_selected_delta_sigma_mean": max(
                    abs(float(row["selected"]["delta_sigma_mean"]))
                    for row in selected_dr_gate["cases"]
                ),
                "maximum_absolute_selected_relative_total": max(
                    abs(float(row["selected"]["relative_total"]))
                    for row in selected_dr_gate["cases"]
                ),
                "current_validation": selected_dr_validation,
            }
            if selected_dr_gate is not None else None
        ),
        "primary_kmax_0p15": primary_report,
        "figures": sorted(path.name for path in args.figure_dir.glob("*.pdf")),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.report),
        "figure_dir": str(args.figure_dir),
        "status": report["status"],
        "primary": {
            "chi2": primary.chi2,
            "dof": primary.dof,
            "p_value": primary.p_value,
            "max_pull": primary.max_pull,
        },
    }, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
