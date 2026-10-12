#!/usr/bin/env python3
"""Direct finite-fNL single-statistic models for ordinary Quijote halos.

This module intentionally exposes callable P0(fNL, theta) and
B000(fNL, theta) models.  No frozen PNG tangent is accepted by any public
evaluation method.  The controlled perturbative content is:

* pre P0: tree density bias with exact-lattice finite local-PNG K1 squared;
* pre/standard B000: halo/reconstruction tree plus leading residual
  stochastic, retaining the authenticated finite tree terms through fNL**2;
* Marisa B000: standard finite tree plus the analytic finite adaptive
  reconstruction coefficients through the same order.

Matter loops, PNG matter-loop uplifts, empirical gains, and P+B likelihoods
are outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


for _thread_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MARISA_B_MAX_THREADS",
):
    os.environ[_thread_name] = "1"
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = ROOT / "codes"
RECOVERY_RUNNER = CODES / "run_quijote_halo_z0p5_local_png_p112_recovery_profiles.py"
FULL_RUNNER = (
    CODES
    / "run_quijote_halo_z0p5_local_png_p112_full_parameter_profiles_v42p875_priorpm200.py"
)
P_MATRIX = (
    ROOT
    / "analysis/quijote_halo_z0p5_pre_p0_fnl250_v1/PRE_P0_FNL250_MATRIX.npz"
)
B_MATRIX_ROOT = ROOT / "analysis/quijote_halo_z0p5_fnl250_standard_marisa/matrices"
B_MATRICES = {
    "standard": B_MATRIX_ROOT / "STANDARD_FNL250_MATRIX.npz",
    "marisa": B_MATRIX_ROOT / "MARISA_FNL250_MATRIX.npz",
}
MARISA_FINITE = (
    ROOT
    / "analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1"
    / "theory/marisa_adaptive_finite_tree_bases.npz"
)
MARISA_FINITE_MANIFEST = MARISA_FINITE.with_suffix(".json")

KMAX = 0.08
P_COUNT = 12
B_INDICES = np.asarray((1, 2, 3, 15, 16, 17, 29, 30, 42), dtype=np.int64)
B_COUNT = int(B_INDICES.size)
DELTA_C = 1.686
P_BPHI = 1.12
B_REC = 1.9326526728794324
P_REC = 1.12
BPHI_REC = 2.0 * DELTA_C * (B_REC - P_REC)

TRUTHS = (-100.0, -50.0, 0.0, 50.0, 100.0)
TRUTH_KEYS = {
    -100.0: "LC_m",
    -50.0: "LC_m50",
    0.0: "fiducial",
    50.0: "LC_p50",
    100.0: "LC_p",
}

PRE_LINEAR_KEYS = (
    "primordial",
    "bphi_f2",
    "advection",
    "bphidelta",
    "bphi_b2",
    "bphi_bK2",
    "stochastic",
)
STANDARD_LINEAR_KEYS = (
    "halo_primordial",
    "halo_bphi_f2",
    "halo_bphi_advection",
    "halo_bphidelta",
    "halo_bphi_b2",
    "halo_bphi_bK2",
    "halo_reconstruction",
    "residual_stochastic_linear",
)
STANDARD_QUADRATIC_KEYS = (
    "halo_bphi_B0",
    "halo_bphi_sq_advection",
    "halo_bphi_sq_F2",
    "halo_bphi_sq_b2",
    "halo_bphi_sq_bK2",
    "halo_bphi_sq_reconstruction",
    "halo_bphi_bphidelta",
    "halo_bphi2_operator",
    "residual_stochastic_quadratic",
)
STANDARD_LINEAR_BPHI_KEYS = (
    "halo_bphi_f2",
    "halo_bphi_advection",
    "halo_bphi_b2",
    "halo_bphi_bK2",
    "halo_reconstruction",
    "residual_stochastic_linear",
)
STANDARD_QUADRATIC_BPHI_KEYS = (
    "halo_bphi_B0",
    "halo_bphi_bphidelta",
)
STANDARD_QUADRATIC_BPHI2_KEYS = (
    "halo_bphi_sq_advection",
    "halo_bphi_sq_F2",
    "halo_bphi_sq_b2",
    "halo_bphi_sq_bK2",
    "halo_bphi_sq_reconstruction",
    "residual_stochastic_quadratic",
)


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


def full_vector(components: dict[str, Any], name: str) -> np.ndarray:
    if name not in components:
        raise KeyError(f"missing direct finite component {name!r}")
    value = np.asarray(components[name], dtype=np.float64)
    if value.shape != (120,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid direct finite component {name}: {value.shape}")
    return value


def bphi_from_b1(b1: float) -> float:
    return 2.0 * DELTA_C * (float(b1) - P_BPHI)


@dataclass(frozen=True)
class DirectRuntime:
    base: Any
    assets: Any
    contexts: dict[str, Any]
    pre_closure: dict[str, Any]
    p_matrix: dict[str, np.ndarray]
    b_matrices: dict[str, dict[str, np.ndarray]]
    marisa_finite: dict[str, np.ndarray]


def load_runtime() -> DirectRuntime:
    for path in (
        RECOVERY_RUNNER,
        FULL_RUNNER,
        P_MATRIX,
        MARISA_FINITE,
        MARISA_FINITE_MANIFEST,
        *B_MATRICES.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(MARISA_FINITE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError("Marisa finite adaptive tree manifest is not pass")

    recovery = import_path("_direct_finite_recovery", RECOVERY_RUNNER)
    full = import_path("_direct_finite_full", FULL_RUNNER)
    base, assets, contexts, _runtime, _old_matrices = full.build_runtime(recovery)
    if set(contexts) != {"pre", "standard"}:
        raise ValueError("direct finite base context registry drift")
    pre_closure = inspect.getclosurevars(contexts["pre"].evaluate_model).nonlocals
    required_closure = {
        "dm_quadratic",
        "dm_total",
        "dm_tree",
        "nbar",
        "reference",
        "response_templates",
        "tree",
        "v0p9",
    }
    if not required_closure.issubset(pre_closure):
        raise ValueError("pre finite model closure is incomplete")

    p_matrix = load_npz(P_MATRIX)
    b_matrices = {name: load_npz(path) for name, path in B_MATRICES.items()}
    marisa_finite = load_npz(MARISA_FINITE)
    if not np.array_equal(
        np.asarray(marisa_finite["indices"], dtype=np.int64), B_INDICES
    ):
        raise ValueError("Marisa finite adaptive indices drift")
    if not all(
        math.isclose(float(marisa_finite[name]), expected, rel_tol=0.0, abs_tol=2e-14)
        for name, expected in (
            ("b_rec", B_REC),
            ("p_rec", P_REC),
            ("bphi_rec", BPHI_REC),
        )
    ):
        raise ValueError("Marisa finite adaptive reconstruction constants drift")
    return DirectRuntime(
        base=base,
        assets=assets,
        contexts=contexts,
        pre_closure=pre_closure,
        p_matrix=p_matrix,
        b_matrices=b_matrices,
        marisa_finite=marisa_finite,
    )


@dataclass(frozen=True)
class DirectP0Model:
    pl: np.ndarray
    pl_over_m: np.ndarray
    pl_over_m2: np.ndarray

    def __post_init__(self) -> None:
        for value in (self.pl, self.pl_over_m, self.pl_over_m2):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (P_COUNT,) or not np.all(np.isfinite(array)):
                raise ValueError("invalid direct P0 lattice template")
            if not np.all(array > 0.0):
                raise ValueError("direct P0 lattice template is not positive")

    def __call__(
        self,
        f_nl: float,
        b1: float,
        pshot_residual: float,
    ) -> np.ndarray:
        """Evaluate the finite tree halo auto-power at the current parameters."""

        bphi = bphi_from_b1(b1)
        return (
            float(b1) ** 2 * self.pl
            + 2.0 * float(f_nl) * float(b1) * bphi * self.pl_over_m
            + float(f_nl) ** 2 * bphi**2 * self.pl_over_m2
            + float(pshot_residual)
        )


class DirectB000Model:
    """Direct finite tree B000 callable for one reconstruction method."""

    def __init__(self, runtime: DirectRuntime, method: str):
        if method not in {"pre", "standard", "marisa"}:
            raise ValueError(method)
        self.runtime = runtime
        self.method = method
        self.base_context = runtime.contexts[
            "pre" if method == "pre" else "standard"
        ]
        self.names = tuple(self.base_context.names)
        if self.names != (
            "b1",
            "b2",
            "gamma2",
            "Ashot_residual",
            "Bshot_residual",
        ):
            raise ValueError(f"unexpected B nuisance registry: {self.names}")

    def nuisance_array(self, mapping: dict[str, float]) -> np.ndarray:
        return np.asarray([float(mapping[name]) for name in self.names], dtype=np.float64)

    def _pre_components(
        self, parameters: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        closure = self.runtime.pre_closure
        b1 = float(parameters["b1"])
        response = closure["v0p9"].pre_png.response_components(
            tree=closure["tree"],
            templates=closure["response_templates"],
            parameters=parameters,
            b1=b1,
            p=1.0,
            nbar=float(closure["nbar"]),
            dm_tree=np.asarray(closure["dm_tree"], dtype=np.float64),
            dm_total=np.asarray(closure["dm_total"], dtype=np.float64),
            dm_quadratic=np.asarray(closure["dm_quadratic"], dtype=np.float64),
            response_model="finite_halo_tree",
            bphi_override=bphi_from_b1(b1),
            preserve_universal_bphidelta=True,
        )
        linear = sum(
            (full_vector(response, name) for name in PRE_LINEAR_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        quadratic = full_vector(response, "halo_quadratic_deterministic") + full_vector(
            response, "halo_quadratic_stochastic"
        )
        return linear, quadratic

    @staticmethod
    def _standard_components(
        components: dict[str, Any], b1: float
    ) -> tuple[np.ndarray, np.ndarray]:
        old_bphi = 2.0 * DELTA_C * (float(b1) - 1.0)
        new_bphi = bphi_from_b1(b1)
        if abs(old_bphi) < 1.0e-10:
            raise ValueError("p=1 bphi is too close to zero for finite component map")
        ratio = new_bphi / old_bphi

        mapped: dict[str, np.ndarray] = {
            name: full_vector(components, name).copy()
            for name in set(STANDARD_LINEAR_KEYS + STANDARD_QUADRATIC_KEYS)
        }
        for name in STANDARD_LINEAR_BPHI_KEYS:
            mapped[name] *= ratio
        for name in STANDARD_QUADRATIC_BPHI_KEYS:
            mapped[name] *= ratio
        for name in STANDARD_QUADRATIC_BPHI2_KEYS:
            mapped[name] *= ratio**2
        linear = sum(
            (mapped[name] for name in STANDARD_LINEAR_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        quadratic = sum(
            (mapped[name] for name in STANDARD_QUADRATIC_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        return linear, quadratic

    def _direct_parts(
        self, mapping: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        nuisance = self.nuisance_array(mapping)
        _unused, parameters, components = self.base_context.evaluate_model(0.0, nuisance)
        gaussian = full_vector(components, "gaussian_tree") + full_vector(
            components, "gaussian_stochastic"
        )
        if self.method == "pre":
            linear, quadratic = self._pre_components(parameters)
            adaptive = {
                "gaussian_linear": np.zeros(B_COUNT),
                "gaussian_quadratic": np.zeros(B_COUNT),
                "png_linear_cross": np.zeros(B_COUNT),
            }
        else:
            linear, quadratic = self._standard_components(
                components, float(parameters["b1"])
            )
            adaptive = {
                "gaussian_linear": np.zeros(B_COUNT),
                "gaussian_quadratic": np.zeros(B_COUNT),
                "png_linear_cross": np.zeros(B_COUNT),
            }
            if self.method == "marisa":
                finite = self.runtime.marisa_finite
                b1 = float(parameters["b1"])
                bphi = bphi_from_b1(b1)
                adaptive = {
                    "gaussian_linear": (
                        b1**4
                        * (BPHI_REC / B_REC)
                        * np.asarray(finite["gaussian_linear_basis"], dtype=np.float64)
                    ),
                    "gaussian_quadratic": (
                        b1**4
                        * (BPHI_REC / B_REC) ** 2
                        * np.asarray(
                            finite["gaussian_quadratic_basis"], dtype=np.float64
                        )
                    ),
                    "png_linear_cross": (
                        b1**3
                        * bphi
                        * (BPHI_REC / B_REC)
                        * np.asarray(
                            finite["png_linear_cross_basis"], dtype=np.float64
                        )
                    ),
                }
                linear = linear.copy()
                quadratic = quadratic.copy()
                linear[B_INDICES] += adaptive["gaussian_linear"]
                quadratic[B_INDICES] += (
                    adaptive["gaussian_quadratic"]
                    + adaptive["png_linear_cross"]
                )
        return gaussian, linear, quadratic, parameters, adaptive

    def __call__(self, f_nl: float, mapping: dict[str, float]) -> np.ndarray:
        """Evaluate B000(fNL, theta) at the current profiler parameters."""

        gaussian, linear, quadratic, _parameters, _adaptive = self._direct_parts(mapping)
        finite = (
            gaussian
            + float(f_nl) * linear
            + float(f_nl) ** 2 * quadratic
        )
        return np.asarray(finite[B_INDICES], dtype=np.float64)

    def parts(self, mapping: dict[str, float]) -> dict[str, np.ndarray]:
        gaussian, linear, quadratic, _parameters, adaptive = self._direct_parts(mapping)
        return {
            "gaussian": np.asarray(gaussian[B_INDICES], dtype=np.float64),
            "finite_linear_coefficient": np.asarray(
                linear[B_INDICES], dtype=np.float64
            ),
            "finite_quadratic_coefficient": np.asarray(
                quadratic[B_INDICES], dtype=np.float64
            ),
            **{f"adaptive_{name}": value for name, value in adaptive.items()},
        }


def p_target_samples(runtime: DirectRuntime, truth: float) -> np.ndarray:
    key = TRUTH_KEYS[float(truth)]
    count = 500 if float(truth) == 0.0 else 250
    samples = np.asarray(runtime.p_matrix[f"{key}_pre_P0"], dtype=np.float64)
    if samples.shape[0] < count or samples.shape[1] != P_COUNT:
        raise ValueError(f"invalid P target samples for {truth}: {samples.shape}")
    return samples[:count]


def b_target_samples(
    runtime: DirectRuntime, method: str, truth: float
) -> np.ndarray:
    matrix = runtime.b_matrices["standard" if method == "pre" else method]
    stage = "pre" if method == "pre" else "post"
    key = TRUTH_KEYS[float(truth)]
    count = 500 if float(truth) == 0.0 else 250
    samples = np.asarray(matrix[f"{key}_{stage}_B000"], dtype=np.float64)
    if samples.shape[0] < count or samples.shape[1] != 120:
        raise ValueError(f"invalid B target samples for {method}/{truth}: {samples.shape}")
    return samples[:count, B_INDICES]


def p_fiducial_samples(runtime: DirectRuntime) -> np.ndarray:
    samples = np.asarray(runtime.p_matrix["fiducial_pre_P0"], dtype=np.float64)
    if samples.shape != (500, P_COUNT):
        raise ValueError("invalid P Fid500")
    return samples


def b_fiducial_samples(runtime: DirectRuntime, method: str) -> np.ndarray:
    matrix = runtime.b_matrices["standard" if method == "pre" else method]
    stage = "pre" if method == "pre" else "post"
    samples = np.asarray(matrix[f"fiducial_{stage}_B000"], dtype=np.float64)
    if samples.shape != (500, 120):
        raise ValueError(f"invalid B Fid500 for {method}")
    return samples[:, B_INDICES]


def b_geometry(runtime: DirectRuntime) -> tuple[np.ndarray, np.ndarray]:
    standard = runtime.b_matrices["standard"]
    marisa = runtime.b_matrices["marisa"]
    pairs = np.asarray(standard["B000_k"], dtype=np.float64)
    edges = np.asarray(standard["B000_k_edges"], dtype=np.float64)
    if not np.allclose(pairs, marisa["B000_k"], rtol=0.0, atol=2.0e-6):
        raise ValueError("standard/Marisa B geometry drift")
    finite_edges = np.asarray(runtime.marisa_finite["edges"], dtype=np.float64)
    expected_edges = edges[B_INDICES].reshape(B_COUNT, 4)
    if not np.array_equal(finite_edges, expected_edges):
        raise ValueError("Marisa finite adaptive edge geometry drift")
    return pairs[B_INDICES], edges[B_INDICES]


def direct_model_closure(runtime: DirectRuntime) -> dict[str, Any]:
    mapping = {
        "b1": 1.93,
        "b2": -0.60,
        "gamma2": -0.44,
        "Ashot_residual": -1.0,
        "Bshot_residual": -0.1,
    }
    records: dict[str, Any] = {}
    for method in ("pre", "standard", "marisa"):
        model = DirectB000Model(runtime, method)
        parts = model.parts(mapping)
        step = 37.0
        zero = model(0.0, mapping)
        plus = model(step, mapping)
        minus = model(-step, mapping)
        odd = (plus - minus) / (2.0 * step)
        even = (plus + minus - 2.0 * zero) / (2.0 * step**2)
        linear_error = float(
            np.linalg.norm(odd - parts["finite_linear_coefficient"])
            / max(float(np.linalg.norm(parts["finite_linear_coefficient"])), 1.0)
        )
        quadratic_error = float(
            np.linalg.norm(even - parts["finite_quadratic_coefficient"])
            / max(float(np.linalg.norm(parts["finite_quadratic_coefficient"])), 1.0)
        )
        records[method] = {
            "odd_coefficient_relative_l2": linear_error,
            "even_coefficient_relative_l2": quadratic_error,
            "quadratic_nonzero": bool(
                np.any(parts["finite_quadratic_coefficient"] != 0.0)
            ),
            "adaptive_quadratic_nonzero": bool(
                method != "marisa"
                or np.any(parts["adaptive_gaussian_quadratic"] != 0.0)
            ),
            "pass": bool(
                linear_error < 2.0e-12
                and quadratic_error < 2.0e-12
                and np.any(parts["finite_quadratic_coefficient"] != 0.0)
                and (
                    method != "marisa"
                    or np.any(parts["adaptive_gaussian_quadratic"] != 0.0)
                )
            ),
        }
    records["pass"] = bool(all(records[name]["pass"] for name in ("pre", "standard", "marisa")))
    return records

