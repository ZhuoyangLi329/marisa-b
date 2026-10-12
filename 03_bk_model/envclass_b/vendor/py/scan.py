#!/usr/bin/env python3
"""High-k direct finite-fNL fully-tree B000 models for validity scans."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import quijote_halo_z0p5_direct_finite_fnl_models as low


ROOT = low.ROOT
CODES = low.CODES
HIGH_BACKEND = CODES / "quijote_halo_z0p5_shared_fully_tree_context_kmax0p20.py"
PRE_TREE = (
    ROOT
    / "analysis/quijote_halo_z0p5_pre_direct_finite_tree_kmax0p20_v1/theory"
    / "pre_halo_tree_finite_png_z0p5_kmax0p20_nrad4_nmu80.npz"
)
PRE_TREE_MANIFEST = PRE_TREE.with_suffix(".json")
MARISA_FINITE = (
    ROOT
    / "analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p20_v1/theory"
    / "marisa_adaptive_finite_tree_bases.npz"
)
MARISA_FINITE_MANIFEST = MARISA_FINITE.with_suffix(".json")

CUTS = (0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20)
CERTIFIED_INDICES = np.asarray(
    (
        1, 2, 3, 4, 5, 6, 8, 9,
        15, 16, 17, 18, 19, 20, 22, 23,
        29, 30, 31, 32, 33, 35, 36,
        42, 43, 44, 45, 47, 48,
        54, 55, 56, 58, 59,
        65, 66, 68, 69,
        75, 77, 78, 92, 93, 99,
    ),
    dtype=np.int64,
)
EXPECTED_COUNTS = {0.08: 9, 0.10: 14, 0.12: 20, 0.14: 27, 0.15: 27, 0.18: 35, 0.20: 44}


def import_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ScanRuntime:
    low_runtime: low.DirectRuntime
    high_runtime: Any
    pre_tree: dict[str, np.ndarray]
    marisa_finite: dict[str, np.ndarray]
    geometry: np.ndarray


def selected_indices(geometry: np.ndarray, cut: float) -> np.ndarray:
    positions = np.max(np.asarray(geometry)[CERTIFIED_INDICES], axis=1) <= float(cut) + 1.0e-12
    indices = CERTIFIED_INDICES[positions]
    expected = EXPECTED_COUNTS[float(cut)]
    if indices.size != expected:
        raise ValueError(f"kmax={cut:.2f}: {indices.size} indices, expected {expected}")
    return indices


def load_runtime() -> ScanRuntime:
    for path in (
        HIGH_BACKEND,
        PRE_TREE,
        PRE_TREE_MANIFEST,
        MARISA_FINITE,
        MARISA_FINITE_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (PRE_TREE_MANIFEST, MARISA_FINITE_MANIFEST):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"high-k direct finite asset is not pass: {path}")
    low_runtime = low.load_runtime()
    backend = import_path("_direct_finite_kmax_scan_backend", HIGH_BACKEND)
    high_runtime = backend.load_runtime()
    with np.load(PRE_TREE, allow_pickle=False) as handle:
        pre_tree = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(MARISA_FINITE, allow_pickle=False) as handle:
        marisa_finite = {name: np.asarray(handle[name]) for name in handle.files}
    geometry = np.asarray(
        low_runtime.b_matrices["standard"]["B000_k"], dtype=np.float64
    )
    if not np.array_equal(
        np.asarray(marisa_finite["indices"], dtype=np.int64), CERTIFIED_INDICES
    ):
        raise ValueError("Marisa high-k finite index registry drift")
    pre_indices = np.asarray(pre_tree["data_indices"], dtype=np.int64)
    if any(int(index) not in set(pre_indices.tolist()) for index in CERTIFIED_INDICES):
        raise ValueError("pre high-k finite tree misses certified indices")
    for cut in CUTS:
        selected_indices(geometry, cut)
    return ScanRuntime(
        low_runtime=low_runtime,
        high_runtime=high_runtime,
        pre_tree=pre_tree,
        marisa_finite=marisa_finite,
        geometry=geometry,
    )


class DirectFiniteScanB000Model:
    """Evaluate one method on any certified subset up to kmax=0.20."""

    def __init__(self, runtime: ScanRuntime, method: str):
        if method not in {"pre", "standard", "marisa"}:
            raise ValueError(method)
        self.runtime = runtime
        self.method = method
        self.base_context = runtime.high_runtime.contexts[0.20][
            "pre" if method == "pre" else "standard"
        ]
        self.names = tuple(self.base_context.names)
        if self.names != ("b1", "b2", "gamma2", "Ashot_residual", "Bshot_residual"):
            raise ValueError("high-k nuisance registry drift")

    def nuisance_array(self, mapping: dict[str, float]) -> np.ndarray:
        return np.asarray([mapping[name] for name in self.names], dtype=np.float64)

    def _pre_coefficients(
        self, parameters: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        closure = self.runtime.low_runtime.pre_closure
        b1 = float(parameters["b1"])
        response = closure["v0p9"].pre_png.response_components(
            tree=self.runtime.pre_tree,
            templates=closure["response_templates"],
            parameters=parameters,
            b1=b1,
            p=1.0,
            nbar=float(closure["nbar"]),
            dm_tree=np.zeros(120, dtype=np.float64),
            dm_total=np.zeros(120, dtype=np.float64),
            dm_quadratic=np.zeros(120, dtype=np.float64),
            response_model="finite_halo_tree",
            bphi_override=low.bphi_from_b1(b1),
            preserve_universal_bphidelta=True,
        )
        linear = sum(
            (low.full_vector(response, name) for name in low.PRE_LINEAR_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        quadratic = low.full_vector(
            response, "halo_quadratic_deterministic"
        ) + low.full_vector(response, "halo_quadratic_stochastic")
        return linear, quadratic

    def parts(self, mapping: dict[str, float]) -> dict[str, np.ndarray]:
        nuisance = self.nuisance_array(mapping)
        _unused, parameters, components = self.base_context.evaluate_model(0.0, nuisance)
        gaussian = low.full_vector(components, "gaussian_tree") + low.full_vector(
            components, "gaussian_stochastic"
        )
        if self.method == "pre":
            linear, quadratic = self._pre_coefficients(parameters)
            adaptive_linear = np.zeros(120, dtype=np.float64)
            adaptive_quadratic = np.zeros(120, dtype=np.float64)
            adaptive_cross = np.zeros(120, dtype=np.float64)
        else:
            linear, quadratic = low.DirectB000Model._standard_components(
                components, float(parameters["b1"])
            )
            adaptive_linear = np.zeros(120, dtype=np.float64)
            adaptive_quadratic = np.zeros(120, dtype=np.float64)
            adaptive_cross = np.zeros(120, dtype=np.float64)
            if self.method == "marisa":
                finite = self.runtime.marisa_finite
                indices = np.asarray(finite["indices"], dtype=np.int64)
                b1 = float(parameters["b1"])
                bphi = low.bphi_from_b1(b1)
                adaptive_linear[indices] = (
                    b1**4
                    * (low.BPHI_REC / low.B_REC)
                    * np.asarray(finite["gaussian_linear_basis"], dtype=np.float64)
                )
                adaptive_quadratic[indices] = (
                    b1**4
                    * (low.BPHI_REC / low.B_REC) ** 2
                    * np.asarray(finite["gaussian_quadratic_basis"], dtype=np.float64)
                )
                adaptive_cross[indices] = (
                    b1**3
                    * bphi
                    * (low.BPHI_REC / low.B_REC)
                    * np.asarray(finite["png_linear_cross_basis"], dtype=np.float64)
                )
                linear = linear + adaptive_linear
                quadratic = quadratic + adaptive_quadratic + adaptive_cross
        return {
            "gaussian": gaussian,
            "linear_coefficient": linear,
            "quadratic_coefficient": quadratic,
            "adaptive_linear": adaptive_linear,
            "adaptive_quadratic": adaptive_quadratic,
            "adaptive_cross": adaptive_cross,
        }

    def __call__(
        self, f_nl: float, mapping: dict[str, float], indices: np.ndarray
    ) -> np.ndarray:
        parts = self.parts(mapping)
        prediction = (
            parts["gaussian"]
            + float(f_nl) * parts["linear_coefficient"]
            + float(f_nl) ** 2 * parts["quadratic_coefficient"]
        )
        return np.asarray(prediction[np.asarray(indices, dtype=np.int64)], dtype=np.float64)


def low_k_overlap(runtime: ScanRuntime) -> dict[str, Any]:
    mapping = {
        "b1": 1.93,
        "b2": -0.60,
        "gamma2": -0.44,
        "Ashot_residual": -1.0,
        "Bshot_residual": -0.1,
    }
    indices = selected_indices(runtime.geometry, 0.08)
    records = {}
    for method in ("pre", "standard", "marisa"):
        high = DirectFiniteScanB000Model(runtime, method)
        low_model = low.DirectB000Model(runtime.low_runtime, method)
        differences = {}
        for f_nl in (-100.0, 0.0, 100.0):
            observed = high(f_nl, mapping, indices)
            expected = low_model(f_nl, mapping)
            differences[f"{f_nl:+.0f}"] = float(
                np.linalg.norm(observed - expected)
                / max(float(np.linalg.norm(expected)), 1.0)
            )
        records[method] = {
            "relative_l2": differences,
            "maximum": max(differences.values()),
            "pass": max(differences.values()) < 2.0e-12,
        }
    records["pass"] = all(records[method]["pass"] for method in ("pre", "standard", "marisa"))
    return records
