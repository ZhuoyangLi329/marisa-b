#!/usr/bin/env python3
"""用当前 active 四列表生成 z=0.5 pre finite-PNG tree 至 kmax=0.30。

代码大纲
========
1. 复用 active production tree integrator，但显式传入当前四列表和独立输出目录。
2. 生成原始 kmax<=0.30 的105个 pair；likelihood registry另排除index0和0.155 shell。
3. 对现有0.20 tree的55个公共raw bins逐字段做active-table版本回归。
4. 输出NPZ/manifest及输入哈希；不读取任何历史目录资产。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model")
CODES = ROOT / "marisa_b_core"
MARISA = CODES
PRE_RUNNER = MARISA / "scripts/production/run_pre_recon_halo_png_uplift_v1.py"
MATRIX = (
    ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/baseline_pre_standard_matrix.npz"
)
ACTIVE_TABLE = (
    ROOT / "analysis/quijote_halo_z0p5_standard_noisy_shift_v1/stochastic_png_k1"
    / "symbolic_canary/quijote_z0p5_plin_pphi_m.dat"
)
NATIVE_BINARY = MARISA / "build/halo_v1/marisa_b_triangle"
REFERENCE_TREE = (
    ROOT / "analysis/quijote_halo_z0p5_pre_direct_finite_tree_kmax0p20_v1/theory"
    / "pre_halo_tree_finite_png_z0p5_kmax0p20_nrad4_nmu80.npz"
)
TAG = "quijote_halo_z0p5_pre_active_direct_finite_tree_kmax0p30_v1"
ANALYSIS = ROOT / "analysis" / TAG
THEORY = ANALYSIS / "theory"
OUTPUT_NPZ = THEORY / "pre_active_finite_tree_kmax0p30_nrad4_nmu80.npz"
NATIVE_JSON = OUTPUT_NPZ.with_suffix(".json")
OUTPUT_JSON = THEORY / "pre_active_finite_tree_kmax0p30_manifest.json"
TREE_B1 = 1.9326526728794324
K_FUND = 2.0 * np.pi / 1000.0
EXCLUDED_SHELL = 0.15539149940013885


def import_path(name: str, path: Path) -> Any:
    """按路径导入 active production runner。"""

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    """计算SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def extended_registry(geometry: np.ndarray, cut: float) -> np.ndarray:
    """应用冻结index0和0.155-shell exclusion。"""

    mask = np.max(geometry, axis=1) <= float(cut) + 1.0e-12
    mask[0] = False
    mask &= ~np.any(np.isclose(geometry, EXCLUDED_SHELL, rtol=0.0, atol=5.0e-7), axis=1)
    return np.flatnonzero(mask)


def main() -> None:
    """生产pre active0.30 theory并做公共bin回归。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers必须在1--8")
    for path in (PRE_RUNNER, MATRIX, ACTIVE_TABLE, NATIVE_BINARY, REFERENCE_TREE):
        if not path.is_file():
            raise FileNotFoundError(path)
    production = PRE_RUNNER.parent
    if str(production) not in sys.path:
        sys.path.insert(0, str(production))
    pre = import_path("_pre_active_direct_finite_k30", PRE_RUNNER)
    geometry = pre.load_measurement_geometry(MATRIX)
    dummy = THEORY / "unused"
    paths = pre.Paths(
        root=MARISA, data_root=ROOT, output_root=ROOT, analysis=ANALYSIS,
        figures=ANALYSIS / "unused_figures", manifest=THEORY / "manifest.json",
        theory=THEORY, fid_matrix=MATRIX, matched_matrix=MATRIX,
        frozen_fitter=dummy, frozen_contract=dummy, b_templates=dummy,
        p_templates=dummy, png_table=ACTIVE_TABLE, native_binary=NATIVE_BINARY,
        dm_linear=dummy, dm_quadratic=dummy, tree_templates=OUTPUT_NPZ,
    )
    product = pre.produce_tree_templates(
        paths, geometry, b1=TREE_B1, nmu=80, nradial=4,
        workers=args.workers, overwrite=args.overwrite, kfund=K_FUND, kmax=0.30,
    )
    with np.load(OUTPUT_NPZ, allow_pickle=False) as handle:
        high = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(REFERENCE_TREE, allow_pickle=False) as handle:
        reference = {name: np.asarray(handle[name]) for name in handle.files}
    high_indices = np.asarray(high["data_indices"], dtype=np.int64)
    reference_indices = np.asarray(reference["data_indices"], dtype=np.int64)
    high_lookup = {int(index): offset for offset, index in enumerate(high_indices)}
    positions = np.asarray([high_lookup[int(index)] for index in reference_indices])
    fields = [
        name for name, value in reference.items()
        if np.asarray(value).shape == (reference_indices.size,)
        and name in high and np.asarray(high[name]).shape == (high_indices.size,)
    ]
    closure = {}
    for name in fields:
        observed = np.asarray(high[name], dtype=float)[positions]
        expected = np.asarray(reference[name], dtype=float)
        closure[name] = float(
            np.linalg.norm(observed - expected) / max(float(np.linalg.norm(expected)), 1.0)
        )
    maximum = max(closure.values())
    registry30 = extended_registry(np.asarray(geometry.weighted_pairs), 0.30)
    registry28 = extended_registry(np.asarray(geometry.weighted_pairs), 0.28)
    missing = [int(index) for index in registry30 if int(index) not in high_lookup]
    checks = {
        "raw_pair_count_105": int(high_indices.size) == 105,
        "likelihood_registry_count_90": int(registry30.size) == 90,
        "kmax0p28_0p30_registry_identity": bool(np.array_equal(registry28, registry30)),
        "all_likelihood_indices_present": not missing,
        "all_numeric_arrays_finite": bool(all(
            np.all(np.isfinite(value))
            for value in high.values()
            if np.asarray(value).dtype.kind in "biufc"
        )),
        "finite_box_projection_version_present": bool(
            np.asarray(high["finite_box_projection_version"]).size == 1
        ),
        "component_recombination_pass": bool(
            product["validation"]["max_relative_recombination_error"] < 3.0e-12
            and product["validation"]["finite_tree_fNL2_max_relative_recombination_error"] < 3.0e-12
            and product["validation"]["finite_tree_fNL2_stochastic_max_relative_recombination_error"] < 3.0e-12
        ),
        "reference0p20_relative_l2": closure,
        "maximum_reference0p20_relative_l2": maximum,
        "maximum_reference0p20_below_one_percent": maximum < 0.01,
    }
    if not all(value for key, value in checks.items() if key not in {
        "reference0p20_relative_l2", "maximum_reference0p20_relative_l2"
    }):
        raise ValueError(f"pre active0.30 checks失败：{checks}, missing={missing}")
    payload = {
        "schema": "quijote-halo-z0p5-pre-active-finite-tree-kmax0p30-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_with_active_table_version_drift",
        "scope": {
            "redshift": 0.5, "kmax_h_mpc": 0.30,
            "raw_indices": high_indices.tolist(), "likelihood_indices": registry30.tolist(),
            "finite_png_order": 2, "fully_tree": True, "matter_loop": False,
            "excluded_index": 0, "excluded_shell_h_mpc": EXCLUDED_SHELL,
        },
        "checks": checks, "producer_payload": product,
        "inputs": {
            "matrix": {"path": str(MATRIX), "sha256": sha256(MATRIX)},
            "active_table": {"path": str(ACTIVE_TABLE), "sha256": sha256(ACTIVE_TABLE)},
            "reference_tree0p20": {"path": str(REFERENCE_TREE), "sha256": sha256(REFERENCE_TREE)},
            "native_binary": {"path": str(NATIVE_BINARY), "sha256": sha256(NATIVE_BINARY)},
            "production_runner": {"path": str(PRE_RUNNER), "sha256": sha256(PRE_RUNNER)},
            "runner": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
            "native_sidecar": {"path": str(NATIVE_JSON), "sha256": sha256(NATIVE_JSON)},
        },
        "output": {"path": str(OUTPUT_NPZ), "sha256": sha256(OUTPUT_NPZ)},
    }
    atomic_json(OUTPUT_JSON, payload)
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
