#!/usr/bin/env python3
"""在冻结四方法 P0 矩阵上扩展 STO-only，生成五方法 matched Fid0/LCp100 P0 矩阵。

代码大纲
========
1. 校验 v2 冻结四方法矩阵的 audit 与 sha256，通过后原样继承其全部数组，
   保证 pre/standard/marisa/sto_marisa 的数值 bit 级不变。
2. 从旧仓库 sto_only_matrix.npz（只读）读取 STO-only 的 Fid500/LCp100 post P0；
   STO-only 生产使用 matter-assisted N+Env 权重（lam1_dm_nenv 目录族），
   位置不动、只挂 INDWEIGHT，与 sto_marisa 共用同一套 STO 权重。
3. 对 STO-only 强制与理论 lattice 相同的 12-bin geometry 门、realization 契约、
   协方差正定检查，与四方法聚合器同一标准。
4. 保存五方法矩阵与 provenance audit；所有输出只写 v2 本部分目录。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "02_reconstruction_on_pk"
FOUR_MATRIX = PART / "inputs/four_method_p0_matrix.npz"
FOUR_AUDIT = PART / "results/four_method_p0_matrix_audit.json"
OLD_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
STO_ONLY = (
    OLD_ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/sto_only_matrix.npz"
)
P_THEORY = PART / "inputs/p0_finite_local_png_lattice.npz"
OUTPUT = PART / "inputs/five_method_p0_matrix.npz"
AUDIT = PART / "results/five_method_p0_matrix_audit.json"
INHERITED_METHODS = ("pre", "standard", "marisa", "sto_marisa")
METHODS = ("pre", "standard", "marisa", "sto_only", "sto_marisa")
VOLUME = 42.875
HARTLAP = (500.0 - 12.0 - 2.0) / 499.0


def sha256_file(path: Path) -> str:
    """返回文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """返回文件路径、大小和 SHA256。"""

    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """原子写压缩 NPZ。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 NPZ 为普通 ndarray 字典。"""

    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def geometry_gate(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    """比较 P0 edges/nmodes/k。"""

    result = {
        "edges_exact": bool(np.array_equal(candidate["edges"], reference["edges"])),
        "nmodes_exact": bool(np.array_equal(candidate["nmodes"], reference["nmodes"])),
        "k_close": bool(np.allclose(candidate["k"], reference["k"], rtol=5e-5, atol=5e-8)),
        "k_max_abs_difference": float(np.max(np.abs(candidate["k"] - reference["k"]))),
    }
    result["pass"] = bool(result["edges_exact"] and result["nmodes_exact"] and result["k_close"])
    return result


def add_method(
    arrays: dict[str, np.ndarray], audit: dict[str, Any], method: str,
    fiducial: np.ndarray, lcp: np.ndarray,
) -> None:
    """写入一种方法的 samples、targets、covariance 和数值诊断。"""

    fiducial = np.asarray(fiducial, dtype=np.float64)
    lcp = np.asarray(lcp, dtype=np.float64)
    if fiducial.shape != (500, 12) or lcp.shape != (100, 12):
        raise ValueError(f"{method} sample shape drift: {fiducial.shape}/{lcp.shape}")
    if not np.all(np.isfinite(fiducial)) or not np.all(np.isfinite(lcp)):
        raise ValueError(f"{method} non-finite P0")
    covariance = np.cov(fiducial, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError(f"{method} covariance not positive definite")
    arrays[f"{method}__fiducial_realizations"] = np.arange(500, dtype=np.int64)
    arrays[f"{method}__LC_p_realizations"] = np.arange(100, dtype=np.int64)
    arrays[f"{method}__fiducial_P0"] = fiducial
    arrays[f"{method}__LC_p_P0"] = lcp
    arrays[f"{method}__target_fNL0_matched100"] = fiducial[:100].mean(axis=0)
    arrays[f"{method}__target_fNL100_matched100"] = lcp.mean(axis=0)
    arrays[f"{method}__covariance_fNL0_fid500_one_box"] = covariance
    audit[method] = {
        "fiducial_count": 500,
        "LC_p_count": 100,
        "target_realizations": [0, 99],
        "covariance_realizations": [0, 499],
        "covariance_minimum_eigenvalue": float(eigenvalues[0]),
        "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "fiducial_mean_range": [float(fiducial.mean(axis=0).min()), float(fiducial.mean(axis=0).max())],
        "LC_p_mean_range": [float(lcp.mean(axis=0).min()), float(lcp.mean(axis=0).max())],
    }


def main() -> None:
    """执行五方法聚合与全部完整性检查。"""

    required = (FOUR_MATRIX, FOUR_AUDIT, STO_ONLY, P_THEORY)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    # 1) 冻结四方法矩阵的链路校验：audit 状态与 sha256 必须一致。
    four_audit_payload = json.loads(FOUR_AUDIT.read_text(encoding="utf-8"))
    if four_audit_payload.get("status") != "pass":
        raise RuntimeError("frozen four-method matrix audit is not pass")
    if four_audit_payload["output"]["sha256"] != sha256_file(FOUR_MATRIX):
        raise RuntimeError("frozen four-method matrix sha256 drift")
    inherited_methods = list(four_audit_payload["methods"])
    if tuple(inherited_methods) != INHERITED_METHODS:
        raise RuntimeError(f"inherited method set drift: {inherited_methods}")

    # 2) 原样继承四方法矩阵的全部数组（bit 级不变，不重算）。
    arrays = load_npz(FOUR_MATRIX)
    for method in INHERITED_METHODS:
        for suffix in (
            "fiducial_P0", "LC_p_P0", "target_fNL0_matched100",
            "target_fNL100_matched100", "covariance_fNL0_fid500_one_box",
        ):
            if f"{method}__{suffix}" not in arrays:
                raise KeyError(f"missing inherited key: {method}__{suffix}")
    if float(arrays["volume_ratio"]) != VOLUME or not np.isclose(float(arrays["hartlap_12"]), HARTLAP):
        raise RuntimeError("inherited covariance scaling drift")

    # 3) 读取 STO-only 源矩阵并对理论 lattice 做 geometry 门。
    sto_only = load_npz(STO_ONLY)
    theory = load_npz(P_THEORY)
    reference = {
        "k": np.asarray(theory["pk_k"]),
        "edges": np.asarray(theory["pk_edges"]),
        "nmodes": np.asarray(theory["pk_nmodes"]),
    }
    sto_only_geometry = {
        "k": np.asarray(sto_only["pk_k"][:12]),
        "edges": np.asarray(sto_only["pk_edges"][:13]),
        "nmodes": np.asarray(sto_only["pk_nmodes"][:12]),
    }
    sto_only_gate = geometry_gate(reference, sto_only_geometry)
    if not sto_only_gate["pass"]:
        raise RuntimeError(f"sto_only geometry drift: {sto_only_gate}")

    # 4) 新增 sto_only 方法条目（500 fiducial / 100 LC_p，前 12 bin）。
    audit: dict[str, Any] = {}
    add_method(
        arrays, audit, "sto_only",
        sto_only["fiducial_post_P0"][:, :12], sto_only["LC_p_post_P0"][:100, :12],
    )
    if f"sto_only__target_fNL0_matched100" not in arrays:
        raise RuntimeError("sto_only keys missing after add_method")

    atomic_npz(OUTPUT, arrays)
    payload = {
        "schema": "quijote-z0p5-five-method-p0-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "methods": list(METHODS),
        "sto_only_definition": (
            "matter-assisted N+Env STO weighting (r10, lambda_th=1, env_source=dm, "
            "weight_source=dm, number weighting, components void/sheet/filament/knot); "
            "halo positions unchanged with INDWEIGHT=sto weights, no gra-recon"
        ),
        "targets": {"fNL0": "fiducial real0--99 mean", "fNL100": "LC_p real0--99 mean"},
        "covariance": {
            "source": "each method's own fNL0 Fid500 single-realization sample covariance",
            "target_mean_divisor": None,
            "volume_ratio": VOLUME,
            "hartlap_12": HARTLAP,
        },
        "inheritance": {
            "frozen_four_method_matrix": {
                **artifact(FOUR_MATRIX),
                "methods": inherited_methods,
                "audit_status": four_audit_payload["status"],
                "arrays_unchanged": True,
                "checks_inherited_from": str(FOUR_AUDIT),
            },
        },
        "checks": {
            "sto_only_geometry": sto_only_gate,
            **audit,
        },
        "inputs": {
            "four_method_matrix": artifact(FOUR_MATRIX),
            "four_method_matrix_audit": artifact(FOUR_AUDIT),
            "sto_only_matrix": artifact(STO_ONLY),
            "P_theory": artifact(P_THEORY),
            "runner": artifact(Path(__file__)),
        },
        "output": artifact(OUTPUT),
    }
    atomic_json(AUDIT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
