#!/usr/bin/env python3
"""聚合 pre/std/Marisa/STO+Marisa 的 matched Fid0/LCp100 P0。

代码大纲
========
1. 从已有 baseline 与 STO matrices 读取 pre/std/STO+Marisa P0。
2. 从新补齐的 ordinary-Marisa post-P0-only 600 个 realization 读取 P0。
3. 对所有方法强制相同 12-bin geometry 和 realization ID 契约。
4. 保存原始 Fid500/LCp100 samples、均值、单盒 covariance 与 provenance audit。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
TAG = "quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1"
ANALYSIS = ROOT / "analysis" / TAG
OUTPUT = ANALYSIS / "FOUR_METHOD_P0_MATRIX.npz"
AUDIT = ANALYSIS / "FOUR_METHOD_P0_MATRIX_AUDIT.json"
BASELINE = (
    ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/baseline_pre_standard_matrix.npz"
)
STO = BASELINE.with_name("sto_marisa_matrix.npz")
MARISA_ROOT = ROOT / "result_quijote_halo_z0p5_marisa_p0only_fid500_lcp100_v1/full"
MARISA_MANIFEST = (
    ROOT / "manifests/quijote_halo_z0p5_marisa_p0only_fid500_lcp100_v1"
    / "20260831/tasks_all.jsonl"
)
P_THEORY = (
    ROOT / "analysis/quijote_halo_z0p5_p0_finite_local_png_lattice_kmax0p08_v1"
    / "P0_FINITE_LOCAL_PNG_LATTICE.npz"
)
METHODS = ("pre", "standard", "marisa", "sto_marisa")
EXPECTED = {"fiducial": 500, "LC_p": 100}
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


def read_post_pk(directory: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """读取并认证一个新 Marisa P0-only product。"""

    pk_path = directory / "post_pk.npz"
    metadata_path = directory / "metadata.json"
    done_path = directory / "DONE"
    missing = [str(path) for path in (pk_path, metadata_path, done_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "quijote-halo-z0p5-marisa-post-p0only-product-v1":
        raise RuntimeError(f"schema drift: {metadata_path}")
    if not metadata["measurement"]["geometry_gate"]["pass"] or not metadata["replay_gate"]["pass"]:
        raise RuntimeError(f"failed product gate: {metadata_path}")
    digest = sha256_file(pk_path)
    if digest != metadata["outputs"]["post_pk_sha256"] or digest != done["post_pk_sha256"]:
        raise RuntimeError(f"post-P0 hash drift: {directory}")
    with np.load(pk_path, allow_pickle=False) as stream:
        geometry = {name: np.asarray(stream[name]) for name in ("k", "edges", "nmodes")}
        values = np.asarray(stream["pk"], dtype=np.float64)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid P0 vector: {pk_path}: {values.shape}")
    return values, geometry, metadata


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
    """执行四方法聚合与全部完整性检查。"""

    required = (BASELINE, STO, MARISA_MANIFEST, P_THEORY)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    baseline = load_npz(BASELINE)
    sto = load_npz(STO)
    theory = load_npz(P_THEORY)
    reference = {
        "k": np.asarray(theory["pk_k"]),
        "edges": np.asarray(theory["pk_edges"]),
        "nmodes": np.asarray(theory["pk_nmodes"]),
    }
    arrays: dict[str, np.ndarray] = {
        "pk_k": reference["k"], "pk_edges": reference["edges"], "pk_nmodes": reference["nmodes"],
        "volume_ratio": np.asarray(VOLUME), "hartlap_12": np.asarray(HARTLAP),
    }
    audit: dict[str, Any] = {}
    baseline_geometry = {
        "k": np.asarray(baseline["pk_k"][:12]),
        "edges": np.asarray(baseline["pk_edges"][:13]),
        "nmodes": np.asarray(baseline["pk_nmodes"][:12]),
    }
    sto_geometry = {
        "k": np.asarray(sto["pk_k"][:12]),
        "edges": np.asarray(sto["pk_edges"][:13]),
        "nmodes": np.asarray(sto["pk_nmodes"][:12]),
    }
    audit["baseline_geometry"] = geometry_gate(reference, baseline_geometry)
    audit["sto_geometry"] = geometry_gate(reference, sto_geometry)
    if not audit["baseline_geometry"]["pass"] or not audit["sto_geometry"]["pass"]:
        raise RuntimeError("existing matrix geometry mismatch")

    add_method(
        arrays, audit, "pre",
        baseline["fiducial_pre_P0"][:, :12], baseline["LC_p_pre_P0"][:100, :12],
    )
    add_method(
        arrays, audit, "standard",
        baseline["fiducial_post_P0"][:, :12], baseline["LC_p_post_P0"][:100, :12],
    )
    add_method(
        arrays, audit, "sto_marisa",
        sto["fiducial_post_P0"][:, :12], sto["LC_p_post_P0"][:100, :12],
    )

    marisa_rows: dict[str, list[np.ndarray]] = {"fiducial": [], "LC_p": []}
    marisa_geometry_checks: list[dict[str, Any]] = []
    metadata_digest = hashlib.sha256()
    for cosmology in ("fiducial", "LC_p"):
        for realization in range(EXPECTED[cosmology]):
            directory = MARISA_ROOT / cosmology / f"real{realization:03d}"
            values, geometry, metadata = read_post_pk(directory)
            if str(metadata["cosmology"]) != cosmology or int(metadata["realization"]) != realization:
                raise RuntimeError(f"Marisa identity drift: {directory}")
            expected_fnl = 0.0 if cosmology == "fiducial" else 100.0
            if float(metadata["input_fNL"]) != expected_fnl:
                raise RuntimeError(f"Marisa fNL drift: {directory}")
            gate = geometry_gate(reference, geometry)
            if not gate["pass"]:
                raise RuntimeError(f"Marisa geometry drift: {directory}: {gate}")
            marisa_geometry_checks.append(gate)
            metadata_digest.update(bytes.fromhex(sha256_file(directory / "metadata.json")))
            marisa_rows[cosmology].append(values)
    add_method(
        arrays, audit, "marisa",
        np.vstack(marisa_rows["fiducial"]), np.vstack(marisa_rows["LC_p"]),
    )
    audit["marisa_products"] = {
        "count": sum(EXPECTED.values()),
        "all_geometry_pass": bool(all(item["pass"] for item in marisa_geometry_checks)),
        "maximum_k_difference": float(max(item["k_max_abs_difference"] for item in marisa_geometry_checks)),
        "metadata_digest": metadata_digest.hexdigest(),
    }

    atomic_npz(OUTPUT, arrays)
    payload = {
        "schema": "quijote-z0p5-four-method-p0-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "methods": METHODS,
        "targets": {"fNL0": "fiducial real0--99 mean", "fNL100": "LC_p real0--99 mean"},
        "covariance": {
            "source": "each method's own fNL0 Fid500 single-realization sample covariance",
            "target_mean_divisor": None,
            "volume_ratio": VOLUME,
            "hartlap_12": HARTLAP,
        },
        "checks": audit,
        "inputs": {
            "baseline": artifact(BASELINE), "sto_marisa": artifact(STO),
            "marisa_manifest": artifact(MARISA_MANIFEST), "P_theory": artifact(P_THEORY),
            "runner": artifact(Path(__file__)),
        },
        "output": artifact(OUTPUT),
    }
    atomic_json(AUDIT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
