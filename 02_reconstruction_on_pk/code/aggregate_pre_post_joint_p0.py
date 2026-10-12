#!/usr/bin/env python3
"""构造 pre+post 联合 24-bin P0 矩阵（四对组合，Fid500/LCp100 matched）。

代码大纲
========
1. 校验五方法矩阵的 audit 与 sha256，确认四对组合的各通道样本存在。
2. 逐 realization 堆叠 [P_pre(12), P_post(12)]：pre 与 post 的 P0 来自同一
   realization 的同一快照/同一任务（生产 metadata 证实），直接列堆叠即得
   逐 realization 联合样本；Fid500 sample covariance 自动包含完整
   pre--post cross 块。
3. 对每对组合写入堆叠 matched100 targets、24x24 单盒 covariance 与
   通道间逐 bin 相关系数摘要；正定与形状检查沿用四/五方法聚合器标准。
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
FIVE_MATRIX = PART / "inputs/five_method_p0_matrix.npz"
FIVE_AUDIT = PART / "results/five_method_p0_matrix_audit.json"
P_THEORY = PART / "inputs/p0_finite_local_png_lattice.npz"
OUTPUT = PART / "inputs/pre_post_joint_p0_matrix.npz"
AUDIT = PART / "results/pre_post_joint_p0_matrix_audit.json"

PAIRS = ("pre_sto_only",)
PAIR_CHANNELS = {
    "pre_sto_only": ("pre", "sto_only"),
}
NBINS = 12
DIM = 24
VOLUME = 42.875
HARTLAP_24 = (500.0 - 24.0 - 2.0) / 499.0


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


def main() -> None:
    """执行四对组合的联合矩阵聚合与完整性检查。"""

    required = (FIVE_MATRIX, FIVE_AUDIT, P_THEORY)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    five_audit_payload = json.loads(FIVE_AUDIT.read_text(encoding="utf-8"))
    if five_audit_payload.get("status") != "pass":
        raise RuntimeError("five-method matrix audit is not pass")
    if five_audit_payload["output"]["sha256"] != sha256_file(FIVE_MATRIX):
        raise RuntimeError("five-method matrix sha256 drift")

    five = load_npz(FIVE_MATRIX)
    theory = load_npz(P_THEORY)

    arrays: dict[str, np.ndarray] = {
        "pk_k": np.asarray(theory["pk_k"]),
        "pk_edges": np.asarray(theory["pk_edges"]),
        "pk_nmodes": np.asarray(theory["pk_nmodes"]),
        "volume_ratio": np.asarray(VOLUME), "hartlap_24": np.asarray(HARTLAP_24),
    }
    audit: dict[str, Any] = {}

    for pair in PAIRS:
        pre_method, post_method = PAIR_CHANNELS[pair]
        pre_fid = np.asarray(five[f"{pre_method}__fiducial_P0"], dtype=np.float64)
        post_fid = np.asarray(five[f"{post_method}__fiducial_P0"], dtype=np.float64)
        pre_lcp = np.asarray(five[f"{pre_method}__LC_p_P0"], dtype=np.float64)
        post_lcp = np.asarray(five[f"{post_method}__LC_p_P0"], dtype=np.float64)
        if pre_fid.shape != (500, NBINS) or post_fid.shape != (500, NBINS):
            raise ValueError(f"{pair} fiducial shape drift: {pre_fid.shape}/{post_fid.shape}")
        if pre_lcp.shape != (100, NBINS) or post_lcp.shape != (100, NBINS):
            raise ValueError(f"{pair} LC_p shape drift: {pre_lcp.shape}/{post_lcp.shape}")

        joint_fid = np.concatenate((pre_fid, post_fid), axis=1)
        joint_lcp = np.concatenate((pre_lcp, post_lcp), axis=1)
        if not np.all(np.isfinite(joint_fid)) or not np.all(np.isfinite(joint_lcp)):
            raise ValueError(f"{pair} non-finite joint P0")

        covariance = np.cov(joint_fid, rowvar=False, ddof=1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= 0:
            raise ValueError(f"{pair} joint covariance not positive definite")

        target_fnl0 = joint_fid[:100].mean(axis=0)
        target_fnl100 = joint_lcp.mean(axis=0)

        arrays[f"{pair}__fiducial_realizations"] = np.arange(500, dtype=np.int64)
        arrays[f"{pair}__LC_p_realizations"] = np.arange(100, dtype=np.int64)
        arrays[f"{pair}__fiducial_joint_P0"] = joint_fid
        arrays[f"{pair}__LC_p_joint_P0"] = joint_lcp
        arrays[f"{pair}__target_fNL0_matched100"] = target_fnl0
        arrays[f"{pair}__target_fNL100_matched100"] = target_fnl100
        arrays[f"{pair}__covariance_fNL0_fid500_one_box"] = covariance

        # 通道间逐 bin realization 相关系数（同一 realization 对），审计摘要用。
        cross_corr = np.asarray([
            float(np.corrcoef(pre_fid[:, b], post_fid[:, b])[0, 1])
            for b in range(NBINS)
        ])
        audit[pair] = {
            "channels": [pre_method, post_method],
            "pairing": "same realization, same snapshot, same production task (per-realization pre_pk/post_pk)",
            "fiducial_count": 500, "LC_p_count": 100,
            "target_realizations": [0, 99], "covariance_realizations": [0, 499],
            "covariance_minimum_eigenvalue": float(eigenvalues[0]),
            "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "pre_post_same_bin_correlation": cross_corr.tolist(),
            "cross_correlation_min": float(np.min(cross_corr)),
            "cross_correlation_max": float(np.max(cross_corr)),
            "cross_correlation_mean": float(np.mean(cross_corr)),
        }

    atomic_npz(OUTPUT, arrays)
    payload = {
        "schema": "quijote-z0p5-pre-post-joint-p0-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "pairs": list(PAIRS),
        "joint_definition": (
            "24-bin vector [P_pre(12), P_post(12)] stacked per matched realization; "
            "full 24x24 Fid500 sample covariance including pre--post cross blocks"
        ),
        "covariance": {
            "source": "joint Fid500 fNL0 single-realization sample covariance",
            "target_mean_divisor": None,
            "volume_ratio": VOLUME,
            "hartlap_24": HARTLAP_24,
        },
        "checks": audit,
        "inputs": {
            "five_method_matrix": artifact(FIVE_MATRIX),
            "five_method_matrix_audit": artifact(FIVE_AUDIT),
            "P_theory": artifact(P_THEORY),
            "runner": artifact(Path(__file__)),
        },
        "output": artifact(OUTPUT),
    }
    atomic_json(AUDIT, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2, sort_keys=True))
    for pair in PAIRS:
        row = audit[pair]
        print(
            f"{pair}: cond={row['covariance_condition_number']:.3g} "
            f"cross-corr [{row['cross_correlation_min']:.3f},{row['cross_correlation_max']:.3f}]"
        )


if __name__ == "__main__":
    main()
