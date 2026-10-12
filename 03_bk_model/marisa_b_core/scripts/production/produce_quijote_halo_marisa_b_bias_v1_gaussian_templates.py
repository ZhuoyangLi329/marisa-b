#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成 Quijote halo MARISA-B bias-v1 Gaussian pre/post 一环模板。

代码大纲
========
1. 读取 z=1 fid500 pre/post B000 aggregate，只从共享的 120 维几何中选择一次
   ``data_indices``；pre/post 因而严格使用同一批 shape 和同一 B000 求积节点。
2. 复用 DM producer 的 ``build_b000_work``/``shell_nodes`` 口径，支持 weighted-k
   单点和两个 k-shell 的 ``k^2 dk`` 平均，再做 ``1/2 int dmu`` 的 B000 投影。
3. 在总次数不超过三次的 10 个 ``(b2,bK2)`` monomial 上，使用条件良好的
   degree-3 Padua 型 10 点设计。每个节点分别调用 native C++ 的
   ``pre_recon_halo_bias_v1_gaussian`` 和
   ``post_recon_halo_bias_v1_gaussian``，不在 Python 中复制一环积分器。
4. 从节点值解出 Btree+B222+B321I+B321II+B411 总理论的多项式系数；默认另外
   计算随机 bias 点，直接比较 native 结果与多项式重组，防止 degree/order、
   stage 或 shape alignment 出错。
5. C++ row 中 ``stochastic_alpha3_basis`` 尚未除以数密度，而 alpha4 basis=1；
   本脚本在 B000 平均后分别除以固定 ``nbar`` 和 ``nbar^2``，输出 joint-fit
   consumer 要求的 residual non-Poisson stochastic basis。
6. pre/post 各写一个互相独立、同名 JSON sidecar 的 NPZ；JSON 明确记录一环
   diagram、bias-v1、固定 cutoff、重构/CIC、P_L、节点条件数和数值自检。native
   halo mode 会先按三条外边长度 canonical sort，再以两条短边定义 q routing；该
   内部排序不改变原始 B000 ``data_indices`` 的对齐。

这里生成的是 Gaussian Gate G 模板，不包含 PNG 响应。halo bias functional 是
``K_n=b1 F_n+b2 D_n+bK2 T_n (n<=4)`` 的固定-cutoff机械一环延拓；脚本不会把它
描述成 Barreira 的 halo 一环推导、renormalized model 或 EFT。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# 必须在 import numpy 前封顶线程数；native 积分和 BLAS 都不得在登录节点超过 8 核。
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def enforce_thread_limit(max_threads: int = 8) -> None:
    """在加载数值库前设置最多 ``max_threads`` 个线程。"""

    for key in THREAD_ENV_KEYS:
        raw = os.environ.get(key)
        try:
            requested = int(raw) if raw is not None and raw.strip() else int(max_threads)
        except ValueError:
            requested = int(max_threads)
        os.environ[key] = str(max(1, min(requested, int(max_threads))))
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


enforce_thread_limit()

import numpy as np

from produce_quijote_dm_marisa_b_b000_theory_vector import (
    build_b000_work,
    prepare_pk_table,
    shell_nodes,
)
from quijote_dm_spt1loop_realspace import QUIJOTE_COSMOLOGY, quijote_power_normalization_audit
from run_marisa_b_triangle import (
    DEFAULT_BUILD_DIR,
    compile_marisa_b,
    run_marisa_b,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MARISA_B_DATA_ROOT", PROJECT_ROOT)).resolve()
DEFAULT_MATRIX = (
    DATA_ROOT
    / "analysis"
    / "quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
    / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid500_b000_pk_matrix.npz"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "analysis" / "theory_vectors" / "marisa_b_halo_bias_v1"
DEFAULT_B1 = 2.7340475186190334
DEFAULT_NBAR = 1.95530218e-4
DEFAULT_R_RECON = 15.0
DEFAULT_B_RECON = 2.7340475186190334
DEFAULT_RECON_CELLSIZE = 8.0
DEFAULT_RECON_CIC_WINDOW_POWER = 4

# K_n 对 b2、bK2 各自线性，所以一环 bispectrum 最多是总三次多项式。
# 采用 total-degree 优先、同一 degree 内 b2 power 递减的稳定且机器可读顺序。
MONOMIAL_POWERS = np.asarray(
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
    ],
    dtype=np.int64,
)

DETERMINISTIC_COMPONENTS = (
    "Btree",
    "B222",
    "B321I",
    "B321II",
    "B411",
    "Bloopterms",
    "B1loop",
    "Btotal",
)
ROW_FIELDS = DETERMINISTIC_COMPONENTS + (
    "stochastic_alpha3_basis",
    "stochastic_alpha4_basis",
)
STAGE_MODES = {
    "pre": "pre_recon_halo_bias_v1_gaussian",
    "post": "post_recon_halo_bias_v1_gaussian",
}

# 新 halo bias-v1 native mode 的精确 routing provenance。这里集中成单一常量，
# 避免 sidecar、NPZ 和 self-test 各自漂移；旧 DM mode 不经过本 producer，也不受影响。
GAUSSIAN_LOOP_ROUTING = (
    "canonical_sorted_external_legs_native_gaussian_diagram_logq_with_spherical_qmin_qmax"
)


@dataclass(frozen=True)
class MeasurementGeometry:
    """fid500 aggregate 中与理论向量对齐所需的最小几何和样本信息。"""

    path: Path
    realization_ids: np.ndarray
    pre_samples: np.ndarray
    post_samples: np.ndarray
    weighted_pairs: np.ndarray
    shell_edges: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    boxsize: float
    meshsize: int


@dataclass(frozen=True)
class StageNodeEvaluation:
    """一个 stage、一个 bias 节点经过 B000 averaging 后的结果。"""

    stage: str
    b2: float
    bK2: float
    vectors: dict[str, np.ndarray]
    conservative_error_proxy: dict[str, np.ndarray]
    chunks: list[dict[str, Any]]
    native_metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """解析生产参数；默认值对应 z=1、R=15 的 fid500 Gate G 口径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="z=1 fid500 pre/post B000 matrix。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--prefix",
        default=(
            "quijote_halo_z1_mmin1e13_r15_marisa_b_bias_v1_gaussian_"
            "diag_kmax0p18_point_nmu12"
        ),
        help="新产物前缀；脚本拒绝覆盖任何同名文件。",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只运行无需 C++/数据文件的 synthetic polynomial、B000 与 nbar 自检。",
    )
    parser.add_argument("--geometry", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--pair-index", action="append", type=int, default=None, help="显式原始 120 维索引；可重复。")
    parser.add_argument("--kmax", type=float, default=0.18, help="要求两个理论 k 都不超过此值。")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--bin-average", choices=("point", "shell"), default="point")
    parser.add_argument("--nmu-b000", type=int, default=12)
    parser.add_argument("--nradial-binavg", type=int, default=4)

    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--pk-table", default=None, help="复用已有两列 P_L(k,z=1) 表。")
    parser.add_argument("--pk-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--pk-kmin", type=float, default=1.0e-4)
    parser.add_argument("--pk-kmax", type=float, default=80.0)
    parser.add_argument("--pk-n", type=int, default=2400)

    parser.add_argument("--b1", type=float, default=DEFAULT_B1)
    parser.add_argument("--nbar", type=float, default=DEFAULT_NBAR)
    parser.add_argument("--b2-node-scale", type=float, default=1.0)
    parser.add_argument("--bk2-node-scale", type=float, default=1.0)
    parser.add_argument("--max-vandermonde-condition", type=float, default=100.0)

    parser.add_argument("--epsrel", type=float, default=1.0e-3)
    parser.add_argument(
        "--p13-epsrel",
        type=float,
        default=1.0e-3,
        help="正式 polynomial extraction 推荐 <=1e-3，并必须通过不可放宽的 1e-4 direct check。",
    )
    parser.add_argument("--qmin", type=float, default=1.0e-4)
    parser.add_argument("--qmax", type=float, default=30.0)
    parser.add_argument("--singular-floor", type=float, default=1.0e-5)
    parser.add_argument("--smoothing-radius", type=float, default=DEFAULT_R_RECON)
    parser.add_argument("--bias-recon", type=float, default=DEFAULT_B_RECON)
    parser.add_argument("--recon-cellsize", type=float, default=DEFAULT_RECON_CELLSIZE)
    parser.add_argument(
        "--recon-cic-window-power",
        type=int,
        default=DEFAULT_RECON_CIC_WINDOW_POWER,
        help="jaxrecon CIC paint/read convention uses product-sinc power 4。",
    )

    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--max-triangles-per-call", type=int, default=72)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "并发执行独立 native chunks；默认 1，硬上限 8。workers>1 时每个 "
            "subprocess 的 OMP/BLAS 线程会强制为 1。"
        ),
    )
    parser.add_argument(
        "--polynomial-check-points",
        type=int,
        default=2,
        help="每个 stage 额外直接计算的随机重组检查点数；正式模板建议 >=2。",
    )
    parser.add_argument("--polynomial-check-seed", type=int, default=20260715)
    parser.add_argument(
        "--polynomial-check-rms-rtol",
        type=float,
        default=1.0e-4,
        help="random direct check 的 RMS/global-RMS 硬门；冻结规格要求 <=1e-4。",
    )
    parser.add_argument(
        "--polynomial-check-max-global-rtol",
        type=float,
        default=1.0e-4,
        help="random direct check 的 max-abs/global-max 硬门；冻结规格要求 <=1e-4。",
    )
    parser.add_argument("--polynomial-check-max-symmetric-rtol", type=float, default=3.0e-2)
    return parser.parse_args()


def infer_pair_indices(k_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由两个 shell edge 推回 upper-triangle 的行列编号。"""

    rounded = np.round(np.asarray(k_edges, dtype=np.float64), 12)
    unique_edges = sorted({tuple(edge) for edge in rounded.reshape(-1, 2)})
    edge_to_index = {edge: index for index, edge in enumerate(unique_edges)}
    pair_i = np.asarray([edge_to_index[tuple(edge)] for edge in rounded[:, 0]], dtype=np.int64)
    pair_j = np.asarray([edge_to_index[tuple(edge)] for edge in rounded[:, 1]], dtype=np.int64)
    return pair_i, pair_j


def load_measurement_geometry(path: Path) -> MeasurementGeometry:
    """读取并严格验证 fid500 aggregate 的 pre/post 共享几何。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到 fid500 matrix：{path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "fiducial_realizations",
            "fiducial_pre_B000",
            "fiducial_post_B000",
            "b000_k",
            "b000_k_edges",
            "boxsize",
            "pk_meshsize",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"fid500 matrix 缺少 keys：{missing}")
        realization_ids = np.asarray(data["fiducial_realizations"], dtype=np.int64)
        pre = np.asarray(data["fiducial_pre_B000"], dtype=np.float64)
        post = np.asarray(data["fiducial_post_B000"], dtype=np.float64)
        weighted_pairs = np.asarray(data["b000_k"], dtype=np.float64)
        shell_edges = np.asarray(data["b000_k_edges"], dtype=np.float64)
        boxsize = float(np.asarray(data["boxsize"]).item())
        meshsize = int(np.asarray(data["pk_meshsize"]).item())

    if pre.ndim != 2 or post.shape != pre.shape:
        raise ValueError(f"pre/post realization matrix 形状不一致：{pre.shape} vs {post.shape}")
    n_realization, n_shape = pre.shape
    if realization_ids.shape != (n_realization,) or np.unique(realization_ids).size != n_realization:
        raise ValueError("realization ID 数量不匹配或有重复。")
    if weighted_pairs.shape != (n_shape, 2) or shell_edges.shape != (n_shape, 2, 2):
        raise ValueError(
            f"B000 geometry 形状异常：k={weighted_pairs.shape}, edges={shell_edges.shape}, n={n_shape}"
        )
    if not all(np.all(np.isfinite(array)) for array in (pre, post, weighted_pairs, shell_edges)):
        raise ValueError("fid500 matrix 含 NaN/Inf。")
    if np.any(shell_edges[:, :, 1] <= shell_edges[:, :, 0]):
        raise ValueError("存在非正宽度的 k shell。")
    pair_i, pair_j = infer_pair_indices(shell_edges)
    return MeasurementGeometry(
        path=path,
        realization_ids=realization_ids,
        pre_samples=pre,
        post_samples=post,
        weighted_pairs=weighted_pairs,
        shell_edges=shell_edges,
        pair_i=pair_i,
        pair_j=pair_j,
        boxsize=boxsize,
        meshsize=meshsize,
    )


def theory_pairs(measurements: MeasurementGeometry, source: str) -> np.ndarray:
    """返回 point projection 使用的 weighted k 或 shell midpoint。"""

    if source == "weighted":
        pairs = measurements.weighted_pairs
    elif source == "center":
        pairs = np.mean(measurements.shell_edges, axis=2)
    else:
        raise ValueError(f"未知 k-source：{source!r}")
    return np.asarray(pairs, dtype=np.float64)


def selected_pair_indices(
    measurements: MeasurementGeometry,
    pairs: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """只选择一次 shape index，随后原样传给 pre 和 post。"""

    n_shape = pairs.shape[0]
    if args.pair_index:
        indices = np.asarray(list(dict.fromkeys(int(value) for value in args.pair_index)), dtype=np.int64)
    else:
        if args.geometry == "diagonal":
            geometry_mask = measurements.pair_i == measurements.pair_j
        elif args.geometry == "full":
            geometry_mask = np.ones(n_shape, dtype=bool)
        else:
            raise ValueError(f"未知 geometry：{args.geometry!r}")
        k_mask = np.max(pairs, axis=1) <= float(args.kmax) + 1.0e-12
        indices = np.nonzero(geometry_mask & k_mask)[0].astype(np.int64)
    if indices.size == 0:
        raise ValueError("shape selection 为空。")
    if np.any(indices < 0) or np.any(indices >= n_shape) or np.unique(indices).size != indices.size:
        raise ValueError(f"pair-index 重复或越界；合法范围 [0,{n_shape})。")
    # 显式 pair-index 仍然执行 kmax 防误用；需要更高 k 时应明确提高 --kmax。
    too_large = indices[np.max(pairs[indices], axis=1) > float(args.kmax) + 1.0e-12]
    if too_large.size:
        raise ValueError(f"显式 pair-index 中有 shape 超过 --kmax={args.kmax:g}：{too_large.tolist()}")
    return indices


def padua_degree3_nodes(b2_scale: float, bk2_scale: float) -> np.ndarray:
    """构造 total-degree 3 的 10 个 Padua 型插值节点。"""

    if not math.isfinite(b2_scale) or b2_scale <= 0.0:
        raise ValueError("--b2-node-scale 必须有限且为正。")
    if not math.isfinite(bk2_scale) or bk2_scale <= 0.0:
        raise ValueError("--bk2-node-scale 必须有限且为正。")
    unit_nodes: list[tuple[float, float]] = []
    degree = 3
    for k in range(degree + 1):
        x = float(np.cos(k * np.pi / degree))
        if abs(x) < 1.0e-15:
            x = 0.0
        if k % 2 == 0:
            y_values = [np.cos((2 * j - 1) * np.pi / (degree + 1)) for j in range(1, degree // 2 + 2)]
        else:
            y_values = [np.cos((2 * j - 2) * np.pi / (degree + 1)) for j in range(1, degree // 2 + 3)]
        for y_raw in y_values:
            y = float(y_raw)
            if abs(y) < 1.0e-15:
                y = 0.0
            unit_nodes.append((x, y))
    nodes = np.asarray(unit_nodes, dtype=np.float64)
    if nodes.shape != (MONOMIAL_POWERS.shape[0], 2):
        raise RuntimeError(f"Padua node 数量错误：{nodes.shape}")
    nodes[:, 0] *= float(b2_scale)
    nodes[:, 1] *= float(bk2_scale)
    return nodes


def polynomial_design(points: np.ndarray, powers: np.ndarray = MONOMIAL_POWERS) -> np.ndarray:
    """构造 ``point x monomial`` Vandermonde 矩阵。"""

    points = np.asarray(points, dtype=np.float64)
    powers = np.asarray(powers, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2 or powers.ndim != 2 or powers.shape[1] != 2:
        raise ValueError("points/powers 必须分别是 (n,2)/(m,2)。")
    return np.asarray(
        [
            [float(b2) ** int(power_b2) * float(bk2) ** int(power_bk2) for power_b2, power_bk2 in powers]
            for b2, bk2 in points
        ],
        dtype=np.float64,
    )


def solve_polynomial_coefficients(nodes: np.ndarray, node_values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """用方阵 solve 精确恢复 10 个 monomial coefficient，并报告节点残差。"""

    design = polynomial_design(nodes)
    values = np.asarray(node_values, dtype=np.float64)
    if design.shape[0] != design.shape[1] or values.shape[0] != design.shape[0]:
        raise ValueError(f"插值 shape 不一致：V={design.shape}, values={values.shape}")
    coefficients = np.linalg.solve(design, values)
    reconstructed = design @ coefficients
    residual = reconstructed - values
    value_scale = max(float(np.max(np.abs(values))), 1.0)
    diagnostics = {
        "vandermonde_condition_2": float(np.linalg.cond(design)),
        "node_reconstruction_max_abs": float(np.max(np.abs(residual))),
        "node_reconstruction_max_abs_over_global_scale": float(np.max(np.abs(residual)) / value_scale),
        "node_reconstruction_rms_over_global_rms": float(
            np.sqrt(np.mean(residual * residual)) / max(np.sqrt(np.mean(values * values)), 1.0e-300)
        ),
    }
    return coefficients, diagnostics


def evaluate_polynomial(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """在一个或多个 ``(b2,bK2)`` 点重组 shape vector。"""

    design = polynomial_design(np.atleast_2d(np.asarray(points, dtype=np.float64)))
    return design @ np.asarray(coefficients, dtype=np.float64)


def accumulate_b000_fields(
    rows: list[dict[str, Any]],
    triangle_meta: list[tuple[int, float]],
    n_selected: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按 DM producer 的 ``factor=weight/2`` 累加所有理论和 stochastic row。"""

    if len(rows) != len(triangle_meta):
        raise ValueError(f"native row/meta 数量不一致：{len(rows)} vs {len(triangle_meta)}")
    vectors = {field: np.zeros(n_selected, dtype=np.float64) for field in ROW_FIELDS}
    error_fields = ("B222", "B321I", "B321II", "B411")
    conservative_error = {field: np.zeros(n_selected, dtype=np.float64) for field in error_fields}
    for row, (local_index, weight) in zip(rows, triangle_meta):
        factor = 0.5 * float(weight)
        for field in ROW_FIELDS:
            vectors[field][int(local_index)] += factor * float(row[field])
        integration = row.get("integration_metadata", {})
        if isinstance(integration, dict):
            for field in ("B222", "B321I", "B411"):
                record = integration.get(field, {})
                if isinstance(record, dict):
                    conservative_error[field][int(local_index)] += abs(factor) * abs(float(record.get("abserr", 0.0)))
            # Gaussian B321II/B123I 由三个 I3=P13-type propagator integral 组合而成；
            # 当前 row 分别公开 P13_k1/k2/k3 stats，而没有单独的 Gaussian
            # B321II stats 字段。因此把三个 I3 abserr 的绝对和作为 conservative
            # driver proxy 留进 provenance。它是 I3 层误差驱动量，不冒充 B 的同量纲
            # propagated error；正式收敛仍由 random direct check 和 eps/qmax scan 验证。
            for p13_field in ("P13_k1", "P13_k2", "P13_k3"):
                record = integration.get(p13_field, {})
                if isinstance(record, dict):
                    conservative_error["B321II"][int(local_index)] += abs(factor) * abs(
                        float(record.get("abserr", 0.0))
                    )
    if not all(np.all(np.isfinite(value)) for value in vectors.values()):
        raise FloatingPointError("B000 accumulation 产生 NaN/Inf。")
    return vectors, conservative_error


def compact_compile_record(record: dict[str, Any]) -> dict[str, Any]:
    """保留编译 provenance，但不在两个 sidecar 中复制大段 stdout。"""

    return {
        "skipped": bool(record.get("skipped", False)),
        "returncode": int(record.get("returncode", 0)),
        "binary": str(record.get("binary", "")),
        "command": [str(item) for item in record.get("command", [])],
        "stderr_tail": str(record.get("stderr", ""))[-4000:],
    }


def evaluate_stage_at_node(
    *,
    stage: str,
    b2: float,
    bK2: float,
    binary: Path,
    pk_table: Path,
    triangles: list[tuple[float, float, float]],
    triangle_meta: list[tuple[int, float]],
    n_selected: int,
    args: argparse.Namespace,
) -> StageNodeEvaluation:
    """分 chunk 调用一个 native halo Gaussian mode，并立即压缩为 B000 vector。"""

    mode = STAGE_MODES[stage]
    chunk_size = max(1, int(args.max_triangles_per_call))
    chunk_work = [
        (start, triangles[start : start + chunk_size])
        for start in range(0, len(triangles), chunk_size)
    ]

    def run_one_chunk(work: tuple[int, list[tuple[float, float, float]]]) -> tuple[
        int,
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        """执行一个独立 subprocess；返回 start 供并发路径恢复确定性顺序。"""

        start, chunk = work
        info = run_marisa_b(
            binary,
            pk_table,
            chunk,
            float(args.epsrel),
            float(args.p13_epsrel),
            float(args.qmin),
            float(args.qmax),
            float(args.smoothing_radius),
            float(args.bias_recon),
            float(args.recon_cellsize),
            int(args.recon_cic_window_power),
            float(args.singular_floor),
            mode,
            "native_cpp",
            b1=float(args.b1),
            b2=float(b2),
            bK2=float(bK2),
        )
        payload = info.get("payload")
        parsed = list(payload.get("results", [])) if isinstance(payload, dict) else []
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        record = {
            "start": int(start),
            "n_triangles": int(len(chunk)),
            "returncode": int(info.get("returncode", -1)),
            "n_parsed_rows": int(len(parsed)),
            "wall_seconds": float(metadata.get("wall_seconds", math.nan)) if isinstance(metadata, dict) else math.nan,
            "stderr_tail": str(info.get("stderr", ""))[-2000:],
        }
        if record["returncode"] != 0 or len(parsed) != len(chunk):
            raise RuntimeError(
                f"MARISA-B {stage} node (b2={b2:g},bK2={bK2:g}) chunk failed：{record}"
            )
        if isinstance(payload, dict):
            if str(payload.get("mode")) != mode or str(payload.get("backend")) != "native_cpp":
                raise RuntimeError(f"native payload mode/backend 不一致：{payload.get('mode')}/{payload.get('backend')}")
        chunk_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        return int(start), parsed, record, chunk_metadata

    # 单 worker 保留原来的直接循环，方便调试且没有 executor 开销。并发路径中
    # future 完成顺序不确定，所以必须按 start 排序后再拼 rows，维持 triangle_meta 对齐。
    if int(args.workers) == 1:
        completed = [run_one_chunk(work) for work in chunk_work]
    else:
        completed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(run_one_chunk, work) for work in chunk_work]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
    completed.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    native_metadata: dict[str, Any] = {}
    for _start, parsed, record, chunk_metadata in completed:
        rows.extend(parsed)
        chunks.append(record)
        if not native_metadata:
            native_metadata = chunk_metadata
    vectors, conservative_error = accumulate_b000_fields(rows, triangle_meta, n_selected)
    return StageNodeEvaluation(
        stage=stage,
        b2=float(b2),
        bK2=float(bK2),
        vectors=vectors,
        conservative_error_proxy=conservative_error,
        chunks=chunks,
        native_metadata=native_metadata,
    )


def stochastic_invariance_and_normalization(
    evaluations: list[StageNodeEvaluation],
    nbar: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """审计 raw basis 与 bias node 无关，并施加 joint-fit 所需 nbar 归一化。"""

    if not math.isfinite(nbar) or nbar <= 0.0:
        raise ValueError("fixed nbar 必须有限且为正。")
    raw_alpha3 = np.stack([item.vectors["stochastic_alpha3_basis"] for item in evaluations], axis=0)
    raw_alpha4 = np.stack([item.vectors["stochastic_alpha4_basis"] for item in evaluations], axis=0)
    mean_alpha3 = np.mean(raw_alpha3, axis=0)
    mean_alpha4 = np.mean(raw_alpha4, axis=0)
    scale3 = max(float(np.max(np.abs(mean_alpha3))), 1.0)
    scale4 = max(float(np.max(np.abs(mean_alpha4))), 1.0)
    diagnostics = {
        "raw_alpha3_node_invariance_max_abs_over_global_scale": float(
            np.max(np.abs(raw_alpha3 - mean_alpha3[None, :])) / scale3
        ),
        "raw_alpha4_node_invariance_max_abs_over_global_scale": float(
            np.max(np.abs(raw_alpha4 - mean_alpha4[None, :])) / scale4
        ),
        "raw_alpha4_max_abs_from_unity": float(np.max(np.abs(mean_alpha4 - 1.0))),
        "alpha3_divisor_nbar": float(nbar),
        "alpha4_divisor_nbar_squared": float(nbar * nbar),
    }
    if diagnostics["raw_alpha3_node_invariance_max_abs_over_global_scale"] > 1.0e-12:
        raise AssertionError("raw alpha3 basis 意外依赖 b2/bK2 node。")
    if diagnostics["raw_alpha4_node_invariance_max_abs_over_global_scale"] > 1.0e-12:
        raise AssertionError("raw alpha4 basis 意外依赖 b2/bK2 node。")
    if diagnostics["raw_alpha4_max_abs_from_unity"] > 1.0e-12:
        raise AssertionError("B000-averaged raw alpha4 basis 不等于 1。")
    return mean_alpha3 / float(nbar), mean_alpha4 / (float(nbar) ** 2), diagnostics


def random_check_points(args: argparse.Namespace) -> np.ndarray:
    """在节点 box 内生成可复现、避开插值节点的随机 direct-check 点。"""

    n_check = int(args.polynomial_check_points)
    if n_check < 0:
        raise ValueError("--polynomial-check-points 不能为负。")
    rng = np.random.default_rng(int(args.polynomial_check_seed))
    points = rng.uniform(-0.8, 0.8, size=(n_check, 2))
    points[:, 0] *= float(args.b2_node_scale)
    points[:, 1] *= float(args.bk2_node_scale)
    return points.astype(np.float64, copy=False)


def compare_polynomial_to_direct(predicted: np.ndarray, direct: np.ndarray) -> dict[str, float]:
    """给出既不过度惩罚过零点、又对整体误差敏感的重组诊断。"""

    predicted = np.asarray(predicted, dtype=np.float64)
    direct = np.asarray(direct, dtype=np.float64)
    if predicted.shape != direct.shape:
        raise ValueError(f"prediction/direct shape 不一致：{predicted.shape} vs {direct.shape}")
    residual = predicted - direct
    global_rms = max(float(np.sqrt(np.mean(direct * direct))), 1.0e-300)
    global_max = max(float(np.max(np.abs(direct))), 1.0e-300)
    symmetric_floor = global_max * 1.0e-10
    symmetric_denominator = np.maximum.reduce(
        (np.abs(predicted), np.abs(direct), np.full(direct.shape, symmetric_floor, dtype=np.float64))
    )
    return {
        "rms_error_over_direct_global_rms": float(np.sqrt(np.mean(residual * residual)) / global_rms),
        "max_abs_error_over_direct_global_max": float(np.max(np.abs(residual)) / global_max),
        "max_pointwise_symmetric_relative_error": float(np.max(np.abs(residual) / symmetric_denominator)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def file_sha256(path: Path) -> str:
    """流式计算输入/源码 hash，避免 provenance 依赖文件时间戳。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def refuse_output_collisions(paths: Iterable[Path]) -> None:
    """任何科学产物同名即停止；调用者必须换 prefix，防止静默覆盖。"""

    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError("拒绝覆盖已有产物，请更换 --prefix：\n" + "\n".join(collisions))


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """先写临时 NPZ，再原子 rename，避免中断留下貌似完整的模板。"""

    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在：{temporary}")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON sidecar。"""

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在：{temporary}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def synthetic_self_test() -> dict[str, Any]:
    """无需外部文件，覆盖节点求解、随机重组、shell 归一化和 nbar 归一化。"""

    rng = np.random.default_rng(20260715)
    nodes = padua_degree3_nodes(1.0, 1.0)
    design = polynomial_design(nodes)
    n_shape = 7
    truth = rng.normal(size=(MONOMIAL_POWERS.shape[0], n_shape))
    values = design @ truth
    recovered, interpolation = solve_polynomial_coefficients(nodes, values)
    coefficient_error = float(np.max(np.abs(recovered - truth)))
    check_points = rng.uniform(-0.8, 0.8, size=(5, 2))
    direct = polynomial_design(check_points) @ truth
    predicted = evaluate_polynomial(recovered, check_points)
    random_error = float(np.max(np.abs(predicted - direct)))

    # 用真实 shell_nodes 验证 k^2 dk weight 已归一；这是 build_b000_work 的径向核心。
    radial_k, radial_weight = shell_nodes(np.asarray([0.04, 0.06]), 4)
    radial_constant = float(np.sum(radial_weight))
    radial_k2_mean = float(np.sum(radial_weight * radial_k * radial_k))
    radial_k2_exact = float((3.0 / 5.0) * (0.06**5 - 0.04**5) / (0.06**3 - 0.04**3))

    # 两个 mu 点的权重和为 2；accumulate 中再乘 1/2，因此 constant B000=constant。
    rows = [
        {
            **{field: 3.0 for field in DETERMINISTIC_COMPONENTS},
            "stochastic_alpha3_basis": 5.0,
            "stochastic_alpha4_basis": 1.0,
            "integration_metadata": {},
        },
        {
            **{field: 3.0 for field in DETERMINISTIC_COMPONENTS},
            "stochastic_alpha3_basis": 5.0,
            "stochastic_alpha4_basis": 1.0,
            "integration_metadata": {},
        },
    ]
    vectors, _ = accumulate_b000_fields(rows, [(0, 1.0), (0, 1.0)], 1)
    mock_evaluations = [
        StageNodeEvaluation("pre", float(i), 0.0, vectors, {}, [], {}) for i in range(MONOMIAL_POWERS.shape[0])
    ]
    alpha3, alpha4, stochastic_diag = stochastic_invariance_and_normalization(mock_evaluations, DEFAULT_NBAR)
    expected_alpha3 = 5.0 / DEFAULT_NBAR
    expected_alpha4 = 1.0 / (DEFAULT_NBAR**2)

    checks = {
        "loop_momentum_routing": GAUSSIAN_LOOP_ROUTING,
        "routing_tag_is_exact": GAUSSIAN_LOOP_ROUTING
        == "canonical_sorted_external_legs_native_gaussian_diagram_logq_with_spherical_qmin_qmax",
        "vandermonde_condition_2": float(np.linalg.cond(design)),
        "coefficient_max_abs_error": coefficient_error,
        "random_recomposition_max_abs_error": random_error,
        "radial_weight_sum": radial_constant,
        "radial_k2_quadrature_minus_exact": radial_k2_mean - radial_k2_exact,
        "constant_B000": float(vectors["Btotal"][0]),
        "alpha3_normalized": float(alpha3[0]),
        "alpha3_expected": float(expected_alpha3),
        "alpha4_normalized": float(alpha4[0]),
        "alpha4_expected": float(expected_alpha4),
        "stochastic_diagnostics": stochastic_diag,
        "interpolation": interpolation,
    }
    passed = bool(
        checks["routing_tag_is_exact"]
        and checks["vandermonde_condition_2"] < 20.0
        and coefficient_error < 1.0e-12
        and random_error < 1.0e-12
        and abs(radial_constant - 1.0) < 1.0e-14
        and abs(radial_k2_mean - radial_k2_exact) < 1.0e-14
        and abs(float(vectors["Btotal"][0]) - 3.0) < 1.0e-14
        and abs(float(alpha3[0]) - expected_alpha3) / expected_alpha3 < 1.0e-14
        and abs(float(alpha4[0]) - expected_alpha4) / expected_alpha4 < 1.0e-14
    )
    return {
        "status": "pass" if passed else "fail",
        "test": "synthetic halo bias-v1 Gaussian template producer",
        "checks": checks,
    }


def build_output_payload(
    *,
    stage: str,
    args: argparse.Namespace,
    measurements: MeasurementGeometry,
    indices: np.ndarray,
    pairs: np.ndarray,
    nodes: np.ndarray,
    evaluations: list[StageNodeEvaluation],
    coefficients_by_component: dict[str, np.ndarray],
    interpolation_by_component: dict[str, dict[str, float]],
    stochastic_alpha3: np.ndarray,
    stochastic_alpha4: np.ndarray,
    stochastic_diagnostics: dict[str, float],
    validation_points: np.ndarray,
    validation_direct: np.ndarray,
    validation_predicted: np.ndarray,
    validation_records: list[dict[str, Any]],
    work_summary: dict[str, Any],
    pk_table: Path,
    pk_info: dict[str, Any],
    unscaled_pk_info: dict[str, Any] | None,
    binary: Path,
    compile_record: dict[str, Any],
    output_npz: Path,
    output_json: Path,
) -> dict[str, Any]:
    """构造能被 joint-fit 严格审计的一环 JSON sidecar。"""

    validation_metrics = compare_polynomial_to_direct(validation_predicted, validation_direct) if validation_points.size else {
        "rms_error_over_direct_global_rms": math.nan,
        "max_abs_error_over_direct_global_max": math.nan,
        "max_pointwise_symmetric_relative_error": math.nan,
        "max_abs_error": math.nan,
    }
    total_wall = float(
        np.nansum([chunk["wall_seconds"] for evaluation in evaluations for chunk in evaluation.chunks])
        + np.nansum([record["wall_seconds"] for record in validation_records])
    )
    source_files = {
        "producer": Path(__file__).resolve(),
        "runner": PROJECT_ROOT / "scripts/production/run_marisa_b_triangle.py",
        "native_cpp": PROJECT_ROOT / "src/marisa_b/marisa_b_native.cpp",
        "triangle_cpp": PROJECT_ROOT / "src/marisa_b/marisa_b_triangle.cpp",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "purpose": "Gate G Gaussian pre/post halo MARISA-B bias-v1 one-loop polynomial template",
        "stage": stage,
        "mode": STAGE_MODES[stage],
        "engine": "MARISA-B native_cpp",
        "model_name": "MARISA-B halo bias-v1 truncated fixed-cutoff one-loop Gaussian",
        "bias_model_tag": "MARISA-B halo bias-v1",
        "theory_order": "one-loop",
        "perturbative_order": "one-loop",
        "loop_order": "one-loop",
        "renormalization": "fixed-cutoff unrenormalized truncated functional",
        "scope_boundary": (
            "Gaussian K_n=b1 F_n+b2 D_n+bK2 T_n for n<=4, mechanically continued through "
            "Btree+B222+B321I+B321II+B411. This is not a Barreira halo one-loop derivation, "
            "not a complete renormalized bias expansion, and not EFT. Native halo runtime "
            "canonical-sorts the three external-leg lengths and defines q routing from the two "
            "shortest legs; legacy DM modes are unchanged."
        ),
        "diagram_basis": ["Btree", "B222", "B321I", "B321II", "B411"],
        "outputs": {"npz": str(output_npz), "json": str(output_json)},
        "inputs": {
            "matrix": str(measurements.path),
            "matrix_sha256": file_sha256(measurements.path),
            "n_realizations": int(measurements.realization_ids.size),
            "realization_id_min": int(np.min(measurements.realization_ids)),
            "realization_id_max": int(np.max(measurements.realization_ids)),
            "pk_table": str(pk_table),
            "pk_table_sha256": file_sha256(pk_table),
            "pk_table_metadata": pk_info,
            "unscaled_pk_table_metadata": unscaled_pk_info,
            "binary": str(binary),
            "binary_sha256": file_sha256(binary),
            "source_sha256": {name: file_sha256(path) for name, path in source_files.items()},
        },
        "fixed_parameters": {
            "b1": float(args.b1),
            "nbar": float(args.nbar),
            "nbar_source": "fid500 halo catalog aggregate metadata/count audit; fixed, not fitted",
        },
        "selection": {
            "geometry": str(args.geometry),
            "k_source": str(args.k_source),
            "kmax": float(args.kmax),
            "explicit_pair_indices": [int(value) for value in args.pair_index] if args.pair_index else None,
            "data_indices": indices.astype(int).tolist(),
            "n_selected": int(indices.size),
            "same_indices_for_pre_post": True,
            "native_external_leg_canonicalization": (
                "internal runtime sort by the three external-leg lengths; the two shortest legs "
                "define native q routing"
            ),
            "data_index_alignment_after_canonicalization": (
                "unchanged: canonical sorting is internal to each symmetric triangle evaluation "
                "and never permutes the original aggregate data_indices or B000 weights"
            ),
            "selected_k_pairs": pairs[indices].tolist(),
            "selected_k_edges": measurements.shell_edges[indices].tolist(),
        },
        "b000_projection": {
            **work_summary,
            "definition": "1/2 integral_{-1}^{1} dmu12; shell mode additionally averages k1,k2 with k^2 dk",
        },
        "integration": {
            "epsrel": float(args.epsrel),
            "p13_epsrel": float(args.p13_epsrel),
            "qmin": float(args.qmin),
            "qmax": float(args.qmax),
            "singular_floor": float(args.singular_floor),
            "hard_cutoff_note": "loop momentum uses a fixed spherical qmin<=q<=qmax regulator",
            "loop_momentum_routing": GAUSSIAN_LOOP_ROUTING,
            "formal_precision_recommendation": (
                "Use epsrel<=1e-3 and p13_epsrel<=1e-3 as the starting point, then tighten "
                "until the <=1e-4 random direct-recomposition gates and shell/nmu/qmax scans pass."
            ),
            "error_proxy_note": (
                "B222/B321I/B411 entries are B-level sums of native abserr. B321II is "
                "reported via the absolute sum of its three native P13/I3 driver abserr values; "
                "it is an I3-level diagnostic, not a same-unit propagated B321II error."
            ),
            "max_triangles_per_call": int(args.max_triangles_per_call),
            "workers": int(args.workers),
            "threads_per_subprocess": 1 if int(args.workers) > 1 else "environment_capped_at_8",
            "total_native_wall_seconds": total_wall,
        },
        "reconstruction": {
            "stage": stage,
            "smoothing_radius": float(args.smoothing_radius),
            "bias_recon": float(args.bias_recon),
            "recon_cellsize": float(args.recon_cellsize),
            "recon_cic_window_power": int(args.recon_cic_window_power),
            "cic_convention": (
                "trusted jaxrecon PlaneParallel f=0 path: CIC paint compensate=False plus CIC read; "
                "product-sinc convention corresponds to recon_cic_window_power=4"
            ),
            "post_only_effect": stage == "post",
        },
        "measurement": {
            "boxsize": float(measurements.boxsize),
            "meshsize": int(measurements.meshsize),
            "value_mode": "jaxpower final; Poisson estimator shot-noise subtraction already applied",
        },
        "stochastic": {
            "interpretation": "stage-specific residual non-Poisson alpha3/alpha4; no full Poisson term is re-added",
            "raw_cpp_alpha3": "B000 average of b1^2*(P1+P2+P3), before number-density normalization",
            "template_alpha3": "raw_cpp_alpha3 / fixed nbar",
            "raw_cpp_alpha4": "1 before number-density normalization",
            "template_alpha4": "raw_cpp_alpha4 / fixed nbar^2",
            "diagnostics": stochastic_diagnostics,
        },
        "polynomial": {
            "coefficient_definition": (
                "Bdet(b2,bK2)[shape] = sum_m polynomial_coefficients[m,shape] "
                "* b2**monomial_powers[m,0] * bK2**monomial_powers[m,1]"
            ),
            "maximum_total_degree": 3,
            "monomial_powers": MONOMIAL_POWERS.astype(int).tolist(),
            "node_family": "degree-3 Padua-type unisolvent nodes",
            "node_scales": {"b2": float(args.b2_node_scale), "bK2": float(args.bk2_node_scale)},
            "nodes": nodes.tolist(),
            "vandermonde_condition_2": float(np.linalg.cond(polynomial_design(nodes))),
            "interpolation_by_component": interpolation_by_component,
            "random_direct_check": {
                "seed": int(args.polynomial_check_seed),
                "points": validation_points.tolist(),
                "n_points": int(validation_points.shape[0]),
                "metrics_Btotal": validation_metrics,
                "thresholds": {
                    "rms_error_over_direct_global_rms": float(args.polynomial_check_rms_rtol),
                    "max_abs_error_over_direct_global_max": float(args.polynomial_check_max_global_rtol),
                    "max_pointwise_symmetric_relative_error_diagnostic_only": float(
                        args.polynomial_check_max_symmetric_rtol
                    ),
                },
                "status": "pass" if validation_points.size else "skipped",
            },
        },
        "linear_power_convention": {
            "z": float(args.z),
            "cosmology": dict(QUIJOTE_COSMOLOGY),
            "growth_mode": "unity_input_pk_at_target_z",
            "pk_amplitude_scale": float(args.pk_amplitude_scale),
            "power_normalization_audit": None if args.pk_table else quijote_power_normalization_audit(float(args.z)),
        },
        "native_metadata": evaluations[0].native_metadata if evaluations else {},
        "node_run_summary": [
            {
                "node_index": int(index),
                "b2": float(evaluation.b2),
                "bK2": float(evaluation.bK2),
                "n_chunks": int(len(evaluation.chunks)),
                "wall_seconds": float(np.nansum([chunk["wall_seconds"] for chunk in evaluation.chunks])),
                "max_conservative_error_proxy": {
                    key: float(np.max(value)) for key, value in evaluation.conservative_error_proxy.items()
                },
            }
            for index, evaluation in enumerate(evaluations)
        ],
        "compile": compact_compile_record(compile_record),
        "contract": {
            "consumer": "fit_quijote_halo_marisa_b_bias_v1_joint.py",
            "loader_alignment_assumption": (
                "consumer aligns only through original data_indices; native external-leg "
                "canonicalization is internal and requires no loader-side reordering"
            ),
            "npz_required": [
                "polynomial_coefficients",
                "monomial_powers",
                "stochastic_basis_alpha3",
                "stochastic_basis_alpha4",
                "data_indices",
            ],
            "coefficient_array_shape": list(coefficients_by_component["Btotal"].shape),
        },
    }
    return payload


def main() -> None:
    """执行 synthetic self-test 或生成一对 pre/post Gaussian template。"""

    enforce_thread_limit()
    args = parse_args()
    if args.self_test:
        payload = synthetic_self_test()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if payload["status"] != "pass":
            raise SystemExit(1)
        return

    scalar_checks = {
        "z": args.z,
        "b1": args.b1,
        "nbar": args.nbar,
        "epsrel": args.epsrel,
        "p13_epsrel": args.p13_epsrel,
        "qmin": args.qmin,
        "qmax": args.qmax,
        "smoothing_radius": args.smoothing_radius,
        "bias_recon": args.bias_recon,
        "recon_cellsize": args.recon_cellsize,
        "singular_floor": args.singular_floor,
    }
    for name, value in scalar_checks.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"--{name.replace('_', '-')} 必须有限。")
    if float(args.nbar) <= 0.0 or float(args.b1) <= 0.0:
        raise ValueError("--nbar/--b1 必须为正。")
    if float(args.qmin) <= 0.0 or float(args.qmax) <= float(args.qmin):
        raise ValueError("必须满足 0 < qmin < qmax。")
    if int(args.nmu_b000) <= 0 or int(args.nradial_binavg) <= 0:
        raise ValueError("B000 quadrature node 数必须为正。")
    if int(args.recon_cic_window_power) < 0:
        raise ValueError("--recon-cic-window-power 不能为负。")
    if not (1 <= int(args.workers) <= 8):
        raise ValueError("--workers 必须在 [1,8]；总并发硬限制为 8 核。")
    if int(args.workers) > 1:
        # run_marisa_b 为每个 subprocess 从当前环境复制 env；在启动线程池前全局
        # 压到 1，便可保证 workers 个进程至多占 workers<=8 个核。Python 线程只等待 I/O。
        for key in THREAD_ENV_KEYS:
            os.environ[key] = "1"
        os.environ["OMP_DYNAMIC"] = "FALSE"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    # 这两个阈值是冻结规格，不允许 CLI 把正式门槛放宽；CLI 只用于要求更严格。
    for name, value in (
        ("polynomial-check-rms-rtol", args.polynomial_check_rms_rtol),
        ("polynomial-check-max-global-rtol", args.polynomial_check_max_global_rtol),
    ):
        if not math.isfinite(float(value)) or not (0.0 < float(value) <= 1.0e-4):
            raise ValueError(f"--{name} 必须在 (0,1e-4]，不能放宽冻结的 random direct-check 门槛。")
    if not math.isfinite(float(args.polynomial_check_max_symmetric_rtol)) or float(
        args.polynomial_check_max_symmetric_rtol
    ) <= 0.0:
        raise ValueError("--polynomial-check-max-symmetric-rtol 必须有限且为正（仅作过零点诊断）。")

    # runner 的 subprocess cwd 固定为 PROJECT_ROOT；所有文件都先转绝对路径，
    # 否则从其它 cwd 传入的相对 --pk-table 会在子进程中被二次拼接。
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, tuple[Path, Path]] = {
        stage: (
            output_dir / f"{args.prefix}_{stage}.npz",
            output_dir / f"{args.prefix}_{stage}.json",
        )
        for stage in STAGE_MODES
    }
    collision_candidates = [path for pair in output_paths.values() for path in pair]
    if not args.pk_table:
        collision_candidates.append(output_dir / f"{args.prefix}_plin_z{args.z:g}.dat")
        if abs(float(args.pk_amplitude_scale) - 1.0) > 1.0e-15:
            collision_candidates.append(
                output_dir / f"{args.prefix}_plin_z{args.z:g}_Ascale{float(args.pk_amplitude_scale):.8g}.dat"
            )
    refuse_output_collisions(collision_candidates)

    measurements = load_measurement_geometry(Path(args.matrix).expanduser().resolve())
    pairs = theory_pairs(measurements, str(args.k_source))
    indices = selected_pair_indices(measurements, pairs, args)
    # build_b000_work 是经过 DM producer 使用的同一实现；只做 key alias，不复制公式。
    build_data = {"hdf5_k_edges": measurements.shell_edges}
    triangles, triangle_meta, work_summary = build_b000_work(
        data=build_data,
        indices=indices,
        theory_pairs=pairs,
        args=args,
    )

    nodes = padua_degree3_nodes(float(args.b2_node_scale), float(args.bk2_node_scale))
    design_condition = float(np.linalg.cond(polynomial_design(nodes)))
    if design_condition > float(args.max_vandermonde_condition):
        raise ValueError(
            f"physical Vandermonde condition={design_condition:.6g} 超过门槛 "
            f"{float(args.max_vandermonde_condition):.6g}；请调整 node scale。"
        )

    binary = (Path(args.build_dir).expanduser().resolve() / "marisa_b_triangle")
    compile_record: dict[str, Any] = {"skipped": True, "returncode": 0, "binary": str(binary)}
    if not args.skip_compile:
        compile_record = compile_marisa_b(binary)
        if int(compile_record.get("returncode", -1)) != 0:
            raise RuntimeError(f"MARISA-B compile failed：{str(compile_record.get('stderr', ''))[-4000:]}")
    if not binary.exists():
        raise FileNotFoundError(f"找不到 MARISA-B binary：{binary}")

    # prepare_pk_table 保留 CLI 原文以便 provenance；native runner 必须收到绝对路径。
    pk_table, pk_info, unscaled_pk_info = prepare_pk_table(args, output_dir)
    pk_table = pk_table.expanduser().resolve()
    evaluations_by_stage: dict[str, list[StageNodeEvaluation]] = {stage: [] for stage in STAGE_MODES}
    # node 外层、stage 内层保证 pre/post 紧邻计算；任一失败都不会留下貌似完整的模板。
    for node_index, (b2, bK2) in enumerate(nodes):
        print(
            f"[node {node_index + 1}/{len(nodes)}] b2={b2:.8g}, bK2={bK2:.8g}",
            file=sys.stderr,
            flush=True,
        )
        for stage in STAGE_MODES:
            evaluations_by_stage[stage].append(
                evaluate_stage_at_node(
                    stage=stage,
                    b2=float(b2),
                    bK2=float(bK2),
                    binary=binary,
                    pk_table=pk_table,
                    triangles=triangles,
                    triangle_meta=triangle_meta,
                    n_selected=int(indices.size),
                    args=args,
                )
            )

    coefficients_by_stage: dict[str, dict[str, np.ndarray]] = {}
    interpolation_by_stage: dict[str, dict[str, dict[str, float]]] = {}
    stochastic_by_stage: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    for stage, evaluations in evaluations_by_stage.items():
        component_coefficients: dict[str, np.ndarray] = {}
        component_diagnostics: dict[str, dict[str, float]] = {}
        for component in DETERMINISTIC_COMPONENTS:
            node_values = np.stack([evaluation.vectors[component] for evaluation in evaluations], axis=0)
            coefficients, interpolation = solve_polynomial_coefficients(nodes, node_values)
            component_coefficients[component] = coefficients
            component_diagnostics[component] = interpolation
        coefficients_by_stage[stage] = component_coefficients
        interpolation_by_stage[stage] = component_diagnostics
        stochastic_by_stage[stage] = stochastic_invariance_and_normalization(evaluations, float(args.nbar))

    validation_points = random_check_points(args)
    validation_direct_by_stage: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGE_MODES}
    validation_records_by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGE_MODES}
    for check_index, (b2, bK2) in enumerate(validation_points):
        print(
            f"[direct check {check_index + 1}/{len(validation_points)}] b2={b2:.8g}, bK2={bK2:.8g}",
            file=sys.stderr,
            flush=True,
        )
        for stage in STAGE_MODES:
            evaluation = evaluate_stage_at_node(
                stage=stage,
                b2=float(b2),
                bK2=float(bK2),
                binary=binary,
                pk_table=pk_table,
                triangles=triangles,
                triangle_meta=triangle_meta,
                n_selected=int(indices.size),
                args=args,
            )
            validation_direct_by_stage[stage].append(evaluation.vectors["Btotal"])
            validation_records_by_stage[stage].append(
                {
                    "check_index": int(check_index),
                    "b2": float(b2),
                    "bK2": float(bK2),
                    "wall_seconds": float(np.nansum([chunk["wall_seconds"] for chunk in evaluation.chunks])),
                }
            )

    payloads: dict[str, dict[str, Any]] = {}
    arrays_by_stage: dict[str, dict[str, np.ndarray]] = {}
    for stage in STAGE_MODES:
        coefficients = coefficients_by_stage[stage]
        node_values_by_component = {
            component: np.stack(
                [evaluation.vectors[component] for evaluation in evaluations_by_stage[stage]], axis=0
            )
            for component in DETERMINISTIC_COMPONENTS
        }
        if validation_points.size:
            validation_direct = np.stack(validation_direct_by_stage[stage], axis=0)
            validation_predicted = evaluate_polynomial(coefficients["Btotal"], validation_points)
            validation_metrics = compare_polynomial_to_direct(validation_predicted, validation_direct)
            if (
                validation_metrics["rms_error_over_direct_global_rms"]
                > float(args.polynomial_check_rms_rtol)
                or validation_metrics["max_abs_error_over_direct_global_max"]
                > float(args.polynomial_check_max_global_rtol)
            ):
                raise AssertionError(
                    f"{stage} random polynomial direct check 未通过：{validation_metrics}"
                )
        else:
            validation_direct = np.empty((0, indices.size), dtype=np.float64)
            validation_predicted = np.empty_like(validation_direct)

        stochastic_alpha3, stochastic_alpha4, stochastic_diagnostics = stochastic_by_stage[stage]
        arrays: dict[str, np.ndarray] = {
            "polynomial_coefficients": coefficients["Btotal"],
            "monomial_powers": MONOMIAL_POWERS,
            "stochastic_basis_alpha3": stochastic_alpha3,
            "stochastic_basis_alpha4": stochastic_alpha4,
            "data_indices": indices.astype(np.int64, copy=False),
            "k_pairs": pairs[indices],
            "k_edges": measurements.shell_edges[indices],
            "bias_nodes": nodes,
            "node_Btotal": node_values_by_component["Btotal"],
            "validation_bias_points": validation_points,
            "validation_direct_Btotal": validation_direct,
            "validation_polynomial_Btotal": validation_predicted,
            "b1": np.asarray(float(args.b1), dtype=np.float64),
            "nbar": np.asarray(float(args.nbar), dtype=np.float64),
            "qmin": np.asarray(float(args.qmin), dtype=np.float64),
            "qmax": np.asarray(float(args.qmax), dtype=np.float64),
            "nmu_b000": np.asarray(int(args.nmu_b000), dtype=np.int64),
            "nradial_binavg": np.asarray(int(args.nradial_binavg), dtype=np.int64),
            "bin_average": np.asarray(str(args.bin_average)),
            "stage": np.asarray(stage),
            "loop_momentum_routing": np.asarray(GAUSSIAN_LOOP_ROUTING),
        }
        for component in DETERMINISTIC_COMPONENTS:
            arrays[f"polynomial_coefficients_{component}"] = coefficients[component]
            arrays[f"node_{component}"] = node_values_by_component[component]
        arrays_by_stage[stage] = arrays

        output_npz, output_json = output_paths[stage]
        payloads[stage] = build_output_payload(
            stage=stage,
            args=args,
            measurements=measurements,
            indices=indices,
            pairs=pairs,
            nodes=nodes,
            evaluations=evaluations_by_stage[stage],
            coefficients_by_component=coefficients,
            interpolation_by_component=interpolation_by_stage[stage],
            stochastic_alpha3=stochastic_alpha3,
            stochastic_alpha4=stochastic_alpha4,
            stochastic_diagnostics=stochastic_diagnostics,
            validation_points=validation_points,
            validation_direct=validation_direct,
            validation_predicted=validation_predicted,
            validation_records=validation_records_by_stage[stage],
            work_summary=work_summary,
            pk_table=pk_table,
            pk_info=pk_info,
            unscaled_pk_info=unscaled_pk_info,
            binary=binary,
            compile_record=compile_record,
            output_npz=output_npz,
            output_json=output_json,
        )

    # 所有数值门都通过后才落盘；先 NPZ 后 sidecar，且每个文件原子写入。
    for stage in STAGE_MODES:
        output_npz, _ = output_paths[stage]
        atomic_savez(output_npz, **arrays_by_stage[stage])
    for stage in STAGE_MODES:
        _, output_json = output_paths[stage]
        atomic_write_json(output_json, payloads[stage])

    summary = {
        "status": "ok",
        "producer": str(Path(__file__).resolve()),
        "n_selected_shapes": int(indices.size),
        "n_b000_triangles_per_bias_node": int(len(triangles)),
        "n_bias_nodes": int(nodes.shape[0]),
        "n_random_direct_checks": int(validation_points.shape[0]),
        "vandermonde_condition_2": design_condition,
        "outputs": {
            stage: {"npz": str(paths[0]), "json": str(paths[1])} for stage, paths in output_paths.items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
