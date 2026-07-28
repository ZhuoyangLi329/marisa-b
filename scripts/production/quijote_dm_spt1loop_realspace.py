#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quijote DM real-space pre-recon 1-loop SPT bispectrum 轻量实现。

代码大纲
========
1. 使用 EdS 标准微扰论递归关系生成对称化的 `F_n/G_n` kernel，
   当前只需要 `F2/F3/F4`。
2. 使用 Quijote fiducial cosmology 和 `cosmoprimo` 生成线性功率谱
   `P_L(k,z=0.5)`；这里的 `P_L` 已经包含红移增长因子，是后续
   归一化检查的核心。
3. 对给定三角形 `(k1,k2,mu12)` 计算 real-space dark matter：
   `B112/tree`、`P13`、`B222`、`B321I`、`B321II`、`B411`。
4. 对 Sugiyama `B000(k1,k2)=1/2 int_{-1}^{1} dmu12 B(k1,k2,k3)`
   做角平均；这个定义和现有 real-space theory 脚本保持一致。
5. 提供两个 CLI mode：
   - `smoke-triangles`：少量三角形 sanity check；
   - `compare-diag`：计算少量 Quijote DM 对角 bin 的 B000，并和
     500 个 fiducial realization 的 pre-recon 均值画 PDF 对比。

注意
====
这是第一版“慢但透明”的实现，目标是先把公式、红移、单位和 B000
投影口径跑通。默认积分节点故意较少，只适合 smoke/diagnostic；
正式曲线需要提高 `--nq/--nmu-loop/--nphi/--nmu-b000` 并做收敛测试。
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MARISA_B_DATA_ROOT", PROJECT_ROOT)).resolve()
DEFAULT_DM_MATRIX = (
    DATA_ROOT
    / "analysis"
    / "jaxrecon_jaxpower_quijote_dm_png_z0p5_2d_k0p3_n15_paperbin_three_cosmo_full"
    / "quijote_dm_png_z0p5_2d_paperbin_b000_three_cosmo_matrix.npz"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "figures" / "diagnostics" / "quijote_dm_spt1loop_realspace"

QUIJOTE_COSMOLOGY = {
    "h": 0.6711,
    "Omega_m": 0.3175,
    "Omega_b": 0.049,
    "sigma8": 0.834,
    "n_s": 0.9624,
}

K_FLOOR = 1.0e-6
TWOPI = 2.0 * math.pi
REACT_XMIN = -9.999561e-01


@dataclass(frozen=True)
class LoopConfig:
    """1-loop 积分网格配置。

    `qmin/qmax` 使用 h/Mpc 单位。ReACT/Copter 常用 `QMINp=1e-4`,
    `QMAXp=30`；这里保留这个口径，但 smoke run 可通过命令行调小
    节点数来控制成本。
    """

    qmin: float = 1.0e-4
    qmax: float = 30.0
    nq: int = 10
    nmu: int = 6
    nphi: int = 6
    q_splits: tuple[float, ...] = ()
    nq_per_segment: int = 0
    mu_splits: tuple[float, ...] = ()
    nmu_per_segment: int = 0
    phi_splits: tuple[float, ...] = ()
    nphi_per_segment: int = 0


@dataclass(frozen=True)
class AdaptiveConfig:
    """ReACT-style 3D adaptive cubature 配置。

    这个配置用于少量三角形或少量 `B000` pair 的高精度验证。积分变量采用
    ReACT `BSPT.cpp::Bloopterms` 的 `(r=q/k1, u=cos(q,k1), phi)`，而不是
    fixed Gauss 后端使用的绝对 `q,mu,phi`。`qmin/qmax` 仍用 h/Mpc 单位。
    """

    qmin: float = 1.0e-4
    qmax: float = 30.0
    epsrel: float = 1.0e-3
    epsabs: float = 1.0e-10
    min_evals: int = 1000
    max_evals: int = 200000
    umin: float = -1.0
    umax: float = 0.99999999


@dataclass
class AdaptiveResult:
    """保存一次 scalar adaptive cubature 的数值结果。"""

    value: float
    abserr: float
    relerr: float
    neval: int
    n_regions: int
    converged: bool
    stopped_reason: str
    max_region_error: float


@dataclass(order=True)
class AdaptiveRegion:
    """按误差从大到小弹出的 adaptive 子区域。"""

    priority: float
    index: int
    center: tuple[float, float, float]
    width: tuple[float, float, float]
    value: float
    error: float
    div_axis: int


def parse_float_token(value: str) -> float:
    """解析浮点参数；额外支持 `pi`、`2*pi`、`pi/2` 这类角度写法。"""

    text = str(value).strip().lower().replace(" ", "")
    if not text:
        raise argparse.ArgumentTypeError("empty float token")
    sign = -1.0 if text.startswith("-") else 1.0
    unsigned = text[1:] if text[0] in "+-" else text
    if unsigned == "pi":
        return float(sign * math.pi)
    if unsigned in {"2pi", "2*pi"}:
        return float(sign * 2.0 * math.pi)
    if unsigned.endswith("*pi"):
        return float(sign * float(unsigned[:-3]) * math.pi)
    if unsigned.endswith("pi") and unsigned[:-2]:
        return float(sign * float(unsigned[:-2]) * math.pi)
    if unsigned.startswith("pi/"):
        return float(sign * math.pi / float(unsigned[3:]))
    if "*pi/" in unsigned:
        numerator, denominator = unsigned.split("*pi/", 1)
        return float(sign * float(numerator) * math.pi / float(denominator))
    return float(value)


def parse_split_list(value: str) -> tuple[float, ...]:
    """解析冒号分隔的分段点列表。"""

    return tuple(parse_float_token(item) for item in str(value).split(":") if item.strip())


def parse_loop_config_string(value: str) -> LoopConfig:
    """解析 loop 配置字符串。

    支持四种格式：
    - `qmax,nq,nmu,nphi`
    - `qmax,nq,nmu,nphi,split1:split2:...,nq_per_segment`
    - `qmax,nq,nmu,nphi,qsplit1:qsplit2:...,nq_per_segment,phisplit1:...,nphi_per_segment`
    - `qmax,nq,nmu,nphi,qsplit1:...,nq_per_segment,musplit1:...,nmu_per_segment,phisplit1:...,nphi_per_segment`

    第二种格式用于分段 log-q Gauss 积分。`nq` 仍保留在配置里作为
    可读标签；真正每段节点数由 `nq_per_segment` 控制。
    第三种格式额外对 `phi` 做分段 Gauss 积分；如果 split 写成 `pi`，
    节点会落在 `(0,pi)` 和 `(pi,2pi)` 两段内部，从而避开精确共面点。
    第四种格式再额外对 loop `mu` 做分段 Gauss 积分，用于诊断
    `q≈k, mu≈1` 附近的端点结构。
    """

    parts = [item.strip() for item in value.split(",")]
    if len(parts) not in (4, 6, 8, 10):
        raise argparse.ArgumentTypeError(
            "config must be qmax,nq,nmu,nphi; "
            "qmax,nq,nmu,nphi,qsplit1:qsplit2:...,nq_per_segment; or "
            "qmax,nq,nmu,nphi,qsplit1:...,nq_per_segment,phisplit1:...,nphi_per_segment; or "
            "qmax,nq,nmu,nphi,qsplit1:...,nq_per_segment,musplit1:...,nmu_per_segment,phisplit1:...,nphi_per_segment"
        )
    qmax, nq, nmu, nphi = parse_float_token(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    if len(parts) == 4:
        return LoopConfig(qmin=1.0e-4, qmax=qmax, nq=nq, nmu=nmu, nphi=nphi)
    q_splits = parse_split_list(parts[4])
    if len(parts) == 6:
        return LoopConfig(qmin=1.0e-4, qmax=qmax, nq=nq, nmu=nmu, nphi=nphi, q_splits=q_splits, nq_per_segment=int(parts[5]))
    if len(parts) == 8:
        phi_splits = parse_split_list(parts[6])
        return LoopConfig(
            qmin=1.0e-4,
            qmax=qmax,
            nq=nq,
            nmu=nmu,
            nphi=nphi,
            q_splits=q_splits,
            nq_per_segment=int(parts[5]),
            phi_splits=phi_splits,
            nphi_per_segment=int(parts[7]),
        )
    mu_splits = parse_split_list(parts[6])
    phi_splits = parse_split_list(parts[8])
    return LoopConfig(
        qmin=1.0e-4,
        qmax=qmax,
        nq=nq,
        nmu=nmu,
        nphi=nphi,
        q_splits=q_splits,
        nq_per_segment=int(parts[5]),
        mu_splits=mu_splits,
        nmu_per_segment=int(parts[7]),
        phi_splits=phi_splits,
        nphi_per_segment=int(parts[9]),
    )


def loop_config_label(config: LoopConfig) -> str:
    """生成简短 loop 配置标签，保持旧配置标签不变。"""

    base = f"qmax{config.qmax:g}_n{config.nq}x{config.nmu}x{config.nphi}"
    label = base
    if config.q_splits:
        n_segments = len(valid_q_splits(config)) + 1
        nq_segment = int(config.nq_per_segment) if int(config.nq_per_segment) > 0 else int(config.nq)
        label = f"{label}_seg{n_segments}x{nq_segment}"
    if config.mu_splits:
        n_segments = len(valid_mu_splits(config)) + 1
        nmu_segment = int(config.nmu_per_segment) if int(config.nmu_per_segment) > 0 else int(config.nmu)
        label = f"{label}_mseg{n_segments}x{nmu_segment}"
    if config.phi_splits:
        n_segments = len(valid_phi_splits(config)) + 1
        nphi_segment = int(config.nphi_per_segment) if int(config.nphi_per_segment) > 0 else int(config.nphi)
        label = f"{label}_pseg{n_segments}x{nphi_segment}"
    return label


def valid_q_splits(config: LoopConfig) -> tuple[float, ...]:
    """返回位于 `(qmin,qmax)` 内的去重分段点。"""

    return tuple(sorted({float(value) for value in config.q_splits if config.qmin < float(value) < config.qmax}))


def valid_mu_splits(config: LoopConfig) -> tuple[float, ...]:
    """返回位于 `(-1,1)` 内的去重 loop-mu 分段点。"""

    return tuple(sorted({float(value) for value in config.mu_splits if -1.0 < float(value) < 1.0}))


def valid_phi_splits(config: LoopConfig) -> tuple[float, ...]:
    """返回位于 `(0,2*pi)` 内的去重 phi 分段点。"""

    return tuple(sorted({float(value) for value in config.phi_splits if 0.0 < float(value) < TWOPI}))


class LinearPower:
    """线性功率谱包装器，统一处理 `k=0` 保护和数组返回。"""

    def __init__(self, pk_func: Callable[[np.ndarray], np.ndarray], z: float, description: str) -> None:
        self.pk_func = pk_func
        self.z = float(z)
        self.description = str(description)

    def __call__(self, k: float | np.ndarray) -> np.ndarray:
        k_array = np.asarray(k, dtype=np.float64)
        k_safe = np.maximum(k_array, K_FLOOR)
        values = np.asarray(self.pk_func(k_safe), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("P_L(k,z) returned non-finite values.")
        if np.any(values < 0.0):
            raise FloatingPointError("P_L(k,z) should be non-negative.")
        return values


def build_quijote_linear_power(z: float) -> tuple[LinearPower, dict[str, float]]:
    """构造 Quijote fiducial `P_L(k,z)`。

    `cosmoprimo` 的 `to_1d(z=z)` 返回指定红移处的线性 matter power。
    因此后续 tree 项应当直接使用 `P_L(z=0.5)^2`，1-loop 项直接使用
    `P_L(z=0.5)^3`；不要再额外乘增长因子。
    """

    try:
        from cosmoprimo import Cosmology
    except ImportError as exc:  # pragma: no cover - 环境缺失时给明确提示。
        raise RuntimeError(
            "cosmoprimo is not available. Activate an environment containing cosmoprimo and "
            "set PYTHONNOUSERSITE=1 before running this script."
        ) from exc

    cosmo = Cosmology(**QUIJOTE_COSMOLOGY, engine="class")
    pk_1d = cosmo.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=float(z))
    description = f"cosmoprimo CLASS delta_cb P_L at z={float(z):g}, Quijote fiducial"
    return LinearPower(pk_1d, z=float(z), description=description), dict(QUIJOTE_COSMOLOGY)


def quijote_power_normalization_audit(z: float, k_values: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)) -> dict[str, object]:
    """记录 `P_L(k,z)` 的红移归一化检查。

    在线性理论中，同一 cosmology 下 `P_L(k,z)/P_L(k,0)` 应该近似为
    与 k 无关的 `D(z)^2/D(0)^2`。把这个检查写进 JSON，后面如果曲线
    量级不对，可以第一时间确认有没有误用 z=0 功率谱或重复乘增长因子。
    """

    try:
        from cosmoprimo import Cosmology
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cosmoprimo is required for the power normalization audit.") from exc

    cosmo = Cosmology(**QUIJOTE_COSMOLOGY, engine="class")
    pk_z = cosmo.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=float(z))
    pk_0 = cosmo.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=0.0)
    k = np.asarray(k_values, dtype=np.float64)
    pz = np.asarray(pk_z(k), dtype=np.float64)
    p0 = np.asarray(pk_0(k), dtype=np.float64)
    ratio = pz / p0
    growth = np.sqrt(ratio)
    return {
        "k_hmpc": k.tolist(),
        "P_L_z": pz.tolist(),
        "P_L_z0": p0.tolist(),
        "P_L_z_over_z0": ratio.tolist(),
        "sqrt_ratio_growth": growth.tolist(),
        "mean_growth": float(np.mean(growth)),
        "max_growth_fractional_scatter": float(np.max(np.abs(growth / np.mean(growth) - 1.0))),
        "interpretation": "P_L used by this script is already evaluated at target z; no extra growth factor is applied.",
    }


def vec_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def flatten_vectors(vectors: Iterable[np.ndarray]) -> tuple[float, ...]:
    """把向量列表转成可缓存的 tuple。

    递归 kernel 会被频繁调用。这里保留 12 位小数，足以覆盖当前
    double precision Gauss 网格，同时显著减少 F4 递归重复计算。
    """

    flat: list[float] = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=np.float64)
        flat.extend(round(float(value), 12) for value in arr)
    return tuple(flat)


def unflatten_vectors(flat: tuple[float, ...]) -> list[np.ndarray]:
    if len(flat) % 3 != 0:
        raise ValueError(f"Kernel cache key length should be multiple of 3, got {len(flat)}")
    return [np.asarray(flat[index : index + 3], dtype=np.float64) for index in range(0, len(flat), 3)]


def alpha_kernel(k_left: np.ndarray, k_right: np.ndarray) -> float:
    """SPT `alpha(k_left,k_right)` kernel。"""

    denom = float(np.dot(k_left, k_left))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(k_left + k_right, k_left) / denom)


def beta_kernel(k_left: np.ndarray, k_right: np.ndarray) -> float:
    """SPT `beta(k_left,k_right)` kernel。"""

    left2 = float(np.dot(k_left, k_left))
    right2 = float(np.dot(k_right, k_right))
    if left2 <= 0.0 or right2 <= 0.0:
        return 0.0
    total2 = float(np.dot(k_left + k_right, k_left + k_right))
    dot = float(np.dot(k_left, k_right))
    return float(total2 * dot / (2.0 * left2 * right2))


@lru_cache(maxsize=400_000)
def kernel_unsym_cached(kind: str, flat: tuple[float, ...]) -> float:
    """未对称化 EdS SPT kernel，按 Bernardeau/Jeong 递归关系计算。"""

    vectors = unflatten_vectors(flat)
    n = len(vectors)
    if n == 1:
        return 1.0
    denom = (2.0 * n + 3.0) * (n - 1.0)
    total = 0.0
    for m in range(1, n):
        left = vectors[:m]
        right = vectors[m:]
        k_left = np.sum(left, axis=0)
        k_right = np.sum(right, axis=0)
        g_left = kernel_unsym_cached("G", flatten_vectors(left))
        f_right = kernel_unsym_cached("F", flatten_vectors(right))
        g_right = kernel_unsym_cached("G", flatten_vectors(right))
        alpha = alpha_kernel(k_left, k_right)
        beta = beta_kernel(k_left, k_right)
        if kind == "F":
            total += g_left * ((2.0 * n + 1.0) * alpha * f_right + 2.0 * beta * g_right) / denom
        elif kind == "G":
            total += g_left * (3.0 * alpha * f_right + 2.0 * n * beta * g_right) / denom
        else:
            raise ValueError(f"Unknown kernel kind {kind!r}")
    return float(total)


@lru_cache(maxsize=400_000)
def kernel_sym_cached(kind: str, flat: tuple[float, ...]) -> float:
    """完全对称化 EdS SPT kernel。"""

    vectors = unflatten_vectors(flat)
    n = len(vectors)
    if n == 1:
        return 1.0
    values = []
    # n<=4，直接遍历全排列最透明；重复排列用 set 去掉。
    for order in set(permutations(range(n))):
        ordered = [vectors[index] for index in order]
        values.append(kernel_unsym_cached(kind, flatten_vectors(ordered)))
    return float(np.mean(values))


def spt_f(vectors: Iterable[np.ndarray]) -> float:
    return kernel_sym_cached("F", flatten_vectors(vectors))


def spt_g(vectors: Iterable[np.ndarray]) -> float:
    return kernel_sym_cached("G", flatten_vectors(vectors))


def react_alpha(k1: float, k2: float, mu: float) -> float:
    """ReACT `SpecialFunctions.cpp::alpha(k1,k2,mu)`."""

    if k1 <= 0.0:
        return 0.0
    return float(1.0 + k2 * mu / k1)


def react_alphas(k1: float, k2: float, mu: float) -> float:
    """ReACT symmetrized alpha."""

    return float(0.5 * (react_alpha(k1, k2, mu) + react_alpha(k2, k1, mu)))


def react_beta1(k1: float, k2: float, mu: float) -> float:
    """ReACT `beta1(k1,k2,mu)`."""

    if k1 <= 0.0 or k2 <= 0.0:
        return 0.0
    return float(mu * (k1 * k1 + k2 * k2 + 2.0 * k1 * k2 * mu) / (2.0 * k1 * k2))


def react_f2eds(k1: float, k2: float, mu: float) -> float:
    return float(5.0 / 7.0 * react_alphas(k1, k2, mu) + 2.0 / 7.0 * react_beta1(k1, k2, mu))


def react_g2eds(k1: float, k2: float, mu: float) -> float:
    return float(3.0 / 7.0 * react_alphas(k1, k2, mu) + 4.0 / 7.0 * react_beta1(k1, k2, mu))


def react_f3edsb(k1: float, k2: float, k3: float, k23: float, k12: float, k13: float, x23: float, x12: float, x13: float) -> float:
    """ReACT optimized EdS `F3edsb`.

    Arguments follow `SpecialFunctions.cpp`: `x23,x12,x13` are the pair
    cosines, and `k23,k12,k13` are the corresponding pair magnitudes.
    """

    if min(k1, k2, k3, k23, k12, k13) <= 0.0:
        return 0.0
    return float(
        (
            2.0 / 63.0 * 2.0 * react_beta1(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 6.0 / 4.0 * react_alphas(k2, k3, x23))
            + 1.0 / 18.0 * react_alpha(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 10.0 / 2.0 * react_alphas(k2, k3, x23))
            + 1.0 / 9.0 * react_alpha(k23, k1, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 6.0 / 4.0 * react_alphas(k2, k3, x23))
            + 2.0 / 63.0 * 2.0 * react_beta1(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 6.0 / 4.0 * react_alphas(k1, k2, x12))
            + 1.0 / 18.0 * react_alpha(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 10.0 / 2.0 * react_alphas(k1, k2, x12))
            + 1.0 / 9.0 * react_alpha(k12, k3, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 6.0 / 4.0 * react_alphas(k1, k2, x12))
            + 2.0 / 63.0 * 2.0 * react_beta1(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 6.0 / 4.0 * react_alphas(k1, k3, x13))
            + 1.0 / 18.0 * react_alpha(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 10.0 / 2.0 * react_alphas(k1, k3, x13))
            + 1.0 / 9.0 * react_alpha(k13, k2, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 6.0 / 4.0 * react_alphas(k1, k3, x13))
        )
        / 3.0
    )


def react_g3edsb(k1: float, k2: float, k3: float, k23: float, k12: float, k13: float, x23: float, x12: float, x13: float) -> float:
    """ReACT optimized EdS `G3edsb`."""

    if min(k1, k2, k3, k23, k12, k13) <= 0.0:
        return 0.0
    return float(
        (
            2.0 / 21.0 * 2.0 * react_beta1(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 6.0 / 4.0 * react_alphas(k2, k3, x23))
            + 1.0 / 42.0 * react_alpha(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 10.0 / 2.0 * react_alphas(k2, k3, x23))
            + 1.0 / 21.0 * react_alpha(k23, k1, (k2 * x12 + k3 * x13) / k23) * (2.0 * react_beta1(k2, k3, x23) + 6.0 / 4.0 * react_alphas(k2, k3, x23))
            + 2.0 / 21.0 * 2.0 * react_beta1(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 6.0 / 4.0 * react_alphas(k1, k2, x12))
            + 1.0 / 42.0 * react_alpha(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 10.0 / 2.0 * react_alphas(k1, k2, x12))
            + 1.0 / 21.0 * react_alpha(k12, k3, (k1 * x13 + k2 * x23) / k12) * (2.0 * react_beta1(k1, k2, x12) + 6.0 / 4.0 * react_alphas(k1, k2, x12))
            + 2.0 / 21.0 * 2.0 * react_beta1(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 6.0 / 4.0 * react_alphas(k1, k3, x13))
            + 1.0 / 42.0 * react_alpha(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 10.0 / 2.0 * react_alphas(k1, k3, x13))
            + 1.0 / 21.0 * react_alpha(k13, k2, (k1 * x12 + k3 * x23) / k13) * (2.0 * react_beta1(k1, k3, x13) + 6.0 / 4.0 * react_alphas(k1, k3, x13))
        )
        / 3.0
    )


def _sqrt_positive(value: float) -> float:
    return math.sqrt(max(0.0, float(value)))


def react_f4edsb(k1: float, k2: float, k3: float, k4: float, x23: float, x24: float, x12: float, x13: float, x14: float) -> float:
    """ReACT optimized EdS `F4edsb` for `k3 dot k4 = XMIN`.

    The argument names use pair labels. This mirrors
    `SpecialFunctions.cpp::F4edsb`, including `REACT_XMIN` for the
    anti-parallel loop pair.
    """

    if min(k1, k2, k3, k4) <= 0.0:
        return 0.0
    x34 = REACT_XMIN
    k1s = k1 * k1
    k2s = k2 * k2
    k3s = k3 * k3
    k4s = k4 * k4
    k1234 = k1s + k2s + k3s + k4s + 2.0 * k1 * k2 * x12 + 2.0 * k1 * k3 * x13 + 2.0 * k2 * k3 * x23 + 2.0 * k1 * k4 * x14 + 2.0 * k2 * k4 * x24 + 2.0 * k3 * k4 * x34
    k123 = k1s + k2s + k3s + 2.0 * k1 * k2 * x12 + 2.0 * k1 * k3 * x13 + 2.0 * k2 * k3 * x23
    k234 = k2s + k3s + k4s + 2.0 * k2 * k3 * x23 + 2.0 * k2 * k4 * x24 + 2.0 * k3 * k4 * x34
    k341 = k3s + k4s + k1s + 2.0 * k3 * k4 * x34 + 2.0 * k3 * k1 * x13 + 2.0 * k4 * k1 * x14
    k412 = k4s + k1s + k2s + 2.0 * k4 * k1 * x14 + 2.0 * k4 * k2 * x24 + 2.0 * k1 * k2 * x12
    k12 = k1s + k2s + 2.0 * k1 * k2 * x12
    k23 = k2s + k3s + 2.0 * k2 * k3 * x23
    k34 = k3s + k4s + 2.0 * k3 * k4 * x34
    k41 = k4s + k1s + 2.0 * k4 * k1 * x14
    k13 = k1s + k3s + 2.0 * k1 * k3 * x13
    k24 = k2s + k4s + 2.0 * k2 * k4 * x24
    if min(k123, k234, k341, k412, k12, k23, k34, k41, k13, k24) <= 1.0e-24:
        return float("nan")

    k12rt = _sqrt_positive(k12)
    k23rt = _sqrt_positive(k23)
    k34rt = _sqrt_positive(k34)
    k41rt = _sqrt_positive(k41)
    k13rt = _sqrt_positive(k13)
    k24rt = _sqrt_positive(k24)

    value = (
        27.0 * (k1s + k1 * k2 * x12 + k1 * k3 * x13 + k1 * k4 * x14) / k1s * react_f3edsb(k2, k3, k4, k34rt, k23rt, k24rt, x34, x23, x24)
        + (27.0 * (k234 + k1 * (k2 * x12 + k3 * x13 + k4 * x14)) / k234 + 6.0 * k1234 * k1 * (k2 * x12 + k3 * x13 + k4 * x14) / k1s / k234) * react_g3edsb(k2, k3, k4, k34rt, k23rt, k24rt, x34, x23, x24)
        + 27.0 * (k2s + k2 * k3 * x23 + k2 * k4 * x24 + k2 * k1 * x12) / k2s * react_f3edsb(k3, k4, k1, k41rt, k34rt, k13rt, x14, x34, x13)
        + (27.0 * (k341 + k2 * (k3 * x23 + k4 * x24 + k1 * x12)) / k341 + 6.0 * k1234 * k2 * (k3 * x23 + k4 * x24 + k1 * x12) / k2s / k341) * react_g3edsb(k3, k4, k1, k41rt, k34rt, k13rt, x14, x34, x13)
        + 27.0 * (k3s + k3 * k4 * x34 + k3 * k1 * x13 + k3 * k2 * x23) / k3s * react_f3edsb(k4, k1, k2, k12rt, k41rt, k24rt, x12, x14, x24)
        + (27.0 * (k412 + k3 * (k4 * x34 + k1 * x13 + k2 * x23)) / k412 + 6.0 * k1234 * k3 * (k4 * x34 + k1 * x13 + k2 * x23) / k3s / k412) * react_g3edsb(k4, k1, k2, k12rt, k41rt, k24rt, x12, x14, x24)
        + 27.0 * (k4s + k4 * k1 * x14 + k4 * k2 * x24 + k4 * k3 * x34) / k4s * react_f3edsb(k1, k2, k3, k23rt, k12rt, k13rt, x23, x12, x13)
        + (27.0 * (k123 + k4 * (k1 * x14 + k2 * x24 + k3 * x34)) / k123 + 6.0 * k1234 * k4 * (k1 * x14 + k2 * x24 + k3 * x34) / k4s / k123) * react_g3edsb(k1, k2, k3, k23rt, k12rt, k13rt, x23, x12, x13)
        + 18.0 * (k12 + k3 * k1 * x13 + k3 * k2 * x23 + k4 * k1 * x14 + k4 * k2 * x24) / k12 * react_g2eds(k1, k2, x12) * react_f2eds(k3, k4, x34)
        + 18.0 * (k23 + k4 * k2 * x24 + k4 * k3 * x34 + k1 * k2 * x12 + k1 * k3 * x13) / k23 * react_g2eds(k2, k3, x23) * react_f2eds(k4, k1, x14)
        + 18.0 * (k34 + k1 * k3 * x13 + k1 * k4 * x14 + k2 * k3 * x23 + k2 * k4 * x24) / k34 * react_g2eds(k3, k4, x34) * react_f2eds(k1, k2, x12)
        + 18.0 * (k13 + k1 * k2 * x12 + k1 * k4 * x14 + k2 * k3 * x23 + k3 * k4 * x34) / k13 * react_g2eds(k1, k3, x13) * react_f2eds(k2, k4, x24)
        + 18.0 * (k24 + k2 * k3 * x23 + k2 * k1 * x12 + k4 * k1 * x14 + k3 * k4 * x34) / k24 * react_g2eds(k2, k4, x24) * react_f2eds(k1, k3, x13)
        + 18.0 * (k41 + k1 * k2 * x12 + k1 * k3 * x13 + k4 * k2 * x24 + k3 * k4 * x34) / k41 * react_g2eds(k1, k4, x14) * react_f2eds(k2, k3, x23)
        + 4.0 * k1234 * (k1 * k3 * x13 + k1 * k4 * x14 + k2 * k3 * x23 + k2 * k4 * x24) / k12 / k34 * react_g2eds(k1, k2, x12) * react_g2eds(k3, k4, x34)
        + 4.0 * k1234 * (k2 * k4 * x24 + k2 * k1 * x12 + k3 * k4 * x34 + k3 * k1 * x13) / k23 / k41 * react_g2eds(k2, k3, x23) * react_g2eds(k4, k1, x14)
        + 4.0 * k1234 * (k2 * k1 * x12 + k2 * k3 * x23 + k4 * k1 * x14 + k4 * k3 * x34) / k24 / k13 * react_g2eds(k2, k4, x24) * react_g2eds(k1, k3, x13)
    ) / 396.0
    return float(value)


def f4edsb_loop_pair(vec_i: np.ndarray, vec_j: np.ndarray, qv: np.ndarray) -> float:
    """Fast ReACT-style `F4(vec_i, vec_j, -q, q)` for B411."""

    ki = vec_norm(vec_i)
    kj = vec_norm(vec_j)
    q = vec_norm(qv)
    if min(ki, kj, q) <= 0.0:
        return 0.0
    x12 = float(np.dot(vec_i, vec_j) / (ki * kj))
    x13 = float(np.dot(vec_i, -qv) / (ki * q))
    x14 = float(np.dot(vec_i, qv) / (ki * q))
    x23 = float(np.dot(vec_j, -qv) / (kj * q))
    x24 = float(np.dot(vec_j, qv) / (kj * q))
    value = react_f4edsb(ki, kj, q, q, x23, x24, x12, x13, x14)
    if math.isfinite(value):
        return value
    return spt_f((vec_i, vec_j, qv, -qv))


def cosine_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    ka = vec_norm(vec_a)
    kb = vec_norm(vec_b)
    if ka <= 0.0 or kb <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (ka * kb))


def f3edsb_vectors(vec_1: np.ndarray, vec_2: np.ndarray, vec_3: np.ndarray) -> float:
    """Fast ReACT-style symmetric `F3(vec_1, vec_2, vec_3)`."""

    k1 = vec_norm(vec_1)
    k2 = vec_norm(vec_2)
    k3 = vec_norm(vec_3)
    if min(k1, k2, k3) <= 0.0:
        return 0.0
    k23 = vec_norm(vec_2 + vec_3)
    k12 = vec_norm(vec_1 + vec_2)
    k13 = vec_norm(vec_1 + vec_3)
    if min(k23, k12, k13) <= 1.0e-12:
        return spt_f((vec_1, vec_2, vec_3))
    value = react_f3edsb(
        k1,
        k2,
        k3,
        k23,
        k12,
        k13,
        cosine_vectors(vec_2, vec_3),
        cosine_vectors(vec_1, vec_2),
        cosine_vectors(vec_1, vec_3),
    )
    if math.isfinite(value):
        return value
    return spt_f((vec_1, vec_2, vec_3))


def f2_analytic(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """解析 F2，用于 sanity check 和可读性。"""

    ka = vec_norm(vec_a)
    kb = vec_norm(vec_b)
    if ka <= 0.0 or kb <= 0.0:
        return 0.0
    mu = float(np.dot(vec_a, vec_b) / (ka * kb))
    return float(5.0 / 7.0 + 0.5 * mu * (ka / kb + kb / ka) + 2.0 / 7.0 * mu * mu)


def triangle_vectors(k1: float, k2: float, mu12: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造闭合三角形三条 Fourier wavevector。

    约定 `k1` 沿 z 轴，`k2` 在 x-z 平面，`k3=-(k1+k2)`。
    因而 `|k3| = sqrt(k1^2+k2^2+2 k1 k2 mu12)`，与现有 B000
    角平均脚本的几何完全一致。
    """

    mu = float(np.clip(mu12, -1.0, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
    v1 = np.asarray([0.0, 0.0, float(k1)], dtype=np.float64)
    v2 = np.asarray([float(k2) * sin_theta, 0.0, float(k2) * mu], dtype=np.float64)
    v3 = -(v1 + v2)
    return v1, v2, v3


def safe_power(pk: LinearPower, vec_or_k: np.ndarray | float) -> float:
    if isinstance(vec_or_k, np.ndarray):
        kval = vec_norm(vec_or_k)
    else:
        kval = float(vec_or_k)
    return float(np.asarray(pk(kval)))


def bispectrum_tree(k1: float, k2: float, mu12: float, pk: LinearPower) -> float:
    """Tree-level matter bispectrum `B112`。"""

    vecs = triangle_vectors(k1, k2, mu12)
    total = 0.0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        total += 2.0 * spt_f((vecs[i], vecs[j])) * safe_power(pk, vecs[i]) * safe_power(pk, vecs[j])
    return float(total)


def loop_nodes(config: LoopConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 log-q、mu、phi 的 Gauss 积分节点和权重。"""

    if config.qmin <= 0.0 or config.qmax <= config.qmin:
        raise ValueError(f"Invalid q range: {config.qmin}, {config.qmax}")
    if config.nq <= 0 or config.nmu <= 0 or config.nphi <= 0:
        raise ValueError(f"Invalid loop node counts: {config}")

    splits = valid_q_splits(config)
    if splits:
        nq_segment = int(config.nq_per_segment) if int(config.nq_per_segment) > 0 else int(config.nq)
        if nq_segment <= 0:
            raise ValueError(f"Invalid nq_per_segment: {config.nq_per_segment}")
        q_chunks = []
        wlog_chunks = []
        xq_seg, wq_seg = np.polynomial.legendre.leggauss(nq_segment)
        bounds = (float(config.qmin), *splits, float(config.qmax))
        for left, right in zip(bounds[:-1], bounds[1:]):
            log_min = math.log(left)
            log_max = math.log(right)
            log_mid = 0.5 * (log_min + log_max)
            log_half = 0.5 * (log_max - log_min)
            q_chunks.append(np.exp(log_mid + log_half * xq_seg))
            wlog_chunks.append(log_half * wq_seg)
        q = np.concatenate(q_chunks)
        wlog = np.concatenate(wlog_chunks)
    else:
        xq, wq = np.polynomial.legendre.leggauss(config.nq)
        log_min = math.log(config.qmin)
        log_max = math.log(config.qmax)
        log_mid = 0.5 * (log_min + log_max)
        log_half = 0.5 * (log_max - log_min)
        q = np.exp(log_mid + log_half * xq)
        wlog = log_half * wq

    mu_splits = valid_mu_splits(config)
    if mu_splits:
        nmu_segment = int(config.nmu_per_segment) if int(config.nmu_per_segment) > 0 else int(config.nmu)
        if nmu_segment <= 0:
            raise ValueError(f"Invalid nmu_per_segment: {config.nmu_per_segment}")
        mu_chunks = []
        wmu_chunks = []
        xu_seg, wu_seg = np.polynomial.legendre.leggauss(nmu_segment)
        bounds = (-1.0, *mu_splits, 1.0)
        for left, right in zip(bounds[:-1], bounds[1:]):
            mu_mid = 0.5 * (float(left) + float(right))
            mu_half = 0.5 * (float(right) - float(left))
            mu_chunks.append(mu_mid + mu_half * xu_seg)
            wmu_chunks.append(mu_half * wu_seg)
        xu = np.concatenate(mu_chunks)
        wu = np.concatenate(wmu_chunks)
    else:
        xu, wu = np.polynomial.legendre.leggauss(config.nmu)

    phi_splits = valid_phi_splits(config)
    if phi_splits:
        nphi_segment = int(config.nphi_per_segment) if int(config.nphi_per_segment) > 0 else int(config.nphi)
        if nphi_segment <= 0:
            raise ValueError(f"Invalid nphi_per_segment: {config.nphi_per_segment}")
        phi_chunks = []
        wphi_chunks = []
        xp_seg, wp_seg = np.polynomial.legendre.leggauss(nphi_segment)
        bounds = (0.0, *phi_splits, TWOPI)
        for left, right in zip(bounds[:-1], bounds[1:]):
            phi_mid = 0.5 * (float(left) + float(right))
            phi_half = 0.5 * (float(right) - float(left))
            phi_chunks.append(phi_mid + phi_half * xp_seg)
            wphi_chunks.append(phi_half * wp_seg)
        phi = np.concatenate(phi_chunks)
        wphi = np.concatenate(wphi_chunks)
    else:
        xp, wp = np.polynomial.legendre.leggauss(config.nphi)
        phi = math.pi * (xp + 1.0)
        wphi = math.pi * wp
    return q, wlog, xu, wu, phi, wphi


def q_vector(q: float, mu_q1: float, phi: float) -> np.ndarray:
    sin_q = math.sqrt(max(0.0, 1.0 - float(mu_q1) * float(mu_q1)))
    return np.asarray(
        [float(q) * sin_q * math.cos(float(phi)), float(q) * sin_q * math.sin(float(phi)), float(q) * float(mu_q1)],
        dtype=np.float64,
    )


def react_p13_shape(r: float) -> float:
    """ReACT/SPT `P13_dd` 一维积分里的 EdS shape 函数。

    ReACT `SPT.cpp::f13_dd` 使用分段展开处理 `r=q/k` 的小 r、r=1 和
    大 r 极限，避免直接公式的数值抵消。这里保持同一口径；输入功率谱
    已经在目标红移，因此不再乘 ReACT 中的 growth normalization。
    """

    r = float(r)
    if r <= 0.0:
        return 0.0
    if r < 1.0e-2:
        return float(-168.0 + (928.0 / 5.0) * r**2 - (4512.0 / 35.0) * r**4 + (416.0 / 21.0) * r**6)
    if abs(r - 1.0) < 1.0e-10:
        return float(-88.0 + 8.0 * (r - 1.0))
    if r > 100.0:
        return float(-488.0 / 5.0 + (96.0 / 5.0) / r**2 - (160.0 / 21.0) / r**4 - (1376.0 / 1155.0) / r**6)
    return float(
        12.0 / r**2
        - 158.0
        + 100.0 * r**2
        - 42.0 * r**4
        + 3.0 / r**3 * (r * r - 1.0) ** 3 * (7.0 * r * r + 2.0) * math.log((1.0 + r) / abs(1.0 - r))
    )


def p13_direct_angular(k: float, pk: LinearPower, config: LoopConfig) -> float:
    """直接角积分版本的 `P13`，用于审计 ReACT 一维公式。

    该积分绕外部 `k` 轴轴对称，因此只需要 `q` 和 `mu` 两维；
    `phi` 已经解析积分出 `2*pi`。
    """

    q_nodes, wlog, mu_nodes, wmu, _, _ = loop_nodes(config)
    kvec = np.asarray([0.0, 0.0, float(k)], dtype=np.float64)
    integral = 0.0
    for q, wq in zip(q_nodes, wlog):
        p_q = safe_power(pk, float(q))
        q3_weight = float(q) ** 3 * float(wq)
        for mu, wu in zip(mu_nodes, wmu):
            qv = q_vector(float(q), float(mu), 0.0)
            measure = q3_weight * float(wu) / (4.0 * math.pi * math.pi)
            integral += measure * spt_f((kvec, qv, -qv)) * p_q
    return float(6.0 * safe_power(pk, float(k)) * integral)


def p13(k: float, pk: LinearPower, config: LoopConfig) -> float:
    """计算 ReACT 口径的 `P13_dd(k)` 一维解析角积分。

    ReACT 写成 `k^3 P_L(k)/(252*4*pi^2) int dr P_L(kr) S(r)`。
    本代码的积分变量是绝对波数 `q`，因此 `dr=dq/k`，用 log-q 节点时
    变成 `k^2 P_L(k)/(252*4*pi^2) int dlogq q P_L(q) S(q/k)`。
    """

    q_nodes, wlog, _, _, _, _ = loop_nodes(config)
    integral = 0.0
    for q, wq in zip(q_nodes, wlog):
        integral += float(wq) * float(q) * safe_power(pk, float(q)) * react_p13_shape(float(q) / float(k))
    return float(safe_power(pk, float(k)) * float(k) ** 2 * integral / (252.0 * 4.0 * math.pi * math.pi))


def loop_integrand_components(
    vecs: tuple[np.ndarray, np.ndarray, np.ndarray],
    qv: np.ndarray,
    pk: LinearPower,
    include_b411: bool = True,
) -> dict[str, float]:
    """返回 `B222/B321I/B411` 的单点 integrand。

    这里不包含积分测度 `d^3q/(2*pi)^3`，但包含 diagram 组合因子
    `8/6/12`。这个函数是生产积分和 ReACT 对齐审计共用的最小公式单元。
    """

    p_ext = [safe_power(pk, vec) for vec in vecs]
    p_q = safe_power(pk, qv)

    # B222: 8 int F2(q,k1-q) F2(-q,k2+q) F2(k1-q,k2+q) PPP
    p1 = vecs[0] - qv
    p2 = vecs[1] + qv
    b222 = (
        8.0
        * f2_analytic(qv, p1)
        * f2_analytic(-qv, p2)
        * f2_analytic(p1, p2)
        * p_q
        * safe_power(pk, p1)
        * safe_power(pk, p2)
    )

    # B321I: 6 int sum_{a!=b} P(b) F2(q,a-q) F3(b,q,a-q) P(q)P(a-q)
    local_321i = 0.0
    for a_index in range(3):
        a_minus_q = vecs[a_index] - qv
        p_a_minus_q = safe_power(pk, a_minus_q)
        f2_part = f2_analytic(qv, a_minus_q)
        for b_index in range(3):
            if b_index == a_index:
                continue
            local_321i += p_ext[b_index] * f2_part * f3edsb_vectors(vecs[b_index], qv, a_minus_q) * p_q * p_a_minus_q
    b321i = 6.0 * local_321i

    b411 = 0.0
    if include_b411:
        # B411: 12 int sum_{pairs} P_i P_j F4(k_i,k_j,q,-q) P(q)
        local_411 = 0.0
        for i, j in ((0, 1), (1, 2), (2, 0)):
            local_411 += p_ext[i] * p_ext[j] * f4edsb_loop_pair(vecs[i], vecs[j], qv) * p_q
        b411 = 12.0 * local_411

    return {"B222": float(b222), "B321I": float(b321i), "B411": float(b411)}


def genz_malik_basic_rule(
    func: Callable[[np.ndarray], float],
    center: np.ndarray,
    width: np.ndarray,
) -> tuple[float, float, int, int]:
    """执行一次 3D Genz-Malik 7/5 basic rule。

    这是 ReACT/Copter `Quadrature.inl::Integrate<3>` 同族的规则。返回的
    `div_axis` 是四阶差分最大的方向，adaptive 主循环会优先沿这个方向
    二分当前误差最大的子区域。
    """

    n_dim = 3
    two_to_n = 1 << n_dim
    lambda2 = math.sqrt(9.0 / 70.0)
    lambda4 = math.sqrt(9.0 / 10.0)
    lambda5 = math.sqrt(9.0 / 19.0)
    wt1 = (12824.0 - 9120.0 * n_dim + 400.0 * n_dim * n_dim) / 19683.0
    wt2 = 980.0 / 6561.0
    wt3 = (1820.0 - 400.0 * n_dim) / 19683.0
    wt4 = 200.0 / 19683.0
    wt5 = 6859.0 / 19683.0 / two_to_n
    wtp1 = (729.0 - 950.0 * n_dim + 50.0 * n_dim * n_dim) / 729.0
    wtp2 = 245.0 / 486.0
    wtp3 = (265.0 - 100.0 * n_dim) / 1458.0
    wtp4 = 25.0 / 729.0
    ratio = (lambda2 / lambda4) ** 2

    neval = 0

    def call(point: np.ndarray) -> float:
        nonlocal neval
        neval += 1
        value = float(func(point))
        if not math.isfinite(value):
            return 0.0
        return value

    rgnvol = float(two_to_n * np.prod(width))
    sum1 = call(center.astype(np.float64, copy=True))

    sum2 = 0.0
    sum3 = 0.0
    difmax = -1.0
    div_axis = 0
    for axis in range(n_dim):
        point = center.copy()
        point[axis] = center[axis] - lambda2 * width[axis]
        f1 = call(point)
        point[axis] = center[axis] + lambda2 * width[axis]
        f2 = call(point)
        point[axis] = center[axis] - lambda4 * width[axis]
        f3 = call(point)
        point[axis] = center[axis] + lambda4 * width[axis]
        f4 = call(point)
        sum2 += f1 + f2
        sum3 += f3 + f4
        df1 = f1 + f2 - 2.0 * sum1
        df2 = f3 + f4 - 2.0 * sum1
        diff = abs(df1 - ratio * df2)
        if diff >= difmax:
            difmax = diff
            div_axis = axis

    sum4 = 0.0
    for axis1 in range(n_dim - 1):
        for axis2 in range(axis1 + 1, n_dim):
            for sign1 in (-1.0, 1.0):
                for sign2 in (-1.0, 1.0):
                    point = center.copy()
                    point[axis1] = center[axis1] + sign1 * lambda4 * width[axis1]
                    point[axis2] = center[axis2] + sign2 * lambda4 * width[axis2]
                    sum4 += call(point)

    sum5 = 0.0
    for signs in np.ndindex(2, 2, 2):
        point = center.copy()
        for axis, bit in enumerate(signs):
            point[axis] = center[axis] + (-1.0 if bit == 0 else 1.0) * lambda5 * width[axis]
        sum5 += call(point)

    rgncmp = rgnvol * (wtp1 * sum1 + wtp2 * sum2 + wtp3 * sum3 + wtp4 * sum4)
    rgnval = rgnvol * (wt1 * sum1 + wt2 * sum2 + wt3 * sum3 + wt4 * sum4 + wt5 * sum5)
    return float(rgnval), float(abs(rgnval - rgncmp)), int(div_axis), int(neval)


def adaptive_integrate_3d(
    func: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    config: AdaptiveConfig,
) -> AdaptiveResult:
    """ReACT-like 3D adaptive cubature，按最大误差区域递归二分。"""

    center = 0.5 * (lower + upper)
    width = 0.5 * (upper - lower)
    value, error, div_axis, neval = genz_malik_basic_rule(func, center, width)
    heap: list[AdaptiveRegion] = [
        AdaptiveRegion(-error, 0, tuple(center.tolist()), tuple(width.tolist()), value, error, div_axis)
    ]
    total_value = float(value)
    total_error = float(error)
    next_index = 1
    stopped_reason = "running"

    while True:
        relerr = total_error / max(abs(total_value), np.finfo(float).tiny)
        if neval >= int(config.min_evals) and (relerr < float(config.epsrel) or total_error < float(config.epsabs)):
            stopped_reason = "tolerance"
            break
        if neval + 2 * 33 > int(config.max_evals):
            stopped_reason = "max_evals"
            break
        if not heap:
            stopped_reason = "empty_heap"
            break

        region = heapq.heappop(heap)
        total_value -= region.value
        total_error -= region.error
        old_center = np.asarray(region.center, dtype=np.float64)
        old_width = np.asarray(region.width, dtype=np.float64)
        axis = int(region.div_axis)
        child_width = old_width.copy()
        child_width[axis] *= 0.5

        for offset in (-1.0, 1.0):
            child_center = old_center.copy()
            child_center[axis] += offset * child_width[axis]
            child_value, child_error, child_axis, child_eval = genz_malik_basic_rule(func, child_center, child_width)
            neval += child_eval
            total_value += child_value
            total_error += child_error
            heapq.heappush(
                heap,
                AdaptiveRegion(
                    -child_error,
                    next_index,
                    tuple(child_center.tolist()),
                    tuple(child_width.tolist()),
                    child_value,
                    child_error,
                    child_axis,
                ),
            )
            next_index += 1

    relerr = total_error / max(abs(total_value), np.finfo(float).tiny)
    return AdaptiveResult(
        value=float(total_value),
        abserr=float(total_error),
        relerr=float(relerr),
        neval=int(neval),
        n_regions=int(len(heap)),
        converged=stopped_reason == "tolerance",
        stopped_reason=stopped_reason,
        max_region_error=float(-heap[0].priority) if heap else 0.0,
    )


def make_reactstyle_loop_integrand(
    *,
    vecs: tuple[np.ndarray, np.ndarray, np.ndarray],
    pk: LinearPower,
    component: str,
) -> Callable[[np.ndarray], float]:
    """构造 ReACT `(r,u,phi)` 变量下的 loop component integrand。

    返回值只包含 `r^2 * component_integrand`，不含外部 prefactor
    `(k1/2pi)^3`。ReACT 的 `phi` 符号和本文件 `q_vector` 约定相反，
    因此这里使用 `phi_python = pi - phi_react`。
    """

    if component not in {"B222", "B321I", "B411", "Bloopterms"}:
        raise ValueError(f"unknown adaptive loop component {component!r}")

    k1_norm = vec_norm(vecs[0])

    def integrand(point: np.ndarray) -> float:
        r = float(point[0])
        u = float(point[1])
        phi_react = float(point[2])
        d_minus = math.sqrt(max(0.0, 1.0 + r * r - 2.0 * r * u))
        d_plus = math.sqrt(max(0.0, 1.0 + r * r + 2.0 * r * u))
        if component in {"B222", "B321I", "Bloopterms"} and d_minus < 1.0e-5:
            return 0.0
        if component in {"B321I", "Bloopterms"} and d_plus < 1.0e-5:
            return 0.0

        q = k1_norm * r
        qv = q_vector(q, u, math.pi - phi_react)
        comps = loop_integrand_components(vecs, qv, pk, include_b411=component in {"B411", "Bloopterms"})
        if component == "B222":
            value = comps["B222"]
        elif component == "B321I":
            value = comps["B321I"]
        elif component == "B411":
            value = comps["B411"]
        else:
            value = comps["B222"] + comps["B321I"] + comps["B411"]
        return float(r * r * value)

    return integrand


def adaptive_loop_components(
    k1: float,
    k2: float,
    mu12: float,
    pk: LinearPower,
    adaptive_config: AdaptiveConfig,
    components: tuple[str, ...] = ("B222", "B321I", "B411"),
    retry_config: AdaptiveConfig | None = None,
    retry_relerr_threshold: float | None = None,
    retry_error_weight: float = 1.0,
    retry_weighted_abserr_threshold: float | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """用 ReACT-style adaptive cubature 计算指定 loop components。

    返回 `(values, meta)`。`values` 是物理量，已乘 `(k1/2pi)^3`；
    `meta` 保存 raw integral、误差估计和 eval 次数，供诊断 JSON 使用。
    如果设置 `retry_config`，则对未收敛、`relerr` 超过阈值，或加权绝对误差
    超过阈值的 component 用第二套配置重算；metadata 中保留 initial/final
    attempt。
    """

    vecs = triangle_vectors(k1, k2, mu12)
    k1_norm = vec_norm(vecs[0])
    lower = np.asarray([float(adaptive_config.qmin) / k1_norm, float(adaptive_config.umin), 0.0], dtype=np.float64)
    upper = np.asarray([float(adaptive_config.qmax) / k1_norm, float(adaptive_config.umax), TWOPI], dtype=np.float64)
    prefactor = k1_norm**3 / (TWOPI**3)
    values: dict[str, float] = {}
    meta: dict[str, dict[str, float]] = {}

    for component in components:
        integrand = make_reactstyle_loop_integrand(vecs=vecs, pk=pk, component=component)
        result = adaptive_integrate_3d(integrand, lower, upper, adaptive_config)
        attempts: list[dict[str, object]] = [{"stage": "initial", **asdict(result)}]
        threshold = float(retry_relerr_threshold) if retry_relerr_threshold is not None else float(adaptive_config.epsrel)
        initial_abserr_scaled = float(abs(prefactor) * result.abserr)
        initial_weighted_abserr = float(abs(retry_error_weight) * initial_abserr_scaled)
        trigger_reasons: list[str] = []
        if not bool(result.converged):
            trigger_reasons.append("not_converged")
        if float(result.relerr) > threshold:
            trigger_reasons.append("relerr")
        if retry_weighted_abserr_threshold is not None and initial_weighted_abserr > float(retry_weighted_abserr_threshold):
            trigger_reasons.append("weighted_abserr")
        should_retry = retry_config is not None and bool(trigger_reasons)
        final_result = result
        if should_retry:
            final_result = adaptive_integrate_3d(integrand, lower, upper, retry_config)
            attempts.append({"stage": "retry", **asdict(final_result)})

        values[component] = float(prefactor * final_result.value)
        final_abserr_scaled = float(abs(prefactor) * final_result.abserr)
        final_weighted_abserr = float(abs(retry_error_weight) * final_abserr_scaled)
        meta[component] = {
            **asdict(final_result),
            "raw_integral": float(final_result.value),
            "raw_abserr": float(final_result.abserr),
            "value": float(values[component]),
            "abserr_scaled": final_abserr_scaled,
            "prefactor": float(prefactor),
            "retried": bool(should_retry),
            "retry_trigger_reasons": trigger_reasons,
            "retry_error_weight": float(retry_error_weight),
            "retry_weighted_abserr_threshold": None
            if retry_weighted_abserr_threshold is None
            else float(retry_weighted_abserr_threshold),
            "weighted_abserr": final_weighted_abserr,
            "n_attempts": int(len(attempts)),
            "total_neval_including_retries": int(sum(int(attempt["neval"]) for attempt in attempts)),
            "initial_converged": bool(result.converged),
            "initial_stopped_reason": str(result.stopped_reason),
            "initial_relerr": float(result.relerr),
            "initial_neval": int(result.neval),
            "initial_abserr_scaled": initial_abserr_scaled,
            "initial_weighted_abserr": initial_weighted_abserr,
            "attempts": attempts,
        }
    return values, meta


def b321ii_from_p13(
    vecs: tuple[np.ndarray, np.ndarray, np.ndarray],
    pk: LinearPower,
    config: LoopConfig,
    p13_cache: dict[object, float] | None,
    p13_config: LoopConfig | None = None,
) -> float:
    """计算 `B321II` propagator correction。

    `B321II` 可由一维 `P13(k)` 写出，不需要 3D adaptive cubature。这里单独
    抽出来，方便 fixed/adaptive 两个后端共享同一个归一化和缓存。
    """

    p13_integration_config = p13_config if p13_config is not None else config
    p_ext = [safe_power(pk, vec) for vec in vecs]
    k_ext = [vec_norm(vec) for vec in vecs]
    if p13_cache is None:
        p13_cache = {}

    def cached_p13(kval: float) -> float:
        key = (round(float(kval), 12), loop_config_label(p13_integration_config))
        if key not in p13_cache:
            p13_cache[key] = p13(float(kval), pk, p13_integration_config)
        return p13_cache[key]

    b321ii = 0.0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        b321ii += f2_analytic(vecs[i], vecs[j]) * (
            p_ext[i] * cached_p13(k_ext[j]) + p_ext[j] * cached_p13(k_ext[i])
        )
    return float(b321ii)


def bispectrum_1loop_components(
    k1: float,
    k2: float,
    mu12: float,
    pk: LinearPower,
    config: LoopConfig,
    p13_cache: dict[object, float] | None = None,
    *,
    p13_config: LoopConfig | None = None,
) -> dict[str, float]:
    """计算单个三角形的 SPT 1-loop 组件。"""

    vecs = triangle_vectors(k1, k2, mu12)
    b321ii = b321ii_from_p13(vecs, pk, config, p13_cache, p13_config=p13_config)

    q_nodes, wlog, mu_nodes, wmu, phi_nodes, wphi = loop_nodes(config)
    b222 = 0.0
    b321i = 0.0
    b411 = 0.0
    measure_norm = 1.0 / (TWOPI**3)

    for q, wq in zip(q_nodes, wlog):
        p_q = safe_power(pk, float(q))
        q3_weight = float(q) ** 3 * float(wq)
        for mu_q1, wu in zip(mu_nodes, wmu):
            for phi, wp in zip(phi_nodes, wphi):
                qv = q_vector(float(q), float(mu_q1), float(phi))
                measure = q3_weight * float(wu) * float(wp) * measure_norm
                components = loop_integrand_components(vecs, qv, pk)
                b222 += measure * components["B222"]
                b321i += measure * components["B321I"]
                b411 += measure * components["B411"]

    return {
        "B222": float(b222),
        "B321I": float(b321i),
        "B321II": float(b321ii),
        "B411": float(b411),
        "B1loop": float(b222 + b321i + b321ii + b411),
    }


def bispectrum_1loop_components_backend(
    k1: float,
    k2: float,
    mu12: float,
    pk: LinearPower,
    config: LoopConfig,
    p13_cache: dict[object, float] | None = None,
    *,
    loop_backend: str = "fixed",
    adaptive_config: AdaptiveConfig | None = None,
    adaptive_components: tuple[str, ...] = ("B222", "B321I", "B411"),
    p13_config: LoopConfig | None = None,
    adaptive_meta_out: list[dict[str, object]] | None = None,
    adaptive_retry_config: AdaptiveConfig | None = None,
    adaptive_retry_relerr_threshold: float | None = None,
    adaptive_retry_error_weight: float = 1.0,
    adaptive_retry_weighted_abserr_threshold: float | None = None,
) -> dict[str, float]:
    """按指定后端计算单三角形 1-loop components。

    `loop_backend="fixed"` 完全等价于原始 fixed Gauss 实现。
    `loop_backend="adaptive"` 时，只对 `adaptive_components` 中列出的
    `B222/B321I/B411` 使用 ReACT-style adaptive cubature；未列出的
    component 仍由 fixed Gauss 填入，因此可以低成本做 hybrid 验证。
    """

    if loop_backend == "fixed":
        return bispectrum_1loop_components(k1, k2, mu12, pk, config, p13_cache=p13_cache, p13_config=p13_config)
    if loop_backend != "adaptive":
        raise ValueError(f"unknown loop_backend {loop_backend!r}; expected 'fixed' or 'adaptive'")

    selected = tuple(dict.fromkeys(str(item) for item in adaptive_components))
    invalid = [item for item in selected if item not in {"B222", "B321I", "B411"}]
    if invalid:
        raise ValueError(f"adaptive_components must be drawn from B222,B321I,B411, got {invalid}")

    if adaptive_config is None:
        adaptive_config = AdaptiveConfig(qmin=float(config.qmin), qmax=float(config.qmax))
    values: dict[str, float] = {}
    if selected:
        adaptive_values, adaptive_meta = adaptive_loop_components(
            k1,
            k2,
            mu12,
            pk,
            adaptive_config,
            selected,
            retry_config=adaptive_retry_config,
            retry_relerr_threshold=adaptive_retry_relerr_threshold,
            retry_error_weight=adaptive_retry_error_weight,
            retry_weighted_abserr_threshold=adaptive_retry_weighted_abserr_threshold,
        )
        values.update(adaptive_values)
        if adaptive_meta_out is not None:
            adaptive_meta_out.append(
                {
                    "k1": float(k1),
                    "k2": float(k2),
                    "mu12": float(mu12),
                    "k3": float(vec_norm(triangle_vectors(k1, k2, mu12)[2])),
                    "adaptive_components": list(selected),
                    "components": adaptive_meta,
                }
            )

    missing = tuple(component for component in ("B222", "B321I", "B411") if component not in values)
    fixed_values: dict[str, float] | None = None
    if missing:
        fixed_values = bispectrum_1loop_components(k1, k2, mu12, pk, config, p13_cache=p13_cache, p13_config=p13_config)
        for component in missing:
            values[component] = float(fixed_values[component])

    if fixed_values is not None:
        b321ii = float(fixed_values["B321II"])
    else:
        b321ii = b321ii_from_p13(triangle_vectors(k1, k2, mu12), pk, config, p13_cache, p13_config=p13_config)

    b222 = float(values["B222"])
    b321i = float(values["B321I"])
    b411 = float(values["B411"])
    return {
        "B222": b222,
        "B321I": b321i,
        "B321II": float(b321ii),
        "B411": b411,
        "B1loop": float(b222 + b321i + b321ii + b411),
    }


def bispectrum_total(
    k1: float,
    k2: float,
    mu12: float,
    pk: LinearPower,
    config: LoopConfig,
    p13_cache: dict[object, float] | None = None,
    *,
    loop_backend: str = "fixed",
    adaptive_config: AdaptiveConfig | None = None,
    adaptive_components: tuple[str, ...] = ("B222", "B321I", "B411"),
    p13_config: LoopConfig | None = None,
    adaptive_meta_out: list[dict[str, object]] | None = None,
    adaptive_retry_config: AdaptiveConfig | None = None,
    adaptive_retry_relerr_threshold: float | None = None,
    adaptive_retry_error_weight: float = 1.0,
    adaptive_retry_weighted_abserr_threshold: float | None = None,
) -> dict[str, float]:
    tree = bispectrum_tree(k1, k2, mu12, pk)
    loop = bispectrum_1loop_components_backend(
        k1,
        k2,
        mu12,
        pk,
        config,
        p13_cache=p13_cache,
        loop_backend=loop_backend,
        adaptive_config=adaptive_config,
        adaptive_components=adaptive_components,
        p13_config=p13_config,
        adaptive_meta_out=adaptive_meta_out,
        adaptive_retry_config=adaptive_retry_config,
        adaptive_retry_relerr_threshold=adaptive_retry_relerr_threshold,
        adaptive_retry_error_weight=adaptive_retry_error_weight,
        adaptive_retry_weighted_abserr_threshold=adaptive_retry_weighted_abserr_threshold,
    )
    return {"Btree": tree, **loop, "Btotal": float(tree + loop["B1loop"])}


def project_b000(
    k1: float,
    k2: float,
    pk: LinearPower,
    config: LoopConfig,
    nmu_b000: int,
    include_loop: bool,
    p13_cache: dict[object, float] | None = None,
    *,
    loop_backend: str = "fixed",
    adaptive_config: AdaptiveConfig | None = None,
    adaptive_components: tuple[str, ...] = ("B222", "B321I", "B411"),
    p13_config: LoopConfig | None = None,
    adaptive_meta_out: list[dict[str, object]] | None = None,
    adaptive_retry_config: AdaptiveConfig | None = None,
    adaptive_retry_relerr_threshold: float | None = None,
    adaptive_retry_weighted_abserr_threshold: float | None = None,
) -> dict[str, float]:
    """把 `B(k1,k2,k3)` 投影到 real-space Sugiyama `B000(k1,k2)`。"""

    mu_nodes, weights = np.polynomial.legendre.leggauss(int(nmu_b000))
    sums = {"Btree": 0.0, "B222": 0.0, "B321I": 0.0, "B321II": 0.0, "B411": 0.0, "B1loop": 0.0, "Btotal": 0.0}
    if p13_cache is None:
        p13_cache = {}
    for mu_index, (mu, weight) in enumerate(zip(mu_nodes, weights)):
        if include_loop:
            meta_start = len(adaptive_meta_out) if adaptive_meta_out is not None else 0
            b000_weight = float(0.5 * float(weight))
            values = bispectrum_total(
                k1,
                k2,
                float(mu),
                pk,
                config,
                p13_cache=p13_cache,
                loop_backend=loop_backend,
                adaptive_config=adaptive_config,
                adaptive_components=adaptive_components,
                p13_config=p13_config,
                adaptive_meta_out=adaptive_meta_out,
                adaptive_retry_config=adaptive_retry_config,
                adaptive_retry_relerr_threshold=adaptive_retry_relerr_threshold,
                adaptive_retry_error_weight=b000_weight,
                adaptive_retry_weighted_abserr_threshold=adaptive_retry_weighted_abserr_threshold,
            )
            if adaptive_meta_out is not None:
                for item in adaptive_meta_out[meta_start:]:
                    item["b000_mu_index"] = int(mu_index)
                    item["b000_weight"] = b000_weight
        else:
            tree = bispectrum_tree(k1, k2, float(mu), pk)
            values = {"Btree": tree, "B222": 0.0, "B321I": 0.0, "B321II": 0.0, "B411": 0.0, "B1loop": 0.0, "Btotal": tree}
        for key in sums:
            sums[key] += 0.5 * float(weight) * float(values[key])
    return {f"B000_{key}": float(value) for key, value in sums.items()}


def project_b000_shape_aware(
    k1: float,
    k2: float,
    pk: LinearPower,
    base_config: LoopConfig,
    endpoint_config: LoopConfig,
    nmu_b000: int,
    include_loop: bool,
    p13_cache: dict[object, float] | None = None,
    *,
    threshold: float = 0.7,
    loop_backend: str = "fixed",
    adaptive_config: AdaptiveConfig | None = None,
    adaptive_components: tuple[str, ...] = ("B222", "B321I", "B411"),
    p13_config: LoopConfig | None = None,
    adaptive_meta_out: list[dict[str, object]] | None = None,
    adaptive_retry_config: AdaptiveConfig | None = None,
    adaptive_retry_relerr_threshold: float | None = None,
    adaptive_retry_weighted_abserr_threshold: float | None = None,
) -> dict[str, float]:
    """用外层三角形形状选择 loop 积分配置来投影 `B000`。

    这是 lightweight integrator 的局部加密入口：外层 `|mu12|>=threshold`
    的 squeezed/flattened 角节点使用 `endpoint_config`，其余角节点使用
    `base_config`。tree 与 `P13` 归一化不依赖这个选择；`P13` cache 因而
    可以安全共享。
    """

    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"shape-aware threshold must be in [0, 1], got {threshold}")

    mu_nodes, weights = np.polynomial.legendre.leggauss(int(nmu_b000))
    sums = {"Btree": 0.0, "B222": 0.0, "B321I": 0.0, "B321II": 0.0, "B411": 0.0, "B1loop": 0.0, "Btotal": 0.0}
    if p13_cache is None:
        p13_cache = {}

    for mu_index, (mu, weight) in enumerate(zip(mu_nodes, weights)):
        mu_value = float(mu)
        config = endpoint_config if abs(mu_value) >= threshold else base_config
        if include_loop:
            meta_start = len(adaptive_meta_out) if adaptive_meta_out is not None else 0
            b000_weight = float(0.5 * float(weight))
            values = bispectrum_total(
                k1,
                k2,
                mu_value,
                pk,
                config,
                p13_cache=p13_cache,
                loop_backend=loop_backend,
                adaptive_config=adaptive_config,
                adaptive_components=adaptive_components,
                p13_config=p13_config,
                adaptive_meta_out=adaptive_meta_out,
                adaptive_retry_config=adaptive_retry_config,
                adaptive_retry_relerr_threshold=adaptive_retry_relerr_threshold,
                adaptive_retry_error_weight=b000_weight,
                adaptive_retry_weighted_abserr_threshold=adaptive_retry_weighted_abserr_threshold,
            )
            if adaptive_meta_out is not None:
                for item in adaptive_meta_out[meta_start:]:
                    item["b000_mu_index"] = int(mu_index)
                    item["b000_weight"] = b000_weight
                    item["shape_aware_config"] = "endpoint" if abs(mu_value) >= threshold else "base"
        else:
            tree = bispectrum_tree(k1, k2, mu_value, pk)
            values = {"Btree": tree, "B222": 0.0, "B321I": 0.0, "B321II": 0.0, "B411": 0.0, "B1loop": 0.0, "Btotal": tree}
        for key in sums:
            sums[key] += 0.5 * float(weight) * float(values[key])
    return {f"B000_{key}": float(value) for key, value in sums.items()}


def load_quijote_dm_pre_mean(
    matrix_path: Path,
    fiducial_sample: str = "common",
) -> dict[str, np.ndarray]:
    """读取 fiducial pre-recon B000 均值。

    ``common`` 保留历史 PNG 对照口径，只使用三宇宙学共同 realization；
    ``all`` 使用矩阵内全部 fiducial realization，适合纯 Gaussian 审计。
    """

    data = np.load(matrix_path)
    sample_key = {
        "common": "fiducial_common_pre_B000",
        "all": "fiducial_pre_B000",
    }.get(str(fiducial_sample))
    if sample_key is None:
        raise ValueError(f"unknown fiducial_sample {fiducial_sample!r}; expected 'common' or 'all'")
    matrix = np.asarray(data[sample_key], dtype=np.float64)
    return {
        "k1": np.asarray(data["k1"], dtype=np.float64),
        "k2": np.asarray(data["k2"], dtype=np.float64),
        "pair_i": np.asarray(data["pair_i"], dtype=np.int64),
        "pair_j": np.asarray(data["pair_j"], dtype=np.int64),
        "k_centers": np.asarray(data["k_centers"], dtype=np.float64),
        "hdf5_k_weighted": np.asarray(data["hdf5_k_weighted"], dtype=np.float64),
        "hdf5_k_edges": np.asarray(data["hdf5_k_edges"], dtype=np.float64),
        "mean": np.mean(matrix, axis=0),
        "stderr": np.std(matrix, axis=0, ddof=1) / math.sqrt(matrix.shape[0]),
        "n_realizations": np.asarray(matrix.shape[0], dtype=np.int64),
        "fiducial_sample": np.asarray(str(fiducial_sample)),
        "fiducial_sample_key": np.asarray(sample_key),
    }


def run_kernel_checks() -> dict[str, float]:
    """基础 kernel sanity check：递归 F2 应等于解析 F2。"""

    rng_vectors = [
        (
            np.asarray([0.01, 0.02, 0.10], dtype=np.float64),
            np.asarray([0.03, -0.04, 0.08], dtype=np.float64),
        ),
        (
            np.asarray([0.04, 0.00, 0.07], dtype=np.float64),
            np.asarray([-0.02, 0.03, 0.05], dtype=np.float64),
        ),
    ]
    diffs = [abs(spt_f(pair) - f2_analytic(pair[0], pair[1])) for pair in rng_vectors]
    return {
        "max_abs_F2_recursive_minus_analytic": float(max(diffs)),
        "kernel_cache_size": float(kernel_sym_cached.cache_info().currsize + kernel_unsym_cached.cache_info().currsize),
    }


def parse_triangle(value: str) -> tuple[float, float, float]:
    """解析 `k1,k2,mu12` 字符串。"""

    parts = [float(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Triangle should be formatted as k1,k2,mu12")
    return parts[0], parts[1], parts[2]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_diag_compare(path: Path, k: np.ndarray, sim: np.ndarray, err: np.ndarray, theory: dict[str, np.ndarray], include_loop: bool) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(k, k * k * sim, yerr=k * k * err, fmt="o", ms=4.0, lw=1.0, color="black", label="Quijote DM pre mean")
    ax.plot(k, k * k * theory["B000_Btree"], marker="s", ms=3.5, lw=1.5, label="SPT tree")
    if include_loop and "B000_Btotal" in theory:
        ax.plot(k, k * k * theory["B000_Btotal"], marker="^", ms=3.5, lw=1.5, label="SPT tree + 1-loop")
        ax.plot(k, k * k * theory["B000_B1loop"], marker=".", ms=3.0, lw=1.0, ls="--", label="SPT 1-loop only")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$k^2 B_{000}(k,k)$")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--z", type=float, default=0.5, help="线性功率谱红移；Quijote DM snapdir_003 为 z=0.5。")
    parser.add_argument("--qmin", type=float, default=1.0e-4, help="1-loop q 积分下限 [h/Mpc]。")
    parser.add_argument("--qmax", type=float, default=30.0, help="1-loop q 积分上限 [h/Mpc]。")
    parser.add_argument("--nq", type=int, default=10, help="log-q Gauss 节点数。")
    parser.add_argument("--q-split", action="append", type=float, default=None, help="可重复；把 log-q 积分按这些 q 值分段。")
    parser.add_argument("--nq-per-segment", type=int, default=0, help="分段时每段 log-q Gauss 节点数；0 表示使用 --nq。")
    parser.add_argument("--nmu-loop", type=int, default=6, help="loop 内部 mu Gauss 节点数。")
    parser.add_argument("--mu-split", action="append", type=float, default=None, help="可重复；把 loop 内部 mu 积分按这些值分段。")
    parser.add_argument("--nmu-per-segment", type=int, default=0, help="分段时每段 mu Gauss 节点数；0 表示使用 --nmu-loop。")
    parser.add_argument("--nphi", type=int, default=6, help="loop 内部 phi Gauss 节点数。")
    parser.add_argument("--phi-split", action="append", type=parse_float_token, default=None, help="可重复；把 phi 积分按这些弧度值分段，支持 pi。")
    parser.add_argument("--nphi-per-segment", type=int, default=0, help="分段时每段 phi Gauss 节点数；0 表示使用 --nphi。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    smoke = sub.add_parser("smoke-triangles", help="计算少量三角形的 tree/1-loop 组件。")
    add_common_args(smoke)
    smoke.add_argument(
        "--triangle",
        action="append",
        type=parse_triangle,
        default=None,
        help="三角形 k1,k2,mu12；可重复。默认跑 0.05/0.10/0.15 的等边。",
    )
    smoke.add_argument("--prefix", default="quijote_dm_spt1loop_smoke_triangles", help="输出文件名前缀。")

    diag = sub.add_parser("compare-diag", help="计算少量对角 B000 并和 Quijote DM pre 均值比较。")
    add_common_args(diag)
    diag.add_argument("--matrix", default=str(DEFAULT_DM_MATRIX), help="Quijote DM 三宇宙学聚合 NPZ。")
    diag.add_argument("--nmu-b000", type=int, default=8, help="B000 外层 mu12 Gauss 节点数。")
    diag.add_argument("--max-diag-bins", type=int, default=4, help="从低 k 开始计算多少个对角 bin。")
    diag.add_argument("--include-loop", action="store_true", help="计算 1-loop；不加时只算 tree，作为快速归一化检查。")
    diag.add_argument(
        "--k-source",
        choices=("weighted", "center"),
        default="weighted",
        help="理论输入使用的 k：weighted 为 HDF5 模式加权有效 k，center 为名义 bin center。",
    )
    diag.add_argument("--prefix", default="quijote_dm_spt1loop_realspace_diag_compare", help="输出文件名前缀。")
    return parser.parse_args()


def loop_config_from_args(args: argparse.Namespace) -> LoopConfig:
    return LoopConfig(
        qmin=float(args.qmin),
        qmax=float(args.qmax),
        nq=int(args.nq),
        nmu=int(args.nmu_loop),
        nphi=int(args.nphi),
        q_splits=tuple(args.q_split or ()),
        nq_per_segment=int(args.nq_per_segment),
        mu_splits=tuple(args.mu_split or ()),
        nmu_per_segment=int(args.nmu_per_segment),
        phi_splits=tuple(args.phi_split or ()),
        nphi_per_segment=int(args.nphi_per_segment),
    )


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = loop_config_from_args(args)
    pk, cosmo_params = build_quijote_linear_power(args.z)
    power_audit = quijote_power_normalization_audit(args.z)
    checks = run_kernel_checks()

    if args.mode == "smoke-triangles":
        triangles = args.triangle
        if triangles is None:
            triangles = [(0.05, 0.05, -0.5), (0.10, 0.10, -0.5), (0.15, 0.15, -0.5)]
        p13_cache: dict[float, float] = {}
        rows = []
        for k1, k2, mu12 in triangles:
            values = bispectrum_total(k1, k2, mu12, pk, config, p13_cache=p13_cache)
            rows.append({"k1": k1, "k2": k2, "mu12": mu12, "k3": vec_norm(triangle_vectors(k1, k2, mu12)[2]), **values})
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "cosmology": cosmo_params,
            "linear_power": pk.description,
            "power_normalization_audit": power_audit,
            "loop_config": asdict(config),
            "kernel_checks": checks,
            "triangles": rows,
            "normalization_note": "P_L is evaluated directly at z; tree scales as P_L(z)^2 and 1-loop as P_L(z)^3.",
        }
        out_json = output_dir / f"{args.prefix}.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.mode == "compare-diag":
        data = load_quijote_dm_pre_mean(Path(args.matrix))
        diag_mask = data["pair_i"] == data["pair_j"]
        diag_indices = np.nonzero(diag_mask)[0][: int(args.max_diag_bins)]
        nominal_k_pairs = np.column_stack((data["k1"], data["k2"]))[diag_indices]
        weighted_k_pairs = data["hdf5_k_weighted"][diag_indices]
        theory_k_pairs = weighted_k_pairs if args.k_source == "weighted" else nominal_k_pairs
        k = np.mean(theory_k_pairs, axis=1)
        sim = data["mean"][diag_indices]
        err = data["stderr"][diag_indices]

        theory_rows = []
        p13_cache: dict[float, float] = {}
        for row, (k1_theory, k2_theory) in zip(diag_indices, theory_k_pairs):
            values = project_b000(
                float(k1_theory),
                float(k2_theory),
                pk,
                config,
                nmu_b000=int(args.nmu_b000),
                include_loop=bool(args.include_loop),
                p13_cache=p13_cache,
            )
            theory_rows.append(values)

        keys = list(theory_rows[0].keys()) if theory_rows else []
        theory = {key: np.asarray([row[key] for row in theory_rows], dtype=np.float64) for key in keys}
        out_npz = output_dir / f"{args.prefix}.npz"
        np.savez(
            out_npz,
            k=k,
            k_source=np.asarray(args.k_source),
            nominal_k_pairs=nominal_k_pairs,
            weighted_k_pairs=weighted_k_pairs,
            theory_k_pairs=theory_k_pairs,
            diag_indices=diag_indices,
            sim_B000=sim,
            sim_B000_stderr=err,
            **theory,
            include_loop=np.asarray(bool(args.include_loop)),
            nmu_b000=np.asarray(int(args.nmu_b000)),
            **{f"loop_{key}": np.asarray(value) for key, value in asdict(config).items()},
        )

        out_pdf = output_dir / f"{args.prefix}.pdf"
        plot_diag_compare(out_pdf, k, sim, err, theory, include_loop=bool(args.include_loop))

        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "outputs": {"npz": str(out_npz), "pdf": str(out_pdf)},
            "matrix": str(Path(args.matrix)),
            "n_realizations": int(data["n_realizations"]),
            "cosmology": cosmo_params,
            "linear_power": pk.description,
            "power_normalization_audit": power_audit,
            "loop_config": asdict(config),
            "nmu_b000": int(args.nmu_b000),
            "include_loop": bool(args.include_loop),
            "p13_cache_size": int(len(p13_cache)),
            "k_source": args.k_source,
            "kernel_checks": checks,
            "diag_indices": diag_indices.tolist(),
            "k": k.tolist(),
            "nominal_k_pairs": nominal_k_pairs.tolist(),
            "weighted_k_pairs": weighted_k_pairs.tolist(),
            "theory_k_pairs": theory_k_pairs.tolist(),
            "sim_B000": sim.tolist(),
            "theory_B000_Btree": theory.get("B000_Btree", np.asarray([])).tolist(),
            "theory_B000_B1loop": theory.get("B000_B1loop", np.asarray([])).tolist(),
            "theory_B000_Btotal": theory.get("B000_Btotal", np.asarray([])).tolist(),
            "normalization_note": "Quijote HDF5 stores raw B000; plot shows k^2 B000 using the same k_source as the theory input. P_L is cosmoprimo linear matter power at z=0.5.",
        }
        out_json = output_dir / f"{args.prefix}.json"
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unhandled mode {args.mode}")


if __name__ == "__main__":
    main()
