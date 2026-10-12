#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
严格三边周期盒功率谱/双谱的 NumPy 参考实现。

代码大纲
========
1. ``build_geometry`` 在完整 FFT lattice 上建立互斥径向 shell，并用 mask 的
   FFT 卷积精确计算每个 ``i <= j <= k`` shell triangle 的 ordered mode count。
2. ``filter_shells`` 把一个实 overdensity 网格分解成各径向 shell 的实空间场。
3. ``measure_power`` 与 ``measure_bispectrum`` 分别测量 ordered tracer-role 的
   shell P 和 B；双谱的三条 Fourier 腿都显式属于冻结 shell，因而是真正的
   all-legs cut。
4. ``bruteforce_*`` 只服务小网格单元测试，直接枚举 Fourier modes，作为 FFT
   卷积实现的独立真值。

Fourier 约定
------------
输入 ``delta_x`` 是周期盒均匀网格上的无量纲 overdensity，NumPy 的
``fftn/ifftn`` 使用默认归一化。令 ``delta_tilde = fftn(delta_x) / Ncell``，则
``delta_tilde = delta(k) / V``。本模块返回通常单位下的

``P = V <delta_tilde delta_tilde>``，
``B = V^2 <delta_tilde delta_tilde delta_tilde>``。

本文件不负责 catalog painting、assignment-window compensation 或 shot-noise
扣除。生产 runner 必须在调用前完成这些步骤，或使用互斥 split 的 cross statistic。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TriangleGeometry:
    """严格三边 shell geometry。

    参数
    ----
    meshsize
        每个笛卡尔方向的网格数；本参考实现只接受立方网格。
    boxsize_mpc_h
        周期盒边长，单位 ``Mpc/h``。
    shell_edges_h_mpc
        一维递增 shell edges。最后一个 upper edge 包含等号，其余 bin 左闭右开。
    shell_index
        每个 FFT mode 所属 shell；``-1`` 表示不属于分析范围。
    triangles
        有非零 exact-lattice count 的 ``(i,j,k)``，并满足 ``i<=j<=k``。
    triangle_counts
        与 ``triangles`` 对齐的 ordered closed-mode triplet 数目。
    k_mean_h_mpc
        每个 triangle 三条 ordered shell 腿的 exact-lattice 平均模长。
    """

    meshsize: int
    boxsize_mpc_h: float
    shell_edges_h_mpc: np.ndarray
    shell_index: np.ndarray
    triangles: np.ndarray
    triangle_counts: np.ndarray
    k_mean_h_mpc: np.ndarray

    @property
    def nshell(self) -> int:
        """返回径向 shell 数目。"""

        return int(self.shell_edges_h_mpc.size - 1)

    @property
    def volume_mpc3_h3(self) -> float:
        """返回周期盒体积，单位 ``(Mpc/h)^3``。"""

        return float(self.boxsize_mpc_h) ** 3


def validate_edges(edges: Iterable[float]) -> np.ndarray:
    """验证并返回 float64 shell edges。

    参数
    ----
    edges
        至少两个有限、严格递增且非负的数。

    返回
    ----
    numpy.ndarray
        连续内存的 float64 一维数组。
    """

    output = np.asarray(tuple(edges), dtype=np.float64)
    if output.ndim != 1 or output.size < 2:
        raise ValueError("shell edges must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(output)):
        raise ValueError("shell edges contain non-finite values")
    if output[0] < 0.0 or not np.all(np.diff(output) > 0.0):
        raise ValueError("shell edges must be non-negative and strictly increasing")
    return np.ascontiguousarray(output)


def fft_wavevectors(meshsize: int, boxsize_mpc_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造完整 FFT lattice 的三个波矢分量和模长。

    参数
    ----
    meshsize
        立方网格每一维的点数。
    boxsize_mpc_h
        周期盒边长，单位 ``Mpc/h``。

    返回
    ----
    tuple
        ``(kx, ky, kz, |k|)``；前三个数组使用 sparse broadcasting，模长为完整
        ``(meshsize, meshsize, meshsize)`` 数组。
    """

    if int(meshsize) != meshsize or int(meshsize) < 2:
        raise ValueError("meshsize must be an integer >= 2")
    if not np.isfinite(boxsize_mpc_h) or float(boxsize_mpc_h) <= 0.0:
        raise ValueError("boxsize_mpc_h must be positive and finite")
    spacing = float(boxsize_mpc_h) / int(meshsize)
    axis = 2.0 * np.pi * np.fft.fftfreq(int(meshsize), d=spacing)
    kx, ky, kz = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, kmag


def assign_shells(kmag: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    """把每个 mode 分配到唯一 shell，分析范围外记为 ``-1``。

    最后一个 bin 对 upper edge 使用浮点容差后的闭区间，从而严格实现预注册的
    ``|k| <= kmax``；零模只有在第一条 edge 恰为零时才可能进入，生产配置从
    ``k_f`` 开始，因此自动排除零模。
    """

    shell_edges = validate_edges(edges)
    values = np.asarray(kmag, dtype=np.float64)
    # FFT wavevectors 与解析 edges 的浮点运算顺序不同。数学上恰落在内部
    # edge（例如 7*kf）的 lattice modes 必须全部进入右侧左闭 bin，不能因
    # Cartesian orientation 的最后一位舍入而分裂到两个 shells。先把机器精度
    # 邻域内的值 canonicalize 到 edge，再应用冻结的左闭右开规则。
    canonical = values.copy()
    for edge in shell_edges[1:-1]:
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(float(edge)))
        canonical[np.abs(canonical - edge) <= tolerance] = edge
    output = np.full(values.shape, -1, dtype=np.int16)
    for index, (lower, upper) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        if index == shell_edges.size - 2:
            tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(float(upper)))
            mask = (canonical >= lower) & (canonical <= upper + tolerance)
        else:
            mask = (canonical >= lower) & (canonical < upper)
        if np.any(output[mask] != -1):
            raise RuntimeError("shell definitions overlap")
        output[mask] = int(index)
    return output


def shell_masks(shell_index: np.ndarray, nshell: int) -> tuple[np.ndarray, ...]:
    """由 shell-index 网格返回一组互斥 float64 Fourier masks。"""

    labels = np.asarray(shell_index)
    if labels.ndim != 3 or len(set(labels.shape)) != 1:
        raise ValueError("shell_index must be a cubic three-dimensional array")
    if int(nshell) < 1:
        raise ValueError("nshell must be positive")
    return tuple(np.asarray(labels == index, dtype=np.float64) for index in range(int(nshell)))


def _closed_triplet_moment(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    """返回三个 Fourier weights 的 closed-triplet sum。

    ``ifftn(mask)`` 的三个乘积在实空间取平均后等于 closed-triplet sum 除以
    ``Ncell^3``。乘回该因子可同时处理纯 mask count 和带 ``|k|`` 的矩。
    """

    if first.shape != second.shape or first.shape != third.shape:
        raise ValueError("all Fourier weights must have identical shapes")
    ncell = int(np.prod(first.shape, dtype=np.int64))
    filtered = [np.fft.ifftn(weight) for weight in (first, second, third)]
    value = np.mean(filtered[0] * filtered[1] * filtered[2]) * float(ncell) ** 3
    if abs(float(np.imag(value))) > 1.0e-7 * max(1.0, abs(float(np.real(value)))):
        raise RuntimeError(f"closed-triplet convolution acquired an imaginary part: {value}")
    return float(np.real(value))


def build_geometry(meshsize: int, boxsize_mpc_h: float, edges: Iterable[float]) -> TriangleGeometry:
    """建立 strict all-legs exact-lattice geometry。

    参数
    ----
    meshsize, boxsize_mpc_h, edges
        网格与 shell 配置，定义同 :class:`TriangleGeometry`。

    返回
    ----
    TriangleGeometry
        仅包含 exact count 大于零的无序 shell triangle；每个 count 本身仍是
        ordered Fourier-mode triplet count，不除以 shell permutation 因子。
    """

    shell_edges = validate_edges(edges)
    _, _, _, kmag = fft_wavevectors(int(meshsize), float(boxsize_mpc_h))
    indices = assign_shells(kmag, shell_edges)
    nshell = int(shell_edges.size - 1)
    # 不同时保留六份 float64 Fourier masks；mesh=256 时这可少占约 0.8 GB。
    # 每个 shell 的普通与 |k|-weighted inverse transform 只做一次。
    real_masks_list: list[np.ndarray] = []
    real_k_masks_list: list[np.ndarray] = []
    for index in range(nshell):
        mask = indices == index
        real_masks_list.append(np.fft.ifftn(mask).real)
        real_k_masks_list.append(np.fft.ifftn(mask * kmag).real)
    real_masks = tuple(real_masks_list)
    real_k_masks = tuple(real_k_masks_list)
    ncell = int(meshsize) ** 3

    triangle_rows: list[tuple[int, int, int]] = []
    count_rows: list[int] = []
    kmean_rows: list[tuple[float, float, float]] = []
    for triangle in combinations_with_replacement(range(nshell), 3):
        i, j, k = triangle
        # einsum 直接累计三数组乘积，避免 ``a*b*c`` 的两个 128 MB 临时数组。
        moment = np.einsum(
            "ijk,ijk,ijk->", real_masks[i], real_masks[j], real_masks[k],
            optimize=False,
        ) / float(ncell)
        count_float = float(moment) * float(ncell) ** 3
        count = int(np.rint(count_float))
        tolerance = 5.0e-7 * max(1.0, abs(count_float))
        if abs(count_float - count) > tolerance:
            raise RuntimeError(
                f"triangle count is not integer within tolerance for {triangle}: {count_float}"
            )
        if count <= 0:
            continue

        means: list[float] = []
        shell_ids = (i, j, k)
        for leg in range(3):
            real_weights = [real_masks[shell] for shell in shell_ids]
            real_weights[leg] = real_k_masks[shell_ids[leg]]
            weighted_sum = (
                float(np.einsum(
                    "ijk,ijk,ijk->",
                    real_weights[0],real_weights[1],real_weights[2],
                    optimize=False,
                ))
                /float(ncell)*float(ncell)**3
            )
            means.append(weighted_sum / float(count))
        triangle_rows.append(triangle)
        count_rows.append(count)
        kmean_rows.append(tuple(means))

    return TriangleGeometry(
        meshsize=int(meshsize),
        boxsize_mpc_h=float(boxsize_mpc_h),
        shell_edges_h_mpc=shell_edges,
        shell_index=indices,
        triangles=np.asarray(triangle_rows, dtype=np.int16).reshape(-1, 3),
        triangle_counts=np.asarray(count_rows, dtype=np.int64),
        k_mean_h_mpc=np.asarray(kmean_rows, dtype=np.float64).reshape(-1, 3),
    )


def validate_real_grid(delta_x: np.ndarray, geometry: TriangleGeometry, label: str) -> np.ndarray:
    """验证一个实、有限、立方 overdensity 网格。"""

    values = np.asarray(delta_x)
    expected = (geometry.meshsize,) * 3
    if values.shape != expected:
        raise ValueError(f"{label} grid has shape {values.shape}; expected {expected}")
    if np.iscomplexobj(values):
        raise ValueError(f"{label} grid must be real")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} grid contains non-finite values")
    return np.asarray(values, dtype=np.float64)


def filter_shells(delta_x: np.ndarray, geometry: TriangleGeometry) -> np.ndarray:
    """把 overdensity 网格分解成所有冻结 Fourier shells。

    返回数组形状为 ``(nshell, meshsize, meshsize, meshsize)``。这是清晰优先的
    reference path；生产 JAX path 可以分批计算，但必须逐项复现本函数结果。
    """

    values = validate_real_grid(delta_x, geometry, "delta_x")
    # 即使上游已经提供 overdensity，也在这里再次移除离散网格均值。这样既与
    # production jaxpower 的 ``mesh - mesh.mean()`` 约定一致，也防止常数场的
    # 非 2 次幂 FFT 在非零 mode 留下机器精度舍入残差。
    values = values - np.mean(values, dtype=np.float64)
    fourier = np.fft.fftn(values)
    output = np.empty((geometry.nshell,) + values.shape, dtype=np.float64)
    for index in range(geometry.nshell):
        filtered = np.fft.ifftn(fourier * (geometry.shell_index == index))
        imaginary = float(np.max(np.abs(filtered.imag)))
        scale = max(1.0, float(np.max(np.abs(filtered.real))))
        if imaginary > 5.0e-12 * scale:
            raise RuntimeError(f"shell {index} violates Hermitian reality: {imaginary}")
        output[index] = filtered.real
    return output


def measure_power(
    filtered_first: np.ndarray,
    filtered_second: np.ndarray,
    geometry: TriangleGeometry,
) -> np.ndarray:
    """测量两个 ordered tracer fields 的 shell cross-power。

    输入必须是 :func:`filter_shells` 的输出；返回长度 ``nshell``、单位
    ``(Mpc/h)^3``。本函数不扣除 auto shot noise。
    """

    expected = (geometry.nshell, geometry.meshsize, geometry.meshsize, geometry.meshsize)
    first = np.asarray(filtered_first, dtype=np.float64)
    second = np.asarray(filtered_second, dtype=np.float64)
    if first.shape != expected or second.shape != expected:
        raise ValueError(f"filtered field shape mismatch; expected {expected}")
    counts = np.asarray(
        [np.count_nonzero(geometry.shell_index == index) for index in range(geometry.nshell)],
        dtype=np.int64,
    )
    if np.any(counts <= 0):
        raise RuntimeError("at least one shell has no Fourier modes")
    values = np.asarray(
        [
            np.einsum(
                "ijk,ijk->", first[index], second[index], optimize=False
            ) / float(geometry.meshsize**3)
            for index in range(geometry.nshell)
        ],
        dtype=np.float64,
    )
    return geometry.volume_mpc3_h3 * values / counts


def measure_factorial_power_fourier(
    first_fourier: np.ndarray,
    second_fourier: np.ndarray,
    geometry: TriangleGeometry,
    *,
    pair_common_object_zero_mode: complex | float | None = None,
) -> dict[str, np.ndarray]:
    """测量 two-field shell power 并可扣除共同对象的 exact self pair。

    Fourier inputs 使用 ``delta(k)/V`` normalization。若两 fields 共享对象，传入
    ``D2_ab(0)=sum_object u_a u_b``；每个 shell 的 self-pair correction 都是
    ``V*D2_ab(0)``。返回 ``raw``、``pair`` 与 ``factorial=raw-pair``。
    """

    first = validate_normalized_fourier(first_fourier, geometry, "first_fourier")
    second = validate_normalized_fourier(second_fourier, geometry, "second_fourier")
    first_filtered = filter_normalized_fourier(first, geometry)
    second_filtered = (
        first_filtered
        if second is first
        else filter_normalized_fourier(second, geometry)
    )
    raw = measure_power(first_filtered, second_filtered, geometry)
    pair = np.zeros_like(raw)
    if pair_common_object_zero_mode is not None:
        scalar = complex(pair_common_object_zero_mode)
        if abs(scalar.imag) > 2.0e-12 * max(1.0, abs(scalar.real)):
            raise ValueError("power pair common-object zero mode must be real")
        pair.fill(geometry.volume_mpc3_h3 * float(scalar.real))
    return {"raw": raw, "pair": pair, "factorial": raw - pair}


def measure_bispectrum(
    filtered_first: np.ndarray,
    filtered_second: np.ndarray,
    filtered_third: np.ndarray,
    geometry: TriangleGeometry,
) -> np.ndarray:
    """测量 ordered tracer-role strict shell bispectrum。

    参数
    ----
    filtered_first, filtered_second, filtered_third
        三个 tracer fields 的 shell-filtered 实空间数组。tracer 顺序不会被内部
        对称化；例如 ``(h,m,m)``、``(m,h,m)`` 会分别保留。
    geometry
        与 fields 完全一致的 exact-lattice geometry。

    返回
    ----
    numpy.ndarray
        与 ``geometry.triangles`` 对齐，单位 ``(Mpc/h)^6``。
    """

    expected = (geometry.nshell, geometry.meshsize, geometry.meshsize, geometry.meshsize)
    fields = tuple(
        np.asarray(field, dtype=np.float64)
        for field in (filtered_first, filtered_second, filtered_third)
    )
    if any(field.shape != expected for field in fields):
        raise ValueError(f"filtered field shape mismatch; expected {expected}")
    values = np.empty(geometry.triangles.shape[0], dtype=np.float64)
    for row, (triangle, count) in enumerate(zip(geometry.triangles, geometry.triangle_counts)):
        i, j, k = (int(value) for value in triangle)
        numerator = np.einsum(
            "ijk,ijk,ijk->",
            fields[0][i],fields[1][j],fields[2][k],
            optimize=False,
        ) / float(geometry.meshsize**3)
        values[row] = geometry.volume_mpc3_h3**2 * float(numerator) / float(count)
    return values


def measure_named_roles(
    filtered_fields: Mapping[str, np.ndarray],
    roles: Iterable[tuple[str, str, str]],
    geometry: TriangleGeometry,
) -> dict[str, np.ndarray]:
    """一次测量多组 ordered tracer roles，并用 ``a__b__c`` 作为稳定 key。"""

    output: dict[str, np.ndarray] = {}
    for role in roles:
        if len(role) != 3:
            raise ValueError(f"bispectrum role must have three entries: {role}")
        missing = [name for name in role if name not in filtered_fields]
        if missing:
            raise KeyError(f"missing filtered fields for role {role}: {missing}")
        key = "__".join(role)
        if key in output:
            raise ValueError(f"duplicate role key: {key}")
        output[key] = measure_bispectrum(
            filtered_fields[role[0]],
            filtered_fields[role[1]],
            filtered_fields[role[2]],
            geometry,
        )
    return output


def validate_normalized_fourier(coefficients: np.ndarray, geometry: TriangleGeometry, label: str) -> np.ndarray:
    """验证一个以 ``delta(k)/V`` 归一化的完整 complex Fourier 网格。"""

    values = np.asarray(coefficients, dtype=np.complex128)
    expected = (geometry.meshsize,) * 3
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} Fourier grid must be finite with shape {expected}")
    negative = (-np.arange(geometry.meshsize, dtype=np.int64)) % geometry.meshsize
    reversed_conjugate = np.conjugate(values[np.ix_(negative, negative, negative)])
    residual = float(np.max(np.abs(values - reversed_conjugate)))
    scale = max(1.0, float(np.max(np.abs(values))))
    if residual > 2.0e-11 * scale:
        raise ValueError(f"{label} Fourier grid is not Hermitian: residual={residual}")
    return values


def filter_normalized_fourier(coefficients: np.ndarray, geometry: TriangleGeometry) -> np.ndarray:
    """把 ``delta(k)/V`` 系数转换为各 shell 的实空间 filtered fields。

    NumPy ``ifftn`` 自带 ``1/Ncell``，因此先乘 ``Ncell``，使返回 field 等于
    ``sum_k [delta(k)/V] exp(ikx)``。该输出可直接交给 :func:`measure_bispectrum`。
    """

    values = validate_normalized_fourier(coefficients, geometry, "coefficients")
    ncell = int(geometry.meshsize) ** 3
    output = np.empty((geometry.nshell,) + values.shape, dtype=np.float64)
    for index in range(geometry.nshell):
        filtered = np.fft.ifftn(
            values * float(ncell) * (geometry.shell_index == index)
        )
        imaginary = float(np.max(np.abs(filtered.imag)))
        scale = max(1.0, float(np.max(np.abs(filtered.real))))
        if imaginary > 5.0e-12 * scale:
            raise RuntimeError(f"normalized shell {index} violates Hermitian reality")
        output[index] = filtered.real
    return output


def reverse_fourier_indices(values: np.ndarray) -> np.ndarray:
    """返回 ``values(-k)``，使用 FFT modulo indices，不做 complex conjugation。"""

    array = np.asarray(values)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError("reverse_fourier_indices requires a cubic 3D array")
    negative = (-np.arange(array.shape[0], dtype=np.int64)) % array.shape[0]
    return array[np.ix_(negative, negative, negative)]


def pair_mode_multiplicity(
    first_shell: int,
    second_shell: int,
    geometry: TriangleGeometry,
) -> np.ndarray:
    """返回 ``C_ij(q)=sum_p 1_i(p)1_j(q-p)`` 的 exact integer 网格。

    该 convolution 是 factorial shot correction 的几何权重。返回 int64 数组，
    shell 顺序交换时严格相同。
    """

    i = int(first_shell)
    j = int(second_shell)
    if not 0 <= i < geometry.nshell or not 0 <= j < geometry.nshell:
        raise IndexError("shell index lies outside geometry")
    first = np.asarray(geometry.shell_index == i, dtype=np.float64)
    second = np.asarray(geometry.shell_index == j, dtype=np.float64)
    ncell = int(geometry.meshsize) ** 3
    convolution = np.fft.fftn(np.fft.ifftn(first) * np.fft.ifftn(second)) * float(ncell)
    imaginary = float(np.max(np.abs(convolution.imag)))
    if imaginary > 5.0e-8:
        raise RuntimeError(f"pair multiplicity acquired imaginary part {imaginary}")
    rounded64 = np.rint(convolution.real).astype(np.int64)
    if float(np.max(np.abs(convolution.real - rounded64))) > 5.0e-7:
        raise RuntimeError("pair multiplicity is not integer within tolerance")
    if np.any(rounded64 < 0):
        raise RuntimeError("pair multiplicity contains negative counts")
    if int(np.max(rounded64)) > np.iinfo(np.int32).max:
        raise RuntimeError("pair multiplicity exceeds int32 storage")
    return rounded64.astype(np.int32)


def build_reversed_pair_multiplicity_cache(
    geometry: TriangleGeometry,
) -> dict[tuple[int, int], np.ndarray]:
    """预计算全部 ``i<=j`` shell pairs 的 ``C_ij(-q)`` integer grids。

    cross-statistics 会重复测量多种 tracer roles；显式 workspace 可让它们共享最多
    21 次 geometry FFT。mesh=256 时该缓存约 1.4 GB，调用者应在一个 realization
    完成后及时释放。
    """

    output: dict[tuple[int, int], np.ndarray] = {}
    for first in range(geometry.nshell):
        for second in range(first, geometry.nshell):
            output[(first, second)] = reverse_fourier_indices(
                pair_mode_multiplicity(first, second, geometry)
            )
    return output


def measure_factorial_bispectrum_prepared(
    filtered_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    fourier_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    geometry: TriangleGeometry,
    *,
    pair_common_object_fields: Mapping[tuple[int, int], np.ndarray] | None = None,
    triple_common_object_zero_mode: complex | float | None = None,
    reversed_pair_multiplicity_cache: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """用已过滤 fields/workspace 测量 factorial B，供多 tracer 批处理复用。

    参数语义与 :func:`measure_factorial_bispectrum_fourier` 相同；新增的
    ``filtered_fields`` 必须与三个 Fourier fields 一一对应。若提供
    ``reversed_pair_multiplicity_cache``，其中数组必须是 ``C_ij(-q)``。
    """

    if len(filtered_fields) != 3 or len(fourier_fields) != 3:
        raise ValueError("prepared factorial estimator requires exactly three legs")
    normalized = tuple(
        validate_normalized_fourier(field, geometry, f"fourier_fields[{index}]")
        for index, field in enumerate(fourier_fields)
    )
    expected = (geometry.nshell, geometry.meshsize, geometry.meshsize, geometry.meshsize)
    filtered = tuple(np.asarray(field, dtype=np.float64) for field in filtered_fields)
    if any(field.shape != expected for field in filtered):
        raise ValueError(f"prepared filtered field shape mismatch; expected {expected}")
    raw = measure_bispectrum(filtered[0], filtered[1], filtered[2], geometry)
    output: dict[str, np.ndarray] = {"raw": raw}
    factorial = raw.copy()
    pair_fields = dict(pair_common_object_fields or {})
    valid_pairs = {(0, 1), (0, 2), (1, 2)}
    if any(tuple(pair) not in valid_pairs for pair in pair_fields):
        raise ValueError("pair_common_object_fields contains an invalid leg pair")
    cache = dict(reversed_pair_multiplicity_cache or {})
    for pair, second_moment in sorted(pair_fields.items()):
        pair = tuple(pair)
        remaining = ({0, 1, 2} - set(pair)).pop()
        d2 = validate_normalized_fourier(second_moment, geometry, f"D2_pair_{pair}")
        d2_negative = reverse_fourier_indices(d2)
        pair_component = np.zeros_like(raw)
        for row, (triangle, count) in enumerate(zip(geometry.triangles, geometry.triangle_counts)):
            pair_shells = tuple(sorted((int(triangle[pair[0]]), int(triangle[pair[1]]))))
            if pair_shells not in cache:
                cache[pair_shells] = reverse_fourier_indices(
                    pair_mode_multiplicity(pair_shells[0], pair_shells[1], geometry)
                )
            multiplicity_negative = cache[pair_shells]
            if multiplicity_negative.shape != geometry.shell_index.shape:
                raise ValueError(f"bad pair multiplicity cache shape for {pair_shells}")
            remaining_mask = geometry.shell_index == int(triangle[remaining])
            summed = np.sum(
                multiplicity_negative[remaining_mask]
                * d2_negative[remaining_mask]
                * normalized[remaining][remaining_mask]
            )
            if abs(float(np.imag(summed))) > 2.0e-10 * max(1.0, abs(float(np.real(summed)))):
                raise RuntimeError(f"pair contraction {pair} is unexpectedly complex")
            pair_component[row] = (
                geometry.volume_mpc3_h3**2
                * float(np.real(summed)) / float(count)
            )
        key = f"pair_{pair[0]}{pair[1]}"
        output[key] = pair_component
        factorial -= pair_component

    triple = np.zeros_like(raw)
    if triple_common_object_zero_mode is not None:
        scalar = complex(triple_common_object_zero_mode)
        if abs(scalar.imag) > 2.0e-12 * max(1.0, abs(scalar.real)):
            raise ValueError("triple common-object zero mode must be real")
        triple.fill(2.0 * geometry.volume_mpc3_h3**2 * float(scalar.real))
        factorial += triple
    output["triple"] = triple
    output["factorial"] = factorial
    return output


def measure_factorial_bispectrum_fourier(
    fourier_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    geometry: TriangleGeometry,
    *,
    pair_common_object_fields: Mapping[tuple[int, int], np.ndarray] | None = None,
    triple_common_object_zero_mode: complex | float | None = None,
) -> dict[str, np.ndarray]:
    """测量带 exact self-pair subtraction 的 ordered factorial bispectrum。

    参数
    ----
    fourier_fields
        三条 tracer legs 的 ``D1_a(k)=delta_a(k)/V`` 完整 Fourier arrays。
    geometry
        strict-three-side exact-lattice geometry。
    pair_common_object_fields
        若两条 legs 共享离散对象，key 为 leg pair ``(0,1)``、``(0,2)`` 或
        ``(1,2)``，value 为
        ``D2_ab(k)=sum_object u_a u_b exp(-ikx)``。不共享对象的 pair 不提供。
        对同一个带符号 F=D-alpha R field，三项都传同一个 ``sum u^2`` field。
    triple_common_object_zero_mode
        三条 legs 共享对象时的 ``D3_abc(0)=sum u_a u_b u_c``；否则省略。

    返回
    ----
    dict
        ``raw``、每个 ``pair_01/pair_02/pair_12`` 被减分量、``triple`` 被加分量
        以及 ``factorial``。全部单位为 ``(Mpc/h)^6``。

    说明
    ----
    对同一 field，inclusion--exclusion 为
    ``distinct = raw - A12 - A13 - A23 + 2*A123``。该恒等式对任意 signed/weighted
    catalog 成立，因而也覆盖 displaced data 加 shifted random 的 post field；
    它不假定 Poisson abundance，也不把 residual stochastic physics 扣掉。
    """

    if len(fourier_fields) != 3:
        raise ValueError("fourier_fields must contain exactly three legs")
    normalized = tuple(
        validate_normalized_fourier(field, geometry, f"fourier_fields[{index}]")
        for index, field in enumerate(fourier_fields)
    )
    filtered_cache: dict[int, np.ndarray] = {}
    filtered_rows: list[np.ndarray] = []
    for field in normalized:
        identity = id(field)
        if identity not in filtered_cache:
            filtered_cache[identity] = filter_normalized_fourier(field, geometry)
        filtered_rows.append(filtered_cache[identity])
    filtered = tuple(filtered_rows)
    return measure_factorial_bispectrum_prepared(
        filtered,
        normalized,
        geometry,
        pair_common_object_fields=pair_common_object_fields,
        triple_common_object_zero_mode=triple_common_object_zero_mode,
    )


def measure_sugiyama_monopole_factorial_fourier(
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    third_moment_zero: complex | float,
    geometry: TriangleGeometry,
) -> dict[str, np.ndarray]:
    """Exact unrestricted-third-leg Sugiyama auto monopole from D1/D2/D3.

    The first two Fourier legs use the frozen radial shells, while the closing
    third leg is unrestricted over the complete FFT mesh.  The normalization
    is the product of the two shell mode counts, exactly matching the Sugiyama
    basis rather than the conditional closed-triplet normalization used by the
    strict three-side observable.
    """

    d1 = validate_normalized_fourier(first_moment, geometry, "Sugiyama_D1")
    d2 = validate_normalized_fourier(second_moment, geometry, "Sugiyama_D2")
    scalar = complex(third_moment_zero)
    if abs(scalar.imag) > 2.0e-12 * max(1.0, abs(scalar.real)):
        raise ValueError("Sugiyama D3 zero mode must be real")
    filtered = filter_normalized_fourier(d1, geometry)
    ncell = int(geometry.meshsize) ** 3
    full = np.fft.ifftn(d1 * float(ncell))
    imaginary = float(np.max(np.abs(full.imag)))
    if imaginary > 5.0e-12 * max(1.0, float(np.max(np.abs(full.real)))):
        raise RuntimeError("Sugiyama unrestricted field is not real")
    full = full.real
    counts = np.asarray([
        np.count_nonzero(geometry.shell_index == shell)
        for shell in range(geometry.nshell)
    ], dtype=np.int64)
    pairs = np.asarray([
        (first, second)
        for first in range(geometry.nshell)
        for second in range(first, geometry.nshell)
    ], dtype=np.int16)
    names = ("raw", "pair_01", "pair_02", "pair_12", "triple", "factorial")
    output = {
        "shell_pairs": pairs,
        "mode_count_product": np.asarray(
            [counts[int(i)] * counts[int(j)] for i, j in pairs], dtype=np.int64
        ),
        **{name: np.empty(len(pairs), dtype=np.float64) for name in names},
    }
    d2_negative = reverse_fourier_indices(d2)
    volume_squared = geometry.volume_mpc3_h3**2
    for row, (first_raw, second_raw) in enumerate(pairs):
        first = int(first_raw)
        second = int(second_raw)
        denominator = float(counts[first] * counts[second])
        raw = volume_squared * float(
            np.einsum(
                "ijk,ijk,ijk->",
                filtered[first], filtered[second], full,
                optimize=False,
            ) / float(ncell)
        ) / denominator
        multiplicity_negative = reverse_fourier_indices(
            pair_mode_multiplicity(first, second, geometry)
        )
        pair_01_complex = np.sum(multiplicity_negative * d2_negative * d1)
        second_mask = geometry.shell_index == second
        first_mask = geometry.shell_index == first
        pair_02_complex = counts[first] * np.sum(
            d2_negative[second_mask] * d1[second_mask]
        )
        pair_12_complex = counts[second] * np.sum(
            d2_negative[first_mask] * d1[first_mask]
        )
        pair_complex = (pair_01_complex, pair_02_complex, pair_12_complex)
        if any(
            abs(float(np.imag(value)))
            > 2.0e-10 * max(1.0, abs(float(np.real(value))))
            for value in pair_complex
        ):
            raise RuntimeError("Sugiyama pair contraction is unexpectedly complex")
        pair_values = [
            volume_squared * float(np.real(value)) / denominator
            for value in pair_complex
        ]
        triple = 2.0 * volume_squared * float(scalar.real)
        output["raw"][row] = raw
        output["pair_01"][row] = pair_values[0]
        output["pair_02"][row] = pair_values[1]
        output["pair_12"][row] = pair_values[2]
        output["triple"][row] = triple
        output["factorial"][row] = raw - sum(pair_values) + triple
    return output


def _mode_rows_by_shell(geometry: TriangleGeometry) -> tuple[np.ndarray, ...]:
    """返回小网格 brute-force 使用的整数 FFT index rows。"""

    return tuple(np.argwhere(geometry.shell_index == index) for index in range(geometry.nshell))


def bruteforce_triangle_counts(geometry: TriangleGeometry) -> np.ndarray:
    """直接枚举小网格的 closed Fourier triplets，返回 exact counts。

    复杂度随低 k mode 数的平方增长，只允许用于单元测试；为防误用，meshsize 大于
    24 时直接报错。
    """

    if geometry.meshsize > 24:
        raise ValueError("bruteforce_triangle_counts is restricted to meshsize <= 24")
    mode_rows = _mode_rows_by_shell(geometry)
    size = int(geometry.meshsize)
    output = np.zeros(geometry.triangles.shape[0], dtype=np.int64)
    for row, (i, j, k) in enumerate(geometry.triangles):
        third_modes = {tuple(mode) for mode in mode_rows[int(k)].tolist()}
        count = 0
        for first in mode_rows[int(i)]:
            for second in mode_rows[int(j)]:
                third = tuple((-(first + second)) % size)
                count += int(third in third_modes)
        output[row] = count
    return output


def bruteforce_bispectrum(
    delta_first: np.ndarray,
    delta_second: np.ndarray,
    delta_third: np.ndarray,
    geometry: TriangleGeometry,
) -> np.ndarray:
    """直接枚举小网格 Fourier triplets 的 ordered bispectrum 真值。"""

    if geometry.meshsize > 24:
        raise ValueError("bruteforce_bispectrum is restricted to meshsize <= 24")
    fields = tuple(
        np.fft.fftn(validate_real_grid(field, geometry, label))
        / float(geometry.meshsize**3)
        for field, label in zip(
            (delta_first, delta_second, delta_third),
            ("delta_first", "delta_second", "delta_third"),
        )
    )
    mode_rows = _mode_rows_by_shell(geometry)
    size = int(geometry.meshsize)
    output = np.empty(geometry.triangles.shape[0], dtype=np.float64)
    for row, ((i, j, k), count) in enumerate(zip(geometry.triangles, geometry.triangle_counts)):
        third_modes = {tuple(mode) for mode in mode_rows[int(k)].tolist()}
        total = 0.0j
        for first in mode_rows[int(i)]:
            first_key = tuple(first)
            for second in mode_rows[int(j)]:
                second_key = tuple(second)
                third_key = tuple((-(first + second)) % size)
                if third_key in third_modes:
                    total += fields[0][first_key] * fields[1][second_key] * fields[2][third_key]
        value = geometry.volume_mpc3_h3**2 * total / float(count)
        if abs(float(value.imag)) > 5.0e-11 * max(1.0, abs(float(value.real))):
            raise RuntimeError(f"brute-force bispectrum is unexpectedly complex: {value}")
        output[row] = float(value.real)
    return output


def preregistered_eq_edges(boxsize_mpc_h: float = 1000.0) -> np.ndarray:
    """返回 M0 冻结的 EQ strict-``kmax=0.08`` shell edges。"""

    if not np.isclose(float(boxsize_mpc_h), 1000.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("the preregistered edges are frozen for a 1000 Mpc/h box")
    fundamental = 2.0 * np.pi / float(boxsize_mpc_h)
    return np.asarray(
        [fundamental, 3 * fundamental, 5 * fundamental, 7 * fundamental,
         9 * fundamental, 11 * fundamental, 0.08],
        dtype=np.float64,
    )


__all__ = [
    "TriangleGeometry",
    "assign_shells",
    "build_reversed_pair_multiplicity_cache",
    "bruteforce_bispectrum",
    "bruteforce_triangle_counts",
    "build_geometry",
    "fft_wavevectors",
    "filter_shells",
    "measure_bispectrum",
    "measure_factorial_bispectrum_fourier",
    "measure_factorial_bispectrum_prepared",
    "measure_factorial_power_fourier",
    "measure_named_roles",
    "measure_power",
    "measure_sugiyama_monopole_factorial_fourier",
    "pair_mode_multiplicity",
    "preregistered_eq_edges",
    "reverse_fourier_indices",
    "shell_masks",
    "filter_normalized_fourier",
    "validate_normalized_fourier",
    "validate_edges",
]
