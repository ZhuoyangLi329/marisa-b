#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""STO 加权方案变体 runner（STO_WEIGHTING_SCAN_PLAN Stage 0 产物）。

派生来源与保证
==============
本文件逐行派生自经过验证的归档实现
``the validated N plus environment implementation``
（2026-09-05 拷贝，基线 sha256 见 GATE-0 审计 JSON）。除下列六处外科手术式
修改外，其余代码与归档验证版完全一致；对 ``method=nenv`` 的调用必须通过
GATE-0 回归（weights/env_labels/post_sto_halo 与既有生产逐元素一致）：

1. 文件头 docstring（本段）；
2. ``parse_args``：``--method`` choices 追加 ``monly``；
3. ``resolve_component_config``：新增 ``monly`` 分支（mass-only，跳过环境
   维度；``mass_bins`` 由参数给出，``base_weight`` 放开为 number/mass/rankmass）；
4. ``build_component_labels``：新增 ``env_enabled`` 关键字；``monly`` 时
   component = mass bin 本身，命名为 M1..Mk；
5. ``run``：``monly`` 时跳过 ``tidal_env_labels``（省 FFT；标签置 0，
   ``env_info`` 记录禁用），``build_component_labels`` 传
   ``env_enabled=False``；
6. ``run``：``monly`` 时 solution 汇总不做 ``(mass_bins,4,density_bins)``
   reshape（分量数不等于 mass_bins*4*density_bins）。

科学内容（与基线一致，仅补充 monly 语义）
==========================================
1. 把 halo catalog paint 到周期 CIC 网格；环境维度用潮汐张量 Hessian 分类
   （void/sheet/filament/knot）；
2. 分量拆分：
   - nenv: N+Env，4 个环境分量，number base weight（生产标准）；
   - monly: **mass-only**，k 个等数量 mass bin 分量，本计划新增；
   - m2env/m4env/m4rho3env: mass x env (x 局部密度) 组合，基线已有；
3. 权重求解：``weight_source=dm`` 时用同 realization DM 平滑场当 teacher
   （解 ``C w = u``，等价最大化与 DM 的 cross-correlation）；
   ``weight_source=halo-split`` 时用 split-half 自监督；
4. 输出 halo 级标签、最终权重与 diagnostics。

原归档验证文件与本文件的关系：归档文件零改动；本文件是 STO_WEIGHTING_SCAN_PLAN
（analysis/quijote_halo_z0p5_stovar_scan_v1/STO_WEIGHTING_SCAN_PLAN.md）
Stage 0.1 的唯一新科学代码。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:  # Quijote snapshots on Perlmutter may require external HDF5 filters.
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover - the filters are environment dependent.
    hdf5plugin = None  # type: ignore[assignment]


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from entrypoint import validate_output_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HALO_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
DEFAULT_DM_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/dm")
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "01_reconstruction_methods/outputs/sto_recon"

BOXSIZE_MPC_H = 1000.0
COORD_UNIT_SCALE_TO_MPC_H = 1.0 / 1000.0
ENV_NAMES = ("void", "sheet", "filament", "knot")


@dataclass(frozen=True)
class StoConfig:
    env_source: str
    env_field_weight: str
    weight_source: str
    split_seed: int
    env_mesh: int
    recon_mesh: int
    gsr_mpc_h: float
    thr_mpc_h: float
    lambda_th: float
    env_sample_shift_cells: int
    stochastic_kmin: float
    stochastic_kmax: float
    stochastic_nbins: int
    covariance_regularization: float


@dataclass(frozen=True)
class ComponentConfig:
    """sto-recon 的 catalog split / base-weight 配置。"""

    method: str
    mass_bins: int
    density_bins: int
    density_radius_mpc_h: float
    base_weight: str
    mass_filename: str
    mass_weight_normalization: str
    rankmass_strength: float
    min_component_count: int
    solution_shrinkage: float
    weight_factor_clip_min: float | None
    weight_factor_clip_max: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quijote halo-only N+Env/M-2+Env sto-recon for one realization.")
    parser.add_argument("--cosmology", default="fiducial", help="Quijote cosmology, e.g. fiducial or LC_p.")
    parser.add_argument("--realization", type=int, default=0)
    parser.add_argument("--halo-root", default=str(DEFAULT_HALO_ROOT))
    parser.add_argument("--mass-root", default=None, help="halo mass 根目录；默认与 --halo-root 相同。")
    parser.add_argument("--mass-filename", default="masses_msun_h.npy", help="每个 realization 中的 mass 文件名。")
    parser.add_argument("--dm-root", default=str(DEFAULT_DM_ROOT), help="仅 legacy --env-source/--weight-source=dm 时使用。")
    parser.add_argument("--snapshot-name", default="snapdir_003")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--manifest", default=None, help="Optional JSONL manifest.")
    parser.add_argument("--task-index", type=int, default=None, help="0-based row index in --manifest.")
    parser.add_argument("--label", default=None, help="Optional extra output subdirectory.")
    parser.add_argument(
        "--method",
        choices=["nenv", "monly", "m2env", "m4env", "m4rho3env"],
        default="m2env",
        help=(
            "sto-recon 方法：nenv 是旧 N+Env（生产标准）；monly 是 mass-only"
            "（本计划新增，component=等数量 mass bin，不含环境维度）；"
            "m2env 是 M-2+Env；m4env 使用 4 个质量 bin；m4rho3env 额外按"
            " halo-only 局部密度分 bin。"
        ),
    )
    parser.add_argument("--mass-bins", type=int, default=None, help="等数量 halo mass bin 数；默认由 --method 决定。")
    parser.add_argument(
        "--density-bins",
        type=int,
        default=None,
        help="halo-only 局部密度等数量 bin 数；m4rho3env 默认 3，其它方法默认 1。",
    )
    parser.add_argument(
        "--density-radius",
        type=float,
        default=15.0,
        help="计算 halo-only 局部密度标签时的 top-hat 平滑半径 [Mpc/h]。",
    )
    parser.add_argument(
        "--base-weight",
        choices=["number", "mass", "rankmass"],
        default="number",
        help="component field 的 halo 基础权重；rankmass 是有界 mass-rank 权重。",
    )
    parser.add_argument(
        "--mass-weight-normalization",
        choices=["mean", "none"],
        default="mean",
        help="base-weight=mass 时是否把 mass 除以全样本平均 mass，以改善数值稳定性。",
    )
    parser.add_argument(
        "--rankmass-strength",
        type=float,
        default=1.0,
        help="base-weight=rankmass 时的质量秩权重强度；1.2 对应约 [0.4, 1.6]。",
    )
    parser.add_argument(
        "--env-source",
        choices=["halo", "dm"],
        default="halo",
        help="Env Hessian 的输入场；当前统一口径使用 halo。",
    )
    parser.add_argument(
        "--env-field-weight",
        choices=["number", "base"],
        default="number",
        help="env-source=halo 时用于画 Hessian 输入场的 halo 权重；number 更接近 galaxy-only 观测口径。",
    )
    parser.add_argument(
        "--weight-source",
        choices=["halo-split", "dm"],
        default="halo-split",
        help="component 权重求解目标；当前统一口径使用 halo-split 自监督，不读 DM。",
    )
    parser.add_argument("--split-seed", type=int, default=20260515, help="halo-split 权重求解的确定性随机种子。")

    parser.add_argument("--env-mesh", type=int, default=128)
    parser.add_argument("--recon-mesh", type=int, default=128)
    parser.add_argument("--gsr", type=float, default=1.0, help="Gaussian smoothing radius for env labels [Mpc/h].")
    parser.add_argument("--thr", type=float, default=15.0, help="Top-hat smoothing radius for weights [Mpc/h].")
    parser.add_argument("--lambda-th", type=float, default=11.25)
    parser.add_argument(
        "--env-sample-shift-cells",
        type=int,
        default=0,
        help=(
            "Integer grid-cell offset used when sampling tidal tensors at halo positions. "
            "The default 0 follows the pmesh CIC grid convention; -1 can be used as an MRACS-style check."
        ),
    )
    parser.add_argument("--stochastic-kmin", type=float, default=0.02)
    parser.add_argument("--stochastic-kmax", type=float, default=0.20)
    parser.add_argument("--stochastic-nbins", type=int, default=18)
    parser.add_argument("--covariance-regularization", type=float, default=1.0e-8)
    parser.add_argument(
        "--min-component-count",
        type=int,
        default=2,
        help="split-half 求解中每个 half 至少需要的 component halo 数；不足的 component 保持 base 权重。",
    )
    parser.add_argument(
        "--solution-shrinkage",
        type=float,
        default=0.0,
        help="把求解出的 component 份额向原始 base 份额收缩的比例，0 表示不收缩。",
    )
    parser.add_argument(
        "--weight-factor-clip-min",
        type=float,
        default=None,
        help="可选：clip 每个 component 的 per-halo 权重因子下限，clip 后重新归一化。",
    )
    parser.add_argument(
        "--weight-factor-clip-max",
        type=float,
        default=None,
        help="可选：clip 每个 component 的 per-halo 权重因子上限，clip 后重新归一化。",
    )

    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--max-dm-particles", type=int, default=None, help="仅 legacy DM 分支 smoke test 使用。")
    parser.add_argument("--dtype", choices=["f4", "f8"], default="f4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果完整输出已经存在，直接成功返回；用于补齐 manifest 时复用已完成 realization。",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if (args.manifest is None) != (args.task_index is None):
        parser.error("--manifest and --task-index must be provided together")
    return args


def load_manifest_task(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.manifest is None:
        return None
    manifest = Path(args.manifest).expanduser().resolve()
    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index != int(args.task_index):
                continue
            row = json.loads(line)
            for key in [
                "cosmology",
                "realization",
                "halo_root",
                "mass_root",
                "mass_filename",
                "dm_root",
                "snapshot_name",
                "result_root",
                "label",
                "method",
                "mass_bins",
                "density_bins",
                "density_radius",
                "base_weight",
                "mass_weight_normalization",
                "rankmass_strength",
                "env_source",
                "env_field_weight",
                "weight_source",
                "split_seed",
                "env_mesh",
                "recon_mesh",
                "gsr",
                "thr",
                "lambda_th",
                "env_sample_shift_cells",
                "stochastic_kmin",
                "stochastic_kmax",
                "stochastic_nbins",
                "covariance_regularization",
                "min_component_count",
                "solution_shrinkage",
                "weight_factor_clip_min",
                "weight_factor_clip_max",
                "chunk_size",
                "max_dm_particles",
                "dtype",
            ]:
                if key in row:
                    setattr(args, key.replace("-", "_"), row[key])
            return row | {"manifest_path": str(manifest), "manifest_row": int(index)}
    raise IndexError(f"task-index {args.task_index} out of range for {manifest}")


def halo_position_path(halo_root: str | Path, cosmology: str, realization: int) -> Path:
    return Path(halo_root).expanduser().resolve() / cosmology / f"real{int(realization):03d}" / "positions_mpc_h.npy"


def halo_mass_path(mass_root: str | Path, cosmology: str, realization: int, mass_filename: str) -> Path:
    """返回单个 realization 的 halo mass 文件路径。"""

    return Path(mass_root).expanduser().resolve() / cosmology / f"real{int(realization):03d}" / str(mass_filename)


def snapshot_dir(dm_root: str | Path, cosmology: str, realization: int, snapshot_name: str) -> Path:
    return Path(dm_root).expanduser().resolve() / cosmology / str(int(realization)) / snapshot_name


def snapshot_files(path: str | Path) -> list[Path]:
    files = sorted(Path(path).glob("snap_*.hdf5"))
    if not files:
        files = sorted(Path(path).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No Quijote HDF5 files found in {path}")
    return files


def read_snapshot_header(path: str | Path) -> dict[str, Any]:
    first = snapshot_files(path)[0]
    with h5py.File(first, "r") as handle:
        attrs = handle["Header"].attrs
        return {
            "boxsize_kpc_h": float(attrs["BoxSize"]),
            "boxsize_mpc_h": float(attrs["BoxSize"]) * COORD_UNIT_SCALE_TO_MPC_H,
            "redshift": float(attrs["Redshift"]),
            "num_files_per_snapshot": int(attrs["NumFilesPerSnapshot"]),
            "num_part_total_type1": int(attrs["NumPart_Total"][1]),
            "omega_m": float(attrs["Omega0"]),
            "omega_lambda": float(attrs["OmegaLambda"]),
            "hubble_param": float(attrs["HubbleParam"]),
        }


def require_power_of_two(mesh: int, name: str) -> int:
    mesh = int(mesh)
    if mesh <= 0 or mesh & (mesh - 1):
        raise ValueError(f"{name} must be a positive power of two, got {mesh}")
    return mesh


def make_output_dir(args: argparse.Namespace) -> Path:
    output = (
        Path(args.result_root).expanduser().resolve()
        / str(args.cosmology)
        / f"real{int(args.realization):03d}"
    )
    if args.label:
        output = output / str(args.label)
    output.mkdir(parents=True, exist_ok=True)
    return validate_output_path(output)


def has_complete_outputs(paths: list[Path]) -> bool:
    """判断 sto-recon 关键输出是否全部存在。

    参数
    ----
    paths
        由 run() 构造的关键输出路径列表，包括 labels、weights、spectra、
        diagnostics、post_sto_halo 和 DONE。

    返回
    ----
    bool
        全部文件存在时返回 True；任何文件缺失都返回 False。
    """

    return all(path.exists() for path in paths)


def paint_positions_to_counts(
    positions: np.ndarray,
    *,
    mesh: int,
    boxsize: float,
    weights: np.ndarray | None = None,
    out: Any | None = None,
) -> Any:
    """Paint positions to a pmesh RealField and return the field."""

    from pmesh.pm import ParticleMesh

    pm = ParticleMesh(Nmesh=[int(mesh)] * 3, BoxSize=float(boxsize), dtype="f8", resampler="cic")
    if out is None:
        out = pm.create(type="real")
        out.value[...] = 0.0
        hold = False
    else:
        hold = True
    mass: float | np.ndarray
    mass = 1.0 if weights is None else np.asarray(weights, dtype="f8")
    return pm.paint(np.asarray(positions, dtype="f8"), mass=mass, hold=hold, out=out)


def field_to_ndarray(field: Any) -> np.ndarray:
    return np.asarray(field.value, dtype=np.float64).copy()


def load_dm_counts(
    path: str | Path,
    *,
    mesh: int,
    boxsize: float,
    chunk_size: int,
    dtype: np.dtype,
    max_particles: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from pmesh.pm import ParticleMesh

    pm = ParticleMesh(Nmesh=[int(mesh)] * 3, BoxSize=float(boxsize), dtype="f8", resampler="cic")
    field = pm.create(type="real")
    field.value[...] = 0.0

    files = snapshot_files(path)
    total_read = 0
    per_file: list[int] = []
    for filename in files:
        file_read = 0
        with h5py.File(filename, "r") as handle:
            coords = handle["PartType1"]["Coordinates"]
            nrows = int(coords.shape[0])
            stop_file = nrows
            if max_particles is not None:
                remaining = int(max_particles) - total_read
                if remaining <= 0:
                    break
                stop_file = min(stop_file, remaining)
            for start in range(0, stop_file, int(chunk_size)):
                stop = min(start + int(chunk_size), stop_file)
                pos = np.asarray(coords[start:stop], dtype=dtype)
                pos = np.mod(pos * COORD_UNIT_SCALE_TO_MPC_H, float(boxsize)).astype("f8", copy=False)
                pm.paint(pos, mass=1.0, hold=True, out=field)
                nread = int(stop - start)
                total_read += nread
                file_read += nread
        per_file.append(file_read)
        if max_particles is not None and total_read >= int(max_particles):
            break

    counts = field_to_ndarray(field)
    info = {
        "snapshot_dir": str(Path(path).expanduser().resolve()),
        "file_count_used": len(per_file),
        "particles_read": int(total_read),
        "particles_read_per_file": per_file,
        "max_dm_particles": None if max_particles is None else int(max_particles),
        "mesh": int(mesh),
        "count_sum": float(np.sum(counts)),
        "count_mean": float(np.mean(counts)),
        "count_max": float(np.max(counts)),
    }
    return counts, info


def counts_to_delta(counts: np.ndarray) -> np.ndarray:
    mean = float(np.mean(counts))
    if not np.isfinite(mean) or mean == 0.0:
        raise ValueError("Cannot build overdensity field with zero/non-finite mean count.")
    return np.asarray(counts, dtype=np.float64) / mean - 1.0


def kgrid_rfft(mesh: int, boxsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell = float(boxsize) / int(mesh)
    kx = 2.0 * np.pi * np.fft.fftfreq(int(mesh), d=cell)
    ky = 2.0 * np.pi * np.fft.fftfreq(int(mesh), d=cell)
    kz = 2.0 * np.pi * np.fft.rfftfreq(int(mesh), d=cell)
    k2 = (
        kx[:, None, None] ** 2
        + ky[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    return kx, ky, kz, k2


def gaussian_window(mesh: int, boxsize: float, radius: float) -> np.ndarray:
    _, _, _, k2 = kgrid_rfft(mesh, boxsize)
    return np.exp(-0.5 * k2 * float(radius) ** 2)


def top_hat_window(mesh: int, boxsize: float, radius: float) -> np.ndarray:
    _, _, _, k2 = kgrid_rfft(mesh, boxsize)
    x = np.sqrt(k2) * float(radius)
    out = np.ones_like(x, dtype=np.float64)
    mask = x > 0.0
    xm = x[mask]
    out[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)
    return out


def sample_grid_nearest(
    field: np.ndarray,
    positions: np.ndarray,
    boxsize: float,
    *,
    shift_cells: int = 0,
) -> np.ndarray:
    mesh = int(field.shape[0])
    scaled = np.asarray(positions, dtype=np.float64) / float(boxsize) * mesh
    index = (np.floor(scaled + 0.5).astype(np.int64) + int(shift_cells)) % mesh
    return field[index[:, 0], index[:, 1], index[:, 2]]


def tidal_env_labels(
    dm_counts: np.ndarray,
    halo_positions: np.ndarray,
    *,
    boxsize: float,
    gsr: float,
    lambda_th: float,
    sample_shift_cells: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mesh = int(dm_counts.shape[0])
    delta = counts_to_delta(dm_counts)
    delta_k = np.fft.rfftn(delta, axes=(0, 1, 2))
    kx, ky, kz, k2 = kgrid_rfft(mesh, boxsize)
    smooth = gaussian_window(mesh, boxsize, gsr)

    comp_values = np.empty((halo_positions.shape[0], 6), dtype=np.float64)
    component_specs = [
        (0, 0, kx[:, None, None], kx[:, None, None]),
        (0, 1, kx[:, None, None], ky[None, :, None]),
        (0, 2, kx[:, None, None], kz[None, None, :]),
        (1, 1, ky[None, :, None], ky[None, :, None]),
        (1, 2, ky[None, :, None], kz[None, None, :]),
        (2, 2, kz[None, None, :], kz[None, None, :]),
    ]
    for icomp, (_i, _j, ki, kj) in enumerate(component_specs):
        factor = np.zeros_like(k2, dtype=np.float64)
        np.divide(ki * kj, k2, out=factor, where=k2 > 0.0)
        tensor = np.fft.irfftn(delta_k * smooth * factor, s=delta.shape, axes=(0, 1, 2))
        comp_values[:, icomp] = sample_grid_nearest(
            tensor,
            halo_positions,
            boxsize,
            shift_cells=int(sample_shift_cells),
        )

    matrices = np.empty((halo_positions.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = comp_values[:, 0]
    matrices[:, 0, 1] = matrices[:, 1, 0] = comp_values[:, 1]
    matrices[:, 0, 2] = matrices[:, 2, 0] = comp_values[:, 2]
    matrices[:, 1, 1] = comp_values[:, 3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = comp_values[:, 4]
    matrices[:, 2, 2] = comp_values[:, 5]
    eigvals = np.linalg.eigvalsh(matrices)
    labels = np.sum(eigvals > float(lambda_th), axis=1).astype(np.int8)
    counts = np.bincount(labels, minlength=4).astype(np.int64)
    info = {
        "env_mesh": int(mesh),
        "gsr_mpc_h": float(gsr),
        "lambda_th": float(lambda_th),
        "sample_shift_cells": int(sample_shift_cells),
        "label_counts": {ENV_NAMES[i]: int(counts[i]) for i in range(4)},
        "label_fractions": {ENV_NAMES[i]: float(counts[i] / max(1, labels.size)) for i in range(4)},
        "eig_min": [float(item) for item in np.min(eigvals, axis=0)],
        "eig_max": [float(item) for item in np.max(eigvals, axis=0)],
        "eig_mean": [float(item) for item in np.mean(eigvals, axis=0)],
        "eig_percentiles": {
            str(p): [float(item) for item in np.percentile(eigvals, p, axis=0)]
            for p in [1, 5, 16, 50, 84, 95, 99]
        },
    }
    return labels, info


def smooth_delta_field(delta: np.ndarray, *, boxsize: float, radius: float) -> np.ndarray:
    window = top_hat_window(int(delta.shape[0]), boxsize, radius)
    return np.fft.irfftn(
        np.fft.rfftn(delta, axes=(0, 1, 2)) * window,
        s=delta.shape,
        axes=(0, 1, 2),
    )


def weighted_shot_noise_volume(weights: np.ndarray, boxsize: float) -> dict[str, float]:
    """Return the expected Poisson shot-noise amplitude for a weighted catalog."""

    arr = np.asarray(weights, dtype=np.float64)
    sum_w = float(np.sum(arr))
    sum_w2 = float(np.sum(arr * arr))
    if not np.isfinite(sum_w) or abs(sum_w) < 1.0e-14:
        raise ValueError("Cannot compute weighted shot noise with zero/non-finite sum of weights.")
    volume = float(boxsize) ** 3
    effective_number = float(sum_w * sum_w / sum_w2) if sum_w2 > 0.0 else float("nan")
    return {
        "volume": volume,
        "sum_w": sum_w,
        "sum_w2": sum_w2,
        "effective_number": effective_number,
        "effective_number_density": effective_number / volume if np.isfinite(effective_number) else float("nan"),
        "shot_noise": float(volume * sum_w2 / (sum_w * sum_w)),
    }


def paint_halo_delta(
    positions: np.ndarray,
    *,
    mesh: int,
    boxsize: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    field = paint_positions_to_counts(positions, mesh=mesh, boxsize=boxsize, weights=weights)
    counts = field_to_ndarray(field)
    weight_sum = float(np.sum(counts))
    mean = float(np.mean(counts))
    if mean == 0.0:
        raise ValueError("empty halo field")
    return counts / mean - 1.0, {
        "mesh": int(mesh),
        "weight_sum": weight_sum,
        "weight_mean_per_cell": mean,
        "weight_max_cell": float(np.max(counts)),
    }


def paint_halo_counts(
    positions: np.ndarray,
    *,
    mesh: int,
    boxsize: float,
    weights: np.ndarray | None = None,
    source_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把 halo catalog 画到 CIC 网格，作为 halo-only Env 或诊断输入场。

    参数
    ----
    positions
        halo 坐标，单位 Mpc/h，shape=(N, 3)。
    mesh, boxsize
        周期盒子的网格数和盒长。
    weights
        可选 halo 权重；None 表示 number-weighted galaxy/halo field。
    source_name
        写入 diagnostics 的场来源说明。

    返回
    ----
    tuple
        `(counts, info)`，counts 是 CIC count/weight 网格，info 记录总权重等审计信息。
    """

    field = paint_positions_to_counts(positions, mesh=mesh, boxsize=boxsize, weights=weights)
    counts = field_to_ndarray(field)
    info = {
        "source_name": str(source_name),
        "mesh": int(mesh),
        "n_halos": int(np.asarray(positions).shape[0]),
        "uses_weights": bool(weights is not None),
        "count_sum": float(np.sum(counts)),
        "count_mean": float(np.mean(counts)),
        "count_max": float(np.max(counts)),
    }
    if weights is not None:
        info["input_weight_summary"] = summarize_array(np.asarray(weights, dtype=np.float64))
    return counts, info


def deterministic_split_mask(n_items: int, *, seed: int, cosmology: str, realization: int) -> tuple[np.ndarray, dict[str, Any]]:
    """构造可复现的 split-half mask，用于 halo-only 自监督权重求解。"""

    token = f"{cosmology}:{int(realization)}"
    offset = 0
    for char in token:
        offset = (offset * 131 + ord(char)) % (2**32)
    final_seed = (int(seed) + offset) % (2**32)
    rng = np.random.default_rng(final_seed)
    score = rng.random(int(n_items))
    mask = score < 0.5
    if not np.any(mask) or np.all(mask):
        order = np.argsort(score, kind="mergesort")
        mask = np.zeros(int(n_items), dtype=bool)
        mask[order[: int(n_items) // 2]] = True
    return mask, {
        "base_seed": int(seed),
        "realization_seed": int(final_seed),
        "cosmology": str(cosmology),
        "realization": int(realization),
        "n_half_a": int(np.sum(mask)),
        "n_half_b": int(np.sum(~mask)),
    }


def resolve_component_config(args: argparse.Namespace) -> ComponentConfig:
    """把命令行/manifest 中的方法选项规范化。

    参数
    ----
    args
        命令行参数，可能已经被 manifest row 覆盖。

    返回
    ----
    ComponentConfig
        后续 split 和权重求解使用的不可变配置。
    """

    method = str(args.method).lower()
    if method == "nenv":
        mass_bins = 1 if args.mass_bins is None else int(args.mass_bins)
        base_weight = "number"
    elif method == "monly":
        # mass-only：component = 等数量 mass bin，无环境维度。
        # mass_bins 缺省 4（与 m4env 对齐）；base_weight 完全放开，
        # 支持 number/mass/rankmass 三种基础权重（本计划的对照维度）。
        mass_bins = 4 if args.mass_bins is None else int(args.mass_bins)
        base_weight = str(args.base_weight)
    elif method == "m2env":
        mass_bins = 2
        base_weight = "mass"
    elif method == "m4env":
        mass_bins = 4 if args.mass_bins is None else int(args.mass_bins)
        base_weight = str(args.base_weight)
    elif method == "m4rho3env":
        mass_bins = 4 if args.mass_bins is None else int(args.mass_bins)
        base_weight = str(args.base_weight)
    else:
        raise ValueError(f"unknown sto-recon method={method}")
    density_bins = 3 if (method == "m4rho3env" and args.density_bins is None) else int(args.density_bins or 1)
    if mass_bins <= 0:
        raise ValueError(f"mass_bins must be positive, got {mass_bins}")
    if density_bins <= 0:
        raise ValueError(f"density_bins must be positive, got {density_bins}")
    min_component_count = int(args.min_component_count)
    if min_component_count < 2:
        raise ValueError(f"min_component_count must be >=2, got {min_component_count}")
    solution_shrinkage = float(args.solution_shrinkage)
    if solution_shrinkage < 0.0 or solution_shrinkage > 1.0:
        raise ValueError(f"solution_shrinkage must be within [0, 1], got {solution_shrinkage}")
    clip_min = None if args.weight_factor_clip_min is None else float(args.weight_factor_clip_min)
    clip_max = None if args.weight_factor_clip_max is None else float(args.weight_factor_clip_max)
    if clip_min is not None and clip_min < 0.0:
        raise ValueError(f"weight_factor_clip_min must be non-negative, got {clip_min}")
    if clip_max is not None and clip_max <= 0.0:
        raise ValueError(f"weight_factor_clip_max must be positive, got {clip_max}")
    if clip_min is not None and clip_max is not None and clip_min >= clip_max:
        raise ValueError(f"weight factor clip min must be < max, got {clip_min} >= {clip_max}")
    rankmass_strength = float(args.rankmass_strength)
    if rankmass_strength < 0.0 or rankmass_strength >= 2.0:
        raise ValueError(f"rankmass_strength must satisfy 0 <= strength < 2, got {rankmass_strength}")
    return ComponentConfig(
        method=method,
        mass_bins=int(mass_bins),
        density_bins=int(density_bins),
        density_radius_mpc_h=float(args.density_radius),
        base_weight=str(base_weight),
        mass_filename=str(args.mass_filename),
        mass_weight_normalization=str(args.mass_weight_normalization),
        rankmass_strength=rankmass_strength,
        min_component_count=min_component_count,
        solution_shrinkage=solution_shrinkage,
        weight_factor_clip_min=clip_min,
        weight_factor_clip_max=clip_max,
    )


def equal_count_mass_bin_labels(masses: np.ndarray, nbins: int) -> tuple[np.ndarray, dict[str, Any]]:
    """按 halo mass 排序后切成等数量 bin。

    参数
    ----
    masses
        shape=(Nhalo,) 的 halo mass，单位 Msun/h。
    nbins
        需要的 mass bin 数；M-2+Env 使用 2。

    返回
    ----
    tuple
        `(labels, info)`，labels 是 0..nbins-1 的 int16 数组，info 记录每个
        bin 的数量和 mass 范围，便于审计。
    """

    masses = np.asarray(masses, dtype=np.float64)
    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError(f"mass bin count must be positive, got {nbins}")
    labels = np.zeros(masses.shape[0], dtype=np.int16)
    if nbins == 1:
        return labels, {
            "mass_bins": 1,
            "policy": "single bin",
            "bin_counts": [int(masses.size)],
            "bin_mass_min_msun_h": [float(np.min(masses)) if masses.size else float("nan")],
            "bin_mass_max_msun_h": [float(np.max(masses)) if masses.size else float("nan")],
        }
    order = np.argsort(masses, kind="mergesort")
    splits = np.array_split(order, nbins)
    bin_counts: list[int] = []
    bin_min: list[float] = []
    bin_max: list[float] = []
    for bin_id, indices in enumerate(splits):
        labels[indices] = int(bin_id)
        bin_counts.append(int(indices.size))
        if indices.size:
            values = masses[indices]
            bin_min.append(float(np.min(values)))
            bin_max.append(float(np.max(values)))
        else:
            bin_min.append(float("nan"))
            bin_max.append(float("nan"))
    return labels, {
        "mass_bins": int(nbins),
        "policy": "stable argsort + numpy.array_split into equal-count bins",
        "bin_counts": bin_counts,
        "bin_mass_min_msun_h": bin_min,
        "bin_mass_max_msun_h": bin_max,
    }


def equal_count_value_bin_labels(values: np.ndarray, nbins: int, *, value_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """按任意一维 halo-only 特征切等数量 bin。

    参数
    ----
    values
        shape=(Nhalo,) 的一维特征，例如 halo-only 平滑局部密度。
    nbins
        需要的 bin 数。
    value_name
        写入 diagnostics 的特征名字。

    返回
    ----
    tuple
        `(labels, info)`。labels 是 0..nbins-1 的 int16 数组；info 记录
        每个 bin 的数量、最小值和最大值，便于审计拆分是否稳定。
    """

    arr = np.asarray(values, dtype=np.float64)
    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError(f"{value_name} bin count must be positive, got {nbins}")
    labels = np.zeros(arr.shape[0], dtype=np.int16)
    if nbins == 1:
        return labels, {
            f"{value_name}_bins": 1,
            "policy": "single bin",
            "bin_counts": [int(arr.size)],
            "bin_value_min": [float(np.min(arr)) if arr.size else float("nan")],
            "bin_value_max": [float(np.max(arr)) if arr.size else float("nan")],
        }
    order = np.argsort(arr, kind="mergesort")
    splits = np.array_split(order, nbins)
    bin_counts: list[int] = []
    bin_min: list[float] = []
    bin_max: list[float] = []
    for bin_id, indices in enumerate(splits):
        labels[indices] = int(bin_id)
        bin_counts.append(int(indices.size))
        if indices.size:
            chunk = arr[indices]
            bin_min.append(float(np.min(chunk)))
            bin_max.append(float(np.max(chunk)))
        else:
            bin_min.append(float("nan"))
            bin_max.append(float("nan"))
    return labels, {
        f"{value_name}_bins": int(nbins),
        "policy": "stable argsort + numpy.array_split into equal-count bins",
        "bin_counts": bin_counts,
        "bin_value_min": bin_min,
        "bin_value_max": bin_max,
    }


def local_density_bin_labels(
    halo_counts: np.ndarray,
    halo_positions: np.ndarray,
    *,
    boxsize: float,
    radius: float,
    nbins: int,
    sample_shift_cells: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用 halo 自身的平滑密度场给每个 halo 打局部密度标签。

    这个标签完全由 halo catalog 计算，不读取 DM。它的作用是把同一
    mass/env 中处于不同局部拥挤程度的 halo 再细分，给 split-half
    自监督求解更多可观测特征。
    """

    nbins = int(nbins)
    if nbins == 1:
        labels = np.zeros(halo_positions.shape[0], dtype=np.int16)
        return labels, {
            "density_bins": 1,
            "radius_mpc_h": float(radius),
            "policy": "single bin; local-density feature disabled",
        }
    delta = counts_to_delta(np.asarray(halo_counts, dtype=np.float64))
    smooth = smooth_delta_field(delta, boxsize=boxsize, radius=float(radius))
    sampled = sample_grid_nearest(
        smooth,
        halo_positions,
        boxsize,
        shift_cells=int(sample_shift_cells),
    )
    labels, split_info = equal_count_value_bin_labels(sampled, nbins, value_name="local_density")
    info = {
        "density_bins": int(nbins),
        "radius_mpc_h": float(radius),
        "sample_shift_cells": int(sample_shift_cells),
        "sampled_delta_summary": summarize_array(sampled),
        "split": split_info,
    }
    return labels, info


def build_component_labels(
    env_labels: np.ndarray,
    mass_bin_labels: np.ndarray,
    *,
    mass_bins: int,
    density_bin_labels: np.ndarray | None = None,
    density_bins: int = 1,
    env_enabled: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """把 mass bin、environment 和可选局部密度 bin 合成为 component label。

    参数
    ----
    env_labels
        0..3 的环境标签；``env_enabled=False`` 时忽略。
    mass_bin_labels
        0..mass_bins-1 的质量 bin 标签。
    mass_bins
        质量 bin 数。
    density_bin_labels
        0..density_bins-1 的局部 halo 密度标签；None 表示单一 density bin。
    density_bins
        局部密度 bin 数。
    env_enabled
        False 表示 mass-only（monly）：component 直接等于 mass bin，
        命名为 M1..Mk，不乘环境维度。

    返回
    ----
    tuple
        `(component_labels, component_names)`；env_enabled 时 component id
        采用 ``(mass_bin * 4 + env_id) * density_bins + density_bin``，
        monly 时直接为 mass_bin。
    """

    env_labels = np.asarray(env_labels, dtype=np.int16)
    mass_bin_labels = np.asarray(mass_bin_labels, dtype=np.int16)
    if env_labels.shape[0] != mass_bin_labels.shape[0]:
        raise ValueError("env_labels and mass_bin_labels length mismatch")
    density_bins = int(density_bins)
    if density_bin_labels is None:
        density_bin_labels = np.zeros(env_labels.shape[0], dtype=np.int16)
    density_bin_labels = np.asarray(density_bin_labels, dtype=np.int16)
    if density_bin_labels.shape[0] != env_labels.shape[0]:
        raise ValueError("density_bin_labels and env_labels length mismatch")
    if not env_enabled:
        # monly：只有 mass 维度；density 维度此时恒为 1（resolve 保证）。
        component_labels = mass_bin_labels.astype(np.int16, copy=True)
        component_names = [f"M{mass_bin + 1}" for mass_bin in range(int(mass_bins))]
        return component_labels, component_names
    component_labels = ((mass_bin_labels * 4 + env_labels) * density_bins + density_bin_labels).astype(
        np.int16,
        copy=False,
    )
    if int(mass_bins) == 1 and density_bins == 1:
        component_names = list(ENV_NAMES)
    else:
        component_names = []
        for mass_bin in range(int(mass_bins)):
            for env_id in range(4):
                for density_id in range(density_bins):
                    name = f"M{mass_bin + 1}-{ENV_NAMES[env_id]}"
                    if density_bins > 1:
                        name = f"{name}-R{density_id + 1}"
                    component_names.append(name)
    return component_labels, component_names


def build_base_weights(
    masses: np.ndarray | None,
    *,
    base_weight: str,
    normalization: str,
    rankmass_strength: float,
    n_halos: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """构造 component field 使用的基础 halo 权重。

    参数
    ----
    masses
        halo mass 数组；base_weight='mass' 时必须提供。
    base_weight
        'number' 表示每个 halo 权重为 1；'mass' 表示使用 mass。
    normalization
        mass 权重的全局归一化方式；默认 mean。
    n_halos
        halo 数量，用于 number 权重。

    返回
    ----
    tuple
        `(base_weights, info)`；base_weights 会进入 component density field。
    """

    if base_weight == "number":
        weights = np.ones(int(n_halos), dtype=np.float64)
        return weights, {"base_weight": "number", "normalization": "none", "weight_summary": summarize_array(weights)}
    if base_weight == "rankmass":
        if masses is None:
            raise ValueError("base_weight='rankmass' requires halo masses")
        raw = np.asarray(masses, dtype=np.float64)
        if raw.shape[0] != int(n_halos):
            raise ValueError(f"mass length {raw.shape[0]} != n_halos {n_halos}")
        order = np.argsort(raw, kind="mergesort")
        rank = np.empty(raw.shape[0], dtype=np.float64)
        rank[order] = (np.arange(raw.shape[0], dtype=np.float64) + 0.5) / float(raw.shape[0])
        strength = float(rankmass_strength)
        weights = 1.0 + strength * (rank - 0.5)
        if np.any(weights <= 0.0):
            raise ValueError(f"rankmass produced non-positive weights with strength={strength}")
        return weights.astype(np.float64, copy=False), {
            "base_weight": "rankmass",
            "normalization": "rank-linear-mean-one",
            "rankmass_strength": strength,
            "raw_mass_summary_msun_h": summarize_array(raw),
            "rank_summary": summarize_array(rank),
            "weight_summary": summarize_array(weights),
            "note": "Halo-only bounded mass-rank weight: w = 1 + strength * (rank - 0.5).",
        }
    if base_weight != "mass":
        raise ValueError(f"unknown base_weight={base_weight}")
    if masses is None:
        raise ValueError("base_weight='mass' requires halo masses")
    raw = np.asarray(masses, dtype=np.float64)
    if raw.shape[0] != int(n_halos):
        raise ValueError(f"mass length {raw.shape[0]} != n_halos {n_halos}")
    if normalization == "mean":
        scale = float(np.mean(raw))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid mean mass scale={scale}")
        weights = raw / scale
    elif normalization == "none":
        scale = 1.0
        weights = raw.copy()
    else:
        raise ValueError(f"unknown mass normalization={normalization}")
    return weights.astype(np.float64, copy=False), {
        "base_weight": "mass",
        "normalization": str(normalization),
        "mass_scale_msun_h": float(scale),
        "raw_mass_summary_msun_h": summarize_array(raw),
        "weight_summary": summarize_array(weights),
    }


def finalize_component_weight_factors(
    raw_active_solution: np.ndarray,
    active_components: list[int],
    component_weight_sums: np.ndarray,
    *,
    component_names: list[str],
    solution_shrinkage: float,
    weight_factor_clip_min: float | None,
    weight_factor_clip_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """把线性求解的 component 份额转为稳定的 per-halo 权重因子。

    参数
    ----
    raw_active_solution
        只包含 active components 的未归一化解，来自 `C^{-1}u`。
    active_components
        `raw_active_solution` 对应的 component id。
    component_weight_sums
        每个 component 的原始 base weight 总和。
    component_names
        component 名字，写 diagnostics 用。
    solution_shrinkage
        将求解份额向原始 base 份额收缩的比例；用于降低 split-half
        自监督中少数 bin 的噪声放大。
    weight_factor_clip_min, weight_factor_clip_max
        可选 per-halo 权重因子 clip 边界；clip 后会重新归一化，使总权重
        与 base catalog 的总权重一致。

    返回
    ----
    tuple
        `(solution_by_component, factor_by_component, fractions, info)`。
        `solution_by_component` 是每个 component 在最终 catalog 总权重中
        占的份额；`factor_by_component` 会乘到 component 内每个 halo 的
        base weight 上。
    """

    component_weight_sums = np.asarray(component_weight_sums, dtype=np.float64)
    total_base_weight = float(np.sum(component_weight_sums))
    if not np.isfinite(total_base_weight) or total_base_weight <= 0.0:
        raise ValueError("base weights must have positive finite sum")
    fractions = component_weight_sums / total_base_weight
    n_components = int(component_weight_sums.size)
    active_components = [int(item) for item in active_components]
    raw_active_solution = np.asarray(raw_active_solution, dtype=np.float64)
    if raw_active_solution.shape[0] != len(active_components):
        raise ValueError("raw_active_solution length does not match active_components")

    raw_solution_by_component = np.zeros(n_components, dtype=np.float64)
    raw_sum = float(np.sum(raw_active_solution))
    if not np.isfinite(raw_sum) or abs(raw_sum) < 1.0e-14 or not np.all(np.isfinite(raw_active_solution)):
        raise RuntimeError(f"Cannot normalize component weights; raw sum={raw_sum}")
    active_solution = raw_active_solution / raw_sum
    for component_id, value in zip(active_components, active_solution):
        raw_solution_by_component[component_id] = value

    active_mask = np.zeros(n_components, dtype=bool)
    active_mask[active_components] = True
    solution_by_component = fractions.copy()
    solution_by_component[active_mask] = raw_solution_by_component[active_mask]

    shrinkage = float(solution_shrinkage)
    if shrinkage > 0.0:
        solution_by_component = (1.0 - shrinkage) * solution_by_component + shrinkage * fractions

    solution_sum = float(np.sum(solution_by_component))
    if not np.isfinite(solution_sum) or abs(solution_sum) < 1.0e-14:
        raise RuntimeError(f"Cannot normalize shrunken component weights; sum={solution_sum}")
    solution_by_component = solution_by_component / solution_sum

    weight_factor_by_component = np.zeros(n_components, dtype=np.float64)
    positive = fractions > 0.0
    weight_factor_by_component[positive] = solution_by_component[positive] / fractions[positive]

    clip_applied = bool(weight_factor_clip_min is not None or weight_factor_clip_max is not None)
    if clip_applied:
        clip_min = -np.inf if weight_factor_clip_min is None else float(weight_factor_clip_min)
        clip_max = np.inf if weight_factor_clip_max is None else float(weight_factor_clip_max)
        before_clip = weight_factor_by_component.copy()
        weight_factor_by_component[positive] = np.clip(weight_factor_by_component[positive], clip_min, clip_max)
        norm = float(np.sum(fractions[positive] * weight_factor_by_component[positive]))
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"Invalid clipped component-weight normalization={norm}")
        weight_factor_by_component[positive] /= norm
        solution_by_component = fractions * weight_factor_by_component
    else:
        before_clip = weight_factor_by_component.copy()

    info = {
        "solution_shrinkage_to_base_fraction": float(shrinkage),
        "inactive_component_policy": "base_fraction",
        "weight_factor_clip_min": None if weight_factor_clip_min is None else float(weight_factor_clip_min),
        "weight_factor_clip_max": None if weight_factor_clip_max is None else float(weight_factor_clip_max),
        "weight_factor_clip_applied": clip_applied,
        "raw_solution_by_component": {component_names[i]: float(raw_solution_by_component[i]) for i in range(n_components)},
        "weight_factor_before_clip_by_component": {
            component_names[i]: float(before_clip[i]) for i in range(n_components)
        },
    }
    return solution_by_component, weight_factor_by_component, fractions, info


def solve_component_weights(
    dm_delta: np.ndarray,
    halo_positions: np.ndarray,
    component_labels: np.ndarray,
    *,
    base_weights: np.ndarray,
    component_names: list[str],
    mesh: int,
    boxsize: float,
    thr: float,
    regularization: float,
    min_component_count: int = 2,
    solution_shrinkage: float = 0.0,
    weight_factor_clip_min: float | None = None,
    weight_factor_clip_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dm_smooth = smooth_delta_field(dm_delta, boxsize=boxsize, radius=thr)
    component_smooth: list[np.ndarray] = []
    labels = np.asarray(component_labels, dtype=np.int16)
    weights0 = np.asarray(base_weights, dtype=np.float64)
    if labels.shape[0] != halo_positions.shape[0] or weights0.shape[0] != halo_positions.shape[0]:
        raise ValueError("component labels/base weights/positions length mismatch")
    n_components = len(component_names)
    component_counts = np.bincount(labels, minlength=n_components).astype(np.int64)
    component_weight_sums = np.bincount(labels, weights=weights0, minlength=n_components).astype(np.float64)
    active_components: list[int] = []
    for component_id in range(n_components):
        mask = labels == component_id
        if int(np.sum(mask)) < int(min_component_count):
            continue
        if component_weight_sums[component_id] <= 0.0:
            continue
        delta_i, _ = paint_halo_delta(
            halo_positions[mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=weights0[mask],
        )
        component_smooth.append(smooth_delta_field(delta_i, boxsize=boxsize, radius=thr))
        active_components.append(component_id)

    if not active_components:
        raise RuntimeError("No non-empty sto-recon components available for weight solving.")

    cov_mm = float(np.mean(dm_smooth * dm_smooth))
    u = np.array([np.mean(dm_smooth * field) for field in component_smooth], dtype=np.float64)
    cmat = np.empty((len(component_smooth), len(component_smooth)), dtype=np.float64)
    for i, field_i in enumerate(component_smooth):
        for j, field_j in enumerate(component_smooth[i:], start=i):
            cmat[i, j] = cmat[j, i] = float(np.mean(field_i * field_j))

    reg_scale = float(np.nanmedian(np.diag(cmat))) if cmat.size else 0.0
    if not np.isfinite(reg_scale) or reg_scale <= 0.0:
        reg_scale = 1.0
    cmat_reg = cmat + float(regularization) * reg_scale * np.eye(cmat.shape[0])
    try:
        raw = np.linalg.solve(cmat_reg, u)
        solver = "solve"
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cmat_reg) @ u
        solver = "pinv"
    raw_sum = float(np.sum(raw))
    if not np.isfinite(raw_sum) or abs(raw_sum) < 1.0e-14:
        raise RuntimeError(f"Cannot normalize N+Env weights; raw sum={raw_sum}")

    solution_by_component, weight_factor_by_component, fractions, solution_regularization = finalize_component_weight_factors(
        raw,
        active_components,
        component_weight_sums,
        component_names=component_names,
        solution_shrinkage=float(solution_shrinkage),
        weight_factor_clip_min=weight_factor_clip_min,
        weight_factor_clip_max=weight_factor_clip_max,
    )
    halo_weights = weights0 * weight_factor_by_component[labels]

    weighted_delta, weighted_info = paint_halo_delta(
        halo_positions,
        mesh=mesh,
        boxsize=boxsize,
        weights=halo_weights,
    )
    weighted_smooth = smooth_delta_field(weighted_delta, boxsize=boxsize, radius=thr)
    pre_delta, pre_info = paint_halo_delta(halo_positions, mesh=mesh, boxsize=boxsize, weights=weights0)
    pre_smooth = smooth_delta_field(pre_delta, boxsize=boxsize, radius=thr)
    cov_hh_pre = float(np.mean(pre_smooth * pre_smooth))
    cov_hm_pre = float(np.mean(pre_smooth * dm_smooth))
    cov_hh_post = float(np.mean(weighted_smooth * weighted_smooth))
    cov_hm_post = float(np.mean(weighted_smooth * dm_smooth))
    r_pre = cov_hm_pre / np.sqrt(cov_mm * cov_hh_pre)
    r_post = cov_hm_post / np.sqrt(cov_mm * cov_hh_post)
    r2_opt = float(u @ raw / cov_mm)

    info = {
        "recon_mesh": int(mesh),
        "thr_mpc_h": float(thr),
        "min_component_count": int(min_component_count),
        "active_component_ids": [int(item) for item in active_components],
        "active_component_names": [component_names[item] for item in active_components],
        "component_counts": {component_names[i]: int(component_counts[i]) for i in range(n_components)},
        "component_base_weight_sums": {component_names[i]: float(component_weight_sums[i]) for i in range(n_components)},
        "component_base_weight_fractions": {component_names[i]: float(fractions[i]) for i in range(n_components)},
        "solution_weight_by_component": {component_names[i]: float(solution_by_component[i]) for i in range(n_components)},
        "per_halo_weight_factor_by_component": {component_names[i]: float(weight_factor_by_component[i]) for i in range(n_components)},
        "per_halo_weight_summary": summarize_array(halo_weights),
        "covariance_regularization": float(regularization),
        "covariance_regularization_scale": float(reg_scale),
        "solver": solver,
        "solution_regularization": solution_regularization,
        "cov_mm_smooth": cov_mm,
        "smooth_bias_pre_cov_hm_over_mm": float(cov_hm_pre / cov_mm) if cov_mm != 0.0 else float("nan"),
        "smooth_bias_post_cov_hm_over_mm": float(cov_hm_post / cov_mm) if cov_mm != 0.0 else float("nan"),
        "cov_dm_component": {component_names[component]: float(val) for component, val in zip(active_components, u)},
        "cov_component_matrix": cmat.tolist(),
        "r_pre_smooth": float(r_pre),
        "r_post_smooth": float(r_post),
        "r2_opt_from_solver": r2_opt,
        "stochasticity_pre_smooth_1_minus_r2": float(1.0 - r_pre**2),
        "stochasticity_post_smooth_1_minus_r2": float(1.0 - r_post**2),
        "smooth_stochasticity_reduced": bool((1.0 - r_post**2) < (1.0 - r_pre**2)),
        "pre_field": pre_info,
        "weighted_field": weighted_info,
    }
    return halo_weights, solution_by_component, info


def solve_component_weights_halo_split(
    halo_positions: np.ndarray,
    component_labels: np.ndarray,
    *,
    base_weights: np.ndarray,
    component_names: list[str],
    split_mask: np.ndarray,
    mesh: int,
    boxsize: float,
    thr: float,
    regularization: float,
    min_component_count: int = 2,
    solution_shrinkage: float = 0.0,
    weight_factor_clip_min: float | None = None,
    weight_factor_clip_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """用 split-half halo proxy 解 M-2+Env component 权重。

    这个函数是当前 halo-only 口径的关键替代：不读取同 realization 的 DM。
    A/B 两个随机半样本互为 teacher/source，使用一半中的 component fields
    去拟合另一半的总 halo 大尺度场。这样同一批 halo 的 Poisson 噪声不会
    同时出现在 source 和 target 中，权重求解仍然只依赖 halo catalog。
    """

    labels = np.asarray(component_labels, dtype=np.int16)
    weights0 = np.asarray(base_weights, dtype=np.float64)
    split_mask = np.asarray(split_mask, dtype=bool)
    if labels.shape[0] != halo_positions.shape[0] or weights0.shape[0] != halo_positions.shape[0]:
        raise ValueError("component labels/base weights/positions length mismatch")
    if split_mask.shape[0] != halo_positions.shape[0]:
        raise ValueError("split mask length mismatch")

    n_components = len(component_names)
    component_counts = np.bincount(labels, minlength=n_components).astype(np.int64)
    component_weight_sums = np.bincount(labels, weights=weights0, minlength=n_components).astype(np.float64)
    counts_a = np.bincount(labels[split_mask], minlength=n_components).astype(np.int64)
    counts_b = np.bincount(labels[~split_mask], minlength=n_components).astype(np.int64)
    active_components = [
        int(component_id)
        for component_id in range(n_components)
        if component_weight_sums[component_id] > 0.0 and counts_a[component_id] >= 2 and counts_b[component_id] >= 2
        and counts_a[component_id] >= int(min_component_count)
        and counts_b[component_id] >= int(min_component_count)
    ]
    if not active_components:
        raise RuntimeError("No split-half non-empty sto-recon components available for halo-only weight solving.")

    def one_direction(source_mask: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        target_delta, _ = paint_halo_delta(
            halo_positions[target_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=weights0[target_mask],
        )
        target_smooth = smooth_delta_field(target_delta, boxsize=boxsize, radius=thr)
        component_smooth: list[np.ndarray] = []
        for component_id in active_components:
            mask = source_mask & (labels == component_id)
            delta_i, _ = paint_halo_delta(
                halo_positions[mask],
                mesh=mesh,
                boxsize=boxsize,
                weights=weights0[mask],
            )
            component_smooth.append(smooth_delta_field(delta_i, boxsize=boxsize, radius=thr))
        u_vec = np.array([np.mean(target_smooth * field) for field in component_smooth], dtype=np.float64)
        c_mat = np.empty((len(component_smooth), len(component_smooth)), dtype=np.float64)
        for i, field_i in enumerate(component_smooth):
            for j, field_j in enumerate(component_smooth[i:], start=i):
                c_mat[i, j] = c_mat[j, i] = float(np.mean(field_i * field_j))
        target_var = float(np.mean(target_smooth * target_smooth))
        return c_mat, u_vec, target_var

    c_ab, u_ab, target_var_ab = one_direction(split_mask, ~split_mask)
    c_ba, u_ba, target_var_ba = one_direction(~split_mask, split_mask)
    cmat = 0.5 * (c_ab + c_ba)
    u = 0.5 * (u_ab + u_ba)
    target_var = 0.5 * (target_var_ab + target_var_ba)

    reg_scale = float(np.nanmedian(np.diag(cmat))) if cmat.size else 0.0
    if not np.isfinite(reg_scale) or reg_scale <= 0.0:
        reg_scale = 1.0
    cmat_reg = cmat + float(regularization) * reg_scale * np.eye(cmat.shape[0])
    try:
        raw = np.linalg.solve(cmat_reg, u)
        solver = "solve"
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cmat_reg) @ u
        solver = "pinv"
    raw_sum = float(np.sum(raw))
    fallback_to_base = False
    if not np.isfinite(raw_sum) or abs(raw_sum) < 1.0e-14 or not np.all(np.isfinite(raw)):
        fallback_to_base = True
        raw = component_weight_sums[active_components] / float(np.sum(component_weight_sums[active_components]))
        raw_sum = float(np.sum(raw))
        solver = "base-fraction-fallback"

    solution_by_component, weight_factor_by_component, fractions, solution_regularization = finalize_component_weight_factors(
        raw,
        active_components,
        component_weight_sums,
        component_names=component_names,
        solution_shrinkage=float(solution_shrinkage),
        weight_factor_clip_min=weight_factor_clip_min,
        weight_factor_clip_max=weight_factor_clip_max,
    )
    halo_weights = weights0 * weight_factor_by_component[labels]

    def split_validation(source_mask: np.ndarray, target_mask: np.ndarray, weights_source: np.ndarray) -> float:
        source_delta, _ = paint_halo_delta(
            halo_positions[source_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=weights_source[source_mask],
        )
        target_delta, _ = paint_halo_delta(
            halo_positions[target_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=weights0[target_mask],
        )
        source_smooth = smooth_delta_field(source_delta, boxsize=boxsize, radius=thr)
        target_smooth = smooth_delta_field(target_delta, boxsize=boxsize, radius=thr)
        cov = float(np.mean(source_smooth * target_smooth))
        var_source = float(np.mean(source_smooth * source_smooth))
        var_target = float(np.mean(target_smooth * target_smooth))
        return cov / np.sqrt(var_source * var_target)

    r_pre_ab = split_validation(split_mask, ~split_mask, weights0)
    r_pre_ba = split_validation(~split_mask, split_mask, weights0)
    r_post_ab = split_validation(split_mask, ~split_mask, halo_weights)
    r_post_ba = split_validation(~split_mask, split_mask, halo_weights)
    r_pre = float(0.5 * (r_pre_ab + r_pre_ba))
    r_post = float(0.5 * (r_post_ab + r_post_ba))
    r2_proxy = float(u @ raw / target_var) if target_var > 0.0 and np.all(np.isfinite(raw)) else float("nan")

    info = {
        "weight_source": "halo-split",
        "recon_mesh": int(mesh),
        "thr_mpc_h": float(thr),
        "min_component_count": int(min_component_count),
        "active_component_ids": [int(item) for item in active_components],
        "active_component_names": [component_names[item] for item in active_components],
        "component_counts": {component_names[i]: int(component_counts[i]) for i in range(n_components)},
        "component_counts_half_a": {component_names[i]: int(counts_a[i]) for i in range(n_components)},
        "component_counts_half_b": {component_names[i]: int(counts_b[i]) for i in range(n_components)},
        "component_base_weight_sums": {component_names[i]: float(component_weight_sums[i]) for i in range(n_components)},
        "component_base_weight_fractions": {component_names[i]: float(fractions[i]) for i in range(n_components)},
        "solution_weight_by_component": {component_names[i]: float(solution_by_component[i]) for i in range(n_components)},
        "per_halo_weight_factor_by_component": {component_names[i]: float(weight_factor_by_component[i]) for i in range(n_components)},
        "per_halo_weight_summary": summarize_array(halo_weights),
        "covariance_regularization": float(regularization),
        "covariance_regularization_scale": float(reg_scale),
        "solver": solver,
        "fallback_to_base": bool(fallback_to_base),
        "solution_regularization": solution_regularization,
        "target_proxy_variance_smooth": float(target_var),
        "cov_proxy_component": {component_names[component]: float(val) for component, val in zip(active_components, u)},
        "cov_component_matrix": cmat.tolist(),
        "r_pre_split_proxy_smooth": r_pre,
        "r_post_split_proxy_smooth": r_post,
        "r2_proxy_from_solver": r2_proxy,
        "split_proxy_1_minus_r2_pre": float(1.0 - r_pre**2),
        "split_proxy_1_minus_r2_post": float(1.0 - r_post**2),
        "split_proxy_improved": bool((1.0 - r_post**2) < (1.0 - r_pre**2)),
        "note": (
            "Halo-only self-supervised solve: source components from one random half "
            "are fitted to the base-weighted large-scale halo field of the other half; "
            "A/B and B/A covariances are averaged."
        ),
    }
    return halo_weights, solution_by_component, info


def summarize_array(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p16": float(np.percentile(arr, 16)),
        "p50": float(np.percentile(arr, 50)),
        "p84": float(np.percentile(arr, 84)),
        "p99": float(np.percentile(arr, 99)),
    }


def binned_cross_spectra(
    dm_delta: np.ndarray,
    pre_delta: np.ndarray,
    post_delta: np.ndarray,
    *,
    boxsize: float,
    kmin: float,
    kmax: float,
    nbins: int,
    shot_noise_pre: float,
    shot_noise_post: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mesh = int(dm_delta.shape[0])
    volume = float(boxsize) ** 3
    norm = volume / float(mesh**6)
    fields = {
        "dm": np.fft.rfftn(dm_delta, axes=(0, 1, 2)),
        "pre": np.fft.rfftn(pre_delta, axes=(0, 1, 2)),
        "post": np.fft.rfftn(post_delta, axes=(0, 1, 2)),
    }
    _, _, _, k2 = kgrid_rfft(mesh, boxsize)
    kval = np.sqrt(k2).ravel()
    edges = np.linspace(float(kmin), float(kmax), int(nbins) + 1)
    which = np.digitize(kval, edges) - 1
    valid = (which >= 0) & (which < int(nbins)) & (kval > 0.0)

    def bin_complex_product(name_a: str, name_b: str) -> np.ndarray:
        values = (fields[name_a] * np.conj(fields[name_b])).real.ravel() * norm
        sums = np.bincount(which[valid], weights=values[valid], minlength=int(nbins))
        counts = np.bincount(which[valid], minlength=int(nbins))
        out = np.full(int(nbins), np.nan, dtype=np.float64)
        mask = counts > 0
        out[mask] = sums[mask] / counts[mask]
        return out

    counts = np.bincount(which[valid], minlength=int(nbins)).astype(np.int64)
    ksum = np.bincount(which[valid], weights=kval[valid], minlength=int(nbins))
    kmean = np.full(int(nbins), np.nan, dtype=np.float64)
    mask = counts > 0
    kmean[mask] = ksum[mask] / counts[mask]
    pmm = bin_complex_product("dm", "dm")
    phh_pre = bin_complex_product("pre", "pre")
    phm_pre = bin_complex_product("pre", "dm")
    phh_post = bin_complex_product("post", "post")
    phm_post = bin_complex_product("post", "dm")
    r_pre = phm_pre / np.sqrt(pmm * phh_pre)
    r_post = phm_post / np.sqrt(pmm * phh_post)
    stoch_pre = 1.0 - r_pre**2
    stoch_post = 1.0 - r_post**2
    residual_pre = phh_pre - phm_pre**2 / pmm
    residual_post = phh_post - phm_post**2 / pmm
    residual_minus_shot_pre = residual_pre - float(shot_noise_pre)
    residual_minus_shot_post = residual_post - float(shot_noise_post)
    residual_over_shot_pre = residual_pre / float(shot_noise_pre)
    residual_over_shot_post = residual_post / float(shot_noise_post)
    spectra = {
        "k_edges": edges,
        "k": kmean,
        "nmodes": counts,
        "pmm": pmm,
        "phh_pre": phh_pre,
        "phm_pre": phm_pre,
        "phh_post": phh_post,
        "phm_post": phm_post,
        "r_pre": r_pre,
        "r_post": r_post,
        "stochasticity_pre": stoch_pre,
        "stochasticity_post": stoch_post,
        "residual_pre": residual_pre,
        "residual_post": residual_post,
        "shot_noise_pre": np.full(int(nbins), float(shot_noise_pre), dtype=np.float64),
        "shot_noise_post": np.full(int(nbins), float(shot_noise_post), dtype=np.float64),
        "residual_minus_shotnoise_pre": residual_minus_shot_pre,
        "residual_minus_shotnoise_post": residual_minus_shot_post,
        "residual_over_shotnoise_pre": residual_over_shot_pre,
        "residual_over_shotnoise_post": residual_over_shot_post,
    }
    valid_bins = np.isfinite(stoch_pre) & np.isfinite(stoch_post) & (counts > 0)
    pre_mean = float(np.nanmean(stoch_pre[valid_bins])) if np.any(valid_bins) else float("nan")
    post_mean = float(np.nanmean(stoch_post[valid_bins])) if np.any(valid_bins) else float("nan")
    resid_pre_mean = float(np.nanmean(residual_pre[valid_bins])) if np.any(valid_bins) else float("nan")
    resid_post_mean = float(np.nanmean(residual_post[valid_bins])) if np.any(valid_bins) else float("nan")
    resid_minus_shot_pre_mean = (
        float(np.nanmean(residual_minus_shot_pre[valid_bins])) if np.any(valid_bins) else float("nan")
    )
    resid_minus_shot_post_mean = (
        float(np.nanmean(residual_minus_shot_post[valid_bins])) if np.any(valid_bins) else float("nan")
    )
    resid_over_shot_pre_mean = (
        float(np.nanmean(residual_over_shot_pre[valid_bins])) if np.any(valid_bins) else float("nan")
    )
    resid_over_shot_post_mean = (
        float(np.nanmean(residual_over_shot_post[valid_bins])) if np.any(valid_bins) else float("nan")
    )
    summary = {
        "mesh": int(mesh),
        "kmin": float(kmin),
        "kmax": float(kmax),
        "nbins": int(nbins),
        "valid_bin_count": int(np.sum(valid_bins)),
        "mean_1_minus_r2_pre": pre_mean,
        "mean_1_minus_r2_post": post_mean,
        "mean_residual_pre": resid_pre_mean,
        "mean_residual_post": resid_post_mean,
        "shot_noise_pre": float(shot_noise_pre),
        "shot_noise_post": float(shot_noise_post),
        "mean_residual_minus_shotnoise_pre": resid_minus_shot_pre_mean,
        "mean_residual_minus_shotnoise_post": resid_minus_shot_post_mean,
        "mean_residual_over_shotnoise_pre": resid_over_shot_pre_mean,
        "mean_residual_over_shotnoise_post": resid_over_shot_post_mean,
        "stochasticity_reduced_by_1_minus_r2": bool(post_mean < pre_mean),
        "stochasticity_reduced_by_residual": bool(resid_post_mean < resid_pre_mean),
        "stochasticity_reduced_by_residual_minus_shotnoise": bool(
            resid_minus_shot_post_mean < resid_minus_shot_pre_mean
        ),
        "stochasticity_reduced_by_residual_over_shotnoise": bool(
            resid_over_shot_post_mean < resid_over_shot_pre_mean
        ),
    }
    return spectra, summary


def binned_split_proxy_spectra(
    halo_positions: np.ndarray,
    *,
    base_weights: np.ndarray,
    post_weights: np.ndarray,
    split_mask: np.ndarray,
    mesh: int,
    boxsize: float,
    kmin: float,
    kmax: float,
    nbins: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """用 split-half halo proxy 生成与旧 spectra.npz 兼容的诊断曲线。"""

    split_mask = np.asarray(split_mask, dtype=bool)

    def one_direction(source_mask: np.ndarray, target_mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        target_delta, _ = paint_halo_delta(
            halo_positions[target_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=base_weights[target_mask],
        )
        pre_delta, _ = paint_halo_delta(
            halo_positions[source_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=base_weights[source_mask],
        )
        post_delta, _ = paint_halo_delta(
            halo_positions[source_mask],
            mesh=mesh,
            boxsize=boxsize,
            weights=post_weights[source_mask],
        )
        pre_shot = weighted_shot_noise_volume(base_weights[source_mask], boxsize)["shot_noise"]
        post_shot = weighted_shot_noise_volume(post_weights[source_mask], boxsize)["shot_noise"]
        return binned_cross_spectra(
            target_delta,
            pre_delta,
            post_delta,
            boxsize=boxsize,
            kmin=kmin,
            kmax=kmax,
            nbins=nbins,
            shot_noise_pre=pre_shot,
            shot_noise_post=post_shot,
        )

    spectra_ab, summary_ab = one_direction(split_mask, ~split_mask)
    spectra_ba, summary_ba = one_direction(~split_mask, split_mask)
    spectra: dict[str, np.ndarray] = {}
    for key, value_ab in spectra_ab.items():
        value_ba = spectra_ba[key]
        if np.issubdtype(np.asarray(value_ab).dtype, np.number):
            spectra[key] = 0.5 * (np.asarray(value_ab) + np.asarray(value_ba))
        else:
            spectra[key] = value_ab
    summary: dict[str, Any] = {}
    for key, value_ab in summary_ab.items():
        value_ba = summary_ba[key]
        if isinstance(value_ab, bool):
            summary[key] = bool(value_ab or value_ba)
        elif isinstance(value_ab, (int, np.integer)):
            summary[key] = int(round(0.5 * (int(value_ab) + int(value_ba))))
        elif isinstance(value_ab, (float, np.floating)):
            summary[key] = float(0.5 * (float(value_ab) + float(value_ba)))
        else:
            summary[key] = value_ab
    summary["diagnostic_target"] = "split-half base-weighted halo field"
    summary["direction_ab"] = to_jsonable(summary_ab)
    summary["direction_ba"] = to_jsonable(summary_ba)
    return spectra, summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def run(args: argparse.Namespace) -> Path:
    manifest_row = load_manifest_task(args)
    args.env_mesh = require_power_of_two(int(args.env_mesh), "env_mesh")
    args.recon_mesh = require_power_of_two(int(args.recon_mesh), "recon_mesh")
    dtype = np.float32 if args.dtype == "f4" else np.float64
    component_config = resolve_component_config(args)
    config = StoConfig(
        env_source=str(args.env_source),
        env_field_weight=str(args.env_field_weight),
        weight_source=str(args.weight_source),
        split_seed=int(args.split_seed),
        env_mesh=int(args.env_mesh),
        recon_mesh=int(args.recon_mesh),
        gsr_mpc_h=float(args.gsr),
        thr_mpc_h=float(args.thr),
        lambda_th=float(args.lambda_th),
        env_sample_shift_cells=int(args.env_sample_shift_cells),
        stochastic_kmin=float(args.stochastic_kmin),
        stochastic_kmax=float(args.stochastic_kmax),
        stochastic_nbins=int(args.stochastic_nbins),
        covariance_regularization=float(args.covariance_regularization),
    )

    halo_path = halo_position_path(args.halo_root, args.cosmology, int(args.realization))
    mass_root = args.halo_root if args.mass_root is None else args.mass_root
    mass_path = halo_mass_path(mass_root, args.cosmology, int(args.realization), component_config.mass_filename)
    dm_path = snapshot_dir(args.dm_root, args.cosmology, int(args.realization), args.snapshot_name)
    if not halo_path.exists():
        raise FileNotFoundError(f"Missing halo positions: {halo_path}")
    requires_mass = component_config.base_weight in {"mass", "rankmass"} or int(component_config.mass_bins) > 1
    if requires_mass and not mass_path.exists():
        raise FileNotFoundError(f"Missing halo masses required by {component_config.method}: {mass_path}")
    uses_dm = str(args.env_source) == "dm" or str(args.weight_source) == "dm"
    if uses_dm and not dm_path.exists():
        raise FileNotFoundError(f"Missing DM snapshot dir for legacy DM branch: {dm_path}")
    output_dir = make_output_dir(args)
    required_outputs = [
        output_dir / "env_labels.npy",
        output_dir / "weights.npy",
        output_dir / "post_sto_halo.npz",
        output_dir / "diagnostics.json",
        output_dir / "spectra.npz",
        output_dir / "DONE",
    ]
    if not args.overwrite and args.skip_existing and has_complete_outputs(required_outputs):
        print(f"[skip-existing] complete sto-recon outputs already exist: {output_dir}", flush=True)
        return output_dir
    if not args.overwrite and any(path.exists() for path in required_outputs):
        raise FileExistsError(f"{output_dir} already has sto-recon outputs; pass --overwrite")
    if args.check_only:
        print(f"check-only ok: halo={halo_path} uses_dm={uses_dm} dm={dm_path if uses_dm else None} output={output_dir}", flush=True)
        return output_dir

    halo_positions = np.asarray(np.load(halo_path), dtype=dtype)
    halo_positions = np.mod(halo_positions, BOXSIZE_MPC_H).astype(dtype, copy=False)
    halo_masses: np.ndarray | None = None
    if requires_mass:
        halo_masses = np.asarray(np.load(mass_path), dtype=np.float64)
        if halo_masses.shape[0] != halo_positions.shape[0]:
            raise ValueError(f"halo positions/masses length mismatch: {halo_positions.shape[0]} != {halo_masses.shape[0]}")
    base_weights, base_weight_info = build_base_weights(
        halo_masses,
        base_weight=component_config.base_weight,
        normalization=component_config.mass_weight_normalization,
        rankmass_strength=component_config.rankmass_strength,
        n_halos=int(halo_positions.shape[0]),
    )

    header: dict[str, Any] | None = None
    if uses_dm:
        header = read_snapshot_header(dm_path)
        if abs(float(header["boxsize_mpc_h"]) - BOXSIZE_MPC_H) > 1.0e-6:
            raise ValueError(f"Unexpected DM boxsize: {header['boxsize_mpc_h']}")

    if str(args.env_source) == "dm":
        env_counts, env_paint_info = load_dm_counts(
            dm_path,
            mesh=config.env_mesh,
            boxsize=BOXSIZE_MPC_H,
            chunk_size=int(args.chunk_size),
            dtype=dtype,
            max_particles=None if args.max_dm_particles is None else int(args.max_dm_particles),
        )
        env_paint_info["env_source"] = "dm"
    else:
        env_weights = None if str(args.env_field_weight) == "number" else base_weights
        env_counts, env_paint_info = paint_halo_counts(
            halo_positions,
            mesh=config.env_mesh,
            boxsize=BOXSIZE_MPC_H,
            weights=env_weights,
            source_name=f"halo-{args.env_field_weight}",
        )
        env_paint_info["env_source"] = "halo"
    if component_config.method == "monly":
        # mass-only：不计算潮汐张量（省 6 次 128^3 FFT/irFFT），环境标签置 0
        # 且不进入 component（见 build_component_labels 的 env_enabled=False）。
        labels = np.zeros(halo_positions.shape[0], dtype=np.int8)
        env_info = {
            "env_disabled": True,
            "note": "method=monly skips tidal classification entirely; env dimension is not used.",
        }
    else:
        labels, env_info = tidal_env_labels(
            env_counts,
            halo_positions,
            boxsize=BOXSIZE_MPC_H,
            gsr=config.gsr_mpc_h,
            lambda_th=config.lambda_th,
            sample_shift_cells=config.env_sample_shift_cells,
        )
    mass_bin_labels, mass_split_info = equal_count_mass_bin_labels(
        np.ones(halo_positions.shape[0], dtype=np.float64) if halo_masses is None else halo_masses,
        int(component_config.mass_bins),
    )
    density_bin_labels, density_split_info = local_density_bin_labels(
        env_counts,
        halo_positions,
        boxsize=BOXSIZE_MPC_H,
        radius=component_config.density_radius_mpc_h,
        nbins=component_config.density_bins,
        sample_shift_cells=config.env_sample_shift_cells,
    )
    component_labels, component_names = build_component_labels(
        labels,
        mass_bin_labels,
        mass_bins=int(component_config.mass_bins),
        density_bin_labels=density_bin_labels,
        density_bins=int(component_config.density_bins),
        env_enabled=component_config.method != "monly",
    )

    split_mask, split_info = deterministic_split_mask(
        int(halo_positions.shape[0]),
        seed=config.split_seed,
        cosmology=str(args.cosmology),
        realization=int(args.realization),
    )
    if str(args.weight_source) == "dm":
        if str(args.env_source) == "dm" and config.recon_mesh == config.env_mesh:
            recon_counts = env_counts
            recon_paint_info = env_paint_info | {"reused_env_counts": True}
        else:
            recon_counts, recon_paint_info = load_dm_counts(
                dm_path,
                mesh=config.recon_mesh,
                boxsize=BOXSIZE_MPC_H,
                chunk_size=int(args.chunk_size),
                dtype=dtype,
                max_particles=None if args.max_dm_particles is None else int(args.max_dm_particles),
            )
            recon_paint_info["reused_env_counts"] = False
        recon_paint_info["weight_source"] = "dm"
        target_delta_for_spectra: np.ndarray | None = counts_to_delta(recon_counts)
        weights, solution_by_component, weight_info = solve_component_weights(
            target_delta_for_spectra,
            halo_positions,
            component_labels,
            base_weights=base_weights,
            component_names=component_names,
            mesh=config.recon_mesh,
            boxsize=BOXSIZE_MPC_H,
            thr=config.thr_mpc_h,
            regularization=config.covariance_regularization,
            min_component_count=component_config.min_component_count,
            solution_shrinkage=component_config.solution_shrinkage,
            weight_factor_clip_min=component_config.weight_factor_clip_min,
            weight_factor_clip_max=component_config.weight_factor_clip_max,
        )
        weight_info["weight_source"] = "dm"
    else:
        recon_paint_info = {
            "weight_source": "halo-split",
            "reused_env_counts": False,
            "note": "No DM field is loaded for the halo-only weight solve.",
        }
        target_delta_for_spectra = None
        weights, solution_by_component, weight_info = solve_component_weights_halo_split(
            halo_positions,
            component_labels,
            base_weights=base_weights,
            component_names=component_names,
            split_mask=split_mask,
            mesh=config.recon_mesh,
            boxsize=BOXSIZE_MPC_H,
            thr=config.thr_mpc_h,
            regularization=config.covariance_regularization,
            min_component_count=component_config.min_component_count,
            solution_shrinkage=component_config.solution_shrinkage,
            weight_factor_clip_min=component_config.weight_factor_clip_min,
            weight_factor_clip_max=component_config.weight_factor_clip_max,
        )
    if component_config.method == "monly":
        # monly 分量数 = mass_bins，不能按 (mass_bins, 4, density_bins) reshape。
        solution_by_env = np.zeros(4, dtype=solution_by_component.dtype)
        solution_by_mass = solution_by_component.astype(
            solution_by_component.dtype, copy=True
        )
        solution_by_density = np.ones(1, dtype=solution_by_component.dtype)
    else:
        solution_grid = solution_by_component.reshape(
            int(component_config.mass_bins),
            4,
            int(component_config.density_bins),
        )
        solution_by_env = solution_grid.sum(axis=(0, 2))
        solution_by_mass = solution_grid.sum(axis=(1, 2))
        solution_by_density = solution_grid.sum(axis=(0, 1))
    pre_delta, pre_field_info = paint_halo_delta(
        halo_positions,
        mesh=config.recon_mesh,
        boxsize=BOXSIZE_MPC_H,
        weights=base_weights,
    )
    post_delta, post_field_info = paint_halo_delta(
        halo_positions,
        mesh=config.recon_mesh,
        boxsize=BOXSIZE_MPC_H,
        weights=weights,
    )
    pre_shot_noise = weighted_shot_noise_volume(base_weights, BOXSIZE_MPC_H)
    post_shot_noise = weighted_shot_noise_volume(weights, BOXSIZE_MPC_H)
    if str(args.weight_source) == "dm":
        assert target_delta_for_spectra is not None
        spectra, stochastic_summary = binned_cross_spectra(
            target_delta_for_spectra,
            pre_delta,
            post_delta,
            boxsize=BOXSIZE_MPC_H,
            kmin=config.stochastic_kmin,
            kmax=config.stochastic_kmax,
            nbins=config.stochastic_nbins,
            shot_noise_pre=pre_shot_noise["shot_noise"],
            shot_noise_post=post_shot_noise["shot_noise"],
        )
    else:
        spectra, stochastic_summary = binned_split_proxy_spectra(
            halo_positions,
            base_weights=base_weights,
            post_weights=weights,
            split_mask=split_mask,
            mesh=config.recon_mesh,
            boxsize=BOXSIZE_MPC_H,
            kmin=config.stochastic_kmin,
            kmax=config.stochastic_kmax,
            nbins=config.stochastic_nbins,
        )

    np.save(output_dir / "env_labels.npy", labels)
    np.save(output_dir / "mass_bin_labels.npy", mass_bin_labels.astype("i2"))
    np.save(output_dir / "density_bin_labels.npy", density_bin_labels.astype("i2"))
    np.save(output_dir / "component_labels.npy", component_labels.astype("i2"))
    np.save(output_dir / "weights.npy", weights.astype("f4"))
    np.save(output_dir / "split_half_labels.npy", split_mask.astype("i1"))
    npz_payload = {
        "positions_mpc_h": halo_positions.astype("f4"),
        "weights": weights.astype("f4"),
        "env_labels": labels.astype("i1"),
        "mass_bin_labels": mass_bin_labels.astype("i2"),
        "density_bin_labels": density_bin_labels.astype("i2"),
        "component_labels": component_labels.astype("i2"),
        "base_weights": base_weights.astype("f4"),
        "solution_weight_by_env": solution_by_env.astype("f8"),
        "solution_weight_by_mass": solution_by_mass.astype("f8"),
        "solution_weight_by_density": solution_by_density.astype("f8"),
        "solution_weight_by_component": solution_by_component.astype("f8"),
        "component_names": np.asarray(component_names, dtype="U32"),
        "method": np.asarray(component_config.method),
        "env_source": np.asarray(str(args.env_source)),
        "weight_source": np.asarray(str(args.weight_source)),
        "split_half_labels": split_mask.astype("i1"),
    }
    if halo_masses is not None:
        npz_payload["masses_msun_h"] = halo_masses.astype("f4")
    np.savez_compressed(output_dir / "post_sto_halo.npz", **npz_payload)
    np.savez(output_dir / "spectra.npz", **spectra)

    pipeline_suffix = "haloonly" if not uses_dm else "dm_supervised_legacy"
    diagnostics = {
        "pipeline_name": f"quijote_halo_sto_recon_{component_config.method}_{pipeline_suffix}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
        "manifest": manifest_row,
        "cosmology": str(args.cosmology),
        "realization": int(args.realization),
        "input": {
            "halo_positions": str(halo_path),
            "dm_snapshot_dir": str(dm_path) if uses_dm else None,
            "halo_masses": str(mass_path) if requires_mass else None,
            "snapshot_header": header,
            "n_halos": int(halo_positions.shape[0]),
            "boxsize_mpc_h": BOXSIZE_MPC_H,
        },
        "config": asdict(config),
        "component_config": asdict(component_config),
        "method_note": (
            "STO weighting-scan adapter (variants_v1, derived from the validated "
            "archive_20260613 nenv implementation; see file header). Env comes from "
            "the tidal Hessian unless --env-source=dm or method=monly (monly skips "
            "the env dimension entirely). The weight solve uses per-realization DM "
            "(--weight-source=dm, production standard for this scan) or split-half "
            "halo self-supervision. nenv=N+Env unit base; monly=mass-bins-only with "
            "configurable base weight; m2env/m4env/m4rho3env combine mass/env/density. "
            "No gra-recon is run here."
        ),
        "uses_dm": bool(uses_dm),
        "env_paint": env_paint_info,
        "recon_target_paint": recon_paint_info,
        "split_half": split_info,
        "environment": env_info,
        "mass_split": mass_split_info,
        "density_split": density_split_info,
        "component_names": component_names,
        "base_weights": base_weight_info,
        "weights": weight_info,
        "pre_field": pre_field_info,
        "post_field": post_field_info,
        "shot_noise": {
            "pre": pre_shot_noise,
            "post": post_shot_noise,
            "note": "Poisson expectation V*sum(w^2)/(sum(w)^2), matching the original rcnc.cpp diagnostic.",
        },
        "stochasticity": stochastic_summary,
        "outputs": {
            "env_labels": str(output_dir / "env_labels.npy"),
            "mass_bin_labels": str(output_dir / "mass_bin_labels.npy"),
            "density_bin_labels": str(output_dir / "density_bin_labels.npy"),
            "component_labels": str(output_dir / "component_labels.npy"),
            "weights": str(output_dir / "weights.npy"),
            "split_half_labels": str(output_dir / "split_half_labels.npy"),
            "post_sto_halo": str(output_dir / "post_sto_halo.npz"),
            "spectra": str(output_dir / "spectra.npz"),
            "diagnostics": str(output_dir / "diagnostics.json"),
            "done": str(output_dir / "DONE"),
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(
        output_dir / "DONE",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostics": str(output_dir / "diagnostics.json"),
            "stochasticity_reduced_by_1_minus_r2": stochastic_summary["stochasticity_reduced_by_1_minus_r2"],
            "smooth_stochasticity_reduced": bool(
                weight_info.get("smooth_stochasticity_reduced", weight_info.get("split_proxy_improved", False))
            ),
            "uses_dm": bool(uses_dm),
        },
    )
    print(json.dumps(to_jsonable({"output_dir": output_dir, "stochasticity": stochastic_summary}), indent=2), flush=True)
    return output_dir


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
