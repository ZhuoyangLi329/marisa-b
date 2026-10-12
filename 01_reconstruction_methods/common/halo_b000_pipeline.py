#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
代码大纲
1. 本脚本对 Quijote FoF halo z=0.5 catalog 执行 real-space jaxrecon + jaxpower B_000。
2. 输入是 readfof 已导出的 halo 位置：
   /pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin3p4e13_positions/{cosmology}/realXXX/positions_mpc_h.npy
   也可以通过 input_mode=sto-weighted 读取 sto-recon 的 post_sto_halo.npz，
   此时 Catalog 的 INDWEIGHT 来自 sto-recon 输出。
3. 当前 halo catalog 没有 RSD，因此 reconstruction 实际使用：
   - growth_rate = 0.0
   - data_field = disp
   - random_field = disp
   物理 growth_rate(z=0.5)=0.7625790038 只写入 metadata 供追溯。
4. 单个 task 的执行顺序：
   - 读取 manifest 或命令行指定的 cosmology/realization；
   - 读取 halo positions.npy 并包装成 desi-clustering Catalog；
   - 生成 deterministic uniform random catalog；
   - 用 data + unshifted random 测 pre-recon B_000；
   - 用 jaxrecon PlaneParallelFFTReconstruction 做 real-space reconstruction；
   - 用 reconstructed data + shifted random 测 post-recon B_000；
   - 写 pre_b000.h5、post_b000.h5、metadata.json、DONE。

函数逻辑关系
1. parse_args / load_manifest_task / get_stage_config：解析运行配置。
2. load_halo_positions / make_halo_catalog：读取和包装 Quijote halo catalog。
3. preflight_b000_binner / measure_b000：检查并测量 B_000。
4. run_jaxrecon：复用 UNIT FastPM runner 的 jaxrecon wrapper。
5. run：串联完整 production task。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

# 对 B_000 binner 保持 x64，避免任何潜在的 nmodes 计数溢出。
os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from mpi4py import MPI

from quijote_dm_2d_paperbin_runner_tools import (
    get_package_report,
    patch_jaxpower_sugiyama_s113_dtype,
    shutdown_jax_distributed,
    to_jsonable,
    wrap_positions_to_periodic_box,
    write_and_validate_spectrum,
)
from quijote_dm_2d_paperbin_runner_tools import (
    generate_rank_random_positions,
    make_centered_edges,
    spectrum_bin_summary,
)
from quijote_dm_2d_paperbin_runner_tools import (
    run_jaxrecon,
    summarize_shifts,
)
from marisa_recon_runner_tools import (
    DEFAULT_MARISA_RECON_ROOT,
    activate_marisa_recon_root,
    run_marisa_jaxrecon,
)


PIPELINE_NAME = "jaxrecon_jaxpower_quijote_halo_z0p5_mmin3p4e13"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSITION_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin3p4e13_positions")
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "01_reconstruction_methods/outputs/standard_gra"
DEFAULT_STO_RESULT_ROOT = PROJECT_ROOT / "01_reconstruction_methods/outputs/sto_recon"

QUIJOTE_BOXSIZE_MPC_H = 1000.0
QUIJOTE_REDSHIFT = 0.5
QUIJOTE_OMEGA_M0 = 0.3175
QUIJOTE_PHYSICAL_GROWTH_RATE_Z0P5 = 0.7625790037720044
QUIJOTE_RECON_GROWTH_RATE_REALSPACE = 0.0
QUIJOTE_HALO_BIAS_LEGACY_DEFAULT = 2.2028557
QUIJOTE_PK_KMIN_FUNDAMENTAL = 2.0 * np.pi / QUIJOTE_BOXSIZE_MPC_H


@dataclass(frozen=True)
class StageConfig:
    """B_000 测量 stage 配置。"""

    name: str
    max_particles_per_rank: int | None
    meshsize: int
    kmin: float
    kmax: float
    kstep: float
    edge_mode: str
    nbins: int | None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run Quijote halo z=0.5 real-space jaxrecon + jaxpower B000."
    )
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--cosmology", default="fiducial", help="fiducial 或 LC_p。")
    parser.add_argument("--realization", type=int, default=0, help="Quijote realization 编号。")
    parser.add_argument("--position-root", default=str(DEFAULT_POSITION_ROOT), help="导出 halo positions 根目录。")
    parser.add_argument(
        "--input-mode",
        choices=["positions", "sto-weighted"],
        default="positions",
        help="positions 使用单位权重原始 halo；sto-weighted 使用 post_sto_halo.npz 的权重。",
    )
    parser.add_argument("--sto-result-root", default=str(DEFAULT_STO_RESULT_ROOT), help="sto-recon 输出根目录。")
    parser.add_argument("--sto-label", default=None, help="sto-recon 输出子目录标签。")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT), help="输出结果根目录。")
    parser.add_argument("--manifest", default=None, help="JSONL manifest。")
    parser.add_argument("--task-index", type=int, default=None, help="manifest 中的 0-based 行号。")
    parser.add_argument("--label", default=None, help="诊断子目录标签。")
    parser.add_argument("--dtype", choices=["f4", "f8"], default="f4")
    parser.add_argument("--meshsize", type=int, default=256, help="B_000 meshsize。")
    parser.add_argument("--kmin", type=float, default=0.01)
    parser.add_argument("--kmax", type=float, default=0.20)
    parser.add_argument("--kstep", type=float, default=0.01)
    parser.add_argument(
        "--edge-mode",
        choices=["centered", "linear"],
        default="centered",
        help="centered uses existing center-based bins; linear uses np.linspace(kmin, kmax, nbins + 1).",
    )
    parser.add_argument("--nbins", type=int, default=None, help="Number of bins for --edge-mode linear.")
    parser.add_argument("--basis", choices=["sugiyama", "sugiyama-diagonal"], default="sugiyama")
    parser.add_argument("--mask-edges-policy", choices=["default", "empty"], default="empty")
    parser.add_argument("--mass-threshold-msun-h", type=float, default=3.4e13)
    parser.add_argument("--pipeline-name", default=PIPELINE_NAME)
    parser.add_argument("--expected-nbins", type=int, default=None)
    parser.add_argument("--redshift", type=float, default=QUIJOTE_REDSHIFT, help="写入 Catalog attrs 和 metadata 的 redshift。")
    parser.add_argument(
        "--physical-growth-rate",
        type=float,
        default=QUIJOTE_PHYSICAL_GROWTH_RATE_Z0P5,
        help="物理 f(z)，real-space recon 默认不用，仅写入 metadata。",
    )
    parser.add_argument("--recon-class", choices=["plane", "marisa-plane"], default="plane")
    parser.add_argument("--growth-rate", type=float, default=QUIJOTE_RECON_GROWTH_RATE_REALSPACE)
    parser.add_argument("--bias", type=float, default=None)
    parser.add_argument("--bias-json", default=None, help="Optional canonical bias JSON for provenance and default bias.")
    parser.add_argument("--smoothing-radius", type=float, default=10.0)
    parser.add_argument("--recon-cellsize", type=float, default=8.0)
    parser.add_argument("--marisa-recon-root", default=str(DEFAULT_MARISA_RECON_ROOT), help="本地 marisa-recon checkout 根目录。")
    parser.add_argument("--fnl-loc", type=float, default=0.0, help="marisa-plane 使用的 local PNG fNL。")
    parser.add_argument("--png-p", type=float, default=1.0, help="desilike local PNG mode='b-p' 的 p 参数。")
    parser.add_argument("--png-delta-c", type=float, default=1.686, help="local PNG bias 使用的球坍缩阈值。")
    parser.add_argument("--png-alpha-kmin", type=float, default=1.0e-4, help="alpha(k) 表格最小 k，单位 h/Mpc。")
    parser.add_argument("--png-alpha-kmax", type=float, default=10.0, help="alpha(k) 表格最大 k，单位 h/Mpc。")
    parser.add_argument("--png-alpha-nk", type=int, default=2048, help="alpha(k) 表格采样点数。")
    parser.add_argument("--png-bias-floor", type=float, default=1.0e-3, help="b_eff(k) 的数值保护下限。")
    parser.add_argument("--nran", type=int, default=30)
    parser.add_argument("--measure-pk", action="store_true", help="同时测量 pre/post P0(k) 并写 pre_pk.npz/post_pk.npz。")
    parser.add_argument("--pk-meshsize", type=int, default=256, help="P(k) FFT meshsize。")
    parser.add_argument("--pk-kmin", type=float, default=QUIJOTE_PK_KMIN_FUNDAMENTAL, help="P(k) 第一条 k-bin edge，默认 2*pi/L。")
    parser.add_argument("--pk-kmax", type=float, default=0.30, help="P(k) 最大 k。")
    parser.add_argument("--pk-kstep", type=float, default=QUIJOTE_PK_KMIN_FUNDAMENTAL, help="P(k) bin 宽，默认 2*pi/L。")
    parser.add_argument("--random-source", choices=["pipeline_numpy", "jaxpower"], default="pipeline_numpy")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--data-field", default="disp")
    parser.add_argument("--random-field", default="disp")
    parser.add_argument("--los", choices=["x", "y", "z"], default="z")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.manifest is not None and args.task_index is None:
        parser.error("--manifest requires --task-index")
    if args.task_index is not None and args.manifest is None:
        parser.error("--task-index requires --manifest")
    return args


def load_manifest_task(args: argparse.Namespace) -> dict[str, Any] | None:
    """从 JSONL manifest 读取当前 task，并覆盖 cosmology/realization。"""

    if args.manifest is None:
        return None
    manifest_path = Path(args.manifest).expanduser().resolve()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == int(args.task_index):
                row = json.loads(line)
                args.cosmology = str(row["cosmology"])
                args.realization = int(row["realization"])
                args.position_root = str(row.get("position_root", args.position_root))
                args.input_mode = str(row.get("input_mode", args.input_mode))
                args.sto_result_root = str(row.get("sto_result_root", args.sto_result_root))
                args.sto_label = row.get("sto_label", args.sto_label)
                args.result_root = str(row.get("result_root", args.result_root))
                args.meshsize = int(row.get("meshsize", args.meshsize))
                args.kmin = float(row.get("kmin", args.kmin))
                args.kmax = float(row.get("kmax", args.kmax))
                args.kstep = float(row.get("kstep", args.kstep))
                args.edge_mode = str(row.get("edge_mode", args.edge_mode))
                nbins = row.get("nbins", args.nbins)
                args.nbins = None if nbins is None else int(nbins)
                args.basis = str(row.get("basis", args.basis))
                args.mask_edges_policy = str(row.get("mask_edges_policy", args.mask_edges_policy))
                args.mass_threshold_msun_h = float(row.get("mass_threshold_msun_h", args.mass_threshold_msun_h))
                args.pipeline_name = str(row.get("pipeline_name", args.pipeline_name))
                expected_nbins = row.get("expected_nbins", args.expected_nbins)
                args.expected_nbins = None if expected_nbins is None else int(expected_nbins)
                args.redshift = float(row.get("redshift", args.redshift))
                args.physical_growth_rate = float(row.get("physical_growth_rate", args.physical_growth_rate))
                args.recon_cellsize = float(row.get("recon_cellsize", args.recon_cellsize))
                args.smoothing_radius = float(row.get("smoothing_radius", args.smoothing_radius))
                args.recon_class = str(row.get("recon_class", args.recon_class))
                args.marisa_recon_root = str(row.get("marisa_recon_root", args.marisa_recon_root))
                args.fnl_loc = float(row.get("fnl_loc", args.fnl_loc))
                args.png_p = float(row.get("png_p", args.png_p))
                args.png_delta_c = float(row.get("png_delta_c", args.png_delta_c))
                args.png_alpha_kmin = float(row.get("png_alpha_kmin", args.png_alpha_kmin))
                args.png_alpha_kmax = float(row.get("png_alpha_kmax", args.png_alpha_kmax))
                args.png_alpha_nk = int(row.get("png_alpha_nk", args.png_alpha_nk))
                args.png_bias_floor = float(row.get("png_bias_floor", args.png_bias_floor))
                args.nran = int(row.get("nran", args.nran))
                args.measure_pk = bool(row.get("measure_pk", args.measure_pk))
                args.pk_meshsize = int(row.get("pk_meshsize", args.pk_meshsize))
                args.pk_kmin = float(row.get("pk_kmin", args.pk_kmin))
                args.pk_kmax = float(row.get("pk_kmax", args.pk_kmax))
                args.pk_kstep = float(row.get("pk_kstep", args.pk_kstep))
                args.growth_rate = float(row.get("growth_rate", args.growth_rate))
                bias = row.get("bias", args.bias)
                args.bias = None if bias is None else float(bias)
                args.bias_json = row.get("bias_json_path", args.bias_json)
                args.data_field = str(row.get("data_field", args.data_field))
                args.random_field = str(row.get("random_field", args.random_field))
                # downstream pilot 与正式生产共用结果根目录；manifest generator
                # 只有在 canonical bias/provenance 不匹配时才设置 overwrite=True。
                # 这里逐 task 接受该明确授权，避免临时 fid10 bias 混入 fid500。
                # A manifest may authorize replacement of an interrupted row,
                # while an explicit CLI --overwrite is the operator's scoped
                # retry override.  Never let a stored false value erase that
                # explicit command-line authorization.
                args.overwrite = bool(args.overwrite or row.get("overwrite", False))
                return row | {"manifest_path": str(manifest_path), "manifest_row": int(index)}
    raise IndexError(f"task-index {args.task_index} 超出 manifest 范围: {manifest_path}")


def load_bias_payload(path: str | Path) -> dict[str, Any]:
    """读取 canonical bias JSON，返回推荐 bias 与追溯字段。"""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if "recommended_bias_value" in payload:
        return {
            "bias_value": float(payload["recommended_bias_value"]),
            "bias_json_path": str(resolved),
            "fit_space": str(payload.get("fit_space", "unknown")),
            "bias_model": str(payload.get("fit_model", "unknown")),
            "recommended_bias_source": payload.get("recommended_bias_source"),
            "legacy_desilike_bias": payload.get("legacy_desilike_bias"),
        }
    if "b1_only" in payload and "b1" in payload["b1_only"]:
        return {
            "bias_value": float(payload["b1_only"]["b1"]),
            "bias_json_path": str(resolved),
            "fit_space": str(payload.get("fit_space", "legacy_desilike")),
            "bias_model": "legacy_desilike",
            "recommended_bias_source": "b1_only",
            "legacy_desilike_bias": float(payload["b1_only"]["b1"]),
        }
    raise KeyError(f"Cannot resolve bias from JSON: {resolved}")


def resolve_bias_configuration(args: argparse.Namespace, manifest_row: dict[str, Any] | None) -> dict[str, Any]:
    """统一得到本次实际使用的 bias 数值和 provenance。"""

    payload = None
    if args.bias_json is not None:
        payload = load_bias_payload(args.bias_json)

    if args.bias is not None:
        bias_value = float(args.bias)
        bias_source = "manual_override" if manifest_row is None else str(manifest_row.get("bias_source", "manifest"))
    elif payload is not None:
        bias_value = float(payload["bias_value"])
        bias_source = "bias_json"
    else:
        bias_value = float(QUIJOTE_HALO_BIAS_LEGACY_DEFAULT)
        bias_source = "legacy_default"

    fit_space = "legacy_default"
    bias_model = "legacy_default"
    recommended_bias_source = None
    legacy_desilike_bias = None
    bias_json_path = None
    if payload is not None:
        fit_space = str(payload.get("fit_space", fit_space))
        bias_model = str(payload.get("bias_model", bias_model))
        recommended_bias_source = payload.get("recommended_bias_source")
        legacy_desilike_bias = payload.get("legacy_desilike_bias")
        bias_json_path = payload.get("bias_json_path")
    elif manifest_row is not None:
        fit_space = str(manifest_row.get("fit_space", fit_space))
        bias_model = str(manifest_row.get("bias_model", bias_model))
        recommended_bias_source = manifest_row.get("recommended_bias_source")
        legacy_desilike_bias = manifest_row.get("legacy_desilike_bias")
        bias_json_path = manifest_row.get("bias_json_path")

    return {
        "bias_value": float(bias_value),
        "bias_source": str(bias_source),
        "bias_json_path": None if bias_json_path is None else str(bias_json_path),
        "fit_space": str(fit_space),
        "bias_model": str(bias_model),
        "recommended_bias_source": recommended_bias_source,
        "legacy_desilike_bias": legacy_desilike_bias,
    }


def get_stage_config(args: argparse.Namespace) -> StageConfig:
    """根据 stage 返回 mesh/bin 配置。smoke 会限制 halo 数并降低 mesh。"""

    if args.stage == "smoke":
        return StageConfig(
            name="smoke",
            max_particles_per_rank=50_000,
            meshsize=128,
            kmin=float(args.kmin),
            kmax=min(float(args.kmax), 0.08),
            kstep=float(args.kstep),
            edge_mode=str(args.edge_mode),
            nbins=int(args.nbins) if args.nbins is not None else None,
        )
    if args.stage == "full":
        return StageConfig(
            name="full",
            max_particles_per_rank=None,
            meshsize=int(args.meshsize),
            kmin=float(args.kmin),
            kmax=float(args.kmax),
            kstep=float(args.kstep),
            edge_mode=str(args.edge_mode),
            nbins=int(args.nbins) if args.nbins is not None else None,
        )
    raise ValueError(f"unknown stage {args.stage}")


def initialize_jax_distributed(mpicomm: MPI.Comm) -> None:
    """初始化 JAX distributed；单进程或登录节点不满足条件时跳过。"""

    try:
        jax.distributed.initialize()
    except (RuntimeError, ValueError) as exc:
        log_root(mpicomm, "JAX distributed initialize skipped: %s", exc)
    else:
        log_root(mpicomm, "JAX distributed initialized.")


def log_root(mpicomm: MPI.Comm, message: str, *args: Any) -> None:
    """只在 MPI rank 0 打印日志。"""

    if int(mpicomm.rank) == 0:
        print(message % args if args else message, flush=True)


def ensure_output_dir(path: Path, mpicomm: MPI.Comm) -> Path:
    """rank 0 创建目录，barrier 同步。"""

    if int(mpicomm.rank) == 0:
        path.mkdir(parents=True, exist_ok=True)
    mpicomm.Barrier()
    return path


def measurement_edges(config: StageConfig) -> np.ndarray:
    """Return k-bin edges for the requested measurement convention."""

    if config.edge_mode == "centered":
        return make_centered_edges(config.kmin, config.kmax, config.kstep)
    if config.edge_mode == "linear":
        if config.nbins is None:
            raise ValueError("--edge-mode linear requires --nbins")
        if int(config.nbins) <= 0:
            raise ValueError(f"invalid --nbins={config.nbins}")
        return np.linspace(float(config.kmin), float(config.kmax), int(config.nbins) + 1, dtype=np.float64)
    raise ValueError(f"unknown edge_mode={config.edge_mode}")


def position_path(position_root: str | Path, cosmology: str, realization: int) -> Path:
    """返回 Quijote halo positions.npy 路径。"""

    return Path(position_root).expanduser().resolve() / cosmology / f"real{int(realization):03d}" / "positions_mpc_h.npy"


def sto_halo_path(sto_result_root: str | Path, cosmology: str, realization: int, sto_label: str | None) -> Path:
    """返回 sto-recon 导出的 post_sto_halo.npz 路径。"""

    path = Path(sto_result_root).expanduser().resolve() / cosmology / f"real{int(realization):03d}"
    if sto_label:
        path = path / str(sto_label)
    return path / "post_sto_halo.npz"


def load_halo_positions(path: str | Path, dtype: np.dtype, max_rows: int | None) -> np.ndarray:
    """读取 halo positions.npy，必要时截断用于 smoke。"""

    positions = np.load(path)
    if max_rows is not None:
        positions = positions[: int(max_rows)]
    return np.asarray(positions, dtype=dtype)


def load_sto_weighted_catalog(path: str | Path, dtype: np.dtype, max_rows: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """读取 sto-recon 输出的 weighted halo catalog。

    参数
    ----
    path
        post_sto_halo.npz 路径。
    dtype
        positions 输出 dtype。
    max_rows
        smoke stage 时最多读取的 halo 行数。

    返回
    ----
    tuple
        `(positions, weights, metadata)`；weights 会作为 Catalog 的 INDWEIGHT。
    """

    with np.load(path) as payload:
        positions = np.asarray(payload["positions_mpc_h"], dtype=dtype)
        weights = np.asarray(payload["weights"], dtype=np.float64)
        method = str(np.asarray(payload["method"]).item()) if "method" in payload else "unknown"
        component_names = [str(item) for item in payload["component_names"].tolist()] if "component_names" in payload else None
    if positions.shape[0] != weights.shape[0]:
        raise ValueError(f"sto positions/weights mismatch in {path}: {positions.shape[0]} != {weights.shape[0]}")
    if max_rows is not None:
        positions = positions[: int(max_rows)]
        weights = weights[: int(max_rows)]
    metadata = {
        "source": str(Path(path).expanduser().resolve()),
        "method": method,
        "n_halos_loaded": int(positions.shape[0]),
        "weight_sum": float(np.sum(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_p01": float(np.percentile(weights, 1)),
        "weight_p50": float(np.percentile(weights, 50)),
        "weight_p99": float(np.percentile(weights, 99)),
    }
    if component_names is not None:
        metadata["component_names"] = component_names
    return positions, weights, metadata


def make_halo_catalog(
    positions: np.ndarray,
    los: str,
    mpicomm: MPI.Comm,
    weights: np.ndarray | None = None,
    redshift: float = QUIJOTE_REDSHIFT,
) -> Any:
    """把 Quijote halo 位置和可选权重包装成 desi-clustering Catalog。"""

    from clustering_statistics.tools import Catalog

    attrs = {
        "boxsize": float(QUIJOTE_BOXSIZE_MPC_H),
        "boxcenter": float(QUIJOTE_BOXSIZE_MPC_H) / 2.0,
        "redshift": float(redshift),
    }
    n = int(positions.shape[0])
    if weights is None:
        indweight = np.ones(n, dtype="f4")
    else:
        indweight = np.asarray(weights, dtype="f4")
        if indweight.shape[0] != n:
            raise ValueError(f"INDWEIGHT length {indweight.shape[0]} != positions length {n}")
    catalog = Catalog(
        {
            "POSITION": np.asarray(positions, dtype="f4"),
            "INDWEIGHT": indweight,
        },
        attrs=attrs,
        mpicomm=mpicomm,
    )
    catalog.attrs["los"] = los
    return catalog


def production_random_seed(pipeline_name: str, cosmology: str, realization: int, stage: str, nran: int) -> int:
    """生成 Quijote halo pipeline 专用 deterministic random seed。"""

    text = f"{pipeline_name}|{cosmology}|real{int(realization):03d}|{stage}|nran={int(nran)}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def validate_jax_runtime_backend() -> dict[str, Any]:
    """Record the actual JAX backend and enforce an optional production gate."""

    default_backend = str(jax.default_backend())
    devices = [
        {
            "platform": str(device.platform),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    required = os.environ.get("QUIJOTE_REQUIRED_JAX_BACKEND")
    passed = required is None or default_backend == required
    result = {
        "default_backend": default_backend,
        "devices": devices,
        "required_backend": required,
        "required_backend_pass": bool(passed),
    }
    if not passed:
        raise RuntimeError(f"JAX backend contract failure: {result}")
    if required == "gpu" and not any(item["platform"] == "gpu" for item in devices):
        raise RuntimeError(f"JAX GPU device contract failure: {result}")
    return result


def apply_scoped_manifest_runtime_contract(manifest_row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Make the fNL250 timeout repair fail closed even from a spooled old wrapper."""

    if manifest_row is None or manifest_row.get("extension_role") != "fnl250_standard_timeout_tail_repair":
        return None
    certified = (
        Path(__file__).resolve().parents[2]
        / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
        / "matrices/baseline_pre_standard_matrix.npz"
    ).resolve()
    os.environ["QUIJOTE_REQUIRED_JAX_BACKEND"] = "gpu"
    os.environ["QUIJOTE_CERTIFIED_GEOMETRY_NPZ"] = str(certified)
    return {
        "extension_role": str(manifest_row["extension_role"]),
        "required_jax_backend": "gpu",
        "certified_geometry_npz": str(certified),
        "source": "runner-enforced from the exact repair manifest role; independent of Slurm wrapper spool time",
    }


def resolve_mask_edges(policy: str) -> list[str] | None:
    """把 manifest 中的 mask_edges_policy 转为 jaxpower 参数。"""

    if policy == "default":
        return None
    if policy == "empty":
        return []
    raise ValueError(f"unknown mask_edges_policy={policy}")


def preflight_b000_binner(
    *,
    boxsize: float,
    meshsize: int,
    edges: np.ndarray,
    basis: str,
    mask_edges_policy: str,
    expected_nbins: int | None,
    mpicomm: MPI.Comm,
) -> dict[str, Any]:
    """检查 B000 binner 的 dtype、bin 数和 nmodes。"""

    from jaxpower.mesh import MeshAttrs
    from jaxpower.mesh3 import BinMesh3SpectrumPoles

    mattrs = MeshAttrs(
        boxsize=float(boxsize) * np.ones(3, dtype="f4"),
        boxcenter=(float(boxsize) / 2.0) * np.ones(3, dtype="f4"),
        meshsize=int(meshsize),
    )
    binner = BinMesh3SpectrumPoles(
        mattrs,
        edges=edges,
        basis=basis,
        ells=[(0, 0, 0)],
        mask_edges=resolve_mask_edges(mask_edges_policy),
    )
    nmodes = np.asarray(binner.nmodes[0])
    xavg = np.asarray(binner.xavg)
    bedges = np.asarray(binner.edges)
    dtype = np.dtype(nmodes.dtype)
    finite = bool(np.all(np.isfinite(nmodes)))
    all_positive = bool(np.all(nmodes > 0))
    nbins = int(nmodes.size)
    if dtype.itemsize < 8 or not finite or not all_positive:
        raise RuntimeError(
            "Quijote halo B000 binner preflight failed: "
            f"basis={basis}, dtype={dtype}, min={int(np.min(nmodes))}, "
            f"max={int(np.max(nmodes))}, all_positive={all_positive}. "
            "Set JAX_ENABLE_X64=1 before importing JAX/jaxpower."
        )
    if expected_nbins is not None and nbins != int(expected_nbins):
        raise RuntimeError(f"B000 binner bin-count mismatch: got {nbins}, expected {expected_nbins}")
    if basis == "sugiyama" and mask_edges_policy == "default" and not bool(np.all(xavg[:, 0] <= xavg[:, 1] + 1e-12)):
        raise RuntimeError("Expected sugiyama default upper-triangle bins with k1 <= k2.")
    summary = {
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "basis": basis,
        "mask_edges_policy": mask_edges_policy,
        "nmodes_dtype": str(dtype),
        "nmodes_itemsize": int(dtype.itemsize),
        "nmodes_min": int(np.min(nmodes)),
        "nmodes_max": int(np.max(nmodes)),
        "nmodes_all_finite": finite,
        "nmodes_all_positive": all_positive,
        "nbins": nbins,
        "xavg": xavg,
        "edges": bedges,
        "nmodes": nmodes,
    }
    log_root(
        mpicomm,
        "Quijote halo B000 binner preflight ok: basis=%s mask=%s nbins=%d nmodes dtype=%s min=%d max=%d",
        basis,
        mask_edges_policy,
        nbins,
        summary["nmodes_dtype"],
        summary["nmodes_min"],
        summary["nmodes_max"],
    )
    return summary


def validate_measured_b000_geometry(
    spectrum: Any,
    certified: dict[str, np.ndarray],
    *,
    label: str,
) -> dict[str, Any]:
    """Require the actual catalog measurement to match its certified binner.

    The standalone preflight and the catalog measurement can take different
    low-level device paths.  A Fourier radius lying within one float32 ulp of a
    bin edge can then migrate between adjacent shells on a different GPU node
    pool.  Comparing the measured spectrum itself prevents a preflight-only
    false pass and forbids mixing different mode sets in one covariance.
    """

    leaf = None
    for _labels, candidate in spectrum.items(level=1):
        leaf = candidate
        break
    if leaf is None:
        raise ValueError(f"{label}: B000 spectrum has no leaves")
    coords = leaf.coords()
    measured_k = np.asarray(coords.get("k")) if isinstance(coords, dict) else None
    if measured_k is None:
        raise ValueError(f"{label}: B000 spectrum has no k coordinates")
    measured_nmodes = np.asarray(leaf.values("nmodes"))
    certified_k = np.asarray(certified["k"])
    certified_nmodes = np.asarray(certified["nmodes"])
    if measured_k.shape != certified_k.shape or measured_nmodes.shape != certified_nmodes.shape:
        raise ValueError(
            f"{label}: B000 geometry shape drift: k={measured_k.shape}/{certified_k.shape}, "
            f"nmodes={measured_nmodes.shape}/{certified_nmodes.shape}"
        )
    absolute = np.abs(measured_k - certified_k)
    relative = absolute / np.maximum(np.abs(certified_k), 1.0e-30)
    failed_k = ~np.isclose(measured_k, certified_k, rtol=5.0e-5, atol=5.0e-8)
    nmodes_equal = bool(np.array_equal(measured_nmodes, certified_nmodes))
    result = {
        "pass": bool(not np.any(failed_k) and nmodes_equal),
        "k_failed_coordinate_count": int(np.count_nonzero(failed_k)),
        "k_maximum_absolute_difference": float(np.max(absolute)),
        "k_maximum_relative_difference": float(np.max(relative)),
        "nmodes_exact": nmodes_equal,
        "nmodes_different_bin_count": int(np.count_nonzero(measured_nmodes != certified_nmodes)),
        "nmodes_maximum_absolute_difference": int(
            np.max(np.abs(measured_nmodes.astype(np.int64) - certified_nmodes.astype(np.int64)))
        ),
    }
    if not result["pass"]:
        raise RuntimeError(f"{label}: measured B000 geometry differs from certified GPU reference: {result}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_certified_measurement_geometry() -> dict[str, Any] | None:
    """Load an explicitly requested, immutable GPU geometry reference."""

    value = os.environ.get("QUIJOTE_CERTIFIED_GEOMETRY_NPZ")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    required = ("b000_k", "b000_k_edges", "b000_nmodes", "pk_k", "pk_edges", "pk_nmodes")
    with np.load(path, allow_pickle=False) as stream:
        missing = [name for name in required if name not in stream.files]
        if missing:
            raise KeyError(f"certified geometry {path} is missing {missing}")
        arrays = {name: np.asarray(stream[name]) for name in required}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "b000": {
            "k": arrays["b000_k"],
            "k_edges": arrays["b000_k_edges"],
            "nmodes": arrays["b000_nmodes"],
        },
        "pk": {
            "k": arrays["pk_k"],
            "edges": arrays["pk_edges"],
            "nmodes": arrays["pk_nmodes"],
        },
    }


def preflight_pk_binner(
    *,
    boxsize: float,
    meshsize: int,
    edges: np.ndarray,
) -> dict[str, Any]:
    """Build the certified P0(k) bin geometry independently of catalog paint."""

    from jaxpower.mesh import MeshAttrs
    from jaxpower.mesh2 import BinMesh2SpectrumPoles

    mattrs = MeshAttrs(
        boxsize=float(boxsize) * np.ones(3, dtype="f4"),
        boxcenter=(float(boxsize) / 2.0) * np.ones(3, dtype="f4"),
        meshsize=int(meshsize),
    )
    binner = BinMesh2SpectrumPoles(mattrs, edges=np.asarray(edges, dtype="f8"), ells=(0,))
    xavg = np.asarray(binner.xavg)
    nmodes = np.asarray(binner.nmodes)
    if nmodes.ndim > 1 and nmodes.shape[0] == 1:
        nmodes = nmodes[0]
    if xavg.ndim != 1 or nmodes.ndim != 1 or xavg.shape != nmodes.shape:
        raise RuntimeError(f"P0 binner preflight shape drift: k={xavg.shape}, nmodes={nmodes.shape}")
    return {
        "xavg": xavg,
        "nmodes": nmodes,
        "edges": np.asarray(binner.edges),
        "nbins": int(xavg.size),
        "nmodes_dtype": str(nmodes.dtype),
        "nmodes_min": int(np.min(nmodes)),
        "nmodes_max": int(np.max(nmodes)),
    }


def validate_measured_pk_geometry(
    spectrum: Any,
    certified: dict[str, np.ndarray],
    *,
    label: str,
) -> dict[str, Any]:
    """Require the actual P0(k) mode set to match its certified binner."""

    pole = next(iter(spectrum))
    measured_k = np.asarray(pole.coords("k"))
    measured_nmodes = np.asarray(pole.values("nmodes"))
    certified_k = np.asarray(certified["k"])
    certified_nmodes = np.asarray(certified["nmodes"])
    if measured_k.shape != certified_k.shape or measured_nmodes.shape != certified_nmodes.shape:
        raise ValueError(
            f"{label}: P0 geometry shape drift: k={measured_k.shape}/{certified_k.shape}, "
            f"nmodes={measured_nmodes.shape}/{certified_nmodes.shape}"
        )
    absolute = np.abs(measured_k - certified_k)
    relative = absolute / np.maximum(np.abs(certified_k), 1.0e-30)
    failed_k = ~np.isclose(measured_k, certified_k, rtol=5.0e-5, atol=5.0e-8)
    nmodes_equal = bool(np.array_equal(measured_nmodes, certified_nmodes))
    result = {
        "pass": bool(not np.any(failed_k) and nmodes_equal),
        "k_failed_coordinate_count": int(np.count_nonzero(failed_k)),
        "k_maximum_absolute_difference": float(np.max(absolute)),
        "k_maximum_relative_difference": float(np.max(relative)),
        "nmodes_exact": nmodes_equal,
        "nmodes_different_bin_count": int(np.count_nonzero(measured_nmodes != certified_nmodes)),
        "nmodes_maximum_absolute_difference": int(
            np.max(np.abs(measured_nmodes.astype(np.int64) - certified_nmodes.astype(np.int64)))
        ),
    }
    if not result["pass"]:
        raise RuntimeError(f"{label}: measured P0 geometry differs from certified GPU reference: {result}")
    return result


@contextmanager
def jaxpower_runtime_compat(mpicomm: MPI.Comm):
    """Bridge two runtime API mismatches in the 20260321 environment.

    The 20260321 cosmodesi environment calls ``jaxpower.get_mesh_attrs`` with
    NumPy positions and ``check=True`` but without forwarding its already known
    MPI communicator.  Current jaxpower then needs that communicator to reduce
    the catalog extent.  In addition, current ``lsstypes.clone`` no longer
    broadcasts the scalar periodic-box normalization across spectrum bins, while
    ``clustering_statistics.compute_box_mesh[23]_spectrum`` still relies on that
    behavior.  Recreate the former scalar broadcast for the single-pole P0/B000
    measurements used here.  Both patches are restored after each measurement;
    all mesh, binning, estimator, and reconstruction settings remain unchanged.
    """

    from jax import numpy as jnp
    import jaxpower
    from jaxpower import mesh as jaxpower_mesh

    original_get_mesh_attrs = jaxpower_mesh.get_mesh_attrs
    original_box2_normalization = jaxpower.compute_box2_normalization
    original_box3_normalization = jaxpower.compute_box3_normalization

    @wraps(original_get_mesh_attrs)
    def get_mesh_attrs_with_mpicomm(*args, **kwargs):
        if kwargs.get("mpicomm") is None:
            kwargs["mpicomm"] = mpicomm
        return original_get_mesh_attrs(*args, **kwargs)

    def broadcast_scalar_normalization(original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            value = original(*args, **kwargs)
            binning = kwargs.get("bin")
            if binning is None:
                raise RuntimeError("Periodic-box normalization compatibility requires bin=...")
            target = (int(np.shape(binning.xavg)[0]),)

            def broadcast_leaf(leaf):
                if isinstance(leaf, list):
                    return [broadcast_leaf(item) for item in leaf]
                if isinstance(leaf, tuple):
                    return tuple(broadcast_leaf(item) for item in leaf)
                shape = tuple(getattr(leaf, "shape", ()))
                if not shape or int(np.prod(shape)) == 1:
                    return jnp.broadcast_to(jnp.reshape(jnp.asarray(leaf), ()), target)
                return leaf

            return broadcast_leaf(value)

        return wrapped

    jaxpower_mesh.get_mesh_attrs = get_mesh_attrs_with_mpicomm
    jaxpower.compute_box2_normalization = broadcast_scalar_normalization(original_box2_normalization)
    jaxpower.compute_box3_normalization = broadcast_scalar_normalization(original_box3_normalization)
    try:
        yield
    finally:
        jaxpower_mesh.get_mesh_attrs = original_get_mesh_attrs
        jaxpower.compute_box2_normalization = original_box2_normalization
        jaxpower.compute_box3_normalization = original_box3_normalization


def measure_b000(
    get_data,
    *,
    boxsize: float,
    meshsize: int,
    edges: np.ndarray,
    los: str,
    basis: str,
    mask_edges_policy: str,
    cache: dict[str, Any],
    mpicomm: MPI.Comm,
):
    """用 jaxpower 测量可配置 basis 的 B000。"""

    from clustering_statistics.spectrum3_tools import compute_box_mesh3_spectrum

    mattrs = {
        "boxsize": boxsize * np.ones(3, dtype="f4"),
        "boxcenter": (boxsize / 2.0) * np.ones(3, dtype="f4"),
        "meshsize": int(meshsize),
    }
    with jaxpower_runtime_compat(mpicomm):
        return compute_box_mesh3_spectrum(
            get_data,
            mattrs=mattrs,
            basis=basis,
            ells=[(0, 0, 0)],
            edges=edges,
            los=los,
            mask_edges=resolve_mask_edges(mask_edges_policy),
            cache=cache,
        )


def make_pk_edges(kmin: float, kmax: float, kstep: float) -> np.ndarray:
    """构造 P(k) bin edges；默认口径是 fundamental mode spacing。"""

    if kstep <= 0:
        raise ValueError("P(k) kstep must be positive")
    if kmax <= kmin:
        raise ValueError("P(k) kmax must be larger than kmin")
    nbin = int(np.floor((float(kmax) - float(kmin)) / float(kstep) + 0.5))
    edges = float(kmin) + float(kstep) * np.arange(nbin + 1, dtype="f8")
    if edges.size < 2:
        raise ValueError("Need at least one P(k) bin")
    return edges


def measure_pk(
    get_data,
    *,
    boxsize: float,
    meshsize: int,
    edges: np.ndarray,
    los: str,
    cache: dict[str, Any],
    mpicomm: MPI.Comm,
):
    """用 jaxpower 测量 P0(k)，支持 reconstructed shifted randoms。"""

    from clustering_statistics.spectrum2_tools import compute_box_mesh2_spectrum

    mattrs = {
        "boxsize": boxsize * np.ones(3, dtype="f4"),
        "boxcenter": (boxsize / 2.0) * np.ones(3, dtype="f4"),
        "meshsize": int(meshsize),
    }
    with jaxpower_runtime_compat(mpicomm):
        return compute_box_mesh2_spectrum(
            get_data,
            mattrs=mattrs,
            ells=(0,),
            edges=np.asarray(edges, dtype="f8"),
            los=los,
            cache=cache,
        )


def pk_spectrum_summary(spectrum) -> dict[str, Any]:
    """抽取 P(k) 输出的轻量摘要。"""

    pole = next(iter(spectrum))
    k = np.asarray(pole.coords("k"), dtype="f8")
    pk = np.asarray(pole.value(), dtype="f8")
    nmodes = np.asarray(pole.values("nmodes"))
    num_shotnoise = np.asarray(pole.values("num_shotnoise"))
    return {
        "n_bins": int(k.size),
        "k_min": float(np.nanmin(k)),
        "k_max": float(np.nanmax(k)),
        "pk_min": float(np.nanmin(pk)),
        "pk_max": float(np.nanmax(pk)),
        "nmodes_dtype": str(nmodes.dtype),
        "nmodes_min": int(np.nanmin(nmodes)),
        "nmodes_max": int(np.nanmax(nmodes)),
        "num_shotnoise_sum": float(np.nansum(num_shotnoise)),
    }


def write_pk_npz(
    filename: Path,
    spectrum,
    *,
    edges: np.ndarray,
    boxsize: float,
    meshsize: int,
    mpicomm: MPI.Comm,
) -> dict[str, Any]:
    """rank 0 写 P0(k) 的 npz 文件，并返回摘要。"""

    jax.block_until_ready(spectrum)
    summary = pk_spectrum_summary(spectrum)
    if int(mpicomm.rank) == 0:
        pole = next(iter(spectrum))
        filename.parent.mkdir(parents=True, exist_ok=True)
        tmp = filename.with_suffix(filename.suffix + ".tmp")
        np.savez_compressed(
            tmp,
            k=np.asarray(pole.coords("k"), dtype="f8"),
            pk=np.asarray(pole.value(), dtype="f8"),
            nmodes=np.asarray(pole.values("nmodes")),
            norm=np.asarray(pole.values("norm")),
            num_shotnoise=np.asarray(pole.values("num_shotnoise")),
            edges=np.asarray(edges, dtype="f8"),
            boxsize=np.array(float(boxsize), dtype="f8"),
            meshsize=np.array(int(meshsize), dtype="i8"),
        )
        os.replace(tmp.with_suffix(tmp.suffix + ".npz"), filename)
    mpicomm.Barrier()
    return summary


def write_metadata(filename: Path, payload: dict[str, Any], mpicomm: MPI.Comm) -> None:
    """写 JSON metadata。"""

    if int(mpicomm.rank) == 0:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    mpicomm.Barrier()


def write_done_marker(filename: Path, metadata: dict[str, Any], mpicomm: MPI.Comm) -> None:
    """写 DONE 标记文件。"""

    if int(mpicomm.rank) == 0:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_name": metadata.get("pipeline_name"),
            "cosmology": metadata.get("cosmology"),
            "realization": metadata.get("realization"),
            "pre_b000": str(metadata.get("outputs", {}).get("pre_b000", "")),
            "post_b000": str(metadata.get("outputs", {}).get("post_b000", "")),
            "metadata": str(metadata.get("outputs", {}).get("metadata", "")),
        }
        tmp = filename.with_suffix(filename.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
        os.replace(tmp, filename)
    mpicomm.Barrier()


def run(args: argparse.Namespace, config: StageConfig, mpicomm: MPI.Comm) -> None:
    """串联 Quijote halo 读取、recon、B_000 测量。"""

    manifest_row = load_manifest_task(args)
    from entrypoint import validate_gra_paths
    validate_gra_paths(args)
    scoped_runtime_contract = apply_scoped_manifest_runtime_contract(manifest_row)
    if args.recon_class == "marisa-plane":
        activate_marisa_recon_root(args.marisa_recon_root)

    from clustering_statistics.tools import Catalog

    bias_config = resolve_bias_configuration(args, manifest_row)
    args.bias = float(bias_config["bias_value"])
    # Manifest rows can override mesh/bin/recon parameters; rebuild stage config
    # after loading the row so preflight and measurements use the requested bins.
    config = get_stage_config(args)
    in_path = position_path(args.position_root, args.cosmology, args.realization)
    sto_path: Path | None = None
    if args.input_mode == "positions" and not in_path.exists():
        raise FileNotFoundError(f"halo positions 不存在: {in_path}")
    if args.input_mode == "sto-weighted":
        sto_path = sto_halo_path(args.sto_result_root, args.cosmology, args.realization, args.sto_label)
        if not sto_path.exists():
            raise FileNotFoundError(f"sto weighted catalog 不存在: {sto_path}")

    dtype = np.float32 if args.dtype == "f4" else np.float64
    edges = measurement_edges(config)
    binner_preflight = preflight_b000_binner(
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        meshsize=config.meshsize,
        edges=edges,
        basis=args.basis,
        mask_edges_policy=args.mask_edges_policy,
        expected_nbins=args.expected_nbins,
        mpicomm=mpicomm,
    )
    if args.check_only:
        log_root(mpicomm, "check-only finished; Quijote halo B000 binner is safe.")
        return

    label = args.label
    output_dir = (
        Path(args.result_root).expanduser().resolve()
        / config.name
        / args.cosmology
        / f"real{int(args.realization):03d}"
    )
    if label:
        output_dir = output_dir / label
    output_dir = ensure_output_dir(output_dir, mpicomm)

    pre_fn = output_dir / "pre_b000.h5"
    post_fn = output_dir / "post_b000.h5"
    pre_pk_fn = output_dir / "pre_pk.npz"
    post_pk_fn = output_dir / "post_pk.npz"
    metadata_fn = output_dir / "metadata.json"
    expected_outputs = [pre_fn, post_fn, metadata_fn, output_dir / "DONE"]
    if args.measure_pk:
        expected_outputs.extend([pre_pk_fn, post_pk_fn])
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"{output_dir} 已有输出，加 --overwrite 覆盖。")
    if args.overwrite:
        done_path = output_dir / "DONE"
        if int(mpicomm.rank) == 0 and done_path.exists():
            invalidated = output_dir / (
                f".DONE.invalidated-{os.environ.get('SLURM_JOB_ID', 'manual')}-"
                f"{os.environ.get('SLURM_ARRAY_TASK_ID', 'na')}"
            )
            os.replace(done_path, invalidated)
        mpicomm.Barrier()
    jax_runtime = validate_jax_runtime_backend()
    certified_geometry = load_certified_measurement_geometry()

    input_catalog_meta: dict[str, Any]
    input_weights: np.ndarray | None
    if args.input_mode == "sto-weighted":
        assert sto_path is not None
        positions, input_weights, input_catalog_meta = load_sto_weighted_catalog(
            sto_path,
            dtype=dtype,
            max_rows=config.max_particles_per_rank,
        )
    else:
        positions = load_halo_positions(in_path, dtype=dtype, max_rows=config.max_particles_per_rank)
        input_weights = None
        input_catalog_meta = {
            "source": str(in_path),
            "method": "unit-weight positions",
            "n_halos_loaded": int(positions.shape[0]),
            "weight_sum": float(positions.shape[0]),
            "weight_mean": 1.0,
            "weight_std": 0.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
        }
    positions = wrap_positions_to_periodic_box(
        positions,
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        dtype=dtype,
        label="quijote_halo_positions_realspace",
        mpicomm=mpicomm,
    )
    local_count = int(positions.shape[0])
    global_count = int(mpicomm.allreduce(local_count, op=MPI.SUM))
    log_label = f"{args.cosmology}_real{int(args.realization):03d}"
    log_root(
        mpicomm,
        "[%s] loaded %d halos with input_mode=%s from %s",
        log_label,
        global_count,
        args.input_mode,
        sto_path if args.input_mode == "sto-weighted" else in_path,
    )

    data = make_halo_catalog(
        positions,
        los=args.los,
        mpicomm=mpicomm,
        weights=input_weights,
        redshift=float(args.redshift),
    )

    random_seed = (
        int(args.random_seed)
        if args.random_seed is not None
        else production_random_seed(args.pipeline_name, args.cosmology, int(args.realization), config.name, int(args.nran))
    )
    random_positions, random_input_meta = generate_rank_random_positions(
        local_data_count=local_count,
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        seed=random_seed,
        dtype=dtype,
        nran=int(args.nran),
        mpicomm=mpicomm,
    )
    random_positions = wrap_positions_to_periodic_box(
        random_positions,
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        dtype=dtype,
        label="randoms_pre",
        mpicomm=mpicomm,
    )
    randoms_pre = Catalog(
        {
            "POSITION": random_positions,
            "INDWEIGHT": np.ones(random_positions.shape[0], dtype="f4"),
        },
        attrs=data.attrs,
        mpicomm=mpicomm,
    )

    cache: dict[str, Any] = {}
    pk_cache: dict[str, Any] = {}
    pk_edges = make_pk_edges(float(args.pk_kmin), float(args.pk_kmax), float(args.pk_kstep)) if args.measure_pk else None
    pk_binner_preflight = (
        preflight_pk_binner(
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            meshsize=int(args.pk_meshsize),
            edges=pk_edges,
        )
        if args.measure_pk and pk_edges is not None
        else None
    )
    pre_pk_summary: dict[str, Any] | None = None
    post_pk_summary: dict[str, Any] | None = None
    pre_pk_geometry_gate: dict[str, Any] | None = None
    post_pk_geometry_gate: dict[str, Any] | None = None
    log_root(mpicomm, "[%s] measuring pre B000", log_label)
    pre_spectrum = measure_b000(
        lambda: {"data": data, "shifted": randoms_pre},
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        meshsize=config.meshsize,
        edges=edges,
        los=args.los,
        basis=args.basis,
        mask_edges_policy=args.mask_edges_policy,
        cache=cache,
        mpicomm=mpicomm,
    )
    pre_geometry_gate = (
        {"required": False, "pass": True}
        if certified_geometry is None
        else {
            "required": True,
            **validate_measured_b000_geometry(
                pre_spectrum,
                certified_geometry["b000"],
                label=f"{log_label}/pre",
            ),
        }
    )
    pre_bins = spectrum_bin_summary(pre_spectrum)
    pre_validation = write_and_validate_spectrum(pre_fn, pre_spectrum, mpicomm)
    del pre_spectrum
    jax.clear_caches()
    if args.measure_pk:
        assert pk_edges is not None
        log_root(mpicomm, "[%s] measuring pre P0(k)", log_label)
        pre_pk = measure_pk(
            lambda: {"data": data, "shifted": randoms_pre},
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            meshsize=int(args.pk_meshsize),
            edges=pk_edges,
            los=args.los,
            cache=pk_cache,
            mpicomm=mpicomm,
        )
        assert pk_binner_preflight is not None
        pre_pk_geometry_gate = (
            {"required": False, "pass": True}
            if certified_geometry is None
            else {
                "required": True,
                **validate_measured_pk_geometry(
                    pre_pk,
                    certified_geometry["pk"],
                    label=f"{log_label}/pre_pk",
                ),
            }
        )
        pre_pk_summary = write_pk_npz(
            pre_pk_fn,
            pre_pk,
            edges=pk_edges,
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            meshsize=int(args.pk_meshsize),
            mpicomm=mpicomm,
        )
        del pre_pk
        jax.clear_caches()

    log_root(
        mpicomm,
        "[%s] %s real-space growth_rate=%.6g bias=%.7f fnl=%.6g fields=(%s,%s)",
        log_label,
        args.recon_class,
        float(args.growth_rate),
        float(args.bias),
        float(args.fnl_loc),
        args.data_field,
        args.random_field,
    )
    if args.recon_class == "marisa-plane":
        data_rec_positions, shifted_rec_positions, recon_meta = run_marisa_jaxrecon(
            data_catalog=data,
            random_positions=random_positions,
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            growth_rate=float(args.growth_rate),
            bias=float(args.bias),
            smoothing_radius=float(args.smoothing_radius),
            recon_cellsize=float(args.recon_cellsize),
            recon_class=args.recon_class,
            data_field=args.data_field,
            random_field=args.random_field,
            nran=int(args.nran),
            random_source=args.random_source,
            los=args.los,
            dtype=dtype,
            mpicomm=mpicomm,
            marisa_recon_root=args.marisa_recon_root,
            fnl_loc=float(args.fnl_loc),
            png_p=float(args.png_p),
            png_delta_c=float(args.png_delta_c),
            png_alpha_kmin=float(args.png_alpha_kmin),
            png_alpha_kmax=float(args.png_alpha_kmax),
            png_alpha_nk=int(args.png_alpha_nk),
            png_alpha_redshift=float(args.redshift),
            png_alpha_cosmology="quijote",
            png_bias_floor=float(args.png_bias_floor),
        )
    else:
        with jaxpower_runtime_compat(mpicomm):
            data_rec_positions, shifted_rec_positions, recon_meta = run_jaxrecon(
                data_catalog=data,
                random_positions=random_positions,
                boxsize=QUIJOTE_BOXSIZE_MPC_H,
                growth_rate=float(args.growth_rate),
                bias=float(args.bias),
                smoothing_radius=float(args.smoothing_radius),
                recon_cellsize=float(args.recon_cellsize),
                recon_class=args.recon_class,
                data_field=args.data_field,
                random_field=args.random_field,
                nran=int(args.nran),
                random_source=args.random_source,
                los=args.los,
                dtype=dtype,
                mpicomm=mpicomm,
            )
    recon_meta["physical_growth_rate_not_used_for_realspace_recon"] = float(args.physical_growth_rate)
    recon_meta["realspace_growth_rate_policy"] = "growth_rate is set to 0.0 because input halo positions have no RSD"
    recon_meta["bias_value"] = float(args.bias)
    recon_meta["fnl_loc"] = float(args.fnl_loc)
    recon_meta["bias_source"] = str(bias_config["bias_source"])
    recon_meta["bias_json_path"] = bias_config["bias_json_path"]
    recon_meta["bias_model"] = str(bias_config["bias_model"])
    recon_meta["fit_space"] = str(bias_config["fit_space"])
    recon_meta["recommended_bias_source"] = bias_config["recommended_bias_source"]
    recon_meta["input_catalog_mode"] = str(args.input_mode)
    del positions, random_positions

    data_rec = Catalog(
        {
            "POSITION": data_rec_positions,
            "INDWEIGHT": np.asarray(data["INDWEIGHT"], dtype="f4"),
        },
        attrs=data.attrs,
        mpicomm=mpicomm,
    )
    shifted_rec = Catalog(
        {
            "POSITION": shifted_rec_positions,
            "INDWEIGHT": np.ones(shifted_rec_positions.shape[0], dtype="f4"),
        },
        attrs=data.attrs,
        mpicomm=mpicomm,
    )
    shifted_count = int(mpicomm.allreduce(int(shifted_rec_positions.shape[0]), op=MPI.SUM))
    del data_rec_positions, shifted_rec_positions

    log_root(mpicomm, "[%s] measuring post B000", log_label)
    post_spectrum = measure_b000(
        lambda: {"data": data_rec, "shifted": shifted_rec},
        boxsize=QUIJOTE_BOXSIZE_MPC_H,
        meshsize=config.meshsize,
        edges=edges,
        los=args.los,
        basis=args.basis,
        mask_edges_policy=args.mask_edges_policy,
        cache=cache,
        mpicomm=mpicomm,
    )
    post_geometry_gate = (
        {"required": False, "pass": True}
        if certified_geometry is None
        else {
            "required": True,
            **validate_measured_b000_geometry(
                post_spectrum,
                certified_geometry["b000"],
                label=f"{log_label}/post",
            ),
        }
    )
    post_bins = spectrum_bin_summary(post_spectrum)
    post_validation = write_and_validate_spectrum(post_fn, post_spectrum, mpicomm)
    del post_spectrum
    jax.clear_caches()
    if args.measure_pk:
        assert pk_edges is not None
        log_root(mpicomm, "[%s] measuring post P0(k)", log_label)
        post_pk = measure_pk(
            lambda: {"data": data_rec, "shifted": shifted_rec},
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            meshsize=int(args.pk_meshsize),
            edges=pk_edges,
            los=args.los,
            cache=pk_cache,
            mpicomm=mpicomm,
        )
        assert pk_binner_preflight is not None
        post_pk_geometry_gate = (
            {"required": False, "pass": True}
            if certified_geometry is None
            else {
                "required": True,
                **validate_measured_pk_geometry(
                    post_pk,
                    certified_geometry["pk"],
                    label=f"{log_label}/post_pk",
                ),
            }
        )
        post_pk_summary = write_pk_npz(
            post_pk_fn,
            post_pk,
            edges=pk_edges,
            boxsize=QUIJOTE_BOXSIZE_MPC_H,
            meshsize=int(args.pk_meshsize),
            mpicomm=mpicomm,
        )
        del post_pk
        jax.clear_caches()

    metadata = {
        "pipeline_name": args.pipeline_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "label": label,
        "manifest": manifest_row,
        "stage": config.__dict__,
        "cosmology": args.cosmology,
        "realization": int(args.realization),
        "input_path": str(in_path),
        "input_mode": str(args.input_mode),
        "sto_weighted_input_path": None if sto_path is None else str(sto_path),
        "input_catalog": input_catalog_meta,
        "quijote_params": {
            "boxsize_mpc_h": QUIJOTE_BOXSIZE_MPC_H,
            "redshift": float(args.redshift),
            "omega_m0": QUIJOTE_OMEGA_M0,
            "data_has_rsd": False,
            "physical_growth_rate": float(args.physical_growth_rate),
            "mass_threshold_msun_h": float(args.mass_threshold_msun_h),
        },
        "bias_value": float(args.bias),
        "fnl_loc": float(args.fnl_loc),
        "recon_class": str(args.recon_class),
        "marisa_recon_root": str(args.marisa_recon_root),
        "png_bias_configuration": {
            "png_p": float(args.png_p),
            "png_delta_c": float(args.png_delta_c),
            "png_alpha_kmin": float(args.png_alpha_kmin),
            "png_alpha_kmax": float(args.png_alpha_kmax),
            "png_alpha_nk": int(args.png_alpha_nk),
            "png_bias_floor": float(args.png_bias_floor),
        },
        "bias_source": str(bias_config["bias_source"]),
        "bias_json_path": bias_config["bias_json_path"],
        "bias_model": str(bias_config["bias_model"]),
        "fit_space": str(bias_config["fit_space"]),
        "legacy_desilike_bias": bias_config["legacy_desilike_bias"],
        "recommended_bias_source": bias_config["recommended_bias_source"],
        "growth_rate_policy": "realspace_no_rsd",
        "runtime_compatibility": {
            "scope": "runtime API plumbing only; estimator, binning, normalization value, and reconstruction are unchanged",
            "jaxpower_get_mesh_attrs_mpicomm_forwarded": True,
            "periodic_box_scalar_normalization_broadcast_to_bins": True,
            "required_by_cosmodesi_environment": "20260321-1.0.0 mutable software stack as observed on 2026-08-20",
            "fiducial_numeric_regression_required": True,
        },
        "scoped_runtime_contract": scoped_runtime_contract,
        "mpi": {
            "rank_count": int(mpicomm.size),
            "local_rank_halo_count": local_count,
            "global_loaded_halo_count": global_count,
            "global_shifted_particle_count": shifted_count,
        },
        "random_input": random_input_meta,
        "reconstruction": recon_meta,
        "measurement": {
            "implementation": "jaxpower via clustering_statistics.spectrum3_tools.compute_box_mesh3_spectrum",
            "basis": args.basis,
            "ells": [(0, 0, 0)],
            "los": args.los,
            "meshsize": int(config.meshsize),
            "edges_requested": edges,
            "mask_edges_policy": args.mask_edges_policy,
            "mask_edges": resolve_mask_edges(args.mask_edges_policy),
            "expected_nbins": args.expected_nbins,
            "binner_preflight": binner_preflight,
            "certified_gpu_geometry": None
            if certified_geometry is None
            else {"path": certified_geometry["path"], "sha256": certified_geometry["sha256"]},
            "pre_bins": pre_bins,
            "post_bins": post_bins,
            "pre_geometry_gate": pre_geometry_gate,
            "post_geometry_gate": post_geometry_gate,
            "post_uses_shifted_randoms": True,
        },
        "pk_measurement": None
        if not args.measure_pk
        else {
            "implementation": "jaxpower via clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
            "ells": [0],
            "los": args.los,
            "meshsize": int(args.pk_meshsize),
            "edges_requested": pk_edges,
            "pre_bins": pre_pk_summary,
            "post_bins": post_pk_summary,
            "binner_preflight": pk_binner_preflight,
            "pre_geometry_gate": pre_pk_geometry_gate,
            "post_geometry_gate": post_pk_geometry_gate,
            "post_uses_shifted_randoms": True,
        },
        "outputs": {
            "pre_b000": str(pre_fn),
            "post_b000": str(post_fn),
            "pre_pk": None if not args.measure_pk else str(pre_pk_fn),
            "post_pk": None if not args.measure_pk else str(post_pk_fn),
            "metadata": str(metadata_fn),
        },
        "validation": {"pre": pre_validation, "post": post_validation},
        "packages": get_package_report() if int(mpicomm.rank) == 0 else {},
        "jax_runtime": jax_runtime,
        "slurm": {
            key: os.environ.get(key)
            for key in [
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NAME",
                "SLURM_NODELIST",
                "SLURM_NTASKS",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
        "command": sys.argv,
    }
    write_metadata(metadata_fn, metadata, mpicomm)
    write_done_marker(output_dir / "DONE", metadata, mpicomm)
    log_root(mpicomm, "[%s] finished: %s", log_label, output_dir)


def main() -> None:
    """脚本入口：初始化 JAX distributed，执行 task，关闭 runtime。"""

    args = parse_args()
    if args.recon_class == "marisa-plane":
        activate_marisa_recon_root(args.marisa_recon_root)
    config = get_stage_config(args)
    mpicomm = MPI.COMM_WORLD
    if args.check_only:
        run(args, config=config, mpicomm=mpicomm)
        return
    skip_jax_distributed = os.environ.get("SKIP_JAX_DISTRIBUTED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if skip_jax_distributed:
        log_root(mpicomm, "JAX distributed disabled by SKIP_JAX_DISTRIBUTED.")
    else:
        initialize_jax_distributed(mpicomm)
    patch_jaxpower_sugiyama_s113_dtype(mpicomm)
    try:
        run(args, config=config, mpicomm=mpicomm)
    finally:
        if not skip_jax_distributed:
            shutdown_jax_distributed(mpicomm)


if __name__ == "__main__":
    main()
