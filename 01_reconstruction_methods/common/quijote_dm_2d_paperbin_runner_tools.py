#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quijote DM 2D paper-bin production runner 的本地 helper。

代码大纲
1. 快照读取：定位 Quijote DM HDF5 snapshot，读取 Header 和 PartType1/Coordinates。
2. Catalog/random 准备：把 DM 粒子包装成 desi-clustering Catalog，并生成确定性
   uniform random catalog。
3. 标准 reconstruction：调用 jaxrecon PlaneParallelFFTReconstruction，输出
   reconstructed data 和 shifted randoms。
4. B000 写入验证：使用 jaxpower spectrum 自带 write() 写 HDF5，然后回读
   value/k/nmodes，确认当前 Fig.5/Fisher 需要的 120-bin 2D 数据向量可读且有限。

这个文件只服务当前 `basis=sugiyama` 的 2D paper-bin DM production。历史 diagonal
runner 已归档，不再作为当前生产依赖。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

try:
    # Quijote DM snapshot 的 Coordinates 使用 HDF5 Blosc filter 32001。
    # 只 import hdf5plugin 就会把 cosmodesi 环境里的 filter 注册给 h5py。
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None  # type: ignore[assignment]

import h5py
import jax
import numpy as np
from mpi4py import MPI


DEFAULT_SNAPSHOT_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/dm")


def to_jsonable(value: Any) -> Any:
    """把 numpy/JAX/Path 对象转成 `json.dump` 可以直接写出的对象。

    参数
    ----
    value
        任意 Python 对象，常见输入包括 dict、list、numpy scalar、numpy array、
        pathlib.Path 和 JAX DeviceArray。

    返回
    ----
    Any
        JSON 兼容对象。数组会转成 list；Path 会转成字符串。
    """

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "block_until_ready"):
        return to_jsonable(np.asarray(jax.device_get(value)))
    return value


def get_package_report() -> dict[str, Any]:
    """记录本次运行关键 Python 包的版本和导入路径。

    返回
    ----
    dict
        package 名称到版本、module 文件路径的映射。缺失包会记录 error。
    """

    report: dict[str, Any] = {}
    for name in ["jax", "jaxlib", "jaxpower", "jaxrecon", "clustering_statistics", "numpy", "h5py", "hdf5plugin"]:
        item: dict[str, Any] = {}
        try:
            item["version"] = importlib.metadata.version(name)
        except Exception as exc:  # noqa: BLE001
            item["version_error"] = repr(exc)
        try:
            module = importlib.import_module(name)
            item["file"] = str(Path(getattr(module, "__file__", "")).resolve())
        except Exception as exc:  # noqa: BLE001
            item["import_error"] = repr(exc)
        report[name] = item
    return report


def log_root(mpicomm: MPI.Comm, message: str, *args: Any) -> None:
    """只在 MPI rank 0 打印日志。"""

    if int(mpicomm.rank) == 0:
        print(message % args if args else message, flush=True)


def get_snapshot_dir(snapshot_root: str | Path, cosmology: str, realization: int, snapshot_name: str) -> Path:
    """返回 Quijote DM snapshot 目录。

    参数
    ----
    snapshot_root
        Quijote DM 根目录，例如 `/pscratch/sd/l/lzy/quijote-png/dm`。
    cosmology
        Quijote PNG cosmology 名称，例如 `fiducial` 或 `LC_p`。
    realization
        realization 编号。
    snapshot_name
        snapshot 目录名，z=0.5 当前固定为 `snapdir_003`。
    """

    return Path(snapshot_root).expanduser().resolve() / str(cosmology) / str(int(realization)) / str(snapshot_name)


def _snapshot_files(snapshot_dir: str | Path) -> list[Path]:
    """列出 snapshot HDF5 分片文件，并在缺失时给出明确错误。"""

    path = Path(snapshot_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"snapshot_dir 不存在: {path}")
    files = sorted(path.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"snapshot_dir 内没有 hdf5 文件: {path}")
    return files


def read_snapshot_header(snapshot_dir: str | Path) -> dict[str, Any]:
    """读取 Quijote snapshot Header。

    返回
    ----
    dict
        包含 boxsize、redshift、分片数、DM 粒子总数等信息。Quijote 原始
        BoxSize 单位是 kpc/h，这里同时给出 Mpc/h。
    """

    files = _snapshot_files(snapshot_dir)
    with h5py.File(files[0], "r") as handle:
        attrs = handle["Header"].attrs
        boxsize_kpc_h = float(attrs["BoxSize"])
        redshift = float(attrs["Redshift"])
        num_files = int(attrs["NumFilesPerSnapshot"])
        num_part_total = np.asarray(attrs["NumPart_Total"], dtype=np.int64)
    if len(files) != num_files:
        raise RuntimeError(f"snapshot 分片数不匹配: found={len(files)} header={num_files} dir={snapshot_dir}")
    return {
        "boxsize_kpc_h": boxsize_kpc_h,
        "boxsize_mpc_h": boxsize_kpc_h / 1000.0,
        "redshift": redshift,
        "num_files_per_snapshot": num_files,
        "num_part_total": num_part_total.tolist(),
        "num_part_total_type1": int(num_part_total[1]),
        "first_file": str(files[0]),
    }


def load_rank_positions(
    *,
    snapshot_dir: str | Path,
    mpicomm: MPI.Comm,
    chunk_size: int,
    dtype: np.dtype,
    max_particles_per_rank: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取当前 MPI rank 负责的 DM 粒子坐标。

    参数
    ----
    snapshot_dir
        单个 realization 的 `snapdir_003` 路径。
    mpicomm
        MPI communicator。当前 Slurm 脚本使用单 rank；这里仍支持多 rank
        按文件编号取模分配。
    chunk_size
        兼容旧 runner 的参数；当前按 HDF5 文件整块读取，不再额外切 chunk。
    dtype
        输出坐标 dtype，一般为 float32。
    max_particles_per_rank
        smoke stage 用的每 rank 截断数；full stage 为 None。

    返回
    ----
    tuple
        `(positions_mpc_h, io_info)`。原始坐标从 kpc/h 转为 Mpc/h。
    """

    files = _snapshot_files(snapshot_dir)
    rank = int(mpicomm.rank)
    size = int(mpicomm.size)
    local_files = [item for index, item in enumerate(files) if index % size == rank]
    arrays: list[np.ndarray] = []
    loaded = 0
    for filename in local_files:
        with h5py.File(filename, "r") as handle:
            dataset = handle["PartType1"]["Coordinates"]
            if max_particles_per_rank is not None:
                remaining = int(max_particles_per_rank) - loaded
                if remaining <= 0:
                    break
                coords = np.asarray(dataset[:remaining], dtype=dtype) / np.asarray(1000.0, dtype=dtype)
            else:
                coords = np.asarray(dataset, dtype=dtype) / np.asarray(1000.0, dtype=dtype)
        if max_particles_per_rank is not None:
            coords = coords[: max(0, int(max_particles_per_rank) - loaded)]
        arrays.append(coords)
        loaded += int(coords.shape[0])
    if arrays:
        positions = np.concatenate(arrays, axis=0).astype(dtype, copy=False)
    else:
        positions = np.empty((0, 3), dtype=dtype)
    local_count = int(positions.shape[0])
    global_count = int(mpicomm.allreduce(local_count, op=MPI.SUM))
    io_info = {
        "snapshot_dir": str(Path(snapshot_dir).expanduser().resolve()),
        "rank": rank,
        "rank_count": size,
        "chunk_size_argument": int(chunk_size),
        "local_file_count": len(local_files),
        "total_file_count": len(files),
        "local_particle_count": local_count,
        "global_particle_count": global_count,
        "max_particles_per_rank": None if max_particles_per_rank is None else int(max_particles_per_rank),
        "coordinate_unit": "Mpc/h",
    }
    return positions, io_info


def wrap_positions_to_periodic_box(
    positions: Any,
    *,
    boxsize: float,
    dtype: np.dtype,
    label: str,
    mpicomm: MPI.Comm,
) -> np.ndarray:
    """把坐标 wrap 回周期盒，并避开 jaxpower 的浮点边界误判。

    参数
    ----
    positions
        numpy 或 JAX 坐标数组，形状 `(N, 3)`。
    boxsize
        盒子边长，单位 Mpc/h。
    dtype
        输出 dtype。
    label
        日志标签。
    mpicomm
        MPI communicator。
    """

    arr = np.asarray(jax.device_get(positions), dtype="f8")
    wrapped64 = np.mod(arr, float(boxsize))
    wrapped64 = np.where(wrapped64 >= float(boxsize), 0.0, wrapped64)
    wrapped64 = np.where(wrapped64 < 0.0, wrapped64 + float(boxsize), wrapped64)
    # jaxpower exchange_particles 会在 device-domain 算术中重新判定坐标所属
    # 子域。float32 的 nextafter(boxsize, 0) 在加减 boxcenter 后可能被舍入到
    # boxsize，从而触发 out-of-box assertion；给边界留 1e-3 Mpc/h 余量。
    wrapped64 = np.minimum(wrapped64, float(boxsize) - 1.0e-3)
    wrapped = wrapped64.astype(dtype, copy=False)
    if np.any(wrapped >= np.asarray(float(boxsize), dtype=dtype)):
        wrapped = np.where(wrapped >= np.asarray(float(boxsize), dtype=dtype), np.asarray(0.0, dtype=dtype), wrapped)
    if wrapped.size:
        local_min = np.min(wrapped, axis=0)
        local_max = np.max(wrapped, axis=0)
    else:
        local_min = np.full(3, np.inf, dtype="f8")
        local_max = np.full(3, -np.inf, dtype="f8")
    global_min = mpicomm.allreduce(local_min, op=MPI.MIN)
    global_max = mpicomm.allreduce(local_max, op=MPI.MAX)
    if int(mpicomm.rank) == 0:
        log_root(mpicomm, "%s wrapped: min=%s max=%s", label, np.asarray(global_min), np.asarray(global_max))
    return wrapped


def make_catalog(positions: np.ndarray, *, boxsize: float, zsnap: float, los: str, mpicomm: MPI.Comm) -> Any:
    """把 DM 粒子坐标包装成 desi-clustering Catalog。

    返回的 Catalog 包含 `POSITION` 和单位 `INDWEIGHT`，attrs 记录 boxsize、
    boxcenter、redshift 和 los。
    """

    from clustering_statistics.tools import Catalog

    attrs = {
        "boxsize": float(boxsize),
        "boxcenter": float(boxsize) / 2.0,
        "redshift": float(zsnap),
        "los": str(los),
    }
    return Catalog(
        {
            "POSITION": np.asarray(positions, dtype="f4"),
            "INDWEIGHT": np.ones(int(positions.shape[0]), dtype="f4"),
        },
        attrs=attrs,
        mpicomm=mpicomm,
    )


def generate_rank_random_positions(
    *,
    local_data_count: int,
    boxsize: float,
    seed: int,
    dtype: np.dtype,
    nran: int,
    mpicomm: MPI.Comm,
) -> tuple[np.ndarray, dict[str, Any]]:
    """为当前 rank 生成 deterministic uniform randoms。

    参数
    ----
    local_data_count
        当前 rank 的 data 粒子数。
    boxsize
        盒子边长，单位 Mpc/h。
    seed
        production_random_seed 给出的全局 seed。
    dtype
        输出 dtype。
    nran
        random/data 数量比。DM 当前固定为 1。
    mpicomm
        MPI communicator。

    返回
    ----
    tuple
        `(random_positions, metadata)`。
    """

    rank_seed = np.random.SeedSequence([int(seed), int(mpicomm.rank)])
    rng = np.random.default_rng(rank_seed)
    local_random_count = int(local_data_count) * int(nran)
    random_positions = rng.random((local_random_count, 3), dtype=np.float32)
    random_positions = (random_positions * np.asarray(float(boxsize), dtype=np.float32)).astype(dtype, copy=False)
    global_random_count = int(mpicomm.allreduce(local_random_count, op=MPI.SUM))
    meta = {
        "seed": int(seed),
        "nran": int(nran),
        "local_random_count": local_random_count,
        "global_random_count": global_random_count,
        "rank_seed_entropy": int(seed),
        "rank_seed_spawn_key": list(rank_seed.spawn_key),
        "generator": "numpy.random.default_rng",
    }
    return random_positions, meta


def make_centered_edges(kmin: float, kmax: float, kstep: float) -> np.ndarray:
    """由 bin center 口径生成 jaxpower edges。

    当前 paper-bin 口径把 `kmin=0.01` 和 `kmax=0.30` 解释为 15 个
    center 的首尾值，`kstep=(0.30-0.01)/14`。第一个 edge 会 clip 到 0，
    因此 metadata 中会看到 `edges_requested[0] = 0.0`。
    """

    if kstep <= 0:
        raise ValueError("kstep must be positive")
    centers = np.arange(float(kmin), float(kmax) + 0.5 * float(kstep), float(kstep), dtype="f8")
    if centers.size < 1:
        raise ValueError("need at least one k bin center")
    edges = np.empty(centers.size + 1, dtype="f8")
    if centers.size == 1:
        edges[0] = max(0.0, centers[0] - 0.5 * float(kstep))
        edges[1] = centers[0] + 0.5 * float(kstep)
        return edges
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = max(0.0, centers[0] - 0.5 * (centers[1] - centers[0]))
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _first_spectrum_leaf(spectrum: Any) -> Any:
    """返回 spectrum 的第一个 leaf，当前 ells 只有 `(0,0,0)`。"""

    for _labels, leaf in spectrum.items(level=1):
        return leaf
    raise ValueError("spectrum has no leaves")


def spectrum_bin_summary(spectrum: Any) -> dict[str, Any]:
    """提取 spectrum 的轻量级 bin 摘要写入 metadata。

    返回
    ----
    dict
        包含 value/k/nmodes 的 shape、finite 状态和前几个 k 坐标。
    """

    leaf = _first_spectrum_leaf(spectrum)
    value = np.asarray(leaf.value())
    coords = leaf.coords()
    k = np.asarray(coords.get("k")) if isinstance(coords, dict) and "k" in coords else None
    summary: dict[str, Any] = {
        "value_shape": list(value.shape),
        "value_dtype": str(value.dtype),
        "value_all_finite": bool(np.all(np.isfinite(value))),
        "value_abs_min": float(np.min(np.abs(value))) if value.size else None,
        "value_abs_max": float(np.max(np.abs(value))) if value.size else None,
    }
    if k is not None:
        summary["k_shape"] = list(k.shape)
        summary["k_first5"] = k[:5].tolist()
    for name in ["nmodes", "norm", "num_shotnoise"]:
        try:
            arr = np.asarray(leaf.get(name))
        except Exception:  # noqa: BLE001
            continue
        summary[f"{name}_shape"] = list(arr.shape)
        if arr.size:
            summary[f"{name}_min"] = float(np.min(arr))
            summary[f"{name}_max"] = float(np.max(arr))
    return summary


def write_and_validate_spectrum(filename: str | Path, spectrum: Any, mpicomm: MPI.Comm) -> dict[str, Any]:
    """写出 jaxpower spectrum HDF5，并回读验证。

    参数
    ----
    filename
        目标 HDF5 文件。
    spectrum
        `compute_box_mesh3_spectrum` 返回的 Mesh3SpectrumPoles。
    mpicomm
        MPI communicator。当前 production 使用单 rank；这里由 rank 0 写文件，
        再广播 validation 结果。
    """

    path = Path(filename).expanduser().resolve()
    validation: dict[str, Any] | None = None
    if int(mpicomm.rank) == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        if tmp.exists():
            tmp.unlink()
        spectrum.write(str(tmp))
        os.replace(tmp, path)
        leaf_value = np.asarray(_first_spectrum_leaf(spectrum).value())
        validation = {
            "value_shape": list(leaf_value.shape),
            "value_dtype": str(leaf_value.dtype),
            "all_finite": bool(np.all(np.isfinite(leaf_value))),
            "abs_min": float(np.min(np.abs(leaf_value))) if leaf_value.size else None,
            "abs_max": float(np.max(np.abs(leaf_value))) if leaf_value.size else None,
        }
        with h5py.File(path, "r") as handle:
            group = handle["0_0_0"] if "0_0_0" in handle else handle["data"]["0_0_0"]
            read_value = np.asarray(group["value"], dtype="f8")
            read_k = np.asarray(group["k"], dtype="f8")
            read_nmodes = np.asarray(group["nmodes"])
        validation.update(
            {
                "readable": True,
                "read_value_shape": list(read_value.shape),
                "read_k_shape": list(read_k.shape),
                "read_nmodes_shape": list(read_nmodes.shape),
                "read_all_finite": bool(np.all(np.isfinite(read_value))),
            }
        )
    validation = mpicomm.bcast(validation, root=0)
    mpicomm.Barrier()
    assert validation is not None
    return validation


def block_if_ready(value: Any) -> None:
    """等待 JAX lazy 计算完成，numpy 输入则直接返回。"""

    try:
        jax.block_until_ready(value)
    except Exception:  # noqa: BLE001
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def summarize_shifts(before: np.ndarray, after: np.ndarray, *, boxsize: float, mpicomm: MPI.Comm) -> dict[str, Any]:
    """计算 periodic 最短位移的全局统计。

    参数
    ----
    before, after
        shift 前后的本地坐标。
    boxsize
        周期盒边长。
    mpicomm
        MPI communicator。
    """

    before = np.asarray(before, dtype="f8")
    after = np.asarray(after, dtype="f8")
    diff = (after - before + 0.5 * float(boxsize)) % float(boxsize) - 0.5 * float(boxsize)
    local_count = int(diff.shape[0])
    local_sum = np.sum(diff, axis=0) if local_count else np.zeros(3, dtype="f8")
    local_abs = np.sum(np.abs(diff), axis=0) if local_count else np.zeros(3, dtype="f8")
    local_sq = np.sum(diff * diff, axis=0) if local_count else np.zeros(3, dtype="f8")
    local_max = np.max(np.abs(diff), axis=0) if local_count else np.zeros(3, dtype="f8")
    count = int(mpicomm.allreduce(local_count, op=MPI.SUM))
    total_sum = np.asarray(mpicomm.allreduce(local_sum, op=MPI.SUM), dtype="f8")
    total_abs = np.asarray(mpicomm.allreduce(local_abs, op=MPI.SUM), dtype="f8")
    total_sq = np.asarray(mpicomm.allreduce(local_sq, op=MPI.SUM), dtype="f8")
    total_max = np.asarray(mpicomm.allreduce(local_max, op=MPI.MAX), dtype="f8")
    if count == 0:
        return {"count": 0, "mean": [0.0, 0.0, 0.0], "mean_abs": [0.0, 0.0, 0.0], "rms": [0.0, 0.0, 0.0], "max_abs": [0.0, 0.0, 0.0]}
    return {
        "count": count,
        "mean": (total_sum / count).tolist(),
        "mean_abs": (total_abs / count).tolist(),
        "rms": np.sqrt(total_sq / count).tolist(),
        "max_abs": total_max.tolist(),
    }


def run_jaxrecon(
    *,
    data_catalog: Any,
    random_positions: np.ndarray | None,
    boxsize: float,
    growth_rate: float,
    bias: float,
    smoothing_radius: float,
    recon_cellsize: float,
    recon_class: str,
    niterations: int | None = None,
    data_field: str,
    random_field: str,
    nran: int,
    random_source: str,
    los: str,
    dtype: np.dtype,
    mpicomm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """运行标准 jaxrecon PlaneParallelFFTReconstruction。

    参数
    ----
    data_catalog
        DM data Catalog。
    random_positions
        当前 rank 的 uniform random 坐标；`random_source=pipeline_numpy` 时必须提供。
    boxsize
        盒子边长，单位 Mpc/h。
    growth_rate, bias, smoothing_radius, recon_cellsize
        标准 reconstruction 参数。DM real-space 当前固定为 f=0、bias=1、
        smoothing=10 Mpc/h、cellsize=5 Mpc/h。
    recon_class
        当前只允许 `plane`。
    niterations
        为了保留旧 metadata 字段而接收；PlaneParallelFFTReconstruction 不使用。
    data_field, random_field
        传给 `read_shifted_positions` 的 field 名称。
    nran, random_source
        shifted random provenance。
    los
        line-of-sight，real-space f=0 时只作 metadata/接口参数。
    dtype
        输出坐标 dtype。
    mpicomm
        MPI communicator。

    返回
    ----
    tuple
        `(data_rec_positions, shifted_random_positions, metadata)`。
    """

    if recon_class != "plane":
        raise ValueError(f"标准 DM runner 只支持 recon_class='plane', got {recon_class}")
    if random_source != "pipeline_numpy":
        raise ValueError("当前 DM 2D production 只支持 deterministic pipeline_numpy randoms")
    if random_positions is None:
        raise ValueError("pipeline_numpy random_source 需要提供 random_positions")

    from clustering_statistics.recon_tools import prepare_jaxpower_particles
    from jaxpower import create_sharding_mesh
    from jaxrecon.zeldovich import PlaneParallelFFTReconstruction, estimate_particle_delta

    recon_mattrs = {
        "boxsize": float(boxsize) * np.ones(3, dtype="f4"),
        "boxcenter": (float(boxsize) / 2.0) * np.ones(3, dtype="f4"),
        "cellsize": float(recon_cellsize),
    }

    times: dict[str, float] = {}
    with create_sharding_mesh(meshsize=recon_mattrs.get("meshsize", None)):
        t0 = MPI.Wtime()
        data = prepare_jaxpower_particles(lambda: {"data": data_catalog}, mattrs=recon_mattrs, return_inverse=True)[0]["data"]
        times["prepare_particles_seconds"] = MPI.Wtime() - t0

        t0 = MPI.Wtime()
        delta = estimate_particle_delta(data, smoothing_radius=float(smoothing_radius))
        block_if_ready(delta.value)
        times["estimate_delta_seconds"] = MPI.Wtime() - t0

        recon_constructor = jax.jit(
            PlaneParallelFFTReconstruction,
            static_argnames=["los"],
            donate_argnums=[0],
        )

        t0 = MPI.Wtime()
        recon = recon_constructor(delta, growth_rate=float(growth_rate), bias=float(bias), los=los)
        data_positions_rec = recon.read_shifted_positions(data.positions, field=data_field)
        data_positions_rec = data.exchange_inverse(data_positions_rec)
        block_if_ready(data_positions_rec)
        times["reconstruct_and_shift_data_seconds"] = MPI.Wtime() - t0

        t0 = MPI.Wtime()
        randoms_positions_rec = recon.read_shifted_positions(random_positions, field=random_field)
        block_if_ready(randoms_positions_rec)
        times["shift_randoms_seconds"] = MPI.Wtime() - t0

    data_rec = wrap_positions_to_periodic_box(
        data_positions_rec,
        boxsize=boxsize,
        dtype=dtype,
        label="jaxrecon data_rec",
        mpicomm=mpicomm,
    )
    shifted_rec = wrap_positions_to_periodic_box(
        randoms_positions_rec,
        boxsize=boxsize,
        dtype=dtype,
        label="jaxrecon shifted_rec",
        mpicomm=mpicomm,
    )
    data_before_local = np.asarray(data_catalog["POSITION"])
    shift_summary = {
        "data": summarize_shifts(data_before_local, data_rec, boxsize=boxsize, mpicomm=mpicomm),
        "random": summarize_shifts(random_positions, shifted_rec, boxsize=boxsize, mpicomm=mpicomm),
    }
    meta = {
        "implementation": "jaxrecon.zeldovich.PlaneParallelFFTReconstruction",
        "recon_class": recon_class,
        "niterations": None if niterations is None else int(niterations),
        "growth_rate": float(growth_rate),
        "bias": float(bias),
        "los": los,
        "smoothing_radius": float(smoothing_radius),
        "cellsize": float(recon_cellsize),
        "data_shift_field": data_field,
        "random_shift_field": random_field,
        "random": {"source": random_source, "field": random_field, "nran": int(nran)},
        "shift_summary": shift_summary,
        "timing_seconds": times,
    }
    return data_rec, shifted_rec, meta


def patch_jaxpower_sugiyama_s113_dtype(mpicomm: MPI.Comm) -> None:
    """修补当前 jaxpower Sugiyama S113 的 JAX scan dtype。

    当前 cosmodesi 环境打开 `JAX_ENABLE_X64=1` 后，Sugiyama S113 shot-noise
    项内部的 spherical-Bessel / `bin.xavg` 运算会产生 `complex128`，但
    jaxpower 原函数用 `mattrs.cdtype` 初始化 scan carry，通常是 `complex64`。
    JAX 要求 scan 输入/输出 carry dtype 完全一致，于是会报
    `complex64[nbin]` vs `complex128[nbin]`。

    这里只复制原函数并把 scan 初值 dtype 改成
    `result_type(mattrs.cdtype, bin.xavg[..., 0], 1j)`。其它公式、binning、
    estimator、shot-noise 项都保持不变。
    """

    from functools import partial

    import jax.numpy as jnp
    import jaxpower.mesh3 as mesh3

    if getattr(mesh3, "_qdm_s113_dtype_patch_applied", False):
        log_root(mpicomm, "jaxpower Sugiyama S113 dtype patch already applied.")
        return

    @partial(jax.jit, donate_argnums=[0, 1, 2], static_argnames=["ells", "los"])
    def _patched_compute_sugiyama_spectrum_S113(rmesh, cmesh, s111=dict(), cmesh_s111=1.0, bin=None, ells=tuple(), los="z"):
        mattrs = rmesh.attrs
        los, vlos = mesh3._format_los(los, ndim=mattrs.ndim)

        rmesh -= rmesh.mean()
        cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.0))
        xvec = mattrs.rcoords(sparse=True)
        svec = mattrs.rcoords(kind="separation", sparse=True)

        @partial(jax.checkpoint, static_argnums=(0, 1))
        def f(Ylms, jl, carry, im):
            los = xvec if vlos is None else vlos
            s111_local, coeff, sym, im = im[-1], im[3], im[4], im[:3]
            tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los).conj()).r2c() * cmesh.conj() - s111_local * cmesh_s111)
            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
            tmp = tmp.c2r() * (jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)).conj()

            def fk(k):
                return jnp.sum(tmp.value * jl[0](snorm * k[0]) * jl[1](snorm * k[1]))

            tmp = jax.lax.map(fk, bin.xavg)
            carry += tmp + sym * tmp.conj()
            return carry, im

        # 在 x64 production 口径下，S113 body 会产生 complex128；这里必须
        # 显式使用 complex128，不能依赖 Python `1j` 的 weak dtype 推导。
        carry_dtype = jnp.complex128
        s113 = []
        for ell1, ell2, ell3 in ells:
            Ylms = [
                [mesh3.get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)]
                for ell in [ell1, ell2, ell3]
            ]
            xs = mesh3._iter_triposh(ell1, ell2, ell3, los=los)
            if xs[0].size:
                xs = xs + [jnp.array([s111[ell3, int(im3) - ell3] for im3 in xs[2]])]
                sign = (1j) ** (ell1 + ell2)
                init = jnp.zeros_like(bin.xavg[..., 0], dtype=carry_dtype)
                s113.append(
                    sign
                    * jax.lax.scan(
                        partial(f, Ylms, [mesh3.get_spherical_jn(ell1), mesh3.get_spherical_jn(ell2)]),
                        init=init,
                        xs=xs,
                    )[0]
                )
            else:
                s113.append(jnp.zeros_like(bin.xavg[..., 0], dtype=carry_dtype))
        return s113

    mesh3._compute_sugiyama_spectrum_S113 = _patched_compute_sugiyama_spectrum_S113
    mesh3._qdm_s113_dtype_patch_applied = True
    log_root(mpicomm, "Applied jaxpower Sugiyama S113 dtype patch for x64 scan carry.")


def shutdown_jax_distributed(mpicomm: MPI.Comm) -> None:
    """关闭 JAX distributed；未初始化时静默跳过。"""

    try:
        jax.distributed.shutdown()
    except Exception as exc:  # noqa: BLE001
        log_root(mpicomm, "JAX distributed shutdown skipped: %s", exc)
