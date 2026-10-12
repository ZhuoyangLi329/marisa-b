#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Project-side helper for running marisa-recon.

代码大纲
1. `activate_marisa_recon_root`：
   校验本目录 png_recon_tools.py 的导入来源；共享 clustering_statistics 由环境提供。
2. `sample_png_bias`：
   在若干固定 k 点上计算 `b_eff(k)`，用于写入 metadata，方便肉眼审计。
3. `run_marisa_jaxrecon`：
   复用现有 Quijote halo runner 的 data/random 准备方式和 output wrapping，
   只把标准 `PlaneParallelFFTReconstruction` 替换为
   `clustering_statistics.png_recon_tools.MarisaPlaneParallelFFTReconstruction`。

设计原则
- 真正的 reconstruction 公式已复制到同目录 png_recon_tools.py；
  本文件只是把 production runner 接到那份本地代码。
- `fnl_loc=0` 时输出应退化到旧的 scalar-bias reconstruction。
- 保留 shifted random convention：post-recon B_000 使用 reconstructed data
  加同一 displacement field shift 后的 uniform random。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from quijote_dm_2d_paperbin_runner_tools import (
    block_if_ready,
    summarize_shifts,
    wrap_positions_to_periodic_box,
)


DEFAULT_MARISA_RECON_ROOT = Path(__file__).resolve().parent


def activate_marisa_recon_root(marisa_recon_root=DEFAULT_MARISA_RECON_ROOT) -> Path:
    """校验 v2 本地 Marisa 核心来源，不切换共享 clustering_statistics 安装。

    参数为核心目录，返回已解析目录；拒绝加载旧 checkout，避免迁移后仍调用旧代码。
    """
    root = Path(marisa_recon_root).expanduser().resolve()
    expected = Path(__file__).resolve().parent
    if root != expected:
        raise ValueError(f"v2 Marisa 核心应位于 {expected}，收到 {root}")
    import png_recon_tools
    if Path(png_recon_tools.__file__).resolve() != root / 'png_recon_tools.py':
        raise RuntimeError('Marisa 核心导入来源错误')
    return root


def sample_png_bias(
    *,
    k_table: np.ndarray,
    alpha_table: np.ndarray,
    b1: float,
    fnl_loc: float,
    p: float,
    delta_c: float,
) -> dict[str, list[float]]:
    """在固定 k 点上采样 `alpha(k)` 和 `b_eff(k)`。

    参数
    ----------
    k_table, alpha_table
        `compute_png_alpha_table` 生成的一维插值表格。
    b1, fnl_loc, p, delta_c
        PNG bias 公式参数。

    返回
    ----
    dict
        JSON-friendly 的 `k_hmpc`、`alpha`、`b_eff` 三个数组。
    """

    k_sample = np.asarray([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], dtype="f8")
    alpha_sample = np.interp(np.log10(k_sample), np.log10(k_table), alpha_table)
    bphi = 2.0 * float(delta_c) * (float(b1) - float(p))
    b_eff = float(b1) + bphi * float(fnl_loc) * alpha_sample
    return {
        "k_hmpc": k_sample.tolist(),
        "alpha": alpha_sample.astype("f8").tolist(),
        "b_eff": np.asarray(b_eff, dtype="f8").tolist(),
    }


def run_marisa_jaxrecon(
    *,
    data_catalog: Any,
    random_positions: np.ndarray | None,
    boxsize: float,
    growth_rate: float,
    bias: float,
    smoothing_radius: float,
    recon_cellsize: float,
    recon_class: str,
    data_field: str,
    random_field: str,
    nran: int,
    random_source: str,
    los: str,
    dtype: np.dtype,
    mpicomm: MPI.Comm,
    marisa_recon_root: str | Path = DEFAULT_MARISA_RECON_ROOT,
    fnl_loc: float = 0.0,
    png_p: float = 1.0,
    png_delta_c: float = 1.686,
    png_alpha_kmin: float = 1.0e-4,
    png_alpha_kmax: float = 10.0,
    png_alpha_nk: int = 2048,
    png_alpha_redshift: float = 0.5,
    png_alpha_cosmology: str = "quijote",
    png_bias_floor: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """运行一次 marisa PNG-aware reconstruction。

    参数
    ----------
    data_catalog
        当前 MPI rank 的 halo Catalog。
    random_positions
        当前 MPI rank 的 uniform random 坐标。`random_source="pipeline_numpy"` 时必须提供。
    boxsize
        周期盒边长，单位 Mpc/h。
    growth_rate
        reconstruction 使用的增长率。当前 real-space halo production 固定为 0。
    bias
        Gaussian scalar bias `b1`，也是 fNL=0 时的 `b_eff`。
    smoothing_radius
        构造 density mesh 前的 Gaussian smoothing 半径，单位 Mpc/h。
    recon_cellsize
        reconstruction mesh 的 cellsize，单位 Mpc/h。
    recon_class
        必须为 `marisa-plane`。
    data_field, random_field
        传给 `read_shifted_positions` 的 displacement field 名称，当前为 `disp`。
    nran, random_source
        shifted random 的生成方式 provenance。
    los
        plane-parallel line-of-sight，当前为 z。
    dtype
        输出坐标 dtype。
    mpicomm
        MPI communicator。
    marisa_recon_root
        本地 marisa-recon checkout 根目录。
    fnl_loc, png_p, png_delta_c, png_alpha_*, png_bias_floor
        PNG bias 公式和 alpha 表格参数。

    返回
    ----
    tuple
        `(data_rec, shifted_rec, metadata)`，前两个是已 wrap 回 box 内的坐标数组。
    """

    root = activate_marisa_recon_root(marisa_recon_root)

    import jax
    from clustering_statistics.recon_tools import prepare_jaxpower_particles
    from png_recon_tools import (
        MarisaPlaneParallelFFTReconstruction,
        compute_png_alpha_table,
        describe_png_bias_model,
    )
    import png_recon_tools
    from jaxpower import create_sharding_mesh, generate_uniform_particles
    from jaxrecon.zeldovich import estimate_particle_delta

    if recon_class != "marisa-plane":
        raise ValueError(f"run_marisa_jaxrecon 仅支持 recon_class='marisa-plane', got {recon_class}")

    alpha_k, alpha_values, alpha_meta = compute_png_alpha_table(
        kmin=float(png_alpha_kmin),
        kmax=float(png_alpha_kmax),
        nk=int(png_alpha_nk),
        redshift=float(png_alpha_redshift),
        cosmology=str(png_alpha_cosmology),
    )
    alpha_meta_dict = asdict(alpha_meta)

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
        times["estimate_delta_seconds"] = MPI.Wtime() - t0

        recon_constructor = jax.jit(
            MarisaPlaneParallelFFTReconstruction,
            static_argnames=["los", "halo_add"],
            donate_argnums=[0],
        )

        t0 = MPI.Wtime()
        recon = recon_constructor(
            delta,
            growth_rate=float(growth_rate),
            bias=float(bias),
            los=los,
            halo_add=0,
            png_alpha_k=alpha_k,
            png_alpha_values=alpha_values,
            fnl_loc=float(fnl_loc),
            png_p=float(png_p),
            png_delta_c=float(png_delta_c),
            bias_floor=float(png_bias_floor),
        )
        data_positions_rec = recon.read_shifted_positions(data.positions, field=data_field)
        data_positions_rec = data.exchange_inverse(data_positions_rec)
        block_if_ready(data_positions_rec)
        times["reconstruct_and_shift_data_seconds"] = MPI.Wtime() - t0

        t0 = MPI.Wtime()
        random_meta: dict[str, Any] = {"source": random_source, "field": random_field, "nran": int(nran)}
        if random_source == "jaxpower":
            randoms = generate_uniform_particles(
                data.attrs,
                size=int(nran) * data.size,
                seed=(42, "index"),
                exchange=True,
                backend="mpi",
                return_inverse=True,
            )
            randoms_positions_rec = recon.read_shifted_positions(randoms.positions, field=random_field)
            randoms_positions_rec = randoms.exchange_inverse(randoms_positions_rec)
            random_before_local = np.asarray(jax.device_get(randoms.exchange_inverse(randoms.positions)))
            random_meta["seed"] = "(42, 'index')"
        else:
            if random_positions is None:
                raise ValueError("pipeline_numpy 需要提供 random_positions")
            randoms_positions_rec = recon.read_shifted_positions(random_positions, field=random_field)
            random_before_local = random_positions
        block_if_ready(randoms_positions_rec)
        times["shift_randoms_seconds"] = MPI.Wtime() - t0

    data_rec = wrap_positions_to_periodic_box(
        data_positions_rec,
        boxsize=boxsize,
        dtype=dtype,
        label="marisa_jaxrecon data_rec",
        mpicomm=mpicomm,
    )
    shifted_rec = wrap_positions_to_periodic_box(
        randoms_positions_rec,
        boxsize=boxsize,
        dtype=dtype,
        label="marisa_jaxrecon shifted_rec",
        mpicomm=mpicomm,
    )
    data_before_local = np.asarray(data_catalog["POSITION"])
    shift_summary = {
        "data": summarize_shifts(data_before_local, data_rec, boxsize=boxsize, mpicomm=mpicomm),
        "random": summarize_shifts(random_before_local, shifted_rec, boxsize=boxsize, mpicomm=mpicomm),
    }

    png_bias_model = describe_png_bias_model(
        b1=float(bias),
        fnl_loc=float(fnl_loc),
        p=float(png_p),
        delta_c=float(png_delta_c),
        alpha_metadata=alpha_meta_dict,
    )
    png_bias_model["sample"] = sample_png_bias(
        k_table=alpha_k,
        alpha_table=alpha_values,
        b1=float(bias),
        fnl_loc=float(fnl_loc),
        p=float(png_p),
        delta_c=float(png_delta_c),
    )

    meta = {
        "implementation": "v2.marisa_gra.png_recon_tools.MarisaPlaneParallelFFTReconstruction",
        "marisa_recon_root": str(root),
        "marisa_recon_module": str(Path(png_recon_tools.__file__).resolve()),
        "recon_class": recon_class,
        "growth_rate": float(growth_rate),
        "bias": float(bias),
        "los": los,
        "smoothing_radius": float(smoothing_radius),
        "cellsize": float(recon_cellsize),
        "data_shift_field": data_field,
        "random_shift_field": random_field,
        "random": random_meta,
        "png_bias_model": png_bias_model,
        "png_bias_floor": float(png_bias_floor),
        "shift_summary": shift_summary,
        "timing_seconds": times,
    }
    return data_rec, shifted_rec, meta
