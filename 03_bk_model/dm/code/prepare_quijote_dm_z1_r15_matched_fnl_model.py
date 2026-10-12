#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 DM z=1、R=15 的一致 G/L/Q 向量及独立精度检查。

执行大纲
========
1. 复用原 profile 的公共 446-realization 几何和已冻结 z=1 谱表。
2. 复用 MARISA-B shell 投影函数；三个扇区使用同一组外部三角形。
3. 通过现有 native 接口计算 Gaussian、linear PNG、B112II 二次系数。
4. 小批量结果写入一个 SQLite 缓存；输入、binary 和源码哈希参与键值，
   因而中断后可精确续算，不能错误复用另一版物理配置。
5. 输出每个精度等级一个小型 JSON，供 profile 和收敛审计直接读取。

约定与限制
==========
这里是连续 shell + 无 alias 的 CIC paint/read 窗口近似，不是假装已经完成
精确格点一环：scalar native 接口把三角形放在标准方向，CIC 的 SO(3) 方位
平均仍需要独立检查。qmin=2pi/L 是球形 loop IR 截断；所有外部边同样要求
k>=2pi/L，去掉部分的 shell/角权重不重新归一化。kmax 限制 k1,k2，k3 不
施加上限，以保持实测 B000 的定义。所有并发 native 子进程只用一个线程，
本机总并发不超过八个。绝不修改 native 物理内核或旧理论文件。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

for _name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes/marisa_b/scripts/production"))
from produce_quijote_dm_marisa_b_local_png_tree_b000_vector import (
    build_b000_mu_work,
    write_png_transfer_table,
)
from run_marisa_b_triangle import run_marisa_b
from run_quijote_dm_z1_r15_1loop_fnl_profiles import SIMULATION_NPZ, sha256

BINARY = ROOT / "marisa_b_core/build/halo_v1/marisa_b_triangle"
SOURCE = ROOT / "codes/marisa_b/src/marisa_b/marisa_b_native.cpp"
TABLE = ROOT / ("analysis/theory_vectors/marisa_b_v0/"
    "marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat")
FALLBACK_PLIN_TABLE = ROOT / ("analysis/theory_vectors/marisa_b_v0/"
    "marisa_b_pre_gaussian_shellbin_diag15_kmax0p3_nrad3_nmu12_20260615_plin_z0.5.dat")
OUTPUT = ROOT / "analysis/quijote_dm_z1_r15_matched_fnl_model_v3"
LEVELS = {
    "coarse": dict(nmu=12, nrad=3, eps=.02, qmc=10),
    "fine": dict(nmu=24, nrad=3, eps=.01, qmc=12),
    "mucheck": dict(nmu=48, nrad=3, eps=.01, qmc=12),
    "radcheck": dict(nmu=24, nrad=5, eps=.01, qmc=12),
    "loopcheck": dict(nmu=24, nrad=3, eps=.002, qmc=14),
}
FIELDS = {
    "G": ("Btree", "B222", "B321I", "B321II", "B411", "B1loop", "Btotal"),
    "L": ("dBdfNL_png_tree", "dBdfNL_png_B122I", "dBdfNL_png_B122II",
          "dBdfNL_png_B113I", "dBdfNL_png_B113II", "dBdfNL_png_1loop", "dBdfNL_png_total"),
    "Q": ("B112II_fNL2_coefficient",),
}


def projection(data: dict, positions: list[int], cfg: dict) -> tuple:
    """把已有 shell 积分的 mu 节点映射到 k3>=kf 的区间，保留原归一化。

    输入是实测几何、对角位置和精度配置；返回 native 三角形及 1/2 积分权重。
    同一输出复用于 G/L/Q，避免各扇区使用不同 shell 或第三边支持。
    """
    diag = np.flatnonzero(data["pair_i"] == data["pair_j"])
    ids = diag[positions]
    triangles, meta, _ = build_b000_mu_work(
        ids, data["hdf5_k_weighted"], cfg["nmu"], bin_average="shell",
        k_edges=data["hdf5_k_edges"], nradial_binavg=cfg["nrad"],
        external_support_min=cfg["qmin"])
    work, weights = [], []
    for (a, b, mu), (local, w) in zip(triangles, meta):
        lo = max(-1., (cfg["qmin"]**2-a*a-b*b)/(2*a*b))
        if lo >= 1:
            continue
        mapped = (1+lo)/2 + (1-lo)*mu/2
        work.append((a, b, mapped))
        weights.append((positions[local], w*(1-lo)/4))
    return work, weights


def evaluate(spec: dict) -> dict:
    """调用一个小批次 native 积分，返回必要成分与误差，不重复保存 stdout。

    spec 中包括精度、阶段、扇区和三角形；调用失败抛错，成功才允许进入缓存。
    stderr 的收敛警告会保存并进入总报告，returncode=0 不等于数值已收敛。
    """
    c, stage, sector = spec["config"], spec["stage"], spec["sector"]
    mode = {"G": f"{stage}_recon_gaussian", "L": f"{stage}_recon_png_1loop",
            "Q": f"{stage}_recon_local_png_b112ii_diagnostic"}[sector]
    # pre 的旧 adaptive PNG 路径在硬 IR 边界上可能耗尽采样预算。
    # reconstruction strength=1/b_rec=1e-30 时 Zrec=F 到机器精度，因而可用
    # 已验证的 post vector/QMC 路径独立积分同一 pre 物理模型。
    zero_shift = stage == "pre" and sector == "L" and c.get("pre_linear_zero_shift_qmc", False)
    if zero_shift:
        mode = "post_recon_png_1loop"
    start = time.monotonic()
    table = Path(spec["table"])
    result = run_marisa_b(
        BINARY, table, spec["triangles"], c["eps"], c["eps"], c["qmin"], c["qmax"],
        c["smoothing_radius"], 1.e30 if zero_shift else 1.,
        c["recon_cellsize"] if stage == "post" else 0., c["cic"] if stage == "post" else 0,
        1.e-5, mode, "native_cpp", png_table=table, png_ir_cutoff=c["qmin"],
        b112ii_integrator="multicenter_qmc", b112ii_qmc_power=c["qmc"],
        b112ii_qmc_replicates=4, png_shape="local", linear_png_integrator="multicenter_qmc")
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"])
    rows = result["payload"]["results"]
    if len(rows) != len(spec["triangles"]):
        raise RuntimeError("native row count does not match requested triangles")
    reduced = []
    for row in rows:
        values = {field: float(row[field]) for field in FIELDS[sector]}
        if not all(np.isfinite(list(values.values()))):
            raise RuntimeError("nonfinite native component")
        values["errors"] = {k: v["abserr"] for k, v in row["integration_metadata"].items()
                            if k in FIELDS[sector]}
        reduced.append(values)
    return dict(rows=reduced, seconds=time.monotonic()-start, stderr=result["stderr"])


def run(args: argparse.Namespace) -> None:
    """运行可续算调度；只让主线程写缓存，最多八个独立单线程积分任务。"""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    simulation_npz = args.simulation_npz.resolve()
    table = args.table.resolve()
    if args.generate_table:
        table_meta = write_png_transfer_table(
            table,
            z=args.redshift,
            kmin=args.pk_kmin,
            kmax=args.pk_kmax,
            n=args.pk_n,
            fallback_plin_table=args.fallback_plin_table.resolve(),
            fallback_a_s=args.a_s,
            fallback_n_s=args.n_s,
            fallback_h=args.h,
        )
        (output / "transfer_table_metadata.json").write_text(
            json.dumps(table_meta, indent=2) + "\n", encoding="utf-8"
        )
    if not simulation_npz.is_file() or not table.is_file():
        raise FileNotFoundError(f"missing simulation or transfer table: {simulation_npz}, {table}")
    cfg = {
        **LEVELS[args.level],
        "qmin": args.qmin,
        "qmax": args.qmax,
        "cic": args.cic,
        "redshift": args.redshift,
        "smoothing_radius": args.smoothing_radius,
        "recon_cellsize": args.recon_cellsize,
    }
    if args.pre_linear_zero_shift_qmc:
        cfg["pre_linear_zero_shift_qmc"] = True
    if args.nmu:
        cfg["nmu"] = args.nmu
    if args.nrad:
        cfg["nrad"] = args.nrad
    with np.load(simulation_npz) as payload:
        data = {k: payload[k] for k in ("pair_i", "pair_j", "hdf5_k_edges", "hdf5_k_weighted")}
        common_realizations = np.asarray(payload["common_realizations"], dtype=np.int64)
    diag = np.flatnonzero(data["pair_i"] == data["pair_j"])
    kval = np.mean(data["hdf5_k_weighted"][diag], axis=1)
    available = np.flatnonzero(kval <= .2 + 1.e-12).tolist()
    positions = sorted(set(args.positions if args.positions is not None else available))
    if not positions or any(p not in available for p in positions):
        raise ValueError("positions must select nonempty diagonal bins with weighted k <= 0.2")
    identity = {"binary": sha256(BINARY), "source": sha256(SOURCE), "table": sha256(table),
                "simulation": sha256(simulation_npz), "projection_version": 1}
    triangles, weights = projection(data, positions, cfg)
    specs = []
    for stage in args.stages:
        for sector in args.sectors:
            for start in range(0, len(triangles), args.chunk_size):
                spec = dict(config=cfg, stage=stage, sector=sector,
                            triangles=triangles[start:start+args.chunk_size], identity=identity,
                            table=str(table))
                key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                specs.append((key, spec, weights[start:start+args.chunk_size]))
    database = sqlite3.connect(output / "integral_cache.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS integrals (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
    results, pending = {}, []
    for key, spec, _ in specs:
        saved = database.execute("SELECT result FROM integrals WHERE key=?", (key,)).fetchone()
        if saved:
            results[key] = json.loads(saved[0])
        else:
            pending.append((key, spec))
    print(f"{args.level}: {len(positions)} bins, {len(triangles)} triangles/sector; "
          f"cached {len(specs)-len(pending)}/{len(specs)} chunks, {args.workers} workers", flush=True)
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate, spec): (key, spec) for key, spec in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, spec = futures[future]
            result = future.result()
            results[key] = result
            database.execute("INSERT OR REPLACE INTO integrals VALUES (?,?)", (key, json.dumps(result)))
            database.commit()
            if number % 20 == 0 or number == len(pending) or result["stderr"]:
                print(f"{args.level} {number}/{len(pending)} {spec['stage']}/{spec['sector']} "
                      f"{result['seconds']:.1f}s chunk, elapsed {time.monotonic()-start:.0f}s "
                      f"warnings={bool(result['stderr'])}", flush=True)
    database.close()
    components = {s: {f: np.zeros(len(diag)) for sector in args.sectors for f in FIELDS[sector]}
                  for s in args.stages}
    errors = {s: {f: np.zeros(len(diag)) for sector in args.sectors for f in FIELDS[sector]}
              for s in args.stages}
    warnings = []
    for key, spec, local_weights in specs:
        result = results[key]
        if result["stderr"]:
            warnings.append(dict(stage=spec["stage"], sector=spec["sector"], text=result["stderr"]))
        for row, (position, weight) in zip(result["rows"], local_weights):
            for field in FIELDS[spec["sector"]]:
                components[spec["stage"]][field][position] += weight*row[field]
                errors[spec["stage"]][field][position] += abs(weight)*row["errors"].get(field, 0.)
    sums = np.zeros(len(diag))
    for p, w in weights:
        sums[p] += w
    scope = args.scope or (
        f"DM z={args.redshift:g} R={args.smoothing_radius:g} "
        f"common{common_realizations.size} matched GLQ"
    )
    document = dict(schema_version=1, scope=scope, level=args.level,
        config=cfg, positions=positions, pair_indices=diag.tolist(), k=kval.tolist(),
        identity=identity, inputs=dict(binary=str(BINARY), source=str(SOURCE), table=str(table),
                                      simulation=str(simulation_npz)),
        common_realizations=common_realizations.tolist(),
        components={s: {k: v.tolist() for k, v in fields.items()} for s, fields in components.items()},
        integrated_error_bounds={s: {k: v.tolist() for k, v in fields.items()} for s, fields in errors.items()},
        warnings=warnings, normalization_weights=sums.tolist(),
        limitations=["Continuous shell, scalar canonical triangle orientation; CIC4 is a no-alias approximation.",
            "Spherical loop IR cutoff is not exact discrete-box loop summation.",
            "Finite shell normalization is retained when external k<2pi/L support is excluded.",
            "kmax applies to k1,k2; the third triangle side can exceed 0.2.",
            "Empty uncomputed entries must be selected only through positions."])
    name = args.name or args.level
    dest = output / f"{name}.json"
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2)+"\n")
    temporary.replace(dest)
    if args.export_bundle:
        # NPZ 仅在需要供现有 profile 使用时输出；诊断精度等级只保留小 JSON。
        if args.stages != ["pre", "post"] or args.sectors != ["G", "L", "Q"] or positions != available:
            raise ValueError("export-bundle requires complete pre/post G/L/Q for all k<=0.2 bins")
        arrays = dict(diagonal_indices=diag, diagonal_k=kval)
        for stage in args.stages:
            for sector, field in (("G", "Btotal"), ("L", "dBdfNL_png_total"),
                                  ("Q", "B112II_fNL2_coefficient")):
                value = np.full(len(diag), np.nan)
                value[positions] = components[stage][field][positions]
                arrays[f"{stage}_{sector}"] = value
        np.savez_compressed(dest.with_suffix(".npz"), **arrays)
    print(f"Saved {dest}", flush=True)


def assemble(args: argparse.Namespace) -> None:
    """将同口径高精度文件与已复核的 pre L 独立积分合并，明确记录替换来源。

    只接受相同输入哈希、bin 几何、物理 cutoff 和积分分辨率；pre 的零强度
    reconstruction 路径必须显式标记。旧文件和原始积分警告均保留在审计来源。
    """
    base = json.loads(args.assemble_from.read_text())
    replacement = json.loads(args.replace_pre_linear_from.read_text())
    if base["identity"] != replacement["identity"] or base["positions"] != replacement["positions"]:
        raise ValueError("final bundle input/geometry identity mismatch")
    for key in ("nmu", "nrad", "qmin", "qmax", "cic", "qmc", "eps"):
        if base["config"][key] != replacement["config"][key]:
            raise ValueError(f"pre-L replacement differs in {key}")
    if not replacement["config"].get("pre_linear_zero_shift_qmc"):
        raise ValueError("replacement must identify the independent zero-shift QMC route")
    for field in FIELDS["L"]:
        base["components"]["pre"][field] = replacement["components"]["pre"][field]
        base["integrated_error_bounds"]["pre"][field] = replacement["integrated_error_bounds"]["pre"][field]
    base["warnings"] = [w for w in base["warnings"] if (w["stage"], w["sector"]) != ("pre", "L")]
    base["warnings"].extend(replacement["warnings"])
    base["assembly_sources"] = {
        "base": {"path": str(args.assemble_from.resolve()), "sha256": sha256(args.assemble_from)},
        "pre_L": {"path": str(args.replace_pre_linear_from.resolve()), "sha256": sha256(args.replace_pre_linear_from)},
        "pre_L_method": "post kernel with reconstruction strength 1e-30, independently integrated with QMC",
    }
    if args.replace_pre_lowk_from:
        lowk_path = args.replace_pre_lowk_from.resolve()
        lowk = json.loads(lowk_path.read_text())
        if base["identity"] != lowk["identity"]:
            raise ValueError("pre low-k replacement has different physics inputs")
        if set(lowk.get("components", {})) != {"pre"}:
            raise ValueError("pre low-k replacement must contain only the pre stage")
        lowk_positions = list(lowk["positions"])
        if not set(lowk_positions).issubset(base["positions"]):
            raise ValueError("pre low-k replacement positions are outside the final bundle")
        for key in ("qmin", "qmax", "cic", "nmu", "nrad"):
            if base["config"][key] != lowk["config"][key]:
                raise ValueError(f"pre low-k replacement differs in fixed setting {key}")
        if not lowk["config"].get("pre_linear_zero_shift_qmc"):
            raise ValueError("pre low-k replacement must use the independent zero-shift QMC route")
        for sector in ("G", "L", "Q"):
            for field in FIELDS[sector]:
                base_values = np.asarray(base["components"]["pre"][field], dtype=float)
                replacement_values = np.asarray(lowk["components"]["pre"][field], dtype=float)
                base_values[lowk_positions] = replacement_values[lowk_positions]
                base["components"]["pre"][field] = base_values.tolist()
                base_errors = np.asarray(base["integrated_error_bounds"]["pre"][field], dtype=float)
                replacement_errors = np.asarray(
                    lowk["integrated_error_bounds"]["pre"][field], dtype=float
                )
                base_errors[lowk_positions] = replacement_errors[lowk_positions]
                base["integrated_error_bounds"]["pre"][field] = base_errors.tolist()
        base["warnings"].extend(lowk["warnings"])
        base["assembly_sources"]["pre_lowk"] = {
            "path": str(lowk_path),
            "sha256": sha256(lowk_path),
        }
        base["assembly_sources"]["pre_lowk_method"] = (
            "positions 2,3,4 replaced in all G/L/Q sectors by the strict loopcheck "
            "template (epsrel=0.002, QMC power=14)"
        )
        base["adopted_pre_lowk_convergence"] = {
            "positions": lowk_positions,
            "config": lowk["config"],
        }
    base["status"] = "assembled; consult MODEL_AUDIT for numerical and simulation validation"
    out = args.output_dir / (args.name or "matched")
    out.with_suffix(".json").write_text(json.dumps(base, indent=2)+"\n")
    arrays = dict(diagonal_indices=np.asarray(base["pair_indices"]), diagonal_k=np.asarray(base["k"]))
    for stage in ("pre", "post"):
        for sector, field in (("G", "Btotal"), ("L", "dBdfNL_png_total"), ("Q", "B112II_fNL2_coefficient")):
            value = np.full(len(base["k"]), np.nan)
            value[base["positions"]] = np.asarray(base["components"][stage][field])[base["positions"]]
            arrays[f"{stage}_{sector}"] = value
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(f"Assembled {out}; remaining warning batches={len(base['warnings'])}")


def main() -> None:
    """解析精度等级和小样本检查参数；默认仅执行 coarse 等级。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=tuple(LEVELS), default="coarse")
    parser.add_argument("--positions", nargs="+", type=int)
    parser.add_argument("--stages", nargs="+", choices=("pre", "post"), default=["pre", "post"])
    parser.add_argument("--sectors", nargs="+", choices=("G", "L", "Q"), default=["G", "L", "Q"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--qmin", type=float, default=2*np.pi/1000.)
    parser.add_argument("--qmax", type=float, default=20.)
    parser.add_argument("--cic", type=int, default=4, choices=(0, 2, 4))
    parser.add_argument("--simulation-npz", type=Path, default=SIMULATION_NPZ)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--redshift", type=float, default=1.0)
    parser.add_argument("--smoothing-radius", type=float, default=15.0)
    parser.add_argument("--recon-cellsize", type=float, default=5.0)
    parser.add_argument("--scope")
    parser.add_argument("--generate-table", action="store_true")
    parser.add_argument("--fallback-plin-table", type=Path, default=FALLBACK_PLIN_TABLE)
    parser.add_argument("--a-s", type=float, default=2.126e-9)
    parser.add_argument("--n-s", type=float, default=0.9624)
    parser.add_argument("--h", type=float, default=0.6711)
    parser.add_argument("--pk-kmin", type=float, default=1.e-6)
    parser.add_argument("--pk-kmax", type=float, default=80.0)
    parser.add_argument("--pk-n", type=int, default=2400)
    parser.add_argument("--nmu", type=int)
    parser.add_argument("--nrad", type=int)
    parser.add_argument("--name")
    parser.add_argument("--export-bundle", action="store_true")
    parser.add_argument("--discrete-tree", action="store_true",
                        help="独立枚举低 k 的全部格点 mode pairs，仅计算 Gtree/Ltree。")
    parser.add_argument("--pre-linear-zero-shift-qmc", action="store_true",
                        help="独立积分检查：pre L 使用重构强度为1e-30的post QMC路径。")
    parser.add_argument("--assemble-from", type=Path)
    parser.add_argument("--replace-pre-linear-from", type=Path)
    parser.add_argument("--replace-pre-lowk-from", type=Path,
                        help="optionally replace pre G/L/Q at the variant's selected low-k positions")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.chunk_size < 1:
        parser.error("workers must be 1..8 and chunk-size must be positive")
    if args.discrete_tree:
        discrete_tree(args)
    elif args.assemble_from:
        if not args.replace_pre_linear_from:
            parser.error("assembly requires --replace-pre-linear-from")
        assemble(args)
    else:
        run(args)


def discrete_tree(args: argparse.Namespace) -> None:
    """精确枚举实测 Fourier shell 的全部 ordered mode pairs。

    默认只做前五个对角 bin（kmax=0.10）。利用格点模长平方为整数，预计算
    P/M 和 CIC 窗口，大幅降低枚举成本。包括 k=0 的计数归一化，但任意零
    外部边的场值置零；这样首 bin 的计数也和 HDF5 一致。输出是树级的独立
    诊断，不能将 tree 替换的混合结果宣传成精确格点一环。
    """
    from scipy.interpolate import CubicSpline
    start_time = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    simulation_npz = args.simulation_npz.resolve()
    table_path = args.table.resolve()
    with np.load(simulation_npz) as d:
        diag = np.flatnonzero(d["pair_i"] == d["pair_j"])
        edges, counts = d["hdf5_k_edges"][diag, 0], d["hdf5_nmodes"][diag]
        weighted = np.mean(d["hdf5_k_weighted"][diag], axis=1)
    positions = args.positions if args.positions is not None else [0, 1, 2, 3, 4]
    if any(weighted[p] > .1 for p in positions):
        raise ValueError("exact discrete tree is restricted to the requested low-k focus <=0.10")
    table = np.loadtxt(table_path)
    spl_p = CubicSpline(np.log(table[:, 0]), table[:, 1], bc_type="natural")
    spl_m = CubicSpline(np.log(table[:, 0]), table[:, 3], bc_type="natural")
    kf = 2*np.pi/1000.
    results = []
    for position in positions:
        lo, hi = edges[position]
        n = int(np.ceil(hi/kf))
        axis = np.arange(-n, n+1)
        lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        norms = np.sum(lattice*lattice, axis=1)
        modes = lattice[(norms*kf*kf >= lo*lo) & (norms*kf*kf < hi*hi)]
        size = len(modes)
        if size*size != int(round(counts[position])):
            raise ValueError(f"pair count mismatch bin {position}: {size*size} vs {counts[position]}")
        kval = kf*np.sqrt(np.arange(12*n*n+1))
        power, transfer = np.zeros(len(kval)), np.ones(len(kval))
        power[1:] = spl_p(np.log(kval[1:]))
        transfer[1:] = spl_m(np.log(kval[1:]))
        potential = power/transfer**2
        gaussian_window = np.exp(-.5*(args.smoothing_radius*kval)**2)
        axis_window = np.sinc(
            np.arange(-2*n, 2*n+1)*args.recon_cellsize*kf/(2*np.pi)
        )**4
        sums = np.zeros(3)
        max_chunk = max(1, 131072//size)
        for begin in range(0, size, max_chunk):
            a = np.repeat(modes[begin:begin+max_chunk], size, axis=0)
            b = np.tile(modes, (min(max_chunk, size-begin), 1))
            vectors = (a, b, -a-b)
            n2 = [np.sum(v*v, axis=1) for v in vectors]
            valid = (n2[0] > 0) & (n2[1] > 0) & (n2[2] > 0)
            safe = [np.maximum(v, 1) for v in n2]
            powers = [power[v] for v in n2]
            windows = [gaussian_window[v2]*np.prod(axis_window[v+2*n], axis=1)
                       for v, v2 in zip(vectors, n2)]
            pre, post = np.zeros(len(a)), np.zeros(len(a))
            for i, j in ((0, 1), (1, 2), (2, 0)):
                dot = np.sum(vectors[i]*vectors[j], axis=1)
                f2 = 5/7 + .5*dot*(1/safe[i]+1/safe[j]) + 2/7*dot*dot/(safe[i]*safe[j])
                shift = -.5*((1+dot/safe[i])*windows[i]+(1+dot/safe[j])*windows[j])
                pp = 2*powers[i]*powers[j]
                pre += pp*f2
                post += pp*(f2+shift)
            m = [transfer[v] for v in n2]
            phi = [potential[v] for v in n2]
            linear = 2*m[0]*m[1]*m[2]*(phi[0]*phi[1]+phi[1]*phi[2]+phi[2]*phi[0])
            sums += np.array([np.sum(pre[valid]), np.sum(post[valid]), np.sum(linear[valid])])
        means = sums/(size*size)
        row = dict(position=position, k=float(weighted[position]), n_modes=size,
                   n_pairs=size*size, mean_mode_k=float(np.mean(np.linalg.norm(modes, axis=1))*kf),
                   pre_Gtree=float(means[0]), post_Gtree=float(means[1]), Ltree=float(means[2]))
        results.append(row)
        print(f"discrete tree bin={position} Npairs={size*size} "
              f"post k²B={weighted[position]**2*means[1]:.6g}, elapsed={time.monotonic()-start_time:.1f}s", flush=True)
    dest = output / "discrete_tree_lowk.json"
    dest.write_text(json.dumps(dict(results=results, table_sha256=sha256(table_path),
        simulation_sha256=sha256(simulation_npz),
        method="Exact ordered lattice pairs, no-alias CIC4, zero-mode field=0, HDF5 normalization",
        limitation="Tree only: loop terms have not been replaced with exact lattice sums."), indent=2)+"\n")
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    main()
