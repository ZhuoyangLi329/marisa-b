#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""运行 MARISA-B 单三角形计算，并保留旧 v0 与新 halo bias-v1 的隔离边界。

执行大纲
--------
1. 解析三角形、功率谱、积分与 bias/reconstruction 参数；
2. 编译 C++ native 核心，或显式复用已经存在的 binary；
3. Python 只负责调度，所有 loop integration 留在 C++；
4. 解析 C++ JSON，并保存逐 diagram 的 JSON/NPZ provenance；
5. 仅在旧 pre-recon DM mode 下可选调用 ReACT oracle 做回归比较。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_marisa_b_triangle import (  # noqa: E402
    PROJECT_ROOT,
    build_marisa_b_triangle,
    enforce_thread_limit,
    thread_limited_env,
)
from audit_quijote_dm_spt_react_bspt_driver import (
    DEFAULT_BUILD_DIR as REACT_DRIVER_BUILD_DIR,
    compile_driver,
    parse_driver_table,
    run_driver,
    write_pk_table,
)


DEFAULT_BUILD_DIR = PROJECT_ROOT / "build" / "marisa_b"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "theory_vectors" / "marisa_b_v0"
DEFAULT_PREFIX = "marisa_b_v0_triangle_smoke"
HALO_BIAS_V1_DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "analysis" / "theory_vectors" / "marisa_b_halo_bias_v1"
)
HALO_BIAS_V1_DEFAULT_PREFIX = "marisa_b_halo_bias_v1_triangle_smoke"

HALO_BIAS_V1_MODES = frozenset(
    {
        "pre_recon_halo_bias_v1_gaussian",
        "post_recon_halo_bias_v1_gaussian",
        "pre_recon_halo_bias_v1_local_png_tree",
        "post_recon_halo_bias_v1_local_png_tree",
        "pre_recon_halo_bias_v1_local_png_1loop_truncated",
        "post_recon_halo_bias_v1_local_png_1loop_truncated",
    }
)

HALO_BIAS_V1_PNG_MODES = frozenset(
    {
        "pre_recon_halo_bias_v1_local_png_tree",
        "post_recon_halo_bias_v1_local_png_tree",
        "pre_recon_halo_bias_v1_local_png_1loop_truncated",
        "post_recon_halo_bias_v1_local_png_1loop_truncated",
    }
)

COMPONENTS = (
    "Btree",
    "B222",
    "B321I",
    "B321II",
    "B411",
    "Bloopterms",
    "B1loop",
    "Btotal",
    "dBdfNL_local_tree",
    "dBdfNL_local_B122I",
    "dBdfNL_local_B122II",
    "dBdfNL_local_B113I",
    "dBdfNL_local_B113II",
    "dBdfNL_local_1loop",
    "dBdfNL_local_total",
    "B112II_fNL2_coefficient",
)

OPTIONAL_HALO_BIAS_V1_COMPONENTS = (
    "dBdfNL_local_B222",
    "dBdfNL_local_B321I",
    "dBdfNL_local_B321II",
    "dBdfNL_local_B411",
    "dBdfNL_local_primordial",
    "dBdfNL_local_bphi_f2",
    "dBdfNL_local_bphi_advection",
    "dBdfNL_local_bphidelta",
    "dBdfNL_local_bphi_b2",
    "dBdfNL_local_bphi_bK2",
    "dBdfNL_local_bphi_reconstruction",
    "Bhalo_tree_fNL2_bphi_B0",
    "Bhalo_tree_fNL2_bphi_sq_advection",
    "Bhalo_tree_fNL2_bphi_sq_F2",
    "Bhalo_tree_fNL2_bphi_sq_b2",
    "Bhalo_tree_fNL2_bphi_sq_bK2",
    "Bhalo_tree_fNL2_bphi_sq_reconstruction",
    "Bhalo_tree_fNL2_bphi_bphidelta",
    "Bhalo_tree_fNL2_bphi2_operator",
    "Bhalo_tree_fNL2_deterministic",
    "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis",
    "stochastic_alpha3_basis",
    "stochastic_alpha4_basis",
    "dBdfNL_stochastic_alpha3_basis",
)


def parse_triangle(value: str) -> tuple[float, float, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Triangle should be formatted as k1,k2,mu12")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangle", action="append", type=parse_triangle, default=None, help="Triangle k1,k2,mu12; may repeat.")
    parser.add_argument("--z", type=float, default=0.5, help="Target redshift for generated Quijote linear power.")
    parser.add_argument("--pk-table", default=None, help="Reuse an existing two-column P_L(k) table.")
    parser.add_argument("--pk-kmin", type=float, default=1.0e-4)
    parser.add_argument("--pk-kmax", type=float, default=80.0)
    parser.add_argument("--pk-n", type=int, default=2400)
    parser.add_argument("--epsrel", type=float, default=1.0e-3)
    parser.add_argument("--p13-epsrel", type=float, default=1.0e-2, help="Native P13 precision; ReACT BSPT::Bloop uses 1e-2 internally.")
    parser.add_argument("--qmin", type=float, default=1.0e-4)
    parser.add_argument("--qmax", type=float, default=30.0)
    parser.add_argument(
        "--png-ir-cutoff",
        type=float,
        default=0.0,
        help="IR cutoff applied to every primordial P_phi propagator; 0 reuses qmin.",
    )
    parser.add_argument(
        "--b112ii-integrator",
        choices=("adaptive", "multicenter_qmc"),
        default="adaptive",
    )
    parser.add_argument("--b112ii-qmc-power", type=int, default=12)
    parser.add_argument("--b112ii-qmc-replicates", type=int, default=4)
    parser.add_argument(
        "--fnl",
        type=float,
        default=0.0,
        help="Finite fNL used only by explicit *_local_png_finite_1loop modes.",
    )
    parser.add_argument("--smoothing-radius", type=float, default=10.0)
    parser.add_argument("--bias-recon", type=float, default=1.0)
    parser.add_argument("--recon-cellsize", type=float, default=0.0)
    parser.add_argument("--recon-cic-window-power", type=int, default=0)
    parser.add_argument("--singular-floor", type=float, default=1.0e-5)
    parser.add_argument("--png-table", default=None, help="Four-column k, P_L, P_phi, M table for local-PNG modes.")
    parser.add_argument("--nowiggle-table", default=None, help="Two-column P_nw(k) table for post_recon_gaussian_ir_nowiggle mode.")
    parser.add_argument("--ir-sigma2", type=float, default=-1.0, help="IR damping sigma^2 for post_recon_gaussian_ir_nowiggle mode.")
    parser.add_argument("--b1", type=float, default=1.0, help="Linear bias multiplying the local PNG tree kernel; DM uses b1=1.")
    parser.add_argument("--b2", type=float, default=0.0, help="halo bias-v1 的 quadratic density bias；旧 DM mode 忽略。")
    parser.add_argument("--bk2", type=float, default=0.0, help="halo bias-v1 的 tidal bias bK2；旧 DM mode 忽略。")
    parser.add_argument("--bphi", type=float, default=0.0, help="local-PNG response bias bphi；只用于 halo PNG mode。")
    parser.add_argument("--bphidelta", type=float, default=0.0, help="local-PNG mixed bias bphi_delta；只用于 halo PNG mode。")
    parser.add_argument("--bphi2", type=float, default=0.0, help="local-PNG phi^2 operator bias b_phi2；只用于 halo PNG tree mode。")
    parser.add_argument("--mode", default="pre_recon_gaussian")
    parser.add_argument("--backend", default="react_gr_bootstrap")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "输出目录；省略时旧 mode 保持 marisa_b_v0，新 halo bias-v1 mode "
            "自动隔离到 marisa_b_halo_bias_v1。"
        ),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="输出前缀；省略时按旧 v0 / 新 halo bias-v1 mode 选择隔离前缀。",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compare-react-driver", action="store_true", help="Run the existing ReACT driver and write component residuals.")
    return parser.parse_args()


def resolve_output_routing(
    mode: str,
    output_dir: str | Path | None,
    prefix: str | None,
) -> tuple[Path, str]:
    """解析 mode-aware 默认产物位置，同时原样保留所有显式用户选择。"""

    is_halo_bias_v1 = str(mode) in HALO_BIAS_V1_MODES
    resolved_dir = Path(output_dir) if output_dir is not None else (
        HALO_BIAS_V1_DEFAULT_OUTPUT_DIR if is_halo_bias_v1 else DEFAULT_OUTPUT_DIR
    )
    resolved_prefix = str(prefix) if prefix is not None else (
        HALO_BIAS_V1_DEFAULT_PREFIX if is_halo_bias_v1 else DEFAULT_PREFIX
    )
    return resolved_dir, resolved_prefix


def validate_output_routing(output_dir: Path, prefix: str, mode: str) -> None:
    """Never allow a new halo-v1 run to land in the frozen v0 namespace."""

    if str(mode) not in HALO_BIAS_V1_MODES:
        return
    resolved_output = output_dir.resolve()
    resolved_v0 = DEFAULT_OUTPUT_DIR.resolve()
    if resolved_output == resolved_v0 or resolved_v0 in resolved_output.parents:
        raise ValueError(
            "halo bias-v1 mode cannot write inside the frozen marisa_b_v0 namespace; "
            "choose a halo-v1 --output-dir"
        )
    if str(prefix).startswith("marisa_b_v0"):
        raise ValueError(
            "halo bias-v1 mode cannot use a marisa_b_v0* prefix; choose a halo-v1 --prefix"
        )


def refuse_halo_bias_v1_output_collisions(paths: list[Path], mode: str) -> None:
    """新 namespace 禁止静默覆盖；旧 mode 的历史覆盖行为保持不变。"""

    if str(mode) not in HALO_BIAS_V1_MODES:
        return
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "拒绝覆盖已有 halo bias-v1 产物，请显式更换 --prefix 或 --output-dir：\n"
            + "\n".join(collisions)
        )


def compile_marisa_b(binary: Path) -> dict[str, object]:
    """Build through the canonical repo-local ReACT adapter helper."""

    gsl_prefix = (
        os.environ.get("GSL_PREFIX", "").strip()
        or os.environ.get("MARISA_B_GSL_ROOT", "").strip()
    )
    return build_marisa_b_triangle(
        binary=binary,
        build_dir=binary.parent / "marisa_b_triangle_adapter",
        gsl_prefix=gsl_prefix or None,
        max_threads=28,
    )


def run_marisa_b(
    binary: Path,
    pk_table: Path,
    triangles: list[tuple[float, float, float]],
    epsrel: float,
    p13_epsrel: float,
    qmin: float,
    qmax: float,
    smoothing_radius: float,
    bias_recon: float,
    recon_cellsize: float,
    recon_cic_window_power: int,
    singular_floor: float,
    mode: str,
    backend: str,
    png_table: Path | None = None,
    nowiggle_table: Path | None = None,
    ir_sigma2: float | None = None,
    b1: float = 1.0,
    b2: float = 0.0,
    bK2: float = 0.0,
    bphi: float = 0.0,
    bphidelta: float = 0.0,
    bphi2: float = 0.0,
    png_ir_cutoff: float | None = None,
    b112ii_integrator: str = "adaptive",
    b112ii_qmc_power: int = 12,
    b112ii_qmc_replicates: int = 4,
    fnl: float = 0.0,
) -> dict[str, object]:
    """调用一次 MARISA-B C++ binary 并返回可审计运行记录。

    参数包含 P_L 表、三角形列表、积分精度、重构设置和 bias 系数；返回字典
    保存命令、return code、stdout/stderr 与解析后的 JSON payload。只有 halo
    bias-v1 mode 才把 b2/bK2 传入 C++，从而兼容旧 binary/旧 mode 的调用口径。
    """
    cmd = [
        str(binary),
        "--pk-table",
        str(pk_table),
        "--epsrel",
        f"{float(epsrel):.17g}",
        "--p13-epsrel",
        f"{float(p13_epsrel):.17g}",
        "--qmin",
        f"{float(qmin):.17g}",
        "--qmax",
        f"{float(qmax):.17g}",
        "--smoothing-radius",
        f"{float(smoothing_radius):.17g}",
        "--bias-recon",
        f"{float(bias_recon):.17g}",
        "--recon-cellsize",
        f"{float(recon_cellsize):.17g}",
        "--recon-cic-window-power",
        str(int(recon_cic_window_power)),
        "--singular-floor",
        f"{float(singular_floor):.17g}",
        "--mode",
        str(mode),
        "--backend",
        str(backend),
    ]
    if png_table is not None:
        cmd.extend(["--png-table", str(png_table)])
    if nowiggle_table is not None:
        cmd.extend(["--nowiggle-table", str(nowiggle_table)])
    if ir_sigma2 is not None:
        cmd.extend(["--ir-sigma2", f"{float(ir_sigma2):.17g}"])
    if png_ir_cutoff is not None:
        cmd.extend(["--png-ir-cutoff", f"{float(png_ir_cutoff):.17g}"])
    if "b112ii" in str(mode) or "local_png_finite_1loop" in str(mode):
        cmd.extend(
            [
                "--b112ii-integrator",
                str(b112ii_integrator),
                "--b112ii-qmc-power",
                str(int(b112ii_qmc_power)),
                "--b112ii-qmc-replicates",
                str(int(b112ii_qmc_replicates)),
            ]
        )
    if "local_png_finite_1loop" in str(mode):
        cmd.extend(["--fnl", f"{float(fnl):.17g}"])
    cmd.extend(["--b1", f"{float(b1):.17g}"])
    if str(mode) in HALO_BIAS_V1_MODES:
        cmd.extend(["--b2", f"{float(b2):.17g}", "--bk2", f"{float(bK2):.17g}"])
    if str(mode) in HALO_BIAS_V1_PNG_MODES:
        cmd.extend(
            [
                "--bphi",
                f"{float(bphi):.17g}",
                "--bphidelta",
                f"{float(bphidelta):.17g}",
                "--bphi2",
                f"{float(bphi2):.17g}",
            ]
        )
    for k1, k2, mu12 in triangles:
        cmd.extend(["--triangle", f"{k1:.17g},{k2:.17g},{mu12:.17g}"])
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=thread_limited_env(),
    )
    payload = None
    if result.returncode == 0:
        payload = json.loads(result.stdout)
    return {
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "payload": payload,
    }


def describe_existing_pk_table(path: Path) -> dict[str, object]:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"P(k) table must have at least two columns: {path}")
    kvals = values[:, 0]
    pvals = values[:, 1]
    return {
        "path": str(path),
        "source": "reused_existing_table",
        "kmin": float(np.min(kvals)),
        "kmax": float(np.max(kvals)),
        "n": int(kvals.size),
        "p_min": float(np.min(pvals)),
        "p_max": float(np.max(pvals)),
    }


def compare_with_react_driver(
    react_rows: list[dict[str, object]],
    marisa_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    react_key = {
        "Btree": "Btree",
        "B222": "B222_react",
        "B321I": "B321I_react",
        "B321II": "B321II_react",
        "B411": "B411_react",
        "Bloopterms": "Bloopterms_react",
        "B1loop": "B1loop_react",
        "Btotal": "Btotal_react",
    }
    for idx, (react, marisa) in enumerate(zip(react_rows, marisa_rows)):
        btree = float(react["Btree"])
        row: dict[str, object] = {
            "triangle_index": int(idx),
            "k1": float(marisa["k1"]),
            "k2": float(marisa["k2"]),
            "mu12": float(marisa["mu12"]),
            "react_Btree": btree,
        }
        for component in react_key:
            mval = float(marisa[component])
            rval = float(react[react_key[component]])
            row[f"delta_{component}_over_react_tree"] = (mval - rval) / btree if btree != 0.0 else math.nan
        out.append(row)
    return out


def save_npz(path: Path, payload: dict[str, object]) -> None:
    rows = payload["results"]
    arrays = {
        "k1": np.asarray([row["k1"] for row in rows], dtype=np.float64),
        "k2": np.asarray([row["k2"] for row in rows], dtype=np.float64),
        "mu12": np.asarray([row["mu12"] for row in rows], dtype=np.float64),
        "k3": np.asarray([row["k3"] for row in rows], dtype=np.float64),
    }
    for component in COMPONENTS:
        arrays[component] = np.asarray([row[component] for row in rows], dtype=np.float64)
    for component in OPTIONAL_HALO_BIAS_V1_COMPONENTS:
        if component in rows[0]:
            arrays[component] = np.asarray([row[component] for row in rows], dtype=np.float64)
    np.savez(path, **arrays)


def main() -> None:
    enforce_thread_limit()
    args = parse_args()
    triangles = args.triangle or [(0.05, 0.05, -0.5)]
    output_dir, output_prefix = resolve_output_routing(
        args.mode, args.output_dir, args.prefix
    )
    validate_output_routing(output_dir, output_prefix, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir)
    binary = build_dir / "marisa_b_triangle"

    collision_candidates = [
        output_dir / f"{output_prefix}.json",
        output_dir / f"{output_prefix}.npz",
    ]
    if not args.pk_table:
        collision_candidates.append(output_dir / f"{output_prefix}_plin_z{args.z:g}.dat")
    refuse_halo_bias_v1_output_collisions(collision_candidates, args.mode)

    if args.pk_table:
        pk_table = Path(args.pk_table)
        pk_info = describe_existing_pk_table(pk_table)
    else:
        pk_table = output_dir / f"{output_prefix}_plin_z{args.z:g}.dat"
        pk_info = write_pk_table(pk_table, args.z, args.pk_kmin, args.pk_kmax, args.pk_n)

    compile_info: dict[str, object] = {"skipped": True, "binary": str(binary)}
    if not args.skip_compile:
        compile_info = compile_marisa_b(binary)
        if compile_info["returncode"] != 0:
            payload = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "compile_failed",
                "pk_table": pk_info,
                "compile": compile_info,
            }
            out_json = output_dir / f"{output_prefix}.json"
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            raise SystemExit(1)

    run_info = run_marisa_b(
        binary,
        pk_table,
        triangles,
        args.epsrel,
        args.p13_epsrel,
        args.qmin,
        args.qmax,
        args.smoothing_radius,
        args.bias_recon,
        args.recon_cellsize,
        args.recon_cic_window_power,
        args.singular_floor,
        args.mode,
        args.backend,
        png_table=Path(args.png_table) if args.png_table else None,
        nowiggle_table=Path(args.nowiggle_table) if args.nowiggle_table else None,
        ir_sigma2=float(args.ir_sigma2) if float(args.ir_sigma2) >= 0.0 else None,
        b1=args.b1,
        b2=args.b2,
        bK2=args.bk2,
        bphi=args.bphi,
        bphidelta=args.bphidelta,
        bphi2=args.bphi2,
        png_ir_cutoff=float(args.png_ir_cutoff),
        b112ii_integrator=str(args.b112ii_integrator),
        b112ii_qmc_power=int(args.b112ii_qmc_power),
        b112ii_qmc_replicates=int(args.b112ii_qmc_replicates),
        fnl=float(args.fnl),
    )
    if run_info["returncode"] != 0 or run_info["payload"] is None:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "run_failed",
            "pk_table": pk_info,
            "compile": compile_info,
            "marisa_b": run_info,
        }
        out_json = output_dir / f"{output_prefix}.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    react_info = None
    comparisons = None
    status = "ok"
    if args.compare_react_driver and args.mode == "pre_recon_gaussian":
        react_binary = Path(REACT_DRIVER_BUILD_DIR) / "react_bspt_driver"
        react_compile = {"skipped": bool(args.skip_compile), "binary": str(react_binary)}
        if not args.skip_compile or not react_binary.exists():
            react_compile = compile_driver(react_binary)
        if int(react_compile.get("returncode", 0)) != 0:
            react_info = {"returncode": 2, "stdout": "", "stderr": "react driver compile failed", "compile": react_compile}
            react_rows = []
        else:
            react_info = run_driver(binary=react_binary, pk_table=pk_table, triangles=triangles, epsrel=args.epsrel)
            react_info["compile"] = react_compile
            react_rows = parse_driver_table(str(react_info["stdout"]))
        if int(react_info["returncode"]) != 0 or len(react_rows) != len(triangles):
            status = "react_compare_failed"
        else:
            comparisons = compare_with_react_driver(react_rows, run_info["payload"]["results"])
            max_abs = max(
                abs(float(value))
                for row in comparisons
                for key, value in row.items()
                if key.startswith("delta_") and math.isfinite(float(value))
            )
            if max_abs > 1.0e-10:
                status = "check"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "purpose": (
            ("MARISA-B halo bias-v1 single-triangle run; " if args.mode in HALO_BIAS_V1_MODES
             else "MARISA-B v0 single-triangle run; ")
            + "Python orchestrates only, C++ performs integration. "
            + f"backend={args.backend}"
        ),
        "triangles": [list(item) for item in triangles],
        "pk_table": pk_info,
        "png_table": str(args.png_table) if args.png_table else None,
        "b1": float(args.b1),
        "qmin": float(args.qmin),
        "qmax": float(args.qmax),
        "png_ir_cutoff": float(args.png_ir_cutoff),
        "b112ii_integrator": str(args.b112ii_integrator),
        "b112ii_qmc_power": int(args.b112ii_qmc_power),
        "b112ii_qmc_replicates": int(args.b112ii_qmc_replicates),
        "fnl": float(args.fnl),
        "smoothing_radius": float(args.smoothing_radius),
        "bias_recon": float(args.bias_recon),
        "recon_cellsize": float(args.recon_cellsize),
        "recon_cic_window_power": int(args.recon_cic_window_power),
        "singular_floor": float(args.singular_floor),
        "compile": compile_info,
        "marisa_b": {key: value for key, value in run_info.items() if key != "stdout"},
        "react_compare": react_info,
        "component_comparisons": comparisons,
        "acceptance_note": (
            "This validates the MARISA-B CLI/schema/Python boundary. "
            "The native_cpp backend is the independent pre-recon Gaussian loop-kernel rewrite; "
            "react_gr_bootstrap is retained only as an oracle/interface test."
        ),
    }
    if args.mode in HALO_BIAS_V1_MODES:
        payload["b2"] = float(args.b2)
        payload["bK2"] = float(args.bk2)
    if args.mode in HALO_BIAS_V1_PNG_MODES:
        payload["bphi"] = float(args.bphi)
        payload["bphidelta"] = float(args.bphidelta)
        payload["bphi2"] = float(args.bphi2)

    out_json = output_dir / f"{output_prefix}.json"
    out_npz = output_dir / f"{output_prefix}.npz"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_npz(out_npz, run_info["payload"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
