#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""调用 ReACT/Copter 原始 `BSPT::Bloop` 做端到端三角形对照。

这个脚本的角色很窄：证明原始 C++ 软件包可以直接执行 GR/EdS
real-space bispectrum 1-loop，并且在同一个 Quijote `P_L(k,z=0.5)`
输入下，与本项目轻量 Python 实现使用同一套归一化约定。

实现口径
========
1. 用 `cosmoprimo` 导出 Quijote fiducial `P_L(k,z=0.5)` 表。
2. 编译 `src/react_adapter/react_bspt_driver.cpp`，并链接 ReACT/Copter 原始源码。
3. driver 内部手动设置 `Dl_spt=dnorm_spt=D_spt=1`，表示输入功率谱
   已经在目标红移，避免 ReACT 再额外乘 growth factor。
4. Python 端计算同一三角形的轻量实现结果，输出 JSON 对照表。

注意：ReACT driver 使用自适应 Cuba/Monte-Carlo 风格积分；本项目轻量
实现默认使用小节点 Gauss 网格。因此本脚本首先用于归一化/接口/量级
审计，不把 loop total 的逐百分点一致性作为硬性通过条件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_marisa_b_triangle import (  # noqa: E402
    prepare_adapter,
    resolve_gsl_flags,
)
from quijote_dm_spt1loop_realspace import (
    LoopConfig,
    bispectrum_total,
    build_quijote_linear_power,
    parse_loop_config_string,
    quijote_power_normalization_audit,
    triangle_vectors,
    vec_norm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MARISA_B_DATA_ROOT", PROJECT_ROOT)).resolve()
REACT_ROOT = PROJECT_ROOT / "external" / "ACTio-ReACTio" / "reactions"
REACT_SRC = REACT_ROOT / "src"
DRIVER_CPP = PROJECT_ROOT / "src" / "react_adapter" / "react_bspt_driver.cpp"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build" / "react_bspt_driver"
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "figures"
    / "diagnostics"
    / "quijote_dm_spt1loop_realspace"
)
MAX_LOCAL_THREADS = 8
THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def _capped_thread_value(value: str | None, max_threads: int = MAX_LOCAL_THREADS) -> str:
    try:
        parsed = int(value) if value is not None and value.strip() else max_threads
    except ValueError:
        parsed = max_threads
    return str(max(1, min(parsed, int(max_threads))))


def thread_limited_env(max_threads: int = MAX_LOCAL_THREADS) -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_LIMIT_ENV_KEYS:
        env[key] = _capped_thread_value(env.get(key), max_threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return env


def enforce_thread_limit(max_threads: int = MAX_LOCAL_THREADS) -> None:
    os.environ.update(thread_limited_env(max_threads))

REACT_SOURCES = [
    "BSPT.cpp",
    "Common.cpp",
    "Cosmology.cpp",
    "InterpolatedPS.cpp",
    "PowerSpectrum.cpp",
    "Quadrature.cpp",
    "SPT.cpp",
    "Spline.cpp",
    "array.cpp",
]


def parse_triangle(value: str) -> tuple[float, float, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Triangle should be formatted as k1,k2,mu12")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangle", action="append", type=parse_triangle, default=None, help="三角形 k1,k2,mu12；可重复。")
    parser.add_argument("--z", type=float, default=0.5, help="Quijote DM 目标红移。")
    parser.add_argument("--pk-kmin", type=float, default=1.0e-4, help="导出给 ReACT driver 的 P(k) 表最小 k。")
    parser.add_argument("--pk-kmax", type=float, default=80.0, help="导出给 ReACT driver 的 P(k) 表最大 k。")
    parser.add_argument("--pk-n", type=int, default=2400, help="导出给 ReACT driver 的 log-k 节点数。")
    parser.add_argument("--epsrel", type=float, default=1.0e-2, help="ReACT BSPT 三维积分相对精度。")
    parser.add_argument("--python-qmax", type=float, default=30.0, help="Python 轻量实现 qmax。")
    parser.add_argument("--python-nq", type=int, default=6, help="Python 轻量实现 log-q 节点数。")
    parser.add_argument("--python-nmu-loop", type=int, default=4, help="Python 轻量实现 loop mu 节点数。")
    parser.add_argument("--python-nphi", type=int, default=4, help="Python 轻量实现 loop phi 节点数。")
    parser.add_argument("--python-config", type=parse_loop_config_string, default=None, help="覆盖 Python loop 配置；支持 q/phi 分段字符串。")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="C++ driver 构建目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    parser.add_argument("--prefix", default="react_bspt_driver_triangle_audit", help="输出文件名前缀。")
    parser.add_argument("--skip-compile", action="store_true", help="跳过编译，直接使用 build-dir 中的二进制。")
    return parser.parse_args()


def write_pk_table(path: Path, z: float, kmin: float, kmax: float, n: int) -> dict[str, object]:
    pk, cosmo = build_quijote_linear_power(z)
    kvals = np.geomspace(float(kmin), float(kmax), int(n))
    pvals = np.asarray(pk(kvals), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# k[h/Mpc] P_L[(Mpc/h)^3], Quijote fiducial, cosmoprimo CLASS delta_cb\n")
        for kval, pval in zip(kvals, pvals):
            stream.write(f"{kval:.17e} {pval:.17e}\n")
    return {
        "path": str(path),
        "kmin": float(kvals[0]),
        "kmax": float(kvals[-1]),
        "n": int(len(kvals)),
        "p_min": float(np.min(pvals)),
        "p_max": float(np.max(pvals)),
        "cosmology": cosmo,
        "linear_power": pk.description,
    }


def compile_driver(binary: Path) -> dict[str, object]:
    binary.parent.mkdir(parents=True, exist_ok=True)
    adapter = prepare_adapter(binary.parent)
    bspt_header_copy = adapter["bspt_header"]
    bspt_cpp_copy = adapter["bspt_cpp"]
    sources = [str(bspt_cpp_copy) if name == "BSPT.cpp" else str(REACT_SRC / name) for name in REACT_SOURCES]
    special_functions_copy = adapter["special_functions"]
    beyond_lcdm_stubs = adapter["gr_stubs"]
    gsl_compile_flags, gsl_link_flags, gsl_metadata = resolve_gsl_flags(None)
    cmd = [
        "g++",
        "-std=gnu++17",
        "-O2",
        "-fopenmp",
        "-ffunction-sections",
        "-fdata-sections",
        "-DHAVE_CONFIG_H",
        f"-DDATADIR=\"{REACT_ROOT / 'data'}\"",
        "-I",
        str(binary.parent),
        "-I",
        str(REACT_ROOT),
        "-I",
        str(REACT_SRC),
        *gsl_compile_flags,
        str(DRIVER_CPP),
        *sources,
        str(special_functions_copy),
        str(beyond_lcdm_stubs),
        *gsl_link_flags,
        "-Wl,--gc-sections",
        "-o",
        str(binary),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=thread_limited_env(),
    )
    return {
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "binary": str(binary),
        "bspt_header_copy": str(bspt_header_copy),
        "bspt_cpp_copy": str(bspt_cpp_copy),
        "special_functions_copy": str(special_functions_copy),
        "beyond_lcdm_stubs": str(beyond_lcdm_stubs),
        "gsl": gsl_metadata,
    }


def run_driver(binary: Path, pk_table: Path, triangles: list[tuple[float, float, float]], epsrel: float) -> dict[str, object]:
    cmd = [str(binary), "--pk-table", str(pk_table), "--epsrel", f"{float(epsrel):.17g}"]
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
    return {
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_driver_table(stdout: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows: list[dict[str, object]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != len(header):
            continue
        row: dict[str, object] = {}
        for key, value in zip(header, parts):
            if key == "growth_mode":
                row[key] = value
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def python_rows(triangles: list[tuple[float, float, float]], z: float, config: LoopConfig) -> list[dict[str, float]]:
    pk, _ = build_quijote_linear_power(z)
    p13_cache: dict[float, float] = {}
    rows = []
    for k1, k2, mu12 in triangles:
        values = bispectrum_total(k1, k2, mu12, pk, config, p13_cache=p13_cache)
        rows.append(
            {
                "k1": float(k1),
                "k2": float(k2),
                "mu12": float(mu12),
                "k3": vec_norm(triangle_vectors(k1, k2, mu12)[2]),
                **{key: float(value) for key, value in values.items()},
            }
        )
    return rows


def compare_rows(react_rows: list[dict[str, object]], py_rows: list[dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for react, py in zip(react_rows, py_rows):
        btree_react = float(react["Btree"])
        btotal_react = float(react["Btotal_react"])
        b1loop_react = float(react["B1loop_react"])
        bloopterms_react = float(react["Bloopterms_react"])
        b321ii_react = float(react["B321II_react"])
        btree_py = float(py["Btree"])
        btotal_py = float(py["Btotal"])
        b1loop_py = float(py["B1loop"])
        bloopterms_py = float(py["B222"] + py["B321I"] + py["B411"])
        b321ii_py = float(py["B321II"])
        rows.append(
            {
                "k1": py["k1"],
                "k2": py["k2"],
                "mu12": py["mu12"],
                "k3": py["k3"],
                "react": react,
                "python": py,
                "relative_tree_difference": (btree_py / btree_react - 1.0) if btree_react != 0.0 else math.nan,
                "relative_total_difference": (btotal_py / btotal_react - 1.0) if btotal_react != 0.0 else math.nan,
                "relative_1loop_difference": (b1loop_py / b1loop_react - 1.0) if b1loop_react != 0.0 else math.nan,
                "loopterms_python_minus_react_over_tree": (bloopterms_py - bloopterms_react) / btree_react if btree_react != 0.0 else math.nan,
                "b321ii_python_minus_react_over_tree": (b321ii_py - b321ii_react) / btree_react if btree_react != 0.0 else math.nan,
                "loop_python_minus_react_over_tree": (b1loop_py - b1loop_react) / btree_react if btree_react != 0.0 else math.nan,
                "react_Bloopterms_over_tree": bloopterms_react / btree_react if btree_react != 0.0 else math.nan,
                "python_Bloopterms_over_tree": bloopterms_py / btree_react if btree_react != 0.0 else math.nan,
                "react_B321II_over_tree": b321ii_react / btree_react if btree_react != 0.0 else math.nan,
                "python_B321II_over_tree": b321ii_py / btree_react if btree_react != 0.0 else math.nan,
                "react_B1loop_over_tree": b1loop_react / btree_react if btree_react != 0.0 else math.nan,
                "python_B1loop_over_tree": b1loop_py / btree_react if btree_react != 0.0 else math.nan,
            }
        )
    return rows


def main() -> None:
    enforce_thread_limit()
    args = parse_args()
    triangles = args.triangle or [(0.05, 0.05, -0.5)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir)
    binary = build_dir / "react_bspt_driver"
    pk_table = output_dir / f"{args.prefix}_plin_z{args.z:g}.dat"

    pk_info = write_pk_table(pk_table, args.z, args.pk_kmin, args.pk_kmax, args.pk_n)
    compile_info = {"skipped": True, "binary": str(binary)}
    if not args.skip_compile:
        compile_info = compile_driver(binary)
        if compile_info["returncode"] != 0:
            payload = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "compile_failed",
                "pk_table": pk_info,
                "compile": compile_info,
            }
            out_json = output_dir / f"{args.prefix}.json"
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            raise SystemExit(1)

    driver_info = run_driver(binary, pk_table, triangles, args.epsrel)
    react_rows = parse_driver_table(driver_info["stdout"])
    py_config = args.python_config or LoopConfig(
        qmin=1.0e-4,
        qmax=float(args.python_qmax),
        nq=int(args.python_nq),
        nmu=int(args.python_nmu_loop),
        nphi=int(args.python_nphi),
    )
    py_rows = python_rows(triangles, args.z, py_config)
    comparisons = compare_rows(react_rows, py_rows)
    tree_ok = all(abs(float(row["relative_tree_difference"])) < 5.0e-4 for row in comparisons)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if driver_info["returncode"] == 0 and tree_ok else "check",
        "purpose": "Original ReACT BSPT::Bloop endpoint audit with the same Quijote P_L(z=0.5) table.",
        "triangles": [list(item) for item in triangles],
        "react_epsrel": float(args.epsrel),
        "python_loop_config": asdict(py_config),
        "pk_table": pk_info,
        "power_normalization_audit": quijote_power_normalization_audit(args.z),
        "compile": compile_info,
        "driver": driver_info,
        "react_rows": react_rows,
        "python_rows": py_rows,
        "comparisons": comparisons,
        "checks": {
            "react_driver_returncode_zero": driver_info["returncode"] == 0,
            "tree_matches_same_pk_to_5e_4": tree_ok,
            "loop_note": "Loop totals compare different numerical integrators; use this field as a diagnostic, not a production convergence claim.",
        },
    }
    out_json = output_dir / f"{args.prefix}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
