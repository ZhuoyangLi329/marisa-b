#!/usr/bin/env python
"""扫描 Quijote DM SPT 1-loop 数值积分收敛性。

这个脚本服务当前 real-space pre-recon DM 1-loop 工作流：

1. 对少量固定三角形扫描 `(qmax,nq,nmu,nphi)`，比较
   `B222/B321I/B321II/B411/B1loop/Btotal`。
2. 可选地对少量 Sugiyama `B000(k1,k2)` pair 做同样扫描，并和
   Quijote fiducial pre-recon mean 比较。

注意：本脚本是诊断工具，不是生产调度器。完整 120 维 1-loop 需要
后续缓存/并行化或批处理。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quijote_dm_spt1loop_realspace import (
    DEFAULT_DM_MATRIX,
    DEFAULT_OUTPUT_DIR,
    LoopConfig,
    bispectrum_total,
    build_quijote_linear_power,
    load_quijote_dm_pre_mean,
    loop_config_label,
    parse_loop_config_string,
    project_b000,
    quijote_power_normalization_audit,
    run_kernel_checks,
)


COMPONENTS = ("B222", "B321I", "B321II", "B411", "B1loop")
B000_COMPONENTS = ("B000_B222", "B000_B321I", "B000_B321II", "B000_B411", "B000_B1loop")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_triangle(value: str) -> tuple[float, float, float]:
    """解析 `k1,k2,mu12`。"""

    parts = [float(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("triangle must be formatted as k1,k2,mu12")
    return parts[0], parts[1], parts[2]


def parse_config(value: str) -> LoopConfig:
    """解析 loop 配置；支持普通、分段 q 和分段 phi 积分格式。"""

    return parse_loop_config_string(value)


def config_label(config: LoopConfig) -> str:
    return loop_config_label(config)


def select_theory_k_pairs(data: dict[str, np.ndarray], indices: np.ndarray, k_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.column_stack((data["k1"], data["k2"]))[indices]
    weighted = data["hdf5_k_weighted"][indices]
    if k_source == "weighted":
        return nominal, weighted, weighted
    if k_source == "center":
        return nominal, weighted, nominal
    raise ValueError(f"Unknown k_source {k_source!r}")


def run_triangle_scan(triangles: list[tuple[float, float, float]], configs: list[LoopConfig], pk) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    p13_caches = {config_label(config): {} for config in configs}
    for tri_index, (k1, k2, mu12) in enumerate(triangles):
        for config in configs:
            label = config_label(config)
            values = bispectrum_total(float(k1), float(k2), float(mu12), pk, config, p13_cache=p13_caches[label])
            rows.append(
                {
                    "triangle_index": int(tri_index),
                    "k1": float(k1),
                    "k2": float(k2),
                    "mu12": float(mu12),
                    "config_label": label,
                    "loop_config": asdict(config),
                    **{key: float(value) for key, value in values.items()},
                    "B1loop_over_Btree": float(values["B1loop"] / values["Btree"]) if values["Btree"] != 0.0 else float("nan"),
                    "p13_cache_size": int(len(p13_caches[label])),
                }
            )
    return rows


def run_b000_scan(
    data: dict[str, np.ndarray],
    pair_indices: list[int],
    k_source: str,
    configs: list[LoopConfig],
    nmu_b000: int,
    pk,
) -> list[dict[str, object]]:
    indices = np.asarray(pair_indices, dtype=np.int64)
    nominal, weighted, theory_pairs = select_theory_k_pairs(data, indices, k_source)
    rows: list[dict[str, object]] = []
    p13_caches = {config_label(config): {} for config in configs}
    for local_index, pair_index in enumerate(indices):
        sim = float(data["mean"][pair_index])
        err = float(data["stderr"][pair_index])
        for config in configs:
            label = config_label(config)
            values = project_b000(
                float(theory_pairs[local_index, 0]),
                float(theory_pairs[local_index, 1]),
                pk,
                config,
                nmu_b000=int(nmu_b000),
                include_loop=True,
                p13_cache=p13_caches[label],
            )
            rows.append(
                {
                    "pair_index": int(pair_index),
                    "pair_i": int(data["pair_i"][pair_index]),
                    "pair_j": int(data["pair_j"][pair_index]),
                    "k_source": k_source,
                    "nominal_k_pair": [float(item) for item in nominal[local_index]],
                    "weighted_k_pair": [float(item) for item in weighted[local_index]],
                    "theory_k_pair": [float(item) for item in theory_pairs[local_index]],
                    "config_label": label,
                    "loop_config": asdict(config),
                    "nmu_b000": int(nmu_b000),
                    "p13_cache_size": int(len(p13_caches[label])),
                    "sim_B000": sim,
                    "sim_B000_stderr": err,
                    **{key: float(value) for key, value in values.items()},
                    "fractional_residual_total": float(values["B000_Btotal"] / sim - 1.0) if sim != 0.0 else float("nan"),
                    "fractional_residual_tree": float(values["B000_Btree"] / sim - 1.0) if sim != 0.0 else float("nan"),
                }
            )
    return rows


def summarize_b000_scan(b000_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """按 pair 汇总 B000 convergence scan，方便快速判断是否稳定。"""

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in b000_rows:
        grouped.setdefault(int(row["pair_index"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for pair_index in sorted(grouped):
        rows = grouped[pair_index]
        first = rows[0]
        k_pair = np.asarray(first["theory_k_pair"], dtype=np.float64)
        total_by_config = {str(row["config_label"]): float(row["fractional_residual_total"]) for row in rows}
        loop_over_tree = {
            str(row["config_label"]): float(row["B000_B1loop"] / row["B000_Btree"]) if float(row["B000_Btree"]) != 0.0 else float("nan")
            for row in rows
        }
        components_over_tree = {
            str(row["config_label"]): {
                component: float(row[component] / row["B000_Btree"]) if float(row["B000_Btree"]) != 0.0 else float("nan")
                for component in B000_COMPONENTS
            }
            for row in rows
        }
        component_spreads = {}
        for component in B000_COMPONENTS:
            values = np.asarray([components_over_tree[str(row["config_label"])][component] for row in rows], dtype=np.float64)
            component_spreads[component] = float(np.nanmax(values) - np.nanmin(values)) if values.size else float("nan")
        totals = np.asarray(list(total_by_config.values()), dtype=np.float64)
        summary_rows.append(
            {
                "pair_index": int(pair_index),
                "pair_i": int(first["pair_i"]),
                "pair_j": int(first["pair_j"]),
                "k_eff": float(np.sqrt(k_pair[0] * k_pair[1])),
                "tree_fractional_residual": float(first["fractional_residual_tree"]),
                "total_fractional_residual_by_config": total_by_config,
                "B1loop_over_Btree_by_config": loop_over_tree,
                "component_over_Btree_by_config": components_over_tree,
                "component_over_Btree_spread": component_spreads,
                "total_fractional_residual_spread": float(np.nanmax(totals) - np.nanmin(totals)) if totals.size else float("nan"),
            }
        )
    return summary_rows


def rows_to_npz_dict(prefix: str, rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """把 list-of-dict 中的数值标量字段转成 npz-friendly arrays。"""

    if not rows:
        return {}
    numeric_keys = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)
    return {f"{prefix}_{key}": np.asarray([row[key] for row in rows], dtype=np.float64) for key in numeric_keys}


def plot_triangle_scan(ax, triangle_rows: list[dict[str, object]]) -> None:
    if not triangle_rows:
        ax.text(0.5, 0.5, "no triangle scan", ha="center", va="center", transform=ax.transAxes)
        return
    labels = list(dict.fromkeys(row["config_label"] for row in triangle_rows))
    x = np.arange(len(labels))
    for component in COMPONENTS:
        values = []
        for label in labels:
            matched = [row for row in triangle_rows if row["config_label"] == label and row["triangle_index"] == 0]
            values.append(matched[0][component] if matched else np.nan)
        ax.plot(x, values, marker="o", ms=3.0, lw=1.1, label=component)
    ax.axhline(0.0, color="0.45", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"$B$ component")
    ax.set_title("single-triangle loop convergence")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, ncol=3)


def plot_b000_scan(ax, b000_rows: list[dict[str, object]]) -> None:
    if not b000_rows:
        ax.text(0.5, 0.5, "no B000 scan", ha="center", va="center", transform=ax.transAxes)
        return
    labels = list(dict.fromkeys(row["config_label"] for row in b000_rows))
    pair_indices = list(dict.fromkeys(int(row["pair_index"]) for row in b000_rows))
    x = np.arange(len(labels))
    for pair_index in pair_indices:
        values = []
        for label in labels:
            matched = [row for row in b000_rows if int(row["pair_index"]) == pair_index and row["config_label"] == label]
            values.append(matched[0]["fractional_residual_total"] if matched else np.nan)
        ax.plot(x, values, marker="o", ms=3.0, lw=1.1, label=f"pair {pair_index}")
    ax.axhline(0.0, color="0.45", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("total / sim - 1")
    ax.set_title("selected B000 residual convergence")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, ncol=3)


def write_pdf(path: Path, triangle_rows: list[dict[str, object]], b000_rows: list[dict[str, object]]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), constrained_layout=True)
    plot_triangle_scan(axes[0], triangle_rows)
    plot_b000_scan(axes[1], b000_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_DM_MATRIX))
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--triangle", action="append", type=parse_triangle, default=None, help="k1,k2,mu12；默认等边 k=0.05。")
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        default=None,
        help="qmax,nq,nmu,nphi；也可用 qmax,nq,nmu,nphi,qsplit1:...,nqseg,phisplit1:...,nphiseg；可重复。",
    )
    parser.add_argument("--skip-b000", action="store_true")
    parser.add_argument("--pair-index", action="append", type=int, default=None, help="B000 pair index；默认 0,1,2。")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--nmu-b000", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="quijote_dm_spt1loop_convergence_scan")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    args = parse_args()
    triangles = args.triangle if args.triangle is not None else [(0.05, 0.05, -0.5)]
    configs = args.config if args.config is not None else [
        LoopConfig(qmin=1.0e-4, qmax=10.0, nq=6, nmu=4, nphi=4),
        LoopConfig(qmin=1.0e-4, qmax=30.0, nq=8, nmu=5, nphi=5),
    ]

    pk, cosmo_params = build_quijote_linear_power(args.z)
    triangle_rows = run_triangle_scan(triangles, configs, pk)
    data = load_quijote_dm_pre_mean(Path(args.matrix))
    pair_indices = args.pair_index if args.pair_index is not None else [0, 1, 2]
    b000_rows = [] if args.skip_b000 else run_b000_scan(data, pair_indices, args.k_source, configs, int(args.nmu_b000), pk)
    b000_summary_rows = summarize_b000_scan(b000_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{args.prefix}.json"
    out_npz = output_dir / f"{args.prefix}.npz"
    out_pdf = output_dir / f"{args.prefix}.pdf"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {"json": str(out_json), "npz": str(out_npz), "pdf": str(out_pdf)},
        "cosmology": cosmo_params,
        "linear_power": pk.description,
        "power_normalization_audit": quijote_power_normalization_audit(args.z),
        "kernel_checks": run_kernel_checks(),
        "triangles": [[float(item) for item in triangle] for triangle in triangles],
        "configs": [asdict(config) for config in configs],
        "b000_pair_indices": [int(item) for item in pair_indices],
        "k_source": args.k_source,
        "nmu_b000": int(args.nmu_b000),
        "triangle_rows": triangle_rows,
        "b000_rows": b000_rows,
        "b000_summary_rows": b000_summary_rows,
        "interpretation_note": "This is a convergence diagnostic. Selected B000 runs are intentionally small and are not production-accuracy full 120-bin loops.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(out_npz, **rows_to_npz_dict("triangle", triangle_rows), **rows_to_npz_dict("b000", b000_rows))
    write_pdf(out_pdf, triangle_rows, b000_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
