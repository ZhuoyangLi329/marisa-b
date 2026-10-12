#!/usr/bin/env python
"""诊断 B000 1-loop 各 diagram 的 q 节点贡献。

代码大纲
========
1. 读取 Quijote DM pre-recon `B000(k1,k2)` 聚合矩阵，按 pair index 选取
   weighted-k 或 center-k 理论输入。
2. 对每个 pair/config 做 Sugiyama `B000` 外层角平均，同时保留每个
   log-q Gauss 节点对 `B222/B321I/B321II/B411` 的贡献。
3. `B222/B321I/B411` 直接复用生产代码的 `loop_integrand_components`；
   `B321II` 使用生产代码同口径的 ReACT `P13` 一维解析 integrand 逐 q
   节点展开。
4. 输出 JSON/NPZ/PDF，重点记录 component/tree、q-node cancellation、
   high-q signed contribution 和主导 q 节点。

本脚本是诊断工具，不用于生产完整 120 维 1-loop。它的目标是定位
中高 k convergence 问题来自哪个 q 区间，而不是提高理论精度。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from quijote_dm_spt1loop_realspace import (
    DEFAULT_DM_MATRIX,
    DEFAULT_OUTPUT_DIR,
    LoopConfig,
    TWOPI,
    bispectrum_tree,
    build_quijote_linear_power,
    load_quijote_dm_pre_mean,
    loop_config_label,
    loop_integrand_components,
    loop_nodes,
    parse_loop_config_string,
    q_vector,
    react_p13_shape,
    safe_power,
    spt_f,
    triangle_vectors,
    vec_norm,
)


COMPONENTS = ("B222", "B321I", "B321II", "B411", "B1loop")
LOOP_INTEGRAL_COMPONENTS = ("B222", "B321I", "B411")


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


def parse_config(value: str) -> LoopConfig:
    """解析 loop 配置；支持普通、q 分段和 phi 分段积分格式。"""

    return parse_loop_config_string(value)


def config_label(config: LoopConfig) -> str:
    return loop_config_label(config)


def theory_pair(data: dict[str, np.ndarray], pair_index: int, k_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.asarray([data["k1"][pair_index], data["k2"][pair_index]], dtype=np.float64)
    weighted = np.asarray(data["hdf5_k_weighted"][pair_index], dtype=np.float64)
    if k_source == "weighted":
        return nominal, weighted, weighted
    if k_source == "center":
        return nominal, weighted, nominal
    raise ValueError(f"Unknown k_source {k_source!r}")


def p13_q_contributions(k: float, pk, config: LoopConfig) -> np.ndarray:
    """返回 ReACT `P13_dd(k)` 在每个 log-q 节点上的贡献。"""

    q_nodes, wlog, _, _, _, _ = loop_nodes(config)
    prefactor = safe_power(pk, float(k)) * float(k) ** 2 / (252.0 * 4.0 * math.pi * math.pi)
    values = []
    for q, wq in zip(q_nodes, wlog):
        values.append(prefactor * float(wq) * float(q) * safe_power(pk, float(q)) * react_p13_shape(float(q) / float(k)))
    return np.asarray(values, dtype=np.float64)


def b321ii_q_contrib_for_triangle(vecs: tuple[np.ndarray, np.ndarray, np.ndarray], pk, config: LoopConfig) -> np.ndarray:
    """把 `B321II` 的 P13 一维积分也按 q 节点展开。"""

    k_ext = [vec_norm(vec) for vec in vecs]
    p_ext = [safe_power(pk, vec) for vec in vecs]
    p13_q = [p13_q_contributions(kval, pk, config) for kval in k_ext]
    total = np.zeros_like(p13_q[0])
    for i, j in ((0, 1), (1, 2), (2, 0)):
        total += spt_f((vecs[i], vecs[j])) * (p_ext[i] * p13_q[j] + p_ext[j] * p13_q[i])
    return total


def profile_b000_pair(data: dict[str, np.ndarray], pair_index: int, k_source: str, config: LoopConfig, nmu_b000: int, pk) -> dict[str, object]:
    """对一个 B000 pair 计算每个 q 节点的 diagram contribution。"""

    nominal, weighted, theory = theory_pair(data, int(pair_index), k_source)
    q_nodes, wlog, mu_nodes, wmu, phi_nodes, wphi = loop_nodes(config)
    outer_mu, outer_w = np.polynomial.legendre.leggauss(int(nmu_b000))
    q_contrib = {component: np.zeros_like(q_nodes, dtype=np.float64) for component in ("B222", "B321I", "B321II", "B411")}
    tree = 0.0
    measure_norm = 1.0 / (TWOPI**3)

    for mu12, w_outer in zip(outer_mu, outer_w):
        vecs = triangle_vectors(float(theory[0]), float(theory[1]), float(mu12))
        tree += 0.5 * float(w_outer) * bispectrum_tree(float(theory[0]), float(theory[1]), float(mu12), pk)
        q_contrib["B321II"] += 0.5 * float(w_outer) * b321ii_q_contrib_for_triangle(vecs, pk, config)

        for q_index, (q, wq) in enumerate(zip(q_nodes, wlog)):
            q3_weight = float(q) ** 3 * float(wq)
            for mu_q1, wu in zip(mu_nodes, wmu):
                for phi, wp in zip(phi_nodes, wphi):
                    qv = q_vector(float(q), float(mu_q1), float(phi))
                    measure = 0.5 * float(w_outer) * q3_weight * float(wu) * float(wp) * measure_norm
                    components = loop_integrand_components(vecs, qv, pk)
                    for component in LOOP_INTEGRAL_COMPONENTS:
                        q_contrib[component][q_index] += measure * components[component]

    q_contrib["B1loop"] = q_contrib["B222"] + q_contrib["B321I"] + q_contrib["B321II"] + q_contrib["B411"]
    component_totals = {component: float(np.sum(q_contrib[component])) for component in COMPONENTS}
    sim = float(data["mean"][pair_index])
    err = float(data["stderr"][pair_index])
    summary = summarize_q_profile(q_nodes, q_contrib, tree)
    return {
        "pair_index": int(pair_index),
        "pair_i": int(data["pair_i"][pair_index]),
        "pair_j": int(data["pair_j"][pair_index]),
        "k_source": k_source,
        "nominal_k_pair": nominal.tolist(),
        "weighted_k_pair": weighted.tolist(),
        "theory_k_pair": theory.tolist(),
        "k_eff": float(np.sqrt(theory[0] * theory[1])),
        "config_label": config_label(config),
        "loop_config": asdict(config),
        "nmu_b000": int(nmu_b000),
        "sim_B000": sim,
        "sim_B000_stderr": err,
        "B000_Btree": float(tree),
        "component_totals": component_totals,
        "component_over_Btree": {component: float(value / tree) if tree != 0.0 else float("nan") for component, value in component_totals.items()},
        "total_fractional_residual": float((tree + component_totals["B1loop"]) / sim - 1.0) if sim != 0.0 else float("nan"),
        "q_nodes": q_nodes.tolist(),
        "q_weights_log": wlog.tolist(),
        "q_contributions": {component: q_contrib[component].tolist() for component in COMPONENTS},
        "q_profile_summary": summary,
    }


def summarize_q_profile(q_nodes: np.ndarray, q_contrib: dict[str, np.ndarray], tree: float) -> dict[str, object]:
    """汇总每个 component 的 q 分布、抵消程度和 high-q 贡献。"""

    thresholds = (0.3, 1.0, 3.0, 10.0)
    summary: dict[str, object] = {}
    for component in COMPONENTS:
        values = np.asarray(q_contrib[component], dtype=np.float64)
        total = float(np.sum(values))
        abs_sum = float(np.sum(np.abs(values)))
        max_abs_index = int(np.argmax(np.abs(values))) if values.size else -1
        item = {
            "total": total,
            "over_Btree": float(total / tree) if tree != 0.0 else float("nan"),
            "abs_sum": abs_sum,
            "cancellation_ratio_abs_total_over_abs_sum": float(abs(total) / abs_sum) if abs_sum > 0.0 else float("nan"),
            "max_abs_q_node": float(q_nodes[max_abs_index]) if max_abs_index >= 0 else float("nan"),
            "max_abs_q_contribution": float(values[max_abs_index]) if max_abs_index >= 0 else float("nan"),
            "max_abs_q_contribution_over_Btree": float(values[max_abs_index] / tree) if tree != 0.0 and max_abs_index >= 0 else float("nan"),
        }
        for threshold in thresholds:
            mask = q_nodes >= float(threshold)
            signed = float(np.sum(values[mask]))
            abs_part = float(np.sum(np.abs(values[mask])))
            item[f"signed_q_ge_{threshold:g}_over_Btree"] = float(signed / tree) if tree != 0.0 else float("nan")
            item[f"abs_q_ge_{threshold:g}_fraction_of_abs_sum"] = float(abs_part / abs_sum) if abs_sum > 0.0 else float("nan")
        summary[component] = item
    return summary


def rows_to_npz(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """保存 ragged-free 的主要 q profile 数组到 NPZ。"""

    arrays: dict[str, np.ndarray] = {}
    for row_index, row in enumerate(rows):
        prefix = f"row{row_index}_pair{row['pair_index']}_{row['config_label']}"
        arrays[f"{prefix}_q_nodes"] = np.asarray(row["q_nodes"], dtype=np.float64)
        for component in COMPONENTS:
            arrays[f"{prefix}_{component}"] = np.asarray(row["q_contributions"][component], dtype=np.float64)
    return arrays


def plot_row(pdf: PdfPages, row: dict[str, object]) -> None:
    """每个 pair/config 输出一页：q-node contribution 与 cumulative。"""

    q = np.asarray(row["q_nodes"], dtype=np.float64)
    tree = float(row["B000_Btree"])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.7), sharex=True, constrained_layout=True)
    for component in COMPONENTS:
        values = np.asarray(row["q_contributions"][component], dtype=np.float64) / tree
        axes[0].plot(q, values, marker="o", ms=3.0, lw=1.0, label=component.replace("B", "B"))
        axes[1].plot(q, np.cumsum(values), marker="o", ms=3.0, lw=1.0, label=component.replace("B", "B"))
    axes[0].axhline(0.0, color="0.45", lw=0.7)
    axes[1].axhline(0.0, color="0.45", lw=0.7)
    axes[0].set_xscale("log")
    axes[0].set_ylabel(r"q-node contribution / $B_{\rm tree}$")
    axes[1].set_ylabel(r"cumulative / $B_{\rm tree}$")
    axes[1].set_xlabel(r"$q\,[h\,\mathrm{Mpc}^{-1}]$")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle(f"pair {row['pair_index']}  k={row['k_eff']:.3f}  {row['config_label']}")
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path: Path, rows: list[dict[str, object]]) -> None:
    configure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for row in rows:
            plot_row(pdf, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_DM_MATRIX))
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--pair-index", action="append", type=int, required=True, help="B000 pair index，可重复。")
    parser.add_argument("--config", action="append", type=parse_config, required=True, help="qmax,nq,nmu,nphi；也可用 qmax,nq,nmu,nphi,qsplit1:...,nqseg,phisplit1:...,nphiseg。")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--nmu-b000", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="quijote_dm_spt1loop_q_profile")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    args = parse_args()
    data = load_quijote_dm_pre_mean(Path(args.matrix))
    pk, cosmo_params = build_quijote_linear_power(float(args.z))
    rows = []
    for pair_index in args.pair_index:
        for config in args.config:
            rows.append(profile_b000_pair(data, int(pair_index), args.k_source, config, int(args.nmu_b000), pk))

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
        "pair_indices": [int(item) for item in args.pair_index],
        "configs": [asdict(config) for config in args.config],
        "k_source": args.k_source,
        "nmu_b000": int(args.nmu_b000),
        "rows": rows,
        "interpretation_note": "Inspect q_profile_summary for high-q dominance and cancellation. This diagnostic reuses the same integrands as production code.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(out_npz, **rows_to_npz(rows))
    write_pdf(out_pdf, rows)
    print(json.dumps({"json": str(out_json), "npz": str(out_npz), "pdf": str(out_pdf), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
