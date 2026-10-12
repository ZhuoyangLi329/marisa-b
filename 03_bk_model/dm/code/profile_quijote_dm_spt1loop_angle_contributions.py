#!/usr/bin/env python
"""诊断 B000 1-loop 三维积分的 loop-angle 节点贡献。

代码大纲
========
1. 读取 Quijote DM pre-recon `B000(k1,k2)` 聚合矩阵，按 pair index 选择
   weighted-k 或 center-k 理论输入。
2. 对给定 segmented q 配置和 loop angular Gauss 节点，逐个 q node 汇总
   `B222/B321I` 在 `(mu_q1, phi)` cell 上的贡献。
3. 每个 q node 记录：
   - signed contribution / Btree
   - angle absolute-sum / Btree
   - cancellation ratio
   - 最大贡献 cell 的 `(mu_q1, phi)` 和数值
4. 输出 JSON/NPZ/PDF。该脚本默认跳过 `B411`，避免为角向诊断反复调用
   F4；生产积分和 ReACT 对齐审计仍默认计算完整 `B411`。

本脚本不替代 convergence scan。它只用来定位为什么某些 angular Gauss
阶数会偶然采到局部结构，特别是 pair 84 的 `5x5` 偏高问题。
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
from matplotlib.backends.backend_pdf import PdfPages

from quijote_dm_spt1loop_realspace import (
    DEFAULT_DM_MATRIX,
    DEFAULT_OUTPUT_DIR,
    TWOPI,
    bispectrum_tree,
    build_quijote_linear_power,
    load_quijote_dm_pre_mean,
    loop_config_label,
    loop_integrand_components,
    loop_nodes,
    parse_loop_config_string,
    q_vector,
    triangle_vectors,
)


COMPONENTS = ("B222", "B321I")


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


def theory_pair(data: dict[str, np.ndarray], pair_index: int, k_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.asarray([data["k1"][pair_index], data["k2"][pair_index]], dtype=np.float64)
    weighted = np.asarray(data["hdf5_k_weighted"][pair_index], dtype=np.float64)
    if k_source == "weighted":
        return nominal, weighted, weighted
    if k_source == "center":
        return nominal, weighted, nominal
    raise ValueError(f"Unknown k_source {k_source!r}")


def profile_angles_for_pair(data: dict[str, np.ndarray], pair_index: int, k_source: str, config, nmu_b000: int, pk) -> dict[str, object]:
    """对一个 B000 pair 生成 q-by-angle contribution profile。"""

    nominal, weighted, theory = theory_pair(data, int(pair_index), k_source)
    q_nodes, wlog, mu_nodes, wmu, phi_nodes, wphi = loop_nodes(config)
    outer_mu, outer_w = np.polynomial.legendre.leggauss(int(nmu_b000))
    measure_norm = 1.0 / (TWOPI**3)

    tree = 0.0
    angle_contrib = {component: np.zeros((len(q_nodes), len(mu_nodes), len(phi_nodes)), dtype=np.float64) for component in COMPONENTS}

    for mu12, w_outer in zip(outer_mu, outer_w):
        vecs = triangle_vectors(float(theory[0]), float(theory[1]), float(mu12))
        tree += 0.5 * float(w_outer) * bispectrum_tree(float(theory[0]), float(theory[1]), float(mu12), pk)
        for q_index, (q, wq) in enumerate(zip(q_nodes, wlog)):
            q3_weight = float(q) ** 3 * float(wq)
            for mu_index, (mu_q1, wu) in enumerate(zip(mu_nodes, wmu)):
                for phi_index, (phi, wp) in enumerate(zip(phi_nodes, wphi)):
                    qv = q_vector(float(q), float(mu_q1), float(phi))
                    measure = 0.5 * float(w_outer) * q3_weight * float(wu) * float(wp) * measure_norm
                    components = loop_integrand_components(vecs, qv, pk, include_b411=False)
                    for component in COMPONENTS:
                        angle_contrib[component][q_index, mu_index, phi_index] += measure * components[component]

    summary = summarize_angle_profile(q_nodes, mu_nodes, phi_nodes, angle_contrib, tree)
    return {
        "pair_index": int(pair_index),
        "pair_i": int(data["pair_i"][pair_index]),
        "pair_j": int(data["pair_j"][pair_index]),
        "k_source": k_source,
        "nominal_k_pair": nominal.tolist(),
        "weighted_k_pair": weighted.tolist(),
        "theory_k_pair": theory.tolist(),
        "k_eff": float(np.sqrt(theory[0] * theory[1])),
        "config_label": loop_config_label(config),
        "loop_config": asdict(config),
        "nmu_b000": int(nmu_b000),
        "B000_Btree": float(tree),
        "q_nodes": q_nodes.tolist(),
        "mu_nodes": mu_nodes.tolist(),
        "phi_nodes": phi_nodes.tolist(),
        "angle_profile_summary": summary,
        "angle_contributions": {component: angle_contrib[component].tolist() for component in COMPONENTS},
        "component_totals_over_Btree": {
            component: float(np.sum(angle_contrib[component]) / tree) if tree != 0.0 else float("nan") for component in COMPONENTS
        },
    }


def summarize_angle_profile(
    q_nodes: np.ndarray,
    mu_nodes: np.ndarray,
    phi_nodes: np.ndarray,
    angle_contrib: dict[str, np.ndarray],
    tree: float,
) -> dict[str, object]:
    """把 full angle profile 压缩成 q-node 和 global summary。"""

    result: dict[str, object] = {}
    for component, cube in angle_contrib.items():
        q_rows = []
        global_abs = np.abs(cube)
        flat_index = int(np.argmax(global_abs)) if global_abs.size else 0
        q_max, mu_max, phi_max = np.unravel_index(flat_index, cube.shape)
        for q_index, q_value in enumerate(q_nodes):
            cells = cube[q_index]
            signed = float(np.sum(cells))
            abs_sum = float(np.sum(np.abs(cells)))
            max_index = int(np.argmax(np.abs(cells))) if cells.size else 0
            mu_index, phi_index = np.unravel_index(max_index, cells.shape)
            q_rows.append(
                {
                    "q": float(q_value),
                    "signed_over_Btree": float(signed / tree) if tree != 0.0 else float("nan"),
                    "abs_sum_over_Btree": float(abs_sum / tree) if tree != 0.0 else float("nan"),
                    "cancellation_ratio_abs_signed_over_abs_sum": float(abs(signed) / abs_sum) if abs_sum > 0.0 else float("nan"),
                    "max_abs_cell_over_Btree": float(cells[mu_index, phi_index] / tree) if tree != 0.0 else float("nan"),
                    "max_abs_cell_mu": float(mu_nodes[mu_index]),
                    "max_abs_cell_phi": float(phi_nodes[phi_index]),
                }
            )
        result[component] = {
            "total_over_Btree": float(np.sum(cube) / tree) if tree != 0.0 else float("nan"),
            "angle_abs_sum_over_Btree": float(np.sum(global_abs) / tree) if tree != 0.0 else float("nan"),
            "global_max_abs_cell": {
                "q": float(q_nodes[q_max]),
                "mu": float(mu_nodes[mu_max]),
                "phi": float(phi_nodes[phi_max]),
                "value_over_Btree": float(cube[q_max, mu_max, phi_max] / tree) if tree != 0.0 else float("nan"),
            },
            "q_rows": q_rows,
        }
    return result


def rows_to_npz(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for row_index, row in enumerate(rows):
        prefix = f"row{row_index}_pair{row['pair_index']}_{row['config_label']}"
        arrays[f"{prefix}_q_nodes"] = np.asarray(row["q_nodes"], dtype=np.float64)
        arrays[f"{prefix}_mu_nodes"] = np.asarray(row["mu_nodes"], dtype=np.float64)
        arrays[f"{prefix}_phi_nodes"] = np.asarray(row["phi_nodes"], dtype=np.float64)
        for component in COMPONENTS:
            arrays[f"{prefix}_{component}_angle_contrib"] = np.asarray(row["angle_contributions"][component], dtype=np.float64)
    return arrays


def plot_row(pdf: PdfPages, row: dict[str, object]) -> None:
    q = np.asarray(row["q_nodes"], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True, constrained_layout=True)
    for component in COMPONENTS:
        q_rows = row["angle_profile_summary"][component]["q_rows"]
        signed = np.asarray([item["signed_over_Btree"] for item in q_rows], dtype=np.float64)
        max_cell = np.asarray([item["max_abs_cell_over_Btree"] for item in q_rows], dtype=np.float64)
        abs_sum = np.asarray([item["abs_sum_over_Btree"] for item in q_rows], dtype=np.float64)
        axes[0].plot(q, signed, marker="o", ms=3.0, lw=1.0, label=f"{component} signed")
        axes[0].plot(q, max_cell, marker="x", ms=3.0, lw=0.9, ls="--", label=f"{component} max cell")
        axes[1].plot(q, abs_sum, marker="o", ms=3.0, lw=1.0, label=f"{component} abs sum")
    for ax in axes:
        ax.axhline(0.0, color="0.45", lw=0.7)
        ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, ncol=2)
    axes[0].set_ylabel(r"signed or max cell / $B_{\rm tree}$")
    axes[1].set_ylabel(r"angle abs sum / $B_{\rm tree}$")
    axes[1].set_xlabel(r"$q\,[h\,\mathrm{Mpc}^{-1}]$")
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
    parser.add_argument("--pair-index", action="append", type=int, required=True)
    parser.add_argument("--config", action="append", type=parse_loop_config_string, required=True, help="qmax,nq,nmu,nphi；也可用 q/phi 分段格式，如 ...,qsplit,nqseg,pi,nphiseg。")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--nmu-b000", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="quijote_dm_spt1loop_angle_profile")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    args = parse_args()
    data = load_quijote_dm_pre_mean(Path(args.matrix))
    pk, cosmo_params = build_quijote_linear_power(float(args.z))
    rows = []
    for pair_index in args.pair_index:
        for config in args.config:
            rows.append(profile_angles_for_pair(data, int(pair_index), args.k_source, config, int(args.nmu_b000), pk))

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
        "interpretation_note": "Angle profile skips B411 by default and diagnoses B222/B321I angular-node sensitivity.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(out_npz, **rows_to_npz(rows))
    write_pdf(out_pdf, rows)
    print(json.dumps({"json": str(out_json), "npz": str(out_npz), "pdf": str(out_pdf), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
