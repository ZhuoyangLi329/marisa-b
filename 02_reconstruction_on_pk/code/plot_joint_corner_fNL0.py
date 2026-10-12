#!/usr/bin/env python3
"""fNL=0 的全参数 triangle 图：joint 5 参数 vs pre vs STO-only（只出 PDF）。

代码大纲
========
1. sha 链路校验后读取三方 posterior（joint 为 5 参数链；pre/STO-only 为
   冻结 3 参数链）。
2. 沿用原仓库 plot_*_ponly_mcmc_contours 的画法：所有方法共享统一的
   parameter_range（全体样本 0.2/99.8 分位数 + padding）、68/95 双层
   contours（fill=False）、对角 draw_marginal、tick 字号 7、grid alpha .1、
   figure 级 legend。
3. 参数序为 joint 的 `(fNL, b1_pre, sn0_pre, b1_post, sn0_post)`；pre 映射
   到 pre 通道参数，STO-only 映射到 post 通道参数；方法没有的参数格留空。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "02_reconstruction_on_pk"
JOINT_WORK = PART / "work/quijote_halo_z0p5_pre_post_joint_p0only_fid0_p100matched100_mcmc_v1"
JOINT_RESULTS = PART / "results/pre_post_joint_p0only_mcmc_results.json"
JOINT_POSTERIORS = JOINT_WORK / "P0ONLY_MCMC_POSTERIORS.npz"
FIVE_RESULTS = PART / "results/five_method_p0only_mcmc_results.json"
FIVE_POSTERIORS = PART / "work/quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_POSTERIORS.npz"
OUTPUT = PART / "figures/joint_vs_pre_vs_sto_only_triangle_fNL0.pdf"

TARGET = "fNL0"  # 默认 target；可由 --target 覆盖。
PARAM_LABELS = (
    r"$f_{\rm NL}$", r"$b_{1,\rm pre}$", r"$b_{1,\rm post}$",
    r"$P_{\rm shot,res}^{\rm pre}$", r"$P_{\rm shot,res}^{\rm post}$",
)
# 每方法样本矩阵的列在 corner 参数序中的位置；没有的参数不出现。
# joint posterior 列序 (fNL, b1_pre, b1_post, sn0_pre, sn0_post) 与
# corner 参数序一致；pre 映射到 pre 通道的 (b1, sn0)，
# STO-only 映射到 post 通道的 (b1, sn0)。
METHOD_COLUMNS = {
    "pre": (0, 1, 3),
    "sto_only": (0, 2, 4),
    "joint": (0, 1, 2, 3, 4),
}
METHOD_SAMPLES = {
    "pre": ("five", "pre__{target}__posterior"),
    "sto_only": ("five", "sto_only__{target}__posterior"),
    "joint": ("joint", "pre_sto_only__{target}__posterior"),
}
COLORS = {"pre": "#4E79A7", "sto_only": "#B07AA1", "joint": "#E15759"}
LABELS = {"pre": "pre", "sto_only": "STO only", "joint": "pre+STO only (joint)"}
BINS = 58
SMOOTHING = 1.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def verified_posteriors() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """sha 链路校验后返回 (five-method, joint) posterior NPZ。"""

    joint_result = json.loads(JOINT_RESULTS.read_text(encoding="utf-8"))
    if joint_result.get("status") != "complete":
        raise RuntimeError("joint results status not complete")
    if sha256(JOINT_POSTERIORS) != joint_result["outputs"]["posteriors"]["sha256"]:
        raise RuntimeError("joint posterior hash drift")
    five_result = json.loads(FIVE_RESULTS.read_text(encoding="utf-8"))
    if five_result.get("status") != "complete":
        raise RuntimeError("five-method results status not complete")
    if sha256(FIVE_POSTERIORS) != five_result["outputs"]["posteriors"]["sha256"]:
        raise RuntimeError("five-method posterior hash drift")
    return load_npz(FIVE_POSTERIORS), load_npz(JOINT_POSTERIORS)


def parameter_range(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    """用稳健分位数定义绘图区间（原仓库同款）。"""

    lower, upper = np.quantile(values, (0.002, 0.998))
    width = upper - lower
    return float(lower - padding * width), float(upper + padding * width)


def draw_marginal(axis, values: np.ndarray, value_range: tuple[float, float], color: str) -> None:
    """绘制平滑一维 marginal（原仓库同款，bins=70 平滑 1.0）。"""

    histogram, edges = np.histogram(values, bins=70, range=value_range, density=True)
    density = gaussian_filter1d(histogram.astype(float), 1.0, mode="nearest")
    centres = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(centres, density, color=color, lw=1.5)
    axis.fill_between(centres, density, color=color, alpha=.12)
    axis.set_xlim(*value_range)


def draw_contours(
    axis, x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
    color: str,
) -> None:
    """在 axis 上绘制95%虚线和68%实线 contours（原仓库同款）。"""

    histogram, x_edges, y_edges = np.histogram2d(
        x, y, bins=BINS, range=(x_range, y_range)
    )
    smooth = gaussian_filter(histogram.astype(float), SMOOTHING, mode="nearest")
    total = float(np.sum(smooth))
    if not total > 0:
        raise ValueError("二维 posterior histogram 为空")
    density = smooth / total
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)

    def threshold(probability: float) -> float:
        position = min(int(np.searchsorted(cumulative, probability)), ordered.size - 1)
        return float(ordered[position])

    threshold68, threshold95 = threshold(0.68), threshold(0.95)
    if not 0 < threshold95 < threshold68:
        raise ValueError("HPD 阈值退化")
    levels = (threshold95, threshold68)
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    axis.contour(
        mesh_x, mesh_y, density, levels=levels,
        colors=(color, color), linestyles=("--", "-"), linewidths=(1.0, 1.5),
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)


def main() -> None:
    """生成指定 target 的三方全参数 triangle PDF。"""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("fNL0", "fNL100"), default=TARGET)
    args = parser.parse_args()
    target = args.target
    output = OUTPUT.with_name(OUTPUT.name.replace("fNL0", target))

    five, joint = verified_posteriors()
    stores = {"five": five, "joint": joint}
    # 每方法 -> {corner 参数序位置: 样本列}。
    columns: dict[str, dict[int, np.ndarray]] = {}
    for method, (store, npz_key_template) in METHOD_SAMPLES.items():
        posterior = np.asarray(stores[store][npz_key_template.format(target=target)], dtype=float)
        columns[method] = {
            corner_index: posterior[:, source_index]
            for source_index, corner_index in enumerate(METHOD_COLUMNS[method])
        }

    dimension = len(PARAM_LABELS)
    # 统一 parameter ranges：每格用所有可用方法样本的并集分位数。
    ranges: list[tuple[float, float]] = []
    for corner_index in range(dimension):
        pooled = np.concatenate([
            columns[method][corner_index]
            for method in METHOD_COLUMNS if corner_index in columns[method]
        ])
        ranges.append(parameter_range(pooled))

    figure, axes = plt.subplots(dimension, dimension, figsize=(11.5, 11.5))
    for row in range(dimension):
        for column in range(dimension):
            axis = axes[row, column]
            if column > row:
                axis.axis("off")
                continue
            available = [
                method for method in METHOD_COLUMNS
                if column in columns[method] and row in columns[method]
            ]
            for method in available:
                if row == column:
                    draw_marginal(axis, columns[method][column], ranges[column], COLORS[method])
                else:
                    draw_contours(
                        axis, columns[method][column], columns[method][row],
                        ranges[column], ranges[row], COLORS[method],
                    )
            if row == dimension - 1:
                axis.set_xlabel(PARAM_LABELS[column])
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(PARAM_LABELS[row])
            elif column > 0:
                axis.set_yticklabels([])
            axis.tick_params(labelsize=7)
            axis.grid(alpha=.1)
    handles = [
        Line2D([0], [0], color=COLORS[method], lw=1.8, label=LABELS[method])
        for method in ("pre", "sto_only", "joint")
    ]
    figure.legend(handles=handles, loc="upper right", frameon=False, ncol=1)
    input_label = "0" if target == "fNL0" else "+100"
    figure.suptitle(
        rf"input fNL$={input_label}$: joint pre+STO-only (5-param) vs pre vs STO-only posteriors",
        y=.995,
    )
    figure.tight_layout(rect=(0, 0, .98, .98))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"status": "complete", "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
