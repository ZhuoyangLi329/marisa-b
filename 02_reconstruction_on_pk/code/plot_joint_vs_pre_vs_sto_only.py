#!/usr/bin/env python3
"""绘制 pre+STO-only joint vs pre vs STO-only 的 fNL-b1 后验 contour（只出 PDF）。

代码大纲
========
1. 校验联合结果与冻结五方法结果的 sha256 链路，读取三方 posterior：
   pre 与 STO-only 取冻结 v1/五方法 NPZ；joint 取联合 NPZ（5 参数）。
2. 单页 1x2（fNL0 / fNL100），在 (fNL, b1) 平面画 68%/95% contours：
   pre（蓝）、STO only（紫）、joint 的 b1_pre 与 b1_post 两条（紫红虚/实）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "02_reconstruction_on_pk"
JOINT_TAG = "quijote_halo_z0p5_pre_post_joint_p0only_fid0_p100matched100_mcmc_v1"
JOINT_WORK = PART / "work" / JOINT_TAG
JOINT_RESULTS = PART / "results/pre_post_joint_p0only_mcmc_results.json"
JOINT_POSTERIORS = JOINT_WORK / "P0ONLY_MCMC_POSTERIORS.npz"
FIVE_RESULTS = PART / "results/five_method_p0only_mcmc_results.json"
FIVE_POSTERIORS = PART / "work/quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_POSTERIORS.npz"
OUTPUT = PART / "figures/joint_vs_pre_vs_sto_only_contours.pdf"

TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
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


def draw_contours(
    axis: Any, x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
    color: str, linestyle: tuple[str, str], linewidth: tuple[float, float],
) -> None:
    """在 axis 上绘制平滑二维密度的 68%/95% contours。"""

    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=BINS, range=(x_range, y_range))
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

    levels = (threshold(0.95), threshold(0.68))
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    axis.contour(
        mesh_x, mesh_y, density, levels=levels,
        colors=(color, color), linestyles=linestyle, linewidths=linewidth,
    )


def parameter_range(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.002, 0.998))
    width = upper - lower
    return float(lower - padding * width), float(upper + padding * width)


def main() -> None:
    """生成三方 (fNL, b1) contour 对比 PDF。"""

    five, joint = verified_posteriors()
    curves = {
        "pre": (("#4E79A7", (":", "-"), (1.0, 1.6)), "pre"),
        "sto_only": (("#B07AA1", (":", "-"), (1.0, 1.6)), "STO only"),
        "joint_b1_pre": (("#E15759", ("--", "-"), (1.2, 1.8)), "pre+STO only (b1 pre)"),
        "joint_b1_post": (("#E15759", ("-.", "-"), (1.2, 1.8)), "pre+STO only (b1 post)"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for axis, target_name in zip(axes, TARGETS):
        samples = {
            "pre": np.asarray(five[f"pre__{target_name}__posterior"], dtype=float),
            "sto_only": np.asarray(five[f"sto_only__{target_name}__posterior"], dtype=float),
            "joint_b1_pre": np.asarray(joint[f"pre_sto_only__{target_name}__posterior"], dtype=float),
            "joint_b1_post": np.asarray(joint[f"pre_sto_only__{target_name}__posterior"], dtype=float),
        }
        all_f = np.concatenate([samples[name][:, 0] for name in samples])
        all_b = np.concatenate([
            samples["pre"][:, 1], samples["sto_only"][:, 1],
            samples["joint_b1_pre"][:, 1], samples["joint_b1_post"][:, 2],
        ])
        xlimits = parameter_range(all_f)
        ylimits = parameter_range(all_b)
        draw_contours(axis, samples["pre"][:, 0], samples["pre"][:, 1], xlimits, ylimits, *curves["pre"][0])
        draw_contours(axis, samples["sto_only"][:, 0], samples["sto_only"][:, 1], xlimits, ylimits, *curves["sto_only"][0])
        draw_contours(axis, samples["joint_b1_pre"][:, 0], samples["joint_b1_pre"][:, 1], xlimits, ylimits, *curves["joint_b1_pre"][0])
        draw_contours(axis, samples["joint_b1_post"][:, 0], samples["joint_b1_post"][:, 2], xlimits, ylimits, *curves["joint_b1_post"][0])
        axis.axvline(TRUTHS[target_name], color="k", ls=":", lw=1.3)
        axis.set_xlim(*xlimits)
        axis.set_ylim(*ylimits)
        axis.set_xlabel(r"$f_{\rm NL}$")
        axis.set_ylabel(r"$b_1$")
        axis.set_title(f"input fNL={TRUTHS[target_name]:.0f}")
        axis.grid(alpha=0.15)
        handles = [
            Line2D([0], [0], color=curves[name][0][0],
                   ls=curves[name][0][1][1], lw=1.8, label=curves[name][1])
            for name in ("pre", "sto_only", "joint_b1_pre", "joint_b1_post")
        ]
        handles.append(Line2D([0], [0], color="k", ls=":", lw=1.3, label="input truth"))
        axis.legend(handles=handles, frameon=False, fontsize=8.5)
    figure.suptitle(r"Joint pre+STO-only (MCMC) vs pre vs STO-only: $f_{\rm NL}$--$b_1$ posteriors, $k_{\max}=0.08$")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
