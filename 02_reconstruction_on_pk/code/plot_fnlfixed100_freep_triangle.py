#!/usr/bin/env python3
"""fNL 固定 100、p 自由校准的 (p, b1, Pshot) 三角图，每方法独立一页（只出 PDF）。

代码大纲
========
1. sha 链路校验后读取 pre 与 STO-only 各自的 (p, b1, Pshot) posterior。
2. 两个拟合是不同 tracer 的独立参数（b1/Pshot/p 互不相同），不做跨方法
   叠加；每方法单独一页三角图，各自独立 parameter_range。
3. 沿用原仓库 triangle 画法：68/95 双层 contours（fill=False）、对角
   draw_marginal、tick 字号 7、grid alpha .1；对角 p 格标注 legacy p=1.12。
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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "02_reconstruction_on_pk"
WORK = PART / "work/quijote_halo_z0p5_pre_sto_only_fnlfixed100_freep_calib_v1"
RESULT_JSON = WORK / "FREEP_RESULTS.json"
POSTERIORS = WORK / "FREEP_POSTERIORS.npz"
OUTPUT = PART / "figures/fnlfixed100_freep_triangles.pdf"

METHODS = ("pre", "standard", "marisa", "sto_only", "sto_marisa")
METHOD_TITLES = {"pre": "pre halo catalog", "standard": "standard-recon catalog", "marisa": "Marisa-recon catalog", "sto_only": "STO-only weighted catalog", "sto_marisa": "STO+Marisa recon catalog"}
PARAM_LABELS = (r"$p$", r"$b_1$", r"$P_{\rm shot,res}$")
METHOD_COLOR = "#4E79A7"
BINS = 58
SMOOTHING = 1.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_posteriors() -> dict[str, np.ndarray]:
    result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError("freep results status not complete")
    if sha256(POSTERIORS) != result["outputs"]["posteriors"]["sha256"]:
        raise RuntimeError("freep posterior hash drift")
    with np.load(POSTERIORS, allow_pickle=False) as stream:
        return {m: np.asarray(stream[f"{m}__posterior"], dtype=float) for m in METHODS}


def parameter_range(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.002, 0.998))
    width = upper - lower
    return float(lower - padding * width), float(upper + padding * width)


def draw_marginal(axis, values, value_range, color):
    histogram, edges = np.histogram(values, bins=70, range=value_range, density=True)
    density = gaussian_filter1d(histogram.astype(float), 1.0, mode="nearest")
    centres = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(centres, density, color=color, lw=1.5)
    axis.fill_between(centres, density, color=color, alpha=.12)
    axis.set_xlim(*value_range)


def draw_contours(axis, x, y, x_range, y_range, color):
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=BINS, range=(x_range, y_range))
    smooth = gaussian_filter(histogram.astype(float), SMOOTHING, mode="nearest")
    total = float(np.sum(smooth))
    if not total > 0:
        raise ValueError("二维 posterior histogram 为空")
    density = smooth / total
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)

    def threshold(probability):
        position = min(int(np.searchsorted(cumulative, probability)), ordered.size - 1)
        return float(ordered[position])

    levels = (threshold(0.95), threshold(0.68))
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    axis.contour(
        mesh_x, mesh_y, density, levels=levels,
        colors=(color, color), linestyles=("--", "-"), linewidths=(1.0, 1.5),
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)


def draw_triangle(pdf, sample: np.ndarray, method: str) -> None:
    ranges = [parameter_range(sample[:, i]) for i in range(3)]
    figure, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
    for row in range(3):
        for column in range(3):
            axis = axes[row, column]
            if column > row:
                axis.axis("off")
                continue
            if row == column:
                draw_marginal(axis, sample[:, column], ranges[column], METHOD_COLOR)
            else:
                draw_contours(
                    axis, sample[:, column], sample[:, row],
                    ranges[column], ranges[row], METHOD_COLOR,
                )
            if row == 0 and column == 0:
                axis.axvline(1.12, color="k", ls=":", lw=1.3)
            if row == 2:
                axis.set_xlabel(PARAM_LABELS[column])
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(PARAM_LABELS[row])
            elif column > 0:
                axis.set_yticklabels([])
            axis.tick_params(labelsize=7)
            axis.grid(alpha=.1)
    figure.suptitle(
        rf"{METHOD_TITLES[method]}: free-$(p,b_1,P_{{\rm shot}})$ calibration, input fNL$=+100$ fixed",
        y=.995,
    )
    figure.tight_layout(rect=(0, 0, 1, .97))
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    samples = verified_posteriors()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        for method in METHODS:
            draw_triangle(pdf, samples[method], method)
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
