#!/usr/bin/env python3
"""绘制 pre 与 STO-only 的 P0-only 对比图（只出 PDF）。

代码大纲
========
1. `pre_vs_sto_only_p0only_contours.pdf`：fNL-b1 后验 68%/95% contour，
   只比较 pre 与 STO only，两个 target 各一 panel；posterior 取自五方法
   结果 NPZ（pre 为冻结 v1 继承、sto_only 为新链，sha256 链路校验）。
2. `pre_vs_sto_only_measured_p0.pdf`：两方法的测量平均 P0(k) 曲线
   （matched100 mean，与 MCMC target 同一数据），配 sto_only/pre ratio
   子图；误差带为各方法自身 Fid500 单盒 covariance 对角元的 mean sigma。
"""

from __future__ import annotations

from datetime import datetime, timezone
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
TAG = "quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1"
WORK = PART / "work" / TAG
MATRIX = PART / "inputs/five_method_p0_matrix.npz"
RESULT_JSON = PART / "results/five_method_p0only_mcmc_results.json"
POSTERIORS = WORK / "P0ONLY_MCMC_POSTERIORS.npz"
CONTOUR_PDF = PART / "figures/pre_vs_sto_only_p0only_contours.pdf"
P0_CURVE_PDF = PART / "figures/pre_vs_sto_only_measured_p0.pdf"

METHODS_PLOT = ("pre", "sto_only")
TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
TARGET_TITLES = {"fNL0": "input fNL=0 (Fiducial matched100)", "fNL100": "input fNL=+100 (LC_p matched100)"}
METHOD_LABELS = {"pre": "pre", "sto_only": "STO only"}
COLORS = {"pre": "#4E79A7", "sto_only": "#B07AA1"}
BINS = 58
SMOOTHING = 1.05


def sha256(path: Path) -> str:
    """返回文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 NPZ。"""

    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def parameter_range(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    """用稳健分位数定义绘图区间。"""

    lower, upper = np.quantile(values, (0.002, 0.998))
    width = upper - lower
    return float(lower - padding * width), float(upper + padding * width)


def density_2d(
    x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], dict[str, float]]:
    """计算平滑二维密度及68%/95% HPD 阈值（与五方法 runner 相同的数学）。"""

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
    level68, level95 = threshold68, threshold95
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    actual68 = float(np.sum(density[density >= level68]))
    actual95 = float(np.sum(density[density >= level95]))
    return (
        mesh_x, mesh_y, density, (level95, level68),
        {"coverage_68": actual68, "coverage_95": actual95},
    )


def draw_contours(
    axis: Any, x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
    color: str,
) -> dict[str, float]:
    """在 axis 上绘制95%虚线和68%实线 contours。"""

    xc, yc, density, levels, coverage = density_2d(x, y, x_range, y_range)
    axis.contourf(
        xc, yc, density, levels=(levels[0], levels[1], float(np.max(density)) * 1.001),
        colors=(color, color), alpha=.15,
    )
    axis.contour(
        xc, yc, density, levels=levels,
        colors=(color, color), linestyles=("--", "-"), linewidths=(1.2, 2.0),
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)
    return coverage


def verified_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """校验结果/矩阵 sha 链路后返回 (matrix, posteriors, results)。"""

    result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError("five-method results status not complete")
    if sha256(MATRIX) != result["inputs"]["matrix"]["sha256"]:
        raise RuntimeError("five-method matrix hash drift")
    if sha256(POSTERIORS) != result["outputs"]["posteriors"]["sha256"]:
        raise RuntimeError("five-method posterior hash drift")
    return load_npz(MATRIX), load_npz(POSTERIORS), result


def write_contour_figure(posteriors: dict[str, np.ndarray]) -> dict[str, Any]:
    """生成 pre vs STO-only 的 fNL-b1 后验 contour PDF（单页 1x2）。"""

    contour_checks: dict[str, Any] = {}
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for axis, target_name in zip(axes, TARGETS):
        samples = {
            method: np.asarray(posteriors[f"{method}__{target_name}__posterior"], dtype=float)
            for method in METHODS_PLOT
        }
        all_f = np.concatenate([sample[:, 0] for sample in samples.values()])
        all_b = np.concatenate([sample[:, 1] for sample in samples.values()])
        xlimits = parameter_range(all_f)
        ylimits = parameter_range(all_b)
        for method in METHODS_PLOT:
            contour_checks[f"{method}__{target_name}"] = draw_contours(
                axis, samples[method][:, 0], samples[method][:, 1], xlimits, ylimits,
                COLORS[method],
            )
        axis.axvline(TRUTHS[target_name], color="k", ls=":", lw=1.3)
        axis.set_xlabel(r"$f_{\rm NL}$")
        axis.set_ylabel(r"$b_1$")
        axis.set_title(TARGET_TITLES[target_name])
        axis.grid(alpha=0.15)
        handles = [
            Line2D([0], [0], color=COLORS[method], lw=1.8, label=METHOD_LABELS[method])
            for method in METHODS_PLOT
        ]
        handles.append(Line2D([0], [0], color="k", ls=":", lw=1.3, label="input truth"))
        axis.legend(handles=handles, frameon=False, fontsize=9)
    figure.suptitle(r"pre vs STO-only $P_0$-only MCMC, $k_{\max}=0.08$")
    figure.tight_layout()
    with PdfPages(CONTOUR_PDF) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    return contour_checks


def write_p0_curve_figure(matrix: dict[str, np.ndarray]) -> dict[str, Any]:
    """生成测量平均 P0 曲线与 ratio 的 PDF（2 行 x 2 列，单页）。"""

    k = np.asarray(matrix["pk_k"], dtype=float)
    figure, axes = plt.subplots(
        2, 2, figsize=(11.6, 7.6), sharex="col",
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )
    summary: dict[str, Any] = {"pk_k": k.tolist()}
    # 先算好两个 target 的比值，取共同的 ratio 显示范围（参考线 1.0 恒在界内）。
    ratios_by_target: dict[str, np.ndarray] = {}
    ratio_errors_by_target: dict[str, np.ndarray] = {}
    for target_name in TARGETS:
        curves: dict[str, np.ndarray] = {}
        sigmas: dict[str, np.ndarray] = {}
        for method in METHODS_PLOT:
            mean = np.asarray(matrix[f"{method}__target_{target_name}_matched100"], dtype=float)
            covariance = np.asarray(matrix[f"{method}__covariance_fNL0_fid500_one_box"], dtype=float)
            curves[method] = mean
            # matched100 mean 的误差 = 单盒 sigma / sqrt(100)。
            sigmas[method] = np.sqrt(np.diag(covariance)) / math.sqrt(100.0)
            summary[f"{method}__{target_name}_mean"] = mean.tolist()
        ratio = curves["sto_only"] / curves["pre"]
        relative = np.sqrt(
            (sigmas["sto_only"] / curves["sto_only"]) ** 2
            + (sigmas["pre"] / curves["pre"]) ** 2
        )
        ratios_by_target[target_name] = ratio
        ratio_errors_by_target[target_name] = ratio * relative
        summary[f"ratio__{target_name}"] = ratio.tolist()
    all_ratios = np.concatenate(list(ratios_by_target.values()))
    all_errors = np.concatenate(list(ratio_errors_by_target.values()))
    ratio_limits = (
        min(0.995, float(np.min(all_ratios - all_errors)) * 0.995),
        max(1.005, float(np.max(all_ratios + all_errors)) * 1.005),
    )
    for column, target_name in enumerate(TARGETS):
        curves = {
            method: np.asarray(matrix[f"{method}__target_{target_name}_matched100"], dtype=float)
            for method in METHODS_PLOT
        }
        sigmas = {
            method: np.sqrt(np.diag(np.asarray(
                matrix[f"{method}__covariance_fNL0_fid500_one_box"], dtype=float
            ))) / math.sqrt(100.0)
            for method in METHODS_PLOT
        }
        top = axes[0, column]
        for method in METHODS_PLOT:
            top.errorbar(
                k, curves[method], yerr=sigmas[method],
                fmt="o-", ms=4, lw=1.5, capsize=2.5,
                color=COLORS[method], label=METHOD_LABELS[method],
            )
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_ylabel(r"$\bar P_0(k)$ $[(\mathrm{Mpc}/h)^3]$")
        top.set_title(TARGET_TITLES[target_name])
        top.grid(alpha=0.15, which="both")
        top.legend(frameon=False, fontsize=9)

        bottom = axes[1, column]
        bottom.errorbar(
            k, ratios_by_target[target_name], yerr=ratio_errors_by_target[target_name],
            fmt="o-", ms=4, lw=1.5, capsize=2.5, color=COLORS["sto_only"],
        )
        bottom.axhline(1.0, color=COLORS["pre"], ls=":", lw=1.3)
        bottom.set_xscale("log")
        bottom.set_xlabel(r"$k$ $[h/\mathrm{Mpc}]$")
        bottom.set_ylabel("STO only / pre")
        bottom.set_ylim(*ratio_limits)
        bottom.grid(alpha=0.15, which="both")
    figure.suptitle(r"Measured mean $P_0(k)$: pre vs STO-only (matched100, error bars: single-box $\sigma/\sqrt{100}$)")
    figure.tight_layout()
    with PdfPages(P0_CURVE_PDF) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    summary["ratio_limits"] = list(ratio_limits)
    return summary


def main() -> None:
    """校验输入后生成两张 PDF 并打印摘要。"""

    matrix, posteriors, result = verified_inputs()
    CONTOUR_PDF.parent.mkdir(parents=True, exist_ok=True)
    contour_checks = write_contour_figure(posteriors)
    curve_summary = write_p0_curve_figure(matrix)
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "contour_pdf": {"path": str(CONTOUR_PDF), "bytes": CONTOUR_PDF.stat().st_size, "sha256": sha256(CONTOUR_PDF)},
            "p0_curve_pdf": {"path": str(P0_CURVE_PDF), "bytes": P0_CURVE_PDF.stat().st_size, "sha256": sha256(P0_CURVE_PDF)},
        },
        "inputs": {
            "matrix": {"path": str(MATRIX), "sha256": sha256(MATRIX)},
            "posteriors": {"path": str(POSTERIORS), "sha256": sha256(POSTERIORS)},
            "results": {"path": str(RESULT_JSON), "sha256": sha256(RESULT_JSON)},
        },
        "contour_checks": contour_checks,
        "p0_curve_summary": {
            name: (values if isinstance(values, list) and len(values) <= 14 else "...")
            for name, values in curve_summary.items()
        },
        "note": "pre 的 posterior 行来自冻结 v1 继承；sto_only 为五方法新链；曲线为 matched100 测量均值。",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
