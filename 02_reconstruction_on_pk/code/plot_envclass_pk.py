#!/usr/bin/env python3
"""环境三类 P0 曲线图（fid 与 LC_p 各一页 + 比值，仅 PDF）。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
PRODUCTS = PART / "envclass_p0only/full"
OUTPUT = PART / "envclass_p0only/calibration/envclass_pk_curves.pdf"
CLASSES = ("void", "sheet", "filament")
COLORS = {"void": "#4E79A7", "sheet": "#F28E2B", "filament": "#59A14F"}
N_REAL = 100


def load(cls: str, cosmo: str) -> tuple[np.ndarray, int]:
    rows = []
    root = PRODUCTS / cosmo
    for real in range(1000):
        path = root / f"real{real:03d}" / f"{cls}_pk.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            rows.append(np.asarray(z["pk"], dtype=np.float64))
        if len(rows) >= N_REAL:
            break
    return np.vstack(rows), len(rows)


def bestfit_curve(cls: str, fNL: float) -> tuple[np.ndarray, dict]:
    """读校准后验的中位数参数算模型曲线（LC_p 侧用 fNL=100）。"""
    import json
    calib = json.loads((PART / "envclass_p0only/calibration/ENVCLASS_CALIBRATION.json").read_text(encoding="utf-8"))
    row = calib["classes"][cls]
    p = row["p"]["median"]; b1 = row["b1"]["median"]; shot = row["Pshot"]["median"]
    with np.load(PART / "envclass_p0only/full/fiducial/real000/void_pk.npz", allow_pickle=False) as z:
        k = np.asarray(z["k"], dtype=float)
    with np.load(PART / "inputs/p0_finite_local_png_lattice.npz", allow_pickle=False) as z:
        k_old = np.asarray(z["pk_k"], dtype=float)
        PL = np.exp(np.interp(np.log(k), np.log(k_old), np.log(np.asarray(z["PL_lattice"], dtype=float))))
        PLM = np.exp(np.interp(np.log(k), np.log(k_old), np.log(np.asarray(z["PL_over_M_lattice"], dtype=float))))
        Pphi = np.exp(np.interp(np.log(k), np.log(k_old), np.log(np.asarray(z["Pphi_lattice"], dtype=float))))
    bphi = 2.0 * 1.686 * (b1 - p)
    model = b1**2 * PL + 2.0 * fNL * b1 * bphi * PLM + fNL**2 * bphi**2 * Pphi + shot
    return model, {"p": p, "b1": b1, "Pshot": shot}


def main() -> None:
    with np.load(PRODUCTS / "fiducial" / "real000" / "void_pk.npz", allow_pickle=False) as z:
        k = np.asarray(z["k"], dtype=float)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        for cosmo, title, fnl in (("fiducial", r"input fNL$=0$ (fiducial)", 0.0),
                                   ("LC_p", r"input fNL$=+100$ (LC_p)", 100.0)):
            data = {}
            for cls in CLASSES:
                samples, n = load(cls, cosmo)
                data[cls] = (samples.mean(axis=0), np.sqrt(np.diag(np.cov(samples, rowvar=False, ddof=1))) / np.sqrt(n), n)
            figure, (top, bottom) = plt.subplots(
                2, 1, figsize=(7.6, 7.8), sharex=True,
                gridspec_kw={"height_ratios": (2.0, 1.0)},
            )
            for cls in CLASSES:
                mean, sigma, n = data[cls]
                top.errorbar(k, mean, yerr=sigma, fmt="o", ms=4, lw=1.2, capsize=2.5,
                             color=COLORS[cls], label=f"{cls} (n={n})")
                model, params = bestfit_curve(cls, fnl)
                top.plot(k, model, "-", lw=1.7, color=COLORS[cls], alpha=0.9,
                         label=f"{cls} best-fit")
            top.set_xscale("log"); top.set_yscale("log")
            top.set_ylabel(r"$\bar P_0(k)$")
            top.set_title(title)
            top.grid(alpha=0.15, which="both")
            top.legend(frameon=False, fontsize=7.5, ncol=2)
            # 下排：测量 / best-fit 残差（按各自误差归一，pull）。
            for cls in CLASSES:
                mean, sigma, _ = data[cls]
                model, _params = bestfit_curve(cls, fnl)
                pull = (mean - model) / sigma
                bottom.plot(k, pull, "o-", ms=4, lw=1.2, color=COLORS[cls], label=cls)
            bottom.axhspan(-1.0, 1.0, color="0.92")
            bottom.axhline(0.0, color="0.35", lw=0.8)
            bottom.set_xscale("log")
            bottom.set_xlabel(r"$k$ $[h/\mathrm{Mpc}]$")
            bottom.set_ylabel(r"$(\bar P_0-\mathrm{fit})/\sigma_{\bar P_0}$")
            bottom.set_ylim(-6, 6)
            bottom.grid(alpha=0.15, which="both")
            bottom.legend(frameon=False, fontsize=9)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(f"written: {OUTPUT}")


if __name__ == "__main__":
    main()
