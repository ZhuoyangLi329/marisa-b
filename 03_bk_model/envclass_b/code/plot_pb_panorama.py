#!/usr/bin/env python3
"""全景图：fNL 约束 vs B-kmax（P-kmax 固定 0.08）——pre 与 sto_marisa。

每方法一个面板，三条曲线：
- B-only（固定 b1 口径，B 端惯例）：误差棒 + 点线
- P+B 联合（b1 自由）：误差棒 + 点线
- P-only（12 bins k<=0.08，不随 B-kmax 变）：水平带（median ± σ）
truth=100 虚线；GoF>0.01 实心点、GoF<=0.01 空心点。PDF only。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
AN = HERE / "analysis"
VALIDITY = Path("/pscratch/sd/l/lzy/reconstruction-png/analysis/"
                "quijote_halo_z0p5_direct_finite_tree_kmax_validity_scan_v42p875"
                "/DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.json")
OUT = HERE / "figures/fnl_vs_bkmax_panorama_pre_stomarisa.pdf"

# ---- B-only pre（旧 validity scan，固定 b1=1.93，truth=+100）----
d = json.loads(VALIDITY.read_text())
b_pre = {}
for r in d["rows"]:
    if r["method"] == "pre" and r["input_fNL"] == 100.0:
        b_pre[r["kmax_h_mpc"]] = (r["map"]["fNL"], r["minos"]["fNL"]["lower_error"],
                                  r["minos"]["fNL"]["upper_error"], r["gof_p_value_data_only"])

# ---- B-only sto_marisa（v4 p-aware 重产，固定 b1=2.2281）----
d4 = json.loads((AN / "STO_MARISA_B_V4_PAWARE_REPRO_SCAN.json").read_text())
b_sm = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                          r["gof_p_value_data_only"]) for r in d4["rows"]}

# ---- P+B 联合（b1 自由）----
dj = json.loads((AN / "JOINT_PB_PRE_VS_STOMARISA_V2_FREEB1.json").read_text())
j_pre = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                           r["gof_p_value_data_only"]) for r in dj["rows"]["pre"]}
j_sm = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                          r["gof_p_value_data_only"]) for r in dj["rows"]["sto_marisa"]}

# ---- P-only（水平线）----
dp = json.loads((AN / "P_ONLY_PRE_VS_STOMARISA.json").read_text())
p_pre = dp["results"]["pre"]
p_sm = dp["results"]["sto_marisa"]

CUTS = sorted(set(b_pre) & set(b_sm))
C_B, C_J, C_P = "#4E79A7", "#B40426", "#7A4EAB"


def series(src):
    k = np.array(CUTS)
    f = np.array([src[c][0] for c in CUTS])
    lo = np.array([src[c][0] + min(src[c][1], src[c][2]) for c in CUTS])
    hi = np.array([src[c][0] + max(src[c][1], src[c][2]) for c in CUTS])
    ok = np.array([src[c][3] > 0.01 for c in CUTS])
    return k, f, f - lo, hi - f, ok


fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True, constrained_layout=True)
panels = [
    ("pre", axes[0], b_pre, j_pre, p_pre, "pre (B-only $b_1$=1.93 fixed; joint free; p=1.179)"),
    ("sto_marisa", axes[1], b_sm, j_sm, p_sm, "sto+marisa (B-only $b_1$=2.228 fixed; joint free; p=1.282)"),
]
for name, ax, bsrc, jsrc, ponly, subtitle in panels:
    kb, fb, lb, ub, okb = series(bsrc)
    kj, fj, lj, uj, okj = series(jsrc)
    ax.axhline(100.0, color="0.35", ls="--", lw=1.0)
    # P-only 水平带
    ax.axhspan(ponly["map_fNL"] - ponly["sigma"], ponly["map_fNL"] + ponly["sigma"],
               color=C_P, alpha=0.12)
    ax.axhline(ponly["map_fNL"], color=C_P, ls=":", lw=1.8)
    ax.text(0.202, ponly["map_fNL"], "P-only", color=C_P, fontsize=9, va="center")
    # B-only / P+B
    for k, f, l, u, ok, color, label, mk, dy in (
            (kb, fb, lb, ub, okb, C_B, "B-only", "o", 1),
            (kj, fj, lj, uj, okj, C_J, "P+B", "s", -1)):
        ax.errorbar(k, f, yerr=[l, u], color=color, marker=mk, ms=6, capsize=3,
                    lw=1.3, ls="-", label=label, mfc=color, mec=color, alpha=0.9)
        for xi, fi, li, ui in zip(k, f, l, u):
            sig = (li + ui) / 2.0
            ax.annotate(f"{fi:.1f}$\\pm${sig:.1f}", (xi, fi),
                        xytext=(0, dy * (max(li, ui) + 6)), textcoords="offset points",
                        ha="center", fontsize=6.2, color=color)
        bad = ~ok
        if bad.any():
            ax.errorbar(k[bad], f[bad], yerr=[l[bad], u[bad]], fmt="none", ecolor=color,
                        capsize=3, alpha=0.45)
            ax.scatter(k[bad], f[bad], facecolors="white", edgecolors=color,
                       s=42, zorder=5)
    ax.set_xlabel(r"$k_{\max}^{\rm B}\ [h\,{\rm Mpc}^{-1}]$  (P block fixed at $k_{\max}=0.08$)")
    ax.set_title(subtitle)
    ax.grid(alpha=0.25)
    ax.set_xticks(CUTS)
    ax.set_xticklabels([f"{c:.2f}" for c in CUTS], fontsize=8)
axes[0].set_ylabel(r"$\hat f_{\rm NL}$  (truth $=100$)")
axes[0].set_ylim(40, 185)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle(r"$f_{\rm NL}$ constraints vs $B$ $k_{\max}$: P-only (band) / B-only / P+B joint"
             "  (filled: GoF>0.01, open: GoF$\\leq$0.01; B-only uses fixed-$b_1$ caliber)", fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, metadata={"Title": "fnl vs B-kmax panorama pre sto_marisa"})
print(f"saved: {OUT}")
