#!/usr/bin/env python3
"""三方法全景图：fNL vs B-kmax（pre / marisa v2 / sto_marisa v4）。PDF only。"""
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
OUT = HERE / "figures/fnl_vs_bkmax_panorama_3methods.pdf"

d = json.loads(VALIDITY.read_text())
b_pre = {r["kmax_h_mpc"]: (r["map"]["fNL"], r["minos"]["fNL"]["lower_error"],
                           r["minos"]["fNL"]["upper_error"], r["gof_p_value_data_only"], True)
         for r in d["rows"] if r["method"] == "pre" and r["input_fNL"] == 100.0}
d4 = json.loads((AN / "STO_MARISA_B_V4_PAWARE_REPRO_SCAN.json").read_text())
b_sm = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                          r["gof_p_value_data_only"], r["minos_fNL"].get("valid", True)) for r in d4["rows"]}
dm = json.loads((AN / "MARISA_V2_TRILOGY.json").read_text())
b_ma = {r["kmax"]: (r["fNL"], -r["sigma"], r["sigma"], r["gof"], True) for r in dm["b_only_rows"]}

dj = json.loads((AN / "JOINT_PB_PRE_VS_STOMARISA_V4_MATCHED250.json").read_text())
j_pre = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                           r["gof_p_value_data_only"], r["minos_fNL"]["valid"]) for r in dj["rows"]["pre"]}
j_sm = {r["kmax_h_mpc"]: (r["map_fNL"], r["minos_fNL"]["lower"], r["minos_fNL"]["upper"],
                          r["gof_p_value_data_only"], r["minos_fNL"]["valid"]) for r in dj["rows"]["sto_marisa"]}
j_ma = {r["kmax"]: (r["fNL"], -r["sigma"], r["sigma"], r["gof"], True) for r in dm["joint_rows"]}

dp = json.loads((AN / "P_ONLY_PRE_VS_STOMARISA.json").read_text())
ponly = {"pre": dp["results"]["pre"], "sto_marisa": dp["results"]["sto_marisa"]}
# marisa P-only 用 trilogy（caveat：矩阵 cov 口径偏保守）
ponly["marisa"] = {"map_fNL": dm["p_only"]["fNL"], "sigma": dm["p_only"]["sigma"]}

C = {"pre": "#4E79A7", "marisa": "#E15759", "sto_marisa": "#B40426"}
CUTS = sorted(set(b_pre))
PANELS = [
    ("pre", b_pre, j_pre, "pre (p=1.179)"),
    ("marisa", b_ma, j_ma, "marisa v2 (p=1.122, reproduced)"),
    ("sto_marisa", b_sm, j_sm, "sto+marisa v4 (p=1.282, P250+B250)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True, constrained_layout=True)
for name, bsrc, jsrc, subtitle in [(p[0], p[1], p[2], p[3]) for p in PANELS]:
    ax = axes[["pre", "marisa", "sto_marisa"].index(name)]
    ax.axhline(100.0, color="0.35", ls="--", lw=1.0)
    po = ponly[name]
    ax.axhspan(po["map_fNL"] - po["sigma"], po["map_fNL"] + po["sigma"], color="#7A4EAB", alpha=0.12)
    ax.axhline(po["map_fNL"], color="#7A4EAB", ls=":", lw=1.6)
    ax.text(0.203, po["map_fNL"], "P-only", color="#7A4EAB", fontsize=8, va="center")
    kb = np.array(CUTS)
    for src, color, label, mk, dy in ((bsrc, C[name], "B-only", "o", 1), (jsrc, C[name], "P+B", "s", -1)):
        f = np.array([src[c][0] for c in CUTS])
        error_valid = np.array([src[c][4] for c in CUTS], dtype=bool)
        lo = np.array([src[c][0] + min(src[c][1], src[c][2]) if src[c][4] else src[c][0]
                       for c in CUTS])
        hi = np.array([src[c][0] + max(src[c][1], src[c][2]) if src[c][4] else src[c][0]
                       for c in CUTS])
        ok = np.array([src[c][3] > 0.01 for c in CUTS])
        ls = "--" if label == "P+B" else "-"
        ax.errorbar(kb, f, yerr=[f - lo, hi - f], color=color, marker=mk, ms=5.5, capsize=3,
                    lw=1.2, ls=ls, label=label, alpha=0.85)
        for xi, fi, li, ui, valid in zip(kb, f, f - lo, hi - f, error_valid):
            text = f"{fi:.0f}$\\pm${(li+ui)/2:.0f}" if valid else f"{fi:.0f}"
            ax.annotate(text, (xi, fi),
                        xytext=(0, dy * (max(li, ui) + 6)), textcoords="offset points",
                        ha="center", fontsize=6, color=color)
        bad = ~ok
        if bad.any():
            ax.scatter(kb[bad], f[bad], facecolors="white", edgecolors=color, s=40, zorder=5)
    ax.set_xlabel(r"$k_{\max}^{\rm B}$  (P block fixed $k_{\max}=0.08$)")
    ax.set_title(subtitle, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xticks(CUTS)
    ax.set_xticklabels([f"{c:.2f}" for c in CUTS], fontsize=7.5)
axes[0].set_ylabel(r"$\hat f_{\rm NL}$  (truth $=100$)")
axes[0].set_ylim(30, 200)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle(r"$f_{\rm NL}$ vs $B$ $k_{\max}$, 3 methods (P-only band / B-only solid / P+B dashed; "
             "open=GoF$\\leq$0.01; B-only fixed-$b_1$, joint free-$b_1$; "
             "marisa joint uses block-diag cov)", fontsize=10.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, metadata={"Title": "fnl vs B-kmax panorama 3 methods"})
print(f"saved: {OUT}")
