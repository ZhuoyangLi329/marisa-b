#!/usr/bin/env python3
"""固定 b1 的 sto_marisa vs pre 全面对比图（PDF，四面板）。

面板 a：fNL 恢复 vs kmax（minos 区间，truth=100 虚线，GoF>0.01 干净区底纹）
面板 b：σ(fNL) vs kmax + 比值 sto_marisa/pre（右轴）
面板 c：GoF p-value vs kmax（log 轴，0.01 阈值）
面板 d：b2 vs kmax
数据：STO_MARISA_B_FIXEDB1_KMAX_SCAN.json（truth+100）与旧 validity scan
（pre，truth+100，b1=1.93 固定）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
STO_JSON = HERE / "analysis/STO_MARISA_B_FIXEDB1_KMAX_SCAN.json"
PRE_JSON = Path("/pscratch/sd/l/lzy/reconstruction-png/analysis/"
                "quijote_halo_z0p5_direct_finite_tree_kmax_validity_scan_v42p875"
                "/DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.json")
OUT = HERE / "figures/sto_marisa_vs_pre_fixedb1_kmax_scan.pdf"

d_sto = json.loads(STO_JSON.read_text())
d_pre = json.loads(PRE_JSON.read_text())

sto = {r["kmax_h_mpc"]: r for r in d_sto["rows"]}
pre = {}
for r in d_pre["rows"]:
    if r["method"] == "pre" and r["input_fNL"] == 100.0:
        pre[r["kmax_h_mpc"]] = r
cuts = sorted(set(sto) & set(pre))

k = np.array(cuts)
sto_f = np.array([sto[c]["map_fNL"] for c in cuts])
sto_lo = np.array([sto[c]["map_fNL"] + min(sto[c]["minos_fNL"]["lower"], sto[c]["minos_fNL"]["upper"]) for c in cuts])
sto_hi = np.array([sto[c]["map_fNL"] + max(sto[c]["minos_fNL"]["lower"], sto[c]["minos_fNL"]["upper"]) for c in cuts])
sto_ok = np.array([sto[c]["minos_fNL"]["valid"] and sto[c]["gof_p_value_data_only"] > 0.01 for c in cuts])
sto_sig = (sto_hi - sto_lo) / 2.0
sto_gof = np.array([sto[c]["gof_p_value_data_only"] for c in cuts])
sto_b2 = np.array([sto[c]["map"]["b2"] for c in cuts])

pre_f = np.array([pre[c]["map"]["fNL"] for c in cuts])
pre_lo = np.array([pre[c]["map"]["fNL"] + pre[c]["minos"]["fNL"]["lower_error"] for c in cuts])
pre_hi = np.array([pre[c]["map"]["fNL"] + pre[c]["minos"]["fNL"]["upper_error"] for c in cuts])
pre_gof = np.array([pre[c]["gof_p_value_data_only"] for c in cuts])
pre_b2 = np.array([pre[c]["map"]["b2"] for c in cuts])
pre_sig = (pre_hi - pre_lo) / 2.0

C_STO = "#B40426"   # sto_marisa 红
C_PRE = "#4E79A7"   # pre 蓝

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)

# ---- a: recovery ----
ax = axes[0, 0]
ax.axhline(100.0, color="0.4", ls="--", lw=1.0, label=r"truth $f_{\rm NL}=100$")
ax.axvspan(0.07, 0.125, color=C_STO, alpha=0.07)
ax.axvspan(0.07, 0.155, color=C_PRE, alpha=0.05)
ax.errorbar(k - 0.002, pre_f, yerr=[pre_f - pre_lo, pre_hi - pre_f], color=C_PRE,
            marker="o", ms=5, capsize=3, lw=1.2, label="pre (b1=1.93 fixed)")
ax.errorbar(k + 0.002, sto_f, yerr=[sto_f - sto_lo, sto_hi - sto_f], color=C_STO,
            marker="s", ms=5, capsize=3, lw=1.2,
            label="sto+marisa (b1=2.228, p=1.282 fixed)")
for i, ok in enumerate(sto_ok):
    if not ok:
        ax.annotate("×", (k[i] + 0.002, sto_f[i]), color=C_STO, fontsize=9,
                    ha="center", va="center", weight="bold")
ax.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$\hat f_{\rm NL}$")
ax.set_title("(a) recovery（× = GoF≤0.01 或区间无效）")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
ax.set_ylim(40, 210)

# ---- b: sigma + ratio ----
ax = axes[0, 1]
ax.plot(k, pre_sig, color=C_PRE, marker="o", ms=5, lw=1.2, label=r"$\sigma(f_{\rm NL})$ pre")
ax.plot(k, sto_sig, color=C_STO, marker="s", ms=5, lw=1.2, label=r"$\sigma(f_{\rm NL})$ sto+marisa")
ax.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$\sigma(f_{\rm NL})$")
ax.set_title("(b) 统计误差与比值")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc="upper right")
axr = ax.twinx()
ratio = np.where(pre_sig > 0, sto_sig / pre_sig, np.nan)
axr.plot(k, ratio, color="0.35", marker="^", ms=5, lw=1.0, ls="--", label="sto+marisa / pre")
axr.axhline(1.0, color="0.6", lw=0.8)
axr.set_ylabel("比值 sto+marisa / pre", color="0.35")
axr.set_ylim(0, 1.4)
axr.legend(frameon=False, fontsize=8, loc="center right")

# ---- c: GoF ----
ax = axes[1, 0]
ax.semilogy(k, np.maximum(pre_gof, 1e-5), color=C_PRE, marker="o", ms=5, lw=1.2, label="pre")
ax.semilogy(k, np.maximum(sto_gof, 1e-5), color=C_STO, marker="s", ms=5, lw=1.2, label="sto+marisa")
ax.axhline(0.01, color="0.3", ls=":", lw=1.2)
ax.text(0.081, 0.013, "p=0.01", fontsize=8, color="0.3")
ax.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"GoF $p$-value (data-only)")
ax.set_title("(c) 拟合质量")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(1e-5, 2)

# ---- d: b2 ----
ax = axes[1, 1]
ax.plot(k, pre_b2, color=C_PRE, marker="o", ms=5, lw=1.2, label="pre")
ax.plot(k, sto_b2, color=C_STO, marker="s", ms=5, lw=1.2, label="sto+marisa")
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$b_2$ (MAP)")
ax.set_title("(d) 二阶局域 bias")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

fig.suptitle(r"Fixed-$b_1$ $B_{000}$ tree models: sto+marisa vs pre（LC$_p$ $f_{\rm NL}=100$ 注入恢复）",
             fontsize=12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, metadata={"Title": "sto_marisa vs pre fixed-b1 B000 kmax scan"})
print(f"saved: {OUT}")
