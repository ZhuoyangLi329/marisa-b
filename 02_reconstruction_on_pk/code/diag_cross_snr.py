#!/usr/bin/env python3
"""互谱块信噪比解剖：为什么 knot 的互谱（sk/fk）没带来增益。

互谱估计量的方差（独立 Poisson 样本、Gaussian 近似）：
  Var(P_cd) ~= [(P_cc+sn_c)(P_dd+sn_d) + P_cd^2] / Nmodes
期望不含 sn，但方差含两类 Poisson 噪声乘积——knot 的 sn_k=V/n_k≈1.3e5
会撑爆 sk/fk 的误差棒。本脚本实测各互谱块（100 套）的信噪比并与
方差公式对照。
"""
import math
from pathlib import Path

import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
PRODUCTS = PART / "envclass_p0only/full"
CACHE = PART / "envclass_p0only/fullspec/bare_auto_cache"
CLASS_FULL = {"v": "void", "s": "sheet", "f": "filament", "k": "knot"}
K_CLASS = {"v": 8.0054e10, "s": 7.9474e10, "f": 7.9472e10, "k": 7.9861e10}
K_CROSS = 7.947e10
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])
PAIRS = [("vs",), ("vf",), ("sf",), ("sk",), ("fk",)]


def n_sn(c, real):
    with np.load(PRODUCTS / "fiducial" / f"real{real:03d}" / f"{CLASS_FULL[c]}_pk.npz") as z:
        norm = float(np.asarray(z["norm"])[0])
        sn = np.asarray(z["num_shotnoise"], dtype=np.float64) / norm
    return math.sqrt(norm * 1.0e9), sn


def auto(c, real):
    bare = np.load(CACHE / "fiducial" / f"real{real:03d}" / f"{c}.npy")
    n_c, sn = n_sn(c, real)
    return K_CLASS[c] * bare / (n_c * n_c * NM) - sn


def cross(cd, real):
    a, b = cd[0], cd[1]
    key1 = f"cross_{CLASS_FULL[a]}{CLASS_FULL[b]}_pk.npz"
    key2 = f"cross_{CLASS_FULL[b]}{CLASS_FULL[a]}_pk.npz"
    base = PRODUCTS / "fiducial" / f"real{real:03d}"
    key = key1 if (base / key1).is_file() else key2
    with np.load(base / key) as z:
        pk = np.asarray(z["pk"], dtype=np.float64)
    na, _ = n_sn(a, real)
    nb, _ = n_sn(b, real)
    return K_CROSS * pk / (na * nb * NM)


reals = list(range(100))
A = {c: np.asarray([auto(c, r) for r in reals]) for c in "vsfk"}
print("n_mean:", {c: round(float(np.mean([n_sn(c, r)[0] for r in reals])), 0) for c in "vsfk"})
sn = {c: np.mean([n_sn(c, r)[1] for r in reals], axis=0) for c in "vsfk"}

print(f"\n{'pair':<5}{'r_cd(bin2-12 均值)':>18}{'实测σ/均值(bin3)':>16}{'方差公式σ/均值':>14}{'sn 项占比':>10}")
for (cd,) in PAIRS:
    a, b = cd[0], cd[1]
    X = np.asarray([cross(cd, r) for r in reals])
    mean = X.mean(axis=0)
    std_real = X.std(axis=0, ddof=1)  # 单套误差
    # 方差公式（单套）：[(Pcc+sn_c)(Pdd+sn_d)+P_cd^2]/Nmodes
    var_pred = ((A[a].mean(axis=0) + sn[a]) * (A[b].mean(axis=0) + sn[b]) + mean**2) / NM
    std_pred = np.sqrt(var_pred)
    var_nosn = (A[a].mean(axis=0) * A[b].mean(axis=0) + mean**2) / NM
    sn_frac = 1.0 - var_nosn / var_pred  # sn 项在方差中的占比
    r_cd = np.mean(mean[1:] / np.sqrt(A[a].mean(axis=0)[1:] * A[b].mean(axis=0)[1:]))
    print(f"{cd:<5}{r_cd:>18.3f}{std_real[3]/abs(mean[3]):>16.3f}"
          f"{std_pred[3]/abs(mean[3]):>14.3f}{np.mean(sn_frac[1:]):>10.2f}")

print("\n== 互谱块的信息量（响应幅度 / 单套误差，粗 SNR，bin1-6）==")
BETA = {"v": -0.056, "s": 1.132, "f": 1.079, "k": 1.721}
PM = np.exp(np.interp(np.log(np.load(PRODUCTS / "fiducial/real000/void_pk.npz")["k"]),
                      np.log(np.load(PART / "inputs/p0_finite_local_png_lattice.npz")["pk_k"]),
                      np.log(np.load(PART / "inputs/p0_finite_local_png_lattice.npz")["PL_over_M_lattice"])))
for (cd,) in PAIRS:
    a, b = cd[0], cd[1]
    X = np.asarray([cross(cd, r) for r in reals])
    std_real = X.std(axis=0, ddof=1)
    resp = (BETA[a] + BETA[b]) * PM  # 每单位 fNL 的响应（近似）
    snr = np.abs(resp) / std_real
    print(f"{cd}: response-SNR bins1-6 = {np.array2string(snr[:6], precision=3, floatmode='fixed')}")
