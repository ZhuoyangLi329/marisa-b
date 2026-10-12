#!/usr/bin/env python3
"""实测响应形状诊断：dP = <P^LC_p> - <P^fid> vs 理论响应形状。

判定问题：PNG 响应项的形状/标度是否带 T_c（fid 谱/PL）因子。
输出：自谱/互谱的 dP 归一形状，与 (a) 2*beta*PL/M 线性形状、
(b) T_c*2*beta*PL/M 带形状 两种理论的逐 bin 比值。
比值平坦 => 该理论形状正确；单调漂移 => 形状失配。
"""
import json
import math
from pathlib import Path

import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
PRODUCTS = PART / "envclass_p0only/full"
CACHE = PART / "envclass_p0only/fullspec/bare_auto_cache"
CALIB = PART / "envclass_p0only/calibration/ENVCLASS_CALIBRATION.json"
LATTICE = PART / "inputs/p0_finite_local_png_lattice.npz"
CLASS_FULL = {"v": "void", "s": "sheet", "f": "filament"}
K_CLASS = {"v": 8.0054e10, "s": 7.9474e10, "f": 7.9472e10}
K_CROSS = 7.947e10
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])

calib = json.loads(CALIB.read_text())
beta = {c: float(calib["classes"][CLASS_FULL[c]]["bphi_over_b1"]["median"]) for c in "vsf"}

with np.load(LATTICE) as z:
    k_old = np.asarray(z["pk_k"], dtype=float)
    PL_old = np.asarray(z["PL_lattice"], dtype=float)
    PM_old = np.asarray(z["PL_over_M_lattice"], dtype=float)
with np.load(PRODUCTS / "fiducial/real000/void_pk.npz") as z:
    k_new = np.asarray(z["k"], dtype=float)


def interp(arr):
    return np.exp(np.interp(np.log(k_new), np.log(k_old), np.log(np.maximum(arr, 1e-300))))


PL, PM = interp(PL_old), interp(PM_old)


def n_sn(c, cosmo, real):
    with np.load(PRODUCTS / cosmo / f"real{real:03d}" / f"{CLASS_FULL[c]}_pk.npz") as z:
        norm = float(np.asarray(z["norm"])[0])
        sn = np.asarray(z["num_shotnoise"], dtype=np.float64) / norm
    return math.sqrt(norm * 1.0e9), sn


def auto(c, cosmo, real):
    bare = np.load(CACHE / cosmo / f"real{real:03d}" / f"{c}.npy")
    n_c, sn = n_sn(c, cosmo, real)
    return K_CLASS[c] * bare / (n_c * n_c * NM) - sn


def cross(cd, cosmo, real):
    a, b = cd[0], cd[1]
    with np.load(PRODUCTS / cosmo / f"real{real:03d}" / f"cross_{CLASS_FULL[a]}{CLASS_FULL[b]}_pk.npz") as z:
        pk = np.asarray(z["pk"], dtype=np.float64)
    na, _ = n_sn(a, cosmo, real)
    nb, _ = n_sn(b, cosmo, real)
    return K_CROSS * pk / (na * nb * NM)


reals = list(range(100))
print("== 自谱响应 dP_cc / 100 vs 理论 ==")
for c in "vsf":
    d_fid = np.mean([auto(c, "fiducial", r) for r in reals], axis=0)
    d_lc = np.mean([auto(c, "LC_p", r) for r in reals], axis=0)
    dp = (d_lc - d_fid)
    T_c = d_fid / PL
    # 理论线性响应（对 b1^2 归一）：2 beta PL/M（fNL=100 的 1/100）
    th_lin = 100.0 * 2.0 * beta[c] * PM          # fNL=100 一阶响应
    th_T = T_c * th_lin                           # 带 T 形状
    scale_lin = np.sum(dp * th_lin) / np.sum(th_lin * th_lin)
    scale_T = np.sum(dp * th_T) / np.sum(th_T * th_T)
    ratio_lin = dp / th_lin
    print(f"\n[{c}] beta={beta[c]:.3f}")
    print(f"  dP/100            : {np.array2string(dp / 100, precision=1)}")
    print(f"  dP/2b*PL/M*100    : {np.array2string(ratio_lin, precision=3, floatmode='fixed')}  (flat=>线性形状对, scale~{scale_lin:.2f})")
    print(f"  dP/(T*2b*PL/M*100): {np.array2string(dp / th_T, precision=3, floatmode='fixed')}  (flat=>带T形状对, scale~{scale_T:.2f})")
    print(f"  T_c               : {np.array2string(T_c, precision=3, floatmode='fixed')}")

print("\n== 互谱响应 dP_cd / 100 vs 理论（含 fNL^2 修正未扣除，一阶对比）==")
for cd in ("sf", "vs", "vf"):
    a, b = cd[0], cd[1]
    d_fid = np.mean([cross(cd, "fiducial", r) for r in reals], axis=0)
    d_lc = np.mean([cross(cd, "LC_p", r) for r in reals], axis=0)
    dp = d_lc - d_fid
    th_lin = 100.0 * (beta[a] + beta[b]) * PM
    T_cd = d_fid / PL
    th_T = T_cd * th_lin
    print(f"\n[{cd}]")
    print(f"  dP/100                 : {np.array2string(dp / 100, precision=1)}")
    print(f"  dP/((ba+bb)PL/M*100)   : {np.array2string(dp / th_lin, precision=3, floatmode='fixed')}")
    print(f"  dP/(T_cd*resp*100)     : {np.array2string(dp / th_T, precision=3, floatmode='fixed')}")
