#!/usr/bin/env python3
"""诊断 vsf 全谱系 MAP 残差结构：逐块 target vs 固定参数模型 + C-S 检查。"""
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
print("calib void keys:", list(calib["classes"]["void"].keys()))


def grab(cls_full, key):
    node = calib["classes"][cls_full].get(key)
    if isinstance(node, dict):
        return float(node.get("median", node.get("value", 1.0)))
    return float(node)


beta = {c: grab(CLASS_FULL[c], "bphi_over_b1") for c in "vsf"}
b1 = {c: grab(CLASS_FULL[c], "b1") for c in "vsf"}
print("beta:", {k: round(v, 3) for k, v in beta.items()})
print("b1:  ", {k: round(v, 3) for k, v in b1.items()})


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
T = {}
for c in "vsf":
    T[c] = np.mean([auto(c, "fiducial", r) for r in reals], axis=0)
for cd in ("vs", "vf", "sf"):
    T[cd] = np.mean([cross(cd, "fiducial", r) for r in reals], axis=0)

with np.load(LATTICE) as z:
    k_old = np.asarray(z["pk_k"], dtype=float)
    PL_old = np.asarray(z["PL_lattice"], dtype=float)
with np.load(PRODUCTS / "fiducial/real000/void_pk.npz") as z:
    k_new = np.asarray(z["k"], dtype=float)
PL = np.exp(np.interp(np.log(k_new), np.log(k_old), np.log(np.maximum(PL_old, 1e-300))))

print("\n== 自谱 target/(b1^2 PL)  低 k 应 ≈1（sn0 小）==")
for c in "vsf":
    print(f"  {c}: {np.array2string(T[c] / (b1[c] ** 2 * PL), precision=3, floatmode='fixed')}")
print("\n== 互谱 target/(b1c*b1d*PL)  应 ≈ 常数 1 ==")
for cd in ("vs", "vf", "sf"):
    bc, bd = cd[0], cd[1]
    print(f"  {cd}: {np.array2string(T[cd] / (b1[bc] * b1[bd] * PL), precision=3, floatmode='fixed')}")
print("\n== Cauchy-Schwarz P_cd^2/(P_cc*P_dd)  应 <=1 ==")
for cd in ("vs", "vf", "sf"):
    c, d = cd[0], cd[1]
    print(f"  {cd}: {np.array2string(T[cd] ** 2 / (T[c] * T[d]), precision=3, floatmode='fixed')}")
print("\n== 互谱/自谱根号比 sqrt 形式（=corr·sqrt(1+sn/P) 混合，参考）==")
for cd in ("vs", "vf", "sf"):
    c, d = cd[0], cd[1]
    corr = T[cd] / np.sqrt(T[c] * T[d])
    print(f"  {cd}: {np.array2string(corr, precision=3, floatmode='fixed')}")
