#!/usr/bin/env python3
"""测类对相关函数 r_cd(k) 并检验其在 fNL=0/100 间的稳定性。

r_cd(k) = <P_cd> / sqrt(<P_cc>*<P_dd>)（FKP 口径，自谱已扣 sn）。
用于 fullspec v4 互谱模板修正：P_cd_model = r_cd(k)*b_c*b_d*[PL+PNG 项]。
检验：fid(fNL=0) vs LC_p(fNL=100) 的 r_cd 差异若 <<1%，则固定 fid 值安全。
"""
import json
import math
from pathlib import Path

import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
PRODUCTS = PART / "envclass_p0only/full"
CACHE = PART / "envclass_p0only/fullspec/bare_auto_cache"
CLASS_FULL = {"v": "void", "s": "sheet", "f": "filament"}
K_CLASS = {"v": 8.0054e10, "s": 7.9474e10, "f": 7.9472e10}
K_CROSS = 7.947e10
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])


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


def mean_over(cosmo, loader, reals):
    return np.mean([loader(r) for r in reals], axis=0)


reals = list(range(100))
out = {}
print(f"{'pair':<5}{'cosmo':<9}r_cd(k) 12 bins")
for cosmo in ("fiducial", "LC_p"):
    A = {c: mean_over(cosmo, lambda r, c=c: auto(c, cosmo, r), reals) for c in "vsf"}
    for cd in ("vs", "vf", "sf"):
        c, d = cd[0], cd[1]
        X = mean_over(cosmo, lambda r, cd=cd: cross(cd, cosmo, r), reals)
        r_cd = X / np.sqrt(A[c] * A[d])
        out[f"{cosmo}/{cd}"] = r_cd.tolist()
        print(f"{cd:<5}{cosmo:<9}{np.array2string(r_cd, precision=4, floatmode='fixed')}")

print("\n== fid vs LC_p 相对差（fNL=100 对 r 的稳定性）==")
for cd in ("vs", "vf", "sf"):
    rf = np.asarray(out[f"fiducial/{cd}"])
    rl = np.asarray(out[f"LC_p/{cd}"])
    denom = np.where(np.abs(rf) > 1e-3, np.abs(rf), 1e-3)
    print(f"{cd}: rel-dev {np.array2string((rl - rf) / denom, precision=3, floatmode='fixed')}")

save = PART / "envclass_p0only/fullspec/r_cd_measurement.json"
save.write_text(json.dumps({k: v for k, v in out.items()}, indent=2))
print(f"\nsaved: {save}")
