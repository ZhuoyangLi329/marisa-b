#!/usr/bin/env python3
"""P(k) only 对照（与 P+B 联合同口径的 P 块单独拟合）。

口径与 JOINT_PB runner 完全一致：pre 用 FNL250 的 pre_P0（250 target/500 cov）；
sto_marisa 用旧矩阵 post_P0（100/500）。参数 {fNL, b1, Pshot}，b1 自由；
p 固定 pre=1.179 / sm=1.2822；P0 tree 模板 12 bins。单点（P 无 kmax 维度）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution

VENDOR = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/vendor")
ANALYSIS = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/analysis")
LATTICE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz")
VOLUME = 42.875
DELTA_C = 1.686
FNL_BOUNDS = (-300.0, 300.0)
NAMES = ("fNL", "b1", "Pshot")
CASES = {"pre": 1.179, "sto_marisa": 1.2822}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    with np.load(LATTICE, allow_pickle=False) as z:
        k_old = np.asarray(z["pk_k"], dtype=float)
        PL0 = np.asarray(z["PL_lattice"], dtype=float)
        PM0 = np.asarray(z["PL_over_M_lattice"], dtype=float)
        PQ0 = np.asarray(z["Pphi_lattice"], dtype=float)
    with np.load(VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as z:
        k12 = np.asarray(z["pk_k"], dtype=float)
        pre_p_fid = np.asarray(z["fiducial_pre_P0"], dtype=np.float64)
        pre_p_lc = np.asarray(z["LC_p_pre_P0"], dtype=np.float64)

    def interp(arr):
        return np.exp(np.interp(np.log(k12), np.log(k_old), np.log(np.maximum(arr, 1e-300))))
    PL, PM, PQ = interp(PL0), interp(PM0), interp(PQ0)

    with np.load(VENDOR / "data/sto_marisa_matrix.npz", allow_pickle=False) as z:
        sm_p_fid = np.asarray(z["fiducial_post_P0"], dtype=np.float64)[:, :12]
        sm_p_lc = np.asarray(z["LC_p_post_P0"], dtype=np.float64)[:, :12]

    data = {"pre": (pre_p_fid, pre_p_lc.mean(axis=0)),
            "sto_marisa": (sm_p_fid, sm_p_lc.mean(axis=0))}
    results = {}
    for name, p_halo in CASES.items():
        fid, target = data[name]
        nfid = fid.shape[0]
        cov_fit = np.cov(fid, rowvar=False, ddof=1) / VOLUME
        hartlap = (float(nfid) - 12 - 2.0) / (float(nfid) - 1.0)
        chol = np.linalg.cholesky(cov_fit) / math.sqrt(hartlap)

        def objective_impl(values):
            m = dict(zip(NAMES, [float(v) for v in values]))
            bphi = 2.0 * DELTA_C * (m["b1"] - p_halo)
            pred = (m["b1"] ** 2 * PL + m["fNL"] * 2.0 * m["b1"] * bphi * PM
                    + m["fNL"] ** 2 * bphi ** 2 * PQ + m["Pshot"])
            white = solve_triangular(chol, pred - target, lower=True)
            return float(white @ white)

        _sig = ", ".join(NAMES)
        _body = ("def _objective(" + _sig + "):\n"
                 "    return _impl((" + ", ".join(NAMES) + ",))\n")
        _ns = {"_impl": objective_impl}
        exec(_body, _ns)
        objective = _ns["_objective"]

        fits = []
        for f0 in (100.0, 0.0, 50.0, 150.0):
            start = {"fNL": f0, "b1": 1.93 if name == "pre" else 2.228, "Pshot": 0.0}
            fit = Minuit(objective, **start)
            fit.errordef = Minuit.LEAST_SQUARES
            fit.limits["fNL"] = FNL_BOUNDS
            fit.limits["b1"] = (0.1, 8.0)
            fit.limits["Pshot"] = (-1.0e5, 1.0e5)
            fit.errors["fNL"] = 10.0
            fit.errors["b1"] = 0.15
            fit.errors["Pshot"] = 1000.0
            fit.strategy, fit.tol = 1, 1e-6
            fit.migrad(ncall=200000, iterate=8, use_simplex=True)
            fits.append(fit)
        valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
        best = min(valid or fits, key=lambda f: float(f.fval))
        best.hesse()
        best.minos("fNL", ncall=100000)
        err = best.merrors["fNL"]
        m = {n: float(best.values[n]) for n in NAMES}
        lo, hi = sorted((m["fNL"] + err.lower, m["fNL"] + err.upper))
        results[name] = {
            "map_fNL": m["fNL"], "sigma": (hi - lo) / 2.0,
            "interval": [lo, hi], "b1": m["b1"], "Pshot": m["Pshot"],
            "gof_p": float(chi2_distribution.sf(float(best.fval), 12 - 3)),
        }
        print(f"[{name:<11} P-only] fNL={m['fNL']:+8.2f} ± {(hi-lo)/2:.2f} b1={m['b1']:.3f} "
              f"gof={results[name]['gof_p']:.4f}", flush=True)

    payload = {
        "schema": "p-only-pre-vs-stomarisa-v1",
        "created_utc": utc_now(),
        "caliber": "same P blocks as JOINT_PB runner; b1 free; p fixed per method",
        "results": results,
    }
    out = ANALYSIS / "P_ONLY_PRE_VS_STOMARISA.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
