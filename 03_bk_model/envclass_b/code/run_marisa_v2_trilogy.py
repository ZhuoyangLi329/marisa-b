#!/usr/bin/env python3
"""触发 B 一键脚本：marisa only v2 三件套 + p 重校准（全自产数据）。

数据（v2 重产 png_p=1.1217、bias=1.9364，fid 500 + LC_p 100）：
- P 块：post_pk.npz 的 fundamental 47 bins 取前 12（k<=0.0785；与 pre 的
  paperbin 12 bins binning 不同，k 中心差 <8%，模板按各自 k 插值，对照时声明）
- B 块：post_b000.h5 的 120 bins（certified 几何）
三个分析：
1. freep 重校准：fNL=100 固定、(p, b1, Pshot) 自由（P0 模板，新 LC_p+新 fid）→ 对照 1.1217 收敛
2. B-only kmax 扫描：b1=1.9364 固定、p_halo=1.1217、adaptive(1.9364,1.1217)、marisa 模板
3. P+B 联合：b1 自由、{fNL,Pshot,b2,γ2,A,B}（P 块 fundamental 12 + B 块 cut）
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
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution

VENDOR = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/vendor")
HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
ANALYSIS = HERE / "analysis"
NEW_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png/"
                "result_quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217/full")
LATTICE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz")
VOLUME = 42.875
DELTA_C = 1.686
B1_FIXED = 1.9364310827111566
P_FIXED = 1.1217
FNL_BOUNDS = (-300.0, 300.0)
PRIOR_WEIGHT = 1.0 / 25.0
B_NAMES = ("fNL", "b2", "gamma2", "Ashot_residual", "Bshot_residual")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _vendor_load(name, filename):
    import types
    path = VENDOR / "py" / filename
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(open(path, encoding="utf-8").read(), str(path), "exec"), module.__dict__)
    return module


def load_block(cosmo, attempts=6, wait=20):
    """堆叠 B(120)（marisa v2 自产；含 Lustre 抖动重试）。P 块改用旧矩阵（见 main）。"""
    import h5py
    import time as _time
    for attempt in range(attempts):
        bs = []
        for real_dir in sorted((NEW_ROOT / cosmo).glob("real*")):
            bf = real_dir / "post_b000.h5"
            try:
                ok = bf.is_file() and (real_dir / "DONE").is_file()
            except OSError:
                ok = False
            if not ok:
                continue
            with h5py.File(bf, "r") as f:
                bs.append(np.asarray(f["0_0_0/value"][()], dtype=np.float64))
        need = 100 if cosmo == "LC_p" else 300
        if len(bs) >= need:
            return np.asarray(bs)
        print(f"load_block({cosmo}) attempt {attempt+1}: {len(bs)} sets, retrying...", flush=True)
        _time.sleep(wait)
    raise RuntimeError(f"insufficient B samples for {cosmo}")


def p_templates(k12):
    with np.load(LATTICE, allow_pickle=False) as z:
        k_old = np.asarray(z["pk_k"], dtype=float)
        PL0 = np.asarray(z["PL_lattice"], dtype=float)
        PM0 = np.asarray(z["PL_over_M_lattice"], dtype=float)
        PQ0 = np.asarray(z["Pphi_lattice"], dtype=float)

    def interp(arr):
        return np.exp(np.interp(np.log(k12), np.log(k_old), np.log(np.maximum(arr, 1e-300))))
    return interp(PL0), interp(PM0), interp(PQ0)


def minuit_fit(objective_impl, names, starts, limits, errors):
    _sig = ", ".join(names)
    _body = ("def _objective(" + _sig + "):\n"
             "    return _impl((" + ", ".join(names) + ",))\n")
    _ns = {"_impl": objective_impl}
    exec(_body, _ns)
    objective = _ns["_objective"]
    fits = []
    for start in starts:
        fit = Minuit(objective, **start)
        fit.errordef = Minuit.LEAST_SQUARES
        for n, lim in limits.items():
            fit.limits[n] = lim
        for n, e in errors.items():
            fit.errors[n] = e
        fit.strategy, fit.tol = 1, 1e-6
        fit.migrad(ncall=200000, iterate=8, use_simplex=True)
        fits.append(fit)
    valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
    return min(valid or fits, key=lambda f: float(f.fval))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    b_fid = load_block("fiducial")
    b_lc = load_block("LC_p")
    # P 块：marisa v2 生产未出 post_pk（measure_pk 未生效），退回五方法 p0 矩阵
    # （paperbin 12 bins，与 sto_marisa 线的 P 块口径对称）；cov 用其 fid500 单套协方差。
    FIVE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/inputs/five_method_p0_matrix.npz")
    with np.load(FIVE, allow_pickle=False) as z:
        target_p = np.asarray(z["marisa__target_fNL100_matched100"], dtype=np.float64)
        cov_p = np.asarray(z["marisa__covariance_fNL0_fid500_one_box"], dtype=np.float64)
    with np.load(VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as z:
        k_fid = np.asarray(z["pk_k"], dtype=np.float64)
    print(f"data: B fid {b_fid.shape[0]}, LC_p {b_lc.shape[0]}; P block from five-method matrix (paperbin 12)")
    PL, PM, PQ = p_templates(k_fid)

    # ---------- 1. freep 重校准（fNL=100 固定，p/b1/Pshot 自由）----------
    # cov_p 来自矩阵（已是单套口径，已含 42.875 换算；Hartlap 惯例补上）
    hart = (500.0 - 12 - 2.0) / 499.0
    chol_p = np.linalg.cholesky(cov_p) / math.sqrt(hart)

    def freep_obj(v):
        p_, b1_, ps_ = [float(x) for x in v]
        bphi = 2.0 * DELTA_C * (b1_ - p_)
        pred = b1_ ** 2 * PL + 100.0 * 2.0 * b1_ * bphi * PM + 100.0 ** 2 * bphi ** 2 * PQ + ps_
        w = solve_triangular(chol_p, pred - target_p, lower=True)
        return float(w @ w)

    best = minuit_fit(freep_obj, ("p", "b1", "Pshot"),
                      starts=[{"p": P_FIXED, "b1": B1_FIXED, "Pshot": 0.0},
                              {"p": 1.0, "b1": 1.93, "Pshot": 0.0}],
                      limits={"p": (-2, 6), "b1": (0.1, 8), "Pshot": (-1e5, 1e5)},
                      errors={"p": 0.1, "b1": 0.15, "Pshot": 1000.0})
    best.hesse()
    p_recal = float(best.values["p"])
    print(f"[freep recal] p={p_recal:.4f} +/- {float(best.errors['p']):.4f} "
          f"b1={float(best.values['b1']):.4f} gof={float(chi2_distribution.sf(float(best.fval), 9)):.3f} "
          f"(input was {P_FIXED})")

    # ---------- 1b. P-only（p 固定 1.1217，fNL/b1/Pshot 自由）----------
    def ponly_obj(v):
        f_, b1_, ps_ = [float(x) for x in v]
        bphi = 2.0 * DELTA_C * (b1_ - P_FIXED)
        pred = b1_ ** 2 * PL + f_ * 2.0 * b1_ * bphi * PM + f_ ** 2 * bphi ** 2 * PQ + ps_
        w = solve_triangular(chol_p, pred - target_p, lower=True)
        return float(w @ w)

    best_p = minuit_fit(ponly_obj, ("fNL", "b1", "Pshot"),
                        starts=[{"fNL": f0, "b1": B1_FIXED, "Pshot": 0.0} for f0 in (100.0, 0.0, 50.0, 150.0)],
                        limits={"fNL": FNL_BOUNDS, "b1": (0.1, 8), "Pshot": (-1e5, 1e5)},
                        errors={"fNL": 10.0, "b1": 0.15, "Pshot": 1000.0})
    best_p.hesse(); best_p.minos("fNL", ncall=100000)
    err = best_p.merrors["fNL"]
    lo, hi = sorted((float(best_p.values["fNL"]) + err.lower, float(best_p.values["fNL"]) + err.upper))
    print(f"[P-only] fNL={float(best_p.values['fNL']):+.2f} +/- {(hi-lo)/2:.2f} "
          f"b1={float(best_p.values['b1']):.3f} gof={float(chi2_distribution.sf(float(best_p.fval), 9)):.3f}")

    # ---------- 2+3. B-only 与 P+B（B 模板机）----------
    low = _vendor_load("quijote_halo_z0p5_direct_finite_fnl_models", "low.py")
    scan = _vendor_load("quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models", "scan.py")
    low.RECOVERY_RUNNER = VENDOR / "py/recovery.py"
    low.FULL_RUNNER = VENDOR / "py/full.py"
    low.P_MATRIX = VENDOR / "data/PRE_P0_FNL250_MATRIX.npz"
    low.B_MATRIX_ROOT = VENDOR / "data"
    low.B_MATRICES = {"standard": VENDOR / "data/STANDARD_FNL250_MATRIX.npz",
                      "marisa": VENDOR / "data/MARISA_FNL250_MATRIX.npz"}
    low.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p08.npz"
    low.MARISA_FINITE_MANIFEST = low.MARISA_FINITE.with_suffix(".json")
    scan.HIGH_BACKEND = VENDOR / "py/high_backend.py"
    scan.PRE_TREE = VENDOR / "data/pre_tree_kmax0p20.npz"
    scan.PRE_TREE_MANIFEST = scan.PRE_TREE.with_suffix(".json")
    scan.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p20.npz"
    scan.MARISA_FINITE_MANIFEST = scan.MARISA_FINITE.with_suffix(".json")
    sys.path.insert(0, str(VENDOR / "py"))
    low.bphi_from_b1 = lambda b1: 2.0 * low.DELTA_C * (float(b1) - P_FIXED)
    runtime = scan.load_runtime()
    low.B_REC = B1_FIXED
    low.P_REC = P_FIXED
    low.BPHI_REC = 2.0 * low.DELTA_C * (B1_FIXED - P_FIXED)
    model = scan.DirectFiniteScanB000Model(runtime, "marisa")
    context = model.base_context
    prior_mean = {n: float(context.prior.mean[list(context.names).index(n)]) for n in context.names}
    whitener = np.asarray(context.prior.whitener, dtype=np.float64)
    prior_order = list(context.names)

    b_rows, j_rows = [], []
    for cut in scan.CUTS:
        indices = scan.selected_indices(runtime.geometry, cut)
        idx = np.asarray(indices, dtype=np.int64)
        b_target = b_lc.mean(axis=0)[idx]
        # ---- B-only（b1 固定）----
        cov_b = np.cov(b_fid[:, idx], rowvar=False, ddof=1) / VOLUME
        hart_b = (float(b_fid.shape[0]) - idx.size - 2.0) / (float(b_fid.shape[0]) - 1.0)
        chol_b = np.linalg.cholesky(cov_b) / math.sqrt(hart_b)

        def b_obj(v):
            m = dict(zip(B_NAMES, [float(x) for x in v]))
            bm = {"b1": B1_FIXED, "b2": m["b2"], "gamma2": m["gamma2"],
                  "Ashot_residual": m["Ashot_residual"], "Bshot_residual": m["Bshot_residual"]}
            pred = model(m["fNL"], bm, idx)
            w = solve_triangular(chol_b, pred - b_target, lower=True)
            dq = float(w @ w)
            pv = np.asarray([bm.get(n, prior_mean[n]) for n in prior_order])
            wv = whitener @ (pv - np.asarray([prior_mean[n] for n in prior_order]))
            return dq + PRIOR_WEIGHT * float(wv @ wv)

        best_b = minuit_fit(b_obj, B_NAMES,
                            starts=[{"fNL": f0, "b2": prior_mean["b2"], "gamma2": prior_mean["gamma2"],
                                     "Ashot_residual": prior_mean["Ashot_residual"],
                                     "Bshot_residual": prior_mean["Bshot_residual"]} for f0 in (100.0, 0.0, 40.0, 160.0)],
                            limits={"fNL": FNL_BOUNDS, "b2": (-4, 6), "gamma2": (-2, 4),
                                    "Ashot_residual": (-1e7, 1e7), "Bshot_residual": (-1e7, 1e7)},
                            errors={"fNL": 10.0, "b2": 0.08, "gamma2": 0.05,
                                    "Ashot_residual": 1e4, "Bshot_residual": 1e4})
        best_b.hesse(); best_b.minos("fNL", ncall=100000)
        e = best_b.merrors["fNL"]
        blo, bhi = sorted((float(best_b.values["fNL"]) + e.lower, float(best_b.values["fNL"]) + e.upper))
        b_rows.append({"kmax": float(cut), "fNL": float(best_b.values["fNL"]),
                       "sigma": (bhi - blo) / 2, "b2": float(best_b.values["b2"]),
                       "gof": float(chi2_distribution.sf(float(best_b.fval), idx.size - len(B_NAMES)))})
        print(f"[B-only kmax={cut:.2f}] fNL={b_rows[-1]['fNL']:+.2f} +/- {b_rows[-1]['sigma']:.2f} "
              f"b2={b_rows[-1]['b2']:+.3f} gof={b_rows[-1]['gof']:.4f}", flush=True)

        # ---- P+B 联合（b1 自由）----
        jt = np.concatenate([target_p, b_target])
        # P-B 跨协方差不可得（P 块无样本堆叠）：块对角近似（保守，声明于 caliber）
        cov_bb = np.cov(b_fid[:, idx], rowvar=False, ddof=1) / VOLUME
        ntot = 12 + idx.size
        cov_j = np.zeros((ntot, ntot))
        cov_j[:12, :12] = cov_p
        cov_j[12:, 12:] = cov_bb
        hart_j = (float(b_fid.shape[0]) - ntot - 2.0) / (float(b_fid.shape[0]) - 1.0)
        chol_j = np.linalg.cholesky(cov_j + 1e-9 * np.eye(ntot) * np.trace(cov_j) / ntot) / math.sqrt(hart_j)
        JN = ("fNL", "b1", "Pshot", "b2", "gamma2", "Ashot_residual", "Bshot_residual")

        def j_obj(v):
            m = dict(zip(JN, [float(x) for x in v]))
            bphi = 2.0 * DELTA_C * (m["b1"] - P_FIXED)
            pp = (m["b1"] ** 2 * PL + m["fNL"] * 2.0 * m["b1"] * bphi * PM
                  + m["fNL"] ** 2 * bphi ** 2 * PQ + m["Pshot"])
            bm = {"b1": m["b1"], "b2": m["b2"], "gamma2": m["gamma2"],
                  "Ashot_residual": m["Ashot_residual"], "Bshot_residual": m["Bshot_residual"]}
            pred = np.concatenate([pp, model(m["fNL"], bm, idx)])
            w = solve_triangular(chol_j, pred - jt, lower=True)
            dq = float(w @ w)
            pv = np.asarray([bm.get(n, prior_mean[n]) for n in prior_order])
            wv = whitener @ (pv - np.asarray([prior_mean[n] for n in prior_order]))
            return dq + PRIOR_WEIGHT * float(wv @ wv)

        best_j = minuit_fit(j_obj, JN,
                            starts=[{"fNL": f0, "b1": B1_FIXED, "Pshot": 0.0,
                                     "b2": prior_mean["b2"], "gamma2": prior_mean["gamma2"],
                                     "Ashot_residual": prior_mean["Ashot_residual"],
                                     "Bshot_residual": prior_mean["Bshot_residual"]} for f0 in (100.0, 0.0, 40.0, 160.0)],
                            limits={"fNL": FNL_BOUNDS, "b1": (0.1, 8), "Pshot": (-1e5, 1e5),
                                    "b2": (-4, 6), "gamma2": (-2, 4),
                                    "Ashot_residual": (-1e7, 1e7), "Bshot_residual": (-1e7, 1e7)},
                            errors={"fNL": 10.0, "b1": 0.15, "Pshot": 1000.0, "b2": 0.08,
                                    "gamma2": 0.05, "Ashot_residual": 1e4, "Bshot_residual": 1e4})
        best_j.hesse(); best_j.minos("fNL", ncall=100000)
        e = best_j.merrors["fNL"]
        jlo, jhi = sorted((float(best_j.values["fNL"]) + e.lower, float(best_j.values["fNL"]) + e.upper))
        j_rows.append({"kmax": float(cut), "fNL": float(best_j.values["fNL"]),
                       "sigma": (jhi - jlo) / 2, "b1": float(best_j.values["b1"]),
                       "gof": float(chi2_distribution.sf(float(best_j.fval), ntot - len(JN)))})
        print(f"[P+B    kmax={cut:.2f}] fNL={j_rows[-1]['fNL']:+.2f} +/- {j_rows[-1]['sigma']:.2f} "
              f"b1={j_rows[-1]['b1']:.3f} gof={j_rows[-1]['gof']:.4f}", flush=True)

    payload = {
        "schema": "marisa-only-v2-paware-trilogy",
        "created_utc": utc_now(),
        "caliber": ("fully self-produced v2 data (png_p=1.1217, bias=1.9364; fid500+LCp100). "
                    "P block from five-method p0 matrix (paperbin 12, symmetric with sto_marisa line); "
                    "B block self-produced v2; joint P+B uses block-diagonal covariance approximation "
                    "(P-B cross-covariance unavailable, conservative)."),
        "freep_recal": {"p": p_recal, "p_err": float(best.errors["p"]),
                        "b1": float(best.values["b1"]), "input_p": P_FIXED},
        "p_only": {"fNL": float(best_p.values["fNL"]), "sigma": (hi - lo) / 2,
                   "b1": float(best_p.values["b1"]), "interval": [lo, hi]},
        "b_only_rows": b_rows, "joint_rows": j_rows,
    }
    out = ANALYSIS / "MARISA_V2_TRILOGY.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    print(f"runner sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
