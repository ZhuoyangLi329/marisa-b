#!/usr/bin/env python3
"""P(k)+B(k) 联合拟合：pre vs sto_marisa 对比（fNL 共享，CUTS 0.08-0.20）。

口径声明（主人"请注意"事项的落实）：
- P 块与 B 块逐 realization 同批：pre 用 FNL250 系列（PRE_P0_FNL250 的
  pre_P0 12 bins + STANDARD_FNL250 的 pre_B000 120 bins，real0-249/0-499 对齐）；
  sto_marisa 的 P 用旧 sto_marisa 矩阵 post_P0 前 12 bins（paperbin，与 pre
  的 12 bins 几何一致）；B 用 v4 p-aware 重产（png_p=1.2822）post_b000。
  P/B 均按 real000-099/0-249 对齐拼接（同 sim 编号）。
- halo p 全局每方法一个：pre=1.179（P0 freep 实测），sto_marisa=1.2822。
  B 端 adaptive 系数：sm 用生产实录 (2.2281, 1.2822)；pre 模板无 adaptive。
- b1 固定：pre=1.93，sm=2.2281。
- 参数 {fNL, Pshot, b2, gamma2, Ashot, Bshot}；弱先验 ×1/25；
  cov = fid500 拼接样本协方差 / 42.875 / Hartlap。
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
LATTICE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz")
NEW_B_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png/"
                  "result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_"
                  "sto_marisa_b000_pk_v2_p1282/full/LC_p")
VOLUME = 42.875
FNL_BOUNDS = (-300.0, 300.0)
PRIOR_WEIGHT = 1.0 / 25.0
NAMES = ("fNL", "b1", "Pshot", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
CASES = {
    "pre": {"b1": 1.93, "p": 1.179, "method": "pre"},
    "sto_marisa": {"b1": 2.2280546353635335, "p": 1.2822, "method": "marisa",
                   "b_rec": 2.2280546353635335, "p_rec": 1.2822},
}


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


def load_new_b() -> np.ndarray:
    import h5py
    rows = []
    for real_dir in sorted(NEW_B_ROOT.glob("real*")):
        path = real_dir / "post_b000.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            rows.append(np.asarray(f["0_0_0/value"][()], dtype=np.float64))
    if len(rows) < 95:
        raise RuntimeError(f"insufficient new B: {len(rows)}")
    return np.asarray(rows)


def p_theory_12():
    """lattice 模板插到 12-bin paperbin k（P0 线同款）。"""
    with np.load(LATTICE, allow_pickle=False) as z:
        k_old = np.asarray(z["pk_k"], dtype=float)
        PL = np.asarray(z["PL_lattice"], dtype=float)
        PM = np.asarray(z["PL_over_M_lattice"], dtype=float)
        Pphi = np.asarray(z["Pphi_lattice"], dtype=float)
    with np.load(VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as z:
        k_new = np.asarray(z["pk_k"], dtype=float)

    def interp(arr):
        return np.exp(np.interp(np.log(k_new), np.log(k_old), np.log(np.maximum(arr, 1e-300))))
    return k_new, interp(PL), interp(PM), interp(Pphi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

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

    k12, PL, PM, Pphi = p_theory_12()
    runtime = scan.load_runtime()

    with np.load(VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as z:
        pre_p_fid = np.asarray(z["fiducial_pre_P0"], dtype=np.float64)      # (500,12)
        pre_p_lc = np.asarray(z["LC_p_pre_P0"], dtype=np.float64)           # (250,12)
    with np.load(VENDOR / "data/STANDARD_FNL250_MATRIX.npz", allow_pickle=False) as z:
        pre_b_fid = np.asarray(z["fiducial_pre_B000"], dtype=np.float64)    # (500,120)
        pre_b_lc = np.asarray(z["LC_p_pre_B000"], dtype=np.float64)         # (250,120)
    with np.load(VENDOR / "data/sto_marisa_matrix.npz", allow_pickle=False) as z:
        sm_p_fid = np.asarray(z["fiducial_post_P0"], dtype=np.float64)[:, :12]   # (500,12)
        sm_p_lc = np.asarray(z["LC_p_post_P0"], dtype=np.float64)[:, :12]        # (100,12)
        sm_b_fid = np.asarray(z["fiducial_post_B000"], dtype=np.float64)         # (500,120)
    sm_b_lc = load_new_b()   # (100,120) v4 重产

    print(f"data: pre P{pre_p_lc.shape} B{pre_b_lc.shape} | sm P{sm_p_lc.shape} B{sm_b_lc.shape}")

    all_rows = {}
    for case_name, cfg in CASES.items():
        low.bphi_from_b1 = lambda b1, p=cfg["p"]: 2.0 * low.DELTA_C * (float(b1) - p)
        if case_name == "sto_marisa":
            low.B_REC = cfg["b_rec"]
            low.P_REC = cfg["p_rec"]
            low.BPHI_REC = 2.0 * low.DELTA_C * (low.B_REC - low.P_REC)
            low.bphi_from_b1 = lambda b1, p=cfg["p"]: 2.0 * low.DELTA_C * (float(b1) - p)
        model = scan.DirectFiniteScanB000Model(runtime, cfg["method"])
        context = model.base_context
        prior_mean = {n: float(context.prior.mean[list(context.names).index(n)]) for n in context.names}
        whitener = np.asarray(context.prior.whitener, dtype=np.float64)
        prior_order = list(context.names)
        b1 = cfg["b1"]
        p_halo = cfg["p"]
        bphi = 2.0 * low.DELTA_C * (b1 - p_halo)
        p_lin = b1 ** 2 * PL
        p_png = 2.0 * b1 * bphi * PM
        p_q = bphi ** 2 * Pphi

        if case_name == "pre":
            p_fid, p_tg = pre_p_fid, pre_p_lc.mean(axis=0)
        else:
            p_fid, p_tg = sm_p_fid, sm_p_lc.mean(axis=0)

        rows = []
        for cut in scan.CUTS:
            indices = scan.selected_indices(runtime.geometry, cut)
            idx = np.asarray(indices, dtype=np.int64)
            b_fid = pre_b_fid if case_name == "pre" else sm_b_fid
            b_tg_all = pre_b_lc.mean(axis=0) if case_name == "pre" else sm_b_lc.mean(axis=0)
            target = np.concatenate([p_tg, b_tg_all[idx]])
            fid_cols = np.concatenate([p_fid, b_fid[:, idx]], axis=1)
            nfid, ndata = fid_cols.shape
            hartlap = (float(nfid) - ndata - 2.0) / (float(nfid) - 1.0)
            cov_fit = np.cov(fid_cols, rowvar=False, ddof=1) / VOLUME
            chol = np.linalg.cholesky(cov_fit) / math.sqrt(hartlap)

            def objective_impl(values):
                m = dict(zip(NAMES, [float(v) for v in values]))
                fb1 = m["b1"]
                fbphi = 2.0 * low.DELTA_C * (fb1 - p_halo)
                p_pred = (fb1 ** 2 * PL + m["fNL"] * 2.0 * fb1 * fbphi * PM
                          + m["fNL"] ** 2 * fbphi ** 2 * Pphi + m["Pshot"])
                bm = {"b1": fb1, "b2": m["b2"], "gamma2": m["gamma2"],
                      "Ashot_residual": m["Ashot_residual"],
                      "Bshot_residual": m["Bshot_residual"]}
                b_pred = model(m["fNL"], bm, idx)
                pred = np.concatenate([p_pred, b_pred])
                white = solve_triangular(chol, pred - target, lower=True)
                data_q = float(white @ white)
                pv = np.asarray([bm.get(n, 0.0) for n in prior_order], dtype=np.float64)
                base = np.asarray([prior_mean[n] for n in prior_order], dtype=np.float64)
                pv = np.where(np.isin(prior_order, list(bm.keys())), pv, base)
                w = whitener @ (pv - base)
                return data_q + PRIOR_WEIGHT * float(w @ w)

            _sig = ", ".join(NAMES)
            _body = ("def _objective(" + _sig + "):\n"
                     "    return _impl((" + ", ".join(NAMES) + ",))\n")
            _ns = {"_impl": objective_impl}
            exec(_body, _ns)
            objective = _ns["_objective"]

            fits = []
            for f0 in (100.0, 0.0, 40.0, 160.0):
                start = {"fNL": f0, "b1": b1, "Pshot": 0.0,
                         "b2": prior_mean["b2"], "gamma2": prior_mean["gamma2"],
                         "Ashot_residual": prior_mean["Ashot_residual"],
                         "Bshot_residual": prior_mean["Bshot_residual"]}
                fit = Minuit(objective, **start)
                fit.errordef = Minuit.LEAST_SQUARES
                fit.limits["fNL"] = FNL_BOUNDS
                fit.limits["b1"] = (0.1, 8.0)
                fit.limits["Pshot"] = (-1.0e5, 1.0e5)
                fit.limits["b2"] = (-4.0, 6.0)
                fit.limits["gamma2"] = (-2.0, 4.0)
                sa = abs(start["Ashot_residual"]) + 1.0e6
                sb = abs(start["Bshot_residual"]) + 1.0e6
                fit.limits["Ashot_residual"] = (-10.0 * sa, 10.0 * sa)
                fit.limits["Bshot_residual"] = (-10.0 * sb, 10.0 * sb)
                fit.errors["fNL"] = 10.0
                fit.errors["b1"] = 0.15
                fit.errors["Pshot"] = 1000.0
                fit.errors["b2"] = 0.08
                fit.errors["gamma2"] = 0.05
                fit.strategy, fit.tol = 1, 1e-6
                fit.migrad(ncall=200000, iterate=8, use_simplex=True)
                fits.append(fit)
            valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
            best = min(valid or fits, key=lambda f: float(f.fval))
            best.hesse()
            best.minos("fNL", ncall=100000)
            err = best.merrors["fNL"]
            m = {n: float(best.values[n]) for n in NAMES}
            fbphi = 2.0 * low.DELTA_C * (m["b1"] - p_halo)
            p_pred = (m["b1"] ** 2 * PL + m["fNL"] * 2.0 * m["b1"] * fbphi * PM
                      + m["fNL"] ** 2 * fbphi ** 2 * Pphi + m["Pshot"])
            bm = {"b1": m["b1"], "b2": m["b2"], "gamma2": m["gamma2"],
                  "Ashot_residual": m["Ashot_residual"], "Bshot_residual": m["Bshot_residual"]}
            b_pred = model(m["fNL"], bm, idx)
            pred = np.concatenate([p_pred, b_pred])
            white = solve_triangular(chol, pred - target, lower=True)
            data_q = float(white @ white)
            rows.append({
                "case": case_name, "kmax_h_mpc": float(cut), "ndata": ndata, "n_fid": nfid,
                "map_fNL": m["fNL"],
                "minos_fNL": {"lower": float(err.lower), "upper": float(err.upper),
                              "valid": bool(err.is_valid)},
                "map": m,
                "data_chi2": data_q, "dof": ndata - len(NAMES),
                "gof_p_value_data_only": float(chi2_distribution.sf(data_q, max(ndata - len(NAMES), 1))),
                "hartlap": hartlap,
            })
            lo, hi = sorted((m["fNL"] + err.lower, m["fNL"] + err.upper))
            print(f"[{case_name:<11} kmax={cut:.2f}] fNL={m['fNL']:+8.2f} [{lo:+.1f},{hi:+.1f}] "
                  f"b1={m['b1']:.3f} gof={rows[-1]['gof_p_value_data_only']:.4f}", flush=True)
        all_rows[case_name] = rows

    payload = {
        "schema": "joint-pb-pre-vs-stomarisa-v2-freeb1",
        "created_utc": utc_now(),
        "caliber": ("P block: 12-bin paperbin k<=0.08 (pre: FNL250 pre_P0; sm: old matrix post_P0). "
                    "B block: pre FNL250 pre_B000; sm = v4 p-aware reproduction (png_p=1.2822). "
                    "Realization-aligned concatenation; halo p unified per method "
                    "(pre=1.179, sm=1.2822); b1 fixed (1.93 / 2.2281)."),
        "cases": {k: {"b1": v["b1"], "p": v["p"]} for k, v in CASES.items()},
        "rows": {k: v for k, v in all_rows.items()},
    }
    out = ANALYSIS / "JOINT_PB_PRE_VS_STOMARISA_V2_FREEB1.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    print(f"runner sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
