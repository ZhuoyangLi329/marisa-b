#!/usr/bin/env python3
"""Phase 3：sto_marisa v4 扫描（p-aware 重产数据 + 两端一致参数）。

数据：
- target = v2_p1282 重产的 LC_p 100 套 post_B000（png_p=1.2822 生产）
- cov = 旧 sto_marisa fid500 post_B000（fnl_loc=0 不受 p 影响，干净）
模型（相对 v3 的变化）：
- adaptive 系数 (b_rec, p_rec) = (2.2281, 1.2822) —— 新生产实录，两端一致
- halo 端 p = 1.2822（暂用 freep 值；重校准后有移动可再跑）
- b1 固定 2.2281
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

OLD_CODES = Path("/pscratch/sd/l/lzy/reconstruction-png/codes")
VENDOR = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/vendor")
HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
NEW_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png/"
                "result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_"
                "sto_marisa_b000_pk_v2_p1282/full/LC_p")
FID_MATRIX = VENDOR / "data/sto_marisa_matrix.npz"  # fid cov（fnl_loc=0 干净）
ANALYSIS = HERE / "analysis"
VOLUME = 42.875
B1_STO = 2.2280546353635335
P_HALO = 1.2822
P_REC = 1.2822
B_REC = 2.2280546353635335
FNL_BOUNDS = (-300.0, 300.0)
PRIOR_WEIGHT = 1.0 / 25.0
NAMES = ("fNL", "b2", "gamma2", "Ashot_residual", "Bshot_residual")


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


def aggregate_new_lcp() -> np.ndarray:
    """堆叠重产 LC_p 100 套 post_B000（h5 的 0_0_0/value）。"""
    import h5py
    rows = []
    for real_dir in sorted(NEW_ROOT.glob("real*")):
        path = real_dir / "post_b000.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            value = np.asarray(f["0_0_0/value"][()], dtype=np.float64)
        if value.shape != (120,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"bad B vector: {path}")
        rows.append(value)
    if len(rows) < 95:
        raise RuntimeError(f"insufficient new LC_p samples: {len(rows)}")
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    lcp = aggregate_new_lcp()
    print(f"new LC_p target: {lcp.shape[0]} sets")
    with np.load(FID_MATRIX, allow_pickle=False) as z:
        fid = np.asarray(z["fiducial_post_B000"], dtype=np.float64)
    print(f"fid cov samples: {fid.shape[0]}")

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

    low.bphi_from_b1 = lambda b1: 2.0 * low.DELTA_C * (float(b1) - P_HALO)
    runtime = scan.load_runtime()
    # 两端一致的重构参数（= v2_p1282 生产实录）
    low.B_REC = B_REC
    low.P_REC = P_REC
    low.BPHI_REC = 2.0 * low.DELTA_C * (low.B_REC - low.P_REC)
    print(f"params: b1={B1_STO:.4f} p_halo={P_HALO} (b_rec,p_rec)=({B_REC:.4f},{P_REC}) bphi_rec={low.BPHI_REC:.4f}")

    model = scan.DirectFiniteScanB000Model(runtime, "marisa")
    context = model.base_context
    prior_mean = {n: float(context.prior.mean[list(context.names).index(n)]) for n in context.names}
    whitener = np.asarray(context.prior.whitener, dtype=np.float64)
    prior_order = list(context.names)
    target_all = lcp.mean(axis=0)

    rows = []
    for cut in scan.CUTS:
        indices = scan.selected_indices(runtime.geometry, cut)
        idx = np.asarray(indices, dtype=np.int64)
        ndata = idx.size
        target = target_all[idx]
        cov_fit = np.cov(fid[:, idx], rowvar=False, ddof=1) / VOLUME
        hartlap = (float(fid.shape[0]) - ndata - 2.0) / (float(fid.shape[0]) - 1.0)
        chol = np.linalg.cholesky(cov_fit) / math.sqrt(hartlap)

        def objective_impl(values):
            m = dict(zip(NAMES, [float(v) for v in values]))
            m["b1"] = B1_STO
            pred = model(m["fNL"], m, idx)
            white = solve_triangular(chol, pred - target, lower=True)
            data_q = float(white @ white)
            pv = np.asarray([m[n] for n in prior_order], dtype=np.float64)
            w = whitener @ (pv - np.asarray([prior_mean[n] for n in prior_order]))
            return data_q + PRIOR_WEIGHT * float(w @ w)

        _sig = ", ".join(NAMES)
        _body = ("def _objective(" + _sig + "):\n"
                 "    return _impl((" + ", ".join(NAMES) + ",))\n")
        _ns = {"_impl": objective_impl}
        exec(_body, _ns)
        objective = _ns["_objective"]

        fits = []
        for f0 in (100.0, 0.0, 40.0, 160.0):
            start = {"fNL": f0, "b2": prior_mean["b2"], "gamma2": prior_mean["gamma2"],
                     "Ashot_residual": prior_mean["Ashot_residual"],
                     "Bshot_residual": prior_mean["Bshot_residual"]}
            fit = Minuit(objective, **start)
            fit.errordef = Minuit.LEAST_SQUARES
            fit.limits["fNL"] = FNL_BOUNDS
            fit.limits["b2"] = (-4.0, 6.0)
            fit.limits["gamma2"] = (-2.0, 4.0)
            sa = abs(start["Ashot_residual"]) + 1.0e6
            sb = abs(start["Bshot_residual"]) + 1.0e6
            fit.limits["Ashot_residual"] = (-10.0 * sa, 10.0 * sa)
            fit.limits["Bshot_residual"] = (-10.0 * sb, 10.0 * sb)
            fit.errors["fNL"] = 10.0
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
        m["b1"] = B1_STO
        pred = model(m["fNL"], m, idx)
        white = solve_triangular(chol, pred - target, lower=True)
        data_q = float(white @ white)
        rows.append({
            "kmax_h_mpc": float(cut), "ndata": ndata,
            "b1_fixed": B1_STO, "p_halo": P_HALO, "p_rec": P_REC, "b_rec": B_REC,
            "map_fNL": m["fNL"],
            "minos_fNL": {"lower": float(err.lower), "upper": float(err.upper),
                          "valid": bool(err.is_valid)},
            "map": {k: m[k] for k in NAMES},
            "data_chi2": data_q, "dof": ndata - len(NAMES),
            "gof_p_value_data_only": float(chi2_distribution.sf(data_q, max(ndata - len(NAMES), 1))),
            "hartlap": hartlap, "n_target": int(lcp.shape[0]), "n_fid": int(fid.shape[0]),
        })
        lo, hi = sorted((m["fNL"] + err.lower, m["fNL"] + err.upper))
        print(f"[v4 kmax={cut:.2f}] fNL={m['fNL']:+8.2f} [{lo:+.1f},{hi:+.1f}] "
              f"b2={m['b2']:+.3f} gof={rows[-1]['gof_p_value_data_only']:.4f}", flush=True)

    payload = {
        "schema": "sto-marisa-b000-v4-paware-reproduction-scan",
        "created_utc": utc_now(),
        "data": {"target": "v2_p1282 LC_p post_B000 (png_p=1.2822 production) 100-real mean",
                 "cov": "old sto_marisa fid500 post_B000 (fnl_loc=0 clean) / 42.875 / Hartlap"},
        "model": "DirectFiniteScanB000Model(marisa); halo p=1.2822; adaptive (b_rec,p_rec)=(2.2281,1.2822) = production",
        "rows": rows,
    }
    out = ANALYSIS / "STO_MARISA_B_V4_PAWARE_REPRO_SCAN.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    print(f"runner sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
