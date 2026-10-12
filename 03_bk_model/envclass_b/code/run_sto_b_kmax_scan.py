#!/usr/bin/env python3
"""sto_only B000 固定 b1 的 kmax 无偏性扫描（主人 2026-09-18 提议）。

科学背景：STO B 当年的 joint 拟合是 b1 自由（失败：b2≈−1.5、响应放大）；
环境类实验证明固定 b1 大幅改善。本实验对全样本 sto_only 场固定
b1_sto=2.197952896776342（fid500 auto-P fit 冻结值）重做 kmax 扫描，
p 固定 P0 校准值 p_sto（fnlfixed100 freep 校准）。对照：pre 固定 b1=1.93
到 kmax 0.15 干净（旧 validity scan）。

模型：DirectFiniteScanB000Model("standard")（sto 是重构场），
bφ 全局 patch 为 2·δc·(b1 − p_sto)。
数据：sto_only_matrix post_B000：target = LC_p 100 套均值（truth+100），
cov = fid 500 套 / 42.875 / Hartlap。
参数：(fNL, b2, γ2, Ashot, Bshot)，弱先验 ×1/25，iminuit 多起点 + minos。
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


def _import_old(name):
    import types
    path = OLD_CODES / f"{name}.py"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(open(path, encoding="utf-8").read(), str(path), "exec"), module.__dict__)
    return module


VENDOR = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/vendor")


def _vendor_load(name, filename):
    path = VENDOR / "py" / filename
    import types
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(open(path, encoding="utf-8").read(), str(path), "exec"), module.__dict__)
    return module


low = _vendor_load("quijote_halo_z0p5_direct_finite_fnl_models", "low.py")
scan = _vendor_load("quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models", "scan.py")

low.RECOVERY_RUNNER = VENDOR / "py/recovery.py"
low.FULL_RUNNER = VENDOR / "py/full.py"
low.P_MATRIX = VENDOR / "data/PRE_P0_FNL250_MATRIX.npz"
low.B_MATRIX_ROOT = VENDOR / "data"
low.B_MATRICES = {
    "standard": VENDOR / "data/STANDARD_FNL250_MATRIX.npz",
    "marisa": VENDOR / "data/MARISA_FNL250_MATRIX.npz",
}
low.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p08.npz"
low.MARISA_FINITE_MANIFEST = low.MARISA_FINITE.with_suffix(".json")
scan.HIGH_BACKEND = VENDOR / "py/high_backend.py"
scan.PRE_TREE = VENDOR / "data/pre_tree_kmax0p20.npz"
scan.PRE_TREE_MANIFEST = scan.PRE_TREE.with_suffix(".json")
scan.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p20.npz"
scan.MARISA_FINITE_MANIFEST = scan.MARISA_FINITE.with_suffix(".json")
sys.path.insert(0, str(VENDOR / "py"))

HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
ANALYSIS = HERE / "analysis"
STO_MATRIX = VENDOR / "data/sto_only_matrix.npz"
FREEP = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work"
             "/quijote_halo_z0p5_pre_sto_only_fnlfixed100_freep_calib_v1/FREEP_PROFILES.json")
VOLUME = 42.875
B1_STO = 2.197952896776342  # fid500 auto-P fit 冻结值
P_STO_FALLBACK = 1.339
FNL_BOUNDS = (-300.0, 300.0)
PRIOR_WEIGHT = 1.0 / 25.0
NAMES = ("fNL", "b2", "gamma2", "Ashot_residual", "Bshot_residual")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_p_sto() -> float:
    try:
        d = json.loads(FREEP.read_text(encoding="utf-8"))
        text = json.dumps(d)
        import re
        hits = re.findall(r'"p_sto[^"]*":\s*{[^}]*"median":\s*([0-9.eE+-]+)', text)
        if not hits:
            hits = re.findall(r'"sto[^"]*p[^"]*":\s*{[^}]*"median":\s*([0-9.eE+-]+)', text)
        if hits:
            return float(hits[0])
    except Exception:
        pass
    return P_STO_FALLBACK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    p_sto = load_p_sto()
    print(f"b1_sto fixed = {B1_STO}  p_sto fixed = {p_sto}")

    # bφ 全局 patch：pre/standard 路径全部吃到 p_sto
    low.bphi_from_b1 = lambda b1: 2.0 * low.DELTA_C * (float(b1) - p_sto)

    runtime = scan.load_runtime()
    model = scan.DirectFiniteScanB000Model(runtime, "standard")
    context = model.base_context
    prior_mean = {n: float(context.prior.mean[list(context.names).index(n)]) for n in context.names}
    whitener = np.asarray(context.prior.whitener, dtype=np.float64)
    prior_order = list(context.names)

    with np.load(STO_MATRIX, allow_pickle=False) as z:
        fid = np.asarray(z["fiducial_post_B000"], dtype=np.float64)   # (500,120)
        lcp = np.asarray(z["LC_p_post_B000"], dtype=np.float64)      # (100,120)
    print(f"data: fid {fid.shape}, LC_p {lcp.shape}")
    target_all = lcp.mean(axis=0)

    rows = []
    for cut in scan.CUTS:
        indices = scan.selected_indices(runtime.geometry, cut)
        idx = np.asarray(indices, dtype=np.int64)
        ndata = idx.size
        target = target_all[idx]
        cov_fit = np.cov(fid[:, idx], rowvar=False, ddof=1) / VOLUME
        hartlap = (500.0 - ndata - 2.0) / 499.0
        chol = np.linalg.cholesky(cov_fit) / math.sqrt(hartlap)

        def mapping_from(values):
            m = dict(zip(NAMES, [float(v) for v in values]))
            m["b1"] = B1_STO
            return m

        def objective_impl(values):
            m = mapping_from(values)
            pred = model(m["fNL"], m, idx)
            white = solve_triangular(chol, pred - target, lower=True)
            data_q = float(white @ white)
            pv = np.asarray([m[n] for n in prior_order], dtype=np.float64)
            w = whitener @ (pv - np.asarray([prior_mean[n] for n in prior_order]))
            return data_q + PRIOR_WEIGHT * float(w @ w)

        # iminuit 需要具名签名；exec 生成（*args 形式 iminuit 不接受）
        _sig = ", ".join(NAMES)
        _body = (
            "def _objective(" + _sig + "):\n"
            "    return _impl((" + ", ".join(NAMES) + ",))\n"
        )
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
        m = mapping_from([best.values[n] for n in NAMES])
        pred = model(m["fNL"], m, idx)
        white = solve_triangular(chol, pred - target, lower=True)
        data_q = float(white @ white)
        row = {
            "method": "sto_only_fixed_b1", "kmax_h_mpc": float(cut),
            "b1_fixed": B1_STO, "p_fixed": p_sto,
            "ndata": ndata, "hartlap": hartlap,
            "map_fNL": float(m["fNL"]),
            "minos_fNL": {"lower": float(err.lower), "upper": float(err.upper),
                          "valid": bool(err.is_valid)},
            "map": {k: float(m[k]) for k in NAMES},
            "data_chi2": data_q, "dof": ndata - len(NAMES),
            "gof_p_value_data_only": float(chi2_distribution.sf(data_q, max(ndata - len(NAMES), 1))),
            "prior_weight": PRIOR_WEIGHT,
        }
        rows.append(row)
        print(f"[sto_only b1={B1_STO:.3f} kmax={cut:.2f}] fNL={row['map_fNL']:+8.2f} "
              f"[{row['map_fNL']+err.lower:+.1f},{row['map_fNL']+err.upper:+.1f}] "
              f"b2={row['map']['b2']:+.3f} gof={row['gof_p_value_data_only']:.4f}", flush=True)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sto-only-b000-fixedb1-kmax-scan-v1",
        "created_utc": utc_now(),
        "model": "DirectFiniteScanB000Model(standard) with global bphi=2*dc*(b1-p_sto); b1 fixed",
        "b1_fixed": B1_STO, "p_fixed": p_sto,
        "data": {"target": "sto_only LC_p post_B000 100-real mean (truth +100)",
                 "cov": "fid 500-real sample / 42.875 / Hartlap"},
        "rows": rows,
    }
    out = ANALYSIS / "STO_B_FIXEDB1_KMAX_SCAN.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    print(f"runner sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
