#!/usr/bin/env python3
"""环境类 B000 的 kmax 无偏性扫描（f / s / f+s 三组合 × 7 档 kmax × b1 两版）。

科学问题（主人 2026-09-18 指示）：按环境拆开 halo 后，单类 tree-level
建模能否推到比全样本更高的 kmax 仍无偏；b2 是否恢复正常（对照 STO
混合体的 −1.5）。

模型
====
复用现役 DirectFiniteScanB000Model（tree bases tracer 无关预计算量）：
  B(fNL, θ) = gaussian(θ) + fNL·linear(θ) + fNL²·quadratic(θ)
环境类唯一改动：PNG 响应的 bφ 注入 2·δc·(b1_c − p_c)，p_c 固定 P0
校准值（ENVCLASS_CALIBRATION.json 各类 p median；全样本默认 1.12）。
gaussian 部分完全复用 certified v0p9 上下文组装。

数据
====
work/{cosmo}/realNNN/{cls}_b000.npz（本目录 run_envclass_b000.py 生产，
120 三角形与 baseline 矩阵几何门对齐）。target = LC_p 100 套平均
（fNL=+100 注入恢复检验）；协方差 = fid 200 套样本协方差 / 42.875，
Hartlap=(200−ndata−2)/199 经 cholesky 缩放（validity scan 同款惯例）。

拟合
====
参数 (fNL, 每类 b2, γ2, Ashot, Bshot[, b1])；b1 两版（自由 / 固定 P0
校准值）；nuisance 带弱高斯先验（v0p9 context prior × 1/25 放宽）。
iminuit 多起点 + minos(fNL)。先验放宽在结果 json 中如实记录。
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

OLD_CODES = Path("/pscratch/sd/lzy/reconstruction-png/codes")


def _import_old(name):
    """显式加载旧仓库模块（spec_from_file_location，绕过 sys.path 玄学）。"""
    import types
    path = OLD_CODES / f"{name}.py"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    if not path.is_file():
        raise ImportError(path)
    sys.modules[name] = module
    source = open(path, "r", encoding="utf-8").read()  # 手动读（open 稳定成功），绕过 importlib get_data
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module

# ---- v2 vendor 加载（旧仓库元数据分钟级抖动，依赖已整体拷贝）----
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

# 路径常量全部指向 vendor 副本（load_runtime 前必须完成）
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
# vendor/py 进 sys.path：recovery/full 内部 import 从稳定子树解析
sys.path.insert(0, str(VENDOR / "py"))

HERE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
WORK = HERE / "work"
CALIB = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk"
             "/envclass_p0only/calibration/ENVCLASS_CALIBRATION.json")
ANALYSIS = HERE / "analysis"
VOLUME = 42.875
N_FID = 200
CUTS = scan.CUTS
FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 8.0)
PRIOR_WEIGHT = 1.0 / 25.0  # nuisance 先验放宽 25 倍（防跑飞，弱先验）
CLASS_DIR = {"filament": "filament", "sheet": "sheet"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class EnvScanModel(scan.DirectFiniteScanB000Model):
    """环境类版 pre 模型：bφ = 2·δc·(b1 − p_c)，p_c 固定 P0 校准值。"""

    def __init__(self, runtime, p_c: float):
        super().__init__(runtime, "pre")
        self.p_c = float(p_c)

    def _pre_coefficients(self, parameters):
        closure = self.runtime.low_runtime.pre_closure
        b1 = float(parameters["b1"])
        response = closure["v0p9"].pre_png.response_components(
            tree=self.runtime.pre_tree,
            templates=closure["response_templates"],
            parameters=parameters,
            b1=b1,
            p=1.0,
            nbar=float(closure["nbar"]),
            dm_tree=np.zeros(120, dtype=np.float64),
            dm_total=np.zeros(120, dtype=np.float64),
            dm_quadratic=np.zeros(120, dtype=np.float64),
            response_model="finite_halo_tree",
            bphi_override=2.0 * low.DELTA_C * (b1 - self.p_c),  # 环境类 p 注入点
            preserve_universal_bphidelta=True,
        )
        linear = sum(
            (low.full_vector(response, name) for name in low.PRE_LINEAR_KEYS),
            np.zeros(120, dtype=np.float64),
        )
        quadratic = low.full_vector(response, "halo_quadratic_deterministic") + low.full_vector(
            response, "halo_quadratic_stochastic")
        return linear, quadratic


def load_env_samples(cls: str, cosmo: str) -> tuple[np.ndarray, list[int]]:
    """堆叠 (N, 120) 环境类 B000 样本，同时返回 realization 列表（跨类交集用）。"""
    rows, reals = [], []
    for real_dir in sorted((WORK / cosmo).glob("real*")):
        path = real_dir / f"{CLASS_DIR[cls]}_b000.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            b = np.asarray(z["B"], dtype=np.float64)
        if b.shape != (120,) or not np.all(np.isfinite(b)):
            raise RuntimeError(f"bad B vector: {path}")
        rows.append(b)
        reals.append(int(real_dir.name.replace("real", "")))
    if not rows:
        raise RuntimeError(f"no samples for {cls}/{cosmo}")
    return np.asarray(rows), reals


def align_common(samples: dict, reals: dict, classes) -> dict:
    """把多类样本裁剪到共同 realization 交集（返回每类裁剪后数组）。"""
    common = set(reals[classes[0]])
    for c in classes[1:]:
        common &= set(reals[c])
    common = sorted(common)
    out = {}
    for c in classes:
        pick = {r: i for i, r in enumerate(reals[c])}
        out[c] = samples[c][[pick[r] for r in common]]
    return out


def prior_terms(model: EnvScanModel):
    """从 v0p9 上下文取 nuisance 先验（mean 与白化器，names 顺序）。"""
    context = model.base_context
    names = list(context.names)  # (b1, b2, gamma2, Ashot_residual, Bshot_residual)
    mean = np.asarray(context.prior.mean, dtype=np.float64)
    whitener = np.asarray(context.prior.whitener, dtype=np.float64)
    return names, mean, whitener


def build_case(combo, cut, samples_lc, samples_fid, models, b1_mode, b1_calib, priors):
    """构造一个 (combo, cut, b1_mode) 的似然并做 iminuit 多起点拟合。"""
    indices = scan.selected_indices(models[combo[0]].runtime.geometry, cut)
    idx = np.asarray(indices, dtype=np.int64)
    n_idx = idx.size

    # 数据向量：target = LC_p 均值；协方差 = fid 样本堆叠（块拼接，全交叉）
    target = np.concatenate([samples_lc[c].mean(axis=0)[idx] for c in combo])
    fid_cols = np.concatenate([samples_fid[c][:, idx] for c in combo], axis=1)
    nfid = fid_cols.shape[0]
    ndata = fid_cols.shape[1]
    if nfid <= ndata + 4:
        raise RuntimeError(f"cov dof insufficient: nfid={nfid} ndata={ndata}")
    hartlap = (float(nfid) - ndata - 2.0) / (float(nfid) - 1.0)
    cov_fit = np.cov(fid_cols, rowvar=False, ddof=1) / VOLUME
    chol = np.linalg.cholesky(cov_fit) / math.sqrt(hartlap)

    # 参数表：fNL + 每类 nuisance（b1 自由版加 b1_c）
    par_names = ["fNL"]
    for c in combo:
        par_names += [f"b2_{c}", f"gamma2_{c}", f"Ashot_{c}", f"Bshot_{c}"]
        if b1_mode == "free":
            par_names += [f"b1_{c}"]

    # 每类 nuisance mapping 构造 + 弱先验
    def class_mapping(c, local):
        mapping = {
            "b1": (local[f"b1_{c}"] if b1_mode == "free" else float(b1_calib[c])),
            "b2": local[f"b2_{c}"],
            "gamma2": local[f"gamma2_{c}"],
            "Ashot_residual": local[f"Ashot_{c}"],
            "Bshot_residual": local[f"Bshot_{c}"],
        }
        return mapping

    def prior_chi2(c, local):
        names, mean, whitener = priors[c]
        values = np.asarray([class_mapping(c, local)[n] for n in names], dtype=np.float64)
        white = whitener @ (values - mean)
        return float(white @ white)

    def prediction(local):
        return np.concatenate([
            models[c](local["fNL"], class_mapping(c, local), idx) for c in combo
        ])

    def objective(local):
        pred = prediction(local)
        white = solve_triangular(chol, pred - target, lower=True)
        data_q = float(white @ white)
        prior_q = PRIOR_WEIGHT * sum(prior_chi2(c, local) for c in combo)
        return data_q + prior_q

    # iminuit 具名签名（exec 生成）
    signature = ", ".join(par_names)
    body = (
        "def _objective({sig}):\n"
        "    return _obj({dict})\n"
    ).format(sig=signature, dict="{**locals()}")
    namespace = {"_obj": objective}
    exec(body, namespace)
    obj = namespace["_objective"]

    # 起点：fNL 多起点，nuisance 从先验均值（b1 起点 = P0 校准值）
    starts = []
    for f0 in (100.0, 0.0, 40.0, 160.0):
        local = {"fNL": f0}
        for c in combo:
            names, mean, _ = priors[c]
            base = dict(zip(names, mean.tolist()))
            local[f"b2_{c}"] = base["b2"]
            local[f"gamma2_{c}"] = base["gamma2"]
            local[f"Ashot_{c}"] = base["Ashot_residual"]
            local[f"Bshot_{c}"] = base["Bshot_residual"]
            if b1_mode == "free":
                local[f"b1_{c}"] = float(b1_calib[c])
        starts.append(local)

    fits = []
    for start in starts:
        fit = Minuit(obj, **{k: float(v) for k, v in start.items()})
        fit.errordef = Minuit.LEAST_SQUARES
        fit.limits["fNL"] = FNL_BOUNDS
        for c in combo:
            fit.limits[f"b2_{c}"] = (-4.0, 6.0)
            fit.limits[f"gamma2_{c}"] = (-2.0, 4.0)
            scale_a = abs(start[f"Ashot_{c}"]) + 1.0e6
            scale_b = abs(start[f"Bshot_{c}"]) + 1.0e6
            fit.limits[f"Ashot_{c}"] = (-10.0 * scale_a, 10.0 * scale_a)
            fit.limits[f"Bshot_{c}"] = (-10.0 * scale_b, 10.0 * scale_b)
            if b1_mode == "free":
                fit.limits[f"b1_{c}"] = B1_BOUNDS
                fit.errors[f"b1_{c}"] = 0.15
            fit.errors[f"b2_{c}"] = 0.08
            fit.errors[f"gamma2_{c}"] = 0.05
        fit.errors["fNL"] = 10.0
        fit.strategy, fit.tol = 1, 1e-6
        fit.migrad(ncall=200000, iterate=8, use_simplex=True)
        fits.append(fit)
    valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
    best = min(valid or fits, key=lambda f: float(f.fval))
    if not best.valid:
        raise RuntimeError(f"fit failed for combo={combo} cut={cut} mode={b1_mode}")
    best.hesse()
    best.minos("fNL", ncall=100000)
    err = best.merrors["fNL"]

    local = {k: float(best.values[k]) for k in par_names}
    pred = prediction(local)
    white = solve_triangular(chol, pred - target, lower=True)
    data_q = float(white @ white)
    n_par = len(par_names)
    row = {
        "combo": "+".join(combo), "kmax_h_mpc": float(cut), "b1_mode": b1_mode,
        "ndata": ndata, "n_par": n_par, "n_fid": nfid, "hartlap": hartlap,
        "map_fNL": local["fNL"],
        "minos_fNL": {"lower": float(err.lower), "upper": float(err.upper),
                      "valid": bool(err.is_valid)},
        "map": local,
        "data_chi2": data_q, "dof": ndata - n_par,
        "gof_p_value_data_only": float(chi2_distribution.sf(data_q, max(ndata - n_par, 1))),
        "b2_values": {c: local[f"b2_{c}"] for c in combo},
        "b1_values": ({c: local[f"b1_{c}"] for c in combo} if b1_mode == "free"
                      else {c: float(b1_calib[c]) for c in combo}),
        "prior_weight": PRIOR_WEIGHT,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combos", default="filament,sheet,filament+sheet")
    args = parser.parse_args()

    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    b1_calib = {c: float(calib["classes"][c]["b1"]["median"]) for c in ("filament", "sheet")}
    p_calib = {c: float(calib["classes"][c]["p"]["median"]) for c in ("filament", "sheet")}
    print(f"p (fixed, P0 calib): {p_calib}")
    print(f"b1 (P0 calib, fixed-mode value): {b1_calib}")

    runtime = scan.load_runtime()
    models = {c: EnvScanModel(runtime, p_calib[c]) for c in ("filament", "sheet")}
    priors = {c: prior_terms(models[c]) for c in ("filament", "sheet")}

    raw_lc = {c: load_env_samples(c, "LC_p") for c in ("filament", "sheet")}
    raw_fid = {c: load_env_samples(c, "fiducial") for c in ("filament", "sheet")}
    reals_lc = {c: raw_lc[c][1] for c in raw_lc}
    reals_fid = {c: raw_fid[c][1] for c in raw_fid}
    samples_lc_raw = {c: raw_lc[c][0] for c in raw_lc}
    samples_fid_raw = {c: raw_fid[c][0] for c in raw_fid}
    both_lc = align_common(samples_lc_raw, reals_lc, ["filament", "sheet"])
    both_fid = align_common(samples_fid_raw, reals_fid, ["filament", "sheet"])
    print(f"common: fid {both_fid['filament'].shape[0]} sets, LC_p {both_lc['filament'].shape[0]} sets")

    combo_list = []
    for token in args.combos.split(","):
        combo_list.append(tuple(t.strip() for t in token.split("+")))

    rows = []
    for combo in combo_list:
        # 单类用各自全量；多类用共同 realization 交集（协方差块对齐前提）
        samples_lc = both_lc if len(combo) > 1 else {c: samples_lc_raw[c] for c in combo}
        samples_fid = both_fid if len(combo) > 1 else {c: samples_fid_raw[c] for c in combo}
        for cut in CUTS:
            for b1_mode in ("fixed", "free"):
                row = build_case(combo, cut, samples_lc, samples_fid, models, b1_mode, b1_calib, priors)
                rows.append(row)
                print(f"[{row['combo']:<17} kmax={cut:.2f} b1={b1_mode:<5}] "
                      f"fNL={row['map_fNL']:+8.2f} "
                      f"[{row['map_fNL']+err_lower(row):+.1f},{row['map_fNL']+err_upper(row):+.1f}] "
                      f"b2={ {k: round(v, 3) for k, v in row['b2_values'].items()} } "
                      f"gof={row['gof_p_value_data_only']:.3f}", flush=True)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "envclass-b000-kmax-validity-scan-v1",
        "created_utc": utc_now(),
        "model": "DirectFiniteScanB000Model(pre) with bphi=2*dc*(b1-p_c), p fixed from P0 calibration",
        "p_fixed": p_calib, "b1_calib": b1_calib,
        "data": {"target": "LC_p 100-real mean B000", "cov": f"fid {N_FID}-real sample / 42.875 / Hartlap"},
        "cuts": list(CUTS),
        "rows": rows,
    }
    out = ANALYSIS / "ENVCLASS_B_KMAX_SCAN.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    print(f"runner sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}")


def err_lower(row):
    return row["minos_fNL"]["lower"]


def err_upper(row):
    return row["minos_fNL"]["upper"]


if __name__ == "__main__":
    main()
