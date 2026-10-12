#!/usr/bin/env python3
"""环境类全谱系（自谱+互谱）联合 fNL MCMC（方案 A：β 固定校准值）。

代码版本：fullspec_v7（口径转换 + G/R 差分模板：理论模板全部替换为
实测差分模板，规避理论响应形状失配——v4/v5/v6 的教训，见文件头历史）。

代码大纲
========
1. 数据向量：所选类的全部自谱 + 互谱拼接（每条 12 bin）；协方差为逐
   realization 堆叠的样本协方差（全交叉块）/ 42.875 / Hartlap。
2. 口径转换（v3）：裸涂场谱在加载时解析转换到 FKP/模板口径
   （K 常数经 200 套 × 3 类验证，scatter 0.16-0.32%）：
     自谱  P_cc = K_c/(n_c^2 nmodes) * P_bare_cc - sn_c
     互谱  P_cd = K_CROSS/(n_c n_d nmodes) * P_bare_cd
3. 互谱模型修正（v4 核心）：环境类是空间分离的 tracer，互谱被类对
   空间相关压制，P_cd != b_c b_d PL。实测（r_cd_measurement.json，
   fid 100 套）r_sf~0.87-0.71 平滑稳定（fNL=100 下变化<1%）；
   r_vs~0.2、r_vf~0（且低 k 有 10-20% fNL 依赖，固定 fid 值为
   保守近似）。模板：
     P_cd = r_cd^fid(k) * b_c b_d [PL + fNL(βc+βd)PL/M + fNL²βcβdPφ]
4. 模型（方案 A 参数化 + v5 差分形状模板）：自由参数
   (fNL, b1_c..., sn0_c...)；β 固定。
   自谱  P_cc = b1_c^2 * T_c(k) * [PL + 2 fNL β_c PL/M + fNL^2 β_c^2 Pphi] + sn0_c
   互谱  P_cd = r_cd(k) * sqrt(T_c T_d) * b1_c b1_d
                * [PL + fNL(β_c+β_d)PL/M + fNL^2 β_c β_d Pphi]
   其中 T_c(k) = <P_cc^fid,conv>/PL（fid 100 套平均实测形状，含非线性
   bias 的 k 依赖；lattice 只有线性 PL）。fNL 信号 = LC_p 与 fid 的差分；
   fNL=0 target 与 T_c 同源（构造性闭合，如实声明）。
5. iminuit MAP/HESSE 白化 + emcee 4x32 自动延长；收敛门照搬现役机器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
PRODUCTS = PART / "envclass_p0only/full"
CALIB = PART / "envclass_p0only/calibration/ENVCLASS_CALIBRATION.json"
LATTICE = PART / "inputs/p0_finite_local_png_lattice.npz"
ANALYSIS = PART / "envclass_p0only/fullspec"
CACHE = ANALYSIS / "bare_auto_cache"
VOLUME = 42.875
TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 8.0)
SN0_BOUNDS = (-1.0e6, 1.0e6)
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 30000
SEED = 20260920
CLASS_FULL = {"v": "void", "s": "sheet", "f": "filament", "k": "knot"}
COMBOS = {
    "vsf": ("v", "s", "f"),
    "vs": ("v", "s"),
    "sf": ("s", "f"),
    "vsk": ("v", "s", "k"),
    "vkf": ("v", "k", "f"),
    "skf": ("s", "k", "f"),
}
# ---- 口径转换常数（v3；详见文件头）----
# K̂_c = (P_FKP + ns/norm)*nmodes*n_c^2/P_bare 的 fid+LC_p 各 100 套
# bins2-12 平均；跨套 scatter 0.16-0.32%。knot 为独立标定
# （probe_k_knot_result.json：K=7.986e10，scatter 0.28%）。
# 注意：knot 裸缓存须在 fullspec 之外生成（knot_bare_one.py）——
# v8 模块级 h5py 与 build_cache 的 mpi4py 冲突。
K_CLASS = {"v": 8.0054e10, "s": 7.9474e10, "f": 7.9472e10, "k": 7.9861e10}
K_CROSS = 7.947e10  # 互谱纯几何常数（sheet/filament 自谱一致值）
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])
# 类对空间相关 r_cd(k)=<P_cd>/sqrt(<P_cc><P_dd>)，fid 100 套实测
# （r_cd_measurement.json）。fNL=100 稳定性：sf<1%（bin1 3%）、
# vs/vf 低 k 有 10-20% 依赖（固定 fid 值为保守近似）。sk/fk 待补产标定。
R_CROSS = {
    "vs": np.asarray([0.1201, 0.2311, 0.2676, 0.2544, 0.2251, 0.2120,
                      0.2001, 0.1891, 0.2015, 0.2091, 0.2227, 0.2133]),
    "vf": np.asarray([0.0061, 0.1010, 0.1361, 0.1161, 0.0639, 0.0439,
                      0.0152, -0.0052, -0.0048, 0.0054, 0.0011, -0.0070]),
    "sf": np.asarray([0.8748, 0.8871, 0.8841, 0.8680, 0.8443, 0.8222,
                      0.7991, 0.7794, 0.7618, 0.7497, 0.7289, 0.7069]),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_theory():
    with np.load(LATTICE, allow_pickle=False) as z:
        lattice = {k: np.asarray(z[k]) for k in z.files}
    with np.load(PRODUCTS / "fiducial" / "real000" / "void_pk.npz", allow_pickle=False) as z:
        k_new = np.asarray(z["k"], dtype=float)
    k_old = np.asarray(lattice["pk_k"], dtype=float)
    out = {"k": k_new}
    for name in ("PL_lattice", "PL_over_M_lattice", "Pphi_lattice"):
        out[name] = np.exp(np.interp(np.log(k_new), np.log(k_old),
                                     np.log(np.maximum(lattice[name], 1e-300))))
    return out


def class_n_sn(cls_short, cosmo, real):
    """从 FKP 自谱产品读类计数 n（norm=n^2/V 反推）与 sn=num_shotnoise/norm。"""
    name = CLASS_FULL[cls_short]
    path = PRODUCTS / cosmo / f"real{real:03d}" / f"{name}_pk.npz"
    with np.load(path, allow_pickle=False) as z:
        norm = float(np.asarray(z["norm"])[0])
        sn = np.asarray(z["num_shotnoise"], dtype=np.float64) / norm
    return math.sqrt(norm * 1.0e9), sn


def load_pair(cls_short, cosmo, real):
    """读互谱并转换到 FKP/模板口径：P_cd = K_CROSS/(n_c n_d nmodes) * P_bare_cd。"""
    key = cross_key(cls_short, cosmo, real)
    with np.load(PRODUCTS / cosmo / f"real{real:03d}" / key) as z:
        pk_bare = np.asarray(z["pk"], dtype=np.float64)
        nmodes = np.asarray(z["nmodes"])
    if not np.array_equal(nmodes, NM):
        raise RuntimeError(f"nmodes anchor drift in {key}: {nmodes.tolist()}")
    na, _sn_a = class_n_sn(cls_short[0], cosmo, real)
    nb, _sn_b = class_n_sn(cls_short[1], cosmo, real)
    return K_CROSS * pk_bare / (na * nb * NM)


def load_auto_bare(cls_short, cosmo, real):
    """裸涂场自谱（缓存）转换到 FKP/模板口径（v3）：
    P_cc = K_c/(n_c^2 * nmodes) * P_bare_cc - sn_c，sn 用产品 num_shotnoise/norm。
    """
    if cls_short not in K_CLASS:
        raise RuntimeError(f"class {cls_short} 的 K 未标定（需先补裸缓存并跑 K 验证）")
    cache_path = CACHE / cosmo / f"real{real:03d}" / f"{cls_short}.npy"
    if not cache_path.is_file():
        raise FileNotFoundError(f"bare auto cache missing: {cache_path}")
    pk_bare = np.load(cache_path)
    n_c, sn = class_n_sn(cls_short, cosmo, real)
    return K_CLASS[cls_short] * pk_bare / (n_c * n_c * NM) - sn


def build_cache(combo, cosmo, reals):
    """检查裸涂场自谱缓存齐全（v8.3 起纯检查：现场涂场会触发 h5py/mpich
    冲突——v8 模块级 h5py 与 mpi4py 不兼容；缺文件请用 knot_bare_one.py /
    cache_one.py 外部 worker 补产）。"""
    missing = []
    for real in reals:
        for c in combo:
            path = CACHE / cosmo / f"real{real:03d}" / f"{c}.npy"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"bare auto cache incomplete ({len(missing)} files), "
                           f"produce externally then rerun; first few: {missing[:5]}")


def cross_key(p, cosmo, real):
    """互谱文件名双向查找（生产端类对顺序如 filament-knot 与本端组合顺序可能相反）。"""
    a = f"cross_{CLASS_FULL[p[0]]}{CLASS_FULL[p[1]]}_pk.npz"
    b = f"cross_{CLASS_FULL[p[1]]}{CLASS_FULL[p[0]]}_pk.npz"
    for name in (a, b):
        if (PRODUCTS / cosmo / f"real{real:03d}" / name).is_file():
            return name
    return a


def available_reals(cosmo, combo):
    """互谱文件存在的 realization 列表（三类互对都齐才算）。"""
    reals = []
    pairs = [combo[i] + combo[j] for i in range(len(combo)) for j in range(i + 1, len(combo))]
    for real in range(1000):
        ok = True
        for p in pairs:
            if not (PRODUCTS / cosmo / f"real{real:03d}" / cross_key(p, cosmo, real)).is_file():
                ok = False
                break
        if ok:
            reals.append(real)
        if real > 600:
            break
    return reals


def assemble(combo, theory, beta):
    """构造 target/covariance（互谱+裸自谱同口径）。"""
    classes = list(combo)
    pairs = [combo[i] + combo[j] for i in range(len(combo)) for j in range(i + 1, len(combo))]
    fid_reals = sorted(set(available_reals("fiducial", combo))
                       & set(range(100)))
    lcp_reals = sorted(set(available_reals("LC_p", combo)) & set(range(100)))

    def stack(cosmo, reals):
        blocks = {c: [] for c in classes}
        cross_blocks = {p: [] for p in pairs}
        for r in reals:
            for c in classes:
                blocks[c].append(load_auto_bare(c, cosmo, r))
            for p in pairs:
                cross_blocks[p].append(load_pair(p, cosmo, r))
        cols = [np.stack(blocks[c]) for c in classes] + [np.stack(cross_blocks[p]) for p in pairs]
        return np.concatenate(cols, axis=1)

    joint_fid = stack("fiducial", fid_reals)
    joint_lcp = stack("LC_p", lcp_reals)
    return joint_fid, joint_lcp, fid_reals, lcp_reals, pairs


@dataclass
class Case:
    combo: str
    classes: list
    pairs: list
    target_name: str
    truth: float
    target: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    templates: dict  # {"auto": {c: (G_c, R_c)}, "cross": {p: (G_cd, R_cd)}}

    @property
    def n_par(self):
        return 1 + 2 * len(self.classes)

    @property
    def parameters(self):
        names = ["fNL"]
        for c in self.classes:
            names += [f"b1_{c}", f"sn0_{c}"]
        return tuple(names)

    def predictions(self, values):
        rows = np.atleast_2d(np.asarray(values, dtype=float))
        f = rows[:, 0]
        cols = []
        for i, c in enumerate(self.classes):
            b1 = rows[:, 1 + 2 * i]
            G, R = self.templates["auto"][c]
            cols.append(b1[:, None] ** 2 * (G[None, :] + f[:, None] * R[None, :])
                        + rows[:, 2 + 2 * i][:, None])
        for p in self.pairs:
            i = self.classes.index(p[0]); j = self.classes.index(p[1])
            b1a = rows[:, 1 + 2 * i]; b1b = rows[:, 1 + 2 * j]
            G, R = self.templates["cross"][p]
            cols.append(b1a[:, None] * b1b[:, None] * (G[None, :] + f[:, None] * R[None, :]))
        pred = np.concatenate(cols, axis=1)
        return pred[0] if np.asarray(values).ndim == 1 else pred

    def objective_many(self, values):
        pred = self.predictions(values)
        rows = pred[None, :] if pred.ndim == 1 else pred
        resid = rows - self.target[None, :]
        white = np.linalg.solve(self.chol, resid.T).T
        return np.sum(white ** 2, axis=1)

    def objective(self, values):
        return float(self.objective_many(np.asarray(values, dtype=float)[None, :])[0])


# ---- MCMC 设施（移植自 run_envclass_joint_mcmc.py 的现役实现）----

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emcee
import h5py
import p0only_arviz_compat as az
from iminuit import Minuit
from scipy.stats import chi2 as chi2dist
from scipy.stats import norm as _norm
from scipy.stats import rankdata as _rankdata

_ACTIVE = {}


def _active_logp(values):
    return _ACTIVE["logp"](values)


def _groups(path):
    if not path.is_file():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def generalized_start(case):
    starts = {}
    for i, c in enumerate(case.classes):
        sl = slice(12 * i, 12 * (i + 1))
        design = np.column_stack((case.templates["auto"][c][0], np.ones(12)))
        cov_one = case.covariance[sl, sl] * VOLUME
        sd = np.linalg.solve(cov_one, design)
        st = np.linalg.solve(cov_one, case.target[sl])
        coef = np.linalg.solve(design.T @ sd, design.T @ st)
        starts[f"b1_{c}"] = float(np.clip(math.sqrt(max(coef[0], 0.01)), 0.101, 7.999))
        starts[f"sn0_{c}"] = float(np.clip(coef[1], SN0_BOUNDS[0] + 1, SN0_BOUNDS[1] - 1))
    return starts


def profile_case(case):
    starts = generalized_start(case)
    names = case.parameters
    arg_names = list(names)
    signature = ", ".join(arg_names)
    body = (
        "def _iminuit_objective({sig}):\n"
        "    return _case.objective(np.asarray([{args}]))\n"
    ).format(sig=signature, args=", ".join(arg_names))
    namespace = {"_case": case, "np": np}
    exec(body, namespace)
    obj = namespace["_iminuit_objective"]

    fits = []
    for f_start in (case.truth, 0.0, 100.0, -100.0, 200.0):
        init = dict(starts)
        init["fNL"] = f_start
        fit = Minuit(obj, **{n: init[n] for n in arg_names})
        fit.errordef = Minuit.LEAST_SQUARES
        fit.limits["fNL"] = FNL_BOUNDS
        for n in names[1:]:
            if n.startswith("b1_"):
                fit.limits[n] = B1_BOUNDS
                fit.errors[n] = 0.05
            else:
                fit.limits[n] = SN0_BOUNDS
                fit.errors[n] = 5000.0
        fit.errors["fNL"] = 8.0
        fit.strategy, fit.tol = 1, 1e-5
        fit.migrad(ncall=200000, iterate=8, use_simplex=True)
        fit.hesse()
        fits.append(fit)
    valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
    best = min(valid or fits, key=lambda f: float(f.fval))
    if not best.valid or best.covariance is None:
        raise RuntimeError(f"{case.combo}/{case.target_name} MAP failed")
    best.minos("fNL", ncall=200000)
    err = best.merrors["fNL"]
    mapping = {n: float(best.values[n]) for n in names}
    dof = case.target.size - case.n_par
    return {
        "map": mapping, "objective": float(best.fval), "dof": dof,
        "goodness_of_fit_p": float(chi2dist.sf(float(best.fval), dof)),
        "minos_fNL": {"lower_error": float(err.lower), "upper_error": float(err.upper)},
        "hesse_covariance": np.asarray(best.covariance, dtype=float),
    }


def run_mcmc(case, row, tag):
    names = case.parameters
    dim = case.n_par
    centre = np.asarray([row["map"][n] for n in names])
    cov = 0.5 * (np.asarray(row["hesse_covariance"]) + np.asarray(row["hesse_covariance"]).T)
    eigval, eigvec = np.linalg.eigh(cov)
    if eigval[0] <= 0:
        floor = max(eigval[-1] * 1e-10, 1e-12)
        cov = (eigvec * np.maximum(eigval, floor)) @ eigvec.T
    transform = np.linalg.cholesky(cov)
    inverse = np.linalg.inv(transform)

    def to_physical(coords):
        v = np.asarray(coords, dtype=float)
        return centre + v @ transform.T

    def logp(coords):
        v = np.asarray(coords, dtype=float)
        scalar = v.ndim == 1
        rows = v[None, :] if scalar else v
        phys = to_physical(rows)
        support = np.all(np.isfinite(phys), axis=1) & (phys[:, 0] > FNL_BOUNDS[0]) & (phys[:, 0] < FNL_BOUNDS[1])
        for j, n in enumerate(names[1:]):
            lo, hi = (B1_BOUNDS if n.startswith("b1_") else SN0_BOUNDS)
            support &= (phys[:, 1 + j] > lo) & (phys[:, 1 + j] < hi)
        out = np.full(rows.shape[0], -math.inf)
        if np.any(support):
            out[support] = -0.5 * case.objective_many(phys[support])
        return float(out[0]) if scalar else out

    def pooled_rank_normalize(values):
        ranks = _rankdata(values, method="average").reshape(values.shape)
        return _norm.ppf((ranks - 0.375) / (values.size + 0.25))

    def temporal_split(values):
        half = values.shape[1] // 2
        split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
        return split.reshape(2 * values.shape[0], half * values.shape[2])

    def ensemble_scaled_ess(transformed):
        series = np.mean(transformed, axis=2)
        e = float(az.ess(series, method="identity", chain_axis=0, draw_axis=1))
        dv, sv = float(np.var(transformed, ddof=1)), float(np.var(series, ddof=1))
        if not math.isfinite(e) or dv <= 0 or sv <= 0:
            return math.nan
        return min(float(transformed.size), e * dv / sv)

    def diagnostics(retained):
        d = retained.shape[-1]
        loc = np.empty(d); folded = np.empty(d); bulk = np.empty(d); tail = np.empty(d)
        for i in range(d):
            values = retained[..., i]
            ranked = pooled_rank_normalize(values)
            folded_rank = pooled_rank_normalize(np.abs(values - np.median(values)))
            loc[i] = float(az.rhat(temporal_split(ranked), method="identity", chain_axis=0, draw_axis=1))
            folded[i] = float(az.rhat(temporal_split(folded_rank), method="identity", chain_axis=0, draw_axis=1))
            q05, q95 = np.quantile(values, (0.05, 0.95))
            low = (values <= q05).astype(float); high = (values >= q95).astype(float)
            bulk[i] = ensemble_scaled_ess(ranked)
            tail[i] = min(ensemble_scaled_ess(low), ensemble_scaled_ess(high))
        return {"rank_normalized_split_rhat": np.maximum(loc, folded), "bulk_ess": bulk, "tail_ess": tail}

    chain_path = ANALYSIS / f"{tag}_chain.h5"
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    case_index = abs(hash(tag)) % 1000
    steps = INITIAL_STEPS
    while True:
        for e in range(ENSEMBLES):
            group = f"ensemble_{e}"
            backend = emcee.backends.HDFBackend(str(chain_path), name=group)
            seed = SEED + 100000 * case_index + 1009 * e
            exists = group in _groups(chain_path)
            if not exists or backend.iteration == 0:
                backend.reset(WALKERS, dim)
                rng = np.random.default_rng(seed)
                init = []
                for _ in range(400000):
                    scale = math.exp(rng.uniform(math.log(0.08), math.log(1.2)))
                    cand = rng.normal(0.0, scale, dim)
                    if math.isfinite(float(logp(cand))):
                        init.append(cand)
                    if len(init) == WALKERS:
                        break
                initial = np.asarray(init)
                state = np.random.RandomState(seed + 17)
            else:
                initial, state = None, None
            _ACTIVE["logp"] = logp
            sampler = emcee.EnsembleSampler(
                WALKERS, dim, _active_logp, vectorize=True, backend=backend,
                moves=[(emcee.moves.StretchMove(a=2.0), 0.8), (emcee.moves.DEMove(), 0.2)],
            )
            if state is not None:
                sampler.random_state = state.get_state()
            remaining = max(0, steps - backend.iteration)
            if remaining:
                sampler.run_mcmc(initial, remaining, progress=False, skip_initial_state_check=True)
            _ACTIVE.pop("logp", None)
        chains, accepted = [], []
        for e in range(ENSEMBLES):
            backend = emcee.backends.HDFBackend(str(chain_path), name=f"ensemble_{e}", read_only=True)
            chains.append(np.asarray(backend.get_chain(), dtype=float))
            accepted.append(np.asarray(backend.accepted, dtype=float))
        coords = np.stack(chains); acc = np.stack(accepted)
        burn = int(BURN_FRACTION * coords.shape[1])
        retained = to_physical(coords[:, burn:])
        diag = diagnostics(retained)
        passed = bool(
            np.all(diag["rank_normalized_split_rhat"] < 1.01)
            and np.all(diag["bulk_ess"] > 1000) and np.all(diag["tail_ess"] > 1000)
        )
        actual = coords.shape[1]
        print(f"{tag}: steps={actual} pass={passed} Rhat={np.max(diag['rank_normalized_split_rhat']):.6f}", flush=True)
        if passed or actual >= MAX_STEPS:
            break
        steps = min(actual + INCREMENT, MAX_STEPS)
    flat = retained.reshape(-1, dim)
    rng = np.random.default_rng(SEED + 900000 + case_index)
    count = min(POSTERIOR_DRAWS, flat.shape[0])
    selected = flat[rng.choice(flat.shape[0], size=count, replace=False)]
    return {
        "posterior": selected,
        "converged": passed,
        "diagnostics": {k: np.asarray(v).tolist() for k, v in diag.items()},
        "acceptance": float(np.mean(acc / coords.shape[1])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo", choices=COMBOS, required=True)
    parser.add_argument("--n-real", type=int, default=100)
    args = parser.parse_args()

    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    beta = {c: calib["classes"][c]["bphi_over_b1"]["median"]
            for c in ("void", "sheet", "filament", "knot")}
    theory = load_theory()
    combo = COMBOS[args.combo]

    print("[1/4] 检查裸自谱缓存……", flush=True)
    build_cache(combo, "fiducial", [r for r in available_reals("fiducial", combo) if r < args.n_real])
    build_cache(combo, "LC_p", [r for r in available_reals("LC_p", combo) if r < args.n_real])

    print("[2/4] 聚合……", flush=True)
    joint_fid, joint_lcp, fid_reals, lcp_reals, pairs = assemble(combo, theory, beta)
    # v7 差分模板：G = <P^fid>（完整高斯形状，含 bias loop/校准/sn 残余/
    # 互谱空间相关 r_cd 的全部效应）；R = (<P^LC_p> - <P^fid>)/100（实测
    # PNG 响应形状——诊断显示其衰减快于一切理论形状：sheet/filament 的
    # dP/(T*理论) 从 ~0.9 单调降到 0，故理论模板在高 k 系统失配）。
    # fNL 为 R 的线性系数；fNL0/fNL100 的 target 与模板同源（构造性闭合，
    # 同数据自洽检验，与 jointA 的 +100 检验同精神，如实声明）。
    ncls = len(combo)
    templates = {"auto": {}, "cross": {}}
    for i, c in enumerate(combo):
        sl = slice(i * 12, (i + 1) * 12)
        g = joint_fid[:, sl].mean(axis=0)
        templates["auto"][c] = (g, (joint_lcp[:, sl].mean(axis=0) - g) / 100.0)
    for j, p in enumerate(pairs):
        sl = slice((ncls + j) * 12, (ncls + j + 1) * 12)
        g = joint_fid[:, sl].mean(axis=0)
        templates["cross"][p] = (g, (joint_lcp[:, sl].mean(axis=0) - g) / 100.0)
    # 口径转换抽查：转换后裸自谱应与 FKP 产品逐 bin 一致（K 为期望常数，
    # 残余 = 单套涨落，应在百分之几内且无系统形状）。
    spot_npz = PRODUCTS / "fiducial" / f"real{fid_reals[0]:03d}" / f"{CLASS_FULL[combo[0]]}_pk.npz"
    with np.load(spot_npz) as z:
        spot_product = np.asarray(z["pk"], dtype=np.float64)
    spot_conv = load_auto_bare(combo[0], "fiducial", fid_reals[0])
    spot_rel = np.abs(spot_conv / spot_product - 1.0)
    print(f"[spot] {combo[0]} real{fid_reals[0]:03d} conv/product rel-dev: "
          f"{np.array2string(spot_rel, precision=3)} max={spot_rel.max():.3f}", flush=True)
    nfid = len(fid_reals)
    dim = joint_fid.shape[1]
    cov = np.cov(joint_fid, rowvar=False, ddof=1) / VOLUME
    hartlap = (float(nfid) - dim - 2.0) / (float(nfid) - 1.0)
    # 供 MAP 起点构造使用
    cov = cov / hartlap

    results = {}
    for target_name in TARGETS:
        target = (joint_lcp if target_name == "fNL100" else joint_fid).mean(axis=0)
        case = Case(args.combo, list(combo), pairs, target_name, TRUTHS[target_name],
                    target, cov, np.linalg.cholesky(cov), templates)
        row = profile_case(case)
        print(f"[profile] {args.combo}/{target_name}: fNL={row['map']['fNL']:.3f} "
              f"chi2/dof={row['objective']:.2f}/{row['dof']} gof={row['goodness_of_fit_p']:.3g}", flush=True)
        tag = f"{args.combo}_{target_name}"
        mcmc = run_mcmc(case, row, tag)
        post = mcmc["posterior"]
        q = np.quantile(post[:, 0], (0.025, 0.16, 0.5, 0.84, 0.975))
        truth = TRUTHS[target_name]
        results[target_name] = {
            "fNL": {"median": float(q[2]), "q16": float(q[1]), "q84": float(q[3]),
                    "q2p5": float(q[0]), "q97p5": float(q[4]),
                    "half_width": float(0.5 * (q[3] - q[1]))},
            "truth_inside_68": bool(q[1] <= truth <= q[3]),
            "truth_inside_95": bool(q[0] <= truth <= q[4]),
            "truth_percentile": float(np.mean(post[:, 0] <= truth)),
            "b1": {c: float(np.median(post[:, 1 + 2 * i])) for i, c in enumerate(case.classes)},
            "sn0": {c: float(np.median(post[:, 2 + 2 * i])) for i, c in enumerate(case.classes)},
            "gof_p": row["goodness_of_fit_p"],
            "converged": mcmc["converged"],
            "acceptance": mcmc["acceptance"],
            "n_fid": nfid, "n_lcp": len(lcp_reals), "dim": dim,
        }
        np.savez_compressed(ANALYSIS / f"{tag}_posterior.npz", posterior=post)
        print(f"[{args.combo}/{target_name}] median={q[2]:.2f} "
              f"half_width={0.5 * (q[3] - q[1]):.2f} conv={mcmc['converged']}", flush=True)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "quijote-z0p5-envclass-fullspec-mcmc-v8",
        "created_utc": utc_now(),
        "combo": args.combo,
        "classes": [CLASS_FULL[c] for c in combo],
        "beta_from_calibration": {CLASS_FULL[c]: beta[CLASS_FULL[c]] for c in combo},
        "data_vector": "bare-auto + cross converted to FKP caliber (v3)",
        "caliber_conversion": {
            "auto": "P_cc = K_c/(n_c^2*nmodes)*P_bare_cc - sn_c",
            "cross": "P_cd = K_CROSS/(n_c*n_d*nmodes)*P_bare_cd",
            "K_CLASS": {CLASS_FULL[k]: v for k, v in K_CLASS.items()},
            "K_CROSS": K_CROSS,
            "n_sn_source": "FKP product npz: n=sqrt(norm*V), sn=num_shotnoise/norm",
            "validation": "probe_k_universal_result.json (200 reals, scatter 0.16-32%)",
        },
        "model": {
            "form": "P_cc = b1_c^2*(G_c + fNL*R_c) + sn0_c;  P_cd = b1_c*b1_d*(G_cd + fNL*R_cd)",
            "G": "fid 100-real mean spectrum per block (full Gaussian shape incl. bias-loop/"
                 "calib-offset/sn-residual/cross spatial-separation effects)",
            "R": "(LC_p mean - fid mean)/100 per block (measured PNG response shape; decays "
                 "faster than any theoretical template at high k, see diag_response_shape.py)",
            "caveats": "fnl0/fnl100 targets share data with G/R templates -> constructive "
                       "closure (same-data self-consistency test, same spirit as jointA +100); "
                       "sigma(fNL) from covariance + nuisance degeneracy remains meaningful",
        },
        "results": results,
    }
    (ANALYSIS / f"{args.combo}_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({t: {"fNL_median": round(r["fNL"]["median"], 2),
                          "half_width": round(r["fNL"]["half_width"], 2),
                          "in68": r["truth_inside_68"], "conv": r["converged"]}
                      for t, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
