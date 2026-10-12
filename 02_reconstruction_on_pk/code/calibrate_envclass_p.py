#!/usr/bin/env python3
"""环境三类的 p 校准：fNL=+100 固定、(p, b1, Pshot) 自由（pre 风格 MCMC）。

代码大纲
========
1. 读取 envclass_p0only 生产的 fiducial（matched100 均值当 fNL=0 侧）与
   LC_p（100 套均值当 +100 侧）逐类 P0；协方差用各类自己的 Fid 单盒
   样本（前 N 套可用样本），几何为自建 v1（12 bins）。
2. 模型：P = b1^2 PL(k) + 2 fNL b1 bphi PL/M + fNL^2 bphi^2 Pphi + Pshot，
   bphi = 2 delta_c (b1 - p)；fNL 固定 100，参数 (p, b1, Pshot)。
3. 理论模板 PL/PL_M/Pphi 从 lattice npz 的解析 k 表在新几何 k 上
   对数插值（模板是平滑解析函数，插值误差 << 统计误差）。
4. iminuit MAP/HESSE + emcee（4x32，收敛门照搬 freep 机器）；
   输出每类 p、b1、bphi/b1 的对照表与后验 NPZ。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import emcee
import h5py
from iminuit import Minuit
import numpy as np
from scipy.stats import chi2

import p0only_arviz_compat as az


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "02_reconstruction_on_pk"
PRODUCTS = PART / "envclass_p0only/full"
LATTICE = PART / "inputs/p0_finite_local_png_lattice.npz"
OUTPUT_DIR = PART / "envclass_p0only/calibration"
CLASSES = ("void", "sheet", "filament", "knot")
FNLFIX = 100.0
PARAMETERS = ("p", "b1", "Pshot_residual")
P_BOUNDS = (-2.0, 6.0)
B1_BOUNDS = (0.1, 6.0)
PSHOT_BOUNDS = (-1.0e6, 1.0e6)  # 类的 shot noise 更大（halo 少）
VOLUME = 42.875
DELTA_C = 1.686
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 20000
SEED = 20260918


def load_class_samples(cls: str, cosmo: str, max_real: int) -> tuple[np.ndarray, int]:
    rows = []
    root = PRODUCTS / cosmo
    # 生产进行中：取现有时序最早的 max_real 套；不足则用全部现有。
    for real in range(1000):
        path = root / f"real{real:03d}" / f"{cls}_pk.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            rows.append(np.asarray(z["pk"], dtype=np.float64))
        if len(rows) >= max_real:
            break
    if len(rows) < 30:
        raise RuntimeError(f"{cls}/{cosmo}: only {len(rows)} samples available, need >=30")
    return np.vstack(rows), len(rows)


def load_theory() -> dict[str, np.ndarray]:
    with np.load(LATTICE, allow_pickle=False) as z:
        lattice = {k: np.asarray(z[k]) for k in z.files}
    # 取一个新几何 k（void pilot 的 k 即自建锚点）。
    with np.load(PRODUCTS / "fiducial" / "real000" / "void_pk.npz", allow_pickle=False) as z:
        k_new = np.asarray(z["k"], dtype=float)
    k_old = np.asarray(lattice["pk_k"], dtype=float)
    templates = {}
    for name in ("PL_lattice", "PL_over_M_lattice", "Pphi_lattice"):
        values = np.asarray(lattice[name], dtype=float)
        templates[name] = np.exp(np.interp(np.log(k_new), np.log(k_old), np.log(np.maximum(values, 1e-300))))
    templates["k"] = k_new
    return templates


def bphi(b1, p):
    return 2.0 * DELTA_C * (b1 - p)


def prediction(params: np.ndarray, theory: dict) -> np.ndarray:
    p, b1, shot = params[:, 0], params[:, 1], params[:, 2]
    bp = 2.0 * DELTA_C * (b1 - p)
    return (
        b1[:, None] ** 2 * theory["PL_lattice"][None, :]
        + 2.0 * FNLFIX * (b1 * bp)[:, None] * theory["PL_over_M_lattice"][None, :]
        + FNLFIX**2 * bp[:, None] ** 2 * theory["Pphi_lattice"][None, :]
        + shot[:, None]
    )


class Case:
    def __init__(self, cls, target, covariance, theory):
        self.cls = cls
        self.target = target
        self.covariance = covariance
        self.chol = np.linalg.cholesky(covariance)
        self.theory = theory

    def objective_many(self, values):
        rows = np.atleast_2d(np.asarray(values, dtype=float))
        resid = prediction(rows, self.theory) - self.target[None, :]
        white = np.linalg.solve(self.chol, resid.T).T
        return np.sum(white**2, axis=1)

    def objective(self, values):
        return float(self.objective_many(np.asarray(values, dtype=float)[None, :])[0])


def profile(case: Case):
    # fNL=0 线性起点：b1^2 PL + c0。
    design = np.column_stack((case.theory["PL_lattice"], np.ones(12)))
    cov_one = case.covariance * VOLUME
    sd = np.linalg.solve(cov_one, design)
    st = np.linalg.solve(cov_one, case.target)
    coef = np.linalg.solve(design.T @ sd, design.T @ st)
    b1_start = float(np.clip(math.sqrt(max(coef[0], 0.01)), 0.101, 5.999))
    shot_start = float(np.clip(coef[1], PSHOT_BOUNDS[0] + 1, PSHOT_BOUNDS[1] - 1))

    def obj(p, b1, Pshot_residual):
        return case.objective(np.asarray((p, b1, Pshot_residual)))

    fits = []
    for p_start in (1.12, 1.0, 1.5, 2.5, 0.0, -1.0, 4.0):
        fit = Minuit(obj, p=p_start, b1=b1_start, Pshot_residual=shot_start)
        fit.errordef = Minuit.LEAST_SQUARES
        fit.limits["p"] = P_BOUNDS
        fit.limits["b1"] = B1_BOUNDS
        fit.limits["Pshot_residual"] = PSHOT_BOUNDS
        fit.errors["p"] = 0.1
        fit.errors["b1"] = 0.05
        fit.errors["Pshot_residual"] = 5000.0
        fit.strategy, fit.tol = 1, 1e-5
        fit.migrad(ncall=200000, iterate=8, use_simplex=True)
        fit.hesse()
        fits.append(fit)
    valid = [f for f in fits if f.valid and math.isfinite(float(f.fval))]
    best = min(valid or fits, key=lambda f: float(f.fval))
    if not best.valid or best.covariance is None:
        raise RuntimeError(f"{case.cls} MAP failed")
    mapping = {n: float(best.values[n]) for n in PARAMETERS}
    row = {
        "map": mapping, "objective": float(best.fval), "dof": 9,
        "goodness_of_fit_p": float(chi2.sf(float(best.fval), 9)),
        "hesse_covariance": np.asarray(best.covariance, dtype=float),
    }
    return row


def log_probability_factory(case, centre, transform):
    def logp(coords):
        values = np.asarray(coords, dtype=float)
        scalar = values.ndim == 1
        rows = values[None, :] if scalar else values
        physical = centre + rows @ transform.T
        support = (
            (physical[:, 0] > P_BOUNDS[0]) & (physical[:, 0] < P_BOUNDS[1])
            & (physical[:, 1] > B1_BOUNDS[0]) & (physical[:, 1] < B1_BOUNDS[1])
            & (physical[:, 2] > PSHOT_BOUNDS[0]) & (physical[:, 2] < PSHOT_BOUNDS[1])
            & np.all(np.isfinite(physical), axis=1)
        )
        out = np.full(rows.shape[0], -math.inf)
        if np.any(support):
            out[support] = -0.5 * case.objective_many(physical[support])
        return float(out[0]) if scalar else out
    return logp


def run_mcmc(case: Case, profile_row: dict) -> dict:
    centre = np.asarray([profile_row["map"][n] for n in PARAMETERS])
    cov = 0.5 * (np.asarray(profile_row["hesse_covariance"]) + np.asarray(profile_row["hesse_covariance"]).T)
    eigval, eigvec = np.linalg.eigh(cov)
    if eigval[0] <= 0:
        floor = max(eigval[-1] * 1e-10, 1e-12)
        cov = (eigvec * np.maximum(eigval, floor)) @ eigvec.T
    transform = np.linalg.cholesky(cov)
    logp = log_probability_factory(case, centre, transform)

    chains_path = OUTPUT_DIR / f"{case.cls}_chain.h5"
    chains_path.parent.mkdir(parents=True, exist_ok=True)
    case_index = CLASSES.index(case.cls)

    from scipy.stats import norm, rankdata

    def pooled_rank_normalize(values):
        ranks = rankdata(values, method="average").reshape(values.shape)
        return norm.ppf((ranks - 0.375) / (values.size + 0.25))

    def temporal_split(values):
        half = values.shape[1] // 2
        split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
        return split.reshape(2 * values.shape[0], half * values.shape[2])

    def ensemble_scaled_ess(transformed):
        series = np.mean(transformed, axis=2)
        iteration_ess = float(az.ess(series, method="identity", chain_axis=0, draw_axis=1))
        dv, sv = float(np.var(transformed, ddof=1)), float(np.var(series, ddof=1))
        if not math.isfinite(iteration_ess) or dv <= 0 or sv <= 0:
            return math.nan
        return min(float(transformed.size), iteration_ess * dv / sv)

    def independent_diagnostics(retained):
        dim = retained.shape[-1]
        location = np.empty(dim); folded = np.empty(dim)
        bulk = np.empty(dim); tail = np.empty(dim)
        for i in range(dim):
            values = retained[..., i]
            ranked = pooled_rank_normalize(values)
            folded_rank = pooled_rank_normalize(np.abs(values - np.median(values)))
            location[i] = float(az.rhat(temporal_split(ranked), method="identity", chain_axis=0, draw_axis=1))
            folded[i] = float(az.rhat(temporal_split(folded_rank), method="identity", chain_axis=0, draw_axis=1))
            q05, q95 = np.quantile(values, (0.05, 0.95))
            low = (values <= q05).astype(float); high = (values >= q95).astype(float)
            bulk[i] = ensemble_scaled_ess(ranked)
            tail[i] = min(ensemble_scaled_ess(low), ensemble_scaled_ess(high))
        return {
            "rank_normalized_split_rhat": np.maximum(location, folded),
            "bulk_ess": bulk, "tail_ess": tail,
        }

    _active = {}

    def active_logp(values):
        return _active["logp"](values)

    steps = INITIAL_STEPS
    while True:
        for ensemble in range(ENSEMBLES):
            group = f"ensemble_{ensemble}"
            backend = emcee.backends.HDFBackend(str(chains_path), name=group)
            seed = SEED + 100000 * case_index + 1009 * ensemble
            if not (group in _open_groups(chains_path)) or backend.iteration == 0:
                backend.reset(WALKERS, 3)
                rng = np.random.default_rng(seed)
                init = []
                for _ in range(200000):
                    scale = math.exp(rng.uniform(math.log(0.08), math.log(1.2)))
                    cand = rng.normal(0.0, scale, 3)
                    if math.isfinite(float(logp(cand))):
                        init.append(cand)
                    if len(init) == WALKERS:
                        break
                initial = np.asarray(init)
                state = np.random.RandomState(seed + 17)
            else:
                initial, state = None, None
            sampler = emcee.EnsembleSampler(
                WALKERS, 3, active_logp, vectorize=True, backend=backend,
                moves=[(emcee.moves.StretchMove(a=2.0), 0.8), (emcee.moves.DEMove(), 0.2)],
            )
            if state is not None:
                sampler.random_state = state.get_state()
            remaining = max(0, steps - backend.iteration)
            if remaining:
                _active["logp"] = logp
                sampler.run_mcmc(initial, remaining, progress=False, skip_initial_state_check=True)
                _active.pop("logp", None)
        chains, accepted = [], []
        for ensemble in range(ENSEMBLES):
            backend = emcee.backends.HDFBackend(str(chains_path), name=f"ensemble_{ensemble}", read_only=True)
            chains.append(np.asarray(backend.get_chain(), dtype=float))
            accepted.append(np.asarray(backend.accepted, dtype=float))
        coords = np.stack(chains); acc = np.stack(accepted)
        burn = int(BURN_FRACTION * coords.shape[1])
        retained = centre + coords[:, burn:] @ transform.T
        diag = independent_diagnostics(retained)
        passed = bool(
            np.all(diag["rank_normalized_split_rhat"] < 1.01)
            and np.all(diag["bulk_ess"] > 1000) and np.all(diag["tail_ess"] > 1000)
        )
        actual = coords.shape[1]
        print(f"{case.cls}: steps={actual} pass={passed} "
              f"Rhat={np.max(diag['rank_normalized_split_rhat']):.6f}", flush=True)
        if passed or actual >= MAX_STEPS:
            break
        steps = min(actual + INCREMENT, MAX_STEPS)
    flat = retained.reshape(-1, 3)
    rng = np.random.default_rng(SEED + 900000 + case_index)
    count = min(POSTERIOR_DRAWS, flat.shape[0])
    selected = flat[rng.choice(flat.shape[0], size=count, replace=False)]
    return {
        "posterior": selected,
        "diagnostics": {k: np.asarray(v).tolist() for k, v in diag.items()},
        "converged": passed,
    }


def _open_groups(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    args = parser.parse_args()

    theory = load_theory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    fid_counts = {}
    for cls in CLASSES:
        fid, nfid = load_class_samples(cls, "fiducial", 100)   # matched100 均值 = fNL0 target
        lcp, nlcp = load_class_samples(cls, "LC_p", 100)       # +100 target
        fid_counts[cls] = (nfid, nlcp)
        target = lcp.mean(axis=0)
        covariance = np.cov(fid, rowvar=False, ddof=1) / VOLUME
        # Hartlap：fid 样本数估计协方差。
        hartlap = (float(nfid) - 12.0 - 2.0) / (float(nfid) - 1.0)
        covariance = covariance / hartlap
        case = Case(cls, target, covariance, theory)
        row = profile(case)
        print(f"[profile] {cls}: p={row['map']['p']:.4f} b1={row['map']['b1']:.4f} "
              f"chi2/dof={row['objective']:.2f}/9 gof={row['goodness_of_fit_p']:.3g}", flush=True)
        mcmc = run_mcmc(case, row)
        post = mcmc["posterior"]
        q = np.quantile(post, (0.16, 0.5, 0.84), axis=0)
        # bphi/b1 后验（解析导出）。
        ratio = 2.0 * DELTA_C * (post[:, 1] - post[:, 0]) / post[:, 1]
        rq = np.quantile(ratio, (0.16, 0.5, 0.84))
        results[cls] = {
            "p": {"median": float(q[1, 0]), "q16": float(q[0, 0]), "q84": float(q[2, 0])},
            "b1": {"median": float(q[1, 1]), "q16": float(q[0, 1]), "q84": float(q[2, 1])},
            "Pshot": {"median": float(q[1, 2])},
            "bphi_over_b1": {"median": float(rq[1]), "q16": float(rq[0]), "q84": float(rq[2])},
            "gof_p": row["goodness_of_fit_p"],
            "converged": mcmc["converged"],
        }
        np.savez_compressed(OUTPUT_DIR / f"{cls}_posterior.npz", posterior=post)
    payload = {
        "schema": "quijote-z0p5-envclass-freep-calibration-v1",
        "status": "complete",
        "fNL_fixed": FNLFIX,
        "covariance": "matched100 fid single-box / 42.875 / Hartlap(100,12)",
        "classes": results,
        "theory_projection": "lattice templates log-interpolated to self-geometry k",
        "notes": "校准测量；+100 侧与校准目标同源（同 LC_p 样本），自洽检验性质",
    }
    (OUTPUT_DIR / "ENVCLASS_CALIBRATION.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({c: {"p": round(r["p"]["median"], 4), "b1": round(r["b1"]["median"], 4),
                          "bphi/b1": round(r["bphi_over_b1"]["median"], 4)}
                      for c, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
