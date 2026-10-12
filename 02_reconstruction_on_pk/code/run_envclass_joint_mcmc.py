#!/usr/bin/env python3
"""环境类 multitracer 联合 fNL MCMC（方案 A：β 固定校准值）。

代码大纲
========
1. 数据向量：选定环境类组合的 P0 拼接（每类 12 bin），协方差为逐
   realization 堆叠的样本协方差（含类间交叉块）/ 42.875 / Hartlap。
2. 模型（每类）：P_c = b1_c^2 [PL + 2 fNL β_c PL/M + fNL^2 β_c^2 Pphi] + sn0_c，
   β_c 固定为校准值（ENVCLASS_CALIBRATION.json），自由参数
   (fNL, b1_c..., sn0_c...)；fNL 全局共享。
3. iminuit MAP/HESSE 白化 + emcee 4x32 自动延长；收敛门照搬现役机器；
   fNL=0 与 +100 两个 target。
4. 输出 JSON/NPZ/中文报告；报告标注 +100 行为同数据自洽检验
   （校准用同一 LC_p 样本）。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
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
CALIB = PART / "envclass_p0only/calibration/ENVCLASS_CALIBRATION.json"
LATTICE = PART / "inputs/p0_finite_local_png_lattice.npz"
ANALYSIS = PART / "envclass_p0only/jointA"
VOLUME = 42.875
DELTA_C = 1.686
TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 8.0)
SN0_BOUNDS = (-1.0e6, 1.0e6)
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 30000
SEED = 20260919
COMBOS = {
    "sf": ("sheet", "filament"),
    "vsf": ("void", "sheet", "filament"),
    "vfk": ("void", "filament", "knot"),
    "fsk": ("filament", "sheet", "knot"),
    "vk": ("void", "knot"),
    "sk": ("sheet", "knot"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        n = float(value)
        return n if math.isfinite(n) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as s:
        json.dump(jsonable(payload), s, indent=2, sort_keys=True)
        s.write("\n")
        Path(s.name).replace(path)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_class_samples(cls, cosmo, max_real):
    """返回 (样本矩阵, 使用的 realization id 列表)。"""
    rows, ids = [], []
    root = PRODUCTS / cosmo
    for real in range(1000):
        path = root / f"real{real:03d}" / f"{cls}_pk.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            rows.append(np.asarray(z["pk"], dtype=np.float64))
        ids.append(real)
        if len(rows) >= max_real:
            break
    if len(rows) < 30:
        raise RuntimeError(f"{cls}/{cosmo}: only {len(rows)} samples")
    return np.vstack(rows), ids


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


@dataclass
class Case:
    combo: str
    classes: tuple
    target_name: str
    truth: float
    target: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    theory: dict
    beta: dict

    @property
    def n_par(self) -> int:
        return 1 + 2 * len(self.classes)

    @property
    def parameters(self) -> tuple:
        names = ["fNL"]
        for c in self.classes:
            names += [f"b1_{c}", f"sn0_{c}"]
        return tuple(names)

    def predictions(self, values):
        rows = np.atleast_2d(np.asarray(values, dtype=float))
        out = []
        for i, c in enumerate(self.classes):
            f = rows[:, 0]
            b1 = rows[:, 1 + 2 * i]
            sn0 = rows[:, 2 + 2 * i]
            beta = self.beta[c]
            out.append(
                b1[:, None] ** 2 * (
                    self.theory["PL_lattice"][None, :]
                    + 2.0 * f[:, None] * beta * self.theory["PL_over_M_lattice"][None, :]
                    + f[:, None] ** 2 * beta ** 2 * self.theory["Pphi_lattice"][None, :]
                ) + sn0[:, None]
            )
        pred = np.concatenate(out, axis=1)
        return pred[0] if np.asarray(values).ndim == 1 else pred

    def objective_many(self, values):
        pred = self.predictions(values)
        rows = pred[None, :] if pred.ndim == 1 else pred
        resid = rows - self.target[None, :]
        white = np.linalg.solve(self.chol, resid.T).T
        return np.sum(white ** 2, axis=1)

    def objective(self, values):
        return float(self.objective_many(np.asarray(values, dtype=float)[None, :])[0])


def load_case(combo, target_name, theory, beta):
    classes = COMBOS[combo]
    fid_stacks, lcp_stacks = {}, {}
    counts = {}
    for c in classes:
        fid, fid_ids = load_class_samples(c, "fiducial", 100)
        lcp, lcp_ids = load_class_samples(c, "LC_p", 100)
        fid_stacks[c] = (fid, fid_ids)
        lcp_stacks[c] = (lcp, lcp_ids)
        counts[c] = (len(fid_ids), len(lcp_ids))
    # 跨类取共有 realization 的交集（生产进行中的乱序补齐会造成少量差异）。
    common_fid = sorted(set.intersection(*[set(ids) for _, ids in fid_stacks.values()]))
    common_lcp = sorted(set.intersection(*[set(ids) for _, ids in lcp_stacks.values()]))

    def stack(samples_by_class, ids, id_list):
        blocks = []
        for c in classes:
            lookup = {r: i for i, r in enumerate(id_list[c])}
            sel = [lookup[r] for r in ids]
            blocks.append(samples_by_class[c][sel])
        return np.concatenate(blocks, axis=1)

    fid_ids_by_class = {c: fid_stacks[c][1] for c in classes}
    lcp_ids_by_class = {c: lcp_stacks[c][1] for c in classes}
    joint_fid = stack({c: fid_stacks[c][0] for c in classes}, common_fid, fid_ids_by_class)
    joint_lcp = stack({c: lcp_stacks[c][0] for c in classes}, common_lcp, lcp_ids_by_class)
    nfid = len(common_fid)
    target = (joint_lcp if target_name == "fNL100" else joint_fid).mean(axis=0)
    covariance = np.cov(joint_fid, rowvar=False, ddof=1) / VOLUME
    dim = joint_fid.shape[1]
    hartlap = (float(nfid) - dim - 2.0) / (float(nfid) - 1.0)
    covariance = covariance / hartlap
    counts = {"n_fid_common": nfid, "n_lcp_common": len(common_lcp)}
    return Case(combo, classes, target_name, TRUTHS[target_name], target,
                covariance, np.linalg.cholesky(covariance), theory, beta), counts


def generalized_start(case):
    starts = {}
    for i, c in enumerate(case.classes):
        sl = slice(12 * i, 12 * (i + 1))
        design = np.column_stack((case.theory["PL_lattice"], np.ones(12)))
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

    # iminuit 需要显式具名签名；用 exec 按参数名生成目标函数。
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
        "goodness_of_fit_p": float(chi2.sf(float(best.fval), dof)),
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


_ACTIVE = {}


def _active_logp(values):
    return _ACTIVE["logp"](values)


def _groups(path):
    if not path.is_file():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo", choices=COMBOS, required=True)
    args = parser.parse_args()

    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    beta = {c: calib["classes"][c]["bphi_over_b1"]["median"] for c in ("void", "sheet", "filament", "knot")}
    theory = load_theory()
    results = {}
    for target_name in TARGETS:
        case, counts = load_case(args.combo, target_name, theory, beta)
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
            "gof_p": row["goodness_of_fit_p"],
            "converged": mcmc["converged"],
            "acceptance": mcmc["acceptance"],
        }
        np.savez_compressed(ANALYSIS / f"{tag}_posterior.npz", posterior=post)
    payload = {
        "schema": "quijote-z0p5-envclass-jointA-mcmc-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "combo": args.combo,
        "classes": list(COMBOS[args.combo]),
        "beta_fixed": {c: beta[c] for c in COMBOS[args.combo]},
        "parameters": "(fNL, b1_c..., sn0_c...); beta_c fixed from calibration",
        "results": results,
        "notes": "+100 为同数据自洽检验（校准与目标同一 LC_p 样本）",
        "inputs": {"calibration": str(CALIB), "lattice": str(LATTICE)},
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    atomic_json(ANALYSIS / f"{args.combo}_RESULTS.json", payload)
    print(json.dumps({t: {"fNL": round(r["fNL"]["median"], 3),
                          "half_width": round(r["fNL"]["half_width"], 3),
                          "in68": r["truth_inside_68"], "gof": round(r["gof_p"], 3)}
                      for t, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
