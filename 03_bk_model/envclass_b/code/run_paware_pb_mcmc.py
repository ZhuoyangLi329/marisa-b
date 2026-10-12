#!/usr/bin/env python3
"""校准 p 的 matched250 P+B 后验：保留原似然，仅增加完整 MCMC。

代码大纲：冻结完整交叉协方差与原模型；验证多项式批量求值；复现 MAP；
四个独立 ensemble 采样并检查收敛；保存链、后验分位数与 PDF。
原 MINOS 产物只读；不加入 IR、不加入 Gaussian 残差模板、不重校准 p。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import emcee
import h5py
import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2

ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
HERE = ROOT / "03_bk_model/envclass_b"
OUT = HERE / "outputs/paware_pb_mcmc_k012_k014_v1"
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(ROOT / "02_reconstruction_on_pk/code"))
import run_joint_pb_matched250 as original
from run_five_method_p0only_mcmc import independent_diagnostics

NAMES = original.NAMES
BNAMES = ("b2", "gamma2", "Ashot_residual", "Bshot_residual")
BOUNDS = np.array([[-300., 300.], [.1, 8.], [-1e5, 1e5], [-4., 6.], [-2., 4.], [-1e7, 1e7], [-1e7, 1e7]])
ENSEMBLES, WALKERS, NDIM = 4, 32, 7
CUTS = (.12, .14)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(method, cut):
    return f"{method}_k{cut:.2f}".replace(".", "p")


def basis_prediction(values, coefficients, p_templates, p_halo):
    """有限树模型的多项式批量求值，必须先通过直接求值逐点验证。"""
    values = np.atleast_2d(values)
    f, b, shot = values[:, :3].T
    bp = 2 * 1.686 * (b - p_halo)
    pp = (b[:, None]**2 * p_templates[0] + (2*f*b*bp)[:, None] * p_templates[1]
          + (f*bp)[:, None]**2 * p_templates[2] + shot[:, None])
    cb = np.polynomial.chebyshev.chebvander(2*(b-.1)/7.9-1, coefficients.shape[0]-1)
    nu = np.column_stack((np.ones(len(values)), values[:, 3:]))
    ff = np.column_stack((np.ones(len(values)), f, f*f))
    features = (cb[:, :, None, None] * nu[:, None, :, None] * ff[:, None, None, :]).reshape(len(values), -1)
    bb = features @ coefficients.reshape(-1, coefficients.shape[-1])
    return np.concatenate((pp, bb), axis=1)


class Case:
    def __init__(self, directory):
        self.directory = directory
        self.meta = json.loads((directory / "contract.json").read_text())
        with np.load(directory / "likelihood.npz", allow_pickle=False) as z:
            for name in z.files:
                setattr(self, name, np.array(z[name]))
        self.inverse_chol = np.linalg.inv(self.chol)

    def terms(self, values):
        values = np.atleast_2d(values)
        pred = basis_prediction(values, self.coefficients, self.p_templates, self.meta["p_halo"])
        w = (pred-self.target) @ self.inverse_chol.T
        nuisance = values[:, [1, 3, 4, 5, 6]]
        pw = (nuisance-self.prior_mean) @ self.prior_whitener.T
        return np.sum(w*w, axis=1), original.PRIOR_WEIGHT*np.sum(pw*pw, axis=1)

    def objective(self, *values):
        a, b = self.terms(values)
        return float(a[0]+b[0])

    def logp(self, coordinates):
        physical = self.centre + np.atleast_2d(coordinates) @ self.transform.T
        valid = np.all(np.isfinite(physical) & (physical > self.bounds[:, 0]) & (physical < self.bounds[:, 1]), axis=1)
        result = np.full(len(physical), -np.inf)
        if valid.any():
            a, b = self.terms(physical[valid])
            result[valid] = -.5*(a+b)
        return result


def prepare(method):
    """严格复用原数据加载器和交叉协方差，不触碰旧产物。"""
    low, scan, runtime = original.configure_models()
    cfg = original.CASES[method]
    low.bphi_from_b1 = lambda b1: 2*low.DELTA_C*(float(b1)-cfg["p"])
    if method == "sto_marisa":
        low.B_REC, low.P_REC = cfg["b_rec"], cfg["p_rec"]
        low.BPHI_REC = 2*low.DELTA_C*(low.B_REC-low.P_REC)
        pt, bt, k, edges = original.load_paware_target()
        with np.load(original.VENDOR / "data/sto_marisa_matrix.npz", allow_pickle=False) as z:
            pf = z["fiducial_post_P0"][:, :12]
            bf = z["fiducial_post_B000"]
    else:
        with np.load(original.VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as z:
            k, pf, pt = z["pk_k"], z["fiducial_pre_P0"], z["LC_p_pre_P0"]
        with np.load(original.VENDOR / "data/STANDARD_FNL250_MATRIX.npz", allow_pickle=False) as z:
            bf, bt = z["fiducial_pre_B000"], z["LC_p_pre_B000"]
    assert pf.shape == (500, 12) and bf.shape == (500, 120)
    assert pt.shape == (250, 12) and bt.shape == (250, 120)
    model = scan.DirectFiniteScanB000Model(runtime, cfg["method"])
    context = model.base_context
    assert tuple(context.names) == ("b1", *BNAMES)
    prior_mean = np.asarray(context.prior.mean)
    prior_whitener = np.asarray(context.prior.whitener)
    templates = np.asarray(original.p_theory(k))
    old = json.loads(original.OUT.read_text())
    # 对 b1 的六阶 Chebyshev 表示及四个 nuisance 的仿射表示；不通过验证即停止。
    nodes = np.cos(np.pi*(np.arange(7)+.5)/7)
    sampled = []
    for x in nodes:
        b1 = .1 + 7.9*(x+1)/2
        mapping = {"b1": b1, **dict.fromkeys(BNAMES, 0.)}
        def parts(mp):
            p = model.parts(mp)
            return np.array([p[n] for n in ("gaussian", "linear_coefficient", "quadratic_coefficient")])
        zero = parts(mapping)
        sampled.append([zero, *[parts({**mapping, name: 1.})-zero for name in BNAMES]])
    coefficients = np.linalg.solve(np.polynomial.chebyshev.chebvander(nodes, 6), np.asarray(sampled).reshape(7, -1)).reshape(7, 5, 3, 120)
    for cut in CUTS:
        directory = OUT / tag(method, cut)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "chain.h5").exists():
            raise RuntimeError("已有链，禁止重新冻结其似然")
        idx = scan.selected_indices(runtime.geometry, cut)
        joint_fid = np.concatenate((pf, bf[:, idx]), axis=1)
        covariance = np.cov(joint_fid, rowvar=False, ddof=1)/original.VOLUME
        hartlap = (500-joint_fid.shape[1]-2)/499
        chol = np.linalg.cholesky(covariance)/np.sqrt(hartlap)
        target = np.concatenate((pt.mean(axis=0), bt.mean(axis=0)[idx]))
        coef = coefficients[..., idx]
        reference = next(row for row in old["rows"][method] if row["kmax_h_mpc"] == cut)
        centre = np.array([reference["map"][n] for n in NAMES])
        bounds = BOUNDS.copy()
        bounds[5:, 1] = 10*(np.abs(prior_mean[3:])+1e6)
        bounds[5:, 0] = -bounds[5:, 1]
        rng = np.random.default_rng(930 + int(cut*100) + (method != "pre"))
        test = np.column_stack((rng.uniform(-290, 290, 200), rng.uniform(.11, 7.99, 200), rng.uniform(-1e4, 1e4, 200), rng.uniform(-3.9, 5.9, 200), rng.uniform(-1.9, 3.9, 200), rng.uniform(-8, 8, (200, 2))))
        test = np.vstack((centre, test))
        fast = basis_prediction(test, coef, templates, cfg["p"])
        direct = fast.copy()
        for i, v in enumerate(test):
            mp = dict(zip(("b1", *BNAMES), v[[1, 3, 4, 5, 6]]))
            direct[i, 12:] = model(v[0], mp, idx)
        whitened_error = solve_triangular(chol, (fast-direct).T, lower=True).T
        max_error = float(np.max(np.abs(whitened_error)))
        if max_error > 1e-6:
            raise RuntimeError(f"多项式与直接模型不一致：{method} {cut} {max_error}")
        white = solve_triangular(chol, direct[0]-target, lower=True)
        data_parity = abs(float(white@white)-reference["data_chi2"])
        if data_parity > 1e-6:
            raise RuntimeError(f"原 MINOS data chi2 未复现：{data_parity}")
        corr = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        meta = {"method": method, "kmax_B": cut, "kmax_P": .08, "volume": original.VOLUME,
                "p_halo": cfg["p"], "p_fixed_not_marginalized": True, "truth_fNL": 100,
                "n_target": 250, "n_fid": 500, "parameter_names": NAMES,
                "full_PB_cross_covariance": True, "cross_correlation_max": np.max(np.abs(corr[:12, 12:])),
                "hartlap": hartlap, "IR_resummation": False, "Gaussian_discrepancy": False,
                "polynomial_max_whitened_error": max_error, "reference_data_chi2_parity": data_parity,
                "reference_MINOS": reference, "original_runner_sha256": digest(Path(original.__file__)),
                "runner_sha256": digest(Path(__file__)), "prior_weight": original.PRIOR_WEIGHT,
                "calibration_uncertainty_propagated": False,
                "fid_covariance_note": "沿用现行冻结 Fid500；STO target 为 p-aware matched250",
                "sampling": {"ensembles": 4, "walkers": 32, "initial_steps": 6000, "max_steps": 20000, "burn_fraction": .5}}
        np.savez(directory / "likelihood.npz", coefficients=coef, p_templates=templates,
                 target=target, covariance=covariance, chol=chol, prior_mean=prior_mean,
                 prior_whitener=prior_whitener, bounds=bounds, centre=centre, indices=idx)
        save_json(directory / "contract.json", meta)
        case = Case(directory)
        fit = Minuit(case.objective, *centre, name=NAMES)
        fit.errordef = 1
        for n, limit, error in zip(NAMES, bounds, (10, .1, 1000, .1, .1, .1, .1)):
            fit.limits[n] = tuple(limit)
            fit.errors[n] = error
        fit.tol = 1e-7
        fit.strategy = 2
        fit.migrad(ncall=100000)
        fit.hesse()
        if not fit.valid or not fit.fmin.has_posdef_covar:
            raise RuntimeError(f"MAP/HESSE 无效：{fit.fmin}")
        centre = np.array(fit.values)
        transform = np.linalg.cholesky(np.array(fit.covariance))
        # 直接模型再次检查新 MAP，避免仅拟合局部插值误差。
        mp = dict(zip(("b1", *BNAMES), centre[[1, 3, 4, 5, 6]]))
        pred = basis_prediction(centre, coef, templates, cfg["p"])[0]
        pred[12:] = model(centre[0], mp, idx)
        w = solve_triangular(chol, pred-target, lower=True)
        pw = prior_whitener @ (centre[[1, 3, 4, 5, 6]]-prior_mean)
        assert abs(float(w@w + original.PRIOR_WEIGHT*(pw@pw))-fit.fval) < 1e-6
        with np.load(directory / "likelihood.npz") as z:
            arrays = dict(z)
        arrays.update(centre=centre, transform=transform)
        np.savez(directory / "likelihood.npz", **arrays)
        meta.update(map=dict(zip(NAMES, centre)), map_objective=fit.fval,
                    likelihood_sha256=digest(directory / "likelihood.npz"))
        save_json(directory / "contract.json", meta)
        print("PREPARED", method, cut, "fNL", centre[0], "parity", data_parity, "poly", max_error, flush=True)


def diagnostics(case, chain, accepted):
    retained = case.centre + chain[:, chain.shape[1]//2:] @ case.transform.T
    result = independent_diagnostics(retained)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau = np.array([emcee.autocorr.integrated_time(c, quiet=True) for c in retained])
    flat = retained.reshape(-1, NDIM)
    scale = flat.std(axis=0)
    half = retained.shape[1]//2
    shift = abs(np.median(retained[:, :half].reshape(-1, NDIM), axis=0)-np.median(retained[:, half:].reshape(-1, NDIM), axis=0))/scale
    spread = np.ptp(np.median(retained, axis=(1, 2)), axis=0)/scale
    ratio = retained.shape[1]/np.max(tau, axis=0)
    edges = case.bounds[:, 0] + .02*np.ptp(case.bounds, axis=1)
    upper = case.bounds[:, 1] - .02*np.ptp(case.bounds, axis=1)
    mass = np.mean((flat < edges) | (flat > upper), axis=0)
    acceptance = accepted.mean()/chain.shape[1]
    checks = {"rhat": bool(np.all(result["rank_normalized_split_rhat"] < 1.01)),
              "fnl_ess": bool(min(result["bulk_ess"][0], result["tail_ess"][0]) > 2000),
              "other_ess": bool(min(np.min(result["bulk_ess"][1:]), np.min(result["tail_ess"][1:])) > 1000),
              "length_tau": bool(np.all(ratio > 50)), "half_shift": bool(np.all(shift < .1)),
              "ensemble_spread": bool(np.all(spread < .2)), "acceptance": bool(.1 < acceptance < .7)}
    result.update(tau=tau, retained_length_over_tau=ratio, half_shift_sigma=shift,
                  ensemble_spread_sigma=spread, boundary_mass=mass, acceptance=acceptance,
                  checks=checks, passed=all(checks.values()), boundary_warning=bool(np.any(mass > .01)),
                  steps=chain.shape[1], burn=chain.shape[1]//2)
    # 边界质量独立报告：撞先验边界不伪称未收敛，也不隐瞒先验敏感性。
    return result, flat


def sample(case_name, smoke=False):
    directory = OUT / case_name
    case = Case(directory)
    assert digest(directory / "likelihood.npz") == case.meta["likelihood_sha256"]
    chain_file = directory / ("smoke_chain.h5" if smoke else "chain.h5")
    max_steps = 200 if smoke else 20000
    first_steps = 200 if smoke else 6000
    for goal in range(first_steps, max_steps+1, 2000):
        for ensemble in range(ENSEMBLES):
            rng = np.random.RandomState(20260930 + ensemble*100 + int(case.meta["kmax_B"]*100) + (case.meta["method"] != "pre")*1000)
            with h5py.File(chain_file, "a") as h:
                name = f"ensemble_{ensemble}"
                g = h.require_group(name)
                if "chain" not in g:
                    g.create_dataset("chain", shape=(0, WALKERS, NDIM), maxshape=(None, WALKERS, NDIM), chunks=(100, WALKERS, NDIM), dtype="f8")
                    g.create_dataset("log_prob", shape=(0, WALKERS), maxshape=(None, WALKERS), chunks=(100, WALKERS), dtype="f8")
                    g.attrs["accepted"] = np.zeros(WALKERS)
                old_steps = len(g["chain"])
                if old_steps >= goal:
                    continue
                if old_steps:
                    rs = json.loads(g.attrs["random_state"])
                    rs = (rs[0], np.array(rs[1], dtype=np.uint32), rs[2], rs[3], rs[4])
                    state = emcee.State(g["chain"][-1], log_prob=g["log_prob"][-1], random_state=rs)
                else:
                    positions = rng.normal(size=(WALKERS, NDIM))*(.5+ensemble*.5)
                    for j in range(WALKERS):
                        while not np.isfinite(case.logp(positions[j])[0]):
                            positions[j] = rng.normal(size=NDIM)*.5
                    state = emcee.State(positions, random_state=rng.get_state())
            while old_steps < goal:
                count = min(1000, goal-old_steps)
                sampler = emcee.EnsembleSampler(WALKERS, NDIM, case.logp, vectorize=True,
                    moves=[(emcee.moves.StretchMove(a=2), .8), (emcee.moves.DEMove(), .2)])
                state = sampler.run_mcmc(state, count, progress=False)
                with h5py.File(chain_file, "a") as h:
                    g = h[f"ensemble_{ensemble}"]
                    for key, val in (("chain", sampler.get_chain()), ("log_prob", sampler.get_log_prob())):
                        g[key].resize(old_steps+count, axis=0)
                        g[key][old_steps:] = val
                    g.attrs["accepted"] += sampler.acceptance_fraction*count
                    g.attrs["random_state"] = json.dumps(clean(state.random_state))
                old_steps += count
                print("SAMPLE", case_name, "ensemble", ensemble, "steps", old_steps, flush=True)
        if smoke:
            print("SMOKE PASS", case_name, flush=True)
            return
        with h5py.File(chain_file) as h:
            chain = np.array([h[f"ensemble_{i}/chain"][:] for i in range(ENSEMBLES)])
            accepted = np.array([h[f"ensemble_{i}"].attrs["accepted"] for i in range(ENSEMBLES)])
        diag, flat = diagnostics(case, chain, accepted)
        quantiles = np.quantile(flat, [.025, .16, .5, .84, .975], axis=0)
        summary = {n: dict(zip(("q025", "q16", "median", "q84", "q975"), quantiles[:, j])) for j, n in enumerate(NAMES)}
        result = {"case": case_name, "status": "converged" if diag["passed"] else "not_converged",
                  "contract": case.meta, "diagnostics": diag, "posterior": summary,
                  "fNL_half_width": (quantiles[3, 0]-quantiles[1, 0])/2,
                  "truth_in_68": bool(quantiles[1, 0] <= 100 <= quantiles[3, 0]),
                  "truth_in_95": bool(quantiles[0, 0] <= 100 <= quantiles[4, 0])}
        # 独立于插值的 MAP 数据 chi2 与当前后验中位数数据 chi2 分开报告。
        data, prior = case.terms(quantiles[2])
        result.update(median_data_chi2=data[0], median_prior_chi2=prior[0],
                      map_gof_p_value=case.meta["reference_MINOS"]["gof_p_value_data_only"])
        save_json(directory / "RESULTS.json", result)
        selection = np.random.default_rng(930).choice(len(flat), min(40000, len(flat)), replace=False)
        np.savez_compressed(directory / "posterior.npz", samples=flat[selection], names=np.array(NAMES))
        print("DIAGNOSTICS", case_name, goal, clean(diag["checks"]), "fNL", clean(summary["fNL"]), flush=True)
        if diag["passed"]:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "sample", "smoke"))
    parser.add_argument("--method", choices=("pre", "sto_marisa"))
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.method)
    else:
        sample(args.case, args.action == "smoke")


if __name__ == "__main__":
    main()
