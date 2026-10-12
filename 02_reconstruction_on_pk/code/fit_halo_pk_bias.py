#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
代码大纲
1. 读取 aggregate_fit_quijote_halo_z0p5_pk_bias.py 生成的 500-realization P0(k) 汇总。
2. 参考 1gpc-UNIT-fnl100-fastpm/agent/task.md ##3.1 的 desilike 功率谱拟合流程。
3. 对 Quijote halo fiducial/fnl0 在 k <= 0.1 h/Mpc 上拟合 bias。

拟合设定
- cosmology 固定为 Quijote fiducial: h=0.6711, Omega_m=0.3175, Omega_b=0.049, sigma8=0.834, ns=0.9624。
- fnl_loc 固定为 0。
- p 固定为 1.2。
- sn0 固定为 0，因为 jaxpower 输出的 pole.value() 已经是 shot-noise corrected 的 P0。
- 输出两个结果：
  1. b1_only: sigmas 固定 0，只拟合 b1。
  2. b1_sigmas: 同时拟合 b1 和 sigmas。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
DEFAULT_INPUT = PROJECT_ROOT / "02_reconstruction_on_pk/inputs/fiducial_pk_bias_fit_input.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "02_reconstruction_on_pk/results/fiducial_pk_bias_fit.json"
Z_SNAPSHOT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Quijote halo P0 bias with desilike.")
    parser.add_argument("--input-npz", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--kmax-fit", type=float, default=0.10)
    parser.add_argument("--kmin-fit", type=float, default=0.0)
    parser.add_argument("--p-fix", type=float, default=1.2)
    parser.add_argument("--niterations", type=int, default=12)
    return parser.parse_args()


def build_data_objects(data: dict[str, np.ndarray], kmin_fit: float, kmax_fit: float):
    """构造当前 desilike observable API 需要的 data vector 和 covariance。"""

    k = np.asarray(data["k"], dtype="f8")
    edges = np.asarray(data["edges"], dtype="f8")
    pk_samples = np.asarray(data["pk_samples"], dtype="f8")
    pk_mean = np.asarray(data["pk_mean"], dtype="f8")
    fit_mask = (k > float(kmin_fit)) & (k <= float(kmax_fit)) & np.isfinite(pk_mean)
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("Not enough fit bins")

    fit_k = k[fit_mask]
    fit_edges = np.concatenate([edges[:-1][fit_mask], [edges[1:][fit_mask][-1]]])
    fit_pk_mean = pk_mean[fit_mask]
    fit_samples = pk_samples[:, fit_mask]
    covariance_mean = np.cov(fit_samples, rowvar=False, ddof=1) / float(fit_samples.shape[0])
    return {
        "data_vector": fit_pk_mean,
        "covariance_mean": covariance_mean,
        "fit_k": fit_k,
        "fit_edges": fit_edges,
        "fit_pk_mean": fit_pk_mean,
        "fit_samples": fit_samples,
    }


def make_likelihood(fit_k: np.ndarray, data_vector: np.ndarray, covariance_mean: np.ndarray, *, p_fix: float, sigmas_free: bool):
    """建立 desilike likelihood。"""

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=Z_SNAPSHOT, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")
    theory.init.params["fnl_loc"].update(fixed=True, value=0.0)
    theory.init.params["p"].update(fixed=True, value=float(p_fix))
    theory.init.params["sn0"].update(fixed=True, value=0.0)
    theory.init.params["sigmas"].update(fixed=not bool(sigmas_free), value=0.0)
    observable = TracerPowerSpectrumMultipolesObservable(
        data=np.asarray(data_vector, dtype="f8"),
        covariance=np.asarray(covariance_mean, dtype="f8"),
        k=[np.asarray(fit_k, dtype="f8")],
        ells=[0],
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()
    likelihood.all_params["fnl_loc"].update(fixed=True, value=0.0)
    likelihood.all_params["p"].update(fixed=True, value=float(p_fix))
    likelihood.all_params["sn0"].update(fixed=True, value=0.0)
    likelihood.all_params["sigmas"].update(fixed=not bool(sigmas_free), value=0.0)
    likelihood.all_params["b1"].update(fixed=False, value=2.45)
    return likelihood


def run_fit(fit_k: np.ndarray, data_vector: np.ndarray, covariance_mean: np.ndarray, *, p_fix: float, sigmas_free: bool, niterations: int) -> dict[str, Any]:
    """运行 MinuitProfiler 并返回 best fit。"""

    from desilike.profilers import MinuitProfiler

    likelihood = make_likelihood(fit_k, data_vector, covariance_mean, p_fix=p_fix, sigmas_free=sigmas_free)
    profiler = MinuitProfiler(likelihood, seed=66)
    profiles = profiler.maximize(niterations=int(niterations))
    best = profiles.bestfit.choice(input=True)
    result = {
        "b1": float(best["b1"]),
        "fnl_loc": float(best["fnl_loc"]),
        "p": float(best["p"]),
        "sn0": float(best["sn0"]),
        "sigmas": float(best["sigmas"]),
        "sigmas_free": bool(sigmas_free),
    }
    try:
        result["logposterior"] = float(profiles.bestfit.choice()["logposterior"])
    except Exception:
        pass
    return result


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz)
    objects = build_data_objects(data, float(args.kmin_fit), float(args.kmax_fit))
    b1_only = run_fit(
        objects["fit_k"],
        objects["data_vector"],
        objects["covariance_mean"],
        p_fix=float(args.p_fix),
        sigmas_free=False,
        niterations=int(args.niterations),
    )
    b1_sigmas = run_fit(
        objects["fit_k"],
        objects["data_vector"],
        objects["covariance_mean"],
        p_fix=float(args.p_fix),
        sigmas_free=True,
        niterations=int(args.niterations),
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_npz": str(Path(args.input_npz).resolve()),
        "kmin_fit": float(args.kmin_fit),
        "kmax_fit": float(args.kmax_fit),
        "n_fit_bins": int(objects["fit_k"].size),
        "n_realizations": int(np.asarray(data["pk_samples"]).shape[0]),
        "fit_k": [float(x) for x in objects["fit_k"]],
        "fit_edges": [[float(a), float(b)] for a, b in zip(objects["fit_edges"][:-1], objects["fit_edges"][1:])],
        "fixed": {"fnl_loc": 0.0, "p": float(args.p_fix), "sn0": 0.0},
        "b1_only": b1_only,
        "b1_sigmas": b1_sigmas,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
