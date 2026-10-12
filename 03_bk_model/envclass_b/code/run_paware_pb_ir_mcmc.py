#!/usr/bin/env python3
"""模块大纲：冻结六组新似然；复用旧采样器；独立目录；逐项数据/先验一致性。

no-IR 只修正重构分母。IR 将同投影的 IR-minus-fixed 理论差加入正确基线，
不改 covariance、prior、bounds、p、体积或数据。所有生成步骤先验收后采样。
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2
import run_paware_pb_mcmc as legacy
from paware_pb_corrected_model import CorrectedModel, BNAMES

OLD = legacy.OUT
OUT = OLD.parent/"paware_pb_ir_brecfix_k012_k014_v2"
legacy.OUT = OUT
OLD_PREDICT = legacy.basis_prediction


def predict(values, coefficients, p_templates, p_halo):
    values = np.atleast_2d(values)
    if p_templates.ndim == 2:
        return OLD_PREDICT(values, coefficients, p_templates, p_halo)
    f, b, shot = values[:, :3].T
    cb = np.polynomial.chebyshev.chebvander(2*(b-.1)/7.9-1, coefficients.shape[0]-1)
    nu = np.column_stack((np.ones(len(values)), values[:, 3:]))
    ff = np.column_stack((np.ones(len(values)), f, f*f))
    features = (cb[:, :, None, None]*nu[:, None, :, None]*ff[:, None, None, :]).reshape(len(values), -1)
    bb = features @ coefficients.reshape(-1, coefficients.shape[-1])
    pc = np.polynomial.chebyshev.chebvander(2*(b-.1)/7.9-1, p_templates.shape[0]-1)
    pp = (pc[:, :, None]*ff[:, None, :]).reshape(len(values), -1) @ p_templates.reshape(-1, 12)
    return np.concatenate((pp+shot[:, None], bb), axis=1)


legacy.basis_prediction = predict


def case_name(method, ir, cut):
    return f"{method}_{'ir' if ir else 'noir'}_k{cut:.2f}".replace(".", "p")


def freeze(method, ir=False):
    degree = 6
    delta = None
    if ir:
        # 必须先加载 v2 IR 包，再建立会导入旧 vendor 包的基线模型。
        # 仅调整导入顺序，不修改已通过验收的模板及其生成代码哈希。
        from paware_pb_ir_templates import load_delta, TEMPLATES, LIBRARY
        delta = load_delta(method)
        delta.contract = dict(delta.contract)
        delta.contract["frozen_assets_sha256"] = {
            str(path): legacy.digest(path) for path in (
                TEMPLATES/f"{method}_accepted.npz", TEMPLATES/f"{method}_accepted.json",
                TEMPLATES/f"{method}_pk.npz", TEMPLATES/"variance.npz", LIBRARY,
                Path(__file__).with_name("paware_pb_ir_templates.py"))}
        degree = delta.degree
    model = CorrectedModel(method)
    nodes = np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
    bnodes = .1+7.9*(nodes+1)/2
    sampled = np.array([model.design(b) for b in bnodes])
    if ir:
        for i, b in enumerate(bnodes):
            sampled[i, :, :, model.indices] += delta.b_design(b).transpose(2, 0, 1)
    coeff = np.linalg.solve(np.polynomial.chebyshev.chebvander(nodes, degree),
                            sampled.reshape(degree+1, -1)).reshape(degree+1, 5, 3, 120)
    for cut in legacy.CUTS:
        name = case_name(method, ir, cut)
        directory = OUT/name
        if directory.exists():
            raise FileExistsError(f"禁止覆盖已有 case: {directory}")
        old = OLD/legacy.tag(method, cut)
        contract = json.loads((old/"contract.json").read_text())
        with np.load(old/"likelihood.npz") as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        assert legacy.digest(old/"likelihood.npz") == contract["likelihood_sha256"]
        idx = arrays["indices"]
        arrays["coefficients"] = coeff[..., idx]
        if ir:
            p_sample = []
            for b in bnodes:
                bp = 2*1.686*(b-model.p)
                original = arrays["p_templates"]*np.array([b*b, 2*b*bp, bp*bp])[:, None]
                p_sample.append(original+delta.p_design(b))
            arrays["p_templates"] = np.linalg.solve(np.polynomial.chebyshev.chebvander(nodes, degree),
                                                    np.array(p_sample).reshape(degree+1, -1)).reshape(degree+1, 3, 12)
        def prediction(x):
            return predict(x, arrays["coefficients"], arrays["p_templates"], model.p)
        def terms(x):
            x = np.atleast_2d(x)
            w = solve_triangular(arrays["chol"], (prediction(x)-arrays["target"]).T, lower=True).T
            pr = (x[:, [1, 3, 4, 5, 6]]-arrays["prior_mean"]) @ arrays["prior_whitener"].T
            return np.sum(w*w, axis=1), legacy.original.PRIOR_WEIGHT*np.sum(pr*pr, axis=1)
        def objective(*x):
            data, prior = terms(x)
            return float(data[0]+prior[0])
        candidates = []
        for f0 in (arrays["centre"][0], 0., 100., 160.):
            start = arrays["centre"].copy(); start[0] = f0
            fit = Minuit(objective, *start, name=legacy.NAMES)
            fit.errordef = 1
            for n, limit, error in zip(legacy.NAMES, arrays["bounds"], (10, .1, 1000, .1, .1, .1, .1)):
                fit.limits[n] = tuple(limit); fit.errors[n] = error
            fit.strategy = 2; fit.tol = 1.e-7
            fit.migrad(ncall=100000); fit.hesse()
            if fit.valid and fit.fmin.has_posdef_covar:
                candidates.append(fit)
        if not candidates:
            raise RuntimeError(f"MAP/HESSE 未通过: {name}")
        fit = min(candidates, key=lambda x: x.fval)
        centre = np.array(fit.values)
        arrays.update(centre=centre, transform=np.linalg.cholesky(np.array(fit.covariance)))
        # 宽先验域直接模型验收，不能只在 MAP 附近验收 b1 插值。
        rng = np.random.default_rng(930)
        test = np.tile(centre, (161, 1))
        test[1:, 0] = rng.uniform(-290, 290, 160)
        test[1:, 1] = rng.uniform(.101, 7.999, 160)
        test[1:, 3:5] = rng.uniform([-3.9, -1.9], [5.9, 3.9], (160, 2))
        fast = prediction(test); direct = fast.copy()
        lookup = np.array([list(model.indices).index(i) for i in idx])
        for i, v in enumerate(test):
            mp = dict(zip(model.names, v[[1, 3, 4, 5, 6]]))
            direct[i, 12:] = model(v[0], mp, idx)
            if ir:
                # 使用独立 holdout 插值合同；模板阶段另做节点原生直接验收。
                d = delta.b_design(v[1], validation=True)[:, :, lookup]
                direct[i, 12:] += np.einsum("a,j,aji->i", np.r_[1., v[3:]], [1., v[0], v[0]**2], d)
                with np.load(old/"likelihood.npz") as z:
                    p = OLD_PREDICT(v, np.zeros((1, 5, 3, len(idx))), z["p_templates"], model.p)[0, :12]
                direct[i, :12] = p + np.array([1., v[0], v[0]**2]) @ delta.p_design(v[1], validation=True)
        error = float(np.max(np.abs(solve_triangular(arrays["chol"], (fast-direct).T, lower=True))))
        if error > 1.e-5:
            raise RuntimeError(f"插值未验收 {name}: {error}")
        unchanged = {}
        with np.load(old/"likelihood.npz") as z:
            for key in ("target", "covariance", "chol", "prior_mean", "prior_whitener", "bounds", "indices"):
                unchanged[key] = bool(np.array_equal(arrays[key], z[key]))
        assert all(unchanged.values())
        data_chi2, prior_chi2 = terms(centre)
        meta = dict(contract)
        meta.update(method=method, case=name, IR_resummation=ir, baseline_reconstruction_corrected=method=="sto_marisa",
                    reconstruction=model.provenance(), baseline_contract=legacy.digest(old/"contract.json"),
                    baseline_likelihood_sha256=legacy.digest(old/"likelihood.npz"), unchanged_arrays=unchanged,
                    polynomial_degree=degree, polynomial_max_whitened_error=error,
                    map=dict(zip(legacy.NAMES, centre)), map_objective=fit.fval,
                    map_data_chi2=float(data_chi2[0]), map_prior_chi2=float(prior_chi2[0]),
                    runner_sha256=legacy.digest(Path(__file__)), corrected_model_sha256=legacy.digest(Path(__file__).with_name("paware_pb_corrected_model.py")))
        meta["old_reference_MINOS"] = meta["reference_MINOS"]
        meta["reference_MINOS"] = {"gof_p_value_data_only": float(chi2.sf(data_chi2[0], len(arrays["target"])-7)),
                                   "note": "new MAP GoF; no new MINOS interval is claimed"}
        if ir:
            meta["IR_contract"] = delta.contract
        directory.mkdir(parents=True)
        np.savez(directory/"likelihood.npz", **arrays)
        meta["likelihood_sha256"] = legacy.digest(directory/"likelihood.npz")
        legacy.save_json(directory/"contract.json", meta)
        print("PREPARED", name, "fNL", centre[0], "chi2", data_chi2[0], "poly_error", error, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "sample", "smoke"))
    parser.add_argument("--method", choices=("pre", "sto_marisa"))
    parser.add_argument("--ir", action="store_true")
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.method == "pre" and not args.ir:
            raise ValueError("pre no-IR 使用保留的旧链，不重复提交")
        freeze(args.method, args.ir)
    else:
        legacy.sample(args.case, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
