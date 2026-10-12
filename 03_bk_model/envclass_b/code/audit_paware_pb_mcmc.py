#!/usr/bin/env python3
"""复核新链与原始直接模型的一致性，并汇总本轮四组结果。

只读取原模型和原 MINOS 结果；图只输出 PDF。MCMC 收敛与物理拟合优度
分别报告：收敛不能补救模型失配，固定 p 的条件后验不传播校准误差。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.linalg import solve_triangular
import run_paware_pb_mcmc as run
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    rows, audits, fnl_chains = [], [], {}
    for method in ("pre", "sto_marisa"):
        low, scan, runtime = run.original.configure_models()
        cfg = run.original.CASES[method]
        low.bphi_from_b1 = lambda b: 2*low.DELTA_C*(float(b)-cfg["p"])
        if method == "sto_marisa":
            low.B_REC, low.P_REC = cfg["b_rec"], cfg["p_rec"]
            low.BPHI_REC = 2*low.DELTA_C*(low.B_REC-low.P_REC)
        model = scan.DirectFiniteScanB000Model(runtime, cfg["method"])
        for cut in run.CUTS:
            name = run.tag(method, cut)
            directory = run.OUT / name
            case = run.Case(directory)
            result = json.loads((directory / "RESULTS.json").read_text())
            assert run.digest(directory / "likelihood.npz") == case.meta["likelihood_sha256"]
            assert run.digest(Path(run.original.__file__)) == case.meta["original_runner_sha256"]
            with h5py.File(directory / "chain.h5") as h:
                coords = np.array([h[f"ensemble_{i}/chain"][:] for i in range(4)])
                logp = np.array([h[f"ensemble_{i}/log_prob"][:] for i in range(4)])
                accepted = np.array([h[f"ensemble_{i}"].attrs["accepted"] for i in range(4)])
            diag, flat = run.diagnostics(case, coords, accepted)
            quantiles = np.quantile(flat, [.025, .16, .5, .84, .975], axis=0)
            stored = np.array([[result["posterior"][n][q] for n in run.NAMES] for q in ("q025", "q16", "median", "q84", "q975")])
            q_error = float(np.max(np.abs(quantiles-stored)))
            assert q_error < 1e-10
            rng = np.random.default_rng(9842)
            select = rng.choice(coords.reshape(-1, 7).shape[0], 100, replace=False)
            x = case.centre + coords.reshape(-1, 7)[select] @ case.transform.T
            target_log = logp.reshape(-1)[select]
            direct_log = []
            for v in x:
                f, b1, shot = v[:3]
                bp = 2*low.DELTA_C*(b1-cfg["p"])
                pl, pm, pq = case.p_templates
                pp = b1*b1*pl + 2*f*b1*bp*pm + (f*bp)**2*pq + shot
                mp = dict(zip(("b1", *run.BNAMES), v[[1, 3, 4, 5, 6]]))
                prediction = np.concatenate((pp, model(f, mp, case.indices)))
                w = solve_triangular(case.chol, prediction-case.target, lower=True)
                p = model.base_context.prior.whitener @ (v[[1, 3, 4, 5, 6]]-model.base_context.prior.mean)
                direct_log.append(-.5*(w@w + run.original.PRIOR_WEIGHT*(p@p)))
            error = float(np.max(np.abs(np.array(direct_log)-target_log)))
            assert error < 1e-7
            assert bool(diag["passed"]) == bool(result["diagnostics"]["passed"])
            assert case.meta["full_PB_cross_covariance"] and case.meta["cross_correlation_max"] > 0
            audit = {"case": name, "passed": True, "direct_model_logp_max_error": error,
                     "posterior_quantile_max_error": q_error, "diagnostics_reproduced": True,
                     "chain_sha256": run.digest(directory / "chain.h5")}
            run.save_json(directory / "AUDIT.json", audit)
            audits.append(audit)
            rows.append({"method": method, "kmax_B": cut, "p": cfg["p"],
                         "fNL": result["posterior"]["fNL"], "half_width": result["fNL_half_width"],
                         "truth_in_68": result["truth_in_68"], "truth_in_95": result["truth_in_95"],
                         "converged": diag["passed"], "max_rhat": np.max(diag["rank_normalized_split_rhat"]),
                         "fNL_bulk_ESS": diag["bulk_ess"][0], "fNL_tail_ESS": diag["tail_ess"][0],
                         "min_length_over_tau": np.min(diag["retained_length_over_tau"]),
                         "boundary_warning": diag["boundary_warning"], "steps": diag["steps"],
                         "MAP_GoF": result["map_gof_p_value"],
                         "MINOS_reference": case.meta["reference_MINOS"], "audit": audit})
            physical = case.centre + coords @ case.transform.T
            fnl_chains[name] = physical[..., 0]
            print("AUDIT PASS", name, "logp error", error, flush=True)
    summary = {"status": "complete" if all(r["converged"] for r in rows) else "not_all_converged",
               "methods": ["pre", "sto_marisa"], "skipped": {"marisa": "新版缺逐 realization P，无法构造完整 P-B 交叉协方差"},
               "volume": 42.875, "kmax_P": .08, "truth_fNL": 100,
               "p_treatment": "各方法校准值固定；未传播 p 的校准不确定性",
               "model": "沿用原 matched250 finite-tree；未加入 IR 或 Gaussian discrepancy",
               "rows": rows, "audits_passed": all(a["passed"] for a in audits)}
    run.save_json(run.OUT / "SUMMARY.json", summary)
    colors = {"pre": "#3569A8", "sto_marisa": "#D36A27"}
    labels = {"pre": "Pre", "sto_marisa": "STO + Marisa"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, cut in zip(axes, run.CUTS):
        for method in ("pre", "sto_marisa"):
            c = fnl_chains[run.tag(method, cut)]
            values = c[:, c.shape[1]//2:].ravel()
            ax.hist(values, bins=np.linspace(60, 165, 95), density=True, histtype="step", lw=1.7, color=colors[method], label=labels[method])
        ax.axvline(100, color="black", ls="--", lw=1, label="Input = 100")
        ax.set(xlabel=r"$f_{\rm NL}$", ylabel="Posterior density", title=rf"$k_{{\max}}^B={cut:.2f}\ h\,\mathrm{{Mpc}}^{{-1}}$")
        ax.legend(frameon=False)
    fig.suptitle(r"Calibrated-$p$ P+B MCMC | full cross covariance | $V=42.875$")
    fig.savefig(run.OUT / "fnl_posterior_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for ax, row in zip(axes.ravel(), rows):
        c = fnl_chains[run.tag(row["method"], row["kmax_B"])]
        for ensemble in range(4):
            ax.plot(np.arange(0, c.shape[1], 10), np.mean(c[ensemble, ::10], axis=1), lw=.6, alpha=.8, label=f"Ensemble {ensemble+1}")
        ax.axvline(c.shape[1]//2, ls=":", color="black")
        ax.axhline(100, ls="--", color="black", lw=.7)
        ax.set(title=f"{labels[row['method']]}, kB={row['kmax_B']:.2f}", xlabel="Step", ylabel=r"Walker-mean $f_{\rm NL}$")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(run.OUT / "fnl_ensemble_traces.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
