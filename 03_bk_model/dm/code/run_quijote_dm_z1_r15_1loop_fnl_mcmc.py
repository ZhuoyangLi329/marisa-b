#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 Quijote DM z=1、R=15 的有限 fNL 模型运行一维 MCMC recovery。

执行大纲
========
1. 复用已经审计的 ``matched.npz`` 理论以及公共 446-realization 数据；
2. 对图中的四个 kmax、pre/post 和三个 input 分别构造同一 Gaussian likelihood；
3. 在统一先验 ``-1000 < fNL < 1000`` 内运行四个独立 emcee ensemble；
4. 用独立 ensemble 的 rank/folded R-hat、bulk/tail ESS、自相关时间、
   半链稳定性、接受率和 prior 边界质量决定是否自动延长；
5. 对一维后验做高分辨率确定性积分，独立核对 MCMC 的 16/50/84 分位数；
6. 原子保存 JSON 和可复核的 retained chains，并用 MCMC equal-tail 68%
   credible interval 重画原四面板 PDF。

统计约定
========
- likelihood 与旧 profile 完全相同：``B=G+fNL*L+fNL^2*Q``；
- target 是每个 cosmology 的 446-realization mean；
- covariance 是对应 stage 的 fiducial 单盒 sample covariance，V=1，
  precision 使用维数相关的 Hartlap 因子，不是 mean covariance；
- 最低两个 diagonal k bin 始终排除；kmax 只限制实测的 k1/k2；
- 图中的中心是 MCMC posterior median，误差棒是 16%--84% equal-tail CI；
- MAP/profile 只用于初始化和审计，不再用于图的中心或误差棒。

本任务是一维后验，本可以直接积分；仍按用户要求真正运行 MCMC，并把精确
积分仅用作交叉验证。四个 ensemble 才是独立链；同一 ensemble 内的 walkers
相互作用，诊断从不把它们伪装成独立 chains。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import warnings
from typing import Any

# 登录节点数值库保持单线程；emcee 的 walker 更新本身已矢量化。
for _key in (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lzy-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "codes/marisa_b/python"))

import run_quijote_dm_z1_r15_1loop_fnl_profiles as profile
from marisa_b import arviz_compat as az

ANALYSIS = ROOT / "analysis/quijote_dm_z1_r15_matched_fnl_model_v3"
THEORY = ANALYSIS / "matched.npz"
OUTPUT_JSON = ANALYSIS / "quijote_dm_z1_r15_1loop_b000_fnl_mcmc_v4.json"
OUTPUT_CHAINS = ANALYSIS / "quijote_dm_z1_r15_1loop_b000_fnl_mcmc_v4_chains.npz"
OUTPUT_PDF = ROOT / "9.11meeting/dm/quijote_dm_z1_r15_1loop_b000_fnl_profile_matched_finite_v3.pdf"

STAGES = profile.STAGES
TARGETS = profile.TARGETS
KMAX = profile.PLOT_RECOVERY_KMAX_GRID
PRIOR = (-1000.0, 1000.0)
ENSEMBLES = 4
WALKERS = 16
INITIAL_STEPS = 8000
INCREMENT = 4000
MAX_STEPS = 24000
BURN_FRACTION = 0.5
SEED = 20260908
EXACT_GRID_SIZE = 200001


def parse_args() -> argparse.Namespace:
    """解析采样规模和输出路径；默认值对应本轮会议 release。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-bundle", type=Path, default=THEORY)
    parser.add_argument("--simulation-npz", type=Path, default=profile.SIMULATION_NPZ)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-chains", type=Path, default=OUTPUT_CHAINS)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--ensembles", type=int, default=ENSEMBLES)
    parser.add_argument("--walkers", type=int, default=WALKERS)
    parser.add_argument("--initial-steps", type=int, default=INITIAL_STEPS)
    parser.add_argument("--increment", type=int, default=INCREMENT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--burn-fraction", type=float, default=BURN_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exact-grid-size", type=int, default=EXACT_GRID_SIZE)
    parser.add_argument("--redshift", type=float, default=1.0)
    parser.add_argument("--smoothing-radius", type=float, default=15.0)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--exclude-first-diagonal-bins", type=int, default=2)
    parser.add_argument(
        "--exclude-last-diagonal-bins", type=int, default=0,
        help=("from the registered k<=0.2 diagonal range, exclude the highest N bins "
              "globally; do not repeat the exclusion for every smaller kmax"),
    )
    args = parser.parse_args()
    profile.require(args.ensembles == 4, "release diagnostics require four independent ensembles")
    profile.require(args.walkers >= 8 and args.walkers % 2 == 0,
                    "walkers must be an even integer >= 8")
    profile.require(args.initial_steps >= 1000 and args.increment >= 1000,
                    "MCMC segments are too short")
    profile.require(args.max_steps >= args.initial_steps, "max_steps < initial_steps")
    profile.require(0.25 <= args.burn_fraction <= 0.75, "invalid burn fraction")
    profile.require(args.exact_grid_size >= 20001 and args.exact_grid_size % 2 == 1,
                    "exact-grid-size must be an odd integer >= 20001")
    profile.require(args.volume > 0.0, "volume must be positive")
    profile.require(args.exclude_first_diagonal_bins >= 0,
                    "exclude-first-diagonal-bins must be non-negative")
    profile.require(args.exclude_last_diagonal_bins >= 0,
                    "exclude-last-diagonal-bins must be non-negative")
    return args


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子保存 JSON，避免中断时留下看似完整的结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(profile.finite_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """一次性压缩保存全部 retained chains 和索引，完成后再改名。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any]:
    """返回路径、大小和 SHA256，供最终 contract 核对。"""

    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": profile.sha256(path)}


def pooled_rank_normalize(values: np.ndarray) -> np.ndarray:
    """对所有 ensemble/draw/walker 合并排序并映射到标准正态。"""

    values = np.asarray(values, dtype=np.float64)
    ranks = rankdata(values.reshape(-1), method="average")
    probabilities = (ranks - 0.375) / (values.size + 0.25)
    return norm.ppf(probabilities).reshape(values.shape)


def ensemble_scaled_ess(transformed: np.ndarray) -> tuple[float, float, float]:
    """计算保留 walker 同步相关的 ESS。

    参数 ``transformed`` 的形状是 (ensemble, draw, walker)。先对同一时刻
    互相作用的 walkers 求均值，只把四个 ensemble 当成独立 chain；随后用
    单 draw 方差与 ensemble-mean 方差之比恢复同一步的有效 walker 信息量。
    返回总 ESS、ensemble-iteration ESS 和信息倍率。
    """

    transformed = np.asarray(transformed, dtype=np.float64)
    series = np.mean(transformed, axis=2)
    iteration_ess = float(az.ess(series, method="identity", chain_axis=0, draw_axis=1))
    draw_variance = float(np.var(transformed, ddof=1))
    series_variance = float(np.var(series, ddof=1))
    if not math.isfinite(iteration_ess) or draw_variance <= 0 or series_variance <= 0:
        return math.nan, iteration_ess, math.nan
    multiplier = draw_variance / series_variance
    return min(float(transformed.size), iteration_ess * multiplier), iteration_ess, multiplier


def exact_posterior(
    target: np.ndarray,
    gaussian: np.ndarray,
    response: np.ndarray,
    quadratic: np.ndarray,
    precision: np.ndarray,
    grid_size: int,
) -> dict[str, Any]:
    """在统一 prior 上确定性积分一维后验，独立复核 MCMC 分位数和边界质量。"""

    grid = np.linspace(PRIOR[0], PRIOR[1], grid_size)
    residual = (target[None, :] - gaussian[None, :] - grid[:, None] * response[None, :]
                - grid[:, None] ** 2 * quadratic[None, :])
    chi2 = np.einsum("ni,ij,nj->n", residual, precision, residual)
    relative = np.exp(-0.5 * (chi2 - np.min(chi2)))
    # 梯形累计积分；端点给半权重，避免用简单 cumsum 引入一个 grid step 偏差。
    dx = float(grid[1] - grid[0])
    cumulative = np.empty_like(relative)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(0.5 * (relative[:-1] + relative[1:]) * dx)
    normalization = float(cumulative[-1])
    cumulative /= normalization
    probabilities = np.asarray([0.025, 0.16, 0.5, 0.84, 0.975])
    quantiles = np.interp(probabilities, cumulative, grid)
    normalized = relative / normalization
    mean = float(np.trapz(grid * normalized, grid))
    variance = float(np.trapz((grid - mean) ** 2 * normalized, grid))
    outer = 0.02 * (PRIOR[1] - PRIOR[0])
    boundary_mask = (grid <= PRIOR[0] + outer) | (grid >= PRIOR[1] - outer)
    # 两段不连续区域分别积分，不能跨越中央区域做一次 trapz。
    left = grid <= PRIOR[0] + outer
    right = grid >= PRIOR[1] - outer
    boundary_mass = float(np.trapz(normalized[left], grid[left])
                          + np.trapz(normalized[right], grid[right]))
    maxima = np.flatnonzero((relative[1:-1] > relative[:-2])
                            & (relative[1:-1] > relative[2:])) + 1
    significant_modes = [float(grid[index]) for index in maxima
                         if relative[index] / np.max(relative) > 1.e-4]
    return {
        "probabilities": probabilities,
        "quantiles": quantiles,
        "mean": mean,
        "standard_deviation": math.sqrt(variance),
        "map": float(grid[int(np.argmax(relative))]),
        "normalization": normalization,
        "prior_boundary_mass_outer_2_percent_each_side": boundary_mass,
        "significant_modes_above_1e_minus_4_relative_height": significant_modes,
        "grid_size": int(grid_size),
        "grid_step": dx,
        "all_density_finite": bool(np.all(np.isfinite(normalized))),
        "boundary_mask_points": int(np.count_nonzero(boundary_mask)),
    }


def independent_diagnostics(
    retained: np.ndarray,
    accepted_counts: np.ndarray,
    total_steps: int,
    exact: dict[str, Any],
) -> dict[str, Any]:
    """用四个独立 ensemble 计算 release convergence gates。

    ``retained`` 形状为 (4, draw, walker)。R-hat 使用先 rank/folded 变换、
    再对每个 ensemble 的 walker 均值做 split-Rhat；tail occupancy 另行检查。
    MCMC 分位数还必须与不依赖 chain 的确定性一维积分一致。
    """

    rank_values = pooled_rank_normalize(retained)
    folded_values = pooled_rank_normalize(np.abs(retained - np.median(retained)))
    rank_series = np.mean(rank_values, axis=2)
    folded_series = np.mean(folded_values, axis=2)
    location_rhat = float(az.rhat(rank_series, method="split", chain_axis=0, draw_axis=1))
    folded_rhat = float(az.rhat(folded_series, method="split", chain_axis=0, draw_axis=1))
    q05, q95 = np.quantile(retained, (0.05, 0.95))
    low = (retained <= q05).astype(np.float64)
    high = (retained >= q95).astype(np.float64)
    tail_fractions = np.stack((np.mean(low, axis=2), np.mean(high, axis=2)), axis=-1)
    tail_rhat = np.asarray(az.rhat(tail_fractions, method="rank", chain_axis=0, draw_axis=1))
    bulk_ess, bulk_iteration_ess, multiplier = ensemble_scaled_ess(rank_values)
    low_ess, low_iteration_ess, _ = ensemble_scaled_ess(low)
    high_ess, high_iteration_ess, _ = ensemble_scaled_ess(high)
    tail_ess = min(low_ess, high_ess)
    tau_rows = []
    for ensemble in range(retained.shape[0]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau = float(np.asarray(emcee.autocorr.integrated_time(
                    retained[ensemble, :, :, None], quiet=True)).reshape(-1)[0])
        except Exception:
            tau = math.nan
        tau_rows.append(tau)
    tau_max = float(np.nanmax(tau_rows))
    length_over_tau = float(retained.shape[1] / tau_max)
    flat = retained.reshape(-1)
    probabilities = np.asarray(exact["probabilities"])
    quantiles = np.quantile(flat, probabilities)
    standard_deviation = float(np.std(flat, ddof=1))
    midpoint = retained.shape[1] // 2
    first = retained[:, :midpoint].reshape(-1)
    second = retained[:, midpoint:].reshape(-1)
    half_shift = float(abs(np.median(first) - np.median(second)) / standard_deviation)
    ensemble_medians = np.median(retained, axis=(1, 2))
    ensemble_spread = float(np.ptp(ensemble_medians) / standard_deviation)
    boundary_width = 0.02 * (PRIOR[1] - PRIOR[0])
    boundary_mass = float(np.mean((flat <= PRIOR[0] + boundary_width)
                                  | (flat >= PRIOR[1] - boundary_width)))
    quantile_difference = quantiles - np.asarray(exact["quantiles"])
    maximum_quantile_difference_sigma = float(
        np.max(np.abs(quantile_difference)) / exact["standard_deviation"])
    mean_acceptance = float(np.mean(accepted_counts / total_steps))
    checks = {
        "rank_and_folded_split_rhat_below_1p01": max(location_rhat, folded_rhat) < 1.01,
        "tail_occupancy_rhat_below_1p01": float(np.max(tail_rhat)) < 1.01,
        "bulk_ess_above_5000": bulk_ess > 5000,
        "tail_ess_above_5000": tail_ess > 5000,
        "retained_length_above_50_tau": length_over_tau > 50,
        "half_chain_median_shift_below_0p1_sigma": half_shift < 0.1,
        "ensemble_median_spread_below_0p2_sigma": ensemble_spread < 0.2,
        "prior_boundary_mass_below_1_percent": boundary_mass < 0.01,
        "acceptance_between_0p10_and_0p80": 0.10 <= mean_acceptance <= 0.80,
        "mcmc_exact_quantiles_within_0p02_sigma": maximum_quantile_difference_sigma < 0.02,
        "exact_posterior_single_significant_mode": len(exact[
            "significant_modes_above_1e_minus_4_relative_height"]) == 1,
    }
    return {
        "rank_location_split_rhat": location_rhat,
        "rank_folded_split_rhat": folded_rhat,
        "rank_normalized_split_rhat": max(location_rhat, folded_rhat),
        "tail_occupancy_rhat": tail_rhat,
        "bulk_ess": bulk_ess,
        "tail_ess_5_95": tail_ess,
        "bulk_ensemble_iteration_ess": bulk_iteration_ess,
        "tail_ensemble_iteration_ess": min(low_iteration_ess, high_iteration_ess),
        "walker_information_multiplier": multiplier,
        "integrated_autocorrelation_time_by_ensemble": tau_rows,
        "maximum_tau": tau_max,
        "retained_length_over_tau": length_over_tau,
        "half_chain_median_shift_sigma": half_shift,
        "ensemble_medians": ensemble_medians,
        "ensemble_median_spread_sigma": ensemble_spread,
        "mean_acceptance_fraction": mean_acceptance,
        "prior_boundary_mass_outer_2_percent_each_side": boundary_mass,
        "quantile_probabilities": probabilities,
        "quantiles": quantiles,
        "posterior_mean": float(np.mean(flat)),
        "posterior_standard_deviation": standard_deviation,
        "mcmc_minus_exact_quantiles": quantile_difference,
        "maximum_mcmc_exact_quantile_difference_sigma": maximum_quantile_difference_sigma,
        "convergence_checks": checks,
        "pass": bool(all(checks.values())),
        "rhat_independence_unit": (
            "four separately seeded emcee ensembles; interacting walkers are averaged only "
            "after pooled rank/fold transforms"
        ),
    }


def initial_positions(
    map_value: float,
    scale: float,
    walkers: int,
    seed: int,
) -> np.ndarray:
    """在 MAP 附近为一个独立 ensemble 生成有限且不贴 prior 边界的 walkers。"""

    rng = np.random.default_rng(seed)
    positions = np.empty((walkers, 1))
    width = max(5.0, 0.5 * scale)
    for index in range(walkers):
        while True:
            candidate = float(map_value + rng.normal(scale=width))
            if PRIOR[0] + 1.e-8 < candidate < PRIOR[1] - 1.e-8:
                positions[index, 0] = candidate
                break
    return positions


def run_case(
    *,
    target: np.ndarray,
    gaussian: np.ndarray,
    response: np.ndarray,
    quadratic: np.ndarray,
    covariance: dict[str, Any],
    truth: float,
    case_seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    """运行一个 stage/kmax/input 的四 ensemble MCMC，并自动延长至通过。"""

    precision = np.asarray(covariance["precision"], dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gaussian = np.asarray(gaussian, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    quadratic = np.asarray(quadratic, dtype=np.float64)
    exact = exact_posterior(target, gaussian, response, quadratic, precision,
                            args.exact_grid_size)
    profile_fit = profile.quadratic_profile(
        target=target, gaussian=gaussian, response=response, quadratic=quadratic,
        covariance=covariance, truth=truth)

    def log_probability(theta: np.ndarray) -> np.ndarray:
        """矢量化均匀先验与 Gaussian log likelihood；输入为 (walker,1)。"""

        values = np.asarray(theta, dtype=np.float64)[:, 0]
        output = np.full(values.shape, -np.inf)
        valid = (values > PRIOR[0]) & (values < PRIOR[1])
        residual = (target[None, :] - gaussian[None, :] - values[valid, None] * response[None, :]
                    - values[valid, None] ** 2 * quadratic[None, :])
        output[valid] = -0.5 * np.einsum("ni,ij,nj->n", residual, precision, residual)
        return output

    samplers: list[emcee.EnsembleSampler] = []
    for ensemble in range(args.ensembles):
        seed = case_seed + 1009 * ensemble
        sampler = emcee.EnsembleSampler(
            args.walkers, 1, log_probability, vectorize=True,
            moves=[(emcee.moves.StretchMove(a=2.0), 0.8), (emcee.moves.DEMove(), 0.2)],
        )
        sampler.random_state = np.random.RandomState(seed + 17).get_state()
        start = initial_positions(profile_fit["profile_fNL"], profile_fit["profile_half_width"],
                                  args.walkers, seed)
        sampler.run_mcmc(start, args.initial_steps, progress=False,
                         skip_initial_state_check=True)
        samplers.append(sampler)

    while True:
        total_steps = int(samplers[0].iteration)
        burn = int(math.floor(args.burn_fraction * total_steps))
        retained = np.stack([
            np.asarray(sampler.get_chain(discard=burn), dtype=np.float64)[:, :, 0]
            for sampler in samplers
        ])
        accepted = np.stack([
            np.asarray(sampler.acceptance_fraction * total_steps, dtype=np.float64)
            for sampler in samplers
        ])
        diagnostics = independent_diagnostics(retained, accepted, total_steps, exact)
        if diagnostics["pass"] or total_steps >= args.max_steps:
            break
        extension = min(args.increment, args.max_steps - total_steps)
        for sampler in samplers:
            sampler.run_mcmc(None, extension, progress=False, skip_initial_state_check=True)

    if not diagnostics["pass"]:
        failed = [name for name, passed in diagnostics["convergence_checks"].items() if not passed]
        raise RuntimeError(f"MCMC nonconvergence at {total_steps} steps: {failed}")
    q025, q16, median, q84, q975 = diagnostics["quantiles"]
    record = {
        "input_fNL": float(truth),
        "posterior_median": median,
        "posterior_lower_68": q16,
        "posterior_upper_68": q84,
        "posterior_lower_95": q025,
        "posterior_upper_95": q975,
        "lower_error_68": median - q16,
        "upper_error_68": q84 - median,
        "truth_inside_equal_tail_68": bool(q16 <= truth <= q84),
        "truth_inside_equal_tail_95": bool(q025 <= truth <= q975),
        "profile_reference": {
            "map": profile_fit["profile_fNL"],
            "delta_chi2_1_interval": [profile_fit["profile_lower"], profile_fit["profile_upper"]],
        },
        "exact_quadrature_audit": exact,
        "diagnostics": diagnostics,
        "chain_layout": {
            "ensembles": args.ensembles,
            "walkers_per_ensemble": args.walkers,
            "total_steps_per_walker": total_steps,
            "burn_steps": burn,
            "retained_steps_per_walker": int(retained.shape[1]),
            "total_retained_draws": int(retained.size),
        },
    }
    return record, retained


def recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """对三个 posterior median 拟合 recovery 斜率和截距，并统计 coverage。"""

    truth = np.asarray([row["input_fNL"] for row in rows])
    median = np.asarray([row["posterior_median"] for row in rows])
    slope, intercept = np.linalg.lstsq(
        np.column_stack((truth, np.ones_like(truth))), median, rcond=None)[0]
    return {
        "slope": float(slope), "intercept": float(intercept),
        "maximum_absolute_bias": float(np.max(np.abs(median - truth))),
        "n_truth_inside_equal_tail_68": int(sum(row["truth_inside_equal_tail_68"] for row in rows)),
        "n_truth_inside_equal_tail_95": int(sum(row["truth_inside_equal_tail_95"] for row in rows)),
        "n_targets": len(rows),
    }


def make_plot(
    destination: Path,
    results: dict[str, dict[str, Any]],
    assets: dict[str, Any],
    redshift: float,
    smoothing_radius: float,
    volume: float,
    excluded_count: int,
    excluded_high_k: np.ndarray,
) -> None:
    """用 MCMC median 和非对称 equal-tail 68% CI 重画原四面板 recovery PDF。"""

    profile.configure_matplotlib()
    styles = {
        "pre": {"marker": "o", "color": "#b8322a", "label": "pre-recon"},
        "post": {"marker": "s", "color": "#1f5a94", "label": "post-recon"},
    }
    truth = np.asarray([value for _, value in TARGETS])
    all_rows = [row for stage in STAGES for cut in KMAX
                for row in results[stage][f"kmax{cut:.2f}"]["rows"]]
    y_lower = min(-320.0, 20 * math.floor((min(row["posterior_lower_68"] for row in all_rows)-10)/20))
    y_upper = max(280.0, 20 * math.ceil((max(row["posterior_upper_68"] for row in all_rows)+10)/20))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), sharex=True, sharey=True)
    for ax, cut in zip(axes.ravel(), KMAX, strict=True):
        key = f"kmax{cut:.2f}"
        ax.plot([-165, 165], [-165, 165], color="0.52", lw=1.8, ls=":",
                label="perfect recovery")
        for stage, shift in (("pre", -2.5), ("post", 2.5)):
            entry = results[stage][key]
            rows = entry["rows"]
            median = np.asarray([row["posterior_median"] for row in rows])
            errors = np.asarray([
                [row["lower_error_68"] for row in rows],
                [row["upper_error_68"] for row in rows],
            ])
            style = styles[stage]
            ax.errorbar(
                truth + shift, median, yerr=errors, fmt=style["marker"],
                color=style["color"], ms=8.0, capsize=4.5, lw=2.0,
                label=style["label"],
            )
        ax.text(
            0.955, 0.065,
            rf"$k_{{\max}}={cut:.2f}\ h\,\mathrm{{Mpc}}^{{-1}}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=20.0,
            weight="semibold",
            color="#C62828",
        )
        ax.set_xlim(-165, 165)
        ax.set_ylim(y_lower, y_upper)
        ax.grid(alpha=0.22, lw=0.9)
        ax.legend(frameon=False, loc="upper left")
    fig.supxlabel(r"input $f_{\rm NL}$", fontsize=25.0, y=0.025)
    fig.supylabel(r"posterior median $f_{\rm NL}$", fontsize=25.0, x=0.018)
    fig.tight_layout(rect=(0.04, 0.065, 1.0, 1.0), h_pad=1.2, w_pad=1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    fig.savefig(temporary)
    plt.close(fig)
    temporary.replace(destination)


def main() -> None:
    """运行 24 个 MCMC case，检查全部 gate 后一次性交付 JSON、chains 和 PDF。"""

    args = parse_args()
    started = time.perf_counter()
    assets = profile.load_assets(
        args.theory_bundle, simulation_npz=args.simulation_npz
    )
    registered = np.flatnonzero(
        assets["diagonal_k"] <= float(np.max(KMAX)) + 1.0e-12
    )
    profile.require(
        registered.size > args.exclude_last_diagonal_bins,
        "exclude-last-diagonal-bins removes every registered diagonal bin",
    )
    excluded_high_positions = (
        registered[-args.exclude_last_diagonal_bins:]
        if args.exclude_last_diagonal_bins
        else np.asarray([], dtype=np.int64)
    )
    allowed_by_high = np.ones(assets["diagonal_k"].size, dtype=bool)
    allowed_by_high[excluded_high_positions] = False
    results: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
    retained_cases: dict[tuple[int, int, int], np.ndarray] = {}
    case_index = 0
    for stage_index, stage in enumerate(STAGES):
        for cut_index, cut in enumerate(KMAX):
            mask = ((np.arange(assets["diagonal_k"].size) >= args.exclude_first_diagonal_bins)
                    & allowed_by_high
                    & (assets["diagonal_k"] <= cut + 1.e-12))
            indices = np.flatnonzero(mask)
            covariance = profile.covariance_contract(
                assets["simulation"][stage]["fiducial"][:, indices], args.volume)
            rows = []
            for target_index, (target_name, truth) in enumerate(TARGETS):
                target = assets["simulation"][stage][target_name][:, indices].mean(axis=0)
                record, retained = run_case(
                    target=target, gaussian=assets["gaussian"][stage][indices],
                    response=assets["png_response"][stage][indices],
                    quadratic=assets["quadratic"][stage][indices], covariance=covariance,
                    truth=truth, case_seed=args.seed + 100000 * case_index, args=args,
                )
                record["target_name"] = target_name
                rows.append(record)
                retained_cases[(stage_index, cut_index, target_index)] = retained
                diagnostic = record["diagnostics"]
                print(
                    f"{stage} kmax={cut:.2f} {target_name}: "
                    f"median={record['posterior_median']:+.3f} "
                    f"CI68=[{record['posterior_lower_68']:+.3f},{record['posterior_upper_68']:+.3f}] "
                    f"steps={record['chain_layout']['total_steps_per_walker']} "
                    f"Rhat={diagnostic['rank_normalized_split_rhat']:.5f} "
                    f"ESS={min(diagnostic['bulk_ess'], diagnostic['tail_ess_5_95']):.0f} "
                    f"exact={diagnostic['maximum_mcmc_exact_quantile_difference_sigma']:.4f}sigma",
                    flush=True,
                )
                case_index += 1
            results[stage][f"kmax{cut:.2f}"] = {
                "kmax": float(cut), "n_data": int(indices.size),
                "selected_diagonal_positions": indices,
                "selected_k": assets["diagonal_k"][indices],
                "hartlap": covariance["hartlap"], "rows": rows,
                "recovery": recovery_summary(rows),
            }

    # 各 case 可能因自动延长有不同长度；用 NaN padding 保留完整 retained chain。
    maximum_retained = max(value.shape[1] for value in retained_cases.values())
    chains = np.full((len(STAGES), len(KMAX), len(TARGETS), args.ensembles,
                      maximum_retained, args.walkers), np.nan, dtype=np.float32)
    retained_lengths = np.zeros((len(STAGES), len(KMAX), len(TARGETS)), dtype=np.int64)
    for index, value in retained_cases.items():
        chains[index + (slice(None), slice(0, value.shape[1]), slice(None))] = value.astype(np.float32)
        retained_lengths[index] = value.shape[1]
    atomic_npz(args.output_chains, {
        "retained_chains_fNL": chains,
        "retained_steps": retained_lengths,
        "stage_names": np.asarray(STAGES),
        "kmax": np.asarray(KMAX),
        "target_names": np.asarray([name for name, _ in TARGETS]),
        "input_fNL": np.asarray([truth for _, truth in TARGETS]),
        "prior_bounds": np.asarray(PRIOR),
    })
    document = {
        "schema_version": 1,
        "scope": (
            f"Quijote DM z={args.redshift:g} R={args.smoothing_radius:g} "
            "diagonal B000 finite-fNL MCMC recovery"
        ),
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": artifact(Path(__file__)),
        "inputs": {name: {"path": str(path), "sha256": assets["hashes"][name]}
                   for name, path in assets["paths"].items()},
        "contract": {
            "model": "G_1loop + fNL*L_1loop + fNL^2*B112II",
            "likelihood": "fixed-covariance Gaussian, logL=-chi2/2",
            "prior": {"fNL": {"distribution": "uniform", "bounds": PRIOR}},
            "redshift": float(args.redshift),
            "smoothing_radius_mpc_h": float(args.smoothing_radius),
            "space": "real",
            "n_common_realizations": int(assets["common_realizations"].size),
            "target": f"{assets['common_realizations'].size}-realization mean",
            "covariance": "stage-specific fiducial single-box sample covariance, Hartlap corrected",
            "volume_gpc3_h3": float(args.volume),
            "exclude_first_diagonal_bins": int(args.exclude_first_diagonal_bins),
            "exclude_last_diagonal_bins": int(args.exclude_last_diagonal_bins),
            "excluded_high_diagonal_positions": excluded_high_positions,
            "excluded_high_diagonal_k_h_mpc": assets["diagonal_k"][excluded_high_positions],
            "kmax_h_mpc": KMAX,
            "posterior_point": "median",
            "errorbar": "equal-tail 16th--84th percentiles",
            "sampler": {
                "name": "emcee", "version": emcee.__version__,
                "environment_python": sys.executable,
                "ensembles": args.ensembles, "walkers": args.walkers,
                "initial_steps": args.initial_steps, "increment": args.increment,
                "maximum_steps": args.max_steps, "burn_fraction": args.burn_fraction,
                "moves": "0.8 StretchMove(a=2.0) + 0.2 DEMove",
                "base_seed": args.seed,
            },
            "diagnostic_note": (
                "four independent ensembles are the chain units; walkers within an ensemble interact"
            ),
        },
        "results": results,
        "global_audit": {
            "all_24_cases_converged": bool(all(
                row["diagnostics"]["pass"] for stage in results.values()
                for entry in stage.values() for row in entry["rows"])),
            "maximum_rank_normalized_split_rhat": float(max(
                row["diagnostics"]["rank_normalized_split_rhat"] for stage in results.values()
                for entry in stage.values() for row in entry["rows"])),
            "minimum_bulk_ess": float(min(
                row["diagnostics"]["bulk_ess"] for stage in results.values()
                for entry in stage.values() for row in entry["rows"])),
            "minimum_tail_ess": float(min(
                row["diagnostics"]["tail_ess_5_95"] for stage in results.values()
                for entry in stage.values() for row in entry["rows"])),
            "minimum_retained_length_over_tau": float(min(
                row["diagnostics"]["retained_length_over_tau"] for stage in results.values()
                for entry in stage.values() for row in entry["rows"])),
            "maximum_mcmc_exact_quantile_difference_sigma": float(max(
                row["diagnostics"]["maximum_mcmc_exact_quantile_difference_sigma"]
                for stage in results.values() for entry in stage.values() for row in entry["rows"])),
        },
        "outputs": {
            "chains": artifact(args.output_chains),
            "pdf": str(args.output_pdf.resolve()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_json, document)
    make_plot(
        args.output_pdf, results, assets, args.redshift, args.smoothing_radius,
        args.volume, args.exclude_first_diagonal_bins,
        assets["diagonal_k"][excluded_high_positions],
    )
    # PDF 在 JSON 之后原子生成，再回写其最终 hash；防止 contract 指向旧图。
    document["outputs"]["pdf"] = artifact(args.output_pdf)
    document["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(args.output_json, document)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_chains}")
    print(f"Replaced {args.output_pdf}")


if __name__ == "__main__":
    main()
