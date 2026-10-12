#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""运行 Quijote DM z=1、R=15 的 one-loop B000 fNL profile recovery。

文件概览
========
本脚本只复用仓库中已经完成的 446 个公共 realization 和 MARISA-B
diagonal shell-bin one-loop 理论，不重新计算昂贵的三角形积分。完整流程为：

1. 读取 fiducial、LC_m、LC_p 的 pre/post ``B000`` realization 矩阵；
2. 对齐 15 个 diagonal ``B000(k,k)`` bin 与 pre/post one-loop 理论；
3. 对多个 ``kmax <= 0.2 h/Mpc`` 子空间构造 fiducial 单盒 covariance；
4. 默认保留旧线性模式；--theory-bundle 使用统一 G+fNL*L+fNL²*Q；
5. 用独立的线性方程求解和显式 profile 网格复核中心及区间；
6. 保存 JSON/NPZ 审计材料，并把四个不同 kmax 的 recovery 面板写入会议 PDF。

命令行参数 ``--exclude-first-diagonal-bins`` 可从所有拟合 mask 中统一排除
最低的若干个 diagonal k bin。改变该参数必须使用新的输出目录和前缀，避免
把不同数据 mask 的结果混在同一个 run contract 中。

统计口径
========
- target 是 446 个公共 realization 的 simulation mean；
- covariance 是同一批 fiducial realization 的单盒 sample covariance；
- 默认报告体积为 ``1 (Gpc/h)^3``，不是 target mean 的 covariance；
- precision 使用与仓库 Fig.5 一致的 Hartlap 因子；
- 每个 kmax 都按实际数据维数重新计算 Hartlap 因子；
- ``fiducial/LC_m/LC_p`` 的输入值分别为 ``0/-100/+100``；
- 旧模式对 fNL 只保留零点的一阶响应；显式 theory bundle 才加入二次项。
- 二次模式求四次 chi² 的全部驻点和 Δchi²=1 根，不假定误差对称。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 登录节点上的数值库不得自行扩展到超过项目约定的 8 个线程。
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lzy-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution
from scipy.optimize import minimize_scalar, brentq


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model")
SIMULATION_NPZ = (
    PROJECT_ROOT
    / "analysis/jaxrecon_jaxpower_quijote_dm_png_z1_r15_2d_k0p3_n15_paperbin_three_cosmo_partial"
    / "quijote_dm_png_z1_r15_2d_paperbin_b000_three_cosmo_matrix_partial446_matrix.npz"
)
PRE_GAUSSIAN_NPZ = (
    PROJECT_ROOT
    / "analysis/theory_vectors/marisa_b_v0"
    / "marisa_b_pre_gaussian_shellbin_z1_r15_diag15_kmax0p3_nrad3_nmu12_partial446_20260616.npz"
)
POST_GAUSSIAN_NPZ = (
    PROJECT_ROOT
    / "analysis/theory_vectors/marisa_b_v0"
    / "marisa_b_post_gaussian_shellbin_z1_r15_diag15_kmax0p3_nrad3_nmu11_qmax20_eps2e2_partial446_20260616.npz"
)
PRE_PNG_NPZ = (
    PROJECT_ROOT
    / "analysis/theory_vectors/marisa_b_v0"
    / "marisa_b_pre_local_png_1loop_z1_shellbin_diag15_kmax0p3_nrad3_nmu24_eps1e3_partial446_20260618.npz"
)
POST_PNG_NPZ = (
    PROJECT_ROOT
    / "analysis/theory_vectors/marisa_b_v0"
    / "marisa_b_post_local_png_1loop_z1_r15_shellbin_diag15_kmax0p3_nrad3_nmu24_qmax20_eps2e2_partial446_20260618.npz"
)

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis/quijote_dm_z1_r15_1loop_fnl_profile_kmax0p20_v1"
DEFAULT_MEETING_DIR = PROJECT_ROOT / "9.11meeting/dm"
DEFAULT_PREFIX = "quijote_dm_z1_r15_1loop_b000_fnl_profile_kmax0p20_v1"

KMAX_GRID = np.asarray([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], dtype=np.float64)
PLOT_RECOVERY_KMAX_GRID = np.asarray([0.08, 0.10, 0.15, 0.20], dtype=np.float64)
PRIMARY_KMAX = 0.15
TARGETS = (
    ("LC_m", -100.0),
    ("fiducial", 0.0),
    ("LC_p", 100.0),
)
STAGES = ("pre", "post")
PINV_RCOND = 1.0e-12


def parse_args() -> argparse.Namespace:
    """解析少量可审计参数；理论和 simulation 输入路径保持冻结。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="likelihood 体积，单位 (Gpc/h)^3；默认与单个 Quijote box 相同。",
    )
    parser.add_argument(
        "--primary-kmax",
        type=float,
        default=PRIMARY_KMAX,
        choices=KMAX_GRID.tolist(),
        help="报告中展示完整 profile 曲线的参考 kmax。",
    )
    parser.add_argument(
        "--exclude-first-diagonal-bins",
        type=int,
        default=0,
        help="从每个 kmax 拟合中排除最低的 N 个 diagonal k bin。",
    )
    parser.add_argument(
        "--exclude-last-diagonal-bins",
        type=int,
        default=0,
        help=("从注册的整体 k<=0.2 diagonal 数据范围排除最高的 N 个 bin；"
              "不会按每个较小 kmax 重复排除。"),
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--meeting-dir", type=Path, default=DEFAULT_MEETING_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--simulation-npz", type=Path, default=SIMULATION_NPZ)
    parser.add_argument("--redshift", type=float, default=1.0)
    parser.add_argument("--smoothing-radius", type=float, default=15.0)
    parser.add_argument("--theory-bundle", type=Path, default=None,
                        help="同 geometry 的 pre/post G/L/Q NPZ；未提供则重现旧线性结果。")
    parser.add_argument("--omit-quadratic", action="store_true",
                        help="仅作消融：有 bundle 时将 Q 置零，以量化 fNL² 的净影响。")
    return parser.parse_args()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """把 NPZ 读成普通字典，确保关闭底层文件句柄。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def sha256(path: Path) -> str:
    """记录输入文件哈希，使会议结果可以追溯到精确资产。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value: Any) -> Any:
    """递归转换 NumPy 标量，并禁止 JSON 中出现 NaN 或 Infinity。"""

    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def require(condition: bool, message: str) -> None:
    """用清晰错误信息终止 provenance 或数组对齐错误。"""

    if not bool(condition):
        raise RuntimeError(message)


def load_assets(
    theory_bundle: Path | None = None,
    omit_quadratic: bool = False,
    simulation_npz: Path = SIMULATION_NPZ,
) -> dict[str, Any]:
    """读取并严格对齐 simulation、Gaussian 基线和 PNG 响应。"""

    paths = {"simulation": simulation_npz.resolve()}
    if theory_bundle is None:
        paths.update({
            "pre_gaussian": PRE_GAUSSIAN_NPZ,
            "post_gaussian": POST_GAUSSIAN_NPZ,
            "pre_png": PRE_PNG_NPZ,
            "post_png": POST_PNG_NPZ,
        })
    arrays = {name: read_npz(path) for name, path in paths.items()}
    sim = arrays["simulation"]

    common = np.asarray(sim["common_realizations"], dtype=np.int64)
    require(common.ndim == 1 and common.size > 2, f"invalid common realization layout: {common.shape}")
    require(np.array_equal(common, np.arange(common.size)),
            f"common realization IDs are not exactly 0..{common.size - 1}")

    pair_i = np.asarray(sim["pair_i"], dtype=np.int64)
    pair_j = np.asarray(sim["pair_j"], dtype=np.int64)
    require(pair_i.shape == pair_j.shape == (120,), "simulation pair layout is not 120-dimensional")
    diagonal_indices = np.flatnonzero(pair_i == pair_j).astype(np.int64)
    require(diagonal_indices.shape == (15,), "expected exactly 15 diagonal B000 bins")

    weighted_k = np.asarray(sim["hdf5_k_weighted"], dtype=np.float64)
    require(weighted_k.shape == (120, 2), "unexpected weighted-k layout")
    diagonal_k = np.mean(weighted_k[diagonal_indices], axis=1)
    require(np.all(np.diff(diagonal_k) > 0.0), "diagonal k values are not strictly increasing")

    gaussian: dict[str, np.ndarray] = {}
    png_response: dict[str, np.ndarray] = {}
    if theory_bundle is None:
        # 四个 legacy 理论文件的有效下标必须与 simulation 的 diagonal 下标完全一致。
        theory_indices = {
            "pre_gaussian": np.asarray(arrays["pre_gaussian"]["computed_pair_indices"], dtype=np.int64),
            "post_gaussian": np.asarray(arrays["post_gaussian"]["pair_indices"], dtype=np.int64),
            "pre_png": np.asarray(arrays["pre_png"]["computed_pair_indices"], dtype=np.int64),
            "post_png": np.asarray(arrays["post_png"]["pair_indices"], dtype=np.int64),
        }
        for name, indices in theory_indices.items():
            require(np.array_equal(indices, diagonal_indices),
                    f"{name} theory indices do not match simulation diagonal")
        gaussian = {
            "pre": np.asarray(arrays["pre_gaussian"]["B000_Btotal"], dtype=np.float64)[diagonal_indices],
            "post": np.asarray(
                arrays["post_gaussian"]["theory_B000_Btotal_post_1loop"], dtype=np.float64
            )[diagonal_indices],
        }
        png_response = {
            "pre": np.asarray(arrays["pre_png"]["B000_dBdfNL_local_total"], dtype=np.float64)[diagonal_indices],
            "post": np.asarray(arrays["post_png"]["B000_dBdfNL_local_total"], dtype=np.float64)[diagonal_indices],
        }
    quadratic = {stage: np.zeros(15) for stage in STAGES}
    theory_contract = None
    if theory_bundle is not None:
        bundle = read_npz(theory_bundle)
        require(np.array_equal(bundle["diagonal_indices"], diagonal_indices), "bundle geometry mismatch")
        require(np.allclose(bundle["diagonal_k"], diagonal_k, rtol=1e-12, atol=1e-12),
                "bundle weighted k mismatch")
        needed = diagonal_k <= .2 + 1e-12
        for stage in STAGES:
            # bundle 外的高 k bin 保留 NaN；从不回退到旧模型拼成一条看似完整的曲线。
            gaussian[stage] = np.asarray(bundle[f"{stage}_G"], dtype=float)
            png_response[stage] = np.asarray(bundle[f"{stage}_L"], dtype=float)
            quadratic[stage] = (np.zeros(15) if omit_quadratic else
                                np.asarray(bundle[f"{stage}_Q"], dtype=float))
            for value in (gaussian[stage], png_response[stage], quadratic[stage]):
                require(np.all(np.isfinite(value[needed])), "bundle missing a fitted bin")
        paths["matched_theory_bundle"] = theory_bundle.resolve()
        meta_path = theory_bundle.with_suffix(".json")
        theory_contract = json.loads(meta_path.read_text()) if meta_path.exists() else None
    for stage in STAGES:
        needed = diagonal_k <= .2 + 1e-12 if theory_bundle is not None else np.ones(15, dtype=bool)
        require(np.all(np.isfinite(gaussian[stage][needed])), f"non-finite {stage} Gaussian theory")
        require(np.all(np.isfinite(png_response[stage][needed])), f"non-finite {stage} PNG theory")
        require(np.linalg.norm(png_response[stage][needed]) > 0.0, f"zero-norm {stage} PNG response")

    simulation: dict[str, dict[str, np.ndarray]] = {stage: {} for stage in STAGES}
    for stage in STAGES:
        for target_name, _ in TARGETS:
            key = f"{target_name}_common_{stage}_B000"
            stack = np.asarray(sim[key], dtype=np.float64)[:, diagonal_indices]
            require(stack.shape == (common.size, 15), f"unexpected stack shape for {key}: {stack.shape}")
            require(np.all(np.isfinite(stack)), f"non-finite simulation values in {key}")
            simulation[stage][target_name] = stack

    return {
        "paths": paths,
        "hashes": {name: sha256(path) for name, path in paths.items()},
        "common_realizations": common,
        "diagonal_indices": diagonal_indices,
        "diagonal_k": diagonal_k,
        "gaussian": gaussian,
        "png_response": png_response,
        "quadratic": quadratic,
        "finite_fnl": theory_bundle is not None and not omit_quadratic,
        "theory_contract": theory_contract,
        "simulation": simulation,
    }


def covariance_contract(fiducial_stack: np.ndarray, volume: float) -> dict[str, Any]:
    """构造单盒 covariance、体积缩放和 Hartlap-corrected precision。"""

    stack = np.asarray(fiducial_stack, dtype=np.float64)
    require(stack.ndim == 2, "fiducial stack must be two-dimensional")
    n_mock, n_data = stack.shape
    require(n_mock > n_data + 2, "not enough mocks for a positive Hartlap factor")
    require(volume > 0.0 and math.isfinite(volume), "volume must be finite and positive")

    covariance_one_box = np.atleast_2d(np.cov(stack, rowvar=False, ddof=1)).astype(np.float64)
    require(np.all(np.isfinite(covariance_one_box)), "covariance contains non-finite values")
    require(np.all(np.diag(covariance_one_box) > 0.0), "covariance diagonal is not positive")

    hartlap = float((n_mock - n_data - 2) / (n_mock - 1))
    covariance_volume = covariance_one_box / float(volume)
    precision = hartlap * np.linalg.pinv(covariance_volume, rcond=PINV_RCOND, hermitian=True)
    precision = 0.5 * (precision + precision.T)
    eigenvalues = np.linalg.eigvalsh(covariance_volume)
    require(float(np.min(eigenvalues)) > 0.0, "selected covariance is not positive definite")

    return {
        "covariance_one_box": covariance_one_box,
        "covariance_volume": covariance_volume,
        "precision": precision,
        "hartlap": hartlap,
        "condition_number": float(np.linalg.cond(covariance_volume)),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "n_mock": int(n_mock),
        "n_data": int(n_data),
    }


def analytic_profile(
    *,
    target: np.ndarray,
    gaussian: np.ndarray,
    response: np.ndarray,
    covariance: dict[str, Any],
    truth: float,
) -> dict[str, Any]:
    """求解线性模型的一参数 profile，并返回完整 goodness-of-fit 信息。"""

    data = np.asarray(target, dtype=np.float64)
    base = np.asarray(gaussian, dtype=np.float64)
    linear = np.asarray(response, dtype=np.float64)
    precision = np.asarray(covariance["precision"], dtype=np.float64)
    offset = data - base

    fisher = float(linear @ precision @ linear)
    require(fisher > 0.0 and math.isfinite(fisher), "profile Fisher information is invalid")
    bestfit = float(linear @ precision @ offset / fisher)
    sigma = float(1.0 / math.sqrt(fisher))
    residual = offset - bestfit * linear
    chi2_min = float(residual @ precision @ residual)
    truth_residual = offset - float(truth) * linear
    chi2_truth = float(truth_residual @ precision @ truth_residual)
    delta_truth = float(chi2_truth - chi2_min)
    dof = int(data.size - 1)
    p_value = float(chi2_distribution.sf(chi2_min, dof)) if dof > 0 else math.nan

    # 独立审计不使用 pseudo-inverse，而直接解 C x = y；profile 中心应与主结果一致。
    covariance_volume = np.asarray(covariance["covariance_volume"], dtype=np.float64)
    solved_response = np.linalg.solve(covariance_volume, linear)
    solved_offset = np.linalg.solve(covariance_volume, offset)
    fisher_unhartlap = float(linear @ solved_response)
    independent_bestfit = float(linear @ solved_offset / fisher_unhartlap)
    independent_sigma = float(1.0 / math.sqrt(float(covariance["hartlap"]) * fisher_unhartlap))

    return {
        "input_fNL": float(truth),
        "profile_fNL": bestfit,
        "profile_lower": bestfit - sigma,
        "profile_upper": bestfit + sigma,
        "profile_half_width": sigma,
        "fisher": fisher,
        "chi2_min": chi2_min,
        "chi2_truth": chi2_truth,
        "truth_delta_chi2": delta_truth,
        "truth_signed_pull": float((bestfit - float(truth)) / sigma),
        "truth_inside_delta_chi2_1": bool(delta_truth <= 1.0 + 1.0e-10),
        "dof": dof,
        "chi2_per_dof": float(chi2_min / dof) if dof > 0 else math.nan,
        "p_value": p_value,
        "bestfit_prediction": base + bestfit * linear,
        "truth_prediction": base + float(truth) * linear,
        "target": data,
        "residual_bestfit": residual,
        "independent_solve": {
            "profile_fNL": independent_bestfit,
            "profile_half_width": independent_sigma,
            "absolute_centre_difference": abs(bestfit - independent_bestfit),
            "absolute_width_difference": abs(sigma - independent_sigma),
        },
    }


def quadratic_profile(
    *, target: np.ndarray, gaussian: np.ndarray, response: np.ndarray,
    quadratic: np.ndarray, covariance: dict[str, Any], truth: float,
) -> dict[str, Any]:
    """求 B=G+fL+f²Q 的全局一参数 profile，记录所有可信区间。

    用 x=f/100 改善多项式系数条件数；chi² 为四次多项式，全部实驻点与
    数值搜索边界一起比较。[-1000,1000] 是显式诊断范围，碰边不算通过。
    独立复核用直接残差、solve(C,...) 和有界标量最小化，不复用四次系数。
    """
    q = np.asarray(quadratic, dtype=float)
    if not np.any(q):
        return analytic_profile(target=target, gaussian=gaussian, response=response,
                                covariance=covariance, truth=truth)
    offset = np.asarray(target) - np.asarray(gaussian)
    precision = covariance["precision"]
    l100, q10000 = 100 * np.asarray(response), 10000 * q
    coefficients = np.array([
        offset @ precision @ offset,
        -2 * l100 @ precision @ offset,
        l100 @ precision @ l100 - 2 * q10000 @ precision @ offset,
        2 * l100 @ precision @ q10000,
        q10000 @ precision @ q10000,
    ])
    roots = np.polynomial.polynomial.polyroots(np.arange(1, 5) * coefficients[1:])
    candidates = [-10., 10.] + [r.real for r in roots if abs(r.imag) < 1e-8 and -10 < r.real < 10]
    values = np.polynomial.polynomial.polyval(candidates, coefficients)
    xbest = float(candidates[int(np.argmin(values))])
    require(abs(xbest) < 9.999, "quadratic optimum reaches fNL diagnostic boundary")
    best = 100 * xbest

    def evaluate(f: float, independent: bool = False) -> float:
        """直接求物理残差的 chi²，独立模式显式求解协方差线性方程。"""
        residual = offset - f * response - f*f*q
        if independent:
            return float(covariance["hartlap"] * residual @ np.linalg.solve(
                covariance["covariance_volume"], residual))
        return float(residual @ precision @ residual)

    chi_min = evaluate(best)
    level = coefficients.copy()
    level[0] -= chi_min + 1
    crossings = np.polynomial.polynomial.polyroots(level)
    edges = [-1000.] + sorted(100*r.real for r in crossings
                              if abs(r.imag) < 1e-7 and -10 < r.real < 10) + [1000.]
    intervals = [[a, b] for a, b in zip(edges[:-1], edges[1:])
                 if evaluate((a+b)/2) <= chi_min + 1 + 1e-8]
    lower, upper = next((a, b) for a, b in intervals if a <= best <= b)
    require(lower > -1000 and upper < 1000, "quadratic confidence interval reaches search boundary")
    # 在所有稠密网格点中找最低一段，再使用直接 likelihood 做独立极小值检查。
    grid = np.linspace(-1000., 1000., 4001)
    grid_values = np.array([evaluate(f, independent=True) for f in grid])
    n = int(np.argmin(grid_values))
    independent = minimize_scalar(lambda f: evaluate(f, independent=True), method="bounded",
                                  bounds=(grid[max(0, n-2)], grid[min(len(grid)-1, n+2)]),
                                  options={"xatol": 1e-9})
    independent_best = float(independent.x)
    width = (upper - lower)/2
    root_margin = max(width * .05, .01)
    independent_edges = [brentq(lambda f: evaluate(f, independent=True) - chi_min - 1,
                                edge-root_margin, edge+root_margin, xtol=1e-10)
                         for edge in (lower, upper)]
    tangent = response + 2*best*q
    fisher = float(tangent @ precision @ tangent)
    truth_chi = evaluate(float(truth))
    dof = int(offset.size - 1)
    return {
        "input_fNL": float(truth), "profile_fNL": best,
        "profile_lower": lower, "profile_upper": upper, "profile_half_width": width,
        "confidence_intervals": intervals, "numerical_search_bounds": [-1000, 1000],
        "fisher": fisher, "chi2_min": chi_min, "chi2_truth": truth_chi,
        "truth_delta_chi2": truth_chi-chi_min,
        "truth_signed_pull": float(np.sign(best-truth)*np.sqrt(max(0., truth_chi-chi_min))),
        "truth_inside_delta_chi2_1": bool(truth_chi-chi_min <= 1 + 1e-10),
        "dof": dof, "chi2_per_dof": chi_min/dof if dof else math.nan,
        "p_value": float(chi2_distribution.sf(chi_min, dof)) if dof else math.nan,
        "bestfit_prediction": gaussian + best*response + best*best*q,
        "truth_prediction": gaussian + truth*response + truth*truth*q,
        "target": target, "residual_bestfit": offset-best*response-best*best*q,
        "independent_solve": {
            "profile_fNL": independent_best, "profile_half_width": (independent_edges[1]-independent_edges[0])/2,
            "absolute_centre_difference": abs(best-independent_best),
            "absolute_width_difference": abs(width-(independent_edges[1]-independent_edges[0])/2),
        },
    }


def recovery_regression(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """用三个输入值概括恢复斜率、截距和偏差。"""

    truth = np.asarray([row["input_fNL"] for row in rows], dtype=np.float64)
    recovered = np.asarray([row["profile_fNL"] for row in rows], dtype=np.float64)
    design = np.column_stack((truth, np.ones_like(truth)))
    slope, intercept = np.linalg.lstsq(design, recovered, rcond=None)[0]
    bias = recovered - truth
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_absolute_bias": float(np.mean(np.abs(bias))),
        "maximum_absolute_bias": float(np.max(np.abs(bias))),
        "n_truth_inside_delta_chi2_1": int(sum(row["truth_inside_delta_chi2_1"] for row in rows)),
        "n_targets": int(len(rows)),
    }


def run_profiles(
    assets: dict[str, Any],
    volume: float,
    primary_kmax: float,
    exclude_first_diagonal_bins: int,
    exclude_last_diagonal_bins: int = 0,
) -> dict[str, Any]:
    """遍历 stage、kmax 和三个输入宇宙学，生成 profile 与审计数组。"""

    diagonal_k = np.asarray(assets["diagonal_k"], dtype=np.float64)
    excluded = int(exclude_first_diagonal_bins)
    excluded_high = int(exclude_last_diagonal_bins)
    require(excluded >= 0, "exclude_first_diagonal_bins must be non-negative")
    require(excluded_high >= 0, "exclude_last_diagonal_bins must be non-negative")
    require(excluded < diagonal_k.size, "exclude_first_diagonal_bins removes every diagonal bin")
    registered = np.flatnonzero(diagonal_k <= float(np.max(KMAX_GRID)) + 1.0e-12)
    require(registered.size > excluded_high, "exclude_last_diagonal_bins removes every registered bin")
    high_positions = registered[-excluded_high:] if excluded_high else np.asarray([], dtype=np.int64)
    allowed_by_low_k = np.arange(diagonal_k.size, dtype=np.int64) >= excluded
    allowed_by_high_k = np.ones(diagonal_k.size, dtype=bool)
    allowed_by_high_k[high_positions] = False
    active_kmax_grid = np.asarray(
        [
            cut
            for cut in KMAX_GRID
            if np.count_nonzero(allowed_by_low_k & allowed_by_high_k
                                & (diagonal_k <= float(cut) + 1.0e-12)) >= 2
        ],
        dtype=np.float64,
    )
    require(active_kmax_grid.size > 0, "no kmax cut retains at least two fitted bins")
    require(
        np.any(np.isclose(active_kmax_grid, float(primary_kmax), rtol=0.0, atol=1.0e-12)),
        "primary_kmax does not retain at least two fitted bins",
    )
    profiles: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
    primary_grids: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}

    for stage in STAGES:
        gaussian_full = np.asarray(assets["gaussian"][stage], dtype=np.float64)
        response_full = np.asarray(assets["png_response"][stage], dtype=np.float64)
        quadratic_full = np.asarray(assets["quadratic"][stage], dtype=np.float64)
        fiducial_full = np.asarray(assets["simulation"][stage]["fiducial"], dtype=np.float64)

        for cut in active_kmax_grid:
            mask = (allowed_by_low_k & allowed_by_high_k
                    & (diagonal_k <= float(cut) + 1.0e-12))
            indices = np.flatnonzero(mask).astype(np.int64)
            require(indices.size >= 2, f"too few bins for kmax={cut}")
            cov = covariance_contract(fiducial_full[:, indices], volume)
            rows: list[dict[str, Any]] = []

            for target_name, truth in TARGETS:
                target_mean = np.mean(assets["simulation"][stage][target_name][:, indices], axis=0)
                result = quadratic_profile(
                    target=target_mean,
                    gaussian=gaussian_full[indices],
                    response=response_full[indices],
                    quadratic=quadratic_full[indices],
                    covariance=cov,
                    truth=truth,
                )
                result["target_name"] = target_name
                rows.append(result)

                # 参考 kmax 保存显式 profile 网格；直接重算 chi2，而不是套解析抛物线。
                if abs(float(cut) - float(primary_kmax)) <= 1.0e-12:
                    grid = np.linspace(-500.0, 500.0, 4001)
                    residual_grid = (
                        target_mean[None, :]
                        - gaussian_full[indices][None, :]
                        - grid[:, None] * response_full[indices][None, :]
                        - grid[:, None]**2 * quadratic_full[indices][None, :]
                    )
                    chi2_grid = np.einsum("ni,ij,nj->n", residual_grid, cov["precision"], residual_grid)
                    delta_grid = chi2_grid - float(np.min(chi2_grid))
                    grid_bestfit = float(grid[int(np.argmin(chi2_grid))])
                    require(
                        abs(grid_bestfit - float(result["profile_fNL"])) <= 0.126,
                        f"explicit grid profile disagrees for {stage}/{target_name}",
                    )
                    primary_grids[stage][target_name] = {
                        "fNL_grid": grid,
                        "delta_chi2": delta_grid,
                        "grid_bestfit": grid_bestfit,
                    }

            max_centre_audit = max(row["independent_solve"]["absolute_centre_difference"] for row in rows)
            max_width_audit = max(row["independent_solve"]["absolute_width_difference"] for row in rows)
            require(max_centre_audit < 1.0e-5, f"independent centre audit failed for {stage}, kmax={cut}")
            require(max_width_audit < 1.0e-7, f"independent width audit failed for {stage}, kmax={cut}")

            profiles[stage][f"kmax{cut:.2f}"] = {
                "kmax": float(cut),
                "selected_diagonal_positions": indices,
                "selected_global_pair_indices": assets["diagonal_indices"][indices],
                "selected_k": diagonal_k[indices],
                "n_data": int(indices.size),
                "n_mock": int(cov["n_mock"]),
                "hartlap": float(cov["hartlap"]),
                "condition_number": float(cov["condition_number"]),
                "minimum_covariance_eigenvalue": float(cov["minimum_eigenvalue"]),
                "rows": rows,
                "recovery": recovery_regression(rows),
                "independent_audit": {
                    "maximum_absolute_centre_difference": float(max_centre_audit),
                    "maximum_absolute_width_difference": float(max_width_audit),
                    "passed": True,
                },
            }

    return {
        "profiles": profiles,
        "primary_grids": primary_grids,
        "active_kmax_grid": active_kmax_grid,
        "excluded_high_positions": high_positions,
    }


def build_summary(profiles: dict[str, dict[str, Any]], primary_kmax: float) -> dict[str, Any]:
    """提取报告需要的主 cut 和安全 kmax 结论。"""

    primary_key = f"kmax{primary_kmax:.2f}"
    stage_summary: dict[str, Any] = {}
    for stage in STAGES:
        passing_cuts = [
            float(entry["kmax"])
            for entry in profiles[stage].values()
            if int(entry["recovery"]["n_truth_inside_delta_chi2_1"]) == len(TARGETS)
        ]
        primary = profiles[stage][primary_key]
        stage_summary[stage] = {
            "largest_scanned_kmax_with_all_truths_inside_delta_chi2_1": max(passing_cuts) if passing_cuts else None,
            "primary_kmax": float(primary_kmax),
            "primary_recovery": primary["recovery"],
            "primary_rows": [
                {
                    key: row[key]
                    for key in (
                        "target_name",
                        "input_fNL",
                        "profile_fNL",
                        "profile_lower",
                        "profile_upper",
                        "truth_delta_chi2",
                        "chi2_min",
                        "dof",
                        "chi2_per_dof",
                        "p_value",
                    )
                }
                for row in primary["rows"]
            ],
        }
    return stage_summary


def configure_matplotlib() -> None:
    """设置适合会议 PPT 的大字号、低文字密度绘图样式。"""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 18.0,
            "axes.labelsize": 24.0,
            "xtick.labelsize": 20.0,
            "ytick.labelsize": 20.0,
            "legend.fontsize": 16.0,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def make_plot(
    output_pdf: Path,
    profiles: dict[str, dict[str, Any]],
    volume: float,
    excluded_diagonal_k: np.ndarray,
    excluded_high_diagonal_k: np.ndarray,
    redshift: float,
    smoothing_radius: float,
    n_common: int,
    finite_fnl: bool = False,
) -> None:
    """生成四个不同 kmax 的 pre/post recovery 会议图。"""

    configure_matplotlib()
    stage_styles = {
        "pre": {"marker": "o", "color": "#b8322a", "label": "pre-recon"},
        "post": {"marker": "s", "color": "#1f5a94", "label": "post-recon"},
    }
    truth_values = np.asarray([truth for _, truth in TARGETS], dtype=np.float64)
    displayed_rows = [row for stage in STAGES for cut in PLOT_RECOVERY_KMAX_GRID
                      for row in profiles[stage][f"kmax{cut:.2f}"]["rows"]]
    # 加入二次项或更换模型后，误差棒可能超出旧固定范围；共用完整 y 轴防止裁切。
    y_lower = min(-280., 20*math.floor((min(r["profile_lower"] for r in displayed_rows)-10)/20))
    y_upper = max(280., 20*math.ceil((max(r["profile_upper"] for r in displayed_rows)+10)/20))

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), sharex=True, sharey=True)
    for ax, cut in zip(axes.ravel(), PLOT_RECOVERY_KMAX_GRID):
        key = f"kmax{cut:.2f}"
        ax.plot([-165, 165], [-165, 165], color="0.52", lw=1.8, ls=":", label="perfect recovery")

        # pre/post 横向错开 2.5，确保同一 input 的中心和误差棒都清楚可见。
        for stage, shift in (("pre", -2.5), ("post", 2.5)):
            entry = profiles[stage][key]
            centres = np.asarray([row["profile_fNL"] for row in entry["rows"]])
            widths = np.array([[row["profile_fNL"]-row["profile_lower"] for row in entry["rows"]],
                               [row["profile_upper"]-row["profile_fNL"] for row in entry["rows"]]])
            style = stage_styles[stage]
            ax.errorbar(
                truth_values + shift,
                centres,
                yerr=widths,
                fmt=style["marker"],
                color=style["color"],
                ms=8.0,
                capsize=4.5,
                lw=2.0,
                label=style["label"],
            )

        ax.text(
            0.955,
            0.065,
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
    fig.supylabel(r"profile best-fit $f_{\rm NL}$", fontsize=25.0, x=0.018)
    fig.tight_layout(rect=(0.04, 0.065, 1.0, 1.0), h_pad=1.2, w_pad=1.0)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)


def save_outputs(
    *,
    args: argparse.Namespace,
    assets: dict[str, Any],
    products: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """保存 JSON、NPZ 和最终会议 PDF。"""

    analysis_dir = args.analysis_dir.resolve()
    meeting_dir = args.meeting_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    meeting_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / f"{args.prefix}.json"
    npz_path = analysis_dir / f"{args.prefix}.npz"
    pdf_path = meeting_dir / f"{args.prefix}.pdf"

    profiles = products["profiles"]
    primary_grids = products["primary_grids"]
    active_kmax_grid = np.asarray(products["active_kmax_grid"], dtype=np.float64)
    summary = build_summary(profiles, float(args.primary_kmax))
    excluded_count = int(args.exclude_first_diagonal_bins)
    excluded_positions = np.arange(excluded_count, dtype=np.int64)
    excluded_k = np.asarray(assets["diagonal_k"], dtype=np.float64)[excluded_positions]
    excluded_high_positions = np.asarray(products["excluded_high_positions"], dtype=np.int64)
    excluded_high_k = np.asarray(assets["diagonal_k"], dtype=np.float64)[excluded_high_positions]
    common = np.asarray(assets["common_realizations"], dtype=np.int64)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            f"Quijote DM z={args.redshift:g} R={args.smoothing_radius:g} "
            "diagonal B000 one-loop fNL profile recovery"
        ),
        "status": "complete",
        "implementation": {"path": str(Path(__file__).resolve()),
                           "sha256": sha256(Path(__file__).resolve())},
        "inputs": {
            name: {"path": str(path), "sha256": assets["hashes"][name]}
            for name, path in assets["paths"].items()
        },
        "contract": {
            "redshift": float(args.redshift),
            "smoothing_radius_mpc_h": float(args.smoothing_radius),
            "space": "real",
            "n_common_realizations": int(common.size),
            "common_realization_range": [int(common[0]), int(common[-1])],
            "statistic": "15 diagonal shell-bin B000(k,k) bins",
            "model": "G_1loop + fNL*L_1loop + fNL^2*B112II" if assets["finite_fnl"] else "G_1loop + fNL*L_1loop",
            "finite_fNL_even_terms": assets["finite_fnl"],
            "theory_bundle_metadata": assets["theory_contract"],
            "targets": {name: truth for name, truth in TARGETS},
            "covariance": (
                f"stage-specific Fiducial common-{common.size} one-realization sample covariance divided by survey volume; "
                "precision multiplied by dimension-dependent Hartlap factor; never covariance of target mean"
            ),
            "volume_gpc3_h3": float(args.volume),
            "requested_kmax_grid_h_mpc": KMAX_GRID,
            "active_kmax_grid_h_mpc": active_kmax_grid,
            "maximum_kmax_h_mpc": float(np.max(active_kmax_grid)),
            "primary_kmax_h_mpc": float(args.primary_kmax),
            "pair_selection": (
                "diagonal pair_i == pair_j, diagonal position >= exclude_first_diagonal_bins, "
                "and weighted k <= kmax"
            ),
            "exclude_first_diagonal_bins": excluded_count,
            "excluded_diagonal_positions": excluded_positions,
            "excluded_diagonal_k_h_mpc": excluded_k,
            "exclude_last_diagonal_bins": int(excluded_high_positions.size),
            "excluded_high_diagonal_positions": excluded_high_positions,
            "excluded_high_diagonal_k_h_mpc": excluded_high_k,
            "profile_parameters": ["fNL"],
            "nuisance_parameters": [],
            "delta_chi2_interval": 1.0,
        },
        "summary": summary,
        "profiles": profiles,
        "audit": {
            "explicit_primary_profile_grid_step": 0.25,
            "independent_method": "np.linalg.solve on covariance, separate from pseudo-inverse precision path",
            "all_independent_checks_passed": True,
            "all_kmax_at_or_below_0p2": bool(np.max(active_kmax_grid) <= 0.2 + 1.0e-12),
        },
        "outputs": {"json": str(json_path), "npz": str(npz_path), "pdf": str(pdf_path)},
        "interpretation_limits": [
            "This is a diagonal-only B000 test, not the full 120-bin upper-triangle likelihood.",
            ("Includes B112II at order fNL^2; convergence and model validity must be assessed separately."
             if assets["finite_fnl"] else "The fNL model is linearized and omits finite-fNL even terms."),
            "kmax limits labelled k1/k2; the B000 projection contains k3 up to approximately 2*kmax.",
            "The fiducial target mean and fiducial covariance use the same common realization set.",
            "Passing at V=1 does not imply passing at a larger survey volume.",
        ],
    }
    json_path.write_text(json.dumps(finite_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    truth = np.asarray([truth for _, truth in TARGETS], dtype=np.float64)
    save_arrays: dict[str, np.ndarray] = {
        "kmax_grid": active_kmax_grid,
        "input_fNL": truth,
        "diagonal_indices": np.asarray(assets["diagonal_indices"], dtype=np.int64),
        "diagonal_k": np.asarray(assets["diagonal_k"], dtype=np.float64),
        "excluded_diagonal_positions": excluded_positions,
        "excluded_diagonal_k": excluded_k,
        "excluded_high_diagonal_positions": excluded_high_positions,
        "excluded_high_diagonal_k": excluded_high_k,
        "pre_gaussian_1loop": np.asarray(assets["gaussian"]["pre"], dtype=np.float64),
        "post_gaussian_1loop": np.asarray(assets["gaussian"]["post"], dtype=np.float64),
        "pre_png_response_1loop": np.asarray(assets["png_response"]["pre"], dtype=np.float64),
        "post_png_response_1loop": np.asarray(assets["png_response"]["post"], dtype=np.float64),
        "pre_quadratic_1loop": np.asarray(assets["quadratic"]["pre"], dtype=np.float64),
        "post_quadratic_1loop": np.asarray(assets["quadratic"]["post"], dtype=np.float64),
    }
    for stage in STAGES:
        entries = [profiles[stage][f"kmax{cut:.2f}"] for cut in active_kmax_grid]
        save_arrays[f"{stage}_n_data"] = np.asarray([entry["n_data"] for entry in entries], dtype=np.int64)
        save_arrays[f"{stage}_hartlap"] = np.asarray([entry["hartlap"] for entry in entries])
        for field in (
            "profile_fNL",
            "profile_lower",
            "profile_upper",
            "profile_half_width",
            "truth_delta_chi2",
            "truth_signed_pull",
            "chi2_min",
            "chi2_per_dof",
            "p_value",
        ):
            save_arrays[f"{stage}_{field}"] = np.asarray(
                [[row[field] for row in entry["rows"]] for entry in entries], dtype=np.float64
            )
        for target_name, _ in TARGETS:
            grid = primary_grids[stage][target_name]
            save_arrays[f"{stage}_{target_name}_primary_fNL_grid"] = np.asarray(grid["fNL_grid"])
            save_arrays[f"{stage}_{target_name}_primary_delta_chi2"] = np.asarray(grid["delta_chi2"])
            save_arrays[f"{stage}_{target_name}_simulation_mean"] = np.mean(
                assets["simulation"][stage][target_name], axis=0
            )
        save_arrays[f"{stage}_fiducial_covariance_one_box_all_diagonal"] = np.cov(
            assets["simulation"][stage]["fiducial"], rowvar=False, ddof=1
        )
    np.savez_compressed(npz_path, **save_arrays)

    make_plot(
        pdf_path,
        profiles,
        float(args.volume),
        excluded_k,
        excluded_high_k,
        float(args.redshift),
        float(args.smoothing_radius),
        int(common.size),
        bool(assets["finite_fnl"]),
    )
    return json_path, npz_path, pdf_path


def main() -> None:
    """运行全部 profile、验证输出并打印会议摘要。"""

    args = parse_args()
    require(float(np.max(KMAX_GRID)) <= 0.2 + 1.0e-12, "registered kmax grid exceeds 0.2")
    assets = load_assets(args.theory_bundle, args.omit_quadratic, args.simulation_npz)
    products = run_profiles(
        assets,
        float(args.volume),
        float(args.primary_kmax),
        int(args.exclude_first_diagonal_bins),
        int(args.exclude_last_diagonal_bins),
    )
    json_path, npz_path, pdf_path = save_outputs(args=args, assets=assets, products=products)
    summary = build_summary(products["profiles"], float(args.primary_kmax))

    print(
        f"DM z={args.redshift:g} R={args.smoothing_radius:g} "
        "one-loop fNL profile recovery complete"
    )
    for stage in STAGES:
        print(f"[{stage}] largest all-truth 68% kmax: {summary[stage]['largest_scanned_kmax_with_all_truths_inside_delta_chi2_1']}")
        for row in summary[stage]["primary_rows"]:
            print(
                f"  {row['target_name']:10s} input={row['input_fNL']:+7.1f} "
                f"profile={row['profile_fNL']:+9.3f} "
                f"interval=[{row['profile_lower']:+9.3f},{row['profile_upper']:+9.3f}] "
                f"truth_dchi2={row['truth_delta_chi2']:.4f}"
            )
    print(f"JSON: {json_path}")
    print(f"NPZ:  {npz_path}")
    print(f"PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
