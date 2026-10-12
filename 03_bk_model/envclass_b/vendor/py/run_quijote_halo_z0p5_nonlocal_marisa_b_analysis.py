#!/usr/bin/env python3
"""Quijote 单盒体积下 EQ/OR-LSS 的 MARISA-B ``B000`` 闭合与限制。

代码大纲
--------
1. 读取并核验冻结的 Fid500 covariance、matched ``+100/-100`` 响应和
   ``B000`` triangle-bin contract；所有 likelihood covariance 都对应
   Quijote 原始单盒体积 ``1 (Gpc/h)^3``。
2. 在固定宇宙学下拟合 Gaussian MARISA-B 模型，只保留
   ``(b1,b2,gamma2,Ashot,Bshot)``；线性功率振幅不是参数。
3. 读取精确 shell-average 的 EQ/OR-LSS ``B111`` 与四个 PNG one-loop
   topology，并用冻结的 reconstruction polynomial 生成 pre/post 响应。
4. 对每个 shape、每个 reconstruction 和每个 ``kmax`` 独立计算
   conditional 与 six-parameter joint Fisher 信息、simulation/theory closure、
   matched bootstrap、50/50 split 以及 post-minus-pre increment。
5. 仅在 ``kmax=0.10 h/Mpc`` 做六参数 profile likelihood，并按预先冻结的
   门槛给 EQ 与 OR-LSS 分别判定 PASS/FAIL；绝不做双 shape 联合限制。
6. 原子写出 JSON/NPZ/CSV/Markdown 和仅 PDF 的诊断图，保留所有中间向量
   以便复算和审计。

本脚本不使用 ``P(k)`` likelihood、STO catalogue、local PNG、经验响应增益、
Gaussian nuisance prior 或 ``k>0.10 h/Mpc`` 的 bin。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "MARISA_B_MAX_THREADS",
):
    os.environ[_name] = "1"
# The old z=1 cache is read-only here.  Its current executable hash changed
# because the same driver gained shape-generic nonlocal support; its frozen
# source projection, inputs, manifest and cached arrays remain authenticated.
os.environ["MARISA_B_ALLOW_FROZEN_CACHE_WITH_STALE_EXECUTABLE"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2 as chi2_distribution


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = ROOT / "codes"
TAG = "quijote_halo_z0p5_nonlocal_marisa_b_kmax0p10"
ANALYSIS = ROOT / "analysis" / TAG
THEORY = ANALYSIS / "theory/NONLOCAL_PNG_RESPONSE_THEORY.npz"
THEORY_MANIFEST = ANALYSIS / "theory/NONLOCAL_PNG_RESPONSE_THEORY_MANIFEST.json"
CONVERGENCE = ANALYSIS / "theory/convergence/NUMERICAL_CONVERGENCE.npz"
CONVERGENCE_REPORT = ANALYSIS / "theory/convergence/NUMERICAL_CONVERGENCE.json"
FROZEN_INPUTS = ANALYSIS / "FROZEN_INPUT_MANIFEST.json"
GAUSSIAN_OUTPUT = ANALYSIS / "GAUSSIAN_BASELINE.json"
GAUSSIAN_ARRAYS = ANALYSIS / "GAUSSIAN_BASELINE_ARRAYS.npz"
OUTPUT = ANALYSIS / "NONLOCAL_MARISA_B_RESULTS.json"
ARRAY_OUTPUT = ANALYSIS / "NONLOCAL_MARISA_B_ARRAYS.npz"
CSV_OUTPUT = ANALYSIS / "NONLOCAL_MARISA_B_RESULTS.csv"
REPORT = ANALYSIS / "NONLOCAL_MARISA_B_REPORT.md"
FIGURES = ROOT / "figures/diagnostics" / TAG

REFINED_PROFILE = CODES / "run_quijote_halo_z0p5_standard_refined_b000_profiles.py"
REFINED_ASSETS = CODES / "quijote_halo_z0p5_standard_refined_assets.py"
FID_MATRIX = (
    ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100/"
    "matrices/baseline_pre_standard_matrix.npz"
)
SHAPE_MATRICES = {
    "EQ": ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_eqp100_eqm100/"
    "matrices/baseline_eq_centered_matrix.npz",
    "OR_LSS": ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_orlssp100_orlssm100/"
    "matrices/baseline_orlss_centered_matrix.npz",
}
RESPONSE_KEYS = {
    "EQ": {
        "pre": "pre_dB000_dfNL_EQp_minus_EQm_samples",
        "standard": "post_dB000_dfNL_EQp_minus_EQm_samples",
    },
    "OR_LSS": {
        "pre": "centered_pre_dB000_dfNL_samples",
        "standard": "centered_post_dB000_dfNL_samples",
    },
}
CUTS = (0.05, 0.06, 0.08, 0.10)
METHODS = ("pre", "standard")
METHOD_LABELS = {"pre": "pre recon", "standard": "post recon"}
SHAPES = ("EQ", "OR_LSS")
NUISANCE_NAMES = ("b1", "b2", "gamma2", "Ashot", "Bshot")
B_REC = 1.9326526728794324
VOLUME = 1.0
BOOTSTRAPS = 4000
FNL_BOUNDS = (-5000.0, 5000.0)
B1_BOUNDS = (0.1, 6.0)

PDF_METADATA = {
    "Author": "MARISA-B nonlocal PNG goal",
    "Creator": Path(__file__).name,
    "Subject": "Quijote halo z=0.5 EQ and OR-LSS B000-only constraints",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def import_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@dataclass
class Runtime:
    """一次分析运行所需的只读输入与外部 MARISA-B context。

    ``contexts`` 按 ``kmax`` 和 reconstruction 保存 likelihood context；
    ``matrix`` 是 Fid500 基线矩阵；``shape_matrices`` 是两种 PNG 的 100 对
    matched derivative；其余字段记录 Gaussian theory asset 与来源模块。
    """

    contexts: dict[float, dict[str, Any]]
    matrix: dict[str, np.ndarray]
    shape_matrices: dict[str, dict[str, np.ndarray]]
    refinement: dict[str, Any]
    modules: dict[str, Any]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def load_runtime() -> Runtime:
    """加载全部冻结输入，并逐项验证 bin 和单盒 covariance contract。

    返回值中的每个 context 已被强制设为 ``VOLUME=1``；若外部 context 的
    triangle indices 或 covariance 与冻结文件不一致，函数立即失败，防止
    在不知情时混入旧体积缩放或另一套 mask。
    """
    for path in (REFINED_PROFILE, REFINED_ASSETS, FID_MATRIX, FROZEN_INPUTS, *SHAPE_MATRICES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = import_path("_nonlocal_profile_context", REFINED_PROFILE)
    projected = import_path("_nonlocal_projected_context", profile.PROJECTED_RUNNER)
    refiner = import_path("_nonlocal_refined_assets", REFINED_ASSETS)
    baseline, physical, base, audit, assets, standard_method, physical_cache = (
        projected.load_modules_and_assets()
    )
    assets, refinement = refiner.refined_assets(assets)
    projected.VOLUME = VOLUME
    base.VOLUME = VOLUME
    audit.VOLUME = VOLUME
    matrix = projected.load_matrix()
    tree = projected.load_pre_tree(matrix)
    dm_tree, dm_total, _dm_audit = projected.load_z05_dm_response(matrix)
    bias = json.loads(projected.BIAS_JSON.read_text(encoding="utf-8"))
    growth_ratio = float(base.legacy.configurations()["z0p5"].growth_power_ratio_to_z1)
    nbar = float(bias["nbar_mean"])
    contexts = {
        cut: profile.build_contexts(
            projected, physical, base, assets, standard_method, physical_cache,
            matrix, tree, dm_tree, dm_total, growth_ratio, nbar, cut, refinement,
        )
        for cut in CUTS
    }
    for cut, pair in contexts.items():
        expected = np.asarray(
            json.loads(FROZEN_INPUTS.read_text(encoding="utf-8"))
            ["B000_bin_contract"]["cuts"][f"{cut:.2f}"]["indices"],
            dtype=int,
        )
        for method, context in pair.items():
            if not np.array_equal(np.asarray(context.indices, dtype=int), expected):
                raise ValueError(f"{method}/{cut}: context index contract drift")
            covariance = np.cov(np.asarray(context.samples), rowvar=False, ddof=1)
            if not np.allclose(context.covariance_single, covariance, rtol=2e-14, atol=1e-6):
                raise ValueError(f"{method}/{cut}: context is not on one-box covariance")
    shape_matrices = {shape: load_npz(path) for shape, path in SHAPE_MATRICES.items()}
    return Runtime(
        contexts=contexts,
        matrix=matrix,
        shape_matrices=shape_matrices,
        refinement=refinement,
        modules={
            "profile": profile, "projected": projected, "refiner": refiner,
            "baseline": baseline, "physical": physical, "base": base,
            "audit": audit, "assets": assets,
        },
    )


def selected(vector: np.ndarray, context: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape == (120,):
        return array[np.asarray(context.indices, dtype=int)]
    if array.shape == (len(context.indices),):
        return array
    raise ValueError(f"unexpected theory-vector shape {array.shape}")


def gaussian_components(
    context: Any,
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    """拆分固定宇宙学的 Gaussian halo bispectrum 预测。

    参数 ``nuisance`` 的顺序严格为
    ``(b1,b2,gamma2,Ashot,Bshot)``。返回 tree、DM one-loop、两个随机项和
    总和；这里没有线性功率振幅自由度，所有 cosmological inputs 均固定。
    """
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if nuisance.shape != (5,):
        raise ValueError("Gaussian nuisance vector must have five entries")
    _prediction, _parameters, components = context.evaluate_model(0.0, nuisance)
    unit_a = nuisance.copy()
    unit_a[3:] = (1.0, 0.0)
    _pa, _qa, components_a = context.evaluate_model(0.0, unit_a)
    unit_b = nuisance.copy()
    unit_b[3:] = (0.0, 1.0)
    _pb, _qb, components_b = context.evaluate_model(0.0, unit_b)
    tree = selected(components["gaussian_tree"], context)
    loop = selected(components["gaussian_matter_loop"], context)
    ashot_unit = selected(components_a["gaussian_stochastic"], context)
    bshot_unit = selected(components_b["gaussian_stochastic"], context)
    stochastic_a = nuisance[3] * ashot_unit
    stochastic_b = nuisance[4] * bshot_unit
    return {
        "tree": tree,
        "loop": loop,
        "Ashot": stochastic_a,
        "Bshot": stochastic_b,
        "stochastic": stochastic_a + stochastic_b,
        "total": tree + loop + stochastic_a + stochastic_b,
        "Ashot_unit": ashot_unit,
        "Bshot_unit": bshot_unit,
    }


def gaussian_model(context: Any, nuisance: np.ndarray) -> np.ndarray:
    return gaussian_components(context, nuisance)["total"]


def covariance_contract(samples: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """从 mock 样本构造 sample covariance、Hartlap 因子和有效 Cholesky。

    ``chol`` 满足用 ``whiten(chol, x)`` 计算
    ``x^T [Hartlap*C^{-1}] x``；输入 covariance 不除以 mock 数，因为目标是
    一个 Quijote 单盒观测，而不是 Fid500 均值的误差。
    """
    samples = np.asarray(samples, dtype=np.float64)
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    n_mock, n_data = samples.shape
    hartlap = (n_mock - n_data - 2.0) / (n_mock - 1.0)
    if hartlap <= 0.0:
        raise ValueError("non-positive Hartlap correction")
    chol = np.linalg.cholesky(covariance) / math.sqrt(hartlap)
    return covariance, hartlap, chol


def whiten(chol: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return solve_triangular(chol, np.asarray(vector, dtype=np.float64), lower=True)


def scaled_rank(matrix: np.ndarray, rcond: float = 1.0e-10) -> int:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0
    norms = np.linalg.norm(matrix, axis=0)
    keep = norms > np.finfo(float).tiny
    if not np.any(keep):
        return 0
    normalized = matrix[:, keep] / norms[keep]
    singular = np.linalg.svd(normalized, compute_uv=False)
    return int(np.count_nonzero(singular > rcond * singular[0]))


def finite_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
) -> np.ndarray:
    """用对称有限差分计算模型对 nuisance 参数的局部导数矩阵。"""
    point = np.asarray(point, dtype=np.float64)
    columns = []
    for index, value in enumerate(point):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((function(plus) - function(minus)) / (2.0 * step))
    return np.column_stack(columns)


def gaussian_fit(context: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """拟合单个 reconstruction 的五参数 Gaussian 基线。

    使用三个 ``b1`` 起点避免局部优化依赖；只对 ``b1`` 施加物理硬边界，
    不加 Gaussian prior。返回可序列化的 GOF 记录及模型、残差、Jacobian 等
    数组，后者会进入 PNG nuisance projection。
    """
    covariance, hartlap, chol = covariance_contract(context.samples)
    target = np.asarray(context.target, dtype=np.float64)

    def prediction(theta: np.ndarray) -> np.ndarray:
        return gaussian_model(context, np.asarray(theta, dtype=np.float64))

    lower = np.asarray((B1_BOUNDS[0], -np.inf, -np.inf, -np.inf, -np.inf))
    upper = np.asarray((B1_BOUNDS[1], np.inf, np.inf, np.inf, np.inf))
    starts = [np.asarray((b1, 0.0, 0.0, 0.0, 0.0)) for b1 in (1.5, 2.0, 2.5)]
    best = None
    for start in starts:
        fit = least_squares(
            lambda theta: whiten(chol, prediction(theta) - target),
            start, bounds=(lower, upper), method="trf", x_scale="jac",
            xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12, max_nfev=5000,
        )
        objective = float(2.0 * fit.cost)
        if best is None or objective < best[0]:
            best = (objective, fit)
    assert best is not None
    objective, fit = best
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Gaussian fit failed: {fit.message}")
    theta = np.asarray(fit.x)
    model = prediction(theta)
    jacobian = finite_jacobian(prediction, theta)
    jacobian_white = np.column_stack([whiten(chol, jacobian[:, column]) for column in range(jacobian.shape[1])])
    rank = scaled_rank(jacobian_white)
    dof = target.size - rank
    names = NUISANCE_NAMES
    components = gaussian_components(context, theta)
    record = {
        "parameter_names": names,
        "map": {name: value for name, value in zip(names, theta)},
        "data_chi2": objective,
        "jacobian_rank": rank,
        "dof": dof,
        "gof_p_value": float(chi2_distribution.sf(objective, dof)) if dof > 0 else None,
        "cosmology_fixed": True,
        "no_Gaussian_parameter_priors": True,
        "bounds": {"b1": B1_BOUNDS},
        "optimizer": {
            "success": bool(fit.success), "message": str(fit.message),
            "nfev": int(fit.nfev), "optimality": float(fit.optimality),
        },
    }
    arrays = {
        "target": target, "model": model, "residual": target - model,
        "covariance": covariance, "chol_fit": chol, "jacobian": jacobian,
        "theta": theta,
        **{f"component_{name}": value for name, value in components.items()},
    }
    return record, arrays


def split_joint(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    physical = theta[:3]
    pre = np.concatenate((physical, theta[3:5]))
    standard = np.concatenate((physical, theta[5:7]))
    return pre, standard


def joint_gaussian_fit(
    pre: Any,
    standard: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """联合拟合 pre/post 的 Gaussian 基线以检验共享 halo bias。

    这里的 ``joint`` 只指同一 Fid500 realization 的 pre 与 post Gaussian
    数据块共享 ``(b1,b2,gamma2)``，两个方法各有 ``Ashot/Bshot``；它不是、
    也不会产生 EQ 与 OR-LSS 的双-shape 联合约束。
    """
    if not np.array_equal(pre.indices, standard.indices):
        raise ValueError("joint pre/standard index mismatch")
    samples = np.column_stack((pre.samples, standard.samples))
    target = np.mean(samples, axis=0)
    covariance, hartlap, chol = covariance_contract(samples)

    def prediction(theta: np.ndarray) -> np.ndarray:
        nuisance_pre, nuisance_standard = split_joint(theta)
        return np.concatenate((
            gaussian_model(pre, nuisance_pre),
            gaussian_model(standard, nuisance_standard),
        ))

    lower = np.asarray((B1_BOUNDS[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf))
    upper = np.asarray((B1_BOUNDS[1], np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
    starts = [
        np.asarray((b1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        for b1 in (1.5, 2.0, 2.5)
    ]
    best = None
    for start in starts:
        fit = least_squares(
            lambda theta: whiten(chol, prediction(theta) - target),
            start, bounds=(lower, upper), method="trf", x_scale="jac",
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=7000,
        )
        objective = float(2.0 * fit.cost)
        if best is None or objective < best[0]:
            best = (objective, fit)
    assert best is not None
    objective, fit = best
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"joint Gaussian fit failed: {fit.message}")
    theta = np.asarray(fit.x)
    jacobian = finite_jacobian(prediction, theta)
    jacobian_white = np.column_stack([whiten(chol, jacobian[:, column]) for column in range(jacobian.shape[1])])
    rank = scaled_rank(jacobian_white)
    dof = target.size - rank
    names = ("b1", "b2", "gamma2", "Ashot_pre", "Bshot_pre", "Ashot_standard", "Bshot_standard")
    nuisance_pre, nuisance_standard = split_joint(theta)
    model = prediction(theta)
    n = len(pre.indices)
    delta_samples = standard.samples - pre.samples
    delta_covariance, delta_hartlap, delta_chol = covariance_contract(delta_samples)
    delta_target = np.mean(delta_samples, axis=0)
    delta_model = model[n:] - model[:n]
    delta_chi2 = float(np.sum(whiten(delta_chol, delta_model - delta_target) ** 2))
    record = {
        "parameter_names": names,
        "map": {name: value for name, value in zip(names, theta)},
        "pre_nuisance": {name: value for name, value in zip(NUISANCE_NAMES, nuisance_pre)},
        "standard_nuisance": {name: value for name, value in zip(NUISANCE_NAMES, nuisance_standard)},
        "data_chi2": objective,
        "jacobian_rank": rank,
        "dof": dof,
        "gof_p_value": float(chi2_distribution.sf(objective, dof)) if dof > 0 else None,
        "paired_increment": {
            "chi2": delta_chi2,
            "dof_no_refit": n,
            "p_value_no_refit": float(chi2_distribution.sf(delta_chi2, n)),
            "hartlap": delta_hartlap,
        },
        "cosmology_fixed": True,
        "no_Gaussian_parameter_priors": True,
        "optimizer": {"success": bool(fit.success), "message": str(fit.message), "nfev": int(fit.nfev)},
    }
    arrays = {
        "target": target, "model": model, "covariance": covariance,
        "chol_fit": chol, "jacobian": jacobian, "theta": theta,
        "delta_target": delta_target, "delta_model": delta_model,
        "delta_covariance": delta_covariance,
    }
    return record, arrays


def run_gaussian(runtime: Runtime) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """在四个 ``kmax`` 上运行固定宇宙学 Gaussian 基线并原子落盘。"""
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for cut in CUTS:
        contexts = runtime.contexts[cut]
        for method in METHODS:
            key = f"{method}__kmax{cut:.2f}"
            fixed_record, fixed_arrays = gaussian_fit(contexts[method])
            rows[key] = fixed_record
            for name, value in fixed_arrays.items():
                arrays[f"{key}__{name}"] = np.asarray(value)
        joint_fixed, joint_fixed_arrays = joint_gaussian_fit(contexts["pre"], contexts["standard"])
        key = f"joint__kmax{cut:.2f}"
        rows[key] = joint_fixed
        for name, value in joint_fixed_arrays.items():
            arrays[f"{key}__{name}"] = np.asarray(value)
        print(
            f"Gaussian kmax={cut:.2f}: pre p={rows[f'pre__kmax{cut:.2f}']['gof_p_value']} "
            f"standard p={rows[f'standard__kmax{cut:.2f}']['gof_p_value']} "
            f"joint p={joint_fixed['gof_p_value']}",
            flush=True,
        )
    target = rows["joint__kmax0.10"]
    gate = bool(
        target["gof_p_value"] is not None
        and float(target["gof_p_value"]) > 0.05
        and float(target["paired_increment"]["p_value_no_refit"]) > 0.05
    )
    payload = {
        "schema": "quijote-z0p5-nonlocal-marisa-b-gaussian-baseline-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "volume_Gpc3_h3": VOLUME,
        "observable": "B000 only",
        "cosmology": "fixed; linear-power amplitude is not an inference parameter",
        "parameter_priors": "none; physical hard bounds only on b1",
        "rows": rows,
        "target_kmax0p10_gate": {
            "joint_fixed_cosmology_p_above_0p05": target["gof_p_value"] > 0.05,
            "paired_increment_p_above_0p05": target["paired_increment"]["p_value_no_refit"] > 0.05,
            "pass": gate,
        },
        "inputs": {
            "Fid500": {"path": FID_MATRIX, "sha256": sha256(FID_MATRIX)},
            "frozen_inputs": {"path": FROZEN_INPUTS, "sha256": sha256(FROZEN_INPUTS)},
            "refined_assets": {"path": REFINED_ASSETS, "sha256": sha256(REFINED_ASSETS)},
            "frozen_cache_executable_drift_policy": (
                "explicit cache-only opt-in; live old executable is never run"
            ),
        },
    }
    atomic_npz(GAUSSIAN_ARRAYS, arrays)
    payload["arrays"] = {"path": GAUSSIAN_ARRAYS, "sha256": sha256(GAUSSIAN_ARRAYS)}
    atomic_json(GAUSSIAN_OUTPUT, payload)
    return payload, arrays


def load_theory() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """读取主理论与收敛产物，并核验状态、哈希及 triangle indices。"""
    for path in (THEORY, THEORY_MANIFEST, CONVERGENCE, CONVERGENCE_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(THEORY_MANIFEST.read_text(encoding="utf-8"))
    convergence = json.loads(CONVERGENCE_REPORT.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or convergence.get("status") != "complete":
        raise ValueError("theory or convergence manifest is incomplete")
    if manifest.get("output", {}).get("sha256") != sha256(THEORY):
        raise ValueError("theory NPZ hash drift")
    if convergence.get("output", {}).get("sha256") != sha256(CONVERGENCE):
        raise ValueError("convergence NPZ hash drift")
    arrays = load_npz(THEORY)
    expected = np.asarray(
        json.loads(FROZEN_INPUTS.read_text(encoding="utf-8"))
        ["B000_bin_contract"]["cuts"]["0.10"]["indices"],
        dtype=int,
    )
    if not np.array_equal(arrays["selected_indices"], expected):
        raise ValueError("theory/frozen selected-index mismatch")
    return arrays, {"theory": manifest, "convergence": convergence}


def polynomial(coefficients: np.ndarray, value: float) -> np.ndarray:
    """用 Horner 法计算 reconstruction-strength 的二次多项式。"""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    result = np.zeros(coefficients.shape[1], dtype=np.float64)
    for power in range(coefficients.shape[0] - 1, -1, -1):
        result = result * value + coefficients[power]
    return result


def theory_component(
    theory: dict[str, np.ndarray],
    shape: str,
    method: str,
    context: Any,
    b1: float,
    component: str,
) -> np.ndarray:
    """生成指定 shape/reconstruction/context 的 halo PNG 响应分量。

    pre 使用 ``lambda=0``，standard 使用 ``lambda=b1/b_rec``。理论文件保存
    matter response；本函数乘 ``b1^3`` 得到本工作冻结模型中的 halo 响应，
    再按 context indices 截取所需 bin。``component`` 可为 tree、loop、四个
    one-loop topology 中任一个，或 tree+loop 的 total。
    """
    prefix = shape.lower()
    if component == "total":
        return theory_component(theory, shape, method, context, b1, "tree") + theory_component(
            theory, shape, method, context, b1, "loop"
        )
    coefficients = np.asarray(
        theory[f"{prefix}_{component}_lambda_coefficients"], dtype=np.float64
    )
    lambda_value = 0.0 if method == "pre" else float(b1) / B_REC
    full = polynomial(coefficients, lambda_value)
    lookup = {
        int(index): location
        for location, index in enumerate(np.asarray(theory["selected_indices"], dtype=int))
    }
    positions = np.asarray([lookup[int(index)] for index in context.indices], dtype=int)
    return float(b1) ** 3 * full[positions]


def metric_from_vectors(theory: np.ndarray, empirical: np.ndarray) -> dict[str, Any]:
    """在已白化空间比较 theory 与 simulation 响应。

    返回两者 Fisher information、各自 ``sigma(fNL)``、将 simulation 投影到
    theory 上的最佳振幅和 precision-weighted cosine；这里的“振幅”只是
    closure 诊断量，绝不是 likelihood 参数，也不会重标定理论。
    """
    theory = np.asarray(theory, dtype=np.float64)
    empirical = np.asarray(empirical, dtype=np.float64)
    tt = float(theory @ theory)
    ee = float(empirical @ empirical)
    te = float(theory @ empirical)
    if tt <= 1.0e-28 or ee <= 1.0e-28:
        return {
            "identifiable": False,
            "information_theory": tt,
            "information_simulation": ee,
            "sigma_theory": None,
            "sigma_simulation": None,
            "simulation_on_theory_amplitude": None,
            "precision_weighted_cosine": None,
        }
    return {
        "identifiable": True,
        "information_theory": tt,
        "information_simulation": ee,
        "sigma_theory": 1.0 / math.sqrt(tt),
        "sigma_simulation": 1.0 / math.sqrt(ee),
        "simulation_on_theory_amplitude": te / tt,
        "precision_weighted_cosine": te / math.sqrt(tt * ee),
    }


def projector_from_nuisance(
    chol: np.ndarray,
    nuisance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """构造白化空间中与 nuisance tangent 正交的投影算符。"""
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if nuisance.ndim != 2:
        raise ValueError("nuisance Jacobian must be a matrix")
    white = np.column_stack([
        whiten(chol, nuisance[:, column]) for column in range(nuisance.shape[1])
    ]) if nuisance.shape[1] else np.zeros((nuisance.shape[0], 0))
    rank = scaled_rank(white)
    projector = np.eye(nuisance.shape[0]) - white @ np.linalg.pinv(white, rcond=1.0e-10)
    if rank >= nuisance.shape[0]:
        projector = np.zeros((nuisance.shape[0], nuisance.shape[0]), dtype=np.float64)
    return projector, white, rank


def response_metric_bundle(
    chol: np.ndarray,
    theory: np.ndarray,
    empirical: np.ndarray,
    bias_jacobian: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """同时计算固定 nuisance 与六参数联合拟合对应的投影响应指标。"""
    theory_white = whiten(chol, theory)
    empirical_white = whiten(chol, empirical)
    bias_projector, bias_white, bias_rank = projector_from_nuisance(chol, bias_jacobian)
    projected_theory = bias_projector @ theory_white
    projected_empirical = bias_projector @ empirical_white
    # 当 nuisance tangent 已张满数据空间时，浮点 pinv 仍会留下约 1e-15 的
    # 伪残差；显式置零，避免把数学上不可识别的低维 cut 报成巨大但有限 sigma。
    if bias_rank >= theory_white.size:
        projected_theory = np.zeros_like(theory_white)
        projected_empirical = np.zeros_like(empirical_white)
    bundles = {
        "conditional": metric_from_vectors(theory_white, empirical_white),
        "joint_six_parameter": metric_from_vectors(
            projected_theory, projected_empirical
        ),
    }
    bundles["joint_six_parameter"]["nuisance_rank"] = bias_rank
    return bundles, {
        "theory_white": theory_white,
        "empirical_white": empirical_white,
        "bias_projector": bias_projector,
        "bias_nuisance_white": bias_white,
    }


def quantiles(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "q2p5": None, "q16": None, "median": None, "q84": None, "q97p5": None, "std": None}
    q = np.quantile(finite, (0.025, 0.16, 0.5, 0.84, 0.975))
    return {
        "count": int(finite.size), "q2p5": float(q[0]), "q16": float(q[1]),
        "median": float(q[2]), "q84": float(q[3]), "q97p5": float(q[4]),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
    }


def bootstrap_response(
    samples: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """对 100 个 matched derivative realization 做成对 bootstrap。

    每次 multinomial 重采样始终让 ``+100`` 与 ``-100`` 使用同一 realization
    权重；输出响应振幅、cosine 和 simulation-derivative Fisher sigma 的
    分位数，不把 bootstrap 均值当成额外数据集。
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape[0] != 100:
        raise ValueError("matched bootstrap requires 100 response realizations")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(100, np.full(100, 0.01), size=BOOTSTRAPS) / 100.0
    means = weights @ samples
    chol = metric_arrays["chol"]
    means_white = solve_triangular(chol, means.T, lower=True).T
    theory_white = metric_arrays["theory_white"]
    records: dict[str, Any] = {}
    distributions: dict[str, np.ndarray] = {}
    projectors = {
        "conditional": np.eye(theory_white.size),
        "joint_six_parameter": metric_arrays["bias_projector"],
    }
    for tier, projector in projectors.items():
        t = projector @ theory_white
        e = means_white @ projector.T
        tt = float(t @ t)
        ee = np.einsum("ij,ij->i", e, e)
        te = e @ t
        if tt <= 1.0e-28:
            amplitude = np.full(BOOTSTRAPS, np.nan)
            cosine = np.full(BOOTSTRAPS, np.nan)
        else:
            amplitude = te / tt
            cosine = te / np.sqrt(np.maximum(tt * ee, np.finfo(float).tiny))
        sigma = 1.0 / np.sqrt(np.maximum(ee, np.finfo(float).tiny))
        records[tier] = {
            "amplitude": quantiles(amplitude),
            "precision_weighted_cosine": quantiles(cosine),
            "simulation_derivative_sigma": quantiles(sigma),
        }
        distributions[f"{tier}__amplitude"] = amplitude
        distributions[f"{tier}__cosine"] = cosine
        distributions[f"{tier}__sigma"] = sigma
    return {
        "n_bootstrap": BOOTSTRAPS,
        "n_matched_realizations": 100,
        "seed": seed,
        "tiers": records,
    }, distributions


def split_response_metrics(
    samples: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """分别用 realization 000--049 与 050--099 检查响应闭合稳定性。"""
    results = {}
    for label, section in (("real000_049", slice(0, 50)), ("real050_099", slice(50, 100))):
        empirical = np.mean(np.asarray(samples)[section], axis=0)
        bundle, _arrays = response_metric_bundle(
            metric_arrays["chol"], metric_arrays["theory"], empirical,
            metric_arrays["bias_jacobian"],
        )
        results[label] = bundle
    return results


def response_residual_gof(
    samples: np.ndarray,
    theory: np.ndarray,
    nuisance: np.ndarray,
) -> dict[str, Any]:
    """用 derivative 均值 covariance 检验 theory-simulation 残差。

    covariance 先由 100 条 derivative 样本估计，再除以 100；随后在 nuisance
    tangent 的正交补中计算 chi-square 和自由度。
    """
    samples = np.asarray(samples, dtype=np.float64)
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1)) / samples.shape[0]
    n_mock, n_data = samples.shape
    hartlap = (n_mock - n_data - 2.0) / (n_mock - 1.0)
    chol = np.linalg.cholesky(covariance) / math.sqrt(hartlap)
    projector, _white, rank = projector_from_nuisance(chol, nuisance)
    residual = whiten(chol, np.mean(samples, axis=0) - theory)
    projected = projector @ residual
    dof = n_data - rank
    value = float(projected @ projected)
    return {
        "chi2": value,
        "dof": dof,
        "p_value": float(chi2_distribution.sf(value, dof)) if dof > 0 else None,
        "nuisance_rank": rank,
        "derivative_mean_covariance": "matched sample covariance divided by 100",
        "Hartlap_100": hartlap,
    }


def fisher_summary(
    chol: np.ndarray,
    response: np.ndarray,
    bias_jacobian: np.ndarray,
    nuisance_names: tuple[str, ...] = NUISANCE_NAMES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """计算 fNL-only 与 ``fNL+五个 halo 参数`` 的联合 Fisher 限制。"""
    zero = np.zeros_like(response)
    bundle, arrays = response_metric_bundle(
        chol, response, zero + response,
        bias_jacobian,
    )
    result = {
        tier: {
            "sigma_fNL": metrics["sigma_theory"],
            "information": metrics["information_theory"],
            "identifiable": metrics["identifiable"],
            **({"nuisance_rank": metrics["nuisance_rank"]} if "nuisance_rank" in metrics else {}),
        }
        for tier, metrics in bundle.items()
    }
    t = arrays["theory_white"]
    nuisance = arrays["bias_nuisance_white"]
    cosines = {}
    if len(nuisance_names) != nuisance.shape[1]:
        raise ValueError("nuisance-name/Jacobian mismatch")
    for location, name in enumerate(nuisance_names):
        vector = nuisance[:, location]
        denominator = math.sqrt(float(t @ t) * float(vector @ vector))
        cosines[name] = float(t @ vector / denominator) if denominator > 0.0 else None
    result["joint_six_parameter"]["individual_nuisance_cosines"] = cosines
    full = np.column_stack((t, nuisance))
    fisher = full.T @ full
    column_norms = np.linalg.norm(full, axis=0)
    scaled_design = full / np.maximum(column_norms, np.finfo(float).tiny)
    singular = np.linalg.svd(scaled_design, compute_uv=False)
    joint_rank = scaled_rank(full)
    parameter_names = ("fNL", *nuisance_names)
    covariance = np.full((len(parameter_names), len(parameter_names)), np.nan)
    correlation = np.full_like(covariance, np.nan)
    if joint_rank == len(parameter_names) and np.all(column_norms > 0.0):
        scaled_fisher = scaled_design.T @ scaled_design
        covariance_scaled = np.linalg.inv(scaled_fisher)
        covariance = covariance_scaled / np.outer(column_norms, column_norms)
        covariance = 0.5 * (covariance + covariance.T)
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        correlation = covariance / np.outer(
            standard_deviations, standard_deviations
        )
        fisher_sigma = result["joint_six_parameter"]["sigma_fNL"]
        if fisher_sigma is not None and not np.isclose(
            math.sqrt(covariance[0, 0]), fisher_sigma, rtol=2.0e-8, atol=1.0e-10
        ):
            raise ValueError("projected and inverse-Fisher fNL sigma disagree")
    result["joint_six_parameter"]["scaled_design_singular_values"] = singular
    result["joint_six_parameter"]["scaled_condition_number"] = (
        float(singular[0] / singular[-1]) if singular[-1] > 0.0 else None
    )
    result["joint_six_parameter"].update({
        "joint_parameter_names": parameter_names,
        "joint_design_rank": joint_rank,
        "joint_covariance_available": joint_rank == len(parameter_names),
        "joint_parameter_sigma": {
            name: (
                math.sqrt(covariance[index, index])
                if np.isfinite(covariance[index, index]) else None
            )
            for index, name in enumerate(parameter_names)
        },
        "joint_parameter_correlation": correlation,
    })
    return result, {
        **arrays,
        "full_fisher": fisher,
        "joint_covariance": covariance,
        "joint_parameter_correlation": correlation,
    }


def closure_one(
    *,
    theory_vector: np.ndarray,
    response_samples: np.ndarray,
    context: Any,
    nuisance: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """完成一个 shape、一个 reconstruction、一个 k-cut 的全部闭合检验。"""
    covariance, hartlap, chol = covariance_contract(context.samples)
    bias_jacobian = finite_jacobian(
        lambda point: gaussian_model(context, point), nuisance
    )
    empirical = np.mean(response_samples, axis=0)
    metrics, metric_arrays = response_metric_bundle(
        chol, theory_vector, empirical, bias_jacobian
    )
    metric_arrays.update({
        "chol": chol, "theory": theory_vector,
        "bias_jacobian": bias_jacobian,
    })
    bootstrap, distributions = bootstrap_response(response_samples, metric_arrays, seed)
    splits = split_response_metrics(response_samples, metric_arrays)
    fisher, fisher_arrays = fisher_summary(
        chol, theory_vector, bias_jacobian
    )
    sim_amp = metrics["joint_six_parameter"]
    theory_sigma = sim_amp["sigma_theory"]
    simulation_sigma = sim_amp["sigma_simulation"]
    sigma_fraction = (
        abs(float(simulation_sigma) / float(theory_sigma) - 1.0)
        if theory_sigma is not None and simulation_sigma is not None else None
    )
    record = {
        "metrics": metrics,
        "fisher": fisher,
        "theory_vs_simulation_joint_six_parameter_sigma_fractional_difference": sigma_fraction,
        "response_residual_gof_bias_projected": response_residual_gof(
            response_samples, theory_vector, bias_jacobian
        ),
        "bootstrap": bootstrap,
        "split_50_50": splits,
        "covariance": {
            "volume_Gpc3_h3": VOLUME, "mock_count": 500,
            "Hartlap": hartlap, "method_specific": True,
        },
    }
    arrays = {
        "theory": theory_vector, "empirical": empirical,
        "response_samples": response_samples,
        "bias_jacobian": bias_jacobian,
        "covariance": covariance, "chol_fit": chol,
        **{f"metric__{name}": value for name, value in metric_arrays.items()},
        **{f"fisher__{name}": value for name, value in fisher_arrays.items()},
        **{f"bootstrap__{name}": value for name, value in distributions.items()},
    }
    return record, arrays


def increment_closure(
    *,
    theory_pre: np.ndarray,
    theory_standard: np.ndarray,
    samples_pre: np.ndarray,
    samples_standard: np.ndarray,
    pre: Any,
    standard: Any,
    joint_theta: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """检验 matched ``standard-pre`` PNG 响应增量。

    simulation 和 Fid500 covariance 都先按 realization 做差，因此自动保留
    pre/post 相关性；投影方向由共享三项 halo bias 加两套 stochastic 参数
    构成，用于回答 reconstruction 新增的响应是否被理论正确预测。
    """
    joint_theta = np.asarray(joint_theta, dtype=np.float64)
    if joint_theta.shape != (7,):
        raise ValueError("fixed-cosmology shared Gaussian map must have seven entries")
    delta_samples = np.asarray(standard.samples) - np.asarray(pre.samples)
    covariance, hartlap, chol = covariance_contract(delta_samples)

    def delta_gaussian(theta: np.ndarray) -> np.ndarray:
        nuisance_pre, nuisance_standard = split_joint(theta)
        return (
            gaussian_model(standard, nuisance_standard)
            - gaussian_model(pre, nuisance_pre)
        )

    bias_jacobian = finite_jacobian(delta_gaussian, joint_theta)
    theory = np.asarray(theory_standard) - np.asarray(theory_pre)
    response_samples = np.asarray(samples_standard) - np.asarray(samples_pre)
    empirical = np.mean(response_samples, axis=0)
    metrics, metric_arrays = response_metric_bundle(
        chol, theory, empirical, bias_jacobian
    )
    metric_arrays.update({
        "chol": chol, "theory": theory,
        "bias_jacobian": bias_jacobian,
    })
    bootstrap, distributions = bootstrap_response(response_samples, metric_arrays, seed)
    fisher, fisher_arrays = fisher_summary(
        chol, theory, bias_jacobian,
        ("b1", "b2", "gamma2", "Ashot_pre", "Bshot_pre", "Ashot_standard", "Bshot_standard"),
    )
    record = {
        "metrics": metrics,
        "fisher": fisher,
        "response_residual_gof_bias_projected": response_residual_gof(
            response_samples, theory, bias_jacobian
        ),
        "bootstrap": bootstrap,
        "split_50_50": split_response_metrics(response_samples, metric_arrays),
        "covariance": {
            "definition": "Fid500 standard-minus-pre paired covariance",
            "volume_Gpc3_h3": VOLUME, "Hartlap": hartlap,
        },
    }
    arrays = {
        "theory": theory, "empirical": empirical,
        "response_samples": response_samples,
        "bias_jacobian": bias_jacobian,
        "covariance": covariance, "chol_fit": chol,
        **{f"metric__{name}": value for name, value in metric_arrays.items()},
        **{f"fisher__{name}": value for name, value in fisher_arrays.items()},
        **{f"bootstrap__{name}": value for name, value in distributions.items()},
    }
    return record, arrays


def profile_fit(
    *,
    context: Any,
    theory: dict[str, np.ndarray],
    shape: str,
    nuisance_start: np.ndarray,
    fisher_sigma: float | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """在 ``kmax=0.10`` 做六参数非线性 profile likelihood。

    参数为 ``(fNL,b1,b2,gamma2,Ashot,Bshot)``；固定 ``fNL`` 时重新优化后五项，
    用 ``Delta chi2=1`` 求左右 68% crossing。理论 PNG 响应随 ``b1^3`` 和
    standard 的 ``lambda=b1/b_rec`` 自洽变化。
    """
    _covariance, hartlap, chol = covariance_contract(context.samples)
    target = np.asarray(context.target, dtype=np.float64)
    nuisance_start = np.asarray(nuisance_start, dtype=np.float64)

    # The reconstruction label contains several suffixes; use membership
    # rather than parsing that internal label.
    method = "pre" if "pre" in context.reconstruction else "standard"

    def model(theta: np.ndarray) -> np.ndarray:
        fnl = float(theta[0])
        nuisance = np.asarray(theta[1:], dtype=np.float64)
        return gaussian_model(context, nuisance) + fnl * theory_component(
            theory, shape, method, context, nuisance[0], "total"
        )

    lower = np.asarray((FNL_BOUNDS[0], B1_BOUNDS[0], -np.inf, -np.inf, -np.inf, -np.inf))
    upper = np.asarray((FNL_BOUNDS[1], B1_BOUNDS[1], np.inf, np.inf, np.inf, np.inf))
    scale = max(float(fisher_sigma) if fisher_sigma is not None else 100.0, 1.0)
    starts = [
        np.concatenate(([fnl], nuisance_start))
        for fnl in (0.0, -scale, scale)
    ]
    best = None
    for start in starts:
        start = np.minimum(np.maximum(start, lower + 1e-10), upper - 1e-10)
        fit = least_squares(
            lambda theta: whiten(chol, model(theta) - target),
            start, bounds=(lower, upper), method="trf", x_scale="jac",
            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=8000,
        )
        objective = float(2.0 * fit.cost)
        if best is None or objective < best[0]:
            best = (objective, fit)
    assert best is not None
    minimum, fit = best
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"PNG profile fit failed: {fit.message}")
    theta_hat = np.asarray(fit.x)

    cache: dict[float, tuple[float, np.ndarray]] = {}

    def fixed_profile(fnl: float) -> float:
        key = float(fnl)
        if key in cache:
            return cache[key][0]
        candidates = [theta_hat[1:]]
        if cache:
            nearest = min(cache, key=lambda item: abs(item - key))
            candidates.append(cache[nearest][1])
        local_lower = lower[1:]
        local_upper = upper[1:]

        def fixed_model(parameters: np.ndarray) -> np.ndarray:
            return model(np.concatenate(([key], parameters)))

        local_best = None
        for start in candidates:
            local = least_squares(
                lambda parameters: whiten(chol, fixed_model(parameters) - target),
                np.minimum(np.maximum(start, local_lower + 1e-10), local_upper - 1e-10),
                bounds=(local_lower, local_upper), method="trf", x_scale="jac",
                xtol=2e-11, ftol=2e-11, gtol=2e-11, max_nfev=6000,
            )
            if not local.success or not np.all(np.isfinite(local.x)):
                continue
            objective = float(2.0 * local.cost)
            if local_best is None or objective < local_best[0]:
                local_best = (objective, np.asarray(local.x))
        if local_best is None:
            raise RuntimeError(f"fixed-fNL profile failed at fNL={key}")
        cache[key] = local_best
        return local_best[0]

    centre_profile = fixed_profile(float(theta_hat[0]))
    minimum = min(minimum, centre_profile)

    def crossing(sign: float) -> float | None:
        width = max(scale, 1.0)
        inner = float(theta_hat[0])
        outer = inner + sign * width
        outer = min(max(outer, FNL_BOUNDS[0]), FNL_BOUNDS[1])
        for _iteration in range(12):
            if fixed_profile(outer) - minimum >= 1.0:
                break
            if outer in FNL_BOUNDS:
                return None
            width *= 1.8
            outer = min(max(inner + sign * width, FNL_BOUNDS[0]), FNL_BOUNDS[1])
        else:
            return None
        left, right = sorted((inner, outer))
        return float(brentq(
            lambda value: fixed_profile(float(value)) - minimum - 1.0,
            left, right, xtol=1e-5, rtol=1e-9, maxiter=80,
        ))

    lower_crossing = crossing(-1.0)
    upper_crossing = crossing(1.0)
    zero_profile = fixed_profile(0.0)
    profile_points = sorted(cache)
    record = {
        "parameter_names": ("fNL", *NUISANCE_NAMES),
        "map": {
            name: value for name, value in zip(("fNL", *NUISANCE_NAMES), theta_hat)
        },
        "minimum_chi2": minimum,
        "profile_chi2_at_fNL0": zero_profile,
        "delta_chi2_fNL0": zero_profile - minimum,
        "delta_chi2_1_interval": {
            "lower": lower_crossing, "upper": upper_crossing,
            "closed": lower_crossing is not None and upper_crossing is not None,
        },
        "fisher_sigma_reference": fisher_sigma,
        "absolute_fNL_hat_over_fisher_sigma": (
            abs(float(theta_hat[0])) / float(fisher_sigma)
            if fisher_sigma is not None and fisher_sigma > 0.0 else None
        ),
        "cosmology_fixed": True,
        "no_parameter_priors": True,
        "numerical_scan_bounds": {"fNL": FNL_BOUNDS, "b1": B1_BOUNDS},
        "hartlap": hartlap,
        "optimizer": {"success": bool(fit.success), "message": str(fit.message), "nfev": int(fit.nfev)},
    }
    arrays = {
        "theta_hat": theta_hat,
        "profile_fNL": np.asarray(profile_points),
        "profile_chi2": np.asarray([cache[value][0] for value in profile_points]),
        "target": target,
        "model_hat": model(theta_hat),
    }
    return record, arrays


def fixed_nuisance(gaussian: dict[str, Any], method: str, cut: float) -> np.ndarray:
    mapping = gaussian["rows"][f"{method}__kmax{cut:.2f}"]["map"]
    return np.asarray([mapping[name] for name in NUISANCE_NAMES], dtype=np.float64)


def fixed_joint_theta(gaussian: dict[str, Any], cut: float) -> np.ndarray:
    names = (
        "b1", "b2", "gamma2", "Ashot_pre", "Bshot_pre",
        "Ashot_standard", "Bshot_standard",
    )
    mapping = gaussian["rows"][f"joint__kmax{cut:.2f}"]["map"]
    return np.asarray([mapping[name] for name in names], dtype=np.float64)


def response_samples(runtime: Runtime, shape: str, method: str, context: Any) -> np.ndarray:
    """读取 100 条 centered matched simulation derivative 并应用同一 bin cut。"""
    values = np.asarray(
        runtime.shape_matrices[shape][RESPONSE_KEYS[shape][method]],
        dtype=np.float64,
    )
    if values.shape != (100, 120):
        raise ValueError(f"{shape}/{method}: centered derivative shape drift")
    return values[:, np.asarray(context.indices, dtype=int)]


def numerical_convergence_audit(
    runtime: Runtime,
    theory: dict[str, np.ndarray],
    gaussian: dict[str, Any],
    manifests: dict[str, Any],
) -> dict[str, Any]:
    """把积分设置变化转换为 covariance-weighted 响应误差。

    收敛计算只覆盖三个代表 triangle，因此本函数明确把它标记为 geometry
    诊断而非完整替代理论；同时汇总 reconstruction polynomial holdout 残差、
    driver 报告的积分误差和 integrand evaluation 数。
    """
    convergence = load_npz(CONVERGENCE)
    component_names = tuple(str(item) for item in convergence["component_names"].tolist())
    total_location = component_names.index("total")
    indices = np.asarray(convergence["indices"], dtype=int)
    rows: dict[str, Any] = {}
    for config in (str(item) for item in convergence["configuration_names"].tolist()):
        rows[config] = {}
        for shape in SHAPES:
            rows[config][shape] = {}
            prefix = shape.lower()
            delta_all = np.asarray(convergence[f"{config}__{prefix}__delta"])
            reference_all = np.asarray(convergence[f"{config}__{prefix}__primary"])
            for method, lambda_location in (("pre", 0), ("standard", 1)):
                context = runtime.contexts[0.10][method]
                local_lookup = {int(index): location for location, index in enumerate(context.indices)}
                local = np.asarray([local_lookup[int(index)] for index in indices], dtype=int)
                _covariance, _hartlap, chol = covariance_contract(context.samples[:, local])
                b1 = fixed_nuisance(gaussian, method, 0.10)[0]
                delta = b1**3 * delta_all[lambda_location, :, total_location]
                reference = b1**3 * reference_all[lambda_location, :, total_location]
                delta_white = whiten(chol, delta)
                reference_white = whiten(chol, reference)
                norm_delta = float(np.linalg.norm(delta_white))
                norm_reference = float(np.linalg.norm(reference_white))
                rows[config][shape][method] = {
                    "representative_indices": indices,
                    "precision_norm_change_per_unit_fNL": norm_delta,
                    "fraction_of_primary_response_precision_norm": (
                        norm_delta / norm_reference if norm_reference > 0.0 else None
                    ),
                    "candidate_vs_primary": metric_from_vectors(
                        reference_white, reference_white + delta_white
                    ),
                }
    interpolation = manifests["theory"]["lambda_interpolation"]["residuals"]
    numerical_errors = {}
    for shape in SHAPES:
        prefix = shape.lower()
        total = np.asarray(theory[f"{prefix}_total_lambda_samples"])
        errors = np.asarray(theory[f"{prefix}_total_numerical_error"])
        numerical_errors[shape] = {
            "maximum_pointwise_reported_error_fraction": float(
                np.max(errors / np.maximum(np.abs(total), 1.0))
            ),
            "minimum_integrand_evaluations": int(np.min(theory[f"{prefix}_neval"])),
            "maximum_integrand_evaluations": int(np.max(theory[f"{prefix}_neval"])),
        }
    return {
        "primary_lambda_holdout_interpolation": interpolation,
        "reported_primary_integration_errors": numerical_errors,
        "configuration_rows": rows,
        "representative_geometry_only": True,
    }


def run_full(
    runtime: Runtime,
    gaussian: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """运行 EQ 与 OR-LSS 各自独立的完整 closure、Fisher 和 profile 分析。

    shape 循环之间没有共享数据向量或 Fisher block，所以不会产生双-shape
    联合限制。返回的 ``acceptance`` 也按 shape 单独冻结，失败后不允许拟合
    一个经验响应系数进行修补。
    """
    theory, manifests = load_theory()
    closure_rows: dict[str, Any] = {}
    increment_rows: dict[str, Any] = {}
    profile_rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for cut in CUTS:
        contexts = runtime.contexts[cut]
        for shape_index, shape in enumerate(SHAPES):
            for method_index, method in enumerate(METHODS):
                context = contexts[method]
                nuisance = fixed_nuisance(gaussian, method, cut)
                total = theory_component(
                    theory, shape, method, context, nuisance[0], "total"
                )
                tree = theory_component(
                    theory, shape, method, context, nuisance[0], "tree"
                )
                loop = theory_component(
                    theory, shape, method, context, nuisance[0], "loop"
                )
                samples = response_samples(runtime, shape, method, context)
                seed = 2026082100 + int(round(cut * 100)) * 100 + shape_index * 10 + method_index
                record, local_arrays = closure_one(
                    theory_vector=total, response_samples=samples,
                    context=context, nuisance=nuisance, seed=seed,
                )
                for component in ("B122I", "B122II", "B113I", "B113II"):
                    local_arrays[f"theory_component_{component}"] = theory_component(
                        theory, shape, method, context, nuisance[0], component
                    )
                local_arrays["theory_tree"] = tree
                local_arrays["theory_loop"] = loop
                tree_fisher, _tree_arrays = fisher_summary(
                    local_arrays["chol_fit"], tree,
                    local_arrays["bias_jacobian"],
                )
                loop_fisher, _loop_arrays = fisher_summary(
                    local_arrays["chol_fit"], loop,
                    local_arrays["bias_jacobian"],
                )
                record["tree_loop_ablation"] = {
                    "tree_only": tree_fisher,
                    "loop_only": loop_fisher,
                    "tree_plus_loop": record["fisher"],
                    "tree_fraction_of_raw_response_norm": float(
                        np.linalg.norm(whiten(local_arrays["chol_fit"], tree))
                        / np.linalg.norm(whiten(local_arrays["chol_fit"], total))
                    ),
                    "loop_fraction_of_raw_response_norm": float(
                        np.linalg.norm(whiten(local_arrays["chol_fit"], loop))
                        / np.linalg.norm(whiten(local_arrays["chol_fit"], total))
                    ),
                }
                key = f"{shape}__{method}__kmax{cut:.2f}"
                closure_rows[key] = {
                    "shape": shape, "method": method, "kmax_h_Mpc": cut,
                    "b1_fiducial": float(nuisance[0]), **record,
                }
                for name, value in local_arrays.items():
                    arrays[f"closure__{key}__{name}"] = np.asarray(value)
            joint_theta = fixed_joint_theta(gaussian, cut)
            b1_joint = float(joint_theta[0])
            pre_theory = theory_component(
                theory, shape, "pre", contexts["pre"], b1_joint, "total"
            )
            standard_theory = theory_component(
                theory, shape, "standard", contexts["standard"], b1_joint, "total"
            )
            pre_samples = response_samples(runtime, shape, "pre", contexts["pre"])
            standard_samples = response_samples(runtime, shape, "standard", contexts["standard"])
            increment, increment_arrays = increment_closure(
                theory_pre=pre_theory, theory_standard=standard_theory,
                samples_pre=pre_samples, samples_standard=standard_samples,
                pre=contexts["pre"], standard=contexts["standard"],
                joint_theta=joint_theta,
                seed=2026083100 + int(round(cut * 100)) * 10 + shape_index,
            )
            increment_key = f"{shape}__kmax{cut:.2f}"
            increment_rows[increment_key] = {
                "shape": shape, "kmax_h_Mpc": cut,
                "shared_b1": b1_joint, **increment,
            }
            for name, value in increment_arrays.items():
                arrays[f"increment__{increment_key}__{name}"] = np.asarray(value)

    for shape in SHAPES:
        for method in METHODS:
            key = f"{shape}__{method}__kmax0.10"
            context = runtime.contexts[0.10][method]
            nuisance = fixed_nuisance(gaussian, method, 0.10)
            sigma = closure_rows[key]["fisher"]["joint_six_parameter"]["sigma_fNL"]
            profile, profile_arrays = profile_fit(
                context=context, theory=theory, shape=shape,
                nuisance_start=nuisance, fisher_sigma=sigma,
            )
            profile_rows[key] = {"shape": shape, "method": method, **profile}
            for name, value in profile_arrays.items():
                arrays[f"profile__{key}__{name}"] = np.asarray(value)
            print(
                f"profile {shape}/{method}: fhat={profile['map']['fNL']:.6g}, "
                f"sigmaF={sigma}", flush=True,
            )

    numerical = numerical_convergence_audit(runtime, theory, gaussian, manifests)
    pre_post_gain: dict[str, Any] = {}
    joint_degradation: dict[str, Any] = {}
    for shape in SHAPES:
        pre_post_gain[shape] = {}
        joint_degradation[shape] = {}
        for cut in CUTS:
            tiers: dict[str, Any] = {}
            for tier in ("conditional", "joint_six_parameter"):
                pre_fisher = closure_rows[f"{shape}__pre__kmax{cut:.2f}"]["fisher"][tier]
                post_fisher = closure_rows[
                    f"{shape}__standard__kmax{cut:.2f}"
                ]["fisher"][tier]
                pre_information = pre_fisher["information"]
                post_information = post_fisher["information"]
                pre_sigma = pre_fisher["sigma_fNL"]
                post_sigma = post_fisher["sigma_fNL"]
                identifiable = bool(
                    pre_fisher["identifiable"] and post_fisher["identifiable"]
                    and pre_information is not None and post_information is not None
                    and pre_information > 0.0 and post_information > 0.0
                )
                tiers[tier] = {
                    "pre_sigma_fNL": pre_sigma,
                    "post_sigma_fNL": post_sigma,
                    "post_over_pre_information": (
                        post_information / pre_information if identifiable else None
                    ),
                    "post_information_gain_fraction": (
                        post_information / pre_information - 1.0 if identifiable else None
                    ),
                    "post_sigma_reduction_fraction": (
                        1.0 - post_sigma / pre_sigma if identifiable else None
                    ),
                    "identifiable_in_both_methods": identifiable,
                }
            pre_post_gain[shape][f"kmax{cut:.2f}"] = tiers
            joint_degradation[shape][f"kmax{cut:.2f}"] = {}
            for method in METHODS:
                fisher = closure_rows[
                    f"{shape}__{method}__kmax{cut:.2f}"
                ]["fisher"]
                conditional = fisher["conditional"]
                joint = fisher["joint_six_parameter"]
                identifiable = bool(
                    conditional["identifiable"] and joint["identifiable"]
                    and conditional["sigma_fNL"] is not None
                    and joint["sigma_fNL"] is not None
                )
                joint_degradation[shape][f"kmax{cut:.2f}"][method] = {
                    "conditional_sigma_fNL": conditional["sigma_fNL"],
                    "joint_sigma_fNL": joint["sigma_fNL"],
                    "joint_over_conditional_sigma": (
                        joint["sigma_fNL"] / conditional["sigma_fNL"]
                        if identifiable else None
                    ),
                    "joint_information_retention_fraction": (
                        joint["information"] / conditional["information"]
                        if identifiable else None
                    ),
                    "identifiable": identifiable,
                }
    acceptance = {}
    gaussian_gate = bool(gaussian["target_kmax0p10_gate"]["pass"])
    for shape in SHAPES:
        gates: dict[str, bool] = {
            "Gaussian_baseline_p_above_0p05": gaussian_gate,
        }
        for method in METHODS:
            row = closure_rows[f"{shape}__{method}__kmax0.10"]
            metric = row["metrics"]["joint_six_parameter"]
            bootstrap = row["bootstrap"]["tiers"]["joint_six_parameter"]["amplitude"]
            profile = profile_rows[f"{shape}__{method}__kmax0.10"]
            gates[f"{method}_projected_cosine_above_0p98"] = bool(
                metric["precision_weighted_cosine"] is not None
                and metric["precision_weighted_cosine"] > 0.98
            )
            gates[f"{method}_projected_amplitude_within_10_percent"] = bool(
                metric["simulation_on_theory_amplitude"] is not None
                and abs(metric["simulation_on_theory_amplitude"] - 1.0) <= 0.10
            )
            gates[f"{method}_bootstrap_95_contains_unity"] = bool(
                bootstrap["q2p5"] is not None
                and bootstrap["q2p5"] <= 1.0 <= bootstrap["q97p5"]
            )
            difference = row[
                "theory_vs_simulation_joint_six_parameter_sigma_fractional_difference"
            ]
            gates[f"{method}_theory_vs_sim_sigma_within_10_percent"] = bool(
                difference is not None and difference <= 0.10
            )
            pull = profile["absolute_fNL_hat_over_fisher_sigma"]
            gates[f"{method}_abs_fNLhat_below_0p5sigma"] = bool(
                pull is not None and pull < 0.5
            )
        increment = increment_rows[f"{shape}__kmax0.10"]["metrics"]["joint_six_parameter"]
        gates["increment_projected_amplitude_within_10_percent"] = bool(
            increment["simulation_on_theory_amplitude"] is not None
            and abs(increment["simulation_on_theory_amplitude"] - 1.0) <= 0.10
        )
        gates["increment_projected_cosine_above_0p95"] = bool(
            increment["precision_weighted_cosine"] is not None
            and increment["precision_weighted_cosine"] > 0.95
        )
        acceptance[shape] = {"gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"}

    payload = {
        "schema": "quijote-z0p5-nonlocal-marisa-b-results-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "scientific_scope": {
            "observable": "B000 only", "shapes": SHAPES,
            "shape_inference": "independent only; no combined-shape likelihood",
            "methods": METHODS, "cuts_h_Mpc": CUTS,
            "volume_Gpc3_h3": VOLUME,
            "linear_power_amplitude": "fixed to the fiducial cosmology",
            "standard_reconstruction": {"R_Mpc_h": 10.0, "b_rec_h": B_REC},
            "excluded": ("P(k)", "STO", "local PNG", "k>0.10", "EFT/new operators", "empirical gain"),
        },
        "parameter_tiers": {
            "conditional": ("fNL",),
            "joint_six_parameter_headline": ("fNL", *NUISANCE_NAMES),
            "joint_constraint_definition": (
                "one six-dimensional Fisher/profile fit per PNG shape and "
                "reconstruction; reported 1D errors and 2D contours come from "
                "that same joint covariance"
            ),
            "fixed_cosmology": True,
            "priors": "none; b1 has physical hard bounds only in profiles",
        },
        "Gaussian_baseline": gaussian,
        "closure": closure_rows,
        "increments": increment_rows,
        "profiles_kmax0p10": profile_rows,
        "pre_post_information_gain": pre_post_gain,
        "joint_constraint_degradation": joint_degradation,
        "numerical_convergence": numerical,
        "acceptance": acceptance,
        "empirical_response_rescaling_used": False,
        "inputs": {
            "frozen": {"path": FROZEN_INPUTS, "sha256": sha256(FROZEN_INPUTS)},
            "theory": {"path": THEORY, "sha256": sha256(THEORY)},
            "theory_manifest": {"path": THEORY_MANIFEST, "sha256": sha256(THEORY_MANIFEST)},
            "convergence": {"path": CONVERGENCE, "sha256": sha256(CONVERGENCE)},
            "Fid500": {"path": FID_MATRIX, "sha256": sha256(FID_MATRIX)},
            **{
                f"{shape}_centered100": {"path": path, "sha256": sha256(path)}
                for shape, path in SHAPE_MATRICES.items()
            },
        },
        "software": {
            "analysis_runner": {
                "path": Path(__file__).resolve(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "input_audit_runner": {
                "path": CODES / "audit_quijote_halo_z0p5_nonlocal_marisa_b_inputs.py",
                "sha256": sha256(
                    CODES / "audit_quijote_halo_z0p5_nonlocal_marisa_b_inputs.py"
                ),
            },
            "theory_runner": {
                "path": CODES / "run_quijote_halo_z0p5_nonlocal_marisa_b_theory.py",
                "sha256": sha256(
                    CODES / "run_quijote_halo_z0p5_nonlocal_marisa_b_theory.py"
                ),
            },
            "convergence_runner": {
                "path": CODES / "run_quijote_halo_z0p5_nonlocal_marisa_b_convergence.py",
                "sha256": sha256(
                    CODES / "run_quijote_halo_z0p5_nonlocal_marisa_b_convergence.py"
                ),
            },
        },
    }
    return payload, arrays


def write_csv(payload: dict[str, Any]) -> None:
    """把每个独立 shape/method/cut/tier 的结果写成扁平 CSV。"""
    fields = (
        "record", "shape", "method", "kmax_h_Mpc", "tier",
        "sigma_fNL_theory", "sigma_fNL_simulation", "amplitude", "cosine",
        "post_information_gain_fraction", "profile_fNL", "profile_pull_sigma",
        "status",
    )
    rows = []
    for key, record in payload["closure"].items():
        for tier, metrics in record["metrics"].items():
            profile = payload["profiles_kmax0p10"].get(key, {})
            rows.append({
                "record": "single_shape", "shape": record["shape"],
                "method": record["method"], "kmax_h_Mpc": record["kmax_h_Mpc"],
                "tier": tier, "sigma_fNL_theory": metrics["sigma_theory"],
                "sigma_fNL_simulation": metrics["sigma_simulation"],
                "amplitude": metrics["simulation_on_theory_amplitude"],
                "cosine": metrics["precision_weighted_cosine"],
                "post_information_gain_fraction": payload[
                    "pre_post_information_gain"
                ][record["shape"]][f"kmax{record['kmax_h_Mpc']:.2f}"][tier][
                    "post_information_gain_fraction"
                ],
                "profile_fNL": profile.get("map", {}).get("fNL"),
                "profile_pull_sigma": profile.get("absolute_fNL_hat_over_fisher_sigma"),
                "status": payload["acceptance"][record["shape"]]["status"] if record["kmax_h_Mpc"] == 0.10 else "diagnostic",
            })
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=CSV_OUTPUT.parent,
        prefix=f".{CSV_OUTPUT.name}.", suffix=".tmp", delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: jsonable(row.get(name)) for name in fields} for row in rows])
        temporary = Path(stream.name)
    temporary.replace(CSV_OUTPUT)


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not identifiable"
    return f"{float(value):.{digits}g}"


def write_report(payload: dict[str, Any]) -> None:
    """写出可直接阅读的 Markdown 摘要、headline 表和冻结门槛。"""
    lines = [
        "# Quijote halo EQ / OR-LSS nonlocal PNG: MARISA-B B000-only result",
        "",
        "## Frozen scope",
        "",
        "- z=0.5 real-space FoF halos, Mmin=1e13 Msun/h.",
        "- Pre reconstruction and ordinary standard reconstruction only; R=10 Mpc/h and operational b_rec=1.9326526728794324.",
        "- Native Quijote volume 1 (Gpc/h)^3, method-specific Fid500 covariance, B000 only, kmax<=0.10 h/Mpc.",
        "- No P(k), STO, local PNG, EFT/new halo operators, external nuisance priors, or empirical response rescaling.",
        "- Cosmology and the linear-power amplitude are fixed; no power-amplitude nuisance is introduced.",
        "- EQ and OR-LSS are constrained independently; there is no dual-shape likelihood or Fisher block.",
        "",
        "## Gaussian baseline",
        "",
    ]
    joint = payload["Gaussian_baseline"]["rows"]["joint__kmax0.10"]
    lines.extend([
        f"The shared-physics pre/standard fit has chi2/dof={joint['data_chi2']:.6g}/{joint['dof']} and p={joint['gof_p_value']:.6g}.",
        f"Its fixed-cosmology map is b1={joint['map']['b1']:.8g}, b2={joint['map']['b2']:.8g}, gamma2={joint['map']['gamma2']:.8g}.",
        f"The paired standard-minus-pre Gaussian increment has p={joint['paired_increment']['p_value_no_refit']:.6g}.",
        "",
        "## Headline one-shape constraints at kmax=0.10 h/Mpc",
        "",
        "| shape | method | conditional sigma | six-parameter joint sigma | sim/theory response amp | projected cosine | fhat/sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for shape in SHAPES:
        for method in METHODS:
            key = f"{shape}__{method}__kmax0.10"
            row = payload["closure"][key]
            profile = payload["profiles_kmax0p10"][key]
            lines.append(
                f"| {shape} | {METHOD_LABELS[method]} | "
                f"{format_number(row['fisher']['conditional']['sigma_fNL'])} | "
                f"{format_number(row['fisher']['joint_six_parameter']['sigma_fNL'])} | "
                f"{format_number(row['metrics']['joint_six_parameter']['simulation_on_theory_amplitude'])} | "
                f"{format_number(row['metrics']['joint_six_parameter']['precision_weighted_cosine'])} | "
                f"{format_number(profile['absolute_fNL_hat_over_fisher_sigma'])} |"
            )
    lines.extend([
        "",
        "## Six-parameter joint maps and Fisher errors at kmax=0.10 h/Mpc",
        "",
        "| shape | method | fNL | b1 | b2 | gamma2 | Ashot | Bshot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for shape in SHAPES:
        for method in METHODS:
            key = f"{shape}__{method}__kmax0.10"
            mapping = payload["profiles_kmax0p10"][key]["map"]
            errors = payload["closure"][key]["fisher"][
                "joint_six_parameter"
            ]["joint_parameter_sigma"]
            entries = [
                f"{format_number(mapping[name])} +/- {format_number(errors[name])}"
                for name in ("fNL", *NUISANCE_NAMES)
            ]
            lines.append(
                f"| {shape} | {METHOD_LABELS[method]} | "
                + " | ".join(entries)
                + " |"
            )
    lines.extend([
        "",
        "## Six-parameter joint degradation at kmax=0.10 h/Mpc",
        "",
        "| shape | method | joint/conditional sigma | retained conditional Fisher information |",
        "|---|---:|---:|---:|",
    ])
    for shape in SHAPES:
        for method in METHODS:
            degradation = payload["joint_constraint_degradation"][shape][
                "kmax0.10"
            ][method]
            retention = degradation["joint_information_retention_fraction"]
            lines.append(
                f"| {shape} | {METHOD_LABELS[method]} | "
                f"{format_number(degradation['joint_over_conditional_sigma'])} | "
                f"{'not identifiable' if retention is None else f'{100.0 * retention:.2f}%'} |"
            )
    lines.extend([
        "",
        "## Independent pre/post Fisher gain scan",
        "",
        "| shape | kmax | conditional pre sigma | conditional post sigma | conditional F gain | headline pre sigma | headline post sigma | headline F gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for shape in SHAPES:
        for cut in CUTS:
            gain = payload["pre_post_information_gain"][shape][f"kmax{cut:.2f}"]
            conditional = gain["conditional"]
            headline = gain["joint_six_parameter"]

            def gain_label(record: dict[str, Any]) -> str:
                value = record["post_information_gain_fraction"]
                return "not identifiable" if value is None else f"{100.0 * value:.1f}%"

            lines.append(
                f"| {shape} | {cut:.2f} | "
                f"{format_number(conditional['pre_sigma_fNL'])} | "
                f"{format_number(conditional['post_sigma_fNL'])} | "
                f"{gain_label(conditional)} | "
                f"{format_number(headline['pre_sigma_fNL'])} | "
                f"{format_number(headline['post_sigma_fNL'])} | "
                f"{gain_label(headline)} |"
            )
    lines.extend(["", "## Frozen acceptance gates", ""])
    for shape in SHAPES:
        acceptance = payload["acceptance"][shape]
        lines.append(f"### {shape}: {acceptance['status']}")
        lines.append("")
        for name, passed in acceptance["gates"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines.append("")
    lines.extend([
        "## Joint-contour convention",
        "",
        "The contour PDF uses the full six-parameter Fisher covariance. Each page holds one PNG shape and directly overlays pre- and post-reconstruction 68% and 95% projections, with each Gaussian centered on its own joint profile map; no EQ/OR-LSS data vectors are combined.",
        "",
        "## Interpretation policy",
        "",
        "A failed shape is reported as a failure of this fixed theory/measurement contract. No response amplitude is fitted or applied to repair it.",
        "The six-parameter joint tier is the headline realistic B-only result; its reported fNL sigma is the one-dimensional marginal of that joint fit, while conditional numbers are diagnostics only and cosmology is fixed.",
        "",
        "## Reproducibility",
        "",
        f"- Machine-readable result: `{OUTPUT}`",
        f"- Arrays: `{ARRAY_OUTPUT}`",
        f"- Table: `{CSV_OUTPUT}`",
        f"- Frozen input manifest: `{FROZEN_INPUTS}`",
        f"- Exact response manifest: `{THEORY_MANIFEST}`",
        f"- Numerical convergence report: `{CONVERGENCE_REPORT}`",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=REPORT.parent,
        prefix=f".{REPORT.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write("\n".join(lines) + "\n")
        temporary = Path(stream.name)
    temporary.replace(REPORT)


def make_plots(
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    gaussian_arrays: dict[str, np.ndarray],
) -> list[Path]:
    """生成 Gaussian、响应、loop、限制和退化诊断；只保留 PDF。"""
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    gaussian_pdf = FIGURES / "gaussian_baseline.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), sharex="col")
    for row, method in enumerate(METHODS):
        key = f"{method}__kmax0.10"
        target = gaussian_arrays[f"{key}__target"]
        model = gaussian_arrays[f"{key}__model"]
        covariance = gaussian_arrays[f"{key}__covariance"]
        sigma = np.sqrt(np.diag(covariance))
        x = np.arange(target.size)
        axes[row, 0].errorbar(x, target, yerr=sigma, fmt="o", ms=3, capsize=2, label="Fid500 mean")
        axes[row, 0].plot(x, model, "-", lw=1.4, label="MARISA-B")
        axes[row, 1].axhline(0.0, color="0.4", lw=0.8)
        axes[row, 1].plot(x, (target - model) / sigma, "o-", ms=3)
        axes[row, 0].set_ylabel(f"{METHOD_LABELS[method]}\n$B_{000}$")
        axes[row, 1].set_ylabel("residual / single-box sigma")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("triangle-bin index within cut")
    axes[1, 1].set_xlabel("triangle-bin index within cut")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(gaussian_pdf, metadata={**PDF_METADATA, "Title": "Gaussian baseline"})
    plt.close(fig)
    outputs.append(gaussian_pdf)

    for shape in SHAPES:
        pdf = FIGURES / f"{shape.lower()}_response_closure.pdf"
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
        for row, method in enumerate(METHODS):
            key = f"{shape}__{method}__kmax0.10"
            theory = arrays[f"closure__{key}__theory"]
            empirical = arrays[f"closure__{key}__empirical"]
            samples = arrays[f"closure__{key}__response_samples"]
            error = np.std(samples, axis=0, ddof=1) / math.sqrt(samples.shape[0])
            x = np.arange(theory.size)
            axes[row, 0].errorbar(x, empirical, yerr=error, fmt="o", ms=3, capsize=2, label="centered simulations")
            axes[row, 0].plot(x, theory, "-", lw=1.5, label="MARISA-B tree+1-loop")
            axes[row, 1].axhline(0.0, color="0.4", lw=0.8)
            axes[row, 1].errorbar(x, empirical - theory, yerr=error, fmt="o", ms=3, capsize=2)
            axes[row, 0].set_ylabel(
                f"{METHOD_LABELS[method]}\n$\\partial B_{000}/\\partial f_{{\\rm NL}}$"
            )
            axes[row, 1].set_ylabel("simulation - theory")
        axes[0, 0].legend(frameon=False, fontsize=8)
        for axis in axes[1]:
            axis.set_xlabel("triangle-bin index within cut")
        for axis in axes.flat:
            axis.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(pdf, metadata={**PDF_METADATA, "Title": f"{shape} response closure"})
        plt.close(fig)
        outputs.append(pdf)

    decomposition_pdf = FIGURES / "tree_loop_decomposition.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    for row, shape in enumerate(SHAPES):
        for column, method in enumerate(METHODS):
            key = f"{shape}__{method}__kmax0.10"
            tree = arrays[f"closure__{key}__theory_tree"]
            loop = arrays[f"closure__{key}__theory_loop"]
            x = np.arange(tree.size)
            axes[row, column].plot(x, tree, "o-", ms=3, label="B111")
            axes[row, column].plot(x, loop, "s-", ms=3, label="four 1-loop topologies")
            axes[row, column].plot(x, tree + loop, "k-", lw=1.5, label="total")
            axes[row, column].set_ylabel(
                f"{shape} {METHOD_LABELS[method]}\nresponse"
            )
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("triangle-bin index within cut")
    fig.tight_layout()
    fig.savefig(decomposition_pdf, metadata={**PDF_METADATA, "Title": "Tree and loop response decomposition"})
    plt.close(fig)
    outputs.append(decomposition_pdf)

    constraints_pdf = FIGURES / "constraints.pdf"
    with PdfPages(constraints_pdf, metadata={**PDF_METADATA, "Title": "Nonlocal PNG constraints"}) as pages:
        for shape in SHAPES:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
            for axis, method in zip(axes, METHODS):
                for tier, label, marker in (
                    ("conditional", "conditional", "o"),
                    ("joint_six_parameter", "six-parameter joint (headline)", "s"),
                ):
                    values = [
                        payload["closure"][f"{shape}__{method}__kmax{cut:.2f}"]["fisher"][tier]["sigma_fNL"]
                        for cut in CUTS
                    ]
                    plotted = np.asarray([np.nan if value is None else value for value in values])
                    axis.plot(CUTS, plotted, marker=marker, label=label)
                axis.set_yscale("log")
                axis.set_xlabel(r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$")
                axis.set_ylabel(r"$\sigma(f_{\rm NL})$")
                axis.grid(alpha=0.18)
                axis.text(
                    0.04, 0.94, METHOD_LABELS[method],
                    transform=axis.transAxes, va="top",
                )
            axes[0].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            pages.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
        for axis, shape in zip(axes, SHAPES):
            for tier, label, marker in (
                ("conditional", "conditional", "o"),
                ("joint_six_parameter", "six-parameter joint (headline)", "s"),
            ):
                values = [
                    payload["pre_post_information_gain"][shape][
                        f"kmax{cut:.2f}"
                    ][tier]["post_information_gain_fraction"]
                    for cut in CUTS
                ]
                axis.plot(
                    CUTS,
                    [np.nan if value is None else 100.0 * value for value in values],
                    marker=marker,
                    label=label,
                )
            axis.axhline(0.0, color="0.4", lw=0.8)
            axis.set_xlabel(r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$")
            axis.set_ylabel("post vs pre Fisher gain [%]")
            axis.text(0.04, 0.94, shape, transform=axis.transAxes, va="top")
            axis.grid(alpha=0.18)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        pages.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
        for row, shape in enumerate(SHAPES):
            for column, method in enumerate(METHODS):
                key = f"{shape}__{method}__kmax0.10"
                f = arrays[f"profile__{key}__profile_fNL"]
                q = arrays[f"profile__{key}__profile_chi2"]
                axes[row, column].plot(f, q - np.min(q), "o-", ms=3)
                axes[row, column].axhline(1.0, color="0.4", ls="--", lw=0.8)
                axes[row, column].set_xlabel(r"$f_{\rm NL}$")
                axes[row, column].set_ylabel(r"$\Delta\chi^2_{\rm prof}$")
                axes[row, column].text(
                    0.04, 0.94, f"{shape} {METHOD_LABELS[method]}",
                    transform=axes[row, column].transAxes, va="top",
                )
                axes[row, column].grid(alpha=0.18)
        fig.tight_layout()
        pages.savefig(fig)
        plt.close(fig)
    outputs.append(constraints_pdf)

    degeneracy_pdf = FIGURES / "parameter_degeneracies.pdf"
    labels = list(NUISANCE_NAMES)
    values = np.asarray([
        [
            payload["closure"][f"{shape}__{method}__kmax0.10"]["fisher"]["joint_six_parameter"]["individual_nuisance_cosines"].get(name, np.nan)
            for name in labels
        ]
        for shape in SHAPES for method in METHODS
    ])
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    image = axis.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(
        np.arange(4),
        [f"{shape} {METHOD_LABELS[method]}" for shape in SHAPES for method in METHODS],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="precision-weighted response–nuisance cosine")
    fig.tight_layout()
    fig.savefig(degeneracy_pdf, metadata={**PDF_METADATA, "Title": "Parameter degeneracies"})
    plt.close(fig)
    outputs.append(degeneracy_pdf)

    contour_pdf = FIGURES / "joint_six_parameter_contours.pdf"
    parameter_names = ("fNL", *NUISANCE_NAMES)
    parameter_labels = (
        r"$f_{\rm NL}$", r"$b_1$", r"$b_2$", r"$\gamma_2$",
        r"$A_{\rm shot}$", r"$B_{\rm shot}$",
    )
    contour_colors = {"pre": "#4477AA", "standard": "#CC6677"}
    with PdfPages(
        contour_pdf,
        metadata={**PDF_METADATA, "Title": "Independent-shape six-parameter contours"},
    ) as pages:
        # 每页只对应一个 shape，并在完全相同的坐标范围内直接叠加 pre/post；
        # 六维 Fisher 同时包含 fNL 与五个 halo/stochastic 参数，shape 之间不联合。
        for shape in SHAPES:
            method_data = {}
            for method in METHODS:
                key = f"{shape}__{method}__kmax0.10"
                covariance = arrays[
                    f"closure__{key}__fisher__joint_covariance"
                ]
                profile_map = payload["profiles_kmax0p10"][key]["map"]
                centers = np.asarray(
                    [profile_map[name] for name in parameter_names],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(covariance)):
                    raise ValueError(f"rank-deficient headline contour for {key}")
                method_data[method] = {
                    "covariance": covariance,
                    "centers": centers,
                    "sigmas": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
                }

            figure, axes = plt.subplots(
                len(parameter_names), len(parameter_names),
                figsize=(12.0, 12.0), squeeze=False,
            )
            for row in range(len(parameter_names)):
                for column in range(len(parameter_names)):
                    axis = axes[row, column]
                    if column > row:
                        axis.axis("off")
                        continue
                    x_limits = (
                        min(
                            method_data[method]["centers"][column]
                            - 3.2 * method_data[method]["sigmas"][column]
                            for method in METHODS
                        ),
                        max(
                            method_data[method]["centers"][column]
                            + 3.2 * method_data[method]["sigmas"][column]
                            for method in METHODS
                        ),
                    )
                    axis.set_xlim(*x_limits)
                    axis.xaxis.set_major_locator(MaxNLocator(3))
                    axis.tick_params(labelsize=6, pad=1)
                    if row == column:
                        grid = np.linspace(*x_limits, 300)
                        for method in METHODS:
                            center = method_data[method]["centers"][column]
                            sigma = method_data[method]["sigmas"][column]
                            density = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
                            axis.plot(
                                grid, density, color=contour_colors[method],
                                lw=1.35, label=METHOD_LABELS[method],
                            )
                            axis.axvline(
                                center, color=contour_colors[method],
                                lw=0.65, alpha=0.8,
                            )
                        axis.set_ylim(0.0, 1.08)
                        axis.set_yticks([])
                    else:
                        y_limits = (
                            min(
                                method_data[method]["centers"][row]
                                - 3.2 * method_data[method]["sigmas"][row]
                                for method in METHODS
                            ),
                            max(
                                method_data[method]["centers"][row]
                                + 3.2 * method_data[method]["sigmas"][row]
                                for method in METHODS
                            ),
                        )
                        axis.set_ylim(*y_limits)
                        axis.yaxis.set_major_locator(MaxNLocator(3))
                        for method in METHODS:
                            covariance = method_data[method]["covariance"]
                            centers = method_data[method]["centers"]
                            color = contour_colors[method]
                            pair = covariance[np.ix_((column, row), (column, row))]
                            eigenvalues, eigenvectors = np.linalg.eigh(pair)
                            order = np.argsort(eigenvalues)[::-1]
                            eigenvalues = np.maximum(eigenvalues[order], 0.0)
                            eigenvectors = eigenvectors[:, order]
                            angle = math.degrees(math.atan2(
                                eigenvectors[1, 0], eigenvectors[0, 0]
                            ))
                            for delta, alpha, linewidth in (
                                (5.991, 0.07, 0.9), (2.30, 0.20, 1.2),
                            ):
                                axis.add_patch(Ellipse(
                                    (centers[column], centers[row]),
                                    width=2.0 * math.sqrt(delta * eigenvalues[0]),
                                    height=2.0 * math.sqrt(delta * eigenvalues[1]),
                                    angle=angle, facecolor=color, edgecolor=color,
                                    alpha=alpha, linewidth=linewidth,
                                ))
                            axis.plot(
                                centers[column], centers[row], ".",
                                color=color, ms=2.5,
                            )
                    if row == len(parameter_names) - 1:
                        axis.set_xlabel(parameter_labels[column], fontsize=8)
                    else:
                        axis.set_xticklabels([])
                    if column == 0 and row > 0:
                        axis.set_ylabel(parameter_labels[row], fontsize=8)
                    elif row > column:
                        axis.set_yticklabels([])
            axes[1, 0].legend(
                handles=(
                    Patch(
                        facecolor=contour_colors["pre"],
                        edgecolor=contour_colors["pre"], alpha=0.20,
                        label="pre recon",
                    ),
                    Patch(
                        facecolor=contour_colors["standard"],
                        edgecolor=contour_colors["standard"], alpha=0.20,
                        label="post recon",
                    ),
                    Patch(facecolor="0.45", edgecolor="0.45", alpha=0.20, label="68%"),
                    Patch(facecolor="0.45", edgecolor="0.45", alpha=0.07, label="95%"),
                ),
                frameon=False, fontsize=7, loc="upper right",
            )
            figure.text(
                0.985, 0.985,
                f"{shape}, pre/post six-parameter Fisher, $k_{{\\max}}=0.10$",
                ha="right", va="top", fontsize=10,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=0.15, w_pad=0.15)
            pages.savefig(figure)
            plt.close(figure)
    outputs.append(contour_pdf)

    acceptance_pdf = FIGURES / "acceptance_summary.pdf"
    with PdfPages(
        acceptance_pdf,
        metadata={**PDF_METADATA, "Title": "Independent-shape constraints and acceptance"},
    ) as pages:
        # 第一页只列最终的四个独立 headline；不构造 EQ+OR-LSS 联合行。
        constraint_rows = []
        for shape in SHAPES:
            for method in METHODS:
                key = f"{shape}__{method}__kmax0.10"
                closure = payload["closure"][key]
                profile = payload["profiles_kmax0p10"][key]
                constraint_rows.append([
                    shape,
                    METHOD_LABELS[method],
                    format_number(closure["fisher"]["conditional"]["sigma_fNL"]),
                    format_number(
                        closure["fisher"]["joint_six_parameter"]["sigma_fNL"]
                    ),
                    format_number(profile["map"]["fNL"]),
                    payload["acceptance"][shape]["status"],
                ])
        fig, axis = plt.subplots(figsize=(10.5, 3.1))
        axis.axis("off")
        table = axis.table(
            cellText=constraint_rows,
            colLabels=(
                "shape", "reconstruction", "conditional sigma(fNL)",
                "headline sigma(fNL)", "profile fNL hat", "shape status",
            ),
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.55)
        fig.tight_layout()
        pages.savefig(fig)
        plt.close(fig)

        # 第二页把所有冻结 gate 原样列出，避免只报一个总 PASS/FAIL。
        gate_names = tuple(payload["acceptance"][SHAPES[0]]["gates"])
        gate_rows = [
            [name, *[
                "PASS" if payload["acceptance"][shape]["gates"][name] else "FAIL"
                for shape in SHAPES
            ]]
            for name in gate_names
        ]
        height = max(3.2, 0.36 * (len(gate_rows) + 2))
        fig, axis = plt.subplots(figsize=(11.0, height))
        axis.axis("off")
        table = axis.table(
            cellText=gate_rows,
            colLabels=("frozen gate", *SHAPES),
            colWidths=(0.72, 0.14, 0.14),
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.35)
        fig.tight_layout()
        pages.savefig(fig)
        plt.close(fig)
    outputs.append(acceptance_pdf)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gaussian", "full"), default="full")
    args = parser.parse_args()
    runtime = load_runtime()
    gaussian, gaussian_arrays = run_gaussian(runtime)
    if args.stage == "full":
        if not THEORY.is_file() or not CONVERGENCE.is_file():
            raise FileNotFoundError("full stage requires completed theory and convergence products")
        payload, arrays = run_full(runtime, gaussian)
        atomic_npz(ARRAY_OUTPUT, arrays)
        write_csv(payload)
        write_report(payload)
        figures = make_plots(payload, arrays, gaussian_arrays)
        payload["outputs"] = {
            "arrays": {"path": ARRAY_OUTPUT, "sha256": sha256(ARRAY_OUTPUT)},
            "csv": {"path": CSV_OUTPUT, "sha256": sha256(CSV_OUTPUT)},
            "report": {"path": REPORT, "sha256": sha256(REPORT)},
            "figures_pdf_only": [
                {"path": path, "sha256": sha256(path)} for path in figures
            ],
        }
        atomic_json(OUTPUT, payload)
        print(json.dumps(jsonable({
            "status": "complete", "output": OUTPUT,
            "acceptance": payload["acceptance"],
        }), sort_keys=True))
        return
    print(json.dumps(jsonable({"status": "complete", "output": GAUSSIAN_OUTPUT}), sort_keys=True))


if __name__ == "__main__":
    main()
