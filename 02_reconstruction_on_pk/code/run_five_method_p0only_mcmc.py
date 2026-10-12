#!/usr/bin/env python3
"""运行五方法、两个输入的 matched100 P0-only 三参数 MCMC（STO-only 扩展）。

代码大纲
========
1. 读取五方法统一 P0 matrix、finite local-PNG lattice theory、冻结四方法
   预注册与 STO-only 修正案。
2. 本轮只对新增的 sto_only 方法构造 Fid500-fNL0 covariance likelihood 并做
   MAP/HESSE/MINOS profile；四方法 v1 的链与结果冻结不动。
3. 在 Hessian-whitened 坐标中运行四个独立 emcee ensemble，自动延长至收敛。
4. 汇总时把冻结 v1 的四方法 posterior（含 sha256 链路校验与分位数 parity）
   与新 sto_only posterior 合并，输出五方法 JSON/CSV/NPZ、中文报告和仅
   PDF 的五方法比较图。
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
import shutil
import tempfile
import time
from typing import Any
import warnings

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import emcee
import h5py
from iminuit import Minuit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import chi2, norm, rankdata

import p0only_arviz_compat as az


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
TAG = "quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1"
ANALYSIS = ROOT / "02_reconstruction_on_pk/work" / TAG
FIGURES = ANALYSIS / "figures"
PART_FIGURES = ROOT / "02_reconstruction_on_pk/figures"
MATRIX = ROOT / "02_reconstruction_on_pk/inputs/five_method_p0_matrix.npz"
MATRIX_AUDIT = ROOT / "02_reconstruction_on_pk/results/five_method_p0_matrix_audit.json"
PREREGISTRATION = ROOT / "02_reconstruction_on_pk/inputs/p0only_preregistration.md"
AMENDMENT = ROOT / "02_reconstruction_on_pk/inputs/p0only_sto_only_amendment.md"
P_THEORY = (
    ROOT / "02_reconstruction_on_pk/inputs"
    / "p0_finite_local_png_lattice.npz"
)
P_THEORY_MANIFEST = P_THEORY.with_suffix(".json")
OLD_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
V1_TAG = "quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1"
V1_POSTERIORS = OLD_ROOT / "analysis" / V1_TAG / "P0ONLY_MCMC_POSTERIORS.npz"
V1_RESULTS = ROOT / "02_reconstruction_on_pk/results/four_method_p0only_mcmc_results.json"
PROFILE_JSON = ANALYSIS / "P0ONLY_PROFILES.json"
PROFILE_NPZ = ANALYSIS / "P0ONLY_PROFILE_ARRAYS.npz"
RESULT_JSON = ANALYSIS / "P0ONLY_MCMC_RESULTS.json"
RESULT_CSV = ANALYSIS / "P0ONLY_MCMC_RESULTS.csv"
RESULT_NPZ = ANALYSIS / "P0ONLY_MCMC_POSTERIORS.npz"
REPORT = ANALYSIS / "P0ONLY_MCMC_REPORT.md"
CHAINS = ANALYSIS / "chains"
TASKS = ANALYSIS / "tasks"
POSTERIORS = ANALYSIS / "posteriors"
MARGINAL_PDF = FIGURES / "five_method_p0only_mcmc_contours.pdf"
RECOVERY_PDF = FIGURES / "five_method_p0only_recovery_comparison.pdf"

INHERITED_METHODS = ("pre", "standard", "marisa", "sto_marisa")
METHODS = ("pre", "standard", "marisa", "sto_only", "sto_marisa")
SAMPLE_METHODS = ("sto_only",)
TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
PARAMETERS = ("fNL", "b1", "Pshot_residual")
METHOD_LABELS = {
    "pre": "pre", "standard": "standard post",
    "marisa": "Marisa post", "sto_only": "STO only", "sto_marisa": "STO+Marisa",
}
COLORS = {
    "pre": "#4E79A7", "standard": "#E15759",
    "marisa": "#F28E2B", "sto_only": "#B07AA1", "sto_marisa": "#59A14F",
}
FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 6.0)
PSHOT_BOUNDS = (-1.0e5, 1.0e5)
VOLUME = 42.875
HARTLAP = (500.0 - 12.0 - 2.0) / 499.0
DELTA_C, P_HALO = 1.686, 1.12
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 40000
SEED = 20260916
BINS = 58
SMOOTHING = 1.05


def utc_now() -> str:
    """返回 UTC ISO 时间。"""

    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """返回文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """返回文件 provenance。"""

    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def jsonable(value: Any) -> Any:
    """递归转换 numpy/Path 为 JSON 类型。"""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    """原子写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_text(path: Path, content: str) -> None:
    """原子写文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """原子写压缩 NPZ。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 NPZ。"""

    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def key(method: str, target: str) -> str:
    """构造稳定 case key。"""

    return f"{method}__{target}"


def bphi(b1: np.ndarray | float) -> np.ndarray | float:
    """动态 physical local-PNG tracer bias。"""

    return 2.0 * DELTA_C * (b1 - P_HALO)


@dataclass
class Case:
    """冻结的三参数 P0 likelihood。"""

    method: str
    target_name: str
    truth: float
    target: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    theory: dict[str, np.ndarray]

    def predictions(self, values: np.ndarray) -> np.ndarray:
        """矢量化计算一个或多个参数点的 finite-PNG P0。"""

        physical = np.asarray(values, dtype=float)
        scalar = physical.ndim == 1
        rows = physical[None, :] if scalar else physical
        f = rows[:, 0]
        b1 = rows[:, 1]
        shot = rows[:, 2]
        bp = np.asarray(bphi(b1))
        prediction = (
            b1[:, None] ** 2 * self.theory["PL_lattice"][None, :]
            + 2.0 * f[:, None] * (b1 * bp)[:, None] * self.theory["PL_over_M_lattice"][None, :]
            + f[:, None] ** 2 * bp[:, None] ** 2 * self.theory["Pphi_lattice"][None, :]
            + shot[:, None]
        )
        return prediction[0] if scalar else prediction

    def objective_many(self, values: np.ndarray) -> np.ndarray:
        """返回多个参数点的 chi2。"""

        predictions = self.predictions(values)
        rows = predictions[None, :] if predictions.ndim == 1 else predictions
        residual = rows - self.target[None, :]
        white = np.linalg.solve(self.cholesky, residual.T).T
        result = np.sum(white**2, axis=1)
        return result

    def objective(self, values: np.ndarray) -> float:
        """返回一个参数点的 chi2。"""

        return float(self.objective_many(np.asarray(values, dtype=float))[0])


def load_case(method: str, target_name: str) -> Case:
    """从统一 matrix 构造一种方法/target 的 likelihood。"""

    if method not in METHODS or target_name not in TARGETS:
        raise ValueError((method, target_name))
    audit = json.loads(MATRIX_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or sha256(MATRIX) != audit["output"]["sha256"]:
        raise RuntimeError("five-method matrix audit/hash failed")
    arrays = load_npz(MATRIX)
    theory = load_npz(P_THEORY)
    target = np.asarray(arrays[f"{method}__target_{target_name}_matched100"], dtype=float)
    covariance_one = np.asarray(arrays[f"{method}__covariance_fNL0_fid500_one_box"], dtype=float)
    covariance = covariance_one / (VOLUME * HARTLAP)
    return Case(
        method, target_name, TRUTHS[target_name], target, covariance,
        np.linalg.cholesky(covariance), theory,
    )


def generalized_start(case: Case) -> tuple[float, float]:
    """在 fNL=0 线性解 b1^2 与常数 shot，生成稳定起点。"""

    covariance_one = case.covariance * VOLUME * HARTLAP
    design = np.column_stack((case.theory["PL_lattice"], np.ones(12)))
    solved_design = np.linalg.solve(covariance_one, design)
    solved_target = np.linalg.solve(covariance_one, case.target)
    coefficient = np.linalg.solve(design.T @ solved_design, design.T @ solved_target)
    b1_start = float(np.clip(math.sqrt(max(float(coefficient[0]), 0.01)), 0.101, 5.999))
    shot_start = float(np.clip(coefficient[1], PSHOT_BOUNDS[0] + 1, PSHOT_BOUNDS[1] - 1))
    return b1_start, shot_start


def configure_minuit(fit: Minuit) -> None:
    """设置共同的 Minuit bounds/steps。"""

    fit.errordef = Minuit.LEAST_SQUARES
    fit.limits["fNL"] = FNL_BOUNDS
    fit.limits["b1"] = B1_BOUNDS
    fit.limits["Pshot_residual"] = PSHOT_BOUNDS
    fit.errors["fNL"] = 8.0
    fit.errors["b1"] = 0.02
    fit.errors["Pshot_residual"] = 100.0
    fit.strategy, fit.tol = 1, 1e-5


def profile_case(case: Case) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """完成 MAP、HESSE、MINOS 和 truth-fixed profile。"""

    b1_start, shot_start = generalized_start(case)

    def objective(fNL: float, b1: float, Pshot_residual: float) -> float:
        return case.objective(np.asarray((fNL, b1, Pshot_residual)))

    fits = []
    f_starts = (case.truth, 0.0, 100.0, -100.0, 200.0)
    for f_start in f_starts:
        fit = Minuit(objective, fNL=f_start, b1=b1_start, Pshot_residual=shot_start)
        configure_minuit(fit)
        fit.migrad(ncall=120000, iterate=8, use_simplex=True)
        fit.hesse()
        fits.append(fit)
    valid = [fit for fit in fits if fit.valid and math.isfinite(float(fit.fval))]
    best = min(valid or fits, key=lambda fit: float(fit.fval))
    if not best.valid or best.covariance is None:
        raise RuntimeError(f"{case.method}/{case.target_name} MAP/HESSE failed")
    best.minos("fNL", ncall=160000)
    error = best.merrors["fNL"]
    mapping = {name: float(best.values[name]) for name in PARAMETERS}
    prediction = case.predictions(np.asarray([mapping[name] for name in PARAMETERS]))

    def truth_objective(b1: float, Pshot_residual: float) -> float:
        return case.objective(np.asarray((case.truth, b1, Pshot_residual)))

    truth_fit = Minuit(
        truth_objective, b1=mapping["b1"], Pshot_residual=mapping["Pshot_residual"]
    )
    truth_fit.errordef = Minuit.LEAST_SQUARES
    truth_fit.limits["b1"] = B1_BOUNDS
    truth_fit.limits["Pshot_residual"] = PSHOT_BOUNDS
    truth_fit.errors["b1"] = 0.02
    truth_fit.errors["Pshot_residual"] = 100.0
    truth_fit.strategy, truth_fit.tol = 1, 1e-5
    truth_fit.migrad(ncall=120000, iterate=8, use_simplex=True)
    truth_fit.hesse()
    if not truth_fit.valid:
        raise RuntimeError(f"{case.method}/{case.target_name} truth profile failed")
    truth_mapping = {
        "fNL": case.truth,
        "b1": float(truth_fit.values["b1"]),
        "Pshot_residual": float(truth_fit.values["Pshot_residual"]),
    }
    truth_objective_value = float(truth_fit.fval)
    eigenvalues = np.linalg.eigvalsh(case.covariance)
    covariance_hesse = np.asarray(best.covariance, dtype=float)
    row = {
        "method": case.method, "target": case.target_name, "truth_fNL": case.truth,
        "map": mapping, "objective": float(best.fval),
        "data_chi2": float(best.fval), "dof": 9,
        "goodness_of_fit_p": float(chi2.sf(float(best.fval), 9)),
        "optimizer_valid": bool(best.valid),
        "hesse_covariance": covariance_hesse,
        "minos_fNL": {
            "lower": mapping["fNL"] + float(error.lower),
            "upper": mapping["fNL"] + float(error.upper),
            "lower_error": float(error.lower), "upper_error": float(error.upper),
            "valid": bool(error.is_valid),
        },
        "truth_profile": {
            "mapping": truth_mapping, "objective": truth_objective_value,
            "delta_chi2": truth_objective_value - float(best.fval),
            "inside_delta_chi2_1": bool(truth_objective_value - float(best.fval) <= 1.0),
            "inside_delta_chi2_4": bool(truth_objective_value - float(best.fval) <= 4.0),
        },
        "covariance": {
            "dimension": 12, "hartlap": HARTLAP, "volume_ratio": VOLUME,
            "minimum_eigenvalue": float(eigenvalues[0]),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "source": "method-specific fNL0 Fid500",
        },
    }
    return row, {
        "target": case.target, "covariance": case.covariance,
        "prediction": prediction,
        "truth_prediction": case.predictions(np.asarray([truth_mapping[name] for name in PARAMETERS])),
    }


def prepare_profiles(methods: tuple[str, ...]) -> None:
    """运行并冻结指定方法的 profiles（本轮默认只有 sto_only）。"""

    for path in (
        MATRIX, MATRIX_AUDIT, PREREGISTRATION, AMENDMENT, P_THEORY, P_THEORY_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for target_name in TARGETS:
        for method in methods:
            case = load_case(method, target_name)
            row, local = profile_case(case)
            name = key(method, target_name)
            rows[name] = row
            for item, value in local.items():
                arrays[f"{name}__{item}"] = value
            print(
                f"profile {name}: fNL={row['map']['fNL']:+.4f} "
                f"{row['minos_fNL']['lower_error']:+.3f}/{row['minos_fNL']['upper_error']:+.3f} "
                f"b1={row['map']['b1']:.5f} p={row['goodness_of_fit_p']:.4g}",
                flush=True,
            )
    atomic_npz(PROFILE_NPZ, arrays)
    payload = {
        "schema": "quijote-z0p5-five-method-p0only-profiles-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "methods_all": METHODS, "methods_run": list(methods), "targets": TARGETS,
            "truths": TRUTHS, "parameters": PARAMETERS, "P_bins": 12, "P_kmax_label": 0.08,
            "covariance": "method-specific fNL0 Fid500 / volume / Hartlap; no target-count divisor",
            "inherited_methods_note": (
                "pre/standard/marisa/sto_marisa 的 profile 与链冻结于 v1，不在本轮重跑"
            ),
        },
        "rows": rows,
        "inputs": {
            "matrix": artifact(MATRIX), "matrix_audit": artifact(MATRIX_AUDIT),
            "preregistration": artifact(PREREGISTRATION), "amendment": artifact(AMENDMENT),
            "theory": artifact(P_THEORY),
            "theory_manifest": artifact(P_THEORY_MANIFEST), "runner": artifact(Path(__file__)),
        },
        "arrays": artifact(PROFILE_NPZ),
    }
    atomic_json(PROFILE_JSON, payload)


def load_profiles() -> dict[str, Any]:
    """读取 profile 并检查冻结输入/runner 哈希。"""

    payload = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or sha256(PROFILE_NPZ) != payload["arrays"]["sha256"]:
        raise RuntimeError("profile status/hash failed")
    for name, path in (
        ("matrix", MATRIX), ("matrix_audit", MATRIX_AUDIT),
        ("preregistration", PREREGISTRATION), ("amendment", AMENDMENT),
        ("theory", P_THEORY),
        ("theory_manifest", P_THEORY_MANIFEST), ("runner", Path(__file__)),
    ):
        if sha256(path) != payload["inputs"][name]["sha256"]:
            raise RuntimeError(f"profile input drift: {name}")
    return payload


@dataclass
class SamplerContext:
    """Hessian-whitened MCMC 坐标与 frozen case。"""

    case: Case
    profile_row: dict[str, Any]
    centre: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray

    def to_physical(self, coordinates: np.ndarray) -> np.ndarray:
        """whitened -> physical。"""

        values = np.asarray(coordinates, dtype=float)
        return self.centre + values @ self.transform.T

    def to_coordinates(self, physical: np.ndarray) -> np.ndarray:
        """physical -> whitened。"""

        values = np.asarray(physical, dtype=float)
        return (values - self.centre) @ self.inverse.T

    def log_probability(self, coordinates: np.ndarray) -> float | np.ndarray:
        """矢量化计算有界平坦先验下的 log posterior。"""

        values = np.asarray(coordinates, dtype=float)
        scalar = values.ndim == 1
        rows = values[None, :] if scalar else values
        physical = self.to_physical(rows)
        support = (
            (physical[:, 0] > FNL_BOUNDS[0]) & (physical[:, 0] < FNL_BOUNDS[1])
            & (physical[:, 1] > B1_BOUNDS[0]) & (physical[:, 1] < B1_BOUNDS[1])
            & (physical[:, 2] > PSHOT_BOUNDS[0]) & (physical[:, 2] < PSHOT_BOUNDS[1])
            & np.all(np.isfinite(physical), axis=1)
        )
        output = np.full(rows.shape[0], -math.inf)
        if np.any(support):
            output[support] = -0.5 * self.case.objective_many(physical[support])
        return float(output[0]) if scalar else output


def build_context(method: str, target_name: str, profiles: dict[str, Any]) -> SamplerContext:
    """用 profile HESSE covariance 构造 sampler context。"""

    case = load_case(method, target_name)
    row = profiles["rows"][key(method, target_name)]
    centre = np.asarray([row["map"][name] for name in PARAMETERS], dtype=float)
    covariance = np.asarray(row["hesse_covariance"], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0:
        floor = max(float(eigenvalues[-1]) * 1e-10, 1e-12)
        covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    transform = np.linalg.cholesky(covariance)
    inverse = np.linalg.inv(transform)
    context = SamplerContext(case, row, centre, transform, inverse)
    parity = abs(-2.0 * float(context.log_probability(np.zeros(3))) - float(row["objective"]))
    if parity > 1e-8:
        raise RuntimeError(f"profile/logp parity failed: {method}/{target_name}: {parity}")
    return context


_ACTIVE: SamplerContext | None = None


def active_log_probability(values: np.ndarray) -> float | np.ndarray:
    """emcee vectorized callback。"""

    if _ACTIVE is None:
        raise RuntimeError("no active sampler context")
    return _ACTIVE.log_probability(values)


def chain_path(method: str, target_name: str) -> Path:
    return CHAINS / f"{key(method, target_name)}.h5"


def contract_path(method: str, target_name: str) -> Path:
    return CHAINS / f"{key(method, target_name)}.contract.json"


def task_path(method: str, target_name: str) -> Path:
    return TASKS / f"{key(method, target_name)}.json"


def posterior_path(method: str, target_name: str) -> Path:
    return POSTERIORS / f"{key(method, target_name)}.npz"


def group_exists(path: Path, group: str) -> bool:
    """检查 HDF backend group。"""

    if not path.is_file():
        return False
    with h5py.File(path, "r") as stream:
        return group in stream


def write_or_validate_contract(path: Path, payload: dict[str, Any]) -> None:
    """首次写 chain contract；恢复时要求逐字段一致。"""

    comparable = jsonable(payload)
    comparable.pop("created_utc", None)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("created_utc", None)
        if existing != comparable:
            raise RuntimeError(f"checkpoint contract drift: {path}")
        return
    atomic_json(path, payload)


def initial_positions(context: SamplerContext, rng: np.random.Generator) -> np.ndarray:
    """生成 full-rank 且位于 prior 支持内的 walkers。"""

    accepted: list[np.ndarray] = []
    for _ in range(200000):
        scale = math.exp(rng.uniform(math.log(0.08), math.log(1.2)))
        candidate = rng.normal(0.0, scale, 3)
        if math.isfinite(float(context.log_probability(candidate))):
            accepted.append(candidate)
        if len(accepted) == WALKERS:
            break
    if len(accepted) != WALKERS:
        raise RuntimeError("insufficient initial walkers")
    positions = np.asarray(accepted)
    if np.linalg.matrix_rank(positions - positions.mean(axis=0)) < 3:
        raise RuntimeError("initial ensemble rank deficient")
    return positions


def sample_to(context: SamplerContext, method: str, target_name: str, steps: int) -> None:
    """把四个独立 ensemble 采样到指定总步数。"""

    path = chain_path(method, target_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    case_index = TARGETS.index(target_name) * len(METHODS) + METHODS.index(method)
    global _ACTIVE
    _ACTIVE = context
    try:
        for ensemble in range(ENSEMBLES):
            group = f"ensemble_{ensemble}"
            exists = group_exists(path, group)
            backend = emcee.backends.HDFBackend(str(path), name=group)
            seed = SEED + 100000 * case_index + 1009 * ensemble
            if not exists or backend.iteration == 0:
                backend.reset(WALKERS, 3)
                initial = initial_positions(context, np.random.default_rng(seed))
                random_state = np.random.RandomState(seed + 17)
            else:
                initial, random_state = None, None
            sampler = emcee.EnsembleSampler(
                WALKERS, 3, active_log_probability, vectorize=True, backend=backend,
                moves=[(emcee.moves.StretchMove(a=2.0), 0.8), (emcee.moves.DEMove(), 0.2)],
            )
            if random_state is not None:
                sampler.random_state = random_state.get_state()
            remaining = max(0, int(steps) - int(backend.iteration))
            if remaining:
                sampler.run_mcmc(initial, remaining, progress=False, skip_initial_state_check=True)
    finally:
        _ACTIVE = None


def load_chain(method: str, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    """读取四个 ensemble 的 chain 与 accepted counts。"""

    chains, accepted = [], []
    for ensemble in range(ENSEMBLES):
        backend = emcee.backends.HDFBackend(
            str(chain_path(method, target_name)), name=f"ensemble_{ensemble}", read_only=True
        )
        chains.append(np.asarray(backend.get_chain(), dtype=float))
        accepted.append(np.asarray(backend.accepted, dtype=float))
    if len({item.shape for item in chains}) != 1:
        raise RuntimeError("ensemble shape drift")
    return np.stack(chains), np.stack(accepted)


def boundary_mass(values: np.ndarray, bounds: tuple[float, float]) -> float:
    """计算先验两端各 2% 区域的 posterior mass。"""

    width = bounds[1] - bounds[0]
    return float(np.mean(
        (values <= bounds[0] + 0.02 * width) | (values >= bounds[1] - 0.02 * width)
    ))


def pooled_rank_normalize(values: np.ndarray) -> np.ndarray:
    """对全部独立 ensemble 的边际样本做 pooled rank normalization。"""

    ranks = rankdata(values, method="average").reshape(values.shape)
    probabilities = (ranks - 0.375) / (values.size + 0.25)
    return norm.ppf(probabilities)


def temporal_split(values: np.ndarray) -> np.ndarray:
    """每个独立 ensemble 时间二分并在半链内 pool walkers。"""

    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    return split.reshape(2 * values.shape[0], half * values.shape[2])


def ensemble_scaled_ess(transformed: np.ndarray) -> float:
    """将 ensemble-iteration ESS 按 walker 信息倍率扩展。"""

    series = np.mean(transformed, axis=2)
    iteration_ess = float(
        az.ess(series, method="identity", chain_axis=0, draw_axis=1)
    )
    draw_variance = float(np.var(transformed, ddof=1))
    series_variance = float(np.var(series, ddof=1))
    if not math.isfinite(iteration_ess) or draw_variance <= 0 or series_variance <= 0:
        return math.nan
    return min(float(transformed.size), iteration_ess * draw_variance / series_variance)


def independent_diagnostics(retained: np.ndarray) -> dict[str, np.ndarray]:
    """计算 rank-Rhat 和 bulk/tail ESS。"""

    dimension = retained.shape[-1]
    location = np.empty(dimension)
    folded = np.empty(dimension)
    bulk = np.empty(dimension)
    tail = np.empty(dimension)
    for parameter in range(dimension):
        values = retained[..., parameter]
        ranked = pooled_rank_normalize(values)
        folded_rank = pooled_rank_normalize(np.abs(values - np.median(values)))
        location[parameter] = float(
            az.rhat(temporal_split(ranked), method="identity", chain_axis=0, draw_axis=1)
        )
        folded[parameter] = float(
            az.rhat(temporal_split(folded_rank), method="identity", chain_axis=0, draw_axis=1)
        )
        q05, q95 = np.quantile(values, (0.05, 0.95))
        low = (values <= q05).astype(float)
        high = (values >= q95).astype(float)
        bulk[parameter] = ensemble_scaled_ess(ranked)
        tail[parameter] = min(ensemble_scaled_ess(low), ensemble_scaled_ess(high))
    return {
        "rank_location_split_rhat": location,
        "rank_folded_split_rhat": folded,
        "rank_normalized_split_rhat": np.maximum(location, folded),
        "bulk_ess": bulk, "tail_ess": tail,
    }


def diagnose(
    context: SamplerContext, coordinates: np.ndarray, accepted: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """转换物理参数并应用全部预注册 convergence gates。"""

    burn = int(BURN_FRACTION * coordinates.shape[1])
    retained = context.to_physical(coordinates[:, burn:])
    diagnostics = independent_diagnostics(retained)
    tau_rows = []
    for ensemble in range(ENSEMBLES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tau = np.asarray(emcee.autocorr.integrated_time(retained[ensemble], quiet=True))
            except Exception:
                tau = np.full(3, math.nan)
        tau_rows.append(tau)
    tau_rows = np.asarray(tau_rows)
    length_over_tau = retained.shape[1] / np.nanmax(tau_rows, axis=0)
    flat = retained.reshape(-1, 3)
    scale = np.std(flat, axis=0, ddof=1)
    midpoint = retained.shape[1] // 2
    first = retained[:, :midpoint].reshape(-1, 3)
    second = retained[:, midpoint:].reshape(-1, 3)
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / scale
    ensemble_medians = np.median(retained, axis=(1, 2))
    ensemble_spread = np.ptp(ensemble_medians, axis=0) / scale
    acceptance = float(np.mean(accepted / coordinates.shape[1]))
    boundaries = {
        "fNL": boundary_mass(flat[:, 0], FNL_BOUNDS),
        "b1": boundary_mass(flat[:, 1], B1_BOUNDS),
        "Pshot_residual": boundary_mass(flat[:, 2], PSHOT_BOUNDS),
    }
    checks = {
        "rhat": bool(np.all(diagnostics["rank_normalized_split_rhat"] < 1.01)),
        "fnl_ess": bool(diagnostics["bulk_ess"][0] > 2000 and diagnostics["tail_ess"][0] > 2000),
        "other_ess": bool(
            np.all(diagnostics["bulk_ess"][1:] > 1000)
            and np.all(diagnostics["tail_ess"][1:] > 1000)
        ),
        "length_tau": bool(np.all(length_over_tau > 50)),
        "half_shift": bool(np.all(half_shift < 0.1)),
        "ensemble_spread": bool(np.all(ensemble_spread < 0.2)),
        "boundaries": bool(all(value < 0.01 for value in boundaries.values())),
        "acceptance": bool(0.1 <= acceptance <= 0.7),
    }
    return {
        **diagnostics,
        "tau_by_ensemble": tau_rows,
        "retained_length_over_tau": length_over_tau,
        "half_chain_median_shift_sigma": half_shift,
        "ensemble_medians": ensemble_medians,
        "ensemble_median_spread_sigma": ensemble_spread,
        "mean_acceptance_fraction": acceptance,
        "boundary_mass": boundaries,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "chain_layout": {
            "ensembles": ENSEMBLES, "walkers": WALKERS,
            "steps": int(coordinates.shape[1]), "burn": burn,
            "retained_samples": int(flat.shape[0]),
        },
    }, retained


def run_sampling(method: str, target_name: str) -> None:
    """采样一个 case，并自动延长到收敛或最大步数。"""

    profiles = load_profiles()
    context = build_context(method, target_name, profiles)
    case_index = TARGETS.index(target_name) * len(METHODS) + METHODS.index(method)
    contract = {
        "schema": "quijote-z0p5-five-method-p0only-chain-v1",
        "created_utc": utc_now(), "method": method, "target": target_name,
        "truth_fNL": context.case.truth, "parameters": PARAMETERS,
        "bounds": {"fNL": FNL_BOUNDS, "b1": B1_BOUNDS, "Pshot_residual": PSHOT_BOUNDS},
        "profile_sha256": sha256(PROFILE_JSON), "profile_npz_sha256": sha256(PROFILE_NPZ),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "amendment_sha256": sha256(AMENDMENT), "runner_sha256": sha256(Path(__file__)),
        "sampler": {
            "ensembles": ENSEMBLES, "walkers": WALKERS,
            "moves": "0.8 StretchMove(a=2)+0.2 DEMove",
            "burn_fraction": BURN_FRACTION, "seed": SEED + 100000 * case_index,
            "coordinates": "constant Hessian-whitened physical parameters",
        },
        "hessian": {
            "centre": context.centre,
            "covariance": context.transform @ context.transform.T,
            "condition_number": float(np.linalg.cond(context.transform @ context.transform.T)),
        },
    }
    write_or_validate_contract(contract_path(method, target_name), contract)
    existing = []
    path = chain_path(method, target_name)
    if path.is_file():
        for ensemble in range(ENSEMBLES):
            if group_exists(path, f"ensemble_{ensemble}"):
                backend = emcee.backends.HDFBackend(
                    str(path), name=f"ensemble_{ensemble}", read_only=True
                )
                existing.append(int(backend.iteration))
    target_steps = max(INITIAL_STEPS, max(existing, default=0))
    started = time.perf_counter()
    while True:
        sample_to(context, method, target_name, target_steps)
        coordinates, accepted = load_chain(method, target_name)
        diagnostic, retained = diagnose(context, coordinates, accepted)
        actual = coordinates.shape[1]
        print(
            f"{key(method,target_name)} steps={actual} pass={diagnostic['pass']} "
            f"Rhat={np.max(diagnostic['rank_normalized_split_rhat']):.6f} "
            f"ESS={min(np.min(diagnostic['bulk_ess']),np.min(diagnostic['tail_ess'])):.1f} "
            f"L/tau={np.nanmin(diagnostic['retained_length_over_tau']):.2f}",
            flush=True,
        )
        if diagnostic["pass"] or actual >= MAX_STEPS:
            break
        target_steps = min(actual + INCREMENT, MAX_STEPS)
    flat = retained.reshape(-1, 3)
    rng = np.random.default_rng(SEED + 900000 + case_index)
    count = min(POSTERIOR_DRAWS, flat.shape[0])
    selected = flat[rng.choice(flat.shape[0], size=count, replace=False)]
    logp = -0.5 * context.case.objective_many(selected)
    atomic_npz(
        posterior_path(method, target_name),
        {
            "posterior": selected, "log_probability": logp,
            "parameter_names": np.asarray(PARAMETERS),
            "target": context.case.target, "covariance": context.case.covariance,
        },
    )
    record = {
        "schema": "quijote-z0p5-five-method-p0only-task-v1",
        "created_utc": utc_now(), "method": method, "target": target_name,
        "status": "complete" if diagnostic["pass"] else "nonconverged_at_max_steps",
        "diagnostics": diagnostic,
        "elapsed_seconds": time.perf_counter() - started,
        "chain": artifact(path), "contract": artifact(contract_path(method, target_name)),
        "posterior": artifact(posterior_path(method, target_name)),
    }
    atomic_json(task_path(method, target_name), record)
    if not diagnostic["pass"]:
        raise RuntimeError(f"{key(method,target_name)} did not converge")


def parameter_summary(values: np.ndarray) -> dict[str, float]:
    """返回 central posterior quantiles。"""

    q = np.quantile(np.asarray(values, dtype=float), (0.025, 0.16, 0.5, 0.84, 0.975))
    return {
        "q2p5": float(q[0]), "q16": float(q[1]), "median": float(q[2]),
        "q84": float(q[3]), "q97p5": float(q[4]),
        "half_width": float(0.5 * (q[3] - q[1])),
    }


def parameter_range(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    """用稳健分位数定义绘图区间。"""

    lower, upper = np.quantile(values, (0.002, 0.998))
    width = upper - lower
    return float(lower - padding * width), float(upper + padding * width)


def density_2d(
    x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], dict[str, float]]:
    """计算平滑二维密度及68%/95% HPD 阈值。"""

    histogram, x_edges, y_edges = np.histogram2d(
        x, y, bins=BINS, range=(x_range, y_range)
    )
    smooth = gaussian_filter(histogram.astype(float), SMOOTHING, mode="nearest")
    total = float(np.sum(smooth))
    if not total > 0:
        raise ValueError("二维 posterior histogram 为空")
    density = smooth / total
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)

    def threshold(probability: float) -> float:
        position = min(int(np.searchsorted(cumulative, probability)), ordered.size - 1)
        return float(ordered[position])

    threshold68, threshold95 = threshold(0.68), threshold(0.95)
    if not 0 < threshold95 < threshold68:
        raise ValueError("HPD 阈值退化")
    level68, level95 = threshold68, threshold95
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    actual68 = float(np.sum(density[density >= level68]))
    actual95 = float(np.sum(density[density >= level95]))
    return (
        mesh_x, mesh_y, density, (level95, level68),
        {"coverage_68": actual68, "coverage_95": actual95},
    )


def draw_contours(
    axis: Any, x: np.ndarray, y: np.ndarray,
    x_range: tuple[float, float], y_range: tuple[float, float],
    color: str, *, fill: bool = True,
) -> dict[str, float]:
    """在 axis 上绘制95%虚线和68%实线 contours。"""

    xc, yc, density, levels, coverage = density_2d(x, y, x_range, y_range)
    if fill:
        axis.contourf(
            xc, yc, density, levels=(levels[0], levels[1], float(np.max(density)) * 1.001),
            colors=(color, color), alpha=.12,
        )
    axis.contour(
        xc, yc, density, levels=levels,
        colors=(color, color), linestyles=("--", "-"), linewidths=(1.0, 1.5),
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)
    return coverage


def draw_marginal(axis: Any, values: np.ndarray, limits: tuple[float, float], color: str, label: str) -> None:
    """绘制平滑一维边际密度。"""

    histogram, edges = np.histogram(values, bins=100, range=limits, density=True)
    density = gaussian_filter1d(histogram.astype(float), 1.1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(centres, density, color=color, lw=1.8, label=label)


def inherited_rows_and_samples() -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """读取冻结 v1 四方法结果与 posterior，做链路校验和分位数 parity。"""

    frozen = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
    if frozen.get("status") != "complete":
        raise RuntimeError("frozen v1 results status not complete")
    if sha256(V1_POSTERIORS) != frozen["outputs"]["posteriors"]["sha256"]:
        raise RuntimeError("frozen v1 posterior hash drift")
    stored = load_npz(V1_POSTERIORS)
    rows: list[dict[str, Any]] = []
    samples: dict[tuple[str, str], np.ndarray] = {}
    combined: dict[str, np.ndarray] = {}
    for row in frozen["rows"]:
        method = row["method"]
        if method not in INHERITED_METHODS:
            continue
        name = key(method, row["target"])
        posterior = np.asarray(stored[f"{name}__posterior"], dtype=float)
        if posterior.shape != (POSTERIOR_DRAWS, 3) or not np.all(np.isfinite(posterior)):
            raise RuntimeError(f"frozen v1 posterior shape/finite failed: {name}")
        summary = parameter_summary(posterior[:, 0])
        for field in ("q2p5", "q16", "median", "q84", "q97p5", "half_width"):
            if abs(summary[field] - float(row["fNL"][field])) > 1e-12:
                raise RuntimeError(f"frozen v1 quantile parity failed: {name}: {field}")
        samples[(method, row["target"])] = posterior
        combined[f"{name}__posterior"] = posterior
        combined[f"{name}__log_probability"] = np.asarray(stored[f"{name}__log_probability"], dtype=float)
        combined[f"{name}__target"] = np.asarray(stored[f"{name}__target"], dtype=float)
        combined[f"{name}__covariance"] = np.asarray(stored[f"{name}__covariance"], dtype=float)
        rows.append({**row, "source": "frozen_v1"})
    if len(rows) != len(INHERITED_METHODS) * len(TARGETS):
        raise RuntimeError("frozen v1 row count mismatch")
    return rows, samples, combined


def write_figures(rows: list[dict[str, Any]], samples: dict[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    """生成 overlay contours 与 recovery/precision PDF。"""

    FIGURES.mkdir(parents=True, exist_ok=True)
    contour_checks: dict[str, Any] = {}
    with PdfPages(MARGINAL_PDF) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for column, target_name in enumerate(TARGETS):
            combined = np.concatenate([samples[(method, target_name)][:, 0] for method in METHODS])
            limits = parameter_range(combined)
            for method in METHODS:
                draw_marginal(
                    axes[column], samples[(method, target_name)][:, 0], limits,
                    COLORS[method], METHOD_LABELS[method],
                )
            axes[column].axvline(TRUTHS[target_name], color="k", ls=":", lw=1.3)
            axes[column].set_xlabel(r"$f_{\rm NL}$")
            axes[column].set_ylabel("marginal density")
            axes[column].set_yticks([])
            axes[column].set_title(f"input {TRUTHS[target_name]:.0f}")
            axes[column].grid(alpha=0.15)
        axes[0].legend(frameon=False, fontsize=9)
        figure.suptitle(r"Five-method $P_0$-only MCMC, $k_{\max}=0.08$")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        for target_name in TARGETS:
            all_f = np.concatenate([samples[(method, target_name)][:, 0] for method in METHODS])
            all_b = np.concatenate([samples[(method, target_name)][:, 1] for method in METHODS])
            xlimits = parameter_range(all_f)
            ylimits = parameter_range(all_b)
            figure, axis = plt.subplots(figsize=(7.2, 5.8))
            for method in METHODS:
                sample = samples[(method, target_name)]
                contour_checks[key(method, target_name)] = draw_contours(
                    axis, sample[:, 0], sample[:, 1], xlimits, ylimits,
                    COLORS[method], fill=False,
                )
            axis.axvline(TRUTHS[target_name], color="k", ls=":", lw=1.3)
            axis.set_xlabel(r"$f_{\rm NL}$")
            axis.set_ylabel(r"$b_1$")
            axis.set_title(f"input fNL={TRUTHS[target_name]:.0f}")
            axis.grid(alpha=0.15)
            handles = [
                Line2D([0], [0], color=COLORS[method], lw=1.8, label=METHOD_LABELS[method])
                for method in METHODS
            ]
            axis.legend(handles=handles, frameon=False, fontsize=9)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    lookup = {(row["method"], row["target"]): row for row in rows}
    x = np.arange(len(METHODS), dtype=float)
    with PdfPages(RECOVERY_PDF) as pdf:
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
        offsets = {"fNL0": -0.12, "fNL100": 0.12}
        target_colors = {"fNL0": "#4E79A7", "fNL100": "#E15759"}
        for target_name in TARGETS:
            selected = [lookup[(method, target_name)] for method in METHODS]
            medians = np.asarray([row["fNL"]["median"] for row in selected])
            lower = medians - np.asarray([row["fNL"]["q16"] for row in selected])
            upper = np.asarray([row["fNL"]["q84"] for row in selected]) - medians
            top.errorbar(
                x + offsets[target_name], medians, yerr=np.vstack((lower, upper)),
                fmt="o", capsize=4, color=target_colors[target_name],
                label=f"input {TRUTHS[target_name]:.0f}",
            )
            ratios = np.asarray([
                row["fNL"]["half_width"] / selected[0]["fNL"]["half_width"]
                for row in selected
            ])
            bottom.plot(
                x + offsets[target_name], ratios, "o-", color=target_colors[target_name],
                label=f"input {TRUTHS[target_name]:.0f}",
            )
        top.axhline(0.0, color="#4E79A7", ls=":", lw=1.0)
        top.axhline(100.0, color="#E15759", ls=":", lw=1.0)
        top.set_ylabel(r"recovered $f_{\rm NL}$ (68%)")
        top.legend(frameon=False)
        top.grid(alpha=0.15)
        bottom.axhline(1.0, color="k", ls=":", lw=1.0)
        bottom.set_ylabel("half-width / pre")
        bottom.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=15)
        bottom.grid(alpha=0.15)
        figure.suptitle(r"Impact of reconstruction on $P_0$-only local-PNG constraints")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return contour_checks


def write_summary() -> None:
    """合并冻结 v1 四方法行与新 sto_only 行，生成五方法表格/报告/PDF。"""

    profiles = load_profiles()
    inherited_rows, samples, combined_arrays = inherited_rows_and_samples()
    combined_arrays["parameter_names"] = np.asarray(PARAMETERS)

    rows: list[dict[str, Any]] = list(inherited_rows)
    for target_name in TARGETS:
        for method in SAMPLE_METHODS:
            task = json.loads(task_path(method, target_name).read_text(encoding="utf-8"))
            if task.get("status") != "complete" or not task["diagnostics"]["pass"]:
                raise RuntimeError(f"incomplete MCMC: {key(method,target_name)}")
            posterior_file = posterior_path(method, target_name)
            if sha256(posterior_file) != task["posterior"]["sha256"]:
                raise RuntimeError(f"posterior hash drift: {key(method,target_name)}")
            arrays = load_npz(posterior_file)
            posterior = np.asarray(arrays["posterior"], dtype=float)
            if posterior.shape != (POSTERIOR_DRAWS, 3) or not np.all(np.isfinite(posterior)):
                raise RuntimeError(f"posterior shape/finite failed: {key(method,target_name)}")
            samples[(method, target_name)] = posterior
            name = key(method, target_name)
            combined_arrays[f"{name}__posterior"] = posterior
            combined_arrays[f"{name}__log_probability"] = arrays["log_probability"]
            combined_arrays[f"{name}__target"] = arrays["target"]
            combined_arrays[f"{name}__covariance"] = arrays["covariance"]
            summaries = {
                pname: parameter_summary(posterior[:, index])
                for index, pname in enumerate(PARAMETERS)
            }
            truth = TRUTHS[target_name]
            summaries["fNL"]["truth_percentile"] = float(np.mean(posterior[:, 0] <= truth))
            summaries["fNL"]["truth_inside_68"] = bool(
                summaries["fNL"]["q16"] <= truth <= summaries["fNL"]["q84"]
            )
            summaries["fNL"]["truth_inside_95"] = bool(
                summaries["fNL"]["q2p5"] <= truth <= summaries["fNL"]["q97p5"]
            )
            profile_row = profiles["rows"][key(method, target_name)]
            rows.append({
                "method": method, "target": target_name, "truth_fNL": truth,
                **summaries,
                "profile_map_fNL": profile_row["map"]["fNL"],
                "profile_truth_delta_chi2": profile_row["truth_profile"]["delta_chi2"],
                "goodness_of_fit_p": profile_row["goodness_of_fit_p"],
                "diagnostics": task["diagnostics"],
                "source": "five_method_new",
            })
    rows.sort(key=lambda row: (TARGETS.index(row["target"]), METHODS.index(row["method"])))

    comparison: dict[str, Any] = {}
    for target_name in TARGETS:
        local = {row["method"]: row for row in rows if row["target"] == target_name}
        pre_width = local["pre"]["fNL"]["half_width"]
        comparison[target_name] = {}
        for method in METHODS:
            ratio = local[method]["fNL"]["half_width"] / pre_width
            comparison[target_name][method] = {
                "half_width_over_pre": ratio,
                "information_over_pre": 1.0 / ratio**2,
                "median_minus_truth": local[method]["fNL"]["median"] - TRUTHS[target_name],
            }

    contour_checks = write_figures(rows, samples)
    atomic_npz(RESULT_NPZ, combined_arrays)

    PART_FIGURES.mkdir(parents=True, exist_ok=True)
    published = {}
    for source in (MARGINAL_PDF, RECOVERY_PDF):
        destination = PART_FIGURES / source.name
        shutil.copy2(source, destination)
        published[source.name] = artifact(destination)

    payload = {
        "schema": "quijote-z0p5-five-method-p0only-mcmc-results-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "methods": METHODS, "targets": TARGETS, "truths": TRUTHS,
            "parameters": PARAMETERS, "P_bins": 12,
            "covariance": "method-specific fNL0 Fid500 / (42.875 * Hartlap_12); no /100",
            "sto_only_definition": (
                "matter-assisted N+Env STO weighting (r10, lambda_th=1, env_source=dm, "
                "weight_source=dm, number weighting); halo positions unchanged, no gra-recon"
            ),
            "row_sources": {
                "pre/standard/marisa/sto_marisa": "frozen four-method v1 (not re-run)",
                "sto_only": "new five-method run under the 2026-09-16 amendment",
            },
        },
        "rows": rows, "comparison_to_pre": comparison,
        "contour_checks": contour_checks,
        "inputs": {
            "matrix": artifact(MATRIX), "matrix_audit": artifact(MATRIX_AUDIT),
            "theory": artifact(P_THEORY), "preregistration": artifact(PREREGISTRATION),
            "amendment": artifact(AMENDMENT),
            "profiles": artifact(PROFILE_JSON), "profile_arrays": artifact(PROFILE_NPZ),
            "frozen_v1_results": artifact(V1_RESULTS),
            "frozen_v1_posteriors": artifact(V1_POSTERIORS),
            "runner": artifact(Path(__file__)),
        },
        "outputs": {
            "posteriors": artifact(RESULT_NPZ),
            "marginal_pdf": artifact(MARGINAL_PDF), "recovery_pdf": artifact(RECOVERY_PDF),
            "published_figures": published,
        },
    }
    atomic_json(RESULT_JSON, payload)
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method", "source", "target", "truth_fNL", "median", "q16", "q84", "q2p5", "q97p5",
            "half_width", "b1_median", "Pshot_median", "truth_percentile",
            "truth_inside_68", "truth_inside_95", "profile_truth_delta_chi2",
            "goodness_of_fit_p", "half_width_over_pre", "information_over_pre",
            "max_rhat", "min_bulk_ess", "min_tail_ess",
        ))
        writer.writeheader()
        for row in rows:
            comp = comparison[row["target"]][row["method"]]
            diagnostic = row["diagnostics"]
            writer.writerow({
                "method": row["method"], "source": row["source"],
                "target": row["target"], "truth_fNL": row["truth_fNL"],
                "median": row["fNL"]["median"], "q16": row["fNL"]["q16"],
                "q84": row["fNL"]["q84"], "q2p5": row["fNL"]["q2p5"],
                "q97p5": row["fNL"]["q97p5"], "half_width": row["fNL"]["half_width"],
                "b1_median": row["b1"]["median"],
                "Pshot_median": row["Pshot_residual"]["median"],
                "truth_percentile": row["fNL"]["truth_percentile"],
                "truth_inside_68": row["fNL"]["truth_inside_68"],
                "truth_inside_95": row["fNL"]["truth_inside_95"],
                "profile_truth_delta_chi2": row["profile_truth_delta_chi2"],
                "goodness_of_fit_p": row["goodness_of_fit_p"],
                "half_width_over_pre": comp["half_width_over_pre"],
                "information_over_pre": comp["information_over_pre"],
                "max_rhat": float(np.max(diagnostic["rank_normalized_split_rhat"])),
                "min_bulk_ess": float(np.min(diagnostic["bulk_ess"])),
                "min_tail_ess": float(np.min(diagnostic["tail_ess"])),
            })
    lines = [
        "# 五方法 P0-only matched100 MCMC（STO-only 扩展）", "",
        "Status: `complete`。四方法行复制自冻结 v1（未重跑）；`sto_only` 为新运行。",
        "",
        "| target | method | fNL median [16,84] | half-width | b1 [16,84] | width/pre | truth in 68/95 | source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        f = row["fNL"]; b = row["b1"]; comp = comparison[row["target"]][row["method"]]
        lines.append(
            f"| {row['target']} | {METHOD_LABELS[row['method']]} | "
            f"{f['median']:.3f} [{f['q16']:.3f},{f['q84']:.3f}] | {f['half_width']:.3f} | "
            f"{b['median']:.5f} [{b['q16']:.5f},{b['q84']:.5f}] | "
            f"{comp['half_width_over_pre']:.4f} | {f['truth_inside_68']}/{f['truth_inside_95']} | "
            f"{row['source']} |"
        )
    lines.extend([
        "", "Covariance：每种方法自己的 fNL=0 Fid500 单盒 covariance，除以 42.875 和 Hartlap；target mean 不除以100。",
        "", "四方法冻结来源：`results/four_method_p0only_mcmc_results.json` 与旧仓库 v1 posterior NPZ（sha256 已校验）。",
        "", f"结果 JSON：`{RESULT_JSON}`。", f"posterior NPZ：`{RESULT_NPZ}`。",
        f"PDF：`{PART_FIGURES / MARGINAL_PDF.name}`、`{PART_FIGURES / RECOVERY_PDF.name}`。", "",
    ])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps(jsonable({
        "status": "complete", "result": str(RESULT_JSON), "report": str(REPORT),
        "figures": [str(PART_FIGURES / MARGINAL_PDF.name), str(PART_FIGURES / RECOVERY_PDF.name)],
    }), indent=2))


def parse_args() -> argparse.Namespace:
    """解析 profiles/sample/summarize/all stage。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profiles", "sample", "summarize", "all"), required=True)
    parser.add_argument(
        "--method", choices=SAMPLE_METHODS,
        help="sample 阶段的目标方法；本轮修正案限定为 sto_only",
    )
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument(
        "--profile-methods", nargs="+", choices=SAMPLE_METHODS, default=list(SAMPLE_METHODS),
        help="profiles 阶段运行的方法（默认只有 sto_only）",
    )
    return parser.parse_args()


def main() -> None:
    """按请求执行一个或全部阶段。"""

    args = parse_args()
    if args.stage in ("profiles", "all"):
        prepare_profiles(tuple(args.profile_methods))
    if args.stage == "sample":
        if args.method is None or args.target is None:
            raise ValueError("--stage sample requires --method and --target")
        run_sampling(args.method, args.target)
    elif args.stage == "all":
        for target_name in TARGETS:
            for method in SAMPLE_METHODS:
                run_sampling(method, target_name)
    if args.stage in ("summarize", "all"):
        write_summary()


if __name__ == "__main__":
    main()
