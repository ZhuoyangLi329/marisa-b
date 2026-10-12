#!/usr/bin/env python3
"""p 固定为校准值的三方法 P0-only MCMC（pre / STO-only / pre+STO-only joint）。

代码大纲
========
1. p 固定：pre 通道 1.1788659、sto 通道 1.3394638（来自
   `run_fnlfixed100_freep_mcmc.py` 校准的中位数，provenance 记录于
   results/inputs）。bphi = 2*delta_c*(b1 - p_fixed)。
2. 三个 case 族 x 两个 target 重新拟合 fNL：
   - pre / sto_only：单通道 12-bin，参数 (fNL, b1, Pshot)；
   - pre_sto_only：24-bin 联合，参数 (fNL, b1_pre, b1_post, sn0_pre, sn0_post)。
   数据、协方差、先验、收敛门全部沿用既有约定。
3. 复用 iminuit profile + HESSE 白化 + emcee 自动延长机器；汇总含独立
   likelihood parity 复核、recovery gate 报告与新主结果表。
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
from typing import Any
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
TAG = "quijote_halo_z0p5_fixedp3method_p0only_fid0_p100matched100_mcmc_v1"
ANALYSIS = PART / "work" / TAG
FIVE_MATRIX = PART / "inputs/five_method_p0_matrix.npz"
FIVE_AUDIT = PART / "results/five_method_p0_matrix_audit.json"
JOINT_MATRIX = PART / "inputs/pre_post_joint_p0_matrix.npz"
JOINT_AUDIT = PART / "results/pre_post_joint_p0_matrix_audit.json"
FREEP_RESULTS = PART / "results/fnlfixed100_freep_results.json"
P_THEORY = PART / "inputs/p0_finite_local_png_lattice.npz"
PROFILE_JSON = ANALYSIS / "FIXEDP_PROFILES.json"
PROFILE_NPZ = ANALYSIS / "FIXEDP_PROFILE_ARRAYS.npz"
RESULT_JSON = ANALYSIS / "FIXEDP_RESULTS.json"
RESULT_CSV = ANALYSIS / "FIXEDP_RESULTS.csv"
RESULT_NPZ = ANALYSIS / "FIXEDP_POSTERIORS.npz"
REPORT = ANALYSIS / "FIXEDP_REPORT.md"
CHAINS = ANALYSIS / "chains"
TASKS = ANALYSIS / "tasks"
POSTERIORS = ANALYSIS / "posteriors"

CASES = ("pre", "sto_only", "pre_sto_only")
TARGETS = ("fNL0", "fNL100")
TRUTHS = {"fNL0": 0.0, "fNL100": 100.0}
# 校准来源：work/..._fnlfixed100_freep_calib_v1/FREEP_RESULTS.json 的 p 中位数。
P_FIXED = {"pre": 1.1788659191136683, "sto_only": 1.3394637946618602}
P_FIXED_ROUNDED = {"pre": 1.18, "sto_only": 1.34}
PARAMETERS_SINGLE = ("fNL", "b1", "Pshot_residual")
PARAMETERS_JOINT = ("fNL", "b1_pre", "b1_post", "sn0_pre", "sn0_post")
CASE_LABELS = {"pre": "pre", "sto_only": "STO only", "pre_sto_only": "pre+STO only (joint)"}
FNL_BOUNDS = (-300.0, 300.0)
B1_BOUNDS = (0.1, 6.0)
PSHOT_BOUNDS = (-1.0e5, 1.0e5)
VOLUME = 42.875
HARTLAP_12 = (500.0 - 12.0 - 2.0) / 499.0
HARTLAP_24 = (500.0 - 24.0 - 2.0) / 499.0
DELTA_C = 1.686
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 40000
SEED = 20261201


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def jsonable(value: Any) -> Any:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def bphi(b1, p):
    return 2.0 * DELTA_C * (b1 - p)


def channel_prediction(f, b1, sn0, p, theory):
    bp = np.asarray(bphi(b1, p))
    return (
        b1[:, None] ** 2 * theory["PL_lattice"][None, :]
        + 2.0 * f[:, None] * (b1 * bp)[:, None] * theory["PL_over_M_lattice"][None, :]
        + f[:, None] ** 2 * bp[:, None] ** 2 * theory["Pphi_lattice"][None, :]
        + sn0[:, None]
    )


@dataclass
class Case:
    """一个 case 的冻结 likelihood（单通道或联合）。"""

    case_name: str
    target_name: str
    truth: float
    target: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    theory: dict[str, np.ndarray]

    @property
    def parameters(self) -> tuple[str, ...]:
        return PARAMETERS_JOINT if self.case_name == "pre_sto_only" else PARAMETERS_SINGLE

    def predictions(self, values: np.ndarray) -> np.ndarray:
        physical = np.asarray(values, dtype=float)
        scalar = physical.ndim == 1
        rows = physical[None, :] if scalar else physical
        if self.case_name == "pre_sto_only":
            pre = channel_prediction(rows[:, 0], rows[:, 1], rows[:, 3], P_FIXED["pre"], self.theory)
            post = channel_prediction(rows[:, 0], rows[:, 2], rows[:, 4], P_FIXED["sto_only"], self.theory)
            prediction = np.concatenate((pre, post), axis=1)
        else:
            p = P_FIXED[self.case_name]
            prediction = channel_prediction(rows[:, 0], rows[:, 1], rows[:, 2], p, self.theory)
        return prediction[0] if scalar else prediction

    def objective_many(self, values: np.ndarray) -> np.ndarray:
        predictions = self.predictions(values)
        rows = predictions[None, :] if predictions.ndim == 1 else predictions
        residual = rows - self.target[None, :]
        white = np.linalg.solve(self.cholesky, residual.T).T
        return np.sum(white**2, axis=1)

    def objective(self, values: np.ndarray) -> float:
        return float(self.objective_many(np.asarray(values, dtype=float))[0])


def load_case(case_name: str, target_name: str) -> Case:
    theory = load_npz(P_THEORY)
    if case_name == "pre_sto_only":
        audit = json.loads(JOINT_AUDIT.read_text(encoding="utf-8"))
        if audit.get("status") != "pass" or sha256(JOINT_MATRIX) != audit["output"]["sha256"]:
            raise RuntimeError("joint matrix audit/hash failed")
        arrays = load_npz(JOINT_MATRIX)
        target = np.asarray(arrays[f"pre_sto_only__target_{target_name}_matched100"], dtype=float)
        covariance = np.asarray(
            arrays["pre_sto_only__covariance_fNL0_fid500_one_box"], dtype=float
        ) / (VOLUME * HARTLAP_24)
    else:
        audit = json.loads(FIVE_AUDIT.read_text(encoding="utf-8"))
        if audit.get("status") != "pass" or sha256(FIVE_MATRIX) != audit["output"]["sha256"]:
            raise RuntimeError("five-method matrix audit/hash failed")
        arrays = load_npz(FIVE_MATRIX)
        target = np.asarray(arrays[f"{case_name}__target_{target_name}_matched100"], dtype=float)
        covariance = np.asarray(
            arrays[f"{case_name}__covariance_fNL0_fid500_one_box"], dtype=float
        ) / (VOLUME * HARTLAP_12)
    return Case(
        case_name, target_name, TRUTHS[target_name], target, covariance,
        np.linalg.cholesky(covariance), theory,
    )


def channel_start(case: Case, channel: str) -> tuple[float, float]:
    """fNL=0 线性化解单通道 b1^2 与常数 sn0 的稳定起点。"""

    covariance_one = case.covariance * VOLUME * (HARTLAP_24 if case.case_name == "pre_sto_only" else HARTLAP_12)
    if case.case_name == "pre_sto_only":
        sl = slice(0, 12) if channel == "pre" else slice(12, 24)
        block = covariance_one[sl, sl]
        target_channel = case.target[sl]
    else:
        block = covariance_one
        target_channel = case.target
    design = np.column_stack((case.theory["PL_lattice"], np.ones(12)))
    solved_design = np.linalg.solve(block, design)
    solved_target = np.linalg.solve(block, target_channel)
    coefficient = np.linalg.solve(design.T @ solved_design, design.T @ solved_target)
    b1_start = float(np.clip(math.sqrt(max(float(coefficient[0]), 0.01)), 0.101, 5.999))
    sn0_start = float(np.clip(coefficient[1], PSHOT_BOUNDS[0] + 1, PSHOT_BOUNDS[1] - 1))
    return b1_start, sn0_start


def profile_case(case: Case) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dimension = len(case.parameters)
    if case.case_name == "pre_sto_only":
        b1_pre, sn0_pre = channel_start(case, "pre")
        b1_post, sn0_post = channel_start(case, "post")
        start = dict(fNL=case.truth, b1_pre=b1_pre, b1_post=b1_post, sn0_pre=sn0_pre, sn0_post=sn0_post)
        bounds = {
            "fNL": FNL_BOUNDS, "b1_pre": B1_BOUNDS, "b1_post": B1_BOUNDS,
            "sn0_pre": PSHOT_BOUNDS, "sn0_post": PSHOT_BOUNDS,
        }
        errors = {"fNL": 8.0, "b1_pre": 0.02, "b1_post": 0.02, "sn0_pre": 100.0, "sn0_post": 100.0}
        dof = 19
    else:
        b1_start, sn0_start = channel_start(case, "channel")
        start = dict(fNL=case.truth, b1=b1_start, Pshot_residual=sn0_start)
        bounds = {"fNL": FNL_BOUNDS, "b1": B1_BOUNDS, "Pshot_residual": PSHOT_BOUNDS}
        errors = {"fNL": 8.0, "b1": 0.02, "Pshot_residual": 100.0}
        dof = 9

    if case.case_name == "pre_sto_only":
        def objective(fNL, b1_pre, b1_post, sn0_pre, sn0_post):
            return case.objective(np.asarray((fNL, b1_pre, b1_post, sn0_pre, sn0_post)))

        def truth_objective(b1_pre, b1_post, sn0_pre, sn0_post):
            return case.objective(
                np.asarray((case.truth, b1_pre, b1_post, sn0_pre, sn0_post))
            )
    else:
        def objective(fNL, b1, Pshot_residual):
            return case.objective(np.asarray((fNL, b1, Pshot_residual)))

        def truth_objective(b1, Pshot_residual):
            return case.objective(np.asarray((case.truth, b1, Pshot_residual)))

    fits = []
    for f_start in (case.truth, 0.0, 100.0, -100.0, 200.0):
        fit = Minuit(objective, **{**start, "fNL": f_start})
        fit.errordef = Minuit.LEAST_SQUARES
        for name, limit in bounds.items():
            fit.limits[name] = limit
        for name, error in errors.items():
            fit.errors[name] = error
        fit.strategy, fit.tol = 1, 1e-5
        fit.migrad(ncall=200000, iterate=8, use_simplex=True)
        fit.hesse()
        fits.append(fit)
    valid = [fit for fit in fits if fit.valid and math.isfinite(float(fit.fval))]
    best = min(valid or fits, key=lambda fit: float(fit.fval))
    if not best.valid or best.covariance is None:
        raise RuntimeError(f"{case.case_name}/{case.target_name} MAP/HESSE failed")
    best.minos("fNL", ncall=200000)
    error = best.merrors["fNL"]
    mapping = {name: float(best.values[name]) for name in case.parameters}

    nuisance = {k: v for k, v in mapping.items() if k != "fNL"}
    truth_fit = Minuit(truth_objective, **nuisance)
    truth_fit.errordef = Minuit.LEAST_SQUARES
    for name in nuisance:
        truth_fit.limits[name] = bounds[name]
        truth_fit.errors[name] = errors[name]
    truth_fit.strategy, truth_fit.tol = 1, 1e-5
    truth_fit.migrad(ncall=200000, iterate=8, use_simplex=True)
    truth_fit.hesse()
    if not truth_fit.valid:
        raise RuntimeError(f"{case.case_name}/{case.target_name} truth profile failed")
    truth_objective_value = float(truth_fit.fval)
    row = {
        "case": case.case_name, "target": case.target_name, "truth_fNL": case.truth,
        "map": mapping, "objective": float(best.fval),
        "data_chi2": float(best.fval), "dof": dof,
        "goodness_of_fit_p": float(chi2.sf(float(best.fval), dof)),
        "optimizer_valid": bool(best.valid),
        "hesse_covariance": np.asarray(best.covariance, dtype=float),
        "minos_fNL": {
            "lower": mapping["fNL"] + float(error.lower),
            "upper": mapping["fNL"] + float(error.upper),
            "lower_error": float(error.lower), "upper_error": float(error.upper),
            "valid": bool(error.is_valid),
        },
        "truth_profile": {
            "objective": truth_objective_value,
            "delta_chi2": truth_objective_value - float(best.fval),
        },
    }
    return row, {"target": case.target, "covariance": case.covariance}


def prepare_profiles() -> None:
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for target_name in TARGETS:
        for case_name in CASES:
            case = load_case(case_name, target_name)
            row, local = profile_case(case)
            name = f"{case_name}__{target_name}"
            rows[name] = row
            for item, value in local.items():
                arrays[f"{name}__{item}"] = value
            print(
                f"profile {name}: fNL={row['map']['fNL']:+.4f} "
                f"{row['minos_fNL']['lower_error']:+.3f}/{row['minos_fNL']['upper_error']:+.3f} "
                f"chi2/dof={row['data_chi2']:.2f}/{row['dof']} p={row['goodness_of_fit_p']:.4g}",
                flush=True,
            )
    atomic_npz(PROFILE_NPZ, arrays)
    atomic_json(PROFILE_JSON, {
        "schema": "quijote-z0p5-fixedp-3method-profiles-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "cases": CASES, "targets": TARGETS,
            "p_fixed": P_FIXED, "p_fixed_rounded": P_FIXED_ROUNDED,
            "parameters_single": PARAMETERS_SINGLE, "parameters_joint": PARAMETERS_JOINT,
        },
        "rows": rows,
        "inputs": {
            "five_matrix": artifact(FIVE_MATRIX), "five_audit": artifact(FIVE_AUDIT),
            "joint_matrix": artifact(JOINT_MATRIX), "joint_audit": artifact(JOINT_AUDIT),
            "theory": artifact(P_THEORY), "freep_calibration": artifact(FREEP_RESULTS),
            "runner": artifact(Path(__file__)),
        },
        "arrays": artifact(PROFILE_NPZ),
    })


def load_profiles() -> dict[str, Any]:
    payload = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or sha256(PROFILE_NPZ) != payload["arrays"]["sha256"]:
        raise RuntimeError("profile status/hash failed")
    for name, path in (
        ("five_matrix", FIVE_MATRIX), ("five_audit", FIVE_AUDIT),
        ("joint_matrix", JOINT_MATRIX), ("joint_audit", JOINT_AUDIT),
        ("theory", P_THEORY), ("runner", Path(__file__)),
    ):
        if sha256(path) != payload["inputs"][name]["sha256"]:
            raise RuntimeError(f"profile input drift: {name}")
    return payload


@dataclass
class SamplerContext:
    case: Case
    profile_row: dict[str, Any]
    centre: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray

    def to_physical(self, coordinates):
        values = np.asarray(coordinates, dtype=float)
        return self.centre + values @ self.transform.T

    def to_coordinates(self, physical):
        values = np.asarray(physical, dtype=float)
        return (values - self.centre) @ self.inverse.T

    def log_probability(self, coordinates):
        values = np.asarray(coordinates, dtype=float)
        scalar = values.ndim == 1
        rows = values[None, :] if scalar else values
        physical = self.to_physical(rows)
        if self.case.case_name == "pre_sto_only":
            bounds_map = [
                FNL_BOUNDS, B1_BOUNDS, B1_BOUNDS, PSHOT_BOUNDS, PSHOT_BOUNDS,
            ]
        else:
            bounds_map = [FNL_BOUNDS, B1_BOUNDS, PSHOT_BOUNDS]
        support = np.all(np.isfinite(physical), axis=1)
        for column, limit in enumerate(bounds_map[: physical.shape[1]]):
            support &= (physical[:, column] > limit[0]) & (physical[:, column] < limit[1])
        output = np.full(rows.shape[0], -math.inf)
        if np.any(support):
            output[support] = -0.5 * self.case.objective_many(physical[support])
        return float(output[0]) if scalar else output


def build_context(case_name: str, target_name: str, profiles: dict[str, Any]) -> SamplerContext:
    case = load_case(case_name, target_name)
    row = profiles["rows"][f"{case_name}__{target_name}"]
    parameters = case.parameters
    centre = np.asarray([row["map"][name] for name in parameters], dtype=float)
    covariance = np.asarray(row["hesse_covariance"], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0:
        floor = max(float(eigenvalues[-1]) * 1e-10, 1e-12)
        covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    transform = np.linalg.cholesky(covariance)
    inverse = np.linalg.inv(transform)
    context = SamplerContext(case, row, centre, transform, inverse)
    parity = abs(-2.0 * float(context.log_probability(np.zeros(len(parameters)))) - float(row["objective"]))
    if parity > 1e-8:
        raise RuntimeError(f"profile/logp parity failed: {case_name}/{target_name}: {parity}")
    return context


_ACTIVE: SamplerContext | None = None


def active_log_probability(values):
    if _ACTIVE is None:
        raise RuntimeError("no active sampler context")
    return _ACTIVE.log_probability(values)


def chain_path(case_name, target_name):
    return CHAINS / f"{case_name}__{target_name}.h5"


def contract_path(case_name, target_name):
    return CHAINS / f"{case_name}__{target_name}.contract.json"


def task_path(case_name, target_name):
    return TASKS / f"{case_name}__{target_name}.json"


def posterior_path(case_name, target_name):
    return POSTERIORS / f"{case_name}__{target_name}.npz"


def group_exists(path: Path, group: str) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as stream:
        return group in stream


def write_or_validate_contract(path: Path, payload: dict[str, Any]) -> None:
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
    dimension = len(context.case.parameters)
    accepted = []
    for _ in range(200000):
        scale = math.exp(rng.uniform(math.log(0.08), math.log(1.2)))
        candidate = rng.normal(0.0, scale, dimension)
        if math.isfinite(float(context.log_probability(candidate))):
            accepted.append(candidate)
        if len(accepted) == WALKERS:
            break
    if len(accepted) != WALKERS:
        raise RuntimeError("insufficient initial walkers")
    positions = np.asarray(accepted)
    if np.linalg.matrix_rank(positions - positions.mean(axis=0)) < dimension:
        raise RuntimeError("initial ensemble rank deficient")
    return positions


def sample_to(context: SamplerContext, case_name: str, target_name: str, steps: int) -> None:
    path = chain_path(case_name, target_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = len(context.case.parameters)
    case_index = TARGETS.index(target_name) * len(CASES) + CASES.index(case_name)
    global _ACTIVE
    _ACTIVE = context
    try:
        for ensemble in range(ENSEMBLES):
            group = f"ensemble_{ensemble}"
            exists = group_exists(path, group)
            backend = emcee.backends.HDFBackend(str(path), name=group)
            seed = SEED + 100000 * case_index + 1009 * ensemble
            if not exists or backend.iteration == 0:
                backend.reset(WALKERS, dimension)
                initial = initial_positions(context, np.random.default_rng(seed))
                random_state = np.random.RandomState(seed + 17)
            else:
                initial, random_state = None, None
            sampler = emcee.EnsembleSampler(
                WALKERS, dimension, active_log_probability, vectorize=True, backend=backend,
                moves=[(emcee.moves.StretchMove(a=2.0), 0.8), (emcee.moves.DEMove(), 0.2)],
            )
            if random_state is not None:
                sampler.random_state = random_state.get_state()
            remaining = max(0, int(steps) - int(backend.iteration))
            if remaining:
                sampler.run_mcmc(initial, remaining, progress=False, skip_initial_state_check=True)
    finally:
        _ACTIVE = None


def load_chain(case_name, target_name):
    chains, accepted = [], []
    for ensemble in range(ENSEMBLES):
        backend = emcee.backends.HDFBackend(
            str(chain_path(case_name, target_name)), name=f"ensemble_{ensemble}", read_only=True
        )
        chains.append(np.asarray(backend.get_chain(), dtype=float))
        accepted.append(np.asarray(backend.accepted, dtype=float))
    if len({item.shape for item in chains}) != 1:
        raise RuntimeError("ensemble shape drift")
    return np.stack(chains), np.stack(accepted)


def boundary_mass(values, bounds):
    width = bounds[1] - bounds[0]
    return float(np.mean(
        (values <= bounds[0] + 0.02 * width) | (values >= bounds[1] - 0.02 * width)
    ))


def pooled_rank_normalize(values):
    from scipy.stats import norm, rankdata
    ranks = rankdata(values, method="average").reshape(values.shape)
    probabilities = (ranks - 0.375) / (values.size + 0.25)
    return norm.ppf(probabilities)


def temporal_split(values):
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    return split.reshape(2 * values.shape[0], half * values.shape[2])


def ensemble_scaled_ess(transformed):
    series = np.mean(transformed, axis=2)
    iteration_ess = float(az.ess(series, method="identity", chain_axis=0, draw_axis=1))
    draw_variance = float(np.var(transformed, ddof=1))
    series_variance = float(np.var(series, ddof=1))
    if not math.isfinite(iteration_ess) or draw_variance <= 0 or series_variance <= 0:
        return math.nan
    return min(float(transformed.size), iteration_ess * draw_variance / series_variance)


def independent_diagnostics(retained):
    dimension = retained.shape[-1]
    location = np.empty(dimension); folded = np.empty(dimension)
    bulk = np.empty(dimension); tail = np.empty(dimension)
    for parameter in range(dimension):
        values = retained[..., parameter]
        ranked = pooled_rank_normalize(values)
        folded_rank = pooled_rank_normalize(np.abs(values - np.median(values)))
        location[parameter] = float(az.rhat(temporal_split(ranked), method="identity", chain_axis=0, draw_axis=1))
        folded[parameter] = float(az.rhat(temporal_split(folded_rank), method="identity", chain_axis=0, draw_axis=1))
        q05, q95 = np.quantile(values, (0.05, 0.95))
        low = (values <= q05).astype(float); high = (values >= q95).astype(float)
        bulk[parameter] = ensemble_scaled_ess(ranked)
        tail[parameter] = min(ensemble_scaled_ess(low), ensemble_scaled_ess(high))
    return {
        "rank_location_split_rhat": location, "rank_folded_split_rhat": folded,
        "rank_normalized_split_rhat": np.maximum(location, folded),
        "bulk_ess": bulk, "tail_ess": tail,
    }


def diagnose(context, coordinates, accepted):
    dimension = len(context.case.parameters)
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
                tau = np.full(dimension, math.nan)
        tau_rows.append(tau)
    tau_rows = np.asarray(tau_rows)
    length_over_tau = retained.shape[1] / np.nanmax(tau_rows, axis=0)
    flat = retained.reshape(-1, dimension)
    scale = np.std(flat, axis=0, ddof=1)
    midpoint = retained.shape[1] // 2
    first = retained[:, :midpoint].reshape(-1, dimension)
    second = retained[:, midpoint:].reshape(-1, dimension)
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / scale
    ensemble_medians = np.median(retained, axis=(1, 2))
    ensemble_spread = np.ptp(ensemble_medians, axis=0) / scale
    acceptance = float(np.mean(accepted / coordinates.shape[1]))
    if context.case.case_name == "pre_sto_only":
        bounds_list = [FNL_BOUNDS, B1_BOUNDS, B1_BOUNDS, PSHOT_BOUNDS, PSHOT_BOUNDS][:dimension]
    else:
        bounds_list = [FNL_BOUNDS, B1_BOUNDS, PSHOT_BOUNDS][:dimension]
    boundaries = {
        name: boundary_mass(flat[:, i], bounds_list[i])
        for i, name in enumerate(context.case.parameters)
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
        **diagnostics, "tau_by_ensemble": tau_rows,
        "retained_length_over_tau": length_over_tau,
        "half_chain_median_shift_sigma": half_shift,
        "ensemble_medians": ensemble_medians,
        "ensemble_median_spread_sigma": ensemble_spread,
        "mean_acceptance_fraction": acceptance,
        "boundary_mass": boundaries, "checks": checks,
        "pass": bool(all(checks.values())),
        "chain_layout": {
            "ensembles": ENSEMBLES, "walkers": WALKERS,
            "steps": int(coordinates.shape[1]), "burn": burn,
            "retained_samples": int(flat.shape[0]),
        },
    }, retained


def run_sampling(case_name: str, target_name: str) -> None:
    profiles = load_profiles()
    context = build_context(case_name, target_name, profiles)
    dimension = len(context.case.parameters)
    case_index = TARGETS.index(target_name) * len(CASES) + CASES.index(case_name)
    contract = {
        "schema": "quijote-z0p5-fixedp-3method-chain-v1",
        "created_utc": utc_now(), "case": case_name, "target": target_name,
        "truth_fNL": context.case.truth, "parameters": context.case.parameters,
        "p_fixed": P_FIXED,
        "bounds": {
            "fNL": FNL_BOUNDS, "b1": B1_BOUNDS, "b1_pre": B1_BOUNDS, "b1_post": B1_BOUNDS,
            "sn0_pre": PSHOT_BOUNDS, "sn0_post": PSHOT_BOUNDS, "Pshot_residual": PSHOT_BOUNDS,
        },
        "profile_sha256": sha256(PROFILE_JSON), "profile_npz_sha256": sha256(PROFILE_NPZ),
        "runner_sha256": sha256(Path(__file__)),
        "sampler": {
            "ensembles": ENSEMBLES, "walkers": WALKERS,
            "moves": "0.8 StretchMove(a=2)+0.2 DEMove",
            "burn_fraction": BURN_FRACTION, "seed": SEED + 100000 * case_index,
            "coordinates": "constant Hessian-whitened physical parameters",
        },
    }
    write_or_validate_contract(contract_path(case_name, target_name), contract)
    path = chain_path(case_name, target_name)
    target_steps = INITIAL_STEPS
    if path.is_file():
        existing = [
            int(emcee.backends.HDFBackend(str(path), name=f"ensemble_{e}", read_only=True).iteration)
            for e in range(ENSEMBLES) if group_exists(path, f"ensemble_{e}")
        ]
        target_steps = max(INITIAL_STEPS, max(existing, default=0))
    while True:
        sample_to(context, case_name, target_name, target_steps)
        coordinates, accepted = load_chain(case_name, target_name)
        diagnostic, retained = diagnose(context, coordinates, accepted)
        actual = coordinates.shape[1]
        print(
            f"{case_name}__{target_name} steps={actual} pass={diagnostic['pass']} "
            f"Rhat={np.max(diagnostic['rank_normalized_split_rhat']):.6f} "
            f"ESS={min(np.min(diagnostic['bulk_ess']),np.min(diagnostic['tail_ess'])):.1f}",
            flush=True,
        )
        if diagnostic["pass"] or actual >= MAX_STEPS:
            break
        target_steps = min(actual + INCREMENT, MAX_STEPS)
    flat = retained.reshape(-1, dimension)
    rng = np.random.default_rng(SEED + 900000 + case_index)
    count = min(POSTERIOR_DRAWS, flat.shape[0])
    selected = flat[rng.choice(flat.shape[0], size=count, replace=False)]
    logp = -0.5 * context.case.objective_many(selected)
    atomic_npz(posterior_path(case_name, target_name), {
        "posterior": selected, "log_probability": logp,
        "parameter_names": np.asarray(context.case.parameters),
        "target": context.case.target, "covariance": context.case.covariance,
    })
    atomic_json(task_path(case_name, target_name), {
        "schema": "quijote-z0p5-fixedp-3method-task-v1",
        "created_utc": utc_now(), "case": case_name, "target": target_name,
        "status": "complete" if diagnostic["pass"] else "nonconverged_at_max_steps",
        "diagnostics": diagnostic,
        "chain": artifact(path), "posterior": artifact(posterior_path(case_name, target_name)),
    })
    if not diagnostic["pass"]:
        raise RuntimeError(f"{case_name}__{target_name} did not converge")


def parameter_summary(values):
    q = np.quantile(np.asarray(values, dtype=float), (0.025, 0.16, 0.5, 0.84, 0.975))
    return {
        "q2p5": float(q[0]), "q16": float(q[1]), "median": float(q[2]),
        "q84": float(q[3]), "q97p5": float(q[4]),
        "half_width": float(0.5 * (q[3] - q[1])),
    }


def write_summary() -> None:
    profiles = load_profiles()
    rows = []
    combined = {}
    samples = {}
    for target_name in TARGETS:
        for case_name in CASES:
            task = json.loads(task_path(case_name, target_name).read_text(encoding="utf-8"))
            if task.get("status") != "complete" or not task["diagnostics"]["pass"]:
                raise RuntimeError(f"incomplete MCMC: {case_name}__{target_name}")
            arrays = load_npz(posterior_path(case_name, target_name))
            posterior = np.asarray(arrays["posterior"], dtype=float)
            if posterior.shape != (POSTERIOR_DRAWS, len(PARAMETERS_SINGLE if case_name != "pre_sto_only" else PARAMETERS_JOINT)):
                raise RuntimeError(f"posterior shape failed: {case_name}__{target_name}")
            name = f"{case_name}__{target_name}"
            combined[f"{name}__posterior"] = posterior
            combined[f"{name}__log_probability"] = arrays["log_probability"]
            samples[(case_name, target_name)] = posterior
            summaries = {
                pname: parameter_summary(posterior[:, i])
                for i, pname in enumerate(
                    PARAMETERS_JOINT if case_name == "pre_sto_only" else PARAMETERS_SINGLE
                )
            }
            truth = TRUTHS[target_name]
            summaries["fNL"]["truth_percentile"] = float(np.mean(posterior[:, 0] <= truth))
            summaries["fNL"]["truth_inside_68"] = bool(summaries["fNL"]["q16"] <= truth <= summaries["fNL"]["q84"])
            summaries["fNL"]["truth_inside_95"] = bool(summaries["fNL"]["q2p5"] <= truth <= summaries["fNL"]["q97p5"])
            # 独立 likelihood parity（抽样 101 点）。
            case = load_case(case_name, target_name)
            indices = np.linspace(0, posterior.shape[0] - 1, 101, dtype=int)
            theory = case.theory
            pts = posterior[indices]
            if case_name == "pre_sto_only":
                pre = channel_prediction(pts[:, 0], pts[:, 1], pts[:, 3], P_FIXED["pre"], theory)
                post = channel_prediction(pts[:, 0], pts[:, 2], pts[:, 4], P_FIXED["sto_only"], theory)
                pred = np.concatenate((pre, post), axis=1)
            else:
                pred = channel_prediction(pts[:, 0], pts[:, 1], pts[:, 2], P_FIXED[case_name], theory)
            resid = pred - case.target[None, :]
            white = np.linalg.solve(np.linalg.cholesky(case.covariance), resid.T).T
            indep_logp = -0.5 * np.sum(white**2, axis=1)
            if float(np.max(np.abs(np.asarray(arrays["log_probability"])[indices] - indep_logp))) > 1e-9:
                raise RuntimeError(f"independent logp parity failed: {name}")
            profile_row = profiles["rows"][name]
            rows.append({
                "case": case_name, "target": target_name, "truth_fNL": truth,
                **summaries,
                "profile_map_fNL": profile_row["map"]["fNL"],
                "profile_truth_delta_chi2": profile_row["truth_profile"]["delta_chi2"],
                "goodness_of_fit_p": profile_row["goodness_of_fit_p"],
                "diagnostics": task["diagnostics"],
            })
    comparison = {}
    for target_name in TARGETS:
        local = {row["case"]: row for row in rows if row["target"] == target_name}
        pre_width = local["pre"]["fNL"]["half_width"]
        comparison[target_name] = {
            case_name: {
                "half_width_over_pre": local[case_name]["fNL"]["half_width"] / pre_width,
                "information_over_pre": (pre_width / local[case_name]["fNL"]["half_width"]) ** 2,
                "median_minus_truth": local[case_name]["fNL"]["median"] - TRUTHS[target_name],
            }
            for case_name in CASES
        }
    atomic_npz(RESULT_NPZ, combined)
    payload = {
        "schema": "quijote-z0p5-fixedp-3method-results-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "cases": CASES, "targets": TARGETS,
            "p_fixed": P_FIXED, "p_fixed_rounded": P_FIXED_ROUNDED,
            "p_calibration_source": str(FREEP_RESULTS),
            "note": (
                "p 来自 fNL=100 固定校准；+100 行的 recovery 因此是同数据自洽检查，"
                "不是独立无偏检验（独立检验需拆分 realization 或其他 fNL 样本）"
            ),
        },
        "rows": rows, "comparison_to_pre": comparison,
        "inputs": {
            "five_matrix": artifact(FIVE_MATRIX), "joint_matrix": artifact(JOINT_MATRIX),
            "theory": artifact(P_THEORY), "freep_calibration": artifact(FREEP_RESULTS),
            "profiles": artifact(PROFILE_JSON), "runner": artifact(Path(__file__)),
        },
        "outputs": {"posteriors": artifact(RESULT_NPZ)},
    }
    atomic_json(RESULT_JSON, payload)
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case", "target", "truth_fNL", "median", "q16", "q84", "q2p5", "q97p5",
            "half_width", "truth_percentile", "truth_inside_68", "truth_inside_95",
            "profile_truth_delta_chi2", "goodness_of_fit_p", "half_width_over_pre",
            "information_over_pre", "max_rhat", "min_bulk_ess",
        ))
        writer.writeheader()
        for row in rows:
            comp = comparison[row["target"]][row["case"]]
            d = row["diagnostics"]
            writer.writerow({
                "case": row["case"], "target": row["target"], "truth_fNL": row["truth_fNL"],
                "median": row["fNL"]["median"], "q16": row["fNL"]["q16"],
                "q84": row["fNL"]["q84"], "q2p5": row["fNL"]["q2p5"],
                "q97p5": row["fNL"]["q97p5"], "half_width": row["fNL"]["half_width"],
                "truth_percentile": row["fNL"]["truth_percentile"],
                "truth_inside_68": row["fNL"]["truth_inside_68"],
                "truth_inside_95": row["fNL"]["truth_inside_95"],
                "profile_truth_delta_chi2": row["profile_truth_delta_chi2"],
                "goodness_of_fit_p": row["goodness_of_fit_p"],
                "half_width_over_pre": comp["half_width_over_pre"],
                "information_over_pre": comp["information_over_pre"],
                "max_rhat": float(np.max(d["rank_normalized_split_rhat"])),
                "min_bulk_ess": float(np.min(d["bulk_ess"])),
            })
    lines = [
        "# p 固定校准值的三方法 P0-only 主结果", "",
        "Status: `complete`。p_pre=1.1789、p_sto=1.3395 固定（fNL=100 校准中位数）。",
        "+100 行为同数据自洽检查（校准用了同一 +100 样本），非独立无偏检验。", "",
        "| target | method | fNL median [16,84] | half-width | width/pre | truth in 68/95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        f = row["fNL"]; comp = comparison[row["target"]][row["case"]]
        lines.append(
            f"| {row['target']} | {CASE_LABELS[row['case']]} | "
            f"{f['median']:.3f} [{f['q16']:.3f},{f['q84']:.3f}] | {f['half_width']:.3f} | "
            f"{comp['half_width_over_pre']:.4f} | {f['truth_inside_68']}/{f['truth_inside_95']} |"
        )
    lines.extend([
        "", "Covariance：单通道 12-bin 各自 Fid500（/42.875/Hartlap12）；joint 24-bin 全 cross（/42.875/Hartlap24）。",
        "", f"结果 JSON：`{RESULT_JSON}`。", f"posterior NPZ：`{RESULT_NPZ}`。", "",
    ])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps(jsonable({
        "status": "complete", "result": str(RESULT_JSON),
        "rows": [
            {"case": r["case"], "target": r["target"],
             "median": r["fNL"]["median"], "half_width": r["fNL"]["half_width"],
             "in68": r["fNL"]["truth_inside_68"], "in95": r["fNL"]["truth_inside_95"]}
            for r in rows
        ],
    }), indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profiles", "sample", "summarize", "all"), required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--target", choices=TARGETS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in ("profiles", "all"):
        prepare_profiles()
    if args.stage == "sample":
        if args.case is None or args.target is None:
            raise ValueError("--stage sample requires --case and --target")
        run_sampling(args.case, args.target)
    elif args.stage == "all":
        for target_name in TARGETS:
            for case_name in CASES:
                run_sampling(case_name, target_name)
    if args.stage in ("summarize", "all"):
        write_summary()


if __name__ == "__main__":
    main()
