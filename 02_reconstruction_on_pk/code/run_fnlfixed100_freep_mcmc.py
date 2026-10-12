#!/usr/bin/env python3
"""fNL 固定为输入真值 +100、p 自由的 P0 校准 MCMC（pre 与 STO-only）。

代码大纲
========
1. 读取五方法矩阵与 lattice 理论（sha 链路校验）。
2. 模型：P = b1^2 PL + 2 fNL b1 bphi PL/M + fNL^2 bphi^2 Pphi + Pshot，
   bphi = 2 delta_c (b1 - p)；**fNL 固定 = 100（输入真值）**，自由参数
   (p, b1, Pshot_residual)。这是校准测量不是盲恢复，无 recovery gate。
3. 每方法用自身 fNL100 matched100 target 与 Fid500 单盒 covariance
   （/42.875/Hartlap12），复用 iminuit profile + HESSE 白化 + emcee
   自动延长至收敛的现役机器。
4. 输出链、posterior、结果 JSON/CSV 与中文报告；汇总内置独立复核
   （分位数 parity、独立 likelihood 重算）。
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
TAG = "quijote_halo_z0p5_pre_sto_only_fnlfixed100_freep_calib_v1"
ANALYSIS = PART / "work" / TAG
MATRIX = PART / "inputs/five_method_p0_matrix.npz"
MATRIX_AUDIT = PART / "results/five_method_p0_matrix_audit.json"
FIVE_RESULTS = PART / "results/five_method_p0only_mcmc_results.json"
P_THEORY = PART / "inputs/p0_finite_local_png_lattice.npz"
PROFILE_JSON = ANALYSIS / "FREEP_PROFILES.json"
PROFILE_NPZ = ANALYSIS / "FREEP_PROFILE_ARRAYS.npz"
RESULT_JSON = ANALYSIS / "FREEP_RESULTS.json"
RESULT_CSV = ANALYSIS / "FREEP_RESULTS.csv"
RESULT_NPZ = ANALYSIS / "FREEP_POSTERIORS.npz"
REPORT = ANALYSIS / "FREEP_REPORT.md"
CHAINS = ANALYSIS / "chains"
TASKS = ANALYSIS / "tasks"
POSTERIORS = ANALYSIS / "posteriors"

METHODS = ("pre", "standard", "marisa", "sto_only", "sto_marisa")
FNLFIX = 100.0
PARAMETERS = ("p", "b1", "Pshot_residual")
P_BOUNDS = (0.0, 3.0)
B1_BOUNDS = (0.1, 6.0)
PSHOT_BOUNDS = (-1.0e5, 1.0e5)
VOLUME = 42.875
HARTLAP = (500.0 - 12.0 - 2.0) / 499.0
DELTA_C = 1.686
ENSEMBLES, WALKERS = 4, 32
INITIAL_STEPS, INCREMENT, MAX_STEPS = 6000, 2000, 20000
BURN_FRACTION, POSTERIOR_DRAWS = 0.5, 40000
SEED = 20261120


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


@dataclass
class Case:
    """fNL 固定、(p, b1, Pshot) 自由的三参数校准 likelihood。"""

    method: str
    target: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    theory: dict[str, np.ndarray]

    def predictions(self, values: np.ndarray) -> np.ndarray:
        physical = np.asarray(values, dtype=float)
        scalar = physical.ndim == 1
        rows = physical[None, :] if scalar else physical
        p, b1, shot = rows[:, 0], rows[:, 1], rows[:, 2]
        bp = np.asarray(bphi(b1, p))
        prediction = (
            b1[:, None] ** 2 * self.theory["PL_lattice"][None, :]
            + 2.0 * FNLFIX * (b1 * bp)[:, None] * self.theory["PL_over_M_lattice"][None, :]
            + FNLFIX**2 * bp[:, None] ** 2 * self.theory["Pphi_lattice"][None, :]
            + shot[:, None]
        )
        return prediction[0] if scalar else prediction

    def objective_many(self, values: np.ndarray) -> np.ndarray:
        predictions = self.predictions(values)
        rows = predictions[None, :] if predictions.ndim == 1 else predictions
        residual = rows - self.target[None, :]
        white = np.linalg.solve(self.cholesky, residual.T).T
        return np.sum(white**2, axis=1)

    def objective(self, values: np.ndarray) -> float:
        return float(self.objective_many(np.asarray(values, dtype=float))[0])


def load_case(method: str) -> Case:
    audit = json.loads(MATRIX_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or sha256(MATRIX) != audit["output"]["sha256"]:
        raise RuntimeError("five-method matrix audit/hash failed")
    arrays = load_npz(MATRIX)
    theory = load_npz(P_THEORY)
    target = np.asarray(arrays[f"{method}__target_fNL100_matched100"], dtype=float)
    covariance = np.asarray(
        arrays[f"{method}__covariance_fNL0_fid500_one_box"], dtype=float
    ) / (VOLUME * HARTLAP)
    return Case(method, target, covariance, np.linalg.cholesky(covariance), theory)


def generalized_start(case: Case) -> tuple[float, float]:
    covariance_one = case.covariance * VOLUME * HARTLAP
    design = np.column_stack((case.theory["PL_lattice"], np.ones(12)))
    solved_design = np.linalg.solve(covariance_one, design)
    solved_target = np.linalg.solve(covariance_one, case.target)
    coefficient = np.linalg.solve(design.T @ solved_design, design.T @ solved_target)
    b1_start = float(np.clip(math.sqrt(max(float(coefficient[0]), 0.01)), 0.101, 5.999))
    shot_start = float(np.clip(coefficient[1], PSHOT_BOUNDS[0] + 1, PSHOT_BOUNDS[1] - 1))
    return b1_start, shot_start


def profile_case(case: Case) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    b1_start, shot_start = generalized_start(case)

    def objective(p: float, b1: float, Pshot_residual: float) -> float:
        return case.objective(np.asarray((p, b1, Pshot_residual)))

    fits = []
    for p_start in (1.12, 1.0, 1.5, 2.0):
        fit = Minuit(objective, p=p_start, b1=b1_start, Pshot_residual=shot_start)
        fit.errordef = Minuit.LEAST_SQUARES
        fit.limits["p"] = P_BOUNDS
        fit.limits["b1"] = B1_BOUNDS
        fit.limits["Pshot_residual"] = PSHOT_BOUNDS
        fit.errors["p"] = 0.1
        fit.errors["b1"] = 0.02
        fit.errors["Pshot_residual"] = 100.0
        fit.strategy, fit.tol = 1, 1e-5
        fit.migrad(ncall=120000, iterate=8, use_simplex=True)
        fit.hesse()
        fits.append(fit)
    valid = [fit for fit in fits if fit.valid and math.isfinite(float(fit.fval))]
    best = min(valid or fits, key=lambda fit: float(fit.fval))
    if not best.valid or best.covariance is None:
        raise RuntimeError(f"{case.method} MAP/HESSE failed")
    best.minos("p", ncall=160000)
    error = best.merrors["p"]
    mapping = {name: float(best.values[name]) for name in PARAMETERS}
    row = {
        "method": case.method,
        "map": mapping, "objective": float(best.fval),
        "data_chi2": float(best.fval), "dof": 9,
        "goodness_of_fit_p": float(chi2.sf(float(best.fval), 9)),
        "optimizer_valid": bool(best.valid),
        "hesse_covariance": np.asarray(best.covariance, dtype=float),
        "minos_p": {
            "lower": mapping["p"] + float(error.lower),
            "upper": mapping["p"] + float(error.upper),
            "lower_error": float(error.lower), "upper_error": float(error.upper),
            "valid": bool(error.is_valid),
        },
    }
    return row, {"target": case.target, "covariance": case.covariance}


def prepare_profiles() -> None:
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        case = load_case(method)
        row, local = profile_case(case)
        rows[method] = row
        for item, value in local.items():
            arrays[f"{method}__{item}"] = value
        print(
            f"profile {method}: p={row['map']['p']:.4f} "
            f"{row['minos_p']['lower_error']:+.4f}/{row['minos_p']['upper_error']:+.4f} "
            f"b1={row['map']['b1']:.5f} chi2/dof={row['data_chi2']:.2f}/9 "
            f"p_gof={row['goodness_of_fit_p']:.4g}",
            flush=True,
        )
    atomic_npz(PROFILE_NPZ, arrays)
    atomic_json(PROFILE_JSON, {
        "schema": "quijote-z0p5-fnlfixed100-freep-profiles-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "methods": METHODS, "parameters": PARAMETERS, "fNL_fixed": FNLFIX,
            "note": "校准测量：fNL 固定为输入真值，无 recovery gate",
        },
        "rows": rows,
        "inputs": {
            "matrix": artifact(MATRIX), "matrix_audit": artifact(MATRIX_AUDIT),
            "theory": artifact(P_THEORY), "runner": artifact(Path(__file__)),
        },
        "arrays": artifact(PROFILE_NPZ),
    })


def load_profiles() -> dict[str, Any]:
    payload = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or sha256(PROFILE_NPZ) != payload["arrays"]["sha256"]:
        raise RuntimeError("profile status/hash failed")
    for name, path in (
        ("matrix", MATRIX), ("matrix_audit", MATRIX_AUDIT),
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
        support = (
            (physical[:, 0] > P_BOUNDS[0]) & (physical[:, 0] < P_BOUNDS[1])
            & (physical[:, 1] > B1_BOUNDS[0]) & (physical[:, 1] < B1_BOUNDS[1])
            & (physical[:, 2] > PSHOT_BOUNDS[0]) & (physical[:, 2] < PSHOT_BOUNDS[1])
            & np.all(np.isfinite(physical), axis=1)
        )
        output = np.full(rows.shape[0], -math.inf)
        if np.any(support):
            output[support] = -0.5 * self.case.objective_many(physical[support])
        return float(output[0]) if scalar else output


def build_context(method: str, profiles: dict[str, Any]) -> SamplerContext:
    case = load_case(method)
    row = profiles["rows"][method]
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
        raise RuntimeError(f"profile/logp parity failed: {method}: {parity}")
    return context


_ACTIVE: SamplerContext | None = None


def active_log_probability(values):
    if _ACTIVE is None:
        raise RuntimeError("no active sampler context")
    return _ACTIVE.log_probability(values)


def chain_path(method: str) -> Path:
    return CHAINS / f"{method}.h5"


def contract_path(method: str) -> Path:
    return CHAINS / f"{method}.contract.json"


def task_path(method: str) -> Path:
    return TASKS / f"{method}.json"


def posterior_path(method: str) -> Path:
    return POSTERIORS / f"{method}.npz"


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
    accepted = []
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


def sample_to(context: SamplerContext, method: str, steps: int) -> None:
    path = chain_path(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    global _ACTIVE
    _ACTIVE = context
    try:
        for ensemble in range(ENSEMBLES):
            group = f"ensemble_{ensemble}"
            exists = group_exists(path, group)
            backend = emcee.backends.HDFBackend(str(path), name=group)
            seed = SEED + 100000 * METHODS.index(method) + 1009 * ensemble
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


def load_chain(method: str) -> tuple[np.ndarray, np.ndarray]:
    chains, accepted = [], []
    for ensemble in range(ENSEMBLES):
        backend = emcee.backends.HDFBackend(
            str(chain_path(method)), name=f"ensemble_{ensemble}", read_only=True
        )
        chains.append(np.asarray(backend.get_chain(), dtype=float))
        accepted.append(np.asarray(backend.accepted, dtype=float))
    if len({item.shape for item in chains}) != 1:
        raise RuntimeError("ensemble shape drift")
    return np.stack(chains), np.stack(accepted)


def boundary_mass(values: np.ndarray, bounds: tuple[float, float]) -> float:
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
        "p": boundary_mass(flat[:, 0], P_BOUNDS),
        "b1": boundary_mass(flat[:, 1], B1_BOUNDS),
        "Pshot_residual": boundary_mass(flat[:, 2], PSHOT_BOUNDS),
    }
    checks = {
        "rhat": bool(np.all(diagnostics["rank_normalized_split_rhat"] < 1.01)),
        "p_ess": bool(diagnostics["bulk_ess"][0] > 2000 and diagnostics["tail_ess"][0] > 2000),
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


def run_sampling(method: str) -> None:
    profiles = load_profiles()
    context = build_context(method, profiles)
    contract = {
        "schema": "quijote-z0p5-fnlfixed100-freep-chain-v1",
        "created_utc": utc_now(), "method": method,
        "fNL_fixed": FNLFIX, "parameters": PARAMETERS,
        "bounds": {"p": P_BOUNDS, "b1": B1_BOUNDS, "Pshot_residual": PSHOT_BOUNDS},
        "profile_sha256": sha256(PROFILE_JSON), "profile_npz_sha256": sha256(PROFILE_NPZ),
        "runner_sha256": sha256(Path(__file__)),
        "sampler": {
            "ensembles": ENSEMBLES, "walkers": WALKERS,
            "moves": "0.8 StretchMove(a=2)+0.2 DEMove",
            "burn_fraction": BURN_FRACTION, "seed": SEED + 100000 * METHODS.index(method),
            "coordinates": "constant Hessian-whitened physical parameters",
        },
    }
    write_or_validate_contract(contract_path(method), contract)
    existing = []
    path = chain_path(method)
    if path.is_file():
        for ensemble in range(ENSEMBLES):
            if group_exists(path, f"ensemble_{ensemble}"):
                backend = emcee.backends.HDFBackend(str(path), name=f"ensemble_{ensemble}", read_only=True)
                existing.append(int(backend.iteration))
    target_steps = max(INITIAL_STEPS, max(existing, default=0))
    while True:
        sample_to(context, method, target_steps)
        coordinates, accepted = load_chain(method)
        diagnostic, retained = diagnose(context, coordinates, accepted)
        actual = coordinates.shape[1]
        print(
            f"{method} steps={actual} pass={diagnostic['pass']} "
            f"Rhat={np.max(diagnostic['rank_normalized_split_rhat']):.6f} "
            f"ESS={min(np.min(diagnostic['bulk_ess']),np.min(diagnostic['tail_ess'])):.1f}",
            flush=True,
        )
        if diagnostic["pass"] or actual >= MAX_STEPS:
            break
        target_steps = min(actual + INCREMENT, MAX_STEPS)
    flat = retained.reshape(-1, 3)
    rng = np.random.default_rng(SEED + 900000 + METHODS.index(method))
    count = min(POSTERIOR_DRAWS, flat.shape[0])
    selected = flat[rng.choice(flat.shape[0], size=count, replace=False)]
    logp = -0.5 * context.case.objective_many(selected)
    atomic_npz(posterior_path(method), {
        "posterior": selected, "log_probability": logp,
        "parameter_names": np.asarray(PARAMETERS),
        "target": context.case.target, "covariance": context.case.covariance,
    })
    atomic_json(task_path(method), {
        "schema": "quijote-z0p5-fnlfixed100-freep-task-v1",
        "created_utc": utc_now(), "method": method,
        "status": "complete" if diagnostic["pass"] else "nonconverged_at_max_steps",
        "diagnostics": diagnostic,
        "chain": artifact(path), "posterior": artifact(posterior_path(method)),
    })
    if not diagnostic["pass"]:
        raise RuntimeError(f"{method} did not converge")


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
    combined = {"parameter_names": np.asarray(PARAMETERS)}
    samples = {}
    for method in METHODS:
        task = json.loads(task_path(method).read_text(encoding="utf-8"))
        if task.get("status") != "complete" or not task["diagnostics"]["pass"]:
            raise RuntimeError(f"incomplete MCMC: {method}")
        arrays = load_npz(posterior_path(method))
        posterior = np.asarray(arrays["posterior"], dtype=float)
        if posterior.shape != (POSTERIOR_DRAWS, 3) or not np.all(np.isfinite(posterior)):
            raise RuntimeError(f"posterior shape/finite failed: {method}")
        samples[method] = posterior
        combined[f"{method}__posterior"] = posterior
        combined[f"{method}__log_probability"] = arrays["log_probability"]
        summaries = {name: parameter_summary(posterior[:, i]) for i, name in enumerate(PARAMETERS)}
        # 独立复核：用独立实现重算 logp（抽样 101 点）。
        theory = load_npz(P_THEORY)
        case = load_case(method)
        indices = np.linspace(0, posterior.shape[0] - 1, 101, dtype=int)
        p_, b1_, s_ = posterior[indices].T
        bp = 2.0 * DELTA_C * (b1_ - p_)
        pred = (
            b1_[:, None] ** 2 * theory["PL_lattice"][None, :]
            + 2.0 * FNLFIX * (b1_ * bp)[:, None] * theory["PL_over_M_lattice"][None, :]
            + FNLFIX**2 * bp[:, None] ** 2 * theory["Pphi_lattice"][None, :]
            + s_[:, None]
        )
        resid = pred - case.target[None, :]
        white = np.linalg.solve(np.linalg.cholesky(case.covariance), resid.T).T
        indep_logp = -0.5 * np.sum(white**2, axis=1)
        saved_logp = np.asarray(arrays["log_probability"], dtype=float)
        if float(np.max(np.abs(saved_logp[indices] - indep_logp))) > 1e-9:
            raise RuntimeError(f"independent logp parity failed: {method}")
        rows.append({
            "method": method, **summaries,
            "profile_map_p": profiles["rows"][method]["map"]["p"],
            "goodness_of_fit_p": profiles["rows"][method]["goodness_of_fit_p"],
            "diagnostics": task["diagnostics"],
        })
    atomic_npz(RESULT_NPZ, combined)
    payload = {
        "schema": "quijote-z0p5-fnlfixed100-freep-results-v1",
        "created_utc": utc_now(), "status": "complete",
        "scope": {
            "methods": METHODS, "parameters": PARAMETERS, "fNL_fixed": FNLFIX,
            "definition": "校准测量：fNL 固定为输入真值 +100，p/b1/Pshot 自由；"
                          "无 recovery gate；协方差沿用五方法约定",
            "prior_p": P_BOUNDS,
        },
        "rows": rows,
        "inputs": {
            "matrix": artifact(MATRIX), "matrix_audit": artifact(MATRIX_AUDIT),
            "theory": artifact(P_THEORY), "five_results": artifact(FIVE_RESULTS),
            "profiles": artifact(PROFILE_JSON), "runner": artifact(Path(__file__)),
        },
        "outputs": {"posteriors": artifact(RESULT_NPZ)},
    }
    atomic_json(RESULT_JSON, payload)
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method", "p_median", "p_q16", "p_q84", "p_half_width",
            "b1_median", "b1_q16", "b1_q84",
            "Pshot_median", "goodness_of_fit_p", "max_rhat", "min_bulk_ess",
        ))
        writer.writeheader()
        for row in rows:
            d = row["diagnostics"]
            writer.writerow({
                "method": row["method"],
                "p_median": row["p"]["median"], "p_q16": row["p"]["q16"],
                "p_q84": row["p"]["q84"], "p_half_width": row["p"]["half_width"],
                "b1_median": row["b1"]["median"], "b1_q16": row["b1"]["q16"],
                "b1_q84": row["b1"]["q84"],
                "Pshot_median": row["Pshot_residual"]["median"],
                "goodness_of_fit_p": row["goodness_of_fit_p"],
                "max_rhat": float(np.max(d["rank_normalized_split_rhat"])),
                "min_bulk_ess": float(np.min(d["bulk_ess"])),
            })
    lines = [
        "# fNL 固定 100、p 自由的 P0 校准（pre / STO-only）", "",
        "Status: `complete`。`(p, b1, Pshot_residual)` 三参数，fNL 固定为输入真值 +100。",
        "这是校准测量，无 recovery gate。", "",
        "| method | p median [16,84] | b1 [16,84] | GoF p |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['p']['median']:.4f} "
            f"[{row['p']['q16']:.4f},{row['p']['q84']:.4f}] | "
            f"{row['b1']['median']:.5f} [{row['b1']['q16']:.5f},{row['b1']['q84']:.5f}] | "
            f"{row['goodness_of_fit_p']:.4g} |"
        )
    lines.extend(["", f"结果 JSON：`{RESULT_JSON}`。", f"posterior NPZ：`{RESULT_NPZ}`。", ""])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps(jsonable({
        "status": "complete", "result": str(RESULT_JSON),
        "rows": [
            {"method": row["method"], "p": row["p"]["median"],
             "p_q16": row["p"]["q16"], "p_q84": row["p"]["q84"]}
            for row in rows
        ],
    }), indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profiles", "sample", "summarize", "all"), required=True)
    parser.add_argument("--method", choices=METHODS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in ("profiles", "all"):
        prepare_profiles()
    if args.stage == "sample":
        if args.method is None:
            raise ValueError("--stage sample requires --method")
        run_sampling(args.method)
    elif args.stage == "all":
        for method in METHODS:
            run_sampling(method)
    if args.stage in ("summarize", "all"):
        write_summary()


if __name__ == "__main__":
    main()
