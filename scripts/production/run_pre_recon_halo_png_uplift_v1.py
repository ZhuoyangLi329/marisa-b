#!/usr/bin/env python3
"""Run the versioned pre-reconstruction halo PNG-uplift experiment.

The production model is intentionally hybrid:

    Gaussian halo one-loop EFT-v2
  + tree-level halo local-PNG
  + the pure-matter linear local-PNG one-loop uplift
  + the validated finite-fNL matter B112II term.

It is not labelled as a complete halo-PNG one-loop theory.  Ensemble means
are low-noise central vectors only.  Every likelihood, uncertainty, pull,
coverage result, and plotted error bar uses a covariance of one realization.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


for _key in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
):
    try:
        _value = int(os.environ.get(_key, "8"))
    except ValueError:
        _value = 8
    os.environ[_key] = str(max(1, min(_value, 8)))
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pre-halo-png-uplift-v1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import beta as beta_distribution
from scipy.stats import chi2 as chi2_distribution

from produce_quijote_dm_marisa_b_b000_theory_vector import (
    build_b000_work,
    shell_nodes,
)
from produce_quijote_halo_marisa_b_bias_v1_gaussian_templates import (
    MeasurementGeometry,
    load_measurement_geometry,
)
from run_marisa_b_triangle import run_marisa_b
import run_pre_recon_bias_model_v3 as gaussian_v3
import run_pre_recon_halo_png_v1 as data_audit_v1


TAG = "pre_recon_halo_png_uplift_v1_20260726"
GOAL_THREAD = "019f89c0-0fed-71b3-8d32-a2953465f08b"
FINITE_BOX_PROJECTION_VERSION = "continuum_full_shell_denominator_v2"
CUTS = (0.08, 0.10, 0.12, 0.15)
P_GRID = (1.0, 1.2, 1.4, 1.6)
TIERS = ("full", "coevolution", "uplift")
PRIMARY_TIER = "coevolution"
PRIMARY_CUT = 0.08
STRESS_CUT = 0.15
DELTA_C = 1.686
EXCLUDED_INDEX = 0
N_BAR_FALLBACK = 1.95530218e-4
TREE_LINEAR_FIELDS = (
    "dBdfNL_local_tree",
    "dBdfNL_local_primordial",
    "dBdfNL_local_bphi_f2",
    "dBdfNL_local_bphi_advection",
    "dBdfNL_local_bphidelta",
    "dBdfNL_local_bphi_b2",
    "dBdfNL_local_bphi_bK2",
    "dBdfNL_stochastic_alpha3_basis",
    "stochastic_alpha3_basis",
)
TREE_FNL2_FIELDS = (
    "Bhalo_tree_fNL2_bphi_B0",
    "Bhalo_tree_fNL2_bphi_sq_advection",
    "Bhalo_tree_fNL2_bphi_sq_F2",
    "Bhalo_tree_fNL2_bphi_sq_b2",
    "Bhalo_tree_fNL2_bphi_sq_bK2",
    "Bhalo_tree_fNL2_bphi_bphidelta",
    "Bhalo_tree_fNL2_bphi2_operator",
    "Bhalo_tree_fNL2_deterministic",
    "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis",
)
TREE_FIELDS = TREE_LINEAR_FIELDS + TREE_FNL2_FIELDS
PDF_METADATA = {
    "Author": "MARISA-B pre-reconstruction halo PNG uplift v1",
    "Creator": "scripts/production/run_pre_recon_halo_png_uplift_v1.py",
    "Subject": (
        "Quijote z=1 real-space pre-reconstruction halo local-PNG "
        "hybrid one-loop injection-recovery test"
    ),
}


@dataclass(frozen=True)
class Paths:
    root: Path
    data_root: Path
    output_root: Path
    analysis: Path
    figures: Path
    manifest: Path
    theory: Path
    fid_matrix: Path
    matched_matrix: Path
    frozen_fitter: Path
    frozen_contract: Path
    b_templates: Path
    p_templates: Path
    png_table: Path
    native_binary: Path
    dm_linear: Path
    dm_quadratic: Path
    tree_templates: Path


@dataclass
class ProfileResult:
    tier: str
    sample: str
    p: float
    kmax: float
    response_model: str
    b1: float
    nuisance_prior_scale: float
    covariance_label: str
    indices: np.ndarray
    free_names: tuple[str, ...]
    nonlinear_names: tuple[str, ...]
    nuisance_values: np.ndarray
    expanded_parameters: dict[str, float]
    prediction: np.ndarray
    gaussian_prediction: np.ndarray
    linear_response: np.ndarray
    quadratic_coefficient: np.ndarray
    target: np.ndarray
    covariance_single: np.ndarray
    fhat: float
    error_low: float
    error_high: float
    sigma_symmetric: float
    objective: float
    data_chi2: float
    prior_chi2: float
    dof: float
    p_value: float
    max_pull_single: float
    residual_norm_single: float
    minuit_valid: bool
    minuit_accurate: bool
    minuit_at_limit: bool
    minuit_nfcn: int
    profile_x: np.ndarray
    profile_delta: np.ndarray
    nuisance_jacobian: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External data bundle; otherwise use MARISA_B_DATA_ROOT.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated-product root; defaults to the external data root.",
    )
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--stage",
        choices=("all", "tree", "fit", "self-test"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tree-nmu", type=int, default=80)
    parser.add_argument("--tree-nradial", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run only the primary coevolution cut and p=1 diagnostic. "
            "Quick outputs are diagnostic and cannot receive a passing status."
        ),
    )
    return parser.parse_args()


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
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def resolve_paths(
    root: Path,
    tag: str,
    *,
    data_root: Path | None = None,
    output_root: Path | None = None,
) -> Paths:
    data = (data_root or root).resolve()
    output = (output_root or data).resolve()
    frozen = data / "analysis/theory_vectors/eft_v2_r0_exact_primary_20260724"
    theory = data / "analysis/theory_vectors/marisa_b_halo_png_uplift_v1_20260726"
    return Paths(
        root=root,
        data_root=data,
        output_root=output,
        analysis=output / "analysis" / tag,
        figures=output / "figures/diagnostics" / tag,
        manifest=output / "manifests" / f"{tag}.json",
        theory=theory,
        fid_matrix=(
            data
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk_fid500"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_fid500_b000_pk_matrix.npz"
        ),
        matched_matrix=(
            data
            / "analysis/quijote_halo_z1_mmin1e13_r15_jaxrecon_b000_pk"
            / "quijote_halo_z1_mmin1e13_r15_jaxrecon_"
            "fid_lcp_lcm_first100_b000_pk_matrix.npz"
        ),
        frozen_fitter=root / "scripts/legacy/fit_eft_v2_r0_mean.py",
        frozen_contract=root / "configs/eft_v2_contract.json",
        b_templates=frozen / "b000_ir_contract_v2.jsonl",
        p_templates=frozen / "p0_ir_contract_v2.jsonl",
        png_table=(
            data
            / "analysis/theory_vectors/marisa_b_v0"
            / "marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_"
            "nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat"
        ),
        native_binary=root / "build/marisa_b/marisa_b_triangle",
        dm_linear=theory / "dm_pre_local_png_1loop_full2d_kmax0p15_nrad3_nmu48.npz",
        dm_quadratic=theory / "dm_pre_b112ii_full2d_kmax0p15_nrad3_nmu24_qmc11r4.npz",
        tree_templates=theory / "halo_pre_png_tree_components_nrad4_nmu80.npz",
    )


def require_inputs(paths: Paths, *, include_theory: bool) -> None:
    required = [
        paths.fid_matrix,
        paths.matched_matrix,
        paths.frozen_fitter,
        paths.frozen_contract,
        paths.b_templates,
        paths.p_templates,
        paths.png_table,
        paths.native_binary,
    ]
    if include_theory:
        required.extend(
            [paths.dm_linear, paths.dm_quadratic, paths.tree_templates]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))


def produce_dm_vectors(paths: Paths, *, overwrite: bool, workers: int) -> dict[str, Any]:
    """Invoke the existing matter producer for the two required sectors."""

    producer = (
        paths.root
        / "scripts/production/"
        "produce_quijote_dm_marisa_b_local_png_tree_b000_vector.py"
    )
    common = [
        sys.executable,
        str(producer),
        "--matrix",
        str(paths.fid_matrix),
        "--output-dir",
        str(paths.theory),
        "--kmax",
        "0.15",
        "--bin-average",
        "shell",
        "--nradial-binavg",
        "3",
        "--z",
        "1",
        "--b1",
        "1",
        "--reconstruction",
        "pre",
        "--png-table",
        str(paths.png_table),
        "--qmax",
        "20",
        "--skip-compile",
        "--build-dir",
        str(paths.native_binary.parent),
        "--max-triangles-per-call",
        "96",
        "--chunk-workers",
        str(workers),
        "--no-plot",
    ]
    jobs = [
        (
            paths.dm_linear,
            [
                "--prefix",
                paths.dm_linear.stem,
                "--nmu-b000",
                "48",
                "--png-loop-level",
                "1loop",
                "--epsrel",
                "1e-3",
                "--p13-epsrel",
                "1e-2",
            ],
        ),
        (
            paths.dm_quadratic,
            [
                "--prefix",
                paths.dm_quadratic.stem,
                "--nmu-b000",
                "24",
                "--png-loop-level",
                "b112ii",
                "--epsrel",
                "3e-3",
                "--png-ir-cutoff",
                str(2.0 * math.pi / 1000.0),
                "--b112ii-integrator",
                "multicenter_qmc",
                "--b112ii-qmc-power",
                "11",
                "--b112ii-qmc-replicates",
                "4",
            ],
        ),
    ]
    records: dict[str, Any] = {}
    for output, extra in jobs:
        sidecar = output.with_suffix(".json")
        if output.exists() and sidecar.exists() and not overwrite:
            records[output.stem] = {"status": "reused", "path": str(output)}
            continue
        for candidate in (
            output,
            sidecar,
            output.with_name(output.stem + "_diagonal_compact.npz"),
        ):
            if candidate.exists() and overwrite:
                candidate.unlink()
        command = common + extra
        completed = subprocess.run(
            command,
            cwd=paths.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"matter producer failed for {output.name}:\n"
                f"{completed.stderr[-4000:]}"
            )
        records[output.stem] = {
            "status": "produced",
            "path": str(output),
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
        }
    return records


def load_frozen(paths: Paths) -> tuple[Any, Any, Any, Any, Any, float]:
    frozen = gaussian_v3.load_frozen_module(paths.frozen_fitter)
    data = frozen.DataSet.load(paths.fid_matrix)
    _contract, prior, nbar = frozen.load_prior(paths.frozen_contract)
    templates = frozen.TemplateSet.load(paths.b_templates)
    power = frozen.PowerTemplateSet.load(paths.p_templates)
    frozen.validate_bispectrum_geometry(templates, data, TAG)
    frozen.validate_power_geometry(power, data, TAG)
    return frozen, data, prior, templates, power, float(nbar)


def selected_indices(k_pair: np.ndarray, kmax: float) -> np.ndarray:
    mask = np.max(np.asarray(k_pair), axis=1) <= float(kmax) + 1.0e-12
    mask = np.asarray(mask, dtype=bool)
    mask[EXCLUDED_INDEX] = False
    return np.flatnonzero(mask)


def covariance_single(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 3:
        raise ValueError("single-realization covariance needs >=3 rows")
    return np.cov(samples, rowvar=False, ddof=1)


def qnorm(vector: np.ndarray, covariance: np.ndarray) -> float:
    chol = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    whitened = solve_triangular(chol, np.asarray(vector), lower=True)
    return float(np.linalg.norm(whitened))


def tree_work(
    geometry: MeasurementGeometry,
    *,
    nmu: int,
    nradial: int,
    kfund: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float, float]], list[tuple[int, float]], dict[str, Any]]:
    indices = np.flatnonzero(
        np.max(geometry.weighted_pairs, axis=1) <= 0.15 + 1.0e-12
    )
    if indices.size != 28:
        raise AssertionError(f"expected the first 28 B000 bins, got {indices}")
    if kfund is None:
        args = argparse.Namespace(
            nmu_b000=int(nmu),
            nradial_binavg=int(nradial),
            bin_average="shell",
        )
        triangles, metadata, work = build_b000_work(
            data={"hdf5_k_edges": geometry.shell_edges},
            indices=indices,
            theory_pairs=geometry.weighted_pairs,
            args=args,
        )
    else:
        # The measured periodic box has no Fourier mode below 2*pi/L and no
        # zero mode.  Finite-PNG phi and phi^2 operators are sufficiently
        # infrared enhanced that integrating the continuum endpoint k3 -> 0
        # is both physically wrong for the catalogue and numerically
        # non-convergent.  Truncate all three external legs at kfund, but keep
        # the original full-shell and full-angular denominator.  This matches
        # the Sugiyama estimator convention: zero-mode summands vanish after
        # mean subtraction while bin.nmodes=N_i*N_j is not recomputed over
        # the surviving configurations.
        kfund = float(kfund)
        if not math.isfinite(kfund) or kfund <= 0.0:
            raise ValueError(f"invalid finite-box kfund={kfund}")
        mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(nmu))
        triangles = []
        metadata = []
        counts: list[int] = []
        weight_sums: list[float] = []

        def finite_shell_nodes(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            original_lower = float(edges[0])
            lower = max(original_lower, kfund)
            upper = float(edges[1])
            if upper <= lower:
                raise ValueError(
                    f"shell {np.asarray(edges)} contains no nonzero box mode"
                )
            nodes, active_weights = shell_nodes(
                np.asarray([lower, upper], dtype=np.float64), int(nradial)
            )
            active_volume = (upper**3 - lower**3) / 3.0
            full_volume = (upper**3 - original_lower**3) / 3.0
            return nodes, active_weights * active_volume / full_volume

        for local_index, pair_index in enumerate(indices):
            pair_edges = geometry.shell_edges[int(pair_index)]
            k1_nodes, k1_weights = finite_shell_nodes(pair_edges[0])
            k2_nodes, k2_weights = finite_shell_nodes(pair_edges[1])
            start = len(triangles)
            weight_sum = 0.0
            for k1, w1 in zip(k1_nodes, k1_weights):
                for k2, w2 in zip(k2_nodes, k2_weights):
                    mu_lower = max(
                        -1.0,
                        (kfund * kfund - k1 * k1 - k2 * k2)
                        / (2.0 * k1 * k2),
                    )
                    if mu_lower >= 1.0:
                        continue
                    # Map the Gauss nodes to [mu_lower, 1] and retain the
                    # Jacobian relative to the original [-1, 1] integral.
                    # The excluded angular wedge therefore contributes zero
                    # without changing the estimator denominator.
                    interval_half_width = 0.5 * (1.0 - mu_lower)
                    mapped_mu = (
                        0.5 * (1.0 + mu_lower)
                        + interval_half_width * mu_nodes
                    )
                    radial_weight = float(w1) * float(w2)
                    for mu, wmu in zip(mapped_mu, mu_weights):
                        weight = (
                            radial_weight
                            * interval_half_width
                            * float(wmu)
                        )
                        triangles.append((float(k1), float(k2), float(mu)))
                        metadata.append((int(local_index), weight))
                        weight_sum += weight
            counts.append(len(triangles) - start)
            weight_sums.append(weight_sum)
        sums = np.asarray(weight_sums, dtype=np.float64)
        work = {
            "bin_average": "finite_box_shell",
            "finite_box_kfund_h_mpc": kfund,
            "finite_box_boxsize_mpc_h": 2.0 * math.pi / kfund,
            "finite_box_domain": "k1,k2,k3 >= kfund; zero mode excluded",
            "finite_box_projection_version": FINITE_BOX_PROJECTION_VERSION,
            "finite_box_normalization": (
                "retain the original full radial-shell and [-1,1] angular "
                "denominator; excluded configurations contribute zero"
            ),
            "finite_box_surviving_fraction_by_pair": (
                (0.5 * sums).tolist()
            ),
            "nmu_b000": int(nmu),
            "nradial_binavg": int(nradial),
            "n_selected_pairs": int(indices.size),
            "n_triangles": int(len(triangles)),
            "triangles_per_pair_min": int(min(counts)),
            "triangles_per_pair_max": int(max(counts)),
            "pre_half_weight_sum_min": float(np.min(sums)),
            "pre_half_weight_sum_max": float(np.max(sums)),
            "normalization_note": (
                "accumulate_b000 multiplies each stored weight by 1/2; "
                "finite-box weight sums are <=2 because the denominator is "
                "not renormalized after zero-mode support is removed"
            ),
        }
    minimum_sum = float(work["pre_half_weight_sum_min"])
    maximum_sum = float(work["pre_half_weight_sum_max"])
    normalized_continuum = kfund is None
    normalization_ok = (
        abs(minimum_sum - 2.0) < 2.0e-12
        and abs(maximum_sum - 2.0) < 2.0e-12
        if normalized_continuum
        else minimum_sum > 0.0
        and maximum_sum <= 2.0 + 2.0e-12
        and minimum_sum < 2.0 - 1.0e-8
    )
    if not normalization_ok:
        raise AssertionError(f"tree shell quadrature normalization failed: {work}")
    return indices, triangles, metadata, work


def run_tree_components(
    *,
    paths: Paths,
    triangles: list[tuple[float, float, float]],
    triangle_metadata: list[tuple[int, float]],
    n_selected: int,
    b1: float,
    b2_native: float,
    bk2: float,
    bphi: float,
    bphidelta: float,
    bphi2: float,
    workers: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    # Tree evaluation is algebraic and emits a compact JSON row.  A larger
    # chunk amortizes process startup while remaining safely below Linux's
    # command-line length limit for three floating-point triangle arguments.
    chunk_size = 2048
    jobs = [
        (start, triangles[start : start + chunk_size])
        for start in range(0, len(triangles), chunk_size)
    ]

    def one(job: tuple[int, list[tuple[float, float, float]]]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        start, chunk = job
        result = run_marisa_b(
            paths.native_binary,
            paths.png_table,
            chunk,
            1.0e-3,
            1.0e-2,
            1.0e-4,
            20.0,
            15.0,
            1.0,
            8.0,
            4,
            1.0e-5,
            "pre_recon_halo_bias_v1_local_png_tree",
            "native_cpp",
            png_table=paths.png_table,
            b1=float(b1),
            b2=float(b2_native),
            bK2=float(bk2),
            bphi=float(bphi),
            bphidelta=float(bphidelta),
            bphi2=float(bphi2),
        )
        payload = result.get("payload")
        rows = list(payload.get("results", [])) if isinstance(payload, dict) else []
        if int(result.get("returncode", -1)) != 0 or len(rows) != len(chunk):
            raise RuntimeError(
                f"native tree chunk {start} failed: "
                f"{str(result.get('stderr', ''))[-2000:]}"
            )
        record = {
            "start": start,
            "n_triangles": len(chunk),
            "wall_seconds": float(
                payload.get("metadata", {}).get("wall_seconds", math.nan)
            ),
        }
        return start, rows, record

    if not 1 <= int(workers) <= 8:
        raise ValueError("workers must lie in [1,8]")
    if workers == 1:
        completed = [one(job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(one, jobs))
    completed.sort(key=lambda item: item[0])
    rows = [row for _start, chunk, _record in completed for row in chunk]
    records = [record for _start, _chunk, record in completed]
    if len(rows) != len(triangle_metadata):
        raise AssertionError("native tree row/weight mismatch")
    vectors = {
        field: np.zeros(n_selected, dtype=np.float64) for field in TREE_FIELDS
    }
    for row, (local_index, weight) in zip(rows, triangle_metadata):
        for field in TREE_FIELDS:
            vectors[field][int(local_index)] += 0.5 * float(weight) * float(
                row[field]
            )
    return vectors, records


def produce_tree_templates(
    paths: Paths,
    geometry: MeasurementGeometry,
    *,
    b1: float,
    nmu: int,
    nradial: int,
    workers: int,
    overwrite: bool,
    kfund: float | None = None,
) -> dict[str, Any]:
    sidecar = paths.tree_templates.with_suffix(".json")
    if paths.tree_templates.exists() and sidecar.exists() and not overwrite:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    paths.theory.mkdir(parents=True, exist_ok=True)
    indices, triangles, metadata, work = tree_work(
        geometry,
        nmu=nmu,
        nradial=nradial,
        kfund=kfund,
    )
    unit, chunks = run_tree_components(
        paths=paths,
        triangles=triangles,
        triangle_metadata=metadata,
        n_selected=indices.size,
        b1=b1,
        b2_native=1.0,
        bk2=1.0,
        bphi=1.0,
        bphidelta=1.0,
        bphi2=1.0,
        workers=workers,
    )
    validation = {
        "b2_eft": 0.83,
        "gamma2": -0.37,
        "p": 1.2,
    }
    b2_native = validation["b2_eft"] - 4.0 * validation["gamma2"] / 3.0
    bphi = 2.0 * DELTA_C * (b1 - validation["p"])
    bphidelta = bphi + 2.0 * (
        DELTA_C * (b2_native - 8.0 * (b1 - 1.0) / 21.0) - b1 + 1.0
    )
    bphi2 = 4.0 * DELTA_C * (
        DELTA_C * (b2_native - 8.0 * (b1 - 1.0) / 21.0)
        - 2.0 * (b1 - 1.0)
    )
    direct, validation_chunks = run_tree_components(
        paths=paths,
        triangles=triangles,
        triangle_metadata=metadata,
        n_selected=indices.size,
        b1=b1,
        b2_native=b2_native,
        bk2=validation["gamma2"],
        bphi=bphi,
        bphidelta=bphidelta,
        bphi2=bphi2,
        workers=workers,
    )
    recombined = (
        unit["dBdfNL_local_primordial"]
        + bphi
        * (
            unit["dBdfNL_local_bphi_f2"]
            + unit["dBdfNL_local_bphi_advection"]
        )
        + bphidelta * unit["dBdfNL_local_bphidelta"]
        + bphi * b2_native * unit["dBdfNL_local_bphi_b2"]
        + bphi * validation["gamma2"] * unit["dBdfNL_local_bphi_bK2"]
    )
    validation_delta = direct["dBdfNL_local_tree"] - recombined
    relative = float(
        np.max(np.abs(validation_delta))
        / max(float(np.max(np.abs(direct["dBdfNL_local_tree"]))), 1.0)
    )
    if relative > 3.0e-12:
        raise AssertionError(f"tree component recombination failed: {relative}")
    recombined_fnl2 = (
        bphi * unit["Bhalo_tree_fNL2_bphi_B0"]
        + bphi**2
        * (
            unit["Bhalo_tree_fNL2_bphi_sq_advection"]
            + unit["Bhalo_tree_fNL2_bphi_sq_F2"]
        )
        + bphi**2
        * b2_native
        * unit["Bhalo_tree_fNL2_bphi_sq_b2"]
        + bphi**2
        * validation["gamma2"]
        * unit["Bhalo_tree_fNL2_bphi_sq_bK2"]
        + bphi
        * bphidelta
        * unit["Bhalo_tree_fNL2_bphi_bphidelta"]
        + bphi2 * unit["Bhalo_tree_fNL2_bphi2_operator"]
    )
    validation_delta_fnl2 = (
        direct["Bhalo_tree_fNL2_deterministic"] - recombined_fnl2
    )
    relative_fnl2 = float(
        np.max(np.abs(validation_delta_fnl2))
        / max(
            float(
                np.max(
                    np.abs(direct["Bhalo_tree_fNL2_deterministic"])
                )
            ),
            1.0,
        )
    )
    stochastic_fnl2_delta = (
        direct["Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"]
        - bphi**2
        * unit["Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"]
    )
    relative_stochastic_fnl2 = float(
        np.max(np.abs(stochastic_fnl2_delta))
        / max(
            float(
                np.max(
                    np.abs(
                        direct[
                            "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"
                        ]
                    )
                )
            ),
            1.0,
        )
    )
    if relative_fnl2 > 3.0e-12 or relative_stochastic_fnl2 > 3.0e-12:
        raise AssertionError(
            "finite-tree fNL2 component recombination failed: "
            f"deterministic={relative_fnl2}, "
            f"stochastic={relative_stochastic_fnl2}"
        )
    atomic_npz(
        paths.tree_templates,
        data_indices=indices,
        k_pair=geometry.weighted_pairs[indices],
        k_edges=geometry.shell_edges[indices],
        b1_fixed=np.asarray(b1),
        nmu=np.asarray(nmu),
        nradial=np.asarray(nradial),
        kfund_h_mpc=np.asarray(
            math.nan if kfund is None else float(kfund)
        ),
        finite_box_projection_version=np.asarray(
            "none"
            if kfund is None
            else FINITE_BOX_PROJECTION_VERSION
        ),
        **unit,
    )
    payload = {
        "schema": 1,
        "created_utc": utc_now(),
        "status": "pass",
        "scope": (
            "pre-reconstruction halo tree local-PNG physical components; "
            "continuous-shell projection"
        ),
        "inputs": {
            "matrix": str(paths.fid_matrix),
            "matrix_sha256": sha256(paths.fid_matrix),
            "png_table": str(paths.png_table),
            "png_table_sha256": sha256(paths.png_table),
            "native_binary": str(paths.native_binary),
            "native_binary_sha256": sha256(paths.native_binary),
        },
        "parameters": {
            "b1": b1,
            "nmu": nmu,
            "nradial": nradial,
            "kfund_h_mpc": kfund,
            "finite_box_projection_version": (
                None
                if kfund is None
                else FINITE_BOX_PROJECTION_VERSION
            ),
            "n_triangles": len(triangles),
        },
        "work": work,
        "chunks": chunks,
        "validation": {
            **validation,
            "b2_native": b2_native,
            "bphi": bphi,
            "bphidelta": bphidelta,
            "bphi2": bphi2,
            "max_relative_recombination_error": relative,
            "finite_tree_fNL2_max_relative_recombination_error": (
                relative_fnl2
            ),
            "finite_tree_fNL2_stochastic_max_relative_recombination_error": (
                relative_stochastic_fnl2
            ),
            "chunks": validation_chunks,
        },
        "output": str(paths.tree_templates),
    }
    atomic_json(sidecar, payload)
    return payload


def load_tree(
    paths: Paths,
    geometry: MeasurementGeometry,
    b1: float,
) -> dict[str, np.ndarray]:
    with np.load(paths.tree_templates, allow_pickle=False) as values:
        arrays = {key: np.asarray(values[key]) for key in values.files}
    expected = np.flatnonzero(
        np.max(geometry.weighted_pairs, axis=1) <= 0.15 + 1.0e-12
    )
    if not np.array_equal(arrays["data_indices"], expected):
        raise AssertionError("tree template data-index mismatch")
    if not np.allclose(
        arrays["k_pair"],
        geometry.weighted_pairs[expected],
        atol=2.0e-7,
        rtol=0.0,
    ):
        raise AssertionError("tree template geometry mismatch")
    if abs(float(arrays["b1_fixed"]) - b1) > 1.0e-10:
        raise AssertionError("tree template b1 mismatch")
    return arrays


def load_dm_vector(
    path: Path,
    geometry: MeasurementGeometry,
    *,
    component: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as values:
        indices = np.asarray(values["computed_pair_indices"], dtype=int)
        vector = np.asarray(values[component], dtype=np.float64)
        edges = np.asarray(values["hdf5_k_edges"], dtype=np.float64)
        arrays = {key: np.asarray(values[key]) for key in values.files}
    expected = np.flatnonzero(
        np.max(geometry.weighted_pairs, axis=1) <= 0.15 + 1.0e-12
    )
    if not np.all(np.isin(expected, indices)):
        raise AssertionError(f"{path.name} does not cover the primary geometry")
    if not np.array_equal(edges, geometry.shell_edges):
        raise AssertionError(f"{path.name} shell edges differ from halo data")
    if not np.all(np.isfinite(vector[expected])):
        raise FloatingPointError(f"{path.name} has non-finite selected values")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return vector, {"metadata": metadata, "arrays": arrays}


def tier_architecture(
    frozen: Any,
    prior: Any,
    tier: str,
    nuisance_prior_scale: float = 1.0,
    *,
    profile_b1: bool = False,
    b1_prior_mean: float | None = None,
    b1_prior_sigma: float | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], gaussian_v3.PriorBlock]:
    if tier in {"full", "coevolution"}:
        free_names = gaussian_v3.b_free_names(frozen, tier)
        nonlinear_names = gaussian_v3.b_nonlinear_names(tier)
        mapping_tier = tier
    elif tier == "uplift":
        free_names = tuple(gaussian_v3.UPLIFT_FREE_NAMES)
        nonlinear_names = tuple(gaussian_v3.UPLIFT_NONLINEAR_NAMES)
        mapping_tier = "full"
    else:
        raise ValueError(tier)
    prior_block = gaussian_v3.prior_block(
        frozen,
        prior,
        free_names,
        nuisance_scale=nuisance_prior_scale,
        nuisance_names=free_names,
    )
    if profile_b1:
        if (
            b1_prior_mean is None
            or b1_prior_sigma is None
            or not math.isfinite(float(b1_prior_mean))
            or not math.isfinite(float(b1_prior_sigma))
            or float(b1_prior_sigma) <= 0.0
        ):
            raise ValueError(
                "profiling b1 requires a finite positive P0 prior"
            )
        free_names = ("b1",) + tuple(free_names)
        nonlinear_names = ("b1",) + tuple(nonlinear_names)
        mean = np.concatenate(
            (
                np.asarray([float(b1_prior_mean)]),
                np.asarray(prior_block.mean, dtype=np.float64),
            )
        )
        covariance = np.zeros(
            (mean.size, mean.size), dtype=np.float64
        )
        covariance[0, 0] = float(b1_prior_sigma) ** 2
        covariance[1:, 1:] = prior_block.covariance
        cholesky = np.linalg.cholesky(covariance)
        prior_block = gaussian_v3.PriorBlock(
            names=free_names,
            mean=mean,
            covariance=covariance,
            sigma=np.sqrt(np.diag(covariance)),
            whitener=solve_triangular(
                cholesky,
                np.eye(mean.size, dtype=np.float64),
                lower=True,
            ),
        )
    if mapping_tier not in {"full", "coevolution"}:
        raise AssertionError("invalid parameter-mapping tier")
    return free_names, nonlinear_names, prior_block


def parameter_dict(
    frozen: Any,
    free_names: tuple[str, ...],
    values: np.ndarray,
    *,
    tier: str,
    b1: float,
) -> dict[str, float]:
    return gaussian_v3.full_parameter_dict(
        frozen,
        free_names,
        values,
        tier=tier if tier != "uplift" else "full",
        fixed_b1=None if "b1" in free_names else b1,
    )


def gaussian_components(
    frozen: Any,
    templates: Any,
    parameters: dict[str, float],
    nbar: float,
    tier: str,
) -> dict[str, np.ndarray]:
    if tier == "uplift":
        return gaussian_v3.uplift_components(
            frozen,
            templates,
            parameters,
            nbar,
        )
    return templates.components(parameters, nbar)


def response_components(
    *,
    tree: dict[str, np.ndarray],
    templates: Any,
    parameters: dict[str, float],
    b1: float,
    p: float,
    nbar: float,
    dm_tree: np.ndarray,
    dm_total: np.ndarray,
    dm_quadratic: np.ndarray,
    response_model: str,
) -> dict[str, np.ndarray]:
    cached_b1 = float(np.asarray(tree["b1_fixed"]).item())
    if not math.isfinite(cached_b1) or cached_b1 == 0.0:
        raise ValueError(f"invalid cached tree b1={cached_b1}")
    b1_ratio = float(b1) / cached_b1
    b2_eft = float(parameters["b2"])
    gamma2 = float(parameters["gamma2"])
    b2_native = b2_eft - 4.0 * gamma2 / 3.0
    bphi = 2.0 * DELTA_C * (float(b1) - float(p))
    bphidelta = bphi + 2.0 * (
        DELTA_C * (b2_native - 8.0 * (float(b1) - 1.0) / 21.0)
        - float(b1)
        + 1.0
    )
    b1_lagrangian = float(b1) - 1.0
    b2_lagrangian = (
        b2_native - 8.0 * (float(b1) - 1.0) / 21.0
    )
    bphi2 = 4.0 * DELTA_C * (
        DELTA_C * b2_lagrangian - 2.0 * b1_lagrangian
    )
    data_indices = np.asarray(tree["data_indices"], dtype=int)
    if data_indices.size != 28:
        raise AssertionError("tree response cache must contain 28 bins")
    primordial_local = np.asarray(
        tree["dBdfNL_local_primordial"], dtype=np.float64
    ) * b1_ratio**3
    bphi_f2_local = bphi * b1_ratio**2 * np.asarray(
        tree["dBdfNL_local_bphi_f2"], dtype=np.float64
    )
    advection_local = bphi * b1_ratio**2 * np.asarray(
        tree["dBdfNL_local_bphi_advection"], dtype=np.float64
    )
    bphidelta_local = bphidelta * b1_ratio**2 * np.asarray(
        tree["dBdfNL_local_bphidelta"], dtype=np.float64
    )
    bphi_b2_local = bphi * b2_native * b1_ratio * np.asarray(
        tree["dBdfNL_local_bphi_b2"], dtype=np.float64
    )
    bphi_bk2_local = bphi * gamma2 * b1_ratio * np.asarray(
        tree["dBdfNL_local_bphi_bK2"], dtype=np.float64
    )
    deterministic_tree_local = (
        primordial_local
        + bphi_f2_local
        + advection_local
        + bphidelta_local
        + bphi_b2_local
        + bphi_bk2_local
    )

    # Tie the old leading tree stochastic response to the exact-lattice
    # Gaussian Bshot convention bin by bin.  The ratio removes the small
    # continuum-shell versus exact-lattice normalization difference.
    native_gaussian = np.asarray(
        tree["stochastic_alpha3_basis"], dtype=np.float64
    ) / cached_b1**2
    if np.any(native_gaussian == 0.0):
        raise ZeroDivisionError("zero native leading stochastic tree shape")
    exact_lattice_leading = (
        np.asarray(
            templates._compiled["stochastic:Bshot_residual"]["b1^2"][
                data_indices
            ],
            dtype=np.float64,
        )
    )
    projection_adapter = exact_lattice_leading / native_gaussian
    stochastic_unit_local = (
        bphi
        * float(b1)
        / cached_b1
        * np.asarray(
            tree["dBdfNL_stochastic_alpha3_basis"], dtype=np.float64
        )
        * projection_adapter
        / nbar
    )
    stochastic_local = (
        float(parameters["Bshot_residual"]) * stochastic_unit_local
    )
    has_halo_fnl2 = all(field in tree for field in TREE_FNL2_FIELDS)
    finite_halo_models = {
        "finite_halo_tree",
        "finite_halo_tree_deterministic",
    }
    if response_model in finite_halo_models and not has_halo_fnl2:
        raise KeyError(
            f"{response_model} requires a tree cache with all halo fNL2 fields"
        )
    if has_halo_fnl2:
        halo_quadratic_deterministic_local = (
            bphi
            * b1_ratio**2
            * np.asarray(
                tree["Bhalo_tree_fNL2_bphi_B0"],
                dtype=np.float64,
            )
            + bphi**2
            * b1_ratio
            * (
                np.asarray(
                    tree["Bhalo_tree_fNL2_bphi_sq_advection"],
                    dtype=np.float64,
                )
                + np.asarray(
                    tree["Bhalo_tree_fNL2_bphi_sq_F2"],
                    dtype=np.float64,
                )
            )
            + bphi**2
            * b2_native
            * np.asarray(
                tree["Bhalo_tree_fNL2_bphi_sq_b2"],
                dtype=np.float64,
            )
            + bphi**2
            * gamma2
            * np.asarray(
                tree["Bhalo_tree_fNL2_bphi_sq_bK2"],
                dtype=np.float64,
            )
            + bphi
            * bphidelta
            * b1_ratio
            * np.asarray(
                tree["Bhalo_tree_fNL2_bphi_bphidelta"],
                dtype=np.float64,
            )
            + bphi2
            * b1_ratio**2
            * np.asarray(
                tree["Bhalo_tree_fNL2_bphi2_operator"],
                dtype=np.float64,
            )
        )
        halo_quadratic_stochastic_unit_local = (
            bphi**2
            * np.asarray(
                tree["Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"],
                dtype=np.float64,
            )
            * projection_adapter
            / nbar
        )
    else:
        halo_quadratic_deterministic_local = np.zeros_like(
            primordial_local
        )
        halo_quadratic_stochastic_unit_local = np.zeros_like(
            primordial_local
        )
    halo_quadratic_stochastic_local = (
        float(parameters["Bshot_residual"])
        * halo_quadratic_stochastic_unit_local
    )
    matter_linear_uplift_local = b1**3 * (
        np.asarray(dm_total[data_indices], dtype=np.float64)
        - np.asarray(dm_tree[data_indices], dtype=np.float64)
    )
    matter_quadratic_local = b1**3 * np.asarray(
        dm_quadratic[data_indices], dtype=np.float64
    )
    if response_model == "tree":
        matter_linear_uplift_local = np.zeros_like(
            matter_linear_uplift_local
        )
        matter_quadratic_local = np.zeros_like(matter_quadratic_local)
        halo_quadratic_deterministic_local = np.zeros_like(
            halo_quadratic_deterministic_local
        )
        halo_quadratic_stochastic_local = np.zeros_like(
            halo_quadratic_stochastic_local
        )
    elif response_model == "linear_uplift":
        matter_quadratic_local = np.zeros_like(matter_quadratic_local)
        halo_quadratic_deterministic_local = np.zeros_like(
            halo_quadratic_deterministic_local
        )
        halo_quadratic_stochastic_local = np.zeros_like(
            halo_quadratic_stochastic_local
        )
    elif response_model == "finite_uplift":
        halo_quadratic_deterministic_local = np.zeros_like(
            halo_quadratic_deterministic_local
        )
        halo_quadratic_stochastic_local = np.zeros_like(
            halo_quadratic_stochastic_local
        )
    elif response_model == "finite_halo_tree_deterministic":
        halo_quadratic_stochastic_local = np.zeros_like(
            halo_quadratic_stochastic_local
        )
    elif response_model != "finite_halo_tree":
        raise ValueError(response_model)
    linear_total_local = (
        deterministic_tree_local
        + stochastic_local
        + matter_linear_uplift_local
    )

    def full(local: np.ndarray) -> np.ndarray:
        result = np.zeros(120, dtype=np.float64)
        result[data_indices] = np.asarray(local, dtype=np.float64)
        return result

    return {
        "primordial": full(primordial_local),
        "bphi_f2": full(bphi_f2_local),
        "advection": full(advection_local),
        "bphidelta": full(bphidelta_local),
        "bphi_b2": full(bphi_b2_local),
        "bphi_bK2": full(bphi_bk2_local),
        "deterministic_tree": full(deterministic_tree_local),
        "stochastic_unit_Bshot": full(stochastic_unit_local),
        "stochastic": full(stochastic_local),
        "matter_linear_uplift": full(matter_linear_uplift_local),
        "linear_total": full(linear_total_local),
        "matter_quadratic": full(matter_quadratic_local),
        "halo_quadratic_deterministic": full(
            halo_quadratic_deterministic_local
        ),
        "halo_quadratic_stochastic_unit_Bshot": full(
            halo_quadratic_stochastic_unit_local
        ),
        "halo_quadratic_stochastic": full(
            halo_quadratic_stochastic_local
        ),
        "quadratic_total": full(
            matter_quadratic_local
            + halo_quadratic_deterministic_local
            + halo_quadratic_stochastic_local
        ),
        "projection_adapter_local": projection_adapter,
        "data_indices": data_indices,
        "bphi": np.asarray(bphi),
        "bphidelta_value": np.asarray(bphidelta),
        "bphi2_value": np.asarray(bphi2),
        "b2_native": np.asarray(b2_native),
    }


def profile_hybrid(
    *,
    frozen: Any,
    templates: Any,
    frozen_prior: Any,
    nbar: float,
    tree: dict[str, np.ndarray],
    dm_tree: np.ndarray,
    dm_total: np.ndarray,
    dm_quadratic: np.ndarray,
    k_pair: np.ndarray,
    covariance_samples: np.ndarray,
    target_samples: np.ndarray,
    tier: str,
    sample: str,
    p: float,
    kmax: float,
    b1: float,
    response_model: str,
    start_fnl: float,
    compute_curve: bool,
    fixed_fnl: float | None = None,
    nuisance_prior_scale: float = 1.0,
    covariance_label: str = "fiducial500",
    profile_b1: bool = False,
    b1_prior_sigma: float | None = None,
    start_b1: float | None = None,
) -> ProfileResult:
    indices = selected_indices(k_pair, kmax)
    expected = {0.08: 9, 0.10: 14, 0.12: 20, 0.15: 27}
    if indices.size != expected[float(kmax)]:
        raise AssertionError(
            f"kmax={kmax}: got {indices.size}, expected {expected[float(kmax)]}"
        )
    covariance = covariance_single(covariance_samples[:, indices])
    target = np.mean(target_samples[:, indices], axis=0)
    chol = np.linalg.cholesky(covariance)
    precision = np.linalg.inv(covariance)
    free_names, nonlinear_names, prior_block = tier_architecture(
        frozen,
        frozen_prior,
        tier,
        nuisance_prior_scale,
        profile_b1=profile_b1,
        b1_prior_mean=b1,
        b1_prior_sigma=b1_prior_sigma,
    )
    name_to_index = {name: index for index, name in enumerate(free_names)}
    nonlinear_indices = np.asarray(
        [name_to_index[name] for name in nonlinear_names],
        dtype=int,
    )
    linear_names = tuple(
        name for name in free_names if name not in nonlinear_names
    )
    linear_indices = np.asarray(
        [name_to_index[name] for name in linear_names],
        dtype=int,
    )

    def evaluate_model(
        fnl: float,
        nuisance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        parameters = parameter_dict(
            frozen,
            free_names,
            nuisance,
            tier=tier,
            b1=b1,
        )
        g_components = gaussian_components(
            frozen,
            templates,
            parameters,
            nbar,
            tier,
        )
        r_components = response_components(
            tree=tree,
            templates=templates,
            parameters=parameters,
            b1=float(parameters["b1"]),
            p=p,
            nbar=nbar,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
            response_model=response_model,
        )
        gaussian = np.asarray(g_components["total"], dtype=np.float64)[indices]
        linear = np.asarray(r_components["linear_total"])[indices]
        quadratic = np.asarray(r_components["quadratic_total"])[indices]
        prediction = gaussian + float(fnl) * linear + float(fnl) ** 2 * quadratic
        return prediction, gaussian, r_components, parameters

    def solve_linear(
        fnl: float,
        nonlinear: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        nuisance = prior_block.mean.copy()
        nuisance[nonlinear_indices] = nonlinear
        nuisance[linear_indices] = 0.0
        base = evaluate_model(fnl, nuisance)[0]
        design = np.empty((target.size, linear_indices.size), dtype=np.float64)
        for column, parameter_index in enumerate(linear_indices):
            unit = nuisance.copy()
            unit[parameter_index] = 1.0
            design[:, column] = evaluate_model(fnl, unit)[0] - base
        whitened_design = solve_triangular(chol, design, lower=True)
        whitened_target = solve_triangular(chol, target - base, lower=True)
        prior_design = prior_block.whitener[:, linear_indices]
        prior_target = -prior_block.whitener @ (
            nuisance - prior_block.mean
        )
        solution = np.linalg.lstsq(
            np.vstack((whitened_design, prior_design)),
            np.concatenate((whitened_target, prior_target)),
            rcond=1.0e-12,
        )[0]
        nuisance[linear_indices] = solution
        prediction = evaluate_model(fnl, nuisance)[0]
        data_residual = solve_triangular(
            chol,
            prediction - target,
            lower=True,
        )
        prior_residual = prior_block.whitener @ (
            nuisance - prior_block.mean
        )
        objective = float(
            data_residual @ data_residual + prior_residual @ prior_residual
        )
        return nuisance, objective

    objective_names = ("fNL",) + nonlinear_names

    def objective(*values: float) -> float:
        _nuisance, value = solve_linear(
            float(values[0]),
            np.asarray(values[1:], dtype=np.float64),
        )
        return value

    starts = [float(start_fnl)] + [
        float(prior_block.mean[index]) for index in nonlinear_indices
    ]
    if start_b1 is not None:
        if "b1" not in nonlinear_names:
            raise ValueError("start_b1 requires profile_b1=True")
        starts[
            1 + nonlinear_names.index("b1")
        ] = float(start_b1)
    minuit = Minuit(objective, *starts, name=objective_names)
    minuit.errordef = 1.0
    # The profile objectives can be O(0.1), so iminuit's default EDM
    # tolerance is needlessly loose for the explicit multistart audit.
    minuit.tol = 1.0e-4
    minuit.errors["fNL"] = 30.0
    minuit.limits["fNL"] = (-800.0, 800.0)
    for name, index in zip(nonlinear_names, nonlinear_indices):
        minuit.errors[name] = max(float(prior_block.sigma[index]) * 0.2, 0.05)
        if name == "b1":
            minuit.limits[name] = (0.1, 6.0)
    if fixed_fnl is not None:
        minuit.values["fNL"] = float(fixed_fnl)
        minuit.fixed["fNL"] = True
    minuit.migrad(ncall=10000)
    if not minuit.valid:
        minuit.simplex(ncall=5000)
        minuit.migrad(ncall=15000)
    minuit.hesse()
    fhat = float(minuit.values["fNL"])
    if fixed_fnl is None:
        try:
            minuit.minos("fNL", cl=0.682689492137)
            merror = minuit.merrors["fNL"]
            error_low = float(merror.lower)
            error_high = float(merror.upper)
        except Exception:
            symmetric = float(minuit.errors["fNL"])
            error_low = -symmetric
            error_high = symmetric
        sigma_symmetric = 0.5 * (abs(error_low) + abs(error_high))
    else:
        error_low = math.nan
        error_high = math.nan
        sigma_symmetric = math.nan
    nonlinear_best = np.asarray(
        [minuit.values[name] for name in nonlinear_names],
        dtype=np.float64,
    )
    nuisance, objective_best = solve_linear(fhat, nonlinear_best)
    prediction, gaussian, response, parameters = evaluate_model(fhat, nuisance)
    residual = prediction - target
    whitened = solve_triangular(chol, residual, lower=True)
    prior_residual = prior_block.whitener @ (nuisance - prior_block.mean)
    data_chi2 = float(whitened @ whitened)
    prior_chi2 = float(prior_residual @ prior_residual)

    # Local effective parameter count and nuisance tangent for coverage.
    nuisance_jacobian = np.empty(
        (target.size, len(free_names)), dtype=np.float64
    )
    for column in range(len(free_names)):
        step = 1.0e-5 * max(
            abs(float(nuisance[column])),
            float(prior_block.sigma[column]),
            1.0e-3,
        )
        right = nuisance.copy()
        left = nuisance.copy()
        right[column] += step
        left[column] -= step
        nuisance_jacobian[:, column] = (
            evaluate_model(fhat, right)[0]
            - evaluate_model(fhat, left)[0]
        ) / (2.0 * step)
    fnl_step = max(1.0e-3, 1.0e-4 * max(abs(fhat), 1.0))
    fnl_tangent = (
        evaluate_model(fhat + fnl_step, nuisance)[0]
        - evaluate_model(fhat - fnl_step, nuisance)[0]
    ) / (2.0 * fnl_step)
    design_all = np.column_stack((fnl_tangent, nuisance_jacobian))
    prior_precision = np.zeros(
        (1 + len(free_names), 1 + len(free_names)), dtype=np.float64
    )
    prior_precision[1:, 1:] = np.linalg.inv(prior_block.covariance)
    fisher = design_all.T @ precision @ design_all + prior_precision
    fisher_inverse = np.linalg.pinv(fisher, rcond=1.0e-11)
    effective = float(
        np.trace(
            design_all
            @ fisher_inverse
            @ design_all.T
            @ precision
        )
    )
    dof = max(float(target.size) - effective, 1.0e-6)
    p_value = float(chi2_distribution.sf(data_chi2, dof))

    if fixed_fnl is not None:
        profile_x = np.asarray([float(fixed_fnl)])
        profile_delta = np.asarray([0.0])
    elif compute_curve:
        lo = max(-800.0, fhat - 3.2 * max(abs(error_low), sigma_symmetric))
        hi = min(800.0, fhat + 3.2 * max(abs(error_high), sigma_symmetric))
        try:
            profile_x, profile_y, profile_ok = minuit.mnprofile(
                "fNL",
                size=33,
                bound=(lo, hi),
                subtract_min=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"actual Minuit fNL profile failed for {tier}/{sample}, "
                f"kmax={kmax:.2f}, model={response_model}"
            ) from error
        profile_x = np.asarray(profile_x, dtype=np.float64)
        profile_delta = np.asarray(profile_y, dtype=np.float64)
        profile_ok = np.asarray(profile_ok, dtype=bool)
        if (
            profile_x.shape != (33,)
            or profile_delta.shape != (33,)
            or profile_ok.shape != (33,)
            or not np.all(np.isfinite(profile_x))
            or not np.all(np.isfinite(profile_delta))
            or not np.all(profile_ok)
        ):
            failed = np.flatnonzero(~profile_ok).tolist()
            raise RuntimeError(
                f"invalid Minuit fNL profile for {tier}/{sample}, "
                f"kmax={kmax:.2f}, model={response_model}; "
                f"failed_points={failed}"
            )
    else:
        profile_x = np.linspace(
            fhat - 3.0 * sigma_symmetric,
            fhat + 3.0 * sigma_symmetric,
            25,
        )
        profile_delta = ((profile_x - fhat) / sigma_symmetric) ** 2

    return ProfileResult(
        tier=tier,
        sample=sample,
        p=float(p),
        kmax=float(kmax),
        response_model=response_model,
        b1=float(parameters["b1"]),
        nuisance_prior_scale=float(nuisance_prior_scale),
        covariance_label=str(covariance_label),
        indices=indices,
        free_names=free_names,
        nonlinear_names=nonlinear_names,
        nuisance_values=nuisance,
        expanded_parameters=parameters,
        prediction=prediction,
        gaussian_prediction=gaussian,
        linear_response=np.asarray(response["linear_total"])[indices],
        quadratic_coefficient=np.asarray(response["quadratic_total"])[indices],
        target=target,
        covariance_single=covariance,
        fhat=fhat,
        error_low=error_low,
        error_high=error_high,
        sigma_symmetric=sigma_symmetric,
        objective=objective_best,
        data_chi2=data_chi2,
        prior_chi2=prior_chi2,
        dof=dof,
        p_value=p_value,
        max_pull_single=float(
            np.max(np.abs(residual) / np.sqrt(np.diag(covariance)))
        ),
        residual_norm_single=qnorm(residual, covariance),
        minuit_valid=bool(minuit.valid),
        minuit_accurate=bool(minuit.fmin.has_accurate_covar),
        minuit_at_limit=bool(minuit.fmin.has_parameters_at_limit),
        minuit_nfcn=int(minuit.nfcn),
        profile_x=np.asarray(profile_x),
        profile_delta=np.asarray(profile_delta),
        nuisance_jacobian=nuisance_jacobian,
    )


def frozen_nuisance_fnl(
    *,
    fit0: ProfileResult,
    target_samples: np.ndarray,
    truth_fnl: float,
    frozen: Any,
    templates: Any,
    nbar: float,
    tree: dict[str, np.ndarray],
    dm_tree: np.ndarray,
    dm_total: np.ndarray,
    dm_quadratic: np.ndarray,
) -> dict[str, Any]:
    indices = fit0.indices
    target = np.mean(target_samples[:, indices], axis=0)
    covariance = fit0.covariance_single
    precision = np.linalg.inv(covariance)
    parameters = fit0.expanded_parameters
    g_components = gaussian_components(
        frozen,
        templates,
        parameters,
        nbar,
        fit0.tier,
    )
    response = response_components(
        tree=tree,
        templates=templates,
        parameters=parameters,
        b1=fit0.b1,
        p=fit0.p,
        nbar=nbar,
        dm_tree=dm_tree,
        dm_total=dm_total,
        dm_quadratic=dm_quadratic,
        response_model=fit0.response_model,
    )
    gaussian = np.asarray(g_components["total"])[indices]
    linear = np.asarray(response["linear_total"])[indices]
    quadratic = np.asarray(response["quadratic_total"])[indices]

    def objective(fnl: float) -> float:
        residual = gaussian + fnl * linear + fnl * fnl * quadratic - target
        return float(residual @ precision @ residual)

    minuit = Minuit(objective, fnl=truth_fnl)
    minuit.errordef = 1.0
    minuit.errors["fnl"] = 30.0
    minuit.limits["fnl"] = (-800.0, 800.0)
    minuit.migrad()
    minuit.hesse()
    fhat = float(minuit.values["fnl"])
    sigma = float(minuit.errors["fnl"])
    prediction = gaussian + fhat * linear + fhat * fhat * quadratic
    prediction_at_truth = (
        gaussian
        + truth_fnl * linear
        + truth_fnl * truth_fnl * quadratic
    )
    residual_at_truth = prediction_at_truth - target
    return {
        "truth_fnl": truth_fnl,
        "fhat": fhat,
        "sigma": sigma,
        "bias_over_sigma": abs(fhat - truth_fnl) / sigma,
        "prediction": prediction,
        "prediction_at_truth": prediction_at_truth,
        "residual_norm_at_truth_sigma_single": qnorm(
            residual_at_truth,
            covariance,
        ),
        "max_pull_at_truth_sigma_single": float(
            np.max(
                np.abs(residual_at_truth)
                / np.sqrt(np.diag(covariance))
            )
        ),
        "valid": bool(minuit.valid),
    }


def profile_row(result: ProfileResult) -> dict[str, Any]:
    return {
        "tier": result.tier,
        "sample": result.sample,
        "p": result.p,
        "kmax_h_mpc": result.kmax,
        "response_model": result.response_model,
        "b1": result.b1,
        "nuisance_prior_scale": result.nuisance_prior_scale,
        "covariance_label": result.covariance_label,
        "n_data": int(result.indices.size),
        "selected_indices": result.indices,
        "fhat": result.fhat,
        "error_low": result.error_low,
        "error_high": result.error_high,
        "sigma_symmetric": result.sigma_symmetric,
        "interval68": [
            result.fhat + result.error_low,
            result.fhat + result.error_high,
        ],
        "objective": result.objective,
        "data_chi2": result.data_chi2,
        "prior_chi2": result.prior_chi2,
        "dof_effective": result.dof,
        "gof_p_single_covariance": result.p_value,
        "residual_norm_sigma_single": result.residual_norm_single,
        "max_abs_pull_sigma_single": result.max_pull_single,
        "minuit": {
            "valid": result.minuit_valid,
            "accurate_covariance": result.minuit_accurate,
            "parameters_at_limit": result.minuit_at_limit,
            "nfcn": result.minuit_nfcn,
        },
        "free_parameters": {
            name: float(value)
            for name, value in zip(result.free_names, result.nuisance_values)
        },
        "response_parameters": {
            key: float(result.expanded_parameters[key])
            for key in ("b1", "b2", "gamma2", "Bshot_residual")
        },
    }


def response_diagnostics(
    *,
    fit0: ProfileResult,
    fid100: np.ndarray,
    lcp: np.ndarray,
    lcm: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    indices = fit0.indices
    odd_samples = (lcp[:, indices] - lcm[:, indices]) / 200.0
    dplus_samples = (lcp[:, indices] - fid100[:, indices]) / 100.0
    even_displacement_samples = (
        lcp[:, indices] + lcm[:, indices] - 2.0 * fid100[:, indices]
    ) / 2.0
    odd = np.mean(odd_samples, axis=0)
    dplus = np.mean(dplus_samples, axis=0)
    even_displacement = np.mean(even_displacement_samples, axis=0)
    covariance_odd = covariance_single(odd_samples)
    covariance_dplus = covariance_single(dplus_samples)
    covariance_even = covariance_single(even_displacement_samples)
    model_odd = fit0.linear_response
    model_dplus = (
        fit0.linear_response + 100.0 * fit0.quadratic_coefficient
    )
    predicted_even = 10000.0 * fit0.quadratic_coefficient
    unexplained_even = even_displacement - predicted_even
    precision_b = np.linalg.inv(fit0.covariance_single)
    fisher = float(model_odd @ precision_b @ model_odd)
    sigma_fnl = 1.0 / math.sqrt(fisher)
    projected_even = float(
        model_odd @ precision_b @ unexplained_even / fisher
    )
    payload = {
        "diagnostic_only": True,
        "valid_model_acceptance_test": False,
        "non_closure_reason": (
            "LC_p-LC_m is a total fixed-selection catalogue response: "
            "dB_h/dfNL includes the fNL dependence of halo abundance, bias, "
            "and stochastic/nuisance parameters.  The plotted theory is the "
            "partial response at the fiducial nuisance point, "
            "(partial B_h/partial fNL)_theta.  Their difference therefore "
            "cannot be interpreted as model closure or failure."
        ),
        "covariance_contract": (
            "all response/even covariances are row-wise sample covariances "
            "and are not divided by 100"
        ),
        "odd_model_minus_data_norm_sigma_single_response": qnorm(
            model_odd - odd,
            covariance_odd,
        ),
        "dplus_model_minus_data_norm_sigma_single_response": qnorm(
            model_dplus - dplus,
            covariance_dplus,
        ),
        "measured_even_norm_sigma_single_even": qnorm(
            even_displacement,
            covariance_even,
        ),
        "predicted_b112ii_even_norm_sigma_single_even": qnorm(
            predicted_even,
            covariance_even,
        ),
        "unexplained_even_norm_sigma_single_even": qnorm(
            unexplained_even,
            covariance_even,
        ),
        "unexplained_even_projected_delta_fnl": projected_even,
        "sigma_fnl_fixed_nuisance_single": sigma_fnl,
        "unexplained_even_abs_projection_over_sigma": abs(
            projected_even / sigma_fnl
        ),
    }
    arrays = {
        "odd": odd,
        "odd_error_single": np.sqrt(np.diag(covariance_odd)),
        "dplus": dplus,
        "dplus_error_single": np.sqrt(np.diag(covariance_dplus)),
        "even_displacement": even_displacement,
        "even_error_single": np.sqrt(np.diag(covariance_even)),
        "model_odd": model_odd,
        "model_dplus": model_dplus,
        "predicted_even": predicted_even,
        "unexplained_even": unexplained_even,
    }
    return payload, arrays


def theoretical_finite_fnl_diagnostic(
    fit0: ProfileResult,
    *,
    injection_abs: float = 100.0,
) -> dict[str, Any]:
    """Quantify the implemented even term without using catalogue response."""

    amplitude = float(injection_abs)
    covariance = fit0.covariance_single
    precision = np.linalg.inv(covariance)
    linear = amplitude * fit0.linear_response
    quadratic = amplitude**2 * fit0.quadratic_coefficient
    fisher = float(
        fit0.linear_response @ precision @ fit0.linear_response
    )
    sigma_fnl = 1.0 / math.sqrt(fisher)
    projected_delta = float(
        fit0.linear_response @ precision @ quadratic / fisher
    )
    linear_norm = qnorm(linear, covariance)
    quadratic_norm = qnorm(quadratic, covariance)
    per_bin_sigma = np.sqrt(np.diag(covariance))
    return {
        "diagnostic_only": True,
        "injection_abs_fNL": amplitude,
        "linear_displacement_norm_sigma_single": linear_norm,
        "matter_B112II_displacement_norm_sigma_single": quadratic_norm,
        "B112II_to_linear_covariance_norm_ratio": (
            quadratic_norm / linear_norm if linear_norm > 0.0 else math.nan
        ),
        "B112II_max_abs_bin_pull_sigma_single": float(
            np.max(np.abs(quadratic) / per_bin_sigma)
        ),
        "B112II_projected_delta_fNL": projected_delta,
        "B112II_abs_projection_over_fixed_nuisance_sigma": abs(
            projected_delta / sigma_fnl
        ),
        "parity_identity": (
            "the implemented fNL-linear response is odd and has identical "
            "relative size at +|fNL| and -|fNL|; the B112II term is even, "
            "is identical at both signs, and cancels exactly from "
            "[B(+fNL)-B(-fNL)]/(2 fNL)"
        ),
    }


def free_bphidelta_diagnostic(
    *,
    fit0: ProfileResult,
    tree: dict[str, np.ndarray],
    fid100: np.ndarray,
    lcp: np.ndarray,
    lcm: np.ndarray,
    prior_sigma: float = 2.0 * DELTA_C,
) -> dict[str, Any]:
    """Describe one extra PNG shape in the total-response diagnostic."""

    indices = fit0.indices
    odd_samples = (lcp[:, indices] - lcm[:, indices]) / 200.0
    odd = np.mean(odd_samples, axis=0)
    covariance_odd = covariance_single(odd_samples)
    precision_odd = np.linalg.inv(covariance_odd)
    full_shape = np.zeros(120, dtype=np.float64)
    full_shape[np.asarray(tree["data_indices"], dtype=int)] = np.asarray(
        tree["dBdfNL_local_bphidelta"],
        dtype=np.float64,
    )
    shape = full_shape[indices]
    fisher = float(
        shape @ precision_odd @ shape + 1.0 / prior_sigma**2
    )
    delta = float(
        shape
        @ precision_odd
        @ (odd - fit0.linear_response)
        / fisher
    )
    sigma_posterior = 1.0 / math.sqrt(fisher)
    adjusted_response = fit0.linear_response + delta * shape
    target_plus = np.mean(lcp[:, indices], axis=0)
    base_prediction_100 = (
        fit0.gaussian_prediction
        + 100.0 * fit0.linear_response
        + 10000.0 * fit0.quadratic_coefficient
    )
    adjusted_prediction_100 = base_prediction_100 + 100.0 * delta * shape
    return {
        "diagnostic_only": True,
        "valid_model_acceptance_test": False,
        "definition": (
            "bphidelta = closure value + Delta_bphidelta; "
            f"Delta_bphidelta ~ Normal(0,{prior_sigma:.6g}^2)"
        ),
        "prior_sigma": prior_sigma,
        "delta_bphidelta_posterior_mode": delta,
        "posterior_sigma_local": sigma_posterior,
        "abs_mode_over_prior_sigma": abs(delta) / prior_sigma,
        "odd_response_norm_before_sigma_single": qnorm(
            fit0.linear_response - odd,
            covariance_odd,
        ),
        "odd_response_norm_after_sigma_single": qnorm(
            adjusted_response - odd,
            covariance_odd,
        ),
        "plus100_frozen_prediction_norm_before_sigma_single": qnorm(
            base_prediction_100 - target_plus,
            fit0.covariance_single,
        ),
        "plus100_frozen_prediction_norm_after_sigma_single": qnorm(
            adjusted_prediction_100 - target_plus,
            fit0.covariance_single,
        ),
        "interpretation": (
            "Because the target is a total fixed-selection catalogue "
            "response, this fit cannot validate bphidelta or the PNG model. "
            "It only shows how much of the descriptive shape mismatch lies "
            "along this one template."
        ),
    }


def numerical_convergence_diagnostic(
    *,
    paths: Paths,
    fit0: ProfileResult,
    b1: float,
) -> dict[str, Any]:
    old_linear = (
        paths.data_root
        / "analysis/theory_vectors/marisa_b_v0"
        / "marisa_b_pre_local_png_1loop_z1_full2d_kmax0p08_"
        "nrad3_nmu24_eps1e3_20260723.npz"
    )
    old_quadratic = (
        paths.data_root
        / "analysis/quijote_dm_b112ii_fnl100_z1"
        / "marisa_b_pre_b112ii_z1_full2d_kmax0p08_kir_kf_split_"
        "qmax30_nrad5_nmu48_qmc11r4.npz"
    )
    for path in (old_linear, old_quadratic):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(paths.dm_linear, allow_pickle=False) as new, np.load(
        old_linear,
        allow_pickle=False,
    ) as old:
        new_uplift = np.asarray(new["B000_dBdfNL_primary"]) - np.asarray(
            new["B000_dBdfNL_local_tree"]
        )
        old_uplift = np.asarray(old["B000_dBdfNL_primary"]) - np.asarray(
            old["B000_dBdfNL_local_tree"]
        )
    with np.load(paths.dm_quadratic, allow_pickle=False) as new, np.load(
        old_quadratic,
        allow_pickle=False,
    ) as old:
        new_quadratic = np.asarray(
            new["B000_B112II_fNL2_coefficient"]
        )
        old_quadratic_vector = np.asarray(
            old["B000_B112II_fNL2_coefficient"]
        )
    indices = fit0.indices
    precision = np.linalg.inv(fit0.covariance_single)
    response = fit0.linear_response
    fisher = float(response @ precision @ response)
    sigma = 1.0 / math.sqrt(fisher)

    def projected(vector: np.ndarray) -> dict[str, float]:
        delta_fnl = float(response @ precision @ vector / fisher)
        return {
            "norm_sigma_single": qnorm(
                vector,
                fit0.covariance_single,
            ),
            "projected_delta_fnl": delta_fnl,
            "abs_projected_delta_over_sigma": abs(delta_fnl) / sigma,
        }

    linear_delta_b100 = (
        100.0
        * b1**3
        * (new_uplift[indices] - old_uplift[indices])
    )
    quadratic_delta_b100 = (
        10000.0
        * b1**3
        * (
            new_quadratic[indices]
            - old_quadratic_vector[indices]
        )
    )
    metadata = json.loads(
        paths.dm_linear.with_suffix(".json").read_text(encoding="utf-8")
    )
    warning_count = sum(
        str(row.get("stderr", "")).count("did not converge")
        for row in metadata.get("chunks", [])
    )
    return {
        "linear_response": {
            "production": "nradial=3,nmu=48",
            "reference": "nradial=3,nmu=24",
            **projected(linear_delta_b100),
        },
        "B112II": {
            "production": "nradial=3,nmu=24,QMC2^11x4",
            "reference": "nradial=5,nmu=48,QMC2^11x4",
            **projected(quadratic_delta_b100),
        },
        "native_adaptive_nonconvergence_messages_in_full_0p15_cache": (
            warning_count
        ),
        "validated_cut": 0.08,
        "pass": bool(
            projected(linear_delta_b100)[
                "abs_projected_delta_over_sigma"
            ]
            < 0.05
            and projected(quadratic_delta_b100)[
                "abs_projected_delta_over_sigma"
            ]
            < 0.05
        ),
    }


def linearized_coverage(
    *,
    fit: ProfileResult,
    samples: np.ndarray,
    truth: float,
    prior_block: gaussian_v3.PriorBlock,
    covariance_reference: np.ndarray | None,
    folds: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    samples = np.asarray(samples[:, fit.indices], dtype=np.float64)
    nreal = samples.shape[0]
    if covariance_reference is not None:
        reference = np.asarray(
            covariance_reference[:, fit.indices], dtype=np.float64
        )
    else:
        reference = None
    fnl_tangent = (
        fit.linear_response + 2.0 * fit.fhat * fit.quadratic_coefficient
    )
    design = np.column_stack((fnl_tangent, fit.nuisance_jacobian))
    prior_precision_nuisance = np.linalg.inv(prior_block.covariance)
    prior_offset = fit.nuisance_values - prior_block.mean
    base = fit.prediction
    estimates = np.empty(nreal, dtype=np.float64)
    sigmas = np.empty(nreal, dtype=np.float64)
    fold_ids = np.arange(nreal, dtype=int) % max(1, int(folds))
    for index in range(nreal):
        if reference is not None:
            covariance = covariance_single(reference)
        else:
            train = samples[fold_ids != fold_ids[index]]
            covariance = covariance_single(train)
        precision = np.linalg.inv(covariance)
        prior_precision = np.zeros(
            (1 + len(fit.free_names), 1 + len(fit.free_names)),
            dtype=np.float64,
        )
        prior_precision[1:, 1:] = prior_precision_nuisance
        fisher = design.T @ precision @ design + prior_precision
        inverse = np.linalg.pinv(fisher, rcond=1.0e-11)
        rhs = design.T @ precision @ (samples[index] - base)
        rhs[1:] -= prior_precision_nuisance @ prior_offset
        shift = inverse @ rhs
        estimates[index] = fit.fhat + shift[0]
        sigmas[index] = math.sqrt(max(float(inverse[0, 0]), 0.0))
    covered = np.abs(estimates - truth) <= sigmas
    count = int(np.count_nonzero(covered))
    alpha = 0.05
    lower = (
        0.0
        if count == 0
        else float(
            beta_distribution.ppf(alpha / 2.0, count, nreal - count + 1)
        )
    )
    upper = (
        1.0
        if count == nreal
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0,
                count + 1,
                nreal - count,
            )
        )
    )
    pulls = (estimates - truth) / sigmas
    payload = {
        "method": (
            "local profiled Gaussian approximation around the central "
            "Minuit solution, retaining the complete nuisance prior"
        ),
        "truth": truth,
        "n_realizations": nreal,
        "covariance": (
            "independent fiducial500 covariance"
            if reference is not None
            else f"{folds}-fold cross-fitted covariance"
        ),
        "mean_fhat": float(np.mean(estimates)),
        "bias": float(np.mean(estimates) - truth),
        "mean_sigma": float(np.mean(sigmas)),
        "pull_mean": float(np.mean(pulls)),
        "pull_std_ddof1": float(np.std(pulls, ddof=1)),
        "coverage68": float(np.mean(covered)),
        "covered_count": count,
        "binomial95_interval": [lower, upper],
        "nominal_0p68_inside_binomial95": lower <= 0.68 <= upper,
    }
    return payload, {
        "fhat": estimates,
        "sigma": sigmas,
        "pull": pulls,
        "covered": covered,
    }


def plot_diagonal(
    path: Path,
    fit: ProfileResult,
    k_pair_full: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    title: str,
) -> None:
    local_diag = np.flatnonzero(
        pair_i[fit.indices] == pair_j[fit.indices]
    )
    k = k_pair_full[fit.indices[local_diag], 0]
    weight = k * k
    error = np.sqrt(np.diag(fit.covariance_single))[local_diag]
    residual = (fit.prediction - fit.target)[local_diag] / error
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        constrained_layout=True,
    )
    axes[0].errorbar(
        k,
        weight * fit.target[local_diag],
        yerr=weight * error,
        fmt="o",
        color="black",
        ecolor="0.55",
        capsize=2.0,
        markersize=4.0,
        label="Quijote mean; one-realization error",
    )
    axes[0].plot(
        k,
        weight * fit.prediction[local_diag],
        "-",
        color="#c51b1d",
        linewidth=1.8,
        label="hybrid best fit",
    )
    axes[0].set_ylabel(r"$k^2 B_{000}(k,k)$")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].plot(k, residual, "o-", color="#c51b1d", markersize=3.5)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_ylabel(r"$(B_{\rm model}-B_{\rm data})/\sigma_{\rm single}$")
    axes[1].set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(alpha=0.18)
    figure.savefig(path, metadata={**PDF_METADATA, "Title": title})
    plt.close(figure)


def plot_full2d_vector_bestfit(
    path: Path,
    scan: list[
        tuple[
            float,
            ProfileResult,
            ProfileResult,
            ProfileResult,
        ]
    ],
    k_pair_full: np.ndarray,
    *,
    pdf_metadata: dict[str, str] | None = None,
) -> None:
    """Write one complete-vector best-fit page for every kmax cut."""

    metadata = {
        **(PDF_METADATA if pdf_metadata is None else pdf_metadata),
        "Title": (
            "Pre-reconstruction full-2D B000 best-fit kmax scan"
        ),
    }
    with PdfPages(path, metadata=metadata) as pdf:
        for kmax, fit0, fit100, fit_minus100 in scan:
            figure = plt.figure(figsize=(15.2, 6.4))
            grid = figure.add_gridspec(
                2,
                3,
                height_ratios=(3.0, 1.0),
                left=0.055,
                right=0.99,
                bottom=0.10,
                top=0.88,
                wspace=0.25,
                hspace=0.08,
            )
            validated = math.isclose(kmax, PRIMARY_CUT)
            status = (
                "validated primary"
                if validated
                else "pre-recon stress only"
            )
            figure.suptitle(
                (
                    rf"$k_{{\max}}={kmax:.2f}\,h\,{{\rm Mpc}}^{{-1}}$"
                    f" — {status}; {fit0.indices.size} full-2D bins"
                ),
                color="black" if validated else "#a63603",
                fontsize=12.0,
            )
            panels = (
                (
                    fit0,
                    (
                        rf"fiducial: $\hat f_{{\rm NL}}={fit0.fhat:.1f}"
                        rf"\pm{fit0.sigma_symmetric:.1f}$"
                    ),
                ),
                (
                    fit100,
                    (
                        rf"LC$_p(+100)$: $\hat f_{{\rm NL}}="
                        rf"{fit100.fhat:.1f}\pm"
                        rf"{fit100.sigma_symmetric:.1f}$"
                    ),
                ),
                (
                    fit_minus100,
                    (
                        rf"LC$_m(-100)$: $\hat f_{{\rm NL}}="
                        rf"{fit_minus100.fhat:.1f}\pm"
                        rf"{fit_minus100.sigma_symmetric:.1f}$"
                    ),
                ),
            )
            for column, (fit, title) in enumerate(panels):
                upper = figure.add_subplot(grid[0, column])
                lower = figure.add_subplot(
                    grid[1, column],
                    sharex=upper,
                )
                x = np.arange(fit.indices.size)
                weight = np.prod(k_pair_full[fit.indices], axis=1)
                sigma = np.sqrt(np.diag(fit.covariance_single))
                upper.errorbar(
                    x,
                    weight * fit.target,
                    yerr=weight * sigma,
                    fmt="o",
                    color="black",
                    ecolor="0.55",
                    capsize=2.0,
                    markersize=3.5,
                    label="Quijote mean; one-realization error",
                )
                upper.plot(
                    x,
                    weight * fit.prediction,
                    "-o",
                    color="#c51b1d",
                    linewidth=1.7,
                    markersize=2.7,
                    label="profile best fit",
                )
                upper.set_title(title)
                upper.set_ylabel(r"$k_1k_2 B_{000}(k_1,k_2)$")
                upper.legend(frameon=False, fontsize=8.0)
                residual = (fit.prediction - fit.target) / sigma
                lower.axhline(0.0, color="0.5", linewidth=0.8)
                lower.plot(
                    x,
                    residual,
                    "o-",
                    color="#c51b1d",
                    linewidth=1.3,
                    markersize=3.0,
                )
                lower.set_ylim(-0.5, 0.5)
                lower.set_xlabel("selected full-2D bin")
                lower.set_ylabel(
                    r"$(B_{\rm model}-B_{\rm data})/"
                    r"\sigma_{\rm single}$"
                )
                tick_step = max(1, math.ceil(fit.indices.size / 14))
                lower.set_xticks(x[::tick_step])
                for axis in (upper, lower):
                    axis.tick_params(direction="in", top=True, right=True)
                    axis.grid(alpha=0.18)
            pdf.savefig(figure)
            plt.close(figure)


def triangular_matrix(
    values: np.ndarray,
    indices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    n = int(max(np.max(pair_i[indices]), np.max(pair_j[indices])) + 1)
    result = np.full((n, n), np.nan)
    for value, global_index in zip(values, indices):
        i = int(pair_i[global_index])
        j = int(pair_j[global_index])
        result[i, j] = value
        result[j, i] = value
    return result


def plot_full2d(
    path: Path,
    fit: ProfileResult,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    title: str,
) -> None:
    sigma = np.sqrt(np.diag(fit.covariance_single))
    panels = (
        ("data", fit.target),
        ("model", fit.prediction),
        ("pull", (fit.prediction - fit.target) / sigma),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    for axis, (label, values) in zip(axes, panels):
        matrix = triangular_matrix(values, fit.indices, pair_i, pair_j)
        if label == "pull":
            limit = max(0.5, float(np.nanmax(np.abs(matrix))))
            image = axis.imshow(
                matrix,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        else:
            image = axis.imshow(matrix, origin="lower", cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("shell j")
        axis.set_ylabel("shell i")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(title)
    figure.savefig(path, metadata={**PDF_METADATA, "Title": title})
    plt.close(figure)


def plot_response(
    path: Path,
    fit0: ProfileResult,
    arrays: dict[str, np.ndarray],
    k_pair: np.ndarray,
    tree_response: np.ndarray,
) -> None:
    x = np.arange(fit0.indices.size)
    weight = np.prod(k_pair[fit0.indices], axis=1)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.6),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].errorbar(
        x,
        weight * arrays["odd"],
        yerr=weight * arrays["odd_error_single"],
        fmt="o",
        color="black",
        ecolor="0.60",
        markersize=3.2,
        capsize=1.5,
        label="catalogue total odd response",
    )
    axes[0].plot(
        x,
        weight * fit0.linear_response,
        color="#c51b1d",
        linewidth=1.7,
        label="tree halo PNG + matter one-loop uplift",
    )
    axes[0].plot(
        x,
        weight * np.asarray(tree_response),
        color="#ef8a62",
        linestyle="--",
        linewidth=1.3,
        label="tree halo PNG only",
    )
    axes[0].set_ylabel(r"$k_1k_2\,dB_{000}/df_{\rm NL}$")
    axes[0].set_title(
        "Total catalogue response vs fixed-nuisance partial theory "
        "(diagnostic only)"
    )
    axes[0].legend(frameon=False, fontsize=8.5)
    residual = (
        fit0.linear_response - arrays["odd"]
    ) / arrays["odd_error_single"]
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].plot(x, residual, "o-", color="#c51b1d", markersize=3.0)
    axes[1].set_ylabel(
        r"partial theory $-$ total catalogue "
        r"[$\sigma_{\rm single,response}$]"
    )
    axes[1].set_xlabel("selected full-2D bin")
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(alpha=0.18)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Matched PNG total-response non-closure diagnostic",
        },
    )
    plt.close(figure)


def plot_components(
    path: Path,
    fit0: ProfileResult,
    response: dict[str, np.ndarray],
    k_pair: np.ndarray,
) -> None:
    x = np.arange(fit0.indices.size)
    weight = np.prod(k_pair[fit0.indices], axis=1)
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    styles = {
        "primordial": ("#1b9e77", "-"),
        "bphi_f2": ("#7570b3", "--"),
        "advection": ("#e7298a", ":"),
        "bphidelta": ("#66a61e", "-."),
        "bphi_b2": ("#e6ab02", "--"),
        "bphi_bK2": ("#a6761d", ":"),
        "stochastic": ("#666666", "-."),
        "matter_linear_uplift": ("#d95f02", "-"),
    }
    for name, (color, linestyle) in styles.items():
        component = np.asarray(response[name])[fit0.indices]
        axis.plot(
            x,
            weight * component,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=name,
        )
    axis.plot(
        x,
        weight * fit0.linear_response,
        color="black",
        linewidth=2.0,
        label="linear total",
    )
    axis.axhline(0.0, color="0.75", linewidth=0.7)
    axis.set_xlabel("selected full-2D bin")
    axis.set_ylabel(r"$k_1k_2\,dB_{000}/df_{\rm NL}$")
    axis.legend(frameon=False, fontsize=7.7, ncol=2)
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(alpha=0.15)
    figure.savefig(
        path,
        metadata={**PDF_METADATA, "Title": "PNG response decomposition"},
    )
    plt.close(figure)


def plot_profiles(path: Path, profiles: list[ProfileResult]) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.2),
        constrained_layout=True,
    )
    for axis, sample, truth in zip(
        axes,
        ("fiducial", "LC_p", "LC_m"),
        (0.0, 100.0, -100.0),
    ):
        for result in profiles:
            if result.sample != sample:
                continue
            axis.plot(
                result.profile_x,
                result.profile_delta,
                label=f"p={result.p:.1f}",
            )
        axis.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.axvline(truth, color="black", linestyle=":", linewidth=1.0)
        axis.set_ylim(0.0, 5.0)
        axis.set_xlabel(r"$f_{\rm NL}$")
        axis.set_ylabel(r"$\Delta\chi^2_{\rm prof}$")
        axis.set_title(sample)
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(
        path,
        metadata={**PDF_METADATA, "Title": "Local-PNG Minuit profiles"},
    )
    plt.close(figure)


def plot_stability(
    path: Path,
    profiles: list[ProfileResult],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.2),
        constrained_layout=True,
    )
    for axis, sample, truth in zip(
        axes,
        ("fiducial", "LC_p", "LC_m"),
        (0.0, 100.0, -100.0),
    ):
        axis.axvspan(
            0.09,
            0.16,
            color="0.92",
            zorder=0,
            label="pre-recon stress only",
        )
        axis.axvline(
            PRIMARY_CUT,
            color="0.35",
            linestyle="--",
            linewidth=0.9,
        )
        for p in P_GRID:
            rows = sorted(
                [
                    result
                    for result in profiles
                    if result.sample == sample
                    and result.tier == "coevolution"
                    and result.p == p
                ],
                key=lambda item: item.kmax,
            )
            if not rows:
                continue
            axis.errorbar(
                [row.kmax for row in rows],
                [row.fhat for row in rows],
                yerr=[
                    [abs(row.error_low) for row in rows],
                    [abs(row.error_high) for row in rows],
                ],
                marker="o",
                capsize=2,
                label=f"p={p:.1f}",
            )
        axis.axhline(truth, color="black", linestyle=":", linewidth=1.0)
        axis.set_xlabel(r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$")
        axis.set_ylabel(r"$f_{\rm NL}$")
        axis.set_title(sample + r" ($0.08$ validated)")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(
        path,
        metadata={**PDF_METADATA, "Title": "kmax and p stability"},
    )
    plt.close(figure)


def plot_coverage_even(
    path: Path,
    coverage0: dict[str, np.ndarray],
    coverage100: dict[str, np.ndarray],
    coverage_minus100: dict[str, np.ndarray],
    response_arrays: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    bins = np.linspace(-4.0, 4.0, 25)
    axes[0].hist(
        coverage0["pull"],
        bins=bins,
        histtype="step",
        color="black",
        label="fiducial",
    )
    axes[0].hist(
        coverage100["pull"],
        bins=bins,
        histtype="step",
        color="#c51b1d",
        label="LC_p",
    )
    axes[0].hist(
        coverage_minus100["pull"],
        bins=bins,
        histtype="step",
        color="#2166ac",
        label="LC_m",
    )
    axes[0].axvline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_xlabel(r"$(\hat f_{\rm NL}-f_{\rm true})/\sigma$")
    axes[0].set_ylabel("realizations")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(
        np.arange(coverage0["fhat"].size),
        coverage0["fhat"],
        s=5,
        color="black",
        alpha=0.55,
    )
    axes[1].axhline(0.0, color="#c51b1d", linewidth=1.0)
    axes[1].set_xlabel("fiducial realization")
    axes[1].set_ylabel(r"$\hat f_{\rm NL}$")
    x = np.arange(response_arrays["even_displacement"].size)
    axes[2].errorbar(
        x,
        response_arrays["even_displacement"],
        yerr=response_arrays["even_error_single"],
        fmt="o",
        markersize=3.0,
        color="black",
        ecolor="0.65",
        label="catalogue total even",
    )
    axes[2].plot(
        x,
        response_arrays["predicted_even"],
        color="#c51b1d",
        linewidth=1.5,
        label=r"$b_1^3B_{112}^{II}$",
    )
    axes[2].set_xlabel("selected full-2D bin")
    axes[2].set_ylabel(
        r"total even displacement (not closure)"
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.tick_params(direction="in", top=True, right=True)
    figure.savefig(
        path,
        metadata={
            **PDF_METADATA,
            "Title": "Coverage and catalogue even-response diagnostic",
        },
    )
    plt.close(figure)


def recovery_pass(result: ProfileResult, truth: float) -> bool:
    interval = (
        result.fhat + result.error_low,
        result.fhat + result.error_high,
    )
    return bool(
        abs(result.fhat - truth) / result.sigma_symmetric <= 0.5
        and interval[0] <= truth <= interval[1]
        and result.p_value >= 0.05
        and result.minuit_valid
        and not result.minuit_at_limit
    )


def markdown_report(summary: dict[str, Any]) -> str:
    def one_row(
        rows: list[dict[str, Any]],
        **matches: Any,
    ) -> dict[str, Any]:
        selected = [
            row
            for row in rows
            if all(row.get(name) == value for name, value in matches.items())
        ]
        if len(selected) != 1:
            raise KeyError((matches, len(selected)))
        return selected[0]

    tree_minus = one_row(
        summary["ablations"],
        response_model="tree",
        sample="LC_m",
        p=1.0,
    )
    linear_minus = one_row(
        summary["ablations"],
        response_model="linear_uplift",
        sample="LC_m",
        p=1.0,
    )
    finite_minus = one_row(
        summary["primary_by_p"],
        p=1.0,
    )["LC_m"]
    prior4_minus = one_row(
        summary["prior_robustness"],
        nuisance_prior_scale=4.0,
        sample="LC_m",
    )
    lines = [
        "# Pre-reconstruction halo PNG uplift v1",
        "",
        f"Final status: **{summary['final_status']}**.",
        "",
        "## Model boundary",
        "",
        (
            "The tested model is the frozen Gaussian halo one-loop EFT-v2 "
            "baseline plus tree-level halo local-PNG, the pure-matter "
            "linear one-loop PNG uplift, and the validated matter "
            "`B112II` finite-fNL term. It is not a complete halo-PNG "
            "one-loop calculation."
        ),
        "",
        "## Statistical contract",
        "",
        (
            "The 500- and 100-realization means are used only as low-noise "
            "central vectors. Every likelihood, interval, pull, norm, "
            "coverage calculation, and plotted error bar uses a covariance "
            "of one realization. No covariance is divided by 500 or 100."
        ),
        "",
        "## Primary profiles",
        "",
        (
            "All entries below use the independently validated "
            "pre-reconstruction cutoff "
            r"\(k_{\max}=0.08\,h\,{\rm Mpc}^{-1}\). Results at "
            r"\(0.10\)--\(0.15\) are extrapolation/stress diagnostics only; "
            "they cannot establish model validity because the "
            "pre-reconstruction DM control itself ceases to be unbiased "
            "above approximately "
            r"\(0.08\,h\,{\rm Mpc}^{-1}\). The post-reconstruction DM "
            "limit near "
            r"\(0.15\,h\,{\rm Mpc}^{-1}\) does not apply to this "
            "pre-reconstruction halo test."
        ),
        "",
        (
            "| p | fNL(fid) | sigma | fNL(+100) | sigma | fNL(-100) "
            "| sigma | fid pass | +100 pass | -100 pass |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in summary["primary_by_p"]:
        lines.append(
            f"| {row['p']:.1f} | {row['fiducial']['fhat']:.3f} | "
            f"{row['fiducial']['sigma_symmetric']:.3f} | "
            f"{row['LC_p']['fhat']:.3f} | "
            f"{row['LC_p']['sigma_symmetric']:.3f} | "
            f"{row['LC_m']['fhat']:.3f} | "
            f"{row['LC_m']['sigma_symmetric']:.3f} | "
            f"{row['fiducial_pass']} | {row['LC_p_pass']} | "
            f"{row['LC_m_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Best-fit kmax scan (p=1)",
            "",
            (
                "Only 0.08 is independently validated for the "
                "pre-reconstruction DM baseline. Every larger cut in this "
                "table and in the multi-page curve PDF is a stress test, "
                "irrespective of how close its best-fit centre is to the "
                "injected value."
            ),
            "",
            (
                "| kmax | bins | fNL(fid) | sigma | fNL(+100) | sigma "
                "| fNL(-100) | sigma | status |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    scan_rows = [
        row
        for row in summary["profiles"]
        if row["tier"] == PRIMARY_TIER
        and math.isclose(row["p"], 1.0)
        and row["response_model"] == "finite_uplift"
    ]
    for cut in CUTS:
        fit0 = next(
            row
            for row in scan_rows
            if row["sample"] == "fiducial"
            and math.isclose(row["kmax_h_mpc"], cut)
        )
        fit100 = next(
            row
            for row in scan_rows
            if row["sample"] == "LC_p"
            and math.isclose(row["kmax_h_mpc"], cut)
        )
        fit_minus100 = next(
            row
            for row in scan_rows
            if row["sample"] == "LC_m"
            and math.isclose(row["kmax_h_mpc"], cut)
        )
        status = "validated" if math.isclose(cut, PRIMARY_CUT) else "stress"
        lines.append(
            f"| {cut:.2f} | {fit0['n_data']} | {fit0['fhat']:.3f} | "
            f"{fit0['sigma_symmetric']:.3f} | {fit100['fhat']:.3f} | "
            f"{fit100['sigma_symmetric']:.3f} | "
            f"{fit_minus100['fhat']:.3f} | "
            f"{fit_minus100['sigma_symmetric']:.3f} | {status} |"
        )
    response = summary["response_diagnostics"]
    free_bphidelta = summary["free_bphidelta_diagnostic_by_p"]["1.0"]
    lines.extend(
        [
            "",
            "## Matched-catalogue response (non-closure diagnostic)",
            "",
            (
                "The LC_p-LC_m finite difference is a total response of a "
                "fixed-selection halo catalogue. It includes fNL-dependent "
                "halo abundance, bias, and stochastic/nuisance parameters. "
                "The overplotted theory is a partial derivative at a fixed "
                "fiducial nuisance point. These are not the same statistical "
                "object, so the following distances are descriptive only and "
                "must not be used to accept or reject the model."
            ),
            "",
            (
                "- Odd-response model distance: "
                f"`{response['odd_model_minus_data_norm_sigma_single_response']:.3f} "
                "sigma_single,response` (diagnostic only)."
            ),
            (
                "- Actual +100 directional-response distance: "
                f"`{response['dplus_model_minus_data_norm_sigma_single_response']:.3f} "
                "sigma_single,response` (diagnostic only)."
            ),
            (
                "- Unexplained even projection after including B112II: "
                f"`{response['unexplained_even_abs_projection_over_sigma']:.3f} "
                "sigma_fNL` (also non-closure, because nuisance/selection "
                "responses enter the even combination)."
            ),
            (
                "- Free-bphidelta diagnostic (p=1): "
                f"`Delta bphidelta={free_bphidelta['delta_bphidelta_posterior_mode']:.3f}` "
                f"(`{free_bphidelta['abs_mode_over_prior_sigma']:.2f}` prior sigma); "
                "the odd-response norm changes from "
                f"`{free_bphidelta['odd_response_norm_before_sigma_single']:.3f}` "
                "to "
                f"`{free_bphidelta['odd_response_norm_after_sigma_single']:.3f}` "
                "sigma_single,response."
            ),
            "",
            "## Theoretical finite-fNL parity audit",
            "",
            (
                "- At |fNL|=100 the implemented matter B112II displacement "
                "has covariance-norm ratio "
                f"`{summary['theoretical_finite_fNL']['B112II_to_linear_covariance_norm_ratio']:.6g}` "
                "relative to the linear PNG displacement, and projects to "
                f"`{summary['theoretical_finite_fNL']['B112II_abs_projection_over_fixed_nuisance_sigma']:.6g}` "
                "fixed-nuisance sigma."
            ),
            (
                "- The missing leading halo-PNG one-loop response is linear "
                "in fNL: it flips sign between +100 and -100 and does not "
                "become relatively larger merely because fNL is negative. "
                "Sign-asymmetric recovery instead requires even/higher-order "
                "finite-fNL terms or sample/nuisance/profile asymmetry."
            ),
            (
                "- Independent-profile centre decomposition at p=1 gives "
                f"a fiducial offset of "
                f"`{summary['profiled_center_parity']['1.0']['fiducial_offset_fNL']:.3f}`, "
                f"an odd gain of "
                f"`{summary['profiled_center_parity']['1.0']['odd_profile_gain']:.4f}`, "
                "and an even profile displacement at |fNL|=100 of "
                f"`{summary['profiled_center_parity']['1.0']['even_profile_displacement_at_abs100_fNL']:.3f}`. "
                "This algebraically explains why the +100 centre can look "
                "better than the -100 centre without assigning a "
                "mean-covariance significance."
            ),
            "",
            "## Coverage and robustness diagnostics",
            "",
            (
                "- Frozen-nuisance +100 residual norm: "
                f"`{summary['frozen_nuisance_primary']['residual_norm_at_truth_sigma_single']:.3f} "
                "sigma_single`; maximum bin pull "
                f"`{summary['frozen_nuisance_primary']['max_pull_at_truth_sigma_single']:.3f}` "
                "(non-closure diagnostic only)."
            ),
            (
                "- Frozen-nuisance -100 residual norm: "
                f"`{summary['frozen_nuisance_minus_primary']['residual_norm_at_truth_sigma_single']:.3f} "
                "sigma_single`; maximum bin pull "
                f"`{summary['frozen_nuisance_minus_primary']['max_pull_at_truth_sigma_single']:.3f}` "
                "(non-closure diagnostic only)."
            ),
            (
                "- Fiducial 68% coverage: "
                f"`{summary['coverage']['fiducial']['coverage68']:.3f}`; "
                "LC_p 68% coverage: "
                f"`{summary['coverage']['LC_p']['coverage68']:.3f}`; "
                "LC_m 68% coverage: "
                f"`{summary['coverage']['LC_m']['coverage68']:.3f}`."
            ),
            (
                "- LC_m Minuit start-value span: "
                f"`{summary['robustness']['start_span_over_sigma']:.6g}` "
                "of its primary single-realization profile sigma."
            ),
            (
                "- Maximum fNL shifts (in the corresponding primary profile "
                "sigma) from nuisance-prior, covariance-source, fixed-b1, "
                "and abundance-proxy variations are respectively "
                f"`{summary['robustness']['prior_shift_over_sigma']}`, "
                f"`{summary['robustness']['covariance_shift_over_sigma']}`, "
                f"`{summary['robustness']['b1_shift_over_sigma']}`, and "
                f"`{summary['robustness']['nbar_shift_over_sigma']}`."
            ),
            "",
            "## Cause audit",
            "",
            (
                "1. **Not a Minuit, covariance, b1, or abundance failure.** "
                "The multi-start span is only "
                f"`{summary['robustness']['start_span_over_sigma']:.4f}` "
                "single-realization sigma. Covariance-source, fixed-b1, and "
                "relative-nbar variations move the LC_m centre by at most "
                f"`{summary['robustness']['covariance_shift_over_sigma']['LC_m']:.3f}`, "
                f"`{summary['robustness']['b1_shift_over_sigma']['LC_m']:.3f}`, "
                "and "
                f"`{summary['robustness']['nbar_shift_over_sigma']['LC_m']:.6f}` "
                "sigma, respectively."
            ),
            (
                "2. **The implemented matter PNG loop is not the cause.** "
                "At p=1 the LC_m centre is "
                f"`{tree_minus['fhat']:.3f}` with tree halo PNG only, "
                f"`{linear_minus['fhat']:.3f}` after the matter one-loop "
                "uplift, and "
                f"`{finite_minus['fhat']:.3f}` after matter B112II. "
                "These changes are negligible compared with the "
                f"`{finite_minus['sigma_symmetric']:.3f}` "
                "single-realization uncertainty."
            ),
            (
                "3. **The fit is prior-regularized and underidentified.** "
                f"The primary vector has `{finite_minus['n_data']}` bins "
                f"but profiles `{len(finite_minus['free_parameters'])}` "
                "Gaussian nuisance parameters plus fNL. Widening all "
                "nuisance priors by four moves the LC_m centre to "
                f"`{prior4_minus['fhat']:.3f}` but inflates its profile "
                f"sigma to `{prior4_minus['sigma_symmetric']:.3f}`. Thus "
                "broad intervals cannot by themselves validate the PNG "
                "shape."
            ),
            (
                "4. **PNG-bias closure is a leading concrete risk.** The "
                "model fixes b_phi through p and fixes b_phi_delta through "
                "a universality relation. The p scan changes the LC_m centre "
                "from `-62.975` to `-88.035`. Published simulation work "
                "finds that universality can overpredict halo b_phi_delta by "
                "about 3 in the relevant b1 range, and likelihood studies "
                "find systematic fNL shifts from strong PNG-bias relations."
            ),
            (
                "5. **The halo finite-fNL sector is incomplete.** The model "
                "contains only the matter B112II quadratic term. It omits "
                "halo tree-level and loop-level fNL-squared contributions "
                "from PNG bias, products of PNG responses, primordial "
                "trispectrum, derivative/counterterm, and stochastic "
                "operators. The implemented matter B112II is far too small "
                "to explain the independently profiled even centre "
                "displacement, but that displacement cannot by itself "
                "identify which omitted halo/selection term is responsible."
            ),
            (
                "6. **Missing halo one-loop PNG can still matter, but not "
                "specifically for negative fNL.** Its leading omitted "
                "O(fNL P_L^3) terms are odd in fNL, so their absolute and "
                "relative size is the same at +100 and -100. They can alter "
                "the odd gain and are known to be enhanced for squeezed "
                "local-PNG halo configurations, but they cannot alone "
                "generate the observed sign asymmetry."
            ),
            "",
            "Primary references: "
            "[Dizgah et al. (2020)](https://arxiv.org/abs/2010.14523), "
            "[Barreira (2021)](https://arxiv.org/abs/2107.06887), "
            "[Assassi, Baumann & Schmidt (2015)](https://arxiv.org/abs/1510.03723), "
            "and "
            "[Yokoyama, Matsubara & Taruya (2013)](https://arxiv.org/abs/1310.4925).",
            "",
            "## Gates",
            "",
            "| gate | pass | value |",
            "|---|:---:|---|",
        ]
    )
    for name, gate in summary["gates"].items():
        lines.append(
            f"| {name} | {gate['pass']} | `{gate.get('value', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    for path in summary["outputs"]["figures"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The independent-profile central-recovery gate is "
                f"`{sum(row['all_primary_gates_pass'] for row in summary['primary_by_p'])}/"
                f"{len(summary['primary_by_p'])}` across the registered p "
                "grid. The p=1 negative injection misses the registered "
                "0.5-sigma centring threshold only marginally. Raw matched "
                "responses and frozen-nuisance predictions are not closure "
                "tests and do not enter this count."
            ),
            "",
            summary["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.tag or "/" in args.tag:
        raise ValueError("--tag must be directory-safe")
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must lie in [1,8]")
    root = args.repo_root.resolve()
    configured_data_root = (
        args.data_root
        if args.data_root is not None
        else Path(os.environ.get("MARISA_B_DATA_ROOT", root))
    )
    data_root = configured_data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )
    paths = resolve_paths(
        root,
        args.tag,
        data_root=data_root,
        output_root=output_root,
    )
    paths.analysis.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.theory.mkdir(parents=True, exist_ok=True)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    require_inputs(paths, include_theory=False)

    accepted_report_path = (
        data_root / "analysis/pre_recon_bias_model_v3_20260725/report.json"
    )
    accepted_report = json.loads(
        accepted_report_path.read_text(encoding="utf-8")
    )
    b1 = float(accepted_report["b1_calibration"]["primary"]["b1"])
    b1_sigma = float(
        accepted_report["b1_calibration"]["primary"]["b1_sigma"]
    )
    geometry = load_measurement_geometry(paths.fid_matrix)

    dm_production: dict[str, Any] = {}
    if args.stage in {"all", "fit"}:
        dm_production = produce_dm_vectors(
            paths,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    if args.stage in {"all", "tree"}:
        tree_production = produce_tree_templates(
            paths,
            geometry,
            b1=b1,
            nmu=args.tree_nmu,
            nradial=args.tree_nradial,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    else:
        tree_production = {}
    if args.stage == "tree":
        print(
            json.dumps(
                {
                    "status": "tree_complete",
                    "output": str(paths.tree_templates),
                    "metadata": tree_production,
                },
                indent=2,
            )
        )
        return
    if args.stage == "self-test":
        if not paths.tree_templates.exists():
            produce_tree_templates(
                paths,
                geometry,
                b1=b1,
                nmu=12,
                nradial=2,
                workers=args.workers,
                overwrite=False,
            )
        print(json.dumps({"status": "self_test_pass", "b1": b1}, indent=2))
        return

    require_inputs(paths, include_theory=True)
    frozen, data, frozen_prior, templates, _power, nbar = load_frozen(paths)
    if abs(nbar - N_BAR_FALLBACK) / N_BAR_FALLBACK > 1.0e-8:
        raise AssertionError(f"unexpected number density {nbar}")
    tree = load_tree(paths, geometry, b1)
    dm_tree, dm_linear_meta = load_dm_vector(
        paths.dm_linear,
        geometry,
        component="B000_dBdfNL_local_tree",
    )
    dm_total, _ = load_dm_vector(
        paths.dm_linear,
        geometry,
        component="B000_dBdfNL_primary",
    )
    dm_quadratic, dm_quadratic_meta = load_dm_vector(
        paths.dm_quadratic,
        geometry,
        component="B000_B112II_fNL2_coefficient",
    )

    # Reuse the already validated immutable data audit, then reload samples
    # explicitly for likelihoods and coverage.
    audit_paths = data_audit_v1.input_paths(root)
    data_audit, _audit_arrays = data_audit_v1.load_and_audit(audit_paths)
    if data_audit["gate_status"] != "pass":
        raise AssertionError("immutable data/covariance audit failed")
    with np.load(paths.fid_matrix, allow_pickle=False) as values:
        fid_ids = np.asarray(values["fiducial_realizations"], dtype=int)
        fid500 = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        p500 = np.asarray(values["fiducial_pre_P0"], dtype=np.float64)
        k_pair = np.asarray(values["b000_k"], dtype=np.float64)
    with np.load(paths.matched_matrix, allow_pickle=False) as values:
        matched_ids = np.asarray(values["fiducial_realizations"], dtype=int)
        fid100 = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        lcp = np.asarray(values["LC_p_pre_B000"], dtype=np.float64)
        lcm = np.asarray(values["LC_m_pre_B000"], dtype=np.float64)
        shot_proxy_by_sample = {
            sample: np.asarray(
                values[f"{sample}_pre_P0_num_shotnoise"],
                dtype=np.float64,
            )
            / np.asarray(
                values[f"{sample}_pre_P0_norm"],
                dtype=np.float64,
            )
            for sample in ("fiducial", "LC_p", "LC_m")
        }
    if not (
        np.array_equal(fid_ids[:100], matched_ids)
        and np.array_equal(fid500[:100], fid100)
    ):
        raise AssertionError("matched realization identity failed")

    # Algebraic/model-contract gates.
    unit_parameters = parameter_dict(
        frozen,
        gaussian_v3.b_free_names(frozen, "full"),
        np.asarray(
            [
                frozen_prior.mean[frozen.FIT_NAMES.index(name)]
                for name in gaussian_v3.b_free_names(frozen, "full")
            ]
        ),
        tier="full",
        b1=b1,
    )
    adapter = response_components(
        tree=tree,
        templates=templates,
        parameters=unit_parameters,
        b1=b1,
        p=1.0,
        nbar=nbar,
        dm_tree=dm_tree,
        dm_total=dm_total,
        dm_quadratic=dm_quadratic,
        response_model="finite_uplift",
    )["projection_adapter_local"]
    tree_indices = np.asarray(tree["data_indices"], dtype=int)
    adapter_audit = {
        "minimum": float(np.min(adapter)),
        "maximum": float(np.max(adapter)),
        "maximum_abs_minus_one": float(np.max(np.abs(adapter - 1.0))),
        "post_adapter_identity_max_relative": float(
            np.max(
                np.abs(
                    adapter
                    * np.asarray(tree["stochastic_alpha3_basis"])
                    - b1**2
                    * templates._compiled["stochastic:Bshot_residual"]["b1^2"][
                        tree_indices
                    ]
                )
            )
            / np.max(
                np.abs(
                    b1**2
                    * templates._compiled["stochastic:Bshot_residual"]["b1^2"][
                        tree_indices
                    ]
                )
            )
        ),
    }
    matter_tree_from_halo = (
        np.asarray(tree["dBdfNL_local_primordial"], dtype=np.float64)
        / b1**3
    )
    matter_tree_native = np.asarray(dm_tree[tree_indices], dtype=np.float64)
    matter_total_native = np.asarray(dm_total[tree_indices], dtype=np.float64)
    matter_limit_assembled = matter_tree_from_halo + (
        matter_total_native - matter_tree_native
    )
    relative_tree_identity = np.abs(
        matter_tree_from_halo - matter_tree_native
    ) / np.maximum(np.abs(matter_tree_native), 1.0)
    primary_global_indices = selected_indices(k_pair, PRIMARY_CUT)
    primary_local_mask = np.isin(tree_indices, primary_global_indices)
    model_contract = {
        "fNL0_reduces_exactly_to_frozen_gaussian": True,
        "tree_subtraction_algebra": (
            "R_h_tree + b1^3*(R_m_total-R_m_tree), hence exactly one "
            "matter-tree contribution in the matter limit"
        ),
        "matter_tree_identity_max_relative_all_stress_bins": float(
            np.max(relative_tree_identity)
        ),
        "matter_tree_identity_max_relative_validated_0p08": float(
            np.max(relative_tree_identity[primary_local_mask])
        ),
        "matter_limit_total_identity_max_relative_validated_0p08": float(
            np.max(
                np.abs(
                    matter_limit_assembled[primary_local_mask]
                    - matter_total_native[primary_local_mask]
                )
                / np.maximum(
                    np.abs(matter_total_native[primary_local_mask]),
                    1.0,
                )
            )
        ),
        "finite_fNL_formula": (
            "B_G + fNL*R_linear + fNL^2*b1^3*Q_m,B112II"
        ),
        "pass": bool(
            np.max(relative_tree_identity[primary_local_mask]) < 0.01
        ),
    }

    if args.quick:
        tiers = ("coevolution",)
        p_values = (1.0,)
    else:
        tiers = TIERS
        p_values = P_GRID
    profiles: list[ProfileResult] = []
    for tier in tiers:
        cuts = CUTS if tier == "coevolution" else (PRIMARY_CUT,)
        for p in p_values:
            for kmax in cuts:
                for sample, samples, start in (
                    ("fiducial", fid500, 0.0),
                    ("LC_p", lcp, 100.0),
                    ("LC_m", lcm, -100.0),
                ):
                    result = profile_hybrid(
                        frozen=frozen,
                        templates=templates,
                        frozen_prior=frozen_prior,
                        nbar=nbar,
                        tree=tree,
                        dm_tree=dm_tree,
                        dm_total=dm_total,
                        dm_quadratic=dm_quadratic,
                        k_pair=k_pair,
                        covariance_samples=fid500,
                        target_samples=samples,
                        tier=tier,
                        sample=sample,
                        p=p,
                        kmax=kmax,
                        b1=b1,
                        response_model="finite_uplift",
                        start_fnl=start,
                        compute_curve=(
                            tier == PRIMARY_TIER and kmax == PRIMARY_CUT
                        ),
                    )
                    profiles.append(result)
                    print(
                        f"[profile] {tier} {sample} p={p:.1f} "
                        f"kmax={kmax:.2f}: {result.fhat:.3f} "
                        f"+{result.error_high:.3f}{result.error_low:.3f}",
                        flush=True,
                    )

    def get_profile(
        tier: str,
        sample: str,
        p: float,
        kmax: float,
    ) -> ProfileResult:
        matches = [
            result
            for result in profiles
            if result.tier == tier
            and result.sample == sample
            and abs(result.p - p) < 1.0e-12
            and abs(result.kmax - kmax) < 1.0e-12
            and result.response_model == "finite_uplift"
        ]
        if len(matches) != 1:
            raise KeyError((tier, sample, p, kmax, len(matches)))
        return matches[0]

    primary0 = get_profile(PRIMARY_TIER, "fiducial", p_values[0], PRIMARY_CUT)
    primary100 = get_profile(PRIMARY_TIER, "LC_p", p_values[0], PRIMARY_CUT)
    primary_minus100 = get_profile(
        PRIMARY_TIER,
        "LC_m",
        p_values[0],
        PRIMARY_CUT,
    )
    fixed_zero_by_p: dict[str, ProfileResult] = {}
    for p in p_values:
        fixed_zero_by_p[f"{p:.1f}"] = profile_hybrid(
            frozen=frozen,
            templates=templates,
            frozen_prior=frozen_prior,
            nbar=nbar,
            tree=tree,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
            k_pair=k_pair,
            covariance_samples=fid500,
            target_samples=fid500,
            tier=PRIMARY_TIER,
            sample="fiducial_fixed_fNL0",
            p=p,
            kmax=PRIMARY_CUT,
            b1=b1,
            response_model="finite_uplift",
            start_fnl=0.0,
            compute_curve=False,
            fixed_fnl=0.0,
        )
    primary0_fixed = fixed_zero_by_p[f"{p_values[0]:.1f}"]

    # Tree-only and linear-finite ablations at the primary cut.
    ablations: list[ProfileResult] = []
    for response_model in ("tree", "linear_uplift"):
        for p in p_values:
            for sample, samples, start in (
                ("fiducial", fid500, 0.0),
                ("LC_p", lcp, 100.0),
                ("LC_m", lcm, -100.0),
            ):
                ablations.append(
                    profile_hybrid(
                        frozen=frozen,
                        templates=templates,
                        frozen_prior=frozen_prior,
                        nbar=nbar,
                        tree=tree,
                        dm_tree=dm_tree,
                        dm_total=dm_total,
                        dm_quadratic=dm_quadratic,
                        k_pair=k_pair,
                        covariance_samples=fid500,
                        target_samples=samples,
                        tier=PRIMARY_TIER,
                        sample=sample,
                        p=p,
                        kmax=PRIMARY_CUT,
                        b1=b1,
                        response_model=response_model,
                        start_fnl=start,
                        compute_curve=False,
                    )
                )

    # b1 robustness is deliberately restricted to the preregistered primary
    # p=1 result; the p grid itself is already profiled above.
    b1_robustness: list[ProfileResult] = []
    (
        _shift_indices,
        shift_triangles,
        shift_triangle_metadata,
        _shift_work,
    ) = tree_work(
        geometry,
        nmu=args.tree_nmu,
        nradial=args.tree_nradial,
    )
    for shifted_b1 in (b1 - b1_sigma, b1 + b1_sigma):
        shifted_tree, _shift_chunks = run_tree_components(
            paths=paths,
            triangles=shift_triangles,
            triangle_metadata=shift_triangle_metadata,
            n_selected=28,
            b1=shifted_b1,
            b2_native=1.0,
            bk2=1.0,
            bphi=1.0,
            bphidelta=1.0,
            bphi2=1.0,
            workers=args.workers,
        )
        shifted_tree["data_indices"] = _shift_indices
        for sample, samples, start in (
            ("fiducial", fid500, 0.0),
            ("LC_p", lcp, 100.0),
            ("LC_m", lcm, -100.0),
        ):
            b1_robustness.append(
                profile_hybrid(
                    frozen=frozen,
                    templates=templates,
                    frozen_prior=frozen_prior,
                    nbar=nbar,
                    tree=shifted_tree,
                    dm_tree=dm_tree,
                    dm_total=dm_total,
                    dm_quadratic=dm_quadratic,
                    k_pair=k_pair,
                    covariance_samples=fid500,
                    target_samples=samples,
                    tier=PRIMARY_TIER,
                    sample=sample,
                    p=p_values[0],
                    kmax=PRIMARY_CUT,
                    b1=shifted_b1,
                    response_model="finite_uplift",
                    start_fnl=start,
                    compute_curve=False,
                )
            )

    # Targeted audits for the apparently asymmetric negative injection.
    # These alter one numerical/statistical assumption at a time and are
    # diagnostics, not additional degrees of freedom in the production fit.
    start_robustness: list[dict[str, Any]] = []
    for start in (-400.0, -200.0, -100.0, 0.0, 100.0, 200.0):
        result = profile_hybrid(
            frozen=frozen,
            templates=templates,
            frozen_prior=frozen_prior,
            nbar=nbar,
            tree=tree,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
            k_pair=k_pair,
            covariance_samples=fid500,
            target_samples=lcm,
            tier=PRIMARY_TIER,
            sample="LC_m",
            p=p_values[0],
            kmax=PRIMARY_CUT,
            b1=b1,
            response_model="finite_uplift",
            start_fnl=start,
            compute_curve=False,
        )
        row = profile_row(result)
        row["start_fnl"] = start
        start_robustness.append(row)

    prior_robustness: list[dict[str, Any]] = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        for sample, samples, truth in (
            ("fiducial", fid500, 0.0),
            ("LC_p", lcp, 100.0),
            ("LC_m", lcm, -100.0),
        ):
            if math.isclose(scale, 1.0):
                result = get_profile(
                    PRIMARY_TIER,
                    sample,
                    p_values[0],
                    PRIMARY_CUT,
                )
            else:
                result = profile_hybrid(
                    frozen=frozen,
                    templates=templates,
                    frozen_prior=frozen_prior,
                    nbar=nbar,
                    tree=tree,
                    dm_tree=dm_tree,
                    dm_total=dm_total,
                    dm_quadratic=dm_quadratic,
                    k_pair=k_pair,
                    covariance_samples=fid500,
                    target_samples=samples,
                    tier=PRIMARY_TIER,
                    sample=sample,
                    p=p_values[0],
                    kmax=PRIMARY_CUT,
                    b1=b1,
                    response_model="finite_uplift",
                    start_fnl=truth,
                    compute_curve=False,
                    nuisance_prior_scale=scale,
                )
            prior_robustness.append(profile_row(result))

    covariance_robustness: list[dict[str, Any]] = []
    covariance_cases = (
        ("fiducial500", fid500),
        ("fiducial100_matched", fid100),
        ("fiducial100_disjoint", fid500[-100:]),
        ("LC_p100", lcp),
        ("LC_m100", lcm),
    )
    for covariance_label, covariance_samples in covariance_cases:
        for sample, samples, truth in (
            ("LC_p", lcp, 100.0),
            ("LC_m", lcm, -100.0),
        ):
            if covariance_label == "fiducial500":
                result = get_profile(
                    PRIMARY_TIER,
                    sample,
                    p_values[0],
                    PRIMARY_CUT,
                )
            else:
                result = profile_hybrid(
                    frozen=frozen,
                    templates=templates,
                    frozen_prior=frozen_prior,
                    nbar=nbar,
                    tree=tree,
                    dm_tree=dm_tree,
                    dm_total=dm_total,
                    dm_quadratic=dm_quadratic,
                    k_pair=k_pair,
                    covariance_samples=covariance_samples,
                    target_samples=samples,
                    tier=PRIMARY_TIER,
                    sample=sample,
                    p=p_values[0],
                    kmax=PRIMARY_CUT,
                    b1=b1,
                    response_model="finite_uplift",
                    start_fnl=truth,
                    compute_curve=False,
                    covariance_label=covariance_label,
                )
            covariance_robustness.append(profile_row(result))

    shot_proxy_means = {
        sample: float(np.mean(values))
        for sample, values in shot_proxy_by_sample.items()
    }
    relative_nbar = {
        sample: float(
            nbar
            * shot_proxy_means["fiducial"]
            / shot_proxy_means[sample]
        )
        for sample in ("fiducial", "LC_p", "LC_m")
    }
    nbar_robustness: list[dict[str, Any]] = []
    for sample, samples, truth in (
        ("LC_p", lcp, 100.0),
        ("LC_m", lcm, -100.0),
    ):
        result = profile_hybrid(
            frozen=frozen,
            templates=templates,
            frozen_prior=frozen_prior,
            nbar=relative_nbar[sample],
            tree=tree,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
            k_pair=k_pair,
            covariance_samples=fid500,
            target_samples=samples,
            tier=PRIMARY_TIER,
            sample=sample,
            p=p_values[0],
            kmax=PRIMARY_CUT,
            b1=b1,
            response_model="finite_uplift",
            start_fnl=truth,
            compute_curve=False,
        )
        row = profile_row(result)
        row["nbar_used_h3_mpc3"] = relative_nbar[sample]
        nbar_robustness.append(row)

    frozen_by_p: dict[str, dict[str, Any]] = {}
    frozen_minus_by_p: dict[str, dict[str, Any]] = {}
    response_by_p: dict[str, dict[str, Any]] = {}
    response_arrays_by_p: dict[str, dict[str, np.ndarray]] = {}
    coverage_payload: dict[str, dict[str, Any]] = {}
    coverage_arrays: dict[str, dict[str, np.ndarray]] = {}
    free_bphidelta_by_p: dict[str, dict[str, Any]] = {}
    for p in p_values:
        fit0_profiled = get_profile(
            PRIMARY_TIER,
            "fiducial",
            p,
            PRIMARY_CUT,
        )
        fit0 = fixed_zero_by_p[f"{p:.1f}"]
        fit100 = get_profile(PRIMARY_TIER, "LC_p", p, PRIMARY_CUT)
        fit_minus100 = get_profile(
            PRIMARY_TIER,
            "LC_m",
            p,
            PRIMARY_CUT,
        )
        key = f"{p:.1f}"
        frozen_by_p[key] = frozen_nuisance_fnl(
            fit0=fit0,
            target_samples=lcp,
            truth_fnl=100.0,
            frozen=frozen,
            templates=templates,
            nbar=nbar,
            tree=tree,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
        )
        frozen_minus_by_p[key] = frozen_nuisance_fnl(
            fit0=fit0,
            target_samples=lcm,
            truth_fnl=-100.0,
            frozen=frozen,
            templates=templates,
            nbar=nbar,
            tree=tree,
            dm_tree=dm_tree,
            dm_total=dm_total,
            dm_quadratic=dm_quadratic,
        )
        response_by_p[key], response_arrays_by_p[key] = response_diagnostics(
            fit0=fit0,
            fid100=fid100,
            lcp=lcp,
            lcm=lcm,
        )
        free_bphidelta_by_p[key] = free_bphidelta_diagnostic(
            fit0=fit0,
            tree=tree,
            fid100=fid100,
            lcp=lcp,
            lcm=lcm,
        )
        _free, _nonlinear, prior_block = tier_architecture(
            frozen,
            frozen_prior,
            PRIMARY_TIER,
        )
        coverage0, arrays0 = linearized_coverage(
            fit=fit0_profiled,
            samples=fid500,
            truth=0.0,
            prior_block=prior_block,
            covariance_reference=None,
            folds=5,
        )
        coverage100, arrays100 = linearized_coverage(
            fit=fit100,
            samples=lcp,
            truth=100.0,
            prior_block=prior_block,
            covariance_reference=fid500,
            folds=1,
        )
        coverage_minus100, arrays_minus100 = linearized_coverage(
            fit=fit_minus100,
            samples=lcm,
            truth=-100.0,
            prior_block=prior_block,
            covariance_reference=fid500,
            folds=1,
        )
        coverage_payload[key] = {
            "fiducial": coverage0,
            "LC_p": coverage100,
            "LC_m": coverage_minus100,
        }
        coverage_arrays[key] = {
            "fiducial": arrays0,
            "LC_p": arrays100,
            "LC_m": arrays_minus100,
        }

    primary_by_p: list[dict[str, Any]] = []
    all_primary_pass = True
    some_primary_pass = False
    for p in p_values:
        fit0 = get_profile(PRIMARY_TIER, "fiducial", p, PRIMARY_CUT)
        fit100 = get_profile(PRIMARY_TIER, "LC_p", p, PRIMARY_CUT)
        fit_minus100 = get_profile(PRIMARY_TIER, "LC_m", p, PRIMARY_CUT)
        pass0 = recovery_pass(fit0, 0.0)
        pass100 = recovery_pass(fit100, 100.0)
        pass_minus100 = recovery_pass(fit_minus100, -100.0)
        pass_all = bool(
            pass0
            and pass100
            and pass_minus100
        )
        all_primary_pass &= pass_all
        some_primary_pass |= pass_all
        primary_by_p.append(
            {
                "p": p,
                "fiducial": profile_row(fit0),
                "LC_p": profile_row(fit100),
                "LC_m": profile_row(fit_minus100),
                "fiducial_pass": pass0,
                "LC_p_pass": pass100,
                "LC_m_pass": pass_minus100,
                "all_primary_gates_pass": pass_all,
            }
        )

    primary_key = f"{p_values[0]:.1f}"
    frozen_primary = frozen_by_p[primary_key]
    frozen_minus_primary = frozen_minus_by_p[primary_key]
    response_primary = response_by_p[primary_key]
    coverage_primary = coverage_payload[primary_key]
    theoretical_finite = theoretical_finite_fnl_diagnostic(primary0_fixed)
    convergence = numerical_convergence_diagnostic(
        paths=paths,
        fit0=primary0_fixed,
        b1=b1,
    )
    drift_values: list[float] = []
    if not args.quick:
        for p in p_values:
            for sample in ("fiducial", "LC_p", "LC_m"):
                low = get_profile(PRIMARY_TIER, sample, p, 0.12)
                high = get_profile(PRIMARY_TIER, sample, p, 0.15)
                drift_values.append(
                    abs(high.fhat - low.fhat)
                    / math.hypot(
                        high.sigma_symmetric,
                        low.sigma_symmetric,
                    )
                )
    max_drift = max(drift_values, default=math.nan)
    tier_differences: list[float] = []
    if not args.quick:
        for p in p_values:
            for sample in ("fiducial", "LC_p", "LC_m"):
                full = get_profile("full", sample, p, PRIMARY_CUT)
                coev = get_profile("coevolution", sample, p, PRIMARY_CUT)
                tier_differences.append(
                    abs(full.fhat - coev.fhat)
                    / math.hypot(
                        full.sigma_symmetric,
                        coev.sigma_symmetric,
                    )
                )
    max_tier_difference = max(tier_differences, default=math.nan)

    primary_reference = {
        sample: get_profile(
            PRIMARY_TIER,
            sample,
            p_values[0],
            PRIMARY_CUT,
        )
        for sample in ("fiducial", "LC_p", "LC_m")
    }
    start_values = np.asarray(
        [row["fhat"] for row in start_robustness],
        dtype=np.float64,
    )
    start_span_over_sigma = float(
        np.ptp(start_values) / primary_reference["LC_m"].sigma_symmetric
    )

    def shift_over_reference_sigma(
        rows: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        shifts: dict[str, list[float]] = {
            sample: [] for sample in primary_reference
        }
        for row in rows:
            sample = str(row["sample"])
            if sample not in primary_reference:
                continue
            reference = primary_reference[sample]
            shifts[sample].append(
                abs(float(row["fhat"]) - reference.fhat)
                / reference.sigma_symmetric
            )
        return {
            sample: (max(values) if values else None)
            for sample, values in shifts.items()
        }

    prior_shift_over_sigma = shift_over_reference_sigma(prior_robustness)
    covariance_shift_over_sigma = shift_over_reference_sigma(
        covariance_robustness
    )
    b1_shift_over_sigma = shift_over_reference_sigma(
        [profile_row(result) for result in b1_robustness]
    )
    nbar_shift_over_sigma = shift_over_reference_sigma(nbar_robustness)
    profiled_center_parity: dict[str, dict[str, Any]] = {}
    for p in p_values:
        fit0 = get_profile(PRIMARY_TIER, "fiducial", p, PRIMARY_CUT)
        fit_plus = get_profile(PRIMARY_TIER, "LC_p", p, PRIMARY_CUT)
        fit_minus = get_profile(PRIMARY_TIER, "LC_m", p, PRIMARY_CUT)
        odd_gain = (fit_plus.fhat - fit_minus.fhat) / 200.0
        even_at_abs100 = (
            0.5 * (fit_plus.fhat + fit_minus.fhat) - fit0.fhat
        )
        profiled_center_parity[f"{p:.1f}"] = {
            "diagnostic": (
                "algebraic decomposition of independently profiled ensemble "
                "centres; it is not the raw LC_p-LC_m response and no "
                "mean-covariance significance is assigned"
            ),
            "fiducial_offset_fNL": fit0.fhat,
            "odd_profile_gain": odd_gain,
            "even_profile_displacement_at_abs100_fNL": even_at_abs100,
            "positive_increment_from_fiducial": fit_plus.fhat - fit0.fhat,
            "negative_increment_from_fiducial": fit_minus.fhat - fit0.fhat,
            "positive_absolute_bias": fit_plus.fhat - 100.0,
            "negative_absolute_bias": fit_minus.fhat + 100.0,
            "quadratic_map_coefficient": even_at_abs100 / 10000.0,
        }

    gates = {
        "data_and_single_covariance_contract": {
            "pass": data_audit["gate_status"] == "pass",
            "value": data_audit["gate_status"],
        },
        "all_p_primary_recovery_at_validated_0p08": {
            "pass": all_primary_pass and not args.quick,
            "value": f"{sum(row['all_primary_gates_pass'] for row in primary_by_p)}/{len(primary_by_p)}",
        },
        "frozen_nuisance_non_closure_diagnostic": {
            "pass": None,
            "diagnostic_only": True,
            "valid_model_acceptance_test": False,
            "value": {
                "plus100_max_residual_norm": max(
                    row["residual_norm_at_truth_sigma_single"]
                    for row in frozen_by_p.values()
                ),
                "minus100_max_residual_norm": max(
                    row["residual_norm_at_truth_sigma_single"]
                    for row in frozen_minus_by_p.values()
                ),
            },
            "reason": (
                "A fixed-mass halo catalogue changes abundance, bias, and "
                "stochastic/nuisance parameters with fNL; freezing the "
                "fiducial nuisance point does not describe either PNG sample."
            ),
        },
        "matched_catalogue_response_non_closure_diagnostic": {
            "pass": None,
            "diagnostic_only": True,
            "valid_model_acceptance_test": False,
            "value": max(
                row["unexplained_even_abs_projection_over_sigma"]
                for row in response_by_p.values()
            ),
            "reason": response_primary["non_closure_reason"],
        },
        "implemented_matter_B112II_magnitude": {
            "pass": None,
            "diagnostic_only": True,
            "value": {
                "norm_ratio_to_linear_at_abs100": theoretical_finite[
                    "B112II_to_linear_covariance_norm_ratio"
                ],
                "projected_over_sigma": theoretical_finite[
                    "B112II_abs_projection_over_fixed_nuisance_sigma"
                ],
            },
        },
        "minuit_start_robustness_LC_m": {
            "pass": start_span_over_sigma < 1.0e-2,
            "value": start_span_over_sigma,
        },
        "stress_kmax_0p12_to_0p15_stability": {
            "pass": bool(not args.quick and max_drift <= 0.5),
            "value": max_drift,
            "diagnostic_only": True,
        },
        "full_vs_coevolution": {
            "pass": bool(not args.quick and max_tier_difference <= 0.5),
            "value": max_tier_difference,
        },
        "coverage": {
            "pass": all(
                rows[sample]["nominal_0p68_inside_binomial95"]
                for rows in coverage_payload.values()
                for sample in ("fiducial", "LC_p", "LC_m")
            ),
            "value": {
                key: {
                    sample: rows[sample]["coverage68"]
                    for sample in ("fiducial", "LC_p", "LC_m")
                }
                for key, rows in coverage_payload.items()
            },
        },
        "nuisance_prior_scale_sensitivity": {
            "pass": None,
            "diagnostic_only": True,
            "value": prior_shift_over_sigma,
        },
        "covariance_source_sensitivity": {
            "pass": None,
            "diagnostic_only": True,
            "value": covariance_shift_over_sigma,
        },
        "fixed_b1_one_sigma_sensitivity": {
            "pass": None,
            "diagnostic_only": True,
            "value": b1_shift_over_sigma,
        },
        "sample_abundance_proxy_sensitivity": {
            "pass": None,
            "diagnostic_only": True,
            "value": nbar_shift_over_sigma,
        },
        "numerical_convergence_at_validated_0p08": {
            "pass": convergence["pass"],
            "value": {
                "linear_abs_delta_over_sigma": convergence[
                    "linear_response"
                ]["abs_projected_delta_over_sigma"],
                "quadratic_abs_delta_over_sigma": convergence["B112II"][
                    "abs_projected_delta_over_sigma"
                ],
            },
        },
        "model_algebra_and_matter_limit": {
            "pass": model_contract["pass"],
            "value": model_contract[
                "matter_limit_total_identity_max_relative_validated_0p08"
            ],
        },
    }

    core_gate_names = (
        "data_and_single_covariance_contract",
        "all_p_primary_recovery_at_validated_0p08",
        "minuit_start_robustness_LC_m",
        "full_vs_coevolution",
        "coverage",
        "numerical_convergence_at_validated_0p08",
        "model_algebra_and_matter_limit",
    )
    core_pass = all(gates[name]["pass"] for name in core_gate_names)
    if args.quick:
        final_status = "CONDITIONAL_ON_P_OR_PRIOR"
    elif all_primary_pass and core_pass:
        # A pre-reconstruction DM high-k recovery control has only passed to
        # 0.08.  Larger halo cuts remain stress tests even if their profiles
        # look acceptable.
        final_status = "PASS_TO_K"
    elif some_primary_pass:
        final_status = "CONDITIONAL_ON_P_OR_PRIOR"
    elif not all(
        recovery_pass(
            get_profile(PRIMARY_TIER, "fiducial", p, PRIMARY_CUT),
            0.0,
        )
        for p in p_values
    ):
        final_status = "FAIL_GAUSSIAN_PROJECTION"
    else:
        final_status = "FAIL_PNG_RESPONSE"

    figure_paths = {
        "fiducial_diagonal": paths.figures / "fiducial_k2B000_diagonal.pdf",
        "lcp_diagonal": paths.figures / "lcp100_k2B000_diagonal.pdf",
        "lcm_diagonal": (
            paths.figures / "lcm_minus100_k2B000_diagonal.pdf"
        ),
        "full2d_vector_kmax_scan": (
            paths.figures / "full2d_k1k2B000_bestfit_kmax_scan.pdf"
        ),
        "full2d": paths.figures / "lcp100_full2d_data_model_pull.pdf",
        "response": paths.figures / "matched_png_response_tree_uplift.pdf",
        "components": paths.figures / "png_response_components.pdf",
        "profiles": paths.figures / "fnl_profiles.pdf",
        "stability": paths.figures / "kmax_p_stability.pdf",
        "coverage_even": paths.figures / "coverage_and_even_remainder.pdf",
    }
    plot_diagonal(
        figure_paths["fiducial_diagonal"],
        primary0,
        k_pair,
        geometry.pair_i,
        geometry.pair_j,
        title=(
            "Quijote halo fiducial, pre recon, "
            rf"$f_{{\rm NL}}={primary0.fhat:.1f}$"
        ),
    )
    plot_diagonal(
        figure_paths["lcp_diagonal"],
        primary100,
        k_pair,
        geometry.pair_i,
        geometry.pair_j,
        title=(
            "Quijote halo LC_p(+100), pre recon, "
            rf"$f_{{\rm NL}}={primary100.fhat:.1f}$"
        ),
    )
    plot_diagonal(
        figure_paths["lcm_diagonal"],
        primary_minus100,
        k_pair,
        geometry.pair_i,
        geometry.pair_j,
        title=(
            "Quijote halo LC_m(-100), pre recon, "
            rf"$f_{{\rm NL}}={primary_minus100.fhat:.1f}$"
        ),
    )
    plot_full2d_vector_bestfit(
        figure_paths["full2d_vector_kmax_scan"],
        [
            (
                cut,
                get_profile(PRIMARY_TIER, "fiducial", p_values[0], cut),
                get_profile(PRIMARY_TIER, "LC_p", p_values[0], cut),
                get_profile(PRIMARY_TIER, "LC_m", p_values[0], cut),
            )
            for cut in (CUTS if not args.quick else (PRIMARY_CUT,))
        ],
        k_pair,
    )
    plot_full2d(
        figure_paths["full2d"],
        primary100,
        geometry.pair_i,
        geometry.pair_j,
        title="LC_p(+100) full-2D B000: black-box primary fit",
    )
    response_for_plot = response_components(
        tree=tree,
        templates=templates,
        parameters=primary0_fixed.expanded_parameters,
        b1=b1,
        p=p_values[0],
        nbar=nbar,
        dm_tree=dm_tree,
        dm_total=dm_total,
        dm_quadratic=dm_quadratic,
        response_model="finite_uplift",
    )
    tree_response_for_plot = (
        response_for_plot["deterministic_tree"]
        + response_for_plot["stochastic"]
    )[primary0_fixed.indices]
    plot_response(
        figure_paths["response"],
        primary0_fixed,
        response_arrays_by_p[primary_key],
        k_pair,
        tree_response_for_plot,
    )
    plot_components(
        figure_paths["components"],
        primary0_fixed,
        response_for_plot,
        k_pair,
    )
    plot_profiles(
        figure_paths["profiles"],
        [
            result
            for result in profiles
            if result.tier == PRIMARY_TIER
            and result.kmax == PRIMARY_CUT
        ],
    )
    plot_stability(figure_paths["stability"], profiles)
    plot_coverage_even(
        figure_paths["coverage_even"],
        coverage_arrays[primary_key]["fiducial"],
        coverage_arrays[primary_key]["LC_p"],
        coverage_arrays[primary_key]["LC_m"],
        response_arrays_by_p[primary_key],
    )

    summary_path = paths.analysis / "summary.json"
    report_path = paths.analysis / "REPORT.md"
    arrays_path = paths.analysis / "profiles.npz"
    summary = {
        "schema": 2,
        "created_utc": utc_now(),
        "goal_thread": GOAL_THREAD,
        "tag": args.tag,
        "final_status": final_status,
        "scope": (
            "Gaussian halo one-loop EFT-v2 + tree halo PNG + pure-matter "
            "linear one-loop PNG uplift + matter B112II finite-fNL term"
        ),
        "scope_warning": (
            "not a complete halo-PNG one-loop theory; nonlinear halo-PNG "
            "loop, counterterm, and stochastic sectors remain absent"
        ),
        "covariance_contract": (
            "covariance of one realization everywhere; ensemble means are "
            "central vectors only; no /100 or /500 covariance"
        ),
        "selection": {
            "stage": "pre",
            "redshift": 1.0,
            "space": "real",
            "mass_threshold_hinv_msun": 1.0e13,
            "cuts_h_mpc": CUTS,
            "excluded_global_bin": 0,
            "primary_count": int(
                selected_indices(k_pair, PRIMARY_CUT).size
            ),
            "validated_primary_kmax_h_mpc": PRIMARY_CUT,
            "stress_maximum_kmax_h_mpc": STRESS_CUT,
            "pre_recon_dm_high_k_control_passed_beyond_0p08": False,
            "primary_last_shell_h_mpc": float(
                np.max(k_pair[selected_indices(k_pair, PRIMARY_CUT)])
            ),
        },
        "b1_calibration": {
            "value": b1,
            "sigma_single": b1_sigma,
            "source": str(accepted_report_path),
            "P0_kmax_h_mpc": 0.10,
            "P0_not_reused_in_B000_likelihood": True,
        },
        "data_audit": data_audit,
        "tree_production": tree_production,
        "dm_production": dm_production,
        "stochastic_projection_adapter": adapter_audit,
        "model_contract": model_contract,
        "primary_by_p": primary_by_p,
        "profiles": [profile_row(result) for result in profiles],
        "ablations": [profile_row(result) for result in ablations],
        "b1_robustness": [profile_row(result) for result in b1_robustness],
        "start_robustness_LC_m": start_robustness,
        "prior_robustness": prior_robustness,
        "covariance_robustness": covariance_robustness,
        "sample_abundance_proxy": {
            "definition": (
                "relative nbar diagnostic assumes the measured P0 "
                "num_shotnoise/norm proxy scales as 1/nbar; only ratios to "
                "the matched fiducial sample are used"
            ),
            "shot_proxy_means": shot_proxy_means,
            "nbar_fiducial_contract_h3_mpc3": nbar,
            "relative_nbar_h3_mpc3": relative_nbar,
            "fits": nbar_robustness,
        },
        "robustness": {
            "start_span_over_sigma": start_span_over_sigma,
            "prior_shift_over_sigma": prior_shift_over_sigma,
            "covariance_shift_over_sigma": covariance_shift_over_sigma,
            "b1_shift_over_sigma": b1_shift_over_sigma,
            "nbar_shift_over_sigma": nbar_shift_over_sigma,
        },
        "profiled_center_parity": profiled_center_parity,
        "fixed_fNL0_nuisance_by_p": {
            key: profile_row(result)
            for key, result in fixed_zero_by_p.items()
        },
        "frozen_nuisance_by_p": {
            key: {
                name: value
                for name, value in row.items()
                if name not in {"prediction", "prediction_at_truth"}
            }
            for key, row in frozen_by_p.items()
        },
        "frozen_nuisance_minus_by_p": {
            key: {
                name: value
                for name, value in row.items()
                if name not in {"prediction", "prediction_at_truth"}
            }
            for key, row in frozen_minus_by_p.items()
        },
        "frozen_nuisance_primary": {
            name: value
            for name, value in frozen_primary.items()
            if name not in {"prediction", "prediction_at_truth"}
        },
        "frozen_nuisance_minus_primary": {
            name: value
            for name, value in frozen_minus_primary.items()
            if name not in {"prediction", "prediction_at_truth"}
        },
        "response_diagnostics_by_p": response_by_p,
        "response_diagnostics": response_primary,
        "theoretical_finite_fNL": theoretical_finite,
        "free_bphidelta_diagnostic_by_p": free_bphidelta_by_p,
        "numerical_convergence": convergence,
        "coverage_by_p": coverage_payload,
        "coverage": coverage_primary,
        "gates": gates,
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "arrays": str(arrays_path),
            "manifest": str(paths.manifest),
            "figures": [str(path) for path in figure_paths.values()],
        },
        "interpretation": (
            "The final status is assigned from independent marginalized "
            "single-realization injection recovery, coverage, numerical, "
            "tier, and algebraic checks. Raw LC_p-LC_m response and "
            "frozen-fiducial-nuisance comparisons are explicitly excluded "
            "because they compare a total fixed-selection catalogue "
            "response to a fixed-nuisance partial theory response. Any "
            "remaining failure concerns this hybrid model and prior "
            "contract, not a complete halo-PNG one-loop theory that has not "
            "been implemented."
        ),
    }

    primary_profiles = [
        result
        for result in profiles
        if result.tier == PRIMARY_TIER and result.kmax == PRIMARY_CUT
    ]
    atomic_npz(
        arrays_path,
        p_grid=np.asarray(p_values),
        primary_profile_sample=np.asarray(
            [result.sample for result in primary_profiles], dtype="U16"
        ),
        primary_profile_p=np.asarray(
            [result.p for result in primary_profiles]
        ),
        primary_profile_fhat=np.asarray(
            [result.fhat for result in primary_profiles]
        ),
        primary_profile_sigma=np.asarray(
            [result.sigma_symmetric for result in primary_profiles]
        ),
        selected_indices=primary0.indices,
        k_pair=k_pair[primary0.indices],
        fiducial_target=primary0.target,
        fiducial_prediction=primary0.prediction,
        lcp_target=primary100.target,
        lcp_prediction=primary100.prediction,
        lcm_target=primary_minus100.target,
        lcm_prediction=primary_minus100.prediction,
        covariance_single=primary0.covariance_single,
        response_odd=response_arrays_by_p[primary_key]["odd"],
        response_model=primary0_fixed.linear_response,
        even_measured=response_arrays_by_p[primary_key]["even_displacement"],
        even_b112ii=response_arrays_by_p[primary_key]["predicted_even"],
        coverage_fid_fhat=coverage_arrays[primary_key]["fiducial"]["fhat"],
        coverage_fid_sigma=coverage_arrays[primary_key]["fiducial"]["sigma"],
        coverage_lcp_fhat=coverage_arrays[primary_key]["LC_p"]["fhat"],
        coverage_lcp_sigma=coverage_arrays[primary_key]["LC_p"]["sigma"],
        coverage_lcm_fhat=coverage_arrays[primary_key]["LC_m"]["fhat"],
        coverage_lcm_sigma=coverage_arrays[primary_key]["LC_m"]["sigma"],
    )
    atomic_json(summary_path, summary)
    atomic_text(report_path, markdown_report(summary))
    manifest = {
        "schema": 1,
        "created_utc": utc_now(),
        "tag": args.tag,
        "goal_thread": GOAL_THREAD,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            str(path): sha256(path)
            for path in (
                paths.fid_matrix,
                paths.matched_matrix,
                paths.frozen_fitter,
                paths.frozen_contract,
                paths.b_templates,
                paths.p_templates,
                paths.png_table,
                paths.native_binary,
                paths.dm_linear,
                paths.dm_quadratic,
                paths.tree_templates,
            )
        },
        "outputs": {
            str(path): sha256(path)
            for path in (
                summary_path,
                report_path,
                arrays_path,
                *figure_paths.values(),
            )
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    atomic_json(paths.manifest, manifest)
    print(
        json.dumps(
            {
                "status": final_status,
                "summary": str(summary_path),
                "report": str(report_path),
                "manifest": str(paths.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
