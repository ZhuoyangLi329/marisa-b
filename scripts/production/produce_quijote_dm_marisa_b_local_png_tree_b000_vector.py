#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Produce Quijote DM pre-recon local-PNG dB000/dfNL with MARISA-B."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def enforce_thread_limit(max_threads: int = 8) -> None:
    for key in THREAD_LIMIT_ENV_KEYS:
        value = os.environ.get(key)
        try:
            parsed = int(value) if value is not None and value.strip() else int(max_threads)
        except ValueError:
            parsed = int(max_threads)
        os.environ[key] = str(max(1, min(parsed, int(max_threads))))
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


enforce_thread_limit()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reconstruction-png")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quijote_dm_spt1loop_realspace import (
    DEFAULT_DM_MATRIX,
    QUIJOTE_COSMOLOGY,
    load_quijote_dm_pre_mean,
)
from run_marisa_b_triangle import (
    DEFAULT_BUILD_DIR,
    compile_marisa_b,
    describe_existing_pk_table,
    run_marisa_b,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MARISA_B_DATA_ROOT", PROJECT_ROOT)).resolve()
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT / "analysis" / "theory_vectors" / "marisa_b_v0"
)
DEFAULT_FD_NPZ = (
    DATA_ROOT
    / "figures"
    / "diagnostics"
    / "quijote_dm_png_z0p5_lcp_lcm_b000_derivative_compare"
    / "quijote_dm_png_z0p5_lcp_lcm_b000_derivative_compare.npz"
)
DEFAULT_PLIN_TABLE = (
    DATA_ROOT
    / "figures"
    / "diagnostics"
    / "quijote_dm_spt1loop_realspace"
    / "react_bspt_b000_diag_weighted_kmax0p305_nmu12_eps1e3_rerun_plin_z0.5.dat"
)
HITOMI_K_PIVOT_HMPC = 0.05
K_FLOOR = 1.0e-6
PNG_COMPONENT_KEYS = (
    "dBdfNL_local_tree",
    "dBdfNL_local_B122I",
    "dBdfNL_local_B122II",
    "dBdfNL_local_B113I",
    "dBdfNL_local_B113II",
    "dBdfNL_local_1loop",
    "dBdfNL_local_total",
    "B112II_fNL2_coefficient",
)
B000_FIELD_BY_COMPONENT = {
    "dBdfNL_local_tree": "B000_dBdfNL_local_tree",
    "dBdfNL_local_B122I": "B000_dBdfNL_local_B122I",
    "dBdfNL_local_B122II": "B000_dBdfNL_local_B122II",
    "dBdfNL_local_B113I": "B000_dBdfNL_local_B113I",
    "dBdfNL_local_B113II": "B000_dBdfNL_local_B113II",
    "dBdfNL_local_1loop": "B000_dBdfNL_local_1loop",
    "dBdfNL_local_total": "B000_dBdfNL_local_total",
    "B112II_fNL2_coefficient": "B000_B112II_fNL2_coefficient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_DM_MATRIX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="marisa_b_local_png_tree_b000_weighted_all120_nmu96")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--kmax", type=float, default=0.305)
    parser.add_argument("--compute-all-pairs", action="store_true")
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--pair-index", action="append", type=int, default=None)
    parser.add_argument("--nmu-b000", type=int, default=96)
    parser.add_argument(
        "--bin-average",
        choices=("point", "shell"),
        default="point",
        help="Use weighted-k point projection or continuous k-shell radial averaging.",
    )
    parser.add_argument(
        "--nradial-binavg",
        type=int,
        default=3,
        help="Gauss-Legendre radial nodes per k shell for --bin-average=shell.",
    )
    parser.add_argument("--epsrel", type=float, default=1.0e-3, help="Native 3D loop integral relative tolerance.")
    parser.add_argument("--p13-epsrel", type=float, default=1.0e-2, help="Native P13/P12-style 1D integral relative tolerance.")
    parser.add_argument(
        "--mu-convergence-nmu",
        action="append",
        type=int,
        default=None,
        help="Optional extra B000 mu-node counts for a convergence report; repeat for several values.",
    )
    parser.add_argument(
        "--max-mu-convergence-triangles",
        type=int,
        default=4096,
        help="Safety cap for extra triangles used by --mu-convergence-nmu.",
    )
    parser.add_argument(
        "--radial-convergence-nradial",
        action="append",
        type=int,
        default=None,
        help="Optional extra radial shell-node counts; repeat for several values.",
    )
    parser.add_argument(
        "--max-radial-convergence-triangles",
        type=int,
        default=8192,
        help="Safety cap for extra triangles used by --radial-convergence-nradial.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.5,
        help=(
            "Redshift used when this runner generates a new P_L/P_phi/M table. "
            "With --png-table, the supplied columns are used verbatim and are "
            "not rescaled by --z."
        ),
    )
    parser.add_argument("--b1", type=float, default=1.0)
    parser.add_argument(
        "--png-loop-level",
        choices=("tree", "1loop", "b112ii"),
        default="tree",
        help="'b112ii' computes only the isolated coefficient B112II/fNL^2.",
    )
    parser.add_argument("--qmin", type=float, default=1.0e-4)
    parser.add_argument("--qmax", type=float, default=30.0)
    parser.add_argument(
        "--png-ir-cutoff",
        type=float,
        default=0.0,
        help="IR cutoff applied to every primordial P_phi propagator; 0 reuses qmin.",
    )
    parser.add_argument(
        "--b112ii-integrator",
        choices=("adaptive", "multicenter_qmc"),
        default="adaptive",
        help="Integrator used only by --png-loop-level b112ii.",
    )
    parser.add_argument("--b112ii-qmc-power", type=int, default=12)
    parser.add_argument("--b112ii-qmc-replicates", type=int, default=4)
    parser.add_argument(
        "--reconstruction",
        choices=("pre", "post"),
        default="pre",
        help="Use pre-recon kernels or finite-R post-recon kernels.",
    )
    parser.add_argument("--smoothing-radius", type=float, default=15.0)
    parser.add_argument("--bias-recon", type=float, default=1.0)
    parser.add_argument("--singular-floor", type=float, default=1.0e-5)
    parser.add_argument("--png-table", default=None, help="Optional existing k, P_L, P_phi, M table.")
    parser.add_argument("--plin-table", default=str(DEFAULT_PLIN_TABLE), help="Fallback two-column P_L table if cosmoprimo is unavailable.")
    parser.add_argument("--a-s", type=float, default=2.126e-9, help="Fallback scalar amplitude A_s used with --plin-table.")
    parser.add_argument("--n-s", type=float, default=float(QUIJOTE_COSMOLOGY["n_s"]), help="Fallback scalar tilt used with --plin-table.")
    parser.add_argument("--h", type=float, default=float(QUIJOTE_COSMOLOGY["h"]), help="Fallback h used with --plin-table.")
    parser.add_argument("--pk-kmin", type=float, default=1.0e-6)
    parser.add_argument("--pk-kmax", type=float, default=80.0)
    parser.add_argument("--pk-n", type=int, default=2400)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--max-triangles-per-call", type=int, default=240)
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=1,
        help="Concurrent C++ chunk subprocesses; default 1 preserves historical behavior.",
    )
    parser.add_argument("--sim-derivative-npz", default=str(DEFAULT_FD_NPZ))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def theory_k_pairs(data: dict[str, np.ndarray], k_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.column_stack((data["k1"], data["k2"])).astype(np.float64, copy=False)
    weighted = np.asarray(data["hdf5_k_weighted"], dtype=np.float64)
    if k_source == "weighted":
        theory = weighted
    elif k_source == "center":
        theory = nominal
    else:
        raise ValueError(f"unknown k_source {k_source!r}")
    return nominal, weighted, theory


def load_b000_mean_geometry(matrix_path: Path) -> dict[str, np.ndarray]:
    """Load either the historical DM aggregate or the halo fid500 geometry.

    The native matter calculation depends only on the shared B000 shell
    geometry.  Halo aggregates use ``b000_k``/``b000_k_edges`` instead of the
    historical DM ``k1``/``hdf5_k_edges`` schema, so accepting both schemas
    avoids copying a temporary matrix merely to run a matter theory vector.
    The sample mean is carried only for optional plotting; it never enters
    the theory calculation.
    """

    with np.load(matrix_path, allow_pickle=False) as values:
        if {
            "fiducial_common_pre_B000",
            "k1",
            "k2",
            "pair_i",
            "pair_j",
            "k_centers",
            "hdf5_k_weighted",
            "hdf5_k_edges",
        }.issubset(values.files):
            return load_quijote_dm_pre_mean(matrix_path)

        required = {
            "fiducial_pre_B000",
            "b000_k",
            "b000_k_edges",
        }
        missing = sorted(required.difference(values.files))
        if missing:
            raise KeyError(
                "matrix is neither a supported DM aggregate nor a halo "
                f"fid500 aggregate; missing halo keys: {missing}"
            )
        samples = np.asarray(values["fiducial_pre_B000"], dtype=np.float64)
        weighted = np.asarray(values["b000_k"], dtype=np.float64)
        edges = np.asarray(values["b000_k_edges"], dtype=np.float64)

    if samples.ndim != 2 or weighted.shape != (samples.shape[1], 2):
        raise ValueError("halo B000 sample/geometry shape mismatch")
    if edges.shape != (samples.shape[1], 2, 2):
        raise ValueError("halo B000 shell-edge shape mismatch")
    rounded = np.round(edges, 12)
    unique_edges = sorted({tuple(edge) for edge in rounded.reshape(-1, 2)})
    edge_to_index = {edge: index for index, edge in enumerate(unique_edges)}
    pair_i = np.asarray(
        [edge_to_index[tuple(edge)] for edge in rounded[:, 0]],
        dtype=np.int64,
    )
    pair_j = np.asarray(
        [edge_to_index[tuple(edge)] for edge in rounded[:, 1]],
        dtype=np.int64,
    )
    centers = np.asarray(
        [0.5 * (edge[0] + edge[1]) for edge in unique_edges],
        dtype=np.float64,
    )
    nominal = np.column_stack((centers[pair_i], centers[pair_j]))
    return {
        "k1": nominal[:, 0],
        "k2": nominal[:, 1],
        "pair_i": pair_i,
        "pair_j": pair_j,
        "k_centers": centers,
        "hdf5_k_weighted": weighted,
        "hdf5_k_edges": edges,
        "mean": np.mean(samples, axis=0),
        "stderr": np.std(samples, axis=0, ddof=1)
        / math.sqrt(samples.shape[0]),
        "n_realizations": np.asarray(samples.shape[0], dtype=np.int64),
        "fiducial_sample": np.asarray("all"),
        "fiducial_sample_key": np.asarray("fiducial_pre_B000"),
        "matrix_schema": np.asarray("halo_fid500_b000"),
    }


def selected_pair_indices(data: dict[str, np.ndarray], theory_pairs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n_pairs = int(data["mean"].shape[0])
    if args.pair_index:
        indices = np.asarray(list(dict.fromkeys(int(item) for item in args.pair_index)), dtype=np.int64)
    elif bool(args.compute_all_pairs):
        indices = np.arange(n_pairs, dtype=np.int64)
    elif bool(args.diagonal_only):
        diag = np.asarray(data["pair_i"] == data["pair_j"], dtype=bool)
        keep = diag & (np.mean(theory_pairs, axis=1) <= float(args.kmax) + 1.0e-12)
        indices = np.nonzero(keep)[0].astype(np.int64)
    else:
        keep = np.max(theory_pairs, axis=1) <= float(args.kmax) + 1.0e-12
        indices = np.nonzero(keep)[0].astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= n_pairs):
        raise ValueError(f"pair_index out of range; valid range is [0,{n_pairs})")
    if indices.size == 0:
        raise ValueError("pair selection produced no pairs")
    return indices


def primordial_power_phi_dimensional(k_hmpc: np.ndarray, *, h: float, a_s: float, n_s: float) -> np.ndarray:
    k_safe = np.maximum(np.asarray(k_hmpc, dtype=np.float64), K_FLOOR)
    k_phys = k_safe * float(h)
    k_pivot_phys = HITOMI_K_PIVOT_HMPC * float(h)
    return (
        (3.0 / 5.0) ** 2
        * (2.0 * np.pi**2 / k_phys**3)
        * float(a_s)
        * (k_phys / k_pivot_phys) ** (float(n_s) - 1.0)
        * float(h) ** 3
    )


def write_png_transfer_table(
    path: Path,
    z: float,
    kmin: float,
    kmax: float,
    n: int,
    *,
    fallback_plin_table: Path,
    fallback_a_s: float,
    fallback_n_s: float,
    fallback_h: float,
) -> dict[str, Any]:
    try:
        from cosmoprimo import Cosmology
    except ImportError:
        values = np.loadtxt(fallback_plin_table, comments="#", dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(f"fallback P_L table must have columns k, P_L: {fallback_plin_table}")
        kvals = np.asarray(values[:, 0], dtype=np.float64)
        plin = np.asarray(values[:, 1], dtype=np.float64)
        p_phi = primordial_power_phi_dimensional(
            kvals,
            h=float(fallback_h),
            a_s=float(fallback_a_s),
            n_s=float(fallback_n_s),
        )
        source = "fallback_reused_P_L_table_plus_explicit_hitomi_dimensional_P_phi"
        primordial_meta = {"A_s": float(fallback_a_s), "n_s": float(fallback_n_s), "k_pivot_hmpc": HITOMI_K_PIVOT_HMPC}
        cosmology_meta = {**dict(QUIJOTE_COSMOLOGY), "h": float(fallback_h), "n_s": float(fallback_n_s)}
        fallback_meta = str(fallback_plin_table)
    else:
        cosmo = Cosmology(**QUIJOTE_COSMOLOGY, engine="class")
        pk_linear = cosmo.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=float(z))
        primordial = cosmo.get_primordial()
        kvals = np.geomspace(float(kmin), float(kmax), int(n))
        plin = np.asarray(pk_linear(kvals), dtype=np.float64)
        p_phi = primordial_power_phi_dimensional(kvals, h=float(cosmo.h), a_s=float(primordial.A_s), n_s=float(primordial.n_s))
        source = "generated_cosmoprimo_CLASS_delta_cb_plus_hitomi_dimensional_P_phi"
        primordial_meta = {"A_s": float(primordial.A_s), "n_s": float(primordial.n_s), "k_pivot_hmpc": HITOMI_K_PIVOT_HMPC}
        cosmology_meta = dict(QUIJOTE_COSMOLOGY)
        fallback_meta = None
    transfer_m = np.sqrt(plin / p_phi)
    if not (np.all(np.isfinite(plin)) and np.all(np.isfinite(p_phi)) and np.all(np.isfinite(transfer_m))):
        raise FloatingPointError("Non-finite values while writing PNG transfer table.")
    if np.any(plin <= 0.0) or np.any(p_phi <= 0.0) or np.any(transfer_m <= 0.0):
        raise FloatingPointError("P_L, P_phi, and M must be positive.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# k[h/Mpc] P_L[(Mpc/h)^3] P_phi[(Mpc/h)^3] M(k,z), Quijote fiducial\n")
        stream.write("# P_phi uses Hitomi dimensional convention: (3/5)^2 2*pi^2/k_phys^3 A_s (k_phys/(0.05*h))^(n_s-1) h^3\n")
        for row in zip(kvals, plin, p_phi, transfer_m):
            stream.write("{:.17e} {:.17e} {:.17e} {:.17e}\n".format(*row))
    return {
        "path": str(path),
        "source": source,
        "fallback_plin_table": fallback_meta,
        "z": float(z),
        "kmin": float(kvals[0]),
        "kmax": float(kvals[-1]),
        "n": int(kvals.size),
        "p_l_min": float(np.min(plin)),
        "p_l_max": float(np.max(plin)),
        "p_phi_min": float(np.min(p_phi)),
        "p_phi_max": float(np.max(p_phi)),
        "m_min": float(np.min(transfer_m)),
        "m_max": float(np.max(transfer_m)),
        "cosmology": cosmology_meta,
        "primordial": primordial_meta,
    }


def describe_png_transfer_table(path: Path) -> dict[str, Any]:
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"PNG transfer table must have columns k, P_L, P_phi, M: {path}")
    return {
        "path": str(path),
        "source": "reused_existing_table",
        "kmin": float(np.min(values[:, 0])),
        "kmax": float(np.max(values[:, 0])),
        "n": int(values.shape[0]),
        "p_l_min": float(np.min(values[:, 1])),
        "p_l_max": float(np.max(values[:, 1])),
        "p_phi_min": float(np.min(values[:, 2])),
        "p_phi_max": float(np.max(values[:, 2])),
        "m_min": float(np.min(values[:, 3])),
        "m_max": float(np.max(values[:, 3])),
    }


def run_marisa_chunks(
    *,
    binary: Path,
    png_table: Path,
    triangles: list[tuple[float, float, float]],
    args: argparse.Namespace,
    cpp_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_size = max(1, int(args.max_triangles_per_call))
    workers = max(1, int(args.chunk_workers))
    specifications = [
        (start, triangles[start : start + chunk_size])
        for start in range(0, len(triangles), chunk_size)
    ]

    def run_one(
        specification: tuple[int, list[tuple[float, float, float]]],
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        start, chunk = specification
        info = run_marisa_b(
            binary,
            png_table,
            chunk,
            float(args.epsrel),
            float(args.p13_epsrel),
            float(args.qmin),
            float(args.qmax),
            float(getattr(args, "smoothing_radius", 15.0)),
            float(getattr(args, "bias_recon", 1.0)),
            0.0,
            0,
            float(args.singular_floor),
            cpp_mode,
            "native_cpp",
            png_table=png_table,
            b1=float(args.b1),
            png_ir_cutoff=float(args.png_ir_cutoff),
            b112ii_integrator=str(args.b112ii_integrator),
            b112ii_qmc_power=int(args.b112ii_qmc_power),
            b112ii_qmc_replicates=int(args.b112ii_qmc_replicates),
        )
        payload = info.get("payload")
        parsed = list(payload.get("results", [])) if isinstance(payload, dict) else []
        chunk_record = {
            "start": int(start),
            "n_triangles": int(len(chunk)),
            "returncode": int(info["returncode"]),
            "stderr": str(info["stderr"]),
            "n_parsed_rows": int(len(parsed)),
            "wall_seconds": float(payload.get("metadata", {}).get("wall_seconds", math.nan)) if isinstance(payload, dict) else math.nan,
        }
        if int(info["returncode"]) != 0 or len(parsed) != len(chunk):
            raise RuntimeError(f"MARISA-B PNG tree chunk failed: {chunk_record}")
        return int(start), parsed, chunk_record

    if workers == 1:
        completed = [run_one(specification) for specification in specifications]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            completed = list(executor.map(run_one, specifications))
    completed.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for _, parsed, chunk_record in completed:
        rows.extend(parsed)
        chunks.append(chunk_record)
    return rows, chunks


def accumulate_b000(rows: list[dict[str, Any]], triangle_meta: list[tuple[int, float]], n_selected: int, component_key: str) -> np.ndarray:
    out = np.zeros(n_selected, dtype=np.float64)
    for row, (local_index, weight) in zip(rows, triangle_meta):
        out[local_index] += 0.5 * float(weight) * float(row[component_key])
    return out


def build_b000_mu_work(
    indices: np.ndarray,
    theory_pairs: np.ndarray,
    nmu_b000: int,
    *,
    bin_average: str,
    k_edges: np.ndarray | None,
    nradial_binavg: int,
    external_support_min: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, float]], dict[str, Any]]:
    """构造 `B000` 投影所需三角形列表，可选择 point 或 shell-bin 平均。"""
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(nmu_b000))
    triangles: list[tuple[float, float, float]] = []
    triangle_meta: list[tuple[int, float]] = []
    per_pair_triangles: list[int] = []
    per_pair_weight_sums: list[float] = []

    if bin_average == "point":
        for local_index, pair_index in enumerate(indices):
            k1, k2 = theory_pairs[int(pair_index)]
            weight_sum = 0.0
            start = len(triangles)
            for mu, weight in zip(mu_nodes, mu_weights):
                triangles.append((float(k1), float(k2), float(mu)))
                triangle_meta.append((int(local_index), float(weight)))
                weight_sum += float(weight)
            per_pair_triangles.append(len(triangles) - start)
            per_pair_weight_sums.append(weight_sum)
    elif bin_average == "shell":
        if k_edges is None:
            raise ValueError("k_edges are required for --bin-average=shell")
        edges = np.asarray(k_edges, dtype=np.float64)
        for local_index, pair_index in enumerate(indices):
            pair_edges = edges[int(pair_index)]
            k1_nodes, k1_weights = shell_nodes(
                pair_edges[0],
                int(nradial_binavg),
                support_min=float(external_support_min),
            )
            k2_nodes, k2_weights = shell_nodes(
                pair_edges[1],
                int(nradial_binavg),
                support_min=float(external_support_min),
            )
            weight_sum = 0.0
            start = len(triangles)
            for k1, w1 in zip(k1_nodes, k1_weights):
                for k2, w2 in zip(k2_nodes, k2_weights):
                    radial_weight = float(w1) * float(w2)
                    for mu, wmu in zip(mu_nodes, mu_weights):
                        weight = radial_weight * float(wmu)
                        triangles.append((float(k1), float(k2), float(mu)))
                        triangle_meta.append((int(local_index), weight))
                        weight_sum += weight
            per_pair_triangles.append(len(triangles) - start)
            per_pair_weight_sums.append(weight_sum)
    else:
        raise ValueError(f"Unknown bin_average mode: {bin_average!r}")

    weight_sums = np.asarray(per_pair_weight_sums, dtype=np.float64)
    work_summary = {
        "bin_average": str(bin_average),
        "nmu_b000": int(nmu_b000),
        "nradial_binavg": int(nradial_binavg) if bin_average == "shell" else None,
        "n_selected_pairs": int(indices.size),
        "n_triangles": int(len(triangles)),
        "triangles_per_pair_min": int(min(per_pair_triangles)) if per_pair_triangles else 0,
        "triangles_per_pair_max": int(max(per_pair_triangles)) if per_pair_triangles else 0,
        "pre_half_weight_sum_min": float(np.min(weight_sums)) if weight_sums.size else math.nan,
        "pre_half_weight_sum_max": float(np.max(weight_sums)) if weight_sums.size else math.nan,
        "external_support_min": float(external_support_min),
        "normalization_note": (
            "accumulate_b000() multiplies stored weights by 1/2. With "
            "external_support_min=0 the pre-half sum is 2; with a positive "
            "support cutoff, the excluded sub-shell has exactly zero theory "
            "but the weights retain the original full-shell normalization."
        ),
    }
    return triangles, triangle_meta, work_summary


def project_b000_components(
    rows: list[dict[str, Any]],
    triangle_meta: list[tuple[int, float]],
    n_selected: int,
) -> dict[str, np.ndarray]:
    """把 C++ 三角形输出累积成选中 pair 的 `B000` component 向量。"""
    return {
        component: accumulate_b000(rows, triangle_meta, n_selected, component)
        for component in PNG_COMPONENT_KEYS
    }


def selected_components_to_full_vectors(
    selected_components: dict[str, np.ndarray],
    indices: np.ndarray,
    n_pairs: int,
) -> dict[str, np.ndarray]:
    """把选中 pair 的 component 结果嵌回完整 120 维向量。"""
    component_vectors: dict[str, np.ndarray] = {}
    for component, values in selected_components.items():
        vector = np.full(n_pairs, np.nan, dtype=np.float64)
        vector[indices] = np.asarray(values, dtype=np.float64)
        component_vectors[component] = vector
    return component_vectors


def shell_nodes(
    k_edges: np.ndarray,
    nradial: int,
    *,
    support_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = map(float, np.asarray(k_edges, dtype=np.float64))
    if not right > left:
        raise ValueError(f"Invalid k shell edges: {k_edges}")
    if int(nradial) <= 0:
        raise ValueError("--nradial-binavg must be positive")
    active_left = max(left, float(support_min))
    if active_left >= right:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    x, w = np.polynomial.legendre.leggauss(int(nradial))
    k = 0.5 * (right + active_left) + 0.5 * (right - active_left) * x
    dk_weight = 0.5 * (right - active_left) * w
    shell_weight = dk_weight * k * k
    norm = (right**3 - left**3) / 3.0
    if not norm > 0.0:
        raise ValueError(f"Invalid k shell normalization for edges: {k_edges}")
    return k.astype(np.float64, copy=False), (shell_weight / norm).astype(np.float64, copy=False)


def run_b000_projection(
    *,
    binary: Path,
    png_table: Path,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    theory_pairs: np.ndarray,
    nmu_b000: int,
    args: argparse.Namespace,
    cpp_mode: str,
    nradial_binavg: int | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int, dict[str, Any]]:
    """按指定 `nmu_b000` 运行一次 B000 投影并返回 component 结果。

    这个函数是主生产和收敛检查共用的唯一投影路径，避免主结果和
    diagnostic 结果因为实现细节不同而不可比。
    """
    resolved_nradial = (
        int(args.nradial_binavg)
        if nradial_binavg is None
        else int(nradial_binavg)
    )
    external_support_min = (
        (
            float(args.png_ir_cutoff)
            if float(args.png_ir_cutoff) > 0.0
            else float(args.qmin)
        )
        if str(args.png_loop_level) == "b112ii"
        else 0.0
    )
    triangles, triangle_meta, work_summary = build_b000_mu_work(
        indices,
        theory_pairs,
        int(nmu_b000),
        bin_average=str(args.bin_average),
        k_edges=np.asarray(data["hdf5_k_edges"], dtype=np.float64),
        nradial_binavg=resolved_nradial,
        external_support_min=external_support_min,
    )
    rows, chunk_summaries = run_marisa_chunks(binary=binary, png_table=png_table, triangles=triangles, args=args, cpp_mode=cpp_mode)
    selected_components = project_b000_components(rows, triangle_meta, int(indices.size))
    return selected_components, chunk_summaries, len(triangles), work_summary


def load_sim_derivative(path: Path, n_pairs: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        key = "pre_central_dB000_dfNL"
        err_key = "pre_central_dB000_dfNL_stderr"
        if key not in data.files:
            return None
        values = np.asarray(data[key], dtype=np.float64)
        if values.shape != (n_pairs,):
            raise ValueError(f"{path} {key} shape {values.shape} does not match n_pairs={n_pairs}")
        stderr = np.asarray(data[err_key], dtype=np.float64) if err_key in data.files else np.full(n_pairs, np.nan)
        return {
            "pre_central_dB000_dfNL": values,
            "pre_central_dB000_dfNL_stderr": stderr,
            "pre_k_pairs": np.asarray(data["pre_k_pairs"], dtype=np.float64) if "pre_k_pairs" in data.files else np.empty((0, 2)),
            "pre_diag_mask": np.asarray(data["pre_diag_mask"], dtype=bool) if "pre_diag_mask" in data.files else np.zeros(n_pairs, dtype=bool),
        }


def comparison_stats(theory: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(theory) & np.isfinite(sim) & (sim != 0.0)
    x = theory[valid]
    y = sim[valid]
    if x.size == 0:
        return {"n": 0, "cosine": math.nan, "corrcoef": math.nan, "median_theory_over_sim": math.nan, "sign_agreement": math.nan}
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    centered = np.corrcoef(x, y)[0, 1] if x.size > 1 else math.nan
    return {
        "n": int(x.size),
        "cosine": float(np.sum(x * y) / denom) if denom != 0.0 else math.nan,
        "corrcoef": float(centered),
        "median_theory_over_sim": float(np.median(x / y)),
        "rms_fractional_difference": float(np.sqrt(np.mean((x / y - 1.0) ** 2))),
        "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
    }


def unique_sorted_nmu_values(main_nmu: int, extra_values: list[int] | None) -> list[int]:
    """整理外层 `mu` 收敛检查需要计算的节点数。"""
    values = [int(main_nmu)]
    if extra_values:
        values.extend(int(item) for item in extra_values)
    values = sorted(set(values))
    if any(item <= 0 for item in values):
        raise ValueError(f"nmu values must be positive: {values}")
    return values


def summarize_mu_convergence(
    *,
    nmu_values: list[int],
    indices: np.ndarray,
    selected_by_nmu: dict[int, dict[str, np.ndarray]],
    primary_component: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """生成外层 B000 角积分收敛报告。

    最高 `nmu` 被当作当前参考值；这里不声称它是数学真值，只用于量化低
    `nmu` 对同一组 pair 的偏差。
    """
    reference_nmu = int(max(nmu_values))
    reference = np.asarray(selected_by_nmu[reference_nmu][primary_component], dtype=np.float64)
    primary_values = np.vstack([np.asarray(selected_by_nmu[nmu][primary_component], dtype=np.float64) for nmu in nmu_values])
    finite_reference = np.isfinite(reference) & (reference != 0.0)
    rel_to_reference = np.full_like(primary_values, np.nan, dtype=np.float64)
    rel_to_reference[:, finite_reference] = (
        primary_values[:, finite_reference] - reference[finite_reference][None, :]
    ) / reference[finite_reference][None, :]

    per_nmu: dict[str, Any] = {}
    for row, nmu in enumerate(nmu_values):
        rel = rel_to_reference[row]
        valid = np.isfinite(rel)
        per_nmu[str(int(nmu))] = {
            "n_valid": int(np.count_nonzero(valid)),
            "max_abs_rel_to_reference": float(np.nanmax(np.abs(rel[valid]))) if np.any(valid) else math.nan,
            "median_rel_to_reference": float(np.nanmedian(rel[valid])) if np.any(valid) else math.nan,
            "rms_rel_to_reference": float(np.sqrt(np.nanmean(rel[valid] ** 2))) if np.any(valid) else math.nan,
        }

    component_arrays: dict[str, np.ndarray] = {
        "mu_convergence_nmu_values": np.asarray(nmu_values, dtype=np.int64),
        "mu_convergence_pair_indices": np.asarray(indices, dtype=np.int64),
        "mu_convergence_primary_values": primary_values,
        "mu_convergence_primary_rel_to_reference": rel_to_reference,
    }
    for component in PNG_COMPONENT_KEYS:
        component_arrays[f"mu_convergence_{B000_FIELD_BY_COMPONENT[component]}"] = np.vstack(
            [np.asarray(selected_by_nmu[nmu][component], dtype=np.float64) for nmu in nmu_values]
        )

    summary = {
        "enabled": True,
        "reference_nmu": reference_nmu,
        "nmu_values": [int(item) for item in nmu_values],
        "pair_indices": [int(item) for item in indices],
        "primary_component": primary_component,
        "per_nmu": per_nmu,
    }
    return summary, component_arrays


def summarize_radial_convergence(
    *,
    nradial_values: list[int],
    indices: np.ndarray,
    selected_by_nradial: dict[int, dict[str, np.ndarray]],
    primary_component: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """生成 shell 径向 Gauss-Legendre 节点收敛报告。"""

    reference_nradial = int(max(nradial_values))
    reference = np.asarray(
        selected_by_nradial[reference_nradial][primary_component],
        dtype=np.float64,
    )
    primary_values = np.vstack(
        [
            np.asarray(
                selected_by_nradial[nradial][primary_component],
                dtype=np.float64,
            )
            for nradial in nradial_values
        ]
    )
    finite_reference = np.isfinite(reference) & (reference != 0.0)
    rel_to_reference = np.full_like(primary_values, np.nan, dtype=np.float64)
    rel_to_reference[:, finite_reference] = (
        primary_values[:, finite_reference]
        - reference[finite_reference][None, :]
    ) / reference[finite_reference][None, :]

    per_nradial: dict[str, Any] = {}
    for row, nradial in enumerate(nradial_values):
        rel = rel_to_reference[row]
        valid = np.isfinite(rel)
        per_nradial[str(int(nradial))] = {
            "n_valid": int(np.count_nonzero(valid)),
            "max_abs_rel_to_reference": (
                float(np.nanmax(np.abs(rel[valid]))) if np.any(valid) else math.nan
            ),
            "median_rel_to_reference": (
                float(np.nanmedian(rel[valid])) if np.any(valid) else math.nan
            ),
            "rms_rel_to_reference": (
                float(np.sqrt(np.nanmean(rel[valid] ** 2)))
                if np.any(valid)
                else math.nan
            ),
        }

    component_arrays: dict[str, np.ndarray] = {
        "radial_convergence_nradial_values": np.asarray(
            nradial_values, dtype=np.int64
        ),
        "radial_convergence_pair_indices": np.asarray(indices, dtype=np.int64),
        "radial_convergence_primary_values": primary_values,
        "radial_convergence_primary_rel_to_reference": rel_to_reference,
    }
    for component in PNG_COMPONENT_KEYS:
        component_arrays[
            f"radial_convergence_{B000_FIELD_BY_COMPONENT[component]}"
        ] = np.vstack(
            [
                np.asarray(
                    selected_by_nradial[nradial][component],
                    dtype=np.float64,
                )
                for nradial in nradial_values
            ]
        )

    summary = {
        "enabled": True,
        "reference_nradial": reference_nradial,
        "nradial_values": [int(item) for item in nradial_values],
        "pair_indices": [int(item) for item in indices],
        "primary_component": primary_component,
        "per_nradial": per_nradial,
    }
    return summary, component_arrays


def compact_diagonal(
    data: dict[str, np.ndarray],
    theory_pairs: np.ndarray,
    computed_mask: np.ndarray,
    component_vectors: dict[str, np.ndarray],
    primary_component: str,
    sim_derivative: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    diag = np.asarray(data["pair_i"] == data["pair_j"], dtype=bool) & computed_mask
    order = np.argsort(np.mean(theory_pairs[diag], axis=1))
    indices = np.nonzero(diag)[0][order]
    arrays = {
        "diag_indices": indices.astype(np.int64),
        "k_eff": np.mean(theory_pairs[indices], axis=1),
    }
    for component, vector in component_vectors.items():
        arrays[B000_FIELD_BY_COMPONENT[component]] = vector[indices]
    arrays["B000_dBdfNL_primary"] = component_vectors[primary_component][indices]
    if sim_derivative is not None:
        arrays["sim_pre_central_dB000_dfNL"] = sim_derivative["pre_central_dB000_dfNL"][indices]
        arrays["sim_pre_central_dB000_dfNL_stderr"] = sim_derivative["pre_central_dB000_dfNL_stderr"][indices]
    return arrays


def triangle_from_sides(a: float, b: float, c: float) -> tuple[float, float, float]:
    mu = (c * c - a * a - b * b) / (2.0 * a * b)
    return float(a), float(b), float(np.clip(mu, -1.0, 1.0))


def validation_checks(binary: Path, png_table: Path, args: argparse.Namespace, cpp_mode: str, component_key: str) -> dict[str, Any]:
    sides = (0.035, 0.055, 0.075)
    if component_key == "B112II_fNL2_coefficient":
        perm_triangles = [
            triangle_from_sides(sides[i], sides[j], sides[k])
            for i, j, k in (
                (0, 1, 2),
                (1, 0, 2),
                (1, 2, 0),
                (2, 1, 0),
                (2, 0, 1),
                (0, 2, 1),
            )
        ]
        permutation_threshold = 1.0e-3
    else:
        perm_triangles = [
            triangle_from_sides(sides[0], sides[1], sides[2]),
            triangle_from_sides(sides[1], sides[2], sides[0]),
            triangle_from_sides(sides[2], sides[0], sides[1]),
        ]
        permutation_threshold = 1.0e-10
    squeezed = [
        triangle_from_sides(0.012, 0.08, 0.08),
        triangle_from_sides(0.024, 0.08, 0.08),
        triangle_from_sides(0.08, 0.08, 0.08),
    ]
    rows, _ = run_marisa_chunks(binary=binary, png_table=png_table, triangles=perm_triangles + squeezed, args=args, cpp_mode=cpp_mode)
    n_permutations = len(perm_triangles)
    perm_values = np.asarray(
        [float(row[component_key]) for row in rows[:n_permutations]],
        dtype=np.float64,
    )
    squeezed_values = np.asarray(
        [float(row[component_key]) for row in rows[n_permutations:]],
        dtype=np.float64,
    )
    permutation_relative = float(
        np.max(np.abs(perm_values - np.mean(perm_values)))
        / np.mean(np.abs(perm_values))
    )
    return {
        "permutation_test": {
            "sides": list(sides),
            "values": perm_values.tolist(),
            "max_abs_delta_over_mean_abs": permutation_relative,
            "threshold": permutation_threshold,
            "passed": bool(permutation_relative < permutation_threshold),
        },
        "squeezed_enhancement_test": {
            "triangles_by_sides": [[0.012, 0.08, 0.08], [0.024, 0.08, 0.08], [0.08, 0.08, 0.08]],
            "values": squeezed_values.tolist(),
            "ratio_kL0p012_over_equilateral": float(squeezed_values[0] / squeezed_values[2]) if squeezed_values[2] != 0.0 else math.nan,
            "ratio_kL0p012_over_kL0p024": float(squeezed_values[0] / squeezed_values[1]) if squeezed_values[1] != 0.0 else math.nan,
            "passed": bool(squeezed_values[0] > squeezed_values[1] > 0.0 and squeezed_values[0] > squeezed_values[2]),
        },
    }


def plot_diagonal(output_pdf: Path, diag: dict[str, np.ndarray], label: str) -> None:
    k = np.asarray(diag["k_eff"], dtype=np.float64)
    theory = np.asarray(diag["B000_dBdfNL_primary"], dtype=np.float64)
    tree = np.asarray(diag.get("B000_dBdfNL_local_tree", np.full_like(theory, np.nan)), dtype=np.float64)
    sim = np.asarray(diag.get("sim_pre_central_dB000_dfNL", np.full_like(theory, np.nan)), dtype=np.float64)
    scale = k * k
    show_tree = np.any(np.isfinite(tree)) and not np.allclose(tree, theory, rtol=1.0e-10, atol=0.0, equal_nan=True)

    def cubic_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 4:
            return x[valid], y[valid]
        from scipy.interpolate import CubicSpline

        order = np.argsort(x[valid])
        x_sorted = x[valid][order]
        y_sorted = y[valid][order]
        x_unique, unique_indices = np.unique(x_sorted, return_index=True)
        if x_unique.size < 4:
            return x_sorted, y_sorted
        y_unique = y_sorted[unique_indices]
        x_smooth = np.linspace(float(x_unique[0]), float(x_unique[-1]), 240)
        spline = CubicSpline(x_unique, y_unique, bc_type="natural")
        return x_smooth, spline(x_smooth)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12})
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    k_smooth, sim_smooth = cubic_curve(k, scale * sim)
    ax_top.plot(k_smooth, sim_smooth, "-", lw=1.8, color="#4C72B0", label="Quijote DM pre (LCp-LCm)/200")
    if show_tree:
        k_smooth, tree_smooth = cubic_curve(k, scale * tree)
        ax_top.plot(k_smooth, tree_smooth, "--", lw=1.5, color="#8172B2", label="MARISA-B local PNG tree")
    k_smooth, theory_smooth = cubic_curve(k, scale * theory)
    ax_top.plot(k_smooth, theory_smooth, "-", lw=1.7, color="#DD8452", label=label)
    ax_top.set_ylabel(r"$k^2\,dB_{000}/df_{\rm NL}$")
    ax_top.grid(alpha=0.25)
    ax_top.legend(frameon=False, fontsize=9)

    valid = np.isfinite(sim) & (sim != 0.0)
    ratio = np.full_like(theory, np.nan)
    ratio[valid] = theory[valid] / sim[valid]
    tree_ratio = np.full_like(tree, np.nan)
    tree_ratio[valid] = tree[valid] / sim[valid]
    ax_bot.axhline(1.0, color="0.4", lw=1.0, ls="--")
    if show_tree:
        k_smooth, tree_ratio_smooth = cubic_curve(k, tree_ratio)
        ax_bot.plot(k_smooth, tree_ratio_smooth, "--", lw=1.2, color="#8172B2")
    k_smooth, ratio_smooth = cubic_curve(k, ratio)
    ax_bot.plot(k_smooth, ratio_smooth, "-", lw=1.3, color="#55A868")
    ax_bot.set_xlabel(r"$k\,[h/{\rm Mpc}]$")
    ax_bot.set_ylabel("theory / sim")
    ax_bot.grid(alpha=0.25)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    enforce_thread_limit()
    args = parse_args()
    if args.png_loop_level == "1loop":
        cpp_mode = (
            "post_recon_local_png_1loop"
            if args.reconstruction == "post"
            else "pre_recon_local_png_1loop"
        )
        primary_component = "dBdfNL_local_total"
        plot_label = "MARISA-B local PNG tree+1loop"
    elif args.png_loop_level == "b112ii":
        cpp_mode = (
            "post_recon_local_png_b112ii_diagnostic"
            if args.reconstruction == "post"
            else "pre_recon_local_png_b112ii_diagnostic"
        )
        primary_component = "B112II_fNL2_coefficient"
        plot_label = r"MARISA-B $\widehat B_{112}^{\rm II}$"
    else:
        if args.reconstruction == "post":
            raise ValueError(
                "--png-loop-level tree has no standalone post-recon mode; "
                "use 1loop or b112ii"
            )
        cpp_mode = "pre_recon_local_png_tree"
        primary_component = "dBdfNL_local_tree"
        plot_label = "MARISA-B local PNG tree"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = Path(args.matrix)
    data = load_b000_mean_geometry(matrix_path)
    nominal_pairs, weighted_pairs, theory_pairs = theory_k_pairs(data, str(args.k_source))
    indices = selected_pair_indices(data, theory_pairs, args)
    n_pairs = int(data["mean"].shape[0])

    build_dir = Path(args.build_dir)
    binary = build_dir / "marisa_b_triangle"
    if args.png_table:
        png_table = Path(args.png_table)
        table_info = describe_png_transfer_table(png_table)
    else:
        png_table = output_dir / f"{args.prefix}_plin_pphi_m_z{args.z:g}.dat"
        table_info = write_png_transfer_table(
            png_table,
            float(args.z),
            float(args.pk_kmin),
            float(args.pk_kmax),
            int(args.pk_n),
            fallback_plin_table=Path(args.plin_table),
            fallback_a_s=float(args.a_s),
            fallback_n_s=float(args.n_s),
            fallback_h=float(args.h),
        )
    pk_info = describe_existing_pk_table(png_table)

    compile_info: dict[str, Any] = {"skipped": True, "binary": str(binary)}
    if not bool(args.skip_compile):
        compile_info = compile_marisa_b(binary)
        if int(compile_info["returncode"]) != 0:
            payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "compile_failed", "compile": compile_info}
            out_json = output_dir / f"{args.prefix}.json"
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            raise SystemExit(1)

    nmu_values = unique_sorted_nmu_values(int(args.nmu_b000), args.mu_convergence_nmu)
    extra_nmu_values = [item for item in nmu_values if item != int(args.nmu_b000)]
    radial_factor = int(args.nradial_binavg) ** 2 if str(args.bin_average) == "shell" else 1
    n_extra_convergence_triangles = int(indices.size) * int(radial_factor) * int(sum(extra_nmu_values))
    if n_extra_convergence_triangles > int(args.max_mu_convergence_triangles):
        raise ValueError(
            "mu convergence request would run "
            f"{n_extra_convergence_triangles} extra triangles, exceeding "
            f"--max-mu-convergence-triangles={args.max_mu_convergence_triangles}. "
            "Use fewer pairs or raise the cap explicitly."
        )
    if args.radial_convergence_nradial and str(args.bin_average) != "shell":
        raise ValueError("--radial-convergence-nradial requires --bin-average shell")
    nradial_values = unique_sorted_nmu_values(
        int(args.nradial_binavg),
        args.radial_convergence_nradial,
    )
    extra_nradial_values = [
        item for item in nradial_values if item != int(args.nradial_binavg)
    ]
    n_extra_radial_convergence_triangles = (
        int(indices.size)
        * int(args.nmu_b000)
        * int(sum(item * item for item in extra_nradial_values))
        if str(args.bin_average) == "shell"
        else 0
    )
    if n_extra_radial_convergence_triangles > int(
        args.max_radial_convergence_triangles
    ):
        raise ValueError(
            "radial convergence request would run "
            f"{n_extra_radial_convergence_triangles} extra triangles, exceeding "
            "--max-radial-convergence-triangles="
            f"{args.max_radial_convergence_triangles}. "
            "Use fewer pairs or raise the cap explicitly."
        )

    selected_components, chunk_summaries, n_triangles, work_summary = run_b000_projection(
        binary=binary,
        png_table=png_table,
        data=data,
        indices=indices,
        theory_pairs=theory_pairs,
        nmu_b000=int(args.nmu_b000),
        args=args,
        cpp_mode=cpp_mode,
    )
    component_vectors = selected_components_to_full_vectors(selected_components, indices, n_pairs)
    theory_vector = component_vectors[primary_component]
    computed_mask = np.zeros(n_pairs, dtype=bool)
    computed_mask[indices] = True

    selected_by_nmu: dict[int, dict[str, np.ndarray]] = {int(args.nmu_b000): selected_components}
    convergence_chunk_summaries: list[dict[str, Any]] = []
    for nmu in extra_nmu_values:
        conv_components, conv_chunks, conv_ntri, conv_work_summary = run_b000_projection(
            binary=binary,
            png_table=png_table,
            data=data,
            indices=indices,
            theory_pairs=theory_pairs,
            nmu_b000=int(nmu),
            args=args,
            cpp_mode=cpp_mode,
        )
        selected_by_nmu[int(nmu)] = conv_components
        convergence_chunk_summaries.append(
            {
                "nmu_b000": int(nmu),
                "n_triangles": int(conv_ntri),
                "work": conv_work_summary,
                "total_cpp_wall_seconds": float(np.nansum([item.get("wall_seconds", math.nan) for item in conv_chunks])),
                "max_chunk_wall_seconds": float(np.nanmax([item.get("wall_seconds", math.nan) for item in conv_chunks])),
                "chunks": conv_chunks,
            }
        )

    mu_convergence_summary: dict[str, Any] = {"enabled": False}
    mu_convergence_arrays: dict[str, np.ndarray] = {}
    if args.mu_convergence_nmu:
        mu_convergence_summary, mu_convergence_arrays = summarize_mu_convergence(
            nmu_values=nmu_values,
            indices=indices,
            selected_by_nmu=selected_by_nmu,
            primary_component=primary_component,
        )

    selected_by_nradial: dict[int, dict[str, np.ndarray]] = {
        int(args.nradial_binavg): selected_components
    }
    radial_convergence_chunk_summaries: list[dict[str, Any]] = []
    for nradial in extra_nradial_values:
        radial_components, radial_chunks, radial_ntri, radial_work_summary = (
            run_b000_projection(
                binary=binary,
                png_table=png_table,
                data=data,
                indices=indices,
                theory_pairs=theory_pairs,
                nmu_b000=int(args.nmu_b000),
                args=args,
                cpp_mode=cpp_mode,
                nradial_binavg=int(nradial),
            )
        )
        selected_by_nradial[int(nradial)] = radial_components
        radial_convergence_chunk_summaries.append(
            {
                "nradial_binavg": int(nradial),
                "n_triangles": int(radial_ntri),
                "work": radial_work_summary,
                "total_cpp_wall_seconds": float(
                    np.nansum(
                        [
                            item.get("wall_seconds", math.nan)
                            for item in radial_chunks
                        ]
                    )
                ),
                "max_chunk_wall_seconds": float(
                    np.nanmax(
                        [
                            item.get("wall_seconds", math.nan)
                            for item in radial_chunks
                        ]
                    )
                ),
                "chunks": radial_chunks,
            }
        )

    radial_convergence_summary: dict[str, Any] = {"enabled": False}
    radial_convergence_arrays: dict[str, np.ndarray] = {}
    if args.radial_convergence_nradial:
        radial_convergence_summary, radial_convergence_arrays = (
            summarize_radial_convergence(
                nradial_values=nradial_values,
                indices=indices,
                selected_by_nradial=selected_by_nradial,
                primary_component=primary_component,
            )
        )

    sim_derivative = (
        None
        if args.png_loop_level == "b112ii"
        else load_sim_derivative(Path(args.sim_derivative_npz), n_pairs)
    )
    if sim_derivative is not None and sim_derivative["pre_k_pairs"].size:
        max_k_delta = float(np.max(np.abs(sim_derivative["pre_k_pairs"] - theory_pairs)))
        if max_k_delta > 1.0e-6:
            raise ValueError(f"simulation derivative k pairs do not align with theory pairs: max delta {max_k_delta}")
    diag_arrays = compact_diagonal(data, theory_pairs, computed_mask, component_vectors, primary_component, sim_derivative)
    checks = validation_checks(binary, png_table, args, cpp_mode, primary_component)

    sim_stats = None
    if sim_derivative is not None:
        sim_stats = {
            "all_computed": comparison_stats(theory_vector, sim_derivative["pre_central_dB000_dfNL"], computed_mask),
            "diagonal": comparison_stats(
                theory_vector,
                sim_derivative["pre_central_dB000_dfNL"],
                computed_mask & np.asarray(data["pair_i"] == data["pair_j"], dtype=bool),
            ),
        }

    out_npz = output_dir / f"{args.prefix}.npz"
    np.savez(
        out_npz,
        pair_index=np.arange(n_pairs, dtype=np.int64),
        computed_pair_indices=indices.astype(np.int64, copy=False),
        computed_mask=computed_mask,
        pair_i=np.asarray(data["pair_i"], dtype=np.int64),
        pair_j=np.asarray(data["pair_j"], dtype=np.int64),
        k_centers=np.asarray(data["k_centers"], dtype=np.float64),
        hdf5_k_edges=np.asarray(data["hdf5_k_edges"], dtype=np.float64),
        k1_nominal=nominal_pairs[:, 0],
        k2_nominal=nominal_pairs[:, 1],
        k1_weighted=weighted_pairs[:, 0],
        k2_weighted=weighted_pairs[:, 1],
        k1_theory=theory_pairs[:, 0],
        k2_theory=theory_pairs[:, 1],
        bin_average=np.asarray(str(args.bin_average)),
        nradial_binavg=np.asarray(int(args.nradial_binavg), dtype=np.int64),
        qmin=np.asarray(float(args.qmin), dtype=np.float64),
        qmax=np.asarray(float(args.qmax), dtype=np.float64),
        png_ir_cutoff=np.asarray(float(args.png_ir_cutoff), dtype=np.float64),
        b112ii_integrator=np.asarray(str(args.b112ii_integrator)),
        b112ii_qmc_power=np.asarray(
            int(args.b112ii_qmc_power), dtype=np.int64
        ),
        b112ii_qmc_replicates=np.asarray(
            int(args.b112ii_qmc_replicates), dtype=np.int64
        ),
        reconstruction=np.asarray(str(args.reconstruction)),
        smoothing_radius=np.asarray(
            float(args.smoothing_radius), dtype=np.float64
        ),
        bias_recon=np.asarray(float(args.bias_recon), dtype=np.float64),
        epsrel=np.asarray(float(args.epsrel), dtype=np.float64),
        **{B000_FIELD_BY_COMPONENT[component]: vector for component, vector in component_vectors.items()},
        **mu_convergence_arrays,
        **radial_convergence_arrays,
        B000_primary=theory_vector,
        B000_dBdfNL_primary=theory_vector,
        sim_pre_central_dB000_dfNL=sim_derivative["pre_central_dB000_dfNL"] if sim_derivative is not None else np.full(n_pairs, np.nan),
        sim_pre_central_dB000_dfNL_stderr=sim_derivative["pre_central_dB000_dfNL_stderr"] if sim_derivative is not None else np.full(n_pairs, np.nan),
    )
    out_diag_npz = None
    if args.png_loop_level != "b112ii":
        out_diag_npz = output_dir / f"{args.prefix}_diagonal_compact.npz"
        np.savez(out_diag_npz, **diag_arrays)

    out_pdf = None
    if not bool(args.no_plot) and args.png_loop_level != "b112ii":
        out_pdf = output_dir / f"{args.prefix}_diag_compare.pdf"
        plot_diagonal(out_pdf, diag_arrays, plot_label)

    summary = {
        "n_pairs_total": int(n_pairs),
        "n_pairs_computed": int(indices.size),
        "n_triangles": int(n_triangles),
        "n_diagonal_compact": int(diag_arrays["k_eff"].size),
        "total_cpp_wall_seconds": float(np.nansum([item.get("wall_seconds", math.nan) for item in chunk_summaries])),
        "max_chunk_wall_seconds": float(np.nanmax([item.get("wall_seconds", math.nan) for item in chunk_summaries])),
        "mu_convergence_extra_triangles": int(n_extra_convergence_triangles),
        "mu_convergence_cpp_wall_seconds": float(
            np.nansum([item.get("total_cpp_wall_seconds", math.nan) for item in convergence_chunk_summaries])
        )
        if convergence_chunk_summaries
        else 0.0,
        "radial_convergence_extra_triangles": int(
            n_extra_radial_convergence_triangles
        ),
        "radial_convergence_cpp_wall_seconds": float(
            np.nansum(
                [
                    item.get("total_cpp_wall_seconds", math.nan)
                    for item in radial_convergence_chunk_summaries
                ]
            )
        )
        if radial_convergence_chunk_summaries
        else 0.0,
        "vector_min": float(np.nanmin(theory_vector[computed_mask])),
        "vector_max": float(np.nanmax(theory_vector[computed_mask])),
    }
    out_json = output_dir / f"{args.prefix}.json"
    table_source = str(table_info.get("source", "unknown"))
    if table_source == "reused_existing_table":
        pl_description = (
            f"P_L column reused verbatim from {png_table}; --z is not applied "
            "to a supplied --png-table, so the table's own provenance is "
            "authoritative"
        )
        pl_redshift_argument_applied = False
    elif table_source.startswith("generated_cosmoprimo"):
        pl_description = (
            "cosmoprimo CLASS delta_cb "
            f"P_L(k,z={float(args.z):g}), Quijote fiducial"
        )
        pl_redshift_argument_applied = True
    else:
        pl_description = (
            f"P_L column reused from {args.plin_table} while constructing "
            f"{png_table}; --z is not applied to the fallback P_L table"
        )
        pl_redshift_argument_applied = False

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "engine": {
            "name": "MARISA-B",
            "backend": "native_cpp",
            "mode": cpp_mode,
            "native_scope": (
                "isolated local-PNG B112II/fNL^2 matter triangle kernel and B000 projection"
                if args.png_loop_level == "b112ii"
                else "local PNG matter dB/dfNL triangle kernel and B000 projection"
            ),
        },
        "outputs": {
            "npz": str(out_npz),
            "diagonal_compact_npz": (
                str(out_diag_npz) if out_diag_npz is not None else None
            ),
            "json": str(out_json),
            "pdf": str(out_pdf) if out_pdf else None,
            "png_table": str(png_table),
        },
        "matrix": str(matrix_path),
        "selection": {
            "k_source": str(args.k_source),
            "kmax": float(args.kmax),
            "compute_all_pairs": bool(args.compute_all_pairs),
            "diagonal_only": bool(args.diagonal_only),
            "explicit_pair_indices": [int(item) for item in args.pair_index] if args.pair_index else None,
            "png_loop_level": str(args.png_loop_level),
            "reconstruction": str(args.reconstruction),
            "primary_component": primary_component,
            "bin_average": str(args.bin_average),
            "nradial_binavg": int(args.nradial_binavg) if str(args.bin_average) == "shell" else None,
        },
        "integration": {
            "nmu_b000": int(args.nmu_b000),
            "epsrel": float(args.epsrel),
            "p13_epsrel": float(args.p13_epsrel),
            "qmin": float(args.qmin),
            "qmax": float(args.qmax),
            "png_ir_cutoff": float(args.png_ir_cutoff),
            "b112ii_integrator": str(args.b112ii_integrator),
            "b112ii_qmc_power": int(args.b112ii_qmc_power),
            "b112ii_qmc_replicates": int(args.b112ii_qmc_replicates),
            "reconstruction": str(args.reconstruction),
            "smoothing_radius": float(args.smoothing_radius),
            "bias_recon": float(args.bias_recon),
            "singular_floor": float(args.singular_floor),
            "max_triangles_per_call": int(args.max_triangles_per_call),
            "chunk_workers": int(args.chunk_workers),
            "mu_convergence_nmu_values": [int(item) for item in nmu_values] if args.mu_convergence_nmu else None,
            "max_mu_convergence_triangles": int(args.max_mu_convergence_triangles),
            "radial_convergence_nradial_values": (
                [int(item) for item in nradial_values]
                if args.radial_convergence_nradial
                else None
            ),
            "max_radial_convergence_triangles": int(
                args.max_radial_convergence_triangles
            ),
        },
        "work": work_summary,
        "mu_convergence": mu_convergence_summary,
        "radial_convergence": radial_convergence_summary,
        "local_png_convention": {
            "tree_formula": "dB111/dfNL = 2*b1^3*[P1*P2*M3/(M1*M2)+P2*P3*M1/(M2*M3)+P3*P1*M2/(M3*M1)]",
            "one_loop_formula": "linear-in-fNL dB/dfNL adds B122^I + B122^II + B113^I + B113^II; B112^II is proportional to fNL^2 and is omitted for the central derivative at fNL=0",
            "finite_fnl_O_PL3_formula": "B112II = fNL^2 * int_q F2(q,k3-q) * T0_hat(k1,k2,q,k3-q) + 2 cyclic; T0_hat contains all 12 local-tau exchange terms",
            "M_definition": "M(k,z)=sqrt(P_L(k,z)/P_phi(k))",
            "P_L": pl_description,
            "P_L_redshift_argument": float(args.z),
            "P_L_redshift_argument_applied": pl_redshift_argument_applied,
            "P_phi": "Hitomi dimensional curvature spectrum in (Mpc/h)^3 with k_pivot=0.05 h/Mpc and (3/5)^2 factor",
            "fallback_A_s_if_no_cosmoprimo": float(args.a_s),
            "b1": float(args.b1),
            "b000_projection": "B000(k1,k2)=1/2 int_{-1}^{1} dmu12 B(k1,k2,k3)",
            "finite_box_shell_policy": (
                "retain the original HDF5 shell volume and angular normalization; "
                "the native png_ir_cutoff sets forbidden external and exchange "
                "primordial modes to zero, matching the estimator audit that "
                "retains zero-mode pairs in the normalization"
            ),
            "references": [
                "Sefusatti 2009 / Sefusatti, Crocce & Desjacques 2010: B111=B0 and delta=M Phi conventions",
                "notes/fig4_b000_derivative_pitfalls.md: Hitomi dimensional P_phi convention",
            ],
        },
        "transfer_table": table_info,
        "pk_table_description_for_cpp": pk_info,
        "summary": summary,
        "validation": checks,
        "simulation_comparison": {"npz": str(args.sim_derivative_npz), "stats": sim_stats},
        "compile": compile_info,
        "chunks": chunk_summaries,
        "mu_convergence_chunks": convergence_chunk_summaries,
        "radial_convergence_chunks": radial_convergence_chunk_summaries,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
