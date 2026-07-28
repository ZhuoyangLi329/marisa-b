#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Produce Quijote DM pre-recon B000 vectors with MARISA-B.

The C++ executable performs the triangle-level SPT calculation. Python only
handles Quijote pair selection, B000 outer mu projection, data-vector alignment,
I/O, and optional ReACT gate checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


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

import numpy as np

from audit_quijote_dm_spt_react_bspt_driver import write_pk_table
from quijote_dm_spt1loop_realspace import (
    DEFAULT_DM_MATRIX,
    QUIJOTE_COSMOLOGY,
    load_quijote_dm_pre_mean,
    quijote_power_normalization_audit,
)
from run_marisa_b_triangle import (
    DEFAULT_BUILD_DIR,
    DEFAULT_OUTPUT_DIR,
    compile_marisa_b,
    describe_existing_pk_table,
    run_marisa_b,
)


COMPONENT_MAP = {
    "B000_Btree": "Btree",
    "B000_B222": "B222",
    "B000_B321I": "B321I",
    "B000_B321II": "B321II",
    "B000_B411": "B411",
    "B000_Bloopterms": "Bloopterms",
    "B000_B1loop": "B1loop",
    "B000_Btotal": "Btotal",
}

COMPARE_COMPONENTS = ("B222", "B321I", "B321II", "B411", "B1loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_DM_MATRIX))
    parser.add_argument("--fiducial-sample", choices=("common", "all"), default="common")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="marisa_b_native_b000_vector_kmax0p305_nmu12")
    parser.add_argument("--k-source", choices=("weighted", "center"), default="weighted")
    parser.add_argument("--kmax", type=float, default=0.305)
    parser.add_argument("--compute-all-pairs", action="store_true")
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--pair-index", action="append", type=int, default=None)
    parser.add_argument("--nmu-b000", type=int, default=12)
    parser.add_argument(
        "--bin-average",
        choices=("point", "shell"),
        default="point",
        help=(
            "B000 projection convention. 'point' keeps the historical weighted-k "
            "single-point projection; 'shell' averages over the two jaxpower k-bin "
            "shells with k^2 dk radial weights and the same outer mu projection."
        ),
    )
    parser.add_argument(
        "--nradial-binavg",
        type=int,
        default=4,
        help="Gauss-Legendre radial nodes per k shell when --bin-average=shell.",
    )
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--epsrel", type=float, default=1.0e-3)
    parser.add_argument("--p13-epsrel", type=float, default=1.0e-2)
    parser.add_argument("--qmin", type=float, default=1.0e-4)
    parser.add_argument("--qmax", type=float, default=30.0)
    parser.add_argument("--singular-floor", type=float, default=1.0e-5)
    parser.add_argument("--backend", choices=("native_cpp", "react_gr_bootstrap"), default="native_cpp")
    parser.add_argument("--pk-table", default=None)
    parser.add_argument(
        "--pk-amplitude-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the linear P_L table by this constant before passing it to "
            "MARISA-B. This is an explicit diagnostic/fix knob for effective "
            "growth or sigma8 normalization tests; default 1 leaves P_L unchanged."
        ),
    )
    parser.add_argument("--pk-kmin", type=float, default=1.0e-4)
    parser.add_argument("--pk-kmax", type=float, default=80.0)
    parser.add_argument("--pk-n", type=int, default=2400)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--max-triangles-per-call", type=int, default=72)
    parser.add_argument("--react-npz", default=None, help="Optional ReACT component/vector NPZ for gate validation.")
    parser.add_argument("--gate-rms", type=float, default=0.01)
    parser.add_argument("--gate-max-abs", type=float, default=0.03)
    return parser.parse_args()


def scale_pk_table(source: Path, target: Path, amplitude_scale: float) -> dict[str, object]:
    """Write a two-column P_L table with the second column multiplied by a scale."""

    if not np.isfinite(float(amplitude_scale)) or float(amplitude_scale) <= 0.0:
        raise ValueError("--pk-amplitude-scale must be finite and positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    n_scaled = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        dst.write(f"# amplitude-scaled copy of {source}\n")
        dst.write(f"# pk_amplitude_scale = {float(amplitude_scale):.17g}\n")
        for line in src:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                dst.write(line)
                continue
            parts = stripped.split()
            if len(parts) < 2:
                dst.write(line)
                continue
            try:
                kval = float(parts[0])
                pval = float(parts[1])
            except ValueError:
                dst.write(line)
                continue
            dst.write(f"{kval:.17e} {pval * float(amplitude_scale):.17e}\n")
            n_scaled += 1
    info = describe_existing_pk_table(target)
    info.update(
        {
            "amplitude_scale": float(amplitude_scale),
            "unscaled_source_path": str(source),
            "n_scaled_rows": int(n_scaled),
            "scale_interpretation": "P_L -> amplitude_scale * P_L; tree scales as A^2 and 1-loop as A^3.",
        }
    )
    return info


def prepare_pk_table(args: argparse.Namespace, output_dir: Path) -> tuple[Path, dict[str, object], dict[str, object] | None]:
    """Prepare the P_L table, optionally applying an explicit amplitude scale."""

    scale = float(args.pk_amplitude_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("--pk-amplitude-scale must be finite and positive")

    if args.pk_table:
        base_pk_table = Path(args.pk_table)
        base_pk_info = describe_existing_pk_table(base_pk_table)
    else:
        base_pk_table = output_dir / f"{args.prefix}_plin_z{args.z:g}.dat"
        base_pk_info = write_pk_table(base_pk_table, float(args.z), float(args.pk_kmin), float(args.pk_kmax), int(args.pk_n))

    if abs(scale - 1.0) <= 1.0e-15:
        base_pk_info = dict(base_pk_info)
        base_pk_info["amplitude_scale"] = 1.0
        return base_pk_table, base_pk_info, None

    scaled_pk_table = output_dir / f"{args.prefix}_plin_z{args.z:g}_Ascale{scale:.8g}.dat"
    scaled_pk_info = scale_pk_table(base_pk_table, scaled_pk_table, scale)
    return scaled_pk_table, scaled_pk_info, base_pk_info


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


def shell_nodes(k_edges: np.ndarray, nradial: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = map(float, np.asarray(k_edges, dtype=np.float64))
    if not right > left:
        raise ValueError(f"invalid k shell edges: {k_edges}")
    if int(nradial) <= 0:
        raise ValueError("--nradial-binavg must be positive")
    x, w = np.polynomial.legendre.leggauss(int(nradial))
    k = 0.5 * (right + left) + 0.5 * (right - left) * x
    dk_weight = 0.5 * (right - left) * w
    shell_weight = dk_weight * k * k
    norm = (right**3 - left**3) / 3.0
    if not norm > 0.0:
        raise ValueError(f"invalid k shell normalization for edges: {k_edges}")
    return k.astype(np.float64, copy=False), (shell_weight / norm).astype(np.float64, copy=False)


def build_b000_work(
    *,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    theory_pairs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, float]], dict[str, object]]:
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(args.nmu_b000))
    triangles: list[tuple[float, float, float]] = []
    triangle_meta: list[tuple[int, float]] = []
    per_pair_triangles: list[int] = []
    per_pair_weight_sums: list[float] = []

    if args.bin_average == "point":
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
    elif args.bin_average == "shell":
        edges = np.asarray(data["hdf5_k_edges"], dtype=np.float64)
        for local_index, pair_index in enumerate(indices):
            pair_edges = edges[int(pair_index)]
            k1_nodes, k1_weights = shell_nodes(pair_edges[0], int(args.nradial_binavg))
            k2_nodes, k2_weights = shell_nodes(pair_edges[1], int(args.nradial_binavg))
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
        raise ValueError(f"unknown bin_average mode: {args.bin_average!r}")

    weight_sums = np.asarray(per_pair_weight_sums, dtype=np.float64)
    work_summary: dict[str, object] = {
        "bin_average": str(args.bin_average),
        "nmu_b000": int(args.nmu_b000),
        "nradial_binavg": int(args.nradial_binavg) if args.bin_average == "shell" else None,
        "n_selected_pairs": int(indices.size),
        "n_triangles": int(len(triangles)),
        "triangles_per_pair_min": int(min(per_pair_triangles)) if per_pair_triangles else 0,
        "triangles_per_pair_max": int(max(per_pair_triangles)) if per_pair_triangles else 0,
        "pre_half_weight_sum_min": float(np.min(weight_sums)) if weight_sums.size else math.nan,
        "pre_half_weight_sum_max": float(np.max(weight_sums)) if weight_sums.size else math.nan,
        "normalization_note": "accumulate_b000 multiplies each stored weight by 1/2; pre_half_weight_sum should be close to 2.",
    }
    return triangles, triangle_meta, work_summary


def run_marisa_chunks(
    *,
    binary: Path,
    pk_table: Path,
    triangles: list[tuple[float, float, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    chunk_size = max(1, int(args.max_triangles_per_call))
    for start in range(0, len(triangles), chunk_size):
        chunk = triangles[start : start + chunk_size]
        info = run_marisa_b(
            binary,
            pk_table,
            chunk,
            float(args.epsrel),
            float(args.p13_epsrel),
            float(args.qmin),
            float(args.qmax),
            10.0,
            1.0,
            0.0,
            0,
            float(args.singular_floor),
            "pre_recon_gaussian",
            str(args.backend),
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
        chunks.append(chunk_record)
        if int(info["returncode"]) != 0 or len(parsed) != len(chunk):
            raise RuntimeError(f"MARISA-B chunk failed: {chunk_record}")
        rows.extend(parsed)
    return rows, chunks


def empty_vectors(n_pairs: int) -> dict[str, np.ndarray]:
    return {key: np.full(n_pairs, np.nan, dtype=np.float64) for key in COMPONENT_MAP}


def accumulate_b000(rows: list[dict[str, object]], triangle_meta: list[tuple[int, float]], n_selected: int) -> list[dict[str, float]]:
    accum = [{key: 0.0 for key in COMPONENT_MAP} for _ in range(n_selected)]
    for row, (local_index, weight) in zip(rows, triangle_meta):
        factor = 0.5 * float(weight)
        for out_key, row_key in COMPONENT_MAP.items():
            accum[local_index][out_key] += factor * float(row[row_key])
    return accum


def make_full_vectors(n_pairs: int, indices: np.ndarray, accum: list[dict[str, float]]) -> dict[str, np.ndarray]:
    vectors = empty_vectors(n_pairs)
    for local_index, pair_index in enumerate(indices):
        for key in COMPONENT_MAP:
            vectors[key][int(pair_index)] = float(accum[local_index][key])
    return vectors


def residual_stats(sim: np.ndarray, theory: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(sim) & np.isfinite(theory) & (sim != 0.0)
    frac = theory[valid] / sim[valid] - 1.0
    return {
        "n": int(frac.size),
        "rms_fractional_residual": float(np.sqrt(np.mean(frac * frac))) if frac.size else math.nan,
        "max_abs_fractional_residual": float(np.max(np.abs(frac))) if frac.size else math.nan,
        "median_fractional_residual": float(np.median(frac)) if frac.size else math.nan,
    }


def align_react_component(react: np.lib.npyio.NpzFile, key: str, n_pairs: int) -> tuple[np.ndarray, np.ndarray, str]:
    raw = np.asarray(react[key], dtype=np.float64)
    if raw.shape == (n_pairs,):
        mask = np.isfinite(raw)
        if "computed_mask" in react.files:
            mask &= np.asarray(react["computed_mask"], dtype=bool)
        return raw, mask, "full_vector"

    for index_key in ("pair_indices", "computed_pair_indices", "pair_index"):
        if index_key not in react.files:
            continue
        indices = np.asarray(react[index_key], dtype=np.int64)
        if indices.shape != raw.shape:
            continue
        if np.any(indices < 0) or np.any(indices >= n_pairs):
            continue
        aligned = np.full(n_pairs, np.nan, dtype=np.float64)
        aligned[indices] = raw
        return aligned, np.isfinite(aligned), f"compact_by_{index_key}"

    raise ValueError(
        f"cannot align ReACT component {key!r}: shape={raw.shape}, n_pairs={n_pairs}; "
        "expected a full vector or a compact array with pair_indices/computed_pair_indices"
    )


def component_gate(vectors: dict[str, np.ndarray], react_npz: Path, computed_mask: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    with np.load(react_npz) as react:
        n_pairs = int(computed_mask.size)
        btree, react_mask, alignment = align_react_component(react, "B000_Btree", n_pairs)
        mask = computed_mask & react_mask & np.isfinite(btree) & (btree != 0.0)
        stats: dict[str, object] = {
            "react_npz": str(react_npz),
            "alignment": alignment,
            "n_compared": int(np.count_nonzero(mask)),
        }
        component_stats: dict[str, object] = {}
        for component in COMPARE_COMPONENTS:
            key = f"B000_{component}"
            if key not in react.files or key not in vectors:
                component_stats[component] = {"status": "missing"}
                continue
            react_component, component_mask, _ = align_react_component(react, key, n_pairs)
            valid = mask & component_mask & np.isfinite(vectors[key])
            delta = (vectors[key][valid] - react_component[valid]) / btree[valid]
            component_stats[component] = {
                "n": int(delta.size),
                "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else math.nan,
                "max_abs": float(np.max(np.abs(delta))) if delta.size else math.nan,
            }
        b1 = component_stats.get("B1loop", {})
        passed = bool(
            int(stats["n_compared"]) > 0
            and float(b1.get("rms", math.inf)) <= float(args.gate_rms)
            and float(b1.get("max_abs", math.inf)) <= float(args.gate_max_abs)
        )
        stats.update(
            {
                "status": "pass" if passed else "fail",
                "thresholds": {"rms": float(args.gate_rms), "max_abs": float(args.gate_max_abs)},
                "component_stats": component_stats,
            }
        )
        return stats


def compact_diagonal_arrays(
    data: dict[str, np.ndarray],
    theory_pairs: np.ndarray,
    computed_mask: np.ndarray,
    vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    diag = np.asarray(data["pair_i"] == data["pair_j"], dtype=bool) & computed_mask
    order = np.argsort(np.mean(theory_pairs[diag], axis=1))
    indices = np.nonzero(diag)[0][order]
    arrays = {
        "diag_indices": indices.astype(np.int64),
        "k_eff": np.mean(theory_pairs[indices], axis=1),
        "sim_B000": np.asarray(data["mean"], dtype=np.float64)[indices],
        "sim_B000_stderr": np.asarray(data["stderr"], dtype=np.float64)[indices],
    }
    for key in COMPONENT_MAP:
        arrays[key] = vectors[key][indices]
    arrays["fractional_residual_tree"] = arrays["B000_Btree"] / arrays["sim_B000"] - 1.0
    arrays["fractional_residual_total"] = arrays["B000_Btotal"] / arrays["sim_B000"] - 1.0
    return arrays


def rows_for_json(indices: np.ndarray, data: dict[str, np.ndarray], theory_pairs: np.ndarray, vectors: dict[str, np.ndarray]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for pair_index in indices:
        idx = int(pair_index)
        edges = np.asarray(data["hdf5_k_edges"][idx], dtype=np.float64)
        row: dict[str, object] = {
            "pair_index": idx,
            "pair_i": int(data["pair_i"][idx]),
            "pair_j": int(data["pair_j"][idx]),
            "k1_theory": float(theory_pairs[idx, 0]),
            "k2_theory": float(theory_pairs[idx, 1]),
            "k1_edges": edges[0].tolist(),
            "k2_edges": edges[1].tolist(),
            "sim_B000": float(data["mean"][idx]),
            "sim_B000_stderr": float(data["stderr"][idx]),
        }
        for key in COMPONENT_MAP:
            row[key] = float(vectors[key][idx])
        row["B1loop_over_Btree"] = float(row["B000_B1loop"]) / float(row["B000_Btree"]) if float(row["B000_Btree"]) != 0.0 else math.nan
        out.append(row)
    return out


def main() -> None:
    enforce_thread_limit()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = Path(args.matrix)
    data = load_quijote_dm_pre_mean(matrix_path, fiducial_sample=str(args.fiducial_sample))
    nominal_pairs, weighted_pairs, theory_pairs = theory_k_pairs(data, str(args.k_source))
    indices = selected_pair_indices(data, theory_pairs, args)
    n_pairs = int(data["mean"].shape[0])

    build_dir = Path(args.build_dir)
    binary = build_dir / "marisa_b_triangle"
    pk_table, pk_info, unscaled_pk_info = prepare_pk_table(args, output_dir)

    compile_info: dict[str, object] = {"skipped": True, "binary": str(binary)}
    if not bool(args.skip_compile):
        compile_info = compile_marisa_b(binary)
        if int(compile_info["returncode"]) != 0:
            out_json = output_dir / f"{args.prefix}.json"
            payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "compile_failed", "compile": compile_info}
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            raise SystemExit(1)

    triangles, triangle_meta, work_summary = build_b000_work(data=data, indices=indices, theory_pairs=theory_pairs, args=args)

    rows, chunk_summaries = run_marisa_chunks(binary=binary, pk_table=pk_table, triangles=triangles, args=args)
    accum = accumulate_b000(rows, triangle_meta, int(indices.size))
    vectors = make_full_vectors(n_pairs, indices, accum)
    computed_mask = np.zeros(n_pairs, dtype=bool)
    computed_mask[indices] = True

    sim = np.asarray(data["mean"], dtype=np.float64)
    stderr = np.asarray(data["stderr"], dtype=np.float64)
    frac_tree = np.full(n_pairs, np.nan, dtype=np.float64)
    frac_total = np.full(n_pairs, np.nan, dtype=np.float64)
    valid_sim = sim != 0.0
    frac_tree[computed_mask & valid_sim] = vectors["B000_Btree"][computed_mask & valid_sim] / sim[computed_mask & valid_sim] - 1.0
    frac_total[computed_mask & valid_sim] = vectors["B000_Btotal"][computed_mask & valid_sim] / sim[computed_mask & valid_sim] - 1.0

    gate = None
    if args.react_npz:
        gate = component_gate(vectors, Path(args.react_npz), computed_mask, args)

    diag_arrays = compact_diagonal_arrays(data, theory_pairs, computed_mask, vectors)
    out_npz = output_dir / f"{args.prefix}.npz"
    np.savez(
        out_npz,
        pair_index=np.arange(n_pairs, dtype=np.int64),
        computed_pair_indices=indices.astype(np.int64, copy=False),
        computed_mask=computed_mask,
        pair_i=np.asarray(data["pair_i"], dtype=np.int64),
        pair_j=np.asarray(data["pair_j"], dtype=np.int64),
        k_centers=np.asarray(data["k_centers"], dtype=np.float64),
        k1_nominal=nominal_pairs[:, 0],
        k2_nominal=nominal_pairs[:, 1],
        k1_weighted=weighted_pairs[:, 0],
        k2_weighted=weighted_pairs[:, 1],
        k1_theory=theory_pairs[:, 0],
        k2_theory=theory_pairs[:, 1],
        hdf5_k_edges=np.asarray(data["hdf5_k_edges"], dtype=np.float64),
        bin_average=np.asarray(str(args.bin_average)),
        nradial_binavg=np.asarray(int(args.nradial_binavg), dtype=np.int64),
        pk_amplitude_scale=np.asarray(float(args.pk_amplitude_scale), dtype=np.float64),
        qmin=np.asarray(float(args.qmin), dtype=np.float64),
        qmax=np.asarray(float(args.qmax), dtype=np.float64),
        epsrel=np.asarray(float(args.epsrel), dtype=np.float64),
        p13_epsrel=np.asarray(float(args.p13_epsrel), dtype=np.float64),
        singular_floor=np.asarray(float(args.singular_floor), dtype=np.float64),
        sim_B000=sim,
        sim_B000_stderr=stderr,
        fractional_residual_tree=frac_tree,
        fractional_residual_total=frac_total,
        **vectors,
    )

    out_diag_npz = output_dir / f"{args.prefix}_diagonal_compact.npz"
    np.savez(out_diag_npz, **diag_arrays)

    summary = {
        "n_pairs_total": int(n_pairs),
        "n_pairs_computed": int(indices.size),
        "n_triangles": int(len(triangles)),
        "n_diagonal_compact": int(diag_arrays["k_eff"].size),
        "work": work_summary,
        "tree_residual_stats": residual_stats(sim, vectors["B000_Btree"], computed_mask),
        "total_residual_stats": residual_stats(sim, vectors["B000_Btotal"], computed_mask),
        "total_cpp_wall_seconds": float(np.nansum([item.get("wall_seconds", math.nan) for item in chunk_summaries])),
        "max_chunk_wall_seconds": float(np.nanmax([item.get("wall_seconds", math.nan) for item in chunk_summaries])),
    }

    out_json = output_dir / f"{args.prefix}.json"
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "engine": {
            "name": "MARISA-B",
            "backend": str(args.backend),
            "native_scope": "pre-recon Gaussian real-space DM SPT tree+1loop",
            "note": "native_cpp does not call BSPT::Bloop/Bloopterms or SPT::P13_dd",
        },
        "outputs": {"npz": str(out_npz), "diagonal_compact_npz": str(out_diag_npz), "json": str(out_json), "pk_table": str(pk_table)},
        "matrix": str(matrix_path),
        "fiducial_sample": str(args.fiducial_sample),
        "n_realizations": int(data["n_realizations"]),
        "selection": {
            "k_source": str(args.k_source),
            "kmax": float(args.kmax),
            "compute_all_pairs": bool(args.compute_all_pairs),
            "diagonal_only": bool(args.diagonal_only),
            "explicit_pair_indices": [int(item) for item in args.pair_index] if args.pair_index else None,
        },
        "integration": {
            "epsrel": float(args.epsrel),
            "p13_epsrel": float(args.p13_epsrel),
            "qmin": float(args.qmin),
            "qmax": float(args.qmax),
            "singular_floor": float(args.singular_floor),
            "nmu_b000": int(args.nmu_b000),
            "bin_average": str(args.bin_average),
            "nradial_binavg": int(args.nradial_binavg) if args.bin_average == "shell" else None,
            "max_triangles_per_call": int(args.max_triangles_per_call),
        },
        "linear_power_convention": {
            "z": float(args.z),
            "source": "cosmoprimo CLASS delta_cb P_L(k,z), or reused table if --pk-table is set",
            "cosmology": dict(QUIJOTE_COSMOLOGY),
            "pk_table": pk_info,
            "unscaled_pk_table": unscaled_pk_info,
            "pk_amplitude_scale": float(args.pk_amplitude_scale),
            "power_normalization_audit": None if args.pk_table else quijote_power_normalization_audit(float(args.z)),
            "amplitude_scale_note": "If pk_amplitude_scale=A, MARISA-B receives P_L -> A P_L; Btree scales as A^2 and B1loop as A^3.",
        },
        "summary": summary,
        "gate": gate,
        "rows_computed": rows_for_json(indices, data, theory_pairs, vectors),
        "compile": compile_info,
        "chunks": chunk_summaries,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
