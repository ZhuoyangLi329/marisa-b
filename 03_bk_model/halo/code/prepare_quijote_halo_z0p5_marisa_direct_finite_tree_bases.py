#!/usr/bin/env python3
"""Build the missing finite-fNL adaptive Marisa tree coefficients at z=0.5.

The fixed-b_rec standard tree model already contains its finite local-PNG
terms through fNL**2.  When the Marisa reconstruction denominator follows

    b_rec(k; fNL_rec) = b_rec + fNL_rec * bphi_rec / M(k),

and fNL_rec is identified with the catalogue fNL, two additional quadratic
tree coefficients are required in addition to the established linear
Gaussian adaptive term:

* one half of the Gaussian second derivative with respect to fNL_rec;
* the cross derivative of the fixed-b_rec linear PNG reconstruction term.

This producer shell-projects all three analytic mode-level bases with the
same exact FFT-lattice/Haar rule used by the existing Marisa adaptive asset.
It does not evaluate a matter loop or introduce a response gain.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model")
CODES = ROOT / "marisa_b_core"
MARISA = CODES
BASE_RUNNER = CODES / "prepare_quijote_halo_z0p5_sto_post_v0p9_corrected_theory.py"
OLD_ADAPTIVE = (
    ROOT
    / "analysis/quijote_halo_z0p5_marisa_only_p112_adaptive_response_kmax0p10"
    / "theory/adaptive_linear_templates_fixed_b1_1p93.npz"
)
TAG = "quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1"
ANALYSIS = ROOT / "analysis" / TAG
RAW = ANALYSIS / "raw"
THEORY = ANALYSIS / "theory"
OUTPUT_NPZ = THEORY / "marisa_adaptive_finite_tree_bases.npz"
OUTPUT_JSON = THEORY / "marisa_adaptive_finite_tree_bases.json"

INDICES = np.asarray((1, 2, 3, 15, 16, 17, 29, 30, 42), dtype=np.int64)
KMAX = 0.08
B1_REFERENCE = 1.93
B_REC = 1.9326526728794324
P_REC = 1.12
DELTA_C = 1.686
BPHI_REC = 2.0 * DELTA_C * (B_REC - P_REC)


def configure_selection(selection: str) -> None:
    """Select an immutable output namespace and certified index registry."""

    global TAG, ANALYSIS, RAW, THEORY, OUTPUT_NPZ, OUTPUT_JSON
    global OLD_ADAPTIVE, INDICES, KMAX
    if selection == "kmax0p08":
        tag = "quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1"
        indices = (1, 2, 3, 15, 16, 17, 29, 30, 42)
        old_tag = "quijote_halo_z0p5_marisa_only_p112_adaptive_response_kmax0p10"
        kmax = 0.08
    elif selection == "kmax0p20":
        tag = "quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p20_v1"
        indices = (
            1, 2, 3, 4, 5, 6, 8, 9,
            15, 16, 17, 18, 19, 20, 22, 23,
            29, 30, 31, 32, 33, 35, 36,
            42, 43, 44, 45, 47, 48,
            54, 55, 56, 58, 59,
            65, 66, 68, 69,
            75, 77, 78, 92, 93, 99,
        )
        old_tag = "quijote_halo_z0p5_marisa_only_p112_adaptive_response_kmax0p20"
        kmax = 0.20
    else:
        raise ValueError(selection)
    TAG = tag
    ANALYSIS = ROOT / "analysis" / TAG
    RAW = ANALYSIS / "raw"
    THEORY = ANALYSIS / "theory"
    OUTPUT_NPZ = THEORY / "marisa_adaptive_finite_tree_bases.npz"
    OUTPUT_JSON = THEORY / "marisa_adaptive_finite_tree_bases.json"
    OLD_ADAPTIVE = (
        ROOT / "analysis" / old_tag / "theory/adaptive_linear_templates_fixed_b1_1p93.npz"
    )
    INDICES = np.asarray(indices, dtype=np.int64)
    KMAX = kmax


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_base() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_direct_finite_marisa_base", BASE_RUNNER
    )
    if specification is None or specification.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module.B_REC = B_REC
    module.BPHI_REC = BPHI_REC
    module.R_SMOOTH = 10.0
    module.KMIN = 2.0 * math.pi / 1000.0
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def run_one(base: Any, index: int, overwrite: bool) -> dict[str, Any]:
    destination = RAW / f"adaptive_finite_tree_bin_{index:03d}.jsonl"
    if overwrite or not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        environment = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MARISA_B_MAX_THREADS",
        ):
            environment[name] = "1"
        completed = subprocess.run(
            base.png_command(index, "adaptive-brec-tree", 0.0),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode != 0 or completed.stderr:
            raise RuntimeError(
                f"adaptive finite tree bin {index} failed: {completed.stderr[-4000:]}"
            )
        temporary.write_text(completed.stdout, encoding="utf-8")
        temporary.replace(destination)

    records = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 2:
        raise ValueError(f"unexpected record count for {destination}")
    header, row = records
    required = (
        "adaptive_brec_denominator_tree_basis",
        "adaptive_brec_gaussian_quadratic_basis",
        "adaptive_brec_png_linear_cross_basis",
    )
    if (
        header.get("schema")
        != "marisa-b-post-r1-adaptive-brec-diagnostic-jsonl-v1"
        or header.get("sector") != "adaptive-brec-tree"
        or int(row.get("index", -1)) != index
        or not math.isclose(
            float(header["reconstruction"]["b_rec_h"]),
            B_REC,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or any(name not in row for name in required)
        or not all(math.isfinite(float(row[name])) for name in required)
    ):
        raise ValueError(f"adaptive finite tree contract failed for {destination}")
    return {
        "index": index,
        "edges": np.asarray(row["edges"], dtype=np.float64),
        "gaussian_linear_basis": float(row[required[0]]),
        "gaussian_quadratic_basis": float(row[required[1]]),
        "png_linear_cross_basis": float(row[required[2]]),
        "path": str(destination),
        "sha256": sha256(destination),
        "header": header,
    }


def produce(workers: int, overwrite: bool) -> dict[str, Any]:
    base = import_base()
    for required in (base.INPUT_TABLE, base.EDGES, base.PNG_DRIVER, OLD_ADAPTIVE):
        if not required.is_file():
            raise FileNotFoundError(required)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, base, int(index), overwrite): int(index)
            for index in INDICES
        }
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    rows.sort(key=lambda row: row["index"])
    if not np.array_equal(
        np.asarray([row["index"] for row in rows], dtype=np.int64), INDICES
    ):
        raise ValueError("adaptive finite tree index drift")

    linear_basis = np.asarray(
        [row["gaussian_linear_basis"] for row in rows], dtype=np.float64
    )
    gaussian_quadratic_basis = np.asarray(
        [row["gaussian_quadratic_basis"] for row in rows], dtype=np.float64
    )
    png_linear_cross_basis = np.asarray(
        [row["png_linear_cross_basis"] for row in rows], dtype=np.float64
    )
    edges = np.asarray([row["edges"] for row in rows], dtype=np.float64)

    with np.load(OLD_ADAPTIVE, allow_pickle=False) as handle:
        old = {name: np.asarray(handle[name]) for name in handle.files}
    old_lookup = {
        int(index): offset
        for offset, index in enumerate(np.asarray(old["indices"], dtype=np.int64))
    }
    old_tree = np.asarray(
        [old["adaptive_tree_selected"][old_lookup[int(index)]] for index in INDICES],
        dtype=np.float64,
    )
    new_tree_reference = (
        B1_REFERENCE**4 * (BPHI_REC / B_REC) * linear_basis
    )
    linear_closure = float(
        np.linalg.norm(new_tree_reference - old_tree)
        / max(float(np.linalg.norm(old_tree)), 1.0)
    )
    if linear_closure >= 2.0e-12:
        raise ValueError(f"established adaptive linear closure failed: {linear_closure}")

    arrays = {
        "indices": INDICES,
        "edges": edges,
        "gaussian_linear_basis": linear_basis,
        "gaussian_quadratic_basis": gaussian_quadratic_basis,
        "png_linear_cross_basis": png_linear_cross_basis,
        "b_rec": np.asarray(B_REC),
        "p_rec": np.asarray(P_REC),
        "bphi_rec": np.asarray(BPHI_REC),
        "b1_reference": np.asarray(B1_REFERENCE),
        "adaptive_gaussian_linear_reference": new_tree_reference,
        "adaptive_gaussian_quadratic_reference": (
            B1_REFERENCE**4
            * (BPHI_REC / B_REC) ** 2
            * gaussian_quadratic_basis
        ),
        "adaptive_png_linear_cross_unit_bphi_reference": (
            B1_REFERENCE**3
            * (BPHI_REC / B_REC)
            * png_linear_cross_basis
        ),
    }
    atomic_npz(OUTPUT_NPZ, arrays)
    payload = {
        "schema": "quijote-halo-z0p5-marisa-direct-finite-tree-bases-v1",
        "created_utc": utc_now(),
        "status": "pass",
        "scope": {
            "redshift": 0.5,
            "kmax_h_mpc": KMAX,
            "indices": INDICES.tolist(),
            "fully_tree_dm": True,
            "finite_png_order": 2,
            "fNL_rec_equals_fNL": True,
            "matter_loop_included": False,
            "empirical_gain": False,
        },
        "normalization": {
            "adaptive_G1": "b1^4*(bphi_rec/b_rec)*gaussian_linear_basis",
            "adaptive_G2": "b1^4*(bphi_rec/b_rec)^2*gaussian_quadratic_basis",
            "adaptive_L_cross": (
                "b1^3*bphi*(bphi_rec/b_rec)*png_linear_cross_basis"
            ),
        },
        "checks": {
            "all_finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in (
                        linear_basis,
                        gaussian_quadratic_basis,
                        png_linear_cross_basis,
                    )
                )
            ),
            "established_linear_asset_relative_l2": linear_closure,
            "established_linear_asset_closes_below_2e_minus_12": (
                linear_closure < 2.0e-12
            ),
        },
        "inputs": {
            "native_header": {
                "path": str(MARISA / "src/marisa_b/marisa_b_native.h"),
                "sha256": sha256(MARISA / "src/marisa_b/marisa_b_native.h"),
            },
            "native_source": {
                "path": str(MARISA / "src/marisa_b/marisa_b_native.cpp"),
                "sha256": sha256(MARISA / "src/marisa_b/marisa_b_native.cpp"),
            },
            "driver_source": {
                "path": str(MARISA / "src/eft_v2/post_r1_png_template_driver.cpp"),
                "sha256": sha256(
                    MARISA / "src/eft_v2/post_r1_png_template_driver.cpp"
                ),
            },
            "driver_executable": {
                "path": str(base.PNG_DRIVER),
                "sha256": sha256(base.PNG_DRIVER),
            },
            "old_linear_asset": {
                "path": str(OLD_ADAPTIVE),
                "sha256": sha256(OLD_ADAPTIVE),
            },
            "raw": [
                {key: row[key] for key in ("index", "path", "sha256")}
                for row in rows
            ],
        },
        "output": {"path": str(OUTPUT_NPZ)},
    }
    atomic_json(OUTPUT_JSON, payload)
    payload["output"]["sha256"] = sha256(OUTPUT_NPZ)
    atomic_json(OUTPUT_JSON, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", choices=("kmax0p08", "kmax0p20"), default="kmax0p08"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configure_selection(args.selection)
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be between 1 and 9")
    payload = produce(args.workers, args.overwrite)
    print(json.dumps(payload["checks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
