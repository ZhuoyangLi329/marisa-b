#!/usr/bin/env python3
"""MARISA-B v0.9 matched/oracle pre/post reconstruction experiment.

The post-reconstruction catalogue is the ordinary fiducial (fNL=0)
catalogue.  Only the forward model follows the conditional matched path
fNL_rec=fNL.  Consequently this program measures an oracle performance bound,
not an operational likelihood for an unknown primordial amplitude.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_SCRIPTS = REPOSITORY_ROOT / "scripts" / "production"
if str(PRODUCTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_SCRIPTS))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import corner
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import solve_triangular

import run_prepost_recon_halo_fnl_mcmc_v1 as inference
import run_post_recon_halo_finite_png_v1 as post_png
import run_pre_recon_bias_model_v3 as pre_gaussian
import run_pre_recon_halo_png_uplift_v1 as pre_png
import run_pre_recon_halo_tree_fnl2_v1 as pre_finite
from produce_quijote_halo_marisa_b_bias_v1_gaussian_templates import (
    load_measurement_geometry,
)


TAG = "marisa_b_v0p9_matched_oracle_20260803"
CUTS = (0.08, 0.10, 0.12, 0.14)
EXPECTED_COUNTS = {0.08: 9, 0.10: 14, 0.12: 20, 0.14: 27}
NAMES = ("b1", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
NONLINEAR_NAMES = ("b1",)
FNL_BOUNDS = (-150.0, 150.0)
B1_BOUNDS = (0.1, 6.0)
B_REC_H = 2.7340475186190334
BPHI_REC = 5.84720823278338
B1_FID = 2.7465779027841823
B1_SIGMA = 0.09252764803869407
KMIN = 2.0 * math.pi / 1000.0
N_BAR = 1.95530218e-4
RESPONSE_VARIANTS = (
    ("p1", 1.0),
    ("m1", -1.0),
    ("p0p5", 0.5),
    ("m0p5", -0.5),
)
DIAGRAMS = ("tree", "B222", "B321I", "B321II", "B411")
LOOP_DIAGRAMS = DIAGRAMS[1:]
STOCHASTIC_RESPONSE_SECTORS = ("Ashot_residual", "Bshot_residual")
RESPONSE_SECTORS = DIAGRAMS + STOCHASTIC_RESPONSE_SECTORS
STATIC_RESPONSE_SECTORS = ("pre_Bshot_residual",)
RAW_RESPONSE_SECTORS = RESPONSE_SECTORS + STATIC_RESPONSE_SECTORS
ANALYTIC_UV_SECTORS = (
    "B321II_bare",
    "B411_bare",
    "B321II_uv_subtraction",
    "B411_uv_subtraction",
    "B321II_uv_restoration",
    "B411_uv_restoration",
)
ANALYTIC_RESPONSE_SECTORS = RESPONSE_SECTORS + ANALYTIC_UV_SECTORS
B1_POWERS = tuple(range(7))
B1_MONOMIALS = tuple("1" if power == 0 else f"b1^{power}" for power in B1_POWERS)
R0_CANARY_INDICES = (1, 33, 75)
FINITE_RESPONSE_CANARY_INDICES = (1, 3, 6)
MCMC_ENSEMBLES = 4
MCMC_WALKERS = 24
MCMC_STEPS = 3000
MCMC_BURN_FRACTION = 0.5

POST_LINEAR_KEYS = (
    "halo_primordial",
    "halo_bphi_f2",
    "halo_bphi_advection",
    "halo_bphidelta",
    "halo_bphi_b2",
    "halo_bphi_bK2",
    "halo_reconstruction",
    "residual_stochastic_linear",
    "matter_one_loop_uplift",
)
POST_QUADRATIC_KEYS = (
    "halo_bphi_B0",
    "halo_bphi_sq_advection",
    "halo_bphi_sq_F2",
    "halo_bphi_sq_b2",
    "halo_bphi_sq_bK2",
    "halo_bphi_sq_reconstruction",
    "halo_bphi_bphidelta",
    "halo_bphi2_operator",
    "residual_stochastic_quadratic",
    "matter_B112II",
)


@dataclass(frozen=True)
class Paths:
    root: Path
    data_root: Path
    output_root: Path
    analysis: Path
    figures: Path
    archive: Path
    raw: Path
    source: Path
    executable: Path
    analytic_executable: Path
    response: Path
    summary: Path
    report: Path
    chains: Path
    analytic_raw: Path
    analytic_manifest: Path
    finite_canary_manifest: Path
    chain_contract: Path
    git_root: Path
    power: Path
    png_table: Path
    edges: Path
    matrix: Path
    post_gaussian: Path
    post_png: Path
    tree_response: Path


def paths_for(
    root: Path,
    *,
    data_root: Path | None = None,
    output_root: Path | None = None,
) -> Paths:
    git_root = root.resolve()
    data = (data_root or root).resolve()
    output = (output_root or data).resolve()
    analysis = output / "analysis" / TAG
    return Paths(
        root=root,
        data_root=data,
        output_root=output,
        analysis=analysis,
        figures=output / "figures/diagnostics" / TAG,
        archive=output / "log" / TAG,
        raw=(
            output / "log" / TAG / "raw_response"
        ),
        source=(
            root / "src/experiments/marisa_b_v0p9_oracle_matter_driver.cpp"
        ),
        executable=root / "build/v0p9/marisa_b_v0p9_oracle_matter_driver",
        analytic_executable=(
            root / "build/v0p9/marisa_b_v0p9_oracle_matter_analytic_driver"
        ),
        response=analysis / "oracle_matter_response.npz",
        summary=analysis / "summary.json",
        report=analysis / "REPORT.md",
        chains=analysis / "chains.h5",
        analytic_raw=(
            output / "log" / TAG / "raw_response_analytic"
        ),
        analytic_manifest=(
            output / "log" / TAG / "analytic_raw_manifest.json"
        ),
        finite_canary_manifest=(
            output / "log" / TAG / "finite_canary_manifest.json"
        ),
        chain_contract=analysis / "chains_contract.json",
        git_root=git_root,
        power=(
            data
            / "analysis/theory_vectors/marisa_b_v0"
            / (
                "marisa_b_pre_gaussian_shellbin_z1_r15_diag15_kmax0p3_"
                "nrad3_nmu12_partial446_20260616_plin_z1.dat"
            )
        ),
        png_table=(
            data
            / "analysis/theory_vectors/marisa_b_v0"
            / (
                "marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_"
                "nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat"
            )
        ),
        edges=git_root / "configs/eft_v2_b000_edges.txt",
        matrix=data / inference.MATRIX_RELATIVE,
        post_gaussian=data / inference.POST_TEMPLATE_RELATIVE,
        post_png=data / inference.POST_PNG_RELATIVE,
        tree_response=(
            data
            / "analysis/post_recon_halo_local_png_brec_fisher_20260728"
            / "adaptive_brec_tree_basis.jsonl"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External v0.9 input/artifact bundle root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated-product root; defaults to the data root.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "compile",
            "compile_analytic",
            "response",
            "finite_canaries",
            "analytic_response",
            "fisher",
            "mcmc",
            "finalize",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--mcmc-workers", type=int, default=8)
    parser.add_argument("--mcmc-walkers", type=int, default=MCMC_WALKERS)
    parser.add_argument("--mcmc-steps", type=int, default=MCMC_STEPS)
    parser.add_argument("--overwrite-response", action="store_true")
    parser.add_argument("--overwrite-chains", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_files(paths: Paths) -> None:
    required = (
        paths.source,
        paths.power,
        paths.png_table,
        paths.edges,
        paths.matrix,
        paths.post_gaussian,
        paths.post_png,
        paths.tree_response,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(map(str, missing)))


def compile_driver(
    paths: Paths,
    *,
    executable: Path | None = None,
) -> dict[str, Any]:
    target = paths.executable if executable is None else executable
    source_root = paths.git_root
    build_root = source_root / "build/halo_v1"
    objects = (
        "eft_v2_parameter_registry.o",
        "eft_v2_model_config.o",
        "eft_v2_template_algebra.o",
        "eft_v2_kernel_primitives.o",
        "eft_v2_bias_operators.o",
        "eft_v2_field_kernel_provider.o",
        "eft_v2_diagram_assembler.o",
        "eft_v2_ir_safe_integrands.o",
        "eft_v2_uv_subtraction.o",
        "eft_v2_direct_evaluator.o",
        "eft_v2_fftlog_dr_oracle.o",
        "eft_v2_counterterms.o",
        "eft_v2_stochastic.o",
        "eft_v2_poisson_reconstruction.o",
        "eft_v2_ir_resummation.o",
        "eft_v2_lattice_shell_rule.o",
        "eft_v2_shell_projector.o",
        "eft_v2_tracer_power.o",
        "halo_v1.o",
        "shell_average.o",
        "marisa_b_native.o",
        "Common.o",
        "PowerSpectrum.o",
        "array.o",
        "Quadrature.o",
    )
    object_paths = tuple(build_root / name for name in objects)
    missing = [path for path in object_paths if not path.is_file()]
    if missing:
        build_environment = os.environ.copy()
        gsl_prefix = Path(build_environment.get("GSL_PREFIX", sys.prefix))
        if (gsl_prefix / "include/gsl").is_dir():
            build_environment.setdefault("GSL_PREFIX", str(gsl_prefix))
        build = subprocess.run(
            [
                "make",
                "-j1",
                *[str(path.relative_to(source_root)) for path in missing],
            ],
            cwd=source_root,
            env=build_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError("driver object build failed:\n" + build.stderr)
    target.parent.mkdir(parents=True, exist_ok=True)
    gsl_prefix = Path(os.environ.get("GSL_PREFIX", sys.prefix))
    gsl_candidates = tuple((gsl_prefix / "lib").glob("*-linux-gnu")) + (
        gsl_prefix / "lib",
    )
    gsl_root = next(
        (candidate for candidate in gsl_candidates if (candidate / "libgsl.so").is_file()),
        None,
    )
    if gsl_root is None:
        raise FileNotFoundError(
            "libgsl.so was not found; set GSL_PREFIX to the GSL installation"
        )
    command = [
        "g++",
        "-std=gnu++17",
        "-O2",
        "-g",
        "-fopenmp",
        "-ffunction-sections",
        "-fdata-sections",
        "-DHAVE_CONFIG_H",
        f"-I{source_root / 'src/halo_v1'}",
        f"-I{source_root / 'src/eft_v2'}",
        f"-I{source_root / 'src/marisa_b'}",
        f"-I{source_root / 'external/ACTio-ReACTio/reactions'}",
        f"-I{source_root / 'external/ACTio-ReACTio/reactions/src'}",
        str(paths.source),
        *map(str, object_paths),
        "-fopenmp",
        "-Wl,--gc-sections",
        f"-L{gsl_root}",
        f"-Wl,-rpath,{gsl_root}",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-o",
        str(target),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("driver compile failed:\n" + completed.stderr)
    return {
        "status": "compiled",
        "elapsed_seconds": time.monotonic() - started,
        "source": str(paths.source),
        "source_sha256": sha256(paths.source),
        "executable": str(target),
        "executable_sha256": sha256(target),
        "linked_object_hashes": {str(path): sha256(path) for path in object_paths},
    }


def selected_indices(k_pair: np.ndarray, kmax: float) -> np.ndarray:
    indices = inference.selected_indices(np.asarray(k_pair), float(kmax))
    if indices.size != EXPECTED_COUNTS[float(kmax)]:
        raise AssertionError("selected-bin contract drift")
    return indices


def raw_path(paths: Paths, variant: str, index: int) -> Path:
    return paths.raw / f"fNLrec_{variant}_bin_{index:03d}.jsonl"


def read_raw(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2 or rows[0].get("record") != "header" or rows[1].get("record") != "bin":
        raise ValueError(f"invalid raw response file {path}")
    return rows[0], rows[1]


def pure_b1_coefficients(polynomial: dict[str, Any]) -> np.ndarray:
    if not isinstance(polynomial, dict):
        raise TypeError("response sector is not a sparse polynomial")
    unexpected = set(polynomial) - set(B1_MONOMIALS)
    if unexpected:
        raise ValueError(f"non-pure-b1 monomials in response: {sorted(unexpected)}")
    result = np.asarray(
        [float(polynomial.get(monomial, 0.0)) for monomial in B1_MONOMIALS],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite pure-b1 response coefficient")
    return result


def compiled_pure_b1_coefficients(
    templates: Any,
    key: str,
    indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros((indices.size, len(B1_POWERS)), dtype=np.float64)
    compiled = templates._compiled[key]
    for column, monomial in enumerate(B1_MONOMIALS):
        values = compiled.get(monomial)
        if values is not None:
            result[:, column] = np.asarray(values, dtype=np.float64)[indices]
    return result


def sparse_mapping_pure_b1_coefficients(
    mapping: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros((indices.size, len(B1_POWERS)), dtype=np.float64)
    unexpected = set(mapping) - set(B1_MONOMIALS)
    if unexpected:
        raise ValueError(f"non-pure-b1 terms in registered sparse map: {unexpected}")
    for column, monomial in enumerate(B1_MONOMIALS):
        values = mapping.get(monomial)
        if values is not None:
            result[:, column] = np.asarray(values, dtype=np.float64)[indices]
    return result


def evaluate_b1_polynomial(coefficients: np.ndarray, b1: float) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape[-1] != len(B1_POWERS):
        raise ValueError("pure-b1 coefficient axis drift")
    powers = np.asarray([float(b1) ** power for power in B1_POWERS])
    return values @ powers


def validate_raw(
    path: Path,
    *,
    variant_value: float,
    index: int,
    expected_edges: np.ndarray,
    paths: Paths,
) -> dict[str, Any]:
    header, row = read_raw(path)
    expected_hashes = {
        "linear_power": sha256(paths.power),
        "edge_file": sha256(paths.edges),
        "png_table": sha256(paths.png_table),
        "driver_executable": sha256(paths.executable),
    }
    if (
        header.get("schema") != "marisa-b-v0p9-oracle-pure-b1-finite-v2"
        or header.get("bin_range") != [index, index + 1]
        or row.get("index") != index
        or not math.isclose(float(header.get("fnl_rec", math.nan)), variant_value, rel_tol=0.0, abs_tol=1.0e-14)
        or not math.isclose(float(header.get("b_rec_h", math.nan)), B_REC_H, rel_tol=0.0, abs_tol=1.0e-14)
        or not math.isclose(float(header.get("bphi_rec", math.nan)), BPHI_REC, rel_tol=0.0, abs_tol=1.0e-14)
        or header.get("pure_b1_polynomial") is not True
        or header.get("source_hashes") != expected_hashes
        or not np.array_equal(np.asarray(row.get("edges")), np.asarray(expected_edges).reshape(-1))
    ):
        raise ValueError(f"raw response contract failed for {path}")
    for name in RAW_RESPONSE_SECTORS:
        pure_b1_coefficients(row.get(name))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "variant": variant_value,
        "index": index,
    }


def produce_response(paths: Paths, workers: int, overwrite: bool) -> dict[str, Any]:
    if not 1 <= workers <= 28:
        raise ValueError("workers must be in [1,28]")
    if not paths.executable.is_file():
        compile_driver(paths)
    shared = inference.load_shared_inputs(
        paths.root,
        paths.data_root,
        paths.output_root,
    )
    indices = selected_indices(shared.k_pair, 0.14)
    paths.raw.mkdir(parents=True, exist_ok=True)

    # The accepted constant-b_rec template supplies the complete r=0 pure-b1
    # coefficient vectors.  Three independently integrated r=0 canaries
    # below check that the new pure-b1 provider and normalization reproduce
    # this source rather than turning the closure into a same-source identity.
    fitter_path = paths.root / inference.FITTER_RELATIVE
    post_fitter = inference.load_module("_marisa_b_v0p9_response_zero_fitter", fitter_path)
    post_data = post_fitter.DataSet.load(paths.matrix)
    zero_templates = post_png.load_partial_templates(
        post_fitter,
        post_data,
        paths.post_gaussian,
    ).tied
    zero_png_templates = post_png.load_png_cache(paths.post_png)
    zero_by_sector = {
        **{
            diagram: compiled_pure_b1_coefficients(
                zero_templates,
                f"diagram:{diagram}",
                indices,
            )
            for diagram in DIAGRAMS
        },
        "Ashot_residual": compiled_pure_b1_coefficients(
            zero_templates,
            "stochastic:Ashot_residual",
            indices,
        ),
        "Bshot_residual": sparse_mapping_pure_b1_coefficients(
            zero_png_templates.residual_gaussian["tied"],
            indices,
        ),
    }

    def execute(job: tuple[str, float, int]) -> dict[str, Any]:
        variant, value, index_value = job
        destination = raw_path(paths, variant, index_value)
        if destination.is_file() and not overwrite:
            record = validate_raw(
                destination,
                variant_value=value,
                index=index_value,
                expected_edges=shared.edges[index_value],
                paths=paths,
            )
            record["reused"] = True
            record["elapsed_seconds"] = 0.0
            return record
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        command = [
            str(paths.executable),
            str(paths.power),
            str(paths.edges),
            str(paths.png_table),
            str(index_value),
            str(index_value + 1),
            f"{value:.17g}",
        ]
        environment = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MARISA_B_MAX_THREADS",
        ):
            environment[name] = "1"
        started = time.monotonic()
        with temporary.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                check=False,
            )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"response job failed for {variant}/bin {index_value}: "
                + completed.stderr[-2000:]
            )
        if completed.stderr:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"unexpected stderr for {variant}/bin {index_value}: "
                + completed.stderr[-1000:]
            )
        temporary.replace(destination)
        record = validate_raw(
            destination,
            variant_value=value,
            index=index_value,
            expected_edges=shared.edges[index_value],
            paths=paths,
        )
        record["reused"] = False
        record["elapsed_seconds"] = elapsed
        return record

    all_records: list[dict[str, Any]] = []
    started = time.monotonic()
    jobs = [("zero", 0.0, int(index)) for index in R0_CANARY_INDICES] + [
        (variant, value, int(index))
        for index in indices
        for variant, value in RESPONSE_VARIANTS
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            all_records.append(record)
            variant, _value, index = futures[future]
            print(
                f"v0.9 oracle response jobs {count}/{len(jobs)}: "
                f"completed {variant}/bin {index}",
                flush=True,
            )

    arrays: dict[str, Any] = {
        "indices": indices.astype(np.int64),
        "b1_fid": np.asarray(B1_FID),
        "brec_h": np.asarray(B_REC_H),
        "bphi_rec": np.asarray(BPHI_REC),
        "kmin": np.asarray(KMIN),
        "b1_powers": np.asarray(B1_POWERS, dtype=np.int64),
    }
    variant_rows: dict[int, dict[str, dict[str, Any]]] = {}
    for index in indices:
        variant_rows[int(index)] = {
            variant: read_raw(raw_path(paths, variant, int(index)))[1]
            for variant, _value in RESPONSE_VARIANTS
        }
    for sector in RESPONSE_SECTORS:
        zero = np.asarray(zero_by_sector[sector], dtype=np.float64)
        coarse = np.asarray(
            [
                0.5
                * (
                    pure_b1_coefficients(variant_rows[int(index)]["p1"][sector])
                    - pure_b1_coefficients(variant_rows[int(index)]["m1"][sector])
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        fine = np.asarray(
            [
                pure_b1_coefficients(variant_rows[int(index)]["p0p5"][sector])
                - pure_b1_coefficients(variant_rows[int(index)]["m0p5"][sector])
                for index in indices
            ],
            dtype=np.float64,
        )
        richardson = (4.0 * fine - coarse) / 3.0
        second_coarse = np.asarray(
            [
                pure_b1_coefficients(variant_rows[int(index)]["p1"][sector])
                - 2.0 * zero[local_index]
                + pure_b1_coefficients(variant_rows[int(index)]["m1"][sector])
                for local_index, index in enumerate(indices)
            ],
            dtype=np.float64,
        )
        second_fine = np.asarray(
            [
                4.0
                * (
                    pure_b1_coefficients(
                        variant_rows[int(index)]["p0p5"][sector]
                    )
                    - 2.0 * zero[local_index]
                    + pure_b1_coefficients(
                        variant_rows[int(index)]["m0p5"][sector]
                    )
                )
                for local_index, index in enumerate(indices)
            ],
            dtype=np.float64,
        )
        second_richardson = (4.0 * second_fine - second_coarse) / 3.0
        arrays[f"{sector}_zero"] = zero
        arrays[f"{sector}_coarse"] = coarse
        arrays[f"{sector}_fine"] = fine
        arrays[f"{sector}_richardson"] = richardson
        arrays[f"{sector}_step_error"] = richardson - fine
        arrays[f"{sector}_second_coarse"] = second_coarse
        arrays[f"{sector}_second_fine"] = second_fine
        arrays[f"{sector}_second_richardson"] = second_richardson
        arrays[f"{sector}_second_step_error"] = second_richardson - second_fine
    pre_bshot_zero = np.asarray(
        [
            pure_b1_coefficients(
                variant_rows[int(index)]["p1"]["pre_Bshot_residual"]
            )
            for index in indices
        ],
        dtype=np.float64,
    )
    for variant, _value in RESPONSE_VARIANTS[1:]:
        candidate = np.asarray(
            [
                pure_b1_coefficients(
                    variant_rows[int(index)][variant]["pre_Bshot_residual"]
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        if not np.array_equal(candidate, pre_bshot_zero):
            raise ValueError("pre leading Bshot shape changed with fNL_rec")
    arrays["pre_Bshot_residual_zero"] = pre_bshot_zero
    arrays["loop_zero"] = sum(arrays[f"{name}_zero"] for name in LOOP_DIAGRAMS)
    arrays["loop_richardson"] = sum(
        arrays[f"{name}_richardson"] for name in LOOP_DIAGRAMS
    )
    arrays["loop_step_error"] = sum(
        arrays[f"{name}_step_error"] for name in LOOP_DIAGRAMS
    )
    arrays["loop_second_richardson"] = sum(
        arrays[f"{name}_second_richardson"] for name in LOOP_DIAGRAMS
    )
    arrays["loop_second_step_error"] = sum(
        arrays[f"{name}_second_step_error"] for name in LOOP_DIAGRAMS
    )
    paths.analysis.mkdir(parents=True, exist_ok=True)
    temporary = paths.response.with_name(f".{paths.response.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(paths.response)
    manifest = {
        "schema": "marisa-b-v0p9-oracle-matter-response-manifest-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "worker_limit": workers,
        "wall_seconds": time.monotonic() - started,
        "job_count": len(all_records),
        "new_job_count": sum(not row["reused"] for row in all_records),
        "selected_indices": indices,
        "variants": RESPONSE_VARIANTS,
        "zero_point": {
            "method": (
                "accepted constant-b_rec pure-b1 coefficient vectors, with "
                "independent r=0 integrations at three canary bins"
            ),
            "source": str(paths.post_gaussian),
            "source_sha256": sha256(paths.post_gaussian),
            "independent_canary_indices": R0_CANARY_INDICES,
            "independent_canary_jobs": [
                row
                for row in all_records
                if row["variant"] == 0.0 and row["index"] in R0_CANARY_INDICES
            ],
        },
        "thread_contract": "at most 28 child processes, one thread each",
        "production_sources": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "driver_source": {
                "path": str(paths.source),
                "sha256": sha256(paths.source),
            },
            "driver_executable": {
                "path": str(paths.executable),
                "sha256": sha256(paths.executable),
            },
        },
        "production_inputs": {
            "linear_power": {
                "path": str(paths.power),
                "sha256": sha256(paths.power),
            },
            "edge_file": {
                "path": str(paths.edges),
                "sha256": sha256(paths.edges),
            },
            "png_table": {
                "path": str(paths.png_table),
                "sha256": sha256(paths.png_table),
            },
        },
        "response": {
            "path": str(paths.response),
            "sha256": sha256(paths.response),
            "derivative_orders": [1, 2],
            "second_derivative_policy": (
                "Gaussian matched-path curvature diagnostic only; excluded "
                "from primary MCMC because dL/dfNL_rec is not supplied"
            ),
        },
        "raw_jobs": sorted(all_records, key=lambda row: (row["index"], row["variant"])),
    }
    atomic_json(paths.archive / "raw_manifest.json", manifest)
    return manifest


def required_finite_canary_jobs() -> tuple[tuple[str, float, int], ...]:
    return tuple(
        [("zero", 0.0, int(index)) for index in R0_CANARY_INDICES]
        + [
            (variant, value, int(index))
            for index in FINITE_RESPONSE_CANARY_INDICES
            for variant, value in RESPONSE_VARIANTS
        ]
    )


def produce_finite_canaries(
    paths: Paths,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Produce only the independently integrated release canaries.

    The executable is rebuilt unconditionally so the source, binary, raw
    records, and manifest form one self-contained provenance contract.
    """
    if not 1 <= workers <= 28:
        raise ValueError("workers must be in [1,28]")
    shared = inference.load_shared_inputs(
        paths.root,
        paths.data_root,
        paths.output_root,
    )
    compile_record: dict[str, Any]
    if paths.finite_canary_manifest.is_file() and not overwrite:
        existing = json.loads(
            paths.finite_canary_manifest.read_text(encoding="utf-8")
        )
        expected_sources = {
            "driver_source": sha256(paths.source),
            "driver_executable": sha256(paths.executable),
        }
        expected_inputs = {
            "linear_power": sha256(paths.power),
            "edge_file": sha256(paths.edges),
            "png_table": sha256(paths.png_table),
        }
        source_match = all(
            existing.get("production_sources", {})
            .get(name, {})
            .get("sha256")
            == digest
            for name, digest in expected_sources.items()
        )
        input_match = all(
            existing.get("production_inputs", {})
            .get(name, {})
            .get("sha256")
            == digest
            for name, digest in expected_inputs.items()
        )
        if not source_match or not input_match:
            raise RuntimeError(
                "finite-canary physics source or input changed; rerun with "
                "--overwrite-response"
            )
        compile_record = dict(existing.get("compile", {}))
    else:
        compile_record = compile_driver(paths)
    paths.raw.mkdir(parents=True, exist_ok=True)
    jobs = required_finite_canary_jobs()

    def execute(job: tuple[str, float, int]) -> dict[str, Any]:
        variant, value, index_value = job
        destination = raw_path(paths, variant, index_value)
        if destination.is_file() and not overwrite:
            record = validate_raw(
                destination,
                variant_value=value,
                index=index_value,
                expected_edges=shared.edges[index_value],
                paths=paths,
            )
            record["reused"] = True
            record["elapsed_seconds"] = 0.0
            return record
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        command = [
            str(paths.executable),
            str(paths.power),
            str(paths.edges),
            str(paths.png_table),
            str(index_value),
            str(index_value + 1),
            f"{value:.17g}",
        ]
        environment = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MARISA_B_MAX_THREADS",
        ):
            environment[name] = "1"
        started = time.monotonic()
        with temporary.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                check=False,
            )
        elapsed = time.monotonic() - started
        if completed.returncode != 0 or completed.stderr:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"finite canary failed for {variant}/bin {index_value}: "
                + completed.stderr[-2000:]
            )
        temporary.replace(destination)
        record = validate_raw(
            destination,
            variant_value=value,
            index=index_value,
            expected_edges=shared.edges[index_value],
            paths=paths,
        )
        record["reused"] = False
        record["elapsed_seconds"] = elapsed
        return record

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(jobs))
    ) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records.append(record)
            variant, _value, index = futures[future]
            print(
                f"v0.9 finite canaries {count}/{len(jobs)}: "
                f"completed {variant}/bin {index}",
                flush=True,
            )

    manifest = {
        "schema": "marisa-b-v0p9-oracle-finite-canary-manifest-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "worker_limit": min(workers, len(jobs)),
        "wall_seconds": time.monotonic() - started,
        "job_count": len(records),
        "new_job_count": sum(not row["reused"] for row in records),
        "required_jobs": jobs,
        "method": (
            "independent finite reconstruction at r=0 and symmetric "
            "r=+/-1,+/-0.5 Richardson steps"
        ),
        "compile": compile_record,
        "production_sources": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "driver_source": {
                "path": str(paths.source),
                "sha256": sha256(paths.source),
            },
            "driver_executable": {
                "path": str(paths.executable),
                "sha256": sha256(paths.executable),
            },
        },
        "production_inputs": {
            "linear_power": {
                "path": str(paths.power),
                "sha256": sha256(paths.power),
            },
            "edge_file": {
                "path": str(paths.edges),
                "sha256": sha256(paths.edges),
            },
            "png_table": {
                "path": str(paths.png_table),
                "sha256": sha256(paths.png_table),
            },
        },
        "raw_jobs": sorted(
            records,
            key=lambda row: (row["index"], row["variant"]),
        ),
    }
    atomic_json(paths.finite_canary_manifest, manifest)
    return manifest


def finite_canary_manifest_provenance(
    paths: Paths,
    shared: Any,
) -> dict[str, Any]:
    if not paths.finite_canary_manifest.is_file():
        raise FileNotFoundError(
            f"missing finite canary manifest: {paths.finite_canary_manifest}; "
            "run --stage finite_canaries"
        )
    manifest = json.loads(
        paths.finite_canary_manifest.read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema")
        != "marisa-b-v0p9-oracle-finite-canary-manifest-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("finite canary manifest is not complete")
    expected_sources = {
        "runner": sha256(Path(__file__).resolve()),
        "driver_source": sha256(paths.source),
        "driver_executable": sha256(paths.executable),
    }
    for name, expected in expected_sources.items():
        if (
            manifest.get("production_sources", {})
            .get(name, {})
            .get("sha256")
            != expected
        ):
            raise ValueError(f"finite canary production source drift for {name}")
    expected_inputs = {
        "linear_power": sha256(paths.power),
        "edge_file": sha256(paths.edges),
        "png_table": sha256(paths.png_table),
    }
    for name, expected in expected_inputs.items():
        if (
            manifest.get("production_inputs", {})
            .get(name, {})
            .get("sha256")
            != expected
        ):
            raise ValueError(f"finite canary production input drift for {name}")
    expected_jobs = required_finite_canary_jobs()
    recorded_jobs = tuple(
        (str(row[0]), float(row[1]), int(row[2]))
        for row in manifest.get("required_jobs", ())
    )
    if recorded_jobs != expected_jobs:
        raise ValueError("finite canary required-job contract drift")
    raw_records = {
        (str(row["path"]), float(row["variant"]), int(row["index"])): row
        for row in manifest.get("raw_jobs", ())
    }
    validated = []
    for variant, value, index in expected_jobs:
        path = raw_path(paths, variant, index)
        record = validate_raw(
            path,
            variant_value=value,
            index=index,
            expected_edges=shared.edges[index],
            paths=paths,
        )
        manifest_record = raw_records.get((str(path), value, index))
        if manifest_record is None or manifest_record.get("sha256") != record["sha256"]:
            raise ValueError(f"finite canary manifest/raw mismatch for {path}")
        validated.append(record)
    return {
        "path": str(paths.finite_canary_manifest),
        "sha256": sha256(paths.finite_canary_manifest),
        "schema": manifest["schema"],
        "validated_raw_jobs": validated,
    }


def analytic_raw_path(paths: Paths, index: int) -> Path:
    return paths.analytic_raw / f"analytic_bin_{index:03d}.jsonl"


def validate_analytic_raw(
    path: Path,
    *,
    index: int,
    expected_edges: np.ndarray,
    paths: Paths,
) -> dict[str, Any]:
    header, row = read_raw(path)
    expected_hashes = {
        "linear_power": sha256(paths.power),
        "edge_file": sha256(paths.edges),
        "png_table": sha256(paths.png_table),
        "driver_executable": sha256(paths.analytic_executable),
    }
    if (
        header.get("schema")
        != "marisa-b-v0p9-oracle-pure-b1-analytic-v1"
        or header.get("bin_range") != [index, index + 1]
        or row.get("index") != index
        or header.get("analytic_direction") is not True
        or header.get("gamma31_is_auxiliary_marker") is not True
        or header.get("per_kernel_common_b1_factored") is not True
        or not math.isclose(
            float(header.get("fnl_rec", math.nan)),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(header.get("b_rec_h", math.nan)),
            B_REC_H,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not math.isclose(
            float(header.get("bphi_rec", math.nan)),
            BPHI_REC,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not math.isclose(
            float(header.get("kmin", math.nan)),
            KMIN,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or header.get("pure_b1_polynomial") is not True
        or header.get("radial_stop") != 7
        or header.get("shell_orders") != [2, 8, 2, 2, 2]
        or header.get("loop_orders") != [8, 32, 24]
        or header.get("q_range") != [0.0001, 6.4]
        or not math.isclose(
            float(header.get("uv_tail_kmax", math.nan)),
            30.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or header.get("source_hashes") != expected_hashes
        or not np.array_equal(
            np.asarray(row.get("edges")),
            np.asarray(expected_edges).reshape(-1),
        )
    ):
        raise ValueError(f"analytic response contract failed for {path}")
    for sector in ANALYTIC_RESPONSE_SECTORS:
        pure_b1_coefficients(row.get(f"{sector}_zero"))
        pure_b1_coefficients(row.get(f"{sector}_direction"))
    pure_b1_coefficients(row.get("pre_Bshot_residual_zero"))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "index": index,
    }


def accepted_zero_coefficients(
    paths: Paths,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    fitter_path = paths.root / inference.FITTER_RELATIVE
    fitter = inference.load_module("_marisa_b_v0p9_analytic_zero_fitter", fitter_path)
    data = fitter.DataSet.load(paths.matrix)
    templates = post_png.load_partial_templates(
        fitter,
        data,
        paths.post_gaussian,
    ).tied
    png_templates = post_png.load_png_cache(paths.post_png)
    return {
        **{
            diagram: compiled_pure_b1_coefficients(
                templates,
                f"diagram:{diagram}",
                indices,
            )
            for diagram in DIAGRAMS
        },
        "Ashot_residual": compiled_pure_b1_coefficients(
            templates,
            "stochastic:Ashot_residual",
            indices,
        ),
        "Bshot_residual": sparse_mapping_pure_b1_coefficients(
            png_templates.residual_gaussian["tied"],
            indices,
        ),
    }


def produce_analytic_response(
    paths: Paths,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not 1 <= workers <= 28:
        raise ValueError("workers must be in [1,28]")
    if not paths.analytic_executable.is_file():
        compile_driver(paths, executable=paths.analytic_executable)
    shared = inference.load_shared_inputs(
        paths.root,
        paths.data_root,
        paths.output_root,
    )
    indices = selected_indices(shared.k_pair, 0.14)
    paths.analytic_raw.mkdir(parents=True, exist_ok=True)
    accepted_zero = accepted_zero_coefficients(paths, indices)

    def execute(index_value: int) -> dict[str, Any]:
        destination = analytic_raw_path(paths, index_value)
        if destination.is_file() and not overwrite:
            record = validate_analytic_raw(
                destination,
                index=index_value,
                expected_edges=shared.edges[index_value],
                paths=paths,
            )
            record["reused"] = True
            record["elapsed_seconds"] = 0.0
            return record
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        command = [
            str(paths.analytic_executable),
            str(paths.power),
            str(paths.edges),
            str(paths.png_table),
            str(index_value),
            str(index_value + 1),
            "analytic",
        ]
        environment = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MARISA_B_MAX_THREADS",
        ):
            environment[name] = "1"
        started = time.monotonic()
        with temporary.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                check=False,
            )
        elapsed = time.monotonic() - started
        if completed.returncode != 0 or completed.stderr:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"analytic response job failed for bin {index_value}: "
                + completed.stderr[-2000:]
            )
        temporary.replace(destination)
        record = validate_analytic_raw(
            destination,
            index=index_value,
            expected_edges=shared.edges[index_value],
            paths=paths,
        )
        record["reused"] = False
        record["elapsed_seconds"] = elapsed
        return record

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute, int(index)): int(index)
            for index in indices
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(
                f"v0.9 analytic oracle response jobs {count}/{len(indices)}: "
                f"completed bin {futures[future]}",
                flush=True,
            )

    arrays: dict[str, Any] = {
        "indices": indices.astype(np.int64),
        "b1_fid": np.asarray(B1_FID),
        "brec_h": np.asarray(B_REC_H),
        "bphi_rec": np.asarray(BPHI_REC),
        "kmin": np.asarray(KMIN),
        "b1_powers": np.asarray(B1_POWERS, dtype=np.int64),
        "response_method": np.asarray("analytic_auxiliary_marker"),
    }
    rows = {
        int(index): read_raw(analytic_raw_path(paths, int(index)))[1]
        for index in indices
    }
    maximum_zero_relative = 0.0
    zero_closure: dict[str, float] = {}
    for sector in RESPONSE_SECTORS:
        zero = np.asarray(
            [
                pure_b1_coefficients(rows[int(index)][f"{sector}_zero"])
                for index in indices
            ],
            dtype=np.float64,
        )
        direction = np.asarray(
            [
                pure_b1_coefficients(
                    rows[int(index)][f"{sector}_direction"]
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        reference = np.asarray(accepted_zero[sector], dtype=np.float64)
        relative = float(
            np.linalg.norm(zero - reference)
            / max(float(np.linalg.norm(reference)), 1.0)
        )
        maximum_zero_relative = max(maximum_zero_relative, relative)
        zero_closure[sector] = relative
        arrays[f"{sector}_zero"] = zero
        arrays[f"{sector}_direction"] = direction
    for sector in ANALYTIC_UV_SECTORS:
        arrays[f"{sector}_zero"] = np.asarray(
            [
                pure_b1_coefficients(rows[int(index)][f"{sector}_zero"])
                for index in indices
            ],
            dtype=np.float64,
        )
        arrays[f"{sector}_direction"] = np.asarray(
            [
                pure_b1_coefficients(
                    rows[int(index)][f"{sector}_direction"]
                )
                for index in indices
            ],
            dtype=np.float64,
        )
    arrays["pre_Bshot_residual_zero"] = np.asarray(
        [
            pure_b1_coefficients(rows[int(index)]["pre_Bshot_residual_zero"])
            for index in indices
        ],
        dtype=np.float64,
    )
    arrays["loop_zero"] = sum(
        arrays[f"{name}_zero"] for name in LOOP_DIAGRAMS
    )
    arrays["loop_direction"] = sum(
        arrays[f"{name}_direction"] for name in LOOP_DIAGRAMS
    )
    if maximum_zero_relative >= 1.0e-8:
        raise ValueError(
            "analytic marker zero point does not reproduce accepted templates"
        )
    paths.analysis.mkdir(parents=True, exist_ok=True)
    temporary = paths.response.with_name(
        f".{paths.response.name}.tmp-{os.getpid()}.npz"
    )
    np.savez_compressed(temporary, **arrays)
    temporary.replace(paths.response)
    manifest = {
        "schema": "marisa-b-v0p9-oracle-analytic-response-manifest-v1",
        "created_utc": utc_now(),
        "status": "complete",
        "worker_limit": workers,
        "wall_seconds": time.monotonic() - started,
        "job_count": len(records),
        "new_job_count": sum(not row["reused"] for row in records),
        "selected_indices": indices,
        "method": (
            "analytic auxiliary-marker product rule after exact per-kernel "
            "common-b1 factorization"
        ),
        "zero_closure_relative_l2": zero_closure,
        "maximum_zero_closure_relative_l2": maximum_zero_relative,
        "thread_contract": "at most 28 child processes, one thread each",
        "production_sources": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "driver_source": {
                "path": str(paths.source),
                "sha256": sha256(paths.source),
            },
            "driver_executable": {
                "path": str(paths.analytic_executable),
                "sha256": sha256(paths.analytic_executable),
            },
        },
        "production_inputs": {
            "linear_power": {
                "path": str(paths.power),
                "sha256": sha256(paths.power),
            },
            "edge_file": {
                "path": str(paths.edges),
                "sha256": sha256(paths.edges),
            },
            "png_table": {
                "path": str(paths.png_table),
                "sha256": sha256(paths.png_table),
            },
        },
        "response": {
            "path": str(paths.response),
            "sha256": sha256(paths.response),
            "derivative_order": 1,
            "finite_difference_canary_required": True,
        },
        "raw_jobs": sorted(records, key=lambda row: row["index"]),
    }
    atomic_json(paths.analytic_manifest, manifest)
    return manifest


def load_tree_response(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header = rows[0]
    vector = np.zeros(120, dtype=np.float64)
    indices = []
    for row in rows[1:]:
        index = int(row["index"])
        vector[index] = float(row["adaptive_brec_denominator_tree_basis"])
        indices.append(index)
    if indices != selected_indices_from_contract():
        raise ValueError("adaptive tree response bin ordering drift")
    return vector, {"path": str(path), "sha256": sha256(path), "header": header}


def selected_indices_from_contract() -> list[int]:
    return [
        1, 2, 3, 4, 5, 6,
        15, 16, 17, 18, 19, 20,
        29, 30, 31, 32, 33,
        42, 43, 44, 45,
        54, 55, 56,
        65, 66,
        75,
    ]


def load_matter_response(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        result = {name: np.asarray(values[name]) for name in values.files}
    required = {
        "indices",
        "b1_fid",
        "brec_h",
        "bphi_rec",
        "kmin",
        "b1_powers",
        "response_method",
        "pre_Bshot_residual_zero",
        "loop_zero",
        "loop_direction",
    }
    required.update(
        f"{sector}_{suffix}"
        for sector in ANALYTIC_RESPONSE_SECTORS
        for suffix in ("zero", "direction")
    )
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"matter response cache misses required keys: {missing}")
    if result["indices"].tolist() != selected_indices_from_contract():
        raise ValueError("matter response index contract drift")
    if result["b1_powers"].tolist() != list(B1_POWERS):
        raise ValueError("matter response pure-b1 polynomial contract drift")
    if str(result["response_method"].item()) != "analytic_auxiliary_marker":
        raise ValueError("matter response method contract drift")
    expected_shape = (len(selected_indices_from_contract()), len(B1_POWERS))
    coefficient_keys = [
        name
        for name in required
        if name.endswith("_zero") or name.endswith("_direction")
    ]
    for name in coefficient_keys:
        if result[name].shape != expected_shape or not np.all(
            np.isfinite(result[name])
        ):
            raise ValueError(f"invalid matter response coefficient array: {name}")
    return result


def prepend_b1_prior(base: Any) -> Any:
    if tuple(base.names) != NAMES[1:]:
        raise ValueError("base prior order drift")
    mean = np.concatenate((np.asarray([B1_FID]), np.asarray(base.mean)))
    covariance = np.zeros((len(NAMES), len(NAMES)), dtype=np.float64)
    covariance[0, 0] = B1_SIGMA**2
    covariance[1:, 1:] = np.asarray(base.covariance)
    whitener = solve_triangular(
        np.linalg.cholesky(covariance),
        np.eye(len(NAMES)),
        lower=True,
    )
    return pre_gaussian.PriorBlock(
        names=NAMES,
        mean=mean,
        covariance=covariance,
        sigma=np.sqrt(np.diag(covariance)),
        whitener=whitener,
    )


def v0p9_prior(module: Any, frozen_prior: Any) -> Any:
    base = pre_gaussian.prior_block(
        module,
        frozen_prior,
        NAMES[1:],
        nuisance_scale=1.0,
        nuisance_names=NAMES[1:],
    )
    return prepend_b1_prior(base)


@dataclass(frozen=True)
class TheoryAssets:
    shared: Any
    prior: Any
    pre_frozen: Any
    pre_templates: Any
    pre_tree: dict[str, np.ndarray]
    pre_dm_tree: np.ndarray
    pre_dm_total: np.ndarray
    pre_dm_quadratic: np.ndarray
    pre_nbar: float
    post_fitter: Any
    post_templates: Any
    post_png_templates: Any
    post_nbar: float
    adaptive_tree_basis: np.ndarray
    adaptive_matter: dict[str, np.ndarray]
    provenance: dict[str, Any]


def response_manifest_provenance(paths: Paths) -> dict[str, Any]:
    manifest_path = paths.analytic_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema")
        != "marisa-b-v0p9-oracle-analytic-response-manifest-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("adaptive matter response manifest is not complete")
    expected_sources = {
        "runner": sha256(Path(__file__).resolve()),
        "driver_source": sha256(paths.source),
        "driver_executable": sha256(paths.analytic_executable),
    }
    recorded_sources = manifest.get("production_sources", {})
    for name, expected in expected_sources.items():
        if recorded_sources.get(name, {}).get("sha256") != expected:
            raise ValueError(
                f"adaptive response production source drift for {name}; "
                "rerun --stage analytic_response (completed raw jobs will be reused)"
            )
    expected_inputs = {
        "linear_power": sha256(paths.power),
        "edge_file": sha256(paths.edges),
        "png_table": sha256(paths.png_table),
    }
    recorded_inputs = manifest.get("production_inputs", {})
    for name, expected in expected_inputs.items():
        if recorded_inputs.get(name, {}).get("sha256") != expected:
            raise ValueError(f"adaptive response production input drift for {name}")
    if manifest.get("response", {}).get("sha256") != sha256(paths.response):
        raise ValueError("adaptive response cache does not match its manifest")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "schema": manifest["schema"],
    }


def load_assets(paths: Paths) -> TheoryAssets:
    shared = inference.load_shared_inputs(
        paths.root,
        paths.data_root,
        paths.output_root,
    )
    pre_paths = pre_finite.paths_for(
        paths.root,
        data_root=paths.data_root,
        output_root=paths.output_root,
    )
    pre_png.require_inputs(pre_paths, include_theory=True)
    geometry = load_measurement_geometry(pre_paths.fid_matrix)
    pre_frozen, _data, frozen_prior, pre_templates, _power, pre_nbar = (
        pre_png.load_frozen(pre_paths)
    )
    pre_tree = pre_png.load_tree(pre_paths, geometry, B1_FID)
    pre_dm_tree, _ = pre_png.load_dm_vector(
        pre_paths.dm_linear,
        geometry,
        component="B000_dBdfNL_local_tree",
    )
    pre_dm_total, _ = pre_png.load_dm_vector(
        pre_paths.dm_linear,
        geometry,
        component="B000_dBdfNL_primary",
    )
    pre_dm_quadratic, _ = pre_png.load_dm_vector(
        pre_paths.dm_quadratic,
        geometry,
        component="B000_B112II_fNL2_coefficient",
    )
    prior = v0p9_prior(pre_frozen, frozen_prior)

    fitter_path = paths.root / inference.FITTER_RELATIVE
    contract_path = paths.root / inference.CONTRACT_EFT_RELATIVE
    post_fitter = inference.load_module("_marisa_b_v0p9_post_fitter", fitter_path)
    post_data = post_fitter.DataSet.load(paths.matrix)
    _contract, post_frozen_prior, post_nbar = post_fitter.load_prior(contract_path)
    post_prior = v0p9_prior(post_fitter, post_frozen_prior)
    if not (
        np.array_equal(prior.mean, post_prior.mean)
        and np.array_equal(prior.covariance, post_prior.covariance)
    ):
        raise ValueError("pre/post v0.9 priors differ")
    post_templates = post_png.load_partial_templates(
        post_fitter,
        post_data,
        paths.post_gaussian,
    )
    post_png_templates = post_png.load_png_cache(paths.post_png)
    adaptive_tree_basis, tree_provenance = load_tree_response(paths.tree_response)
    adaptive_matter = load_matter_response(paths.response)
    response_manifest = response_manifest_provenance(paths)
    if not math.isclose(float(np.asarray(adaptive_matter["brec_h"])), B_REC_H):
        raise ValueError("adaptive matter response used a different b_rec")
    if not math.isclose(float(np.asarray(adaptive_matter["bphi_rec"])), BPHI_REC):
        raise ValueError("adaptive matter response used a different bphi_rec")
    return TheoryAssets(
        shared=shared,
        prior=prior,
        pre_frozen=pre_frozen,
        pre_templates=pre_templates,
        pre_tree=pre_tree,
        pre_dm_tree=pre_dm_tree,
        pre_dm_total=pre_dm_total,
        pre_dm_quadratic=pre_dm_quadratic,
        pre_nbar=float(pre_nbar),
        post_fitter=post_fitter,
        post_templates=post_templates,
        post_png_templates=post_png_templates,
        post_nbar=float(post_nbar),
        adaptive_tree_basis=adaptive_tree_basis,
        adaptive_matter=adaptive_matter,
        provenance={
            "matrix": {"path": str(paths.matrix), "sha256": sha256(paths.matrix)},
            "post_gaussian": {
                "path": str(paths.post_gaussian),
                "sha256": sha256(paths.post_gaussian),
            },
            "post_png": {"path": str(paths.post_png), "sha256": sha256(paths.post_png)},
            "adaptive_tree": tree_provenance,
            "adaptive_matter": {
                "path": str(paths.response),
                "sha256": sha256(paths.response),
            },
            "adaptive_matter_manifest": response_manifest,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "driver_source": {
                "path": str(paths.source),
                "sha256": sha256(paths.source),
            },
            "pre_gaussian": {
                "path": str(pre_paths.b_templates),
                "sha256": sha256(pre_paths.b_templates),
            },
            "pre_tree_png": {
                "path": str(pre_paths.tree_templates),
                "sha256": sha256(pre_paths.tree_templates),
            },
            "pre_dm_linear": {
                "path": str(pre_paths.dm_linear),
                "sha256": sha256(pre_paths.dm_linear),
            },
            "pre_dm_quadratic": {
                "path": str(pre_paths.dm_quadratic),
                "sha256": sha256(pre_paths.dm_quadratic),
            },
        },
    )


def sum_vectors(components: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    return sum(
        (np.asarray(components[name], dtype=np.float64) for name in names),
        np.zeros(120, dtype=np.float64),
    )


def post_v0p9_png_components(
    templates: Any,
    parameters: dict[str, float],
    nbar: float,
) -> dict[str, np.ndarray]:
    complete = templates.components(
        parameters,
        nbar,
        "tied",
        "legacy_accepted_pre",
    )
    linear = sum_vectors(complete, POST_LINEAR_KEYS)
    quadratic = sum_vectors(complete, POST_QUADRATIC_KEYS)
    return {
        **{name: np.asarray(complete[name]) for name in POST_LINEAR_KEYS},
        **{name: np.asarray(complete[name]) for name in POST_QUADRATIC_KEYS},
        "linear_total": linear,
        "quadratic_total": quadratic,
    }


def post_v0p9_gaussian_components(
    templates: Any,
    png_templates: Any,
    parameters: dict[str, float],
    nbar: float,
) -> dict[str, np.ndarray]:
    """Halo tree, pure-matter loop uplift, and leading stochastic shapes.

    With every nonlinear tracer bias set to zero, the post-reconstruction
    pure-matter diagrams contain powers of b1/B_REC_H generated by the shift
    field.  Keeping only the literal ``b1^3`` polynomial coefficient would
    silently discard those reconstructed-matter pieces.  Evaluating each loop
    diagram on the pure-b1 subspace implements

        b1^3 B_m,loop(lambda=b1/B_REC_H)

    while retaining the full halo tree at the fitted b2 and gamma2.  The
    Bshot direction is the registered leading marked-field map from the PNG
    cache, not the same-monomial fragment of the full mixed one-loop
    stochastic sector.
    """

    cache: dict[str, float] = {"1": 1.0}
    tree = templates._evaluate_shape("diagram:tree", parameters, cache)
    pure = {name: 0.0 for name in parameters}
    pure["b1"] = float(parameters["b1"])
    pure_cache: dict[str, float] = {"1": 1.0}
    loop_parts = {
        diagram: templates._evaluate_shape(
            f"diagram:{diagram}",
            pure,
            pure_cache,
        )
        for diagram in LOOP_DIAGRAMS
    }
    loop = sum(loop_parts.values(), np.zeros(120, dtype=np.float64))
    a_shape = templates._compiled["stochastic:Ashot_residual"].get("1")
    b_shape = png_templates.residual_gaussian["tied"].get("b1^2")
    if a_shape is None or b_shape is None:
        raise KeyError("post v0.9 leading stochastic shapes are missing")
    stochastic = (
        float(parameters["Ashot_residual"]) * a_shape / nbar**2
        + float(parameters["Bshot_residual"])
        * float(parameters["b1"]) ** 2
        * b_shape
        / nbar
    )
    return {
        "tree": tree,
        **loop_parts,
        "loop": loop,
        "stochastic": stochastic,
        "total": tree + loop + stochastic,
    }


def pre_v0p9_gaussian_components(
    frozen: Any,
    templates: Any,
    parameters: dict[str, float],
    nbar: float,
    leading_bshot_coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    base = pre_png.gaussian_components(
        frozen,
        templates,
        parameters,
        nbar,
        "uplift",
    )
    a_shape = templates._compiled["stochastic:Ashot_residual"].get("1")
    if a_shape is None:
        raise KeyError("pre v0.9 leading Ashot shape is missing")
    leading_bshot = evaluate_b1_polynomial(
        leading_bshot_coefficients,
        float(parameters["b1"]),
    )
    stochastic = (
        float(parameters["Ashot_residual"]) * a_shape / nbar**2
        + float(parameters["Bshot_residual"]) * leading_bshot / nbar
    )
    return {
        **base,
        "stochastic": stochastic,
        "total": np.asarray(base["tree"]) + np.asarray(base["loop"]) + stochastic,
    }


def pre_v0p9_response_components(
    *,
    frozen: Any,
    tree: dict[str, np.ndarray],
    templates: Any,
    parameters: dict[str, float],
    nbar: float,
    dm_tree: np.ndarray,
    dm_total: np.ndarray,
    dm_quadratic: np.ndarray,
    leading_bshot_coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    response = pre_png.response_components(
        tree=tree,
        templates=templates,
        parameters=parameters,
        b1=float(parameters["b1"]),
        p=1.0,
        nbar=nbar,
        dm_tree=dm_tree,
        dm_total=dm_total,
        dm_quadratic=dm_quadratic,
        response_model="finite_halo_tree",
    )
    b1 = float(parameters["b1"])
    historical = (
        b1**2
        * np.asarray(
            templates._compiled["stochastic:Bshot_residual"]["b1^2"],
            dtype=np.float64,
        )
    )
    leading = evaluate_b1_polynomial(leading_bshot_coefficients, b1)
    supported = np.any(np.asarray(leading_bshot_coefficients) != 0.0, axis=1)
    if np.any(historical[supported] == 0.0):
        raise ZeroDivisionError("zero historical pre Bshot projection adapter")
    ratio = np.ones(120, dtype=np.float64)
    ratio[supported] = leading[supported] / historical[supported]
    result = {name: np.asarray(value).copy() for name, value in response.items()}
    for name in (
        "stochastic_unit_Bshot",
        "stochastic",
        "halo_quadratic_stochastic_unit_Bshot",
        "halo_quadratic_stochastic",
    ):
        result[name] *= ratio
    result["linear_total"] = (
        result["deterministic_tree"]
        + result["stochastic"]
        + result["matter_linear_uplift"]
    )
    result["quadratic_total"] = (
        result["matter_quadratic"]
        + result["halo_quadratic_deterministic"]
        + result["halo_quadratic_stochastic"]
    )
    result["leading_stochastic_projection_ratio"] = ratio
    return result


def full_parameter_dict(module: Any, nuisance: np.ndarray) -> dict[str, float]:
    return pre_gaussian.full_parameter_dict(
        module,
        NAMES,
        np.asarray(nuisance, dtype=np.float64),
        tier="full",
        fixed_b1=None,
    )


def build_context(
    assets: TheoryAssets,
    reconstruction: str,
    kmax: float,
    *,
    oracle: bool,
) -> inference.CollapsedContext:
    if reconstruction not in {"pre", "post"}:
        raise ValueError(reconstruction)
    if reconstruction == "pre" and oracle:
        raise ValueError("pre reconstruction has no oracle denominator")
    indices = selected_indices(assets.shared.k_pair, kmax)
    source = assets.shared.pre_samples if reconstruction == "pre" else assets.shared.post_samples
    samples = np.asarray(source[:, indices], dtype=np.float64)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    target = np.mean(samples, axis=0)
    matter_lookup = {
        int(index): location
        for location, index in enumerate(np.asarray(assets.adaptive_matter["indices"], dtype=int))
    }
    pre_leading_bshot_coefficients = np.zeros(
        (120, len(B1_POWERS)),
        dtype=np.float64,
    )
    for global_index, local_index in matter_lookup.items():
        pre_leading_bshot_coefficients[global_index] = np.asarray(
            assets.adaptive_matter["pre_Bshot_residual_zero"][local_index],
            dtype=np.float64,
        )

    if reconstruction == "pre":
        def evaluate(
            fnl: float,
            nuisance: np.ndarray,
        ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
            parameters = pre_png.parameter_dict(
                assets.pre_frozen,
                NAMES,
                nuisance,
                tier="uplift",
                b1=B1_FID,
            )
            gaussian = pre_v0p9_gaussian_components(
                assets.pre_frozen,
                assets.pre_templates,
                parameters,
                assets.pre_nbar,
                pre_leading_bshot_coefficients,
            )
            response = pre_v0p9_response_components(
                frozen=assets.pre_frozen,
                tree=assets.pre_tree,
                templates=assets.pre_templates,
                parameters=parameters,
                nbar=assets.pre_nbar,
                dm_tree=assets.pre_dm_tree,
                dm_total=assets.pre_dm_total,
                dm_quadratic=assets.pre_dm_quadratic,
                leading_bshot_coefficients=pre_leading_bshot_coefficients,
            )
            prediction = (
                np.asarray(gaussian["total"])
                + float(fnl) * np.asarray(response["linear_total"])
                + float(fnl) ** 2 * np.asarray(response["quadratic_total"])
            )
            return prediction[indices], parameters, {**response, "gaussian_total": gaussian["total"]}
    else:
        def evaluate(
            fnl: float,
            nuisance: np.ndarray,
        ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
            parameters = full_parameter_dict(assets.post_fitter, nuisance)
            gaussian = post_v0p9_gaussian_components(
                assets.post_templates.tied,
                assets.post_png_templates,
                parameters,
                assets.post_nbar,
            )
            response = post_v0p9_png_components(
                assets.post_png_templates,
                parameters,
                assets.post_nbar,
            )
            adaptive_tree = np.zeros(120, dtype=np.float64)
            adaptive_loop = np.zeros(120, dtype=np.float64)
            # For the deliberately retained leading estimator-filtered
            # stochastic basis, both Ashot (constant) and the marked-field
            # Bshot generating shape are independent of the reconstruction
            # denominator.  Their oracle derivative is analytically zero.
            # The finite-difference sectors in the response cache are kept
            # solely as a null test and must never inject numerical residue
            # into the scientific model.
            adaptive_stochastic = np.zeros(120, dtype=np.float64)
            if oracle:
                b1 = float(parameters["b1"])
                for global_index, local_index in matter_lookup.items():
                    adaptive_tree[global_index] = float(
                        evaluate_b1_polynomial(
                            assets.adaptive_matter["tree_direction"][local_index],
                            b1,
                        )
                    )
                    adaptive_loop[global_index] = float(
                        evaluate_b1_polynomial(
                            assets.adaptive_matter["loop_direction"][local_index],
                            b1,
                        )
                    )
            matched_linear = (
                np.asarray(response["linear_total"])
                + adaptive_tree
                + adaptive_loop
                + adaptive_stochastic
            )
            prediction = (
                np.asarray(gaussian["total"])
                + float(fnl) * matched_linear
                + float(fnl) ** 2 * np.asarray(response["quadratic_total"])
            )
            components = {
                **response,
                "fixed_brec_linear_total": response["linear_total"],
                "adaptive_tree": adaptive_tree,
                "adaptive_matter_loop": adaptive_loop,
                "adaptive_stochastic": adaptive_stochastic,
                "linear_total": matched_linear,
                "gaussian_total": gaussian["total"],
            }
            return prediction[indices], parameters, components

    return inference.CollapsedContext(
        reconstruction=reconstruction,
        tier=("v0p9_oracle" if oracle else "v0p9_fixed"),
        kmax=float(kmax),
        names=NAMES,
        nonlinear_names=NONLINEAR_NAMES,
        prior=assets.prior,
        indices=indices,
        target=target,
        covariance_single=covariance,
        covariance_mock_count=samples.shape[0],
        k_pair=assets.shared.k_pair,
        edges=assets.shared.edges,
        realization_ids=assets.shared.realization_ids,
        samples=samples,
        evaluate_model=evaluate,
        input_hashes={name: str(value.get("sha256", "")) for name, value in assets.provenance.items()},
        fnl_bounds=FNL_BOUNDS,
    )


def whitened_norm(vector: np.ndarray, covariance: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            solve_triangular(
                np.linalg.cholesky(covariance),
                np.asarray(vector, dtype=np.float64),
                lower=True,
            )
        )
    )


def theory_audit(paths: Paths, assets: TheoryAssets) -> dict[str, Any]:
    finite_canary_provenance = finite_canary_manifest_provenance(
        paths,
        assets.shared,
    )
    indices = selected_indices(assets.shared.k_pair, 0.14)
    selected_samples = assets.shared.post_samples[:, indices]
    covariance = np.cov(selected_samples, rowvar=False, ddof=1)
    response_indices = np.asarray(assets.adaptive_matter["indices"], dtype=int)
    if not np.array_equal(indices, response_indices):
        raise ValueError("response and likelihood bins differ")

    parameters = full_parameter_dict(
        assets.post_fitter,
        np.asarray([B1_FID, 0.0, 0.0, 0.0, 0.0]),
    )
    cache: dict[str, float] = {"1": 1.0}
    r0_closure: dict[str, Any] = {}
    maximum_r0_relative = 0.0
    for sector in RESPONSE_SECTORS:
        if sector == "Bshot_residual":
            full_vector = assets.post_png_templates._evaluate_sparse(
                assets.post_png_templates.residual_gaussian["tied"],
                parameters,
            )
        else:
            key = (
                f"diagram:{sector}"
                if sector in DIAGRAMS
                else f"stochastic:{sector}"
            )
            full_vector = assets.post_templates.tied._evaluate_shape(
                key,
                parameters,
                cache,
            )
        polynomial_vector = evaluate_b1_polynomial(
            assets.adaptive_matter[f"{sector}_zero"],
            B1_FID,
        )
        reference = np.asarray(full_vector, dtype=np.float64)[indices]
        delta = polynomial_vector - reference
        scale = max(float(np.linalg.norm(reference)), 1.0)
        relative = float(np.linalg.norm(delta) / scale)
        maximum_r0_relative = max(maximum_r0_relative, relative)
        r0_closure[sector] = {
            "relative_l2": relative,
            "difference_sigma_single": whitened_norm(delta, covariance),
        }

    response_lookup = {
        int(index): location for location, index in enumerate(response_indices)
    }
    canary_locations = np.asarray(
        [response_lookup[index] for index in R0_CANARY_INDICES],
        dtype=int,
    )
    canary_covariance = covariance[np.ix_(canary_locations, canary_locations)]
    independent_r0_canary: dict[str, Any] = {}
    maximum_canary_relative = 0.0
    # The current finite-canary executable projects Ashot with the same exact
    # Haar shell weights, so both retained leading stochastic shapes are valid
    # independent zero-point canaries alongside the deterministic diagrams.
    for sector in DIAGRAMS + STOCHASTIC_RESPONSE_SECTORS:
        value_deltas = []
        coefficient_delta_norm = 0.0
        coefficient_reference_norm = 0.0
        for global_index, local_index in zip(R0_CANARY_INDICES, canary_locations):
            _header, row = read_raw(raw_path(paths, "zero", global_index))
            measured = pure_b1_coefficients(row[sector])
            reference = np.asarray(
                assets.adaptive_matter[f"{sector}_zero"][local_index],
                dtype=np.float64,
            )
            delta = measured - reference
            coefficient_delta_norm += float(delta @ delta)
            coefficient_reference_norm += float(reference @ reference)
            value_deltas.append(float(evaluate_b1_polynomial(delta, B1_FID)))
        relative = math.sqrt(coefficient_delta_norm) / max(
            math.sqrt(coefficient_reference_norm),
            1.0,
        )
        maximum_canary_relative = max(maximum_canary_relative, relative)
        independent_r0_canary[sector] = {
            "coefficient_relative_l2": relative,
            "evaluated_difference_sigma_single": whitened_norm(
                np.asarray(value_deltas),
                canary_covariance,
            ),
        }

    tree_coefficients = np.asarray(
        assets.adaptive_matter["tree_direction"],
        dtype=np.float64,
    )
    tree_expected_coefficients = np.zeros_like(tree_coefficients)
    tree_expected_coefficients[:, 4] = (
        BPHI_REC / B_REC_H * assets.adaptive_tree_basis[indices]
    )
    tree_coefficient_delta = tree_coefficients - tree_expected_coefficients
    tree_coefficient_relative = float(
        np.linalg.norm(tree_coefficient_delta)
        / max(float(np.linalg.norm(tree_expected_coefficients)), 1.0)
    )
    tree_multi_b1 = {}
    maximum_tree_value_relative = 0.0
    for trial_b1 in (1.0, B1_FID, 4.0):
        measured = evaluate_b1_polynomial(tree_coefficients, trial_b1)
        expected = (
            trial_b1**4
            * BPHI_REC
            / B_REC_H
            * assets.adaptive_tree_basis[indices]
        )
        delta = measured - expected
        relative = float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(expected)), 1.0)
        )
        maximum_tree_value_relative = max(maximum_tree_value_relative, relative)
        tree_multi_b1[f"{trial_b1:.12g}"] = {
            "relative_l2": relative,
            "difference_sigma_single": whitened_norm(delta, covariance),
        }
    tree_from_matter = evaluate_b1_polynomial(tree_coefficients, B1_FID)
    tree_registered = (
        B1_FID**4
        * BPHI_REC
        / B_REC_H
        * assets.adaptive_tree_basis[indices]
    )
    tree_delta = tree_from_matter - tree_registered
    tree_closure = {
        "relative_l2": max(tree_coefficient_relative, maximum_tree_value_relative),
        "coefficient_relative_l2": tree_coefficient_relative,
        "expected_nonzero_power": 4,
        "maximum_multi_b1_relative_l2": maximum_tree_value_relative,
        "multi_b1": tree_multi_b1,
        "difference_sigma_single": whitened_norm(tree_delta, covariance),
    }
    loop_response = evaluate_b1_polynomial(
        assets.adaptive_matter["loop_direction"],
        B1_FID,
    )
    loop_response_norm = whitened_norm(loop_response, covariance)

    finite_indices = np.asarray(FINITE_RESPONSE_CANARY_INDICES, dtype=int)
    finite_locations = np.asarray(
        [response_lookup[int(index)] for index in finite_indices],
        dtype=int,
    )
    finite_covariance = covariance[np.ix_(finite_locations, finite_locations)]
    finite_rows = {
        int(index): {
            variant: read_raw(raw_path(paths, variant, int(index)))[1]
            for variant, _value in RESPONSE_VARIANTS
        }
        for index in finite_indices
    }
    finite_difference_canary: dict[str, Any] = {}
    maximum_analytic_minus_richardson_sigma = 0.0
    maximum_richardson_step_sigma = 0.0
    finite_coefficients: dict[str, dict[str, np.ndarray]] = {}
    for sector in RESPONSE_SECTORS:
        coarse = np.asarray(
            [
                0.5
                * (
                    pure_b1_coefficients(finite_rows[int(index)]["p1"][sector])
                    - pure_b1_coefficients(finite_rows[int(index)]["m1"][sector])
                )
                for index in finite_indices
            ],
            dtype=np.float64,
        )
        fine = np.asarray(
            [
                pure_b1_coefficients(
                    finite_rows[int(index)]["p0p5"][sector]
                )
                - pure_b1_coefficients(
                    finite_rows[int(index)]["m0p5"][sector]
                )
                for index in finite_indices
            ],
            dtype=np.float64,
        )
        richardson = (4.0 * fine - coarse) / 3.0
        analytic = np.asarray(
            assets.adaptive_matter[f"{sector}_direction"][finite_locations],
            dtype=np.float64,
        )
        finite_coefficients[sector] = {
            "coarse": coarse,
            "fine": fine,
            "richardson": richardson,
            "analytic": analytic,
        }
        multi_b1 = {}
        sector_maximum_difference = 0.0
        sector_maximum_step = 0.0
        for trial_b1 in (1.0, B1_FID, 4.0):
            analytic_minus_richardson = evaluate_b1_polynomial(
                analytic - richardson,
                trial_b1,
            )
            richardson_minus_fine = evaluate_b1_polynomial(
                richardson - fine,
                trial_b1,
            )
            difference_sigma = whitened_norm(
                analytic_minus_richardson,
                finite_covariance,
            )
            step_sigma = whitened_norm(
                richardson_minus_fine,
                finite_covariance,
            )
            sector_maximum_difference = max(
                sector_maximum_difference,
                difference_sigma,
            )
            sector_maximum_step = max(sector_maximum_step, step_sigma)
            multi_b1[f"{trial_b1:.12g}"] = {
                "analytic_minus_richardson_sigma_single": difference_sigma,
                "richardson_minus_fine_sigma_single": step_sigma,
            }
        if sector in DIAGRAMS:
            maximum_analytic_minus_richardson_sigma = max(
                maximum_analytic_minus_richardson_sigma,
                sector_maximum_difference,
            )
            maximum_richardson_step_sigma = max(
                maximum_richardson_step_sigma,
                sector_maximum_step,
            )
        finite_difference_canary[sector] = {
            "indices": finite_indices,
            "multi_b1": multi_b1,
            "maximum_analytic_minus_richardson_sigma_single": (
                sector_maximum_difference
            ),
            "maximum_richardson_minus_fine_sigma_single": sector_maximum_step,
        }

    uv_identities: dict[str, Any] = {}
    maximum_uv_identity_relative = 0.0
    for diagram in ("B321II", "B411"):
        for suffix in ("zero", "direction"):
            renormalized = np.asarray(
                assets.adaptive_matter[f"{diagram}_{suffix}"],
                dtype=np.float64,
            )
            expected = (
                np.asarray(
                    assets.adaptive_matter[f"{diagram}_bare_{suffix}"],
                    dtype=np.float64,
                )
                - np.asarray(
                    assets.adaptive_matter[
                        f"{diagram}_uv_subtraction_{suffix}"
                    ],
                    dtype=np.float64,
                )
                + np.asarray(
                    assets.adaptive_matter[
                        f"{diagram}_uv_restoration_{suffix}"
                    ],
                    dtype=np.float64,
                )
            )
            relative = float(
                np.linalg.norm(renormalized - expected)
                / max(float(np.linalg.norm(expected)), 1.0)
            )
            maximum_uv_identity_relative = max(
                maximum_uv_identity_relative,
                relative,
            )
            uv_identities[f"{diagram}_{suffix}"] = {
                "relative_l2": relative,
                "maximum_absolute": float(
                    np.max(np.abs(renormalized - expected))
                ),
            }

    stochastic_zero_response = {}
    for sector, normalization_power in (
        ("Ashot_residual", 2),
        ("Bshot_residual", 1),
    ):
        prior_index = assets.prior.names.index(sector)
        scale = float(assets.prior.sigma[prior_index]) / assets.post_nbar**normalization_power
        zero_coefficients = np.asarray(
            assets.adaptive_matter[f"{sector}_zero"],
            dtype=np.float64,
        )
        coefficient_scale = max(float(np.max(np.abs(zero_coefficients))), 1.0)
        coefficient_tolerance = 1.0e-12 * coefficient_scale
        maximum_direction = float(
            np.max(
                np.abs(
                    np.asarray(
                        assets.adaptive_matter[f"{sector}_direction"],
                        dtype=np.float64,
                    )
                )
            )
        )
        response_vector = scale * evaluate_b1_polynomial(
            assets.adaptive_matter[f"{sector}_direction"], B1_FID
        )
        stochastic_zero_response[sector] = {
            "analytic_expectation": "exactly zero",
            "normalization": "one prior sigma",
            "coefficient_scale": coefficient_scale,
            "absolute_coefficient_tolerance": coefficient_tolerance,
            "maximum_absolute_direction_coefficient": maximum_direction,
            "response_sigma_single": whitened_norm(response_vector, covariance),
            "pass": bool(maximum_direction <= coefficient_tolerance),
        }

    gaussian_second_coefficients = np.zeros(
        (finite_indices.size, len(B1_POWERS)),
        dtype=np.float64,
    )
    gaussian_second_step_coefficients = np.zeros_like(
        gaussian_second_coefficients
    )
    for sector in DIAGRAMS:
        zero = np.asarray(
            assets.adaptive_matter[f"{sector}_zero"][finite_locations],
            dtype=np.float64,
        )
        rows = finite_rows
        second_coarse = np.asarray(
            [
                pure_b1_coefficients(rows[int(index)]["p1"][sector])
                - 2.0 * zero[local_index]
                + pure_b1_coefficients(rows[int(index)]["m1"][sector])
                for local_index, index in enumerate(finite_indices)
            ],
            dtype=np.float64,
        )
        second_fine = np.asarray(
            [
                4.0
                * (
                    pure_b1_coefficients(rows[int(index)]["p0p5"][sector])
                    - 2.0 * zero[local_index]
                    + pure_b1_coefficients(rows[int(index)]["m0p5"][sector])
                )
                for local_index, index in enumerate(finite_indices)
            ],
            dtype=np.float64,
        )
        second_richardson = (4.0 * second_fine - second_coarse) / 3.0
        gaussian_second_coefficients += second_richardson
        gaussian_second_step_coefficients += second_richardson - second_fine
    half_gaussian_second = 0.5 * evaluate_b1_polynomial(
        gaussian_second_coefficients,
        B1_FID,
    )
    half_gaussian_second_step_error = 0.5 * evaluate_b1_polynomial(
        gaussian_second_step_coefficients,
        B1_FID,
    )
    gaussian_curvature_diagnostic = {
        "coefficient": "one-half d2G/dfNL_rec2 at fNL_rec=0",
        "not_used_in_primary_mcmc": True,
        "missing_for_complete_matched_quadratic_order": "dL/dfNL_rec",
        "amplitude_projection": {
            str(amplitude): {
                "curvature_sigma_single": whitened_norm(
                    amplitude**2 * half_gaussian_second,
                    finite_covariance,
                ),
                "richardson_minus_fine_sigma_single": whitened_norm(
                    amplitude**2 * half_gaussian_second_step_error,
                    finite_covariance,
                ),
            }
            for amplitude in (20.0, 50.0, 100.0)
        },
        "canary_indices": finite_indices,
    }

    table = np.loadtxt(paths.png_table)
    transfer = table[:, 3]
    wave = table[:, 0]
    if not (wave[0] <= KMIN <= wave[-1]) or np.any(transfer <= 0.0):
        raise ValueError("PNG transfer table does not support the fundamental mode")
    transfer_at_kmin = float(
        np.exp(np.interp(np.log(KMIN), np.log(wave), np.log(transfer)))
    )
    # Include the exact finite-box fundamental mode rather than silently
    # replacing it by the first (slightly larger) tabulated k value.
    domain_transfer = np.concatenate(
        (np.asarray([transfer_at_kmin]), transfer[wave > KMIN])
    )
    denominator_rows = []
    for fnl in (-150.0, -100.0, -50.0, -20.0, 20.0, 50.0, 100.0, 150.0):
        values = B_REC_H + fnl * BPHI_REC / domain_transfer
        denominator_rows.append(
            {
                "fNL": fnl,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "positive": bool(np.all(values > 0.0)),
            }
        )
    exact_positive_lower_bound = float(
        -B_REC_H * np.min(domain_transfer) / BPHI_REC
    )

    test_binary = paths.git_root / "build/halo_v1/test_eft_v2_bias_operators"
    algebra_environment = os.environ.copy()
    if not algebra_environment.get("GSL_PREFIX"):
        gsl_prefix = Path(sys.prefix)
        if (
            (gsl_prefix / "include/gsl").is_dir()
            and (gsl_prefix / "lib/libgsl.so").is_file()
        ):
            algebra_environment["GSL_PREFIX"] = str(gsl_prefix)
    test_build = subprocess.run(
        ["make", "-j1", "build/halo_v1/test_eft_v2_bias_operators"],
        cwd=paths.git_root,
        text=True,
        capture_output=True,
        env=algebra_environment,
        check=False,
    )
    if test_build.returncode != 0:
        raise RuntimeError(
            "adaptive denominator algebra-test build failed:\n"
            + test_build.stderr[-4000:]
        )
    test = subprocess.run(
        [str(test_binary)],
        cwd=paths.git_root,
        text=True,
        capture_output=True,
        check=False,
    )
    checks = {
        "cached_r0_pure_b1_evaluation_matches_full_template_below_1e8": bool(
            maximum_r0_relative < 1.0e-8
        ),
        "independent_r0_canary_coefficients_match_below_1e8": bool(
            maximum_canary_relative < 1.0e-8
        ),
        "matter_tree_matches_registered_tree_below_1e8": bool(
            tree_closure["relative_l2"] < 1.0e-8
        ),
        "analytic_response_matches_finite_richardson_below_0p02_sigma_single": bool(
            maximum_analytic_minus_richardson_sigma < 0.02
        ),
        "finite_canary_richardson_step_below_0p02_sigma_single": bool(
            maximum_richardson_step_sigma < 0.02
        ),
        "analytic_uv_renormalization_identities_below_1e10": bool(
            maximum_uv_identity_relative < 1.0e-10
        ),
        "selected_stochastic_response_coefficients_are_zero": bool(
            all(row["pass"] for row in stochastic_zero_response.values())
        ),
        "source_leg_and_limit_tests_pass": bool(test.returncode == 0),
        "all_registered_finite_denominators_positive": bool(
            all(row["positive"] for row in denominator_rows)
        ),
    }
    return {
        "finite_canary_provenance": finite_canary_provenance,
        "r0_closure": r0_closure,
        "maximum_r0_relative_l2": maximum_r0_relative,
        "independent_r0_canary": independent_r0_canary,
        "maximum_independent_r0_canary_relative_l2": maximum_canary_relative,
        "tree_response_closure": tree_closure,
        "loop_response_norm_sigma_single": loop_response_norm,
        "analytic_finite_difference_canary": finite_difference_canary,
        "maximum_analytic_minus_richardson_sigma_single": (
            maximum_analytic_minus_richardson_sigma
        ),
        "maximum_finite_richardson_step_sigma_single": (
            maximum_richardson_step_sigma
        ),
        "analytic_uv_renormalization_identities": uv_identities,
        "maximum_analytic_uv_identity_relative_l2": (
            maximum_uv_identity_relative
        ),
        "selected_stochastic_zero_response": stochastic_zero_response,
        "finite_ashot_projection_note": (
            "the current finite-canary driver and analytic driver both use "
            "the exact Haar shell-weight sum; Ashot is included in the "
            "independent r=0 closure and has analytic zero response"
        ),
        "gaussian_matched_path_curvature_diagnostic": (
            gaussian_curvature_diagnostic
        ),
        "denominator_sentinels": denominator_rows,
        "transfer_at_exact_fundamental_mode": transfer_at_kmin,
        "exact_denominator_lower_fNL_bound_at_kmin": exact_positive_lower_bound,
        "local_likelihood_note": (
            "The primary MCMC uses the fiducial tangent G+f*dG/df_rec and "
            "does not evaluate a finite-f denominator.  The exact denominator "
            "sentinel is nevertheless reported and may fail at the extreme "
            "negative edge of the registered top-hat prior."
        ),
        "algebra_test": {
            "binary": str(test_binary),
            "sha256": sha256(test_binary),
            "gsl_prefix": algebra_environment.get("GSL_PREFIX"),
            "build_returncode": test_build.returncode,
            "build_stdout_tail": test_build.stdout[-2000:],
            "build_stderr_tail": test_build.stderr[-2000:],
            "returncode": test.returncode,
            "stdout_tail": test.stdout[-2000:],
            "stderr_tail": test.stderr[-2000:],
        },
        "checks": checks,
    }


def gaussian_baseline_record(
    context: inference.CollapsedContext,
) -> tuple[dict[str, Any], Any]:
    map_record, conditional = inference.gaussian_nuisance_map(context)
    residual = np.asarray(conditional.prediction) - np.asarray(context.target)
    sigma = np.sqrt(np.diag(context.covariance_single))
    norm = whitened_norm(residual, context.covariance_single)
    maximum_pull = float(np.max(np.abs(residual / sigma)))
    record = {
        "map": map_record,
        "residual_norm_sigma_single": norm,
        "maximum_diagonal_pull_sigma_single": maximum_pull,
        "valid": bool(map_record["valid"]),
        "pass": bool(map_record["valid"] and norm < 1.0 and maximum_pull < 1.0),
    }
    return record, conditional


def compact_fisher(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "fixed_nuisance_sigma_fNL",
        "all_nuisance_sigma_fNL",
        "b1_fixed_all_other_nuisance_sigma_fNL",
        "information_retention_after_all_nuisance",
        "degradation_all_over_fixed",
        "scaled_fisher_minimum_eigenvalue",
        "scaled_fisher_condition_number",
        "strongest_individual_degeneracies",
    )
    return {key: record[key] for key in keys}


def response_components_at_map(
    context: inference.CollapsedContext,
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    _prediction, _parameters, components = context.evaluate_model(0.0, nuisance)
    result = {}
    for name in (
        "linear_total",
        "fixed_brec_linear_total",
        "adaptive_tree",
        "adaptive_matter_loop",
        "adaptive_stochastic",
    ):
        if name in components:
            result[name] = np.asarray(components[name], dtype=np.float64)[context.indices]
    return result


def save_pdf(figure: plt.Figure, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "MARISA-B v0.9 matched/oracle diagnostic",
            "Subject": "single-realization covariance; Quijote halo fNL=0",
        },
    )
    plt.close(figure)


def plot_fisher(rows: list[dict[str, Any]], path: Path) -> None:
    cuts = np.asarray([row["kmax_h_mpc"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.9, 4.25), sharex=True)
    metrics = (
        ("fixed_nuisance_sigma_fNL", "fixed nuisance"),
        ("all_nuisance_sigma_fNL", "all nuisance marginalized"),
    )
    curves = (
        ("pre", "pre v0.9", "black", "--", "o"),
        ("post_fixed", r"post, fixed $b_{\rm rec}$", "#b2182b", ":", "s"),
        ("post_oracle", r"post matched/oracle", "#2166ac", "-", "D"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for key, curve_label, colour, linestyle, marker in curves:
            axis.plot(
                cuts,
                [row[key][metric] for row in rows],
                color=colour,
                linestyle=linestyle,
                marker=marker,
                lw=1.8,
                label=curve_label,
            )
        axis.set_xlabel(r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$")
        axis.set_ylabel(rf"Fisher $\sigma(f_{{\rm NL}})$ ({label})")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8.7)
    figure.suptitle(
        "MARISA-B v0.9 matched/oracle Fisher forecast\n"
        "Quijote fNL=0; covariance of one realization",
        fontsize=12,
    )
    figure.tight_layout()
    save_pdf(figure, path, "MARISA-B v0.9 Fisher comparison")


def plot_response_decomposition(
    pre_context: inference.CollapsedContext,
    post_context: inference.CollapsedContext,
    pre_nuisance: np.ndarray,
    post_nuisance: np.ndarray,
    assets: TheoryAssets,
    path: Path,
) -> None:
    pre_response = response_components_at_map(pre_context, pre_nuisance)["linear_total"]
    post = response_components_at_map(post_context, post_nuisance)
    k_pair = assets.shared.k_pair[post_context.indices]
    scale = k_pair[:, 0] * k_pair[:, 1]
    x = np.arange(post_context.indices.size)
    pre_sigma = np.sqrt(np.diag(pre_context.covariance_single))
    post_sigma = np.sqrt(np.diag(post_context.covariance_single))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    curves = (
        (pre_response, "pre total", "black", "--", pre_sigma),
        (
            post["fixed_brec_linear_total"],
            "post fixed-brec PNG",
            "#b2182b",
            ":",
            post_sigma,
        ),
        (
            post["adaptive_tree"],
            "oracle Gaussian tree correction",
            "#4393c3",
            "-.",
            post_sigma,
        ),
        (
            post["adaptive_matter_loop"],
            "oracle matter-loop correction",
            "#f28e2b",
            "-.",
            post_sigma,
        ),
        (
            post["adaptive_stochastic"],
            "oracle leading-stochastic correction (analytic zero)",
            "#6a3d9a",
            "-.",
            post_sigma,
        ),
        (
            post["linear_total"],
            "post oracle total",
            "#2166ac",
            "-",
            post_sigma,
        ),
    )
    for vector, label, colour, linestyle, sigma in curves:
        axes[0].plot(
            x,
            scale * vector,
            color=colour,
            ls=linestyle,
            lw=1.7,
            label=label,
        )
        axes[1].plot(
            x,
            vector / sigma,
            color=colour,
            ls=linestyle,
            lw=1.5,
        )
    for axis in axes:
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"$k_1k_2\,\partial B_{000}/\partial f_{\rm NL}$")
    axes[1].set_ylabel(r"$(\partial B/\partial f_{\rm NL})/\sigma_{\rm single,diag}$")
    axes[1].set_xlabel("retained full-2D bin index")
    axes[0].legend(frameon=False, fontsize=8.4, ncol=2)
    axes[0].set_title(
        r"$k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$; first global bin excluded"
    )
    figure.tight_layout()
    save_pdf(figure, path, "MARISA-B v0.9 oracle response decomposition")


def run_fisher(paths: Paths) -> dict[str, Any]:
    assets = load_assets(paths)
    audit = theory_audit(paths, assets)
    rows: list[dict[str, Any]] = []
    plot_payload: tuple[Any, ...] | None = None
    all_baselines_pass = True
    all_conditional_validations_pass = True
    minimum_eigenvalue = math.inf
    maximum_derivative_closure = 0.0
    for kmax in CUTS:
        contexts = {
            "pre": build_context(assets, "pre", kmax, oracle=False),
            "post_fixed": build_context(assets, "post", kmax, oracle=False),
            "post_oracle": build_context(assets, "post", kmax, oracle=True),
        }
        baseline_pre, conditional_pre = gaussian_baseline_record(contexts["pre"])
        baseline_post, conditional_post = gaussian_baseline_record(contexts["post_fixed"])
        all_baselines_pass = all_baselines_pass and baseline_pre["pass"] and baseline_post["pass"]
        nuisance_map = {
            "pre": conditional_pre.nuisance,
            "post_fixed": conditional_post.nuisance,
            "post_oracle": conditional_post.nuisance,
        }
        conditional_validation = {
            name: inference.validate_context(
                context,
                np.random.default_rng(
                    20260803
                    + int(round(1000.0 * kmax))
                    + 100 * tuple(contexts).index(name)
                ),
                random_points=3,
            )
            for name, context in contexts.items()
        }
        all_conditional_validations_pass = (
            all_conditional_validations_pass
            and all(row["pass"] for row in conditional_validation.values())
        )
        fisher_rows: dict[str, dict[str, Any]] = {}
        responses: dict[str, np.ndarray] = {}
        derivative_checks: dict[str, Any] = {}
        for name, context in contexts.items():
            jacobian, derivative = inference.model_jacobian_at_zero(
                context,
                nuisance_map[name],
            )
            fisher, _matrix = inference.fisher_from_jacobian(context, context, jacobian)
            fisher_rows[name] = compact_fisher(fisher)
            responses[name] = jacobian[:, 0]
            derivative_checks[name] = derivative
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(fisher["scaled_fisher_minimum_eigenvalue"]),
            )
            maximum_derivative_closure = max(
                maximum_derivative_closure,
                float(derivative["maximum_relative_step_closure"]),
                float(derivative["registered_linear_total_relative_closure"]),
            )
        covariance_swap = {
            "pre_response_pre_covariance": inference.fixed_fisher_sigma(
                responses["pre"], contexts["pre"]
            ),
            "pre_response_post_covariance": inference.fixed_fisher_sigma(
                responses["pre"], contexts["post_oracle"]
            ),
            "oracle_post_response_pre_covariance": inference.fixed_fisher_sigma(
                responses["post_oracle"], contexts["pre"]
            ),
            "oracle_post_response_post_covariance": inference.fixed_fisher_sigma(
                responses["post_oracle"], contexts["post_oracle"]
            ),
        }
        rows.append(
            {
                "kmax_h_mpc": kmax,
                "n_data": contexts["pre"].target.size,
                "baseline": {"pre": baseline_pre, "post": baseline_post},
                "pre": fisher_rows["pre"],
                "post_fixed": fisher_rows["post_fixed"],
                "post_oracle": fisher_rows["post_oracle"],
                "derivative_checks": derivative_checks,
                "conditional_validation": conditional_validation,
                "covariance_response_swap": covariance_swap,
            }
        )
        if math.isclose(kmax, 0.14):
            plot_payload = (
                contexts["pre"],
                contexts["post_oracle"],
                conditional_pre.nuisance,
                conditional_post.nuisance,
            )
    required_theory_checks = {
        name: bool(value)
        for name, value in audit["checks"].items()
        if name != "all_registered_finite_denominators_positive"
    }
    checks = {
        "all_gaussian_baselines_pass_single_covariance_gate": all_baselines_pass,
        "all_conditional_affine_and_direct_optimizer_validations_pass": (
            all_conditional_validations_pass
        ),
        "all_required_theory_response_gates_pass": all(
            required_theory_checks.values()
        ),
        "all_fisher_matrices_positive_definite": minimum_eigenvalue > 0.0,
        "model_derivative_closure_below_1e8": maximum_derivative_closure < 1.0e-8,
        "single_realization_covariance_everywhere": True,
        "expected_bin_counts": all(
            row["n_data"] == EXPECTED_COUNTS[row["kmax_h_mpc"]] for row in rows
        ),
    }
    paths.figures.mkdir(parents=True, exist_ok=True)
    fisher_figure = paths.figures / "v0p9_fisher_vs_kmax.pdf"
    response_figure = paths.figures / "v0p9_response_decomposition.pdf"
    plot_fisher(rows, fisher_figure)
    assert plot_payload is not None
    plot_response_decomposition(*plot_payload, assets, response_figure)
    payload = {
        "schema": "marisa-b-v0p9-matched-oracle-fisher-v1",
        "created_utc": utc_now(),
        "status": "complete" if all(checks.values()) else "completed_with_failed_gate",
        "scope": "conditional matched/oracle forecast at fiducial fNL=0",
        "covariance": "unscaled sample covariance of one 1 (Gpc/h)^3 realization",
        "theory_audit": audit,
        "kmax_scan": rows,
        "checks": checks,
        "minimum_scaled_fisher_eigenvalue": minimum_eigenvalue,
        "maximum_model_derivative_relative_closure": maximum_derivative_closure,
        "provenance": assets.provenance,
        "figures": {
            "fisher": {"path": str(fisher_figure), "sha256": sha256(fisher_figure)},
            "response": {"path": str(response_figure), "sha256": sha256(response_figure)},
        },
    }
    atomic_json(paths.analysis / "fisher.json", payload)
    return payload


def posterior_full_draws(
    context: inference.CollapsedContext,
    diagnosis: dict[str, Any],
    chain_path: Path,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    _retained, theta = inference.retained_chain_samples(
        context,
        chain_path=chain_path,
        diagnosis=diagnosis,
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        theta.shape[0],
        size=min(count, theta.shape[0]),
        replace=False,
    )
    result = np.empty((selected.size, 1 + len(context.names)), dtype=np.float64)
    for output_index, pool_index in enumerate(selected):
        local_theta = np.asarray(theta[pool_index], dtype=np.float64)
        conditional = context.conditional(local_theta)
        nuisance = conditional.nuisance.copy()
        covariance = np.linalg.inv(conditional.hessian)
        nuisance[context.linear_indices] += (
            np.linalg.cholesky(covariance)
            @ rng.normal(size=context.linear_indices.size)
        )
        nuisance[context.nonlinear_indices] = local_theta[1:]
        result[output_index, 0] = local_theta[0]
        result[output_index, 1:] = nuisance
    return result


def plot_mcmc_scan(rows: list[dict[str, Any]], path: Path) -> None:
    cuts = np.asarray([row["kmax_h_mpc"] for row in rows])
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.7, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1.0)},
    )
    for key, label, colour, marker in (
        ("pre", "pre v0.9", "black", "o"),
        ("post_oracle", "post oracle-tangent v0.9", "#2166ac", "D"),
    ):
        medians = np.asarray([row[key]["median"] for row in rows])
        low = np.asarray([row[key]["equal_tail_68"][0] for row in rows])
        high = np.asarray([row[key]["equal_tail_68"][1] for row in rows])
        widths = 0.5 * (high - low)
        top.errorbar(
            cuts,
            medians,
            yerr=np.vstack((medians - low, high - medians)),
            color=colour,
            marker=marker,
            lw=1.6,
            capsize=3,
            label=label,
        )
        bottom.plot(cuts, widths, color=colour, marker=marker, lw=1.7, label=label)
        bottom.plot(
            cuts,
            [row[key]["fisher_marginal_sigma"] for row in rows],
            color=colour,
            ls="--",
            lw=1.1,
            alpha=0.75,
        )
    top.axhline(0.0, color="0.5", lw=0.9)
    top.axhspan(FNL_BOUNDS[0], FNL_BOUNDS[1], color="0.7", alpha=0.045)
    top.set_ylabel(r"$f_{\rm NL}$ posterior (68%)")
    top.grid(alpha=0.18)
    top.legend(frameon=False)
    bottom.set_xlabel(r"$k_{\max}\,[h\,{\rm Mpc}^{-1}]$")
    bottom.set_ylabel("68% half-width")
    bottom.grid(alpha=0.18)
    bottom.legend(
        handles=[
            Line2D([], [], color="black", marker="o", label="pre MCMC"),
            Line2D([], [], color="#2166ac", marker="D", label="post oracle MCMC"),
            Line2D([], [], color="0.4", ls="--", label="marginal Fisher"),
        ],
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        "MARISA-B v0.9: pre/post oracle-tangent MCMC sensitivity\n"
        r"Quijote $f_{\rm NL}=0$; hard prior $[-150,150]$; single-box covariance"
    )
    figure.tight_layout()
    save_pdf(figure, path, "MARISA-B v0.9 pre/post MCMC scan")


def plot_corner(
    pre_samples: np.ndarray,
    post_samples: np.ndarray,
    path: Path,
) -> None:
    labels = (
        r"$f_{\rm NL}$",
        r"$b_1$",
        r"$b_2$",
        r"$b_{K^2}\;(\gamma_2)$",
        r"$A_{\rm shot}$",
        r"$B_{\rm shot}$",
    )
    combined = np.vstack((pre_samples, post_samples))
    ranges: list[tuple[float, float]] = []
    for column in range(combined.shape[1]):
        if column == 0:
            ranges.append(FNL_BOUNDS)
        else:
            low, high = np.quantile(combined[:, column], (0.003, 0.997))
            padding = 0.06 * max(float(high - low), 1.0e-9)
            ranges.append((float(low - padding), float(high + padding)))
    figure = corner.corner(
        pre_samples,
        labels=labels,
        range=ranges,
        bins=36,
        color="black",
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        truths=(0.0, B1_FID, None, None, None, None),
        truth_color="0.45",
        max_n_ticks=4,
        hist_kwargs={"density": True, "lw": 1.4},
        contour_kwargs={"linewidths": 1.2, "linestyles": ("--", "-")},
    )
    corner.corner(
        post_samples,
        fig=figure,
        range=ranges,
        bins=36,
        color="#2166ac",
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        max_n_ticks=4,
        hist_kwargs={"density": True, "lw": 1.4},
        contour_kwargs={"linewidths": 1.2, "linestyles": ("--", "-")},
    )
    figure.legend(
        handles=[
            Line2D([], [], color="black", lw=2.2, label="pre v0.9"),
            Line2D([], [], color="#2166ac", lw=2.2, label="post oracle-tangent v0.9"),
            Line2D([], [], color="0.35", ls="-", label="68% contour"),
            Line2D([], [], color="0.35", ls="--", label="95% contour"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )
    figure.suptitle(
        r"MARISA-B v0.9, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$"
        "\nconditional-linear nuisance draws included",
        y=1.01,
    )
    save_pdf(figure, path, "MARISA-B v0.9 six-parameter corner")


def bestfit_payload(
    context: inference.CollapsedContext,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    profile = inference.minuit_centre(context, marginal=False)
    theta = np.asarray(profile["theta"], dtype=np.float64)
    conditional = context.conditional(theta)
    residual = np.asarray(conditional.prediction) - np.asarray(context.target)
    sigma = np.sqrt(np.diag(context.covariance_single))
    payload = {
        "profiler_centre_only": True,
        "profiler_errors_used": False,
        "theta_names": context.theta_names,
        "theta": theta,
        "nuisance_names": context.names,
        "conditional_nuisance": conditional.nuisance,
        "residual_norm_sigma_single": whitened_norm(
            residual,
            context.covariance_single,
        ),
        "maximum_diagonal_pull_sigma_single": float(np.max(np.abs(residual / sigma))),
        "valid": bool(profile["valid"]),
    }
    arrays = {
        "target": np.asarray(context.target),
        "model": np.asarray(conditional.prediction),
        "sigma": sigma,
        "pull": residual / sigma,
    }
    return payload, arrays


def plot_bestfits(
    contexts: dict[str, inference.CollapsedContext],
    arrays: dict[str, dict[str, np.ndarray]],
    assets: TheoryAssets,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 6.8),
        sharex="col",
        gridspec_kw={"height_ratios": (2.25, 1.0)},
    )
    for column, (key, title) in enumerate(
        (("pre", "pre reconstruction"), ("post_oracle", "post matched/oracle"))
    ):
        context = contexts[key]
        values = arrays[key]
        k_pair = assets.shared.k_pair[context.indices]
        scale = k_pair[:, 0] * k_pair[:, 1]
        x = np.arange(context.indices.size)
        top = axes[0, column]
        lower = axes[1, column]
        top.errorbar(
            x,
            scale * values["target"],
            yerr=scale * values["sigma"],
            fmt="o",
            ms=3.5,
            color="black",
            ecolor="0.7",
            elinewidth=0.8,
            capsize=1.5,
            label="Quijote fid mean; single-box error",
        )
        top.plot(x, scale * values["model"], color="#b2182b", lw=1.8, label="v0.9 best fit")
        lower.axhline(0.0, color="0.5", lw=0.8)
        lower.plot(x, values["pull"], color="#b2182b", lw=1.3, marker="o", ms=2.5)
        lower.set_ylim(-0.5, 0.5)
        lower.set_xlabel("retained full-2D bin index")
        lower.set_ylabel(r"$(B_{\rm th}-B_{\rm data})/\sigma_{\rm single}$")
        top.set_ylabel(r"$k_1k_2 B_{000}$")
        top.set_title(title)
        top.grid(alpha=0.15)
        lower.grid(alpha=0.15)
        top.legend(frameon=False, fontsize=8.2)
    figure.suptitle(
        r"Quijote halo $f_{\rm NL}=0$, $k_{\max}=0.14\,h\,{\rm Mpc}^{-1}$"
        "\nblack=data, red=theory; covariance of one realization",
        fontsize=12,
    )
    figure.tight_layout()
    save_pdf(figure, path, "MARISA-B v0.9 pre/post best-fit B000")


def restricted_fnl_summary(samples: np.ndarray, half_width: float) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    selected = values[np.abs(values) <= float(half_width)]
    if selected.size < 2:
        raise ValueError("too few retained samples for restricted-fNL summary")
    equal = np.quantile(selected, (0.16, 0.5, 0.84))
    return {
        "support": [-float(half_width), float(half_width)],
        "conditional_sample_count": int(selected.size),
        "posterior_mass_in_support": float(selected.size / values.size),
        "equal_tail_68": (float(equal[0]), float(equal[2])),
        "median": float(equal[1]),
        "hdi_68": inference.shortest_hdi(selected, 0.68),
        "standard_deviation": float(np.std(selected, ddof=1)),
    }


def build_mcmc_chain_contract(
    paths: Paths,
    fisher_path: Path,
    assets: TheoryAssets,
    contexts: dict[tuple[str, float], inference.CollapsedContext],
    walkers: int,
) -> dict[str, Any]:
    contract = {
        "schema": "marisa-b-v0p9-matched-oracle-chain-contract-v1",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fisher_sha256": sha256(fisher_path),
        "theory_provenance": assets.provenance,
        "contexts": {
            f"{name}_kmax{kmax:.2f}": context.metadata()
            for (name, kmax), context in contexts.items()
        },
        "prior": {
            "names": assets.prior.names,
            "mean": assets.prior.mean,
            "covariance": assets.prior.covariance,
        },
        "sampler": {
            "ensembles": MCMC_ENSEMBLES,
            "walkers": int(walkers),
            "theta_names": ("fNL",) + NONLINEAR_NAMES,
            "moves": "0.8 StretchMove(a=2) + 0.2 DEMove",
            "steps_are_extendable_and_not_part_of_fingerprint": True,
        },
    }
    return {
        "schema": contract["schema"],
        "fingerprint": sha256_payload(contract),
        "contract": contract,
    }


def prepare_mcmc_chain_contract(
    paths: Paths,
    expected: dict[str, Any],
) -> dict[str, Any]:
    fingerprint = str(expected["fingerprint"])
    if paths.chains.is_file():
        if not paths.chain_contract.is_file():
            raise RuntimeError(
                "existing chains lack a model contract; rerun with "
                "--overwrite-chains"
            )
        recorded = json.loads(paths.chain_contract.read_text(encoding="utf-8"))
        recorded_contract = recorded.get("contract")
        if not isinstance(recorded_contract, dict) or (
            sha256_payload(recorded_contract) != recorded.get("fingerprint")
        ):
            raise RuntimeError(
                "existing chain sidecar fails its internal fingerprint; "
                "rerun with --overwrite-chains"
            )
        if recorded.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "existing chains belong to a different model fingerprint; "
                "rerun with --overwrite-chains"
            )
        with h5py.File(paths.chains, "r") as chain_file:
            stored = chain_file.attrs.get("marisa_b_model_fingerprint")
            if isinstance(stored, bytes):
                stored = stored.decode("utf-8")
            if stored != fingerprint:
                raise RuntimeError(
                    "HDF5 chain fingerprint is missing or stale; rerun with "
                    "--overwrite-chains"
                )
        return recorded

    record = {
        **expected,
        "created_utc": utc_now(),
        "chain_path": str(paths.chains),
    }
    atomic_json(paths.chain_contract, record)
    paths.chains.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(paths.chains, "w") as chain_file:
        chain_file.attrs["marisa_b_model_fingerprint"] = fingerprint
        chain_file.attrs["marisa_b_contract_schema"] = str(expected["schema"])
    return record


def run_mcmc(paths: Paths, workers: int, walkers: int, steps: int) -> dict[str, Any]:
    if not 1 <= workers <= 28:
        raise ValueError("MCMC workers must be in [1,28]")
    if walkers < 2 * (1 + len(NONLINEAR_NAMES)):
        raise ValueError("too few MCMC walkers")
    fisher_path = paths.analysis / "fisher.json"
    if not fisher_path.is_file():
        run_fisher(paths)
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    assets = load_assets(paths)
    if fisher.get("provenance") != jsonable(assets.provenance):
        run_fisher(paths)
        fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
        if fisher.get("provenance") != jsonable(assets.provenance):
            raise RuntimeError("Fisher provenance remained stale after regeneration")
    current_canary = finite_canary_manifest_provenance(paths, assets.shared)
    if (
        fisher.get("theory_audit", {}).get("finite_canary_provenance")
        != jsonable(current_canary)
    ):
        raise RuntimeError(
            "finite-canary provenance changed after Fisher; rerun --stage fisher"
        )
    if not all(
        bool(fisher["checks"][name])
        for name in (
            "all_gaussian_baselines_pass_single_covariance_gate",
            "all_conditional_affine_and_direct_optimizer_validations_pass",
            "all_required_theory_response_gates_pass",
            "all_fisher_matrices_positive_definite",
            "model_derivative_closure_below_1e8",
            "single_realization_covariance_everywhere",
            "expected_bin_counts",
        )
    ):
        raise RuntimeError("registered Fisher/baseline gate failed; MCMC suppressed")

    contexts: dict[tuple[str, float], inference.CollapsedContext] = {}
    diagnoses: dict[tuple[str, float], dict[str, Any]] = {}
    run_records: dict[str, Any] = {}
    for kmax in CUTS:
        contexts[("pre", kmax)] = build_context(assets, "pre", kmax, oracle=False)
        contexts[("post_oracle", kmax)] = build_context(
            assets,
            "post",
            kmax,
            oracle=True,
        )
    chain_contract = prepare_mcmc_chain_contract(
        paths,
        build_mcmc_chain_contract(
            paths,
            fisher_path,
            assets,
            contexts,
            walkers,
        ),
    )
    for order, ((name, kmax), context) in enumerate(contexts.items()):
        requested = int(steps)
        diagnosis: dict[str, Any] | None = None
        records = []
        for attempt in range(3):
            record = inference.run_mcmc_context(
                context,
                chain_path=paths.chains,
                ensembles=MCMC_ENSEMBLES,
                walkers=walkers,
                steps=requested,
                workers=workers,
                seed=20260803 + 10000 * order,
            )
            diagnosis = inference.diagnose_context(
                context,
                chain_path=paths.chains,
                burn_fraction=MCMC_BURN_FRACTION,
            )
            records.append(
                {
                    "requested_steps": requested,
                    "run": record,
                    "pass": diagnosis["pass"],
                    "convergence_checks": diagnosis["convergence_checks"],
                }
            )
            if diagnosis["pass"]:
                break
            requested *= 2
        assert diagnosis is not None
        diagnoses[(name, kmax)] = diagnosis
        run_records[f"{name}_kmax{kmax:.2f}"] = records
        print(
            f"v0.9 MCMC {name} kmax={kmax:.2f}: "
            f"pass={diagnosis['pass']} steps={requested}",
            flush=True,
        )

    fisher_by_cut = {float(row["kmax_h_mpc"]): row for row in fisher["kmax_scan"]}
    rows = []
    all_converged = True
    for kmax in CUTS:
        row: dict[str, Any] = {"kmax_h_mpc": kmax, "n_data": EXPECTED_COUNTS[kmax]}
        for name in ("pre", "post_oracle"):
            diagnosis = diagnoses[(name, kmax)]
            all_converged = all_converged and bool(diagnosis["pass"])
            summary = diagnosis["fnl_summary"]
            equal = tuple(float(value) for value in summary["equal_tail_68"])
            row[name] = {
                "median": float(summary["median"]),
                "equal_tail_68": equal,
                "equal_tail_68_half_width": 0.5 * (equal[1] - equal[0]),
                "hdi_68": summary["hdi_68"],
                "equal_tail_95": summary["equal_tail_95"],
                "standard_deviation": summary["standard_deviation"],
                "boundary_mass_outer_2_percent_each_side": summary[
                    "boundary_mass_outer_2_percent_each_side"
                ],
                "posterior_mass_abs_fNL_le_50": None,
                "posterior_mass_abs_fNL_le_100": None,
                "fisher_marginal_sigma": float(
                    fisher_by_cut[kmax][name]["all_nuisance_sigma_fNL"]
                ),
                "converged": bool(diagnosis["pass"]),
                "maximum_rank_rhat": float(
                    np.max(diagnosis["rank_normalized_split_rhat"])
                ),
                "minimum_bulk_ess": float(np.min(diagnosis["bulk_ess"])),
                "minimum_tail_ess": float(np.min(diagnosis["tail_ess_5_95"])),
                "minimum_retained_length_over_tau": float(
                    np.min(diagnosis["retained_length_over_tau"])
                ),
            }
            _retained, flat = inference.retained_chain_samples(
                contexts[(name, kmax)],
                chain_path=paths.chains,
                diagnosis=diagnosis,
            )
            fnl = flat[:, 0]
            mass_50 = float(np.mean(np.abs(fnl) <= 50.0))
            mass_100 = float(np.mean(np.abs(fnl) <= 100.0))
            row[name]["posterior_mass_abs_fNL_le_50"] = mass_50
            row[name]["posterior_mass_abs_fNL_le_100"] = mass_100
            row[name]["posterior_mass_50_lt_abs_fNL_le_100"] = mass_100 - mass_50
            row[name]["posterior_mass_100_lt_abs_fNL_le_150"] = 1.0 - mass_100
            row[name]["restricted_support_summaries"] = {
                "abs_fNL_le_50": restricted_fnl_summary(fnl, 50.0),
                "abs_fNL_le_100": restricted_fnl_summary(fnl, 100.0),
            }
            row[name]["local_tangent_scope_audit"] = {
                "exact_expansion_point_fNL": 0.0,
                "posterior_mass_abs_fNL_le_50": mass_50,
                "posterior_mass_abs_fNL_le_100": mass_100,
                "hard_acceptance_gate": False,
                "interpretation": (
                    "descriptive locality audit only; restricted summaries "
                    "do not turn the tangent likelihood into an exact finite-"
                    "fNL matched-reconstruction posterior"
                ),
            }
        row["post_over_pre_width68"] = (
            row["post_oracle"]["equal_tail_68_half_width"]
            / row["pre"]["equal_tail_68_half_width"]
        )
        rows.append(row)

    primary_contexts = {
        "pre": contexts[("pre", 0.14)],
        "post_oracle": contexts[("post_oracle", 0.14)],
    }
    full_samples = {
        name: posterior_full_draws(
            context,
            diagnoses[(name, 0.14)],
            paths.chains,
            count=50000,
            seed=20260803 + index * 711,
        )
        for index, (name, context) in enumerate(primary_contexts.items())
    }
    bestfits = {}
    bestfit_arrays = {}
    for name, context in primary_contexts.items():
        bestfits[name], bestfit_arrays[name] = bestfit_payload(context)

    paths.figures.mkdir(parents=True, exist_ok=True)
    mcmc_figure = paths.figures / "v0p9_mcmc_vs_kmax.pdf"
    corner_figure = paths.figures / "v0p9_corner_pre_post_kmax0p14.pdf"
    bestfit_figure = paths.figures / "v0p9_bestfit_b000_pre_post.pdf"
    plot_mcmc_scan(rows, mcmc_figure)
    plot_corner(full_samples["pre"], full_samples["post_oracle"], corner_figure)
    plot_bestfits(primary_contexts, bestfit_arrays, assets, bestfit_figure)
    payload = {
        "schema": "marisa-b-v0p9-matched-oracle-mcmc-v1",
        "created_utc": utc_now(),
        "status": "complete" if all_converged else "completed_with_nonconverged_chain",
        "all_contexts_converged": all_converged,
        "hard_fNL_prior": FNL_BOUNDS,
        "b1_prior": {"mean": B1_FID, "sigma": B1_SIGMA, "free": True},
        "covariance": "single-realization sample covariance; never divided by 500",
        "profiler_policy": "centres and best-fit curves only; no profiler error is used",
        "likelihood_scope": (
            "local fiducial-tangent sensitivity model: exact matched/oracle "
            "first derivative at fNL=0, not an exact finite-fNL adaptive-"
            "reconstruction likelihood"
        ),
        "finite_path_terms_not_claimed": (
            "dL/dfNL_rec and one-half d2G/dfNL_rec2 at quadratic order"
        ),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fisher": {
            "path": str(fisher_path),
            "sha256": sha256(fisher_path),
        },
        "theory_provenance": assets.provenance,
        "chain_contract": {
            "path": str(paths.chain_contract),
            "sha256": sha256(paths.chain_contract),
            "fingerprint": chain_contract["fingerprint"],
            "schema": chain_contract["schema"],
        },
        "kmax_scan": rows,
        "bestfits_kmax0p14": bestfits,
        "primary_full_parameter_order": ("fNL",) + NAMES,
        "primary_full_draw_quantiles": {
            name: np.quantile(samples, (0.16, 0.5, 0.84), axis=0)
            for name, samples in full_samples.items()
        },
        "run_records": run_records,
        "figures": {
            "mcmc": {"path": str(mcmc_figure), "sha256": sha256(mcmc_figure)},
            "corner": {"path": str(corner_figure), "sha256": sha256(corner_figure)},
            "bestfit": {"path": str(bestfit_figure), "sha256": sha256(bestfit_figure)},
        },
        "chains": {"path": str(paths.chains), "sha256": sha256(paths.chains)},
    }
    atomic_json(paths.analysis / "mcmc.json", payload)
    return payload


def report_text(summary: dict[str, Any]) -> str:
    fisher = summary["fisher"]
    mcmc = summary["mcmc"]
    lines = [
        "# MARISA-B v0.9 matched/oracle result",
        "",
        "## Scope",
        "",
        (
            "This is a conditional oracle bound using the Quijote z=1, R=15, "
            "fNL=0 halo catalogues.  The post forward model follows "
            "fNL_rec=fNL, but no unknown-fNL catalogue has been reconstructed "
            "with an operational estimator."
        ),
        "",
        "The covariance and every plotted error bar are for one 1 (Gpc/h)^3 realization; the covariance is never divided by 500.",
        "",
        "## Fisher scan",
        "",
        "| kmax | pre fixed | pre marginal | post fixed-brec marginal | post oracle fixed | post oracle marginal |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fisher["kmax_scan"]:
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} "
            f"| {row['pre']['fixed_nuisance_sigma_fNL']:.3f} "
            f"| {row['pre']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_fixed']['all_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_oracle']['fixed_nuisance_sigma_fNL']:.3f} "
            f"| {row['post_oracle']['all_nuisance_sigma_fNL']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## MCMC scan",
            "",
            (
                "These chains use the registered local oracle-tangent "
                "likelihood.  They are a finite-prior sensitivity test, not "
                "an exact finite-fNL matched-reconstruction posterior; the "
                "quadratic product-rule terms dL/dr and d2G/dr2/2 are not "
                "claimed."
            ),
            "",
            "| kmax | pre fNL (68%) | post oracle fNL (68%) | post/pre half-width |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in mcmc["kmax_scan"]:
        pre = row["pre"]
        post = row["post_oracle"]
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} "
            f"| {pre['median']:.2f} [{pre['equal_tail_68'][0]:.2f}, {pre['equal_tail_68'][1]:.2f}] "
            f"| {post['median']:.2f} [{post['equal_tail_68'][0]:.2f}, {post['equal_tail_68'][1]:.2f}] "
            f"| {row['post_over_pre_width68']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Local-tangent support audit",
            "",
            (
                "Posterior mass and conditional restricted summaries are "
                "reported diagnostically; they do not make the local "
                "tangent likelihood exact at finite fNL."
            ),
            "",
            "| kmax | pre mass <=50 / <=100 | post mass <=50 / <=100 | pre restricted-50 68% | post restricted-50 68% |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mcmc["kmax_scan"]:
        pre = row["pre"]
        post = row["post_oracle"]
        pre50 = pre["restricted_support_summaries"]["abs_fNL_le_50"]
        post50 = post["restricted_support_summaries"]["abs_fNL_le_50"]
        lines.append(
            f"| {row['kmax_h_mpc']:.2f} "
            f"| {pre['posterior_mass_abs_fNL_le_50']:.3f} / "
            f"{pre['posterior_mass_abs_fNL_le_100']:.3f} "
            f"| {post['posterior_mass_abs_fNL_le_50']:.3f} / "
            f"{post['posterior_mass_abs_fNL_le_100']:.3f} "
            f"| [{pre50['equal_tail_68'][0]:.2f}, {pre50['equal_tail_68'][1]:.2f}] "
            f"| [{post50['equal_tail_68'][0]:.2f}, {post50['equal_tail_68'][1]:.2f}] |"
        )
    theory_audit_payload = fisher["theory_audit"]
    curvature = theory_audit_payload[
        "gaussian_matched_path_curvature_diagnostic"
    ]
    lines.extend(
        [
            "",
            "## Gaussian matched-path curvature diagnostic",
            "",
            (
                "This is the separately measurable G_rr/2 contribution.  "
                "It is not included in the primary MCMC because the complete "
                "quadratic product rule also requires dL/dr."
            ),
            "",
            "| abs(fNL) | curvature norm [sigma_single] | Richardson error [sigma_single] |",
            "|---:|---:|---:|",
        ]
    )
    for amplitude in sorted(curvature["amplitude_projection"], key=float):
        record = curvature["amplitude_projection"][amplitude]
        lines.append(
            f"| {float(amplitude):.0f} "
            f"| {record['curvature_sigma_single']:.6g} "
            f"| {record['richardson_minus_fine_sigma_single']:.6g} |"
        )
    denominator = theory_audit_payload
    lines.extend(
        [
            "",
            "## Important finite-amplitude limit",
            "",
            (
                "The local oracle tangent is exact at fNL=0.  The literal "
                "finite denominator becomes non-positive below "
                f"fNL={denominator['exact_denominator_lower_fNL_bound_at_kmin']:.3f} "
                "at the lowest supported displacement mode.  Therefore the "
                "small negative sliver of the nominal [-150,150] prior is not "
                "an exact finite-reconstruction model, even though the local "
                "tangent likelihood remains finite there."
            ),
            "",
            "## Acceptance",
            "",
        ]
    )
    for name, passed in summary["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `v0p9_fisher_vs_kmax.pdf`",
            "- `v0p9_response_decomposition.pdf`",
            "- `v0p9_mcmc_vs_kmax.pdf`",
            "- `v0p9_corner_pre_post_kmax0p14.pdf`",
            "- `v0p9_bestfit_b000_pre_post.pdf`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(paths: Paths) -> dict[str, Any]:
    fisher_path = paths.analysis / "fisher.json"
    mcmc_path = paths.analysis / "mcmc.json"
    if not fisher_path.is_file() or not mcmc_path.is_file():
        raise FileNotFoundError("fisher.json and mcmc.json are required")
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    mcmc = json.loads(mcmc_path.read_text(encoding="utf-8"))
    if fisher.get("schema") != "marisa-b-v0p9-matched-oracle-fisher-v1":
        raise ValueError("unexpected Fisher schema")
    if mcmc.get("schema") != "marisa-b-v0p9-matched-oracle-mcmc-v1":
        raise ValueError("unexpected MCMC schema")
    assets = load_assets(paths)
    if fisher.get("provenance") != jsonable(assets.provenance):
        raise RuntimeError("Fisher provenance is stale relative to current inputs")
    current_canary = finite_canary_manifest_provenance(paths, assets.shared)
    recorded_canary = fisher.get("theory_audit", {}).get(
        "finite_canary_provenance"
    )
    if recorded_canary != jsonable(current_canary):
        raise RuntimeError("Fisher finite-canary provenance is stale")
    if mcmc.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("MCMC runner provenance is stale")
    if mcmc.get("theory_provenance") != fisher.get("provenance"):
        raise RuntimeError("MCMC and Fisher theory provenance differ")
    if mcmc.get("fisher", {}).get("sha256") != sha256(fisher_path):
        raise RuntimeError("MCMC was not generated from the current Fisher result")
    if not paths.chains.is_file() or not paths.chain_contract.is_file():
        raise FileNotFoundError("MCMC chain or chain contract is missing")
    if mcmc.get("chains", {}).get("sha256") != sha256(paths.chains):
        raise RuntimeError("MCMC chain hash differs from the reported chain")
    if (
        mcmc.get("chain_contract", {}).get("sha256")
        != sha256(paths.chain_contract)
    ):
        raise RuntimeError("MCMC chain-contract hash differs from the report")
    chain_contract = json.loads(paths.chain_contract.read_text(encoding="utf-8"))
    fingerprint = str(chain_contract.get("fingerprint", ""))
    if not isinstance(chain_contract.get("contract"), dict) or (
        sha256_payload(chain_contract["contract"]) != fingerprint
    ):
        raise RuntimeError("MCMC chain sidecar fails its internal fingerprint")
    if mcmc.get("chain_contract", {}).get("fingerprint") != fingerprint:
        raise RuntimeError("MCMC chain fingerprint differs from the report")
    with h5py.File(paths.chains, "r") as chain_file:
        stored_fingerprint = chain_file.attrs.get(
            "marisa_b_model_fingerprint"
        )
        if isinstance(stored_fingerprint, bytes):
            stored_fingerprint = stored_fingerprint.decode("utf-8")
        if stored_fingerprint != fingerprint:
            raise RuntimeError("HDF5 chain fingerprint differs from its contract")
    theory_checks = fisher["theory_audit"]["checks"]
    acceptance = {
        "data_covariance_and_bin_contract": bool(
            fisher["checks"]["single_realization_covariance_everywhere"]
            and fisher["checks"]["expected_bin_counts"]
        ),
        "gaussian_baseline_gate": bool(
            fisher["checks"]["all_gaussian_baselines_pass_single_covariance_gate"]
        ),
        "fisher_geometry_and_derivative_gate": bool(
            fisher["checks"]["all_fisher_matrices_positive_definite"]
            and fisher["checks"]["model_derivative_closure_below_1e8"]
            and fisher["checks"][
                "all_conditional_affine_and_direct_optimizer_validations_pass"
            ]
        ),
        "r0_and_tree_response_closure": bool(
            theory_checks[
                "cached_r0_pure_b1_evaluation_matches_full_template_below_1e8"
            ]
            and theory_checks[
                "independent_r0_canary_coefficients_match_below_1e8"
            ]
            and theory_checks["matter_tree_matches_registered_tree_below_1e8"]
        ),
        "analytic_response_and_finite_canary": bool(
            theory_checks[
                "analytic_response_matches_finite_richardson_below_0p02_sigma_single"
            ]
            and theory_checks[
                "finite_canary_richardson_step_below_0p02_sigma_single"
            ]
            and theory_checks[
                "analytic_uv_renormalization_identities_below_1e10"
            ]
            and theory_checks[
                "selected_stochastic_response_coefficients_are_zero"
            ]
        ),
        "kernel_algebra_limits": bool(theory_checks["source_leg_and_limit_tests_pass"]),
        "finite_denominator_entire_nominal_prior": bool(
            theory_checks["all_registered_finite_denominators_positive"]
        ),
        "mcmc_convergence": bool(mcmc["all_contexts_converged"]),
        "profiler_widths_not_used": True,
        "oracle_scope_explicit": True,
    }
    summary = {
        "schema": "marisa-b-v0p9-matched-oracle-release-v1",
        "created_utc": utc_now(),
        "goal_thread": "019f89c0-0fed-71b3-8d32-a2953465f08b",
        "status": "accepted" if all(acceptance.values()) else "completed_with_failed_acceptance_gate",
        "scientific_label": (
            "conditional fNL=0 matched/oracle forecast; not an operational "
            "unknown-fNL reconstruction likelihood"
        ),
        "model": (
            "halo tree + pure-matter SPT one-loop uplift + finite local-PNG "
            "halo tree through fNL^2 + matter PNG one-loop/B112II uplift + "
            "two leading stochastic amplitudes"
        ),
        "fisher": fisher,
        "mcmc": mcmc,
        "acceptance": acceptance,
        "known_limit": (
            "The literal adaptive denominator crosses zero in a narrow "
            "negative-fNL edge interval; primary inference is a local tangent."
        ),
        "outputs": {
            "analysis": str(paths.analysis),
            "figures": str(paths.figures),
            "chains": str(paths.chains),
            "chain_contract": str(paths.chain_contract),
            "finite_canary_raw_archive": str(paths.raw),
            "finite_canary_manifest": str(paths.finite_canary_manifest),
            "analytic_raw_archive": str(paths.analytic_raw),
            "analytic_response_manifest": str(paths.analytic_manifest),
        },
    }
    atomic_json(paths.summary, summary)
    temporary = paths.report.with_name(f".{paths.report.name}.tmp-{os.getpid()}")
    temporary.write_text(report_text(summary), encoding="utf-8")
    temporary.replace(paths.report)
    return summary


def main() -> None:
    args = parse_args()
    paths = paths_for(
        args.repo_root.expanduser().resolve(),
        data_root=(
            args.data_root.expanduser().resolve()
            if args.data_root is not None
            else None
        ),
        output_root=(
            args.output_root.expanduser().resolve()
            if args.output_root is not None
            else None
        ),
    )
    require_files(paths)
    if not 1 <= args.workers <= 28 or not 1 <= args.mcmc_workers <= 28:
        raise ValueError("all worker limits must be in [1,28]")
    if args.overwrite_chains and paths.chains.is_file():
        paths.chains.unlink()
    if args.overwrite_chains and paths.chain_contract.is_file():
        paths.chain_contract.unlink()
    if args.overwrite_chains:
        (paths.analysis / "mcmc.json").unlink(missing_ok=True)
    stages = (
        (
            "finite_canaries",
            "compile_analytic",
            "analytic_response",
            "fisher",
            "mcmc",
            "finalize",
        )
        if args.stage == "all"
        else (args.stage,)
    )
    results = {}
    for stage in stages:
        if stage == "compile":
            results[stage] = compile_driver(paths)
        elif stage == "compile_analytic":
            results[stage] = compile_driver(
                paths,
                executable=paths.analytic_executable,
            )
        elif stage == "response":
            results[stage] = produce_response(
                paths,
                workers=args.workers,
                overwrite=args.overwrite_response,
            )
        elif stage == "finite_canaries":
            results[stage] = produce_finite_canaries(
                paths,
                workers=args.workers,
                overwrite=args.overwrite_response,
            )
        elif stage == "analytic_response":
            results[stage] = produce_analytic_response(
                paths,
                workers=args.workers,
                overwrite=args.overwrite_response,
            )
        elif stage == "fisher":
            results[stage] = run_fisher(paths)
        elif stage == "mcmc":
            results[stage] = run_mcmc(
                paths,
                workers=args.mcmc_workers,
                walkers=args.mcmc_walkers,
                steps=args.mcmc_steps,
            )
        elif stage == "finalize":
            results[stage] = finalize(paths)
        else:
            raise ValueError(stage)
    print(json.dumps(jsonable({"status": "done", "stages": tuple(results)}), indent=2))


if __name__ == "__main__":
    main()
