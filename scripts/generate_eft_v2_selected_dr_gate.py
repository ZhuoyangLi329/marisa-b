#!/usr/bin/env python3
"""Generate the current-source EFT-v2 selected-triangle DR certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


CASES = (
    {
        "label": "scalene",
        "triangle": (0.045, 0.067, 0.083),
        "frequency": 64,
        "previous_frequency": 48,
        "direct": (16, 32, 24),
        "previous_direct": (12, 24, 16),
    },
    {
        "label": "equilateral",
        "triangle": (0.067, 0.067, 0.067),
        "frequency": 128,
        "previous_frequency": 96,
        "direct": (24, 48, 32),
        "previous_direct": (16, 32, 24),
    },
    {
        "label": "squeezed",
        "triangle": (0.025, 0.067, 0.083),
        "frequency": 64,
        "previous_frequency": 48,
        "direct": (32, 64, 48),
        "previous_direct": (24, 48, 32),
    },
)
SELECTED_BIAS = {
    "b1": 2.2,
    "b2": 0.3,
    "gamma2": -0.2,
    "b3": 0.1,
}
MATTER_BIAS = {
    "b1": 1.0,
    "b2": 0.0,
    "gamma2": 0.0,
    "b3": 0.0,
}
SOURCE_PATHS = {
    "r0_template_driver_cpp": "src/eft_v2/r0_template_driver.cpp",
    "parameter_registry_cpp": "src/eft_v2/parameter_registry.cpp",
    "parameter_registry_h": "src/eft_v2/parameter_registry.h",
    "template_algebra_cpp": "src/eft_v2/template_algebra.cpp",
    "template_algebra_h": "src/eft_v2/template_algebra.h",
    "kernel_primitives_cpp": "src/eft_v2/kernel_primitives.cpp",
    "kernel_primitives_h": "src/eft_v2/kernel_primitives.h",
    "field_kernel_provider_cpp": "src/eft_v2/field_kernel_provider.cpp",
    "field_kernel_provider_h": "src/eft_v2/field_kernel_provider.h",
    "fftlog_dr_oracle_cpp": "src/eft_v2/fftlog_dr_oracle.cpp",
    "fftlog_dr_oracle_h": "src/eft_v2/fftlog_dr_oracle.h",
    "direct_evaluator_cpp": "src/eft_v2/direct_evaluator.cpp",
    "direct_evaluator_h": "src/eft_v2/direct_evaluator.h",
    "ir_safe_integrands_cpp": "src/eft_v2/ir_safe_integrands.cpp",
    "ir_safe_integrands_h": "src/eft_v2/ir_safe_integrands.h",
    "tracer_power_cpp": "src/eft_v2/tracer_power.cpp",
    "tracer_power_h": "src/eft_v2/tracer_power.h",
    "uv_subtraction_cpp": "src/eft_v2/uv_subtraction.cpp",
    "uv_subtraction_h": "src/eft_v2/uv_subtraction.h",
    "bias_operators_cpp": "src/eft_v2/bias_operators.cpp",
    "bias_operators_h": "src/eft_v2/bias_operators.h",
    "diagram_assembler_cpp": "src/eft_v2/diagram_assembler.cpp",
    "diagram_assembler_h": "src/eft_v2/diagram_assembler.h",
    "generated_b222_b321i_tables_h": (
        "src/eft_v2/generated_b222_b321i_tables.h"
    ),
    "generated_b411_table_h": "src/eft_v2/generated_b411_table.h",
    "halo_v1_cpp": "src/halo_v1/halo_v1.cpp",
    "halo_v1_h": "src/halo_v1/halo_v1.h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--omp-threads", type=int, default=28)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {value}")
        ),
    )


def run_key(
    command: list[str],
    driver_hash: str,
    power_hash: str,
    threads: int,
) -> str:
    payload = json.dumps(
        {
            "command": command,
            "driver_sha256": driver_hash,
            "linear_power_sha256": power_hash,
            "omp_threads": threads,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run_driver(
    command: list[str],
    cache_path: Path,
    driver_hash: str,
    power_hash: str,
    threads: int,
    resume: bool,
) -> dict[str, Any]:
    key = run_key(command, driver_hash, power_hash, threads)
    if cache_path.exists():
        if not resume:
            raise FileExistsError(
                f"{cache_path} exists; pass --resume to validate and reuse"
            )
        cached = strict_json(cache_path.read_text())
        if (
            cached.get("schema")
            != "marisa-b-eft-v2-selected-dr-raw-run-v1"
            or cached.get("run_key") != key
            or cached.get("command") != command
            or cached.get("driver_sha256") != driver_hash
            or cached.get("linear_power_sha256") != power_hash
            or cached.get("omp_threads") != threads
            or not isinstance(cached.get("result"), dict)
        ):
            raise RuntimeError(f"invalid cached result {cache_path}")
        print(
            json.dumps(
                {
                    "record": "selected_dr_run",
                    "status": "reused",
                    "cache": str(cache_path),
                    "elapsed_seconds": cached.get("elapsed_seconds"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return cached

    print(
        json.dumps(
            {
                "record": "selected_dr_run",
                "status": "running",
                "cache": str(cache_path),
                "selector": command[1],
                "triangle": [float(value) for value in command[3:6]],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"driver failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    if completed.stderr.strip():
        raise RuntimeError(
            f"driver wrote stderr despite success: {completed.stderr.strip()}"
        )
    result = strict_json(completed.stdout)
    if not isinstance(result, dict):
        raise RuntimeError("driver output is not one JSON object")
    record = {
        "schema": "marisa-b-eft-v2-selected-dr-raw-run-v1",
        "run_key": key,
        "driver_sha256": driver_hash,
        "linear_power_sha256": power_hash,
        "omp_threads": threads,
        "command": command,
        "elapsed_seconds": elapsed,
        "result": result,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "record": "selected_dr_run",
                "status": "complete",
                "cache": str(cache_path),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def selected_command(
    driver: Path,
    power: Path,
    mode: str,
    triangle: tuple[float, float, float],
    frequency: int,
    direct: tuple[int, int, int],
) -> list[str]:
    if mode == "analytic":
        selector = "--selected-analytic"
        fft_kmin, fft_kmax = 1.0e-5, 10.0
        fft_frequency = frequency
        fft_grid = max(256, 4 * frequency)
        fft_bias = "-0.3"
    elif mode == "direct":
        selector = "--selected-direct"
        fft_kmin, fft_kmax = 1.0e-6, 1.0e3
        fft_frequency = 512
        fft_grid = 2048
        fft_bias = "auto"
    else:
        raise ValueError(f"unknown mode {mode}")
    return [
        str(driver),
        selector,
        str(power),
        *(format(value, ".17g") for value in triangle),
        format(fft_kmin, ".17g"),
        format(fft_kmax, ".17g"),
        str(fft_frequency),
        str(fft_grid),
        fft_bias,
        "1e-12",
        "1e-8",
        "10",
        "0.0001",
        "30",
        *(str(value) for value in direct),
        "0.3",
        "64",
        "64",
    ]


def monomial_value(monomial: str, parameters: dict[str, float]) -> float:
    if monomial == "1":
        return 1.0
    result = 1.0
    for factor in monomial.split("*"):
        name, exponent_text = factor.split("^")
        if name not in parameters:
            raise KeyError(f"unregistered bias parameter {name}")
        result *= parameters[name] ** int(exponent_text)
    return result


def polynomial_value(
    polynomial: dict[str, float],
    parameters: dict[str, float],
) -> float:
    return sum(
        float(coefficient) * monomial_value(monomial, parameters)
        for monomial, coefficient in polynomial.items()
    )


def templates(raw: dict[str, Any], mode: str) -> dict[str, Any]:
    result = raw["result"]
    if result.get("schema") != "marisa-b-eft-v2-selected-dr-v2":
        raise RuntimeError("selected driver returned an unexpected schema")
    branch = result.get(mode)
    if not isinstance(branch, dict):
        raise RuntimeError(f"selected output lacks {mode}")
    value = branch.get("templates")
    if not isinstance(value, dict):
        raise RuntimeError(f"selected {mode} output lacks templates")
    return value


def validate_raw_result(
    raw: dict[str, Any],
    mode: str,
    triangle: tuple[float, float, float],
    direct_quadrature: tuple[int, int, int],
) -> None:
    result = raw["result"]
    stored_triangle = np.asarray(result.get("triangle"), dtype=float)
    expected_triangle = np.sort(np.asarray(triangle, dtype=float))
    if (
        stored_triangle.shape != (3,)
        or not np.all(np.isfinite(stored_triangle))
        or not np.allclose(
            stored_triangle,
            expected_triangle,
            rtol=0.0,
            atol=2.0e-15,
        )
    ):
        raise RuntimeError("selected result has the wrong triangle")
    if mode == "analytic":
        if "analytic" not in result or "direct" in result:
            raise RuntimeError("analytic run contains the wrong branches")
        return
    if mode != "direct":
        raise ValueError(f"unknown selected-result mode {mode}")
    branch = result.get("direct")
    if (
        not isinstance(branch, dict)
        or branch.get("quadrature") != list(direct_quadrature)
        or branch.get("qmin") != 1.0e-4
        or branch.get("qmax") != 30.0
        or branch.get("mu_ren") != 0.3
        or branch.get("asymptotic_factor") != 64.0
        or branch.get("ir_safe") is not True
    ):
        raise RuntimeError("direct run has the wrong integration contract")
    tail = branch.get("p13_uv_tail_restoration")
    if not isinstance(tail, dict) or tail.get("enabled") is not True:
        raise RuntimeError("direct run lacks the P13 UV-tail restoration")


def compare_projection(
    analytic: dict[str, Any],
    direct: dict[str, Any],
    parameters: dict[str, float],
    sigma: float,
    relative_limit: float,
    sigma_limit: float,
) -> dict[str, Any]:
    analytic_total = polynomial_value(analytic["total"], parameters)
    direct_total = polynomial_value(direct["total"], parameters)
    delta = analytic_total - direct_total
    relative = delta / analytic_total
    topology_deltas = {
        topology: (
            polynomial_value(analytic[topology], parameters)
            - polynomial_value(direct[topology], parameters)
        )
        for topology in ("B222", "B321I", "B321II", "B411")
    }
    return {
        "analytic_total": analytic_total,
        "direct_total": direct_total,
        "delta_total": delta,
        "relative_total": relative,
        "delta_sigma_mean": delta / sigma,
        "topology_deltas": topology_deltas,
        "pass": bool(
            abs(relative) <= relative_limit
            and abs(delta / sigma) <= sigma_limit
        ),
    }


def nearest_error_scale(
    matrix: np.lib.npyio.NpzFile,
    triangle: tuple[float, float, float],
) -> tuple[int, list[float], float]:
    pair = np.asarray(triangle[:2], dtype=float)
    stored = np.asarray(matrix["b000_k"], dtype=float)
    index = int(np.argmin(np.sum((stored - pair) ** 2, axis=1)))
    sigma = float(matrix["fiducial_pre_B000_std"][index] / np.sqrt(500.0))
    return index, stored[index].tolist(), sigma


def current_hashes(
    driver: Path,
    source_root: Path,
    power: Path,
) -> dict[str, str]:
    hashes = {
        "eft_v2_r0_template_driver_binary": sha256(driver),
        "linear_power": sha256(power),
    }
    for name, relative in SOURCE_PATHS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
    return hashes


def main() -> None:
    args = parse_args()
    if not 1 <= args.omp_threads <= 28:
        raise ValueError("--omp-threads must be in 1..28")
    for path in (
        args.driver,
        args.power,
        args.matrix,
        args.contract,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.source_root.is_dir():
        raise NotADirectoryError(args.source_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    contract = strict_json(args.contract.read_text())
    registered_bias = {
        str(item["id"]) for item in contract["bias_parameters"]
    }
    if not set(SELECTED_BIAS).issubset(registered_bias):
        raise RuntimeError(
            "selected projection contains an unregistered bias parameter"
        )
    selected_bias_full = {
        name: SELECTED_BIAS.get(name, 0.0)
        for name in registered_bias
    }
    matter_bias_full = {
        name: MATTER_BIAS.get(name, 0.0)
        for name in registered_bias
    }
    gate = contract["gates"]
    thresholds = {
        "relative_total_max": float(gate["direct_dr_relative_total"]),
        "absolute_sigma_mean_max": float(
            gate["direct_dr_absolute_sigma_mean"]
        ),
        "analytic_frequency_step_sigma_mean_max": float(
            gate["direct_dr_absolute_sigma_mean"]
        ),
        "direct_grid_step_sigma_mean_max": float(
            gate["direct_dr_absolute_sigma_mean"]
        ),
    }
    hashes = current_hashes(args.driver, args.source_root, args.power)
    if hashes["linear_power"] != contract["source_hashes"]["linear_power_z1"]:
        raise RuntimeError("linear-power hash does not match contract")
    matrix_hash = sha256(args.matrix)
    if matrix_hash != contract["source_hashes"]["fiducial500_matrix"]:
        raise RuntimeError("matrix hash does not match contract")

    matrix = np.load(args.matrix, allow_pickle=False)
    case_reports = []
    raw_hashes = {}
    for case in CASES:
        label = case["label"]
        triangle = case["triangle"]
        fine_direct = case["direct"]
        commands = {
            "analytic": selected_command(
                args.driver,
                args.power,
                "analytic",
                triangle,
                case["frequency"],
                fine_direct,
            ),
            "analytic_previous": selected_command(
                args.driver,
                args.power,
                "analytic",
                triangle,
                case["previous_frequency"],
                fine_direct,
            ),
            "direct": selected_command(
                args.driver,
                args.power,
                "direct",
                triangle,
                case["frequency"],
                fine_direct,
            ),
            "direct_previous": selected_command(
                args.driver,
                args.power,
                "direct",
                triangle,
                case["frequency"],
                case["previous_direct"],
            ),
        }
        raw = {}
        for variant, command in commands.items():
            cache = args.work_dir / f"{label}_{variant}.json"
            raw[variant] = run_driver(
                command,
                cache,
                hashes["eft_v2_r0_template_driver_binary"],
                hashes["linear_power"],
                args.omp_threads,
                args.resume,
            )
            raw_hashes[f"{label}_{variant}"] = sha256(cache)

        validate_raw_result(
            raw["analytic"], "analytic", triangle, fine_direct
        )
        validate_raw_result(
            raw["analytic_previous"], "analytic", triangle, fine_direct
        )
        validate_raw_result(
            raw["direct"], "direct", triangle, fine_direct
        )
        validate_raw_result(
            raw["direct_previous"],
            "direct",
            triangle,
            case["previous_direct"],
        )
        analytic = templates(raw["analytic"], "analytic")
        analytic_previous = templates(
            raw["analytic_previous"], "analytic"
        )
        direct = templates(raw["direct"], "direct")
        direct_previous = templates(
            raw["direct_previous"], "direct"
        )
        index, nearest_pair, sigma = nearest_error_scale(
            matrix, triangle
        )
        selected = compare_projection(
            analytic,
            direct,
            selected_bias_full,
            sigma,
            thresholds["relative_total_max"],
            thresholds["absolute_sigma_mean_max"],
        )
        matter = compare_projection(
            analytic,
            direct,
            matter_bias_full,
            sigma,
            thresholds["relative_total_max"],
            thresholds["absolute_sigma_mean_max"],
        )
        current_analytic_selected = polynomial_value(
            analytic["total"], selected_bias_full
        )
        previous_analytic_selected = polynomial_value(
            analytic_previous["total"], selected_bias_full
        )
        current_direct_selected = polynomial_value(
            direct["total"], selected_bias_full
        )
        previous_direct_selected = polynomial_value(
            direct_previous["total"], selected_bias_full
        )
        analytic_step = (
            current_analytic_selected - previous_analytic_selected
        ) / sigma
        direct_step = (
            current_direct_selected - previous_direct_selected
        ) / sigma
        case_pass = bool(
            selected["pass"]
            and matter["pass"]
            and abs(analytic_step)
            <= thresholds["analytic_frequency_step_sigma_mean_max"]
            and abs(direct_step)
            <= thresholds["direct_grid_step_sigma_mean_max"]
        )
        case_reports.append(
            {
                "label": label,
                "triangle_h_mpc": list(triangle),
                "fftlog_frequency_count": case["frequency"],
                "previous_frequency_count": case["previous_frequency"],
                "analytic_frequency_step_sigma_mean": analytic_step,
                "direct_quadrature": list(case["direct"]),
                "direct_integration_nodes": int(
                    raw["direct"]["result"]["direct"][
                        "integration_nodes"
                    ]
                ),
                "previous_direct_quadrature": list(
                    case["previous_direct"]
                ),
                "direct_grid_step_sigma_mean": direct_step,
                "nearest_measurement_bin": index,
                "nearest_measurement_pair_h_mpc": nearest_pair,
                "sigma_mean": sigma,
                "selected": selected,
                "matter": matter,
                "p13_uv_tail_restoration": raw["direct"]["result"][
                    "direct"
                ].get("p13_uv_tail_restoration"),
                "pass": case_pass,
            }
        )

    certificate = {
        "schema": "marisa-b-eft-v2-r0-selected-dr-gate-v2",
        "status": (
            "pass"
            if all(case["pass"] for case in case_reports)
            else "fail"
        ),
        "scope": (
            "pre-reconstruction, real-space, Gaussian one-loop "
            "galaxy bispectrum"
        ),
        "gate": thresholds,
        "bias_projection": {
            "selected": {
                **SELECTED_BIAS,
                "all_other_registered_bias_parameters": 0.0,
            },
            "matter": {
                "b1": 1.0,
                "all_other_bias_parameters": 0.0,
            },
        },
        "fftlog": {
            "k_range_h_mpc": [1.0e-5, 10.0],
            "endpoint_inclusive_sampling": True,
            "interpolated_power_available": False,
            "mode_coefficient_relative_tolerance": 1.0e-12,
            "B321II_path": (
                "factorized renormalized source-power P13 with "
                "explicit qmax-to-infinity UV-tail restoration"
            ),
        },
        "direct": {
            "q_range_h_mpc": [1.0e-4, 30.0],
            "mu_ren_h_mpc": 0.3,
            "asymptotic_factor": 64.0,
            "ir_safe": True,
            "uv_subtract": True,
            "exact_opposite_pair_K4": True,
            "p13_uv_tail_restoration": True,
        },
        "measurement_error_scale": {
            "matrix": str(args.matrix),
            "matrix_sha256": matrix_hash,
            "convention": (
                "sqrt(diag(fiducial_pre_B000_cov / 500)); "
                "covariance of the 500-realization mean"
            ),
            "mapping": (
                "Each fixed triangle uses the nearest stored B000 radial "
                "pair only to set sigma_mean; it is not a shell projection."
            ),
        },
        "cases": case_reports,
        "input_hashes": hashes,
        "raw_run_hashes": raw_hashes,
        "interpretation": (
            "Selected-template direct-subtracted versus analytic FFTLog/DR "
            "gate for scalene, equilateral, and squeezed shapes, with both "
            "selected halo-bias and matter projections. Full shell-projected "
            "R0 statistics remain a separate gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            certificate,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "status": certificate["status"],
                "case_status": {
                    case["label"]: case["pass"]
                    for case in case_reports
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
