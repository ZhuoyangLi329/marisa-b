#!/usr/bin/env python3
"""Verify the frozen and deterministic MARISA-B inference regression."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np


REGISTRY_RELATIVE = Path("tests/data/inference_regression_v1.json")
RUNNER_RELATIVE = Path(
    "scripts/production/run_prepost_recon_halo_fnl_mcmc_v1.py"
)
DATA_VERIFIER_RELATIVE = Path("scripts/verify_production_data.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(
    actual: float,
    expected: float,
    *,
    absolute: float,
    relative: float = 0.0,
    label: str,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=float(relative),
        abs_tol=float(absolute),
    ):
        raise AssertionError(
            f"{label}: got {actual!r}, expected {expected!r}, "
            f"rtol={relative}, atol={absolute}"
        )


def import_runner(repository_root: Path) -> Any:
    package_root = repository_root / "python"
    production_root = repository_root / "scripts/production"
    for path in (package_root, production_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner_path = repository_root / RUNNER_RELATIVE
    specification = importlib.util.spec_from_file_location(
        "_marisa_b_inference_regression_runner",
        runner_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import {runner_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def import_data_verifier(repository_root: Path) -> Any:
    path = repository_root / DATA_VERIFIER_RELATIVE
    specification = importlib.util.spec_from_file_location(
        "_marisa_b_production_data_verifier",
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def verify_data_manifest(
    repository_root: Path,
    data_root: Path,
    registry: dict[str, Any],
) -> dict[str, Any]:
    contract = registry["data_manifest"]
    manifest_path = repository_root / contract["relative_uri"]
    require(manifest_path.is_file(), f"missing data manifest: {manifest_path}")
    require(
        sha256(manifest_path) == contract["sha256"],
        "production data manifest SHA256 changed",
    )
    manifest = load_json(manifest_path)
    assets = manifest.get("assets")
    require(isinstance(assets, list), "data manifest assets must be a list")
    require(
        len(assets) == int(contract["expected_asset_count"]),
        "production data manifest must contain the registered nine assets",
    )
    verification = import_data_verifier(repository_root).verify(
        manifest_path,
        data_root,
    )
    require(
        verification["status"] == "pass",
        "production data verification failed: "
        + "; ".join(verification["failures"]),
    )
    require(
        manifest["policy"]["covariance_kind"] == "single_realization",
        "data manifest covariance is not single-realization",
    )
    return {
        "status": "pass",
        "manifest_sha256": contract["sha256"],
        "asset_count": len(verification["checked"]),
        "asset_ids": [
            row["id"] for row in verification["checked"]
        ],
    }


def load_frozen_artifacts(
    data_root: Path,
    registry: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {}
    for name, contract in registry["frozen_artifacts"].items():
        path = data_root / contract["relative_uri"]
        require(path.is_file(), f"missing frozen artifact: {path}")
        actual_sha = sha256(path)
        require(
            actual_sha == contract["sha256"],
            f"frozen artifact SHA256 changed: {name}",
        )
        document = load_json(path)
        require(
            document.get("schema") == contract["schema"],
            f"frozen artifact schema changed: {name}",
        )
        documents[name] = document
        report[name] = {
            "relative_uri": contract["relative_uri"],
            "sha256": actual_sha,
            "schema": contract["schema"],
        }
    return documents, report


def verify_frozen_profiler(
    summary: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    covariance = summary["covariance_contract"]
    require(
        covariance["division_by_number_of_realizations"] is False,
        "frozen profiler used covariance of the mean",
    )
    require(summary["status"] == "pass", "frozen profiler did not pass")
    tolerance = float(registry["tolerances"]["frozen_json_absolute"])
    for anchor in registry["anchors"]["pre_finite_png_profiler_centres"]:
        rows = [
            row
            for row in summary["profiles"]
            if row["tier"] == "coevolution"
            and row["response_model"] == "finite_halo_tree"
            and row["sample"] == anchor["sample"]
            and math.isclose(
                float(row["kmax_h_mpc"]),
                float(anchor["kmax_h_mpc"]),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ]
        require(len(rows) == 1, f"profiler anchor is not unique: {anchor}")
        row = rows[0]
        require(int(row["n_data"]) == int(anchor["n_data"]), "bin count changed")
        close(
            row["fhat"],
            anchor["fNL"],
            absolute=tolerance,
            label=f"profiler fNL {anchor['sample']} {anchor['kmax_h_mpc']}",
        )
        close(
            row["b1"],
            anchor["b1"],
            absolute=tolerance,
            label=f"profiler b1 {anchor['sample']} {anchor['kmax_h_mpc']}",
        )


def verify_frozen_mcmc(
    summary: dict[str, Any],
    state: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    scope = summary["scientific_scope"]
    require(
        scope["covariance"]
        == "covariance of one realization; never divided by 500",
        "frozen MCMC covariance contract changed",
    )
    require(
        "centre only" in scope["profiler_policy"],
        "frozen MCMC treats profiler width as posterior uncertainty",
    )
    require(summary["statistical_audit_pass"] is True, "MCMC audit failed")
    require(
        summary["scientific_release_gate"]["pass"] is False,
        "provisional post model was incorrectly promoted",
    )
    require(
        state["stages"]["validate/pre_coevolution_kmax0p14"]["pass"] is True
        and state["stages"]["validate/post_coevolution_kmax0p14"]["pass"] is True,
        "frozen likelihood closure failed",
    )
    tolerance = float(registry["tolerances"]["frozen_json_absolute"])
    for reconstruction, anchor in registry["anchors"]["mcmc_kmax0p14"].items():
        record = summary["posterior"]["coevolution"][reconstruction]["0.14"]
        fnl = record["fnl"]
        for name, source in (
            ("median_fNL", fnl["median"]),
            ("equal_tail_68_half_width", fnl["equal_tail_68_half_width"]),
            ("maximum_rank_rhat", record["maximum_rank_rhat"]),
            ("minimum_bulk_ess", record["minimum_bulk_ess"]),
            ("minimum_tail_ess", record["minimum_tail_ess"]),
        ):
            close(
                source,
                anchor[name],
                absolute=tolerance,
                label=f"MCMC {reconstruction} {name}",
            )
        np.testing.assert_allclose(
            fnl["equal_tail_68"],
            anchor["equal_tail_68"],
            rtol=0.0,
            atol=tolerance,
        )
        require(
            fnl["posterior_status"] == anchor["posterior_status"],
            f"MCMC posterior status changed: {reconstruction}",
        )


def fisher_reference(
    summary: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in summary["kmax_scan"]
        if math.isclose(
            float(row["kmax_h_mpc"]),
            float(anchor["kmax_h_mpc"]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
    ]
    require(len(rows) == 1, f"Fisher anchor is not unique: {anchor}")
    return rows[0][anchor["reconstruction"]]["tight_b1_fisher"]


def verify_frozen_fisher(
    summary: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    require(
        "pending" in summary["status"],
        "frozen Fisher summary lost its provisional status",
    )
    require(
        all(summary["implementation_checks"]["checks"].values()),
        "frozen Fisher implementation check failed",
    )
    tolerance = float(registry["tolerances"]["frozen_json_absolute"])
    for anchor in registry["anchors"]["fisher"]:
        reference = fisher_reference(summary, anchor)
        for name in (
            "fixed_nuisance_sigma_fNL",
            "all_nuisance_sigma_fNL",
            "b1_fixed_all_other_nuisance_sigma_fNL",
            "scaled_fisher_minimum_eigenvalue",
        ):
            close(
                reference[name],
                anchor[name],
                absolute=tolerance,
                label=(
                    f"frozen Fisher {anchor['reconstruction']} "
                    f"{anchor['kmax_h_mpc']} {name}"
                ),
            )


def verify_current_runner(
    repository_root: Path,
    data_root: Path,
    triangle_binary: Path,
    registry: dict[str, Any],
    temporary_root: Path,
) -> dict[str, Any]:
    require(
        triangle_binary.is_file(),
        f"missing triangle binary: {triangle_binary}",
    )
    require(
        os.access(triangle_binary, os.X_OK),
        f"triangle binary is not executable: {triangle_binary}",
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(temporary_root / "matplotlib"),
    )
    runner = import_runner(repository_root)
    original_paths_for = runner.pre_model.paths_for

    def paths_with_selected_binary(*args: Any, **kwargs: Any) -> Any:
        return replace(
            original_paths_for(*args, **kwargs),
            native_binary=triangle_binary,
        )

    runner.pre_model.paths_for = paths_with_selected_binary
    try:
        shared = runner.load_shared_inputs(
            repository_root,
            data_root,
            temporary_root,
        )
        audit = runner.audit_shared_inputs(temporary_root, shared)
        require(audit["pass"] is True, "current shared-input audit failed")
        require(
            audit["covariance_contract"][
                "covariance_of_single_realization"
            ]
            is True
            and audit["covariance_contract"][
                "covariance_divided_by_500"
            ]
            is False,
            "current runner covariance contract changed",
        )

        contexts: dict[tuple[str, float], Any] = {}
        likelihood_report: dict[str, Any] = {}
        map_report: dict[str, Any] = {}
        for reconstruction in ("pre", "post"):
            context = runner.build_context(
                shared,
                reconstruction,
                "coevolution",
                0.14,
            )
            contexts[(reconstruction, 0.14)] = context
            validation = runner.validate_context(
                context,
                np.random.default_rng(20260727),
                random_points=4,
            )
            require(
                validation["pass"] is True,
                f"current likelihood closure failed: {reconstruction}",
            )
            tolerance = registry["tolerances"]
            require(
                validation["maximum_affine_error_sigma_single"]
                < tolerance["maximum_affine_error_sigma_single"],
                f"affine closure failed: {reconstruction}",
            )
            require(
                validation["maximum_direct_objective_delta"]
                < tolerance["maximum_direct_objective_delta"],
                f"direct objective closure failed: {reconstruction}",
            )
            require(
                validation["logdet_spread_across_random_points"]
                > tolerance["minimum_logdet_spread"],
                f"logdet negative control failed: {reconstruction}",
            )
            reference = registry["anchors"]["current_joint_map"][
                f"{reconstruction}_kmax0p14"
            ]
            result = validation["profile_centre_only"]
            require(result["valid"] is True, "current MAP is invalid")
            require(
                list(result["theta_names"]) == reference["theta_names"],
                f"current MAP coordinates changed: {reconstruction}",
            )
            for index, (actual, expected) in enumerate(
                zip(result["theta"], reference["theta"])
            ):
                absolute = (
                    tolerance["map_fNL_absolute"]
                    if index == 0
                    else tolerance["map_nuisance_absolute"]
                )
                close(
                    actual,
                    expected,
                    absolute=absolute,
                    label=f"current MAP {reconstruction} theta[{index}]",
                )
            close(
                result["objective"],
                reference["objective"],
                absolute=tolerance["map_objective_absolute"],
                label=f"current MAP {reconstruction} objective",
            )
            likelihood_report[reconstruction] = {
                "pass": True,
                "maximum_affine_error_sigma_single": validation[
                    "maximum_affine_error_sigma_single"
                ],
                "maximum_direct_objective_delta": validation[
                    "maximum_direct_objective_delta"
                ],
                "logdet_spread": validation[
                    "logdet_spread_across_random_points"
                ],
            }
            map_report[reconstruction] = {
                "theta_names": list(result["theta_names"]),
                "theta": [float(value) for value in result["theta"]],
                "objective": float(result["objective"]),
            }

        calibration = runner.b1_calibration_contract(data_root)
        b1_mean = float(calibration["primary"]["b1"])
        b1_sigma = float(calibration["primary"]["b1_sigma"])
        fisher_report = []
        for anchor in registry["anchors"]["fisher"]:
            key = (anchor["reconstruction"], float(anchor["kmax_h_mpc"]))
            context = contexts.get(key)
            if context is None:
                context = runner.build_context(
                    shared,
                    anchor["reconstruction"],
                    "coevolution",
                    float(anchor["kmax_h_mpc"]),
                )
                contexts[key] = context
            tight = runner.context_with_b1_prior(
                context,
                mean=b1_mean,
                sigma=b1_sigma,
            )
            _gaussian_map, conditional = runner.gaussian_nuisance_map(tight)
            jacobian, derivative = runner.model_jacobian_at_zero(
                tight,
                conditional.nuisance,
            )
            require(
                max(
                    derivative["maximum_relative_step_closure"],
                    derivative["registered_linear_total_relative_closure"],
                )
                < 1.0e-10,
                f"Fisher response closure failed: {key}",
            )
            result, _matrix = runner.fisher_from_jacobian(
                tight,
                tight,
                jacobian,
            )
            for name in (
                "fixed_nuisance_sigma_fNL",
                "all_nuisance_sigma_fNL",
                "b1_fixed_all_other_nuisance_sigma_fNL",
                "scaled_fisher_minimum_eigenvalue",
            ):
                close(
                    result[name],
                    anchor[name],
                    absolute=registry["tolerances"]["fisher_absolute"],
                    relative=registry["tolerances"]["fisher_relative"],
                    label=(
                        f"current Fisher {anchor['reconstruction']} "
                        f"{anchor['kmax_h_mpc']} {name}"
                    ),
                )
            fisher_report.append(
                {
                    "reconstruction": anchor["reconstruction"],
                    "kmax_h_mpc": anchor["kmax_h_mpc"],
                    "fixed_nuisance_sigma_fNL": float(
                        result["fixed_nuisance_sigma_fNL"]
                    ),
                    "all_nuisance_sigma_fNL": float(
                        result["all_nuisance_sigma_fNL"]
                    ),
                    "scaled_fisher_minimum_eigenvalue": float(
                        result["scaled_fisher_minimum_eigenvalue"]
                    ),
                }
            )
    finally:
        runner.pre_model.paths_for = original_paths_for

    return {
        "shared_input_audit": {
            "pass": True,
            "matrix_shape": audit["matrix_shape"],
            "mask_counts": {
                key: int(value["count"])
                for key, value in audit["mask_contract"].items()
            },
        },
        "likelihood_closure": likelihood_report,
        "joint_map": map_report,
        "fisher": fisher_report,
        "sampler_smoke": registry["sampler_smoke"],
    }


def parse_args() -> argparse.Namespace:
    default_repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repository,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(os.environ["MARISA_B_DATA_ROOT"])
            if "MARISA_B_DATA_ROOT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--triangle-binary",
        type=Path,
        default=default_repository / "build/halo_v1/marisa_b_triangle",
    )
    arguments = parser.parse_args()
    if arguments.data_root is None:
        parser.error(
            "--data-root is required when MARISA_B_DATA_ROOT is unset"
        )
    return arguments


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repository_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    triangle_binary = args.triangle_binary.expanduser().resolve()
    registry_path = repository_root / REGISTRY_RELATIVE
    registry = load_json(registry_path)
    require(
        registry.get("schema")
        == "marisa-b-inference-regression-registry-v1",
        "unsupported inference-regression registry",
    )
    data_report = verify_data_manifest(
        repository_root,
        data_root,
        registry,
    )
    frozen, frozen_report = load_frozen_artifacts(data_root, registry)
    verify_frozen_profiler(
        frozen["pre_finite_png_profiler"],
        registry,
    )
    verify_frozen_mcmc(
        frozen["prepost_mcmc_summary"],
        frozen["prepost_mcmc_state"],
        registry,
    )
    verify_frozen_fisher(
        frozen["prepost_fisher_summary"],
        registry,
    )
    with tempfile.TemporaryDirectory(
        prefix="marisa_b_inference_regression_"
    ) as temporary:
        current = verify_current_runner(
            repository_root,
            data_root,
            triangle_binary,
            registry,
            Path(temporary),
        )
    return {
        "schema": "marisa-b-inference-regression-result-v1",
        "status": "pass",
        "elapsed_seconds": time.perf_counter() - started,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256(registry_path),
        },
        "production_data": data_report,
        "frozen_artifacts": frozen_report,
        "current_runner": current,
        "scientific_limits": registry["scientific_limits"],
    }


def main() -> int:
    try:
        result = run(parse_args())
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "marisa-b-inference-regression-result-v1",
                    "status": "fail",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
