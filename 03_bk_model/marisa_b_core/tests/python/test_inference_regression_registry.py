"""Inference regression registry semantics without external Quijote data."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPOSITORY_ROOT / "tests/data/inference_regression_v1.json"
)


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_schema_and_frozen_artifact_identities() -> None:
    registry = load_registry()
    assert registry["schema"] == (
        "marisa-b-inference-regression-registry-v1"
    )
    assert registry["schema_version"] == 1
    assert registry["data_manifest"]["expected_asset_count"] == 9
    artifacts = registry["frozen_artifacts"]
    assert set(artifacts) == {
        "pre_finite_png_profiler",
        "prepost_mcmc_summary",
        "prepost_mcmc_state",
        "prepost_fisher_summary",
    }
    for artifact in artifacts.values():
        assert not Path(artifact["relative_uri"]).is_absolute()
        assert len(artifact["sha256"]) == 64
        int(artifact["sha256"], 16)


def test_registry_requires_single_realization_covariance() -> None:
    covariance = load_registry()["covariance_contract"]
    assert covariance["kind"] == "single_realization"
    assert covariance["realization_count"] == 500
    assert covariance["division_by_realization_count"] is False
    assert covariance["ensemble_mean_role"] == "central_vector_only"
    assert covariance["precision_correction"] == "hartlap"


def test_profiler_carries_centres_but_no_posterior_width() -> None:
    registry = load_registry()
    policy = registry["profiler_policy"]
    assert policy["role"] == "joint_map_centre_only"
    assert policy["reported_uncertainty"] is False
    assert (
        policy["posterior_uncertainty_source"]
        == "converged_marginalized_mcmc"
    )
    profiler_anchors = registry["anchors"][
        "pre_finite_png_profiler_centres"
    ]
    assert profiler_anchors
    forbidden = ("error", "interval", "sigma", "width", "posterior")
    for anchor in profiler_anchors:
        assert {"kmax_h_mpc", "sample", "n_data", "fNL", "b1"} == set(
            anchor
        )
        assert not any(
            token in key.lower()
            for key in anchor
            for token in forbidden
        )
    for anchor in registry["anchors"]["current_joint_map"].values():
        assert set(anchor) == {
            "reconstruction",
            "tier",
            "kmax_h_mpc",
            "theta_names",
            "theta",
            "objective",
        }


def test_mcmc_fisher_and_scientific_boundaries_are_explicit() -> None:
    registry = load_registry()
    mcmc = registry["anchors"]["mcmc_kmax0p14"]
    assert set(mcmc) == {"pre", "post"}
    assert mcmc["pre"]["equal_tail_68_half_width"] > 0.0
    assert mcmc["post"]["equal_tail_68_half_width"] > 0.0
    assert registry["sampler_smoke"]["status"] == "skipped"
    assert "not a convergence test" in registry["sampler_smoke"]["reason"]
    limits = " ".join(registry["scientific_limits"]).lower()
    assert "provisional_unvalidated_numerical_convergence" in limits
    assert "quarantined" in limits
    assert "profiler" in limits and "no posterior uncertainty" in limits
    assert "fisher" in limits and "not marginalized posterior" in limits
