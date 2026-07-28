"""Machine-readable inference and external-data contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]


def test_inference_example_validates_and_uses_single_covariance() -> None:
    schema = json.loads(
        (ROOT / "schemas/inference_run.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (ROOT / "configs/inference_prepost_example.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(example, schema)
    assert example["statistics"]["covariance_kind"] == "single_realization"
    assert example["parameters"]["b1"]["mode"] == "free"
    assert example["parameters"]["fNL"]["lower"] == -150.0
    assert example["parameters"]["fNL"]["upper"] == 150.0


def test_external_manifest_has_unique_relative_hash_bound_assets() -> None:
    manifest = json.loads(
        (ROOT / "configs/production_data_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = manifest["assets"]
    assert manifest["policy"]["tracked_in_source_git"] is False
    assert manifest["policy"]["covariance_kind"] == "single_realization"
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["relative_uri"] for row in rows}) == len(rows)
    for row in rows:
        path = Path(row["relative_uri"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert len(row["sha256"]) == 64
        assert row["bytes"] > 0
