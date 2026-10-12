"""仓库与数据根目录解析的 synthetic 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

import marisa_b.paths as paths


def test_repo_root_is_default_data_root(tmp_path: Path) -> None:
    resolved = paths.resolve_repository_paths(
        repo_root=tmp_path,
        environment={},
    )
    assert resolved.repo_root == tmp_path.resolve()
    assert resolved.data_root == tmp_path.resolve()


def test_explicit_data_root_overrides_environment(tmp_path: Path) -> None:
    resolved = paths.resolve_repository_paths(
        repo_root=tmp_path,
        data_root="explicit-data",
        environment={
            paths.DATA_ROOT_ENVIRONMENT_VARIABLE: "environment-data"
        },
    )
    assert resolved.data_root == (
        tmp_path / "explicit-data"
    ).resolve()


def test_environment_data_root_and_safe_data_path(tmp_path: Path) -> None:
    resolved = paths.resolve_repository_paths(
        repo_root=tmp_path,
        environment={
            paths.DATA_ROOT_ENVIRONMENT_VARIABLE: "external-data"
        },
    )
    assert resolved.data_root == (
        tmp_path / "external-data"
    ).resolve()
    assert resolved.data_path("templates", "pre.jsonl") == (
        tmp_path / "external-data/templates/pre.jsonl"
    ).resolve()
    with pytest.raises(ValueError):
        resolved.data_path("..", "escape.dat")
    with pytest.raises(ValueError):
        resolved.data_path(Path("/tmp/absolute-escape.dat"))


def test_implicit_repo_root_comes_from_package_location() -> None:
    expected = Path(paths.__file__).resolve().parents[2]
    resolved = paths.resolve_repository_paths(environment={})
    assert resolved.repo_root == expected
    assert "history" not in resolved.repo_root.parts
