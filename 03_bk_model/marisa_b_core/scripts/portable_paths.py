#!/usr/bin/env python3
"""Backward-compatible path resolver for a standalone MARISA-B clone.

New code should use :mod:`marisa_b.paths`.  This module keeps the historical
script API available while removing the old assumption that the Git checkout
is nested three levels below a portable bundle.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPOSITORY_ROOT
INFERRED_PORTABLE_ROOT = REPOSITORY_ROOT


class PathConfigurationError(RuntimeError):
    """Raised when a required external input is unavailable."""


def _as_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def data_root(cli_value: str | Path | None = None) -> Path:
    """Return the external data root from CLI or ``MARISA_B_DATA_ROOT``.

    Falling back to the repository root is useful for source-only fixtures.
    Production manifests still require callers to set an explicit external
    root before accessing Quijote matrices or generated theory products.
    """

    if cli_value is not None and str(cli_value).strip():
        return _as_path(cli_value, Path.cwd())
    environment_value = os.environ.get("MARISA_B_DATA_ROOT", "").strip()
    if environment_value:
        return _as_path(environment_value, Path.cwd())
    return REPOSITORY_ROOT


def portable_root(cli_value: str | Path | None = None) -> Path:
    """Deprecated alias retained for historical scripts."""

    return data_root(cli_value)


def _load_config(root: Path) -> tuple[dict[str, Any], Path]:
    explicit = os.environ.get("MARISA_B_PATHS_CONFIG", "").strip()
    config_path = (
        _as_path(explicit, Path.cwd())
        if explicit
        else REPOSITORY_ROOT / "config" / "paths.local.toml"
    )
    if not config_path.is_file():
        return {}, config_path
    with config_path.open("rb") as stream:
        return tomllib.load(stream), config_path


def configured_path(
    *,
    cli_value: str | Path | None = None,
    environment: str,
    section: str,
    key: str,
    relative_default: str | Path | None,
    portable_root_value: str | Path | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve one input from CLI, environment, local config, then data root."""

    root = data_root(portable_root_value)
    config, config_path = _load_config(root)

    if cli_value is not None and str(cli_value).strip():
        return _as_path(cli_value, Path.cwd())

    environment_value = os.environ.get(environment, "").strip()
    if environment_value:
        return _as_path(environment_value, Path.cwd())

    section_values = config.get(section, {})
    if key in section_values:
        configured = str(section_values[key]).strip()
        if configured:
            return _as_path(configured, root)
        if required:
            raise PathConfigurationError(
                f"{section}.{key} is empty in {config_path}; set "
                f"--{key.replace('_', '-')} or {environment}"
            )
        return None

    if relative_default is not None:
        return _as_path(relative_default, root)
    if required:
        raise PathConfigurationError(
            f"missing required path {section}.{key}; set {environment} "
            f"or {config_path}"
        )
    return None


def contract_path(
    value: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Resolve a source contract path relative to the repository."""

    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (repository_root / path).resolve()
    )


def contract_display_path(
    path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
