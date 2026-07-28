#!/usr/bin/env python3
"""Verify an external MARISA-B production-data bundle against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "production_data_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"unsupported data-manifest version in {path}")
    if not isinstance(payload.get("assets"), list):
        raise ValueError(f"{path} has no asset list")
    return payload


def verify(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    checked: list[dict[str, Any]] = []
    for row in manifest["assets"]:
        identifier = str(row.get("id", ""))
        relative_uri = Path(str(row.get("relative_uri", "")))
        if not identifier or not str(relative_uri) or relative_uri.is_absolute():
            failures.append(f"invalid asset declaration: {identifier!r}")
            continue
        resolved = (data_root / relative_uri).resolve()
        try:
            resolved.relative_to(data_root.resolve())
        except ValueError:
            failures.append(f"asset escapes data root: {identifier}")
            continue
        if not resolved.is_file():
            failures.append(f"missing asset: {identifier} ({relative_uri})")
            continue
        actual_bytes = resolved.stat().st_size
        actual_sha256 = sha256(resolved)
        if actual_bytes != int(row["bytes"]):
            failures.append(
                f"size mismatch: {identifier}: {actual_bytes} != {row['bytes']}"
            )
        if actual_sha256 != row["sha256"]:
            failures.append(f"SHA256 mismatch: {identifier}")
        checked.append(
            {
                "id": identifier,
                "relative_uri": relative_uri.as_posix(),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    return {
        "status": "pass" if not failures else "fail",
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "assets_expected": len(manifest["assets"]),
        "assets_checked": len(checked),
        "checked": checked,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External bundle root; otherwise use MARISA_B_DATA_ROOT.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    root = arguments.data_root
    if root is None:
        configured = os.environ.get("MARISA_B_DATA_ROOT", "").strip()
        if not configured:
            raise SystemExit(
                "set --data-root or MARISA_B_DATA_ROOT; production data are "
                "intentionally not stored in source Git"
            )
        root = Path(configured)
    report = verify(arguments.manifest.resolve(), root.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
