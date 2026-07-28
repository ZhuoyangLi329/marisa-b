#!/usr/bin/env python3
"""Validate and merge disjoint EFT-v2 R0 JSONL template shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--allow-interrupted-prefix",
        action="store_true",
        help=(
            "With --allow-partial, accept a contiguous stored prefix of a "
            "declared shard range and normalize its output range."
        ),
    )
    parser.add_argument("--expected-count", type=int, default=120)
    parser.add_argument("--source-power", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--source-file", action="append", default=[], metavar="LABEL=PATH",
        help="Record a labeled implementation-source SHA256 in the merged header.",
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        if mode == "w":
            raw = path.open("wb")
            compressed = gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0,
            )
            return io.TextIOWrapper(compressed, encoding="utf-8")
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def require_finite_json(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location}: non-finite floating-point value")


def read_part(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open_text(path, "r") as stream:
        rows = []
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=reject_nonstandard_constant,
                    object_pairs_hook=reject_duplicate_keys,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: top-level JSON record must be an object"
                )
            require_finite_json(row, f"{path}:{line_number}")
            rows.append(row)
    headers = [row for row in rows if row.get("record") == "header"]
    bins = [row for row in rows if row.get("record") == "bin"]
    if len(headers) != 1 or len(headers) + len(bins) != len(rows):
        raise ValueError(f"{path}: expected one header followed only by bin records")
    return headers[0], bins


def validate_part_indices(
    path: Path,
    header: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_count: int,
    allow_interrupted_prefix: bool = False,
) -> None:
    if not rows:
        raise ValueError(f"{path}: shard contains no bin records")
    claimed_range = header.get("bin_range")
    if (
        not isinstance(claimed_range, list)
        or len(claimed_range) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in claimed_range
        )
    ):
        raise ValueError(f"{path}: header bin_range must contain two integers")
    start, stop = claimed_range
    if not (0 <= start < stop <= expected_count):
        raise ValueError(
            f"{path}: header bin_range {claimed_range} lies outside "
            f"0..{expected_count - 1}"
        )
    indices: list[int] = []
    for row_number, row in enumerate(rows, 1):
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"{path}: bin record {row_number} has a non-integer index")
        if not (start <= index < stop):
            raise ValueError(
                f"{path}: bin index {index} lies outside header range {claimed_range}"
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{path}: duplicate bin indices within one shard")
    claimed_indices = header.get("bin_indices")
    if claimed_indices is None:
        expected_indices = list(range(start, stop))
        if indices != expected_indices:
            stored_prefix = list(range(start, start + len(indices)))
            if not allow_interrupted_prefix or indices != stored_prefix:
                raise ValueError(
                    f"{path}: header claims contiguous bins {start}..{stop - 1}, "
                    f"but records are {indices}"
                )
    else:
        if (
            not isinstance(claimed_indices, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in claimed_indices
            )
            or claimed_indices != sorted(set(claimed_indices))
        ):
            raise ValueError(
                f"{path}: header bin_indices must be sorted unique integers"
            )
        if claimed_indices != indices:
            raise ValueError(
                f"{path}: header bin_indices do not match the stored records"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparable_header(header: dict[str, Any]) -> dict[str, Any]:
    result = dict(header)
    result.pop("bin_range", None)
    result.pop("bin_indices", None)
    result.pop("merged_parts", None)
    return result


def main() -> None:
    args = parse_args()
    if args.expected_count < 1:
        raise ValueError("--expected-count must be positive")
    if args.allow_interrupted_prefix and not args.allow_partial:
        raise ValueError(
            "--allow-interrupted-prefix requires --allow-partial"
        )
    reference: dict[str, Any] | None = None
    bins: dict[int, dict[str, Any]] = {}
    part_provenance: list[dict[str, Any]] = []
    for path in args.inputs:
        header, rows = read_part(path)
        validate_part_indices(
            path,
            header,
            rows,
            args.expected_count,
            allow_interrupted_prefix=args.allow_interrupted_prefix,
        )
        row_indices = [int(row["index"]) for row in rows]
        actual_range = [row_indices[0], row_indices[-1] + 1]
        provenance = {
            "bin_range": actual_range,
            "sha256": sha256(path),
        }
        declared_range = list(header.get("bin_range", []))
        if declared_range != actual_range:
            provenance["declared_bin_range"] = declared_range
            provenance["interrupted_prefix"] = True
        part_provenance.append(provenance)
        if reference is None:
            reference = header
        elif comparable_header(header) != comparable_header(reference):
            raise ValueError(f"{path}: shard header is incompatible with the first input")
        for row in rows:
            index = int(row["index"])
            if index in bins:
                raise ValueError(f"duplicate shell-bin index {index}")
            bins[index] = row
    assert reference is not None
    indices = sorted(bins)
    if not args.allow_partial and indices != list(range(args.expected_count)):
        raise ValueError(
            f"complete merge requires bins 0..{args.expected_count - 1}; "
            f"got {len(indices)} bins"
        )
    reference = dict(reference)
    reference["bin_range"] = [indices[0], indices[-1] + 1]
    if indices != list(range(indices[0], indices[-1] + 1)):
        reference["bin_indices"] = indices
    else:
        reference.pop("bin_indices", None)
    reference["merged_parts"] = sorted(
        part_provenance, key=lambda row: (row["bin_range"], row["sha256"]),
    )
    source_hashes = dict(reference.get("source_hashes", {}))
    if args.source_power is not None:
        source_hashes["linear_power"] = sha256(args.source_power)
    if args.contract is not None:
        source_hashes["eft_v2_contract"] = sha256(args.contract)
    for specification in args.source_file:
        if "=" not in specification:
            raise ValueError(
                f"invalid --source-file {specification!r}; expected LABEL=PATH"
            )
        label, filename = specification.split("=", 1)
        if not label or label in source_hashes:
            raise ValueError(f"duplicate or empty source-file label {label!r}")
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f"source file does not exist: {path}")
        source_hashes[label] = sha256(path)
    if source_hashes:
        reference["source_hashes"] = dict(sorted(source_hashes.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_text(args.output, "w") as stream:
        stream.write(
            json.dumps(
                reference, allow_nan=False, separators=(",", ":"), sort_keys=True,
            )
            + "\n"
        )
        for index in indices:
            stream.write(
                json.dumps(
                    bins[index],
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        json.dumps(
            {"output": str(args.output), "bin_count": len(indices)},
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
