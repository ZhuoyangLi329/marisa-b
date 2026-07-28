#!/usr/bin/env python3
"""Verify the standalone MARISA-B source, license, and Git boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REACT = ROOT / "external" / "ACTio-ReACTio"
EXPECTED_REACT_COMMIT = "a36dda02db6b537a4ddcc7bf044c446c10a58373"
EXPECTED_REACT_TREE_HASH = (
    "d2f3b86ac85016aa1c386dd91299284ea72a83ae0480652be02f60c15655d48d"
)
EXPECTED_NATIVE_HASHES = {
    "src/marisa_b/marisa_b_native.cpp": (
        "d3ddf662ca5bb8a36c8126dd72dfb4c19b6b61c87ec6b65047c4c1361bc112b2"
    ),
    "src/marisa_b/marisa_b_native.h": (
        "28743d88b165f74edf2708d995a066336841c3c6a0ea72364b4df985f80eb184"
    ),
}
EXPECTED_RELEASE_HASHES = {
    "LICENSE": (
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
    ),
    "external/ACTio-ReACTio/LICENSE": (
        "b8b9c247a9d453f0669a8c6104b08293b31851a6cdf0f0dcbe08b5bf6dcaf5a1"
    ),
    "external/ACTio-ReACTio/reactions/COPYING": (
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
    ),
    "src/eft_v2/generated_b222_b321i_tables.h": (
        "6cf00d544606acbc7f8317bb84180b65a841a7175e5e513d673265e7508bbcc5"
    ),
    "src/eft_v2/generated_b411_table.h": (
        "8e08e579fec06ab19b6bb9d8a48a4aef5db3173c68d42afd1f2bbae8638c46c6"
    ),
    "scripts/generate_eft_v2_b411_table.py": (
        "42618405f57d5816b320acda48ec4d1d428549ec82eac040c6b46fc78b7a6ef2"
    ),
}
FORBIDDEN_TRACKED_PATTERNS = (
    re.compile(
        r"(^|/)(__pycache__|build|runs|analysis|figures|manifests|log|history)/"
    ),
    re.compile(
        r"\.(pyc|o|a|so|npz|npy|h5|hdf5|jsonl|pdf|png|out|err)$",
        re.IGNORECASE,
    ),
)
FORBIDDEN_CURRENT_TREE_TEXT = re.compile(
    "|".join(
        re.escape(value)
        for value in (
            "/home/" + "marisa",
            "/p" + "scratch/",
            "/global/" + "homes/",
            "history/" + "worktrees",
            "PORTABLE_" + "DM_DIR",
        )
    )
)
SECRET_TEXT = re.compile(
    r"(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|sk-[A-Za-z0-9]{32,})"
)
SHA256_TEXT = re.compile(r"^[0-9a-f]{64}$")
COMMIT_TEXT = re.compile(r"^[0-9a-f]{40}$")
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def git(*arguments: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=check,
        text=True,
        capture_output=True,
    )
    return result.stdout


def git_bytes(*arguments: str, cwd: Path = ROOT) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def tracked_files() -> list[str]:
    return [
        entry
        for entry in git("ls-files", "-z").split("\0")
        if entry
    ]


def text_files(paths: Iterable[str]) -> Iterable[tuple[str, str]]:
    for relative in paths:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield relative, payload


def verify(allow_dirty: bool) -> dict[str, object]:
    failures: list[str] = []
    tracked = tracked_files()

    for required in (
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "provenance" / "third_party_sources.tsv",
        ROOT / "provenance" / "MARISA_B_IMPORT.tsv",
    ):
        if not required.is_file():
            failures.append(f"missing release metadata: {required.relative_to(ROOT)}")

    native_sources = [
        path
        for path in tracked
        if path.endswith("marisa_b_native.cpp")
    ]
    if native_sources != ["src/marisa_b/marisa_b_native.cpp"]:
        failures.append(
            "expected exactly one canonical marisa_b_native.cpp; got "
            + repr(native_sources)
        )
    for relative, expected in EXPECTED_NATIVE_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"canonical native hash mismatch: {relative}")
    for relative, expected in EXPECTED_RELEASE_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"release input hash mismatch: {relative}")

    for relative in tracked:
        if any(pattern.search(relative) for pattern in FORBIDDEN_TRACKED_PATTERNS):
            failures.append(f"forbidden generated artifact is tracked: {relative}")
        path = ROOT / relative
        if (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > MAX_TRACKED_FILE_BYTES
        ):
            failures.append(
                "tracked source file exceeds 5 MiB release boundary: "
                f"{relative}"
            )

    for relative, payload in text_files(tracked):
        match = FORBIDDEN_CURRENT_TREE_TEXT.search(payload)
        if match:
            failures.append(
                f"machine-local or historical path in {relative}: {match.group(0)}"
            )
        if SECRET_TEXT.search(payload):
            failures.append(f"credential-like text in tracked file: {relative}")

    if not REACT.is_dir():
        failures.append("ACTio-ReACTio submodule is not initialized")
    else:
        try:
            react_head = git("rev-parse", "HEAD", cwd=REACT).strip()
        except subprocess.CalledProcessError:
            react_head = ""
        if react_head != EXPECTED_REACT_COMMIT:
            failures.append(
                f"ACTio-ReACTio commit mismatch: {react_head or '<missing>'}"
            )
        try:
            react_tree_hash = hashlib.sha256(
                git_bytes("archive", "HEAD", cwd=REACT)
            ).hexdigest()
        except subprocess.CalledProcessError:
            react_tree_hash = ""
        if react_tree_hash != EXPECTED_REACT_TREE_HASH:
            failures.append(
                "ACTio-ReACTio archived tree hash mismatch: "
                f"{react_tree_hash or '<missing>'}"
            )
        react_status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            cwd=REACT,
            check=False,
        )
        if react_status:
            failures.append("ACTio-ReACTio submodule is dirty")

    ledger = ROOT / "provenance" / "third_party_sources.tsv"
    third_party_entries = 0
    if ledger.is_file():
        rows = read_tsv(ledger)
        third_party_entries = len(rows)
        required_fields = {
            "project",
            "upstream_url",
            "upstream_identity",
            "license",
            "compiled_files",
        }
        for index, row in enumerate(rows, start=2):
            missing = sorted(
                field for field in required_fields if not row.get(field)
            )
            if missing:
                failures.append(
                    f"incomplete third-party ledger row {index}: {missing}"
                )

    import_ledger = ROOT / "provenance" / "MARISA_B_IMPORT.tsv"
    first_party_entries = 0
    if import_ledger.is_file():
        rows = read_tsv(import_ledger)
        first_party_entries = len(rows)
        required_fields = {
            "destination_path",
            "source_snapshot",
            "source_path",
            "source_sha256",
            "imported_blob_sha256",
            "import_identity",
            "imported_utc",
            "import_commit",
            "current_role",
            "notes",
        }
        seen_destinations: set[str] = set()
        for index, row in enumerate(rows, start=2):
            missing = sorted(
                field for field in required_fields if not row.get(field)
            )
            if missing:
                failures.append(
                    f"incomplete first-party ledger row {index}: {missing}"
                )
                continue
            destination = row["destination_path"]
            if destination in seen_destinations:
                failures.append(
                    f"duplicate first-party destination in row {index}: "
                    f"{destination}"
                )
            seen_destinations.add(destination)
            destination_path = Path(destination)
            if destination_path.is_absolute() or ".." in destination_path.parts:
                failures.append(
                    f"unsafe first-party destination in row {index}: "
                    f"{destination}"
                )
                continue
            if not (ROOT / destination_path).is_file():
                failures.append(
                    f"missing first-party destination in row {index}: "
                    f"{destination}"
                )
            source_hash = row["source_sha256"]
            imported_hash = row["imported_blob_sha256"]
            commit = row["import_commit"]
            if not SHA256_TEXT.fullmatch(source_hash):
                failures.append(
                    f"invalid source SHA256 in first-party row {index}"
                )
            if not SHA256_TEXT.fullmatch(imported_hash):
                failures.append(
                    f"invalid imported SHA256 in first-party row {index}"
                )
            if not COMMIT_TEXT.fullmatch(commit):
                failures.append(
                    f"invalid import commit in first-party row {index}: "
                    f"{commit}"
                )
                continue
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    "HEAD",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ancestor.returncode != 0:
                failures.append(
                    f"import commit is not an ancestor in row {index}: "
                    f"{commit}"
                )
                continue
            try:
                imported_blob = git_bytes(
                    "show",
                    f"{commit}:{destination}",
                )
            except subprocess.CalledProcessError:
                failures.append(
                    f"destination absent at import commit in row {index}: "
                    f"{commit}:{destination}"
                )
                continue
            actual_imported_hash = hashlib.sha256(imported_blob).hexdigest()
            if actual_imported_hash != imported_hash:
                failures.append(
                    f"imported blob hash mismatch in row {index}: "
                    f"{destination}"
                )

    status_text = git("status", "--porcelain=v1", "--untracked-files=all")
    if status_text and not allow_dirty:
        failures.append("Git worktree is not clean")

    head = git("rev-parse", "HEAD").strip()
    return {
        "status": "pass" if not failures else "fail",
        "root": str(ROOT),
        "git_head": head,
        "git_clean": not bool(status_text),
        "allow_dirty": allow_dirty,
        "tracked_files_checked": len(tracked),
        "third_party_entries_checked": third_party_entries,
        "first_party_entries_checked": first_party_entries,
        "react_commit": (
            git("rev-parse", "HEAD", cwd=REACT).strip()
            if REACT.is_dir()
            else None
        ),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    report = verify(arguments.allow_dirty)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
