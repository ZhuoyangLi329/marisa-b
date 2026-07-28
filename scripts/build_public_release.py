#!/usr/bin/env python3
"""Create a two-commit, history-clean MARISA-B release repository and bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import inspect
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_BRANCH = "release/v0.1.0-rc1"
DEFAULT_TAG = "v0.1.0-rc1"
IMPORT_LEDGER = Path("provenance") / "MARISA_B_IMPORT.tsv"


def run(
    repository: Path,
    *arguments: str,
    capture: bool = True,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_git_archive(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"unsafe Git archive member: {member.name}")
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(destination, filter="fully_trusted")
        else:
            archive.extractall(destination)


def gitlinks(repository: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for line in run(repository, "ls-files", "--stage").splitlines():
        metadata, path = line.split("\t", maxsplit=1)
        mode, object_id, _stage = metadata.split()
        if mode == "160000":
            result.append((path, object_id))
    return result


def commit(
    repository: Path,
    message: str,
    *,
    author_name: str,
    author_email: str,
) -> str:
    run(
        repository,
        "-c",
        f"user.name={author_name}",
        "-c",
        f"user.email={author_email}",
        "commit",
        "-m",
        message,
    )
    return run(repository, "rev-parse", "HEAD").strip()


def rewrite_import_ledger(repository: Path, import_commit: str) -> int:
    path = repository / IMPORT_LEDGER
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("first-party import ledger has no header")
    for row in rows:
        destination = repository / row["destination_path"]
        if not destination.is_file():
            raise FileNotFoundError(destination)
        row["import_commit"] = import_commit
        row["imported_blob_sha256"] = sha256(destination)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    source = arguments.source.resolve()
    destination = arguments.destination.resolve()
    bundle = arguments.bundle.resolve()
    if destination.exists():
        raise FileExistsError(f"release destination already exists: {destination}")
    if bundle.exists():
        raise FileExistsError(f"release bundle already exists: {bundle}")
    if run(source, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source repository must be clean")
    source_head = run(source, "rev-parse", "HEAD").strip()
    source_gitlinks = gitlinks(source)
    archive = git_bytes(source, "archive", "--format=tar", source_head)
    safe_extract_git_archive(archive, destination)

    subprocess.run(
        ["git", "init", "-b", arguments.branch, str(destination)],
        check=True,
        text=True,
        capture_output=True,
    )
    run(
        destination,
        "add",
        "-A",
        "--",
        ".",
        f":(exclude){IMPORT_LEDGER.as_posix()}",
    )
    for path, object_id in source_gitlinks:
        run(
            destination,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{object_id},{path}",
        )
    source_commit = commit(
        destination,
        "publish normalized MARISA-B source tree",
        author_name=arguments.author_name,
        author_email=arguments.author_email,
    )

    imported_rows = rewrite_import_ledger(destination, source_commit)
    run(destination, "add", str(IMPORT_LEDGER))
    provenance_commit = commit(
        destination,
        "register first-party source provenance",
        author_name=arguments.author_name,
        author_email=arguments.author_email,
    )

    run(
        destination,
        "submodule",
        "update",
        "--init",
        "--recursive",
        capture=False,
    )
    verifier = destination / "scripts" / "verify_repository_boundary.py"
    subprocess.run(
        [sys.executable, str(verifier)],
        cwd=destination,
        check=True,
        text=True,
        capture_output=True,
    )
    run(
        destination,
        "-c",
        f"user.name={arguments.author_name}",
        "-c",
        f"user.email={arguments.author_email}",
        "tag",
        "-a",
        arguments.tag,
        "-m",
        "MARISA-B v0.1.0 release candidate 1",
    )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    run(
        destination,
        "bundle",
        "create",
        str(bundle),
        f"refs/heads/{arguments.branch}",
        f"refs/tags/{arguments.tag}",
    )
    bundle_verification = run(destination, "bundle", "verify", str(bundle))
    return {
        "status": "pass",
        "source_head": source_head,
        "release_repository": str(destination),
        "release_branch": arguments.branch,
        "release_tag": arguments.tag,
        "source_commit": source_commit,
        "provenance_commit": provenance_commit,
        "commit_count": int(run(destination, "rev-list", "--all", "--count")),
        "first_party_entries": imported_rows,
        "gitlinks": [
            {"path": path, "commit": object_id}
            for path, object_id in source_gitlinks
        ],
        "bundle": str(bundle),
        "bundle_sha256": sha256(bundle),
        "bundle_verification": bundle_verification.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--author-name", default="MARISA-B Release")
    parser.add_argument(
        "--author-email",
        default="marisa-b-release@users.noreply.github.com",
    )
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
