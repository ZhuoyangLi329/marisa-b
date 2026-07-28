#!/usr/bin/env python3
"""Check the generated ReACT adapter and the no-integration CLI smoke path."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_marisa_b_triangle as triangle_build  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = triangle_build.prepare_adapter(args.build_dir)

    header = adapter["bspt_header"].read_text(encoding="utf-8")
    cpp = adapter["bspt_cpp"].read_text(encoding="utf-8")
    special = adapter["special_functions"].read_text(encoding="utf-8")
    stubs = adapter["gr_stubs"].read_text(encoding="utf-8")

    require(
        header.count("void BlooptermComponentsGR(") == 1,
        "BSPT component declaration must appear exactly once",
    )
    require(
        cpp.count("void BSPT::BlooptermComponentsGR(") == 1,
        "BSPT component definition must appear exactly once",
    )
    require(
        cpp.count(triangle_build.BSPT_COMPONENT_METHOD) == 1,
        "the historical BSPT component method must be inserted verbatim once",
    )
    require(
        triangle_build.SCOL_INCLUDE not in special,
        "the generated SpecialFunctions copy still includes SCOL.h",
    )
    require(
        stubs == triangle_build.BEYOND_LCDM_GR_STUBS,
        "the generated GR stubs differ from the canonical adapter",
    )
    require(
        hashlib.sha256(
            triangle_build.BSPT_COMPONENT_METHOD.encode("utf-8")
        ).hexdigest()
        == "556b3eb85d1b337e6968ffd63515556483cc9a6189013e26f46722d131dfe639",
        "the historical BSPT component formula text changed",
    )
    require(
        hashlib.sha256(
            triangle_build.BEYOND_LCDM_GR_STUBS.encode("utf-8")
        ).hexdigest()
        == "0c8eeb849f82cc3a9bcbcf15e0c9260d209067fc4385e253d2ebdc551bbd0d51",
        "the historical GR stub text changed",
    )
    react_driver = (
        PROJECT_ROOT / "src" / "react_adapter" / "react_bspt_driver.cpp"
    )
    require(
        react_driver.is_file()
        and hashlib.sha256(react_driver.read_bytes()).hexdigest()
        == "91a5194de78f9d00cb15ec6360def3ef5a855db50fc724f4770ce3cbac7bb98e",
        "the imported ReACT diagnostic driver is missing or changed",
    )
    require(args.binary.is_file(), f"triangle binary is missing: {args.binary}")

    symbols = subprocess.run(
        ["nm", "-C", str(args.binary)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        symbols.returncode == 0
        and "BSPT::BlooptermComponentsGR(" in symbols.stdout,
        "compiled binary does not retain the BSPT component diagnostics",
    )

    result = subprocess.run(
        [str(args.binary), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=triangle_build.thread_limited_env(triangle_build.MAX_THREADS),
    )
    require(result.returncode == 0, f"--help returned {result.returncode}")
    require(
        "Usage: marisa_b_triangle" in result.stdout,
        "--help output does not contain the CLI usage line",
    )
    print("MARISA-B triangle adapter/build checks passed: 12")


if __name__ == "__main__":
    main()
