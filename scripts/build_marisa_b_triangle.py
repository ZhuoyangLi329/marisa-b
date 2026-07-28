#!/usr/bin/env python3
"""Build the canonical MARISA-B triangle CLI with its ReACT GR adapter.

The upstream ReACT checkout is kept pristine.  This helper reproduces the
historical build-time adapter used by the production triangle runner:

* add the diagnostics-only ``BSPT::BlooptermComponentsGR`` declaration and
  definition to build-directory copies of ``BSPT.h`` and ``BSPT.cpp``;
* copy ``SpecialFunctions.cpp`` while removing its unused ``SCOL.h`` include;
* provide the same GR-only BeyondLCDM link stubs.

No loop integrand or normalization formula is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REACT_ROOT = PROJECT_ROOT / "external" / "ACTio-ReACTio" / "reactions"
REACT_SRC = REACT_ROOT / "src"
MARISA_DIR = PROJECT_ROOT / "src" / "marisa_b"
MARISA_TRIANGLE_CPP = MARISA_DIR / "marisa_b_triangle.cpp"
MARISA_NATIVE_CPP = MARISA_DIR / "marisa_b_native.cpp"

MAX_THREADS = 28
THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)

REACT_SOURCES = (
    "BSPT.cpp",
    "Common.cpp",
    "Cosmology.cpp",
    "InterpolatedPS.cpp",
    "PowerSpectrum.cpp",
    "Quadrature.cpp",
    "SPT.cpp",
    "Spline.cpp",
    "array.cpp",
)

BSPT_HEADER_MARKER = (
    "    // 1-loop terms except for B123\n"
    "    real Bloopterms(int model, real k1, real k2, real k3, real x) const;\n"
)
BSPT_COMPONENT_DECLARATION = (
    "    // Diagnostics-only GR/EdS split of Bloopterms = "
    "B222 + B321I + B411.\n"
    "    void BlooptermComponentsGR(real k1, real k2, real k3, real x, "
    "real& B222, real& B321I, real& B411) const;\n"
)
BSPT_CPP_MARKER = (
    "\n\n///////////////////// NUMERICAL TREE LEVEL SPECTRUM "
    "////////////////////////"
)
BSPT_COMPONENT_METHOD = r'''

void BSPT::BlooptermComponentsGR(real k1, real k2, real k3, real x, real& B222, real& B321I, real& B411) const {
      real KMAX = QMAXp/k1;
      real KMIN = QMINp/k1;
      real tp = 2.*M_PI;
      real c[3] = {KMIN,-1.,0.};
      real d[3] = {KMAX,0.99999999,tp};
      real prefac = pow6(Dl_spt/dnorm_spt)*pow3(k1/tp);
      B222 = prefac*Integrate<3>(bind(B_222, cref(P_L), k1, k2, x, k3, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3), c, d, epsrel);
      B321I = prefac*Integrate<3>(bind(B_321I, cref(P_L), k1, k2, x, k3, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3), c, d, epsrel);
      B411 = prefac*Integrate<3>(bind(B_411, cref(P_L), k1, k2, x, k3, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3), c, d, epsrel);
}
'''

SCOL_INCLUDE = '#include "SCOL.h"\n'

BEYOND_LCDM_GR_STUBS = r'''
#include <cmath>

static double lcdm_h(double a, double omega0) {
    return std::sqrt(omega0 / (a * a * a) + (1.0 - omega0));
}

double bespokehub(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    return lcdm_h(a, omega0);
}

double bespokehubd(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    return -3.0 * omega0 / (2.0 * a * a * a);
}

double bespokehubdd(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    return 9.0 * omega0 / (2.0 * a * a * a);
}

double HAg(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    return lcdm_h(a, omega0);
}

double HA1g(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    return -3.0 * omega0 / (2.0 * a * a * a);
}

double HA2g(double a, double omega0, double extpars[], int model) {
    (void)extpars;
    (void)model;
    const double h = lcdm_h(a, omega0);
    return 3.0 * omega0 / (2.0 * h * h * a * a * a);
}

double HA2g2(double a, double omega0, double extpars[], int model) {
    return HA2g(a, omega0, extpars, model);
}

double myfricF(double a, double omega0, double extpars[], int model) {
    (void)a;
    (void)omega0;
    (void)extpars;
    (void)model;
    return 0.0;
}

double mu(double a, double k0, double omega0, double extpars[], int model) {
    (void)a;
    (void)k0;
    (void)omega0;
    (void)extpars;
    (void)model;
    return 1.0;
}

double gamma2(double a, double omega0, double k0, double k1, double k2, double u1, double extpars[], int model) {
    (void)a;
    (void)omega0;
    (void)k0;
    (void)k1;
    (void)k2;
    (void)u1;
    (void)extpars;
    (void)model;
    return 0.0;
}

double gamma3(double a, double omega0, double k0, double k1, double k2, double k3, double u1, double u2, double u3, double extpars[], int model) {
    (void)a;
    (void)omega0;
    (void)k0;
    (void)k1;
    (void)k2;
    (void)k3;
    (void)u1;
    (void)u2;
    (void)u3;
    (void)extpars;
    (void)model;
    return 0.0;
}
'''.lstrip()


def _bounded_thread_count(value: int | str | None) -> int:
    try:
        parsed = int(value) if value is not None else MAX_THREADS
    except (TypeError, ValueError):
        parsed = MAX_THREADS
    return max(1, min(parsed, MAX_THREADS))


def thread_limited_env(max_threads: int | str | None = MAX_THREADS) -> dict[str, str]:
    """Return an environment in which every common thread pool is capped."""

    capped = _bounded_thread_count(max_threads)
    env = os.environ.copy()
    for key in THREAD_LIMIT_ENV_KEYS:
        try:
            requested = int(env[key]) if env.get(key, "").strip() else capped
        except ValueError:
            requested = capped
        env[key] = str(max(1, min(requested, capped)))
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return env


def enforce_thread_limit(max_threads: int | str | None = MAX_THREADS) -> None:
    os.environ.update(thread_limited_env(max_threads))


def _require_exactly_once(text: str, marker: str, source: Path) -> None:
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one adapter marker in {source}, found {count}"
        )


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def write_bspt_component_copies(build_dir: Path) -> tuple[Path, Path]:
    """Create the exact historical diagnostics patch without touching ReACT."""

    header_source = REACT_SRC / "BSPT.h"
    cpp_source = REACT_SRC / "BSPT.cpp"
    header_target = build_dir / "BSPT.h"
    cpp_target = build_dir / "BSPT_component_diag.cpp"

    header_text = header_source.read_text(encoding="utf-8")
    _require_exactly_once(header_text, BSPT_HEADER_MARKER, header_source)
    if "BlooptermComponentsGR" in header_text:
        raise RuntimeError(
            "upstream BSPT.h already defines BlooptermComponentsGR; "
            "refusing to apply the adapter twice"
        )
    patched_header = header_text.replace(
        BSPT_HEADER_MARKER,
        BSPT_HEADER_MARKER + BSPT_COMPONENT_DECLARATION,
        1,
    )
    if patched_header.count("void BlooptermComponentsGR(") != 1:
        raise RuntimeError("BSPT declaration patch was not applied exactly once")

    cpp_text = cpp_source.read_text(encoding="utf-8")
    _require_exactly_once(cpp_text, BSPT_CPP_MARKER, cpp_source)
    if "BSPT::BlooptermComponentsGR" in cpp_text:
        raise RuntimeError(
            "upstream BSPT.cpp already defines BlooptermComponentsGR; "
            "refusing to apply the adapter twice"
        )
    patched_cpp = cpp_text.replace(
        BSPT_CPP_MARKER,
        BSPT_COMPONENT_METHOD + BSPT_CPP_MARKER,
        1,
    )
    if patched_cpp.count("void BSPT::BlooptermComponentsGR(") != 1:
        raise RuntimeError("BSPT definition patch was not applied exactly once")

    _write_text_if_changed(header_target, patched_header)
    _write_text_if_changed(cpp_target, patched_cpp)
    return header_target, cpp_target


def write_special_functions_gr_copy(build_dir: Path) -> Path:
    """Copy SpecialFunctions.cpp with only its unused SCOL include removed."""

    source = REACT_SRC / "SpecialFunctions.cpp"
    target = build_dir / "SpecialFunctions_gr_no_scol.cpp"
    text = source.read_text(encoding="utf-8")
    _require_exactly_once(text, SCOL_INCLUDE, source)
    patched = text.replace(SCOL_INCLUDE, "", 1)
    if SCOL_INCLUDE in patched:
        raise RuntimeError("SCOL include removal was not exact")
    _write_text_if_changed(target, patched)
    return target


def write_beyond_lcdm_gr_stubs(build_dir: Path) -> Path:
    """Write the historical GR-only link stubs verbatim."""

    target = build_dir / "BeyondLCDM_gr_stubs.cpp"
    _write_text_if_changed(target, BEYOND_LCDM_GR_STUBS)
    return target


def prepare_adapter(build_dir: Path) -> dict[str, Path]:
    build_dir.mkdir(parents=True, exist_ok=True)
    bspt_header, bspt_cpp = write_bspt_component_copies(build_dir)
    special_functions = write_special_functions_gr_copy(build_dir)
    gr_stubs = write_beyond_lcdm_gr_stubs(build_dir)
    return {
        "bspt_header": bspt_header,
        "bspt_cpp": bspt_cpp,
        "special_functions": special_functions,
        "gr_stubs": gr_stubs,
    }


def _existing_gsl_prefix_flags(prefix: Path) -> tuple[list[str], list[str], dict[str, str]]:
    bases = (prefix, prefix / "usr")
    include_dirs = [
        base / "include"
        for base in bases
        if (base / "include" / "gsl" / "gsl_sf_bessel.h").is_file()
    ]
    lib_dirs: list[Path] = []
    for base in bases:
        lib_dirs.extend(sorted((base / "lib").glob("*-linux-gnu")))
        lib_dirs.extend([base / "lib64", base / "lib"])
    lib_dirs = [
        path
        for path in lib_dirs
        if any(path.glob("libgsl.so*")) and any(path.glob("libgslcblas.so*"))
    ]
    if not include_dirs or not lib_dirs:
        raise FileNotFoundError(
            f"GSL_PREFIX does not contain GSL headers and libraries: {prefix}"
        )
    include_dir = include_dirs[0]
    lib_dir = lib_dirs[0]
    compile_flags = [f"-I{include_dir}"]
    link_flags = [
        f"-L{lib_dir}",
        "-Wl,--disable-new-dtags",
        f"-Wl,-rpath,{lib_dir}",
        "-lgsl",
        "-lgslcblas",
        "-lm",
    ]
    metadata = {
        "source": "prefix",
        "prefix": str(prefix),
        "include": str(include_dir),
        "lib": str(lib_dir),
    }
    return compile_flags, link_flags, metadata


def resolve_gsl_flags(
    gsl_prefix: Path | str | None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Resolve conventional or Debian-extracted GSL_PREFIX layouts."""

    prefix_text = str(gsl_prefix).strip() if gsl_prefix is not None else ""
    if not prefix_text:
        prefix_text = os.environ.get("GSL_PREFIX", "").strip()
    if not prefix_text:
        prefix_text = os.environ.get("MARISA_B_GSL_ROOT", "").strip()
    if prefix_text:
        return _existing_gsl_prefix_flags(Path(prefix_text).expanduser().resolve())

    pkg = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "gsl"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pkg.returncode == 0:
        flags = shlex.split(pkg.stdout)
        compile_flags = [flag for flag in flags if flag.startswith(("-I", "-D"))]
        link_flags = [flag for flag in flags if flag not in compile_flags]
        return compile_flags, link_flags, {"source": "pkg-config"}
    return [], ["-lgsl", "-lgslcblas", "-lm"], {"source": "system-default"}


def _stringify_command(command: Iterable[str]) -> list[str]:
    return [str(item) for item in command]


def build_marisa_b_triangle(
    *,
    binary: Path,
    build_dir: Path,
    native_object: Path | None = None,
    gsl_prefix: Path | str | None = None,
    cxx: str | None = None,
    max_threads: int | str | None = MAX_THREADS,
    smoke_help: bool = False,
) -> dict[str, object]:
    """Prepare the adapter, compile the CLI, and optionally run ``--help``."""

    capped_threads = _bounded_thread_count(max_threads)
    binary = Path(binary).resolve()
    build_dir = Path(build_dir).resolve()
    adapter = prepare_adapter(build_dir)
    binary.parent.mkdir(parents=True, exist_ok=True)
    cxx_command = shlex.split(cxx or os.environ.get("CXX", "g++"))
    if not cxx_command:
        raise ValueError("CXX command is empty")
    gsl_compile, gsl_link, gsl_metadata = resolve_gsl_flags(gsl_prefix)

    react_sources = [
        adapter["bspt_cpp"]
        if name == "BSPT.cpp"
        else REACT_SRC / name
        for name in REACT_SOURCES
    ]
    native_input = (
        Path(native_object).resolve()
        if native_object is not None
        else MARISA_NATIVE_CPP
    )
    if not native_input.is_file():
        raise FileNotFoundError(f"canonical native input is missing: {native_input}")

    command = [
        *cxx_command,
        "-std=gnu++17",
        "-O3",
        "-fopenmp",
        "-ffunction-sections",
        "-fdata-sections",
        "-DHAVE_CONFIG_H",
        f'-DDATADIR="{REACT_ROOT / "data"}"',
        f"-I{build_dir}",
        f"-I{REACT_ROOT}",
        f"-I{REACT_SRC}",
        f"-I{MARISA_DIR}",
        *gsl_compile,
        str(MARISA_TRIANGLE_CPP),
        str(native_input),
        *(str(source) for source in react_sources),
        str(adapter["special_functions"]),
        str(adapter["gr_stubs"]),
        *gsl_link,
        "-Wl,--gc-sections",
        "-o",
        str(binary),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=thread_limited_env(capped_threads),
    )
    report: dict[str, object] = {
        "command": _stringify_command(command),
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "binary": str(binary),
        "build_dir": str(build_dir),
        "native_input": str(native_input),
        "adapter": {key: str(value) for key, value in adapter.items()},
        "gsl": gsl_metadata,
        "max_threads": capped_threads,
    }
    if result.returncode != 0:
        return report

    if smoke_help:
        help_result = subprocess.run(
            [str(binary), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=thread_limited_env(capped_threads),
        )
        report["help_smoke"] = {
            "command": [str(binary), "--help"],
            "returncode": int(help_result.returncode),
            "stdout": help_result.stdout,
            "stderr": help_result.stderr,
            "usage_present": "Usage: marisa_b_triangle" in help_result.stdout,
        }
        if help_result.returncode != 0 or "Usage: marisa_b_triangle" not in help_result.stdout:
            report["returncode"] = 2
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "marisa_b_triangle_adapter",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=PROJECT_ROOT / "build" / "marisa_b_triangle",
    )
    parser.add_argument(
        "--native-object",
        type=Path,
        default=None,
        help="Reuse Makefile's canonical marisa_b_native.o instead of recompiling it.",
    )
    parser.add_argument(
        "--gsl-prefix",
        default=None,
        help="GSL prefix; supports PREFIX/include and Debian PREFIX/usr/include layouts.",
    )
    parser.add_argument("--cxx", default=None)
    parser.add_argument("--max-threads", type=int, default=MAX_THREADS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-help", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        adapter = prepare_adapter(args.build_dir)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "build_dir": str(args.build_dir),
                    "adapter": {key: str(value) for key, value in adapter.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    report = build_marisa_b_triangle(
        binary=args.binary,
        build_dir=args.build_dir,
        native_object=args.native_object,
        gsl_prefix=args.gsl_prefix,
        cxx=args.cxx,
        max_threads=args.max_threads,
        smoke_help=args.smoke_help,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if int(report["returncode"]) != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
