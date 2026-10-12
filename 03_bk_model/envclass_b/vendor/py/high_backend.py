#!/usr/bin/env python3
"""构造 z=0.5 pre/standard 的共享 fully-tree Gaussian context。

代码大纲
========
1. 复用活动 MARISA-B halo-tree/stochastic 与 standard-reconstruction kernel assets。
2. 构造 kmax=0.10/0.15/0.20 的 pre/standard contexts 和 Fid500 covariance。
3. 只提供 context；不读取旧 nonlocal theory、convergence、profile 或 MCMC 结果。
4. 下游必须显式选取 `gaussian_tree + gaussian_stochastic`，不得使用 matter loop。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import run_quijote_halo_z0p5_nonlocal_marisa_b_analysis as context_base


ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = ROOT / "codes"
CUTS = (0.10, 0.15, 0.20)
EXPECTED_COUNTS = {0.10: 14, 0.15: 27, 0.20: 44}
STANDARD_K20_ANALYSIS = ROOT / "analysis/quijote_halo_z0p5_r10_standard_fnl_model_kmax0p20"
STANDARD_LOADER = CODES / "run_quijote_halo_z0p5_standard_fnl_model_audit.py"
CERTIFIED_INDICES = np.asarray((
    1, 2, 3, 4, 5, 6, 8, 9,
    15, 16, 17, 18, 19, 20, 22, 23,
    29, 30, 31, 32, 33, 35, 36,
    42, 43, 44, 45, 47, 48,
    54, 55, 56, 58, 59,
    65, 66, 68, 69,
    75, 77, 78, 92, 93, 99,
), dtype=np.int64)


def import_path(name: str, path: Path) -> Any:
    """按绝对路径导入活动 Python 模块。"""

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_runtime() -> Any:
    """构造三个 cut 的 contexts，不读取任何归档/旧 nonlocal 结果。"""

    for path in (
        context_base.REFINED_PROFILE,
        context_base.REFINED_ASSETS,
        context_base.FID_MATRIX,
        *context_base.SHAPE_MATRICES.values(),
        STANDARD_LOADER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = import_path("_shared_tree_profile_context", context_base.REFINED_PROFILE)
    projected = import_path("_shared_tree_projected_context", profile.PROJECTED_RUNNER)
    refiner = import_path("_shared_tree_refined_assets", context_base.REFINED_ASSETS)
    _baseline_old, physical, _corrected_old, _audit_old, _assets_old, _method_old, physical_cache = (
        projected.load_modules_and_assets()
    )
    standard_loader = import_path("_shared_tree_standard_loader", STANDARD_LOADER)
    standard_loader.TAG = "quijote_halo_z0p5_r10_standard_fnl_model_kmax0p20"
    standard_loader.ANALYSIS = STANDARD_K20_ANALYSIS
    standard_loader.THEORY = STANDARD_K20_ANALYSIS / "theory"
    standard_loader.TREE_CACHE = (
        STANDARD_K20_ANALYSIS / "tree_theory_cache/z0p5_standard_kmax0p20/compiled_tree_templates.npz"
    )
    standard_loader.PNG_CACHE = (
        standard_loader.THEORY / "finite_png_templates_z0p5_r10_brec1p9326527_kmax0p20.npz"
    )
    standard_loader.ORACLE_CACHE = (
        standard_loader.THEORY / "oracle_matter_response_z0p5_r10_brec1p9326527_kmax0p20.npz"
    )
    standard_loader.THEORY_MANIFEST = standard_loader.THEORY / "theory_manifest.json"
    standard_loader.WRAPPER_MANIFEST = standard_loader.THEORY / "standard_wrapper_manifest.json"
    standard_loader.CUTS = CUTS
    standard_loader.EXPECTED_COUNTS = EXPECTED_COUNTS
    standard_loader.CACHE_INDICES = CERTIFIED_INDICES.copy()
    corrected, audit = standard_loader.configure_base()
    assets = standard_loader.load_assets(corrected)
    standard_method, _method_record, _p0_arrays = standard_loader.load_standard_method(corrected, audit)
    assets, refinement = refiner.refined_assets(assets)
    projected.VOLUME = context_base.VOLUME
    corrected.VOLUME = context_base.VOLUME
    audit.VOLUME = context_base.VOLUME
    matrix = projected.load_matrix()
    tree = projected.load_pre_tree(matrix)
    dm_tree, dm_total, _dm_audit = projected.load_z05_dm_response(matrix)
    bias = json.loads(projected.BIAS_JSON.read_text(encoding="utf-8"))
    growth_ratio = float(corrected.legacy.configurations()["z0p5"].growth_power_ratio_to_z1)
    nbar = float(bias["nbar_mean"])
    contexts = {
        cut: profile.build_contexts(
            projected, physical, corrected, assets, standard_method, physical_cache,
            matrix, tree, dm_tree, dm_total, growth_ratio, nbar, cut, refinement,
        )
        for cut in CUTS
    }
    geometry_key = "B000_k" if "B000_k" in matrix else "b000_k"
    geometry = np.asarray(matrix[geometry_key], dtype=float)
    for cut, pair in contexts.items():
        expected = CERTIFIED_INDICES[
            np.max(geometry[CERTIFIED_INDICES], axis=1) <= cut + 1e-12
        ]
        if expected.size != EXPECTED_COUNTS[cut]:
            raise RuntimeError(f"cut {cut}: expected registry size drift")
        for method, context in pair.items():
            if not np.array_equal(np.asarray(context.indices, dtype=int), expected):
                raise RuntimeError(f"{method}/{cut}: context index drift")
            covariance = np.cov(np.asarray(context.samples), rowvar=False, ddof=1)
            if not np.allclose(context.covariance_single, covariance, rtol=2e-14, atol=1e-6):
                raise RuntimeError(f"{method}/{cut}: Fid500 covariance drift")
    shape_matrices = {
        shape: context_base.load_npz(path)
        for shape, path in context_base.SHAPE_MATRICES.items()
    }
    return context_base.Runtime(
        contexts=contexts,
        matrix=matrix,
        shape_matrices=shape_matrices,
        refinement=refinement,
        modules={
            "baseline": standard_loader,
            "physical": physical,
            "base": corrected,
            "audit": audit,
            "assets": assets,
            "projected": projected,
        },
    )


def main() -> None:
    """运行轻量 smoke 并打印 context 维数。"""

    runtime = load_runtime()
    print(json.dumps({
        f"{cut:.2f}": {
            method: int(len(context.indices))
            for method, context in runtime.contexts[cut].items()
        }
        for cut in CUTS
    }, sort_keys=True))


if __name__ == "__main__":
    main()
