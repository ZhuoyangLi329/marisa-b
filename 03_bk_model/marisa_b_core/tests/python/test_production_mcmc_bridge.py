"""生产 MCMC runner 必须只使用 package 中的唯一 weighted quantile。"""

from __future__ import annotations

import ast
from pathlib import Path


def test_production_runner_imports_unique_weighted_quantile() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runner = (
        repo_root
        / "scripts/production/run_prepost_recon_halo_fnl_mcmc_v1.py"
    )
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "weighted_quantile"
    ]
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "marisa_b.statistics"
        for alias in node.names
    ]
    assert definitions == []
    assert imports.count("weighted_quantile") == 1
