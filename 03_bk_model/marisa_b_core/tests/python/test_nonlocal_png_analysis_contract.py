"""项目级 nonlocal-PNG 分析脚本的轻量数值 contract 测试。

代码大纲
--------
1. 按文件路径导入分析 runner，不读取大型 simulation 产品。
2. 验证 precision-weighted 响应振幅、cosine 与 Fisher sigma 的定义。
3. 验证 nuisance projector 确实删除指定方向，并给出可手算的边缘化结果。
4. 验证 pre/standard reconstruction polynomial 取值、``b1^3`` halo 映射及
   tree+loop 求和，同时检查最终源代码不含 ``A_P`` 或双-shape Fisher 接口。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


RUNNER = (
    Path(__file__).resolve().parents[4]
    / "codes/run_quijote_halo_z0p5_nonlocal_marisa_b_analysis.py"
)


def load_runner():
    """从共享 workspace 导入 runner，并注册模块以支持 dataclass 初始化。"""
    specification = importlib.util.spec_from_file_location(
        "_test_nonlocal_png_analysis", RUNNER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_response_metric_has_expected_amplitude_cosine_and_sigma():
    """平行向量的 projection amplitude 应为 2，cosine 应为 1。"""
    runner = load_runner()
    result = runner.metric_from_vectors(
        np.asarray((3.0, 4.0)), np.asarray((6.0, 8.0))
    )
    assert result["simulation_on_theory_amplitude"] == 2.0
    assert result["precision_weighted_cosine"] == 1.0
    assert result["sigma_theory"] == 0.2
    assert result["sigma_simulation"] == 0.1


def test_nuisance_projection_matches_hand_calculation():
    """投影掉第一坐标后，响应只剩第二坐标且 sigma(fNL)=1。"""
    runner = load_runner()
    chol = np.eye(2)
    response = np.asarray((1.0, 1.0))
    nuisance = np.asarray(((1.0,), (0.0,)))
    result, arrays = runner.fisher_summary(
        chol, response, nuisance, nuisance_names=("test_nuisance",)
    )
    assert np.isclose(result["conditional"]["sigma_fNL"], 1.0 / np.sqrt(2.0))
    assert np.isclose(result["joint_six_parameter"]["sigma_fNL"], 1.0)
    assert np.allclose(arrays["bias_projector"] @ response, (0.0, 1.0))
    assert result["joint_six_parameter"]["joint_covariance_available"]
    assert np.all(np.isfinite(arrays["joint_covariance"]))


def test_saturated_nuisance_space_is_reported_unidentifiable():
    """nuisance 张满数据空间时必须返回 None，而不是浮点伪残差的大 sigma。"""
    runner = load_runner()
    result, arrays = runner.fisher_summary(
        np.eye(2),
        np.asarray((1.0, -2.0)),
        np.eye(2),
        nuisance_names=("nuisance_1", "nuisance_2"),
    )
    joint = result["joint_six_parameter"]
    assert not joint["identifiable"]
    assert joint["sigma_fNL"] is None
    assert not joint["joint_covariance_available"]
    assert np.array_equal(arrays["bias_projector"], np.zeros((2, 2)))


def test_reconstruction_polynomial_and_halo_mapping():
    """pre 取 lambda=0，standard 取 b1/b_rec，最后统一乘 b1^3。"""
    runner = load_runner()
    theory = {
        "selected_indices": np.asarray((1, 2), dtype=int),
        "eq_tree_lambda_coefficients": np.asarray(
            ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        ),
        "eq_loop_lambda_coefficients": np.asarray(
            ((0.5, 1.0), (1.5, 2.0), (2.5, 3.0))
        ),
    }
    context = SimpleNamespace(indices=np.asarray((2, 1), dtype=int))
    b1 = runner.B_REC
    pre_tree = runner.theory_component(theory, "EQ", "pre", context, b1, "tree")
    standard_tree = runner.theory_component(
        theory, "EQ", "standard", context, b1, "tree"
    )
    standard_total = runner.theory_component(
        theory, "EQ", "standard", context, b1, "total"
    )
    assert np.allclose(pre_tree / b1**3, (2.0, 1.0))
    assert np.allclose(standard_tree / b1**3, (12.0, 9.0))
    assert np.allclose(standard_total / b1**3, (18.0, 13.5))


def test_final_source_has_no_power_amplitude_or_dual_shape_fisher():
    """用户冻结范围不允许 ``A_P`` 与 EQ+OR-LSS 联合 Fisher。"""
    source = RUNNER.read_text(encoding="utf-8")
    assert "A_P" not in source
    assert "joint_shape" not in source
    assert "joint_correlation" not in source
