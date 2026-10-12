"""MARISA-B 的轻量 Python 公共接口。

当前包只承载可独立测试的统计语义、参数映射和路径解析。昂贵的理论
积分与版本化分析 runner 仍保留在各自入口中，后续再逐步迁移。
"""

from .parameters import COEVOLUTION_DEPENDENT, coevolution_parameters
from .model import (
    FinitePngComponents,
    ModelSelection,
    combine_finite_png,
)
from .paths import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    RepositoryPaths,
    resolve_repository_paths,
)
from .statistics import (
    SUPPORTED_PRECISION_CORRECTIONS,
    hartlap_factor,
    mean_covariance,
    single_realization_covariance,
    validate_precision_correction,
    weighted_quantile,
)

__all__ = [
    "COEVOLUTION_DEPENDENT",
    "DATA_ROOT_ENVIRONMENT_VARIABLE",
    "FinitePngComponents",
    "ModelSelection",
    "RepositoryPaths",
    "SUPPORTED_PRECISION_CORRECTIONS",
    "coevolution_parameters",
    "combine_finite_png",
    "hartlap_factor",
    "mean_covariance",
    "resolve_repository_paths",
    "single_realization_covariance",
    "validate_precision_correction",
    "weighted_quantile",
]

__version__ = "0.1.0"
