"""MARISA-B halo bias 参数关系的纯 Python 实现。"""

from __future__ import annotations

import math
from collections.abc import Mapping


COEVOLUTION_DEPENDENT = (
    "gamma21",
    "gamma21x",
    "gamma211",
    "gamma22",
    "gamma31",
)

_COEVOLUTION_REQUIRED = (
    "b1",
    "b2",
    "gamma2",
    "gamma2x",
    "gamma3",
)


def _finite_parameter(
    parameters: Mapping[str, float],
    name: str,
) -> float:
    """读取一个必需参数并拒绝缺失或非有限值。"""

    if name not in parameters:
        raise KeyError(f"missing coevolution parameter {name!r}")
    value = float(parameters[name])
    if not math.isfinite(value):
        raise ValueError(
            f"coevolution parameter {name!r} must be finite"
        )
    return value


def coevolution_parameters(
    parameters: Mapping[str, float],
) -> dict[str, float]:
    """在输入参数副本上应用五条 coevolution 关系。

    Parameters
    ----------
    parameters
        至少包含 ``b1``、``b2``、``gamma2``、``gamma2x`` 和
        ``gamma3`` 的参数映射，允许携带任意额外 nuisance 参数。

    Returns
    -------
    dict
        输入的独立副本，其中五个 dependent bias 被当前生产关系覆盖。

    Notes
    -----
    公式与现有 pre-reconstruction bias-v3 runner 中采用的
    Eggemeier et al. (2021) Eqs. (20)--(24)、``NLE_L=0`` 口径一致。
    原始输入映射不会被修改。
    """

    values = {
        name: _finite_parameter(parameters, name)
        for name in _COEVOLUTION_REQUIRED
    }
    result = dict(parameters)
    b1 = values["b1"]
    b2 = values["b2"]
    gamma2 = values["gamma2"]
    gamma2x = values["gamma2x"]
    gamma3 = values["gamma3"]
    gamma21 = (
        (2.0 / 21.0) * (b1 - 1.0)
        + (6.0 / 7.0) * gamma2
    )
    result["gamma21"] = gamma21
    result["gamma21x"] = (
        (2.0 / 21.0) * b2
        + (6.0 / 7.0) * gamma2x
    )
    result["gamma211"] = (
        (5.0 / 77.0) * (b1 - 1.0)
        + (15.0 / 14.0) * gamma2
        - (9.0 / 7.0) * gamma3
        + gamma21
    )
    result["gamma22"] = (
        -(6.0 / 539.0) * (b1 - 1.0)
        - (9.0 / 49.0) * gamma2
    )
    result["gamma31"] = (
        -(4.0 / 11.0) * (b1 - 1.0)
        - 6.0 * gamma2
    )
    return result
