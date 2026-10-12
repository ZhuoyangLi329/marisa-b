"""五条 coevolution bias 关系的纯 Python 回归测试。"""

from __future__ import annotations

import pytest

from marisa_b.parameters import (
    COEVOLUTION_DEPENDENT,
    coevolution_parameters,
)


def test_coevolution_relations_match_registered_formulas() -> None:
    source = {
        "b1": 2.5,
        "b2": 0.7,
        "gamma2": -0.3,
        "gamma2x": 0.4,
        "gamma3": -0.2,
        "stochastic": 9.0,
        **{name: 999.0 for name in COEVOLUTION_DEPENDENT},
    }
    untouched = dict(source)
    result = coevolution_parameters(source)
    gamma21 = (2.0 / 21.0) * 1.5 + (6.0 / 7.0) * -0.3
    assert result["gamma21"] == pytest.approx(gamma21)
    assert result["gamma21x"] == pytest.approx(
        (2.0 / 21.0) * 0.7 + (6.0 / 7.0) * 0.4
    )
    assert result["gamma211"] == pytest.approx(
        (5.0 / 77.0) * 1.5
        + (15.0 / 14.0) * -0.3
        - (9.0 / 7.0) * -0.2
        + gamma21
    )
    assert result["gamma22"] == pytest.approx(
        -(6.0 / 539.0) * 1.5 - (9.0 / 49.0) * -0.3
    )
    assert result["gamma31"] == pytest.approx(
        -(4.0 / 11.0) * 1.5 - 6.0 * -0.3
    )
    assert result["stochastic"] == 9.0
    assert source == untouched


def test_coevolution_requires_finite_independent_parameters() -> None:
    valid = {
        "b1": 2.0,
        "b2": 0.0,
        "gamma2": 0.0,
        "gamma2x": 0.0,
        "gamma3": 0.0,
    }
    missing = dict(valid)
    del missing["gamma3"]
    with pytest.raises(KeyError):
        coevolution_parameters(missing)
    invalid = dict(valid, gamma2=float("nan"))
    with pytest.raises(ValueError):
        coevolution_parameters(invalid)
