"""Finite-PNG model-coordinate and recombination regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from marisa_b.model import (
    FinitePngComponents,
    ModelSelection,
    combine_finite_png,
)


def test_model_selection_covers_production_and_crosscheck() -> None:
    for reconstruction in ("pre", "post"):
        production = ModelSelection("coevolution", reconstruction)
        crosscheck = ModelSelection("full", reconstruction)
        assert production.png_order == "quadratic"
        assert crosscheck.png_order == "quadratic"
    with pytest.raises(ValueError):
        ModelSelection("uplift", "pre")  # type: ignore[arg-type]


def test_finite_png_golden_vectors_and_parity() -> None:
    gaussian = np.asarray([2.0, -3.0, 5.0])
    linear = np.asarray([0.1, 0.2, -0.4])
    quadratic = np.asarray([0.001, -0.002, 0.003])
    components = FinitePngComponents.from_arrays(
        gaussian,
        linear,
        quadratic,
    )
    plus = components.predict(100.0)
    zero = components.predict(0.0)
    minus = components.predict(-100.0)
    np.testing.assert_allclose(zero, gaussian)
    np.testing.assert_allclose(
        plus,
        np.asarray([22.0, -3.0, -5.0]),
    )
    np.testing.assert_allclose(
        minus,
        np.asarray([2.0, -43.0, 75.0]),
    )
    np.testing.assert_allclose((plus - minus) / 200.0, linear)
    np.testing.assert_allclose(
        (plus + minus) / 2.0 - zero,
        100.0**2 * quadratic,
    )
    np.testing.assert_allclose(
        combine_finite_png(gaussian, linear, quadratic, 100.0),
        plus,
    )


def test_finite_png_rejects_invalid_components() -> None:
    with pytest.raises(ValueError):
        FinitePngComponents.from_arrays([1.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        combine_finite_png([1.0], [0.0], [0.0], float("nan"))
