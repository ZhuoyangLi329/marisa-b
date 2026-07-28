"""Small, explicit model contracts shared by MARISA-B inference runners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


BiasTier = Literal["coevolution", "full"]
Reconstruction = Literal["pre", "post"]
PngOrder = Literal["gaussian", "linear", "quadratic"]


@dataclass(frozen=True)
class ModelSelection:
    """Discrete coordinates of the current halo model."""

    bias_tier: BiasTier
    reconstruction: Reconstruction
    png_order: PngOrder = "quadratic"

    def __post_init__(self) -> None:
        if self.bias_tier not in {"coevolution", "full"}:
            raise ValueError(f"unsupported bias tier {self.bias_tier!r}")
        if self.reconstruction not in {"pre", "post"}:
            raise ValueError(
                f"unsupported reconstruction {self.reconstruction!r}"
            )
        if self.png_order not in {"gaussian", "linear", "quadratic"}:
            raise ValueError(f"unsupported PNG order {self.png_order!r}")


def _component(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


@dataclass(frozen=True)
class FinitePngComponents:
    """Gaussian, linear-PNG, and quadratic-PNG prediction vectors."""

    gaussian: NDArray[np.float64]
    linear: NDArray[np.float64]
    quadratic: NDArray[np.float64]

    @classmethod
    def from_arrays(
        cls,
        gaussian: ArrayLike,
        linear: ArrayLike,
        quadratic: ArrayLike,
    ) -> "FinitePngComponents":
        result = cls(
            gaussian=_component(gaussian, "gaussian"),
            linear=_component(linear, "linear"),
            quadratic=_component(quadratic, "quadratic"),
        )
        if not (
            result.gaussian.shape
            == result.linear.shape
            == result.quadratic.shape
        ):
            raise ValueError("finite-PNG components have mismatched shapes")
        return result

    def predict(self, fnl: float) -> NDArray[np.float64]:
        """Return ``G + fNL L + fNL² Q`` for a finite scalar ``fNL``."""

        value = float(fnl)
        if not math.isfinite(value):
            raise ValueError("fNL must be finite")
        prediction = (
            self.gaussian
            + value * self.linear
            + value**2 * self.quadratic
        )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("finite-PNG prediction is not finite")
        return np.asarray(prediction, dtype=np.float64)


def combine_finite_png(
    gaussian: ArrayLike,
    linear: ArrayLike,
    quadratic: ArrayLike,
    fnl: float,
) -> NDArray[np.float64]:
    """Validated functional wrapper for finite-PNG recombination."""

    return FinitePngComponents.from_arrays(
        gaussian,
        linear,
        quadratic,
    ).predict(fnl)
