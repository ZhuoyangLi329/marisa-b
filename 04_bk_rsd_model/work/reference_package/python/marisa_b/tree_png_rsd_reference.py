"""Tree-level redshift-space finite-local-PNG halo bispectrum oracle.

This module is a self-contained reference implementation for extending the
MARISA-B real-space halo tree model to plane-parallel redshift space.  It is
restricted to

    B_h^s(f_NL) = G + f_NL L + f_NL**2 Q

and does not evaluate any one-loop contribution.  The implementation keeps
MARISA-B's native bias convention:

* ``b2`` multiplies ``delta**2 / 2``;
* ``bK2`` multiplies ``S2 = (khat1.khat2)**2 - 1/3``;
* ``bphi2`` is the independent phi**2 operator coefficient, not ``bphi**2``;
* ``Q`` is the coefficient of ``f_NL**2`` (one half of the second derivative).

The default post-reconstruction map matches the current MARISA-B/Shirasaki
same-shift convention: the redshift-space halo density is smoothed, divided
by ``b_rec``, and used to shift both halos and randoms without removing the
linear Kaiser factor.  An adaptive local-PNG reconstruction denominator can
be expanded consistently through f_NL**2.

The code is intended as a numerical/scientific oracle for a later native C++
implementation.  It deliberately excludes survey selection effects,
wide-angle terms, Alcock-Paczynski remapping, FoG counterterms, and a claimed
finite-PNG stochastic closure.  Those belong to separate modelling layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from numpy.typing import ArrayLike, NDArray

ReconstructionDenominator = Literal["bias_only", "kaiser"]


@dataclass(frozen=True)
class PolyFnl2:
    """Polynomial ``c0 + f_NL c1 + f_NL**2 c2``, truncated at second order."""

    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            object.__setattr__(self, item.name, value)

    def __add__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        rhs = as_poly(other)
        return PolyFnl2(self.c0 + rhs.c0, self.c1 + rhs.c1, self.c2 + rhs.c2)

    __radd__ = __add__

    def __sub__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        rhs = as_poly(other)
        return PolyFnl2(self.c0 - rhs.c0, self.c1 - rhs.c1, self.c2 - rhs.c2)

    def __rsub__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        return as_poly(other).__sub__(self)

    def __neg__(self) -> "PolyFnl2":
        return PolyFnl2(-self.c0, -self.c1, -self.c2)

    def __mul__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        rhs = as_poly(other)
        return PolyFnl2(
            self.c0 * rhs.c0,
            self.c0 * rhs.c1 + self.c1 * rhs.c0,
            self.c0 * rhs.c2 + self.c1 * rhs.c1 + self.c2 * rhs.c0,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "PolyFnl2":
        if self.c0 == 0.0:
            raise ZeroDivisionError("a polynomial with zero constant term is not invertible")
        return PolyFnl2(
            1.0 / self.c0,
            -self.c1 / self.c0**2,
            self.c1**2 / self.c0**3 - self.c2 / self.c0**2,
        )

    def __truediv__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        return self * as_poly(other).reciprocal()

    def __rtruediv__(self, other: float | "PolyFnl2") -> "PolyFnl2":
        return as_poly(other) * self.reciprocal()

    def evaluate(self, f_nl: float) -> float:
        value = float(f_nl)
        if not math.isfinite(value):
            raise ValueError("f_nl must be finite")
        return self.c0 + value * self.c1 + value**2 * self.c2


@dataclass(frozen=True)
class FinitePngComponents:
    """Coefficients of ``G + f_NL L + f_NL**2 Q``."""

    G: float
    L: float
    Q: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            object.__setattr__(self, item.name, value)

    @classmethod
    def from_poly(cls, poly: PolyFnl2) -> "FinitePngComponents":
        return cls(poly.c0, poly.c1, poly.c2)

    def as_poly(self) -> PolyFnl2:
        return PolyFnl2(self.G, self.L, self.Q)

    def predict(self, f_nl: float) -> float:
        return self.as_poly().evaluate(f_nl)

    def __add__(self, other: "FinitePngComponents") -> "FinitePngComponents":
        return FinitePngComponents(
            self.G + other.G,
            self.L + other.L,
            self.Q + other.Q,
        )

    def __mul__(self, scalar: float) -> "FinitePngComponents":
        value = float(scalar)
        return FinitePngComponents(value * self.G, value * self.L, value * self.Q)

    __rmul__ = __mul__


@dataclass(frozen=True)
class HaloBiasV1:
    """Current MARISA-B halo-bias-v1 parameters."""

    b1: float
    b2: float
    bK2: float
    bphi: float
    bphidelta: float
    bphi2: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            object.__setattr__(self, item.name, value)


@dataclass(frozen=True)
class RsdConfig:
    """Plane-parallel RSD settings.

    ``f_growth`` is the logarithmic growth rate d ln D / d ln a.  It is
    intentionally not named ``f`` to avoid confusion with ``f_NL``.
    """

    f_growth: float
    los: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.f_growth):
            raise ValueError("f_growth must be finite")
        vector = _unit_vector(self.los, "los")
        object.__setattr__(self, "los", tuple(float(x) for x in vector))

    @property
    def los_array(self) -> NDArray[np.float64]:
        return np.asarray(self.los, dtype=np.float64)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Tree-level post-reconstruction source-kernel configuration.

    The production-compatible default is ``denominator_mode='bias_only'`` and
    ``f_nl_recon_ratio=0``.  This is the same-shift/no-linear-RSD-removal map
    currently represented in MARISA-B.

    ``denominator_mode='kaiser'`` is exposed only as an explicit alternative
    for future data-pipeline matching.  It changes the shift estimator to
    divide by b_rec + f_rec mu**2 and must not be enabled unless the catalog
    reconstruction actually used that prescription.

    If ``f_nl_recon_ratio=eta`` is non-zero, the denominator is

        d(k; f_NL) = d0(k) + eta f_NL bphi_rec / M(k),

    and is expanded analytically through f_NL**2.
    """

    smoothing_radius: float
    bias_recon: float
    cell_size: float = 0.0
    cic_window_power: int = 0
    denominator_mode: ReconstructionDenominator = "bias_only"
    f_growth_recon: float = 0.0
    f_nl_recon_ratio: float = 0.0
    bphi_recon: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.smoothing_radius) or self.smoothing_radius < 0.0:
            raise ValueError("smoothing_radius must be finite and non-negative")
        if not math.isfinite(self.bias_recon) or self.bias_recon == 0.0:
            raise ValueError("bias_recon must be finite and non-zero")
        if not math.isfinite(self.cell_size) or self.cell_size < 0.0:
            raise ValueError("cell_size must be finite and non-negative")
        if int(self.cic_window_power) != self.cic_window_power or self.cic_window_power < 0:
            raise ValueError("cic_window_power must be a non-negative integer")
        if self.denominator_mode not in {"bias_only", "kaiser"}:
            raise ValueError("unsupported denominator_mode")
        for name in ("f_growth_recon", "f_nl_recon_ratio", "bphi_recon"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RsdTreeResult:
    """Tree result plus audit components."""

    total: FinitePngComponents
    gravitational_and_bias: FinitePngComponents
    primordial: FinitePngComponents
    pair_terms: tuple[FinitePngComponents, FinitePngComponents, FinitePngComponents]
    z1: tuple[PolyFnl2, PolyFnl2, PolyFnl2]
    z2: tuple[PolyFnl2, PolyFnl2, PolyFnl2]


@dataclass(frozen=True)
class DesiTreeStochasticBases:
    """Gaussian tree-level stochastic shapes used in EFT-style analyses.

    The full contribution is

        B_stoch = Bshot_basis * B_shot / nbar
                + Pshot_basis * P_shot / nbar
                + Ashot_basis * A_shot / nbar**2.

    These are *Gaussian* RSD stochastic shapes.  They do not assert a unique
    finite-PNG stochastic closure.
    """

    Bshot_basis: float
    Pshot_basis: float
    Ashot_basis: float = 1.0


@dataclass(frozen=True)
class KernelPair:
    """First- and second-order kernels for one cyclic pair."""

    z1_i: PolyFnl2
    z1_j: PolyFnl2
    z2_ij: PolyFnl2


def as_poly(value: float | PolyFnl2) -> PolyFnl2:
    return value if isinstance(value, PolyFnl2) else PolyFnl2(float(value), 0.0, 0.0)


def _unit_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return array / norm


def _vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector")
    if float(np.linalg.norm(array)) <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return array


def validate_closed_triangle(
    vectors: ArrayLike,
    *,
    relative_tolerance: float = 2.0e-11,
) -> NDArray[np.float64]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite array with shape (3,3)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("all external wavevectors must be non-zero")
    scale = max(float(np.max(norms)), 1.0)
    if float(np.linalg.norm(np.sum(array, axis=0))) > relative_tolerance * scale:
        raise ValueError("external wavevectors do not close")
    return array


def _three_positive(values: Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain three finite positive values")
    return array


def cosine_between(k1: ArrayLike, k2: ArrayLike) -> float:
    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))


def f2_kernel(k1: ArrayLike, k2: ArrayLike) -> float:
    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    ka = float(np.linalg.norm(a))
    kb = float(np.linalg.norm(b))
    nu = cosine_between(a, b)
    return 5.0 / 7.0 + 0.5 * nu * (ka / kb + kb / ka) + 2.0 / 7.0 * nu**2


def g2_kernel(k1: ArrayLike, k2: ArrayLike) -> float:
    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    ka = float(np.linalg.norm(a))
    kb = float(np.linalg.norm(b))
    nu = cosine_between(a, b)
    return 3.0 / 7.0 + 0.5 * nu * (ka / kb + kb / ka) + 4.0 / 7.0 * nu**2


def tidal_s2(k1: ArrayLike, k2: ArrayLike) -> float:
    nu = cosine_between(k1, k2)
    return nu**2 - 1.0 / 3.0


def mu_los(k: ArrayLike, rsd: RsdConfig) -> float:
    vector = _vector3(k, "k")
    return float(np.dot(vector, rsd.los_array) / np.linalg.norm(vector))


def z1_rsd_poly(
    k: ArrayLike,
    transfer_m: float,
    bias: HaloBiasV1,
    rsd: RsdConfig,
) -> PolyFnl2:
    m = float(transfer_m)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("transfer_m must be finite and positive")
    mu = mu_los(k, rsd)
    return PolyFnl2(
        bias.b1 + rsd.f_growth * mu**2,
        bias.bphi / m,
        0.0,
    )


def real_space_k2_poly(
    k1: ArrayLike,
    k2: ArrayLike,
    transfer_m1: float,
    transfer_m2: float,
    bias: HaloBiasV1,
) -> PolyFnl2:
    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    m1 = float(transfer_m1)
    m2 = float(transfer_m2)
    if m1 <= 0.0 or m2 <= 0.0 or not math.isfinite(m1 + m2):
        raise ValueError("transfer functions must be finite and positive")
    k1n = float(np.linalg.norm(a))
    k2n = float(np.linalg.norm(b))
    nu = cosine_between(a, b)
    u1 = 1.0 / m1
    u2 = 1.0 / m2
    h = bias.b1 * f2_kernel(a, b) + 0.5 * bias.b2 + bias.bK2 * tidal_s2(a, b)
    advection = nu * (k1n / k2n * u1 + k2n / k1n * u2)
    linear_png = 0.5 * (bias.bphi * advection + bias.bphidelta * (u1 + u2))
    quadratic_png = 0.5 * bias.bphi2 * u1 * u2
    return PolyFnl2(h, linear_png, quadratic_png)


def z2_rsd_poly(
    k1: ArrayLike,
    k2: ArrayLike,
    transfer_m1: float,
    transfer_m2: float,
    bias: HaloBiasV1,
    rsd: RsdConfig,
) -> PolyFnl2:
    """Minimal no-selection/no-velocity-bias redshift-space Z2 kernel."""

    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    total = a + b
    total_norm = float(np.linalg.norm(total))
    if total_norm <= 1.0e-14:
        raise ValueError("the pair sum is singular")

    z1_a = z1_rsd_poly(a, transfer_m1, bias, rsd)
    z1_b = z1_rsd_poly(b, transfer_m2, bias, rsd)
    base = real_space_k2_poly(a, b, transfer_m1, transfer_m2, bias)

    total_parallel = float(np.dot(total, rsd.los_array))
    mu_total = total_parallel / total_norm
    velocity_growth = rsd.f_growth * mu_total**2 * g2_kernel(a, b)

    ka = float(np.linalg.norm(a))
    kb = float(np.linalg.norm(b))
    mu_a = mu_los(a, rsd)
    mu_b = mu_los(b, rsd)
    mapping = 0.5 * rsd.f_growth * total_parallel * (
        (mu_a / ka) * z1_b + (mu_b / kb) * z1_a
    )
    return base + velocity_growth + mapping


def gaussian_cic_window(k: ArrayLike, recon: ReconstructionConfig) -> float:
    vector = _vector3(k, "k")
    knorm = float(np.linalg.norm(vector))
    window = math.exp(-0.5 * (knorm * recon.smoothing_radius) ** 2)
    if recon.cell_size > 0.0 and recon.cic_window_power > 0:
        one_dimensional = np.sinc(vector * recon.cell_size / (2.0 * math.pi))
        window *= float(np.prod(one_dimensional) ** recon.cic_window_power)
    return window


def reconstruction_shift_poly(
    total_wavevector: ArrayLike,
    block_wavevector: ArrayLike,
    block_transfer_m: float,
    rsd: RsdConfig,
    recon: ReconstructionConfig,
) -> PolyFnl2:
    """One reconstructed displacement block, expanded through f_NL**2."""

    total = _vector3(total_wavevector, "total_wavevector")
    block = _vector3(block_wavevector, "block_wavevector")
    q2 = float(np.dot(block, block))
    numerator = -float(np.dot(total, block)) / q2 * gaussian_cic_window(block, recon)

    d0 = recon.bias_recon
    if recon.denominator_mode == "kaiser":
        d0 += recon.f_growth_recon * mu_los(block, rsd) ** 2
    d1 = recon.f_nl_recon_ratio * recon.bphi_recon / float(block_transfer_m)
    denominator = PolyFnl2(d0, d1, 0.0)
    return numerator / denominator


def z2_post_recon_rsd_poly(
    k1: ArrayLike,
    k2: ArrayLike,
    transfer_m1: float,
    transfer_m2: float,
    bias: HaloBiasV1,
    rsd: RsdConfig,
    recon: ReconstructionConfig,
) -> PolyFnl2:
    """Same-shift/no-linear-RSD-removal reconstructed Z2 kernel."""

    a = _vector3(k1, "k1")
    b = _vector3(k2, "k2")
    total = a + b
    z1_a = z1_rsd_poly(a, transfer_m1, bias, rsd)
    z1_b = z1_rsd_poly(b, transfer_m2, bias, rsd)
    z2_pre = z2_rsd_poly(a, b, transfer_m1, transfer_m2, bias, rsd)
    shift_a = reconstruction_shift_poly(total, a, transfer_m1, rsd, recon)
    shift_b = reconstruction_shift_poly(total, b, transfer_m2, rsd, recon)
    return z2_pre + 0.5 * (shift_a + shift_b) * z1_a * z1_b


def primordial_b0_hat(power_linear: Sequence[float], transfer_m: Sequence[float]) -> float:
    p = _three_positive(power_linear, "power_linear")
    m = _three_positive(transfer_m, "transfer_m")
    pairs = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    return float(sum(2.0 * p[i] * p[j] * m[k] / (m[i] * m[j]) for i, j, k in pairs))


def finite_png_halo_bispectrum_rsd(
    vectors: ArrayLike,
    power_linear: Sequence[float],
    transfer_m: Sequence[float],
    bias: HaloBiasV1,
    rsd: RsdConfig,
    *,
    reconstruction: ReconstructionConfig | None = None,
) -> RsdTreeResult:
    """Evaluate the oriented pre/post tree-level finite-PNG halo bispectrum.

    Parameters are evaluated at the three external magnitudes.  The returned
    coefficients are exact through f_NL**2 within the supplied bias basis.
    """

    vecs = validate_closed_triangle(vectors)
    p = _three_positive(power_linear, "power_linear")
    m = _three_positive(transfer_m, "transfer_m")
    z1 = tuple(z1_rsd_poly(vecs[i], m[i], bias, rsd) for i in range(3))

    pairs = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    pair_polys: list[PolyFnl2] = []
    z2_polys: list[PolyFnl2] = []
    grav = PolyFnl2()
    for i, j, _ in pairs:
        if reconstruction is None:
            z2 = z2_rsd_poly(vecs[i], vecs[j], m[i], m[j], bias, rsd)
        else:
            z2 = z2_post_recon_rsd_poly(
                vecs[i], vecs[j], m[i], m[j], bias, rsd, reconstruction
            )
        z2_polys.append(z2)
        pair = 2.0 * p[i] * p[j] * z1[i] * z1[j] * z2
        pair_polys.append(pair)
        grav = grav + pair

    b0hat = primordial_b0_hat(p, m)
    primordial = PolyFnl2(0.0, b0hat, 0.0) * z1[0] * z1[1] * z1[2]
    total = grav + primordial
    return RsdTreeResult(
        total=FinitePngComponents.from_poly(total),
        gravitational_and_bias=FinitePngComponents.from_poly(grav),
        primordial=FinitePngComponents.from_poly(primordial),
        pair_terms=tuple(FinitePngComponents.from_poly(x) for x in pair_polys),  # type: ignore[arg-type]
        z1=z1,  # type: ignore[arg-type]
        z2=tuple(z2_polys),  # type: ignore[arg-type]
    )


def desi_gaussian_tree_stochastic_bases(
    vectors: ArrayLike,
    power_linear: Sequence[float],
    b1: float,
    rsd: RsdConfig,
) -> DesiTreeStochasticBases:
    """Return the Gaussian tree stochastic shapes used in EFT analyses.

    This is Eq. (59)-type bookkeeping written without nbar and nuisance
    amplitudes.  At f_growth=0 the B-shot shape reduces to MARISA-B's current
    ``b1**2 * sum(P_i)`` raw alpha3 basis, while the P-shot shape vanishes.
    """

    vecs = validate_closed_triangle(vectors)
    p = _three_positive(power_linear, "power_linear")
    b1_value = float(b1)
    if not math.isfinite(b1_value):
        raise ValueError("b1 must be finite")
    b_basis = 0.0
    p_basis = 0.0
    for index in range(3):
        mu2 = mu_los(vecs[index], rsd) ** 2
        z1_gaussian = b1_value + rsd.f_growth * mu2
        b_basis += p[index] * b1_value * z1_gaussian
        p_basis += p[index] * rsd.f_growth * mu2 * z1_gaussian
    return DesiTreeStochasticBases(float(b_basis), float(p_basis), 1.0)


def triangle_vectors_from_shape_orientation(
    k1: float,
    k2: float,
    k3: float,
    mu1: float,
    phi: float,
    *,
    los: ArrayLike = (0.0, 0.0, 1.0),
) -> NDArray[np.float64]:
    """Construct a closed triangle with fixed shape and LOS orientation.

    ``mu1`` is the cosine between k1 and the LOS.  ``phi`` rotates k2 around
    k1.  The convention gives

        mu2 = mu1*nu12 - sqrt(1-mu1**2)*sqrt(1-nu12**2)*cos(phi).
    """

    lengths = np.asarray([k1, k2, k3], dtype=np.float64)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("k1, k2, k3 must be finite and positive")
    if k1 + k2 < k3 - 1.0e-13 or k1 + k3 < k2 - 1.0e-13 or k2 + k3 < k1 - 1.0e-13:
        raise ValueError("the supplied lengths do not satisfy the triangle inequality")
    mu = float(mu1)
    if not math.isfinite(mu) or abs(mu) > 1.0 + 1.0e-13:
        raise ValueError("mu1 must lie in [-1,1]")
    mu = float(np.clip(mu, -1.0, 1.0))
    if not math.isfinite(phi):
        raise ValueError("phi must be finite")

    n = _unit_vector(los, "los")
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, n))) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    ex = reference - np.dot(reference, n) * n
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)

    transverse = math.sqrt(max(0.0, 1.0 - mu**2))
    khat1 = transverse * ex + mu * n
    perp1 = mu * ex - transverse * n
    perp2 = ey

    nu12 = (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)
    if abs(nu12) > 1.0 + 2.0e-12:
        raise ValueError("invalid triangle opening cosine")
    nu12 = float(np.clip(nu12, -1.0, 1.0))
    opening_sine = math.sqrt(max(0.0, 1.0 - nu12**2))
    khat2 = nu12 * khat1 + opening_sine * (
        math.cos(phi) * perp1 + math.sin(phi) * perp2
    )

    vec1 = k1 * khat1
    vec2 = k2 * khat2
    vec3 = -(vec1 + vec2)
    result = np.vstack((vec1, vec2, vec3))
    if not np.isclose(np.linalg.norm(vec3), k3, rtol=2.0e-11, atol=2.0e-13):
        raise RuntimeError("internal triangle construction failed")
    return result


def project_scoccimarro_multipole(
    k1: float,
    k2: float,
    k3: float,
    power_linear: Sequence[float],
    transfer_m: Sequence[float],
    bias: HaloBiasV1,
    rsd: RsdConfig,
    *,
    ell: int = 0,
    n_mu: int = 18,
    n_phi: int = 20,
    reconstruction: ReconstructionConfig | None = None,
) -> FinitePngComponents:
    """Continuum orientation projection for the common B_ell convention.

    The normalization is

        B_ell = (2 ell + 1)/(4 pi) int_{-1}^1 dmu1 int_0^{2pi} dphi
                L_ell(mu1) B(k1,k2,k3;mu1,phi).

    ``ell=0`` is the ordinary angle-averaged monopole.  Higher multipoles are
    the Scoccimarro/Legendre convention tied to k1, not a TriPoSH B_l1l2L
    normalization.
    """

    if int(ell) != ell or ell < 0:
        raise ValueError("ell must be a non-negative integer")
    if n_mu < 2 or n_phi < 2:
        raise ValueError("n_mu and n_phi must be at least two")

    mu_nodes, mu_weights = leggauss(int(n_mu))
    phi_nodes_raw, phi_weights_raw = leggauss(int(n_phi))
    phi_nodes = math.pi * (phi_nodes_raw + 1.0)
    phi_weights = math.pi * phi_weights_raw
    legendre_coefficients = np.zeros(int(ell) + 1)
    legendre_coefficients[int(ell)] = 1.0

    accumulator = PolyFnl2()
    for mu, w_mu in zip(mu_nodes, mu_weights):
        legendre = float(legval(mu, legendre_coefficients))
        for phi, w_phi in zip(phi_nodes, phi_weights):
            vectors = triangle_vectors_from_shape_orientation(
                k1, k2, k3, float(mu), float(phi), los=rsd.los_array
            )
            result = finite_png_halo_bispectrum_rsd(
                vectors,
                power_linear,
                transfer_m,
                bias,
                rsd,
                reconstruction=reconstruction,
            )
            accumulator = accumulator + (w_mu * w_phi * legendre) * result.total.as_poly()
    normalization = (2.0 * ell + 1.0) / (4.0 * math.pi)
    return FinitePngComponents.from_poly(normalization * accumulator)


def damping_from_covariance(k: ArrayLike, covariance: ArrayLike) -> float:
    """Return exp[-k_i Sigma_ij k_j] in MARISA-B's scalar convention."""

    vector = _vector3(k, "k")
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 3x3 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-13):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) < -1.0e-12:
        raise ValueError("covariance must be positive semidefinite")
    return math.exp(-float(vector @ matrix @ vector))


def pre_recon_rsd_ir_covariance(sigma2: float, rsd: RsdConfig) -> NDArray[np.float64]:
    """Leading plane-parallel RSD BAO displacement covariance.

    This follows MARISA-B's existing real-space convention in which a scalar
    covariance gives exp(-k**2 sigma2).  With R = I + f_growth n n^T,

        Sigma_s = sigma2 R R^T,

    and the damping exponent is

        -k**2 sigma2 [1 + f_growth(2+f_growth) mu**2].

    Post-reconstruction requires a reconstruction-specific covariance and is
    intentionally not guessed here.
    """

    value = float(sigma2)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("sigma2 must be finite and non-negative")
    n = rsd.los_array
    response = np.eye(3) + rsd.f_growth * np.outer(n, n)
    return value * (response @ response.T)


__all__ = [
    "DesiTreeStochasticBases",
    "FinitePngComponents",
    "HaloBiasV1",
    "KernelPair",
    "PolyFnl2",
    "ReconstructionConfig",
    "RsdConfig",
    "RsdTreeResult",
    "cosine_between",
    "damping_from_covariance",
    "desi_gaussian_tree_stochastic_bases",
    "f2_kernel",
    "finite_png_halo_bispectrum_rsd",
    "g2_kernel",
    "gaussian_cic_window",
    "mu_los",
    "pre_recon_rsd_ir_covariance",
    "primordial_b0_hat",
    "project_scoccimarro_multipole",
    "real_space_k2_poly",
    "reconstruction_shift_poly",
    "tidal_s2",
    "triangle_vectors_from_shape_orientation",
    "validate_closed_triangle",
    "z1_rsd_poly",
    "z2_post_recon_rsd_poly",
    "z2_rsd_poly",
]
