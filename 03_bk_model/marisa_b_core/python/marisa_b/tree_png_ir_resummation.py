"""Tree-level IR resummation for the finite-local-PNG halo bispectrum.

This module is a deliberately self-contained Python oracle for MARISA-B's
real-space halo ``G + fNL L + fNL**2 Q`` tree sector.  It does not call or
match any one-loop contribution.

The fiducial scheme, ``strict_lo``, implements the controlled leading-IR,
leading-wiggle result

    X_IR = X_nw + sum_i exp(-k_i**2 Sigma**2) dX/d eps_i |_{eps=0},

for every deterministic and stochastic tree component X.  The legwise
variation holds the primordial spectrum fixed,

    P_i(eps) = P_nw,i + eps P_w,i,
    M_i(eps) = M_nw,i sqrt(1 + eps P_w,i / P_nw,i),

so that P_phi = P_L / M**2 is unchanged.  This consistently resums BAO
wiggles in P_L, M, 1/M, the primordial B111 term, PNG bias terms, advection,
and the fixed-b_rec reconstruction source kernel.

``factorized_closure`` instead evaluates the complete fixed-order tree
functional at P_IR = P_nw + exp(-k**2 Sigma**2) P_w and the corresponding
M_IR.  It agrees with ``strict_lo`` through O(P_w), but retains unproven
higher-wiggle products and is intended as a systematic variant, not the
fiducial result.

Conventions follow ``src/marisa_b/marisa_b_native.cpp``:

* b2 multiplies delta^2 / 2;
* bk2 multiplies mu^2 - 1/3;
* bphi2 is the independent phi^2-bias coefficient, not bphi**2;
* Q is the coefficient of fNL**2, i.e. one half of the second derivative;
* stochastic quantities returned here are raw bases, before alpha/nbar
  amplitudes are applied by the likelihood layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson
from scipy.special import spherical_jn

IrScheme = Literal["strict_lo", "factorized_closure"]


# Atomic term names intentionally track the current native ComponentResult
# fields where possible, making per-component regression straightforward.
GAUSSIAN_TERMS = (
    "Btree_gaussian_pre",
    "Btree_gaussian_reconstruction",
)
LINEAR_TERMS = (
    "dBdfNL_local_primordial",
    "dBdfNL_local_bphi_f2",
    "dBdfNL_local_bphi_advection",
    "dBdfNL_local_bphidelta",
    "dBdfNL_local_bphi_b2",
    "dBdfNL_local_bphi_bK2",
    "dBdfNL_local_bphi_reconstruction",
)
QUADRATIC_TERMS = (
    "Bhalo_tree_fNL2_bphi_B0",
    "Bhalo_tree_fNL2_bphi_sq_advection",
    "Bhalo_tree_fNL2_bphi_sq_F2",
    "Bhalo_tree_fNL2_bphi_sq_b2",
    "Bhalo_tree_fNL2_bphi_sq_bK2",
    "Bhalo_tree_fNL2_bphi_sq_reconstruction",
    "Bhalo_tree_fNL2_bphi_bphidelta",
    "Bhalo_tree_fNL2_bphi2_operator",
)
STOCHASTIC_TERMS = (
    "stochastic_alpha3_basis",
    "dBdfNL_stochastic_alpha3_basis",
    "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis",
    "stochastic_alpha4_basis",
)
ALL_TERMS = GAUSSIAN_TERMS + LINEAR_TERMS + QUADRATIC_TERMS + STOCHASTIC_TERMS


@dataclass(frozen=True)
class TreeComponents:
    """Coefficients of ``G + fNL L + fNL**2 Q``."""

    gaussian: float
    linear: float
    quadratic: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            object.__setattr__(self, field.name, value)

    @property
    def G(self) -> float:
        return self.gaussian

    @property
    def L(self) -> float:
        return self.linear

    @property
    def Q(self) -> float:
        return self.quadratic

    def predict(self, fnl: float) -> float:
        value = float(fnl)
        if not math.isfinite(value):
            raise ValueError("fNL must be finite")
        return self.gaussian + value * self.linear + value**2 * self.quadratic


@dataclass(frozen=True)
class StochasticBases:
    """Raw stochastic tree bases before alpha/nbar amplitudes are applied."""

    alpha3_gaussian: float
    alpha3_linear_png: float
    alpha3_quadratic_png: float
    alpha4_constant: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True)
class HaloTreeBias:
    """MARISA-B native halo-bias-v1 convention.

    Callers working in the EFT ``(b2_EFT, gamma2)`` basis must first use the
    existing MARISA-B conversion ``b2_native=b2_EFT-4 gamma2/3`` and
    ``bk2=gamma2``.
    """

    b1: float
    b2: float
    bk2: float
    bphi: float
    bphidelta: float
    bphi2: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Fixed-b_rec reconstruction source-kernel settings.

    ``None`` means pre-reconstruction.  A non-zero CIC window reproduces the
    orientation-dependent source kernel in the native code, but a scalar
    ``sigma2`` is not a complete IR damping model for anisotropic CIC.  Use
    CIC-off for the first production validation.
    """

    smoothing_radius: float
    bias_recon: float
    cell_size: float = 0.0
    cic_window_power: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.smoothing_radius) or self.smoothing_radius < 0.0:
            raise ValueError("smoothing_radius must be finite and non-negative")
        if not math.isfinite(self.bias_recon) or self.bias_recon == 0.0:
            raise ValueError("bias_recon must be finite and non-zero")
        if not math.isfinite(self.cell_size) or self.cell_size < 0.0:
            raise ValueError("cell_size must be finite and non-negative")
        if int(self.cic_window_power) != self.cic_window_power:
            raise ValueError("cic_window_power must be an integer")
        if self.cic_window_power < 0:
            raise ValueError("cic_window_power must be non-negative")


@dataclass(frozen=True)
class TreeIrConfig:
    """Tree-only IR configuration.

    No ``(1+k^2 Sigma^2)`` matching option is provided: that counterterm is
    required only when a hard one-loop contribution is included and matched.
    """

    sigma2: float
    scheme: IrScheme = "strict_lo"

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma2) or self.sigma2 < 0.0:
            raise ValueError("sigma2 must be finite and non-negative")
        if self.scheme not in {"strict_lo", "factorized_closure"}:
            raise ValueError(f"unsupported IR scheme {self.scheme!r}")


@dataclass(frozen=True)
class TreeIrResult:
    """IR-resummed deterministic/stochastic components and audit metadata."""

    deterministic: TreeComponents
    stochastic: StochasticBases
    operator_terms: Mapping[str, float]
    damping: tuple[float, float, float]
    sigma2: float
    scheme: IrScheme

    def __post_init__(self) -> None:
        missing = set(ALL_TERMS).difference(self.operator_terms)
        if missing:
            raise ValueError(f"operator_terms is missing {sorted(missing)}")
        frozen = MappingProxyType(
            {name: float(self.operator_terms[name]) for name in ALL_TERMS}
        )
        object.__setattr__(self, "operator_terms", frozen)
        if len(self.damping) != 3 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.damping
        ):
            raise ValueError("damping must contain three finite values in [0,1]")

    def predict_deterministic(self, fnl: float) -> float:
        return self.deterministic.predict(fnl)


@dataclass(frozen=True)
class _Dual:
    """First-order dual number for an exact legwise O(P_w) derivative."""

    value: float
    tangent: float = 0.0

    def __add__(self, other: Scalar) -> _Dual:
        rhs = _as_dual(other)
        return _Dual(self.value + rhs.value, self.tangent + rhs.tangent)

    __radd__ = __add__

    def __sub__(self, other: Scalar) -> _Dual:
        rhs = _as_dual(other)
        return _Dual(self.value - rhs.value, self.tangent - rhs.tangent)

    def __rsub__(self, other: Scalar) -> _Dual:
        lhs = _as_dual(other)
        return _Dual(lhs.value - self.value, lhs.tangent - self.tangent)

    def __mul__(self, other: Scalar) -> _Dual:
        rhs = _as_dual(other)
        return _Dual(
            self.value * rhs.value,
            self.tangent * rhs.value + self.value * rhs.tangent,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Scalar) -> _Dual:
        rhs = _as_dual(other)
        if rhs.value == 0.0:
            raise ZeroDivisionError("dual-number division by zero")
        inverse = 1.0 / rhs.value
        return _Dual(
            self.value * inverse,
            (self.tangent * rhs.value - self.value * rhs.tangent) * inverse**2,
        )

    def __rtruediv__(self, other: Scalar) -> _Dual:
        return _as_dual(other).__truediv__(self)

    def __pow__(self, exponent: int) -> _Dual:
        if not isinstance(exponent, int):
            raise TypeError("dual numbers only support integer powers")
        if exponent == 0:
            return _Dual(1.0, 0.0)
        if exponent < 0:
            return _Dual(1.0, 0.0) / (self ** (-exponent))
        return _Dual(
            self.value**exponent,
            exponent * self.value ** (exponent - 1) * self.tangent,
        )

    def __neg__(self) -> _Dual:
        return _Dual(-self.value, -self.tangent)


Scalar: TypeAlias = float | _Dual
RawTerms: TypeAlias = dict[str, Scalar]


def _as_dual(value: Scalar) -> _Dual:
    return value if isinstance(value, _Dual) else _Dual(float(value))


def _vector3(value: ArrayLike, name: str, *, positive: bool = False) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _closed_vectors(value: ArrayLike) -> NDArray[np.float64]:
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.shape != (3, 3):
        raise ValueError("vectors must have shape (3,3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("triangle vectors must be non-zero")
    scale = float(np.max(norms))
    if np.linalg.norm(np.sum(vectors, axis=0)) > 2.0e-12 * scale:
        raise ValueError("triangle vectors are not closed")
    return vectors


def triangle_vectors(k1: float, k2: float, mu12: float) -> NDArray[np.float64]:
    """Construct the closed orientation used by the native triangle helper."""

    k1_value = float(k1)
    k2_value = float(k2)
    mu_value = float(mu12)
    if not math.isfinite(k1_value) or k1_value <= 0.0:
        raise ValueError("k1 must be finite and positive")
    if not math.isfinite(k2_value) or k2_value <= 0.0:
        raise ValueError("k2 must be finite and positive")
    if not math.isfinite(mu_value) or not -1.0 <= mu_value <= 1.0:
        raise ValueError("mu12 must lie in [-1,1]")
    transverse = k2_value * math.sqrt(max(0.0, 1.0 - mu_value**2))
    first = np.array([0.0, 0.0, k1_value], dtype=np.float64)
    second = np.array([transverse, 0.0, k2_value * mu_value], dtype=np.float64)
    third = -(first + second)
    return np.stack((first, second, third))


def _f2(ki: float, kj: float, mu: float) -> float:
    return 5.0 / 7.0 + 0.5 * mu * (ki / kj + kj / ki) + 2.0 * mu**2 / 7.0


def _cic_window(vector: NDArray[np.float64], config: ReconstructionConfig) -> float:
    if config.cell_size <= 0.0 or config.cic_window_power <= 0:
        return 1.0
    arguments = vector * config.cell_size / (2.0 * math.pi)
    # np.sinc(x)=sin(pi*x)/(pi*x), matching the native sinc convention.
    return float(np.prod(np.sinc(arguments))) ** config.cic_window_power


def _shift_factor(
    total: NDArray[np.float64],
    block: NDArray[np.float64],
    config: ReconstructionConfig,
) -> float:
    q2 = float(np.dot(block, block))
    if q2 <= 1.0e-24:
        return 0.0
    q = math.sqrt(q2)
    window = math.exp(-0.5 * (q * config.smoothing_radius) ** 2)
    window *= _cic_window(block, config)
    return -float(np.dot(total, block)) * window / (q2 * config.bias_recon)


def _zero_terms() -> RawTerms:
    terms: RawTerms = {name: 0.0 for name in ALL_TERMS}
    terms["stochastic_alpha4_basis"] = 1.0
    return terms


def _finite_png_halo_tree_terms_raw(
    kvec: NDArray[np.float64],
    power: Sequence[Scalar],
    transfer: Sequence[Scalar],
    bias: HaloTreeBias,
    reconstruction: ReconstructionConfig | None,
) -> RawTerms:
    """Evaluate atomic native-compatible tree terms with float or dual inputs."""

    terms = _zero_terms()
    norms = np.linalg.norm(kvec, axis=1)
    inverse_m: list[Scalar] = [1.0 / item for item in transfer]

    b0_hat: Scalar = 0.0
    for index in range(3):
        terms["stochastic_alpha3_basis"] += bias.b1**2 * power[index]
        terms["dBdfNL_stochastic_alpha3_basis"] += (
            bias.b1 * bias.bphi * power[index] * inverse_m[index]
        )
        terms["Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"] += (
            bias.bphi**2 * power[index] * inverse_m[index] ** 2
        )

    for i, j, ell in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        mu = float(np.dot(kvec[i], kvec[j]) / (norms[i] * norms[j]))
        f2 = _f2(float(norms[i]), float(norms[j]), mu)
        s2 = mu**2 - 1.0 / 3.0
        ui = inverse_m[i]
        uj = inverse_m[j]
        u_sum = ui + uj
        u_product = ui * uj
        advection = mu * (
            norms[i] / norms[j] * ui + norms[j] / norms[i] * uj
        )
        pp = power[i] * power[j]

        terms["Btree_gaussian_pre"] += 2.0 * bias.b1**2 * pp * (
            bias.b1 * f2 + 0.5 * bias.b2 + bias.bk2 * s2
        )

        terms["dBdfNL_local_bphi_f2"] += (
            2.0 * bias.b1**2 * bias.bphi * pp * f2 * u_sum
        )
        terms["dBdfNL_local_bphi_advection"] += (
            bias.b1**2 * bias.bphi * pp * advection
        )
        terms["dBdfNL_local_bphidelta"] += (
            bias.b1**2 * bias.bphidelta * pp * u_sum
        )
        terms["dBdfNL_local_bphi_b2"] += (
            bias.b1 * bias.b2 * bias.bphi * pp * u_sum
        )
        terms["dBdfNL_local_bphi_bK2"] += (
            2.0 * bias.b1 * bias.bk2 * bias.bphi * pp * s2 * u_sum
        )

        terms["Bhalo_tree_fNL2_bphi_sq_advection"] += (
            bias.b1 * bias.bphi**2 * pp * u_sum * advection
        )
        terms["Bhalo_tree_fNL2_bphi_sq_F2"] += (
            2.0 * bias.b1 * bias.bphi**2 * pp * f2 * u_product
        )
        terms["Bhalo_tree_fNL2_bphi_sq_b2"] += (
            bias.b2 * bias.bphi**2 * pp * u_product
        )
        terms["Bhalo_tree_fNL2_bphi_sq_bK2"] += (
            2.0 * bias.bk2 * bias.bphi**2 * pp * s2 * u_product
        )
        terms["Bhalo_tree_fNL2_bphi_bphidelta"] += (
            bias.b1 * bias.bphi * bias.bphidelta * pp * u_sum**2
        )
        terms["Bhalo_tree_fNL2_bphi2_operator"] += (
            bias.b1**2 * bias.bphi2 * pp * u_product
        )

        if reconstruction is not None:
            total = kvec[i] + kvec[j]
            shift_sum = _shift_factor(total, kvec[i], reconstruction)
            shift_sum += _shift_factor(total, kvec[j], reconstruction)
            terms["Btree_gaussian_reconstruction"] += (
                pp * shift_sum * bias.b1**4
            )
            terms["dBdfNL_local_bphi_reconstruction"] += (
                2.0 * pp * shift_sum * bias.b1**3 * bias.bphi * u_sum
            )
            terms["Bhalo_tree_fNL2_bphi_sq_reconstruction"] += (
                pp
                * shift_sum
                * bias.b1**2
                * bias.bphi**2
                * (ui**2 + 4.0 * ui * uj + uj**2)
            )

        b0_hat += 2.0 * pp * transfer[ell] / (transfer[i] * transfer[j])

    terms["dBdfNL_local_primordial"] = bias.b1**3 * b0_hat
    terms["Bhalo_tree_fNL2_bphi_B0"] = (
        bias.b1**2 * bias.bphi * sum(inverse_m, 0.0) * b0_hat
    )
    return terms


def _values(raw: Mapping[str, Scalar]) -> dict[str, float]:
    return {
        name: float(value.value if isinstance(value, _Dual) else value)
        for name, value in raw.items()
    }


def _tangents(raw: Mapping[str, Scalar]) -> dict[str, float]:
    return {
        name: float(value.tangent if isinstance(value, _Dual) else 0.0)
        for name, value in raw.items()
    }


def _add_scaled(
    destination: dict[str, float],
    source: Mapping[str, float],
    scale: float,
) -> None:
    for name in ALL_TERMS:
        destination[name] += scale * source[name]


def _components_from_terms(terms: Mapping[str, float]) -> TreeComponents:
    return TreeComponents(
        gaussian=sum(terms[name] for name in GAUSSIAN_TERMS),
        linear=sum(terms[name] for name in LINEAR_TERMS),
        quadratic=sum(terms[name] for name in QUADRATIC_TERMS),
    )


def _stochastic_from_terms(terms: Mapping[str, float]) -> StochasticBases:
    return StochasticBases(
        alpha3_gaussian=terms["stochastic_alpha3_basis"],
        alpha3_linear_png=terms["dBdfNL_stochastic_alpha3_basis"],
        alpha3_quadratic_png=terms[
            "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"
        ],
        alpha4_constant=terms["stochastic_alpha4_basis"],
    )


def finite_png_halo_tree_operator_terms(
    vectors: ArrayLike,
    linear_power: ArrayLike,
    transfer_m: ArrayLike,
    bias: HaloTreeBias,
    reconstruction: ReconstructionConfig | None = None,
) -> dict[str, float]:
    """Return all deterministic and stochastic raw tree terms."""

    kvec = _closed_vectors(vectors)
    power = _vector3(linear_power, "linear_power", positive=True)
    transfer = _vector3(transfer_m, "transfer_m", positive=True)
    return _values(
        _finite_png_halo_tree_terms_raw(
            kvec, power.tolist(), transfer.tolist(), bias, reconstruction
        )
    )


def finite_png_halo_tree_components(
    vectors: ArrayLike,
    linear_power: ArrayLike,
    transfer_m: ArrayLike,
    bias: HaloTreeBias,
    reconstruction: ReconstructionConfig | None = None,
) -> TreeComponents:
    """Evaluate the fixed-order deterministic finite-PNG halo tree."""

    terms = finite_png_halo_tree_operator_terms(
        vectors, linear_power, transfer_m, bias, reconstruction
    )
    return _components_from_terms(terms)


def finite_png_stochastic_bases(
    vectors: ArrayLike,
    linear_power: ArrayLike,
    transfer_m: ArrayLike,
    bias: HaloTreeBias,
    reconstruction: ReconstructionConfig | None = None,
) -> StochasticBases:
    """Evaluate native raw stochastic bases.

    ``vectors`` and ``reconstruction`` are accepted for API symmetry; the
    current native stochastic bases are independent of reconstruction.
    """

    terms = finite_png_halo_tree_operator_terms(
        vectors, linear_power, transfer_m, bias, reconstruction
    )
    return _stochastic_from_terms(terms)


def no_wiggle_transfer(
    linear_power: ArrayLike,
    no_wiggle_power: ArrayLike,
    transfer_m: ArrayLike,
) -> NDArray[np.float64]:
    """Derive M_nw while preserving P_phi=P_L/M**2 exactly."""

    power = _vector3(linear_power, "linear_power", positive=True)
    no_wiggle = _vector3(no_wiggle_power, "no_wiggle_power", positive=True)
    transfer = _vector3(transfer_m, "transfer_m", positive=True)
    return transfer * np.sqrt(no_wiggle / power)


def _strict_leg_response(
    kvec: NDArray[np.float64],
    smooth_power: NDArray[np.float64],
    wiggle_power: NDArray[np.float64],
    smooth_transfer: NDArray[np.float64],
    bias: HaloTreeBias,
    reconstruction: ReconstructionConfig | None,
    leg: int,
) -> dict[str, float]:
    power: list[Scalar] = [float(item) for item in smooth_power]
    transfer: list[Scalar] = [float(item) for item in smooth_transfer]
    power[leg] = _Dual(smooth_power[leg], wiggle_power[leg])
    transfer[leg] = _Dual(
        smooth_transfer[leg],
        0.5
        * smooth_transfer[leg]
        * wiggle_power[leg]
        / smooth_power[leg],
    )
    raw = _finite_png_halo_tree_terms_raw(
        kvec, power, transfer, bias, reconstruction
    )
    return _tangents(raw)


def ir_resummed_finite_png_halo_tree(
    vectors: ArrayLike,
    linear_power: ArrayLike,
    no_wiggle_power: ArrayLike,
    transfer_m: ArrayLike,
    bias: HaloTreeBias,
    ir: TreeIrConfig,
    reconstruction: ReconstructionConfig | None = None,
) -> TreeIrResult:
    """IR-resum the complete current tree-level finite-PNG halo basis.

    The returned result includes deterministic G/L/Q, all current native
    stochastic raw bases, and the resummed atomic operator terms.
    """

    kvec = _closed_vectors(vectors)
    wave_numbers = np.linalg.norm(kvec, axis=1)
    full_power = _vector3(linear_power, "linear_power", positive=True)
    smooth_power = _vector3(no_wiggle_power, "no_wiggle_power", positive=True)
    full_transfer = _vector3(transfer_m, "transfer_m", positive=True)
    smooth_transfer = no_wiggle_transfer(
        full_power, smooth_power, full_transfer
    )
    wiggle_power = full_power - smooth_power
    damping_array = np.exp(-wave_numbers**2 * ir.sigma2)

    if ir.scheme == "strict_lo":
        terms = _values(
            _finite_png_halo_tree_terms_raw(
                kvec,
                smooth_power.tolist(),
                smooth_transfer.tolist(),
                bias,
                reconstruction,
            )
        )
        for leg in range(3):
            response = _strict_leg_response(
                kvec,
                smooth_power,
                wiggle_power,
                smooth_transfer,
                bias,
                reconstruction,
                leg,
            )
            _add_scaled(terms, response, float(damping_array[leg]))
    else:
        resummed_power = smooth_power + damping_array * wiggle_power
        if np.any(resummed_power <= 0.0):
            raise ValueError("factorized resummed power is non-positive")
        # Hold P_phi fixed under the dressed transfer.
        resummed_transfer = full_transfer * np.sqrt(
            resummed_power / full_power
        )
        terms = _values(
            _finite_png_halo_tree_terms_raw(
                kvec,
                resummed_power.tolist(),
                resummed_transfer.tolist(),
                bias,
                reconstruction,
            )
        )

    return TreeIrResult(
        deterministic=_components_from_terms(terms),
        stochastic=_stochastic_from_terms(terms),
        operator_terms=terms,
        damping=tuple(float(item) for item in damping_array),
        sigma2=ir.sigma2,
        scheme=ir.scheme,
    )


def bao_displacement_variance(
    q: ArrayLike,
    no_wiggle_power: ArrayLike,
    *,
    r_bao: float,
    k_ir: float,
    residual_filter: ArrayLike
    | Callable[[NDArray[np.float64]], ArrayLike]
    | None = None,
) -> float:
    r"""Compute the real-space BAO displacement variance.

    .. math::

       \Sigma^2 = \frac{1}{6\pi^2}\int_0^{k_{\rm IR}}dq\,P_{nw}(q)
       A(q)^2[1-j_0(qr_{BAO})+2j_2(qr_{BAO})].

    ``A(q)=1`` pre-reconstruction.  For MARISA-B's fixed-b_rec isotropic
    post convention, use :func:`gaussian_reconstruction_residual`.  Cartesian
    CIC and adaptive scale-dependent reconstruction require a tensor/general
    displacement treatment and should not be represented by this scalar.
    """

    q_array = np.asarray(q, dtype=np.float64)
    power = np.asarray(no_wiggle_power, dtype=np.float64)
    if q_array.ndim != 1 or power.shape != q_array.shape:
        raise ValueError("q and no_wiggle_power must be same-length 1D arrays")
    if q_array.size < 3:
        raise ValueError("at least three q samples are required")
    if not np.all(np.isfinite(q_array)) or not np.all(np.isfinite(power)):
        raise ValueError("q or no_wiggle_power contains non-finite values")
    if np.any(q_array <= 0.0) or np.any(power < 0.0):
        raise ValueError("q must be positive and no_wiggle_power non-negative")
    if np.any(np.diff(q_array) <= 0.0):
        raise ValueError("q must be strictly increasing")
    if not math.isfinite(r_bao) or r_bao <= 0.0:
        raise ValueError("r_bao must be finite and positive")
    if not math.isfinite(k_ir) or k_ir <= 0.0:
        raise ValueError("k_ir must be finite and positive")

    mask = q_array <= k_ir
    if np.count_nonzero(mask) < 3:
        raise ValueError("fewer than three q samples lie below k_ir")
    q_used = q_array[mask]
    power_used = power[mask]
    x = q_used * r_bao
    bao_window = 1.0 - spherical_jn(0, x) + 2.0 * spherical_jn(2, x)

    if residual_filter is None:
        residual = np.ones_like(q_used)
    elif callable(residual_filter):
        residual = np.asarray(residual_filter(q_used), dtype=np.float64)
    else:
        supplied = np.asarray(residual_filter, dtype=np.float64)
        if supplied.shape == q_array.shape:
            residual = supplied[mask]
        elif supplied.shape == q_used.shape:
            residual = supplied
        else:
            raise ValueError("residual_filter has incompatible shape")
    if residual.shape != q_used.shape or not np.all(np.isfinite(residual)):
        raise ValueError("residual_filter produced invalid values")

    integrand = power_used * residual**2 * bao_window / (6.0 * math.pi**2)
    sigma2 = float(simpson(integrand, x=q_used))
    if not math.isfinite(sigma2) or sigma2 < -1.0e-12:
        raise ValueError("computed Sigma^2 is invalid")
    return max(0.0, sigma2)


def gaussian_reconstruction_residual(
    q: ArrayLike,
    *,
    smoothing_radius: float,
    tracer_bias: float,
    bias_recon: float,
) -> NDArray[np.float64]:
    r"""Return ``A(q)=1-(b1/b_rec) exp[-q^2 R^2/2]``."""

    wave_numbers = np.asarray(q, dtype=np.float64)
    if np.any(~np.isfinite(wave_numbers)) or np.any(wave_numbers < 0.0):
        raise ValueError("q must be finite and non-negative")
    if not math.isfinite(smoothing_radius) or smoothing_radius < 0.0:
        raise ValueError("smoothing_radius must be finite and non-negative")
    if not math.isfinite(tracer_bias):
        raise ValueError("tracer_bias must be finite")
    if not math.isfinite(bias_recon) or bias_recon == 0.0:
        raise ValueError("bias_recon must be finite and non-zero")
    smoothing = np.exp(-0.5 * (wave_numbers * smoothing_radius) ** 2)
    return 1.0 - (tracer_bias / bias_recon) * smoothing
