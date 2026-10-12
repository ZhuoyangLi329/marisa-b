from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from marisa_b.tree_png_rsd_reference import (  # noqa: E402
    FinitePngComponents,
    HaloBiasV1,
    PolyFnl2,
    ReconstructionConfig,
    RsdConfig,
    damping_from_covariance,
    desi_gaussian_tree_stochastic_bases,
    f2_kernel,
    finite_png_halo_bispectrum_rsd,
    g2_kernel,
    pre_recon_rsd_ir_covariance,
    primordial_b0_hat,
    project_scoccimarro_multipole,
    real_space_k2_poly,
    reconstruction_shift_poly,
    tidal_s2,
    triangle_vectors_from_shape_orientation,
    validate_closed_triangle,
    z1_rsd_poly,
    z2_post_recon_rsd_poly,
    z2_rsd_poly,
)


VECTORS = np.array(
    [
        [0.052, 0.013, 0.027],
        [-0.018, 0.047, -0.011],
        [-0.034, -0.060, -0.016],
    ],
    dtype=float,
)
P = np.array([1180.0, 970.0, 830.0])
M = np.array([0.82, 1.07, 0.91])
BIAS = HaloBiasV1(
    b1=2.08,
    b2=0.41,
    bK2=-0.27,
    bphi=3.25,
    bphidelta=1.36,
    bphi2=2.42,
)
RSD = RsdConfig(f_growth=0.79, los=(0.2, -0.3, 0.9327379053))
RECON = ReconstructionConfig(
    smoothing_radius=15.0,
    bias_recon=2.4,
    cell_size=0.0,
    cic_window_power=0,
)


def _manual_real_space_components(vectors, power, transfer, bias):
    pairs = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    z1 = [PolyFnl2(bias.b1, bias.bphi / transfer[i], 0.0) for i in range(3)]
    total = PolyFnl2()
    for i, j, _ in pairs:
        z2 = real_space_k2_poly(
            vectors[i], vectors[j], transfer[i], transfer[j], bias
        )
        total += 2.0 * power[i] * power[j] * z1[i] * z1[j] * z2
    total += PolyFnl2(0.0, primordial_b0_hat(power, transfer), 0.0) * z1[0] * z1[1] * z1[2]
    return FinitePngComponents.from_poly(total)


def _standard_gaussian_z2(k1, k2, bias, f_growth, los):
    k1n = np.linalg.norm(k1)
    k2n = np.linalg.norm(k2)
    total = k1 + k2
    kn = np.linalg.norm(total)
    n = np.asarray(los, dtype=float)
    n /= np.linalg.norm(n)
    mu1 = np.dot(k1, n) / k1n
    mu2 = np.dot(k2, n) / k2n
    mu = np.dot(total, n) / kn
    z1_1 = bias.b1 + f_growth * mu1**2
    z1_2 = bias.b1 + f_growth * mu2**2
    return (
        bias.b1 * f2_kernel(k1, k2)
        + 0.5 * bias.b2
        + bias.bK2 * tidal_s2(k1, k2)
        + f_growth * mu**2 * g2_kernel(k1, k2)
        + 0.5
        * f_growth
        * mu
        * kn
        * (mu1 / k1n * z1_2 + mu2 / k2n * z1_1)
    )


def _rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ]
    )


def test_triangle_validation_rejects_nonclosure():
    validate_closed_triangle(VECTORS)
    broken = VECTORS.copy()
    broken[2, 0] += 1.0e-3
    with pytest.raises(ValueError, match="do not close"):
        validate_closed_triangle(broken)


def test_poly_arithmetic_and_reciprocal_are_second_order_exact():
    a = PolyFnl2(1.3, -0.4, 0.2)
    b = PolyFnl2(2.1, 0.7, -0.1)
    product = a * b
    expected = PolyFnl2(
        1.3 * 2.1,
        1.3 * 0.7 - 0.4 * 2.1,
        1.3 * -0.1 + -0.4 * 0.7 + 0.2 * 2.1,
    )
    assert product == expected
    quotient = a / b
    for h in (1.0e-4, 3.0e-5):
        exact = a.evaluate(h) / b.evaluate(h)
        assert quotient.evaluate(h) == pytest.approx(exact, rel=2.0e-9, abs=2.0e-12)


def test_fgrowth_zero_recovers_current_real_space_pre_model():
    expected = _manual_real_space_components(VECTORS, P, M, BIAS)
    actual = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, BIAS, RsdConfig(0.0, RSD.los)
    ).total
    assert actual.G == pytest.approx(expected.G, rel=2.0e-14)
    assert actual.L == pytest.approx(expected.L, rel=2.0e-14)
    assert actual.Q == pytest.approx(expected.Q, rel=2.0e-14)


def test_gaussian_limit_matches_standard_z1_z2_tree_formula():
    no_png_bias = HaloBiasV1(BIAS.b1, BIAS.b2, BIAS.bK2, 0.0, 0.0, 0.0)
    result = finite_png_halo_bispectrum_rsd(VECTORS, P, M, no_png_bias, RSD)
    manual = 0.0
    n = RSD.los_array
    pairs = ((0, 1), (1, 2), (2, 0))
    for i, j in pairs:
        mui = np.dot(VECTORS[i], n) / np.linalg.norm(VECTORS[i])
        muj = np.dot(VECTORS[j], n) / np.linalg.norm(VECTORS[j])
        z1i = BIAS.b1 + RSD.f_growth * mui**2
        z1j = BIAS.b1 + RSD.f_growth * muj**2
        z2 = _standard_gaussian_z2(
            VECTORS[i], VECTORS[j], BIAS, RSD.f_growth, n
        )
        manual += 2.0 * P[i] * P[j] * z1i * z1j * z2
    assert result.total.G == pytest.approx(manual, rel=2.0e-14)
    # Even with zero PNG bias, local primordial B contributes linearly in f_NL.
    assert result.total.L == pytest.approx(
        primordial_b0_hat(P, M)
        * np.prod(
            [
                BIAS.b1
                + RSD.f_growth
                * (np.dot(VECTORS[i], n) / np.linalg.norm(VECTORS[i])) ** 2
                for i in range(3)
            ]
        ),
        rel=2.0e-14,
    )
    assert result.total.Q == pytest.approx(0.0, abs=1.0e-12)


def test_rsd_z2_png_linear_coefficient_contains_mapping_cross_term():
    z2 = z2_rsd_poly(VECTORS[0], VECTORS[1], M[0], M[1], BIAS, RSD)
    real = real_space_k2_poly(VECTORS[0], VECTORS[1], M[0], M[1], BIAS)
    total = VECTORS[0] + VECTORS[1]
    n = RSD.los_array
    total_parallel = np.dot(total, n)
    mu1 = np.dot(VECTORS[0], n) / np.linalg.norm(VECTORS[0])
    mu2 = np.dot(VECTORS[1], n) / np.linalg.norm(VECTORS[1])
    expected_extra = (
        0.5
        * RSD.f_growth
        * total_parallel
        * (
            mu1 / np.linalg.norm(VECTORS[0]) * BIAS.bphi / M[1]
            + mu2 / np.linalg.norm(VECTORS[1]) * BIAS.bphi / M[0]
        )
    )
    assert z2.c1 - real.c1 == pytest.approx(expected_extra, rel=2.0e-14)
    assert z2.c2 == pytest.approx(real.c2, rel=2.0e-14)


def test_primordial_piece_has_correct_rsd_external_legs():
    result = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD)
    z1_0 = [z.c0 for z in result.z1]
    z1_1 = [z.c1 for z in result.z1]
    b0 = primordial_b0_hat(P, M)
    expected_l = b0 * z1_0[0] * z1_0[1] * z1_0[2]
    expected_q = b0 * (
        z1_1[0] * z1_0[1] * z1_0[2]
        + z1_0[0] * z1_1[1] * z1_0[2]
        + z1_0[0] * z1_0[1] * z1_1[2]
    )
    assert result.primordial.G == 0.0
    assert result.primordial.L == pytest.approx(expected_l, rel=2.0e-14)
    assert result.primordial.Q == pytest.approx(expected_q, rel=2.0e-14)


def test_post_reconstruction_leaves_z1_unchanged_and_modifies_z2():
    pre = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD)
    post = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, BIAS, RSD, reconstruction=RECON
    )
    assert post.z1 == pre.z1
    assert any(a != b for a, b in zip(post.z2, pre.z2))
    # Primordial tree is unchanged because Z1_rec = Z1.
    assert post.primordial == pre.primordial


def test_post_z2_matches_explicit_same_shift_formula():
    z2_pre = z2_rsd_poly(VECTORS[0], VECTORS[1], M[0], M[1], BIAS, RSD)
    z1_0 = z1_rsd_poly(VECTORS[0], M[0], BIAS, RSD)
    z1_1 = z1_rsd_poly(VECTORS[1], M[1], BIAS, RSD)
    total = VECTORS[0] + VECTORS[1]
    r0 = reconstruction_shift_poly(total, VECTORS[0], M[0], RSD, RECON)
    r1 = reconstruction_shift_poly(total, VECTORS[1], M[1], RSD, RECON)
    expected = z2_pre + 0.5 * (r0 + r1) * z1_0 * z1_1
    actual = z2_post_recon_rsd_poly(
        VECTORS[0], VECTORS[1], M[0], M[1], BIAS, RSD, RECON
    )
    assert actual == expected


def test_smoothing_radius_to_infinity_returns_pre_reconstruction():
    off = ReconstructionConfig(
        smoothing_radius=1.0e6,
        bias_recon=RECON.bias_recon,
    )
    pre = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD).total
    post = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, BIAS, RSD, reconstruction=off
    ).total
    assert post.G == pytest.approx(pre.G, rel=1.0e-14, abs=1.0e-8)
    assert post.L == pytest.approx(pre.L, rel=1.0e-14, abs=1.0e-8)
    assert post.Q == pytest.approx(pre.Q, rel=1.0e-14, abs=1.0e-8)


def test_adaptive_reconstruction_shift_coefficients_match_exact_rational_derivatives():
    adaptive = ReconstructionConfig(
        smoothing_radius=12.0,
        bias_recon=2.3,
        denominator_mode="bias_only",
        f_nl_recon_ratio=0.7,
        bphi_recon=2.9,
    )
    total = VECTORS[0] + VECTORS[1]
    poly = reconstruction_shift_poly(total, VECTORS[0], M[0], RSD, adaptive)
    q = VECTORS[0]
    numerator = (
        -np.dot(total, q)
        / np.dot(q, q)
        * math.exp(-0.5 * (np.linalg.norm(q) * adaptive.smoothing_radius) ** 2)
    )
    d0 = adaptive.bias_recon
    d1 = adaptive.f_nl_recon_ratio * adaptive.bphi_recon / M[0]
    assert poly.c0 == pytest.approx(numerator / d0, rel=2.0e-14)
    assert poly.c1 == pytest.approx(-numerator * d1 / d0**2, rel=2.0e-14)
    assert poly.c2 == pytest.approx(numerator * d1**2 / d0**3, rel=2.0e-14)


def test_full_adaptive_post_coefficients_match_finite_differences_of_exact_model():
    adaptive = ReconstructionConfig(
        smoothing_radius=13.0,
        bias_recon=2.25,
        f_nl_recon_ratio=0.6,
        bphi_recon=3.1,
    )
    result = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, BIAS, RSD, reconstruction=adaptive
    ).total

    def exact(fnl):
        n = RSD.los_array
        z1 = []
        for i in range(3):
            mu = np.dot(VECTORS[i], n) / np.linalg.norm(VECTORS[i])
            z1.append(BIAS.b1 + RSD.f_growth * mu**2 + fnl * BIAS.bphi / M[i])
        out = 0.0
        for i, j, _ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            z2_pre = z2_rsd_poly(
                VECTORS[i], VECTORS[j], M[i], M[j], BIAS, RSD
            ).evaluate(fnl)
            total = VECTORS[i] + VECTORS[j]
            shifts = []
            for index in (i, j):
                q = VECTORS[index]
                numerator = (
                    -np.dot(total, q)
                    / np.dot(q, q)
                    * math.exp(-0.5 * (np.linalg.norm(q) * adaptive.smoothing_radius) ** 2)
                )
                denominator = (
                    adaptive.bias_recon
                    + adaptive.f_nl_recon_ratio
                    * fnl
                    * adaptive.bphi_recon
                    / M[index]
                )
                shifts.append(numerator / denominator)
            z2 = z2_pre + 0.5 * sum(shifts) * z1[i] * z1[j]
            out += 2.0 * P[i] * P[j] * z1[i] * z1[j] * z2
        out += fnl * primordial_b0_hat(P, M) * z1[0] * z1[1] * z1[2]
        return out

    h = 3.0e-5
    g = exact(0.0)
    l = (exact(h) - exact(-h)) / (2.0 * h)
    q = (exact(h) - 2.0 * exact(0.0) + exact(-h)) / (2.0 * h**2)
    assert result.G == pytest.approx(g, rel=2.0e-13)
    assert result.L == pytest.approx(l, rel=3.0e-8)
    assert result.Q == pytest.approx(q, rel=3.0e-8)


def test_rotation_of_triangle_and_los_is_covariant():
    rotation = _rotation_matrix([0.3, 0.5, -0.2], 0.73)
    rotated_vectors = VECTORS @ rotation.T
    rotated_los = rotation @ RSD.los_array
    original = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD).total
    rotated = finite_png_halo_bispectrum_rsd(
        rotated_vectors,
        P,
        M,
        BIAS,
        RsdConfig(RSD.f_growth, tuple(rotated_los)),
    ).total
    assert rotated.G == pytest.approx(original.G, rel=2.0e-13)
    assert rotated.L == pytest.approx(original.L, rel=2.0e-13)
    assert rotated.Q == pytest.approx(original.Q, rel=2.0e-13)


def test_leg_permutation_invariance():
    reference = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD).total
    permutation = np.array([2, 0, 1])
    permuted = finite_png_halo_bispectrum_rsd(
        VECTORS[permutation], P[permutation], M[permutation], BIAS, RSD
    ).total
    assert permuted.G == pytest.approx(reference.G, rel=2.0e-13)
    assert permuted.L == pytest.approx(reference.L, rel=2.0e-13)
    assert permuted.Q == pytest.approx(reference.Q, rel=2.0e-13)


def test_shape_orientation_constructor_has_requested_mu_and_closure():
    k1, k2, k3 = 0.07, 0.055, 0.063
    mu1 = -0.37
    phi = 1.27
    vectors = triangle_vectors_from_shape_orientation(k1, k2, k3, mu1, phi)
    validate_closed_triangle(vectors)
    assert np.linalg.norm(vectors[0]) == pytest.approx(k1, rel=2.0e-14)
    assert np.linalg.norm(vectors[1]) == pytest.approx(k2, rel=2.0e-14)
    assert np.linalg.norm(vectors[2]) == pytest.approx(k3, rel=2.0e-14)
    assert vectors[0, 2] / k1 == pytest.approx(mu1, rel=2.0e-14)


def test_monopole_at_zero_growth_equals_orientation_independent_real_space_value():
    lengths = np.linalg.norm(VECTORS, axis=1)
    expected = _manual_real_space_components(VECTORS, P, M, BIAS)
    projected = project_scoccimarro_multipole(
        *lengths,
        P,
        M,
        BIAS,
        RsdConfig(0.0),
        ell=0,
        n_mu=8,
        n_phi=8,
    )
    assert projected.G == pytest.approx(expected.G, rel=2.0e-13)
    assert projected.L == pytest.approx(expected.L, rel=2.0e-13)
    assert projected.Q == pytest.approx(expected.Q, rel=2.0e-13)


def test_monopole_quadrature_converges_for_rsd_case():
    lengths = np.linalg.norm(VECTORS, axis=1)
    coarse = project_scoccimarro_multipole(
        *lengths, P, M, BIAS, RSD, n_mu=16, n_phi=18
    )
    fine = project_scoccimarro_multipole(
        *lengths, P, M, BIAS, RSD, n_mu=24, n_phi=26
    )
    assert coarse.G == pytest.approx(fine.G, rel=5.0e-12)
    assert coarse.L == pytest.approx(fine.L, rel=5.0e-12)
    assert coarse.Q == pytest.approx(fine.Q, rel=5.0e-12)


def test_desi_stochastic_bases_have_correct_real_space_limit():
    real = desi_gaussian_tree_stochastic_bases(
        VECTORS, P, BIAS.b1, RsdConfig(0.0, RSD.los)
    )
    assert real.Bshot_basis == pytest.approx(BIAS.b1**2 * np.sum(P), rel=2.0e-14)
    assert real.Pshot_basis == pytest.approx(0.0, abs=1.0e-14)
    assert real.Ashot_basis == 1.0
    rsd = desi_gaussian_tree_stochastic_bases(VECTORS, P, BIAS.b1, RSD)
    assert rsd.Bshot_basis != pytest.approx(real.Bshot_basis)
    assert rsd.Pshot_basis > 0.0


def test_pre_recon_ir_covariance_matches_scalar_and_anisotropic_formula():
    sigma2 = 22.0
    covariance = pre_recon_rsd_ir_covariance(sigma2, RSD)
    k = VECTORS[0]
    mu = np.dot(k, RSD.los_array) / np.linalg.norm(k)
    expected_exponent = sigma2 * np.dot(k, k) * (
        1.0 + RSD.f_growth * (2.0 + RSD.f_growth) * mu**2
    )
    assert damping_from_covariance(k, covariance) == pytest.approx(
        math.exp(-expected_exponent), rel=2.0e-14
    )
    real_cov = pre_recon_rsd_ir_covariance(sigma2, RsdConfig(0.0, RSD.los))
    assert damping_from_covariance(k, real_cov) == pytest.approx(
        math.exp(-sigma2 * np.dot(k, k)), rel=2.0e-14
    )


def test_kaiser_denominator_is_explicitly_distinct_from_current_bias_only_map():
    bias_only = ReconstructionConfig(15.0, 2.4)
    kaiser = ReconstructionConfig(
        15.0,
        2.4,
        denominator_mode="kaiser",
        f_growth_recon=RSD.f_growth,
    )
    total = VECTORS[0] + VECTORS[1]
    a = reconstruction_shift_poly(total, VECTORS[0], M[0], RSD, bias_only)
    b = reconstruction_shift_poly(total, VECTORS[0], M[0], RSD, kaiser)
    assert a.c0 != pytest.approx(b.c0)
    assert a.c1 == b.c1 == 0.0
    assert a.c2 == b.c2 == 0.0


def test_global_los_reversal_leaves_parity_even_tree_unchanged():
    forward = finite_png_halo_bispectrum_rsd(VECTORS, P, M, BIAS, RSD).total
    reverse = finite_png_halo_bispectrum_rsd(
        VECTORS,
        P,
        M,
        BIAS,
        RsdConfig(RSD.f_growth, tuple(-RSD.los_array)),
    ).total
    assert reverse.G == pytest.approx(forward.G, rel=2.0e-14)
    assert reverse.L == pytest.approx(forward.L, rel=2.0e-14)
    assert reverse.Q == pytest.approx(forward.Q, rel=2.0e-14)


def test_bphi2_is_an_independent_operator_not_bphi_squared():
    only_phi2 = HaloBiasV1(
        b1=BIAS.b1,
        b2=BIAS.b2,
        bK2=BIAS.bK2,
        bphi=0.0,
        bphidelta=0.0,
        bphi2=BIAS.bphi2,
    )
    without_phi2 = HaloBiasV1(
        b1=BIAS.b1,
        b2=BIAS.b2,
        bK2=BIAS.bK2,
        bphi=0.0,
        bphidelta=0.0,
        bphi2=0.0,
    )
    with_operator = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, only_phi2, RSD
    ).total
    without_operator = finite_png_halo_bispectrum_rsd(
        VECTORS, P, M, without_phi2, RSD
    ).total
    assert with_operator.G == pytest.approx(without_operator.G, rel=2.0e-14)
    assert with_operator.L == pytest.approx(without_operator.L, rel=2.0e-14)
    assert with_operator.Q != pytest.approx(without_operator.Q)


def test_odd_legendre_multipoles_vanish_for_global_los_parity_symmetric_model():
    lengths = np.linalg.norm(VECTORS, axis=1)
    for ell in (1, 3):
        projected = project_scoccimarro_multipole(
            *lengths,
            P,
            M,
            BIAS,
            RSD,
            ell=ell,
            n_mu=24,
            n_phi=26,
        )
        assert projected.G == pytest.approx(0.0, abs=2.0e-5)
        assert projected.L == pytest.approx(0.0, abs=2.0e-5)
        assert projected.Q == pytest.approx(0.0, abs=2.0e-5)
