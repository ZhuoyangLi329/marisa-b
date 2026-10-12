from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import spherical_jn

from marisa_b.post_recon_tree_png_ir import (
    AdaptiveReconstructionConfig,
    PostReconIrSettings,
    damping_expansion_from_variance,
    ir_resummed_post_recon_finite_png_tree,
    post_recon_displacement_variance_at_fnl,
    post_recon_displacement_variance_expansion,
    post_recon_source_operator_terms,
    _components_from_post_terms,
    _post_source_terms_raw,
    _strict_post_leg_response,
)
from marisa_b.tree_png_ir_resummation import (
    HaloTreeBias,
    ReconstructionConfig,
    TreeComponents,
    bao_displacement_variance,
    no_wiggle_transfer,
    triangle_vectors,
)


def _external_setup():
    vectors = triangle_vectors(0.115, 0.162, -0.37)
    p_nw = np.array([8100.0, 5400.0, 6550.0])
    p_w = np.array([180.0, -125.0, 95.0])
    p_full = p_nw + p_w
    p_phi = np.array([4.4e4, 1.05e4, 2.2e4])
    m_full = np.sqrt(p_full / p_phi)
    bias = HaloTreeBias(
        b1=2.05,
        b2=0.31,
        bk2=-0.16,
        bphi=0.42,
        bphidelta=0.73,
        bphi2=-0.24,
    )
    return vectors, p_full, p_nw, m_full, bias


def _ir_grid(n: int = 501):
    q = np.geomspace(1.0e-3, 0.22, n)
    p_nw = 1.3e4 * (q / 0.05) ** 0.96 * np.exp(-q / 0.18)
    wiggle_fraction = 0.035 * np.sin(108.0 * q) * np.exp(-(q / 0.22) ** 2)
    p_full = p_nw * (1.0 + wiggle_fraction)
    # Moderate positive transfer for stable finite-difference tests while
    # retaining a scale-dependent 1/M response.
    m_nw = 0.22 + (q / 0.075) ** 2
    m_full = m_nw * np.sqrt(p_full / p_nw)
    return q, p_full, p_nw, m_full


def _settings(**kwargs):
    values = dict(r_bao=105.0, k_ir=0.16, q_min=1.0e-3)
    values.update(kwargs)
    return PostReconIrSettings(**values)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _components_array(value: TreeComponents) -> np.ndarray:
    return np.array([value.G, value.L, value.Q])


def test_fixed_denominator_isotropic_variance_matches_direct_integral():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=14.0, bias_recon=1.82
    )
    settings = _settings(k_ir=float(q[-70]))
    result = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
    )

    m_nw = m_full * np.sqrt(p_nw / p_full)
    W = np.exp(-0.5 * (q * reconstruction.smoothing_radius) ** 2)
    a = bias.b1 / reconstruction.bias_recon
    n = bias.bphi / (reconstruction.bias_recon * m_nw)
    r0 = 1.0 - W * a
    r1 = -W * n
    c = (r0**2, 2.0 * r0 * r1, r1**2)
    x = q * settings.r_bao
    bao = 1.0 - spherical_jn(0, x) + 2.0 * spherical_jn(2, x)
    mask = q <= settings.k_ir
    expected = tuple(
        simpson(
            p_nw[mask] * item[mask] * bao[mask] / (6.0 * math.pi**2),
            x=q[mask],
        )
        for item in c
    )
    # The implementation inserts an interpolated node exactly at k_ir.  The
    # grid is dense, so the direct masked integral agrees at high accuracy.
    np.testing.assert_allclose(
        [result.sigma_f0[0], result.sigma_f1[0], result.sigma_f2[0]],
        expected,
        rtol=2.0e-5,
        atol=1.0e-10,
    )
    assert result.sigma_f0[0] == result.sigma_f0[1] == result.sigma_f0[2]
    assert not result.anisotropic


def test_infinite_smoothing_recovers_pre_reconstruction_variance():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=1.0e6, bias_recon=2.0
    )
    settings = _settings(k_ir=float(q[-70]))
    result = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
    )
    sigma_pre = bao_displacement_variance(
        q,
        p_nw,
        r_bao=settings.r_bao,
        k_ir=settings.k_ir,
    )
    assert math.isclose(result.sigma_f0[0], sigma_pre, rel_tol=2.0e-5)
    np.testing.assert_allclose(result.sigma_f1, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.sigma_f2, 0.0, atol=1.0e-14)


def test_zero_smoothing_matched_adaptive_removes_linear_displacement():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=0.0, bias_recon=bias.b1
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=bias.bphi, fnl_recon_ratio=1.0
    )
    result = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        _settings(),
        adaptive,
    )
    np.testing.assert_allclose(result.sigma_f0, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.sigma_f1, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.sigma_f2, 0.0, atol=2.0e-15)


def test_matched_adaptive_cancels_all_fnl_damping_responses_at_finite_R():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=13.0, bias_recon=bias.b1
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=bias.bphi, fnl_recon_ratio=1.0
    )
    result = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        _settings(),
        adaptive,
    )
    assert result.sigma_f0[0] > 0.0
    np.testing.assert_allclose(result.sigma_f1, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(result.sigma_f2, 0.0, atol=2.0e-14)


def test_fixed_denominator_variance_polynomial_matches_direct_finite_fnl():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.0, bias_recon=1.9
    )
    settings = _settings()
    expansion = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
    )
    h = 0.17
    zero = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors,
            q,
            p_full,
            p_nw,
            m_full,
            bias,
            reconstruction,
            settings,
            0.0,
        )
    )
    plus = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors,
            q,
            p_full,
            p_nw,
            m_full,
            bias,
            reconstruction,
            settings,
            h,
        )
    )
    minus = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors,
            q,
            p_full,
            p_nw,
            m_full,
            bias,
            reconstruction,
            settings,
            -h,
        )
    )
    np.testing.assert_allclose(zero, expansion.sigma_f0, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        (plus - minus) / (2.0 * h),
        expansion.sigma_f1,
        rtol=2e-12,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        (plus + minus - 2.0 * zero) / (2.0 * h**2),
        expansion.sigma_f2,
        rtol=2e-11,
        atol=2e-12,
    )


def test_adaptive_variance_taylor_coefficients_match_direct_small_fnl():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.0, bias_recon=1.87
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=0.31, fnl_recon_ratio=0.64
    )
    settings = _settings()
    expansion = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
        adaptive,
    )
    h = 2.0e-4
    zero = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, 0.0, adaptive
        )
    )
    plus = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, h, adaptive
        )
    )
    minus = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, -h, adaptive
        )
    )
    np.testing.assert_allclose(
        (plus - minus) / (2.0 * h),
        expansion.sigma_f1,
        rtol=3.0e-7,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        (plus + minus - 2.0 * zero) / (2.0 * h**2),
        expansion.sigma_f2,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_adaptive_source_terms_match_native_basis_normalization():
    vectors, p_full, _, m_full, bias = _external_setup()
    reconstruction = ReconstructionConfig(
        smoothing_radius=11.0, bias_recon=1.76
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=0.39, fnl_recon_ratio=0.61
    )
    fixed = post_recon_source_operator_terms(
        vectors, p_full, m_full, bias, reconstruction
    )
    result = post_recon_source_operator_terms(
        vectors, p_full, m_full, bias, reconstruction, adaptive
    )

    inv_m = 1.0 / m_full
    ratio = adaptive.bphi_recon / reconstruction.bias_recon
    g1 = 0.0
    g2 = 0.0
    qcross = 0.0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        total = vectors[i] + vectors[j]
        terms = []
        for index in (i, j):
            k2 = np.dot(vectors[index], vectors[index])
            W = math.exp(
                -0.5
                * k2
                * reconstruction.smoothing_radius**2
            )
            terms.append(
                -np.dot(total, vectors[index])
                * W
                / (k2 * reconstruction.bias_recon)
            )
        shift_i, shift_j = terms
        pp = p_full[i] * p_full[j]
        shift_over_m = shift_i * inv_m[i] + shift_j * inv_m[j]
        gaussian_linear = -pp * shift_over_m
        gaussian_quadratic = pp * (
            shift_i * inv_m[i] ** 2 + shift_j * inv_m[j] ** 2
        )
        png_cross = (
            -2.0
            * pp
            * (inv_m[i] + inv_m[j])
            * shift_over_m
        )
        g1 += adaptive.eta * bias.b1**4 * ratio * gaussian_linear
        g2 += adaptive.eta**2 * bias.b1**4 * ratio**2 * gaussian_quadratic
        qcross += (
            adaptive.eta
            * bias.b1**3
            * bias.bphi
            * ratio
            * png_cross
        )

    assert result["dBdfNL_rec_brec_gaussian_linear"] == pytest.approx(g1)
    assert result["Btree_fNL2_rec_brec_gaussian_quadratic"] == pytest.approx(g2)
    assert result["Btree_fNL2_rec_brec_png_linear_cross"] == pytest.approx(qcross)
    for name in fixed:
        if name not in {
            "dBdfNL_rec_brec_gaussian_linear",
            "Btree_fNL2_rec_brec_gaussian_quadratic",
            "Btree_fNL2_rec_brec_png_linear_cross",
        }:
            assert result[name] == fixed[name]


def test_damping_expansion_is_exact_taylor_of_exponential():
    vectors, _, _, _, _ = _external_setup()
    from marisa_b.post_recon_tree_png_ir import DisplacementVarianceExpansion

    displacement = DisplacementVarianceExpansion(
        sigma_f0=(3.1, 2.8, 3.7),
        sigma_f1=(0.2, -0.14, 0.07),
        sigma_f2=(0.015, 0.021, -0.011),
        anisotropic=True,
        q_min=1.0e-3,
        k_ir=0.15,
        includes_shift_noise=False,
    )
    damping = damping_expansion_from_variance(vectors, displacement)
    k = np.linalg.norm(vectors, axis=1)
    h = 1.0e-3
    exact = lambda f: np.exp(
        -k**2
        * (
            np.asarray(displacement.sigma_f0)
            + f * np.asarray(displacement.sigma_f1)
            + f**2 * np.asarray(displacement.sigma_f2)
        )
    )
    D0 = exact(0.0)
    D1 = (exact(h) - exact(-h)) / (2.0 * h)
    D2 = (exact(h) + exact(-h) - 2.0 * D0) / (2.0 * h**2)
    np.testing.assert_allclose(damping.D0, D0, rtol=2e-15)
    np.testing.assert_allclose(np.asarray(damping.D0) * damping.d1, D1, rtol=2e-8)
    np.testing.assert_allclose(np.asarray(damping.D0) * damping.d2, D2, rtol=8e-6)


def test_complete_G_L_Q_matches_direct_strict_finite_fnl_derivatives():
    vectors, p_full, p_nw, m_full, bias = _external_setup()
    q, pq_full, pq_nw, mq_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.5, bias_recon=1.84
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=0.29, fnl_recon_ratio=0.57
    )
    settings = _settings()
    result = ir_resummed_post_recon_finite_png_tree(
        vectors,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        q,
        pq_full,
        pq_nw,
        mq_full,
        settings,
        adaptive,
    )

    m_nw = no_wiggle_transfer(p_full, p_nw, m_full)
    p_w = p_full - p_nw
    raw_nw = _post_source_terms_raw(
        vectors,
        p_nw.tolist(),
        m_nw.tolist(),
        bias,
        reconstruction,
        adaptive,
    )
    nw = _components_from_post_terms(
        {name: float(value) for name, value in raw_nw.items()}
    )
    responses = []
    for leg in range(3):
        response = _strict_post_leg_response(
            vectors,
            p_nw,
            p_w,
            m_nw,
            bias,
            reconstruction,
            adaptive,
            leg,
        )
        responses.append(_components_from_post_terms(response))

    def direct(fnl: float) -> float:
        sigma = post_recon_displacement_variance_at_fnl(
            vectors,
            q,
            pq_full,
            pq_nw,
            mq_full,
            bias,
            reconstruction,
            settings,
            fnl,
            adaptive,
        )
        k = np.linalg.norm(vectors, axis=1)
        damping = np.exp(-k**2 * np.asarray(sigma))
        return nw.predict(fnl) + sum(
            damping[leg] * responses[leg].predict(fnl)
            for leg in range(3)
        )

    h = 2.0e-4
    zero = direct(0.0)
    plus = direct(h)
    minus = direct(-h)
    direct_coefficients = np.array(
        [
            zero,
            (plus - minus) / (2.0 * h),
            (plus + minus - 2.0 * zero) / (2.0 * h**2),
        ]
    )
    np.testing.assert_allclose(
        _components_array(result.deterministic),
        direct_coefficients,
        rtol=4.0e-6,
        atol=1.5e-2,
    )
    assert abs(result.damping_mixing.linear_from_gaussian) > 0.0
    assert abs(result.damping_mixing.total_quadratic) > 0.0


def test_no_scale_dependent_reconstruction_response_has_zero_mixing():
    vectors, p_full, p_nw, m_full, bias = _external_setup()
    bias_no_phi = HaloTreeBias(
        b1=bias.b1,
        b2=bias.b2,
        bk2=bias.bk2,
        bphi=0.0,
        bphidelta=bias.bphidelta,
        bphi2=bias.bphi2,
    )
    q, pq_full, pq_nw, mq_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=13.0, bias_recon=1.9
    )
    result = ir_resummed_post_recon_finite_png_tree(
        vectors,
        p_full,
        p_nw,
        m_full,
        bias_no_phi,
        reconstruction,
        q,
        pq_full,
        pq_nw,
        mq_full,
        _settings(),
    )
    np.testing.assert_allclose(
        _components_array(result.deterministic),
        _components_array(result.source_only),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert result.damping_mixing.linear_from_gaussian == 0.0
    assert result.damping_mixing.total_quadratic == 0.0


def test_cartesian_quadrature_without_cic_matches_isotropic_result():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid(181)
    reconstruction = ReconstructionConfig(
        smoothing_radius=13.0, bias_recon=1.88
    )
    isotropic = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        _settings(mesh_damping="isotropic"),
    )
    cartesian = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        _settings(
            mesh_damping="cartesian",
            angular_mu_order=28,
            angular_phi_order=56,
        ),
    )
    np.testing.assert_allclose(
        cartesian.sigma_f0, isotropic.sigma_f0, rtol=2.5e-6, atol=2e-9
    )
    np.testing.assert_allclose(
        cartesian.sigma_f1, isotropic.sigma_f1, rtol=3.0e-6, atol=2e-9
    )
    np.testing.assert_allclose(
        cartesian.sigma_f2, isotropic.sigma_f2, rtol=4.0e-6, atol=2e-9
    )


def test_cartesian_cic_preserves_absolute_orientation_dependence():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid(151)
    reconstruction = ReconstructionConfig(
        smoothing_radius=10.0,
        bias_recon=1.82,
        cell_size=72.0,
        cic_window_power=4,
    )
    settings = _settings(
        mesh_damping="cartesian",
        angular_mu_order=24,
        angular_phi_order=48,
    )
    reference = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
    )
    rotation = _rotation(np.array([1.0, 2.0, -0.7]), 0.43)
    rotated = post_recon_displacement_variance_expansion(
        vectors @ rotation.T,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
    )
    assert reference.anisotropic and rotated.anisotropic
    difference = np.max(
        np.abs(np.asarray(reference.sigma_f0) - np.asarray(rotated.sigma_f0))
    )
    assert difference > 1.0e-6


def test_isotropic_mode_refuses_to_discard_active_cic_silently():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid(101)
    reconstruction = ReconstructionConfig(
        smoothing_radius=10.0,
        bias_recon=1.8,
        cell_size=50.0,
        cic_window_power=4,
    )
    with pytest.raises(ValueError, match="discard the active Cartesian CIC"):
        post_recon_displacement_variance_expansion(
            vectors,
            q,
            p_full,
            p_nw,
            m_full,
            bias,
            reconstruction,
            _settings(mesh_damping="isotropic"),
        )


def test_shift_noise_is_optional_positive_and_taylor_consistent():
    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.0, bias_recon=1.91
    )
    adaptive = AdaptiveReconstructionConfig(
        bphi_recon=0.25, fnl_recon_ratio=0.8
    )
    settings = _settings()
    noise = 120.0 * np.exp(-(q / 0.16) ** 2)
    without = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
        adaptive,
    )
    with_noise = post_recon_displacement_variance_expansion(
        vectors,
        q,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        settings,
        adaptive,
        shift_noise_power=noise,
    )
    assert with_noise.includes_shift_noise
    assert with_noise.sigma_f0[0] > without.sigma_f0[0]

    h = 2.0e-4
    exact0 = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, 0.0, adaptive,
            shift_noise_power=noise,
        )
    )
    exactp = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, h, adaptive,
            shift_noise_power=noise,
        )
    )
    exactm = np.asarray(
        post_recon_displacement_variance_at_fnl(
            vectors, q, p_full, p_nw, m_full, bias,
            reconstruction, settings, -h, adaptive,
            shift_noise_power=noise,
        )
    )
    np.testing.assert_allclose(
        (exactp - exactm) / (2.0 * h),
        with_noise.sigma_f1,
        rtol=3e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        (exactp + exactm - 2.0 * exact0) / (2.0 * h**2),
        with_noise.sigma_f2,
        rtol=3e-5,
        atol=2e-7,
    )


def test_stochastic_damping_mixing_requires_explicit_closure():
    vectors, p_full, p_nw, m_full, bias = _external_setup()
    q, pq_full, pq_nw, mq_full = _ir_grid()
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.0, bias_recon=1.86
    )
    result = ir_resummed_post_recon_finite_png_tree(
        vectors,
        p_full,
        p_nw,
        m_full,
        bias,
        reconstruction,
        q,
        pq_full,
        pq_nw,
        mq_full,
        _settings(),
    )
    closed = result.stochastic_if_common_alpha3()
    assert closed.alpha3_gaussian == result.source_stochastic.alpha3_gaussian
    assert closed.alpha3_linear_png == pytest.approx(
        result.source_stochastic.alpha3_linear_png
        + result.stochastic_mixing.linear_from_gaussian_alpha3
    )
    assert closed.alpha3_quadratic_png == pytest.approx(
        result.source_stochastic.alpha3_quadratic_png
        + result.stochastic_mixing.quadratic_from_linear_alpha3
        + result.stochastic_mixing.quadratic_from_gaussian_alpha3
    )


def test_local_png_ir_cutoff_must_be_positive_and_inside_grid():
    with pytest.raises(ValueError, match="q_min must be finite and positive"):
        PostReconIrSettings(r_bao=105.0, k_ir=0.15, q_min=0.0)

    vectors, _, _, _, bias = _external_setup()
    q, p_full, p_nw, m_full = _ir_grid(101)
    reconstruction = ReconstructionConfig(
        smoothing_radius=12.0, bias_recon=1.9
    )
    settings = PostReconIrSettings(
        r_bao=105.0, k_ir=0.16, q_min=1.0e-5
    )
    with pytest.raises(ValueError, match="must bracket"):
        post_recon_displacement_variance_expansion(
            vectors,
            q,
            p_full,
            p_nw,
            m_full,
            bias,
            reconstruction,
            settings,
        )


def test_native_ir_sigma2_convention_is_one_wiggle_D0():
    vectors, _, _, _, _ = _external_setup()
    from marisa_b.post_recon_tree_png_ir import DisplacementVarianceExpansion

    sigma = 4.25
    displacement = DisplacementVarianceExpansion(
        sigma_f0=(sigma, sigma, sigma),
        sigma_f1=(0.0, 0.0, 0.0),
        sigma_f2=(0.0, 0.0, 0.0),
        anisotropic=False,
        q_min=1.0e-3,
        k_ir=0.15,
        includes_shift_noise=False,
    )
    damping = damping_expansion_from_variance(vectors, displacement)
    k = np.linalg.norm(vectors, axis=1)
    # Native damping_ir is exp[-k^2 ir_sigma2/2] and its one-wiggle path uses
    # damping_ir^2.  Hence ir_sigma2 maps directly to sigma here.
    native_half = np.exp(-0.5 * k**2 * sigma)
    np.testing.assert_allclose(damping.D0, native_half**2, rtol=2e-15)
