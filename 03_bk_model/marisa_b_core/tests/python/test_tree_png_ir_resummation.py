from __future__ import annotations

import math

import numpy as np

from marisa_b.tree_png_ir_resummation import (
    GAUSSIAN_TERMS,
    LINEAR_TERMS,
    QUADRATIC_TERMS,
    HaloTreeBias,
    ReconstructionConfig,
    TreeIrConfig,
    bao_displacement_variance,
    finite_png_halo_tree_components,
    finite_png_halo_tree_operator_terms,
    finite_png_stochastic_bases,
    gaussian_reconstruction_residual,
    ir_resummed_finite_png_halo_tree,
    no_wiggle_transfer,
    triangle_vectors,
)


def _setup(wiggle_scale: float = 1.0):
    vectors = triangle_vectors(0.12, 0.17, -0.31)
    no_wiggle = np.array([8000.0, 5200.0, 6200.0])
    base_wiggle = np.array([200.0, -150.0, 100.0])
    power = no_wiggle + wiggle_scale * base_wiggle
    # Build M from a fixed primordial spectrum so P/M^2 is controlled.
    primordial = np.array([4.65e4, 9.76e3, 1.98e4])
    transfer = np.sqrt(power / primordial)
    bias = HaloTreeBias(
        b1=2.1,
        b2=0.35,
        bk2=-0.18,
        bphi=3.4,
        bphidelta=1.2,
        bphi2=-0.7,
    )
    return vectors, power, no_wiggle, transfer, bias


def _array(components):
    return np.array(
        [components.gaussian, components.linear, components.quadratic]
    )


def test_no_wiggle_transfer_preserves_primordial_power():
    _, power, no_wiggle, transfer, _ = _setup()
    transfer_nw = no_wiggle_transfer(power, no_wiggle, transfer)
    np.testing.assert_allclose(
        no_wiggle / transfer_nw**2,
        power / transfer**2,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_operator_terms_sum_to_components():
    vectors, power, _, transfer, bias = _setup()
    terms = finite_png_halo_tree_operator_terms(
        vectors, power, transfer, bias
    )
    components = finite_png_halo_tree_components(
        vectors, power, transfer, bias
    )
    expected = np.array(
        [
            sum(terms[name] for name in GAUSSIAN_TERMS),
            sum(terms[name] for name in LINEAR_TERMS),
            sum(terms[name] for name in QUADRATIC_TERMS),
        ]
    )
    np.testing.assert_allclose(_array(components), expected, rtol=0.0, atol=0.0)


def test_pre_tree_matches_explicit_native_G_L_Q_formula():
    vectors, power, _, transfer, bias = _setup()
    result = finite_png_halo_tree_components(vectors, power, transfer, bias)
    norms = np.linalg.norm(vectors, axis=1)
    inverse_m = 1.0 / transfer
    gaussian = 0.0
    linear = 0.0
    quadratic = 0.0
    b0_hat = 0.0

    for i, j, ell in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        mu = np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j])
        f2 = 5.0 / 7.0 + 0.5 * mu * (
            norms[i] / norms[j] + norms[j] / norms[i]
        ) + 2.0 * mu**2 / 7.0
        s2 = mu**2 - 1.0 / 3.0
        h2 = bias.b1 * f2 + 0.5 * bias.b2 + bias.bk2 * s2
        ui = inverse_m[i]
        uj = inverse_m[j]
        u_sum = ui + uj
        u_product = ui * uj
        advection = mu * (
            norms[i] / norms[j] * ui + norms[j] / norms[i] * uj
        )
        pp = power[i] * power[j]
        gaussian += 2.0 * bias.b1**2 * pp * h2
        linear += pp * (
            2.0 * bias.b1 * bias.bphi * u_sum * h2
            + bias.b1**2 * bias.bphi * advection
            + bias.b1**2 * bias.bphidelta * u_sum
        )
        quadratic += pp * (
            2.0 * bias.bphi**2 * u_product * h2
            + bias.b1 * bias.bphi**2 * u_sum * advection
            + bias.b1 * bias.bphi * bias.bphidelta * u_sum**2
            + bias.b1**2 * bias.bphi2 * u_product
        )
        b0_hat += 2.0 * pp * transfer[ell] / (
            transfer[i] * transfer[j]
        )

    linear += bias.b1**3 * b0_hat
    quadratic += (
        bias.b1**2 * bias.bphi * np.sum(inverse_m) * b0_hat
    )
    np.testing.assert_allclose(
        _array(result),
        [gaussian, linear, quadratic],
        rtol=3.0e-15,
        atol=2.0e-7,
    )


def test_stochastic_bases_match_native_contract():
    vectors, power, _, transfer, bias = _setup()
    result = finite_png_stochastic_bases(
        vectors, power, transfer, bias
    )
    np.testing.assert_allclose(
        [
            result.alpha3_gaussian,
            result.alpha3_linear_png,
            result.alpha3_quadratic_png,
            result.alpha4_constant,
        ],
        [
            bias.b1**2 * np.sum(power),
            bias.b1 * bias.bphi * np.sum(power / transfer),
            bias.bphi**2 * np.sum(power / transfer**2),
            1.0,
        ],
        rtol=3.0e-15,
        atol=2.0e-10,
    )


def test_post_minus_pre_matches_native_reconstruction_terms():
    vectors, power, _, transfer, bias = _setup()
    config = ReconstructionConfig(smoothing_radius=13.0, bias_recon=1.9)
    pre = finite_png_halo_tree_components(vectors, power, transfer, bias)
    post = finite_png_halo_tree_components(
        vectors, power, transfer, bias, config
    )
    norms = np.linalg.norm(vectors, axis=1)
    inverse_m = 1.0 / transfer
    delta = np.zeros(3)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        total = vectors[i] + vectors[j]
        wi = math.exp(-0.5 * (norms[i] * config.smoothing_radius) ** 2)
        wj = math.exp(-0.5 * (norms[j] * config.smoothing_radius) ** 2)
        shift = -np.dot(total, vectors[i]) * wi / (
            norms[i] ** 2 * config.bias_recon
        )
        shift += -np.dot(total, vectors[j]) * wj / (
            norms[j] ** 2 * config.bias_recon
        )
        ui = inverse_m[i]
        uj = inverse_m[j]
        pp = power[i] * power[j]
        delta[0] += pp * shift * bias.b1**4
        delta[1] += (
            2.0 * pp * shift * bias.b1**3 * bias.bphi * (ui + uj)
        )
        delta[2] += (
            pp
            * shift
            * bias.b1**2
            * bias.bphi**2
            * (ui**2 + 4.0 * ui * uj + uj**2)
        )
    np.testing.assert_allclose(
        _array(post) - _array(pre), delta, rtol=2.0e-13, atol=2.0e-7
    )


def test_post_tends_to_pre_for_infinite_smoothing():
    vectors, power, _, transfer, bias = _setup()
    pre = finite_png_halo_tree_components(vectors, power, transfer, bias)
    post = finite_png_halo_tree_components(
        vectors,
        power,
        transfer,
        bias,
        ReconstructionConfig(smoothing_radius=1.0e6, bias_recon=2.0),
    )
    np.testing.assert_allclose(_array(post), _array(pre), rtol=2e-14, atol=1e-8)


def test_no_wiggle_input_is_invariant_for_both_schemes():
    vectors, _, no_wiggle, transfer_full, bias = _setup()
    primordial = np.array([4.65e4, 9.76e3, 1.98e4])
    transfer_nw = np.sqrt(no_wiggle / primordial)
    fixed = finite_png_halo_tree_components(
        vectors, no_wiggle, transfer_nw, bias
    )
    for scheme in ("strict_lo", "factorized_closure"):
        result = ir_resummed_finite_png_halo_tree(
            vectors,
            no_wiggle,
            no_wiggle,
            transfer_nw,
            bias,
            TreeIrConfig(sigma2=8.0, scheme=scheme),
        )
        np.testing.assert_allclose(
            _array(result.deterministic), _array(fixed), rtol=3e-15, atol=1e-8
        )


def test_factorized_zero_sigma_recovers_full_fixed_order_tree():
    vectors, power, no_wiggle, transfer, bias = _setup()
    fixed = finite_png_halo_tree_components(vectors, power, transfer, bias)
    result = ir_resummed_finite_png_halo_tree(
        vectors,
        power,
        no_wiggle,
        transfer,
        bias,
        TreeIrConfig(sigma2=0.0, scheme="factorized_closure"),
    )
    np.testing.assert_allclose(
        _array(result.deterministic), _array(fixed), rtol=3e-15, atol=1e-8
    )


def test_strict_lo_is_exact_legwise_first_order_response():
    vectors, power, no_wiggle, transfer, bias = _setup()
    transfer_nw = no_wiggle_transfer(power, no_wiggle, transfer)
    wiggle = power - no_wiggle
    smooth = finite_png_halo_tree_components(
        vectors, no_wiggle, transfer_nw, bias
    )
    result = ir_resummed_finite_png_halo_tree(
        vectors,
        power,
        no_wiggle,
        transfer,
        bias,
        TreeIrConfig(sigma2=0.0, scheme="strict_lo"),
    )

    epsilon = 2.0e-4
    plus_power = no_wiggle + epsilon * wiggle
    minus_power = no_wiggle - epsilon * wiggle
    plus_transfer = transfer_nw * np.sqrt(plus_power / no_wiggle)
    minus_transfer = transfer_nw * np.sqrt(minus_power / no_wiggle)
    plus = finite_png_halo_tree_components(
        vectors, plus_power, plus_transfer, bias
    )
    minus = finite_png_halo_tree_components(
        vectors, minus_power, minus_transfer, bias
    )
    derivative = (_array(plus) - _array(minus)) / (2.0 * epsilon)
    exact = _array(result.deterministic) - _array(smooth)
    np.testing.assert_allclose(exact, derivative, rtol=2e-8, atol=2e-4)


def test_strict_primordial_term_matches_three_dressed_transfer_legs():
    vectors, power, no_wiggle, transfer, bias = _setup()
    primordial_only = HaloTreeBias(
        b1=bias.b1,
        b2=0.0,
        bk2=0.0,
        bphi=0.0,
        bphidelta=0.0,
        bphi2=0.0,
    )
    sigma2 = 4.2
    result = ir_resummed_finite_png_halo_tree(
        vectors,
        power,
        no_wiggle,
        transfer,
        primordial_only,
        TreeIrConfig(sigma2=sigma2, scheme="strict_lo"),
    )
    transfer_nw = no_wiggle_transfer(power, no_wiggle, transfer)
    b0_nw = 0.0
    for i, j, ell in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        b0_nw += 2.0 * no_wiggle[i] * no_wiggle[j] * transfer_nw[ell] / (
            transfer_nw[i] * transfer_nw[j]
        )
    k = np.linalg.norm(vectors, axis=1)
    d = np.exp(-k**2 * sigma2)
    expected = primordial_only.b1**3 * b0_nw * (
        1.0 + 0.5 * np.sum(d * (power - no_wiggle) / no_wiggle)
    )
    assert math.isclose(
        result.operator_terms["dBdfNL_local_primordial"],
        expected,
        rel_tol=3e-14,
    )


def test_quadratic_png_stochastic_basis_is_bao_invariant():
    vectors, power, no_wiggle, transfer, bias = _setup()
    fixed = bias.bphi**2 * np.sum(power / transfer**2)
    for scheme in ("strict_lo", "factorized_closure"):
        result = ir_resummed_finite_png_halo_tree(
            vectors,
            power,
            no_wiggle,
            transfer,
            bias,
            TreeIrConfig(sigma2=7.0, scheme=scheme),
        )
        assert math.isclose(
            result.stochastic.alpha3_quadratic_png,
            fixed,
            rel_tol=3e-15,
        )


def test_factorized_minus_strict_is_second_order_in_wiggle_amplitude():
    differences = []
    for amplitude in (0.04, 0.08):
        vectors, power, no_wiggle, transfer, bias = _setup(amplitude)
        strict = ir_resummed_finite_png_halo_tree(
            vectors,
            power,
            no_wiggle,
            transfer,
            bias,
            TreeIrConfig(sigma2=5.0, scheme="strict_lo"),
        )
        factorized = ir_resummed_finite_png_halo_tree(
            vectors,
            power,
            no_wiggle,
            transfer,
            bias,
            TreeIrConfig(sigma2=5.0, scheme="factorized_closure"),
        )
        differences.append(
            np.linalg.norm(
                _array(factorized.deterministic)
                - _array(strict.deterministic)
            )
        )
    assert differences[0] > 0.0
    assert math.isclose(differences[1] / differences[0], 4.0, rel_tol=0.08)


def test_displacement_variance_and_reconstruction_residual():
    q = np.geomspace(1.0e-4, 0.4, 4001)
    power = 1.5e4 * q * np.exp(-q / 0.12)
    sigma_pre = bao_displacement_variance(
        q, power, r_bao=110.0, k_ir=0.15
    )
    residual = gaussian_reconstruction_residual(
        q,
        smoothing_radius=15.0,
        tracer_bias=2.0,
        bias_recon=2.0,
    )
    sigma_post = bao_displacement_variance(
        q,
        power,
        r_bao=110.0,
        k_ir=0.15,
        residual_filter=residual,
    )
    assert sigma_pre > 0.0
    assert 0.0 <= sigma_post < sigma_pre


def test_permutation_invariance_without_cic():
    vectors, power, no_wiggle, transfer, bias = _setup()
    reference = ir_resummed_finite_png_halo_tree(
        vectors,
        power,
        no_wiggle,
        transfer,
        bias,
        TreeIrConfig(sigma2=5.0),
    )
    permutation = np.array([2, 0, 1])
    permuted = ir_resummed_finite_png_halo_tree(
        vectors[permutation],
        power[permutation],
        no_wiggle[permutation],
        transfer[permutation],
        bias,
        TreeIrConfig(sigma2=5.0),
    )
    np.testing.assert_allclose(
        _array(permuted.deterministic),
        _array(reference.deterministic),
        rtol=2e-14,
        atol=2e-7,
    )
