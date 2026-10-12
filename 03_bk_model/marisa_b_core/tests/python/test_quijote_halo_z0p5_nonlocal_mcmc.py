from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


RUNNER = (
    Path(__file__).resolve().parents[3]
    / "run_quijote_halo_z0p5_nonlocal_marisa_b_mcmc.py"
)


@pytest.fixture(scope="module")
def module():
    specification = importlib.util.spec_from_file_location(
        "_test_nonlocal_bounded_mcmc", RUNNER
    )
    assert specification is not None and specification.loader is not None
    loaded = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = loaded
    specification.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def assets(module):
    return module.load_assets()


def test_production_task_contract_is_complete_and_unique(module):
    specs = module.production_specs()
    assert len(specs) == 32
    assert len({spec.key for spec in specs}) == 32
    assert sum(spec.covariance == "mean100" for spec in specs) == 8
    assert sum(
        spec.covariance == "native"
        and spec.prior == "wide"
        and spec.nuisance_prior == "wide5"
        for spec in specs
    ) == 8
    assert sum(spec.prior == "narrow" for spec in specs) == 8
    assert sum(spec.nuisance_prior == "narrow3" for spec in specs) == 8


def test_flat_prior_boxes_are_the_frozen_five_and_three_scale_ranges(module):
    primary = module.TaskSpec("EQ", -100.0, "pre", "native", "wide", "wide5")
    sensitivity = module.TaskSpec(
        "EQ", -100.0, "pre", "native", "wide", "narrow3"
    )
    assert primary.linear_bounds == (
        (-15.0, 15.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)
    )
    assert sensitivity.linear_bounds == (
        (-9.0, 9.0), (-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)
    )
    assert np.all(np.isfinite(np.asarray(primary.full_bounds)))


def test_profile_maps_and_models_are_reproduced(module, assets):
    for spec in module.base_specs():
        context = module.PosteriorContext(assets, spec)
        key = spec.base_key
        theta = np.asarray(
            assets.arrays[f"{key}__ensemble_mean_100__theta_map"]
        )
        conditional = context.conditional(theta[:2])
        expected = np.asarray(
            assets.arrays[f"{key}__ensemble_mean_100__model_map"]
        )
        difference = module.solve_triangular(
            context.chol_native_fit,
            conditional.prediction - expected,
            lower=True,
        )
        assert np.max(np.abs(conditional.linear_mean - theta[2:])) < 1e-8
        assert np.linalg.norm(difference) < 1e-9
        assert np.isfinite(context.log_explicit(theta))


def test_production_support_enforces_every_flat_boundary(module, assets):
    spec = module.TaskSpec("EQ", -100.0, "pre", "native", "wide", "wide5")
    context = module.PosteriorContext(assets, spec)
    theta = np.asarray(
        assets.arrays[f"{spec.base_key}__ensemble_mean_100__theta_map"]
    ).copy()
    assert np.isfinite(context.log_explicit(theta))
    for index, bounds in enumerate(spec.full_bounds):
        outside = theta.copy()
        outside[index] = bounds[1] + 1e-6
        assert context.log_explicit(outside) == -np.inf


def test_noncentered_transform_is_exact_and_keeps_physical_boxes(module, assets):
    for spec in module.base_specs():
        context = module.PosteriorContext(assets, spec)
        theta = np.asarray(
            assets.arrays[f"{spec.base_key}__ensemble_mean_100__theta_map"]
        )
        coordinate = context.physical_to_noncentered(theta)
        recovered = context.noncentered_to_physical(coordinate)
        assert np.allclose(recovered, theta, rtol=0.0, atol=3e-11)
        _residual, _design, lower, _mean = context._system(*theta[:2])
        expected = context.log_explicit(theta) - np.sum(
            np.log(np.diag(lower))
        )
        assert context.log_noncentered(coordinate) == pytest.approx(
            expected, rel=0.0, abs=1e-10
        )
        batch = np.vstack((coordinate, coordinate))
        assert np.allclose(
            context.log_noncentered(batch), expected, rtol=0.0, atol=1e-10
        )

    context = module.PosteriorContext(
        assets,
        module.TaskSpec("EQ", -100.0, "pre", "native", "wide", "wide5"),
    )
    coordinate = context.physical_to_noncentered(np.asarray(
        assets.arrays["EQ__m100__pre__ensemble_mean_100__theta_map"]
    ))
    coordinate[2:] = 1e6
    assert context.log_noncentered(coordinate) == -np.inf


def test_native_initializer_falls_back_to_bounded_physical_map(module, assets):
    spec = module.TaskSpec(
        "EQ", -100.0, "pre", "native", "wide", "wide5"
    )
    context = module.PosteriorContext(assets, spec)
    physical = np.asarray(
        assets.arrays[f"{spec.base_key}__ensemble_mean_100__theta_map"]
    )
    maps = [{"physical": physical, "unit": context.to_unit(physical)}]
    reference = module.build_quadrature_reference(context, nodes=101)
    positions = module.production_initial_positions(
        context, reference, maps, 12, np.random.default_rng(194)
    )
    assert positions.shape == (12, 6)
    assert np.linalg.matrix_rank(positions - np.mean(positions, axis=0)) == 6
    assert np.all(np.isfinite(context.log_noncentered(positions)))


def test_rank_split_rhat_uses_pooled_marginals_not_walker_means(module):
    rng = np.random.default_rng(23)
    retained = rng.normal(size=(4, 400, 12, 2))
    diagnostics = module.independent_ensemble_diagnostics(retained)
    expected = []
    for parameter in range(2):
        values = retained[..., parameter]
        ranked = module.pooled_rank_normalize(values)
        folded = module.pooled_rank_normalize(
            np.abs(values - np.median(values))
        )
        half = values.shape[1] // 2

        def split(values):
            return np.concatenate(
                (values[:, :half], values[:, -half:]), axis=0
            ).reshape(8, -1)

        expected.append(max(
            module.az.rhat(split(ranked), method="identity"),
            module.az.rhat(split(folded), method="identity"),
        ))
    assert np.allclose(
        diagnostics["rank_normalized_split_rhat"], expected,
        rtol=0.0, atol=1e-14,
    )


def test_unit_transform_and_covariance_scaling_are_exact(module, assets):
    mean_spec = module.TaskSpec("OR_LSS", 100.0, "standard", "mean100", "wide")
    native_spec = module.TaskSpec("OR_LSS", 100.0, "standard", "native", "wide")
    mean = module.PosteriorContext(assets, mean_spec)
    native = module.PosteriorContext(assets, native_spec)
    theta = np.asarray(
        assets.arrays[f"{mean_spec.base_key}__ensemble_mean_100__theta_map"]
    )
    assert np.allclose(mean.from_unit(mean.to_unit(theta)), theta, rtol=0.0, atol=1e-12)
    assert mean.profile_objective(theta[:2]) == pytest.approx(
        module.N_ENSEMBLE * native.profile_objective(theta[:2]),
        rel=2e-13,
        abs=2e-12,
    )
