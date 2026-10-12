from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


CODES = Path(__file__).resolve().parents[3]
RUNNER = CODES / "run_quijote_halo_z0p5_nonlocal_marisa_b_fixed_pk_b1.py"


@pytest.fixture(scope="module")
def module():
    sys.path.insert(0, str(CODES))
    specification = importlib.util.spec_from_file_location(
        "_test_nonlocal_fixed_pk_b1", RUNNER
    )
    assert specification is not None and specification.loader is not None
    loaded = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = loaded
    specification.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def assets(module):
    return module.base.load_assets()


def test_certified_pk_bias_provenance(module):
    value, payload = module.load_pk_bias()
    assert value == pytest.approx(1.9326526728794324, rel=0.0, abs=1e-14)
    assert payload["recommended_bias_source"] == "free_sn"
    assert payload["fit_model"] == "P_hh = b1^2 P_lin + c0"
    assert payload["n_realizations"] == 500
    assert payload["kmax_fit"] == pytest.approx(0.10)


def test_fixed_task_contract_has_three_prior_boxes_per_target(module):
    specs = module.production_specs()
    assert len(specs) == 24
    assert len({spec.key for spec in specs}) == 24
    assert sum(
        spec.prior == "wide" and spec.nuisance_prior == "wide5"
        for spec in specs
    ) == 8
    assert sum(spec.prior == "narrow" for spec in specs) == 8
    assert sum(spec.nuisance_prior == "narrow3" for spec in specs) == 8


def test_fixed_design_is_rank_five_and_matches_six_parameter_model(
    module, assets
):
    fixed_b1, _ = module.load_pk_bias()
    for shape in module.base.SHAPES:
        for method in module.base.METHODS:
            spec = module.FixedSpec(shape, -100.0, method)
            context = module.FixedB1Context(assets, spec, fixed_b1)
            assert np.linalg.matrix_rank(context.white_design) == 5
            point = np.asarray(context.bounded_map()["parameters"])
            full = context.full_six_parameters(point[None, :])
            assert full[0, 1] == fixed_b1
            assert np.allclose(
                context.model_batch(point[None, :]),
                context.parent.model_batch(full),
                rtol=0.0,
                atol=0.0,
            )


def test_noncentered_transform_and_jacobian_are_exact(module, assets):
    fixed_b1, _ = module.load_pk_bias()
    context = module.FixedB1Context(
        assets, module.FixedSpec("EQ", -100.0, "pre"), fixed_b1
    )
    point = np.asarray(context.bounded_map()["parameters"])
    coordinate = context.physical_to_noncentered(point)
    recovered = context.noncentered_to_physical(coordinate)
    assert np.allclose(recovered, point, rtol=0.0, atol=1e-12)
    expected = context.log_explicit(point) - 0.5 * context.logdet_hessian
    assert context.log_noncentered(coordinate) == pytest.approx(
        expected, rel=0.0, abs=1e-10
    )
    assert np.allclose(
        context.log_noncentered(np.vstack((coordinate, coordinate))),
        expected,
        rtol=0.0,
        atol=1e-10,
    )


def test_physical_boxes_are_enforced_and_initializer_is_finite(module, assets):
    fixed_b1, _ = module.load_pk_bias()
    context = module.FixedB1Context(
        assets, module.FixedSpec("OR_LSS", 100.0, "standard"), fixed_b1
    )
    point = np.asarray(context.bounded_map()["parameters"])
    assert np.isfinite(context.log_explicit(point))
    for index, bounds in enumerate(context.spec.bounds):
        outside = point.copy()
        outside[index] = bounds[1] + 1e-6
        assert context.log_explicit(outside) == -np.inf
    positions = module.initial_positions(
        context, 10, np.random.default_rng(819)
    )
    assert positions.shape == (10, 5)
    assert np.linalg.matrix_rank(positions - np.mean(positions, axis=0)) == 5
    assert np.all(np.isfinite(context.log_noncentered(positions)))


def test_fixed_posterior_statistics_preserve_five_dimensions(module):
    samples = np.random.default_rng(91).normal(size=(2000, 5))
    summary = module.posterior_statistics(samples)
    assert np.asarray(summary["quantiles"]).shape == (5, 5)
    assert np.asarray(summary["hdi68"]).shape == (5, 2)
    assert np.asarray(summary["hdi95"]).shape == (5, 2)
    assert np.asarray(summary["correlation"]).shape == (5, 5)
