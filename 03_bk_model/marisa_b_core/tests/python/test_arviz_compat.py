import numpy as np

from marisa_b import arviz_compat


def test_split_rhat_matches_direct_variance_formula():
    values = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
            [0.25, 1.25, 2.25, 3.25, 4.25, 5.25],
        ]
    )
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]))
    draws = split.shape[1]
    within = np.mean(np.var(split, axis=1, ddof=1))
    between = draws * np.var(np.mean(split, axis=1), ddof=1)
    expected = np.sqrt((between / within + draws - 1.0) / draws)
    assert arviz_compat.rhat(values, method="split") == expected


def test_rank_rhat_and_identity_ess_preserve_parameter_axes():
    rng = np.random.default_rng(2031)
    values = rng.normal(size=(3, 4, 500, 2))
    rank_rhat = arviz_compat.rhat(
        values,
        method="rank",
        chain_axis=1,
        draw_axis=2,
    )
    identity_ess = arviz_compat.ess(
        values,
        method="identity",
        chain_axis=1,
        draw_axis=2,
    )
    assert rank_rhat.shape == (3, 2)
    assert identity_ess.shape == (3, 2)
    assert np.all(np.isfinite(rank_rhat))
    assert np.all(np.isfinite(identity_ess))


def test_constant_identity_ess_is_total_draw_count():
    values = np.ones((4, 25))
    assert arviz_compat.ess(values, method="identity") == values.size
