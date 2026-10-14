import numpy as np

from fruithsi.ordinal import CumulativeLogitPLS, cumlogit_proba, mean_abs_class_error, ordinal_code


def test_ordinal_code_runs_firm_to_soft():
    assert ordinal_code({1: 0, 2: 1, 3: 2}) == {1: 2.0, 2: 1.0, 3: 0.0}


def test_mean_abs_class_error_counts_two_steps_and_averages_per_fruit():
    y = np.array([0, 0, 2, 2])
    groups = np.array([1, 1, 2, 2])
    scores = np.eye(3)[[2, 2, 2, 2]]  # unripe fruit called overripe, overripe fruit correct
    assert mean_abs_class_error(y, scores, groups) == 1.0


def test_cumlogit_proba_is_a_distribution():
    p = cumlogit_proba(np.random.default_rng(0).normal(size=(50, 3)), np.ones(3), [-0.5, 0.7])
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1)


def test_cumulative_logit_pls_learns_ordered_classes():
    rng = np.random.default_rng(0)
    n_fruit = 36
    y_f = np.repeat([0, 1, 2], 12)
    groups = np.repeat(np.arange(n_fruit), 2)
    y = np.repeat(y_f, 2)
    X = np.outer(y, rng.normal(size=30)) + rng.normal(0, 0.3, (72, 30)) + 5
    train = groups % 6 != 0
    m = CumulativeLogitPLS(dict(zip(groups, y, strict=True)), "raw").fit(
        X[train], y[train], groups[train])
    assert (m.decision_function(X[~train]).argmax(1) == y[~train]).mean() > 0.8
