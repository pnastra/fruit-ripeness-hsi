import numpy as np

from fruithsi import CLASSES
from fruithsi.firmness import FirmnessPLS, class_scores, fit_cutpoints, regression_metrics

U, P, OVER = (CLASSES.index(c) for c in ("unripe", "perfect", "overripe"))


def test_class_scores_cut_firmness_and_average_linearly():
    f = np.array([1.0, 5.0, 9.0])
    assert list(class_scores(f, 3, 7).argmax(1)) == [OVER, P, U]
    # the mean of two images' scores = the scores of their mean firmness
    pair = class_scores(np.array([2.0, 10.0]), 3, 7).mean(0)
    np.testing.assert_allclose(pair, class_scores(np.array([6.0]), 3, 7)[0])


def test_fit_cutpoints_recovers_separable_classes():
    f = np.r_[np.linspace(0, 1, 20), np.linspace(2, 3, 20), np.linspace(4, 5, 20)]
    y = np.repeat([OVER, P, U], 20)
    lo, hi = fit_cutpoints(f, y)
    assert 1 <= lo < 2 and 3 <= hi < 4


def test_firmness_pls_learns_a_linear_target_and_uses_only_training_fruits():
    rng = np.random.default_rng(0)
    n_fruit = 30
    t = rng.uniform(2000, 18000, n_fruit)
    y_f = np.where(t > 12000, U, np.where(t > 7000, P, OVER))
    groups = np.repeat(np.arange(n_fruit), 2)
    w = rng.normal(size=40)
    X = np.outer(np.repeat(t, 2), w) / 1e4 + rng.normal(0, 0.01, (60, 40)) + 5
    train = groups < 24
    firm = {g: t[g] for g in range(24)}  # test fruits' firmness is not even available
    m = FirmnessPLS(firm, transform_name="raw").fit(X[train], np.repeat(y_f, 2)[train],
                                                     groups[train])
    pred = m.predict_firmness(X[~train])
    assert regression_metrics(np.repeat(t, 2)[~train], pred)["r2"] > 0.9
