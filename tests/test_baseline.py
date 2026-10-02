import numpy as np

from fruithsi.baseline import PLSDA, Majority
from fruithsi.evaluate import confusion, fruit_level, metrics, wilson_interval

rng = np.random.default_rng(0)
N_FRUITS, N_BANDS = 30, 40
GROUPS = np.repeat(np.arange(N_FRUITS), 2)
Y = np.repeat(np.arange(N_FRUITS) % 3, 2)
# class-specific smooth spectra plus noise: an easy problem a working PLS-DA must solve
_BASE = np.stack([np.sin(np.linspace(0, 3, N_BANDS) * (c + 1)) for c in range(3)]) + 2
X = _BASE[Y] + rng.normal(0, 0.05, (len(Y), N_BANDS))


def test_majority_predicts_most_frequent_class():
    y = np.array([0, 1, 1, 1, 2])
    scores = Majority().fit(np.zeros((5, 3)), y).decision_function(np.zeros((4, 3)))
    assert scores.shape == (4, 3)
    assert (scores.argmax(1) == 1).all()


def test_plsda_learns_separable_classes_on_unseen_fruits():
    train = GROUPS < 20
    model = PLSDA("snv").fit(X[train], Y[train], GROUPS[train])
    pred = model.decision_function(X[~train]).argmax(1)
    assert (pred == Y[~train]).mean() > 0.9
    assert 1 <= model.n_components_ <= model.max_components


def test_plsda_works_without_groups():
    model = PLSDA("sg1").fit(X, Y, None)
    assert model.decision_function(X).shape == (len(Y), 3)


def test_fruit_level_averages_both_sides():
    # front says class 0 (0.6), back says class 1 (0.9): the fruit-level average picks class 1
    scores = np.array([[0.6, 0.4, 0.0], [0.0, 0.9, 0.1]])
    ids, y_f, s_f = fruit_level(scores, np.array([1, 1]), np.array([7, 7]))
    assert ids.tolist() == [7] and y_f.tolist() == [1] and s_f.argmax(1).tolist() == [1]
    m = metrics(np.array([1, 1]), scores, np.array([7, 7]))
    assert m["image_acc"] == 0.5 and m["fruit_acc"] == 1.0


def test_confusion_shape_and_wilson():
    cm = confusion(np.array([0, 1, 2, 2]), np.eye(3)[[0, 1, 2, 1]])
    assert cm.shape == (3, 3) and cm.sum() == 4 and cm[2, 1] == 1
    low, high = wilson_interval(20, 40)
    assert low < 0.5 < high and 0 <= low and high <= 1
    assert wilson_interval(0, 10)[0] == 0
