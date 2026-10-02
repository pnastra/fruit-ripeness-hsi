import numpy as np
import pytest

from fruithsi import CLASSES
from fruithsi.preprocess import load_spectra
from fruithsi.split import (
    grouped_folds,
    grouped_holdout,
    random_folds,
    random_holdout,
    shared_groups,
)

# 40 fruits x 2 images (front, back), labels per fruit like the real data: 13 / 16 / 11
FRUIT_LABELS = np.array([0] * 13 + [1] * 16 + [2] * 11)
GROUPS = np.repeat(np.arange(1, 41), 2)
Y = np.repeat(FRUIT_LABELS, 2)


def test_grouped_holdout_has_no_fruit_in_both_sets():
    for seed in range(10):
        train, test = grouped_holdout(Y, GROUPS, 0.2, seed)
        assert shared_groups(GROUPS, train, test) == set()
        assert len(np.unique(GROUPS[test])) == 8
        assert len(train) + len(test) == len(Y)
        assert set(Y[test]) == {0, 1, 2}  # stratified: every class is in the test fruits


def test_grouped_folds_no_overlap_and_partition_each_repeat():
    seen = {}
    for repeat, train, test in grouped_folds(Y, GROUPS, 5, 3, seed=0):
        assert shared_groups(GROUPS, train, test) == set()
        seen.setdefault(repeat, []).extend(test.tolist())
    assert sorted(seen) == [0, 1, 2]
    for tests in seen.values():  # every image is tested exactly once per repeat
        assert sorted(tests) == list(range(len(Y)))


def test_grouped_folds_reshuffle_between_repeats():
    first = {r: tuple(te) for r, _, te in grouped_folds(Y, GROUPS, 5, 2, seed=0)}
    assert first[0] != first[1]


def test_random_split_leaks_fruits():
    """The leakage demo only demonstrates something if the random split really does leak."""
    train, test = random_holdout(Y, 0.2, seed=0)
    assert len(shared_groups(GROUPS, train, test)) > 0
    assert any(shared_groups(GROUPS, tr, te) for _, tr, te in random_folds(Y, 5, 1, seed=0))


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/spectra.parquet").exists(),
    reason="run `make features` first",
)
def test_real_data_grouped_splits_have_no_overlap():
    df, _, _ = load_spectra()
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    g = df.fruit_id.to_numpy()
    assert shared_groups(g, *grouped_holdout(y, g)) == set()
    assert all(not shared_groups(g, tr, te) for _, tr, te in grouped_folds(y, g, 5, 5))
