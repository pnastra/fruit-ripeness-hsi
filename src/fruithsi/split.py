"""Train/test splits. Reported metrics use the grouped splits (by physical fruit).

The random-by-image splits exist only to demonstrate leakage: the front and back images of one
fruit are near-duplicates, so a random split puts the same fruit in both train and test.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

Split = tuple[np.ndarray, np.ndarray]  # (train_idx, test_idx) into the image arrays


def grouped_holdout(y, groups, test_fraction: float = 0.2, seed: int = 0) -> Split:
    """Hold out ~test_fraction of the *fruits*, stratified by class (one label per fruit)."""
    y, groups = np.asarray(y), np.asarray(groups)
    fruit_ids, first = np.unique(groups, return_index=True)
    train_f, test_f = train_test_split(
        fruit_ids, test_size=test_fraction, stratify=y[first], random_state=seed
    )
    return np.flatnonzero(np.isin(groups, train_f)), np.flatnonzero(np.isin(groups, test_f))


def grouped_folds(y, groups, n_splits: int = 5, n_repeats: int = 5, seed: int = 0) -> Iterator:
    """Yield (repeat, train_idx, test_idx): stratified group k-fold, reshuffled each repeat.

    Within one repeat every image is in exactly one test fold, so the pooled out-of-fold
    predictions of a repeat cover all fruits once.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train, test in cv.split(np.zeros(len(y)), y, groups):
            yield r, train, test


def random_holdout(y, test_fraction: float = 0.2, seed: int = 0) -> Split:
    """Leakage demo: random split by image, ignoring which fruit an image belongs to."""
    y = np.asarray(y)
    train, test = train_test_split(
        np.arange(len(y)), test_size=test_fraction, stratify=y, random_state=seed
    )
    return np.sort(train), np.sort(test)


def random_folds(y, n_splits: int = 5, n_repeats: int = 5, seed: int = 0) -> Iterator:
    """Leakage demo: repeated stratified k-fold by image (same protocol, no grouping)."""
    y = np.asarray(y)
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train, test in cv.split(np.zeros(len(y)), y):
            yield r, train, test


def shared_groups(groups, train_idx, test_idx) -> set:
    """Fruit ids that appear in both train and test (must be empty for a valid grouped split)."""
    groups = np.asarray(groups)
    return set(groups[train_idx]) & set(groups[test_idx])
