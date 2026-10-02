"""Metrics: image-level and fruit-level accuracy / macro-F1, confusion matrix, Wilson interval."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from fruithsi import CLASSES

LABELS = list(range(len(CLASSES)))


def fruit_level(scores, y, groups):
    """Average the scores of a fruit's images (front + back) -> (fruit_ids, y_fruit, scores)."""
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    fruit_ids, first = np.unique(groups, return_index=True)
    mean_scores = np.stack([scores[groups == g].mean(axis=0) for g in fruit_ids])
    return fruit_ids, y[first], mean_scores


def metrics(y, scores, groups) -> dict[str, float]:
    """Accuracy and macro-F1, per image and per fruit (scores: (n, n_classes), argmax = class)."""
    pred = np.argmax(scores, axis=1)
    _, y_f, s_f = fruit_level(scores, y, groups)
    pred_f = np.argmax(s_f, axis=1)
    return {
        "image_acc": accuracy_score(y, pred),
        "image_f1": f1_score(y, pred, average="macro", labels=LABELS, zero_division=0),
        "fruit_acc": accuracy_score(y_f, pred_f),
        "fruit_f1": f1_score(y_f, pred_f, average="macro", labels=LABELS, zero_division=0),
    }


def confusion(y, scores) -> np.ndarray:
    """Rows = true class, columns = predicted class, in CLASSES order."""
    return confusion_matrix(y, np.argmax(scores, axis=1), labels=LABELS)


def wilson_interval(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (honest about tiny n)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))
