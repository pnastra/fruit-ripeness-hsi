"""Baselines: majority class and PLS-DA, evaluated with grouped (and, for the demo, random) splits.

Every model has `fit(X, y, groups)` and `decision_function(X) -> (n, n_classes)` scores, so the same
protocol code evaluates all of them (the CNN in M6 follows the same interface).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fruithsi import CLASSES
from fruithsi.evaluate import LABELS, confusion, metrics, wilson_interval
from fruithsi.preprocess import load_pixels, load_spectra, subset_mean, transform
from fruithsi.split import (
    grouped_folds,
    grouped_holdout,
    random_folds,
    random_holdout,
    shared_groups,
)

N_CLASSES = len(CLASSES)
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "baselines"
METRIC_NAMES = ("image_acc", "image_f1", "fruit_acc", "fruit_f1")


class Majority:
    """Always predicts the most frequent training class (scores = class frequencies)."""

    def fit(self, X, y, groups=None, idx=None):
        self.freq_ = np.bincount(y, minlength=N_CLASSES) / len(y)
        return self

    def decision_function(self, X):
        return np.tile(self.freq_, (len(X), 1))


class PLSDA:
    """PLS discriminant analysis: PLS regression onto one-hot labels, class = argmax score.

    `n_components` is chosen inside `fit` by an inner cross-validation on the training data only
    (grouped by fruit when `groups` is given), picking the best macro-F1 (ties -> fewer components).
    """

    def __init__(self, transform_name="snv", n_components=None, max_components=10,
                 inner_splits=4, seed=0):
        self.transform_name = transform_name
        self.n_components = n_components
        self.max_components = max_components
        self.inner_splits = inner_splits
        self.seed = seed

    @staticmethod
    def _onehot(y):
        return np.eye(N_CLASSES)[y]

    def _fit_pls(self, Z, y, k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn warns when a component has ~0 variance
            return PLSRegression(n_components=k, scale=False).fit(Z, self._onehot(y))

    def _select_components(self, Z, y, groups):
        if groups is None:
            cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed)
            folds = list(cv.split(Z, y))
        else:
            cv = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=self.seed)
            folds = list(cv.split(Z, y, groups))
        ks = range(1, min(self.max_components, len(Z) - 1) + 1)
        scores = []
        for k in ks:
            f1s = []
            for tr, te in folds:
                pred = self._fit_pls(Z[tr], y[tr], k).predict(Z[te]).argmax(1)
                f1s.append(f1_score(y[te], pred, average="macro", labels=LABELS, zero_division=0))
            scores.append(np.mean(f1s))
        return list(ks)[int(np.argmax(scores))]  # argmax returns the first (fewest) on ties

    def fit(self, X, y, groups=None, idx=None):
        Z = transform(X, self.transform_name)
        y = np.asarray(y)
        self.n_components_ = self.n_components or self._select_components(Z, y, groups)
        self.pls_ = self._fit_pls(Z, y, self.n_components_)
        return self

    def decision_function(self, X):
        return self.pls_.predict(transform(X, self.transform_name))


class PixelAugmented:
    """Wrap a model so it trains on plain mean spectra plus `n_draws` pixel-subset means per image.

    Only the training images are augmented; `decision_function` sees plain spectra, so test fruits
    are never augmented. `pixels[i]` must be the pixel array of image i of the full X (`idx`
    gives the global image indices of the rows passed to `fit`).
    """

    def __init__(self, make_base: Callable, pixels, n_draws=20, frac=0.05, seed=0):
        self.make_base, self.pixels = make_base, pixels
        self.n_draws, self.frac, self.seed = n_draws, frac, seed

    def fit(self, X, y, groups=None, idx=None):
        rng = np.random.default_rng(self.seed)
        draws = np.stack([subset_mean(self.pixels[i], rng, self.frac)
                          for i in idx for _ in range(self.n_draws)])
        X_aug = np.concatenate([X, draws])
        y_aug = np.concatenate([y, np.repeat(y, self.n_draws)])
        g_aug = None
        if groups is not None:
            g_aug = np.concatenate([groups, np.repeat(groups, self.n_draws)])
        self.model_ = self.make_base().fit(X_aug, y_aug, g_aug)
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    @property
    def n_components_(self):
        return self.model_.n_components_


def run_cv(make_model: Callable, X, y, groups, folds) -> dict:
    """Fit on each train fold, score the test fold; metrics are computed per repeat.

    One repeat's test folds partition all images, so its pooled out-of-fold scores cover every
    fruit exactly once. Returns per-repeat metrics, summed confusion matrix and chosen components.
    """
    oof: dict[int, np.ndarray] = {}
    components, histories = [], []
    for repeat, train, test in folds:
        model = make_model().fit(X[train], y[train], groups[train], idx=train)
        oof.setdefault(repeat, np.full((len(y), N_CLASSES), np.nan))[test] = (
            model.decision_function(X[test])
        )
        if hasattr(model, "n_components_"):
            components.append(model.n_components_)
        if hasattr(model, "history_"):
            histories.append(model.history_)
    per_repeat = [metrics(y, oof[r], groups) for r in sorted(oof)]
    return {
        "per_repeat": per_repeat,
        "confusion": sum(confusion(y, oof[r]) for r in oof),
        "components": components,
        "histories": histories,
        "oof": oof,  # {repeat: (n_images, n_classes) out-of-fold scores}
    }


def run_holdout(make_model: Callable, X, y, groups, split) -> dict:
    train, test = split
    model = make_model().fit(X[train], y[train], groups[train], idx=train)
    out = metrics(y[test], model.decision_function(X[test]), groups[test])
    out["n_test_fruits"] = int(len(np.unique(groups[test])))
    out["n_components"] = getattr(model, "n_components_", None)
    return out


def summarise(cv: dict) -> dict[str, float]:
    """Mean and sd over repeats for each metric, plus a Wilson interval for fruit accuracy."""
    out = {}
    for name in METRIC_NAMES:
        vals = np.array([r[name] for r in cv["per_repeat"]])
        out[f"cv_{name}_mean"], out[f"cv_{name}_std"] = float(vals.mean()), float(vals.std())
    k = out["cv_fruit_acc_mean"] * 40  # 40 labelled fruits
    out["cv_fruit_acc_ci_low"], out["cv_fruit_acc_ci_high"] = wilson_interval(k, 40)
    return out


def _confusion_figure(cm: np.ndarray, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=30)
    ax.set_yticks(range(N_CLASSES), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def _loss_figure(histories: list[dict], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, h in enumerate(histories):
        ax.plot(h["train"], color="tab:blue", alpha=0.25, lw=0.8,
                label="train loss (one line per fold)" if k == 0 else None)
        ax.plot(h["val"], color="tab:orange", alpha=0.25, lw=0.8,
                label="validation loss" if k == 0 else None)
        ax.plot(h["best_epoch"], h["val"][h["best_epoch"]], "k.", ms=4,
                label="early-stopping epoch" if k == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def evaluate_run(name: str, make_model: Callable, X, y, groups, *, grouped: bool,
                 n_splits=5, n_repeats=5, seed=0, params: dict | None = None) -> dict:
    """Evaluate one model under one protocol and log it as one MLflow run."""
    import mlflow

    if grouped:
        folds = list(grouped_folds(y, groups, n_splits, n_repeats, seed))
        split = grouped_holdout(y, groups, 0.2, seed)
        # the whole point of the grouped protocol: never the same fruit on both sides
        assert all(not shared_groups(groups, tr, te) for _, tr, te in folds)
        assert not shared_groups(groups, *split)
    else:
        folds = list(random_folds(y, n_splits, n_repeats, seed))
        split = random_holdout(y, 0.2, seed)

    cv = run_cv(make_model, X, y, groups, folds)
    hold = run_holdout(make_model, X, y, groups, split)
    summary = summarise(cv)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "split": "grouped_by_fruit" if grouped else "random_by_image (LEAKAGE DEMO)",
            "n_splits": n_splits, "n_repeats": n_repeats, "seed": seed,
            "n_images": len(y), "n_fruits": len(np.unique(groups)), **(params or {}),
        })
        if cv["components"]:
            mlflow.log_param("cv_components_chosen", sorted(cv["components"]))
        mlflow.log_metrics(summary)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in hold.items()
                            if k.endswith(("acc", "f1"))})
        mlflow.log_metric("holdout_n_test_fruits", hold["n_test_fruits"])
        mlflow.log_figure(_confusion_figure(cv["confusion"], f"{name}\nCV, summed over repeats"),
                          "confusion_cv.png")
        if cv["histories"]:
            mlflow.log_metric("cv_best_epoch_mean", float(np.mean([h["best_epoch"]
                                                                   for h in cv["histories"]])))
            mlflow.log_figure(_loss_figure(cv["histories"], f"{name}: loss per CV fold"),
                              "loss_curves.png")
    return {"name": name, **summary, **{f"holdout_{k}": v for k, v in hold.items()}}


def permutation_test(make_model: Callable, X, y, groups, n_permutations: int = 200,
                     seed: int = 0) -> dict:
    """Is the grouped-CV fruit accuracy better than chance for this pipeline?

    Shuffles the labels across fruits (a fruit's two images keep the same label), reruns the
    identical grouped CV protocol each time, and compares the observed accuracy with that null.
    A null centred on chance also confirms that the grouped protocol itself does not leak.
    """
    def score(labels):
        folds = list(grouped_folds(labels, groups, 5, 5, 0))
        return summarise(run_cv(make_model, X, labels, groups, folds))["cv_fruit_acc_mean"]

    observed = score(y)
    fruit_ids, first = np.unique(groups, return_index=True)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = dict(zip(fruit_ids, rng.permutation(y[first]), strict=True))
        null.append(score(np.array([shuffled[g] for g in groups])))
    return summarise_permutations(observed, null, seeds=[seed])


def summarise_permutations(observed: float, null, seeds: list[int]) -> dict:
    """Summary of a null distribution; `null` is kept so chunks can be merged later."""
    null = np.asarray(null, dtype=float)
    return {
        "observed_fruit_acc": observed, "n_permutations": int(len(null)), "seeds": seeds,
        "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float(((null >= observed).sum() + 1) / (len(null) + 1)),
        "null": [round(float(v), 4) for v in null],
    }


def merge_permutations(old: dict, new: dict) -> dict:
    """Combine two chunks run with different seeds (same observed accuracy, more shuffles)."""
    if set(old["seeds"]) & set(new["seeds"]):
        raise ValueError("this seed was already run: use a different --perm-seed")
    if abs(old["observed_fruit_acc"] - new["observed_fruit_acc"]) > 1e-9:
        raise ValueError("the observed accuracy changed: rerun from scratch")
    return summarise_permutations(
        old["observed_fruit_acc"], old["null"] + new["null"], old["seeds"] + new["seeds"]
    )


def reset_experiment(name: str) -> None:
    """Delete the experiment's earlier runs so that rerunning a script replaces them."""
    import mlflow

    client = mlflow.MlflowClient()
    for run in client.search_runs([client.get_experiment_by_name(name).experiment_id]):
        client.delete_run(run.info.run_id)


def main() -> None:
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    reset_experiment(EXPERIMENT)

    df, X, _ = load_spectra()
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    pixels = load_pixels(df.hdr_path)  # needs `make features` (pixels.npz)
    print(f"{len(y)} labelled images, {len(np.unique(groups))} fruits, {X.shape[1]} bands")

    runs = [
        ("majority_grouped", Majority, True, {"model": "majority"}),
        ("plsda_snv_grouped", lambda: PLSDA("snv"), True, {"model": "plsda", "transform": "snv"}),
        ("plsda_snv_random", lambda: PLSDA("snv"), False, {"model": "plsda", "transform": "snv"}),
        ("plsda_sg1_grouped", lambda: PLSDA("sg1"), True, {"model": "plsda", "transform": "sg1"}),
        ("plsda_snv_grouped_aug",
         lambda: PixelAugmented(lambda: PLSDA("snv"), pixels, n_draws=20, frac=0.05), True,
         {"model": "plsda", "transform": "snv", "augmentation": "subset_mean x20, frac 0.05"}),
    ]
    rows = [evaluate_run(n, f, X, y, groups, grouped=g, params=p) for n, f, g, p in runs]

    cols = ["name"] + [f"cv_{m}_mean" for m in METRIC_NAMES] + ["cv_fruit_acc_ci_low",
            "cv_fruit_acc_ci_high", "holdout_fruit_acc", "holdout_image_acc"]
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
    print("\nMLflow UI: uv run mlflow ui --backend-store-uri", MLFLOW_URI)


if __name__ == "__main__":
    main()
