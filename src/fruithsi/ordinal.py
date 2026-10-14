"""V4: ordinal classification (unripe < perfect < overripe).

Two pre-registered models (PLAN §12, V1 row), each on the v1 features and on per-pixel SNV:
- ordinal PLS: PLS regression on the class rank, cut-points learned on training folds;
- cumulative logit (proportional odds) on the PLS scores.
Extra metric: mean absolute class error (calling an unripe fruit overripe costs 2, not 1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from fruithsi import CLASSES
from fruithsi.baseline import (
    MLFLOW_URI,
    PLSDA,
    PixelAugmented,
    _confusion_figure,
    reset_experiment,
    run_cv,
    summarise,
)
from fruithsi.evaluate import fruit_level
from fruithsi.firmness import FirmnessPLS
from fruithsi.preprocess import load_pixels, load_pixnorm, load_spectra, snv, transform
from fruithsi.split import grouped_folds, shared_groups

EXPERIMENT = "v2_ordinal"
N_CLASSES = len(CLASSES)
RIDGE = 1e-3  # fixed; only keeps the coefficients finite if training classes separate


def ordinal_code(y_by_fruit: dict) -> dict:
    """fruit_id -> 2 - class index, so that larger = firmer = less ripe (FirmnessPLS direction)."""
    return {g: float(N_CLASSES - 1 - k) for g, k in y_by_fruit.items()}


class OrdinalPLS(FirmnessPLS):
    """PLS regression on the class rank + learned cut-points (FirmnessPLS with a rank target)."""

    def __init__(self, y_by_fruit: dict, transform_name="snv", **kw):
        super().__init__(ordinal_code(y_by_fruit), transform_name, **kw)


def _cumlogit_nll(params, T, y, ridge):
    k = T.shape[1]
    beta, theta0, d = params[:k], params[k], params[k + 1]
    eta = T @ beta
    cdf = np.stack([expit(theta0 - eta), expit(theta0 + np.exp(d) - eta)], 1)
    p = np.column_stack([cdf[:, 0], cdf[:, 1] - cdf[:, 0], 1 - cdf[:, 1]])
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).sum() + ridge * beta @ beta


def cumlogit_proba(T, beta, theta) -> np.ndarray:
    eta = T @ beta
    cdf = np.stack([expit(theta[0] - eta), expit(theta[1] - eta)], 1)
    return np.column_stack([cdf[:, 0], cdf[:, 1] - cdf[:, 0], 1 - cdf[:, 1]])


class CumulativeLogitPLS:
    """Proportional-odds model P(Y <= j) = sigmoid(theta_j - t . beta) on PLS scores t.

    The PLS part (rank target, components by inner grouped CV) is OrdinalPLS's; scores are
    standardised with training statistics. decision_function returns class probabilities.
    """

    def __init__(self, y_by_fruit: dict, transform_name="snv", ridge=RIDGE):
        self.pls = OrdinalPLS(y_by_fruit, transform_name)
        self.transform_name, self.ridge = transform_name, ridge

    def _scores(self, X):
        T = self.pls.pls_.transform(transform(X, self.transform_name))
        return (T - self.mu_) / self.sd_

    def fit(self, X, y, groups, idx=None):
        self.pls.fit(X, y, groups)
        self.n_components_ = self.pls.n_components_
        T = self.pls.pls_.transform(transform(X, self.transform_name))
        self.mu_, self.sd_ = T.mean(0), T.std(0) + 1e-12
        T = (T - self.mu_) / self.sd_
        y = np.asarray(y)
        x0 = np.r_[np.zeros(T.shape[1]), -0.5, 0.0]
        res = minimize(_cumlogit_nll, x0, args=(T, y, self.ridge), method="L-BFGS-B")
        k = T.shape[1]
        self.beta_ = res.x[:k]
        self.theta_ = np.array([res.x[k], res.x[k] + np.exp(res.x[k + 1])])
        return self

    def decision_function(self, X):
        return cumlogit_proba(self._scores(X), self.beta_, self.theta_)


def mean_abs_class_error(y, scores, groups) -> float:
    """Fruit-level mean |true rank - predicted rank| (ranks = CLASSES indices)."""
    _, y_f, s_f = fruit_level(scores, y, groups)
    return float(np.abs(y_f - s_f.argmax(1)).mean())


def main() -> None:
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    reset_experiment(EXPERIMENT)

    df, X_mean, _ = load_spectra()
    pdf, X_pix, _ = load_pixnorm("pixnorm")
    assert list(pdf.hdr_path) == list(df.hdr_path)
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    y_by_fruit = dict(zip(groups, y, strict=True))
    pixels = load_pixels(df.hdr_path)
    pixels_snv = [snv(p.astype(np.float64)) for p in pixels]
    feats = {"snv": (X_mean, "snv", pixels, "snv of the mean spectrum (v1 features)"),
             "pixnorm": (X_pix, "raw", pixels_snv, "mean of per-pixel SNV spectra")}

    def aug(base, px):
        return lambda: PixelAugmented(base, px, n_draws=20, frac=0.05)

    runs = [("v1_reference_plsda_snv_aug", "snv", lambda t: PLSDA(t), "plsda (v1 shipped model)")]
    for f in feats:
        runs.append((f"ordinal_pls_{f}_aug", f, lambda t: OrdinalPLS(y_by_fruit, t), "ordinal_pls"))
        runs.append((f"cumlogit_pls_{f}_aug", f, lambda t: CumulativeLogitPLS(y_by_fruit, t),
                     "cumulative_logit_on_pls_scores"))

    folds = list(grouped_folds(y, groups, 5, 5, 0))
    assert all(not shared_groups(groups, tr, te) for _, tr, te in folds)
    rows = []
    for name, f, make_base, model_desc in runs:
        X, tname, px, desc = feats[f]
        cv = run_cv(aug(lambda make_base=make_base, tname=tname: make_base(tname), px),
                    X, y, groups, folds)
        summary = summarise(cv)
        cm = cv["confusion"]
        extra = {"cv_mace_mean": float(np.mean([mean_abs_class_error(y, cv["oof"][r], groups)
                                                for r in cv["oof"]])),
                 **{f"cv_recall_{c}": float(cm[i, i] / cm[i].sum())
                    for i, c in enumerate(CLASSES)}}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": model_desc, "features": desc, "split": "grouped_by_fruit",
                               "n_splits": 5, "n_repeats": 5, "seed": 0,
                               "augmentation": "subset_mean x20, frac 0.05",
                               "cv_components_chosen": sorted(cv["components"])})
            mlflow.log_metrics({**summary, **extra})
            mlflow.log_figure(_confusion_figure(cm, f"{name}\nCV, summed over repeats"),
                              "confusion_cv.png")
        rows.append({"name": name, **summary, **extra, "confusion": cm.tolist()})
        print(f"done {name}", flush=True)

    cols = ["name", "cv_fruit_f1_mean", "cv_fruit_f1_std", "cv_fruit_acc_mean", "cv_mace_mean",
            "cv_recall_unripe", "cv_recall_perfect", "cv_recall_overripe"]
    out = pd.DataFrame(rows)
    print(out[cols].round(3).to_string(index=False))
    for r in rows:
        print(r["name"], "confusion (rows true u/p/o, cols pred):", r["confusion"])


if __name__ == "__main__":
    main()
