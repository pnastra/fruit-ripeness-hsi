"""V3: PLS regression on firmness, turned into ripeness classes with cut-points from training folds.

Pre-registered in PLAN §12 (V1 row): raw firmness scale, the v1 grouped protocol and augmentation,
two feature sets (SNV of the mean spectrum; mean of per-pixel SNV spectra).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fruithsi import CLASSES
from fruithsi.baseline import (
    MLFLOW_URI,
    PixelAugmented,
    _confusion_figure,
    reset_experiment,
    summarise,
)
from fruithsi.evaluate import LABELS, confusion, metrics
from fruithsi.preprocess import load_pixels, load_pixnorm, load_spectra, snv, transform
from fruithsi.split import grouped_folds, shared_groups

EXPERIMENT = "v2_firmness"
UNRIPE, PERFECT, OVERRIPE = (CLASSES.index(c) for c in ("unripe", "perfect", "overripe"))
N_CUT_CANDIDATES = 60


def class_scores(f: np.ndarray, c_lo: float, c_hi: float) -> np.ndarray:
    """Scores (n, 3) in CLASSES order whose argmax cuts firmness at c_lo < c_hi.

    Firm = unripe, soft = overripe. The scores are linear in f, so averaging a fruit's image
    scores gives the same class as cutting the fruit's mean predicted firmness.
    """
    s = np.zeros((len(f), len(CLASSES)))
    s[:, UNRIPE], s[:, OVERRIPE] = f - c_hi, c_lo - f
    return s


def fit_cutpoints(f_pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The two cut-points (on predicted firmness) with the best macro-F1; ties -> first found."""
    cands = np.unique(np.quantile(f_pred, np.linspace(0.02, 0.98, N_CUT_CANDIDATES)))
    best, best_f1 = (cands[0], cands[-1]), -1.0
    for i, lo in enumerate(cands):
        for hi in cands[i + 1:]:
            pred = class_scores(f_pred, lo, hi).argmax(1)
            f1 = f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
            if f1 > best_f1:
                best, best_f1 = (float(lo), float(hi)), f1
    return best


class FirmnessPLS:
    """PLS regression on firmness + cut-points -> class scores (same interface as PLSDA).

    `firmness` maps fruit_id -> measured firmness; targets are looked up through `groups`, which
    also works for the augmented rows of PixelAugmented. Components (fewest RMSE) and cut-points
    both come from an inner grouped CV on the training data only.
    """

    def __init__(self, firmness: dict, transform_name="snv", max_components=10, inner_splits=4,
                 seed=0):
        self.firmness, self.transform_name = firmness, transform_name
        self.max_components, self.inner_splits, self.seed = max_components, inner_splits, seed

    @staticmethod
    def _fit_pls(Z, t, k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PLSRegression(n_components=k, scale=False).fit(Z, t)

    def fit(self, X, y, groups, idx=None):
        Z, y = transform(X, self.transform_name), np.asarray(y)
        t = np.array([self.firmness[g] for g in groups], dtype=float)
        cv = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        folds = list(cv.split(Z, y, groups))
        ks = list(range(1, min(self.max_components, len(Z) - 1) + 1))
        oof = np.zeros((len(ks), len(t)))
        for tr, te in folds:
            for j, k in enumerate(ks):
                oof[j, te] = self._fit_pls(Z[tr], t[tr], k).predict(Z[te]).ravel()
        rmse = np.sqrt(((oof - t) ** 2).mean(1))
        j = int(np.argmin(rmse))
        self.n_components_ = ks[j]
        self.cutpoints_ = fit_cutpoints(oof[j], y)
        self.pls_ = self._fit_pls(Z, t, self.n_components_)
        return self

    def predict_firmness(self, X):
        return self.pls_.predict(transform(X, self.transform_name)).ravel()

    def decision_function(self, X):
        return class_scores(self.predict_firmness(X), *self.cutpoints_)


def regression_metrics(t, f) -> dict[str, float]:
    t, f = np.asarray(t, float), np.asarray(f, float)
    ss = ((t - t.mean()) ** 2).sum()
    return {"rmse": float(np.sqrt(((t - f) ** 2).mean())),
            "r2": float(1 - ((t - f) ** 2).sum() / ss),
            "spearman": float(spearmanr(t, f).statistic)}


def run_firmness_cv(make_model, X, y, groups, firmness: dict, n_splits=5, n_repeats=5, seed=0):
    """The v1 grouped protocol; also collects out-of-fold predicted firmness."""
    folds = list(grouped_folds(y, groups, n_splits, n_repeats, seed))
    assert all(not shared_groups(groups, tr, te) for _, tr, te in folds)
    scores, f_oof, components = {}, {}, []
    for r, tr, te in folds:
        model = make_model().fit(X[tr], y[tr], groups[tr], idx=tr)
        scores.setdefault(r, np.zeros((len(y), len(CLASSES))))[te] = model.decision_function(X[te])
        inner = getattr(model, "model_", model)  # unwrap PixelAugmented
        f_oof.setdefault(r, np.zeros(len(y)))[te] = inner.predict_firmness(X[te])
        components.append(model.n_components_)
    fruit_ids, first = np.unique(groups, return_index=True)
    t_fruit = np.array([firmness[g] for g in fruit_ids])
    reg = []
    for r in sorted(f_oof):
        f_fruit = np.array([f_oof[r][groups == g].mean() for g in fruit_ids])
        reg.append(regression_metrics(t_fruit, f_fruit))
    return {
        "per_repeat": [metrics(y, scores[r], groups) for r in sorted(scores)],
        "confusion": sum(confusion(y, scores[r]) for r in scores),
        "components": components, "regression": reg, "f_oof": f_oof,
        "fruit_ids": fruit_ids, "t_fruit": t_fruit, "y_fruit": y[first],
    }


def _pred_vs_measured_figure(results: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"unripe": "tab:green", "perfect": "tab:orange", "overripe": "tab:brown"}
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4), sharey=True)
    for ax, (name, cv) in zip(np.atleast_1d(axes), results.items(), strict=True):
        groups = cv["groups"]
        f = np.mean([[cv["f_oof"][r][groups == g].mean() for g in cv["fruit_ids"]]
                     for r in cv["f_oof"]], axis=0)  # mean over repeats, per fruit
        for c in CLASSES:
            k = cv["y_fruit"] == CLASSES.index(c)
            ax.scatter(cv["t_fruit"][k], f[k], color=colors[c], label=c, edgecolor="w")
        lim = [min(cv["t_fruit"].min(), f.min()), max(cv["t_fruit"].max(), f.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        m = np.mean([r["r2"] for r in cv["regression"]])
        rho = np.mean([r["spearman"] for r in cv["regression"]])
        ax.set_title(f"{name}\nR² {m:.2f}, Spearman ρ {rho:.2f} (mean over repeats)", fontsize=9)
        ax.set_xlabel("measured firmness")
    np.atleast_1d(axes)[0].set_ylabel("predicted firmness (out-of-fold, grouped CV)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    import mlflow
    import pandas as pd

    from fruithsi.io import PROCESSED_DIR

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    reset_experiment(EXPERIMENT)

    df, X_mean, _ = load_spectra()
    pdf, X_pix, _ = load_pixnorm("pixnorm")
    assert list(pdf.hdr_path) == list(df.hdr_path)
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    firmness = df.groupby("fruit_id").firmness.first().to_dict()
    pixels = load_pixels(df.hdr_path)
    pixels_snv = [snv(p.astype(np.float64)) for p in pixels]

    variants = {
        "firmness_snv_aug": (X_mean, "snv", pixels, "snv of the mean spectrum (v1 features)"),
        "firmness_pixnorm_aug": (X_pix, "raw", pixels_snv, "mean of per-pixel SNV spectra"),
    }
    rows, results = [], {}
    for name, (X, tname, px, desc) in variants.items():
        def make(tname=tname, px=px):
            return PixelAugmented(lambda: FirmnessPLS(firmness, tname), px, n_draws=20, frac=0.05)

        cv = run_firmness_cv(make, X, y, groups, firmness)
        cv["groups"] = groups
        results[name] = cv
        summary = summarise(cv)
        reg = {f"cv_firm_{k}_mean": float(np.mean([r[k] for r in cv["regression"]]))
               for k in ("rmse", "r2", "spearman")}
        cm = cv["confusion"]
        recall = {f"cv_recall_{c}": float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": "pls_regression_firmness -> cut-points", "features": desc,
                               "target": "firmness (raw scale)", "split": "grouped_by_fruit",
                               "n_splits": 5, "n_repeats": 5, "seed": 0,
                               "augmentation": "subset_mean x20, frac 0.05",
                               "cv_components_chosen": sorted(cv["components"])})
            mlflow.log_metrics({**summary, **reg, **recall})
            mlflow.log_figure(_confusion_figure(cm, f"{name}\nCV, summed over repeats"),
                              "confusion_cv.png")
        rows.append({"name": name, **summary, **reg, **recall})

    fig = _pred_vs_measured_figure(results)
    out = PROCESSED_DIR.parent.parent / "docs" / "firmness_pred_vs_measured.png"
    fig.savefig(out, dpi=130)
    cols = ["name", "cv_fruit_f1_mean", "cv_fruit_acc_mean", "cv_firm_rmse_mean",
            "cv_firm_r2_mean", "cv_firm_spearman_mean", "cv_recall_unripe", "cv_recall_perfect",
            "cv_recall_overripe"]
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
