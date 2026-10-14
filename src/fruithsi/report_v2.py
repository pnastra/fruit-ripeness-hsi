"""V5: the v2 comparison table, the pre-registered decision, and a VIP plot for firmness.

Also runs the two PLS-DA-on-pixnorm rows (hypothesis (a)), which V3/V4 did not need. Everything else
is read from MLflow (experiments baselines, v2_firmness, v2_ordinal).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from fruithsi.export import vip_scores
from fruithsi.firmness import FirmnessPLS
from fruithsi.ordinal import mean_abs_class_error
from fruithsi.predict import MODEL_PATH
from fruithsi.preprocess import load_pixels, load_pixnorm, load_spectra, snv
from fruithsi.split import grouped_folds

DOCS = Path("docs")
EXPERIMENT = "v2_plsda"
V1_F1 = 0.516  # shipped v1 model, pre-registered reference (PLAN §12, V1 row)
BAR = V1_F1 + 0.05
# (experiment, run, model, features, augmented) in table order
ROWS = [
    ("baselines", "majority_grouped", "majority", "-", False),
    ("v2_ordinal", "v1_reference_plsda_snv_aug", "PLS-DA (v1 shipped)", "mean, SNV", True),
    (EXPERIMENT, "plsda_pixnorm_aug", "PLS-DA", "per-pixel SNV", True),
    ("baselines", "plsda_snv_grouped", "PLS-DA, no aug (reference for the next row)",
     "mean, SNV", False),
    (EXPERIMENT, "plsda_pixnorm_core_noaug", "PLS-DA, no aug", "per-pixel SNV, rim dropped", False),
    ("v2_firmness", "firmness_snv_aug", "firmness PLS -> class", "mean, SNV", True),
    ("v2_firmness", "firmness_pixnorm_aug", "firmness PLS -> class", "per-pixel SNV", True),
    ("v2_ordinal", "ordinal_pls_snv_aug", "ordinal PLS", "mean, SNV", True),
    ("v2_ordinal", "ordinal_pls_pixnorm_aug", "ordinal PLS", "per-pixel SNV", True),
    ("v2_ordinal", "cumlogit_pls_snv_aug", "cumulative logit on PLS", "mean, SNV", True),
    ("v2_ordinal", "cumlogit_pls_pixnorm_aug", "cumulative logit on PLS", "per-pixel SNV", True),
]
CANDIDATES = {r[1] for r in ROWS[2:] if not r[1].startswith("plsda_snv")}


def run_pixnorm_plsda(y, groups, pixels_snv) -> None:
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    reset_experiment(EXPERIMENT)
    folds = list(grouped_folds(y, groups, 5, 5, 0))
    runs = {
        "plsda_pixnorm_aug": (load_pixnorm("pixnorm")[1],
                              lambda: PixelAugmented(lambda: PLSDA("raw"), pixels_snv, 20, 0.05)),
        "plsda_pixnorm_core_noaug": (load_pixnorm("pixnorm_core")[1], lambda: PLSDA("raw")),
    }
    for name, (X, make) in runs.items():
        cv = run_cv(make, X, y, groups, folds)
        cm = cv["confusion"]
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": "plsda", "features": name, "split": "grouped_by_fruit",
                               "n_splits": 5, "n_repeats": 5, "seed": 0,
                               "cv_components_chosen": sorted(cv["components"])})
            mlflow.log_metrics({
                **summarise(cv),
                "cv_mace_mean": float(np.mean([mean_abs_class_error(y, cv["oof"][r], groups)
                                               for r in cv["oof"]])),
                **{f"cv_recall_{c}": float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)},
            })
            mlflow.log_figure(_confusion_figure(cm, name), "confusion_cv.png")
        print(f"done {name}", flush=True)


def load_table():
    import mlflow
    import pandas as pd

    rows = []
    for exp, run, model, feats, aug in ROWS:
        r = mlflow.search_runs(experiment_names=[exp],
                               filter_string=f"tags.mlflow.runName = '{run}'").iloc[0]
        get = lambda k, r=r: r.get(f"metrics.{k}", np.nan)  # noqa: E731
        rows.append({"model": model, "features": feats, "aug": "yes" if aug else "no", "run": run,
                     "fruit_f1": get("cv_fruit_f1_mean"), "fruit_f1_sd": get("cv_fruit_f1_std"),
                     "fruit_acc": get("cv_fruit_acc_mean"), "mace": get("cv_mace_mean"),
                     "recall_unripe": get("cv_recall_unripe")})
    return pd.DataFrame(rows)


def vip_figure(wl, X_mean, X_pix, y, groups, firmness, path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v1 = np.asarray(json.loads(Path(MODEL_PATH).read_text())["vip"])
    fits = {"firmness PLS, mean SNV": FirmnessPLS(firmness, "snv").fit(X_mean, y, groups),
            "firmness PLS, per-pixel SNV": FirmnessPLS(firmness, "raw").fit(X_pix, y, groups)}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(wl, v1, color="#3d9970", lw=1.8, label="PLS-DA VIP (v1 shipped)")
    out = {}
    for (name, m), c in zip(fits.items(), ["#b5651d", "#7f7f7f"], strict=True):
        vip = vip_scores(m.pls_)
        ax.plot(wl, vip, color=c, lw=1.2, label=f"{name} ({m.n_components_} comp.)")
        out[name] = {"n_components": m.n_components_,
                     "top_nm": sorted(np.round(wl[np.argsort(vip)[::-1][:5]]).tolist()),
                     "spearman_vs_v1": float(np.corrcoef(np.argsort(np.argsort(vip)),
                                                         np.argsort(np.argsort(v1)))[0, 1])}
    ax.axhline(1, color="k", lw=0.6, ls=":")
    for nm, lab in [(680, "chlorophyll"), (970, "water")]:
        ax.axvline(nm, color="k", lw=0.5, alpha=0.4)
        ax.text(nm + 3, ax.get_ylim()[1] * 0.92, lab, fontsize=7)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("VIP")
    ax.set_title("Band importance: firmness models (fitted on all 80 images) vs v1", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return out


def main() -> None:
    import argparse

    import mlflow

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-runs", action="store_true", help="reuse the v2_plsda runs")
    args = ap.parse_args()
    mlflow.set_tracking_uri(MLFLOW_URI)

    df, X_mean, wl = load_spectra()
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    pixels_snv = [snv(p.astype(np.float64)) for p in load_pixels(df.hdr_path)]
    if not args.skip_runs:
        run_pixnorm_plsda(y, groups, pixels_snv)

    table = load_table()
    table["passes_bar"] = table.run.isin(CANDIDATES) & (table.fruit_f1 >= BAR)
    winners = table[table.passes_bar]
    perm = {}
    for run in winners.run:  # pre-registered: permutation test only for a candidate over the bar
        raise NotImplementedError(f"{run} passed the bar: add its permutation test here")
    firmness = df.groupby("fruit_id").firmness.first().to_dict()
    vip = vip_figure(wl, X_mean, load_pixnorm("pixnorm")[1], y, groups, firmness,
                     DOCS / "firmness_vip.png")
    table.to_csv(DOCS / "results_v2.csv", index=False)
    print(table.drop(columns="run").round(3).to_string(index=False))
    print(json.dumps(vip, indent=1))
    print("winners:", list(winners.run) or "none", perm)


if __name__ == "__main__":
    main()
