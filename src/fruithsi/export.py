"""Fit the final PLS-DA on all labelled images and export it for serving (`models/plsda_snv.json`).

Final model = the variant chosen by cross-validated macro-F1 (PLS-DA, SNV, pixel-subset
augmentation). Its probabilities come from a one-parameter temperature fitted on the out-of-fold
scores of the grouped CV, so the confidences reflect how weak the signal really is.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from fruithsi import CLASSES
from fruithsi.baseline import PLSDA, PixelAugmented, run_cv, summarise
from fruithsi.predict import MODEL_PATH, PLSDAModel
from fruithsi.preprocess import load_pixels, load_spectra
from fruithsi.split import grouped_folds

N_DRAWS, FRAC = 20, 0.05  # same augmentation as the evaluated run `plsda_snv_grouped_aug`
MODEL_VERSION = "plsda-snv-aug-0.1.0"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(scores: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = softmax(scores / temperature)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising the log loss of softmax(scores / T) (1-parameter calibration)."""
    res = minimize_scalar(
        lambda t: log_loss(scores, y, np.exp(t)), bounds=(-5, 5), method="bounded"
    )
    return float(np.exp(res.x))


def vip_scores(pls) -> np.ndarray:
    """Variable importance in projection of a fitted PLSRegression (mean square = 1)."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = (q**2).sum(axis=0) * (t**2).sum(axis=0)  # explained Y variance per component
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(w.shape[0] * (w_norm**2 @ ss) / ss.sum())


def affine_map(pls, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """(coef, intercept) with scores = Z @ coef.T + intercept, read off `pls.predict` itself.

    predict() is affine in Z, so predicting the zero vector gives the intercept and predicting the
    identity matrix gives the coefficients. Robust to how sklearn stores centring internally.
    """
    intercept = pls.predict(np.zeros((1, n_bands)))[0]
    coef = (pls.predict(np.eye(n_bands)) - intercept).T
    return coef, intercept


def main(out: Path = MODEL_PATH) -> None:
    df, X, wl = load_spectra()
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    pixels = load_pixels(df.hdr_path)

    def make():
        return PixelAugmented(lambda: PLSDA("snv"), pixels, n_draws=N_DRAWS, frac=FRAC)

    # 1) honest performance + out-of-fold scores (grouped CV: no fruit in train and test)
    n_repeats = 5
    cv = run_cv(make, X, y, groups, list(grouped_folds(y, groups, 5, n_repeats, 0)))
    summary = summarise(cv)
    oof = np.concatenate([cv["oof"][r] for r in sorted(cv["oof"])])
    labels = np.tile(y, n_repeats)

    # 2) temperature so that probabilities are calibrated on those out-of-fold scores
    temperature = fit_temperature(oof, labels)
    loss_t, loss_1 = log_loss(oof, labels, temperature), log_loss(oof, labels, 1.0)
    p = softmax(oof / temperature)
    calib = {
        "oof_log_loss": loss_t, "oof_log_loss_T1": loss_1, "chance_log_loss": float(np.log(3)),
        "mean_confidence": float(p.max(axis=1).mean()),
        "oof_image_accuracy": float((p.argmax(axis=1) == labels).mean()),
    }

    # 3) final model on all 80 images (40 fruits); components chosen by inner grouped CV
    final = make().fit(X, y, groups, idx=np.arange(len(y)))
    pls = final.model_.pls_
    coef, intercept = affine_map(pls, X.shape[1])

    meta = {
        "format_version": 1,
        "name": "plsda-snv-aug",
        "version": MODEL_VERSION,
        "created": date.today().isoformat(),
        "classes": list(CLASSES),
        "bands_nm": [round(float(w), 2) for w in wl],
        "transform": "snv",
        "n_components": int(final.model_.n_components_),
        "temperature": temperature,
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "vip": vip_scores(pls).tolist(),
        "training": {
            "dataset": "DeepHS Fruit 2023, Mango, Specim FX10 (VIS camera)",
            "n_images": len(y), "n_fruits": int(len(np.unique(groups))),
            "augmentation": f"pixel-subset means, {N_DRAWS} draws/image, fraction {FRAC}",
        },
        "cv_metrics": {**summary, "protocol": "StratifiedGroupKFold by fruit, 5 folds x 5 repeats",
                       "majority_fruit_acc": 0.4},
        "calibration": calib,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1))

    # 4) the exported numpy predictor must reproduce the sklearn model exactly
    exported = PLSDAModel.load(out)
    np.testing.assert_allclose(exported.scores(X), final.decision_function(X), atol=1e-9)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB), {meta['n_components']} components")
    print(f"CV fruit-level accuracy {summary['cv_fruit_acc_mean']:.3f} "
          f"(95% CI {summary['cv_fruit_acc_ci_low']:.2f}-{summary['cv_fruit_acc_ci_high']:.2f}), "
          f"macro-F1 {summary['cv_fruit_f1_mean']:.3f}")
    print(f"temperature {temperature:.3f}; OOF log loss {loss_t:.3f} (T=1: {loss_1:.3f}, "
          f"chance {np.log(3):.3f}); mean confidence {calib['mean_confidence']:.3f} "
          f"vs OOF image accuracy {calib['oof_image_accuracy']:.3f}")


if __name__ == "__main__":
    main()
