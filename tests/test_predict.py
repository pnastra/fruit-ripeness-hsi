import json
from pathlib import Path

import numpy as np
import pytest

from fruithsi import CLASSES
from fruithsi.baseline import PLSDA
from fruithsi.export import affine_map, fit_temperature, softmax, vip_scores
from fruithsi.io import PROCESSED_DIR
from fruithsi.predict import MODEL_PATH, PLSDAModel, snv
from fruithsi.preprocess import load_spectra
from fruithsi.preprocess import snv as snv_training

B = 30
rng = np.random.default_rng(0)
Y = np.repeat(np.arange(3), 20)
_BASE = np.stack([np.sin(np.linspace(0, 4, B) * (c + 1)) for c in range(3)]) + 2
X = _BASE[Y] + rng.normal(0, 0.1, (len(Y), B))


def make_model(tmp_path: Path, temperature: float = 0.5):
    """Fit a small PLS-DA, export it the way export.py does, and load the numpy predictor."""
    fitted = PLSDA("snv", n_components=3).fit(X, Y)
    coef, intercept = affine_map(fitted.pls_, B)
    meta = {
        "classes": list(CLASSES), "bands_nm": np.linspace(500, 1000, B).tolist(),
        "transform": "snv", "coef": coef.tolist(), "intercept": intercept.tolist(),
        "temperature": temperature, "version": "test",
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(meta))
    return fitted, PLSDAModel.load(path)


def test_numpy_snv_matches_training_snv():
    np.testing.assert_allclose(snv(X), snv_training(X))


def test_exported_model_reproduces_sklearn_scores(tmp_path):
    fitted, model = make_model(tmp_path)
    np.testing.assert_allclose(model.scores(X), fitted.decision_function(X), atol=1e-9)


def test_probabilities_and_predict_output(tmp_path):
    _, model = make_model(tmp_path)
    p = model.predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    out = model.predict(X[0])
    assert out["class"] in CLASSES
    assert out["confidence"] == pytest.approx(max(out["probabilities"].values()))
    assert sum(out["probabilities"].values()) == pytest.approx(1)
    assert (model.predict_proba(X).argmax(1) == Y).mean() > 0.9  # easy synthetic classes


def test_higher_temperature_means_less_confident(tmp_path):
    _, sharp = make_model(tmp_path, temperature=0.2)
    _, soft = make_model(tmp_path, temperature=5.0)
    assert sharp.predict_proba(X).max(axis=1).mean() > soft.predict_proba(X).max(axis=1).mean()


def test_input_validation(tmp_path):
    _, model = make_model(tmp_path)
    with pytest.raises(ValueError, match="expected 30 bands"):
        model.predict(np.ones(B + 1))
    with pytest.raises(ValueError, match="NaN"):
        model.predict(np.full(B, np.nan))
    with pytest.raises(ValueError, match="flat"):
        model.predict(np.full(B, 0.3))


def test_fit_temperature_recovers_the_true_temperature():
    r = np.random.default_rng(1)
    logits = r.normal(0, 2, (6000, 3))
    true_t = 2.0
    p = softmax(logits / true_t)
    labels = np.array([r.choice(3, p=row) for row in p])
    assert fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.1)


def test_vip_scores_have_mean_square_one():
    pls = PLSDA("snv", n_components=3).fit(X, Y).pls_
    vip = vip_scores(pls)
    assert vip.shape == (B,) and (vip >= 0).all()
    assert np.mean(vip**2) == pytest.approx(1.0)


# --- the committed model file --------------------------------------------------------------
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make model` first")


@needs_model
def test_committed_model_contract():
    model = PLSDAModel.load()
    assert model.classes == list(CLASSES)
    assert model.n_bands == 192 and model.coef.shape == (3, 192) and model.intercept.shape == (3,)
    assert 0.1 < model.temperature < 10
    assert 476 < model.bands_nm[0] < 477 and 998 < model.bands_nm[-1] < 999
    meta = json.loads(MODEL_PATH.read_text())
    assert meta["training"]["n_fruits"] == 40 and len(meta["vip"]) == 192
    assert meta["cv_metrics"]["majority_fruit_acc"] == 0.4


@needs_model
@pytest.mark.skipif(not (PROCESSED_DIR / "spectra.parquet").exists(), reason="run `make features`")
def test_committed_model_predicts_real_spectra():
    model = PLSDAModel.load()
    df, spectra, wl = load_spectra()
    np.testing.assert_allclose(wl, model.bands_nm, atol=0.06)  # same bands as the training data
    p = model.predict_proba(spectra)
    assert p.shape == (80, 3) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    in_sample = (np.array(CLASSES)[p.argmax(1)] == df.ripeness.to_numpy()).mean()
    assert in_sample > 0.5  # in-sample only: well above the 40% majority rate
