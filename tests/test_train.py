from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fruithsi.baseline import PixelAugmented
from fruithsi.dataset import SpectraDataset
from fruithsi.evaluate import metrics
from fruithsi.io import PROCESSED_DIR
from fruithsi.model import SpectralCNN
from fruithsi.preprocess import load_pixels, load_spectra, subset_mean
from fruithsi.train import CNNClassifier, CNNConfig

B = 64
rng = np.random.default_rng(0)
N_FRUITS = 30
GROUPS = np.repeat(np.arange(N_FRUITS), 2)
Y = np.repeat(np.arange(N_FRUITS) % 3, 2)
_BASE = np.stack([np.sin(np.linspace(0, 6, B) * (c + 1)) for c in range(3)]) + 2
X = _BASE[Y] + rng.normal(0, 0.05, (len(Y), B))
# fake per-image pixel arrays: 200 noisy pixels around each image's spectrum
PIXELS = [x + rng.normal(0, 0.2, (200, B)) for x in X]
FAST = CNNConfig(augment=False, max_epochs=60, patience=60)


def test_model_shape_and_size():
    model = SpectralCNN()
    assert model(torch.randn(5, 1, 192)).shape == (5, 3)
    assert sum(p.numel() for p in model.parameters()) < 5000


def test_eval_dataset_is_deterministic():
    ds = SpectraDataset(X, Y)
    a, label = ds[3]
    assert a.shape == (1, B) and a.dtype == torch.float32 and label.dtype == torch.long
    assert torch.equal(a, ds[3][0])


def test_training_dataset_draws_fresh_pixel_subset_each_call():
    ds = SpectraDataset(X, Y, PIXELS, frac=0.1, transform_name="raw")
    first, second = ds[0][0], ds[0][0]
    assert not torch.equal(first, second)  # a new random subset every call
    full_mean = torch.as_tensor(PIXELS[0].mean(0), dtype=torch.float32)
    assert (first.squeeze() - full_mean).abs().max() < 0.2  # but still the same fruit


def test_standardize_is_applied():
    mean, std = X.mean(0), X.std(0) + 1e-8
    ds = SpectraDataset(X, Y, transform_name="raw", standardize=(mean, std))
    batch = torch.stack([ds[i][0] for i in range(len(ds))]).squeeze(1).numpy()
    np.testing.assert_allclose(batch.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(batch.std(0), 1, atol=1e-3)


def test_subset_mean_shape_and_range():
    pix = np.arange(20, dtype=float).reshape(10, 2)
    out = subset_mean(pix, np.random.default_rng(0), frac=0.3)
    assert out.shape == (2,) and pix.min() <= out.min() and out.max() <= pix.max()


def test_cnn_learns_separable_classes_and_loss_falls():
    train = GROUPS < 22
    idx = np.flatnonzero(train)
    model = CNNClassifier(FAST).fit(X[train], Y[train], GROUPS[train], idx=idx)
    h = model.history_
    assert h["train"][-1] < 0.5 * h["train"][0]
    pred = model.decision_function(X[~train])
    assert metrics(Y[~train], pred, GROUPS[~train])["image_acc"] > 0.9
    assert np.allclose(pred.sum(axis=1), 1)  # softmax probabilities


def test_cnn_is_reproducible_with_a_fixed_seed():
    idx = np.arange(len(Y))
    cfg = replace(FAST, max_epochs=8, patience=8)
    a = CNNClassifier(cfg, seed=3).fit(X, Y, GROUPS, idx=idx).decision_function(X)
    b = CNNClassifier(cfg, seed=3).fit(X, Y, GROUPS, idx=idx).decision_function(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_early_stopping_triggers_and_restores_best_epoch():
    cfg = replace(FAST, max_epochs=300, patience=3, val_smoothing=1.0)
    model = CNNClassifier(cfg).fit(X, Y, GROUPS, idx=np.arange(len(Y)))
    h = model.history_
    assert len(h["val"]) < 300
    assert h["val"][h["best_epoch"]] == min(h["val"][: h["best_epoch"] + 1])


def test_training_with_augmentation_runs_and_needs_pixels():
    idx = np.arange(len(Y))
    cfg = replace(FAST, augment=True, frac=0.1, max_epochs=5, patience=5, transform_name="raw")
    CNNClassifier(cfg, PIXELS).fit(X, Y, GROUPS, idx=idx)
    with pytest.raises(ValueError):
        CNNClassifier(cfg).fit(X, Y, GROUPS, idx=idx)


def test_dataloader_batches():
    batch, labels = next(iter(DataLoader(SpectraDataset(X, Y), batch_size=7)))
    assert batch.shape == (7, 1, B) and labels.shape == (7,)


class _Recorder:
    """Stub model: remembers what it was fit on."""

    def fit(self, X, y, groups=None, idx=None):
        self.fit_X, self.fit_y, self.fit_groups = X, y, groups
        return self

    def decision_function(self, X):
        self.scored_X = X
        return np.zeros((len(X), 3))


def test_pixel_augmentation_only_expands_training_data():
    rec = _Recorder()
    model = PixelAugmented(lambda: rec, PIXELS, n_draws=4, frac=0.1)
    idx = np.arange(10)
    model.fit(X[idx], Y[idx], GROUPS[idx], idx=idx)
    assert len(rec.fit_X) == 10 * (1 + 4)
    assert len(rec.fit_y) == len(rec.fit_groups) == 50
    assert set(rec.fit_groups) == set(GROUPS[idx])  # draws inherit their fruit id
    np.testing.assert_array_equal(rec.fit_X[:10], X[idx])  # originals kept unchanged
    model.decision_function(X[10:12])
    np.testing.assert_array_equal(rec.scored_X, X[10:12])  # test spectra are never augmented


def test_load_pixels_round_trip(tmp_path: Path):
    pixels = np.arange(30, dtype=np.float32).reshape(10, 3)
    offsets = np.array([0, 4, 10])
    names = np.array(["a.hdr", "b.hdr"])
    np.savez(tmp_path / "p.npz", pixels=pixels, offsets=offsets, hdr_paths=names)
    out = load_pixels(["b.hdr", "a.hdr"], path=tmp_path / "p.npz")
    np.testing.assert_array_equal(out[0], pixels[4:10])
    np.testing.assert_array_equal(out[1], pixels[0:4])


@pytest.mark.skipif(
    not (PROCESSED_DIR / "pixels.npz").exists() or not (PROCESSED_DIR / "spectra.parquet").exists(),
    reason="run `make features` first",
)
def test_pixels_match_mean_spectra():
    df, spectra, _ = load_spectra()
    pixels = load_pixels(df.hdr_path)
    assert len(pixels) == len(df) == 80
    for p, s in zip(pixels, spectra, strict=True):
        np.testing.assert_allclose(p.mean(axis=0), s, rtol=1e-3, atol=1e-4)
