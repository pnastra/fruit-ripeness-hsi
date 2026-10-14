import numpy as np
import pandas as pd
import pytest

from fruithsi.io import PROCESSED_DIR, RAW_DIR, load_camera_info
from fruithsi.preprocess import (
    BAND_RANGE_NM,
    TRANSFORMS,
    fruit_mask,
    mean_spectrum,
    select_bands,
    snv,
    transform,
)

WL = np.linspace(400, 1000, 61)  # fake camera, 10 nm steps


def disc_cube(radius=20, size=64, value=0.6):
    """Zero background with a bright disc 'fruit' whose spectrum ramps linearly with band."""
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < radius**2
    cube = np.zeros((size, size, len(WL)), dtype=np.float32)
    cube[disc] = value * np.linspace(0.5, 1.5, len(WL))
    return cube, disc


def test_mask_finds_disc_and_ignores_rim():
    cube, disc = disc_cube()
    mask = fruit_mask(cube, WL)
    assert mask.sum() < disc.sum()  # eroded
    assert not (mask & ~disc).any()  # nothing outside the fruit
    assert mask.sum() > 0.6 * disc.sum()


def test_mask_keeps_largest_blob_only():
    cube, disc = disc_cube()
    cube[2:5, 2:5, :] = 0.6  # small bright speck in the background
    assert not fruit_mask(cube, WL, erode=0)[2:5, 2:5].any()


def test_mean_spectrum():
    cube, _ = disc_cube()
    spec = mean_spectrum(cube, fruit_mask(cube, WL))
    expected = 0.6 * np.linspace(0.5, 1.5, len(WL))
    np.testing.assert_allclose(spec, expected, rtol=1e-4)  # float32 cube
    with pytest.raises(ValueError):
        mean_spectrum(cube, np.zeros(cube.shape[:2], dtype=bool))


def test_select_bands():
    keep = select_bands(WL)
    assert WL[keep].min() >= BAND_RANGE_NM[0] and WL[keep].max() <= BAND_RANGE_NM[1]


def test_snv_and_transforms():
    x = np.random.default_rng(0).random((5, 40)) + 1
    z = snv(x)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=1), 1)
    for method in TRANSFORMS:
        assert transform(x, method).shape == x.shape
    with pytest.raises(ValueError):
        transform(x, "nope")


def test_snv_removes_multiplicative_scale():
    x = np.random.default_rng(1).random((1, 40)) + 1
    np.testing.assert_allclose(snv(3.0 * x), snv(x))


@pytest.mark.skipif(
    not (PROCESSED_DIR / "spectra.parquet").exists() or not (RAW_DIR / "Mango").exists(),
    reason="run `make features` first",
)
def test_spectra_parquet():
    df = pd.read_parquet(PROCESSED_DIR / "spectra.parquet")
    wl = load_camera_info()["VIS"]["wavelengths"]
    bands = [c for c in df.columns if c.startswith("nm_")]
    assert len(bands) == select_bands(wl).sum() == 192
    assert len(df) == 562
    assert not df[bands].isna().any().any()
    assert np.isfinite(df[bands].to_numpy()).all()
    assert (df.n_pixels > 200).all()


def test_pixel_norm_matches_hand_computation():
    rng = np.random.default_rng(0)
    cube = rng.random((6, 6, 10)) + 0.5
    mask = np.zeros((6, 6), bool)
    mask[1:4, 2:5] = True
    keep = np.arange(10) >= 2
    px = cube[mask][:, keep]
    hand = ((px - px.mean(1, keepdims=True)) / px.std(1, keepdims=True)).mean(0)
    np.testing.assert_allclose(mean_spectrum(cube, mask, keep, pixel_norm=True), hand)
    np.testing.assert_allclose(mean_spectrum(cube, mask, keep), px.mean(0))


def test_pixel_norm_ignores_per_pixel_brightness():
    cube, disc = disc_cube()
    cube[disc] *= np.random.default_rng(2).uniform(0.5, 2.0, disc.sum())[:, None]
    m = fruit_mask(cube, WL)
    plain = mean_spectrum(cube, m, pixel_norm=True)
    np.testing.assert_allclose(plain, snv(np.linspace(0.5, 1.5, len(WL)))[0], atol=1e-5)


@pytest.mark.skipif(
    not (PROCESSED_DIR / "spectra_pixnorm.parquet").exists(), reason="run `make features-v2` first"
)
def test_pixnorm_parquet():
    from fruithsi.preprocess import PIXNORM_VARIANTS, load_pixnorm, load_spectra

    v1, _, wl1 = load_spectra()
    for variant in PIXNORM_VARIANTS:
        df, X, wl = load_pixnorm(variant)
        assert X.shape == (80, 192) and np.isfinite(X).all()
        np.testing.assert_array_equal(wl, wl1)
        assert list(df.hdr_path) == list(v1.hdr_path)
        np.testing.assert_allclose(X.mean(1), 0, atol=1e-9)  # mean of SNV pixels has mean 0
    core, _, _ = load_pixnorm("pixnorm_core")
    assert (core.n_pixels < v1.n_pixels).all()
