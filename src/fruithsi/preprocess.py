"""Mask the fruit, average to one spectrum per image, optional SNV / Savitzky-Golay transforms."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

CAMERA = "VIS"
# Bands below ~475 nm have SNR < 10 (second-difference noise estimate over 60 cubes, see PLAN §12).
BAND_RANGE_NM = (475.0, 1000.0)
MASK_BAND_NM = 850.0  # fruit is bright here; the background is exactly zero
# Not Otsu: the 850 nm histogram is a continuum (dim limb to bright centre), not two clusters,
# so Otsu cuts through the fruit. 0.10 sits just above the background and follows the true edge.
MASK_THRESHOLD = 0.10
MASK_ERODE_PX = 3  # drop the outer rim: fruit/background mixing and steep-angle limb
MIN_MASK_PIXELS = 200  # of 64x64 = 4096; smaller masks are flagged as suspicious


def select_bands(wavelengths: np.ndarray, band_range_nm=BAND_RANGE_NM) -> np.ndarray:
    """Boolean array marking the bands to keep (edge bands dropped)."""
    lo, hi = band_range_nm
    return (wavelengths >= lo) & (wavelengths <= hi)


def fruit_mask(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = MASK_THRESHOLD,
    erode: int = MASK_ERODE_PX,
) -> np.ndarray:
    """Boolean (lines, samples) mask of the fruit.

    Threshold the ~850 nm band, keep the largest connected blob, fill holes, then erode `erode`
    pixels so the rim (fruit mixed with background, steep-angle limb) is not averaged in.
    """
    band = cube[:, :, int(np.abs(wavelengths - MASK_BAND_NM).argmin())]
    mask = band > threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    mask = labels == (1 + int(np.argmax(sizes)))
    mask = ndimage.binary_fill_holes(mask)
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    return mask


def mean_spectrum(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average spectrum over the masked pixels -> (bands,)."""
    if not mask.any():
        raise ValueError("empty mask")
    return cube[mask].mean(axis=0)


def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard normal variate: each spectrum (row) minus its mean, divided by its std."""
    spectra = np.atleast_2d(spectra)
    return (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(axis=1, keepdims=True)


def savgol(spectra: np.ndarray, window: int = 11, polyorder: int = 2, deriv: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing (deriv=0) or derivative along the band axis."""
    return savgol_filter(np.atleast_2d(spectra), window, polyorder, deriv=deriv, axis=1)


TRANSFORMS = ("raw", "snv", "sg1", "snv_sg1")


def transform(spectra: np.ndarray, method: str = "raw") -> np.ndarray:
    """Apply one of TRANSFORMS to an array of spectra (n, bands)."""
    if method == "raw":
        return np.atleast_2d(spectra)
    if method == "snv":
        return snv(spectra)
    if method == "sg1":
        return savgol(spectra)
    if method == "snv_sg1":
        return savgol(snv(spectra))
    raise ValueError(f"unknown transform {method!r}; choose from {TRANSFORMS}")


def band_columns(wavelengths: np.ndarray) -> list[str]:
    return [f"nm_{w:.1f}" for w in wavelengths]


def build_spectra(raw_dir=None, camera: str = CAMERA):
    """Mean spectrum per image of `camera` -> DataFrame (metadata + one column per kept band)."""
    import pandas as pd

    from fruithsi.io import RAW_DIR, build_inventory, load_camera_info, read_cube

    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    wavelengths = load_camera_info(raw_dir)[camera]["wavelengths"]
    keep = select_bands(wavelengths)
    inv = build_inventory(raw_dir)
    inv = inv[inv.camera == camera].reset_index(drop=True)

    spectra, n_pixels = [], []
    for hdr in inv.hdr_path:
        cube = read_cube(raw_dir / hdr)
        mask = fruit_mask(cube, wavelengths)
        spectra.append(mean_spectrum(cube, mask)[keep])
        n_pixels.append(int(mask.sum()))

    out = inv.drop(columns=["camera"]).assign(n_pixels=n_pixels)
    bands = pd.DataFrame(np.asarray(spectra), columns=band_columns(wavelengths[keep]))
    return pd.concat([out, bands], axis=1)


def load_spectra(path=None, labelled_only: bool = True):
    """Read spectra.parquet -> (DataFrame, X (n, bands), wavelengths (bands,))."""
    import pandas as pd

    from fruithsi.io import PROCESSED_DIR

    df = pd.read_parquet(PROCESSED_DIR / "spectra.parquet" if path is None else path)
    if labelled_only:
        df = df[df.labelled].reset_index(drop=True)
    bands = [c for c in df.columns if c.startswith("nm_")]
    return df, df[bands].to_numpy(dtype=np.float64), np.array([float(b[3:]) for b in bands])


PIXELS_FILE = "pixels.npz"


def build_pixel_spectra(raw_dir=None, camera: str = CAMERA):
    """Masked per-pixel spectra of every labelled image -> (pixels, offsets, hdr_paths).

    `pixels` is one float32 array (total_pixels, bands) with the same band selection as the mean
    spectra; image i owns rows offsets[i]:offsets[i + 1]. Only labelled images are kept (the
    pixel-subset augmentation is only used for training on labelled images).
    """
    from fruithsi.io import RAW_DIR, build_inventory, load_camera_info, read_cube

    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    wavelengths = load_camera_info(raw_dir)[camera]["wavelengths"]
    keep = select_bands(wavelengths)
    inv = build_inventory(raw_dir)
    inv = inv[(inv.camera == camera) & inv.labelled].reset_index(drop=True)

    chunks = []
    for hdr in inv.hdr_path:
        cube = read_cube(raw_dir / hdr)
        chunks.append(cube[fruit_mask(cube, wavelengths)][:, keep].astype(np.float32))
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks])])
    return np.concatenate(chunks), offsets, inv.hdr_path.to_numpy(dtype=str)


def load_pixels(hdr_paths, path=None) -> list[np.ndarray]:
    """Per-image pixel arrays (n_pixels, bands), in the order of `hdr_paths`."""
    from fruithsi.io import PROCESSED_DIR

    with np.load(PROCESSED_DIR / PIXELS_FILE if path is None else path) as f:
        pixels, offsets, names = f["pixels"], f["offsets"], f["hdr_paths"]
    index = {str(n): i for i, n in enumerate(names)}
    return [pixels[offsets[index[h]] : offsets[index[h] + 1]] for h in hdr_paths]


def subset_mean(pixels: np.ndarray, rng: np.random.Generator, frac: float = 0.4) -> np.ndarray:
    """Mean spectrum of a random subset (fraction `frac`) of an image's fruit pixels.

    The augmentation: same fruit, same label, slightly different input on every draw.
    """
    n = max(1, int(round(frac * len(pixels))))
    return pixels[rng.choice(len(pixels), size=n, replace=False)].mean(axis=0)


def main() -> None:
    from fruithsi.io import PROCESSED_DIR

    df = build_spectra()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "spectra.parquet"
    df.to_parquet(out, index=False)
    bands = [c for c in df.columns if c.startswith("nm_")]
    print(f"wrote {out}: {len(df)} images x {len(bands)} bands ({bands[0]} .. {bands[-1]})")
    small = (df.n_pixels < MIN_MASK_PIXELS).sum()
    print(
        f"mask pixels: min {df.n_pixels.min()}, median {df.n_pixels.median():.0f}, "
        f"max {df.n_pixels.max()}; below {MIN_MASK_PIXELS}: {small}"
    )
    pixels, offsets, names = build_pixel_spectra()
    out = PROCESSED_DIR / PIXELS_FILE
    np.savez(out, pixels=pixels, offsets=offsets, hdr_paths=names)
    print(f"wrote {out}: {len(names)} labelled images, {len(pixels)} pixels x {pixels.shape[1]}")


if __name__ == "__main__":
    main()
