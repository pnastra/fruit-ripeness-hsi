"""Serving side: load the exported PLS-DA model and predict with plain NumPy.

Deliberately imports nothing but numpy and the standard library (no scikit-learn, scipy, torch or
pandas), so the API and its Docker image stay tiny. A PLS regression is a linear model, so the
exported file only needs the preprocessing settings and one coefficient matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/plsda_snv.json")


def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard normal variate per spectrum (same formula as `fruithsi.preprocess.snv`)."""
    spectra = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    # a constant spectrum has std ~1e-17 from rounding, not exactly 0: compare with a tolerance
    if (std <= 1e-12 + 1e-9 * np.abs(mean)).any():
        raise ValueError("flat spectrum (zero variance): cannot apply SNV")
    return (spectra - mean) / std


class PLSDAModel:
    """Exported PLS-DA: scores = SNV(x) @ coef.T + intercept; proba = softmax(scores / T)."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.classes: list[str] = meta["classes"]
        self.bands_nm = np.asarray(meta["bands_nm"], dtype=np.float64)
        self.coef = np.asarray(meta["coef"], dtype=np.float64)  # (n_classes, n_bands)
        self.intercept = np.asarray(meta["intercept"], dtype=np.float64)  # (n_classes,)
        self.temperature = float(meta["temperature"])
        self.version: str = meta["version"]
        if meta["transform"] != "snv":
            raise ValueError(f"unsupported transform {meta['transform']!r}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> PLSDAModel:
        return cls(json.loads(Path(path).read_text()))

    @property
    def n_bands(self) -> int:
        return len(self.bands_nm)

    def _check(self, spectra) -> np.ndarray:
        spectra = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
        if spectra.shape[1] != self.n_bands:
            raise ValueError(f"expected {self.n_bands} bands, got {spectra.shape[1]}")
        if not np.isfinite(spectra).all():
            raise ValueError("spectrum contains NaN or infinite values")
        return spectra

    def scores(self, spectra) -> np.ndarray:
        """Raw PLS-DA class scores, shape (n, n_classes)."""
        return snv(self._check(spectra)) @ self.coef.T + self.intercept

    def predict_proba(self, spectra) -> np.ndarray:
        """Class probabilities: temperature-scaled softmax of the scores, shape (n, n_classes)."""
        z = self.scores(spectra) / self.temperature
        z = z - z.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, spectrum) -> dict:
        """One spectrum -> {"class", "confidence", "probabilities"}."""
        p = self.predict_proba(spectrum)[0]
        return {
            "class": self.classes[int(p.argmax())],
            "confidence": float(p.max()),
            "probabilities": {c: float(v) for c, v in zip(self.classes, p, strict=True)},
        }
