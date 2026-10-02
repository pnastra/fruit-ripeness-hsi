"""PyTorch Dataset over mean spectra, with optional pixel-subset-mean augmentation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from fruithsi.preprocess import subset_mean, transform


class SpectraDataset(Dataset):
    """One item = (spectrum as a (1, bands) float tensor, class index).

    Without `pixels` (evaluation): item i is the preprocessed plain mean spectrum of image i,
    identical on every call.

    With `pixels` (training augmentation): item i is a *new* random pixel-subset mean of image i
    on every call, then preprocessed. Same fruit and label, slightly different input each epoch.

    `standardize=(mean, std)` (per band, computed on the training images only) is applied after
    preprocessing: it removes the shape every spectrum shares, so the network sees what differs
    between samples.

    The random generator is stored on the dataset, so use it with DataLoader(num_workers=0):
    worker processes would each copy the same generator state and draw identical augmentations.
    """

    def __init__(self, X, y, pixels=None, *, frac=0.4, transform_name="snv", seed=0,
                 standardize=None):
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.pixels, self.frac, self.transform_name = pixels, frac, transform_name
        self.standardize = standardize
        self.rng = np.random.default_rng(seed)
        # fixed spectra are preprocessed once; augmented ones are preprocessed per draw
        self.Z = None
        if pixels is None:
            self.Z = self._standardize(transform(np.asarray(X), transform_name))

    def _standardize(self, spectra):
        if self.standardize is None:
            return spectra
        mean, std = self.standardize
        return (spectra - mean) / std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        if self.pixels is None:
            spectrum = self.Z[i]
        else:
            draw = subset_mean(self.pixels[i], self.rng, self.frac)
            spectrum = self._standardize(transform(draw, self.transform_name))[0]
        return torch.as_tensor(spectrum, dtype=torch.float32).unsqueeze(0), self.y[i]
