"""A small 1D-CNN over the band axis of one spectrum."""

from __future__ import annotations

import torch
from torch import nn


class SpectralCNN(nn.Module):
    """(batch, 1, bands) -> (batch, n_classes) logits.

    Each block is Conv1d -> BatchNorm -> ReLU -> MaxPool(2): the convolution looks at a window of
    neighbouring bands (absorption features), pooling halves the band axis. The band axis is then
    pooled to a fixed `n_segments` (not to 1) and flattened, so the classifier still knows
    *roughly where* in the spectrum a feature occurs: wavelength position is the information in a
    spectrum, and a global average would throw it away. Stays small (~1.5k parameters).
    """

    def __init__(self, channels=(8, 16), kernel=7, dropout=0.3, n_classes=3, n_segments=8):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(n_segments),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in * n_segments, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
