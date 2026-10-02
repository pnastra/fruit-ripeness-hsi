"""1D-CNN training (loop with early stopping), evaluated with the same protocol as the baselines."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fruithsi import CLASSES
from fruithsi.baseline import METRIC_NAMES, MLFLOW_URI, evaluate_run, reset_experiment
from fruithsi.dataset import SpectraDataset
from fruithsi.model import SpectralCNN
from fruithsi.preprocess import load_pixels, load_spectra, transform
from fruithsi.split import grouped_holdout, random_holdout

EXPERIMENT = "cnn"


@dataclass(frozen=True)
class CNNConfig:
    transform_name: str = "snv"
    augment: bool = True  # pixel-subset-mean augmentation of the training images
    frac: float = 0.05  # fraction of the fruit pixels averaged per draw (~124 px; 0.4 was too weak)
    channels: tuple = (8, 16)
    kernel: int = 7
    dropout: float = 0.5
    label_smoothing: float = 0.1  # soft targets: stops the net from becoming overconfident
    lr: float = 1e-3
    weight_decay: float = 5e-2
    batch_size: int = 16
    max_epochs: int = 150
    patience: int = 25  # epochs without improvement (smoothed val loss) before stopping
    val_smoothing: float = 0.3  # EMA weight of the newest val loss: ~7 val fruits are very noisy
    val_fraction: float = 0.2  # share of the training *fruits* held out for early stopping


# At most three configurations (PLAN §7: no hyperparameter search).
CONFIGS = {
    "snv_aug": CNNConfig(),
    "snv_noaug": CNNConfig(augment=False),
    "snv_sg1_aug": CNNConfig(transform_name="snv_sg1"),
}


def _tensor(spectra: np.ndarray, device) -> torch.Tensor:
    """(n, bands) numpy -> (n, 1, bands) float tensor: the (batch, channel, length) Conv1d wants."""
    return torch.as_tensor(spectra, dtype=torch.float32).unsqueeze(1).to(device)


class CNNClassifier:
    """fit / decision_function wrapper so the CNN runs through the same protocol as PLS-DA."""

    def __init__(self, config: CNNConfig, pixels=None, seed: int = 0, device: str = "cpu"):
        self.config, self.pixels, self.seed, self.device = config, pixels, seed, device

    def _prepare(self, X) -> np.ndarray:
        """Plain (never augmented) spectra -> preprocessed and standardised with training stats."""
        return (transform(np.asarray(X), self.config.transform_name) - self.mean_) / self.std_

    def fit(self, X, y, groups=None, idx=None):
        cfg, device = self.config, self.device
        X, y = np.asarray(X), np.asarray(y)
        if cfg.augment and (self.pixels is None or idx is None):
            raise ValueError("augmentation needs the pixel arrays and the global image indices")

        # seed everything that is random: weight init, batch order, augmentation draws
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Early-stopping validation set: whole fruits held out of the *training* data.
        if groups is None:
            train, val = random_holdout(y, cfg.val_fraction, self.seed)
        else:
            train, val = grouped_holdout(y, groups, cfg.val_fraction, self.seed)

        # per-band standardisation from the training images only (never val/test)
        z_train = transform(X[train], cfg.transform_name)
        self.mean_, self.std_ = z_train.mean(axis=0), z_train.std(axis=0) + 1e-8

        pixels = [self.pixels[i] for i in idx[train]] if cfg.augment else None
        dataset = SpectraDataset(X[train], y[train], pixels, frac=cfg.frac,
                                 transform_name=cfg.transform_name, seed=self.seed,
                                 standardize=(self.mean_, self.std_))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(self.seed))
        x_val = _tensor(self._prepare(X[val]), device)
        y_val = torch.as_tensor(y[val], dtype=torch.long).to(device)

        model = SpectralCNN(cfg.channels, cfg.kernel, cfg.dropout, len(CLASSES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)  # training loss
        val_loss_fn = nn.CrossEntropyLoss()  # plain CE for validation, comparable across configs

        history = {"train": [], "val": []}
        best_loss, best_state, best_epoch, wait, smoothed = np.inf, None, 0, 0, None
        for epoch in range(cfg.max_epochs):
            model.train()  # dropout on, BatchNorm uses batch statistics
            total = 0.0
            for xb, yb in loader:  # one pass over the (freshly augmented) training images
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()  # clear gradients left over from the previous batch
                loss = loss_fn(model(xb), yb)  # forward pass + loss
                loss.backward()  # gradients of the loss w.r.t. every weight
                optimizer.step()  # AdamW update
                total += loss.item() * len(yb)
            model.eval()  # dropout off, BatchNorm uses running statistics
            with torch.no_grad():  # no gradient bookkeeping needed for validation
                val_loss = val_loss_fn(model(x_val), y_val).item()
            history["train"].append(total / len(dataset))
            history["val"].append(val_loss)

            # judge improvement on an exponential moving average of the validation loss
            smoothed = val_loss if smoothed is None else (
                cfg.val_smoothing * val_loss + (1 - cfg.val_smoothing) * smoothed)
            if smoothed < best_loss - 1e-4:  # improved: remember these weights
                best_loss, best_epoch, wait = smoothed, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= cfg.patience:  # early stopping
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.history_ = {**history, "best_epoch": best_epoch}
        return self

    def decision_function(self, X):
        """Class probabilities (softmax) for plain (never augmented) mean spectra."""
        z = _tensor(self._prepare(X), self.device)
        with torch.no_grad():
            return torch.softmax(self.model_(z), dim=1).cpu().numpy()


def main() -> None:
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    reset_experiment(EXPERIMENT)

    df, X, _ = load_spectra()
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    pixels = load_pixels(df.hdr_path)  # needs `make features` (pixels.npz)

    rows = []
    for name, cfg in CONFIGS.items():
        print(f"running cnn_{name} ...", flush=True)
        params = {"model": "cnn", **{k: str(v) for k, v in asdict(cfg).items()}}
        rows.append(evaluate_run(f"cnn_{name}", lambda cfg=cfg: CNNClassifier(cfg, pixels),
                                 X, y, groups, grouped=True, params=params))
    cols = ["name"] + [f"cv_{m}_mean" for m in METRIC_NAMES] + ["cv_fruit_acc_ci_low",
            "cv_fruit_acc_ci_high", "holdout_fruit_acc"]
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
