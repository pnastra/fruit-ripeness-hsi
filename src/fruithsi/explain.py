"""Band importance: PLS VIP scores (shipped model) vs CNN gradient x input, on one wavelength axis.

Both are descriptive: with 40 fruits and weak class signal they hint at where the models look,
they do not prove which wavelengths carry ripeness information.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fruithsi import CLASSES
from fruithsi.predict import MODEL_PATH
from fruithsi.preprocess import load_pixels, load_spectra
from fruithsi.train import CONFIGS, CNNClassifier

DOCS = Path("docs")
CNN_CONFIG = "snv_sg1_aug"  # the best CNN configuration (by CV macro-F1)
SEEDS = range(5)
REFERENCE_NM = {"chlorophyll absorption (~680 nm)": 680, "water absorption (~970 nm)": 970}
RED_EDGE_NM = (700, 750)


def cnn_attribution(X, y, groups, pixels, config, seeds=SEEDS) -> np.ndarray:
    """Mean |gradient x input| per band of the true-class logit, one row per training seed.

    The CNN sees per-band standardised spectra, so the attribution is in that input space
    (still one value per band, i.e. per wavelength). Each row is scaled to a maximum of 1.
    """
    rows = []
    for seed in seeds:
        model = CNNClassifier(config, pixels, seed=seed).fit(
            X, y, groups, idx=np.arange(len(y))
        )
        z = torch.as_tensor(model._prepare(X), dtype=torch.float32).unsqueeze(1)
        z.requires_grad_(True)
        logits = model.model_(z)  # eval mode: dropout off, BatchNorm uses running statistics
        logits[torch.arange(len(y)), torch.as_tensor(y)].sum().backward()
        attr = (z.grad * z).abs().squeeze(1).mean(dim=0).detach().numpy()  # (bands,)
        rows.append(attr / attr.max())
    return np.stack(rows)


def plot(wl, vip, cnn, spectrum, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax.axvspan(*RED_EDGE_NM, color="gray", alpha=0.12, label="red edge (700-750 nm)")
    for label, nm in REFERENCE_NM.items():
        ax.axvline(nm, color="gray", ls="--", lw=0.9)
        ax.text(nm + 3, 1.02, label.split(" (")[0], rotation=90, va="top", fontsize=7, color="gray")
    ax.plot(wl, vip / vip.max(), color="#3d9970", lw=1.8, label="PLS VIP (shipped model)")
    mean, lo, hi = cnn.mean(axis=0), cnn.min(axis=0), cnn.max(axis=0)
    ax.plot(wl, mean, color="#4c78a8", lw=1.8,
            label=f"CNN gradient x input (mean of {len(cnn)} seeds)")
    ax.fill_between(wl, lo, hi, color="#4c78a8", alpha=0.18, label="CNN min-max over seeds")
    ax.set_ylabel("importance (each scaled to max 1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax2.plot(wl, spectrum, color="k", lw=1.2)
    ax2.set_ylabel("mean SNV\nspectrum")
    ax2.set_xlabel("wavelength (nm)")
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    meta = json.loads(MODEL_PATH.read_text())
    wl = np.asarray(meta["bands_nm"])
    vip = np.asarray(meta["vip"])

    df, X, wl_data = load_spectra()
    np.testing.assert_allclose(wl, wl_data, atol=0.06)
    y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df.fruit_id.to_numpy()
    pixels = load_pixels(df.hdr_path)

    cnn = cnn_attribution(X, y, groups, pixels, CONFIGS[CNN_CONFIG])
    cnn_mean = cnn.mean(axis=0)

    from fruithsi.preprocess import snv

    DOCS.mkdir(exist_ok=True)
    plot(wl, vip, cnn, snv(X).mean(axis=0), DOCS / "band_importance.png")
    np.savetxt(DOCS / "band_importance.csv", np.column_stack([wl, vip, vip / vip.max(), cnn_mean]),
               delimiter=",", header="wavelength_nm,vip,vip_scaled,cnn_grad_x_input_scaled",
               fmt=["%.1f", "%.4f", "%.4f", "%.4f"], comments="")

    def top(v, k=5):
        return sorted(float(w) for w in wl[np.argsort(v)[::-1][:k]])

    pairs = [(i, j) for i in range(len(cnn)) for j in range(i + 1, len(cnn))]
    seed_corr = [spearmanr(cnn[i], cnn[j])[0] for i, j in pairs]
    stats = {
        "cnn_config": CNN_CONFIG, "n_seeds": len(cnn),
        "spearman_vip_vs_cnn": float(spearmanr(vip, cnn_mean)[0]),
        "spearman_between_cnn_seeds_mean": float(np.mean(seed_corr)),
        "top5_vip_nm": top(vip), "top5_cnn_nm": top(cnn_mean),
        "mean_importance_by_region": {
            name: {"vip_scaled": float((vip / vip.max())[(wl >= lo) & (wl <= hi)].mean()),
                   "cnn_scaled": float(cnn_mean[(wl >= lo) & (wl <= hi)].mean())}
            for name, (lo, hi) in {"500-600": (500, 600), "600-680": (600, 680),
                                   "680-700": (680, 700), "700-750 red edge": (700, 750),
                                   "750-900 NIR plateau": (750, 900),
                                   "900-1000 water": (900, 1000)}.items()
        },
    }
    (DOCS / "band_importance_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {DOCS}/band_importance.png, .csv, _stats.json")


if __name__ == "__main__":
    main()
