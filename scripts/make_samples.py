"""Write a few derived mean spectra to samples/ as ready-to-send /predict request bodies.

Only derived mean spectra (192 numbers each), never images. They come from the labelled training
images, so they are for trying the API, not for judging the model.
"""

import json
from pathlib import Path

from fruithsi.preprocess import load_spectra

OUT = Path("samples")
# (file name, class, side): the first fruit (lowest id) of each class, plus two back sides
PICKS = [
    ("unripe_front", "unripe", "front"), ("perfect_front", "perfect", "front"),
    ("overripe_front", "overripe", "front"), ("unripe_back", "unripe", "back"),
    ("overripe_back", "overripe", "back"),
]


def main() -> None:
    df, X, _ = load_spectra()
    OUT.mkdir(exist_ok=True)
    index = {}
    for name, cls, side in PICKS:
        row = df[(df.ripeness == cls) & (df.side == side)].sort_values("fruit_id").iloc[0]
        spectrum = [round(float(v), 4) for v in X[row.name]]
        (OUT / f"{name}.json").write_text(json.dumps({"spectrum": spectrum}))
        index[f"{name}.json"] = {"true_class": cls, "fruit_id": int(row.fruit_id), "side": side}
    (OUT / "index.json").write_text(json.dumps({
        "note": "Derived mean spectra (476.5-998.2 nm, 192 bands) of labelled TRAINING images, "
                "for trying /predict. Not for evaluating the model.",
        "samples": index,
    }, indent=1))
    print(f"wrote {len(PICKS)} samples to {OUT}/")


if __name__ == "__main__":
    main()
