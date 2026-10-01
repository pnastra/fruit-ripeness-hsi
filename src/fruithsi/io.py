"""Read ENVI hyperspectral cubes and the DeepHS annotations, and build the image inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FRUIT = "Mango"

# The three annotation files together cover every image exactly once. Their record `id`s collide
# across files, so images are keyed by (camera, header path), never by id.
ANNOTATION_FILES = {
    "train": "train_all_v2.json",
    "val": "val_v2.json",
    "test": "test_v2.json",
}

# e.g. Mango/VIS/day_10_m3/mango_day_10_m3_33_back.hdr -> day 10, fruit 33, back
_NAME = re.compile(r"mango_day_(?P<day>\d+)_m3_(?P<fruit>\d+)_(?P<side>front|back)\.hdr$")


def parse_name(path: str | Path) -> tuple[int, int, str]:
    """Return (day, fruit_id, side) from a Mango header file name."""
    m = _NAME.search(str(path))
    if m is None:
        raise ValueError(f"not a Mango header file name: {path}")
    return int(m["day"]), int(m["fruit"]), m["side"]


def read_cube(hdr_path: str | Path) -> np.ndarray:
    """Read one ENVI cube as a float32 array of shape (lines, samples, bands).

    The `.bin` data file sits next to the `.hdr` header with the same stem.
    """
    import spectral.io.envi as envi

    hdr_path = Path(hdr_path)
    img = envi.open(str(hdr_path), image=str(hdr_path.with_suffix(".bin")))
    return np.asarray(img.load(), dtype=np.float32)


def load_camera_info(raw_dir: str | Path = RAW_DIR) -> dict[str, dict]:
    """Camera id -> {"name": str, "wavelengths": np.ndarray (nm)}.

    The ENVI headers carry no wavelengths; they live in the annotation files.
    """
    path = Path(raw_dir) / "annotations" / ANNOTATION_FILES["train"]
    cameras = json.loads(path.read_text())["cameras"]
    return {
        c["id"]: {"name": c["name"], "wavelengths": np.asarray(c["wavelengths"], dtype=np.float64)}
        for c in cameras
    }


def build_inventory(raw_dir: str | Path = RAW_DIR, fruit: str = FRUIT):
    """One row per image of `fruit`: ids, camera, day, side, path, official split and labels.

    `fruit_id` is the physical fruit (1-40 for Mango): the same fruits are imaged every day until
    each is measured destructively, so the day is NOT part of the id.
    Unlabelled images (all days before a fruit's last) have NaN/None labels.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    rows = []
    for split, fname in ANNOTATION_FILES.items():
        data = json.loads((raw_dir / "annotations" / fname).read_text())
        labels = {a["record_id"]: a for a in data["annotations"]}
        for rec in data["records"]:
            if rec["fruit"] != fruit:
                continue
            hdr = rec["files"]["header_file"]
            day, fruit_id, side = parse_name(hdr)
            ann = labels.get(rec["id"], {})
            rows.append(
                {
                    "camera": rec["camera_type"],
                    "day": day,
                    "fruit_id": fruit_id,
                    "side": side,
                    "hdr_path": hdr,
                    "official_split": split,
                    "ripeness": ann.get("ripeness_state"),
                    "ripeness_fine": ann.get("ripeness_state_fine"),
                    "firmness": ann.get("firmness"),
                    "storage_days": ann.get("storage_days"),
                }
            )
    df = pd.DataFrame(rows)
    df["labelled"] = df["ripeness"].notna()
    return df.sort_values(["camera", "fruit_id", "day", "side"]).reset_index(drop=True)


def main() -> None:
    """Print cube info for one example, then write and summarise the inventory."""
    cameras = load_camera_info()
    inv = build_inventory()

    example = inv[(inv.camera == "VIS") & inv.labelled].iloc[0]
    cube = read_cube(RAW_DIR / example.hdr_path)
    wl = cameras[example.camera]["wavelengths"]
    print(f"example: {example.hdr_path}")
    print(f"camera : {example.camera} = {cameras[example.camera]['name']}")
    print(f"shape  : {cube.shape} (lines, samples, bands), dtype {cube.dtype}")
    print(f"bands  : {len(wl)} from {wl[0]:.1f} to {wl[-1]:.1f} nm")
    print(f"values : min {cube.min():.3f}, max {cube.max():.3f}, mean {cube.mean():.3f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "inventory.parquet"
    inv.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(inv)} images")

    lab = inv[inv.labelled]
    print("\nlabelled images per camera x class:")
    print(lab.groupby(["camera", "ripeness"]).size().unstack(fill_value=0))
    print("\nlabelled physical fruits per camera x class:")
    print(lab.groupby(["camera", "ripeness"]).fruit_id.nunique().unstack(fill_value=0))


if __name__ == "__main__":
    main()
