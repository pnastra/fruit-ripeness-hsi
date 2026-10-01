from pathlib import Path

import numpy as np
import pytest

from fruithsi.io import RAW_DIR, build_inventory, load_camera_info, parse_name, read_cube

HEADER = """ENVI
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bip
byte order = 0
"""

needs_data = pytest.mark.skipif(not (RAW_DIR / "Mango").exists(), reason="dataset not downloaded")


def test_parse_name():
    path = "Mango/VIS/day_10_m3/mango_day_10_m3_33_back.hdr"
    assert parse_name(path) == (10, 33, "back")
    with pytest.raises(ValueError):
        parse_name("Mango/VIS/other.hdr")


def test_read_cube_synthetic(tmp_path: Path):
    """Write a tiny bip float32 cube in the dataset's format and read it back."""
    lines, samples, bands = 4, 5, 7
    cube = np.random.default_rng(0).random((lines, samples, bands), dtype=np.float32)
    (tmp_path / "x.hdr").write_text(HEADER.format(lines=lines, samples=samples, bands=bands))
    cube.tofile(tmp_path / "x.bin")

    out = read_cube(tmp_path / "x.hdr")

    assert out.shape == (lines, samples, bands)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, cube)


@needs_data
def test_read_real_cube():
    cameras = load_camera_info()
    hdr = RAW_DIR / "Mango/VIS/day_10_m3/mango_day_10_m3_33_back.hdr"
    cube = read_cube(hdr)
    assert cube.shape == (64, 64, 224)
    assert cube.shape[-1] == len(cameras["VIS"]["wavelengths"])
    assert np.isfinite(cube).all()


@needs_data
def test_inventory():
    inv = build_inventory()
    assert len(inv) == 1124
    assert set(inv.camera) == {"VIS", "VIS_COR"}
    # same physical fruits are imaged every day, so fruit ids repeat across days
    assert inv.fruit_id.nunique() == 40
    lab = inv[inv.labelled & (inv.camera == "VIS")]
    assert lab.fruit_id.nunique() == 40
    assert len(lab) == 80  # front + back
    assert lab.groupby("fruit_id").ripeness.nunique().eq(1).all()
    assert set(lab.ripeness) == {"unripe", "perfect", "overripe"}
