from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fruithsi.baseline import merge_permutations, summarise_permutations
from fruithsi.explain import cnn_attribution
from fruithsi.report import RUNS, comparison_chart, write_markdown
from fruithsi.train import CNNConfig

B = 40
rng = np.random.default_rng(0)
GROUPS = np.repeat(np.arange(30), 2)
Y = np.repeat(np.arange(30) % 3, 2)
_BASE = np.stack([np.sin(np.linspace(0, 5, B) * (c + 1)) for c in range(3)]) + 2
X = _BASE[Y] + rng.normal(0, 0.05, (len(Y), B))


def test_cnn_attribution_shape_and_scaling():
    cfg = replace(CNNConfig(), augment=False, max_epochs=5, patience=5, transform_name="snv")
    attr = cnn_attribution(X, Y, GROUPS, None, cfg, seeds=range(2))
    assert attr.shape == (2, B)
    assert (attr >= 0).all() and np.allclose(attr.max(axis=1), 1)


def fake_table():
    rows = [{"model": label, "run": name, "fruit_acc": 0.5, "fruit_acc_sd": 0.03, "ci_low": 0.35,
             "ci_high": 0.65, "fruit_f1": 0.48, "image_acc": 0.5, "image_f1": 0.47,
             "holdout_fruit_acc": 0.4} for name, label in RUNS]
    return pd.DataFrame(rows)


def test_markdown_table_and_permutation_line(tmp_path):
    perm = {"n_permutations": 200, "null_mean": 0.35, "null_q95": 0.46,
            "observed_fruit_acc": 0.535, "p_value": 0.02}
    path = tmp_path / "results.md"
    write_markdown(fake_table(), path, perm)
    text = path.read_text()
    assert text.count("\n| ") >= len(RUNS)  # one table row per run (+ header)
    assert "0.500 (0.35-0.65)" in text and "p = 0.020" in text
    assert "does **not** beat PLS-DA" in text


def test_comparison_chart_is_written(tmp_path):
    comparison_chart(fake_table(), tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000


def test_permutation_summary_and_merge():
    a = summarise_permutations(0.5, [0.3, 0.4, 0.45, 0.2], seeds=[0])
    assert a["p_value"] == pytest.approx(1 / 5)  # no shuffle reaches 0.5: (0 + 1) / (4 + 1)
    b = summarise_permutations(0.5, [0.55, 0.1], seeds=[1])
    merged = merge_permutations(a, b)
    assert merged["n_permutations"] == 6 and merged["seeds"] == [0, 1]
    assert merged["p_value"] == pytest.approx(2 / 7)  # one shuffle (0.55) reaches 0.5
    with pytest.raises(ValueError, match="seed"):
        merge_permutations(a, a)  # same seed twice would double-count the same shuffles
    with pytest.raises(ValueError, match="observed"):
        merge_permutations(a, summarise_permutations(0.6, [0.3], seeds=[2]))
