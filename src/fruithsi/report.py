"""Results table and comparison chart from the MLflow runs -> docs/ (aggregate numbers only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fruithsi import CLASSES
from fruithsi.baseline import (
    MLFLOW_URI,
    PLSDA,
    PixelAugmented,
    merge_permutations,
    permutation_test,
)
from fruithsi.preprocess import load_pixels, load_spectra

DOCS = Path("docs")
SHIPPED = "plsda_snv_grouped_aug"

# (run name, label); the order is the order of the table
RUNS = [
    ("majority_grouped", "Majority class"),
    ("plsda_snv_grouped", "PLS-DA (SNV)"),
    ("plsda_sg1_grouped", "PLS-DA (Savitzky-Golay 1st derivative)"),
    (SHIPPED, "PLS-DA (SNV) + augmentation (shipped)"),
    ("cnn_snv_noaug", "1D-CNN (SNV)"),
    ("cnn_snv_aug", "1D-CNN (SNV) + augmentation"),
    ("cnn_snv_sg1_aug", "1D-CNN (SNV + SG 1st derivative) + augmentation"),
    ("plsda_snv_random", "PLS-DA (SNV), RANDOM split by image (leakage demo, not a result)"),
]


def load_results():
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(MLFLOW_URI)
    frames = [mlflow.search_runs(experiment_names=[e]) for e in ("baselines", "cnn")]
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.rename(
        columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run")
    )
    runs = runs.set_index("run")
    rows = []
    for name, label in RUNS:
        r = runs.loc[name]
        rows.append({
            "model": label, "run": name,
            "fruit_acc": r.cv_fruit_acc_mean, "fruit_acc_sd": r.cv_fruit_acc_std,
            "ci_low": r.cv_fruit_acc_ci_low, "ci_high": r.cv_fruit_acc_ci_high,
            "fruit_f1": r.cv_fruit_f1_mean, "image_acc": r.cv_image_acc_mean,
            "image_f1": r.cv_image_f1_mean, "holdout_fruit_acc": r.holdout_fruit_acc,
        })
    return pd.DataFrame(rows)


def comparison_chart(table, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1]
    colors = ["#e8a33d" if "random" in r else "#bbbbbb" if "majority" in r
              else "#3d9970" if "plsda" in r else "#4c78a8" for r in t.run]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(t.model.str.replace(" (leakage demo, not a result)", "", regex=False), t.fruit_acc,
            color=colors, xerr=[t.fruit_acc - t.ci_low, t.ci_high - t.fruit_acc], capsize=3)
    ax.axvline(0.4, color="k", ls=":", lw=0.8)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("fruit-level accuracy, grouped CV (mean of 5 repeats; whisker = 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def write_markdown(table, path: Path, perm: dict | None) -> None:
    def fmt(r):
        return f"{r.fruit_acc:.3f} ({r.ci_low:.2f}-{r.ci_high:.2f})"

    lines = [
        "# Results (Mango, VIS camera, 3 classes, 40 labelled fruits)", "",
        "Headline protocol: repeated stratified group k-fold **by fruit** (5 folds x 5 repeats). "
        "Fruit-level accuracy averages the front and back image of each fruit. The interval is a "
        "95% Wilson interval for 40 fruits; one fruit is 2.5 points. Majority-class rate: 0.40.",
        "",
        "| Model | Fruit acc. (95% CI) | Fruit macro-F1 | Image acc. | sd over repeats |",
        "|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.model} | {fmt(r)} | {r.fruit_f1:.3f} | {r.image_acc:.3f} | "
                     f"{r.fruit_acc_sd:.3f} |")
    lines += [
        "", "![comparison](results_comparison.png)", "", "**Reading the table**",
        "- PLS-DA beating the majority class is supported by the permutation test below, although "
        "the intervals overlap; every other difference in the table is within the noise.",
        "- The 1D-CNN does **not** beat PLS-DA (small data favours chemometrics). Its best "
        "configuration was picked among only three, so that number is slightly optimistic.",
        "- Augmentation (pixel-subset means) brings no measurable gain for either model.",
        "- The random split inflates PLS-DA because a fruit's two images land on both sides; "
        "it is shown only to demonstrate leakage.",
        "- A single grouped hold-out (8 test fruits) is too noisy to report.",
    ]
    if perm:
        lines += [
            f"- Label-permutation test ({perm['n_permutations']} shuffles, shipped model, same "
            f"protocol): null mean {perm['null_mean']:.3f}, "
            f"95th percentile {perm['null_q95']:.3f}, "
            f"observed {perm['observed_fruit_acc']:.3f}, p = {perm['p_value']:.3f}. "
            "The signal is real but modest."
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutation", type=int, default=0,
                        help="run the label-permutation test with this many shuffles (slow, "
                             "~9 s each); chunks with different --perm-seed are merged")
    parser.add_argument("--perm-seed", type=int, default=0)
    args = parser.parse_args()

    DOCS.mkdir(exist_ok=True)
    perm_path = DOCS / "permutation_test.json"
    if args.permutation:
        df, X, _ = load_spectra()
        y = df.ripeness.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        groups = df.fruit_id.to_numpy()
        pixels = load_pixels(df.hdr_path)

        def make():
            return PixelAugmented(lambda: PLSDA("snv"), pixels, n_draws=20, frac=0.05)

        result = permutation_test(make, X, y, groups, args.permutation, seed=args.perm_seed)
        if perm_path.exists():
            result = merge_permutations(json.loads(perm_path.read_text()), result)
        perm_path.write_text(json.dumps(result, indent=1))
    perm = json.loads(perm_path.read_text()) if perm_path.exists() else None

    table = load_results()
    table.to_csv(DOCS / "results.csv", index=False, float_format="%.4f")
    comparison_chart(table, DOCS / "results_comparison.png")
    write_markdown(table, DOCS / "results.md", perm)
    print(table[["model", "fruit_acc", "ci_low", "ci_high", "fruit_f1", "image_acc"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {DOCS}/results.md, results.csv, results_comparison.png"
          + ("" if perm else "  (no permutation test yet: run with --permutation 200)"))


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
