# Results (Mango, VIS camera, 3 classes, 40 labelled fruits)

Headline protocol: repeated stratified group k-fold **by fruit** (5 folds x 5 repeats). Fruit-level accuracy averages the front and back image of each fruit. The interval is a 95% Wilson interval for 40 fruits; one fruit is 2.5 points. Majority-class rate: 0.40.

| Model | Fruit acc. (95% CI) | Fruit macro-F1 | Image acc. | sd over repeats |
|---|---|---|---|---|
| Majority class | 0.400 (0.26-0.55) | 0.190 | 0.400 | 0.000 |
| PLS-DA (SNV) | 0.505 (0.36-0.65) | 0.490 | 0.508 | 0.043 |
| PLS-DA (Savitzky-Golay 1st derivative) | 0.515 (0.37-0.66) | 0.493 | 0.530 | 0.034 |
| PLS-DA (SNV) + augmentation (shipped) | 0.535 (0.38-0.68) | 0.516 | 0.520 | 0.041 |
| 1D-CNN (SNV) | 0.435 (0.29-0.59) | 0.405 | 0.422 | 0.041 |
| 1D-CNN (SNV) + augmentation | 0.425 (0.29-0.58) | 0.390 | 0.432 | 0.027 |
| 1D-CNN (SNV + SG 1st derivative) + augmentation | 0.460 (0.32-0.61) | 0.435 | 0.445 | 0.058 |
| PLS-DA (SNV), RANDOM split by image (leakage demo, not a result) | 0.630 (0.48-0.76) | 0.613 | 0.598 | 0.070 |

![comparison](results_comparison.png)

**Reading the table**
- PLS-DA beating the majority class is supported by the permutation test below, although the intervals overlap; every other difference in the table is within the noise.
- The 1D-CNN does **not** beat PLS-DA (small data favours chemometrics). Its best configuration was picked among only three, so that number is slightly optimistic.
- Augmentation (pixel-subset means) brings no measurable gain for either model.
- The random split inflates PLS-DA because a fruit's two images land on both sides; it is shown only to demonstrate leakage.
- A single grouped hold-out (8 test fruits) is too noisy to report.
- Label-permutation test (60 shuffles, shipped model, same protocol): null mean 0.350, 95th percentile 0.460, observed 0.535, p = 0.016. The signal is real but modest.
