# Version 2 results: nothing beat v1

**Decision: v1 stays in production. V6 (shipping v2) does not happen.**

The analysis plan, hypotheses and decision rule were written into `PLAN.md` §12 (V1 row, 14 Oct 2026)
before any v2 result existed. The rule: a variant ships only if its fruit-level macro-F1 is at least
**0.566** (v1's 0.516 + 0.05) under the v1 protocol **and** a label-permutation test gives p < 0.05.
No variant reached the first condition, so no permutation test was run.

Protocol (same as v1): repeated StratifiedGroupKFold by fruit (5 folds x 5 repeats, seed 0),
40 labelled mangoes (80 images), metrics at fruit level (front + back scores averaged), the same
pixel-subset augmentation on training folds only. The v1 row was rerun under the same folds and
reproduces its logged score exactly.

| Model | Features | Aug | Fruit macro-F1 (sd) | Fruit acc | Mean abs. class error | Unripe recall |
|---|---|---|---|---|---|---|
| Majority class | - | no | 0.190 | 0.400 | - | - |
| **PLS-DA (v1, shipped)** | mean, SNV | yes | **0.516** (0.046) | **0.535** | **0.63** | 0.32 |
| PLS-DA | per-pixel SNV | yes | 0.485 (0.060) | 0.505 | 0.66 | 0.29 |
| PLS-DA (v1 features, reference for the next row) | mean, SNV | no | 0.490 (0.050) | 0.505 | - | - |
| PLS-DA | per-pixel SNV, rim dropped | no | 0.482 (0.032) | 0.500 | 0.67 | 0.33 |
| Firmness PLS -> class | mean, SNV | yes | 0.405 (0.061) | 0.415 | - | 0.41 |
| Firmness PLS -> class | per-pixel SNV | yes | 0.457 (0.056) | 0.460 | - | 0.39 |
| Ordinal PLS | mean, SNV | yes | 0.342 (0.077) | 0.360 | 0.77 | 0.38 |
| Ordinal PLS | per-pixel SNV | yes | 0.342 (0.039) | 0.375 | 0.76 | 0.33 |
| Cumulative logit on PLS scores | mean, SNV | yes | 0.368 (0.017) | 0.405 | 0.69 | 0.27 |
| Cumulative logit on PLS scores | per-pixel SNV | yes | 0.370 (0.028) | 0.405 | 0.69 | 0.25 |

sd = across the 5 CV repeats. The 95% interval for an accuracy on 40 fruits is about +-15 points,
so differences of a few points between rows are noise. The rim-dropped variant has no augmentation
(the stored pixel spectra have no coordinates), so its reference is the no-augmentation v1 row.
Data: `docs/results_v2.csv`; runs: MLflow experiments `v2_plsda`, `v2_firmness`, `v2_ordinal`.

## What we learned

- **(a) Per-pixel normalisation does not help.** Applying SNV to each pixel before averaging
  changes the features very little (within-fruit share of variance 0.147 -> 0.144, class
  separability 0.076 -> 0.078) and PLS-DA on them scores slightly below v1 (0.485 vs 0.516,
  within noise). Dropping the outer rim is no better (0.482 vs 0.490 without augmentation).
- **(b) Firmness is not predictable from these spectra.** Out-of-fold R2 is below zero (RMSE
  about 5,000, the size of the fruit-to-fruit spread); predictions sit in a flat band. So turning
  predicted firmness into classes loses to classifying directly. See `firmness_pred_vs_measured.png`.
- **(b) Ordinal models are worse, even on the ordinal metric.** v1 already calls 36 of 130
  unripe images *overripe*: the spectra do not place the classes on one unripe-to-overripe axis,
  and models that force one axis lose the information that separates them. Mean absolute class
  error rises from 0.63 (v1) to 0.69-0.77.
- **Band importance agrees with v1.** The firmness models (fitted on all 80 images, see
  `firmness_vip.png`) weight the same bands as the v1 PLS-DA (rank correlation of VIP 0.86-0.87):
  the chlorophyll red edge at 686-694 nm and the noisy 998 nm edge band; they put less weight on the
  950-960 nm water band. Caveat: with negative out-of-fold R2 and 9-10 components, these weights
  describe a model that overfits; read them as "the same spectral features", not as a firmness signal.

## Why v1 still holds

The limit is the data, not the model family: 40 fruits, spectrally close classes (separability
about 0.08) and a +-15-point interval. Better targets and better normalisation did not move the
score; more fruits (or another camera/fruit) are the likely next step, not more models.
