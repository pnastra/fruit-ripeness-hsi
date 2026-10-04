# Band importance: PLS VIP vs CNN gradient x input

![band importance](band_importance.png)

Regenerate with `make explain` (numbers: `band_importance_stats.json`, curves: `band_importance.csv`).
Both curves are scaled to a maximum of 1 and are descriptive only: with 40 fruits and a weak class
signal they show where the models look, not which wavelengths truly carry ripeness information.

| | PLS VIP (shipped model) | CNN gradient x input (5 seeds) |
|---|---|---|
| Top-5 bands (nm) | 688.2, 690.9, 956.0, 995.4, 998.2 | 964.4, 975.6, 978.5, 981.3, 998.2 |
| Mean importance 680-700 nm | 0.70 | 0.40 |
| Mean importance 700-750 nm (red edge) | 0.60 | 0.40 |
| Mean importance 750-900 nm (NIR plateau) | 0.30 | 0.44 |
| Mean importance 900-1000 nm (water) | 0.50 | 0.58 |

Agreement: Spearman correlation between the two curves is -0.07; between the CNN's own seeds it is
only 0.24 on average, so the CNN attribution is unstable.

**Do the key wavelengths match known chemistry?** For PLS, partly and plausibly: VIP is highest in
the 680-750 nm region (chlorophyll absorption and the red edge, whose position follows chlorophyll
content, which falls as mangoes ripen) and near the 950-1000 nm water-absorption region, with a smaller
bump around 540-560 nm (green reflectance, pigment changes), and it is lowest on the NIR plateau
(750-900 nm), which mostly reflects fruit structure and geometry. The CNN attribution is nearly flat and
only agrees with PLS on the water band, so it gives little independent confirmation. The very highest
VIP value sits at the last band (998 nm), at the noisier edge of the sensor, and should not be
over-interpreted.
