# fruit-ripeness-hsi

[![CI](https://github.com/pnastra/fruit-ripeness-hsi/actions/workflows/ci.yml/badge.svg)](https://github.com/pnastra/fruit-ripeness-hsi/actions/workflows/ci.yml)

Predict fruit ripeness from **hyperspectral images**, comparing a 1D-CNN honestly against a
classic chemometrics baseline (PLS-DA), served as a container on Google Cloud Run.

> Work in progress. See `PLAN.md` for the milestone plan.

## Data and license status

Data: DeepHS Fruit 2023 datasets, University of Tübingen Cognitive Systems Lab
(<https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/>).

**License: not stated** on the download page, in the dataset readme, or in the
[code repository](https://github.com/cogsys-tuebingen/deephs_fruit) (checked 29 Sep 2026).
I emailed the dataset authors on 1 Oct 2026 to ask under what terms the data may be used (status: awaiting reply). Until then this repo
contains no dataset images; only derived outputs (predictions, mean spectra, plots) are shown.

Citation: Varga, L. A., Makowski, J., Zell, A. (2021). *Measuring the Ripeness of Fruit with
Hyperspectral Imaging and Deep Learning.* IJCNN 2021.

The code in this repository is MIT licensed (see `LICENSE`).

## Run locally

```
uv sync
uv run pytest
```
