# PLAN — fruit-ripeness-hsi

A 2-week portfolio project: predict fruit ripeness from **hyperspectral images**, compared honestly against a classic chemometrics baseline, packaged in Docker and deployed as a container on **Google Cloud Run**, with CI/CD from GitHub Actions.

**Audience:** hiring managers for agritech / food-tech / computer vision IC roles.
**What it should prove:** spectral-imaging ML (my PhD area) with sound evaluation (grouped split, a baseline to beat), plus production-style MLOps: a container on a real cloud, automated tests and deploys, and basic prediction monitoring.
**Builds on project 1** (`crop-yield-forecast`): same conventions (`uv`, `src/`, Makefile, MLflow, tests, teaching mode). New here: imaging data, a neural network, cloud deployment, CI/CD.

**Timebox:** about 45–50 hours over 2 weeks (target ship date: 18 Oct 2026). Week 1 = data and models, week 2 = ship. No new scope in week 2.

---

## 1. Working rules for Claude Code

- Read this file at the start of every session, then read §12 (Progress log) to see where we stopped.
- Work on **one milestone per session**. Each milestone is sized for one sitting (2–5 h). Tick its checkboxes when done.
- **"Done when"** under each milestone is the exit test. Don't call a milestone done until it passes.
- **Stay in scope.** Anything in "Out of scope" needs my explicit OK.
- Ask before adding a dependency that isn't listed in §3.
- **Never commit** secrets (GCP keys, `.env`), raw data (`data/raw/`), MLflow runs, model checkpoints larger than 20 MB, or `.venv/`.
- **Reported metrics come only from the grouped split** (by fruit ID, §4). The random split exists only to show leakage.
- Keep code in `src/`, not in notebooks. Notebooks are for EDA and narrative only.
- Before calling a milestone done, run `uv run ruff check .` and `uv run pytest`.
- **Data is big.** Never download more than the one fruit chosen in M1 without asking me.
- At the end of each session, add 2–4 lines to §12 and commit.

### Teaching mode (always on)
I'm learning PyTorch training loops, Google Cloud Run, Artifact Registry and GitHub Actions in this project, so explain as you go:
- **Before each command or file change:** 1–2 lines on *what* it does and *why*.
- **Go deeper on:** PyTorch `Dataset`/`DataLoader` and the training loop, `gcloud` commands, IAM/service accounts, GitHub Actions YAML and secrets, Workload Identity Federation. Keep it brief for NumPy, scikit-learn, PLS/chemometrics and pandas (I know these well).
- **Pause and ask before:** anything that creates or changes cloud resources or billing, any `git push`, creating secrets, installing anything outside the project, or downloading data over 5 GB.
- **At the end of each milestone:** a 3–5 bullet recap plus the commands I'd run to reproduce it.
- If something fails, explain what the error means before fixing it.

---

## 2. Environment

| Item | Status |
|---|---|
| Mac, Git, VS Code, Claude Code, `uv`, Docker Desktop | ✅ (from project 1) |
| GitHub account | ✅ |
| Free disk space for the chosen fruit (zip + extracted, about 2–3× the zip size) | ⬜ M1 |
| Google Cloud account, billing enabled, **budget alert at $1** | ✅ project `fruit-ripeness-hsi`, budget 1 THB (alerts only) |
| `gcloud` CLI | ✅ 588.0.0 at `/usr/local/share/google-cloud-sdk/bin` (not on the non-interactive PATH) |
| Kaggle/Colab GPU | Only if CPU/MPS training is too slow (M6) |

**Python:** 3.12 (safest for PyTorch). Use `uv python pin 3.12`.

---

## 3. Dependencies (managed by `uv`)

- **Core:** numpy, pandas, scikit-learn, scipy, pyarrow, spectral (ENVI reader), torch, mlflow, matplotlib
- **Serving:** fastapi, uvicorn, pydantic
- **Dev:** pytest, ruff, jupyter / ipykernel, httpx
- **Stretch only:** captum, streamlit

The Docker image installs **serving deps only** plus CPU-only PyTorch, to keep it small for Cloud Run.

---

## 4. Data

- **Source:** DeepHS Fruit 2023 datasets, University of Tübingen Cognitive Systems Lab: https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/ . Code and paper: https://github.com/cogsys-tuebingen/deephs_fruit (Varga, Makowski & Zell, IJCNN 2021).
- **Zip sizes (checked 29 Sep 2026):** Avocado 72 GB, Kiwi 44 GB, Papaya 22 GB, Mango 2.7 GB, Kaki 2.2 GB. Annotations 162 KB (use `annotations-upd-2024-01-09.zip`). A torrent of everything is also offered.
- **Layout:** the readme says to extract each zip into its own folder: one folder per fruit with `VIS`, `VIS_COR` and `NIR` subfolders, plus `annotations/`.
- **Cameras:** Specim FX10, INNO-SPEC Redeye 1.7, Corning microHSI 410. **Use one camera only** so all samples share the same bands.
- **Labels:** destructive measurements (firmness, sugar content) and a ripeness class. Confirm in M3.
- **Start with Mango or Kaki** (small downloads). Kiwi/Avocado only with my OK.
- **Split:** grouped by fruit ID (each physical fruit is in train *or* test, never both), because the same fruit is imaged several times.
- **License: not stated** on the download page or in the readme (checked 29 Sep 2026). Resolve in M0. Until confirmed, don't commit or redistribute images; the app shows only derived outputs (predictions, mean spectra, plots). Cite the paper.

---

## 5. Repo structure

```
fruit-ripeness-hsi/
├── PLAN.md
├── CLAUDE.md               # one line: @PLAN.md
├── README.md               # demo link, results, model card, credits
├── LICENSE                 # MIT (code only)
├── .gitignore
├── pyproject.toml / uv.lock
├── Makefile                # data, features, train-baseline, train-cnn, serve, docker-build, docker-run, deploy
├── Dockerfile
├── .github/workflows/
│   ├── ci.yml              # ruff + pytest on every push/PR
│   └── deploy.yml          # build → Artifact Registry → Cloud Run on push to main
├── data/raw/               # gitignored
├── data/processed/         # gitignored
├── notebooks/01_eda.ipynb
├── src/fruithsi/
│   ├── io.py               # read ENVI cubes + annotations
│   ├── preprocess.py       # mask the fruit, mean spectrum, SNV / Savitzky–Golay
│   ├── split.py            # grouped split by fruit ID
│   ├── baseline.py         # majority class, PLS-DA
│   ├── dataset.py          # PyTorch Dataset
│   ├── model.py            # small 1D-CNN
│   ├── train.py            # training loop + MLflow
│   ├── evaluate.py         # metrics, confusion matrix
│   └── explain.py          # band importance
├── api/main.py             # FastAPI: /health, /predict, prediction logging
├── samples/                # a few derived mean spectra for trying /predict (not raw images)
├── models/model.pt         # small, committed if < 20 MB
└── tests/
```

---

## 6. Milestones

Thirteen small milestones (M0–M12): about 23 h in week 1 and 22 h in week 2. Hours are targets; if one runs 50% over, stop and note it in §12.

### Week 1 — Data and models

#### M0 — Repo skeleton and license check (≈2.5 h)
- [x] Create the public GitHub repo `fruit-ripeness-hsi` (MIT); `uv init`, `uv python pin 3.12`, deps from §3, commit `uv.lock`
- [x] `.gitignore` (§9), `CLAUDE.md`, `Makefile` stubs, `src/fruithsi/` package, one trivial passing test
- [x] License: check the annotations zip, dataset readme and GitHub repo. If none, draft a short email to the authors (I send it) and note the status in the README.

**Done when:** `uv run pytest` passes, the first commit is pushed, and the README states the license status.

#### M1 — Download one fruit (≈2 h, mostly waiting)
- [x] Check free disk space (need 2–3× the zip size)
- [x] `make data`: annotations + **Mango or Kaki**, resume-capable download (`curl -C -`), extract to `data/raw/<fruit>/`
- [x] Record the folder tree (depth 3) and file counts in §12

**Done when:** the data is extracted and the tree is in §12.

#### M2 — Load, inventory, go/no-go (≈3 h)
- [x] `io.py`: read one ENVI cube; print shape, wavelengths, camera
- [x] Inventory table (fruit ID, camera, day, path, labels) → `data/processed/inventory.parquet`
- [x] Count labelled fruits per ripeness class per camera; pick one camera
- [x] **Go:** at least ~30 labelled fruits and at least 2 classes. **No-go:** try the other small fruit once; if it still fails, stop and ask me.

**Done when:** the decision (fruit, camera, task) is in §12, and a test checks the loader on one cube.

#### M3 — Segmentation and mean spectra (≈4 h)
- [x] Simple fruit/background mask (threshold on intensity or a band ratio); save one mask example for the README
- [x] `preprocess.py`: one mean spectrum per image → `data/processed/spectra.parquet`; drop noisy edge bands
- [x] SNV and Savitzky–Golay as options

**Done when:** `spectra.parquet` exists, and a test checks the band count and that there are no NaNs.

#### M4 — Quick EDA (≈3 h, hard timebox)
- [x] Label distribution; images per physical fruit
- [x] Mean spectra by class (raw vs SNV vs SG derivative) in one figure
- [x] Write 3–5 findings for the README and "Decisions for M5" at the end of the notebook

**Done when:** the decisions are recorded and the notebook runs top to bottom.

#### M5 — Split, baselines, leakage demo (≈4 h)
- [x] `split.py`: grouped hold-out by fruit ID (plus `GroupKFold` for CV)
- [x] `baseline.py`: majority class and **PLS-DA**; metrics accuracy, macro-F1, confusion matrix; MLflow `sqlite:///mlflow.db`
- [x] Leakage demo: PLS-DA on a random split by image, logged as its own run
- [x] Test: no fruit ID in both train and test

**Done when:** three runs are in MLflow (majority, PLS-DA grouped, PLS-DA random).

#### M6 — 1D-CNN (≈5 h)
- [x] `dataset.py`, `model.py` (small 1D-CNN), `train.py` (loop, early stopping, fixed seeds, CPU/MPS)
- [x] Train on the grouped split; try **at most 3 configurations** (no grid search); log to MLflow

**Done when:** the loss curve is sensible and the best CNN run is in MLflow.

### Week 2 — Ship

#### M7 — Compare, save, explain (≈4 h)
- [x] Results table: majority vs PLS-DA vs 1D-CNN (grouped) + PLS-DA random split
- [x] Save the shipped model + preprocessing parameters (bands, SNV settings, class names); test that it loads and predicts (**adapted: PLS-DA is shipped as `models/plsda_snv.json`, plain NumPy, no torch; see §12**)
- [x] One band-importance plot: PLS VIP scores and CNN gradient × input on the same axes; 2–3 sentences on whether the key wavelengths match known chemistry (e.g. chlorophyll ~680 nm, water bands)

**Done when:** the table and plot are saved for the README.

#### M8 — API (≈3 h)
- [x] `api/main.py`: `GET /health`, `POST /predict` (validated spectrum length → class probabilities)
- [x] Each prediction logged as one JSON line to stdout (timestamp, model version, class, confidence)
- [x] Put 3–5 derived sample spectra in `samples/`; `TestClient` tests, including wrong length → 422

**Done when:** `make serve` works locally and the API tests pass.

#### M9 — Docker (≈3 h)
- [x] `Dockerfile`: `python:3.12-slim`, serving deps only (**no torch needed: PLS-DA is shipped as numpy**); image under ~1 GB; build with `--platform linux/amd64`
- [x] `make docker-build && make docker-run`; call `/predict` with a sample

**Done when:** the container answers `/predict` locally and the image size is recorded.

#### M10 — Google Cloud setup and first deploy (≈4 h) — **ask me before each step**
- [x] GCP project, billing, **$1 budget alert**; install `gcloud`; enable Cloud Run and Artifact Registry
- [x] Artifact Registry repo in `asia-southeast1`; push the image
- [x] `gcloud run deploy` (min 0, max 1, 1 GiB, unauthenticated); check `/health` and `/docs` from a cold start; find a prediction line in Cloud Logging

**Done when:** the public URL works and is written in §12.

#### M11 — CI/CD (≈5 h) — **ask me before creating anything**
- [x] `ci.yml`: ruff + pytest on every push and PR; CI badge in the README
- [x] `deploy.yml`: on push to `main`, build → Artifact Registry → Cloud Run, authenticated with **Workload Identity Federation** (explain each IAM step)
- [x] Push a small visible change and confirm it redeploys

**Done when:** a push to `main` runs CI and updates the live service with no manual steps.

#### M12 — README and model card (≈4 h)
- [x] Top: live `/docs` link, screenshot, one-sentence summary, CI badge
- [x] Findings (M4), results table and band-importance plot (M7)
- [x] Architecture diagram: GitHub → Actions → Artifact Registry → Cloud Run → Cloud Logging
- [x] Model card: intended use, limits (one fruit, one camera, lab conditions, small sample, not for commercial grading), data source, license status
- [x] Credits, citation, how to run locally

**Done when:** a stranger could understand and rerun the project from the README alone.

### Stretch (only after M12, or as version 2)
- [ ] Streamlit demo UI calling the Cloud Run API
- [ ] Random forest baseline and a small hyperparameter grid
- [ ] A second fruit or camera, to test transfer
- [ ] Firmness regression head
- [ ] Captum attributions

---

## 7. Out of scope

- Downloading Avocado or Kiwi unless M2's go/no-go fails and I approve
- Mixing cameras in one model
- Training on full-resolution cubes end to end (use mean spectra; patches only as a stretch)
- LLM/RAG layers
- Kubernetes, Terraform, or any paid GCP service beyond the free tier
- Model registries beyond MLflow's local store
- Hyperparameter search (at most 3 CNN configurations)
- A demo UI (the FastAPI `/docs` page is the demo; Streamlit is a stretch goal)

---

## 8. Definition of done

- [x] Public GitHub repo with README, MIT license (code), passing CI badge
- [x] A live Cloud Run URL
- [x] The CNN is compared honestly with PLS-DA; if it doesn't win, the README says so and explains why
- [x] Every push to `main` redeploys automatically
- [x] No secrets, raw images or large checkpoints in the git history
- [x] The dataset license status is stated in the README
- [x] The GCP budget alert is in place, and the service scales to zero

---

## 9. `.gitignore`

```
# Python / uv
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# Data
data/raw/
data/processed/
*.zip
*.hdr
*.bin
*.raw

# MLflow
mlruns/
mlflow.db
mlartifacts/

# Models (commit only the small final model, explicitly)
*.ckpt
checkpoints/

# Secrets
.env
*.json.key
gcp-*.json
*.token

# macOS / editors
.DS_Store
.vscode/
```

---

## 10. Google Cloud Run for a first-time user (M10–M11)

1. Create a Google Cloud account and project; enable billing (the free tier still needs a card). **Set a budget alert at $1 immediately** (Billing → Budgets & alerts).
2. Install `gcloud`, run `gcloud auth login` and `gcloud config set project <project-id>`.
3. Enable APIs: `run.googleapis.com`, `artifactregistry.googleapis.com`.
4. Create an Artifact Registry Docker repo in `asia-southeast1`; `gcloud auth configure-docker asia-southeast1-docker.pkg.dev`.
5. **Mac ARM note:** build for Cloud Run's platform with `docker build --platform linux/amd64 ...`.
6. Push the image, then `gcloud run deploy fruit-ripeness --image <image> --region asia-southeast1 --allow-unauthenticated --min-instances 0 --max-instances 1 --memory 1Gi`.
7. For GitHub Actions, use Workload Identity Federation (`google-github-actions/auth`) instead of a JSON key.
8. Delete old images in Artifact Registry now and then; storage above the free allowance costs money.

---

## 11. Risks

| Risk | Fallback |
|---|---|
| The small fruit has too few labels for one camera | Try the other small fruit; then Kiwi (44 GB) with my OK; then EuroSAT |
| The dataset license forbids reuse | Keep the repo, show only aggregate plots and metrics, and link to the dataset instead of hosting samples |
| The CNN doesn't beat PLS-DA | Report it honestly — small data favours chemometrics, which is a credible finding |
| Download is slow or breaks | Resume with `curl -C -`, or use the torrent |
| PyTorch image too large for Cloud Run | CPU-only wheel, slim base, or export to ONNX Runtime |
| Cloud Run costs | Min instances 0, max 1, budget alert, delete old images |
| Week 1 overruns | Cut the leakage demo to one sentence and use the best of the first CNN runs; don't push the Week 2 start |
| CD (M11) gets stuck on IAM | Ship with CI only and a manual `make deploy`; add CD in version 2 |

---

## 12. Progress log

Claude Code adds 2–4 lines at the end of every session: date, milestone, what was done, decisions, what's next.

| Date | Milestone | Notes |
|---|---|---|
| 2026-09-29 | M0 (local part) | Skeleton, uv deps (serving in main deps; `train`/`dev` groups), lint+test pass, first commit local. License: none stated anywhere; email drafted, not sent. |
| 2026-09-29 | M0 done | Public repo https://github.com/pnastra/fruit-ripeness-hsi pushed. License email still to be sent by me. |
| 2026-09-29 | M1 done | Chose **Mango** (2.7 GB zip, 4.1 GB extracted; 167 GB free). `make data` = `scripts/download_data.sh` (resumable, idempotent). Tree (depth 3) below. Next: M2 (io.py, inventory, go/no-go). |
| 2026-10-01 | M2 done | **Go. Decision: Mango, camera VIS (Specim FX10, 397.7-1003.8 nm, 224 bands, 64x64 px crops), task = 3-class ripeness (unripe / perfect / overripe).** README license status updated (email sent 1 Oct). |
| | | Why VIS: labels identical for both cameras; VIS has the wider range (chlorophyll ~680 nm and water ~970 nm). VIS_COR = Corning microHSI 410, 408-901 nm, 249 bands. |
| 2026-10-01 | M3 done | `preprocess.py` (mask, mean spectrum, SNV, Savitzky-Golay via `transform()`), `make features` -> `data/processed/spectra.parquet` (562 VIS images x 192 bands, 476.5-998.2 nm). Local notebook `notebooks/local/inspect_samples.ipynb` (gitignored) for browsing samples. |
| | | Mask = 850 nm band > 0.10, largest blob, fill holes, erode 3 px: 2245-2682 px per image. **Not Otsu** (histogram is a continuum, Otsu cut the fruit). Bands outside 475-1000 nm dropped (SNR < 10 below ~470 nm). Spectra file stores raw means only; SNV/SG are applied on demand. Mask figure saved to `reports/figures/` (gitignored until the license reply). Next: M4 (EDA, hard timebox). |
| 2026-10-01 | M4 done | `notebooks/01_eda.ipynb` (aggregate plots only, runs top to bottom): label counts, mean spectra raw/SNV/SG1, Fisher ratio, PCA, front-vs-back distance, chlorophyll index; Findings and "Decisions for M5" are at the end of the notebook. Local-only: `notebooks/local/spatial_heterogeneity.ipynb` (index maps, waits for license). |
| | | Headline: classes are spectrally close (separability 0.06-0.08, PCA overlaps), so expect modest accuracy; majority baseline = 40% of fruits. M5 plan: repeated GroupKFold by fruit_id as headline + single hold-out, PLS-DA on SNV (SG1 as the one alternative). M6 needs my OK for pixel-subset augmentation. |
| 2026-10-01 | M4 pushed; M6 decision | M4 pushed. **Approved for M6: pixel-subset-mean augmentation** (train folds only; test fruits always get the plain whole-fruit mean; give PLS-DA the same augmentation or report both so the comparison is level). Needs pixel spectra of the 80 labelled VIS images saved to `data/processed/` (gitignored, ~150 MB). Next: M5 in a fresh session. |
| 2026-10-02 | M5 done | `split.py`, `evaluate.py`, `baseline.py`, `make train-baseline` -> 4 MLflow runs in experiment `baselines` (`sqlite:///mlflow.db`, rerun replaces them). Protocol: repeated StratifiedGroupKFold by fruit (5 folds x 5 repeats) is the headline; one grouped hold-out (8 test fruits) shown next to it. PLS-DA = SNV default, SG1 the one alternative, components by inner grouped CV. 22 tests pass. |
| | | **Results (fruit-level acc, mean over 5 repeats, 95% Wilson CI, n=40):** majority 0.40 (26-55%); **PLS-DA SNV 0.505 (36-65%), macro-F1 0.49**; PLS-DA SG1 0.515; PLS-DA SNV with RANDOM split (leakage) 0.63 (48-76%). Single grouped hold-out is useless as evidence (8 fruits): PLS-DA 0.375 fruit acc, image acc 0.25. Label-permutation test (200 shuffles, same protocol): null mean 0.347, q95 0.46, p = 0.015, so the signal is real but modest and the grouped protocol does not leak. |
| | | Confusion (grouped CV, per repeat of 80 images): recall unripe 0.33, perfect 0.61, overripe 0.56; unripe is mostly called perfect. Inner CV picks 8-9 PLS components (of max 10) for ~64 training images: likely overfitting. PLS-DA does not yet get the pixel-subset augmentation; do that in M6 so both models are compared level. |
| 2026-10-02 | M6 done | `dataset.py`, `model.py` (1,411 params), `train.py`, `make train-cnn` -> 3 runs in MLflow experiment `cnn`; `make features` now also writes `data/processed/pixels.npz` (145 MB, gitignored; per-pixel spectra of the 80 labelled images). `make train-baseline` adds `plsda_snv_grouped_aug` so PLS-DA gets the same augmentation. 35 tests pass. |
| | | **Results (grouped CV, fruit-level acc, mean over 5 repeats):** majority 0.400; PLS-DA SNV 0.505; **PLS-DA SNV + aug 0.535 (F1 0.52)**; PLS-DA SG1 0.515; CNN snv_aug 0.425; CNN snv_noaug 0.435; **CNN snv_sg1_aug 0.460 (F1 0.435), best CNN run**. Repeat-to-repeat sd is 0.03-0.06 and the 95% CI for n=40 is about +-15 points, so only PLS-DA > majority is a real gap; CNN vs majority and aug vs no-aug are not distinguishable. **The CNN does not beat PLS-DA**; README must say so (small data favours chemometrics). Best CNN chosen by CV macro-F1 among only 3 configs: slightly optimistic. |
| | | Things found while getting the CNN to learn (all fixed using TRAINING data only, never test scores): (1) global average pooling hid *where* in the spectrum a feature is -> now pool to 8 segments + flatten; (2) raw SNV inputs share one dominant shape, so even a 30k-param net could not fit 50 spectra -> per-band standardisation with training-fold statistics; (3) validation = ~7 fruits is very noisy, early stopping hit epoch 0 -> EMA-smoothed val loss (0.3), dropout 0.5, wd 0.05, label smoothing 0.1 (diagnosed on 6 inner splits of one outer training set); a few folds still stop at epoch 0. |
| | | **Augmentation finding:** random pixel subsets of one fruit barely differ (frac 0.4 gave noise = 2.3% of the fruit-to-fruit spread), so frac was cut to 0.05 (~124 px, 8%) for both CNN and PLS-DA. Even so it brings no measurable gain. A stronger variant (random contiguous patches, which would capture the centre-to-rim gradient and skin patches) changes the approved method: ask before trying. Open question for M7: which model to ship (PLS-DA aug is best but needs no torch; the plan's API/Docker assume a torch model). |
| 2026-10-04 | M7 done | **Decision: ship PLS-DA (SNV + augmentation)**, chosen beforehand by best CV macro-F1 (0.516); the CNN is reported as a comparison only. `make model` -> `models/plsda_snv.json` (21 KB: 192 bands, 8 PLS components, affine map, temperature, VIP, CV metrics, calibration); `src/fruithsi/predict.py` is numpy-only (SNV, one matrix product, temperature softmax) so the API/Docker need no torch, scipy or sklearn. The export asserts the numpy predictor reproduces sklearn exactly (tested). `make report` -> `docs/results.md`, `results.csv`, `results_comparison.png`; `make explain` -> `docs/band_importance.png/.csv/.md`. 48 tests pass. |
| | | **Probabilities:** PLS-DA gives scores, so the API probabilities are softmax(scores / T) with T = 0.50 fitted on grouped out-of-fold scores: OOF log loss 0.939 vs 1.099 for chance; mean confidence 0.54 vs OOF image accuracy 0.52 (well calibrated, but expect confidences of only ~0.4-0.6). Final model is trained on all 80 labelled images (40 fruits). **Permutation test for the shipped model:** 60 shuffles (2 chunks, seeds 0 and 1), null mean 0.350, q95 0.460, max 0.49 vs observed 0.535: p = 0.016 (floor for 60 shuffles). |
| | | **Band importance (honest):** PLS VIP is highest at 680-750 nm (chlorophyll absorption and red edge), near 950-1000 nm (water) and a bump at 540-560 nm, lowest on the NIR plateau: chemically plausible. The CNN gradient x input is nearly flat and unstable (seed-to-seed Spearman 0.24) and uncorrelated with VIP (-0.07); it agrees only on the water band. The top VIP value is at the noisy 998 nm edge band. |
| | | **Open items / notes:** (1) `pyproject.toml` still lists torch and scipy as main dependencies; move them to the `train` group in M9 when the Dockerfile fixes the serving deps (main = numpy, fastapi, uvicorn, pydantic). (2) Environment quirk: background processes only make progress while a tool call is active (a 230 CPU-second job took 27 min of wall time); run long jobs with short polling calls, or in chunks (the permutation test is chunkable via `--perm-seed`). (3) The README must state: PLS-DA ships, the CNN lost, intervals are about +-15 points, grouped CV by fruit, license email pending. |
| 2026-10-04 | M8 done | `api/main.py` (FastAPI): `GET /health`, `POST /predict` (`{"spectrum": [192 floats]}` -> predicted_class, confidence, probabilities, model_version), `/` redirects to `/docs` (pre-filled with a sample spectrum). One JSON log line per prediction on stdout (severity, message, timestamp, model_version, predicted_class, confidence; the input is never logged). `samples/` = 5 derived mean spectra of TRAINING images (`scripts/make_samples.py`; not for evaluation). `make serve` verified with curl: 200 / 422 / log line. 63 tests pass. |
| | | Bug found by a test and fixed: FastAPI's default 422 response echoes the offending input, so a body containing `NaN` made the error response un-encodable and returned a 500; a custom handler now returns only loc/msg/type. Sample confidence is low (e.g. 0.38 for a training-set "perfect" fruit), as expected from the weak signal. |
| 2026-10-04 | M9 done | `Dockerfile` (python:3.12-slim, uv 0.10.7, `uv sync --frozen --no-default-groups`, non-root user, `$PORT`, HEALTHCHECK) + `.dockerignore` (keeps `data/`, `mlruns/`, notebooks out of the build context). `make docker-build` (`--platform linux/amd64`, emulated on this ARM Mac, 13 s) and `make docker-run`. **Image size: 109 MB** (limit was ~1 GB): only fastapi, uvicorn, pydantic, starlette, numpy. Tested in the container: /health 200, /predict on a sample, wrong length and NaN -> 422, one JSON log line in `docker logs`, runs as uid 1000, healthy, honours `PORT=9090`. `pyproject.toml`: torch and scipy moved to the `train` group (main deps = numpy, fastapi, uvicorn, pydantic); local env unchanged (train is a default group). Docker Desktop had to be started with `open -a Docker`. |
| | | **Stopped here on purpose: M10 needs the user** (GCP account, billing, $1 budget alert, interactive `gcloud auth login`; the plan says to ask before each step). Next: M10, then M11 (secrets/IAM, also ask first), then M12 README (live URL and CI badge come from M10/M11). |
| 2026-10-07 | M10 (deployed) | **Live: https://fruit-ripeness-581425276724.asia-southeast1.run.app** (`/docs` is the demo; also reachable as fruit-ripeness-64afcstziq-as.a.run.app). User did billing, 1 THB budget alert, gcloud install, API enabling. I (one approval per step): Artifact Registry Docker repo `fruit-ripeness` in asia-southeast1; `gcloud auth configure-docker asia-southeast1-docker.pkg.dev` (credHelper line in ~/.docker/config.json, backup kept); pushed `.../fruit-ripeness/api:bb819d9` and `:v1` (same digest, ~105 MB); `gcloud run deploy fruit-ripeness` (public via allUsers run.invoker, min 0 / max 1, 1 GiB, 1 CPU) -> revision fruit-ripeness-00001-v7v. |
| | | Verified live: /health 200 (0.58 s first request, 0.10 s warm), /docs 200, / -> 307 /docs, /predict 200 in 0.11 s, wrong length 422; the prediction appears in Cloud Logging as a structured INFO entry (jsonPayload: predicted_class, confidence, model_version). `make deploy` added (refuses with uncommitted changes; tags the image with the git SHA). **Open:** true cold-start check of /health and /docs after scale-to-zero (do at the start of the next exchange; the harness blocks long sleeps). Runs as the default compute service account (broad Editor rights): use a dedicated minimal service account in M11. |
| 2026-10-08 | M10 done | Cold start measured after a day idle: first /health 3.5 s (3.15 s server-side; a new uvicorn startup is in the logs just before it), then ~0.1 s warm; /docs 200. |
| 2026-10-08 | M11 (setup) | Approved step by step. GCP: enabled `iamcredentials`, `sts` (+ `iam`, needed to create service accounts and pools). Runtime SA `fruit-ripeness-run` (no roles), service switched to it (revision 00002-wdl; logs still flow). Deploy SA `github-deployer`: no project roles, only `artifactregistry.writer` on repo fruit-ripeness, `run.developer` on service fruit-ripeness, `iam.serviceAccountUser` on fruit-ripeness-run. WIF pool `github` + OIDC provider `github-actions` (issuer token.actions.githubusercontent.com, condition `assertion.repository == 'pnastra/fruit-ripeness-hsi'`), `workloadIdentityUser` for that repo's principalSet on github-deployer (first try PERMISSION_DENIED from pool propagation delay; retry worked). GitHub repo variables WIF_PROVIDER, DEPLOY_SA, RUN_SA (not secrets; no key file exists). |
| | | Code: `.github/workflows/ci.yml` (push + PR: uv sync --frozen, ruff, pytest) and `deploy.yml` (workflow_run after CI succeeds on main: auth via WIF, build + push image tagged with the commit SHA, `gcloud run deploy` as fruit-ripeness-run, smoke test /health and /predict). torch now comes from the CPU-only index on Linux (`[tool.uv.sources]`), removing 15 CUDA packages + triton from CI installs; macOS unchanged. |
| 2026-10-08 | M11 done | First push: CI failed at setup (`astral-sh/setup-uv@v10` does not exist: that action publishes no moving major tag) -> pinned `v10.2.0`; then CI green and the first automatic Deploy succeeded (1 min 14 s: WIF auth, build, push, deploy, smoke test); the Deploy run for the failed CI was correctly skipped. **Exit test passed:** pushing 27832d3 (CI badge in README + API version 0.1.0 -> 0.1.1) ran CI then Deploy with no manual steps; live `/openapi.json` went 0.1.0 -> 0.1.1 (revision 00004-bwf, image tagged with the full commit SHA, running as fruit-ripeness-run). |
| | | **Open (ask first):** Artifact Registry holds 231 MB after 3 images (~60 MB added per push; free tier 0.5 GB) -> add a cleanup policy keeping the last ~3 images. Optional: pin `runs-on: ubuntu-24.04` (ubuntu-latest moves to Ubuntu 26 from 19 Oct); docs-only pushes also redeploy (harmless, but each adds an image). Next: M12 (README and model card). |
| 2026-10-08 | M12 done | README rewritten for a hiring-manager reader: one-paragraph summary at the top with the live `/docs` link, CI badge and a screenshot of the live docs page (`docs/api_docs.png`, headless Chrome, shows v0.1.1); results table + chart, 5 EDA findings, band-importance plot and chemistry reading, method table, Mermaid architecture diagram (GitHub -> Actions CI -> Deploy via WIF -> Artifact Registry -> Cloud Run -> Cloud Logging), model card (intended use, not-for, limits, performance, confidence, license), API usage, full rerun commands, repository layout, data/license/citation (official BibTeX from the authors' repo; the 2021 paper covers avocado/kiwi, the mango data is the 2023 release). |
| | | Definition of done checked: CI badge green, live URL, CNN-vs-PLS-DA comparison stated honestly, push to main redeploys, budget alert + scale to zero. **History audit:** largest blob ever committed is uv.lock (720 KB); no dataset files (only `data/raw/.gitkeep`); 0 secret-like strings; the only images are aggregate plots and the API screenshot. Still open (ask first): Artifact Registry cleanup policy, optional `ubuntu-24.04` pin; license reply from the dataset authors (update the README when it arrives; then the mask figure in `reports/figures/` could be published). |
| 2026-10-08 | Housekeeping | Artifact Registry cleanup policy (approved): keep the 3 most recent versions, delete the rest (runs in the background ~daily). CI builds are one registry entry per image; local builds now use `--provenance=false` too, so the policy never splits an image into index/child parts (the first manual image bb819d9 had 3 entries and will be the first to go; the live image is always the newest, so it is always kept; rollback is limited to the last 3 images). CI and Deploy pinned to `ubuntu-24.04`. Dataset license: still no reply from the authors (README already says "awaiting reply"). |

M1 data tree (`data/raw/`, gitignored). Mango has **VIS and VIS_COR only, no NIR** (M2 correction: VIS_COR is NOT a corrected variant, it is a different camera, see M2 row):
```
annotations/            8 json files (train_all, train_only_labeled, val, test; each with a _v2)
Mango/VIS/              1124 files = 562 cubes (.bin + .hdr), 11 folders day_{1,2,3,4,5,7,8,9,10,11,12}_m3
Mango/VIS_COR/          1124 files = 562 cubes, same 11 day folders
_zips/                  Mango.zip, annotations-upd-2024-01-09.zip
```
File names look like `mango_day_10_m3_33_back.hdr` (day, fruit number, front/back side).

M2 findings (matter for M3-M5):
- **Fruit ID = fruit number alone (1-40).** The same 40 mangoes are imaged every day (n per day: 40,38,37,35,32,29,24,19,13,9,5) until each is measured destructively. `(day, number)` would leak across days.
- **Only 40 labelled fruits** = 80 images (front + back), each labelled once on its last imaging day: 16 perfect / 13 unripe / 11 overripe. The other 482 VIS images (earlier days) are unlabelled. Go threshold (>=30 fruits, >=2 classes) met, but marginal: a single grouped hold-out would have ~8 fruits, so report grouped CV (GroupKFold) alongside it in M5.
- **The official train/val/test split leaks**: front and back of the same fruit land in different splits. Do not use it; build our own split by fruit ID. Record `id`s also collide across the three annotation files, so key on (camera, header path).
- ENVI headers have no wavelengths; they come from `cameras` in the annotation JSON. Values look like reflectance (-0.09 to 0.89).
- Extra labels available for later: firmness, storage_days (stretch: firmness regression).

