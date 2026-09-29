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
| Google Cloud account, billing enabled, **budget alert at $1** | ⬜ M10 |
| `gcloud` CLI | ⬜ M10 |
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
- [ ] Create the public GitHub repo `fruit-ripeness-hsi` (MIT); `uv init`, `uv python pin 3.12`, deps from §3, commit `uv.lock`
- [ ] `.gitignore` (§9), `CLAUDE.md`, `Makefile` stubs, `src/fruithsi/` package, one trivial passing test
- [ ] License: check the annotations zip, dataset readme and GitHub repo. If none, draft a short email to the authors (I send it) and note the status in the README.

**Done when:** `uv run pytest` passes, the first commit is pushed, and the README states the license status.

#### M1 — Download one fruit (≈2 h, mostly waiting)
- [ ] Check free disk space (need 2–3× the zip size)
- [ ] `make data`: annotations + **Mango or Kaki**, resume-capable download (`curl -C -`), extract to `data/raw/<fruit>/`
- [ ] Record the folder tree (depth 3) and file counts in §12

**Done when:** the data is extracted and the tree is in §12.

#### M2 — Load, inventory, go/no-go (≈3 h)
- [ ] `io.py`: read one ENVI cube; print shape, wavelengths, camera
- [ ] Inventory table (fruit ID, camera, day, path, labels) → `data/processed/inventory.parquet`
- [ ] Count labelled fruits per ripeness class per camera; pick one camera
- [ ] **Go:** at least ~30 labelled fruits and at least 2 classes. **No-go:** try the other small fruit once; if it still fails, stop and ask me.

**Done when:** the decision (fruit, camera, task) is in §12, and a test checks the loader on one cube.

#### M3 — Segmentation and mean spectra (≈4 h)
- [ ] Simple fruit/background mask (threshold on intensity or a band ratio); save one mask example for the README
- [ ] `preprocess.py`: one mean spectrum per image → `data/processed/spectra.parquet`; drop noisy edge bands
- [ ] SNV and Savitzky–Golay as options

**Done when:** `spectra.parquet` exists, and a test checks the band count and that there are no NaNs.

#### M4 — Quick EDA (≈3 h, hard timebox)
- [ ] Label distribution; images per physical fruit
- [ ] Mean spectra by class (raw vs SNV vs SG derivative) in one figure
- [ ] Write 3–5 findings for the README and "Decisions for M5" at the end of the notebook

**Done when:** the decisions are recorded and the notebook runs top to bottom.

#### M5 — Split, baselines, leakage demo (≈4 h)
- [ ] `split.py`: grouped hold-out by fruit ID (plus `GroupKFold` for CV)
- [ ] `baseline.py`: majority class and **PLS-DA**; metrics accuracy, macro-F1, confusion matrix; MLflow `sqlite:///mlflow.db`
- [ ] Leakage demo: PLS-DA on a random split by image, logged as its own run
- [ ] Test: no fruit ID in both train and test

**Done when:** three runs are in MLflow (majority, PLS-DA grouped, PLS-DA random).

#### M6 — 1D-CNN (≈5 h)
- [ ] `dataset.py`, `model.py` (small 1D-CNN), `train.py` (loop, early stopping, fixed seeds, CPU/MPS)
- [ ] Train on the grouped split; try **at most 3 configurations** (no grid search); log to MLflow

**Done when:** the loss curve is sensible and the best CNN run is in MLflow.

### Week 2 — Ship

#### M7 — Compare, save, explain (≈4 h)
- [ ] Results table: majority vs PLS-DA vs 1D-CNN (grouped) + PLS-DA random split
- [ ] Save `models/model.pt` + preprocessing parameters (bands, SNV/SG settings, class names); test that it loads and predicts
- [ ] One band-importance plot: PLS VIP scores and CNN gradient × input on the same axes; 2–3 sentences on whether the key wavelengths match known chemistry (e.g. chlorophyll ~680 nm, water bands)

**Done when:** the table and plot are saved for the README.

#### M8 — API (≈3 h)
- [ ] `api/main.py`: `GET /health`, `POST /predict` (validated spectrum length → class probabilities)
- [ ] Each prediction logged as one JSON line to stdout (timestamp, model version, class, confidence)
- [ ] Put 3–5 derived sample spectra in `samples/`; `TestClient` tests, including wrong length → 422

**Done when:** `make serve` works locally and the API tests pass.

#### M9 — Docker (≈3 h)
- [ ] `Dockerfile`: `python:3.12-slim`, CPU-only torch, serving deps only; image under ~1 GB; build with `--platform linux/amd64`
- [ ] `make docker-build && make docker-run`; call `/predict` with a sample

**Done when:** the container answers `/predict` locally and the image size is recorded.

#### M10 — Google Cloud setup and first deploy (≈4 h) — **ask me before each step**
- [ ] GCP project, billing, **$1 budget alert**; install `gcloud`; enable Cloud Run and Artifact Registry
- [ ] Artifact Registry repo in `asia-southeast1`; push the image
- [ ] `gcloud run deploy` (min 0, max 1, 1 GiB, unauthenticated); check `/health` and `/docs` from a cold start; find a prediction line in Cloud Logging

**Done when:** the public URL works and is written in §12.

#### M11 — CI/CD (≈5 h) — **ask me before creating anything**
- [ ] `ci.yml`: ruff + pytest on every push and PR; CI badge in the README
- [ ] `deploy.yml`: on push to `main`, build → Artifact Registry → Cloud Run, authenticated with **Workload Identity Federation** (explain each IAM step)
- [ ] Push a small visible change and confirm it redeploys

**Done when:** a push to `main` runs CI and updates the live service with no manual steps.

#### M12 — README and model card (≈4 h)
- [ ] Top: live `/docs` link, screenshot, one-sentence summary, CI badge
- [ ] Findings (M4), results table and band-importance plot (M7)
- [ ] Architecture diagram: GitHub → Actions → Artifact Registry → Cloud Run → Cloud Logging
- [ ] Model card: intended use, limits (one fruit, one camera, lab conditions, small sample, not for commercial grading), data source, license status
- [ ] Credits, citation, how to run locally

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

- [ ] Public GitHub repo with README, MIT license (code), passing CI badge
- [ ] A live Cloud Run URL
- [ ] The CNN is compared honestly with PLS-DA; if it doesn't win, the README says so and explains why
- [ ] Every push to `main` redeploys automatically
- [ ] No secrets, raw images or large checkpoints in the git history
- [ ] The dataset license status is stated in the README
- [ ] The GCP budget alert is in place, and the service scales to zero

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
| | | |
