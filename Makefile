.PHONY: data inventory features model report explain train-baseline train-cnn serve docker-build docker-run deploy test lint

test:
	uv run pytest

lint:
	uv run ruff check .

data:            # M1: FRUIT=Mango by default
	./scripts/download_data.sh

inventory:       # M2: cube info + data/processed/inventory.parquet
	uv run python -m fruithsi.io

features:        # M3: data/processed/spectra.parquet
	uv run python -m fruithsi.preprocess

train-baseline: # M5: majority + PLS-DA -> MLflow (mlflow.db)
	uv run python -m fruithsi.baseline

model:           # M7: fit final PLS-DA on all data -> models/plsda_snv.json
	uv run python -m fruithsi.export

report:          # M7: docs/results.md + chart from MLflow (add ARGS="--permutation 200")
	uv run python -m fruithsi.report $(ARGS)

explain:         # M7: docs/band_importance.png (VIP vs CNN gradient x input)
	uv run python -m fruithsi.explain

train-cnn:       # M6: 3 CNN configurations -> MLflow experiment "cnn"
	uv run python -m fruithsi.train

serve:           # M8: local API at http://127.0.0.1:8000 (docs at /docs)
	uv run uvicorn api.main:app --reload --port 8000

docker-build:    # M9: image for Cloud Run (amd64; slower on an ARM Mac: emulated)
	docker build --platform linux/amd64 -t fruit-ripeness:local .

docker-run:      # M9: http://localhost:8080/docs
	docker run --rm -p 8080:8080 fruit-ripeness:local

# --- Google Cloud (M10). Needs gcloud on PATH and `gcloud auth configure-docker` done once.
PROJECT ?= fruit-ripeness-hsi
REGION  ?= asia-southeast1
IMAGE   := $(REGION)-docker.pkg.dev/$(PROJECT)/fruit-ripeness/api
TAG     ?= $(shell git rev-parse --short HEAD)

deploy:          # M10: build HEAD, push to Artifact Registry, deploy to Cloud Run (creates a new revision)
	@test -z "$$(git status --porcelain)" || (echo "commit your changes first: the image tag must match the code" && exit 1)
	docker build --platform linux/amd64 -t $(IMAGE):$(TAG) .
	docker push $(IMAGE):$(TAG)
	gcloud run deploy fruit-ripeness --image $(IMAGE):$(TAG) --region $(REGION) --project $(PROJECT) \
		--allow-unauthenticated --min-instances 0 --max-instances 1 --memory 1Gi
