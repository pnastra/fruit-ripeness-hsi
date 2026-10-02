.PHONY: data inventory features train-baseline train-cnn serve docker-build docker-run deploy test lint

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

train-cnn:       # M6: 3 CNN configurations -> MLflow experiment "cnn"
	uv run python -m fruithsi.train

serve:           # M8
	@echo "not implemented yet (M8)"

docker-build:    # M9
	@echo "not implemented yet (M9)"

docker-run:      # M9
	@echo "not implemented yet (M9)"

deploy:          # M10
	@echo "not implemented yet (M10)"
