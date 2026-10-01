.PHONY: data inventory features train-baseline train-cnn serve docker-build docker-run deploy test lint

test:
	uv run pytest

lint:
	uv run ruff check .

data:            # M1: FRUIT=Mango by default
	./scripts/download_data.sh

inventory:       # M2: cube info + data/processed/inventory.parquet
	uv run python -m fruithsi.io

features:        # M3
	@echo "not implemented yet (M3)"

train-baseline:  # M5
	@echo "not implemented yet (M5)"

train-cnn:       # M6
	@echo "not implemented yet (M6)"

serve:           # M8
	@echo "not implemented yet (M8)"

docker-build:    # M9
	@echo "not implemented yet (M9)"

docker-run:      # M9
	@echo "not implemented yet (M9)"

deploy:          # M10
	@echo "not implemented yet (M10)"
