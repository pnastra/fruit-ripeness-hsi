.PHONY: data features train-baseline train-cnn serve docker-build docker-run deploy test lint

test:
	uv run pytest

lint:
	uv run ruff check .

data:            # M1
	@echo "not implemented yet (M1)"

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
