# Serving image: FastAPI + numpy only (the shipped PLS-DA is a plain-numpy JSON model,
# so no torch / scikit-learn / scipy is needed). Build for Cloud Run's platform:
#   docker build --platform linux/amd64 -t fruit-ripeness:local .
FROM python:3.12-slim

# uv installs exactly what uv.lock pins
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

WORKDIR /app

# 1) dependencies first, so this layer is cached until pyproject.toml / uv.lock change.
#    --no-default-groups skips the train and dev groups; --no-install-project skips our own code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) code, model and sample spectra (no data, notebooks or training outputs: see .dockerignore)
COPY src ./src
COPY api ./api
COPY models/plsda_snv.json ./models/plsda_snv.json
COPY samples ./samples

# run as a non-root user
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# Cloud Run injects $PORT; `exec` lets uvicorn receive SIGTERM directly
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
