"""FastAPI service for the shipped PLS-DA ripeness model. /docs is the demo.

    GET  /health   liveness + model info
    POST /predict  one mean spectrum -> class probabilities (one JSON log line per prediction)
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fruithsi.predict import PLSDAModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "plsda_snv.json"
EXAMPLE_FILE = ROOT / "samples" / "perfect_front.json"

DESCRIPTION = """
Predicts the ripeness class (**unripe / perfect / overripe**) of a mango from one **mean
reflectance spectrum**: 192 values, the mean over the fruit pixels of a hyperspectral image
(Specim FX10), bands 476.5 to 998.2 nm in ascending order.

**Research demo, not a grading tool.** The model (PLS-DA) was trained on 40 mangoes from one lab
dataset (DeepHS Fruit 2023) and reaches about 0.5 fruit-level accuracy under cross-validation
(majority class: 0.4), so confidences are modest. Use the pre-filled example in `/predict`
(a training-set spectrum) to try it.
"""


@lru_cache(maxsize=1)
def get_model() -> PLSDAModel:
    """Load the exported model once (path overridable with the MODEL_PATH environment variable)."""
    return PLSDAModel.load(os.environ.get("MODEL_PATH", DEFAULT_MODEL))


def _example() -> dict:
    return json.loads(EXAMPLE_FILE.read_text()) if EXAMPLE_FILE.exists() else {}


Model = Annotated[PLSDAModel, Depends(get_model)]  # injected, loaded once


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [_example()]} if _example() else {})

    spectrum: list[float] = Field(
        description="Mean reflectance per band (192 values, 476.5 to 998.2 nm, ascending)."
    )

    @field_validator("spectrum")
    @classmethod
    def check_spectrum(cls, values: list[float]) -> list[float]:
        model = get_model()
        if len(values) != model.n_bands:
            lo, hi = model.bands_nm[0], model.bands_nm[-1]
            raise ValueError(
                f"expected {model.n_bands} values ({lo:.1f} to {hi:.1f} nm), got {len(values)}"
            )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("all values must be finite numbers")
        return values


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(description="Probability of the predicted class.")
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    classes: list[str]
    n_bands: int
    band_range_nm: tuple[float, float]


app = FastAPI(title="Mango ripeness from hyperspectral spectra", version="0.1.1",
              description=DESCRIPTION)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with where / what / why only. FastAPI's default also echoes the offending input, which
    cannot be JSON-encoded when it contains NaN (the client would get a 500 instead of a 422)."""
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health(model: Model) -> HealthResponse:
    return HealthResponse(
        status="ok", model_version=model.version, classes=model.classes, n_bands=model.n_bands,
        band_range_nm=(float(model.bands_nm[0]), float(model.bands_nm[-1])),
    )


def log_prediction(model_version: str, predicted_class: str, confidence: float) -> None:
    """One JSON object per line on stdout; Cloud Logging turns it into a structured log entry."""
    print(json.dumps({
        "severity": "INFO", "message": "prediction",
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "model_version": model_version, "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
    }), flush=True)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model: Model) -> PredictResponse:
    try:
        out = model.predict(request.spectrum)
    except ValueError as err:  # e.g. a flat spectrum cannot be normalised
        raise HTTPException(status_code=422, detail=str(err)) from err
    log_prediction(model.version, out["class"], out["confidence"])
    return PredictResponse(
        predicted_class=out["class"], confidence=out["confidence"],
        probabilities=out["probabilities"], model_version=model.version,
    )
