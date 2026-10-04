import json
from pathlib import Path

import numpy as np
import pytest
from api.main import app, get_model
from fastapi.testclient import TestClient

from fruithsi import CLASSES

SAMPLES = Path("samples")
client = TestClient(app)


def body(name: str = "perfect_front.json") -> dict:
    return json.loads((SAMPLES / name).read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and data["n_bands"] == 192
    assert data["classes"] == list(CLASSES)
    assert data["band_range_nm"][0] < 477 and data["band_range_nm"][1] > 998
    assert data["model_version"] == get_model().version


@pytest.mark.parametrize("name", sorted(p.name for p in SAMPLES.glob("*_*.json")))
def test_predict_on_every_sample(name):
    r = client.post("/predict", json=body(name))
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] in CLASSES
    assert data["confidence"] == pytest.approx(max(data["probabilities"].values()))
    assert sum(data["probabilities"].values()) == pytest.approx(1)
    assert set(data["probabilities"]) == set(CLASSES)
    assert data["model_version"] == get_model().version


def test_predict_matches_the_model_directly():
    spectrum = body()["spectrum"]
    api = client.post("/predict", json={"spectrum": spectrum}).json()["probabilities"]
    direct = get_model().predict(spectrum)["probabilities"]
    for c in CLASSES:
        assert api[c] == pytest.approx(direct[c])


def test_wrong_length_is_422():
    for n in (0, 1, 191, 193):
        r = client.post("/predict", json={"spectrum": [0.3] * n})
        assert r.status_code == 422, n
    detail = client.post("/predict", json={"spectrum": [0.3] * 10}).json()["detail"]
    assert "expected 192 values" in json.dumps(detail)


def test_other_invalid_bodies_are_422():
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"spectrum": "abc"}).status_code == 422
    assert client.post("/predict", json={"spectrum": ["x"] * 192}).status_code == 422
    nan_body = '{"spectrum": [' + ",".join(["NaN"] + ["0.3"] * 191) + "]}"
    r = client.post("/predict", content=nan_body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_flat_spectrum_is_422_not_a_crash():
    r = client.post("/predict", json={"spectrum": [0.3] * 192})
    assert r.status_code == 422 and "flat" in r.json()["detail"]


def test_prediction_is_logged_as_one_json_line(capsys):
    r = client.post("/predict", json=body("unripe_front.json"))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1
    log = json.loads(lines[0])
    assert log["message"] == "prediction" and log["severity"] == "INFO"
    assert log["model_version"] == get_model().version
    assert log["predicted_class"] == r.json()["predicted_class"]
    assert log["confidence"] == pytest.approx(r.json()["confidence"], abs=1e-4)
    assert log["timestamp"].endswith("+00:00")
    assert "spectrum" not in log  # inputs are not logged


def test_failed_requests_are_not_logged_as_predictions(capsys):
    client.post("/predict", json={"spectrum": [0.3] * 5})
    client.post("/predict", json={"spectrum": [0.3] * 192})
    assert capsys.readouterr().out.strip() == ""


def test_root_redirects_to_docs_and_docs_has_an_example():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()["components"]["schemas"]["PredictRequest"]
    assert len(schema["examples"][0]["spectrum"]) == 192


def test_samples_match_the_training_spectra_layout():
    index = json.loads((SAMPLES / "index.json").read_text())["samples"]
    assert len(index) == 5 and {v["true_class"] for v in index.values()} == set(CLASSES)
    for name in index:
        spectrum = np.asarray(body(name)["spectrum"])
        assert spectrum.shape == (192,) and np.isfinite(spectrum).all()


def test_validation_errors_do_not_echo_the_input():
    r = client.post("/predict", json={"spectrum": [0.123456] * 10})
    assert r.status_code == 422
    assert "0.123456" not in r.text
    assert set(r.json()["detail"][0]) == {"loc", "msg", "type"}
