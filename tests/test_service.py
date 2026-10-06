"""API tests against the real data pack and trained model (skipped when either is missing)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kestrel_fraud.config import DATA_DIR, OUTPUT_DIR
from kestrel_fraud.data import find_file
from kestrel_fraud.service import app

pytestmark = pytest.mark.needs_model


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # the context manager runs startup, which loads the model
        yield c


@pytest.fixture(scope="module")
def test_claims() -> pd.DataFrame:
    return pd.read_csv(find_file(DATA_DIR, "test_unlabelled.csv"))


@pytest.fixture(scope="module")
def predictions() -> pd.Series:
    return pd.read_csv(OUTPUT_DIR / "predictions.csv").set_index("claim_id")["score"]


def as_request(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in row.items()}


def test_health_and_screen(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "Check risk" in page.text


def test_api_reproduces_batch_predictions(client, test_claims, predictions):
    for _, row in test_claims.sample(40, random_state=1).iterrows():
        got = client.post("/score", json=as_request(row)).json()["risk_score"]
        assert got == pytest.approx(predictions[row["claim_id"]], abs=1e-4), row["claim_id"]


def test_reasons_are_readable_and_consistent(client, test_claims):
    for _, row in test_claims.sample(30, random_state=2).iterrows():
        reasons = client.post("/score", json=as_request(row)).json()["reasons"]
        up = {r["factor"] for r in reasons["raising_risk"]}
        down = {r["factor"] for r in reasons["lowering_risk"]}
        assert not up & down, "a factor must not both raise and lower risk"
        assert all(r["text"] and "=" not in r["text"] for r in reasons["raising_risk"] + reasons["lowering_risk"])


def test_examples_round_trip(client):
    high = client.post("/score", json=client.get("/example?kind=high").json()).json()
    low = client.post("/score", json=client.get("/example?kind=low").json()).json()
    assert high["risk_score"] > high["review_threshold"] > low["risk_score"]
    assert high["reasons"]["raising_risk"]


def test_timezone_aware_input_is_read_as_ist(client):
    claim = client.get("/example?kind=high").json()
    ist = client.post("/score", json={**claim, "submitted_at": "2026-08-28T03:17:00"}).json()["risk_score"]
    utc = client.post("/score", json={**claim, "submitted_at": "2026-08-27T21:47:00Z"}).json()["risk_score"]
    assert ist == utc


@pytest.mark.parametrize("change, field", [
    ({"partner_id": "SP0000"}, "partner_id"),
    ({"sku": "XX-00"}, "sku"),
    ({"claim_amount_inr": -5}, "claim_amount_inr"),
    ({"photo_attached": "maybe"}, "photo_attached"),
])
def test_bad_input_is_rejected_with_a_clear_message(client, test_claims, change, field):
    r = client.post("/score", json={**as_request(test_claims.iloc[0]), **change})
    assert r.status_code == 422 and field in str(r.json()["detail"])


def test_service_without_model_fails_politely(client, test_claims):
    saved = app.state.bundle, app.state.bundle_error
    app.state.bundle, app.state.bundle_error = None, "Model not trained yet. Run `python -m kestrel_fraud.train`."
    try:
        assert client.get("/health").json()["status"] == "model_missing"
        r = client.post("/score", json=as_request(test_claims.iloc[0]))
        assert r.status_code == 503 and "python -m kestrel_fraud.train" in r.json()["detail"]
    finally:
        app.state.bundle, app.state.bundle_error = saved
