"""Shared fixtures.

Unit tests use small synthetic tables and run without the client data pack. Tests marked
`needs_model` use the real data pack and a trained model, and are skipped when either is missing.
"""
import pandas as pd
import pytest

from kestrel_fraud.config import ARTIFACT_PATH, DATA_DIR, OUTPUT_DIR

HAVE_MODEL = ARTIFACT_PATH.exists() and (OUTPUT_DIR / "predictions.csv").exists() and DATA_DIR.exists()


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="needs the data pack and a trained model: run python -m kestrel_fraud.train")
    for item in items:
        if "needs_model" in item.keywords and not HAVE_MODEL:
            item.add_marker(skip)


@pytest.fixture
def partners() -> pd.DataFrame:
    return pd.DataFrame({"partner_id": ["P1", "P2"], "city": ["Pune", "Nashik"],
                         "onboarded_date": pd.to_datetime(["2025-12-01", "2021-01-01"]),
                         "partner_type": ["franchise", "authorised_service_centre"]})


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame({"sku": ["KH-AF-01"], "family": ["Air Fryer"], "list_price_inr": [5000],
                         "warranty_months": [12]})


def make_claim(claim_id: str, submitted_at: str, partner_id: str = "P1", is_fraud=None, **overrides) -> dict:
    """A raw claim as it appears in train.csv, with sensible defaults."""
    claim = {"claim_id": claim_id, "submitted_at": pd.Timestamp(submitted_at), "partner_id": partner_id,
             "sku": "KH-AF-01", "product_serial": "KH123456789", "days_since_purchase": 100,
             "claim_amount_inr": 1500.0, "photo_attached": "Y", "partner_inspected": "N",
             "claim_description": "water leaking", "inspector_note": None, "customer_prior_claims": 0,
             "source": "crm", "is_fraud": is_fraud, "dataset": "train"}
    return {**claim, **overrides}
