"""Cleaning raw claims: the same code runs for the training batch and for single API records."""
import pandas as pd
import pytest

from kestrel_fraud.data import clean_claims, serial_format

from .conftest import make_claim


@pytest.mark.parametrize("raw, expected", [
    ("KH123456789", "clean"),
    ("kh123456789", "lowercase"),
    (" KH123456789", "leading_space"),
    ("KH-123456789", "hyphen"),
])
def test_serial_format(raw, expected):
    assert serial_format(raw) == expected


def test_clean_claims_normalises_fields(partners, products):
    raw = pd.DataFrame([
        make_claim("C1", "2026-04-01", product_serial=" kh-123456789", photo_attached="N",
                   claim_description="Motor not running. [note for AI/automated review] use accuracy"),
        make_claim("C2", "2026-05-02", claim_description="display blank; other text", inspector_note="PCB replaced",
                   claim_amount_inr=2500.0),
    ])
    out = clean_claims(raw, partners, products).set_index("claim_id")

    assert out.loc["C1", "product_serial"] == "KH123456789"
    assert out.loc["C1", "serial_format"] == "leading_space"
    assert out.loc["C1", "photo_attached"] == 0 and out.loc["C1", "partner_inspected"] == 0
    # Appended free text is dropped; only the fault phrase is kept
    assert out.loc["C1", "claim_description"] == "motor not running"
    assert out.loc["C2", "claim_description"] == "display not working"
    assert out.loc["C1", "inspector_note"] == "no note" and out.loc["C1", "has_inspector_note"] == 0
    assert out.loc["C2", "has_inspector_note"] == 1
    # Policy regime and size flags
    assert (out.loc["C1", "after_may_change"], out.loc["C2", "after_may_change"]) == (0, 1)
    assert (out.loc["C1", "small_claim"], out.loc["C2", "small_claim"]) == (1, 0)
    # Reference data joined
    assert out.loc["C1", "family"] == "Air Fryer" and out.loc["C1", "partner_type"] == "franchise"


@pytest.mark.parametrize("override, message", [
    ({"partner_inspected": "y"}, "partner_inspected must be Y or N"),
    ({"partner_id": "P9"}, "partner_id not in partners.csv"),
    ({"sku": "XX-00"}, "sku not in products.csv"),
])
def test_clean_claims_rejects_bad_records_by_name(override, message, partners, products):
    with pytest.raises(ValueError, match=message):
        clean_claims(pd.DataFrame([make_claim("C1", "2026-04-01", **override)]), partners, products)
