"""Point-in-time correctness: a claim's features must not depend on anything after it."""
import pandas as pd
import pytest

from kestrel_fraud.data import clean_claims
from kestrel_fraud.features import build_features

from .conftest import make_claim


def features_for(claims: list[dict], partners, products) -> pd.DataFrame:
    return build_features(clean_claims(pd.DataFrame(claims), partners, products)).set_index("claim_id")


@pytest.fixture
def history() -> list[dict]:
    return [
        make_claim("C1", "2026-01-01", is_fraud=1.0),
        make_claim("C2", "2026-01-20", is_fraud=0.0),
        make_claim("C3", "2026-02-15"),                       # being scored: no outcome yet
        make_claim("C4", "2026-03-30", is_fraud=1.0),
    ]


def test_fraud_history_respects_30_day_label_lag(history, partners, products):
    f = features_for(history, partners, products)
    # C2 (20 Jan): C1's outcome (1 Jan) is only 19 days old, so not yet known
    assert f.loc["C2", "p_frauds_90d"] == 0 and f.loc["C2", "p_past_labelled"] == 0
    # C3 (15 Feb): outcomes known up to 16 Jan, so C1 counts but C2 does not
    assert f.loc["C3", "p_frauds_90d"] == 1 and f.loc["C3", "p_past_labelled"] == 1


def test_later_outcomes_do_not_leak_backwards(history, partners, products):
    before = features_for(history, partners, products)
    history[3]["is_fraud"] = 0.0                              # change a later outcome
    history.append(make_claim("C5", "2026-04-02", is_fraud=1.0))
    after = features_for(history, partners, products)
    pd.testing.assert_frame_equal(before.loc[["C1", "C2", "C3"]], after.loc[["C1", "C2", "C3"]])


def test_partner_window_counts_only_earlier_claims_of_same_partner(history, partners, products):
    history.append(make_claim("X1", "2026-02-10", partner_id="P2"))    # other partner: must not count
    f = features_for(history, partners, products)
    assert f.loc["C1", "p_claims_30d"] == 0
    assert f.loc["C3", "p_claims_30d"] == 1                   # only C2 in [16 Jan, 15 Feb)
    assert f.loc["C3", "p_claims_90d"] == 2                   # C1 and C2


@pytest.mark.parametrize("amount, expected", [(1799, 0), (1800, 1), (1999, 1), (2000, 0)])
def test_near_threshold_band(amount, expected, partners, products):
    f = features_for([make_claim("C1", "2026-06-01", claim_amount_inr=float(amount))], partners, products)
    assert f.loc["C1", "near_threshold"] == expected


def test_new_partner_interaction_only_after_policy_change(partners, products):
    f = features_for([make_claim("A", "2026-04-01"), make_claim("B", "2026-06-01"),
                      make_claim("C", "2026-06-02", partner_id="P2")], partners, products)
    assert f.loc["A", "partner_is_new"] == 1 and f.loc["A", "new_x_post"] == 0
    assert f.loc["B", "new_x_post"] == 1
    assert f.loc["C", "partner_is_new"] == 0 and f.loc["C", "new_x_post"] == 0


def test_duplicate_claim_ids_are_rejected(partners, products):
    claims = clean_claims(pd.DataFrame([make_claim("C1", "2026-01-01"), make_claim("C1", "2026-01-05")]),
                          partners, products)
    with pytest.raises(ValueError, match="claim_id must be unique"):
        build_features(claims)
