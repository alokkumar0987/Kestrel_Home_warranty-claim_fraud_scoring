"""Time-based split with a staleness freeze, and the bootstrap interval behind the expected score."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from kestrel_fraud.data import clean_claims
from kestrel_fraud.validation import bootstrap_ci, time_split

from .conftest import make_claim


def test_freeze_hides_outcomes_from_validation_features_but_keeps_labels(partners, products):
    history = clean_claims(pd.DataFrame([
        make_claim("C1", "2026-02-20", is_fraud=1.0),
        make_claim("C2", "2026-04-20", is_fraud=0.0),
    ]), partners, products)

    _, current = time_split(history, "2026-04-01")
    train, stale = time_split(history, "2026-04-01", freeze="2026-02-01")

    assert current.set_index("claim_id").loc["C2", "p_frauds_90d"] == 1   # C1's outcome is 59 days old: known
    assert stale.set_index("claim_id").loc["C2", "p_frauds_90d"] == 0     # frozen before C1: not known
    assert train.set_index("claim_id").loc["C1", "is_fraud"] == 1          # true labels are restored
    assert stale.set_index("claim_id").loc["C2", "is_fraud"] == 0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.1).astype(int)
    scores = y + rng.normal(0, 0.8, size=400)
    lo, hi = bootstrap_ci(y, scores, average_precision_score, n_boot=200)
    assert lo < average_precision_score(y, scores) < hi


def test_grouped_bootstrap_matches_plain_bootstrap_for_singleton_groups():
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.2).astype(int)
    scores = rng.random(200)
    plain = bootstrap_ci(y, scores, average_precision_score, n_boot=50, seed=3)
    grouped = bootstrap_ci(y, scores, average_precision_score, n_boot=50, seed=3, groups=np.arange(200))
    assert plain == grouped


def test_bootstrap_ci_fails_clearly_when_only_one_class():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_ci(np.zeros(50), np.random.default_rng(0).random(50), average_precision_score, n_boot=20)
