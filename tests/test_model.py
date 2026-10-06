"""Score correction and the saved model bundle."""
import numpy as np
import pandas as pd
import pytest

from kestrel_fraud.model import BundleError, ModelBundle, prior_correct, weight_inflation


def test_prior_correct_is_identity_without_inflation():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    np.testing.assert_allclose(prior_correct(p, 1.0), p)


def test_prior_correct_keeps_ranking_and_lowers_scores():
    p = np.random.default_rng(0).uniform(0, 1, 200)
    corrected = prior_correct(p, 50.0)
    assert (np.argsort(corrected) == np.argsort(p)).all()
    assert (corrected <= p).all()


def test_weight_inflation_matches_balanced_class_weights():
    # 10 frauds, 90 genuine, no post-policy rows: balanced weights make the classes equal,
    # so the odds are inflated from 10/90 to 1, a factor of 9
    train = pd.DataFrame({"is_fraud": [1] * 10 + [0] * 90, "after_may_change": 0})
    assert weight_inflation(train, post_may_weight=3) == pytest.approx(9.0)


def test_missing_bundle_gives_actionable_error(tmp_path):
    with pytest.raises(BundleError, match="python -m kestrel_fraud.train"):
        ModelBundle.load(tmp_path / "missing.joblib")
