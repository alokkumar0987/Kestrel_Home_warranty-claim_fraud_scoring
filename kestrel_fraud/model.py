"""The estimator, its training weights, and the saved model bundle used by the service."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LOGISTIC_C, POST_MAY_WEIGHT
from .features import CATEGORICAL, FEATURES, NUMERIC

BUNDLE_FORMAT = 2   # bump when ModelBundle fields change


class BundleError(RuntimeError):
    """The saved model cannot be used; the message says what to do."""


def make_estimator() -> Pipeline:
    """Logistic regression on scaled numeric + one-hot categorical features.

    Chosen over XGBoost: equal on the top 40 a month, better deeper in the list, and its
    coefficients give exact, explainable reasons (notebook §7.4).
    """
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(prep, LogisticRegression(C=LOGISTIC_C, class_weight="balanced", max_iter=3000))


def sample_weights(train: pd.DataFrame, post_may_weight: float = POST_MAY_WEIGHT) -> np.ndarray:
    """Post-policy claims count more: they look like the claims being scored."""
    return np.where(train["after_may_change"] == 1, post_may_weight, 1.0)


def fit_estimator(train: pd.DataFrame, post_may_weight: float = POST_MAY_WEIGHT) -> Pipeline:
    """Fit the estimator on labelled claims, with post-policy claims weighted up."""
    return make_estimator().fit(train[FEATURES], train["is_fraud"],
                                logisticregression__sample_weight=sample_weights(train, post_may_weight))


def weight_inflation(train: pd.DataFrame, post_may_weight: float = POST_MAY_WEIGHT) -> float:
    """Factor by which class and sample weights inflated the training odds of fraud."""
    y = train["is_fraud"].to_numpy()
    n, n_pos = len(y), y.sum()
    class_w = np.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))   # sklearn's "balanced"
    w = class_w * sample_weights(train, post_may_weight)
    return float((w[y == 1].sum() / w[y == 0].sum()) / (n_pos / (n - n_pos)))


def prior_correct(p: np.ndarray, inflation: float) -> np.ndarray:
    """Remove the training-weight inflation from predicted probabilities.

    Monotone, so the ranking (and ROC-AUC / PR-AUC) is unchanged, and label-free: nothing is
    fitted on outcomes. It puts scores back on a probability-like scale so a 0.5 threshold
    is not swamped by false alarms.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    odds = p / (1 - p) / inflation
    return odds / (1 + odds)


def risk_scores(estimator: Pipeline, features: pd.DataFrame, inflation: float) -> np.ndarray:
    """The risk score used everywhere (batch, validation, API): model output with weight inflation removed."""
    return prior_correct(estimator.predict_proba(features[FEATURES])[:, 1], inflation)


@dataclass
class ModelBundle:
    """Everything the service needs to score one claim."""
    estimator: Pipeline
    inflation: float
    review_threshold: float          # score of the 40th claim per month in the scored period
    reference_scores: np.ndarray     # sorted scores of the scored period, for percentiles
    feature_means: np.ndarray        # mean transformed training row: the baseline for reasons
    typical: dict                    # typical partner values, quoted in reasons
    examples: dict                   # claim ids to try on the screen: {"high": ..., "low": ...}
    history: pd.DataFrame            # cleaned known claims: partner history for new claims
    partners: pd.DataFrame
    products: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Risk scores for claims that already have model features."""
        return risk_scores(self.estimator, features, self.inflation)

    def save(self, path: Path) -> None:
        """Write the bundle with the version facts that `load` checks."""
        self.metadata.update(format=BUNDLE_FORMAT, sklearn_version=sklearn.__version__, features=FEATURES,
                             saved_at=datetime.now().isoformat(timespec="seconds"))
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        """Load a bundle, or raise BundleError saying how to fix it (missing, corrupt or out of date)."""
        retrain = "Run `python -m kestrel_fraud.train` (see README), then restart."
        if not path.exists():
            raise BundleError(f"Model not trained yet. {retrain}")
        try:
            bundle = joblib.load(path)
        except Exception as exc:   # corrupt file or pickle from incompatible library versions
            raise BundleError(f"Saved model could not be loaded ({type(exc).__name__}). {retrain}") from exc
        meta = getattr(bundle, "metadata", {})
        if not isinstance(bundle, cls) or meta.get("format") != BUNDLE_FORMAT:
            raise BundleError(f"Saved model is from an older version of this code. {retrain}")
        if meta.get("features") != FEATURES:
            raise BundleError(f"Saved model uses a different feature set. {retrain}")
        if meta.get("sklearn_version") != sklearn.__version__:
            raise BundleError(f"Saved model was built with scikit-learn {meta.get('sklearn_version')}, "
                              f"this environment has {sklearn.__version__}. {retrain}")
        return bundle
