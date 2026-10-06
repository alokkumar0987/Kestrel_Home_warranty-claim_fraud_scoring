"""Time-based validation and the review-desk metric, shared by evaluate.py and the tests."""
from collections.abc import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from .config import REVIEWS_PER_MONTH
from .features import build_features


def time_split(history: pd.DataFrame, cutoff: str | pd.Timestamp,
               freeze: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on labelled claims before `cutoff`, validate on labelled claims from `cutoff`.

    Outcomes from `freeze` (default: the cutoff) on are hidden while features are built, so
    validation claims see partner history frozen there, as real claims will see it frozen at the
    export date. An earlier `freeze` simulates scoring months after the export (staleness backtest).
    """
    cutoff = pd.Timestamp(cutoff)
    freeze = cutoff if freeze is None else pd.Timestamp(freeze)
    truth = history.set_index("claim_id")["is_fraud"]
    masked = history.copy()
    masked.loc[masked["submitted_at"] >= freeze, "is_fraud"] = np.nan

    fe = build_features(masked)
    fe = fe[fe["dataset"] == "train"].copy()
    fe["is_fraud"] = fe["claim_id"].map(truth)
    fe = fe[fe["is_fraud"].notna()].astype({"is_fraud": int})
    return fe[fe["submitted_at"] < cutoff].copy(), fe[fe["submitted_at"] >= cutoff].copy()


def review_top_k(claims: pd.DataFrame, scores: np.ndarray, k: int = REVIEWS_PER_MONTH) -> dict:
    """What the desk would find if it reviewed the k highest-scoring claims."""
    picked = claims.assign(score=scores).nlargest(k, "score")
    caught = int(picked["is_fraud"].sum())
    rupees = float(picked.loc[picked["is_fraud"] == 1, "claim_amount_inr"].sum())
    return {"picked": picked, "k": k, "caught": caught, "frauds": int(claims["is_fraud"].sum()),
            "rupees": rupees, "rupees_per_review": rupees / k}


Metric = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, metric: Metric, n_boot: int = 1000, seed: int = 0,
                 groups: np.ndarray | None = None) -> tuple[float, float]:
    """95% percentile interval of `metric(y, scores)` over claims resampled with replacement.

    With `groups`, whole groups are resampled (e.g. the same claim scored under several scenarios).
    """
    rng = np.random.default_rng(seed)
    _, group_index = np.unique(np.arange(len(y)) if groups is None else groups, return_inverse=True)
    # Row positions of each group, ascending, without scanning every row once per group
    order = np.argsort(group_index, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(group_index))[:-1])
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(len(members), size=len(members))])
        if 0 < y[idx].sum() < len(idx):          # the metric needs both classes
            values.append(metric(y[idx], scores[idx]))
    if not values:
        raise ValueError("No bootstrap sample contained both classes; the interval is undefined.")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def wilson_interval(successes: int, trials: int) -> str:
    """A proportion with its 95% Wilson interval, formatted for the report."""
    lo, hi = proportion_confint(successes, trials, method="wilson")
    return f"{successes}/{trials} = {successes / trials:.0%} (95% CI {lo:.0%}–{hi:.0%})"
