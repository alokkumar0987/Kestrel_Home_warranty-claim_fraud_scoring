"""Point-in-time features.

Every feature of a claim uses only what was known when it was submitted:
- partner behaviour uses the partner's earlier claims (labels not needed),
- partner fraud history uses outcomes of claims at least LABEL_LAG_DAYS old.
The lag matters: without it, training rows would see outcomes from the day before while
test rows only see outcomes up to the export date (train-serve skew, notebook §6).
"""
import numpy as np
import pandas as pd

from .config import (
    AUTO_APPROVE_LIMIT_INR,
    DAYS_PER_MONTH,
    LABEL_LAG_DAYS,
    NEAR_LIMIT_FLOOR_INR,
    NEW_PARTNER_DAYS,
    PRIOR_FRAUD_RATE,
    PRIOR_WEIGHT,
)

# Order matches the notebook (§6.7). Reordering changes the solver's path and so the fitted model slightly.
NUMERIC = [
    # claim
    "log_amount", "amount_to_price", "near_threshold", "warranty_used_frac", "customer_prior_claims",
    "partner_inspected", "has_inspector_note", "small_claim", "after_may_change", "submit_dow",
    # partner profile
    "partner_tenure_days", "partner_is_new",
    # partner behaviour, trailing windows
    "p_claims_30d", "p_near_threshold_share_30d", "p_uninspected_share_30d", "p_amount_ratio_mean_30d",
    "p_claims_90d", "p_near_threshold_share_90d", "p_uninspected_share_90d", "p_amount_ratio_mean_90d",
    # partner fraud history, lagged
    "p_frauds_90d", "p_past_labelled", "p_past_fraud_rate",
    # interactions
    "new_x_post", "near_thr_x_uninsp",
]
CATEGORICAL = ["partner_type", "family", "claim_description", "inspector_note", "serial_format"]
FEATURES = NUMERIC + CATEGORICAL


def _partner_window_features(claims: pd.DataFrame, days: int) -> pd.DataFrame:
    """Partner behaviour over the `days` before each claim. closed='left' excludes the claim itself."""
    window = f"{days}D"
    by_partner = claims.set_index("submitted_at").groupby("partner_id")

    def roll(col: str, how: str) -> np.ndarray:
        return by_partner[col].transform(lambda s: getattr(s.rolling(window, closed="left"), how)()).values

    return pd.DataFrame({
        "claim_id": claims["claim_id"].values,
        f"p_claims_{days}d": roll("claim_id", "count"),
        f"p_near_threshold_share_{days}d": roll("near_threshold", "mean"),
        f"p_uninspected_share_{days}d": roll("not_inspected", "mean"),
        f"p_amount_ratio_mean_{days}d": roll("amount_to_price", "mean"),
    })


def _labels_before(claims: pd.DataFrame, cutoff: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per claim: the partner's (frauds, labelled claims) among claims submitted strictly before `cutoff`."""
    # astype(float): a set with no outcomes at all arrives as an object column of None
    events = (claims.loc[claims["is_fraud"].notna(), ["partner_id", "submitted_at", "is_fraud"]]
                    .astype({"is_fraud": float}).sort_values("submitted_at"))
    events["cum_frauds"] = events.groupby("partner_id")["is_fraud"].cumsum()
    events["cum_labelled"] = events.groupby("partner_id").cumcount() + 1

    query = pd.DataFrame({"row": claims.index, "partner_id": claims["partner_id"], "cutoff": cutoff})
    found = pd.merge_asof(query.sort_values("cutoff"), events.drop(columns="is_fraud"),
                          left_on="cutoff", right_on="submitted_at", by="partner_id", allow_exact_matches=False)
    found = found.set_index("row").reindex(claims.index)
    return found["cum_frauds"].fillna(0).values, found["cum_labelled"].fillna(0).values


def build_features(claims: pd.DataFrame, label_lag_days: int = LABEL_LAG_DAYS) -> pd.DataFrame:
    """Add model features to cleaned claims. Each row's history is the other rows of `claims`.

    To score one new claim, pass it together with the same partner's earlier claims.
    """
    if not claims["claim_id"].is_unique:
        raise ValueError("claim_id must be unique: duplicates would corrupt the per-claim feature merge.")
    claims = claims.sort_values(["submitted_at", "claim_id"]).reset_index(drop=True)
    amount = claims["claim_amount_inr"]

    # Claim
    claims["log_amount"] = np.log1p(amount)
    claims["amount_to_price"] = amount / claims["list_price_inr"]
    claims["near_threshold"] = ((amount >= NEAR_LIMIT_FLOOR_INR) & (amount < AUTO_APPROVE_LIMIT_INR)).astype(int)
    claims["warranty_used_frac"] = claims["days_since_purchase"] / (claims["warranty_months"] * DAYS_PER_MONTH)
    claims["not_inspected"] = 1 - claims["partner_inspected"]
    claims["near_thr_x_uninsp"] = claims["near_threshold"] * claims["not_inspected"]
    claims["submit_dow"] = claims["submitted_at"].dt.dayofweek

    # Partner profile at claim time. New-partner risk only appeared after the policy change,
    # hence the interaction (hypothesis notebook).
    claims["partner_tenure_days"] = (claims["submitted_at"] - claims["onboarded_date"]).dt.days
    claims["partner_is_new"] = (claims["partner_tenure_days"] < NEW_PARTNER_DAYS).astype(int)
    claims["new_x_post"] = claims["partner_is_new"] * claims["after_may_change"]

    # Partner behaviour: no labels, so every earlier claim counts (including unlabelled ones)
    for days in (30, 90):
        claims = claims.merge(_partner_window_features(claims, days), on="claim_id", how="left")
    window_cols = [c for c in claims.columns if c.startswith("p_") and c.endswith(("_30d", "_90d"))]
    claims[window_cols] = claims[window_cols].fillna(0)   # no earlier claims in the window: no evidence

    # Partner fraud history: outcomes of claims at least `label_lag_days` old.
    # p_frauds_90d counts frauds among claims 30-90 days old: recent, because the fraudsters changed in May 2026.
    frauds_known, labelled_known = _labels_before(claims, claims["submitted_at"] - pd.Timedelta(days=label_lag_days))
    frauds_90d_ago, _ = _labels_before(claims, claims["submitted_at"] - pd.Timedelta(days=90))
    claims["p_past_labelled"] = labelled_known
    claims["p_frauds_90d"] = frauds_known - frauds_90d_ago
    claims["p_past_fraud_rate"] = ((frauds_known + PRIOR_FRAUD_RATE * PRIOR_WEIGHT)
                                   / (labelled_known + PRIOR_WEIGHT))
    return claims
