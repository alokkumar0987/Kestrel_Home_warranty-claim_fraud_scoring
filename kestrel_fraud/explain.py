"""Plain-language reasons for a score, written for a claims officer.

The model is a logistic regression, so each feature's push on a claim's score is exact:
    push = coefficient x (this claim's transformed value - the average training claim's value)
in log-odds. Related features are summed into one factor (GROUPS) so that overlapping features
cannot appear as contradicting reasons, e.g. "first-year partner" lowering risk while
"first-year partner after May" raises it.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import AUTO_APPROVE_LIMIT_INR, PRIOR_FRAUD_RATE
from .features import CATEGORICAL, NUMERIC

# Not shown to the officer: a calendar flag, the weekday, and how many claims were investigated
HIDDEN = {"after_may_change", "submit_dow", "p_past_labelled"}

GROUPS = {
    "partner fraud history": ["p_frauds_90d", "p_past_fraud_rate"],
    "partner tenure": ["new_x_post", "partner_is_new", "partner_tenure_days"],
    "partner claim volume": ["p_claims_30d", "p_claims_90d"],
    "partner claim pattern": ["p_near_threshold_share_30d", "p_near_threshold_share_90d", "p_uninspected_share_30d",
                              "p_uninspected_share_90d", "p_amount_ratio_mean_30d", "p_amount_ratio_mean_90d"],
    "claim amount": ["near_thr_x_uninsp", "near_threshold", "small_claim", "amount_to_price", "log_amount"],
    "inspection": ["partner_inspected", "has_inspector_note", "inspector_note"],
}
_GROUP_OF = {feature: group for group, members in GROUPS.items() for feature in members}

SERIAL_STYLE = {"clean": "in the standard format", "lowercase": "in lower case",
                "leading_space": "with a leading space", "hyphen": "with a hyphen"}
LIMIT = f"Rs {AUTO_APPROVE_LIMIT_INR:,}"

# One sentence per feature: (value, claim row, typical partner values) -> text
Template = Callable[[object, pd.Series, dict], str]
TEMPLATES: dict[str, Template] = {
    "p_frauds_90d": lambda v, r, t: (
        f"Partner {r.partner_id} had {int(v)} confirmed fraud claim(s) among its claims from 30-90 days earlier."
        if v > 0 else f"Partner {r.partner_id} has no confirmed fraud among its claims from 30-90 days earlier."),
    "p_past_fraud_rate": lambda v, r, t: (
        f"Partner {r.partner_id}'s track record: estimated fraud rate {v:.1%} over {int(r.p_past_labelled)} "
        f"investigated claims (company average {PRIOR_FRAUD_RATE:.1%})."),
    "partner_tenure_days": lambda v, r, t: f"Partner {r.partner_id} joined Kestrel {int(v)} days before this claim.",
    "partner_is_new": lambda v, r, t: (
        "Partner is in its first year with Kestrel." if v else "Partner has been with Kestrel over a year."),
    "new_x_post": lambda v, r, t: (
        "First-year partner claiming after the 1 May 2026 auto-approval rule: the pattern behind most recent fraud."
        if v else "Not a first-year partner claiming after the May 2026 rule."),
    "p_claims_30d": lambda v, r, t: (
        f"Partner submitted {int(v)} claim(s) in the previous 30 days (typical partner: {t['p_claims_30d']:.0f})."),
    "p_claims_90d": lambda v, r, t: (
        f"Partner submitted {int(v)} claim(s) in the previous 90 days (typical partner: {t['p_claims_90d']:.0f})."),
    "near_threshold": lambda v, r, t: (
        f"Amount Rs {r.claim_amount_inr:,.0f} is just under the {LIMIT} auto-approval limit." if v
        else f"Amount Rs {r.claim_amount_inr:,.0f} is not just under the {LIMIT} limit."),
    "near_thr_x_uninsp": lambda v, r, t: (
        f"Just under {LIMIT} and not inspected." if v else "Not a near-limit claim that skipped inspection."),
    "partner_inspected": lambda v, r, t: "Partner inspection signed off." if v else "No partner inspection sign-off.",
    "has_inspector_note": lambda v, r, t: "Inspector wrote a note." if v else "No inspector note.",
    "amount_to_price": lambda v, r, t: f"Claim is {v:.0%} of the product's list price (Rs {r.list_price_inr:,.0f}).",
    "log_amount": lambda v, r, t: f"Claim amount Rs {r.claim_amount_inr:,.0f}.",
    "small_claim": lambda v, r, t: (
        f"Claim under {LIMIT}: auto-approved without inspection since 1 May 2026." if v
        else f"Claim of {LIMIT} or more: still needs inspection."),
    "customer_prior_claims": lambda v, r, t: f"Customer has {int(v)} earlier warranty claim(s).",
    "warranty_used_frac": lambda v, r, t: (
        f"Claim made {min(v, 1):.0%} of the way through the warranty "
        f"({int(r.days_since_purchase)} days after purchase)."),
    "p_near_threshold_share_30d": lambda v, r, t: (
        f"{v:.0%} of the partner's claims in the previous 30 days were just under {LIMIT}."),
    "p_near_threshold_share_90d": lambda v, r, t: (
        f"{v:.0%} of the partner's claims in the previous 90 days were just under {LIMIT}."),
    "p_uninspected_share_30d": lambda v, r, t: (
        f"{v:.0%} of the partner's claims in the previous 30 days skipped inspection."),
    "p_uninspected_share_90d": lambda v, r, t: (
        f"{v:.0%} of the partner's claims in the previous 90 days skipped inspection."),
    "p_amount_ratio_mean_30d": lambda v, r, t: (
        f"Partner's claims in the previous 30 days averaged {v:.0%} of product price."),
    "p_amount_ratio_mean_90d": lambda v, r, t: (
        f"Partner's claims in the previous 90 days averaged {v:.0%} of product price."),
    "claim_description": lambda v, r, t: f"Fault described as '{v}'.",
    "inspector_note": lambda v, r, t: f"Inspector note: '{v}'." if v != "no note" else "No inspector note.",
    "serial_format": lambda v, r, t: f"Serial number typed {SERIAL_STYLE.get(v, v)}.",
    "partner_type": lambda v, r, t: f"Partner is a {str(v).replace('_', ' ')}.",
    "family": lambda v, r, t: f"Product: {v}.",
}


def describe(feature: str, row: pd.Series, typical: dict) -> str:
    """One readable sentence for a feature's value on this claim."""
    template = TEMPLATES.get(feature)
    return template(row[feature], row, typical) if template else f"{feature} = {row[feature]}"


def _source_feature_names(estimator: Pipeline) -> list[str]:
    """The original feature behind each transformed column: numeric names, then one entry per category."""
    encoder = estimator[0].named_transformers_["cat"]
    return NUMERIC + [col for col, cats in zip(CATEGORICAL, encoder.categories_, strict=True) for _ in cats]


def contributions(estimator: Pipeline, feature_means: np.ndarray, row: pd.DataFrame) -> pd.Series:
    """Log-odds push of each original feature for one claim, relative to the average training claim."""
    transformed = estimator[0].transform(row)
    x = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed).ravel()
    push = estimator[-1].coef_[0] * (x - feature_means)
    return pd.Series(push, index=_source_feature_names(estimator)).groupby(level=0).sum()


def reasons(estimator: Pipeline, feature_means: np.ndarray, row: pd.DataFrame, typical: dict,
            n_up: int = 4, n_down: int = 2, min_push: float = 0.1) -> dict:
    """Top factors raising and lowering this claim's risk, each with one readable sentence."""
    push = contributions(estimator, feature_means, row).drop(labels=list(HIDDEN), errors="ignore")
    totals = push.groupby(lambda f: _GROUP_OF.get(f, f)).sum()
    claim = row.iloc[0]

    def factor(group: str, total: float) -> dict:
        members = push[[f for f in GROUPS.get(group, [group]) if f in push.index]]
        # Describe the group by the member that drives it, in the group's direction
        lead = members[np.sign(members) == np.sign(total)].abs().idxmax()
        text = describe(lead, claim, typical)
        if lead in CATEGORICAL and claim[lead] != "no note":
            # A category on its own ("Product: Robot Vacuum.") does not say which way it points
            # The weight is the model's effect with other factors held equal, not a raw fraud rate
            text += f" In past claims this went with {'more' if total > 0 else 'less'} fraud, other factors equal."
        return {"factor": group, "feature": lead, "text": text,
                "weight": round(float(total), 2)}

    up = totals[totals > min_push].sort_values(ascending=False).head(n_up)
    down = totals[totals < -min_push].sort_values().head(n_down)
    return {"raising_risk": [factor(g, t) for g, t in up.items()],
            "lowering_risk": [factor(g, t) for g, t in down.items()]}
