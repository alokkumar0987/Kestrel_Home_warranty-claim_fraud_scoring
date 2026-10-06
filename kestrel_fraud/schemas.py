"""Request and response models for the scoring API (also documented at /docs)."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One warranty claim, with the same fields as test_unlabelled.csv."""
    claim_id: str = Field("NEW-CLAIM", description="Claim number")
    submitted_at: datetime = Field(..., description="Submission time. Without a timezone it is read as IST.")
    partner_id: str = Field(..., examples=["SP3004"])
    sku: str = Field(..., examples=["KH-AF-02"])
    product_serial: str = Field(..., description="Serial as typed by the partner")
    days_since_purchase: int = Field(..., ge=0)
    claim_amount_inr: float = Field(..., gt=0)
    photo_attached: Literal["Y", "N"]
    partner_inspected: Literal["Y", "N"]
    claim_description: str = Field(..., description="Fault description, e.g. 'water leaking'")
    inspector_note: str | None = None
    customer_prior_claims: int = Field(0, ge=0, description="Customer's earlier warranty claims")
    source: Literal["crm", "legacy_zoho"] = "crm"


class Reason(BaseModel):
    factor: str = Field(..., description="Group of related features, e.g. 'partner fraud history'")
    feature: str = Field(..., description="The feature that drives this factor")
    text: str = Field(..., description="One sentence for a claims officer")
    weight: float = Field(..., description="Push on the score in log-odds; positive raises risk")


class Reasons(BaseModel):
    raising_risk: list[Reason]
    lowering_risk: list[Reason]


class ModelInfo(BaseModel):
    type: str
    trained_on_claims: int
    trained_on_frauds: int
    trained_to: str = Field(..., description="Date of the latest labelled claim used in training")


class ScoreResponse(BaseModel):
    claim_id: str
    risk_score: float = Field(..., description="Higher = more likely fraudulent. A ranking, not an exact probability.")
    riskier_than_pct_of_recent_claims: float
    review_threshold: float = Field(..., description="Score of the 40th riskiest claim per month in the scored period")
    recommendation: str
    reasons: Reasons
    note: str
    model: ModelInfo
