"""Scoring service: one endpoint that scores a single claim, and one screen that calls it.

    uvicorn kestrel_fraud.service:app --port 8000      then open http://localhost:8000

POST /score                  one claim as JSON -> risk score, recommendation, reasons
GET  /example?kind=high|low  a real test-period claim, to try the screen
GET  /health                 whether a trained model is loaded
No external API or key is used. Without a trained model the service still starts, and every
scoring call returns 503 with instructions.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from . import __version__
from .config import ARTIFACT_PATH, LOCAL_TZ
from .data import CLAIM_COLUMNS, clean_claims
from .explain import reasons
from .features import build_features
from .model import BundleError, ModelBundle
from .schemas import Claim, ScoreResponse

log = logging.getLogger("uvicorn.error")   # uvicorn's configured logger, so startup messages appear in its output
STATIC_DIR = Path(__file__).parent / "static"
# Model facts worth showing per score; the feature list and versions stay in the bundle for load checks
MODEL_SUMMARY = ["trained_on_claims", "trained_on_frauds", "trained_to"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. A missing model is not fatal: scoring calls then return 503."""
    try:
        app.state.bundle, app.state.bundle_error = ModelBundle.load(ARTIFACT_PATH), None
        log.info("Loaded model from %s", ARTIFACT_PATH)
    except BundleError as exc:
        app.state.bundle, app.state.bundle_error = None, str(exc)
        log.warning("Starting without a model: %s", exc)
    yield


app = FastAPI(title="Kestrel warranty-claim fraud score", version=__version__, lifespan=lifespan)


def get_bundle(request: Request) -> ModelBundle:
    """Dependency: the loaded model, or 503 with the reason it is missing."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        detail = getattr(request.app.state, "bundle_error", None) or "Model not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return bundle


def _to_local_naive(ts: pd.Series) -> pd.Series:
    """Training timestamps are naive IST. Convert timezone-aware input to IST before dropping the zone."""
    return ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None) if ts.dt.tz is not None else ts


def _recommendation(risk: float, threshold: float) -> str:
    if risk >= threshold:
        return "Send to the investigation desk before paying."
    if risk >= threshold / 3:
        return "Borderline: pay, but check if the desk has spare capacity this month."
    return "Low risk: pay as normal."


def score_claim(claim: Claim, bundle: ModelBundle) -> dict:
    """Score one claim against the stored partner history; 422 for a partner or SKU the model has not seen."""
    if not (bundle.partners["partner_id"] == claim.partner_id).any():
        raise HTTPException(status_code=422, detail=f"Unknown partner_id '{claim.partner_id}'.")
    if not (bundle.products["sku"] == claim.sku).any():
        raise HTTPException(status_code=422, detail=f"Unknown sku '{claim.sku}'.")

    record = pd.DataFrame([claim.model_dump()])
    record["submitted_at"] = _to_local_naive(pd.to_datetime(record["submitted_at"]))
    record = clean_claims(record, bundle.partners, bundle.products).assign(dataset="new", is_fraud=np.nan)

    # Partner features depend only on the same partner's claims. A stored copy of this claim is
    # dropped so that re-scoring a known claim reproduces its batch score.
    history = bundle.history
    past = history[(history["partner_id"] == claim.partner_id) & (history["claim_id"] != claim.claim_id)]
    features = build_features(pd.concat([past, record], ignore_index=True))
    row = features[features["claim_id"] == claim.claim_id]

    risk = float(bundle.score(row)[0])
    percentile = np.searchsorted(bundle.reference_scores, risk, side="right") / len(bundle.reference_scores)
    return {
        "claim_id": claim.claim_id,
        "risk_score": round(risk, 4),
        "riskier_than_pct_of_recent_claims": round(float(percentile) * 100, 1),
        "review_threshold": round(bundle.review_threshold, 4),
        "recommendation": _recommendation(risk, bundle.review_threshold),
        "reasons": reasons(bundle.estimator, bundle.feature_means, row, bundle.typical),
        "note": ("The risk score ranks claims; it is not an exact probability. The desk can review about 40 claims "
                 "a month: the threshold is the score of the 40th riskiest claim per month in Jul-Sep 2026."),
        "model": {"type": "logistic regression",
                  **{k: bundle.metadata[k] for k in MODEL_SUMMARY if k in bundle.metadata}},
    }


Bundle = Annotated[ModelBundle, Depends(get_bundle)]


@app.post("/score", response_model=ScoreResponse)
def score(claim: Claim, bundle: Bundle):
    """Score one claim: risk score, how it ranks against recent claims, a recommendation and reasons."""
    return score_claim(claim, bundle)


@app.get("/example", response_model=Claim)
def example(bundle: Bundle, kind: Annotated[Literal["high", "low"], Query()] = "high"):
    """A real test-period claim in request format: the riskiest one, or a typical one."""
    row = bundle.history.loc[bundle.history["claim_id"] == bundle.examples[kind], CLAIM_COLUMNS].iloc[0]
    claim = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}   # numpy -> Python types
    claim["submitted_at"] = pd.Timestamp(claim["submitted_at"]).to_pydatetime()
    # Undo cleaning so the claim looks as a partner would submit it
    for col in ["photo_attached", "partner_inspected"]:
        claim[col] = "Y" if claim[col] == 1 else "N"
    claim["inspector_note"] = None if claim["inspector_note"] == "no note" else claim["inspector_note"]
    return Claim(**claim)


@app.get("/health")
def health(request: Request):
    """Whether a trained model is loaded, and if not, what to do."""
    bundle = getattr(request.app.state, "bundle", None)
    return {"status": "ok" if bundle else "model_missing",
            "detail": None if bundle else getattr(request.app.state, "bundle_error", None)}


@app.get("/", include_in_schema=False)
def screen():
    """The one-page screen for claims staff."""
    return FileResponse(STATIC_DIR / "index.html")
