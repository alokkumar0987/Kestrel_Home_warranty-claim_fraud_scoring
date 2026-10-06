"""Train the final model on every labelled claim and score the test claims.

    python -m kestrel_fraud.train [--data Data]

Writes:
    artifacts/model.joblib                 model bundle used by the service
    outputs/predictions.csv                one score per test claim, in sample_submission.csv shape
    outputs/review_list_jul_sep_2026.csv   top claims per month for the investigation desk
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_PATH, DATA_DIR, OUTPUT_DIR, REVIEWS_PER_MONTH
from .data import DataPack, load_pack, prepare_history
from .features import build_features
from .model import ModelBundle, fit_estimator, risk_scores, weight_inflation

log = logging.getLogger("kestrel_fraud.train")

REVIEW_COLUMNS = [
    "claim_id", "submitted_at", "month", "score", "rank_in_month", "claim_amount_inr",
    "partner_id", "partner_is_new", "partner_tenure_days", "partner_type", "city",
    # risk features a reviewer can check
    "p_frauds_90d", "p_past_fraud_rate", "p_claims_30d", "p_claims_90d", "new_x_post",
    "near_threshold", "near_thr_x_uninsp", "partner_inspected", "amount_to_price", "customer_prior_claims",
]


def train_bundle(pack: DataPack) -> tuple[ModelBundle, pd.DataFrame]:
    """Fit on all labelled claims; return the bundle and the scored test claims."""
    history = prepare_history(pack)
    features = build_features(history)
    train = features[(features["dataset"] == "train") & features["is_fraud"].notna()].astype({"is_fraud": int})
    test = features[features["dataset"] == "test"].copy()

    estimator, inflation = fit_estimator(train), weight_inflation(train)
    test["score"] = risk_scores(estimator, test, inflation)
    test["month"] = test["submitted_at"].dt.strftime("%Y-%m")
    ranked = test.sort_values("score", ascending=False)["claim_id"].to_numpy()

    bundle = ModelBundle(
        estimator=estimator, inflation=inflation,
        # The score of the 40th riskiest claim per month, averaged over the scored months
        review_threshold=float(test.groupby("month")["score"]
                               .apply(lambda s: s.nlargest(REVIEWS_PER_MONTH).min()).mean()),
        reference_scores=np.sort(test["score"].to_numpy()),
        # Baseline for reasons: the average training claim, in the estimator's transformed space
        feature_means=np.asarray(estimator[0].transform(train).mean(axis=0)).ravel(),
        typical={col: float(train[col].median()) for col in ["p_claims_30d", "p_claims_90d"]},
        examples={"high": ranked[0], "low": ranked[len(ranked) // 2]},   # riskiest and a typical claim
        history=history, partners=pack.partners, products=pack.products,
        metadata={"trained_on_claims": len(train), "trained_on_frauds": int(train["is_fraud"].sum()),
                  "trained_to": str(train["submitted_at"].max().date())},
    )
    return bundle, test


def write_predictions(test: pd.DataFrame, sample_submission: pd.DataFrame, path: Path) -> None:
    """One row per test claim, same order and columns as sample_submission.csv."""
    predictions = sample_submission[["claim_id"]].merge(test[["claim_id", "score"]], on="claim_id", how="left")
    problems = {
        "claim ids or order differ from sample_submission":
            not predictions["claim_id"].equals(sample_submission["claim_id"]),
        "duplicate claim ids": not predictions["claim_id"].is_unique,
        "missing scores": predictions["score"].isna().any(),
        "scores outside [0, 1]": not predictions["score"].between(0, 1).all(),
    }
    if any(problems.values()):
        raise ValueError(f"Predictions failed checks: {[name for name, bad in problems.items() if bad]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False, float_format="%.10g")


def write_review_list(test: pd.DataFrame, path: Path) -> None:
    """Top claims per month for the investigation desk, with the facts a reviewer would check."""
    review = (test.sort_values("score", ascending=False).groupby("month").head(REVIEWS_PER_MONTH)
                  .sort_values(["month", "score"], ascending=[True, False]))
    review["rank_in_month"] = review.groupby("month").cumcount() + 1
    review[REVIEW_COLUMNS].to_csv(path, index=False)


def main() -> None:
    """CLI entry point: train, save the model bundle, write predictions and the review list."""
    parser = argparse.ArgumentParser(description="Train the fraud model and write predictions.")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder with the client data pack")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pack = load_pack(args.data)
    bundle, test = train_bundle(pack)
    bundle.save(ARTIFACT_PATH)
    write_predictions(test, pack.sample_submission, OUTPUT_DIR / "predictions.csv")
    write_review_list(test, OUTPUT_DIR / "review_list_jul_sep_2026.csv")

    log.info("Trained on %s claims (%s fraud) to %s", bundle.metadata["trained_on_claims"],
             bundle.metadata["trained_on_frauds"], bundle.metadata["trained_to"])
    log.info("Saved model to %s", ARTIFACT_PATH)
    log.info("Saved %s predictions and the review list to %s", len(test), OUTPUT_DIR)
    log.info("Review threshold (40th claim a month): %.3f", bundle.review_threshold)


if __name__ == "__main__":
    main()
