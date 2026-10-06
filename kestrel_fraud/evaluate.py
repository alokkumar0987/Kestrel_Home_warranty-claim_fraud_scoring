"""Evidence that the model works, and how often it does not.

    python -m kestrel_fraud.evaluate [--data Data] [--service-sample 200]

Re-runs the time-based validation with the packaged code and writes evidence/validation_report.md.
Nothing here uses Jul-Sep test outcomes (we do not have them).
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import DATA_DIR, EVIDENCE_DIR, GOODWILL_COST_INR, OUTPUT_DIR, POLICY_CHANGE, REVIEWS_PER_MONTH
from .data import find_file, load_pack, prepare_history
from .model import fit_estimator, risk_scores, weight_inflation
from .validation import bootstrap_ci, review_top_k, time_split, wilson_interval

log = logging.getLogger("kestrel_fraud.evaluate")
K = REVIEWS_PER_MONTH
FOLD_A_CUTOFF = POLICY_CHANGE                    # train before the policy change, validate May-June
FOLD_B_CUTOFF = pd.Timestamp("2026-06-01")       # main fold: May outcomes known, validate June


def score_fold(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Fit on `train` and return risk scores for `valid`."""
    return risk_scores(fit_estimator(train), valid, weight_inflation(train))


def comparison_table(valid: pd.DataFrame, methods: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per ranking method: what the desk would catch reviewing its top 40."""
    rows = []
    for name, scores in methods.items():
        r = review_top_k(valid, scores)
        rows.append({"Method": name, "Fraud caught (top 40)": f"{r['caught']} / {r['frauds']}",
                     "Precision@40": f"{r['caught'] / K:.0%}", "Rs fraud caught": f"{r['rupees']:,.0f}",
                     "Rs / review": f"{r['rupees_per_review']:,.0f}", "Genuine customers held": K - r["caught"],
                     "PR-AUC": f"{average_precision_score(valid['is_fraud'], scores):.3f}",
                     "ROC-AUC": f"{roc_auc_score(valid['is_fraud'], scores):.3f}"})
    return pd.DataFrame(rows)


def case_profile(claims: pd.DataFrame) -> pd.Series:
    """What a group of claims looks like, for comparing caught, missed and false-alarm cases."""
    return pd.Series({
        "claims": len(claims),
        "partner in first year": f"{claims['partner_is_new'].mean():.0%}",
        "partner has visible confirmed fraud (30-90 days)": f"{(claims['p_frauds_90d'] > 0).mean():.0%}",
        "claim under Rs 2,000": f"{claims['small_claim'].mean():.0%}",
        "not inspected": f"{(1 - claims['partner_inspected']).mean():.0%}",
        "median claim (Rs)": f"{claims['claim_amount_inr'].median():,.0f}",
    })


# Test months and how stale partner fraud history is when they are scored (export frozen at 30 June).
# Simulated on June by freezing outcomes 0, 1 or 2 months before the 1 June cutoff.
STALENESS_SCENARIOS = {"July (history current)": None,
                       "August (1 month stale)": "2026-05-01",
                       "September (2 months stale)": "2026-04-01"}


def expected_score(history: pd.DataFrame, estimator: Pipeline, inflation: float, valid: pd.DataFrame) -> str:
    """Estimate the test-set score from June: bootstrap interval plus a staleness backtest.

    The Fold B model scores the same June claims with partner fraud history frozen earlier, as
    August and September claims will be. Pooling the three scorings mimics the pooled Jul-Sep score.
    """
    rows, ys, scores = [], [], []
    for month, freeze in STALENESS_SCENARIOS.items():
        stale = time_split(history, FOLD_B_CUTOFF, freeze=freeze)[1].set_index("claim_id").loc[valid["claim_id"]]
        s = risk_scores(estimator, stale, inflation)
        y = stale["is_fraud"].to_numpy()
        lo, hi = bootstrap_ci(y, s, average_precision_score)
        rows.append({"Scored as": month, "PR-AUC": f"{average_precision_score(y, s):.3f}",
                     "PR-AUC 95% CI": f"{lo:.2f}–{hi:.2f}", "ROC-AUC": f"{roc_auc_score(y, s):.3f}",
                     "Fraud caught (top 40)": review_top_k(stale.reset_index(), s)["caught"],
                     "Claims with visible partner fraud": f"{(stale['p_frauds_90d'] > 0).mean():.1%}"})
        ys.append(y)
        scores.append(s)
    y, s = np.concatenate(ys), np.concatenate(scores)
    lo, hi = bootstrap_ci(y, s, average_precision_score, groups=np.tile(np.arange(len(valid)), len(ys)))
    return "\n".join([
        "## Expected score on the test set (Jul–Sep 2026)\n",
        "The test outcomes are hidden, so the expected score is estimated from June. Two measurements replace "
        "judgement: a **bootstrap** (June claims resampled 1,000 times) for how much one month of 22 frauds can "
        "move, and a **staleness backtest**. The test set's partner fraud history is frozen at 30 June, so August "
        "and September claims are scored without the latest outcomes. The backtest scores the same June claims with "
        "the same model, with history frozen 1 and 2 months earlier.\n",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        f"- **Pooled over the three scenarios** (assuming the hidden score is one PR-AUC over all Jul–Sep claims): "
        f"PR-AUC "
        f"**{average_precision_score(y, s):.3f}** (95% CI {lo:.2f}–{hi:.2f}), ROC-AUC {roc_auc_score(y, s):.3f}. "
        f"A random ranking scores PR-AUC = fraud rate = {y.mean():.3f}.",
        "- One month of staleness costs the whole drop: it hides May's outcomes, the first from the post-May "
        "fraudsters. A second month only hides April's, from the pre-May pattern, which barely helps rank June. "
        "The September scenario is harsher than the real test (it loses all post-May outcomes; the real test keeps "
        "May–June), so it is a pessimistic case.",
        "- **Not measured, so not in the number:** first-year and unseen partners rising through Jul–Sep, and a "
        "further change in the fraud pattern. Both push the real score down; Fold A shows how far a pattern change "
        "can go.\n",
    ])


def service_check(data_dir: Path, n: int) -> str:
    """Score test claims one at a time through the API and compare with predictions.csv."""
    # Imported here so the report can be built without loading the web app unless the check runs
    from fastapi.testclient import TestClient

    from .service import app

    predictions_path = OUTPUT_DIR / "predictions.csv"
    if not predictions_path.exists():
        return "Skipped: run `python -m kestrel_fraud.train` first."
    with TestClient(app) as client:
        if client.get("/health").json()["status"] != "ok":
            return "Skipped: model not trained."
        test = pd.read_csv(find_file(data_dir, "test_unlabelled.csv"))
        expected = pd.read_csv(predictions_path).set_index("claim_id")["score"]
        sample = test.sample(min(n, len(test)), random_state=7)
        diffs, millis = [], []
        for _, row in sample.iterrows():
            body = {k: (None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in row.items()}
            start = time.perf_counter()
            got = client.post("/score", json=body).json()["risk_score"]
            millis.append((time.perf_counter() - start) * 1000)
            diffs.append(abs(got - expected[row["claim_id"]]))
    return (f"{len(sample)} test claims scored one at a time through `POST /score`; largest difference from "
            f"`predictions.csv` = {max(diffs):.1e} (the API rounds to 4 decimals). "
            f"Median latency {np.median(millis):.0f} ms, max {max(millis):.0f} ms on a laptop.")


def build_report(data_dir: Path, service_sample: int) -> str:
    """Run both validation folds and the service check, and return the report as Markdown."""
    history = prepare_history(load_pack(data_dir))

    # Fold A: trained before the policy change, validated after it
    train_a, valid_a = time_split(history, FOLD_A_CUTOFF)
    auc_a = roc_auc_score(valid_a["is_fraud"], score_fold(train_a, valid_a))

    # Fold B (main): May outcomes known, validated on June
    train_b, valid_b = time_split(history, FOLD_B_CUTOFF)
    estimator_b, inflation_b = fit_estimator(train_b), weight_inflation(train_b)
    scores = risk_scores(estimator_b, valid_b, inflation_b)
    top = review_top_k(valid_b, scores)
    picked, y = top["picked"], valid_b["is_fraud"].to_numpy()
    fraud_rs = valid_b.loc[valid_b["is_fraud"] == 1, "claim_amount_inr"].sum()
    missed = valid_b[(valid_b["is_fraud"] == 1) & ~valid_b["claim_id"].isin(picked["claim_id"])]
    false_alarms, hits = picked[picked["is_fraud"] == 0], picked[picked["is_fraud"] == 1]
    flagged = valid_b["claim_id"].isin(picked["claim_id"]).to_numpy()
    methods = {
        "Model (logistic regression)": scores,
        "Rule: partners with recent confirmed fraud":
            (valid_b["p_frauds_90d"] + valid_b["p_past_fraud_rate"]).to_numpy(),
        "Rule: newest partners first": (valid_b["partner_is_new"] - valid_b["claim_amount_inr"] / 1e6).to_numpy(),
        "Rule: biggest claims first": valid_b["claim_amount_inr"].to_numpy(),
    }
    caught, frauds = top["caught"], top["frauds"]

    return "\n".join([
        f"# Validation report\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by `python -m kestrel_fraud.evaluate`.\n",
        "## How it was checked\n",
        "- **Time-based validation only.** The test claims (Jul–Sep 2026) all come after the 1 May 2026 rule that "
        "auto-approves claims under Rs 2,000, and fraud behaves differently after it. A random split would mix the two "
        "periods and overstate performance.",
        "- **Main check (Fold B):** train on claims to 31 May 2026, score every June 2026 claim, review the top 40 "
        "(the investigation desk's monthly capacity). Partner fraud history is frozen at 31 May and uses only outcomes "
        "at least 30 days old, as it will be for the real test months.",
        f"- **Sample size:** June has {len(valid_b)} labelled claims and **{frauds} frauds** (Rs {fraud_rs:,.0f}). "
        "All June numbers below are observed counts for one month, with wide uncertainty.\n",
        "## Result (June 2026, top 40 reviewed)\n",
        comparison_table(valid_b, methods).to_markdown(index=False, disable_numparse=True),
        "",
        f"- Precision (reviewed claims that were fraud): {wilson_interval(caught, K)}",
        f"- Recall (June frauds caught): {wilson_interval(caught, frauds)}",
        f"- Rupees: Rs {top['rupees']:,.0f} of Rs {fraud_rs:,.0f} ({top['rupees'] / fraud_rs:.0%}), "
        f"about Rs {top['rupees_per_review']:,.0f} per review.",
        f"- **How often it is wrong:** {K - caught} of the 40 reviewed claims ({(K - caught) / K:.0%}) were genuine "
        f"(each costs about Rs {GOODWILL_COST_INR} in goodwill), and {len(missed)} of {frauds} frauds "
        f"({len(missed) / frauds:.0%}) were not reviewed.",
        f"- Accuracy, for reference: flagging nothing = {(y == 0).mean():.1%}; reviewing the top 40 = "
        f"{(flagged == y).mean():.1%}; thresholding the score at 0.5 = {((scores > 0.5) == y).mean():.1%}. "
        "Accuracy rewards flagging nothing, so it is not used to judge the model.\n",
        "## The kinds of case it gets wrong\n",
        pd.DataFrame({"Caught frauds": case_profile(hits), "Missed frauds": case_profile(missed),
                      "False alarms (genuine, reviewed)": case_profile(false_alarms)}).to_markdown(),
        "",
        "- **Missed frauds** come from partners with no confirmed fraud yet visible, or are the old pattern (a large "
        f"claim). The largest missed claim was Rs {missed['claim_amount_inr'].max():,.0f}, "
        f"{missed['claim_amount_inr'].max() / fraud_rs:.0%} of June's fraud value on its own.",
        f"- **False alarms** look like the frauds: first-year partner, under Rs 2,000, not inspected. Only "
        f"{(false_alarms['p_frauds_90d'] > 0).mean():.0%} come from partners with visible confirmed fraud. On paper "
        "the model cannot tell a genuine small, uninspected claim from a new outlet apart from a fraudulent one; that "
        "is what the desk review is for. Every reviewed claim in June came from a first-year partner.\n",
        expected_score(history, estimator_b, inflation_b, valid_b),
        "## The pattern changed once already (Fold A)\n",
        f"Trained only on claims before 1 May 2026, the same model ranks May–June claims **backwards**: ROC-AUC "
        f"{auc_a:.3f} (reversed, {1 - auc_a:.3f}). Before May, fraud went with large claims from long-standing "
        "partners; after May, with small claims from first-year partners. Technical causes (dates, label encoding, "
        "probability column, AUC computation, categories, point-in-time history) were checked in the notebook "
        "(section 7.2). This is the biggest risk for the live model: if fraud shifts again, the ranking can fail until "
        "it is retrained on new outcomes.\n",
        "## Service check\n",
        service_check(data_dir, service_sample) + "\n",
        "Automated tests: `python -m pytest` (unit tests on cleaning, point-in-time features and score correction "
        "run without the data pack; API tests also check that the API reproduces `predictions.csv`, reasons are "
        "present and consistent, bad input is rejected with a clear message, and the service fails politely without "
        "a model).\n",
    ])


def main() -> None:
    """CLI entry point: write evidence/validation_report.md."""
    parser = argparse.ArgumentParser(description="Write the validation report.")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder with the client data pack")
    parser.add_argument("--service-sample", type=int, default=200, help="test claims to send through the API")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = EVIDENCE_DIR / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(args.data, args.service_sample), encoding="utf-8")
    log.info("Saved %s", path)


if __name__ == "__main__":
    main()
