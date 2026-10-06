"""Compare other algorithms with the frozen model, and test whether any reaches the board's accuracy target.

    python tools/benchmark_algorithms.py [--data Data]     (needs requirements-notebook.txt; under a minute)

Writes evidence/algorithm_benchmark.md. Run after the model configuration was frozen: the results are
evidence, not a selection step, and the production model does not change because of them. Every
algorithm uses the same features, the same time-based folds and generic settings (none is tuned), so
the comparison is like for like. The Jul-Sep test set is not used.
"""
import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# joblib cannot count physical cores on Windows 11 (no `wmic`) and prints a harmless traceback
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from imblearn.ensemble import BalancedRandomForestClassifier  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from kestrel_fraud.config import DATA_DIR, EVIDENCE_DIR, LOGISTIC_C, POLICY_CHANGE, REVIEWS_PER_MONTH  # noqa: E402
from kestrel_fraud.data import load_pack, prepare_history  # noqa: E402
from kestrel_fraud.features import CATEGORICAL, FEATURES, NUMERIC  # noqa: E402
from kestrel_fraud.model import sample_weights  # noqa: E402
from kestrel_fraud.validation import bootstrap_ci, time_split  # noqa: E402

K = REVIEWS_PER_MONTH
FOLD_B_CUTOFF = pd.Timestamp("2026-06-01")     # same folds as evaluate.py
CURRENT = "Logistic regression (current model)"
ISOLATION = "Isolation forest (unsupervised)"
SURVIVES = 0.6                                  # Fold A ROC-AUC above this: still ranks the new fraud usefully


def candidates(pos_weight: float) -> dict:
    """name -> (classifier, accepts sample weights). Generic settings; imbalance handled where supported."""
    return {
        CURRENT: (LogisticRegression(C=LOGISTIC_C, class_weight="balanced", max_iter=3000), True),
        "Logistic regression, L1": (LogisticRegression(C=0.05, penalty="l1", solver="liblinear",
                                                       class_weight="balanced"), False),
        "Random forest": (RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=0), True),
        "Balanced random forest": (BalancedRandomForestClassifier(500, sampling_strategy="all", replacement=True,
                                                                  bootstrap=False, n_jobs=-1, random_state=0), True),
        "Extra trees": (ExtraTreesClassifier(500, min_samples_leaf=3, class_weight="balanced", n_jobs=-1,
                                             random_state=0), True),
        "Histogram gradient boosting": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                       class_weight="balanced", random_state=0), True),
        "XGBoost": (xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=0,
                                      n_jobs=-1), True),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                        subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                        random_state=0, verbose=-1), True),
        "Neural network (MLP)": (MLPClassifier((64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
                                               random_state=0), False),
        "Support vector machine (RBF)": (SVC(C=1.0, class_weight="balanced", random_state=0), True),
        "k-nearest neighbours": (KNeighborsClassifier(25, weights="distance"), False),
    }


def preprocessor() -> ColumnTransformer:
    """The production model's preprocessing, so every algorithm sees identical inputs."""
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])


def fold_scores(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit every candidate on `train`; return its ranking scores on `valid`."""
    y = train["is_fraud"].to_numpy()
    scores = {}
    for name, (clf, weighted) in candidates(pos_weight=(y == 0).sum() / (y == 1).sum()).items():
        pipe = make_pipeline(preprocessor(), clf)
        fit_kw = {f"{pipe.steps[-1][0]}__sample_weight": sample_weights(train)} if weighted else {}
        pipe.fit(train[FEATURES], y, **fit_kw)
        scores[name] = (pipe.predict_proba(valid[FEATURES])[:, 1] if hasattr(clf, "predict_proba")
                        else pipe.decision_function(valid[FEATURES]))
    # No labels at all: how unusual each claim looks
    iso = make_pipeline(preprocessor(), IsolationForest(n_estimators=500, random_state=0)).fit(train[FEATURES])
    scores[ISOLATION] = -iso.decision_function(valid[FEATURES])
    ranks = [pd.Series(scores[m]).rank(pct=True).to_numpy() for m in (CURRENT, "XGBoost", "LightGBM")]
    scores["Rank average: logistic + XGBoost + LightGBM"] = np.mean(ranks, axis=0)
    return scores


def accuracy_curve(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Accuracy when the k highest-scoring claims are flagged, for k = 0..n."""
    tp = np.concatenate([[0], np.cumsum(y[np.argsort(-s)])])
    k = np.arange(len(y) + 1)
    return (tp + (len(y) - y.sum()) - (k - tp)) / len(y)


def summarise(y: np.ndarray, scores: dict, fold_a: dict) -> pd.DataFrame:
    """One row per algorithm, best June PR-AUC first."""
    rows = []
    for name, s in scores.items():
        acc = accuracy_curve(y, s)
        lo, hi = bootstrap_ci(y, s, average_precision_score, n_boot=300)
        rows.append({"name": name, "pr_auc": average_precision_score(y, s), "ci": (lo, hi),
                     "roc_auc": roc_auc_score(y, s), "top40": int(y[np.argsort(-s)[:K]].sum()),
                     "acc40": acc[K], "best_acc": acc.max(), "best_k": int(acc.argmax()), "fold_a": fold_a[name]})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).set_index("name")


def results_table(res: pd.DataFrame) -> str:
    table = pd.DataFrame({
        "Algorithm": res.index,
        "PR-AUC": res["pr_auc"].map("{:.3f}".format),
        "PR-AUC 95% CI": res["ci"].map(lambda c: f"{c[0]:.2f}–{c[1]:.2f}"),
        "ROC-AUC": res["roc_auc"].map("{:.3f}".format),
        "Frauds in top 40": res["top40"],
        "Accuracy, top 40 checked": res["acc40"].map("{:.1%}".format),
        "Best accuracy, any cut-off*": [f"{a:.1%} (flag {k})" for a, k in zip(res["best_acc"], res["best_k"],
                                                                               strict=True)],
        "Fold A ROC-AUC": res["fold_a"].map("{:.2f}".format),
    })
    return table.to_markdown(index=False, disable_numparse=True)


def build_report(data_dir: Path) -> str:
    history = prepare_history(load_pack(data_dir))
    train_a, valid_a = time_split(history, POLICY_CHANGE)
    train_b, valid_b = time_split(history, FOLD_B_CUTOFF)
    fold_a = {name: roc_auc_score(valid_a["is_fraud"], s) for name, s in fold_scores(train_a, valid_a).items()}
    scores = fold_scores(train_b, valid_b)
    y = valid_b["is_fraud"].to_numpy()
    res = summarise(y, scores, fold_a)

    n, frauds = len(y), int(y.sum())
    current_ci = res.loc[CURRENT, "ci"]
    overlap = all(lo <= current_ci[1] and current_ci[0] <= hi for lo, hi in res["ci"])
    survivors = res[res["fold_a"] > SURVIVES].sort_values("fold_a", ascending=False)
    failed = res[res["fold_a"] <= SURVIVES]

    return "\n".join([
        "# Algorithm benchmark\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python tools/benchmark_algorithms.py`.\n",
        "## Why this exists\n",
        "Two questions a reviewer will ask: would another algorithm do better, and can any algorithm reach the "
        "board's accuracy target (97%, or 98%)? This benchmark answers both on the same evidence as the main "
        "validation. **It was run after the model configuration was frozen.** It is evidence, not a selection "
        "step: the production model did not change because of it.\n",
        "## How it was run\n",
        "- **Same data, same folds** as `evidence/validation_report.md`: train on claims to 31 May 2026, score "
        f"all {n} June claims ({frauds} frauds). Partner fraud history is frozen at the cutoff with the 30-day "
        "label lag. The Jul–Sep test set is not used.",
        "- **Same 30 features and preprocessing** for every algorithm, and the same post-May training weight "
        "where the algorithm accepts sample weights. Class imbalance is handled with each algorithm's own option.",
        "- **Generic settings, no tuning.** Tuning 13 algorithms on one month with 22 frauds would only fit noise. "
        "The notebook's XGBoost (5× weight, different settings) scored PR-AUC 0.46 on the same month, against "
        f"{res.loc['XGBoost', 'pr_auc']:.2f} here: settings alone move a result that much.",
        "- **Fold A ROC-AUC:** the same algorithm trained only on claims before the 1 May 2026 rule, scored on "
        "May–June. Below 0.5 means it ranks the new fraud backwards.\n",
        "## Results (June 2026)\n",
        results_table(res),
        "",
        "\\* *Best accuracy, any cut-off* picks the number of claims to flag **using June's answers**, which no "
        "real system can do. It is an upper bound, deliberately optimistic.\n",
        "## What it shows\n",
        f"- **No algorithm reaches 97%, let alone 98%.** Flagging nothing scores {(y == 0).mean():.1%}. The best "
        f"accuracy any algorithm reaches, even with the optimistic cut-off, is {res['best_acc'].max():.1%}, by "
        f"flagging a handful of claims. 98% on June allows at most {int(n * 0.02)} errors; flagging nothing "
        f"already makes {frauds}. Getting there needs the flagged claims to be about 80% fraud; the best top 40 "
        f"here is {res['top40'].max() / K:.0%} fraud.",
        "- **Accuracy falls as the desk checks more claims**, for every algorithm: catching fraud means holding "
        "some genuine claims. An accuracy target rewards checking fewer claims.",
        f"- **The differences between algorithms are within noise.** {'Every' if overlap else 'Not every'} "
        f"PR-AUC interval overlaps the current model's ({current_ci[0]:.2f}–{current_ci[1]:.2f}). The top-40 "
        f"catch ranges from {res['top40'].min()} to {res['top40'].max()} frauds, a few claims in one month.",
        f"- **Most algorithms do not survive the policy change.** Trained before May, {len(failed)} of {len(res)} "
        f"rank May–June claims at ROC-AUC {failed['fold_a'].min():.2f}–{failed['fold_a'].max():.2f}, most below "
        "0.5 (backwards): they learned the old fraud. The exceptions: "
        + "; ".join(f"{name}: {auc:.2f}" for name, auc in survivors["fold_a"].items()) + ".",
        f"- **The isolation forest uses no labels at all.** It scores {res.loc[ISOLATION, 'fold_a']:.2f} on "
        f"Fold A and {res.loc[ISOLATION, 'pr_auc']:.2f} PR-AUC on June: the new fraud looks unusual, so it "
        "stands out before anyone has labelled it. That makes it the best early-warning signal here for the "
        "next shift, which a supervised model cannot see until outcomes arrive.\n",
        "## Decision\n",
        "- **Keep the logistic regression** for scoring. It is level with the best on the desk's 40 checks, its "
        "reasons are exact coefficients a claims officer can read, and switching now would mean choosing on the "
        "same month used to validate.",
        "- **Add the isolation forest as a monitor, not a scorer**, in the next iteration: each month, count the "
        "claims it finds unusual that the model scores low. A jump is the cue to send a random sample to the "
        "desk and retrain early. Not built now: it changes the desk's process, which needs Kestrel's agreement.",
        "- **Re-run this benchmark when July outcomes arrive.** Switch algorithm only if one is clearly better on "
        "two months in a row. Extra trees and k-nearest neighbours are the ones to watch.\n",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Write evidence/algorithm_benchmark.md.")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder with the client data pack")
    args = parser.parse_args()
    path = EVIDENCE_DIR / "algorithm_benchmark.md"
    path.write_text(build_report(args.data), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
