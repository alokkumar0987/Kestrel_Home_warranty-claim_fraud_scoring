# Kestrel Home: warranty-claim fraud scoring

Scores each warranty claim for fraud risk so Kestrel's investigation desk (about 40 reviews a month) looks at the riskiest claims before they are paid. Includes a small web service with one screen, the analysis notebooks, and evidence of how well it works.

**No paid API or key is needed.** The model is a logistic regression that runs locally.

## Results at a glance

| | |
|---|---|
| **What it does** | Ranks every claim by fraud risk so the desk checks the riskiest 40 a month |
| **June 2026 trial** (trained only on earlier claims) | 14 of 22 frauds in the top 40; Rs 19,765 caught, about Rs 494 per check. The 40 largest claims would catch 1 |
| **Accuracy** | 95.2% while catching 14 frauds; flagging nothing scores 96.9%. No algorithm tested reaches 97% (`evidence/algorithm_benchmark.md`) |
| **Expected test score** | PR-AUC about 0.27 (range 0.20–0.35); random is about 0.03 |
| **Model** | Logistic regression on 30 point-in-time features; exact, readable reasons; no paid API |

## Run it on a clean machine

Needs **Python 3.11–3.13** and the client data pack. Commands are for Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`.

```powershell
# 1. Get the code and create an environment
git clone https://github.com/alokkumar0987/Kestrel_Home_warranty-claim_fraud_scoring.git kestrel && cd kestrel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Put the client data pack in a folder called Data/ (it is not in the repo, see "Data" below)
#    Data/ must contain the *train.csv, *test_unlabelled.csv, *partners.csv, *products.csv, *sample_submission.csv files

# 3. Train the model and write predictions (about 10 seconds)
python -m kestrel_fraud.train --data Data

# 4. Start the service and open the screen
uvicorn kestrel_fraud.service:app --port 8000
# then open http://localhost:8000
```

On the screen, press **Load a high-risk example** (or **Load a typical example**), then **Check risk**.

If step 3 is skipped, the service still starts. `/health` reports `model_missing`, and the screen shows a message saying how to train the model.

## What you get

| Path | What it is |
|---|---|
| `outputs/predictions.csv` | One score per test claim, in `sample_submission.csv` shape. Higher = more likely fraudulent. Written by step 3 |
| `outputs/review_list_jul_sep_2026.csv` | Top 40 claims per month, Jul–Sep 2026, for the investigation desk. Written by step 3 |
| `evidence/validation_report.md` | How it was validated, results, error rates, the kinds of case it gets wrong |
| `evidence/algorithm_benchmark.md` | 12 other algorithms on the same folds, and whether any reaches the accuracy target |
| `memo_to_ritu.md` | One-page decision memo for the client |
| `submission-form.md` | Answers to the submission form |
| `notebooks/` | Full analysis (outputs cleared, see "Data") |

`outputs/` is not committed (policy §10). The submitted `predictions.csv` is at the Drive link in `submission-form.md`. The review list and the memo with the outlets named are for Kestrel only; step 3 regenerates the review list.

## The API

`POST /score` takes one claim as JSON, with the same fields as `test_unlabelled.csv`:

```json
{
  "claim_id": "NEW-CLAIM-001", "submitted_at": "2026-08-28T10:15:00", "partner_id": "SP3004", "sku": "KH-RV-02",
  "product_serial": "KH123456789", "days_since_purchase": 120, "claim_amount_inr": 1890,
  "photo_attached": "Y", "partner_inspected": "N", "claim_description": "water leaking",
  "inspector_note": null, "customer_prior_claims": 0, "source": "crm"
}
```

It returns the risk score, how it compares with recent claims, a recommendation (send to the desk / borderline / pay), and plain-language reasons that raise or lower the risk. Unknown partners or SKUs and malformed fields get a `422` with a message naming the field.

Other endpoints: `GET /example?kind=high|low` (a real test-period claim to try), `GET /health`, `GET /docs` (interactive API docs).

## Check that it works

```powershell
python -m pytest                                 # 38 tests; the 28 unit tests need no data pack
python -m kestrel_fraud.evaluate --data Data     # rebuilds evidence/validation_report.md (about 35 seconds)
python tools/benchmark_algorithms.py             # rebuilds evidence/algorithm_benchmark.md (needs requirements-notebook.txt)
```

- **Unit tests** (synthetic data, run anywhere): cleaning and clear errors for bad records, the predictions-file checks, the 30-day label lag, the staleness freeze and bootstrap, that later outcomes never leak into earlier claims' features, the Rs 1,800–2,000 band, score correction keeps the ranking.
- **API tests** (need the data pack and step 3; skipped otherwise): the API reproduces `predictions.csv`, reasons never contradict each other, timezone handling, bad input gets a clear `422`, and no model gives a polite `503`.

Configuration: `KESTREL_DATA_DIR` and `KESTREL_ARTIFACT` override the data folder and model file. All business rules and modelling choices live in `kestrel_fraud/config.py`.

## How the model works (short)

- **Validation is time-based.** The test claims (Jul–Sep 2026) all come after the 1 May 2026 rule that auto-approves claims under Rs 2,000. Fraud behaves differently after that rule, so the model is validated on June 2026 after training on earlier claims.
- **30 features**: claim (amount vs product price, just under the Rs 2,000 limit, inspection), partner profile (tenure), partner behaviour in the last 30/90 days, and the partner's confirmed fraud history. Fraud history only uses outcomes at least 30 days old, so training sees history as stale as it will be in use.
- **Logistic regression**, post-May claims weighted 3×. After the configuration was frozen, 12 other algorithms (tree ensembles, gradient boosting, neural network, SVM, k-NN, an unsupervised isolation forest) were run on the same folds: none is better beyond noise, and none reaches 97% accuracy even with a cut-off chosen on the answers (`evidence/algorithm_benchmark.md`).
- **Score**: model output with the training-weight inflation removed by formula (ranking unchanged). Treat it as a risk ranking, not an exact probability.

Details and every number: `notebooks/Kestrel_Home.ipynb`, `evidence/validation_report.md` and `evidence/algorithm_benchmark.md`.

## Data

The client's ops policy (§10) says customer and operational data must not be published or uploaded to public repositories. So this repo contains **no client data**, and no partner or claim IDs in its documents: `Data/`, `artifacts/` (the model file embeds claim history) and `outputs/` are git-ignored, and the notebooks are committed with outputs cleared. To reproduce, put the data pack in `Data/` and follow the steps above. After editing a notebook locally, run `python tools/export_notebooks.py` to refresh the output-free copies.

## Architecture

End to end, from the client's data pack to the investigation desk and back. `config.py` sits under every step: policy dates, the Rs 2,000 limit, 40 reviews a month, costs, paths and modelling choices.

```
  CLIENT DATA PACK   Data/  (never committed: ops policy §10)
  train.csv · test_unlabelled.csv · partners.csv · products.csv · sample_submission.csv
                                          │
                                          ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ 1. CLEAN                                                           data.py      │
 │    find files by suffix → drop re-submitted claims (keep the first) →           │
 │    normalise serials and Y/N → strip text injected into descriptions →          │
 │    reject unknown partners/SKUs → join partners.csv + products.csv              │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                          │  cleaned claim history (train + test)
                                          ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ 2. FEATURES (point-in-time)                                    features.py      │
 │    claim: amount vs price, Rs 1,800–2,000 band, inspection, warranty used       │
 │    partner: tenure · claims in the 30/90 days before · confirmed fraud from     │
 │    outcomes at least 30 days old (no later outcome can leak into a claim)       │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
 ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
 │ 3a. TRAIN               │ │ 3b. VALIDATE            │ │ 3c. BENCHMARK           │
 │ train.py + model.py     │ │ evaluate.py             │ │ tools/                  │
 │ logistic regression,    │ │ + validation.py         │ │ benchmark_algorithms.py │
 │ post-May claims ×3,     │ │ train to 31 May, score  │ │ 12 other algorithms on  │
 │ weight inflation        │ │ June, top 40, bootstrap,│ │ the same folds; after   │
 │ removed (risk_scores)   │ │ staleness backtest      │ │ the freeze, evidence    │
 └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
              │                           │                           │
              ▼                           ▼                           ▼
 artifacts/model.joblib       evidence/                    evidence/
 outputs/predictions.csv      validation_report.md         algorithm_benchmark.md
 outputs/review_list_*.csv
              │
              │  ModelBundle loaded once at startup
              │  (versioned; if missing, the service still starts and returns 503 + how to fix)
              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ 4. SERVE                                              service.py (FastAPI)      │
 │                                                                                 │
 │  POST /score ─► schemas.Claim ─► clean_claims ─► build_features ─► risk_scores  │
 │  one claim      422 if invalid   same code as    partner history                │
 │  as JSON                         the batch       + this claim                   │
 │                                                                         │       │
 │  response ◄── recommendation ◄── explain.reasons ◄──────────────────────┘       │
 │  score, rank,  vs the 40-a-month  coefficients → plain sentences                │
 │  reasons       review threshold   (grouped, never contradicting)                │
 │                                                                                 │
 │  GET /  screen (static/index.html) · GET /example · GET /health · GET /docs     │
 └─────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ 5. USE                                                   investigation desk     │
 │    review the top 40 claims a month (keep ~5 slots for random established       │
 │    partners) → record fraud / not fraud                                         │
 └─────────────────────────────────────────────────────────────────────────────────┘
              │
              │  monthly CRM export with the new outcomes → replace the files in Data/
              └──────────────────────────► back to 1. CLEAN  (retrain every month)
```

Steps 1 and 2 are the same code for the batch and for a single API claim, so a claim scored through the API gets exactly its batch score (`tests/test_service.py` checks this). Step 3c is evidence only: it never changes the production model.

## Layout

```
.
├── kestrel_fraud/                  the product, in pipeline order
│   ├── __init__.py                 package version
│   ├── config.py                   paths (env-overridable), business rules from the policy, modelling choices
│   ├── data.py                     load the data pack, clean claims (one code path for batch and API)
│   ├── features.py                 point-in-time features (claim, partner behaviour, lagged fraud history)
│   ├── model.py                    estimator, training weights, risk_scores(), versioned ModelBundle
│   ├── explain.py                  plain-language reasons from the model's coefficients
│   ├── validation.py               time-based split, staleness freeze, top-40 metric, bootstrap, Wilson CI
│   ├── train.py                    CLI: train on all labelled claims → artifacts/, outputs/
│   ├── evaluate.py                 CLI: time-based validation → evidence/validation_report.md
│   ├── schemas.py                  API request and response models (shown at /docs)
│   ├── service.py                  FastAPI app: /score, /example, /health, and the screen
│   └── static/index.html           the screen
│
├── tests/                          38 tests: 28 unit tests on synthetic data + 10 API tests on the real pack
│   ├── conftest.py                 synthetic partners/products, claim factory, skip rule for API tests
│   ├── test_data.py                cleaning; bad records rejected by name
│   ├── test_features.py            30-day label lag, no leakage from later outcomes, partner windows
│   ├── test_model.py               score correction keeps the ranking; missing model gives a clear error
│   ├── test_validation.py          staleness freeze, bootstrap interval
│   ├── test_train.py               predictions file matches sample_submission.csv or training stops
│   └── test_service.py             API reproduces predictions.csv, reasons, timezone, 422, 503
│
├── evidence/
│   ├── validation_report.md        June results, error rates, cases it gets wrong, expected test score
│   └── algorithm_benchmark.md      12 other algorithms on the same folds; can any reach 97–98% accuracy
│
├── notebooks/                      the analysis, outputs cleared
│   ├── Kestrel_Home.ipynb          end to end: target, cleaning, features, validation, final model
│   └── Ritu_Hypothesis_New_Partners.ipynb   fraud rates, chi-square, logistic regression, partner-level test
│
├── tools/
│   ├── export_notebooks.py         copies working notebooks into notebooks/ with outputs stripped
│   └── benchmark_algorithms.py     writes evidence/algorithm_benchmark.md (research; not used by the product)
│
├── Analysis_&_planning/            handwritten requirements and plan, written before any AI tool was used
├── ai_usage/prompt-log.txt         every prompt sent to Claude Code, timestamped (IDs removed)
├── memo_to_ritu.md                 one-page decision memo for the client
├── submission-form.md              answers to the submission form, with the Drive and GitHub links
├── README.md · CLAUDE.md           how to run it · guidance for coding agents working in this repo
├── requirements.txt                product, training and tests (pinned)
├── requirements-notebook.txt       extra packages for the notebooks and the benchmark
├── pyproject.toml                  pytest and ruff settings, package metadata
└── .gitignore · .gitattributes     keep client data out; keep line endings consistent

Not committed (git-ignored, ops policy §10):
  Data/                             the client data pack
  artifacts/model.joblib            trained model; embeds the cleaned claim history
  outputs/                          predictions.csv, review_list_jul_sep_2026.csv, notebook/ (the notebook's own copies)
  drive_upload/                     files shared privately instead of published
  .claude/                          local agent settings and the working notebooks with outputs
```

`train.py`, `evaluate.py` and the service share all logic through `data`, `features` and `model`, so the batch predictions and the API cannot drift apart (a test checks they agree). The notebooks are the exploratory record; the package is the source of truth.
