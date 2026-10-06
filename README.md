# Kestrel Home: warranty-claim fraud scoring

Scores each warranty claim for fraud risk so Kestrel's investigation desk (about 40 reviews a month) looks at the riskiest claims before they are paid. Includes a small web service with one screen, the analysis notebooks, and evidence of how well it works.

**No paid API or key is needed.** The model is a logistic regression that runs locally.

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

```
                        Data/  (client data pack, not in the repo)
          train.csv · test_unlabelled.csv · partners.csv · products.csv
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ data.py      find files by suffix → drop re-submitted duplicates (keep   │
 │              first) → clean claims (serials, Y/N, descriptions, injected │
 │              text) → join partners + products                            │
 └──────────────────────────────────────────────────────────────────────────┘
                                     │  cleaned claim history
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ features.py  point-in-time features: claim · partner tenure · partner    │
 │              behaviour (30/90 days before) · partner fraud history       │
 │              (outcomes ≥ 30 days old only)                               │
 └──────────────────────────────────────────────────────────────────────────┘
                │                                          │
                ▼                                          ▼
 ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
 │ train.py                        │      │ evaluate.py                     │
 │ model.py: logistic regression,  │      │ validation.py: train to 31 May, │
 │ post-May ×3, score correction   │      │ score June, top 40, bootstrap,  │
 │ fit on all labelled claims      │      │ staleness backtest              │
 └─────────────────────────────────┘      └─────────────────────────────────┘
        │                    │                             │
        ▼                    ▼                             ▼
 artifacts/model.joblib   outputs/predictions.csv   evidence/validation_report.md
 (ModelBundle: model,     outputs/review_list_...
  threshold, history)
        │
        │ loaded once at startup (versioned; 503 + instructions if missing)
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ service.py  (FastAPI)                                                    │
 │                                                                          │
 │  POST /score ──► schemas.Claim ──► data.clean_claims ──► features on     │
 │   one claim      (validate, 422)    (same code as batch)  partner history│
 │                                                           + this claim   │
 │                                                               │          │
 │  response ◄── recommendation ◄── explain.reasons ◄── model.score         │
 │  (score, rank,  (vs 40-a-month    (coefficients →                        │
 │   reasons)       threshold)        plain sentences)                      │
 │                                                                          │
 │  GET /  (static/index.html) ── calls /example and /score                 │
 │  GET /health · GET /example · GET /docs                                  │
 └──────────────────────────────────────────────────────────────────────────┘
```

`config.py` sits under all of it: policy dates, the Rs 2,000 limit, 40 reviews a month, costs, paths and modelling choices. Because the service runs the same `clean_claims` and `build_features` as training, a claim scored through the API gets exactly its batch score (`tests/test_service.py` checks this).

## Layout

```
kestrel_fraud/            the product, in pipeline order
  config.py               paths (env-overridable), business rules, modelling choices
  data.py                 load the data pack, clean claims (same code for batch and API)
  features.py             point-in-time features
  model.py                estimator, training weights, score correction, versioned ModelBundle
  explain.py              plain-language reasons from the model's coefficients
  validation.py           time-based split and review-desk metrics
  train.py                CLI: train on all labelled claims, write artifacts/ and outputs/
  evaluate.py             CLI: write evidence/validation_report.md
  schemas.py              API request/response models (shown at /docs)
  service.py              FastAPI app: /score, /example, /health, and the screen
  static/index.html       the screen
tests/                    unit tests (no data needed) + API tests
notebooks/                analysis, outputs cleared: Kestrel_Home (end to end), Ritu_Hypothesis_New_Partners
tools/export_notebooks.py strips outputs from working notebooks into notebooks/
tools/benchmark_algorithms.py  writes evidence/algorithm_benchmark.md (research, not used by the product)
evidence/                 validation report and algorithm benchmark
memo_to_ritu.md, submission-form.md    deliverables
Analysis_&_planning/      my handwritten requirements and plan, written before using any AI tool
outputs/notebook/         the notebook's own copies of the outputs (checked against outputs/, never overwrite it)
```

`train.py` and `evaluate.py` share all logic with the service through `data`, `features` and `model`, so the batch predictions and the API cannot drift apart (a test checks they agree). The notebooks are the exploratory record; the package is the source of truth.
