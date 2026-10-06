# Validation report

Generated 2026-10-06 12:36 by `python -m kestrel_fraud.evaluate`.

## How it was checked

- **Time-based validation only.** The test claims (Jul–Sep 2026) all come after the 1 May 2026 rule that auto-approves claims under Rs 2,000, and fraud behaves differently after it. A random split would mix the two periods and overstate performance.
- **Main check (Fold B):** train on claims to 31 May 2026, score every June 2026 claim, review the top 40 (the investigation desk's monthly capacity). Partner fraud history is frozen at 31 May and uses only outcomes at least 30 days old, as it will be for the real test months.
- **Sample size:** June has 713 labelled claims and **22 frauds** (Rs 45,673). All June numbers below are observed counts for one month, with wide uncertainty.

## Result (June 2026, top 40 reviewed)

| Method                                     | Fraud caught (top 40)   | Precision@40   | Rs fraud caught   | Rs / review   | Genuine customers held   | PR-AUC   | ROC-AUC   |
|:-------------------------------------------|:------------------------|:---------------|:------------------|:--------------|:-------------------------|:---------|:----------|
| Model (logistic regression)                | 14 / 22                 | 35%            | 19,765            | 494           | 26                       | 0.371    | 0.913     |
| Rule: partners with recent confirmed fraud | 10 / 22                 | 25%            | 14,538            | 363           | 30                       | 0.161    | 0.789     |
| Rule: newest partners first                | 6 / 22                  | 15%            | 5,481             | 137           | 34                       | 0.176    | 0.891     |
| Rule: biggest claims first                 | 1 / 22                  | 2%             | 17,866            | 447           | 39                       | 0.050    | 0.461     |

- Precision (reviewed claims that were fraud): 14/40 = 35% (95% CI 22%–50%)
- Recall (June frauds caught): 14/22 = 64% (95% CI 43%–80%)
- Rupees: Rs 19,765 of Rs 45,673 (43%), about Rs 494 per review.
- **How often it is wrong:** 26 of the 40 reviewed claims (65%) were genuine (each costs about Rs 380 in goodwill), and 8 of 22 frauds (36%) were not reviewed.
- Accuracy, for reference: flagging nothing = 96.9%; reviewing the top 40 = 95.2%; thresholding the score at 0.5 = 96.4%. Accuracy rewards flagging nothing, so it is not used to judge the model.

## The kinds of case it gets wrong

|                                                  | Caught frauds   | Missed frauds   | False alarms (genuine, reviewed)   |
|:-------------------------------------------------|:----------------|:----------------|:-----------------------------------|
| claims                                           | 14              | 8               | 26                                 |
| partner in first year                            | 100%            | 75%             | 100%                               |
| partner has visible confirmed fraud (30-90 days) | 64%             | 12%             | 27%                                |
| claim under Rs 2,000                             | 100%            | 88%             | 92%                                |
| not inspected                                    | 100%            | 88%             | 92%                                |
| median claim (Rs)                                | 1,344           | 1,223           | 1,554                              |

- **Missed frauds** come from partners with no confirmed fraud yet visible, or are the old pattern (a large claim). The largest missed claim was Rs 17,866, 39% of June's fraud value on its own.
- **False alarms** look like the frauds: first-year partner, under Rs 2,000, not inspected. Only 27% come from partners with visible confirmed fraud. On paper the model cannot tell a genuine small, uninspected claim from a new outlet apart from a fraudulent one; that is what the desk review is for. Every reviewed claim in June came from a first-year partner.

## Expected score on the test set (Jul–Sep 2026)

The test outcomes are hidden, so the expected score is estimated from June. Two measurements replace judgement: a **bootstrap** (June claims resampled 1,000 times) for how much one month of 22 frauds can move, and a **staleness backtest**. The test set's partner fraud history is frozen at 30 June, so August and September claims are scored without the latest outcomes. The backtest scores the same June claims with the same model, with history frozen 1 and 2 months earlier.

| Scored as                  |   PR-AUC | PR-AUC 95% CI   |   ROC-AUC |   Fraud caught (top 40) | Claims with visible partner fraud   |
|:---------------------------|---------:|:----------------|----------:|------------------------:|:------------------------------------|
| July (history current)     |    0.371 | 0.21–0.58       |     0.913 |                      14 | 5.8%                                |
| August (1 month stale)     |    0.252 | 0.15–0.45       |     0.896 |                      10 | 2.9%                                |
| September (2 months stale) |    0.253 | 0.15–0.45       |     0.897 |                      10 | 1.4%                                |

- **Pooled over the three scenarios** (assuming the hidden score is one PR-AUC over all Jul–Sep claims): PR-AUC **0.285** (95% CI 0.16–0.46), ROC-AUC 0.902. A random ranking scores PR-AUC = fraud rate = 0.031.
- One month of staleness costs the whole drop: it hides May's outcomes, the first from the post-May fraudsters. A second month only hides April's, from the pre-May pattern, which barely helps rank June. The September scenario is harsher than the real test (it loses all post-May outcomes; the real test keeps May–June), so it is a pessimistic case.
- **Not measured, so not in the number:** first-year and unseen partners rising through Jul–Sep, and a further change in the fraud pattern. Both push the real score down; Fold A shows how far a pattern change can go.

## The pattern changed once already (Fold A)

Trained only on claims before 1 May 2026, the same model ranks May–June claims **backwards**: ROC-AUC 0.125 (reversed, 0.875). Before May, fraud went with large claims from long-standing partners; after May, with small claims from first-year partners. Technical causes (dates, label encoding, probability column, AUC computation, categories, point-in-time history) were checked in the notebook (section 7.2). This is the biggest risk for the live model: if fraud shifts again, the ranking can fail until it is retrained on new outcomes.

## Service check

200 test claims scored one at a time through `POST /score`; largest difference from `predictions.csv` = 5.0e-05 (the API rounds to 4 decimals). Median latency 63 ms, max 84 ms on a laptop.

Automated tests: `python -m pytest` (unit tests on cleaning, point-in-time features and score correction run without the data pack; API tests also check that the API reproduces `predictions.csv`, reasons are present and consistent, bad input is rejected with a clear message, and the service fails politely without a model).
