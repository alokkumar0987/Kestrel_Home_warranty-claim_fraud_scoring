# Algorithm benchmark

Generated 2026-10-06 23:37 by `python tools/benchmark_algorithms.py`.

## Why this exists

Two questions a reviewer will ask: would another algorithm do better, and can any algorithm reach the board's accuracy target (97%, or 98%)? This benchmark answers both on the same evidence as the main validation. **It was run after the model configuration was frozen.** It is evidence, not a selection step: the production model did not change because of it.

## How it was run

- **Same data, same folds** as `evidence/validation_report.md`: train on claims to 31 May 2026, score all 713 June claims (22 frauds). Partner fraud history is frozen at the cutoff with the 30-day label lag. The Jul–Sep test set is not used.
- **Same 30 features and preprocessing** for every algorithm, and the same post-May training weight where the algorithm accepts sample weights. Class imbalance is handled with each algorithm's own option.
- **Generic settings, no tuning.** Tuning 13 algorithms on one month with 22 frauds would only fit noise. The notebook's XGBoost (5× weight, different settings) scored PR-AUC 0.46 on the same month, against 0.26 here: settings alone move a result that much.
- **Fold A ROC-AUC:** the same algorithm trained only on claims before the 1 May 2026 rule, scored on May–June. Below 0.5 means it ranks the new fraud backwards.

## Results (June 2026)

| Algorithm                                   | PR-AUC   | PR-AUC 95% CI   | ROC-AUC   | Frauds in top 40   | Accuracy, top 40 checked   | Best accuracy, any cut-off*   | Fold A ROC-AUC   |
|:--------------------------------------------|:---------|:----------------|:----------|:-------------------|:---------------------------|:------------------------------|:-----------------|
| Extra trees                                 | 0.475    | 0.27–0.68       | 0.908     | 15                 | 95.5%                      | 97.5% (flag 4)                | 0.33             |
| Isolation forest (unsupervised)             | 0.430    | 0.24–0.64       | 0.881     | 11                 | 94.4%                      | 97.5% (flag 4)                | 0.81             |
| Balanced random forest                      | 0.410    | 0.24–0.62       | 0.910     | 14                 | 95.2%                      | 97.2% (flag 6)                | 0.24             |
| Support vector machine (RBF)                | 0.410    | 0.20–0.60       | 0.853     | 13                 | 95.0%                      | 97.3% (flag 3)                | 0.72             |
| Rank average: logistic + XGBoost + LightGBM | 0.409    | 0.24–0.61       | 0.864     | 13                 | 95.0%                      | 97.3% (flag 3)                | 0.11             |
| Logistic regression (current model)         | 0.371    | 0.21–0.59       | 0.913     | 14                 | 95.2%                      | 97.1% (flag 1)                | 0.12             |
| Random forest                               | 0.371    | 0.18–0.58       | 0.859     | 14                 | 95.2%                      | 97.1% (flag 1)                | 0.34             |
| Neural network (MLP)                        | 0.352    | 0.20–0.56       | 0.874     | 11                 | 94.4%                      | 97.3% (flag 3)                | 0.15             |
| LightGBM                                    | 0.350    | 0.18–0.54       | 0.839     | 11                 | 94.4%                      | 97.3% (flag 3)                | 0.18             |
| k-nearest neighbours                        | 0.349    | 0.20–0.54       | 0.910     | 16                 | 95.8%                      | 97.1% (flag 1)                | 0.47             |
| Logistic regression, L1                     | 0.309    | 0.18–0.54       | 0.900     | 10                 | 94.1%                      | 97.1% (flag 3)                | 0.10             |
| XGBoost                                     | 0.260    | 0.14–0.48       | 0.796     | 12                 | 94.7%                      | 96.9% (flag 0)                | 0.12             |
| Histogram gradient boosting                 | 0.257    | 0.16–0.47       | 0.758     | 15                 | 95.5%                      | 96.9% (flag 0)                | 0.27             |

\* *Best accuracy, any cut-off* picks the number of claims to flag **using June's answers**, which no real system can do. It is an upper bound, deliberately optimistic.

## What it shows

- **No algorithm reaches 97%, let alone 98%.** Flagging nothing scores 96.9%. The best accuracy any algorithm reaches, even with the optimistic cut-off, is 97.5%, by flagging a handful of claims. 98% on June allows at most 14 errors; flagging nothing already makes 22. Getting there needs the flagged claims to be about 80% fraud; the best top 40 here is 40% fraud.
- **Accuracy falls as the desk checks more claims**, for every algorithm: catching fraud means holding some genuine claims. An accuracy target rewards checking fewer claims.
- **The differences between algorithms are within noise.** Every PR-AUC interval overlaps the current model's (0.21–0.59). The top-40 catch ranges from 10 to 16 frauds, a few claims in one month.
- **Most algorithms do not survive the policy change.** Trained before May, 11 of 13 rank May–June claims at ROC-AUC 0.10–0.47, most below 0.5 (backwards): they learned the old fraud. The exceptions: Isolation forest (unsupervised): 0.81; Support vector machine (RBF): 0.72.
- **The isolation forest uses no labels at all.** It scores 0.81 on Fold A and 0.43 PR-AUC on June: the new fraud looks unusual, so it stands out before anyone has labelled it. That makes it the best early-warning signal here for the next shift, which a supervised model cannot see until outcomes arrive.

## Decision

- **Keep the logistic regression** for scoring. It is level with the best on the desk's 40 checks, its reasons are exact coefficients a claims officer can read, and switching now would mean choosing on the same month used to validate.
- **Add the isolation forest as a monitor, not a scorer**, in the next iteration: each month, count the claims it finds unusual that the model scores low. A jump is the cue to send a random sample to the desk and retrain early. Not built now: it changes the desk's process, which needs Kestrel's agreement.
- **Re-run this benchmark when July outcomes arrive.** Switch algorithm only if one is clearly better on two months in a row. Extra trees and k-nearest neighbours are the ones to watch.
