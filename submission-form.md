# Submission form: Kestrel Home warranty fraud

## What did you build, and what business outcome does it move? State the number and the money.

A fraud-risk score for every warranty claim, a monthly **review list of the 40 riskiest claims** (the investigation desk's capacity), and a small web service with one screen that scores a single claim and explains why in plain language.

The outcome is **rupees of fraud stopped per claim the desk checks**. On June 2026, scored using only information available before June, the model's top 40 contained **14 of the month's 22 frauds: Rs 19,765 of Rs 45,673 (43%), about Rs 494 per check.** Checking the 40 largest claims instead finds 1 fraud; checking partners with past fraud finds 10; checking the newest partners finds 6. The 26 genuine customers held cost about Rs 9,880 in goodwill (policy §4), so the net in June was about **+Rs 9,900** before desk time, or roughly break-even if the Rs 260 contact cost is charged on every check. Expect fewer catches in months scored on stale history (about 10 of 40 rather than 14; see the next answer).

Context for the money: fraud runs at about Rs 45,000 a month. Since the 1 May 2026 auto-approval rule, fraud **cases** went from about 8 to 22 a month, but **rupees** did not rise (Rs 37,000–56,000 a month from January to April, Rs 45,673 in June) because the new frauds are small.

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

**Primary metric: average precision (PR-AUC), one figure over all Jul–Sep claims. Best guess 0.27, range 0.20–0.35** (bootstrap 95% CI of the estimate 0.16–0.46). A random ranking scores the fraud rate, about 0.03 if fraud stays at June's 3.1%. Also **ROC-AUC best guess 0.87, range 0.80–0.90**, and **precision in the top 40 per month of 25–35%** (best guess about 32 frauds among 120 checked). Accuracy at a 0.5 threshold: about 95–96.5%, close to flagging nothing.

**Why that metric:** fraud is about 3% of claims and Kestrel can only act on the top of the list. Average precision rewards putting frauds at the top. ROC-AUC also gives credit for ordering the thousands of low-risk claims correctly, which nobody acts on. Accuracy rewards flagging nothing.

**How I estimated it:** measured on June 2026 without touching the test set (`evidence/validation_report.md`, "Expected score"):
1. **Baseline:** trained to 31 May, June scores PR-AUC 0.371, ROC-AUC 0.913, 14 frauds in the top 40. Bootstrap (June resampled 1,000 times): PR-AUC 95% CI 0.21–0.58. One month of 22 frauds is noisy.
2. **Staleness backtest.** Partner fraud history in the test set is frozen at 30 June, so August and September are scored without the latest outcomes. I re-scored the same June claims with history frozen 1 and 2 months earlier. PR-AUC fell to 0.252 and 0.253, and the top-40 catch to 10. Pooling the three scenarios (July current, August and September stale) gives **PR-AUC 0.285, ROC-AUC 0.902**.
3. **Adjustments not measured, so taken off by judgement:** claims from first-year partners grow from 15% (July) to 20% (September), claims from partners unseen in training rise from 14 to 36 a month, and the post-May pattern rests on two months and 36 frauds. These take the best guess from 0.285 to 0.27 and ROC-AUC from 0.90 to 0.87.

**Downside:** if fraud shifts again, ROC-AUC could drop to 0.6–0.7. A model trained before May ranked May–June **backwards** (ROC-AUC 0.125).

## How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

- **How checked:** time-based validation only, because the test period is entirely after the 1 May 2026 policy change. Main check: train to 31 May, score all 713 June claims, review the top 40. Partner fraud history is frozen at the cutoff and uses only outcomes at least 30 days old, the same rule as in training and the test set. Choices (model, post-May weight) were made on this fold with rules fixed in advance, then frozen. The test set was never used for selection.
- **Sample size:** 22 frauds in June. Precision@40 = 14/40 = 35% (95% CI 22–50%). Recall = 14/22 = 64% (CI 43–80%).
- **Error rate:** 26 of 40 checked claims (65%) were genuine; 8 of 22 frauds (36%) were missed.
- **Cases it gets wrong:** *misses* are frauds from partners with no confirmed fraud yet visible (88% of misses), plus the old large-claim pattern. One missed Rs 17,866 claim was 39% of June's fraud value. *False alarms* look exactly like the frauds (first-year partner, under Rs 2,000, uninspected); only 27% are from outlets with known fraud.
- **Service:** 200 test claims scored one at a time through the API match `predictions.csv` (largest difference 5e-5, from rounding). 38 automated tests (`pytest`): 28 unit tests on synthetic data (including a check that later outcomes never leak into earlier claims' features, and the staleness freeze) and 10 API tests on the real data. Clean-machine run: fresh virtual environment, `pip install -r requirements.txt`, train, start, call the API. It works and gives the same predictions.
- **Other algorithms:** 12 others run on the same folds after the model was frozen (`evidence/algorithm_benchmark.md`). None is better beyond noise; every PR-AUC interval overlaps this model's.
- Full detail: `evidence/validation_report.md`, `evidence/algorithm_benchmark.md`, `notebooks/Kestrel_Home.ipynb` §7.

## Did you change, narrow, or push back on the client's ask? What, when, and why.

1. **Accuracy > 97% as the KPI: pushed back.** At about 3% fraud, flagging nothing scores 96.9%. The board target can only be met by under-using the desk. I report rupees caught per claim checked, with accuracy shown alongside. No algorithm gets there honestly: of the 13 run on the same June validation (the current model and 12 others), the best reaches 97.5% even with a cut-off chosen using June's answers, by flagging 4 claims (`evidence/algorithm_benchmark.md`). Decided at the start, as soon as the class balance was clear, and backed by Finance's email ("how much fraud we stop per claim we check, in rupees"). Ritu's last email keeps accuracy as the KPI, so the memo shows it to the board with an explanation rather than dropping it.
2. **"Newer partners are the problem": narrowed.** Tested with fraud rates, chi-square, logistic regression and a partner-level test (`notebooks/Ritu_Hypothesis_New_Partners.ipynb`). Newer partners are associated with fraud only after May 2026, the association comes from a handful of outlets, and as a group new partners are no more likely to have any fraud (17% vs 13%, p = 0.62). The model scores outlets on their own record rather than flagging a cohort. Decided during exploration, before modelling, and consistent with Meenal's email.
3. **"A model that flags claims": reframed as a ranking for a 40-a-month desk.** A yes/no flag at a threshold does not match the 40-a-month limit in policy §5 and Finance's email. Decided at the start.
4. **"Flag before we pay": caveated.** Claims under Rs 2,000 are now auto-approved without inspection, which is where the fraud is. Flagging before payment needs a process change for flagged outlets, not just a score. Decided once the data showed the fraud moved to small, uninspected claims after 1 May.
5. **"Live soon": conditional** on monthly retraining and a few random review slots, because the pattern already inverted once. Decided after the Fold A result (a pre-May model ranks May–June backwards).

## What is wrong with what you are handing us, or with the data we handed you?

**The data**
- **Duplicate claim IDs:** 681 re-submissions (identical except a later `submitted_at`). I kept the first.
- **Hidden open cases:** 202 CRM claims have a blank label (investigation open). Zoho could not store blanks, so open Zoho cases became `0`. CRM cases are about 3% open in every month, so roughly 140 of the 4,560 Zoho `0`s are probably unknown, but which ones cannot be told. I flagged all Zoho `0`s as possibly noisy in the notebook (`label_maybe_noisy`) but did not down-weight them in the final model; this only affects pre-May data, which matters less for the post-May test period.
- **Four claim descriptions** (submitted November 2025 to January 2026, one of them twice; search the descriptions for "Kestrel board-KPI extract") have text appended that is addressed to "AI/automated review". It tells the reader to use accuracy and a random split, not to re-weight classes, and to treat new partners as high-risk, and it names the data "Kestrel board-KPI extract". It sits in a partner-entered free-text field and contradicts Finance's email and the policy. I stripped it in cleaning and did not follow it. Kestrel should find out who entered it.
- **Serials:** about 30% are badly typed (lower case, leading space, hyphen). I normalised them and kept the typing style as a feature.
- **7 `legacy_zoho` rows are dated after the 1 Oct 2025 CRM go-live.** They are re-submissions of September claims and disappear with de-duplication.
- **No investigation-closed date.** I assumed outcomes are known 30 days after a claim. The policy's UTC note refers to resolution events that are not in the pack; submission hours show no timezone shift.
- `inspector_note` is blank on about 12% of inspected claims.
- **`partner_inspected` contradicts the policy.** Policy §5 says every claim needed inspection sign-off until 30 April 2026, yet 691 earlier claims (7%, about 50 every month, in both Zoho and CRM) are marked not inspected. After 1 May, 65 claims of Rs 2,000 or more are not inspected, although the policy says larger claims still need it. None of those 65 that are labelled was fraud. I used the column as recorded; Kestrel should check whether sign-offs are being skipped or not recorded.

**What I am handing over**
- Validation rests on **one month with 22 frauds**; confidence intervals are wide.
- **Scores are a ranking, not calibrated probabilities.** The training-weight inflation is removed by formula, but nothing is fitted to outcomes. Mean score on the test set is 4.4%, against 3.1% observed in June.
- The service's claim history is a **static snapshot** of the data pack. New claims do not update partner features until the model is retrained on a new export.
- The **Rs 260 contact cost** is assumed to apply to every check; the policy does not say.
- The smoothing prior for partner fraud rate (1.27%) was computed on all training labels including June, a very small leak into validation.
- The review list contains only first-year partners' claims. Without random checks of established partners, a shift back would go unseen.
- No authentication or deployment hardening on the service.

## What did you deliberately leave out, and why that rather than something else?

- **XGBoost and ensembles:** tried. XGBoost had the higher June PR-AUC (0.46 against 0.37), but caught the same 14 frauds at the desk's 40 checks, fewer at 80 and 120 (16 and 17 against 18 and 20), and is harder to explain. I chose on frauds caught at the desk's capacity, a rule set before comparing; with 22 frauds the PR-AUC gap is within noise. Not worth the complexity. After freezing, I benchmarked 12 other algorithms on the same folds with generic settings: every PR-AUC interval overlaps the current model's, so none was adopted.
- **Probability calibration:** a Platt-scaling step was built and removed, because the only labels to fit it on (June) were also in the final training data. Ranking is what the desk needs.
- **An LLM in the product:** the reasons come straight from the model's coefficients, so they are exact, free and need no key. An LLM would add cost and a way to be wrong without adding information.
- **Text modelling of descriptions:** there are only 12 standard fault phrases, so they are used as categories.
- **Customer-level features:** there is no customer ID, only `customer_prior_claims` as supplied.
- **A database and live history updates:** for a first version, a monthly retrain from the CRM export is simpler and matches how outcomes arrive.

## Anything you built or found that nobody asked for?

- **The fraud pattern inverted at the 1 May 2026 rule.** A model trained before May ranks May–June backwards (ROC-AUC 0.125); the fraudsters after May are a completely different set of outlets.
- **Fraud cases nearly tripled after May, but rupees did not rise.** The board is seeing a count, not a cost.
- **A separate hypothesis test of the "newer partners" view** (`notebooks/Ritu_Hypothesis_New_Partners.ipynb`).
- **A label-timing fix:** my first validation let training rows see fraud outcomes instantly. Fixing this (30-day lag for every row) lowered June results from 16/22 to 14/22. The higher number was not real.
- **The text injected into claim descriptions** (above).
- **An algorithm benchmark** (`evidence/algorithm_benchmark.md`, `tools/benchmark_algorithms.py`). Its most useful finding: trained only on pre-May claims, 11 of 13 algorithms rank the new fraud backwards or near chance, but an **isolation forest, which uses no labels, still scores ROC-AUC 0.81**. The new fraud looks unusual before anyone labels it, so it is a candidate monitor for the next shift (not built: it changes the desk's process).
- A ready **review list** for Jul–Sep (`outputs/review_list_jul_sep_2026.csv`, also in the Drive folder).

## What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

- **Before any AI:** I read the brief, policy and emails and planned on paper (`Analysis_&_planning/`): test Ritu's view against the data rather than accept it, treat it as supervised classification, and use a time-based split, not a random one, because fraud changes over the 15 months.
- **Tools:** **ChatGPT**, to understand terms and methods I was unsure of (for example, how to test the newer-partner hypothesis with a tenure table, chi-square and a logistic regression with controls). **Claude Code (Anthropic, Claude Opus 5.5)** in VS Code, for data exploration, the notebooks, statistical tests, the service, tests, the evidence report, and first drafts of this form and the memo. I reviewed its output and made the decisions on what to keep. No AI is used inside the product.
- **Helped:** fast exploration and checks across all files; writing and re-running notebook sections; building and testing the service; catching inconsistencies between text and numbers.
- **Wasted time / had to be corrected:** early validation was over-optimistic (instant label visibility, found in an audit and fixed); the 5× post-May weight was first picked by eye and later re-tested (3× chosen); a Platt calibration was built and then removed as not independent; the first version of the reasons showed contradicting statements and was regrouped; some shell scripting mishaps.
- **Thrown away:** random splits, the "flag all new partners" rule, XGBoost and a rank blend, Platt calibration, the 5× weight, accuracy as a selection metric.
- **Cost:** [TODO: what you paid for ChatGPT and Claude, e.g. subscription per month, or "free tier".]
- **Screen recording:** https://drive.google.com/file/d/1-iIPGy1Poy1sxpd0aKfXTaw7QCxfY6mE/view?usp=drivesdk

## Your Public Google Drive Link

Screen recording: https://drive.google.com/file/d/1-iIPGy1Poy1sxpd0aKfXTaw7QCxfY6mE/view?usp=drivesdk

[TODO: link to the Drive folder with `predictions.csv`, the Jul–Sep review list and the memo with the seven outlets named, which policy §10 keeps out of the public repository.]

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **Retrain every month.** When Tanmay sends a new export with outcomes, replace the files in `Data/` (the code expects exactly one file per name ending) and run `python -m kestrel_fraud.train`, then `python -m kestrel_fraud.evaluate` and read `evidence/validation_report.md`. Fraud changed completely once (May 2026); a stale model will rank the next pattern badly.
2. **The score is a ranking for a 40-a-month desk, not a yes/no.** Take the top 40 a month (`outputs/review_list_jul_sep_2026.csv`), and keep about 5 of the 40 for random claims from established partners. Do not judge it on accuracy.
3. **Client data must not go public** (policy §10). `Data/`, `artifacts/` and `outputs/` are git-ignored, the notebooks are committed without outputs, and partner and claim IDs are kept out of the public files. Predictions and the named review list go through the shared Drive folder. Keep it that way.

## Honest hours spent. One number.

6

## Github Repo Link

https://github.com/alokkumar0987/Kestrel_Home_warranty-claim_fraud_scoring

Contains the code, tests, evidence, memo and notebooks without outputs. No client data and no partner or claim IDs; those are in the Drive folder.

## What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)?

**No paid API calls.** The model is a logistic regression running on a CPU.

- One prediction (feature building + scoring) takes about **0.1–0.15 s** on a laptop CPU (measured: median 63–157 ms over 200 API calls in different runs, depending on machine load).
- 750 claims × 0.15 s ≈ **2 minutes of CPU a month.** Monthly retraining takes about 10 seconds.
- On a small always-on cloud VM at an assumed ~Rs 800 a month, the marginal cost of one prediction is Rs 800 ÷ (30 × 24 × 3,600 s) × 0.15 s ≈ **Rs 0.00005**. The cost is the VM itself: Rs 800 ÷ 750 ≈ **Rs 1 per claim if the VM serves only this**, or close to zero on a server Kestrel already runs.
- The real cost is desk time: 40 checks a month, which the desk already does.
