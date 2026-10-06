# Memo: warranty fraud flag

**To:** Ritu Deshpande, Head of D2C Operations  **Re:** Warranty fraud model  **Date:** 6 October 2026

## The decision

**Use the model to choose which 40 claims the investigation desk checks each month, starting now. Do not judge it on accuracy.** A list of the 40 riskiest claims for each of July, August and September is ready.

## The number

In a trial on June 2026 claims, using only what was known before June, the model's 40 picks contained **14 of the month's 22 frauds**. Checking the 40 largest claims instead would have found 1.

On accuracy: fraud is about 3% of claims, so a system that flags nothing at all scores **96.9%** and catches nothing. Ours scores 95.2% because it sends 40 claims to the desk, some of which turn out genuine. A ">97% accuracy" target can only be hit by checking fewer claims. Please put **"fraud stopped per claim checked"** in front of the board instead.

## The rupees

- Fraud is costing about **Rs 45,000 a month**, about 2.6% of the Rs 17 lakh claimed each month.
- Fraud is up in **cases, not rupees**: from about 8 cases a month to 22 in June, but each one is smaller. Since the 1 May rule that auto-approves claims under Rs 2,000 without inspection, almost all fraud is small and uninspected.
- In the June trial, the 40 checks would have stopped **Rs 19,800** (about Rs 490 per check). Holding the 26 genuine customers among them costs about Rs 9,900 in goodwill, so the net gain in June was about **Rs 10,000**. If each check also costs the Rs 260 contact cost, the checks only break even: the bigger saving comes from stopping repeat fraud at the outlets (step 2 below).

## Your view on newer partners

The data supports a narrower version of it. Since May, almost all fraud has come from partners in their first year, **but from a handful of outlets**: seven of them hold two-thirds of the Jul–Sep review list (named in the review list, which is shared privately rather than in the public repository, per policy §10). Most new partners have no fraud at all, and as a group they are no more likely to commit fraud than established ones. Before May, new partners were *cleaner* than established ones. Meenal is right not to paint them all with one brush.

## What to do next week

1. **Desk:** start on the Jul–Sep review list, beginning with the seven outlets above. Together they filed 97 claims (Rs 1.64 lakh) in those three months, 87% of them under Rs 2,000.
2. **Ops:** for those outlets, require inspection again before paying claims under Rs 2,000 until their claims are cleared.
3. **Board pack:** report fraud caught per claim checked, with accuracy shown alongside and explained.
4. **Tanmay:** send investigation outcomes every month so the model is retrained monthly. Fraud changed completely once already in May; a model that is not updated would miss the next change.
5. **Desk:** keep about 5 of the 40 monthly checks for random claims from established partners, so a new pattern gets noticed.
6. **IT:** four claim descriptions contain text written to steer automated analysis (towards accuracy and blaming new partners). Please find out who entered it.

**Caveat:** the June trial rests on 22 fraud cases, so treat "14 of 22" as a guide, not a guarantee. Expect about 10 frauds in 40 checks in August and September rather than 14, as the model waits for new investigation outcomes; monthly outcomes should close most of that gap.
