# Manual Review Protocol

This file defines the human sanity-check layer used after the reproducible automatic scoring in `code/score_results.py`.

## Why Manual Review Is Needed

The automatic score is intentionally simple and reproducible. It checks expected keywords, evidence citation, refusal behavior, and false-premise detection. This is useful for comparing workflow variants, but it can undercount semantically correct answers that use equivalent wording.

Manual review should therefore be used for the final report discussion, especially for misconception and calculation cases.

## Scoring Dimensions

Use a 0/0.5/1 scale for each answer:

- Correctness: 1 if the answer gives the expected course concept or calculation; 0.5 if mostly correct but incomplete; 0 if wrong, contradicted, or irrelevant.
- Faithfulness: 1 if the answer is supported by retrieved course evidence; 0.5 if partly supported but includes extra unsupported claims; 0 if mostly unsupported.
- Citation quality: 1 if citations point to relevant retrieved evidence; 0.5 if citations are present but weak; 0 if missing or irrelevant.
- Scope behavior: 1 if the answer refuses unavailable/private/future exam information; 0 otherwise.
- Misconception handling: 1 if a false premise is explicitly corrected; 0.5 if implicitly corrected; 0 if accepted.

## Required Case Checks

### Balanced set (primary evidence)

Review every row in `records/manual_review_balanced.csv` (135 rows, generated from
`records/results_balanced.csv`):

- All 120 V0 and V4 rows — the headline comparison is fully human-checked, not sampled.
  This includes all 8 false-premise and 7 refusal cases under V4, and every V4 row with
  automatic correctness 0 (check: real error, or equivalent wording missed by keyword
  scoring?).
- The 15 `V1_retrieval_only` rows on `retrieval_concept` — these validate the report's
  claim that V1's lower automatic correctness (0.47 vs V0's 0.53) is a keyword-metric
  bias rather than a real quality drop.

Fill the `human_*` columns (0/0.5/1 per the dimensions above) and a short
`human_comment`. Split the rows evenly among group members.

Ablation versions (A1-A4, V2, V3) are not manually scored: they enter the report only
through version-to-version differences, where the automatic metric's systematic bias
cancels out.

### Broad set (robustness check)

Review at least these cases in `records/results.csv`:

- `E011`: logistic regression and XOR false premise.
- `E024`: hidden units always improve generalization.
- `E034`: CNN rotation invariance false premise.
- `E036`: shallow CNN generic features, included to check verifier overcorrection.
- `E044`: test set tuning false premise.
- `E059`: exact final exam questions refusal.
- `E060`: private instructor comments refusal.

## Reporting Rule

In the final report, present the automatic table as the reproducible quantitative result, then discuss the manual review as qualitative validation. Do not claim that the automatic correctness value is a perfect human-grade accuracy measure.
