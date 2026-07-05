# Results Snapshot

Generated from `records/summary.csv` after the local Ollama run.

| Version | Correctness | Faithfulness | Citation Hit | False-Premise Detection | Refusal Accuracy | Avg Calls | Avg Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline | 0.408 | 0.608 | 0.000 | 0.000 | 1.000 | 1.00 | 2.00 |
| V1 retrieval only | 0.492 | 0.975 | 0.917 | 0.000 | 0.500 | 1.00 | 3.43 |
| V2 router + retrieval | 0.508 | 0.975 | 0.950 | 0.200 | 0.500 | 2.00 | 4.52 |
| V3 retrieval + verifier | 0.558 | 0.983 | 0.917 | 1.000 | 0.500 | 1.80 | 6.00 |
| V4 full workflow | 0.592 | 0.975 | 0.950 | 1.000 | 1.000 | 2.77 | 6.78 |
| A1 full without router | 0.583 | 0.975 | 0.950 | 1.000 | 0.500 | 1.80 | 5.76 |
| A2 full without verifier | 0.525 | 0.975 | 0.950 | 0.200 | 1.000 | 2.00 | 4.45 |
| A3 full top-k=2 | 0.533 | 1.000 | 0.817 | 1.000 | 1.000 | 2.77 | 5.98 |
| A4 without calculation guard | 0.567 | 0.975 | 0.950 | 1.000 | 1.000 | 2.87 | 7.14 |

## Interpretation Notes

These numbers are automatic initial scores. They are useful for trends, but the final report should include a manual sanity check because keyword-based scoring can undercount semantically correct answers.

Main observed trends:

- Retrieval greatly improves grounding: citation hit rises from 0.000 to 0.917-0.950.
- The verifier/misconception guard fixes false-premise questions that baseline and retrieval-only versions often mishandle.
- The deterministic calculation guard raises calculation-case correctness to 1.000; removing it lowers overall correctness from 0.592 to 0.567 and calculation correctness from 1.000 to 0.700.
- The final V4 workflow gives the best automatic correctness score while retaining explicit scope detection.
- Reducing retrieval depth to top-k=2 lowers citation hit rate, which supports the top-k ablation.
- The full workflow improves reliability on false-premise and unanswerable cases but costs more LLM calls and latency.

Important caveat: correctness is an automatic keyword-style score. It is intentionally reproducible but conservative; for example, several semantically correct misconception corrections receive 0.5 because they use equivalent wording rather than every expected term. The final report should pair this table with manual case-level review.

## Recommended Case Studies

### False Premise: Logistic Regression and XOR

Case `E011`: "Since logistic regression uses a sigmoid, why can it solve XOR without hidden layers?"

- Baseline accepts the false premise and claims logistic regression can solve XOR.
- Full workflow corrects it: logistic regression has a linear decision boundary, and sigmoid alone does not add hidden nonlinear transformations.

### False Premise: CNN Rotation Invariance

Case `E034`: "Since CNNs are naturally rotation invariant, why does training on upside-down images not hurt testing on normal images?"

- Baseline/retrieval-only refuse or miss the misconception.
- Full workflow corrects the premise: standard CNNs are not naturally invariant to 180-degree rotation, and train-test distribution mismatch can hurt performance.

### Unanswerable: Exact Final Exam Questions

Case `E059`: "What exact questions will appear in the final exam?"

- Full workflow refuses with "not enough information in the provided course materials".
- This supports the responsible-use claim that the system should not invent private or unavailable information.

### Calculation: CNN Output Shape

Case `E030`: "For a 32x32x3 image with 8 filters of size 5x5, stride 1, no padding, and no bias, what are the output shape and parameter count?"

- Without the calculation guard, the local LLM can produce invalid shapes or parameter counts.
- Full workflow computes the formula directly: output shape `28x28x8`, parameter count `5*5*3*8 = 600`.
