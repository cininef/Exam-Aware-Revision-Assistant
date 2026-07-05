# Results Slide Content

## Main Result Table

| Workflow | Correct. | Faithful. | Citation | False Premise | Refusal | Calls |
|---|---:|---:|---:|---:|---:|---:|
| V0 baseline | 0.408 | 0.608 | 0.000 | 0.000 | 1.000 | 1.00 |
| V1 retrieval | 0.492 | 0.975 | 0.917 | 0.000 | 0.500 | 1.00 |
| V2 router + retrieval | 0.508 | 0.975 | 0.950 | 0.200 | 0.500 | 2.00 |
| V3 retrieval + verifier | 0.558 | 0.983 | 0.917 | 1.000 | 0.500 | 1.80 |
| V4 full workflow | 0.592 | 0.975 | 0.950 | 1.000 | 1.000 | 2.77 |
| A4 no formula guard | 0.567 | 0.975 | 0.950 | 1.000 | 1.000 | 2.87 |

## Talk Track

- Retrieval fixes grounding: citation hit rises from 0.000 to 0.917-0.950.
- The router helps scope detection and deterministic course-topic query construction.
- The verifier/misconception guard fixes false premises: 0.000 to 1.000 on false-premise cases.
- The calculation/formula guard fixes arithmetic substitutions: calculation correctness reaches 1.000.
- Full workflow handles unanswerable cases while preserving faithfulness.
- The tradeoff is cost/latency: V4 uses about 2.77 calls per question.

## Best Case Study

Question:

> Since logistic regression uses a sigmoid, why can it solve XOR without hidden layers?

Baseline:

> Claims logistic regression can solve XOR because sigmoid gives non-linear probability mapping.

Full workflow:

> The premise is incorrect: logistic regression has a linear decision boundary, and XOR is not linearly separable. A sigmoid maps a linear score to a probability; it does not add hidden nonlinear transformations.
