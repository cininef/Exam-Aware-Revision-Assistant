# Report Draft Skeleton

## Title

Exam-Aware COMP5541 Revision Assistant: Improving Course-Material QA with Scope Detection, Evidence Retrieval, and Misconception Verification

## Abstract

We evaluate whether a small LLM workflow can improve course-specific revision question answering over a single-call baseline. The system answers COMP5541 questions using official materials from Lectures 2-9. We compare a baseline, four harnesses, a combined workflow, and ablations on a fixed 60-case evaluation set. The main metrics are correctness, faithfulness, citation hit rate, false-premise detection, refusal accuracy, and cost/latency.

## Key Result

The full workflow improved automatic correctness from 0.408 to 0.592, faithfulness from 0.608 to 0.975, and citation hit rate from 0.000 to 0.950. The verifier/misconception guard improved false-premise detection from 0.000 to 1.000. The deterministic calculation guard raised calculation-case correctness to 1.000; removing it reduced calculation correctness to 0.700. The tradeoff is higher latency and more LLM calls.

## Main Tables to Fill

### Overall Performance

| Version | Correctness | Faithfulness | Citation Hit | False-Premise Detection | Refusal Accuracy | Avg Calls | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline | 0.408 | 0.608 | 0.000 | 0.000 | 1.000 | 1.00 | 2.00 |
| V1 retrieval | 0.492 | 0.975 | 0.917 | 0.000 | 0.500 | 1.00 | 3.43 |
| V2 router + retrieval | 0.508 | 0.975 | 0.950 | 0.200 | 0.500 | 2.00 | 4.52 |
| V3 retrieval + verifier | 0.558 | 0.983 | 0.917 | 1.000 | 0.500 | 1.80 | 6.00 |
| V4 full | 0.592 | 0.975 | 0.950 | 1.000 | 1.000 | 2.77 | 6.78 |

### Ablation

| Version | What is removed/changed | Expected effect | Observed effect |
|---|---|---|---|
| A1 full without router | Router/query rewrite | Lower retrieval quality on complex questions | Correctness 0.583, close to V4; router mainly helps refusal/scope framing |
| A2 full without verifier | Verifier | Lower faithfulness and false-premise handling | False-premise detection drops to 0.200 |
| A3 full top-k=2 | Retrieval depth | Lower citation hit on broad questions | Citation hit drops from 0.950 to 0.817 |
| A4 full without calculation guard | Formula execution | More arithmetic/formula mistakes | Overall correctness drops to 0.567; calculation correctness drops to 0.700 |

### Manual Review Status

The automatic correctness score is conservative because it uses deterministic keyword matching. Before final submission, manually review the representative cases listed in `docs/evaluation_manual_review.md`, especially false-premise and calculation examples.
