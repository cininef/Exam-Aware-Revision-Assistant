# Evaluation Set Design

This project uses two complementary evaluation sets.

## Broad Coverage Set

`records/evaluation_set.csv` is the original broad course-coverage set. It covers Lectures 2-9 and tutorials, with many concept questions. It is useful for measuring general course QA behavior, but it is intentionally broad rather than balanced by workflow component.

Current distribution:

- concept: 34
- comparison: 12
- calculation: 5
- false_premise: 5
- formula: 2
- unanswerable: 2

## Harness-Balanced Set

`records/evaluation_set_balanced.csv` is designed for Track B ablation. It contains 60 cases split into four equal groups of 15:

- `retrieval_concept`: direct course-material questions testing whether retrieval grounds factual answers.
- `reasoning_comparison`: "why" and comparison questions testing router/query formulation and evidence-grounded explanation.
- `calculation_formula`: formula, shape, parameter, and small symbolic questions testing the deterministic formula guard.
- `verification_scope`: false-premise and unanswerable questions testing scope routing, misconception correction, and refusal behavior.

This balanced design is better for proving each harness contribution because each workflow component has enough targeted cases to evaluate:

- H1 Scope Router / Question-Type Detector: mainly `reasoning_comparison` and `verification_scope`.
- H2 Course RAG Retriever: mainly `retrieval_concept` and all answerable groups.
- H3 Evidence Verifier / Misconception Guard: mainly false-premise cases in `verification_scope`.
- H4 Formula / Calculation Guard: mainly `calculation_formula`.

The balanced set should be used as the main ablation evidence if time permits. The broad set can remain as a robustness check.

## Metrics

The metrics are chosen to match the failure modes each harness is supposed to control:

| Metric | Why it is needed | Harness relevance |
|---|---|---|
| Correctness score | Measures whether the answer solves the revision question | Overall task success |
| Faithfulness score | Measures whether answer claims are grounded in course evidence | Retrieval and verifier |
| Citation hit@k | Measures whether top-k retrieved chunks include the expected source/topic | Retriever and router |
| False-premise detection | Measures whether the workflow rejects misleading premises | Router and verifier |
| Refusal accuracy | Measures whether out-of-scope questions are refused | Scope router and verifier |
| Calculation/formula accuracy | Measures formula, shape, and parameter-count reliability | Formula guard |
| Cost proxy | Measures LLM calls and token estimates | Workflow overhead |
| Latency | Measures runtime trade-off | Workflow overhead |
| Human-checked score | Catches cases where keyword auto-scoring is too strict | Final evidence quality |

## Comparison Plan

All workflow versions are run on the same evaluation set. This makes the comparison controlled:

| Comparison | Question answered |
|---|---|
| `V1_retrieval_only - V0_baseline_single_call` | Does retrieval improve a plain LLM? |
| `V2_router_rewrite_retrieval - V1_retrieval_only` | Does the router/query rewrite improve retrieval usage? |
| `A4_full_without_calculation_guard - V2_router_rewrite_retrieval` | Does the verifier/misconception guard add value beyond routed retrieval? |
| `V4_full_router_retrieval_verifier - A4_full_without_calculation_guard` | Does the formula guard improve calculation/formula cases? |
| `V4_full_router_retrieval_verifier - A3_full_topk2` | Is top-k=5 better than a smaller evidence budget? |
| `V4_full_router_retrieval_verifier - V0_baseline_single_call` | Does the final combined workflow improve over the baseline? |
