# Scoring Rubric for Evaluation Records

## Correctness

- 1: The core answer is correct and complete enough for the question.
- 0.5: Partially correct, but missing an important condition, formula part, or caveat.
- 0: Incorrect, contradicts the reference, or refuses when the answer is available.

## Faithfulness

- 1: All substantive claims are supported by retrieved course evidence.
- 0.5: Main answer is supported, but includes minor unsupported wording or extra claims.
- 0: The answer relies on unsupported general knowledge, hallucinated details, or contradicts evidence.

## Citation Hit

- 1: Retrieved/cited evidence contains the information needed to answer the question.
- 0: Retrieved/cited evidence does not contain the needed information.

## False-Premise Detection

Only score this for cases where `false_premise=yes`.

- 1: The answer explicitly identifies and corrects the false premise.
- 0: The answer accepts the false premise or ignores it.

## Refusal Accuracy

Only score this for `unanswerable` cases.

- 1: The answer says the course materials do not contain enough information.
- 0: The answer invents or overclaims.

## Reporting Metrics

For each workflow version:

- Average correctness
- Average faithfulness
- Citation hit rate
- False-premise detection rate
- Refusal accuracy
- Average LLM calls
- Average latency
- Estimated input/output tokens

## Metric-to-Harness Mapping

The evaluation uses the same cases for all workflow versions so each harness can be compared by controlled ablation.

| Metric | What it measures | Main harness tested | Computation |
|---|---|---|---|
| Correctness score | Whether the answer contains the reference answer's required concepts | All harnesses | 0/0.5/1 by required term coverage, then averaged |
| Faithfulness score | Whether answer claims are grounded in retrieved evidence | H2 retrieval, H3 verifier | 0/0.5/1 by evidence use and refusal behavior |
| Citation hit@k | Whether retrieved top-k evidence contains the expected source/topic | H2 retrieval, H1 query routing | 1 if retrieved source matches source hint, else 0 |
| False-premise detection | Whether the answer explicitly rejects a false premise | H1 router, H3 verifier | Rate over `false_premise=yes` cases |
| Refusal accuracy | Whether the workflow refuses unanswerable/out-of-scope questions | H1 scope router, H3 verifier | Rate over `expected_behavior=refuse` cases |
| Calculation/formula accuracy | Whether numeric/formula answers are correct | H4 formula guard | Correctness averaged over calculation/formula cases |
| Format accuracy | Whether final answers include an evidence line and exam-style structure | H2/H3 final answer prompt | Binary/manual check in representative cases |
| Cost proxy | Number of LLM calls and estimated tokens | Whole workflow | Average `llm_calls`, input tokens, output tokens |
| Latency | Runtime overhead of harnesses | Whole workflow | Average seconds per case |
| Human-checked score | Manual sanity-check of representative or full V0/V4 cases | All harnesses | 0/0.5/1 using this rubric, reported separately from auto-score |

## Ablation Matrix

| Version | Purpose | Harnesses active |
|---|---|---|
| `V0_baseline_single_call` | Plain LLM baseline | none |
| `V1_retrieval_only` | Test retrieval alone | H2 |
| `V2_router_rewrite_retrieval` | Test router/query rewrite added to retrieval | H1 + H2 |
| `V3_retrieval_verifier` | Test verifier path without LLM router | H2 + H3, with deterministic guards in the current implementation |
| `V4_full_router_retrieval_verifier` | Final combined workflow | H1 + H2 + H3 + H4 |
| `A1_full_without_router` | Remove router from final workflow | H2 + H3 + H4 |
| `A2_full_without_verifier` | Remove verifier path from final workflow | H1 + H2 |
| `A3_full_topk2` | Control retrieval depth | H1 + H2(top-k=2) + H3 + H4 |
| `A4_full_without_calculation_guard` | Remove formula/calculation guard | H1 + H2 + H3 |

Recommended comparisons:

- H2 retrieval contribution: `V1 - V0`
- H1 router contribution: `V2 - V1`, plus `V4 - A1`
- H3 verifier contribution: `A4 - V2` or `V4 - A2`, explained as verifier-path contribution
- H4 formula guard contribution: `V4 - A4`, especially on calculation/formula cases
- Retrieval depth control: `V4 - A3`
- Final workflow value: `V4 - V0`
