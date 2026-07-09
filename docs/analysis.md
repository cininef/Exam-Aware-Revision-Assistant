# Analysis and Records Map

This file maps the Track B "complete records" requirement to the exact files, workflow versions, and score columns in this repository.

## Complete Records Checklist

| Requirement from brief | Our file(s) | Exact row/column meaning |
|---|---|---|
| Evaluation set | `records/evaluation_set_balanced.csv`; `records/evaluation_set.csv` | Each row is a test case with `case_id`, `category`, `lecture_scope`, `question`, `reference_answer`, `source_hint`, `required_terms`, `false_premise`, and `expected_behavior`. |
| Baseline outputs | `records/results_balanced.csv`; broad check in `records/results.csv` | Rows where `version = V0_baseline_single_call`. This is the single-call LLM baseline with no retrieval, router, verifier, or formula guard. |
| Each harness output | `records/results_balanced.csv` | Rows where `version` is `V1`, `V2`, `V3`, and related ablations. These show how the workflow behaves as harnesses are added or removed. |
| Final workflow outputs | `records/results_balanced.csv` | Rows where `version = V4_full_router_retrieval_verifier`. This is the final workflow with router, retrieval, verifier, and formula guard. |
| Ablation/control outputs | `records/results_balanced.csv` | Rows where `version = A1_full_without_router`, `A2_full_without_verifier`, `A3_full_topk2`, or `A4_full_without_calculation_guard`. |
| Scores | `records/results_balanced.csv`; summarized in `records/summary_balanced.csv` | Per-row score columns are `correctness_score`, `faithfulness_score`, `citation_hit`, `false_premise_detected`, `refusal_correct`. Summary means are in `summary_balanced.csv`. |
| Judge prompt if used | Not used | We did not use LLM-as-judge. Scores are deterministic automatic scores plus human sanity-check notes. |
| Human sanity-check notes | `records/manual_review_balanced.csv`; protocol in `docs/evaluation_manual_review.md` | Manual review subset: all V0 rows, all V4 rows, and V1 rows for `retrieval_concept` cases. Fill `human_*` columns and comments. |
| Cost/latency observations | `records/results_balanced.csv`; `records/summary_balanced.csv` | `latency_s`, `llm_calls`, `input_tokens_est`, and `output_tokens_est` record runtime and cost proxies. |
| Model/access metadata | `records/model_metadata.md`; `docs/ai_disclosure.md` | Records provider, model, access route, date, temperature, retrieval backend, and AI-tool disclosure. |

## Workflow Version Map

| Version | What it is | Harnesses active | Why it is included |
|---|---|---|---|
| `V0_baseline_single_call` | Plain LLM baseline | none | Shows what a single LLM call can and cannot do. |
| `V1_retrieval_only` | Retrieve evidence, then answer | H2 retrieval | Tests whether RAG improves grounding and citations. |
| `V2_router_rewrite_retrieval` | Router/query rewrite plus retrieval | H1 router + H2 retrieval | Tests whether scope/type routing and course-aware query formation help retrieval. |
| `V3_retrieval_verifier` | Retrieval plus verifier path | H2 retrieval + H3 verifier, with deterministic guards in the current implementation | Tests verification without the LLM router. |
| `V4_full_router_retrieval_verifier` | Final workflow | H1 router + H2 retrieval + H3 verifier + H4 formula guard | Main system reported as final result. |
| `A1_full_without_router` | Final workflow without router | H2 + H3 + H4 | Ablation for H1 router. |
| `A2_full_without_verifier` | Final workflow without verifier path | H1 + H2 | Ablation for H3 verifier path. The formula guard also drops out because it is inside the verifier path. |
| `A3_full_topk2` | Final workflow with top-k = 2 | H1 + H2(top-k=2) + H3 + H4 | Retrieval-depth control. |
| `A4_full_without_calculation_guard` | Final workflow without formula guard | H1 + H2 + H3 | Ablation for H4 formula/calculation guard. |

## Score Columns

| Column | Range | How it is produced | What it means |
|---|---:|---|---|
| `correctness_score` | 0, 0.5, 1 | `code/score_results.py` checks how many `required_terms` appear in the answer. Refusal cases are correct if the answer refuses. | Whether the answer covers the reference answer's key facts. |
| `faithfulness_score` | 0, 0.5, 1 | `code/score_results.py` checks whether the answer uses retrieved evidence/citations or correctly refuses. | Whether the answer appears grounded in course evidence. This is an automatic proxy, not a perfect groundedness judge. |
| `citation_hit` | 0 or 1 | `code/score_results.py` compares `retrieved_sources` with `source_hint`. | Whether top-k retrieval found the expected source/topic. |
| `false_premise_detected` | 0 or 1, blank for non-false-premise cases | Computed only when `false_premise = yes`. The answer must explicitly signal an incorrect premise. | Whether the workflow catches misleading questions. |
| `refusal_correct` | 0 or 1, blank for answerable cases | Computed only when `expected_behavior = refuse`. The answer must say not enough information/cannot answer. | Whether the workflow refuses unavailable/private/future-exam questions. |
| `latency_s` | seconds | Recorded during workflow execution. | Runtime cost. |
| `llm_calls` | integer | Counted by workflow code. | API/local model call cost. |
| `input_tokens_est`, `output_tokens_est` | estimated token counts | Estimated by `code/src/utils.py`. | Cost proxy for API-based runs. |
| `human_comment` | text | Initially auto-filled, manually edited during review. | Short note for human sanity-check and error analysis. |

## Current Balanced-Set Summary

Source file: `records/summary_balanced.csv`.

| Version | Correctness | Faithfulness | Citation hit | Avg latency (s) | Avg LLM calls | False-premise detection | Refusal correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| `V0_baseline_single_call` | 0.533 | 0.633 | 0.000 | 2.699 | 1.000 | 0.000 | 1.000 |
| `V1_retrieval_only` | 0.467 | 0.983 | 0.817 | 5.228 | 1.000 | 0.125 | 0.571 |
| `V2_router_rewrite_retrieval` | 0.467 | 0.983 | 0.833 | 6.579 | 2.000 | 0.125 | 0.714 |
| `V3_retrieval_verifier` | 0.550 | 0.992 | 0.817 | 8.485 | 1.783 | 1.000 | 0.571 |
| `V4_full_router_retrieval_verifier` | 0.575 | 0.992 | 0.833 | 9.450 | 2.717 | 1.000 | 0.857 |
| `A1_full_without_router` | 0.567 | 0.992 | 0.833 | 8.266 | 1.783 | 1.000 | 0.714 |
| `A2_full_without_verifier` | 0.483 | 0.983 | 0.833 | 6.424 | 2.000 | 0.125 | 0.857 |
| `A3_full_topk2` | 0.567 | 1.000 | 0.767 | 8.002 | 2.717 | 1.000 | 0.857 |
| `A4_full_without_calculation_guard` | 0.525 | 0.992 | 0.833 | 9.964 | 2.833 | 1.000 | 0.857 |

## Main Comparisons

Use the same evaluation set and same model/settings for every comparison.

| Comparison | What it tests | Current result |
|---|---|---|
| `V1 - V0` | H2 retrieval | Faithfulness improves from 0.633 to 0.983 and citation hit from 0.000 to 0.817. Automatic correctness drops from 0.533 to 0.467, which is why V1 retrieval-concept rows are included in manual review. |
| `V2 - V1` | H1 router/query rewrite on retrieval-only pipeline | Citation hit improves slightly from 0.817 to 0.833; refusal accuracy improves from 0.571 to 0.714. Correctness is unchanged. |
| `V4 - A1` | H1 router inside the final workflow | Correctness improves from 0.567 to 0.575 and refusal accuracy from 0.714 to 0.857, with extra LLM calls. |
| `V4 - A2` | H3 verifier path | Correctness improves from 0.483 to 0.575 and false-premise detection from 0.125 to 1.000. |
| `V4 - A4` | H4 formula/calculation guard | Correctness improves from 0.525 to 0.575. This comparison should also be reported on calculation/formula cases specifically. |
| `V4 - A3` | Retrieval depth top-k=5 vs top-k=2 | Citation hit improves from 0.767 to 0.833 and correctness from 0.567 to 0.575. |
| `V4 - V0` | Final workflow vs baseline | Correctness improves from 0.533 to 0.575, faithfulness from 0.633 to 0.992, citation hit from 0.000 to 0.833, and false-premise detection from 0.000 to 1.000. |

## How to Explain the Metrics

The automatic scores are not a claim that the system has perfect human-level grading. They are reproducible proxies:

- Correctness uses required-term coverage, so it can under-score semantically correct but differently worded answers.
- Faithfulness checks evidence usage and is therefore a proxy for groundedness, not a full factuality judge.
- Citation hit measures whether the retrieval stage reached the expected lecture/source.
- Human review is used to catch automatic-score edge cases and to support qualitative error analysis.

This is why the final report should present automatic scores as the quantitative backbone, then use manual review notes for representative successes, failures, and cases where automatic scoring is conservative.

## Files to Open During Report Writing

- Main test cases: `records/evaluation_set_balanced.csv`
- Main per-case outputs and scores: `records/results_balanced.csv`
- Main summary table: `records/summary_balanced.csv`
- Manual review subset: `records/manual_review_balanced.csv`
- Broad robustness outputs: `records/results.csv`
- Broad robustness summary: `records/summary.csv`
- Scoring rules: `docs/scoring_rubric.md`
- Manual review protocol: `docs/evaluation_manual_review.md`
- Material source list: `docs/material_manifest.md` and the README "Materials Used" section
