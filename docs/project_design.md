# Project Design

## Track Choice

We choose Track B because the project goal is to improve one shared LLM workflow and prove whether each added harness helps. The task is small enough to reproduce, but rich enough to evaluate carefully.

## Task

Given a COMP5541 revision question, answer using only the official course materials from Lectures 2-9 and available tutorials. The system should cite supporting evidence. If the materials do not contain enough information, it should say so instead of guessing.

## Why This Task Is Nontrivial

A single LLM call can answer from general machine-learning knowledge rather than the course materials. It can also:

- ignore course scope,
- miss lecture-specific phrasing,
- fail calculation-style questions,
- accept false premises,
- invent details for unanswerable questions.

## Harnesses

### H1: Scope and Question-Type Router

The router predicts whether the question is in scope, identifies the likely lecture/topic, labels the question type, and rewrites the query for retrieval.

Expected effect: improve retrieval hit rate and improve refusal behavior for out-of-scope questions.

### H2: Course Evidence Retrieval

The retrieval layer first converts official materials into page-aware overlapping chunks. Each chunk stores the source file, source group, page number, chunk number, stable chunk id, and text. The default backend uses local TF-IDF, which is cheap and reproducible. An optional Chroma vector-store backend supports semantic retrieval over the same chunks.

Expected effect: improve correctness and citation grounding by forcing the answer to use course evidence.

### H3: Misconception and Faithfulness Verifier

The verifier reviews the draft answer against retrieved evidence. It corrects false premises, removes unsupported claims, or refuses if evidence is insufficient.

Expected effect: improve faithfulness and false-premise detection, at the cost of extra latency and one more LLM call.

### H4: Deterministic Calculation and Formula Guard

For parseable formula questions, the workflow checks whether the question contains enough explicit symbolic or numeric information to compute the answer directly. Examples include linear-regression parameter count, ReLU derivative sign cases, chain-rule terms, convolution output shape, convolution parameter count, and pooling output shape.

Expected effect: reduce arithmetic and formula-substitution mistakes made by the local LLM, especially on calculation-style exam questions.

## Controlled Comparisons

All versions run on the same 60-case evaluation set:

- V0 baseline single call
- V1 retrieval only
- V2 router + retrieval
- V3 retrieval + verifier
- V4 full workflow
- A1 full without router
- A2 full without verifier
- A3 full with top-k = 2
- A4 full without calculation/formula guard

## Metrics

- Correctness: answer matches the reference answer.
- Faithfulness: answer claims are supported by retrieved course evidence.
- Citation hit: cited/retrieved evidence contains the answer.
- False-premise detection: system explicitly rejects incorrect premises.
- Refusal accuracy: system refuses unanswerable or out-of-scope questions.
- Cost/latency: LLM calls, estimated tokens, and wall-clock latency.

## Expected Findings

The likely pattern is:

- Retrieval improves correctness over baseline.
- Router improves complex and topic-specific questions by improving retrieval queries.
- Verifier improves faithfulness and false-premise cases.
- Combined workflow should be strongest overall but slower.
- Top-k ablation shows whether retrieval depth matters.
- Calculation-guard ablation shows whether deterministic formula execution reduces arithmetic errors.
