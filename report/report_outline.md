# Report Outline

Maximum 10 pages excluding references and appendices.

## 1. Introduction

- Track B selection
- Task: exam-aware COMP5541 revision QA over Lectures 2-9
- Motivation: students need grounded revision answers, not generic LLM guesses
- Main claim: routing, retrieval, verification, and deterministic formula checking improve reliability over a single-call baseline

## 2. Methodology

- Knowledge base and material scope
- Baseline V0
- H1 scope/question-type router
- H2 course evidence retrieval
- H3 misconception/faithfulness verifier
- H4 deterministic formula/calculation guard
- Full workflow diagram

## 3. Evaluation Design

- 60-case evaluation set
- Case categories: concept, formula, comparison, calculation, false premise, unanswerable
- Metrics: correctness, faithfulness, citation hit, false-premise detection, refusal accuracy, cost/latency
- Controlled comparison versions and ablations
- Model/API details and access date

## 4. Results and Analysis

- Main quantitative table by workflow version
- Category-level table
- Ablation table
- Calculation subgroup table
- Latency/cost table
- Representative success cases
- Representative failure cases
- Use `report/results_snapshot.md` as the starting point for the numeric tables.

## 5. Limitations and Responsible Use

- Course-material coverage limits
- Retrieval errors
- LLM verifier limitations
- Manual scoring subjectivity
- Privacy and academic integrity
- Cost/latency tradeoff

## 6. Conclusion

- Which harness helped most
- Whether combined workflow beat baseline
- When the workflow still fails

## 7. Acknowledgement and AI Disclosure

- Member-specific contributions
- AI tools used and purpose
