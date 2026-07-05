# 7-Minute Slides Outline

## Slide 1: Problem and Claim

Single-call LLMs can give confident but unsupported course-specific answers. We test whether an exam-aware workflow improves COMP5541 revision QA.

## Slide 2: Task and Data

- Task: answer COMP5541 L2-L9 revision questions using official course materials
- Knowledge base: lectures/tutorials
- Evaluation: 60 fixed cases from concept, formula, calculation, comparison, false-premise, unanswerable categories

## Slide 3: Workflow

Show diagram:

question -> router/query rewrite -> retrieve evidence -> answer -> verifier/revise -> formula guard -> final answer

## Slide 4: Experimental Versions

- V0 baseline
- V1 retrieval
- V2 router + retrieval
- V3 retrieval + verifier
- V4 full
- Ablations: no router, no verifier, top-k=2, no formula guard

## Slide 5: Metrics

- Correctness
- Faithfulness
- Citation hit
- False-premise detection
- Refusal accuracy
- Cost/latency

## Slide 6: Results Table

Use the compact table in `slides/results_slide_content.md`.

## Slide 7: Case Study

Use one false-premise example:

"Since logistic regression uses sigmoid, why can it solve XOR without hidden layers?"

Show baseline failure, retrieval evidence, verifier correction.

## Slide 8: Ablation and Tradeoff

Explain which harness earned its place and what cost it added.

Include the calculation-guard result: calculation correctness reaches 1.000, while removing the guard drops it to 0.700.

## Slide 9: Limitations and Conclusion

- Retrieval misses can still happen
- Verifier is not perfect
- Course scope is limited to available materials
- Full workflow improves reliability but costs more calls
