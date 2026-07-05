# Presentation Script Draft

## Opening

Our Track B task is a COMP5541 revision QA system. The baseline is a single LLM call. The problem is that a single call often answers from general machine-learning knowledge instead of the course materials, and it may accept false premises.

## Workflow

We added four small harnesses. First, a router detects scope, topic, and question type. Second, a retriever searches official lecture and tutorial chunks. Third, a verifier checks whether the answer is supported and whether the question contains a false premise. Fourth, a deterministic formula guard handles parseable calculation questions so the LLM does not have to do arithmetic from free text.

## Evaluation

We built 60 fixed cases across Lectures 2-9. The categories include concept, formula, comparison, calculation, false premise, and unanswerable questions. Every version runs on the same cases.

## Evidence

We compare baseline, each harness, the full workflow, and ablations. This lets us ask not just whether the full system is better, but which step actually contributes.

## Conclusion

The full workflow improves automatic correctness from 0.408 to 0.592, citation hit rate from 0.000 to 0.950, and false-premise detection from 0.000 to 1.000. The formula guard fixes calculation failures, raising calculation-case correctness to 1.000. The main tradeoff is latency and call count: V4 uses about 2.77 calls per question. The remaining weakness is concept-answer completeness, where a small local model may omit expected course terms even when the answer is mostly correct.
