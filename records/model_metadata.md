# Model and Experiment Metadata

## Final Experiment Run

- Date: 2026-07-02
- Updated run: 2026-07-03
- Provider: Ollama local runtime
- Model: `qwen2.5:3b`
- Access route: local Ollama API at `http://localhost:11434`
- Temperature: `0`
- Retriever backend: `tfidf`
- Top-k evidence: `5`
- Router policy: LLM-assisted scope/type classification with deterministic course-topic retrieval query
- Additional guard: deterministic calculation/formula checking for parseable formula questions
- Evaluation set: `records/evaluation_set.csv`
- Number of cases: `60`
- Workflow versions per case: `9`
- Total result rows: `540`

## Notes

Gemini free tier was tested but the project quota for `gemini-2.5-flash-lite` was limited to 20 free requests per day, which is not enough for the full experiment. The final run therefore uses a local no-cost model so the full controlled comparison can be reproduced without paid API usage.

An attempted `qwen3:4b` Ollama download was cancelled because the network speed was too low for a same-session sanity check. The final recorded full run remains `qwen2.5:3b`.

The scoring columns in `records/results.csv` were filled by `code/score_results.py` using deterministic heuristics. The manual sanity-check protocol is in `docs/evaluation_manual_review.md`; use it before treating automatic correctness as final human-grade accuracy.
