# Code README

Runnable code for the Track B experiments. The top-level [README](../README.md) explains
the project by Track B step; this file maps those steps to code and gives the finalized
run pipeline.

## Code Map (file ↔ Track B step)

| File | Track B step | What it implements |
|---|---|---|
| `src/ingest.py` | Step 3 (H2) | PDF/notebook/markdown → per-page text → 220-word chunks (45 overlap) → `chunks.jsonl` |
| `src/retriever.py` | Step 3 (H2) | TF-IDF retriever (default) and optional Chroma backend; evidence formatting with source/page/chunk id |
| `src/workflows.py` | Steps 2-3 | `baseline()` (V0), H1 `route_question()`, H2 answer prompt, H3 `retrieval_with_verifier()` + `misconception_guard()`, H4 `calculation_guard()`, V4 `full_workflow()` with ablation switches |
| `src/evaluate.py` | Step 6 | Runs all 9 versions on the same evaluation set; writes per-case results and a summary |
| `score_results.py` | Step 5 | Deterministic automatic scoring (correctness / faithfulness / citation / false-premise / refusal) |
| `src/llm.py` | infra | Provider clients (Ollama / Gemini / OpenAI-compatible / dummy), retries, response cache |
| `src/config.py`, `src/utils.py` | infra | Paths and small helpers |
| `run_ingest.py`, `run_evaluation.py`, `build_chroma.py` | entry points | Thin CLI wrappers |

Do not edit prompt strings in `src/workflows.py`: cached responses in `data/cache/` are
keyed by exact prompt text, and the recorded results were produced with these prompts.

## Finalized Run Pipeline

All recorded experiments use one model for every version: `qwen2.5:3b` via local
Ollama, temperature 0, TF-IDF retriever, top-k 5.

```bash
# 0. install
python3 -m pip install -r code/requirements.txt
ollama pull qwen2.5:3b

# 1. ingest course materials (rerun only if raw materials change)
python3 code/run_ingest.py

# 2. main run: balanced ablation set -> separate output files
cd code
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:3b TOP_K=5 TEMPERATURE=0 \
python3 run_evaluation.py \
  --eval-set ../records/evaluation_set_balanced.csv \
  --results ../records/results_balanced.csv \
  --summary ../records/summary_balanced.csv
cd ..

# 3. automatic scoring
PYTHONPATH=code python3 code/score_results.py \
  --results records/results_balanced.csv \
  --summary records/summary_balanced.csv
```

The broad robustness set (`records/evaluation_set.csv` → `records/results.csv`) is
already run and archived; rerunning `python3 code/run_evaluation.py` with the same env
vars reproduces it from cache.

Quick no-API dry run: `LLM_PROVIDER=dummy python3 code/run_evaluation.py --limit 2`.

## Optional: Other Backends and Providers

Semantic retrieval via Chroma (same interface, measured the same way):

```bash
python3 -m pip install -r code/requirements-vector.txt
python3 code/build_chroma.py --rebuild
RETRIEVER_BACKEND=chroma LLM_PROVIDER=ollama ... python3 code/run_evaluation.py
```

Hosted providers for the optional cross-model check (record exact model, route, and
access date in `records/model_metadata.md`):

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=... GEMINI_MODEL=gemini-2.5-flash-lite ...
LLM_PROVIDER=openai_compatible OPENAI_COMPATIBLE_BASE_URL=... OPENAI_COMPATIBLE_API_KEY=... ...
```

Responses are cached per provider+model+prompt, so a quota-limited run can be resumed
across days without repeating calls.

## Output Files

- `data/processed/text/`, `data/processed/index/chunks.jsonl` — ingestion outputs
- `data/cache/` — cached LLM responses (deterministic reruns)
- `records/results*.csv` — one row per case × version, with score columns
- `records/summary*.csv` — per-version aggregate metrics

Important score columns in `records/results*.csv` (filled by `score_results.py`, then
manually sanity-checked per [docs/evaluation_manual_review.md](../docs/evaluation_manual_review.md)):
`correctness_score` (0/0.5/1), `faithfulness_score` (0/0.5/1), `citation_hit` (0/1),
`false_premise_detected` (0/1 on false-premise cases), `refusal_correct` (0/1 on
unanswerable cases), `human_comment`.
