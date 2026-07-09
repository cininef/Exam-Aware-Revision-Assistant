# ExamGuard-RAG: An Exam-Aware Course-Material QA Workflow for COMP5541

**Track:** Track B - Design and Improve a Small LLM Workflow  
**Team members:** [Name / Student ID], [Name / Student ID], [Name / Student ID]  
**Main artifacts:** `records/results_balanced.csv`, `records/summary_balanced.csv`, `records/manual_review_balanced.csv`, `code/`, `README.md`

## Abstract

This project studies whether a small, controlled LLM workflow can improve course-specific question answering over a single-call baseline. The task is COMP5541 exam revision QA: given a student question, answer only from official course materials from Lectures 2-9 and tutorials, cite retrieved evidence, correct false premises, and refuse questions that the materials cannot answer. We designed four harnesses: a scope/question-type router, a course-material retriever, an evidence verifier with misconception handling, and a deterministic formula/calculation guard. We evaluated nine workflow versions on a balanced 60-case evaluation set and a broader 60-case robustness set. On the balanced set, the final workflow improves faithfulness from 0.633 to 0.992, citation hit rate from 0.000 to 0.833, and false-premise detection from 0.000 to 1.000 compared with the single-call baseline. Automatic correctness improves more modestly from 0.533 to 0.575, and our analysis shows that keyword scoring undercounts several semantically correct but differently worded answers. The main conclusion is not that the workflow solves all course QA, but that the harnesses measurably improve grounding, evidence traceability, false-premise handling, and calculation reliability, at the cost of more LLM calls and latency.

## 1. Introduction and Need

We chose Track B because the goal is not to compare many LLMs, but to improve one shared LLM-based system and prove whether the added workflow controls actually help. Our shared task is:

> Given a COMP5541 revision question, answer using only official course materials from Lectures 2-9 and tutorials. The answer should cite evidence, correct false premises, and refuse unavailable/private/future-exam information.

This task matters because students often ask revision questions in natural language, but a plain LLM call may answer from general machine-learning knowledge instead of the specific course slides. In an exam setting, that is unsafe: the system may use terminology not emphasized in the course, miss the slide-specific caveat, accept a misleading premise, or invent an answer about future exam questions.

A good output means:

- correct enough for the course context,
- grounded in retrieved course evidence,
- traceable to slide/tutorial sources,
- explicit when the premise is wrong,
- explicit when the answer is unavailable in the course materials,
- reliable on small formula and calculation questions.

### Track B Step Map

| Track B requirement | What we did | Main evidence file |
|---|---|---|
| 1. State the need | Course-specific exam revision QA over official COMP5541 materials. | `README.md`, this report Section 1 |
| 2. Build a baseline | `V0_baseline_single_call`: one LLM call, no harness. | `records/results_balanced.csv` rows with `version=V0_baseline_single_call` |
| 3. Build harnesses and final workflow | H1 router, H2 retrieval, H3 verifier, H4 formula guard, combined as V4. | `code/src/workflows.py`, `code/src/retriever.py` |
| 4. Design evaluation set | Balanced 60-case set plus broad 60-case robustness set. | `records/evaluation_set_balanced.csv`, `records/evaluation_set.csv` |
| 5. Choose metrics | Correctness, faithfulness, citation hit@k, false-premise detection, refusal accuracy, cost/latency, human sanity check. | `docs/scoring_rubric.md`, `code/score_results.py` |
| 6. Run comparisons and ablations | V0, V1, V2, V3, V4, A1-A4 all run on the same set. | `records/results_balanced.csv`, `records/summary_balanced.csv` |
| 7. Analyze honestly | We report both wins and weak points: strong grounding/safety gains, modest keyword correctness gain, latency cost, scoring limitations. | this report Sections 6-9, `docs/analysis.md` |

The initial design idea came from identifying these failure modes. A single LLM call fails because it is "unharnessed": it answers directly from model memory. Each harness was added to control one concrete failure mode:

- if the model answers from memory, retrieve evidence first;
- if the question is misleading or out of scope, route and verify it;
- if the model cites weak evidence, show exact source/page/chunk;
- if the model makes arithmetic mistakes, use deterministic formula execution.

## 2. Data, Scope, and Preprocessing

### 2.1 Materials Used

The retrieval knowledge base contains official course materials only:

- Course slides: Lecture 2 to Lecture 9.
- Tutorials: linear algebra, calculus, ML development, PyTorch, CNN image classification, RNN/LSTM name classification, and GAN MNIST tutorial.
- Assessment brief: included only as course context.

The main raw source directories are:

- `data/raw/course_slides/`
- `data/raw/tutorials/`
- `data/raw/assessment_briefs/`

Files under `data/raw/evaluation_sources/` are used for designing test cases and reference answers, not as the main answer evidence. Project brief files are used only to understand the Track B requirements. Personal assignment reports, private notes, and previous generated outputs are excluded.

### 2.2 Preprocessing Pipeline

The ingestion pipeline is implemented in `code/src/ingest.py`.

Processing steps:

1. Convert PDFs, markdown/text files, and notebooks into plain text.
2. Save extracted text to `data/processed/text/`.
3. For PDFs, preserve page boundaries from `pdftotext`.
4. Normalize whitespace.
5. Split each page into overlapping chunks.
6. Save chunks to `data/processed/index/chunks.jsonl`.

Default chunking:

- 220 words per chunk,
- 45-word overlap,
- very short chunks under 80 characters discarded.

Each chunk stores:

```text
chunk_id, source_file, source_group, page, chunk_number, text
```

This preprocessing is important because the workflow needs not only an answer, but also a way to show where the answer came from. Page-aware chunks allow the demo and records to display slide/tutorial source, page, chunk id, and retrieval score.

## 3. Baseline and Harness Design

### 3.1 Baseline: V0 Single-Call LLM

The baseline is `V0_baseline_single_call`, implemented by `baseline()` in `code/src/workflows.py`.

```text
Question -> LLM -> Answer
```

It uses the same model and temperature as other versions, but has no retrieval, router, verifier, or formula guard. This is the system every harness must beat.

### 3.2 H1: Scope Router / Question-Type Detector

The router is implemented by `route_question()` and `heuristic_route()`.

It predicts:

- whether the question is in scope,
- the question type: concept, formula, comparison, calculation, false premise, or unanswerable,
- the likely lecture/topic,
- a course-aware retrieval query.

Why we designed it:

The same natural-language question can require different behavior. "Why does CNN help?" needs an explanatory answer. "Since CNNs are naturally rotation invariant..." should be treated as a false premise. "What exact questions will appear in the final exam?" should be refused. The router gives the rest of the workflow this control signal instead of treating every prompt as ordinary QA.

Implementation detail:

The router can call the LLM, but the retrieval query is kept deterministic using a topic keyword table. We made this choice after observing that a small local model sometimes rewrites course terms into plausible but non-canonical labels, which hurts sparse retrieval.

### 3.3 H2: Course Evidence Retrieval / RAG

The retriever is implemented in `code/src/retriever.py`. The default backend is TF-IDF over the processed chunks. An optional Chroma vector backend is also provided, but the recorded run uses TF-IDF for reproducibility and zero API cost.

The answer prompt inserts top-k evidence chunks:

```text
[Evidence 1] source file, page, chunk id, score
chunk text

[Evidence 2] ...
```

The model is instructed to answer only from the evidence, use exam-style wording, show formulas for calculation questions, and end with an evidence line.

Why we designed it:

RAG is the most direct harness for a course-material QA task. It reduces reliance on model memory, creates evidence traceability, and makes citation hit@k measurable.

### 3.4 H3: Evidence Verifier / Misconception Guard

The verifier is implemented by `retrieval_with_verifier()` and `misconception_guard()`.

Workflow:

```text
retrieve evidence -> draft answer -> verifier checks draft against evidence -> final answer
```

The verifier checks:

- whether the answer is supported by evidence,
- whether a false premise should be corrected,
- whether the answer should refuse due to insufficient evidence,
- whether the model is overcorrecting an ordinary question as a false premise.

We also include a small deterministic misconception guard for recurring course misconceptions, such as:

- logistic regression and XOR,
- polynomial degree always improving test performance,
- hidden units always improving generalization,
- CNNs being naturally rotation invariant,
- tuning on the test set.

Why we designed it:

Retrieval alone can still pass the wrong draft through. In our task, the most damaging errors are not just missing facts, but accepting a misleading premise. H3 is designed to catch that.

### 3.5 H4: Deterministic Formula / Calculation Guard

The formula guard is implemented by `calculation_guard()`.

It fires only when the question contains enough explicit numeric or symbolic information. Covered cases include:

- linear-regression parameter count with a bias term,
- ReLU derivative for negative inputs,
- chain-rule term for a simple neuron,
- convolution output width formula,
- convolution output shape and parameter count,
- pooling output shape.

Why we designed it:

Small local LLMs often make arithmetic and formula-substitution errors even when the retrieved evidence contains the right formula. Calculation questions have deterministic answers, so using a rule-based solver is more reliable than asking the LLM to freely generate the arithmetic.

### 3.6 Final Workflow: V4

The final workflow is `V4_full_router_retrieval_verifier`.

```text
Question
-> H1 Scope Router / Question-Type Detector
-> H2 Course Evidence Retrieval
-> Draft Answer
-> H4 Formula/Calculation Guard when applicable
-> H3 Evidence Verifier / Misconception Guard
-> Final Answer
```

V4 is not a harness by itself. It is the combined workflow that stacks the harnesses.

## 4. Evaluation Design

### 4.1 Evaluation Sets

We use two fixed 60-case evaluation sets.

The main set is `records/evaluation_set_balanced.csv`. It is balanced by harness target:

| Category | Cases | Main purpose |
|---|---:|---|
| `retrieval_concept` | 15 | Test whether H2 retrieval finds course facts. |
| `reasoning_comparison` | 15 | Test H1 router/query and evidence-grounded explanations. |
| `calculation_formula` | 15 | Test H4 formula/calculation guard. |
| `verification_scope` | 15 | Test H1 scope behavior and H3 false-premise/refusal handling. |

The broad robustness set is `records/evaluation_set.csv`. It covers Lectures 2-9 more broadly, but is less balanced: it contains many concept questions. We report balanced-set results as the main ablation evidence and keep the broad set as a robustness check.

### 4.2 How the Balanced Questions Were Made

The balanced set was designed to make each harness testable rather than to cherry-pick easy examples.

For `retrieval_concept`, we selected concepts directly from the slides, such as softmax, dropout, KNN, RNN hidden states, and generative models. Each case has a reference answer, a source hint, and required terms.

Example:

```text
Question: What does softmax convert raw class scores into?
Reference answer: Softmax converts raw scores into positive class probabilities that sum to one.
Source hint: Lecture 3 softmax
Required terms: raw scores; probabilities; sum to one
```

For `reasoning_comparison`, we created "why/how/compare" questions that require mechanism-level explanation. These are harder than definitions because the answer must connect multiple course terms.

Example:

```text
Question: Why are convolutional layers more parameter-efficient than fully connected layers for images?
Reference answer: Convolutional layers use local connectivity and weight sharing instead of separate weights for every input-output pair.
Required terms: local connectivity; weight sharing; fewer parameters
```

For `calculation_formula`, we converted slide formulas into small exam-style calculations with deterministic answers.

Example:

```text
Question: For a 32x32x3 image with 16 filters of size 3x3, stride 1, padding 1, and no bias, what is the output shape and parameter count?
Reference answer: 32x32x16 and 432 parameters.
Required terms: 32x32x16; 432
```

For `verification_scope`, we created false-premise and refusal cases. These check whether the workflow corrects misleading assumptions and refuses unavailable information.

Example false-premise case:

```text
Question: Since logistic regression uses sigmoid, why can it solve XOR without hidden layers?
Expected behavior: correct_premise
```

Example refusal case:

```text
Question: What exact questions will appear in the COMP5541 final exam?
Expected behavior: refuse
```

### 4.3 Metrics

Scores are generated by `code/score_results.py`.

| Metric | How it is scored | Why it fits the task |
|---|---|---|
| Correctness | 0/0.5/1 by required-term coverage, with refusal cases treated separately | Measures answer completeness against reference answers. |
| Faithfulness | 0/0.5/1 based on evidence use and refusal behavior | Measures whether answer is grounded in course evidence. |
| Citation hit@k | 1 if retrieved source matches `source_hint`, else 0 | Measures whether retrieval found the expected lecture/source. |
| False-premise detection | Rate over `false_premise=yes` cases | Measures whether misleading questions are corrected. |
| Refusal accuracy | Rate over `expected_behavior=refuse` cases | Measures whether unavailable/private/future-exam questions are refused. |
| Cost/latency | LLM calls, estimated tokens, wall-clock latency | Measures workflow overhead. |
| Human sanity check | Manual 0/0.5/1 review on selected rows | Catches cases where keyword scoring is too strict. |

We did not use an LLM-as-judge. This keeps scoring reproducible and avoids adding another model's bias. The limitation is that keyword correctness can under-score semantically correct answers, so we added manual sanity-check notes.

### 4.4 Versions and Ablations

All versions run on the same evaluation set with the same model/settings.

| Version | Purpose |
|---|---|
| `V0_baseline_single_call` | Single-call baseline. |
| `V1_retrieval_only` | H2 retrieval alone. |
| `V2_router_rewrite_retrieval` | H1 router + H2 retrieval. |
| `V3_retrieval_verifier` | H2 retrieval + H3 verifier path. |
| `V4_full_router_retrieval_verifier` | Final workflow with H1-H4. |
| `A1_full_without_router` | Remove H1 router from final workflow. |
| `A2_full_without_verifier` | Remove H3 verifier path from final workflow. |
| `A3_full_topk2` | Change retrieval depth to top-k=2. |
| `A4_full_without_calculation_guard` | Remove H4 formula/calculation guard. |

Important interpretation note: in this implementation, H4 is located inside the verifier path. Therefore `A2_full_without_verifier` also drops the formula guard. We use `A4` as the clean formula-guard ablation.

## 5. Experimental Setup

Recorded runs use:

- model: `qwen2.5:3b` via local Ollama,
- access route: `http://localhost:11434`,
- temperature: 0,
- retrieval backend: TF-IDF,
- top-k: 5 for main versions,
- cache: `data/cache/`.

We chose a local model because the experiment needs many calls: 60 cases × 9 versions, with some versions requiring multiple LLM calls. A local model avoids free-tier quota issues and makes the run reproducible. The model is small, so we do not claim absolute state-of-the-art answer quality. Instead, we test whether harnesses improve a weak but accessible baseline.

## 6. Results

### 6.1 Overall Balanced-Set Results

Source: `records/summary_balanced.csv`.

| Version | Correctness | Faithfulness | Citation hit | False-premise detect | Refusal correct | Avg calls | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline | 0.533 | 0.633 | 0.000 | 0.000 | 1.000 | 1.00 | 2.70 |
| V1 retrieval only | 0.467 | 0.983 | 0.817 | 0.125 | 0.571 | 1.00 | 5.23 |
| V2 router + retrieval | 0.467 | 0.983 | 0.833 | 0.125 | 0.714 | 2.00 | 6.58 |
| V3 retrieval + verifier | 0.550 | 0.992 | 0.817 | 1.000 | 0.571 | 1.78 | 8.49 |
| V4 full workflow | 0.575 | 0.992 | 0.833 | 1.000 | 0.857 | 2.72 | 9.45 |
| A1 without router | 0.567 | 0.992 | 0.833 | 1.000 | 0.714 | 1.78 | 8.27 |
| A2 without verifier | 0.483 | 0.983 | 0.833 | 0.125 | 0.857 | 2.00 | 6.42 |
| A3 top-k=2 | 0.567 | 1.000 | 0.767 | 1.000 | 0.857 | 2.72 | 8.00 |
| A4 without formula guard | 0.525 | 0.992 | 0.833 | 1.000 | 0.857 | 2.83 | 9.96 |

### 6.2 Main Findings

**Retrieval improves grounding more than automatic correctness.**  
Compared with V0, V1 raises faithfulness from 0.633 to 0.983 and citation hit from 0.000 to 0.817. However, automatic correctness drops from 0.533 to 0.467. Case inspection shows why: several retrieval answers are grounded but use wording that does not exactly match the required-term scorer. This is why V1 retrieval-concept rows are included in the manual review set.

**The verifier is the strongest safety harness.**  
Comparing V4 with A2, correctness improves from 0.483 to 0.575 and false-premise detection improves from 0.125 to 1.000. This supports H3's role: retrieval alone often answers misleading questions directly, while the verifier catches and corrects the premise. Importantly, the detection score does not depend on the hardcoded misconception rules: the rule table covered 5 of the 8 false-premise cases, and the remaining 3 (RNN vanishing gradients, attention and order information, GAN likelihood) were caught by the verifier LLM pass alone, also at 1.000.

**The formula guard fixes concrete calculation failures.**  
Comparing V4 with A4, overall correctness improves from 0.525 to 0.575, and calculation-category correctness improves from 0.500 to 0.700. The guard fired on 7 of the 15 calculation cases and answered all 7 correctly (1.000), while the LLM alone scored 0.438 on the remaining 8; firing is recorded per case in the `notes` column of `records/results_balanced.csv`. In case B038, the baseline and no-guard versions produce the wrong CNN output shape/parameter count, while V4 computes `32x32x16` and `432` correctly. This supports H4 as a targeted deterministic harness.

**The router helps scope/refusal more than general correctness.**  
V4 vs A1 shows a small correctness gain, 0.567 to 0.575, but refusal accuracy improves from 0.714 to 0.857. This suggests the router's value is not broad answer generation; it mainly helps classify scope and question type.

**Top-k retrieval depth matters modestly.**  
V4 vs A3 shows citation hit improves from 0.767 at top-k=2 to 0.833 at top-k=5. The correctness gain is small, but the citation hit difference supports using a larger evidence budget for varied course questions.

**The final workflow beats the baseline on reliability dimensions, but not dramatically on keyword correctness.**  
V4 improves correctness from 0.533 to 0.575, faithfulness from 0.633 to 0.992, citation hit from 0.000 to 0.833, and false-premise detection from 0.000 to 1.000. The main quality gain is evidence grounding and safety behavior, not a large jump in automatic keyword correctness.

### 6.3 Category-Level Results

Correctness by balanced category:

| Category | V0 | V1 | V2 | V4 | A4 |
|---|---:|---:|---:|---:|---:|
| retrieval_concept | 0.533 | 0.400 | 0.533 | 0.533 | 0.533 |
| reasoning_comparison | 0.400 | 0.433 | 0.433 | 0.433 | 0.433 |
| calculation_formula | 0.667 | 0.600 | 0.500 | 0.700 | 0.500 |
| verification_scope | 0.533 | 0.433 | 0.400 | 0.633 | 0.633 |

The category table reveals why the total correctness increase is modest. The final workflow helps most on `calculation_formula` and `verification_scope`, but concept and reasoning scores remain limited by small-model answer completeness and keyword scoring. For example, an answer may correctly describe cross entropy as minimizing the difference between predicted class probabilities and the true distribution, but receive a low automatic score if it omits the exact required phrase "target distribution" or "loss."

Citation hit by category:

| Category | V0 | V1 | V2 | V4 |
|---|---:|---:|---:|---:|
| retrieval_concept | 0.000 | 0.933 | 1.000 | 1.000 |
| reasoning_comparison | 0.000 | 0.933 | 0.800 | 0.800 |
| calculation_formula | 0.000 | 1.000 | 1.000 | 1.000 |
| verification_scope | 0.000 | 0.400 | 0.533 | 0.533 |

This supports the retrieval claim. Even when automatic correctness is conservative, retrieval clearly changes the system from no source traceability to high source hit rates.

### 6.4 Representative Cases

**B038: CNN output shape and parameter count.**  
Question: "For a 32x32x3 image with 16 filters of size 3x3, stride 1, padding 1, and no bias, what is the output shape and parameter count?"

- V0 gives an incorrect shape because it ignores padding.
- A4 without formula guard also gives an incorrect formula/shape.
- V4 returns: `32x32x16` and `3*3*3*16 = 432`.

This is the clearest example of H4's value.

**B047: logistic regression and XOR false premise.**  
Question: "Since logistic regression uses sigmoid, why can it solve XOR without hidden layers?"

- V0 accepts the false premise and claims sigmoid enables non-linear decision boundaries.
- A2 without verifier also accepts the false premise.
- V4 corrects it: logistic regression has a linear decision boundary, and XOR is not linearly separable without hidden nonlinear transformations.

This supports H3's misconception-handling role.

**B054: exact final exam questions.**  
Question: "What exact questions will appear in the COMP5541 final exam?"

- V0 and V4 both refuse. This is a useful negative result: the baseline already handles some obvious refusal cases, so the workflow does not improve every category.
- The workflow's added value is more visible on false-premise and evidence-grounding cases.

**B023: convolutional layers vs fully connected layers.**  
The answer is directionally correct but automatic correctness is low because the wording does not consistently include all required terms such as "weight sharing" and "local connectivity." This illustrates why manual review is needed for reasoning/comparison cases.

### 6.5 Broad-Set Robustness Check

The broad set (`records/evaluation_set.csv`) is less balanced and contains more concept questions. It is useful as a robustness check because it asks whether the same pattern holds outside the harness-targeted balanced set.

On the broad set, V4 improves automatic correctness from 0.408 to 0.592, faithfulness from 0.608 to 0.975, citation hit from 0.000 to 0.950, and false-premise detection from 0.000 to 1.000 compared with V0. The calculation guard also has a clear effect: removing it reduces correctness on the five broad-set calculation cases from 1.000 to 0.700 (1.000 to 0.786 when the two formula cases are included). These broad-set numbers are stronger than the balanced-set headline correctness because the category mix is different. We therefore use the balanced set for the main ablation logic and the broad set only as a secondary robustness check.

## 7. Human Sanity Check

The automatic score is deterministic and reproducible, but it is not a perfect human grading metric. It is especially strict for concept and reasoning questions because it checks required-term coverage.

We prepared `records/manual_review_balanced.csv` with 135 rows:

- all 60 V0 baseline rows,
- all 60 V4 final workflow rows,
- 15 V1 retrieval-only rows for `retrieval_concept`.

The review dimensions are:

- human correctness,
- human faithfulness,
- human citation quality,
- human scope behavior,
- human misconception handling,
- human comment.

The final report should present automatic scores as the quantitative backbone and human review as a sanity check for representative successes and failures. Calculation/formula questions need less manual effort because many have deterministic numeric answers; concept, reasoning, false-premise, and refusal cases need more human judgment.

## 8. Optional Demo UI

We implemented an optional Streamlit demo in `code/demo_app.py`. It is not part of the evidence proof, but it makes the workflow inspectable for presentation.

The demo lets the user enter a COMP5541 question and displays:

- H1 router result: in-scope flag, question type, lecture topic, and retrieval query;
- H2 retrieved evidence: slide/tutorial source, page, chunk id, score, and text;
- H4 formula/calculation guard output when a deterministic pattern is detected;
- H3 verifier output and final answer.

Run:

```bash
cd code
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:3b TOP_K=5 TEMPERATURE=0 \
streamlit run demo_app.py --server.port 8501
```

This demo supports the optional extension in the brief, but it is not a substitute for the evaluation tables and ablations.

## 9. Limitations and Responsible Use

**Small model limitation.**  
The recorded run uses a local 3B model. This keeps the experiment free and reproducible, but limits answer fluency and completeness. A stronger hosted model may improve concept/reasoning answers, but the controlled harness comparison should still hold.

**Automatic scoring limitation.**  
Correctness uses keyword-style required-term matching. This can under-score semantically correct answers. Human sanity checks are therefore necessary before making final claims.

**Retrieval limitation.**  
TF-IDF is deterministic and cheap, but can miss semantically relevant chunks when wording differs. The optional Chroma backend may improve some cases but would need a controlled rerun.

**Verifier limitation.**  
The verifier can improve false-premise handling, but it can also add latency and may overcorrect if the question is ordinary. We include an overcorrection check, but this is not a full formal guarantee.

**Formula guard limitation.**  
H4 is intentionally narrow. It only fires on parseable formula patterns. It should not be presented as a general symbolic math engine.

**Privacy and academic integrity.**  
The system refuses private information and future exam questions. It should be used as a revision aid, not as a way to invent official answers or replace studying the course materials.

**Cost/latency tradeoff.**  
The final workflow improves grounding and safety but costs more calls and latency: V4 averages 2.72 LLM calls and 9.45 seconds, compared with V0's 1 call and 2.70 seconds.

## 10. Conclusion

The Track B claim is supported with controlled evidence. The full workflow does not massively improve automatic keyword correctness, but it substantially improves the reliability dimensions that matter for course-material QA: faithfulness, citation traceability, false-premise detection, and deterministic calculation behavior. The ablations show that the verifier and formula guard have the clearest targeted benefits, retrieval is essential for evidence grounding, and the router helps scope/refusal behavior more than general correctness. The main remaining gap is human review of concept/reasoning cases and possible cross-model robustness checks with a stronger hosted model if quota permits.

## Acknowledgement and Contribution

Each member contributed one harness within the shared task:

- Member 1: H1 scope router / question-type detector and evaluation-set design.
- Member 2: H2 course-material retrieval, ingestion, chunking, and citation display.
- Member 3: H3 evidence verifier / misconception guard and manual-review protocol.
- Shared extension: H4 deterministic formula/calculation guard and optional Streamlit demo.

Final report should replace this section with real names, student IDs, and more specific individual work.

## AI-Tool Disclosure

AI tools were used for coding assistance, debugging, wording support, and drafting. The team selected the task, materials, harness design, evaluation criteria, and interpretation of results. Exact disclosure is maintained in `docs/ai_disclosure.md`.
