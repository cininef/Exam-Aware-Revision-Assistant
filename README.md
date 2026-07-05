# Exam-Aware COMP5541 Revision Assistant (Track B)

Track B project: design and improve a small LLM workflow, and prove with a controlled
evaluation that each added harness helps.

This README walks through the Track B required steps in order and states, for each
step, exactly what we did and where the artifact lives. Reproduction commands are at
the end.

---

## Step 0: Choose the Shared Task

We chose one shared task that fits our background as students of this course:
**answering COMP5541 revision questions from the official course materials
(Lectures 2-9 and tutorials), with evidence citations.**

The whole project revolves around this single task. The knowledge base contains only
official course materials (see [docs/material_manifest.md](docs/material_manifest.md));
personal assignment solutions are deliberately excluded. Each member owns one harness
(see [docs/contribution_plan.md](docs/contribution_plan.md)).

## Step 1: State the Need

**Who needs it:** students revising for the COMP5541 exam, who ask questions about a
fixed body of course text and need answers they can trust and trace back to the slides.

**Why a single LLM call is not enough.** It:

- answers from general machine-learning knowledge instead of the course's actual slides,
- cannot cite where in the course an answer comes from,
- accepts false premises embedded in revision questions ("Since CNNs are naturally
  rotation invariant, why ..."),
- invents answers to questions the materials cannot answer (out-of-scope questions),
- makes arithmetic and formula-substitution errors on exam-style calculation questions.

**What a good output means:** a correct, exam-style answer grounded in retrieved course
evidence, with an explicit evidence citation, an explicit correction when the premise
is false, and an explicit refusal when the course materials cannot answer the question.

## Step 2: Build the Baseline

`V0_baseline_single_call`: the question is sent to the LLM in a single call — no
retrieval, no routing, no verification, same system prompt and temperature 0 as every
other version. Implemented as `baseline()` in
[code/src/workflows.py](code/src/workflows.py).

This is the workflow every harness must beat. On the broad set it scores 0.41
correctness, 0.61 faithfulness, and 0.00 citation hit — the failure modes from Step 1
are real and measurable.

## Step 3: Build the Harnesses and the Combined Workflow

Each member designed one harness. All four are implemented in
[code/src/workflows.py](code/src/workflows.py):

**H1 — Scope router / question-type detector** (`route_question`, `heuristic_route`).
An LLM call classifies the question's scope (in/out of course materials) and type
(concept / comparison / calculation / false_premise / unanswerable). The retrieval
query itself is built deterministically from a course-topic keyword table, because a
small local model tends to rewrite course terms into non-canonical labels that hurt
sparse retrieval. Out-of-scope questions are diverted to a refusal prompt before any
retrieval happens.

**H2 — Course RAG retriever** ([code/src/ingest.py](code/src/ingest.py),
[code/src/retriever.py](code/src/retriever.py)). Course PDFs, notebooks, and markdown
are converted to text, split per page, and chunked (220-word windows, 45-word overlap)
into `data/processed/index/chunks.jsonl`. The default retriever is TF-IDF (1-2 grams);
an optional Chroma vector store gives semantic retrieval with the same interface. The
top-k chunks are inserted into the prompt with source file, page, and chunk id, and the
model is instructed to answer only from that evidence and to end with an "Evidence:"
line.

**H3 — Evidence verifier / misconception guard** (`retrieval_with_verifier`,
`misconception_guard`). A second LLM pass checks the draft answer against the retrieved
evidence: it corrects false premises, enforces the refusal wording when evidence is
insufficient, and is explicitly instructed not to invent a false-premise correction for
ordinary questions (with a deterministic overcorrection check on top). A small
deterministic rule table handles five known course misconceptions; the remaining
false-premise cases are handled by the verifier LLM itself, so the harness is not just
a lookup table.

**H4 — Calculation / formula guard** (`calculation_guard`). A deterministic solver for
parseable exam-style formula questions: linear-regression parameter counts,
convolution/pooling output shapes (including padding), convolution parameter counts,
ReLU derivatives, and chain-rule terms. It only fires when the question contains enough
explicit numeric/symbolic information; on the balanced set it covers 7 of 15
calculation cases and everything else falls through to the LLM.

**Combined workflow** `V4_full_router_retrieval_verifier` (`full_workflow`) stacks all
four: question → H1 route → H2 retrieve → draft answer → H4 guard → H3 verify → final
answer.

## Step 4: Design the Evaluation Set

Two fixed 60-case sets. Every case has: `question`, `reference_answer` (hand-written),
`source_hint` (expected lecture), `required_terms` (for automatic scoring),
`false_premise` flag, and `expected_behavior` (`answer` / `correct_premise` /
`refuse`).

- **`records/evaluation_set_balanced.csv` — the main ablation set.** 4 categories × 15
  cases, each category targeting one harness: `retrieval_concept` (H2),
  `reasoning_comparison` (H1 + evidence reasoning), `calculation_formula` (H4), and
  `verification_scope` (8 false-premise + 7 unanswerable cases, testing H1 scope and
  H3). Balanced design means each harness has enough targeted cases for its ablation
  difference to be visible instead of being averaged away by concept questions.
- **`records/evaluation_set.csv` — broad robustness set.** Course-wide coverage
  (34 concept / 12 comparison / 5 calculation / 5 false-premise / 2 formula /
  2 unanswerable). Already fully run; kept as a robustness check.

Design rationale: [records/evaluation_design.md](records/evaluation_design.md).

## Materials Used

The retrieval knowledge base is built only from the directories listed in
[code/src/config.py](code/src/config.py): `data/raw/course_slides`,
`data/raw/tutorials`, and `data/raw/assessment_briefs`. Files under
`data/raw/evaluation_sources` are used to design test cases and reference answers, but
are not the main evidence source for answering user questions. Project-brief PDFs are
used only to understand the Track B requirements.

### Retrieval Knowledge Base Files

Course slides:

- `data/raw/course_slides/Lecture2-2026-COMP5541(1).pdf`
  - Used topics: KNN, non-parametric prediction, validation/test split, linear
    regression, parameter counts, bias-variance, polynomial complexity, MSE.
- `data/raw/course_slides/Lecture3-2026.pdf`
  - Used topics: one-hot encoding, logistic regression, sigmoid, softmax, cross
    entropy, L1/L2 regularization, SGD/minibatches, momentum, AdaGrad, Adam.
- `data/raw/course_slides/Lecture 4.pdf`
  - Used topics: neural-network motivation, linear-model limitations, nonlinear
    activations, artificial neurons, computation graphs, chain rule,
    backpropagation, ReLU derivative, hidden-unit capacity.
- `data/raw/course_slides/Lecture 5.pdf`
  - Used topics: CNN motivation, local connectivity, weight sharing, filter depth,
    convolution output shape formula, convolution parameter count, pooling,
    receptive field, transfer learning, CNN feature hierarchy.
- `data/raw/course_slides/Lecture 6(2).pdf`
  - Used topics: practical neural-network training, activation saturation, image
    preprocessing, data augmentation, Kaiming initialization, learning rate
    behavior, dropout, overfitting/sanity checks, validation vs test use.
- `data/raw/course_slides/Lecture 7(1).pdf`
  - Used topics: sequential data, RNN hidden state, parameter sharing across time,
    encoder-decoder setup, vanilla RNN limitations, LSTM gates and cell state.
- `data/raw/course_slides/Lecture 8(2).pdf`
  - Used topics: attention, context vectors, attention weights, encoder-decoder
    bottleneck, Transformer parallelism, unsupervised learning, dimensionality
    reduction.
- `data/raw/course_slides/Lecture 9 .pdf`
  - Used topics: generative-model taxonomy, explicit vs implicit density,
    PixelRNN/PixelCNN autoregressive modeling, conditional pixel generation, GAN
    generator/discriminator setup, GAN training behavior and limitations.

Tutorials:

- `data/raw/tutorials/Tutorial 1-Review on Linear Algebra.pdf`
- `data/raw/tutorials/T2-Review on Calculus.pdf`
- `data/raw/tutorials/T3-Machine Learning Development_Q.pdf`
- `data/raw/tutorials/T4-pytorch.pdf`
- `data/raw/tutorials/T5_Image_Classification_with_CNNs(1).pdf`
- `data/raw/tutorials/RNN_LSTM_Name_Classification_Tutorial (1).ipynb`
- `data/raw/tutorials/T8_GAN_MNIST_Tutorial.ipynb`
  - Used as supporting material for calculus/linear algebra review, ML development
    protocol, PyTorch implementation details, CNN training, RNN/LSTM coding, and
    DCGAN/MNIST tutorial concepts such as unsupervised GAN training and labels being
    ignored by the adversarial objective.

Assessment/context material included in the retrieval source directories:

- `data/raw/assessment_briefs/COMP5541_2026_assignment.pdf`
  - Used only as course-context evidence, not as a source of personal assignment
    solutions.

### Evaluation-Design Files

These files help create evaluation questions, expected answers, and source hints:

- `data/raw/evaluation_sources/COMP5541_quiz_revision_map.md`
- `data/raw/evaluation_sources/exam.md`
- `data/raw/evaluation_sources/quiz1.md`
- `data/raw/evaluation_sources/quiz2.md`
- `data/raw/evaluation_sources/quiz3.md`

### Project-Requirement Files

These files are read to align the submission with the Track B requirements, not used
as answer evidence:

- `data/raw/project_brief/COMP5541_Group_Project_2026.pdf`
- `data/raw/project_brief/COMP5541_Group_Project_Slides.pdf`

### Generated Text and Index Files

Ingestion converts the raw materials into extracted text and page-aware chunks:

- Extracted text: `data/processed/text/*.txt`
- Retrieval chunks: `data/processed/index/chunks.jsonl`
- Optional vector store: `data/processed/chroma/`

The system does not use personal assignment reports, private notes, generated answers
from previous runs, or files outside the official course/tutorial/project materials as
retrieval evidence.

## Step 5: Choose the Metrics

Scoring rules: [docs/scoring_rubric.md](docs/scoring_rubric.md). Automatic scoring:
[code/score_results.py](code/score_results.py) (deterministic keyword/rule based, so
scores are reproducible). Manual sanity-check protocol:
[docs/evaluation_manual_review.md](docs/evaluation_manual_review.md).

| Metric | What it proves | Main harness |
|---|---|---|
| Correctness (0 / 0.5 / 1) | Task success: required-term coverage vs reference answer | all |
| Faithfulness (0 / 0.5 / 1) | Answer is grounded in retrieved course evidence | H2, H3 |
| Citation hit@k | Retrieved top-k contains the expected source | H2, H1 |
| False-premise detection rate | Misleading premises are explicitly corrected | H1, H3 |
| Refusal accuracy | Unanswerable questions are refused, not invented | H1, H3 |
| Calculation/formula correctness | Numeric/formula answers are right | H4 |
| Cost proxy (LLM calls, est. tokens) + latency | What each harness costs | all |
| Human-checked score | Catches cases where keyword auto-scoring is too strict | all |

Known limitation, disclosed up front: the automatic faithfulness score checks evidence
usage/format and saturates near 1.0 for all retrieval versions; real groundedness is
covered by the human sanity check.

## Step 6: Run Comparisons and the Ablation

All 9 versions run on the same evaluation set, same model (`qwen2.5:3b` via local
Ollama), temperature 0, top-k 5, with every LLM response cached in `data/cache/` so
runs are deterministic and free to repeat.

| Version | Workflow | Harnesses |
|---|---|---|
| `V0_baseline_single_call` | single LLM call | none |
| `V1_retrieval_only` | retrieve → answer | H2 |
| `V2_router_rewrite_retrieval` | route → retrieve → answer | H1 + H2 |
| `V3_retrieval_verifier` | retrieve → answer → verify | H2 + H3 (+ guards) |
| `V4_full_router_retrieval_verifier` | full workflow | H1 + H2 + H3 + H4 |
| `A1_full_without_router` | V4 minus router | H2 + H3 + H4 |
| `A2_full_without_verifier` | V4 minus verifier | H1 + H2 |
| `A3_full_topk2` | V4 with top-k = 2 | retrieval-depth control |
| `A4_full_without_calculation_guard` | V4 minus formula guard | H1 + H2 + H3 |

Each member's harness is measured individually against the baseline, and the stacked
workflow is measured against each ablation:

- H2 retrieval: V1 − V0
- H1 router: V2 − V1, and V4 − A1
- H3 verifier: V4 − A2
- H4 formula guard: V4 − A4 (on calculation cases)
- Retrieval-depth control: V4 − A3
- Whole workflow: V4 − V0

## Step 7: Analyze the Results Honestly

Primary evidence: the balanced set, 60 cases × 9 versions
(`records/summary_balanced.csv`). Headline (V0 → V4): correctness 0.53 → 0.58,
faithfulness 0.63 → 0.99, citation hit 0.00 → 0.83, false-premise detection
0.00 → 1.00, at a cost of 1 → 2.7 LLM calls and 2.7 s → 9.5 s per question.

Per-harness evidence:

- **H4 formula guard:** calculation correctness 0.50 (A4, guard off) → 0.70 (V4). The
  guard fired on 7 of 15 calculation cases and got all 7 correct (1.0); the LLM
  handled the remaining 8 at 0.44. Guard firing is recorded per case in the `notes`
  column of `records/results_balanced.csv`.
- **H3 verifier:** false-premise detection 0.125 without the verifier (A2) → 1.00 with
  it (V4/V3). The misconception rule table covered 5 of the 8 false-premise cases; the
  other 3 were caught by the verifier LLM pass alone (also 1.00), so H3 is not just a
  lookup table. Verification-scope correctness: 0.47 (A2) → 0.63 (V4).
- **H1 router:** refusal accuracy 0.57 (V1) → 0.71 (V2, router added) → 0.86 (V4);
  removing the router from the full workflow drops it back to 0.71 (A1). Its overall
  correctness effect is small — the router's job is scope control, not answer quality.
- **H2 retrieval:** transforms verifiability — faithfulness 0.63 → 0.98 and citation
  hit 0.00 → 0.82 (V0 → V1) — and enables every downstream harness (the verifier and
  guards operate on retrieved evidence).

Honest negatives, stated as findings:

- **Auto-correctness alone slightly favors the baseline style:** V1 scores 0.47 vs
  V0's 0.53, because keyword-based correctness rewards verbose general-knowledge
  answers, while evidence-only answers lose points when retrieval pulls noisy chunks.
  But V0's answers are unverifiable (faithfulness 0.63, citation 0.00) — for a
  revision assistant, correctness must be read together with groundedness, and the
  human sanity check covers this metric bias.
- **V0's perfect refusal (1.00) is a system-prompt artifact:** the shared system
  prompt already instructs refusal, and with no evidence in context the baseline
  refuses everything unanswerable. Adding retrieved evidence actually tempts the model
  to answer (V1 drops to 0.57); the router restores refusal to 0.86 (V4).
- **Retrieval depth buys coverage, not raw correctness:** top-k=2 (A3) matches V4's
  overall correctness (0.57 vs 0.58) but lowers citation hit (0.77 vs 0.83).

Robustness check: on the broad course-coverage set (`records/summary.csv`) the same
ordering holds with a larger headline gap — correctness 0.41 (V0) → 0.59 (V4),
faithfulness 0.61 → 0.98, citation hit 0.00 → 0.95.

---

## Reproducing the Experiments

### 0. Setup

```bash
cd GroupProject_ExamAware_COMP5541
python3 -m pip install -r code/requirements.txt
```

Requires `pdftotext` (poppler) for ingestion and a local [Ollama](https://ollama.com)
with `ollama pull qwen2.5:3b` for the recorded run. A no-API dry run works with
`LLM_PROVIDER=dummy`.

### 1. Ingest course materials

```bash
python3 code/run_ingest.py
```

### 2. Run the main (balanced) evaluation

The broad set is already run and archived (`records/results.csv`). The only remaining
run is the balanced set, written to separate files so nothing is overwritten:

```bash
cd code
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:3b TOP_K=5 TEMPERATURE=0 \
python3 run_evaluation.py \
  --eval-set ../records/evaluation_set_balanced.csv \
  --results ../records/results_balanced.csv \
  --summary ../records/summary_balanced.csv
```

(To re-run the broad set: `python3 code/run_evaluation.py` with the same env vars —
cached responses make it fast and deterministic.)

### 3. Score

```bash
PYTHONPATH=code python3 code/score_results.py \
  --results records/results_balanced.csv \
  --summary records/summary_balanced.csv
```

### 4. Human sanity check

Follow [docs/evaluation_manual_review.md](docs/evaluation_manual_review.md) on
representative V0/V4 rows and fill `human_comment` before treating automatic scores as
final.

## Model Choice and Access Record

All 9 workflow versions use **one model** — `qwen2.5:3b` via local Ollama API
(`http://localhost:11434`), temperature 0, TF-IDF retriever, top-k 5 — because the
ablation is only a controlled comparison if the model, prompts, data, and settings are
held fixed while single harnesses are switched on/off.

Why this model (choosing a suitable free model is part of the Track B work):

- **Feasibility:** a full run needs ~1,100+ LLM calls (60 cases × 9 versions × up to 3
  calls). We measured the Gemini free tier at 20 requests/day for our project — not
  enough; a local model has no quota and reruns are free from cache.
- **Reproducibility:** local model + temperature 0 + response cache makes every
  recorded number exactly reproducible by the teaching team.
- **Fit for the claim:** Track B rewards proving that a harness fixes a weak baseline,
  not using a big model. A small model has real, measurable failure modes (V0
  correctness 0.41, citation hit 0.00), which is precisely what the harnesses must fix.

Optional cross-model robustness check (only if free quota permits): run V0 and V4 only
on a hosted free-tier model (e.g. `gemini-2.5-flash-lite` or an OpenAI-compatible
provider) to show the conclusions are not specific to a 3B model. The cache lets a
quota-limited run resume across days. If quota does not permit, this is stated as a
limitation instead — see [records/model_metadata.md](records/model_metadata.md) for
exact dates and the recorded quota evidence.

## Repository Map

- `code/` — runnable ingestion, workflow, evaluation, and scoring code; a file ↔
  Track B step map is in [code/README.md](code/README.md)
- `records/` — evaluation sets, per-case results, summaries, model metadata
- `docs/` — scoring rubric, evaluation design, manual review protocol, material
  manifest, contribution plan, AI-tool disclosure
- `report/`, `slides/` — report and 7-minute presentation structure
- `data/` — raw course materials, processed chunks, LLM response cache

## AI-Tool Disclosure

See [docs/ai_disclosure.md](docs/ai_disclosure.md), as required by the project brief.
