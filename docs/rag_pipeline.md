# RAG Pipeline Details

## 1. Document Ingestion

Implemented in `code/src/ingest.py`.

Supported source formats:

- PDF lecture slides and assignment brief
- Markdown and text files
- Jupyter notebooks

Processing steps:

1. Convert each source file to text.
2. Save extracted text under `data/processed/text/`.
3. Split PDF text by page using form-feed markers from `pdftotext`.
4. Normalize whitespace.
5. Split each page into overlapping word chunks.
6. Save all chunks to `data/processed/index/chunks.jsonl`.

Default chunking:

- 220 words per chunk
- 45-word overlap
- discard very short chunks under 80 characters

Each chunk has:

```text
chunk_id
source_file
source_group
page
chunk_number
text
```

## 2. Retriever Backends

Implemented in `code/src/retriever.py`.

### TF-IDF Backend

Default backend:

```bash
RETRIEVER_BACKEND=tfidf
```

It uses `sklearn.feature_extraction.text.TfidfVectorizer` with:

- lowercase normalization
- English stop words
- unigram and bigram features
- maximum 50,000 features

This backend is cheap, deterministic, and does not require API calls.

### Chroma Vector Backend

Optional backend:

```bash
python3 -m pip install -r code/requirements-vector.txt
python3 code/build_chroma.py --rebuild
RETRIEVER_BACKEND=chroma
```

Chroma persists vectors under:

```text
data/processed/chroma/
```

The Chroma collection is populated from the same `chunks.jsonl`, so both retrievers use identical chunk boundaries and metadata.

## 3. Prompt Construction

Implemented in `code/src/workflows.py`.

For retrieval-augmented answering, the system inserts the top-k evidence chunks:

```text
[Evidence 1] source file, page, chunk id, score
chunk text

[Evidence 2] ...
```

The answer prompt requires the model to:

- use only the evidence,
- say not enough information if evidence is insufficient,
- give exam-style answers with key course terms,
- show formulas and substitutions for calculation questions,
- end with an evidence line.

## 4. Deterministic Formula Guard

Implemented in `code/src/workflows.py`.

For parseable calculation questions, the workflow applies deterministic formula execution after retrieval. This is a separate harness from retrieval because arithmetic and formula substitution are common local-LLM failure modes.

Currently covered cases include:

- linear-regression parameter count with a bias term,
- ReLU derivative for negative inputs,
- chain-rule term for a single neuron,
- convolution output width formula,
- convolution output shape and parameter count,
- pooling output shape.

The guard only fires when the question contains enough explicit symbolic or numeric information. Otherwise, the answer remains evidence-grounded LLM output.

## 5. Workflow Variants

The same RAG components are reused in:

- `V1_retrieval_only`
- `V2_router_rewrite_retrieval`
- `V3_retrieval_verifier`
- `V4_full_router_retrieval_verifier`
- retrieval ablations
- `A4_full_without_calculation_guard`

This makes the Track B comparison controlled: the evaluation set is fixed and only the harness configuration changes.
