from __future__ import annotations

import json
import os
import time
from html import escape
from dataclasses import dataclass

import streamlit as st

from src import config
from src.llm import BaseLLM, LLMResponse, build_llm
from src.retriever import BaseRetriever, RetrievalResult, build_retriever, format_context
from src.workflows import (
    SYSTEM,
    _answer_prompt,
    calculation_guard,
    has_false_premise_signal,
    misconception_guard,
    route_question,
)


@dataclass
class DemoTrace:
    question: str
    route: dict[str, str]
    route_response: LLMResponse | None
    retrieved: list[RetrievalResult]
    draft_response: LLMResponse | None
    guard_name: str
    guard_answer: str | None
    verifier_response: LLMResponse | None
    final_answer: str
    latency_s: float
    llm_calls: int


SAMPLE_QUESTIONS = [
    "Why are convolutional layers more parameter-efficient than fully connected layers for images?",
    "For a 32x32x3 image with 16 filters of size 3x3, stride 1, padding 1, and no bias, what is the output shape and parameter count?",
    "Since logistic regression uses sigmoid, why can it solve XOR without hidden layers?",
    "What exact questions will appear in the COMP5541 final exam?",
]


def local_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --accent: #1677ff;
          --ink: #18202c;
          --muted: #637083;
          --line: #d9e1ec;
          --panel: #f7f9fc;
        }
        .block-container {
          max-width: 1180px;
          padding-top: 1.8rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        .app-subtitle {
          color: var(--muted);
          font-size: 1.02rem;
          margin-top: -0.6rem;
          margin-bottom: 1.1rem;
        }
        .step-band {
          border: 1px solid var(--line);
          border-left: 4px solid var(--accent);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.85rem 1rem;
          margin: 0.7rem 0;
        }
        .step-title {
          font-weight: 700;
          color: var(--ink);
          margin-bottom: 0.25rem;
        }
        .step-meta {
          color: var(--muted);
          font-size: 0.9rem;
        }
        .evidence-title {
          font-weight: 700;
          color: var(--ink);
        }
        .evidence-meta {
          color: var(--muted);
          font-size: 0.86rem;
          margin-bottom: 0.35rem;
        }
        .answer-box {
          border: 1px solid var(--line);
          background: #ffffff;
          border-radius: 8px;
          padding: 1rem;
          white-space: pre-wrap;
          color: var(--ink);
          line-height: 1.5;
        }
        .small-note {
          color: var(--muted);
          font-size: 0.88rem;
        }
        div[data-testid="stMetric"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.55rem 0.75rem;
          background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_retriever() -> BaseRetriever:
    return build_retriever()


@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model: str) -> BaseLLM:
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if provider == "ollama" and model:
        os.environ["OLLAMA_MODEL"] = model
    if provider == "openai" and model:
        os.environ["OPENAI_MODEL"] = model
    if provider == "gemini" and model:
        os.environ["GEMINI_MODEL"] = model
    return build_llm()


def verifier_prompt(
    question: str,
    route: dict[str, str],
    context: str,
    draft_answer: str,
) -> str:
    question_type = route.get("question_type", "concept").lower()
    false_premise_policy = (
        "The router marked this as false_premise, so explicitly test and correct the premise."
        if question_type == "false_premise"
        else (
            "The router did not mark this as false_premise. Do not reframe an ordinary "
            "why/compare/concept question as an incorrect premise unless the wording explicitly "
            "uses a false universal such as always, guarantee, naturally, or since and the "
            "evidence directly contradicts it."
        )
    )
    return f"""
You are the final verifier for a COMP5541 course-material QA system.

Question:
{question}

Router metadata:
{json.dumps(route, ensure_ascii=True)}

Evidence text:
{context}

Draft answer:
{draft_answer}

Produce the final answer only.

Verification rules:
1. Use ONLY the evidence text. Do not rely on general ML knowledge.
2. False-premise policy: {false_premise_policy}
3. If the premise is false or contradicted by evidence, begin with "The premise is incorrect:" and correct it.
4. If evidence is insufficient, say "not enough information in the provided course materials".
5. Do not say the draft is supported if the evidence contradicts it.
6. Do not include meta-comments such as "the draft answer".
7. End with an "Evidence:" line naming the evidence numbers used.
"""


def run_demo_trace(llm: BaseLLM, retriever: BaseRetriever, question: str, top_k: int) -> DemoTrace:
    start = time.perf_counter()
    route, route_response = route_question(llm, question, use_llm=True)

    if route.get("in_scope", "yes").lower() == "no":
        prompt = (
            "The available materials are COMP5541 Machine Learning and Data Analytics course "
            "materials from Lectures 2-9 and tutorials. Do not mention other courses.\n\n"
            f"Question: {question}\n\n"
            "If the exact answer is not present in those materials, reply exactly with a brief "
            "'not enough information in the provided course materials' explanation."
        )
        refusal = llm.complete(prompt, SYSTEM)
        responses = [r for r in [route_response, refusal] if r]
        return DemoTrace(
            question=question,
            route=route,
            route_response=route_response,
            retrieved=[],
            draft_response=None,
            guard_name="Scope refusal",
            guard_answer=refusal.text,
            verifier_response=None,
            final_answer=refusal.text,
            latency_s=time.perf_counter() - start,
            llm_calls=len(responses),
        )

    query = route.get("search_query") or question
    retrieved = retriever.search(query, k=top_k)
    context = format_context(retrieved)
    draft_response = llm.complete(_answer_prompt(question, context), SYSTEM)

    calculated = calculation_guard(question, retrieved)
    if calculated:
        responses = [r for r in [route_response, draft_response] if r]
        return DemoTrace(
            question=question,
            route=route,
            route_response=route_response,
            retrieved=retrieved,
            draft_response=draft_response,
            guard_name="H4 calculation/formula guard",
            guard_answer=calculated,
            verifier_response=None,
            final_answer=calculated,
            latency_s=time.perf_counter() - start,
            llm_calls=len(responses),
        )

    guarded = misconception_guard(question, retrieved)
    if guarded:
        responses = [r for r in [route_response, draft_response] if r]
        return DemoTrace(
            question=question,
            route=route,
            route_response=route_response,
            retrieved=retrieved,
            draft_response=draft_response,
            guard_name="H3 misconception guard",
            guard_answer=guarded,
            verifier_response=None,
            final_answer=guarded,
            latency_s=time.perf_counter() - start,
            llm_calls=len(responses),
        )

    verifier_response = llm.complete(verifier_prompt(question, route, context, draft_response.text), SYSTEM)
    final_answer = verifier_response.text
    if route.get("question_type", "concept").lower() != "false_premise" and not has_false_premise_signal(question):
        if verifier_response.text.lower().lstrip().startswith("the premise is incorrect"):
            final_answer = draft_response.text

    responses = [r for r in [route_response, draft_response, verifier_response] if r]
    return DemoTrace(
        question=question,
        route=route,
        route_response=route_response,
        retrieved=retrieved,
        draft_response=draft_response,
        guard_name="No deterministic guard fired",
        guard_answer=None,
        verifier_response=verifier_response,
        final_answer=final_answer,
        latency_s=time.perf_counter() - start,
        llm_calls=len(responses),
    )


def step_band(title: str, meta: str) -> None:
    st.markdown(
        f"""
        <div class="step-band">
          <div class="step-title">{escape(title)}</div>
          <div class="step-meta">{escape(meta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_box(text: str) -> None:
    st.markdown(f'<div class="answer-box">{escape(text)}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="ExamGuard-RAG Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_css()

    st.title("ExamGuard-RAG")
    st.markdown(
        '<div class="app-subtitle">Interactive COMP5541 course-material QA demo with visible router, retrieval, verifier, and formula-guard traces.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Runtime")
        provider = st.selectbox("LLM provider", ["ollama", "dummy", "gemini", "openai"], index=0)
        default_model = {
            "ollama": "qwen2.5:3b",
            "dummy": "dry-run",
            "gemini": "gemini-2.5-flash-lite",
            "openai": "gpt-5.4-mini",
        }[provider]
        model = st.text_input("Model", value=default_model)
        top_k = st.slider("Retrieved evidence chunks", min_value=2, max_value=8, value=5, step=1)
        st.caption("Use `dummy` for a fast UI check. Use `ollama` for the recorded zero-cost local demo.")

    sample = st.selectbox("Try a sample question", SAMPLE_QUESTIONS)
    question = st.text_area(
        "Question",
        value=sample,
        height=110,
        placeholder="Ask a COMP5541 revision question from Lectures 2-9 or tutorials.",
    )

    run = st.button("Run workflow", type="primary", use_container_width=False)
    if not run:
        st.markdown(
            '<div class="small-note">The demo shows the same harness structure used in the evaluation: H1 route, H2 retrieve, H4 formula guard when applicable, and H3 verify.</div>',
            unsafe_allow_html=True,
        )
        return

    if not question.strip():
        st.warning("Enter a question first.")
        return

    try:
        retriever = get_retriever()
        llm = get_llm(provider, model)
        with st.spinner("Running harness workflow..."):
            trace = run_demo_trace(llm, retriever, question.strip(), top_k)
    except Exception as exc:
        st.error(str(exc))
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Provider", getattr(llm, "provider", provider))
    metric_cols[1].metric("Model", getattr(llm, "model", model))
    metric_cols[2].metric("LLM calls", trace.llm_calls)
    metric_cols[3].metric("Latency", f"{trace.latency_s:.2f}s")

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.subheader("Harness Trace")
        step_band(
            "H1 Scope Router / Question Type",
            f"in_scope={trace.route.get('in_scope')} | type={trace.route.get('question_type')} | topic={trace.route.get('lecture_topic')}",
        )
        st.json(trace.route)

        step_band(
            "H2 Course Evidence Retrieval",
            f"query={trace.route.get('search_query', trace.question)} | top_k={top_k} | retrieved={len(trace.retrieved)}",
        )
        if trace.retrieved:
            for index, result in enumerate(trace.retrieved, start=1):
                with st.expander(
                    f"Evidence {index}: {result.source_file}, page {result.page} | score {result.score:.4f}",
                    expanded=index <= 2,
                ):
                    st.markdown(f'<div class="evidence-meta">chunk {result.chunk_id}</div>', unsafe_allow_html=True)
                    st.write(result.text)
        else:
            st.info("No retrieval was needed because the router sent the question to scope refusal.")

        step_band("H4 Formula / Calculation Guard", trace.guard_name)
        if trace.guard_answer:
            answer_box(trace.guard_answer)
        else:
            st.markdown('<div class="small-note">No parseable deterministic calculation pattern was detected.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Answer Construction")
        if trace.draft_response:
            with st.expander("Draft answer before verifier", expanded=False):
                answer_box(trace.draft_response.text)

        step_band(
            "H3 Evidence Verifier / Finalizer",
            "Checks the draft against retrieved slide evidence, corrects false premises, and enforces refusal behavior.",
        )
        if trace.verifier_response:
            with st.expander("Verifier raw output", expanded=False):
                answer_box(trace.verifier_response.text)

        st.subheader("Final Answer")
        answer_box(trace.final_answer)

    st.divider()
    st.markdown(
        f'<div class="small-note">Knowledge base: {config.CHUNKS_PATH}. Retrieval backend: {config.env("RETRIEVER_BACKEND", "tfidf")}.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
