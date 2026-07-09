from __future__ import annotations

import json

from .llm import BaseLLM
from .retriever import BaseRetriever
from .workflows import (
    WorkflowOutput,
    add_route_cost,
    full_workflow,
    retrieval_answer,
    retrieval_with_verifier,
    route_question,
)


HARNESS_LABELS = {
    "H1": ("Harness 1 · Router", "Scope router and retrieval query rewrite"),
    "H2": ("Harness 2 · RAG", "Evidence-grounded answer prompt"),
    "H3": ("Harness 3 · Verifier", "Evidence verifier and misconception guard"),
    "H4": ("Harness 4 · Full", "Full workflow with calculation guard"),
}


def run_harnesses(
    llm: BaseLLM,
    retriever: BaseRetriever,
    question: str,
    k: int = 5,
) -> tuple[dict[str, str], list[WorkflowOutput], WorkflowOutput]:
    route, route_response = route_question(llm, question, use_llm=True)
    query = route.get("search_query") or question

    h1 = retrieval_answer(
        llm,
        retriever,
        question,
        query=query,
        k=k,
        version="H1_router",
    )
    h1 = add_route_cost(h1, route_response)
    h1.notes = f"route={json.dumps(route, ensure_ascii=True)}"

    h2 = retrieval_answer(llm, retriever, question, k=k, version="H2_rag")

    h3 = retrieval_with_verifier(
        llm,
        retriever,
        question,
        query=query,
        k=k,
        version="H3_verifier",
        route=route,
        use_calculation_guard=False,
    )

    h4 = full_workflow(
        llm,
        retriever,
        question,
        k=k,
        version="H4_full",
    )

    return route, [h1, h2, h3, h4], h4


def harness_payload(output: WorkflowOutput) -> dict[str, object]:
    harness_id = output.version.split("_", 1)[0]
    title, description = HARNESS_LABELS.get(harness_id, (output.version, ""))
    return {
        "id": harness_id,
        "version": output.version,
        "title": title,
        "description": description,
        "answer": output.answer,
        "llm_calls": output.llm_calls,
        "latency_s": round(output.latency_s, 3),
        "notes": output.notes,
    }


def harnesses_to_dict(
    route: dict[str, str],
    outputs: list[WorkflowOutput],
    final_output: WorkflowOutput,
) -> dict[str, object]:
    return {
        "route": route,
        "final_answer": final_output.answer,
        "final_version": final_output.version,
        "harnesses": [harness_payload(item) for item in outputs],
        "final_metrics": {
            "llm_calls": final_output.llm_calls,
            "latency_s": round(final_output.latency_s, 3),
        },
    }
