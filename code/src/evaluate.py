from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import config
from .llm import build_llm
from .retriever import build_retriever
from .workflows import (
    add_route_cost,
    baseline,
    full_workflow,
    retrieval_answer,
    retrieval_with_verifier,
    route_question,
)


def run_all_versions(
    limit: int | None = None,
    evaluation_set_path: Path | None = None,
    results_path: Path | None = None,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    config.ensure_dirs()
    evaluation_set_path = evaluation_set_path or config.EVALUATION_SET_PATH
    results_path = results_path or config.RESULTS_PATH
    summary_path = summary_path or config.SUMMARY_PATH
    eval_df = pd.read_csv(evaluation_set_path)
    if limit:
        eval_df = eval_df.head(limit)

    llm = build_llm()
    retriever = build_retriever()
    rows: list[dict[str, object]] = []
    top_k = int(config.env("TOP_K", "5") or "5")

    versions = [
        "V0_baseline_single_call",
        "V1_retrieval_only",
        "V2_router_rewrite_retrieval",
        "V3_retrieval_verifier",
        "V4_full_router_retrieval_verifier",
        "A1_full_without_router",
        "A2_full_without_verifier",
        "A3_full_topk2",
        "A4_full_without_calculation_guard",
    ]

    total_cases = len(eval_df)
    for case_index, (_, case) in enumerate(eval_df.iterrows(), start=1):
        question = str(case["question"])
        print(f"[{case_index}/{total_cases}] Running case {case['case_id']}: {question[:80]}", flush=True)
        outputs = []
        outputs.append(baseline(llm, question))
        outputs.append(retrieval_answer(llm, retriever, question, k=top_k))

        route, route_response = route_question(llm, question, use_llm=True)
        rewritten = route.get("search_query") or question
        out = retrieval_answer(
            llm,
            retriever,
            question,
            query=rewritten,
            k=top_k,
            version="V2_router_rewrite_retrieval",
        )
        out = add_route_cost(out, route_response)
        out.notes = f"route={route}"
        outputs.append(out)

        outputs.append(retrieval_with_verifier(llm, retriever, question, k=top_k))
        outputs.append(full_workflow(llm, retriever, question, k=top_k))
        outputs.append(
            full_workflow(
                llm,
                retriever,
                question,
                k=top_k,
                use_router=False,
                use_verifier=True,
                version="A1_full_without_router",
            )
        )
        outputs.append(
            full_workflow(
                llm,
                retriever,
                question,
                k=top_k,
                use_router=True,
                use_verifier=False,
                version="A2_full_without_verifier",
            )
        )
        outputs.append(
            full_workflow(
                llm,
                retriever,
                question,
                k=2,
                use_router=True,
                use_verifier=True,
                version="A3_full_topk2",
            )
        )
        outputs.append(
            full_workflow(
                llm,
                retriever,
                question,
                k=top_k,
                use_router=True,
                use_verifier=True,
                use_calculation_guard=False,
                version="A4_full_without_calculation_guard",
            )
        )

        for output in outputs:
            row = case.to_dict()
            row.update(asdict(output))
            row.update(
                {
                    "correctness_score": "",
                    "faithfulness_score": "",
                    "citation_hit": "",
                    "false_premise_detected": "",
                    "refusal_correct": "",
                    "human_comment": "",
                }
            )
            rows.append(row)

        partial = pd.DataFrame(rows)
        partial.to_csv(results_path, index=False, quoting=csv.QUOTE_MINIMAL)
        write_summary(partial, versions, summary_path=summary_path)
        print(f"[{case_index}/{total_cases}] Saved {len(partial)} result rows", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_summary(results, versions, summary_path=summary_path)
    return results


def write_summary(results: pd.DataFrame, versions: list[str], summary_path: Path | None = None) -> None:
    summary_path = summary_path or config.SUMMARY_PATH
    summary_rows = []
    for version in versions:
        subset = results[results["version"] == version]
        summary_rows.append(
            {
                "version": version,
                "n_cases": len(subset),
                "avg_latency_s": subset["latency_s"].astype(float).mean() if len(subset) else "",
                "avg_llm_calls": subset["llm_calls"].astype(float).mean() if len(subset) else "",
                "avg_input_tokens_est": subset["input_tokens_est"].astype(float).mean() if len(subset) else "",
                "avg_output_tokens_est": subset["output_tokens_est"].astype(float).mean() if len(subset) else "",
                "manual_correctness_avg": "fill_after_scoring",
                "manual_faithfulness_avg": "fill_after_scoring",
                "manual_citation_hit_rate": "fill_after_scoring",
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def main(
    limit: int | None = None,
    evaluation_set_path: Path | None = None,
    results_path: Path | None = None,
    summary_path: Path | None = None,
) -> None:
    results = run_all_versions(
        limit=limit,
        evaluation_set_path=evaluation_set_path,
        results_path=results_path,
        summary_path=summary_path,
    )
    print(f"Wrote {len(results)} rows to {results_path or config.RESULTS_PATH}")
    print(f"Wrote summary to {summary_path or config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
