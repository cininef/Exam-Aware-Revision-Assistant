from __future__ import annotations

import re
import argparse
from pathlib import Path

import pandas as pd

from src import config


def split_terms(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [term.strip().lower() for term in str(value).split(";") if term.strip()]


def contains_term(answer: str, term: str) -> bool:
    term = term.lower()
    answer = answer.lower()
    words = [w for w in re.split(r"[^a-z0-9]+", term) if w]
    if not words:
        return False
    return all(word in answer for word in words)


def score_correctness(answer: str, required_terms: object, expected_behavior: str) -> float:
    answer_l = answer.lower()
    if expected_behavior == "refuse":
        return 1.0 if "not enough information" in answer_l or "cannot" in answer_l else 0.0
    terms = split_terms(required_terms)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if contains_term(answer, term))
    ratio = hits / len(terms)
    if ratio >= 0.75:
        return 1.0
    if ratio >= 0.4:
        return 0.5
    return 0.0


def score_faithfulness(answer: str, retrieved_sources: object, version: str) -> float:
    answer_l = answer.lower()
    if "not enough information" in answer_l:
        return 1.0
    if pd.isna(retrieved_sources) or not str(retrieved_sources).strip():
        return 0.0 if version != "V0_baseline_single_call" else 0.5
    if "evidence" in answer_l or "page" in answer_l or "chunk" in answer_l:
        return 1.0
    return 0.5


def score_citation_hit(retrieved_sources: object, source_hint: object) -> int:
    if pd.isna(retrieved_sources) or pd.isna(source_hint):
        return 0
    sources = str(retrieved_sources).lower()
    hint = str(source_hint).lower()
    lecture_match = re.search(r"lecture\s+(\d+)", hint)
    if lecture_match:
        lecture = lecture_match.group(1)
        return int(f"lecture {lecture}" in sources or f"lecture{lecture}" in sources)
    return int(any(token in sources for token in re.split(r"[^a-z0-9]+", hint) if len(token) > 4))


def main(results_path: Path | None = None, summary_path: Path | None = None) -> None:
    results_path = results_path or config.RESULTS_PATH
    summary_path = summary_path or config.SUMMARY_PATH
    results = pd.read_csv(results_path)
    results["correctness_score"] = [
        score_correctness(str(row.answer), row.required_terms, str(row.expected_behavior))
        for row in results.itertuples(index=False)
    ]
    results["faithfulness_score"] = [
        score_faithfulness(str(row.answer), row.retrieved_sources, str(row.version))
        for row in results.itertuples(index=False)
    ]
    results["citation_hit"] = [
        score_citation_hit(row.retrieved_sources, row.source_hint)
        for row in results.itertuples(index=False)
    ]
    results["false_premise_detected"] = [
        int(
            str(row.false_premise).lower() == "yes"
            and (
                "premise" in str(row.answer).lower()
                or "incorrect" in str(row.answer).lower()
                or "wrong" in str(row.answer).lower()
            )
        )
        if str(row.false_premise).lower() == "yes"
        else ""
        for row in results.itertuples(index=False)
    ]
    results["refusal_correct"] = [
        int(
            str(row.expected_behavior).lower() == "refuse"
            and (
                "not enough information" in str(row.answer).lower()
                or "cannot" in str(row.answer).lower()
            )
        )
        if str(row.expected_behavior).lower() == "refuse"
        else ""
        for row in results.itertuples(index=False)
    ]
    results["human_comment"] = results["human_comment"].fillna("auto-score; manually sanity-check before final submission")
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("version")
        .agg(
            n_cases=("case_id", "count"),
            correctness=("correctness_score", "mean"),
            faithfulness=("faithfulness_score", "mean"),
            citation_hit=("citation_hit", "mean"),
            avg_latency_s=("latency_s", "mean"),
            avg_llm_calls=("llm_calls", "mean"),
        )
        .reset_index()
    )
    false_cases = results[results["false_premise"].astype(str).str.lower().eq("yes")]
    if not false_cases.empty:
        fp = false_cases.groupby("version")["false_premise_detected"].mean().reset_index()
        summary = summary.merge(fp, on="version", how="left")
    refuse_cases = results[results["expected_behavior"].astype(str).str.lower().eq("refuse")]
    if not refuse_cases.empty:
        rf = refuse_cases.groupby("version")["refusal_correct"].mean().reset_index()
        summary = summary.merge(rf, on="version", how="left")
    summary.to_csv(summary_path, index=False)
    print(f"Scored {len(results)} rows -> {results_path}")
    print(f"Wrote scored summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to per-case output CSV. Defaults to records/results.csv.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to scored summary CSV. Defaults to records/summary.csv.",
    )
    args = parser.parse_args()
    main(results_path=args.results, summary_path=args.summary)
