"""Minimal web UI for the COMP5541 revision assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.llm import build_llm
from src.retriever import BaseRetriever, build_retriever, format_context
from src.slides import render_page_png
from src.web_compare import harnesses_to_dict, run_harnesses

STATIC_DIR = Path(__file__).parent / "static"
load_dotenv(STATIC_DIR.parent / ".env")
BENCHMARK_VERSIONS = {
    "H1": "V2_router_rewrite_retrieval",
    "H2": "V1_retrieval_only",
    "H3": "V3_retrieval_verifier",
    "H4": "V4_full_router_retrieval_verifier",
}

app = FastAPI(title="COMP5541 Revision Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_retriever: BaseRetriever | None = None
_llm = None


def _get_retriever() -> BaseRetriever:
    global _retriever
    if _retriever is None:
        config.ensure_dirs()
        _retriever = build_retriever()
    return _retriever


def _get_llm():
    global _llm
    if _llm is None:
        config.ensure_dirs()
        _llm = build_llm()
    return _llm


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Passage(BaseModel):
    source: str
    page: int
    score: float
    text: str


class HarnessResult(BaseModel):
    id: str
    version: str
    title: str
    description: str
    answer: str
    llm_calls: int
    latency_s: float
    notes: str = ""


class BenchmarkRow(BaseModel):
    id: str
    version: str
    correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    citation_hit: Optional[float] = None
    avg_latency_s: Optional[float] = None
    avg_llm_calls: Optional[float] = None


class QueryResponse(BaseModel):
    passages: list[Passage]
    text: str
    answer: str = ""
    route: dict[str, str] = Field(default_factory=dict)
    harnesses: list[HarnessResult] = Field(default_factory=list)
    final_metrics: dict[str, Union[float, int]] = Field(default_factory=dict)
    benchmark: list[BenchmarkRow] = Field(default_factory=list)


def _load_benchmark() -> list[BenchmarkRow]:
    summary_path = config.SUMMARY_PATH
    if not summary_path.exists():
        return []

    df = pd.read_csv(summary_path)
    rows: list[BenchmarkRow] = []
    for harness_id, version in BENCHMARK_VERSIONS.items():
        subset = df[df["version"] == version]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            BenchmarkRow(
                id=harness_id,
                version=version,
                correctness=_safe_float(row.get("correctness")),
                faithfulness=_safe_float(row.get("faithfulness")),
                citation_hit=_safe_float(row.get("citation_hit")),
                avg_latency_s=_safe_float(row.get("avg_latency_s")),
                avg_llm_calls=_safe_float(row.get("avg_llm_calls")),
            )
        )
    return rows


def _safe_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class StatusResponse(BaseModel):
    provider: str
    model: str
    ready: bool
    message: str


def _ollama_ready(base_url: str, model: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        names = {item.get("name", "") for item in response.json().get("models", [])}
        aliases = {name.split(":")[0] for name in names if name}
        target = model.split(":")[0]
        if model in names or any(name.startswith(f"{model}:") for name in names) or target in aliases:
            return True, f"Ollama connected · {model}"
        installed = ", ".join(sorted(names)) or "none"
        return False, f"Ollama is running but model {model} was not found. Installed: {installed}"
    except Exception as exc:
        return False, f"Cannot connect to Ollama ({base_url}). Run: ollama serve — {exc}"


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    provider = (config.env("LLM_PROVIDER") or "dummy").lower()

    if provider == "dummy":
        return StatusResponse(
            provider=provider,
            model="dry-run",
            ready=False,
            message=(
                "LLM not configured. Set LLM_PROVIDER=ollama (or gemini / openai) in code/.env, "
                "then restart with bash code/run_web.sh."
            ),
        )

    if provider == "ollama":
        model = config.env("OLLAMA_MODEL", "qwen2.5:3b") or "qwen2.5:3b"
        base_url = config.env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
        ready, message = _ollama_ready(base_url, model)
        return StatusResponse(provider=provider, model=model, ready=ready, message=message)

    if provider == "gemini":
        model = config.env("GEMINI_MODEL", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite"
        if not (config.env("GEMINI_API_KEY") or "").strip():
            return StatusResponse(
                provider=provider,
                model=model,
                ready=False,
                message="GEMINI_API_KEY is not set. Add it in code/.env and restart the server.",
            )
        return StatusResponse(provider=provider, model=model, ready=True, message=f"Using {provider} / {model}")

    if provider in {"openai_compatible", "compatible", "deepseek", "groq"}:
        model = config.env("OPENAI_COMPATIBLE_MODEL", "") or ""
        if not (config.env("OPENAI_COMPATIBLE_API_KEY") or "").strip():
            return StatusResponse(
                provider=provider,
                model=model,
                ready=False,
                message="OPENAI_COMPATIBLE_API_KEY is not set. Add it in code/.env and restart the server.",
            )
        return StatusResponse(provider=provider, model=model, ready=True, message=f"Using {provider} / {model}")

    if provider == "openai":
        model = config.env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
        if not (config.env("OPENAI_API_KEY") or "").strip():
            return StatusResponse(
                provider=provider,
                model=model,
                ready=False,
                message="OPENAI_API_KEY is not set. Add it in code/.env and restart the server.",
            )
        return StatusResponse(provider=provider, model=model, ready=True, message=f"Using {provider} / {model}")

    return StatusResponse(
        provider=provider,
        model="unknown",
        ready=False,
        message=f"Unknown LLM_PROVIDER={provider}",
    )


@app.get("/api/benchmark", response_model=list[BenchmarkRow])
async def benchmark() -> list[BenchmarkRow]:
    return _load_benchmark()


@app.post("/api/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retriever = _get_retriever()
        llm = _get_llm()
        top_k = int(config.env("TOP_K", "5") or "5")
        results = retriever.search(question, k=top_k)
        route, harness_outputs, final_output = run_harnesses(llm, retriever, question, k=top_k)
        comparison = harnesses_to_dict(route, harness_outputs, final_output)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Course index not found. Run `python3 code/run_ingest.py` first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    passages = [
        Passage(
            source=result.source_file,
            page=result.page,
            score=result.score,
            text=result.text,
        )
        for result in results
    ]
    return QueryResponse(
        passages=passages,
        text=format_context(results),
        answer=str(comparison["final_answer"]),
        route=route,
        harnesses=[HarnessResult(**item) for item in comparison["harnesses"]],
        final_metrics=comparison["final_metrics"],
        benchmark=_load_benchmark(),
    )


@app.get("/api/slide")
async def slide_image(
    source: str = Query(min_length=1),
    page: int = Query(ge=1),
) -> Response:
    try:
        png = render_page_png(source, page)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
