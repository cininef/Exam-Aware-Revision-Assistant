from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEXT_DIR = PROCESSED_DIR / "text"
INDEX_DIR = PROCESSED_DIR / "index"
CHROMA_DIR = PROCESSED_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "outputs"
RECORDS_DIR = PROJECT_ROOT / "records"

KNOWLEDGE_SOURCE_DIRS = [
    RAW_DIR / "course_slides",
    RAW_DIR / "tutorials",
    RAW_DIR / "assessment_briefs",
]

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
CHROMA_COLLECTION = "comp5541_course_chunks"
EVALUATION_SET_PATH = RECORDS_DIR / "evaluation_set.csv"
RESULTS_PATH = RECORDS_DIR / "results.csv"
SUMMARY_PATH = RECORDS_DIR / "summary.csv"


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def ensure_dirs() -> None:
    for path in [TEXT_DIR, INDEX_DIR, CHROMA_DIR, CACHE_DIR, OUTPUT_DIR, RECORDS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
