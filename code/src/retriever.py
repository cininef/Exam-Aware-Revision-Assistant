from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .utils import read_jsonl


@dataclass
class RetrievalResult:
    chunk_id: str
    source_file: str
    source_group: str
    page: int
    score: float
    text: str

    def citation(self) -> str:
        return f"{self.source_file}, page {self.page}, chunk {self.chunk_id}"


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    def __init__(self, chunks_path: Path = config.CHUNKS_PATH):
        self.rows = read_jsonl(chunks_path)
        if not self.rows:
            raise FileNotFoundError(
                f"No chunks found at {chunks_path}. Run: python3 code/run_ingest.py"
            )
        self.texts = [row["text"] for row in self.rows]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        if scores.size == 0:
            return []
        top_indices = np.argsort(-scores)[:k]
        results: list[RetrievalResult] = []
        for index in top_indices:
            row = self.rows[int(index)]
            results.append(
                RetrievalResult(
                    chunk_id=str(row["chunk_id"]),
                    source_file=str(row["source_file"]),
                    source_group=str(row["source_group"]),
                    page=int(row["page"]),
                    score=float(scores[index]),
                    text=str(row["text"]),
                )
            )
        return results


class ChromaRetriever(BaseRetriever):
    """Persistent semantic vector retriever.

    Chroma is optional because it adds a heavier dependency and may download a
    default embedding model the first time it runs. The TF-IDF retriever remains
    the zero-setup fallback.
    """

    def __init__(
        self,
        chunks_path: Path = config.CHUNKS_PATH,
        persist_dir: Path = config.CHROMA_DIR,
        collection_name: str = config.CHROMA_COLLECTION,
        rebuild: bool = False,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "Chroma backend requested but chromadb is not installed. "
                "Run: python3 -m pip install -r code/requirements-vector.txt"
            ) from exc

        self.rows = read_jsonl(chunks_path)
        if not self.rows:
            raise FileNotFoundError(
                f"No chunks found at {chunks_path}. Run: python3 code/run_ingest.py"
            )
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        if rebuild:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "COMP5541 course-material chunks"},
        )
        if self.collection.count() == 0:
            self._populate()

    def _populate(self, batch_size: int = 128) -> None:
        ids = [str(row["chunk_id"]) for row in self.rows]
        documents = [str(row["text"]) for row in self.rows]
        metadatas = [
            {
                "source_file": str(row["source_file"]),
                "source_group": str(row["source_group"]),
                "page": int(row["page"]),
                "chunk_number": int(row["chunk_number"]),
            }
            for row in self.rows
        ]
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        result = self.collection.query(query_texts=[query], n_results=k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows: list[RetrievalResult] = []
        for chunk_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
            distance_value = float(distance)
            score = 1.0 / (1.0 + max(distance_value, 0.0))
            rows.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    source_file=str(metadata.get("source_file", "")),
                    source_group=str(metadata.get("source_group", "")),
                    page=int(metadata.get("page", 0)),
                    score=score,
                    text=str(doc),
                )
            )
        return rows


def build_retriever(rebuild_chroma: bool = False) -> BaseRetriever:
    backend = (config.env("RETRIEVER_BACKEND", "tfidf") or "tfidf").lower()
    if backend == "chroma":
        return ChromaRetriever(rebuild=rebuild_chroma)
    if backend == "tfidf":
        return TfidfRetriever()
    raise ValueError(f"Unknown RETRIEVER_BACKEND={backend}. Use 'tfidf' or 'chroma'.")


def format_context(results: list[RetrievalResult]) -> str:
    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        parts.append(
            f"[Evidence {i}] {result.citation()} | score={result.score:.4f}\n"
            f"{result.text}"
        )
    return "\n\n".join(parts)
