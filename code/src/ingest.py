from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config
from .utils import normalize_space, safe_stem, stable_id, write_jsonl


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".ipynb"}


def convert_pdf(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\f".join(pages)


def convert_ipynb(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    pieces: list[str] = []
    for i, cell in enumerate(data.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        if source.strip():
            pieces.append(f"[cell {i}: {cell_type}]\n{source}")
    return "\n\n".join(pieces)


def convert_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return convert_pdf(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".ipynb":
        return convert_ipynb(path)
    raise ValueError(f"Unsupported file type: {path}")


def source_group(path: Path) -> str:
    for parent in path.parents:
        if parent.name in {"course_slides", "tutorials", "assessment_briefs"}:
            return parent.name
    return "unknown"


def iter_knowledge_files() -> list[Path]:
    files: list[Path] = []
    for directory in config.KNOWLEDGE_SOURCE_DIRS:
        if directory.exists():
            files.extend(
                sorted(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
    return files


def split_pages(text: str) -> list[str]:
    pages = text.split("\f")
    if len(pages) == 1:
        return [text]
    return pages


def chunk_words(text: str, max_words: int = 220, overlap: int = 45) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(max_words: int = 220, overlap: int = 45) -> list[dict[str, object]]:
    config.ensure_dirs()
    rows: list[dict[str, object]] = []
    for path in iter_knowledge_files():
        raw_text = convert_file(path)
        text_path = config.TEXT_DIR / f"{safe_stem(path)}.txt"
        text_path.write_text(raw_text, encoding="utf-8")

        for page_number, page in enumerate(split_pages(raw_text), start=1):
            page = normalize_space(page)
            if len(page) < 80:
                continue
            for chunk_number, chunk in enumerate(chunk_words(page, max_words, overlap), start=1):
                chunk = normalize_space(chunk)
                if len(chunk) < 80:
                    continue
                source = path.name
                chunk_id = stable_id(f"{source}:{page_number}:{chunk_number}:{chunk}")
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": source,
                        "source_group": source_group(path),
                        "page": page_number,
                        "chunk_number": chunk_number,
                        "text": chunk,
                    }
                )
    write_jsonl(config.CHUNKS_PATH, rows)
    return rows


def main() -> None:
    files = iter_knowledge_files()
    if not files:
        print("No source files found. Add course materials under:")
        for directory in config.KNOWLEDGE_SOURCE_DIRS:
            print(f"  - {directory}")
        print("Supported types: .pdf, .md, .txt, .ipynb")
        print("See docs/material_manifest.md for the expected file list.")
        return

    rows = build_chunks()
    print(f"Processed {len(files)} file(s).")
    print(f"Wrote {len(rows)} chunks to {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
